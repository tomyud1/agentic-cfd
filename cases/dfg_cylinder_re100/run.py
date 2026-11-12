"""Build and run the DFG 2D-2 cylinder case (Re = 100) on the mesh levels in numerics.yaml; save raw outputs.

Usage (repo root, inside WSL):
  python -m cases.dfg_cylinder_re100.run mesh M1 [M2 ...]        # build + checkMesh only (inspect)
  python -m cases.dfg_cylinder_re100.run preflight                # short runs on M1 (from rest) and M4 (mapped)
  python -m cases.dfg_cylinder_re100.run production NAME ...      # named runs (see RUNS below), in order given
  python -m cases.dfg_cylinder_re100.run scaling [CORES ...]      # MPI scaling test on the finest mesh
  python -m cases.dfg_cylinder_re100.run psolver                  # pressure linear-solver test on M3
  python -m cases.dfg_cylinder_re100.run live NAME ...            # run NAME with the live periodic stop -> live_NAME
  python -m cases.dfg_cylinder_re100.run live_check               # replay the stop rule on M1..M4; compare live_ runs
Outputs: results/dfg_cylinder_re100/{preflight.json, runs/<name>.json + .npz, perf/<name>.json}
"""
import json
import math
import platform
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from cfdlib import foam, monitor, perf

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS_DIR = OUT / "runs"
PERF_DIR = OUT / "perf"
COURANT_LIMIT = 2.0     # PIMPLE with 2 outer correctors: stable and accurate up to Co ~ 1.6 here (time-step study)


def load():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    return problem, numerics


def run_root():
    return Path(foam.foam_env("FOAM_RUN")) / "agentic" / HERE.name


# ---------------------------------------------------------------------------------------------
# Mesh: O-grid around the cylinder inside a square box, 8 rectangular blocks around it
# ---------------------------------------------------------------------------------------------
def geometric_first(length, n, ratio):
    """First cell of n cells over `length` with last/first = ratio (blockMesh simpleGrading)."""
    if abs(ratio - 1) < 1e-12 or n == 1:
        return length / n
    r = ratio ** (1.0 / (n - 1))
    return length * (r - 1) / (r ** n - 1)


def mesh_spec(level, problem, numerics):
    g, mm = problem["given"], numerics["mesh"]
    H, L, D, xc, yc = (g[k]["value"] for k in ("H", "L", "D", "xc", "yc"))
    R = D / 2
    k = mm["levels"].index(level)
    nc = mm["nc"][k]
    nr, nl, nb = (int(round(nc * mm[key])) for key in ("nr_over_nc", "nl_over_nc", "nb_over_nc"))
    nt, nd = mm["nt_M1"] * 2 ** k, mm["nd_M1"] * 2 ** k
    b0, b1 = mm["box"]
    X, Y = [0.0, b0, b1, L], [0.0, b0, b1, H]
    dz = mm["dz"]
    pts = [(X[i], Y[j]) for j in range(4) for i in range(4)]            # P(i, j) = 4 j + i
    s = R / math.sqrt(2)
    pts += [(xc - s, yc - s), (xc + s, yc - s), (xc + s, yc + s), (xc - s, yc + s)]   # SW SE NE NW = 16..19
    nv = len(pts)
    vertices = [(x, y, 0.0) for x, y in pts] + [(x, y, dz) for x, y in pts]
    P = lambda i, j: 4 * j + i
    SW, SE, NE, NW = 16, 17, 18, 19
    Ed, Er = mm["downstream_ratio"], mm["radial_ratio"]

    def blk(name, quad, n, gr=(1, 1)):
        return dict(name=name, v=list(quad) + [q + nv for q in quad], n=n, g=gr)

    blocks = [
        blk("below-upstream", (P(0, 0), P(1, 0), P(1, 1), P(0, 1)), (nl, nb)),
        blk("below-box", (P(1, 0), P(2, 0), P(2, 1), P(1, 1)), (nc, nb)),
        blk("below-downstream", (P(2, 0), P(3, 0), P(3, 1), P(2, 1)), (nd, nb), (Ed, 1)),
        blk("upstream", (P(0, 1), P(1, 1), P(1, 2), P(0, 2)), (nl, nc)),
        blk("downstream", (P(2, 1), P(3, 1), P(3, 2), P(2, 2)), (nd, nc), (Ed, 1)),
        blk("above-upstream", (P(0, 2), P(1, 2), P(1, 3), P(0, 3)), (nl, nt)),
        blk("above-box", (P(1, 2), P(2, 2), P(2, 3), P(1, 3)), (nc, nt)),
        blk("above-downstream", (P(2, 2), P(3, 2), P(3, 3), P(2, 3)), (nd, nt), (Ed, 1)),
        # O-grid: local x along the box side, local y radially inwards (cells shrink towards the cylinder)
        blk("O-south", (P(1, 1), P(2, 1), SE, SW), (nc, nr), (1, 1 / Er)),
        blk("O-east", (P(2, 1), P(2, 2), NE, SE), (nc, nr), (1, 1 / Er)),
        blk("O-north", (P(2, 2), P(1, 2), NW, NE), (nc, nr), (1, 1 / Er)),
        blk("O-west", (P(1, 2), P(1, 1), SW, NW), (nc, nr), (1, 1 / Er)),
    ]
    arcs = []
    for a, b, (mx, my) in ((SW, SE, (xc, yc - R)), (SE, NE, (xc + R, yc)), (NE, NW, (xc, yc + R)), (NW, SW, (xc - R, yc))):
        arcs += [(a, b, (mx, my, 0.0)), (a + nv, b + nv, (mx, my, dz))]

    def side(a, b):
        return [a, b, b + nv, a + nv]

    patches = {
        "inlet": dict(type="patch", faces=[side(P(0, j + 1), P(0, j)) for j in range(3)]),
        "outlet": dict(type="patch", faces=[side(P(3, j), P(3, j + 1)) for j in range(3)]),
        "walls": dict(type="wall", faces=[side(P(i, 0), P(i + 1, 0)) for i in range(3)]
                                         + [side(P(i + 1, 3), P(i, 3)) for i in range(3)]),
        "cylinder": dict(type="wall", faces=[side(SE, SW), side(NE, SE), side(NW, NE), side(SW, NW)]),
        "frontAndBack": dict(type="empty", faces=[[b["v"][0], b["v"][3], b["v"][2], b["v"][1]] for b in blocks]
                                                 + [b["v"][4:] for b in blocks]),
    }
    cells = sum(b["n"][0] * b["n"][1] for b in blocks)
    # Characteristic cell sizes (for the report): cylinder surface, first radial cell, box, downstream end
    h_surf = (math.pi * D / 4) / nc
    h_rad = geometric_first(b1 - yc - R, nr, 1 / Er)      # along the O-grid axis; last (cylinder) cell:
    r_ = (1 / Er) ** (1 / (nr - 1))
    h_wall = h_rad * r_ ** (nr - 1)
    h_box = (b1 - b0) / nc
    h_out = geometric_first(L - b1, nd, Ed) * Ed
    return dict(level=level, nc=nc, nr=nr, nl=nl, nb=nb, nt=nt, nd=nd, cells=cells, vertices=vertices,
                blocks=blocks, arcs=arcs, patches=patches, dz=dz,
                h=dict(surface=h_surf, wall=h_wall, box=h_box, outlet=h_out))


# ---------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------
def dt_of(level, numerics, factor=1.0):
    k = numerics["mesh"]["levels"].index(level)
    return numerics["time"]["dt_M1"] / 2 ** k * factor


def fmt_t(t):
    return f"{round(t, 9):g}"


def params_for(level, problem, numerics, dt, start_time, end_time, write_interval, purge, cores=1,
               div_U=None, p_solver=None, p_tol=None, p_relTol=None, n_outer=None):
    g = problem["given"]
    ms = mesh_spec(level, problem, numerics)
    ls, pim, sch = numerics["linear_solvers"], numerics["pimple"], numerics["schemes"]
    return dict(
        ms, H=g["H"]["value"], Um=g["Um"]["value"], nu=g["nu"]["value"], rho=g["rho"]["value"],
        xc=g["xc"]["value"], yc=g["yc"]["value"], dt=dt, start_time=fmt_t(start_time), end_time=fmt_t(end_time),
        write_interval=write_interval, purge_write=purge, cores=cores,
        ddt=sch["ddt"], div_U=div_U or sch["div_U"],
        p_solver=p_solver or ls["p"]["solver"], p_tol=p_tol or ls["p"]["tol"],
        p_relTol=ls["p"]["relTol"] if p_relTol is None else p_relTol, U_tol=ls["U"]["tol"],
        n_outer=n_outer or pim["nOuterCorrectors"], n_corr=pim["nCorrectors"], n_nonorth=pim["nNonOrthogonalCorrectors"],
        probes=numerics["probes"])


def build(name, level, problem, numerics, start_time=0.0, **kw):
    """Render a case, run blockMesh + checkMesh, and name the initial-field directory after start_time."""
    prm = params_for(level, problem, numerics, start_time=start_time, **kw)
    case = foam.render_case(HERE / "template", run_root() / name, prm)
    foam.run("blockMesh", case)
    foam.run("checkMesh", case)
    if start_time:
        (case / "0").rename(case / fmt_t(start_time))
    return case, prm


def map_from(case, source):
    """Initial field = latest time of `source` (a run of this case, possibly on another mesh)."""
    foam.run(f"mapFields {source} -consistent -sourceTime latestTime", case, "mapFields")


def solve(case, cores, timeout=None, log_name=None, watch=None):
    """Run pimpleFoam (in parallel if cores > 1). watch: live check (cfdlib.monitor.LiveStop)."""
    solver = "pimpleFoam"
    name = log_name or solver
    if cores > 1:
        foam.run("decomposePar -force", case, "decomposePar")
        hw = " --use-hwthread-cpus" if cores > 8 else ""      # more ranks than physical cores: use the 16 hardware threads
        wall = foam.run(f"mpirun{hw} -np {cores} {solver} -parallel", case, name, timeout=timeout, watch=watch)
        foam.run("reconstructPar", case, "reconstructPar")
        for d in case.glob("processor*"):
            shutil.rmtree(d)
    else:
        wall = foam.run(solver, case, name, timeout=timeout, watch=watch)
    return wall


def _dat(case, fo, fname):
    hits = sorted((case / "postProcessing" / fo).rglob(fname))
    if not hits:
        return None
    rows = []
    for h in hits:                       # restarts write one folder per start time
        rows += foam.read_dat(h)
    return rows


def collect(case):
    """Time series written by the function objects (every time step)."""
    out = {}
    fc = _dat(case, "forcesCyl", "force.dat")
    fw = _dat(case, "forcesWalls", "force.dat")
    # force.dat (v2606): time, total xyz, pressure xyz, viscous xyz as nine scalar columns
    fc, fw = np.array(fc, dtype=float), np.array(fw, dtype=float)
    out["t"] = fc[:, 0]
    out["Fcyl"] = fc[:, 1:4]                            # total force on the cylinder
    out["Fcyl_p"] = fc[:, 4:7]
    out["Fcyl_v"] = fc[:, 7:10]
    out["Fwall"] = fw[:, 1:4]
    for key, fo, fname in (("mom_in", "momFlux_inlet", "surfaceFieldValue.dat"),
                           ("mom_out", "momFlux_outlet", "surfaceFieldValue.dat"),
                           ("p_in", "pForce_inlet", "surfaceFieldValue.dat"),
                           ("p_out", "pForce_outlet", "surfaceFieldValue.dat"),
                           ("q_in", "flow_inlet", "surfaceFieldValue.dat"),
                           ("q_out", "flow_outlet", "surfaceFieldValue.dat"),
                           ("momentum", "momentum", "volFieldValue.dat")):
        rows = _dat(case, fo, fname)
        out[key] = np.array([r[1] for r in rows])
        out[key + "_t"] = np.array([r[0] for r in rows])
    pr = _dat(case, "probes", "U")
    out["probe_t"] = np.array([r[0] for r in pr])
    out["probe_U"] = np.array([r[1] for r in pr])
    return out


def record(name, case, prm, cores, wall, extra=None):
    """Save time series + metadata of a finished run."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    ts = collect(case)
    np.savez_compressed(RUNS_DIR / f"{name}.npz", **ts)
    log = case / "log.pimpleFoam"
    health = monitor.assess(log, courant_limit=COURANT_LIMIT)
    pf = perf.summarize(case, "pimpleFoam", cells=prm["cells"], cores=cores)
    (PERF_DIR / f"{name}.json").write_text(json.dumps(pf, indent=1))
    cm = (case / "log.checkMesh").read_text()
    meta = dict(name=name, level=prm["level"], cells=prm["cells"], h=prm["h"], dt=prm["dt"],
                start_time=float(prm["start_time"]), end_time=float(prm["end_time"]), cores=cores,
                div_U=prm["div_U"], p_solver=prm["p_solver"], case_dir=str(case), wall_time=wall,
                mesh_ok="Mesh OK." in cm, checkmesh=_checkmesh_numbers(cm), health=health, perf=pf,
                openfoam=foam.foam_env("WM_PROJECT_VERSION"), python=platform.python_version(),
                run_root=str(run_root()))
    meta.update(extra or {})
    (RUNS_DIR / f"{name}.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"{name}: {prm['cells']} cells, {cores} cores, {wall:.1f} s, health {health['status']}, "
          f"{pf['us_per_cell_step']:.3g} us/cell/step", flush=True)
    return meta


def _checkmesh_numbers(text):
    import re
    out = {}
    for key, pat in (("max_nonorth", r"Mesh non-orthogonality Max: ([\d.eE+-]+)"),
                     ("max_skew", r"Max skewness = ([\d.eE+-]+)"),
                     ("max_aspect", r"Max aspect ratio = ([\d.eE+-]+)")):
        m = re.search(pat, text)
        out[key] = float(m.group(1)) if m else None
    return out


# Production runs: name -> (level, dt factor, source run for the initial field, run length key, options)
def run_table(numerics):
    t = numerics["time"]
    lv = numerics["mesh"]["levels"]
    table = {}
    t0 = 0.0
    prev = None
    for i, L in enumerate(lv):
        length = t["develop_end"] if i == 0 else t["run_length"]
        if i == len(lv) - 1:
            length = t.get("run_length_finest", length)
        table[L] = dict(level=L, factor=1.0, source=prev, start=t0, end=t0 + length,
                        cores=numerics["parallel"]["cores"][L], final=(i == len(lv) - 1))
        prev, t0 = L, t0 + length
    ts = numerics["timestep_study"]
    base = table[ts["mesh"]]
    for f in ts["factors"]:
        table[f"{ts['mesh']}_dtx{f:g}"] = dict(level=ts["mesh"], factor=f, source=ts["start_from"], start=base["end"],
                                               end=base["end"] + t["run_length"], cores=1, final=False)
    s = numerics["sensitivity"]
    b = table[s["mesh"]]
    table[f"{s['mesh']}_LU"] = dict(level=s["mesh"], factor=1.0, source=b["source"], start=b["start"], end=b["end"],
                                    cores=b["cores"], final=False, div_U=s["div_U"])
    return table


def do_run(name, timeout=2400):
    problem, numerics = load()
    spec = run_table(numerics)[name]
    t = numerics["time"]
    dt = dt_of(spec["level"], numerics, spec["factor"])
    length = spec["end"] - spec["start"]
    wi, purge = (t["write_interval_final"], t["keep_writes"]) if spec["final"] else (length, 0)
    case, prm = build(name, spec["level"], problem, numerics, start_time=spec["start"], dt=dt,
                      end_time=spec["end"], write_interval=wi, purge=purge, cores=spec["cores"],
                      div_U=spec.get("div_U"))
    if spec["source"]:
        map_from(case, run_root() / spec["source"])
    wall = solve(case, spec["cores"], timeout=timeout)
    return record(name, case, prm, spec["cores"], wall, extra=dict(source=spec["source"], dt_factor=spec["factor"]))


# ---------------------------------------------------------------------------------------------
# Live stop (cfdlib.monitor.LiveStop): a level ends once the case's periodicity rule holds
# ---------------------------------------------------------------------------------------------
def periodic_rule(problem, numerics, min_length=1.5, periods_kept=14):
    """The acceptance periodicity rule as a stopping rule: over the last cycles_measured lift cycles the
    peaks and the period must change less than the acceptance tolerances (or have reached their floor,
    see monitor.settled). Needs at least min_length seconds of the run and cycles_measured + 1 cycles.

    The verdict depends only on the last 9 cycles (monitor.settled: 4 for the spread, 5 for the drift
    test, 9 for the floor test), so once the period is known only the last `periods_kept` periods (at
    least 3 s, the window of the mid-level of the cycle detection) are analysed: the cost of a check
    stays the same however long the run gets, and the verdict is the same as on the whole history."""
    from . import signals
    acc, n = numerics["acceptance"], numerics["time"]["cycles_measured"]
    tol = dict(cl_max=acc["periodic_cl_peak"], cl_min=acc["periodic_cl_peak"], cd_max=acc["periodic_cd_peak"],
               cd_min=acc["periodic_cd_peak"], period=acc["periodic_period_rel"])
    last_period = [None]

    def rule(t, cd, cl):
        if len(t) < 10 or t[-1] - t[0] < min_length:
            return None
        if last_period[0]:
            i0 = int(np.searchsorted(t, t[-1] - max(3.0, periods_kept * last_period[0])))
            t, cd, cl = t[i0:], cd[i0:], cl[i0:]
        cyc = signals.cycles(t, cd, cl, window=min(3.0, t[-1] - t[0]))
        if len(cyc) < n + 1:
            return None
        last_period[0] = cyc[-1]["period"]
        q = {k: [c[k] for c in cyc] for k in ("cl_max", "cl_min", "cd_max", "cd_min")}
        q["period"] = [c["period"] / cyc[-1]["period"] for c in cyc]   # relative change of the period
        d = monitor.judge(q, tol, n)
        d.update(t=float(t[-1]), cycles=len(cyc), cd_max=cyc[-1]["cd_max"], cl_max=cyc[-1]["cl_max"])
        return d
    return rule


def live_decider(problem, numerics):
    """decide(case) for monitor.LiveStop: drag and lift from the run's force.dat so far (read
    incrementally, monitor.DatTail), then periodic_rule."""
    from . import signals
    rule, dz = periodic_rule(problem, numerics), numerics["mesh"]["dz"]
    tails = {}

    def decide(case):
        tail = tails.setdefault(str(case), monitor.DatTail(case, "forcesCyl", "force.dat"))
        fc = tail()
        if len(fc) < 10:
            return None
        t, cd, cl = signals.coefficients(dict(t=fc[:, 0], Fcyl=fc[:, 1:4]), problem["given"], dz)
        return rule(t, cd, cl)
    return decide


def do_live(name, timeout=4000):
    """Run table entry NAME with the live stop, as live_NAME: same mesh, time step and source; it starts
    from the source's last written time and its planned length becomes the maximum."""
    problem, numerics = load()
    spec = run_table(numerics)[name]
    src = spec["source"]
    start = float(foam.time_dirs(run_root() / src)[-1].name) if src else 0.0
    length = spec["end"] - spec["start"]
    nm = f"live_{name}"
    dt = dt_of(spec["level"], numerics, spec["factor"])
    case, prm = build(nm, spec["level"], problem, numerics, start_time=start, dt=dt, end_time=start + length,
                      write_interval=length, purge=0, cores=spec["cores"], div_U=spec.get("div_U"))
    if src:
        map_from(case, run_root() / src)
    ls = monitor.LiveStop(case, live_decider(problem, numerics), log=lambda m: print(f"{nm}: {m}", flush=True))
    wall = solve(case, spec["cores"], timeout=timeout, watch=ls)
    t_end = float(foam.time_dirs(case)[-1].name)
    return record(nm, case, prm, spec["cores"], wall,
                  extra=dict(source=src, dt_factor=spec["factor"], live_stop=ls.summary(), t_end_actual=t_end,
                             max_length=length))


KEYS = ("cd_max", "cd_min", "cl_max", "cl_min", "f")


def do_live_check(names=("M1", "M2", "M3", "M4"), every=0.05):
    """Replay the stop rule on the recorded fixed-length runs (when would each have stopped, and with what
    numbers?) and compare any live_ runs with them. Writes results/dfg_cylinder_re100/live_stop.json."""
    from . import signals
    problem, numerics = load()
    rule, n, dz = periodic_rule(problem, numerics), numerics["time"]["cycles_measured"], numerics["mesh"]["dz"]
    out = dict(rule="periodicity tolerances of the acceptance rules over the last cycles_measured cycles, or "
                    "their floor (cfdlib.monitor.settled); the replay checks every 0.05 s of simulated time",
               runs={})
    for L in names:
        d = np.load(RUNS_DIR / f"{L}.npz")
        meta = json.loads((RUNS_DIR / f"{L}.json").read_text())
        t, cd, cl = signals.coefficients(d, problem["given"], dz)
        full = signals.summary(signals.cycles(t, cd, cl), n)
        hit = None
        for tc in np.arange(t[0] + every, t[-1] + 1e-9, every):
            m = t <= tc
            r = rule(t[m], cd[m], cl[m])
            if r and r["stop"]:
                hit = r
                break
        frac = (hit["t"] - t[0]) / (t[-1] - t[0]) if hit else 1.0
        rec = dict(start=float(t[0]), end=float(t[-1]), wall=meta["wall_time"], replay_stop=hit,
                   replay_fraction=frac, replay_wall_est=frac * meta["wall_time"], full={k: full[k] for k in KEYS})
        msg = f"{L}: {t[0]:.3f}..{t[-1]:.3f} s, {meta['wall_time']:.0f} s wall; replay: "
        if hit:
            m = t <= hit["t"]
            early = signals.summary(signals.cycles(t[m], cd[m], cl[m]), n)
            rec["replay_vs_full"] = {k: early[k] - full[k] for k in KEYS}
            msg += (f"stop at {hit['t']:.3f} ({hit['status']}) -> ~{rec['replay_wall_est']:.0f} s; "
                    + ", ".join(f"{k} {v:+.1e}" for k, v in rec["replay_vs_full"].items()))
        else:
            msg += "never stops"
        print(msg)
        lp = RUNS_DIR / f"live_{L}.npz"
        if lp.exists():
            lm = json.loads((RUNS_DIR / f"live_{L}.json").read_text())
            tl, cdl, cll = signals.coefficients(np.load(lp), problem["given"], dz)
            live = signals.summary(signals.cycles(tl, cdl, cll), n)
            rec["live"] = dict(start=float(tl[0]), end=float(tl[-1]), wall=lm["wall_time"],
                               decision=lm["live_stop"]["decision"], health=lm["health"]["status"],
                               vs_full={k: live[k] - full[k] for k in KEYS})
            print(f"   live_{L}: {tl[0]:.3f}..{tl[-1]:.3f} s, {lm['wall_time']:.0f} s wall, health "
                  f"{rec['live']['health']}; " + ", ".join(f"{k} {v:+.1e}" for k, v in rec["live"]["vs_full"].items()))
        out["runs"][L] = rec
    (OUT / "live_stop.json").write_text(json.dumps(out, indent=1, default=float))


def do_preflight():
    """Short runs: M1 from rest (stability, outputs, timing) and the finest mesh from the mapped M1 start field."""
    problem, numerics = load()
    lv = numerics["mesh"]["levels"]
    t_short = numerics["preflight"]["end_time_short"]
    out = []
    for L, cores in ((lv[0], 1), (lv[-1], numerics["parallel"]["cores"][lv[-1]])):
        dt = dt_of(L, numerics)
        full = numerics["time"]["develop_end"] if L == lv[0] else numerics["time"]["run_length"]
        case, prm = build(f"pf_{L}", L, problem, numerics, dt=dt, end_time=t_short, write_interval=t_short,
                          purge=0, cores=cores)
        t0 = time.time()
        try:
            wall = solve(case, cores, timeout=900)
            crashed = False
        except RuntimeError as e:
            wall, crashed = time.time() - t0, True
            print(e)
        if not (case / "log.pimpleFoam").exists():
            raise SystemExit(f"pre-flight {L}: the solver never started (see the logs in {case})")
        a = monitor.assess(case / "log.pimpleFoam", end_time=full, courant_limit=COURANT_LIMIT)
        ts = None
        try:
            ts = collect(case)
        except Exception as e:                 # noqa: BLE001 - a missing output is a pre-flight finding
            print("collect failed:", e)
        pf = perf.summarize(case, "pimpleFoam", cells=prm["cells"], cores=cores)
        steps_full = full / dt
        a.update(name=L, cells=prm["cells"], cores=cores, dt=dt, wall=wall, crashed=crashed, case=str(case),
                 end_time_short=t_short, full_length=full, perf=pf,
                 est_full=pf["us_per_cell_step"] * 1e-6 * prm["cells"] * steps_full if pf["us_per_cell_step"] else None,
                 n_force_rows=len(ts["t"]) if ts else 0, n_steps=round(t_short / dt),
                 force_last=ts["Fcyl"][-1].tolist() if ts else None,
                 q_in=ts["q_in"][-1] if ts else None, q_out=ts["q_out"][-1] if ts else None,
                 checkmesh=_checkmesh_numbers((case / "log.checkMesh").read_text()),
                 mesh_ok="Mesh OK." in (case / "log.checkMesh").read_text())
        out.append(a)
        print(f"preflight {L}: {a['status']}, {prm['cells']} cells, {cores} cores, {wall:.1f} s for {t_short} s, "
              f"{pf['us_per_cell_step']} us/cell/step; projected {a['est_full']} s for {full} s; "
              f"force rows {a['n_force_rows']}/{a['n_steps']}; q_in {a['q_in']} q_out {a['q_out']}; mesh {a['checkmesh']}")
        for c in a["checks"]:
            print(f"   {c['status']:4s} {c['name']}: {c['value']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(out, indent=1, default=str))


def do_mesh(levels):
    problem, numerics = load()
    for L in levels:
        case, prm = build(f"mesh_{L}", L, problem, numerics, dt=dt_of(L, numerics), end_time=0.1, write_interval=1,
                          purge=0)
        cm = (case / "log.checkMesh").read_text()
        print(L, prm["cells"], prm["h"], _checkmesh_numbers(cm), "Mesh OK." in cm)


def do_scaling(cores_list, steps=200):
    """Same short run on the finest mesh with 1, 2, 4, 8 ... MPI ranks; start field mapped from the developed M1."""
    problem, numerics = load()
    L = numerics["mesh"]["levels"][-1]
    dt = dt_of(L, numerics)
    t0 = numerics["time"]["develop_end"]
    old = PERF_DIR / "scaling.json"
    res = [r for r in (json.loads(old.read_text()) if old.exists() else []) if r["cores"] not in cores_list]
    for n in cores_list:
        case, prm = build(f"scale_{L}_np{n}", L, problem, numerics, start_time=t0, dt=dt, end_time=t0 + steps * dt,
                          write_interval=1e3, purge=0, cores=n)
        map_from(case, run_root() / numerics["mesh"]["levels"][0])
        wall = solve(case, n, timeout=1800)
        pf = perf.summarize(case, "pimpleFoam", cells=prm["cells"], cores=n)
        ts = collect(case)
        pf.update(wall_solver=wall, Fcyl_last=ts["Fcyl"][-1].tolist(), loop_us_per_cell_step=_loop_us(case, prm["cells"]))
        res.append(pf)
        print(f"np={n}: {wall:.1f} s, {pf['us_per_cell_step']:.3g} us/cell/step (loop {pf['loop_us_per_cell_step']:.3g}), "
              f"cpu eff {pf['cpu_efficiency']}, p iters {pf['lin_iters'].get('p')}, F {pf['Fcyl_last']}", flush=True)
        PERF_DIR.mkdir(parents=True, exist_ok=True)
        (PERF_DIR / "scaling.json").write_text(json.dumps(res, indent=1))


def _loop_us(case, cells):
    """Wall microseconds per cell per step inside the time loop (excludes start-up), from ClockTime."""
    import re
    log = (case / "log.pimpleFoam").read_text(errors="replace")
    ck = [float(x) for x in re.findall(r"ClockTime = ([\d.eE+-]+) s", log)]
    steps = len(re.findall(r"^Time = ", log, re.M))
    return 1e6 * (ck[-1] - ck[0]) / (cells * (steps - 1)) if len(ck) > 1 and steps > 1 else None


def do_psolver(level="M3", steps=200):
    """Pressure linear solver / tolerance variants on one mesh, same start field (mapped developed M1)."""
    problem, numerics = load()
    dt = dt_of(level, numerics)
    t0 = numerics["time"]["develop_end"]
    variants = [("GAMG", 1e-7, 0.01), ("PCG", 1e-7, 0.01), ("GAMG", 1e-7, 0.0), ("GAMG", 1e-6, 0.01), ("GAMG", 1e-9, 0.01)]
    res = []
    for solver, tol, rel in variants:
        nm = f"psolver_{level}_{solver}_{tol:g}_{rel:g}"
        case, prm = build(nm, level, problem, numerics, start_time=t0, dt=dt, end_time=t0 + steps * dt, write_interval=1e3,
                          purge=0, cores=1, p_solver=solver, p_tol=tol, p_relTol=rel)
        map_from(case, run_root() / numerics["mesh"]["levels"][0])
        wall = solve(case, 1)
        pf = perf.summarize(case, "pimpleFoam", cells=prm["cells"], cores=1)
        ts = collect(case)
        pf.update(variant=dict(solver=solver, tol=tol, relTol=rel), Fcyl=ts["Fcyl"].tolist(),
                  loop_us_per_cell_step=_loop_us(case, prm["cells"]))
        res.append(pf)
        print(f"{nm}: {wall:.1f} s, loop {pf['loop_us_per_cell_step']:.3g} us/cell/step, p iters {pf['lin_iters'].get('p')}, "
              f"final F {ts['Fcyl'][-1]}", flush=True)
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    (PERF_DIR / f"psolver_{level}.json").write_text(json.dumps(res, indent=1))


def do_experiment(name, level, factor, n_outer, source, cores=1, length=None):
    """Exploratory run (time-integration tests): same setup with dt x factor and n_outer outer correctors."""
    problem, numerics = load()
    src = run_table(numerics)[source]
    start, length = src["end"], length or numerics["time"]["run_length"]
    dt = dt_of(level, numerics, factor)
    case, prm = build(name, level, problem, numerics, start_time=start, dt=dt, end_time=start + length,
                      write_interval=length, purge=0, cores=cores, n_outer=n_outer)
    map_from(case, run_root() / source)
    wall = solve(case, cores, timeout=2400)
    return record(name, case, prm, cores, wall, extra=dict(source=source, dt_factor=factor, n_outer=n_outer, experiment=True))


def do_wake(name):
    """Fields of a finished run at the written time closest to a lift maximum: vorticity + cell centres -> npz."""
    from . import signals
    problem, numerics = load()
    case = run_root() / name
    d = np.load(RUNS_DIR / f"{name}.npz")
    t, cd, cl = signals.coefficients(d, problem["given"], numerics["mesh"]["dz"])
    cyc = signals.cycles(t, cd, cl)
    times = [float(x.name) for x in foam.time_dirs(case)]
    tmax = [c["t_clmax"] for c in cyc]
    best = min(((abs(tw - tm), tw, tm) for tw in times for tm in tmax if tw > times[0]), key=lambda z: z[0])
    tw = best[1]
    tname = next(x.name for x in foam.time_dirs(case) if abs(float(x.name) - tw) < 1e-9)
    foam.run(f"postProcess -func vorticity -time {tname}", case, "vorticity")
    foam.run(f"postProcess -func writeCellCentres -time {tname}", case, "writeCellCentres")
    C = foam.read_internal_field(case / tname / "C")
    W = foam.read_internal_field(case / tname / "vorticity")
    U = foam.read_internal_field(case / tname / "U")
    cl_at = float(np.interp(tw, t, cl))
    np.savez_compressed(RUNS_DIR / f"{name}_wake.npz", x=C[:, 0], y=C[:, 1], wz=W[:, 2], ux=U[:, 0], uy=U[:, 1],
                        t=tw, t_clmax=best[2], cl=cl_at)
    print(f"wake picture of {name} at t = {tw} (lift maximum at {best[2]:.5f}, offset {best[0]:.4g} s), cl = {cl_at:.5f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    args = sys.argv[2:]
    if mode == "mesh":
        do_mesh(args or load()[1]["mesh"]["levels"])
    elif mode == "preflight":
        do_preflight()
    elif mode == "production":
        jobs = 1
        if "--jobs" in args:
            k = args.index("--jobs")
            jobs = int(args[k + 1])
            args = args[:k] + args[k + 2:]
        if jobs == 1:
            for nm in args:
                do_run(nm)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                list(ex.map(do_run, args))
    elif mode == "scaling":
        do_scaling([int(a) for a in args] or [1, 2, 4, 8])
    elif mode == "exp":
        # exp NAME LEVEL FACTOR NOUTER SOURCE [CORES]  (several runs: separate them with '--')
        groups, cur = [], []
        for x in args:
            if x == "--":
                groups.append(cur)
                cur = []
            else:
                cur.append(x)
        groups.append(cur)
        jobs = [(gq[0], gq[1], float(gq[2]), int(gq[3]), gq[4], int(gq[5]) if len(gq) > 5 else 1) for gq in groups if gq]
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            list(ex.map(lambda j: do_experiment(*j), jobs))
    elif mode == "wake":
        do_wake(args[0])
    elif mode == "live":
        for nm in args:
            do_live(nm)
    elif mode == "live_check":
        do_live_check()
    elif mode == "psolver":
        do_psolver(*(args[:1] or []))
    else:
        raise SystemExit(f"unknown mode {mode}")
