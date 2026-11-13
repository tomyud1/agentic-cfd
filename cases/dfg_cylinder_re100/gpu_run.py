"""The DFG 2D-2 cylinder case (Re = 100) on gpucfd: the same problem and protocol as run.py's OpenFOAM levels.

Usage (repo root, inside WSL):
  python -m cases.dfg_cylinder_re100.gpu_run preflight     # short runs on G1 (from rest) and G3 (from G1): cost, health
  python -m cases.dfg_cylinder_re100.gpu_run production [NAME ...]  # G1 0..15 s from rest; G2, G2_dtx0.5, G3 (run_table)
  python -m cases.dfg_cylinder_re100.gpu_run clustered [C1 C2 C3]   # the protocol on grids clustered around the cylinder
  python -m cases.dfg_cylinder_re100.gpu_run clustered C3i # any other run of the table, here C3 with the IMEX scheme
  python -m cases.dfg_cylinder_re100.gpu_run timing C3     # cost per step of a run before making it
  python -m cases.dfg_cylinder_re100.gpu_run tolcheck C3i C3 [LENGTH [TOL ...]]   # the pressure tolerance run C3i may
                                                           # use, from the last field of run C3 (a run on the same grid)
Every run is a row of the table in numerics.yaml (gpucfd.runs): its grid level, variant, scheme and source run.
Outputs: results/dfg_cylinder_re100/runs/<G..>.json + .npz (cfdlib.gpu.write_run), gpu_preflight.json;
final velocity fields in ~/gpucfd_runs/dfg_cylinder_re100/ (outside git, like OpenFOAM's run directories).

npz keys: t, Fcyl (force per unit density and unit depth on the cylinder, x y z), probe_v (v at the wake
probe), q_in / q_out (volume flow per unit depth through inlet / outlet, out of the domain positive), umax.
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from cfdlib import gpu
from . import signals as S

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS_DIR = OUT / "runs"
FIELDS = Path.home() / "gpucfd_runs" / HERE.name


def load():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    return problem, numerics


def run_spec(name, numerics):
    """One row of the run table (numerics.yaml, gpucfd.runs), completed: dict(name, level, family ("uniform" or
    "clustered"), variant (its name or None), changes (the variant's settings), scheme, source, dt_factor)."""
    G = numerics["gpucfd"]
    if name not in G["runs"]:
        raise ValueError(f"unknown run {name}: add a row for it to numerics.yaml (gpucfd.runs)")
    r = G["runs"][name] or {}
    level, v = r.get("level", name), r.get("variant")
    if level not in G["levels"] + G["clustered"]["levels"]:
        raise ValueError(f"run {name}: unknown level {level}")
    return dict(name=name, level=level, family="uniform" if level in G["levels"] else "clustered", variant=v,
                changes=dict(G["clustered"]["variants"][v]) if v else {}, scheme=r.get("scheme", G["scheme"]),
                source=r.get("source"), dt_factor=r.get("dt_factor", 1.0))


def runs_where(numerics, **want):
    """Names of the runs of the table whose completed row has these values, in the table's order:
    runs_where(numerics, family="clustered", scheme="rk3", variant="s") -> ["C1s", "C2s", "C3s"]."""
    out = []
    for name in numerics["gpucfd"]["runs"]:
        spec = run_spec(name, numerics)
        if all(spec[k] == v for k, v in want.items()):
            out.append(name)
    return out


def make(run, problem, numerics):
    """A simulation for a run of the table: its grid, the cylinder's place and the time scheme."""
    from gpucfd import gcfd, bodies as B
    g = {k: v["value"] for k, v in problem["given"].items()}
    G = numerics["gpucfd"]
    spec = run_spec(run, numerics)
    H, Um = g["H"], g["Um"]
    grid = None
    xc = g["xc"]
    if spec["family"] == "uniform":
        k = G["levels"].index(spec["level"])
        n = (G["n_G1"][0] * 2 ** k, G["n_G1"][1] * 2 ** k, 1)
    else:  # grids clustered around the cylinder (numerics.yaml, gpucfd.clustered)
        from gpucfd import grids
        C, var = G["clustered"], spec["changes"]
        k = C["levels"].index(spec["level"])
        n0, band = var.get("n_C1", C["n_C1"]), var.get("band", C["band"])
        n = (n0[0] * 2 ** k, n0[1] * 2 ** k, 1)
        h = g["D"] / (C["D_h_C1"] * 2 ** k)
        grid = (grids.clustered(n[0], g["L"], band[0], h, C["grow"][0]),
                grids.clustered(n[1], H, band[1], h, C["grow"][1]), None)
        xc = g["xc"] + var.get("x_shift", 0.0)  # the cylinder moved relative to the grid (variant s)

    def inflow(X, Y, Z):
        return (4 * Um * Y * (H - Y) / H ** 2, 0 * X, 0 * X)
    bnd = {"x-": ("inflow", inflow), "x+": "outflow", "y-": "wall", "y+": "wall"}
    s = gcfd.NavierStokes(n, (g["L"], H, 1.0), g["nu"], scheme=spec["scheme"], boundaries=bnd,
                          bodies=[B.Cylinder((xc, g["yc"]), g["D"] / 2)], grid=grid)
    return s, inflow


def from_rest(s, inflow):
    """Start field: the inflow profile everywhere in the fluid (zero in the body), projected."""
    q = []
    for c in range(2):
        X, Y, Z = s.face_coords(c)
        q.append(np.where(s.fluid[c], inflow(X, Y, Z)[c], 0.0))
    s.set(q)
    return s.project()


def observer(problem, numerics):
    g = {k: v["value"] for k, v in problem["given"].items()}
    G = numerics["gpucfd"]
    box = (tuple(G["force_box"][0]), tuple(G["force_box"][1]), (0.0, 1.0))
    px, py = numerics["probes"][0]

    def probe_v(s, q):
        v = q[1][0]
        if s.stretched:  # v lives at the cell centres in x and the faces in y: linear between the true positions
            xs, ys = s._axis(0, False), s._axis(1, True)
            i, j = int(np.searchsorted(xs, px)) - 1, int(np.searchsorted(ys, py)) - 1
            a, b = (px - xs[i]) / (xs[i + 1] - xs[i]), (py - ys[j]) / (ys[j + 1] - ys[j])
            return (1 - a) * (1 - b) * v[j, i] + a * (1 - b) * v[j, i + 1] + (1 - a) * b * v[j + 1, i] + a * b * v[j + 1, i + 1]
        hx, hy = s.length[0] / s.n[0], s.length[1] / s.n[1]   # v lives at x = (i + 1/2) hx, y = j hy
        fx, fy = px / hx - 0.5, py / hy
        i, j = int(np.floor(fx)), int(np.floor(fy))
        a, b = fx - i, fy - j
        return (1 - a) * (1 - b) * v[j, i] + a * (1 - b) * v[j, i + 1] + (1 - a) * b * v[j + 1, i] + a * b * v[j + 1, i + 1]

    def observe(s):
        q = s.get()
        F = s.forces(box)
        return dict(Fcyl=F, cd=2 * F[0] / ((2 * g["Um"] / 3) ** 2 * g["D"]), cl=2 * F[1] / ((2 * g["Um"] / 3) ** 2 * g["D"]),
                    probe_v=probe_v(s, q), q_in=s.flux("x-"), q_out=s.flux("x+"),
                    umax=float(np.sqrt(np.max(q[0][0] ** 2) + np.max(q[1][0] ** 2))))
    return observe


def periodic_watch(numerics):
    """Live check: the case's periodicity rule over the last cycles_measured lift cycles."""
    acc, n = numerics["acceptance"], numerics["time"]["cycles_measured"]

    def watch(t, d):
        tt = d["t"]
        if tt[-1] - tt[0] < 1.5:
            return None
        cyc = S.cycles(tt, d["cd"], d["cl"], window=min(3.0, tt[-1] - tt[0]))
        if len(cyc) < n + 1:
            return None
        sp = S.summary(cyc, n)["spread"]
        ok = (max(sp["cl_max"], sp["cl_min"]) < acc["periodic_cl_peak"] and max(sp["cd_max"], sp["cd_min"]) < acc["periodic_cd_peak"]
              and sp["period"] / cyc[-1]["period"] < acc["periodic_period_rel"])
        return f"periodic (cl spread {max(sp['cl_max'], sp['cl_min']):.1e})" if ok else None
    return watch


def save_field(s, name):
    FIELDS.mkdir(parents=True, exist_ok=True)
    q = s.get()
    np.savez_compressed(FIELDS / f"{name}.npz", u=q[0], v=q[1])


def load_field(name, run, problem, numerics):
    """A simulation made for `run` (a row of the table) holding the saved last field of run `name`
    (the same run, or another one on the same grid)."""
    s, _ = make(run, problem, numerics)
    d = np.load(FIELDS / f"{name}.npz")
    s.set([d["u"], d["v"]])
    s.project()
    s._name = name
    return s


def one(name, run, problem, numerics, t0, t1, src=None, factor=None, ptol=None, runs_dir=RUNS_DIR, log=print,
        stop_after=None):
    """Make the run `run` of the table and record it as `name` (the same name, except for trial runs), from rest
    (src None) or from the last field of simulation src; returns (s, record).
    factor: time-step multiplier (default: the row's dt_factor).
    stop_after: end the run at the first periodicity check that passes at least this long after t0 (t1 is
    then the longest the run may go)."""
    from gpucfd import steady
    G = numerics["gpucfd"]
    ptol = G["pressure_tol"] if ptol is None else ptol
    spec = run_spec(run, numerics)
    factor = spec["dt_factor"] if factor is None else factor
    s, inflow = make(run, problem, numerics)
    div0 = from_rest(s, inflow) if src is None else steady.prolong(src, s)
    step = gpu.time_step(s, G["u_dt"], G["cfl"], factor)   # the step, both limits and which one set it (kept in the record)
    dt, h = step["dt"], step["h_min"]
    log(f"{name}: grid {s.n[0]} x {s.n[1]} ({int(np.prod(s.n))} cells), D/h {0.1 / h:.1f}, dt {dt:.3g} "
        f"(convective {step['convective']:.3g}, viscous {'none' if step['viscous'] is None else format(step['viscous'], '.3g')}; "
        f"set by the {step['set_by']} limit), t {t0:g} -> {t1:g}, start div {div0:.1e}", flush=True)
    watch = periodic_watch(numerics)
    if stop_after is not None:
        inner = watch

        def watch(t, d):
            return inner(t, d) if t - t0 >= stop_after - 1e-9 else None
    res = gpu.run_transient(s, dt, t0, t1, G["sample_dt"], observer(problem, numerics), u_ref=problem["given"]["Um"]["value"],
                            log=lambda m: print(m, flush=True), tag=f"{name} ", watch=watch,
                            ptol=ptol, cfl_limit=1.0, stop_on_watch=stop_after is not None,
                            partial=Path(runs_dir) / f"{name}.partial.npz")  # a killed run keeps its record
    rec = gpu.write_run(runs_dir, name, res, dict(level=spec["level"], variant=spec["variant"], n=list(s.n[:2]),
                                                  cells=int(np.prod(s.n)), h=h, D_h=0.1 / h,
                                                  source=getattr(src, "_name", None), dt_factor=factor, pressure_tol=ptol,
                                                  scheme=s.scheme, time_step=step, sample_dt=G["sample_dt"],
                                                  force_per_unit_depth=True))
    save_field(s, name)
    s._name = name
    p = rec["perf"]
    log(f"{name}: done, health {rec['health']['status']}, GPU {p['gpu_s']:.1f} s ({p['ms_per_step']:.2f} ms/step, "
        f"{p['steps']} steps, {p['poisson_its']:.1f} pressure iterations per step), observe {p['observe_s']:.1f} s, "
        f"wall {p['wall_s']:.1f} s; periodic from {rec['watch_first']['t'] if rec['watch_first'] else 'never'}", flush=True)
    return s, rec


def run_table(numerics):
    """name -> (source run, t0, t1) for the runs on equal cells: the same protocol as run.py's OpenFOAM levels
    (the first from rest for develop_end seconds, each other one run_length seconds from its source's end)."""
    t, table = numerics["time"], {}
    for nm in runs_where(numerics, family="uniform"):       # the table lists a source before the runs that start from it
        src = run_spec(nm, numerics)["source"]
        t0 = 0.0 if src is None else table[src][2]
        table[nm] = (src, t0, t0 + (t["develop_end"] if src is None else t["run_length"]))
    return table


def production(names):
    """Named runs in the order given; each starts from its source run's saved last field."""
    problem, numerics = load()
    table = run_table(numerics)
    for nm in names:
        src, t0, t1 = table[nm]
        source = load_field(src, src, problem, numerics) if src else None
        s, _ = one(nm, nm, problem, numerics, t0, t1, src=source)
        s.close()
        if source is not None:
            source.close()


def clustered(names):
    """The same protocol on the grids clustered around the cylinder (numerics.yaml, gpucfd.clustered): C1 from
    rest for develop_end seconds; each finer level from the saved last field of its source run (the table in
    numerics.yaml, gpucfd.runs), for at least run_length seconds and until the case's periodicity rule holds
    (checked live every second), at most max_length. One run does what a fixed run plus extend() did for G2 and G3."""
    problem, numerics = load()
    C, t = numerics["gpucfd"]["clustered"], numerics["time"]
    for nm in names:
        spec = run_spec(nm, numerics)
        if spec["family"] != "clustered":
            raise ValueError(f"{nm} is not a run on a clustered grid (use: production)")
        imex = spec["scheme"] == "imex"
        # per level and scheme (the tolerance rule applied on each level)
        tol = (C["imex"]["pressure_tol"] if imex else C.get("pressure_tol", {})).get(spec["level"])
        if imex and tol is None:
            raise ValueError(f"{nm}: no pressure tolerance yet (gpucfd.clustered.imex.pressure_tol): run tolcheck first")
        prev = spec["source"]
        if prev is None:
            s, _ = one(nm, nm, problem, numerics, 0.0, t["develop_end"], ptol=tol)
            s.close()
            continue
        t0 = json.loads((RUNS_DIR / f"{prev}.json").read_text())["end_time"]
        source = load_field(prev, prev, problem, numerics)
        s, _ = one(nm, nm, problem, numerics, t0, t0 + C["max_length"], src=source, stop_after=t["run_length"], ptol=tol)
        s.close()
        source.close()


def uniform_retol(names):
    """A uniform level again with the pressure tolerance its own rule gives (numerics.yaml,
    gpucfd.pressure_tol_level), from the coarser level's saved last field, stopping like the clustered levels
    (at least run_length, then the first periodicity check that passes). Overwrites runs/<name>: move the first
    record to gpu_tuning/ before calling this."""
    problem, numerics = load()
    G, t = numerics["gpucfd"], numerics["time"]
    for nm in names:
        prev = run_spec(nm, numerics)["source"]
        t0 = json.loads((RUNS_DIR / f"{prev}.json").read_text())["end_time"]
        source = load_field(prev, prev, problem, numerics)
        s, _ = one(nm, nm, problem, numerics, t0, t0 + G["clustered"]["max_length"], src=source, stop_after=t["run_length"],
                   ptol=G["pressure_tol_level"][nm])
        s.close()
        source.close()


def tolcheck(level, src_name, tols=(1e-10, 1e-9, 1e-8, 1e-7), length=0.35):
    """The pressure tolerance a run may use: the one way to check it (cfdlib.gpu.tolerance_check with the case's
    rule, numerics.yaml gpucfd.pressure_tol_rule: drag and lift coefficients within that of the strictest run, over
    `length` seconds, about one shedding cycle), from the saved last field of run src_name, which must be on the
    same grid. Writes gpu_tuning/tol_<run>.json. tolerance_record() reads it, and the files of the first checks.
    level: the run of the table to check (C3, C3i, ...)."""
    problem, numerics = load()
    G = numerics["gpucfd"]
    t0 = json.loads((RUNS_DIR / f"{src_name}.json").read_text())["end_time"]
    s, _ = make(level, problem, numerics)
    dt = s.max_dt(G["u_dt"], G["cfl"])
    s.close()
    res = gpu.tolerance_check(lambda: load_field(src_name, level, problem, numerics), dt, t0, length, G["sample_dt"],
                              observer(problem, numerics), problem["given"]["Um"]["value"], tols=tols, keys=["cd", "cl"],
                              rule=G["pressure_tol_rule"], log=lambda m: print(f"{level}: {m}", flush=True), cfl_limit=1.0)
    res.update(level=level, source=src_name, dt=dt, length=length)
    (OUT / "gpu_tuning" / f"tol_{level}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"{level}: chosen {res['chosen']:g}", flush=True)


def tolerance_record(run, numerics=None, tdir=None):
    """The stored pressure-tolerance check of a run, in the layout of cfdlib.gpu.tolerance_check, whichever
    command made it: gpu_tuning/tol_<run>.json (tolcheck) or, for the first checks of 2026-11-12 and 13,
    ptol_<run>.json (made by commands that tolcheck replaced; they store no rule, so the case's is applied)."""
    numerics = numerics or load()[1]
    tdir = Path(tdir or OUT / "gpu_tuning")
    for stem in (f"tol_{run}", f"ptol_{run}"):
        if (tdir / f"{stem}.json").exists():
            return gpu.read_tolerance_check(tdir / f"{stem}.json", rule=numerics["gpucfd"]["pressure_tol_rule"])
    raise FileNotFoundError(f"no pressure-tolerance check stored for run {run} in {tdir}")


def timing(level, steps=200):
    """Cost of a level before it is run: `steps` time steps from the coarser level's saved field (not recorded)."""
    from gpucfd import steady
    problem, numerics = load()
    G = numerics["gpucfd"]
    prev = run_spec(level, numerics)["source"]
    source = load_field(prev, prev, problem, numerics)
    s, _ = make(level, problem, numerics)
    steady.prolong(source, s)
    dt = s.max_dt(G["u_dt"], G["cfl"])
    s.run(dt, 20, G["pressure_tol"])
    st = s.run(dt, steps, G["pressure_tol"])
    print(f"{level}: {s.n[0]} x {s.n[1]}, dt {dt:.4g}, {st['ms'] / steps:.2f} ms per step, {st['poisson_its']:.1f} pressure "
          f"iterations per projection; 1 s of flow = {1 / dt:.0f} steps = {st['ms'] / steps / dt / 1000:.0f} GPU-s", flush=True)
    s.close()
    source.close()


def extend(name, max_length=12.0):
    """Continue run `name` from its saved last field, with the same time step, until the live periodicity check
    passes (at most max_length seconds more), and merge the continuation into its records: the same result as
    one longer run. Needed when the coarser start field is far from the finer level's answer (G1's drag is 16 %
    low), so the fixed 3 s of the OpenFOAM protocol do not reach the periodic state."""
    problem, numerics = load()
    G = numerics["gpucfd"]
    rec0 = json.loads((RUNS_DIR / f"{name}.json").read_text())
    d0 = dict(np.load(RUNS_DIR / f"{name}.npz"))
    s = load_field(name, name, problem, numerics)
    dt0, t0 = rec0["dt"], rec0["end_time"]
    n = int(round(max_length / dt0))
    print(f"{name}: continuing from t {t0:g} with dt {dt0:.4g}, until periodic (at most {max_length:g} s)", flush=True)
    res = gpu.run_transient(s, dt0, t0, t0 + n * dt0, G["sample_dt"], observer(problem, numerics),
                            u_ref=problem["given"]["Um"]["value"], log=lambda m: print(m, flush=True), tag=f"{name}+ ",
                            watch=periodic_watch(numerics), stop_on_watch=True, ptol=rec0["pressure_tol"], cfl_limit=1.0)
    p0, p1 = rec0["perf"], res["perf"]
    steps = p0["steps"] + p1["steps"]
    perf = gpu.perf_record(p0["gpu_s"] + p1["gpu_s"], p0["wall_s"] + p1["wall_s"], p0["observe_s"] + p1["observe_s"], steps,
                           p0["cells"], (p0["poisson_its"] * p0["steps"] + p1["poisson_its"] * p1["steps"]) / steps,
                           p0["sample_steps"])
    checks = []
    for c0, c1 in zip(rec0["health"]["checks"], res["health"]["checks"]):
        worse = c1 if ["ok", "warn", "fail"].index(c1["status"]) > ["ok", "warn", "fail"].index(c0["status"]) else c0
        checks.append(dict(worse, value=f"{c0['value']} / {c1['value']}"))
    status = max((c["status"] for c in checks), key=["ok", "warn", "fail"].index)
    merged = dict(t=np.concatenate([d0["t"], res["t"][1:]]),
                  series={k: np.concatenate([d0[k], res["series"][k][1:]]) for k in res["series"]},
                  perf=perf, health=dict(status=status, checks=checks, steps=steps, last_time=float(res["t"][-1])),
                  dt=dt0, watch_first=res["watch_first"])
    meta = {k: rec0[k] for k in ("level", "variant", "n", "cells", "h", "D_h", "source", "dt_factor", "pressure_tol", "scheme",
                                 "time_step", "sample_dt", "force_per_unit_depth") if k in rec0}
    meta["segments"] = rec0.get("segments", [[rec0["start_time"], t0]]) + [[t0, float(res["t"][-1])]]
    rec = gpu.write_run(RUNS_DIR, name, merged, meta)
    save_field(s, name)
    s.close()
    print(f"{name}: now t {rec['start_time']:g} -> {rec['end_time']:g}, GPU {perf['gpu_s']:.1f} s, wall {perf['wall_s']:.1f} s, "
          f"health {status}, periodic at {res['watch_first']['t'] if res['watch_first'] else 'never'}", flush=True)


def preflight():
    """G1 from rest for 0.5 s and G3 from G1 for 0.05 s: stability, outputs, cost per step and per sample."""
    problem, numerics = load()
    G = numerics["gpucfd"]
    out = []
    s1, r1 = one("pf_G1", G["levels"][0], problem, numerics, 0.0, 0.5)      # trial runs: the grids of G1 and G3, own names
    s3, r3 = one("pf_G3", G["levels"][-1], problem, numerics, 0.5, 0.55, src=s1)
    for r in (r1, r3):
        p = r["perf"]
        n_samp = len(np.load(RUNS_DIR / f"{r['name']}.npz")["t"])
        r["observe_ms_per_sample"] = 1000 * p["observe_s"] / n_samp
        out.append(r)
        print(f"{r['name']}: {r['health']['status']}, {p['ms_per_step']:.2f} ms/step, {r['observe_ms_per_sample']:.1f} ms per sample, "
              + "; ".join(f"{c['name']}: {c['value']}" for c in r["health"]["checks"]))
    for nm in ("pf_G1", "pf_G3"):  # pre-flight records are not runs of the study
        for ext in ("json", "npz"):
            (RUNS_DIR / f"{nm}.{ext}").unlink()
    (OUT / "gpu_preflight.json").write_text(json.dumps(out, indent=1, default=float))
    s1.close()
    s3.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    if mode == "production":
        production(sys.argv[2:] or list(run_table(load()[1])))
    elif mode == "extend":
        for nm in sys.argv[2:]:
            extend(nm)
    elif mode == "tolcheck":
        extra = [float(x) for x in sys.argv[4:]]
        tolcheck(sys.argv[2], sys.argv[3], **(dict(length=extra[0]) if extra else {}), **(dict(tols=extra[1:]) if extra[1:] else {}))
    elif mode == "preflight":
        preflight()
    elif mode == "clustered":
        clustered(sys.argv[2:] or load()[1]["gpucfd"]["clustered"]["levels"])
    elif mode == "timing":
        timing(sys.argv[2])
    elif mode == "uniform_retol":
        uniform_retol(sys.argv[2:])
    else:
        raise SystemExit(f"unknown mode {mode}")
