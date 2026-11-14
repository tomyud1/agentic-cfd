"""OpenFOAM on case 17's turbulent channel, for the comparison with gpucfd (plan: plan.yaml next to this file).

    PYTHONPATH=. python dev/compare/turbulent_channel_re180/openfoam.py build     # the case, from gpucfd's last field
    PYTHONPATH=. python dev/compare/turbulent_channel_re180/openfoam.py tune      # the timed tries of plan.yaml
    PYTHONPATH=. python dev/compare/turbulent_channel_re180/openfoam.py run       # start-up and averaging
    PYTHONPATH=. python dev/compare/turbulent_channel_re180/openfoam.py extend 10 # that many time units more
    PYTHONPATH=. python dev/compare/turbulent_channel_re180/openfoam.py stats     # the record and its assessment
    PYTHONPATH=. python dev/compare/turbulent_channel_re180/openfoam.py time_new  # cost per step on the new grid
    (inside WSL from the repository root; needs OpenFOAM; `build` and `time_new` also need the GPU library, to
    read gpucfd's field with its own coordinates)

Same cells, same time step, same driving force, no model. The case is built by the exploration's builder
(dev/exploration/les_channel/openfoam_time.py), which was checked there; this script adds the statistics:
bulk velocity and wall shear every few steps (function objects), and the time-averaged velocity and its
second moments in every cell (fieldAverage), averaged over the planes here.
The profile gaps and outputs are formed as for gpucfd (dev/optimize/turbulent_channel_re180/stage.py, the
case's stats.py): same reference, same limits. One difference of kind: OpenFOAM keeps all values at the
cell centres, gpucfd keeps the wall-normal velocity on faces; both are compared with the reference at the
heights where they hold their values.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "dev" / "exploration" / "les_channel"))
sys.path.insert(0, str(ROOT / "dev" / "optimize" / "turbulent_channel_re180"))
import openfoam_time as OT  # noqa: E402
import stage as S  # noqa: E402
from cases.turbulent_channel_re180 import stats as ST  # noqa: E402
from cfdlib import foam, monitor, signals  # noqa: E402

PLAN = yaml.safe_load((HERE / "plan.yaml").read_text(encoding="utf-8"))
RUNS = HERE / "runs"
G3 = json.loads((ROOT / "results" / "turbulent_channel_re180" / "runs" / "G3.json").read_text(encoding="utf-8"))
WRITE = 6.0                                    # time units between stored states
DT = WRITE / round(WRITE / float(G3["dt"]))    # gpucfd's mean step, moved by less than 1e-4 of itself so that the stored
                                               # states (and the end of the run) fall on whole steps
QS = S.QS
HEAD = OT.HEAD

SOLVERS = dict(
    tutorial="""
    p { solver GAMG; tolerance 0; relTol 0.1; smoother GaussSeidel; }
    pFinal { solver GAMG; smoother DICGaussSeidel; tolerance 1e-06; relTol 0; }""",
    pcg="""
    p { solver PCG; preconditioner DIC; tolerance 0; relTol 0.1; }
    pFinal { solver PCG; preconditioner DIC; tolerance 1e-06; relTol 0; }""",
    gamg_dic="""
    p { solver GAMG; tolerance 0; relTol 0.1; smoother DICGaussSeidel; cacheAgglomeration true; nCellsInCoarsestLevel 100; }
    pFinal { solver GAMG; smoother DICGaussSeidel; tolerance 1e-06; relTol 0; cacheAgglomeration true; nCellsInCoarsestLevel 100; }""",
)


def case_dir(tag="G3"):
    return Path(foam.foam_env("FOAM_RUN")) / "agentic" / "turbulent_channel_re180" / f"of_{tag}"


def fv_solution(case, name):
    (case / "system/fvSolution").write_text(HEAD.format(cls="dictionary", obj="fvSolution") + "\nsolvers\n{" + SOLVERS[name] + """
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0.1; }
    UFinal { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0; }
}
PIMPLE { nOuterCorrectors 1; nCorrectors 2; nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }
""")


def control_dict(case, t_end, average_from, write=WRITE):
    n = int(PLAN["run"]["sample_steps"])
    every = f"executeControl timeStep; executeInterval {n}; writeControl timeStep; writeInterval {n};"
    (case / "system/controlDict").write_text(HEAD.format(cls="dictionary", obj="controlDict") + f"""
application pimpleFoam; startFrom latestTime; startTime 0; stopAt endTime; endTime {t_end!r}; deltaT {DT!r};
writeControl runTime; writeInterval {write!r}; purgeWrite 2; writeFormat ascii; writePrecision 12; timeFormat general; timePrecision 10;
runTimeModifiable true;
functions
{{
    wallShearStress {{ type wallShearStress; libs (fieldFunctionObjects); patches (bottomWall topWall);
                      executeControl timeStep; executeInterval {n}; writeControl writeTime; log false; }}
    tauLow {{ type surfaceFieldValue; libs (fieldFunctionObjects); regionType patch; name bottomWall; operation areaAverage;
             fields (wallShearStress); writeFields false; log false; {every} }}
    tauHigh {{ type surfaceFieldValue; libs (fieldFunctionObjects); regionType patch; name topWall; operation areaAverage;
              fields (wallShearStress); writeFields false; log false; {every} }}
    bulk {{ type volFieldValue; libs (fieldFunctionObjects); regionType all; operation volAverage; fields (U);
           writeFields false; log false; {every} }}
    average {{ type fieldAverage; libs (fieldFunctionObjects); timeStart {average_from!r}; writeControl writeTime; log false;
              fields ( U {{ mean on; prime2Mean on; base time; }} ); }}
}}
""")


def gpucfd_field(spec, path):
    """gpucfd's stored field (u, v, w on their faces) with the coordinates of those faces, as the builder wants it."""
    g = S.setup()[0]
    s = S.make(spec, g)
    try:
        f = np.load(path)
        src = {f"q{c}": f[k] for c, k in enumerate(("u", "v", "w"))} if "u" in f.files else {f"q{c}": f[f"q{c}"] for c in range(3)}
        src.update({f"x{c}{d}": s._axis(d, d == c) for c in range(3) for d in range(3)})
        y = np.concatenate((s._axis(1, True), [s.length[1]]))
    finally:
        s.close()
    return src, y


def do_build(tag="G3", spec=None, field=None, steps=60):
    spec = spec or PLAN["grid"]
    field = field or Path.home() / "gpucfd_runs" / "turbulent_channel_re180" / "G3.npz"
    src, y = gpucfd_field(spec, field)
    case = case_dir(tag)
    bulk = OT.build(case, tuple(spec["n"]), y, DT, steps, src)
    print(f"built {case}: {spec['n']}, time step {DT:.4e} (gpucfd's mean step on 192 x 64 x 64), bulk velocity of the start field {bulk:.4f}")
    return case


def do_tune():
    case = case_dir()
    T = PLAN["tuning"]
    rows = {}
    for name in T["tries"]:
        fv_solution(case, name)
        r = OT.timed(case, T["cores"], T["steps"])
        rows[name] = r
        print(f"  {name}: {1e3 * r['s_per_step']:.0f} ms per step on {T['cores']} cores, largest Courant number {r['courant_max']:.2f}", flush=True)
    best = min(rows, key=lambda k: rows[k]["s_per_step"])
    RUNS.mkdir(exist_ok=True)
    (RUNS / "tune.json").write_text(json.dumps(dict(rows=rows, best=best, dt=DT, cores=T["cores"]), indent=1))
    print(f"fastest: {best}")


def series(case):
    """Bulk velocity and the two wall shears against time, from the function objects (all restarts joined)."""
    def read(name, file):
        rows = []
        for d in sorted((case / "postProcessing" / name).iterdir(), key=lambda p: float(p.name)):
            rows += [(r[0], r[1][0]) for r in foam.read_dat(d / file) if len(r) > 1 and isinstance(r[1], tuple)]
        a = np.array(sorted(dict(rows).items()))                 # a restart repeats times: keep the later value
        return a[:, 0], a[:, 1]
    t, ub = read("bulk", "volFieldValue.dat")
    t1, lo = read("tauLow", "surfaceFieldValue.dat")
    t2, hi = read("tauHigh", "surfaceFieldValue.dat")
    n = min(len(t), len(t1), len(t2))
    assert np.allclose(t[:n], t1[:n]) and np.allclose(t[:n], t2[:n])
    return dict(t=t[:n], bulk=ub[:n], tau_low=np.abs(lo[:n]), tau_high=np.abs(hi[:n]))


def progress(case, log):
    def watch():
        try:
            d = series(case)
            k = d["t"] >= d["t"][-1] - 2.0
            log(f"t {d['t'][-1]:.2f}: bulk {d['bulk'][-1]:.4f}, wall shear over the last 2 units {0.5 * np.mean(d['tau_low'][k] + d['tau_high'][k]):.4f}")
            late = d["t"] >= d["t"][-1] - 5.0
            if d["t"][-1] > 20.0 and 0.5 * np.mean(d["tau_low"][late] + d["tau_high"][late]) < 0.6:
                log("the flow is going laminar (wall shear below 0.6 of the driving force over 5 time units): stopping")
                foam.stop(case)
        except Exception as e:                                   # noqa: BLE001  (no data yet, or a file being written)
            log(f"no progress line: {type(e).__name__}: {e}")
    return watch


def logger():
    RUNS.mkdir(exist_ok=True)
    fh = open(RUNS / "openfoam_G3.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    return log


def do_run(extend=None, test=False):
    """The full run of the plan; extend: so many time units more from the last state; test: 30 steps with the
    averaging switched on after 10, to see that the statistics are written and read (its files are removed
    by the full run)."""
    import shutil
    case = case_dir()
    log = logger() if not test else print
    R = PLAN["run"] if not test else dict(startup=10 * DT, average=20 * DT)
    cores = PLAN["tuning"]["cores"]
    best = json.loads((RUNS / "tune.json").read_text())["best"]
    if extend is None:
        fv_solution(case, best)
        control_dict(case, R["startup"] + R["average"], R["startup"], WRITE if not test else 10 * DT)
        (case / "system/controlDict").write_text((case / "system/controlDict").read_text().replace("startFrom latestTime", "startFrom startTime"))
        shutil.rmtree(case / "postProcessing", ignore_errors=True)
        for old in foam.time_dirs(case):
            if float(old.name) > 0:
                shutil.rmtree(old)
        foam.run("decomposePar -force", case, "decomposePar")
        t_end = R["startup"] + R["average"]
    else:
        last = max(float(d.name) for d in foam.time_dirs(case / "processor0"))
        t_end = last + float(extend)
        control_dict(case, t_end, R["startup"])
    log(f"OpenFOAM run to t = {t_end:g} on {cores} cores, pressure solver '{best}', time step {DT:.4e}")
    wall = foam.run(f"mpirun -np {cores} pimpleFoam -parallel", case, "pimpleFoam" if extend is None else f"pimpleFoam_ext{int(t_end)}",
                    timeout=12 * 3600, watch=progress(case, log), watch_every=900.0)
    log(f"RUN DONE: wall time {wall / 3600:.2f} h")
    if test:
        d = series(case)
        foam.run("reconstructPar -latestTime -fields '(UMean UPrime2Mean)'", case, "reconstructPar")
        last = foam.time_dirs(case)[-1]
        um, r2 = foam.read_internal_field(last / "UMean"), foam.read_internal_field(last / "UPrime2Mean")
        print(f"test: {len(d['t'])} samples, t {d['t'][0]:.5f} to {d['t'][-1]:.5f}, bulk {d['bulk'][-1]:.4f}, wall shear low {d['tau_low'][-1]:.4f} "
              f"high {d['tau_high'][-1]:.4f}; averaged fields at t = {last.name}: UMean {um.shape}, mean of its x component {um[:, 0].mean():.4f}, "
              f"UPrime2Mean {r2.shape}, largest xx {r2[:, 0].max():.3g}")


def planes(a, n):
    """A cell field (cell number = i + nx (j + ny k)) as (nz, ny, nx)."""
    return a.reshape(n[2], n[1], n[0])


def do_stats():
    g, ref, limits, numerics = S.setup()
    acc, Rf = numerics["acceptance"], ref["scalars"]
    case = case_dir()
    n = PLAN["grid"]["n"]
    R = PLAN["run"]
    d = series(case)
    foam.run("reconstructPar -latestTime -fields '(UMean UPrime2Mean)'", case, "reconstructPar")
    last = foam.time_dirs(case)[-1]
    t_end = float(last.name)
    t0 = R["startup"]
    um = foam.read_internal_field(last / "UMean")
    r2 = foam.read_internal_field(last / "UPrime2Mean")            # xx xy xz yy yz zz, about each cell's own time mean
    U = [planes(um[:, c], n) for c in range(3)]
    Up = [u.mean(axis=(0, 2), keepdims=True) for u in U]

    def second(i, j, col):                                         # about the plane's mean: plus the scatter of the cells' time means
        return (planes(r2[:, col], n) + (U[i] - Up[i]) * (U[j] - Up[j])).mean(axis=(0, 2))
    prof = dict(U=Up[0][0, :, 0], uu=second(0, 0, 0), vv=second(1, 1, 3), ww=second(2, 2, 5), uv=second(0, 1, 1))
    src, yf = gpucfd_field(PLAN["grid"], Path.home() / "gpucfd_runs" / "turbulent_channel_re180" / "G3.npz")
    y = 0.5 * (yf[:-1] + yf[1:])
    ut, nu, h = g["u_tau"], g["nu"], n[1] // 2
    yp = y[:h] * ut / nu
    fold = dict(U=ST.fold_c(prof["U"][None])[0], u_rms=np.sqrt(ST.fold_c(prof["uu"][None])[0]), v_rms=np.sqrt(ST.fold_c(prof["vv"][None])[0]),
                w_rms=np.sqrt(ST.fold_c(prof["ww"][None])[0]),
                uv=0.5 * (-prof["uv"][:h] + prof["uv"][::-1][:h]))  # the turbulent shear stress -u'v', odd about the centre plane
    G = {}
    for q in QS:
        sim = fold[q] / (ut ** 2 if q == "uv" else ut)
        rf = ref["at"](q, yp)
        ok = np.isfinite(rf)
        diff = sim[ok] - rf[ok]
        k = int(np.argmax(np.abs(diff)))
        G[q] = dict(signed=float(diff[k]), y_plus=float(yp[ok][k]), y=[float(v) for v in yp[ok]], diff=[float(v) for v in diff])
    from gpucfd import turbulence
    B = turbulence.bulk_average(d["t"], d["bulk"], 0.5 * (d["tau_low"] + d["tau_high"]), g["f"], g["delta"], t_start=t0)
    keep = d["t"] >= t0
    imbalance = float(signals.stationary(d["t"][keep], 0.5 * (d["tau_low"] + d["tau_high"])[keep])["mean"] / (g["f"] * g["delta"]) - 1.0)
    ub, ue = float(B["mean"]), float(B["stderr"])
    d1, d2 = g["delta"] - y[h - 1], g["delta"] - y[h - 2]
    centre = float((fold["U"][-1] * d2 ** 2 - fold["U"][-2] * d1 ** 2) / (d2 ** 2 - d1 ** 2) / ut)
    pk_y, pk = (float(v) for v in ST.peak(yp, fold["u_rms"] / ut))
    cf = 2.0 * ut ** 2 / ub ** 2
    checks = dict(bulk=(100 * (ub / Rf["bulk"]["value"] - 1), acc["bulk_pct"], "%"),
                  centre=(100 * (centre / Rf["centre"]["value"] - 1), acc["centre_pct"], "%"),
                  cf=(100 * (cf / Rf["cf"]["value"] - 1), acc["cf_pct"], "%"),
                  urms_peak=(100 * (pk / Rf["urms_peak"]["value"] - 1), acc["urms_peak_pct"], "%"),
                  urms_peak_y=(pk_y - Rf["urms_peak_y"]["value"], acc["urms_peak_yplus"], " wall units"))
    for q in QS:
        checks[q] = (G[q]["signed"], limits[q], f"@{G[q]['y_plus']:.0f}")
    checks = {k: dict(value=float(v[0]), limit=float(v[1]), ok=bool(abs(v[0]) <= v[1]), unit=v[2]) for k, v in checks.items()}
    logs = sorted(case.glob("log.pimpleFoam*"))
    steps, exec_s, co = 0, 0.0, 0.0
    for lg in logs:
        arr = monitor.log_arrays(monitor.parse_log(lg)["steps"])
        steps += len(arr["exec"])
        exec_s += float(arr["exec"][-1])
        co = max(co, float(np.nanmax(arr["co"])))
    tune = json.loads((RUNS / "tune.json").read_text())
    rec = dict(tool="OpenFOAM v" + foam.foam_env("WM_PROJECT_VERSION"), solver="pimpleFoam, no model, backward, linear, PISO with 2 pressure corrections",
               pressure_solver=tune["best"], cores=tune["cores"], n=list(n), cells=int(np.prod(n)), dt=DT, t_start=t0, t_end=t_end,
               span=t_end - t0, steps=steps, wall_s=exec_s, ms_per_step=1e3 * exec_s / steps, s_per_unit=exec_s / t_end, courant_max=co,
               bulk=ub, bulk_stderr=ue, bulk_reliable=bool(B.get("reliable", False)), imbalance=imbalance, centre=centre, cf=cf,
               urms_peak=pk, urms_peak_y=pk_y, checks=checks, all_ok=all(c["ok"] for c in checks.values()), gaps=G,
               usable=bool(abs(imbalance) <= PLAN["usable"]["imbalance"] and ue / ub <= PLAN["usable"]["bulk_stderr"]))
    RUNS.mkdir(exist_ok=True)
    (RUNS / "openfoam_G3.json").write_text(json.dumps(rec, indent=1, default=float))
    np.savez(RUNS / "openfoam_G3.npz", y=y, **{f"p_{k}": v for k, v in prof.items()}, **d)
    print(f"OpenFOAM {n}: averaged t = {t0:g} to {t_end:g}, bulk {ub:.4f} +- {ue:.4f}, wall shear {imbalance:+.2%} from the driving force, "
          f"{rec['ms_per_step']:.0f} ms per step, {rec['s_per_unit']:.0f} s per time unit, {exec_s / 3600:.2f} h on {tune['cores']} cores, "
          f"largest Courant number {co:.2f}")
    print("   " + "  ".join(f"{k} {c['value']:+.3f}{c['unit']} (limit {c['limit']:g})" + ("" if c["ok"] else "*") for k, c in checks.items()))
    print(f"   usable by the plan: {rec['usable']}")


def do_time_new():
    N = PLAN["new_grid"]
    case = do_build("new", dict(n=N["n"], wall_cell_plus=N["wall_cell_plus"]),
                    Path.home() / "gpucfd_runs" / "turbulent_channel_re180" / "opt_x15z15.npz", N["timed_steps"])
    best = json.loads((RUNS / "tune.json").read_text())["best"]
    fv_solution(case, best)
    r = OT.timed(case, PLAN["tuning"]["cores"], N["timed_steps"])
    (RUNS / "time_new.json").write_text(json.dumps(dict(n=N["n"], dt=DT, pressure_solver=best, **r), indent=1))
    print(f"new grid {N['n']}: {1e3 * r['s_per_step']:.0f} ms per step on {r['cores']} cores, largest Courant number {r['courant_max']:.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        do_build()
    elif cmd == "tune":
        do_tune()
    elif cmd == "run":
        do_run()
    elif cmd == "test":
        do_run(test=True)
    elif cmd == "extend":
        do_run(extend=float(sys.argv[2]))
    elif cmd == "stats":
        do_stats()
    elif cmd == "time_new":
        do_time_new()
    else:
        sys.exit(__doc__)
