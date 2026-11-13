"""Build and run the turbulent flat plate (Spalart-Allmaras) on NASA's grids.

Usage (repository root, inside WSL):
  python -m cases.turbulent_flat_plate_sa.run preflight             # G1 to convergence + a short run on G5
  python -m cases.turbulent_flat_plate_sa.run production [G1 ...]   # live-stopped runs, coarse to fine, grid sequencing
  python -m cases.turbulent_flat_plate_sa.run trials [name ..]      # relaxation trials on G1 (the period-2 cycle)
  python -m cases.turbulent_flat_plate_sa.run sensitivity [name ..] # the variants of numerics.yaml
Outputs: results/turbulent_flat_plate_sa/preflight.json, runs/<name>.json + runs/<name>.npz, runs/DONE_<mode>
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from cfdlib import foam, monitor, perf, structgrid

from .common import HERE, OUT, RUNS, given, load, read_grid


def run_root():
    return Path(foam.foam_env("FOAM_RUN")) / "agentic" / HERE.name


# ------------------------------------------------------------------------------------------------
# Case
# ------------------------------------------------------------------------------------------------
def grid_info(i):
    """Geometry of level i: points, cell centres, plate range and the probe cells."""
    x, y = read_grid(i)
    xs, ys = x[0], y[:, 0]
    xc, yc = 0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1])
    plate = structgrid.index_range(xs, lo=0.0)
    return dict(x=x, y=y, xs=xs, ys=ys, xc=xc, yc=yc, plate=plate)


def params(problem, num, g, **override):
    G = given(problem)
    c, s, r, l = num["convergence"], num["schemes"], num["relaxation"], num["linear_solvers"]
    pr = num["probes"]
    cols = [int(np.searchsorted(g["xs"], xs)) for xs in pr["stations"]]        # the column just downstream of the station
    cols = [min(k, len(g["xc"]) - 1) for k in cols]
    wall = [(f"{g['xc'][k]:.12g}", f"{g['yc'][0]:.12g}") for k in cols]
    outer = [(f"{g['xc'][k]:.12g}", f"{g['yc'][int(np.argmin(np.abs(g['yc'] - yo)))]:.12g}")
             for k, yo in zip(cols, pr["outer_y"])]
    p = dict(U_inf=G["U_inf"], nu=G["nu"], rho=G["rho"], nuTilda_inf=f"{G['nut_in'] * G['nu']:.12g}",
             plate_length=G["x_out"], ft2=num["model"]["ft2"], sigmaNut=num["model"]["sigmaNut"], Cs=num["model"]["Cs"],
             div_U=s["div_U"], div_nuTilda=s["div_nuTilda"], **num["far_field"],
             consistent=r["consistent"], relax_U=r["U"], relax_nuTilda=r["nuTilda"], relax_p=r["p"],
             p_solver=l["p_solver"], p_tol=l["p_tol"], p_relTol=l["p_relTol"], p_maxIter=l["p_maxIter"],
             u_tol=l["u_tol"], u_relTol=l["u_relTol"],
             max_iterations=c["max_iterations"], write_every=c["write_every"],
             write_precision=num["mesh"]["write_precision"], probes_wall=wall, probes_outer=outer)
    p.update(override)
    return p


def build(name, problem, num, i, source=None, cores=1, **override):
    """Render the case of level i, write NASA's grid as its mesh, start from `source` (a coarser or same-grid case)."""
    g = grid_info(i)
    par = params(problem, num, g, **override)
    case = foam.render_case(HERE / "template", run_root() / name, par)
    info = structgrid.write_polymesh(case, g["x"], g["y"], {
        "inlet": ("i-", None), "outlet": ("i+", None), "top": ("j+", None),
        "symmetry": ("j-", (0, g["plate"][0])), "plate": ("j-", g["plate"])},
        types={"plate": "wall", "symmetry": "symmetryPlane"})
    assert info["digits_needed"] <= num["mesh"]["write_precision"], info
    foam.run("checkMesh", case)
    if source is not None:
        foam.map_fields(case, source)
        # mapFields copies the source's boundary values onto this case's fixed-value patches, so a variant that
        # changes an inflow value would silently run with the old one (insight): prescribe them again.
        foam.run(f"foamDictionary 0/U -entry boundaryField/inlet/value -set 'uniform ({par['U_inf']} 0 0)'", case, "setInletU")
        foam.run(f"foamDictionary 0/nuTilda -entry boundaryField/inlet/value -set 'uniform {par['nuTilda_inf']}'", case, "setInletNuTilda")
    if cores > 1:
        (case / "system" / "decomposeParDict").write_text(
            "FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\n"
            f"numberOfSubdomains {cores};\nmethod simple;\ncoeffs {{ n ({cores} 1 1); }}\n")
        foam.run("decomposePar -force", case, "decomposePar")
    return case, g, info


def solve(case, num, cores, decide=None, timeout=None, watch_every=5.0):
    live = monitor.LiveStop(case, decide) if decide else None
    cmd = num["solver"] if cores == 1 else f"mpirun -np {cores} {num['solver']} -parallel"
    wall = foam.run(cmd, case, log_name=num["solver"], watch=live, watch_every=watch_every, timeout=timeout)
    if cores > 1:
        foam.run("reconstructPar -latestTime", case, "reconstructPar")
    return wall, live


# ------------------------------------------------------------------------------------------------
# Live stop
# ------------------------------------------------------------------------------------------------
def monitored(case, tails):
    """The monitored values of a run so far: (iterations, array (n, 5), names). Read incrementally."""
    key = str(case)
    t = tails.setdefault(key, dict(cd=monitor.DatTail(case, "forces", "coefficient.dat"),
                                   u=monitor.DatTail(case, "probesU", "U"),
                                   nt=monitor.DatTail(case, "probesNut", "nuTilda")))
    cd, u, nt = t["cd"](), t["u"](), t["nt"]()
    n = min(len(cd), len(u), len(nt))
    if n == 0:
        return np.empty(0), np.empty((0, 5)), []
    v = np.column_stack([cd[:n, 1], u[:n, 1], u[:n, 4], nt[:n, 1], nt[:n, 2]])   # Cd is the first coefficient column
    return cd[:n, 0], v, ["Cd", "u_wall(x1)", "u_wall(x2)", "nuTilda(x1)", "nuTilda(x2)"]


def tolerances(num, v):
    t = num["convergence"]["tol_rel"]
    rel = np.array([t["cd"], t["u_wall"], t["u_wall"], t["nuTilda"], t["nuTilda"]])
    return rel * np.abs(v[-1])


def live_decider(num):
    """Stop once the estimated iteration error of every monitored value is below its relative tolerance."""
    c = num["convergence"]
    tails, streak = {}, {}

    def decide(case):
        it, v, names = monitored(case, tails)
        if len(v) < 6 * c["every"]:
            return None
        tol = tolerances(num, v)
        r = monitor.steady_check(v, tol, c["every"], names)
        # period-2 guard: the largest change between consecutive iterations over the last 2 x every iterations
        alt = np.abs(np.diff(v[-2 * c["every"] - 1:], axis=0)).max(axis=0)
        flip = [n for n, a, t in zip(names, alt, tol) if a > t]
        if flip:
            r.update(ok=False, reason=f"changing between consecutive iterations: {', '.join(flip)} "
                                      f"(largest {float((alt / tol).max()):.3g} x tolerance)")
        r["alternation"] = dict(zip(names, (alt / np.abs(v[-1])).tolist()))
        key = str(case)
        streak[key] = streak.get(key, 0) + 1 if r["ok"] else 0
        r.update(stop=streak[key] >= c["hold"], t=float(it[-1]))
        return r
    return decide


# ------------------------------------------------------------------------------------------------
# Reading results
# ------------------------------------------------------------------------------------------------
def model_coeffs(log_text):
    """The model coefficients the solver printed at start-up (printCoeffs on), and whether f_t2 is active."""
    m = re.search(r"SpalartAllmarasCoeffs\s*\{([^}]*)\}", log_text)
    coeffs = {}
    if m:
        for k, v in re.findall(r"(\w+)\s+([^;\s]+);", m.group(1)):
            coeffs[k] = v
    ft2 = re.search(r"ft2 term: (\w+)", log_text)
    return coeffs, (ft2.group(1) if ft2 else None)


def collect(case, name, G, num, g, wall, cores, live, level, variant=None):
    """Post-process one finished run into runs/<name>.json and .npz (everything the analysis needs)."""
    case = Path(case)
    t = foam.time_dirs(case)[-1]
    ni, nj = len(g["xc"]), len(g["yc"])
    sh = (nj, ni)
    U = foam.read_internal_field(t / "U")
    arr = dict(xs=g["xs"], ys=g["ys"], ux=U[:, 0].reshape(sh), uy=U[:, 1].reshape(sh),
               p=foam.read_internal_field(t / "p").reshape(sh),
               nut=foam.read_internal_field(t / "nut").reshape(sh),
               nuTilda=foam.read_internal_field(t / "nuTilda").reshape(sh))
    i0 = g["plate"][0]
    tw = np.atleast_2d(foam.read_boundary_field(t / "wallShearStress", "plate"))
    arr["tau_x"] = -tw[:, 0]                                  # the function object gives the stress on the wall with a minus sign
    arr["yplus"] = np.broadcast_to(foam.read_boundary_field(t / "yPlus", "plate"), (ni - i0,)).copy()
    # boundary fluxes and values for the mass and momentum balance (the solver's own face fluxes)
    n_face = dict(inlet=nj, outlet=nj, top=ni)
    for patch, n in n_face.items():
        arr[f"phi_{patch}"] = np.broadcast_to(np.asarray(foam.read_boundary_field(t / "phi", patch), float), (n,)).copy()
    arr["u_outlet"] = np.broadcast_to(np.atleast_2d(foam.read_boundary_field(t / "U", "outlet")), (nj, 3))[:, 0].copy()
    try:
        arr["u_top"] = np.broadcast_to(np.atleast_2d(foam.read_boundary_field(t / "U", "top")), (ni, 3))[:, 0].copy()
    except ValueError:                                        # a patch type that stores no value (the lid variant): no flux there
        arr["u_top"] = arr["ux"][-1].copy()
    # histories
    it, v, names = monitored(case, {})
    arr["hist_it"], arr["hist"] = it, v
    log = case / f"log.{num['solver']}"
    text = log.read_text(errors="replace")
    res = foam.residual_history(log, ("Ux", "Uy", "p", "nuTilda"))
    for k, r in res.items():
        arr[f"res_{k}"] = np.array(r, float)
    coeffs, ft2 = model_coeffs(text)
    cd_rows = monitor.read_dat_live(case, "forces", "coefficient.dat")
    out = dict(name=name, level=level, variant=variant, case_dir=str(case), cells=ni * nj, points=[ni + 1, nj + 1],
               cores=cores, wall_time=wall, iterations=int(float(t.name)), cd_forces=float(cd_rows[-1][1]),
               live_stop=live.summary() if live else None, monitored=names,
               health=monitor.assess(log), mesh_ok="Mesh OK." in (case / "log.checkMesh").read_text(),
               perf=perf.summarize(case, num["solver"], ni * nj, cores), model_coeffs=coeffs, ft2=ft2,
               final_residuals={k: (r[-1] if r else None) for k, r in res.items()},
               bounding=len(re.findall(r"bounding nuTilda", text)))
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{name}.npz", **arr)
    (RUNS / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def report_line(rec):
    d = (rec.get("live_stop") or {}).get("decision") or {}
    return (f"{rec['name']}: {rec['cells']} cells, {rec['iterations']} iterations, {rec['wall_time']:.1f} s on "
            f"{rec['cores']} cores, Cd {rec['cd_forces']:.9g}, health {rec['health']['status']}, "
            f"stop: {d.get('status', 'no live stop')} {d.get('reason', '')}")


# ------------------------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------------------------
def do_preflight():
    problem, num = load()
    G = given(problem)
    levels = num["mesh"]["levels"]
    res = []
    # 1. the coarsest grid to convergence with the live rule: what a converged iteration costs, and its answer
    case, g, info = build("pfG1", problem, num, 0)
    wall, live = solve(case, num, 1, live_decider(num), timeout=900, watch_every=2.0)
    rec = collect(case, "pfG1", G, num, g, wall, 1, live, levels[0])
    print(report_line(rec), flush=True)
    res.append(dict(kind="full", level=levels[0], mesh=dict(info, patches=info["patches"]), record="pfG1"))
    # 2. the finest grid, a short run from rest on its production rank count
    last = len(levels) - 1
    cores = num["mesh"]["cores"][last]
    n_short = num["preflight"]["iterations"]
    # monitor.preflight() copies the case without its processor folders, so a decomposed case cannot use it
    # (insight): the short run is its own case here, with the iteration count as its end.
    case, g, info = build("pfG5", problem, num, last, cores=cores, max_iterations=n_short, write_every=n_short)
    try:
        solve(case, num, cores, timeout=900)
    except RuntimeError as e:                                 # a crash is a finding; assess() reports it
        print(str(e)[-600:])
    pcase = case
    pf = monitor.assess(pcase / f"log.{num['solver']}")
    for w in foam.ignored_settings(pcase):
        pf["checks"].append(dict(name="Settings the solver ignores", status="warn", value="ignored", meaning=w))
    files = {}
    for d in sorted((pcase / "postProcessing").glob("*/*/*")):
        lines = d.read_text().splitlines()
        files[str(d.relative_to(pcase))] = lines[:4] + lines[-2:]
    p = perf.summarize(pcase, num["solver"], info["cells"], cores)
    pf.update(kind="short", level=levels[last], cells=info["cells"], cores=cores, mesh=info, postProcessing_head=files,
              perf=p, seconds_per_iteration=p["wall_s"] / n_short if p.get("wall_s") else None,
              checkMesh=[l for l in (case / "log.checkMesh").read_text().splitlines()
                         if re.search(r"Max aspect ratio|non-orthogonality|Mesh OK|Max skewness|\*\*\*", l)])
    print(f"preflight {levels[last]} ({info['cells']} cells, {cores} ranks): {pf['status']}, "
          f"{pf['seconds_per_iteration']:.4g} s per iteration")
    for ch in pf["checks"]:
        print(f"   {ch['status']:4s} {ch['name']}: {ch['value']}")
    for k, v in files.items():
        print("  ", k, "|", " || ".join(l[:160] for l in v))
    print("   pressure iterations per solve:", p["lin_iters"])
    res.append(pf)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(res, indent=1, default=str))


def do_production(levels):
    problem, num = load()
    G = given(problem)
    all_levels = num["mesh"]["levels"]
    decide = live_decider(num)
    i_first = all_levels.index(levels[0])
    prev = run_root() / all_levels[i_first - 1] if i_first > 0 else None
    RUNS.mkdir(parents=True, exist_ok=True)
    marker = RUNS / "DONE_production"
    marker.unlink(missing_ok=True)
    for lv in levels:
        i = all_levels.index(lv)
        cores = num["mesh"]["cores"][i]
        case, g, _ = build(lv, problem, num, i, source=prev, cores=cores)
        wall, live = solve(case, num, cores, decide)
        rec = collect(case, lv, G, num, g, wall, cores, live, lv)
        print(report_line(rec), flush=True)
        prev = case
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def do_sensitivity(names):
    problem, num = load()
    G = given(problem)
    all_levels = num["mesh"]["levels"]
    decide = live_decider(num)
    marker = RUNS / "DONE_sensitivity"
    marker.unlink(missing_ok=True)
    for vn in names:
        spec = num["sensitivity"][vn]
        for lv in spec["levels"]:
            i = all_levels.index(lv)
            cores = num["mesh"]["cores"][i]
            name = f"{vn}_{lv}"
            case, g, _ = build(name, problem, num, i, source=run_root() / lv, cores=cores, **spec["set"])
            wall, live = solve(case, num, cores, decide)
            rec = collect(case, name, G, num, g, wall, cores, live, lv, variant=vn)
            print(report_line(rec), flush=True)
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def fixed_count(n):
    """decide() for LiveStop that ends a run after n iterations through the stop file. Not through endTime: a
    trial that must show what the iteration does should end like a production run does."""
    tails = {}

    def decide(case):
        it, _, _ = monitored(case, tails)
        if not len(it):
            return None
        return dict(stop=bool(it[-1] >= n), status="ok", reason="fixed count", t=float(it[-1]), quantities={})
    return decide


def do_trials(names):
    """The relaxation trials of the pre-flight on the coarsest grid (numerics.yaml, trials)."""
    problem, num = load()
    G = given(problem)
    lv = num["mesh"]["levels"][0]
    for name in names:
        spec = num["trials"][name]
        case, g, _ = build(name, problem, num, 0, relax_U=spec["U"], relax_nuTilda=spec["nuTilda"])
        wall, live = solve(case, num, 1, fixed_count(spec["iterations"]), timeout=600, watch_every=0.5)
        rec = collect(case, name, G, num, g, wall, 1, live, lv, variant=f"relaxation U {spec['U']}, nuTilda {spec['nuTilda']}")
        print(report_line(rec), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    _, n = load()
    if mode == "preflight":
        do_preflight()
    elif mode == "production":
        do_production(sys.argv[2:] or n["mesh"]["levels"])
    elif mode == "trials":
        do_trials(sys.argv[2:] or list(n["trials"]))
    elif mode == "sensitivity":
        do_sensitivity(sys.argv[2:] or list(n["sensitivity"]))
    else:
        raise SystemExit(f"unknown mode {mode}")
