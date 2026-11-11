"""Build and run Stokes' first problem for every (mesh, time step) pair in numerics.yaml; save raw outputs.

Usage (repo root, inside WSL):
  python -m cases.stokes_first_problem.run preflight                 # short runs, coarsest + finest level
  python -m cases.stokes_first_problem.run production [--jobs N] [NAME ...]   # full runs (default: all)
Outputs: results/stokes_first_problem/preflight.json, runs/<name>.json + runs/<name>.npz
Run names: N<cells>_dt<time step in ms>ms, e.g. N2000_dt2.5ms.
"""
import json
import platform
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from cfdlib import foam, monitor
from cfdlib.report import tex_num

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"


def load():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    return problem, numerics


def run_root():
    return Path(foam.foam_env("FOAM_RUN")) / "agentic" / HERE.name


def name_of(Ny, dt):
    return f"N{Ny}_dt{dt * 1000:g}ms"


def all_runs(numerics):
    """Unique (Ny, dt) pairs of every study, in the order: family, mesh, time step."""
    seen, out = set(), []
    for study in ("family", "mesh", "timestep"):
        for Ny, dt in numerics["runs"][study]:
            if (Ny, dt) not in seen:
                seen.add((Ny, dt))
                out.append((Ny, dt))
    return out


def params_for(Ny, dt, problem, numerics, end_time=None):
    g, d = problem["given"], numerics["domain"]
    return dict(U=g["U"]["value"], nu=g["nu"]["value"], H=d["H"], Lx=d["Lx"], Lz=d["Lz"], Ny=Ny, dt=dt,
                end_time=end_time or g["t_end"]["value"], write_interval=numerics["write_interval"],
                U_tol=numerics["linear_solvers"]["U_tol"], p_tol=numerics["linear_solvers"]["p_tol"])


def build(Ny, dt, problem, numerics, prefix="", end_time=None):
    """Render and mesh one case. Pre-flight uses its own directories (prefix 'pf_')."""
    case = foam.render_case(HERE / "template", run_root() / f"{prefix}{name_of(Ny, dt)}",
                            params_for(Ny, dt, problem, numerics, end_time))
    foam.run("blockMesh", case)
    foam.run("checkMesh", case)
    return case


ROUNDOFF_CONT = 1e-12      # 'sum local' continuity error below this is round-off (it is ~1e-19 here)
ROUNDOFF_V = 1e-10         # |v|/U below this is round-off (v is exactly 0 in this flow)


def health(log, case, U, **kw):
    """monitor.assess plus absolute-level checks for quantities that are exactly zero in this flow.

    assess() judges continuity errors and residuals by their TREND (late / early). Here the continuity
    error and v are round-off (1e-28 .. 1e-17), so their ratios are noise and give false alarms
    (insight 3). Such a check is downgraded to ok only when the ABSOLUTE level is round-off, and the
    original verdict is kept in the check's text.
    """
    a = monitor.assess(log, **kw)
    text = Path(log).read_text(errors="replace")
    cont = [float(x) for x in re.findall(r"sum local = ([-0-9.eE+]+)", text)]
    cont_max = max(cont) if cont else None
    tds = [t for t in foam.time_dirs(case) if float(t.name) > 0]
    v_max = float(np.max(np.abs(foam.read_internal_field(tds[-1] / "U")[:, 1])) / U) if tds else None
    for c in a["checks"]:
        orig = c["status"]
        if c["name"] == "Continuity-error trend" and c["status"] != "ok" and cont_max is not None and cont_max < ROUNDOFF_CONT:
            c.update(status="ok", value=c["value"] + r" of round-off; largest value \(" + tex_num(cont_max, 2) + r"\)",
                     meaning=c["meaning"] + " Here the error itself is round-off (below "
                             + r"\(" + tex_num(ROUNDOFF_CONT) + r"\)), so its trend carries no information "
                             "(automatic trend check said: " + orig + ").")
        if c["name"] == "Residual trend: Uy" and c["status"] != "ok" and v_max is not None and v_max < ROUNDOFF_V:
            c.update(status="ok", value=c["value"] + r"; but \(\max|v|/U = " + tex_num(v_max, 2) + r"\)",
                     meaning="v is exactly zero in this flow; its normalised residual is round-off noise and its trend "
                             r"means nothing. The absolute level \(|v|/U < " + tex_num(ROUNDOFF_V) + r"\) is checked instead "
                             "(automatic trend check said: " + orig + ").")
    a["checks"].append(dict(name="Round-off level of zero quantities", status="ok" if (
        cont_max is not None and cont_max < ROUNDOFF_CONT and v_max is not None and v_max < ROUNDOFF_V) else "warn",
        value=r"continuity \(" + tex_num(cont_max or 0, 2) + r"\), \(|v|/U = " + tex_num(v_max or 0, 2) + r"\)",
        meaning="Mass conservation error and cross-flow velocity must stay at round-off: v = 0 exactly in this flow."))
    a["status"] = "fail" if any(c["status"] == "fail" for c in a["checks"]) else \
                  "warn" if any(c["status"] == "warn" for c in a["checks"]) else "ok"
    a.update(cont_max=cont_max, v_max=v_max)
    return a


def _dat(case, fo, fname):
    hits = sorted((case / "postProcessing" / fo).rglob(fname))
    return foam.read_dat(hits[0]) if hits else None


def do_preflight():
    problem, numerics = load()
    fam = numerics["runs"]["family"]
    t_short, t_full = numerics["preflight"]["end_time"], problem["given"]["t_end"]["value"]
    out = []
    for Ny, dt in (fam[0], fam[-1]):
        case = build(Ny, dt, problem, numerics, prefix="pf_")
        pf = monitor.preflight(case, numerics["solver"], end_time_short=t_short, end_time_full=t_full, timeout=600)
        pfc = Path(pf["case"])
        h = health(pfc / f"log.{numerics['solver']}", pfc, problem["given"]["U"]["value"], end_time=t_full)
        pf.update(checks=h["checks"], status=h["status"], cont_max=h["cont_max"], v_max=h["v_max"])
        tau, mom = _dat(pfc, "tauPlate", "surfaceFieldValue.dat"), _dat(pfc, "momentum", "volFieldValue.dat")
        n_steps = round(t_short / dt)
        ok_fo = bool(tau) and bool(mom) and len(tau) >= n_steps and len(mom) >= n_steps
        pf["checks"].append(dict(
            name="Outputs recorded every time step", status="ok" if ok_fo else "fail",
            value=f"{len(tau or [])} wall-stress rows, {len(mom or [])} momentum rows for {n_steps} steps",
            meaning="The wall shear stress and the momentum integral must be written at every step: they are "
                    "requested outputs and the momentum-balance check needs them."))
        if not ok_fo:
            pf["status"] = "fail"
        pf.update(Ny=Ny, dt=dt, name=name_of(Ny, dt), n_steps=n_steps,
                  tau_first=tau[:3] if tau else None, tau_last=tau[-1] if tau else None,
                  mom_last=mom[-1] if mom else None)
        out.append(pf)
        print(f"preflight {name_of(Ny, dt)}: {pf['status']}; {pf['wall_time']} s for {t_short} s; "
              f"projected {pf['est_full_runtime']} s for {t_full} s")
        for ch in pf["checks"]:
            print(f"   {ch['status']:4s} {ch['name']}: {ch['value']}")
        print("   tau rows:", pf["tau_first"], "...", pf["tau_last"], "| momentum:", pf["mom_last"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(out, indent=1, default=str))
    if any(p["status"] == "fail" for p in out):
        raise SystemExit("Pre-flight FAILED: fix the setup before production runs.")


def do_one(Ny, dt):
    problem, numerics = load()
    d = numerics["domain"]
    name = name_of(Ny, dt)
    t0 = time.time()
    case = build(Ny, dt, problem, numerics)
    wall = foam.run(numerics["solver"], case, timeout=900)
    foam.run("postProcess -func writeCellCentres -latestTime", case, "writeCellCentres")
    log = case / f"log.{numerics['solver']}"
    tds = [t for t in foam.time_dirs(case) if float(t.name) > 0]
    Cy = foam.read_internal_field(tds[-1] / "C")[:, 1]
    order = np.argsort(Cy)
    times = np.array([float(t.name) for t in tds])
    U_prof = np.array([foam.read_internal_field(t / "U")[order, 0] for t in tds])     # (n_times, Ny)
    tau = _dat(case, "tauPlate", "surfaceFieldValue.dat")
    mom = _dat(case, "momentum", "volFieldValue.dat")
    area = d["Lx"] * d["Lz"]
    tau_t = np.array([r[0] for r in tau])
    tau_x = np.array([r[1][0] for r in tau])                       # kinematic, tau/rho, x component
    mom_t = np.array([r[0] for r in mom])
    mom_x = np.array([r[1][0] for r in mom]) / area                # integral of u dy  [m^2/s]
    np.savez_compressed(RUNS / f"{name}.npz", y=Cy[order], times=times, u=U_prof,
                        tau_t=tau_t, tau_x=tau_x, mom_t=mom_t, mom_x=mom_x)
    res = foam.residual_history(log, ("Ux", "p"))
    meta = dict(name=name, Ny=Ny, dt=dt, h=d["H"] / Ny, case_dir=str(case), wall_time=wall,
                total_time=time.time() - t0, steps=len(tau_t), mesh_ok="Mesh OK." in (case / "log.checkMesh").read_text(),
                health=health(log, case, problem["given"]["U"]["value"]), final_residuals={f: (v[-1] if v else None) for f, v in res.items()},
                max_Ux_residual=max(res["Ux"]) if res["Ux"] else None,
                openfoam=foam.foam_env("WM_PROJECT_VERSION"), python=platform.python_version(),
                run_root=str(run_root()))
    (RUNS / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(f"{name}: {meta['steps']} steps, solver {wall:.1f} s, health {meta['health']['status']}", flush=True)
    return meta


def do_production(pairs, jobs):
    RUNS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(lambda p: do_one(*p), pairs))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    if mode == "preflight":
        do_preflight()
    elif mode == "production":
        args, jobs = sys.argv[2:], 1
        if "--jobs" in args:
            k = args.index("--jobs")
            jobs = int(args[k + 1])
            args = args[:k] + args[k + 2:]
        _, num = load()
        pairs = all_runs(num)
        if args:
            pairs = [p for p in pairs if name_of(*p) in args]
        do_production(pairs, jobs)
    else:
        raise SystemExit(f"unknown mode {mode}")
