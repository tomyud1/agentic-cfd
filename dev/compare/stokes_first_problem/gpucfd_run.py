"""Case 3 with gpucfd: Stokes' first problem on the case's cells and time steps, with the four outputs the
case asks for, formed by the case's own code (analyze.analyse_run) against the same exact solution.
The plan (runs, scheme, stop, what counts as agreement) is plan.yaml, written before the runs.

    PYTHONPATH=. python dev/compare/stokes_first_problem/gpucfd_run.py

Writes gpucfd.json next to this file: one value per requirement, the run times, the checks.
"""
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import yaml

from cases.stokes_first_problem import analytical
from cases.stokes_first_problem.analyze import analyse_run
from cases.stokes_first_problem.run import name_of
from cfdlib import foam, verification as V
from cfdlib.report import m, tex_pct as tp
from gpucfd import gcfd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CASE = ROOT / "cases" / HERE.name
OUT = HERE / "gpucfd.json"
CFL_X = 0.4                    # U dt / hx: sets the (free) length of the periodic direction


def solve(Ny, step, U, nu, H, tend, T, plan):
    """From rest to tend with a fixed step. Returns the data analyse_run reads (the layout of the case's
    runs/*.npz: kinematic wall stress and momentum after every step, profiles at the times T) and the run record."""
    Nx, h = plan["domain"]["Nx"], H / Ny
    hx = U * step / CFL_X
    n = int(round(tend / step))
    at = {int(round(t / step)): t for t in T}
    t0 = time.perf_counter()
    s = gcfd.NavierStokes((Nx, Ny, 1), (Nx * hx, H, hx), nu, walls=(False, True, False), wall_velocity={"y-": (U, 0, 0)},
                          scheme=plan["scheme"], htol=plan["diffusion_tolerance"])
    tau, mom, prof, gpu, spread = np.empty(n), np.empty(n), {}, 0.0, 0.0
    for i in range(1, n + 1):
        st = s.run(step, 1, plan["diffusion_tolerance"])
        gpu += st["ms"] / 1e3
        u = s.get()[0][0]                                          # [j, i]
        tau[i - 1] = nu * (U - u[0, 0]) / (h / 2)                  # magnitude, as wallShearStress reports it
        mom[i - 1] = np.sum(u[:, 0]) * h
        if i in at:
            prof[at[i]] = u[:, 0].copy()
            spread = max(spread, float(np.max(np.abs(u - u[:, :1])) / U))
    s.close()
    wall = time.perf_counter() - t0
    times = np.array(sorted(prof))
    d = dict(y=(np.arange(Ny) + 0.5) * h, times=times, u=np.array([prof[t] for t in times]),
             tau_t=step * np.arange(1, n + 1), tau_x=tau, mom_t=step * np.arange(1, n + 1), mom_x=mom)
    # momentum balance per step for this time scheme: change of momentum against the mean wall stress of the step
    M = np.concatenate([[0.0], mom])
    bal = np.abs(np.diff(M)[1:] / step - 0.5 * (tau[1:] + tau[:-1])) / tau[-1]
    run = dict(name=name_of(Ny, step), Ny=Ny, dt=step, h=h, cells=Nx * Ny, steps=n, gpu_s=gpu, wall_s=wall,
               x_spread=spread, max_div=st["max_div"], balance_trapezoid_max=float(bal.max()),
               balance_trapezoid_after_1s=float(bal[int(round(1.0 / step)):].max()))
    return d, run


def main():
    plan = yaml.safe_load((HERE / "plan.yaml").read_text(encoding="utf-8"))
    g = yaml.safe_load((CASE / "problem.yaml").read_text(encoding="utf-8"))["given"]
    numerics = yaml.safe_load((CASE / "numerics.yaml").read_text(encoding="utf-8"))
    of = json.loads((ROOT / "results" / HERE.name / "results.json").read_text(encoding="utf-8"))["data"]["runs"]
    U, nu, rho, tend = (g[k]["value"] for k in ("U", "nu", "rho", "t_end"))
    H, acc = numerics["domain"]["H"], numerics["acceptance"]
    T = [float(t) for t in numerics["profile_times"]]
    _, ex = analytical.evaluate(g, H)

    busy = [p[3][:60] for p in foam.running()]
    t0 = time.perf_counter()
    name = gcfd.device()
    startup = time.perf_counter() - t0
    print(f"GPU: {name}; start-up {startup:.2f} s; other solver processes on the CPU: {len(busy)}", flush=True)

    fam, runs = [], []
    for Ny, step in numerics["runs"]["family"]:
        d, run = solve(Ny, step, U, nu, H, tend, T, plan)
        meta = dict(run, wall_time=run["wall_s"], health=dict(status="ok"), mesh_ok=True, case_dir="")
        r = analyse_run(meta, d, ex, U, nu, rho, T, float(analytical.FRACTION))
        fam.append(r)
        runs.append(run)
        print(f"{run['name']:>16}: wall {run['wall_s']:.1f} s (gpu {run['gpu_s']:.1f}) | profile " +
              ", ".join(f"{r['eu'][t]:.2e}" for t in T) + f" | tau {r['e_tau']:.2e} | depth {r['e_delta_max']:.2e} | "
              f"momentum {r['e_P']:.2e} | tau within 1 % from t = {r['t_1pct']:g} s | far {r['far']:.1e} | "
              f"balance {run['balance_trapezoid_max']:.1e} (after 1 s {run['balance_trapezoid_after_1s']:.1e})", flush=True)
    fine, ofine = fam[-1], dict(of[fam[-1]["name"]])
    ofine.update(eu_max=max(ofine["eu"].values()), e_delta_max=max(ofine["e_delta"].values()))
    of_wall = {r["name"]: json.loads((ROOT / "results" / HERE.name / "runs" / f"{r['name']}.json").read_text())["wall_time"]
               for r in runs}                                      # the case's run records keep OpenFOAM's times
    hs = [x["h"] for x in fam]
    order = dict(profile_20s=V.order_from_errors(hs, [x["eu"][T[-1]] for x in fam]),
                 profile_1s=V.order_from_errors(hs, [x["eu"][T[0]] for x in fam]),
                 tau=V.order_from_errors(hs, [x["e_tau"] for x in fam]), momentum=V.order_from_errors(hs, [x["e_P"] for x in fam]))

    # ---- the case's rules on the finest run, and the two tools side by side -----------------------------
    pairs = dict(a=("eu_max", acc["profile_max_err"]), b=("e_tau", acc["tau_err_tend"]),
                 c=("e_delta_max", acc["delta_err"]), d=("e_P", acc["momentum_err"]))
    fail, warn, side = [], [], {}
    for lab, (key, lim) in pairs.items():
        gv, fv = fine[key], ofine[key]
        if gv > lim:
            fail.append(f"({lab}) {key} {gv:.2e} above {lim}")
        side[lab] = dict(gpucfd=gv, openfoam=fv, limit=lim, ratio=gv / fv, at_most_twice=bool(gv <= 2 * fv),
                         same_side_of_the_rule=bool((gv <= lim) == (fv <= lim)))
    if fine["far"] > acc["far_field_max"]:
        fail.append("far-field check: the layer is not deep enough")
    p = order["profile_20s"][-1]
    if not acc["order_range"][0] <= p <= acc["order_range"][1]:
        warn.append(f"observed order {p:.2f} outside {acc['order_range']}")
    verdict = "FAIL" if fail else "WARN" if warn else "PASS"

    answers = dict(a=dict(value=fine["eu_max"], diff=m(r"\le " + tp(fine["eu_max"]))),
                   b=dict(value=fine["e_tau"], diff=m(tp(fine["e_tau"]))),
                   c=dict(value=fine["e_delta_max"], diff=m(r"\le " + tp(fine["e_delta_max"]))),
                   d=dict(value=fine["e_P"], diff=m(tp(fine["e_P"]))))
    keep = lambda r: dict(name=r["name"], Ny=r["Ny"], dt=r["dt"], eu={str(t): v for t, v in r["eu"].items()}, e_tau=r["e_tau"],
                          tau_end=r["tau_end"], e_delta={str(t): v for t, v in r["e_delta"].items()}, e_P=r["e_P"], P_end=r["P_end"],
                          t_1pct=r["t_1pct"], far=r["far"])
    ratios = [x["ratio"] for x in side.values()]
    plain = (f"gpucfd with the same cells and time steps: its four errors are {min(ratios):.2f} to {max(ratios):.2f} times OpenFOAM's, "
             f"and its finest run takes {runs[-1]['wall_s']:.0f} s.")
    rec = dict(tool="gpucfd", case=HERE.name, date=dt.date.today().isoformat(), device=name, startup_s=startup, plain=plain,
               busy_cpu=bool(busy), busy_processes=busy, answers=answers, runs=runs, family=[keep(r) for r in fam], order=order,
               by_case_rules=dict(verdict=verdict, fail=fail, warn=warn),
               against_openfoam=dict(finest=side, openfoam_wall_s=of_wall,
                                     openfoam_t_1pct=ofine["t_1pct"],
                                     agree=bool(all(x["at_most_twice"] and x["same_side_of_the_rule"] for x in side.values()))))
    OUT.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"by the case's rules: {verdict}; fail {fail}; warn {warn}")
    print("orders (last pair): " + ", ".join(f"{k} {v[-1]:.2f}" for k, v in order.items()))
    for lab, x in side.items():
        print(f"  ({lab}) gpucfd {x['gpucfd']:.3e}, OpenFOAM {x['openfoam']:.3e}, ratio {x['ratio']:.2f}, limit {x['limit']}")
    print(f"equal accuracy by the plan's rule: {rec['against_openfoam']['agree']}\nrecord: {OUT}")


if __name__ == "__main__":
    main()
