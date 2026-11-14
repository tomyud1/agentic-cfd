"""Case 1 with gpucfd: plane Poiseuille flow on the case's grids, with the four outputs the case asks for,
formed as in cases/plane_poiseuille/analyze.py against the same exact solution. The plan (grids, time step,
stop rule, tuning budget, what counts as agreement) is plan.yaml, written before the runs.

    PYTHONPATH=. python dev/compare/plane_poiseuille/gpucfd_run.py

Writes gpucfd.json next to this file: one value per requirement, the run times per grid, the checks.
"""
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import yaml

from cases.plane_poiseuille import analytical
from cfdlib import foam, verification as V
from cfdlib.report import m, tex_pct as tp
from gpucfd import gcfd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CASE = ROOT / "cases" / HERE.name
OUT = HERE / "gpucfd.json"
CFL_X = 0.8                    # u_max dt / hx: sets the (free) length of the periodic direction


def solve(Ny, c, g, G, mu, u_max, plan):
    """March from rest until the wall friction balances the driving force. Returns y, u(y) and the run record."""
    H, rho, nu = (g[k]["value"] for k in ("H", "rho", "nu"))
    Nx, h, stop = plan["grids"]["Nx"], H / Ny, plan["stop"]
    step = c * H * h / nu
    hx = u_max * step / CFL_X
    t0 = time.perf_counter()
    s = gcfd.NavierStokes((Nx, Ny, 1), (Nx * hx, H, hx), nu, walls=(False, True, False), force=(G / rho, 0.0, 0.0),
                          scheme=plan["scheme"], htol=plan["diffusion_tolerance"])
    steps, gpu, good, bal = 0, 0.0, 0, np.inf
    while steps < stop["cap_steps"] and good < 2:
        st = s.run(step, stop["check_every"], plan["diffusion_tolerance"])
        steps += stop["check_every"]
        gpu += st["ms"] / 1e3
        u = s.get()[0][0]                                          # [j, i]: u on the x-faces, at the cell-centre heights
        tau = mu * (u[0, 0] + u[-1, 0]) / (h / 2)                  # both walls
        bal = abs(tau - G * H) / (G * H)
        good = good + 1 if bal < stop["force_balance"] else 0
    v = s.get()[1][0]
    s.close()
    return (np.arange(Ny) + 0.5) * h, u[:, 0], dict(
        Ny=Ny, cells=Nx * Ny, c=c, dt=step, steps=steps, converged=bool(good >= 2), force_balance=float(bal),
        x_spread=float(np.max(np.abs(u - u[:, :1])) / u_max), v_max=float(np.max(np.abs(v)) / u_max),
        max_div=st["max_div"], gpu_s=gpu, wall_s=time.perf_counter() - t0)


def outputs(y, u, run, ex, u_exact, G, H, mu):
    """As analyze.py's per-mesh comparison."""
    n, h, ue = len(u), H / len(u), u_exact(y)
    u_c = 0.5 * (u[n // 2 - 1] + u[n // 2])
    q = float(np.sum(u) * h)
    tau = mu * 0.5 * (u[0] + u[-1]) / (h / 2)
    return dict(Ny=run["Ny"], h=h, err_profile=V.relative_l2(u, ue, ex["u_max"]), u_c=float(u_c),
                err_umax=abs(u_c - ex["u_max"]) / ex["u_max"], q=q, err_q=abs(q - ex["q"]) / ex["q"],
                tau=float(tau), err_tau=abs(tau - ex["tau_w"]) / ex["tau_w"], force_balance=run["force_balance"],
                offset_mean=float(np.mean(u - ue)), offset_spread=float(np.ptp(u - ue)), offset_predicted=G * h ** 2 / (8 * mu))


def main():
    plan = yaml.safe_load((HERE / "plan.yaml").read_text(encoding="utf-8"))
    g = yaml.safe_load((CASE / "problem.yaml").read_text(encoding="utf-8"))["given"]
    of = json.loads((ROOT / "results" / HERE.name / "results.json").read_text(encoding="utf-8"))["data"]
    theory, u_exact = analytical.evaluate(g)
    ex = {d["key"]: d["value"] for d in theory["derived"]}
    G, mu, H = theory["G"], theory["mu"], g["H"]["value"]

    busy = [p[3][:60] for p in foam.running()]
    t0 = time.perf_counter()
    name = gcfd.device()
    startup = time.perf_counter() - t0
    print(f"GPU: {name}; start-up {startup:.2f} s; other solver processes on the CPU: {len(busy)}", flush=True)

    # ---- the tuning budget: three tries of the time-step constant on one grid ---------------------------
    tries = []
    for c in plan["time_step"]["c_tries"]:
        _, _, run = solve(32, c, g, G, mu, ex["u_max"], plan)
        tries.append(run)
        print(f"try c = {c:g}: {run['steps']} steps, converged {run['converged']}, balance {run['force_balance']:.1e}, "
              f"wall {run['wall_s']:.2f} s", flush=True)
    ok = [r for r in tries if r["converged"]]
    if not ok:
        raise SystemExit("no try met the stop rule: the plan's budget is used up; nothing is recorded")
    c = min(ok, key=lambda r: r["wall_s"])["c"]
    print(f"chosen: c = {c:g}", flush=True)

    rows, runs = [], []
    for Ny in plan["grids"]["Ny"]:
        y, u, run = solve(Ny, c, g, G, mu, ex["u_max"], plan)
        x = outputs(y, u, run, ex, u_exact, G, H, mu)
        rows.append(x)
        runs.append(run)
        print(f"Ny {Ny:4d}: {run['steps']} steps, converged {run['converged']}, balance {run['force_balance']:.1e}, "
              f"gpu {run['gpu_s']:.2f} s, wall {run['wall_s']:.2f} s | profile {x['err_profile']:.3e} u_max {x['err_umax']:.1e} "
              f"q {x['err_q']:.3e} tau {x['err_tau']:.1e}", flush=True)
    hs = [x["h"] for x in rows]
    order_profile = V.order_from_errors(hs, [x["err_profile"] for x in rows])
    order_q = V.order_from_errors(hs, [x["err_q"] for x in rows])
    fine = rows[-1]

    # ---- the case's rules, and the two tools on equal grids ---------------------------------------------
    fail = []
    if not all(r["converged"] for r in runs):
        fail.append("a run reached the step cap before it settled")
    if fine["err_profile"] > 0.005:
        fail.append("finest-grid profile error above 0.5 %")
    if not 1.5 <= order_profile[-1] <= 2.5:
        fail.append(f"observed order {order_profile[-1]:.2f} outside 1.5 to 2.5")
    if max(x["force_balance"] for x in rows) > 1e-6:
        fail.append("momentum balance not satisfied")
    verdict = "FAIL" if fail else "PASS"
    ofr = {r["Ny"]: r for r in of["rows"]}
    per_grid = [dict(Ny=x["Ny"], profile=abs(x["err_profile"] - ofr[x["Ny"]]["err_profile"]), q=abs(x["err_q"] - ofr[x["Ny"]]["err_q"]),
                     u_max=x["err_umax"], tau=x["err_tau"], openfoam_u_max=ofr[x["Ny"]]["err_umax"], openfoam_tau=ofr[x["Ny"]]["err_tau"])
                for x in rows]
    tol = 1e-8
    agree = all(max(p["profile"], p["q"], p["u_max"], p["tau"]) < tol for p in per_grid)

    answers = dict(a=dict(value=fine["err_profile"], diff="", sim="RMS difference " + m(tp(fine["err_profile"])) + " of " + m(r"u_{\max}")),
                   b=dict(value=fine["err_umax"], diff=m(tp(fine["err_umax"]))),
                   c=dict(value=fine["err_q"], diff=m(tp(fine["err_q"]))),
                   d=dict(value=fine["err_tau"], diff=m(tp(fine["err_tau"]))))
    plain = (f"gpucfd on the same cells gives the same four numbers (largest difference between the tools "
             + m(tp(max(max(p["profile"], p["q"], p["u_max"], p["tau"]) for p in per_grid), 2)) + f"), its finest run in {runs[-1]['wall_s']:.2f} s.")
    rec = dict(tool="gpucfd", case=HERE.name, date=dt.date.today().isoformat(), device=name, startup_s=startup, plain=plain,
               busy_cpu=bool(busy), busy_processes=busy, answers=answers, runs=runs, grids=rows,
               order_profile=order_profile, order_q=order_q, tuning=dict(tries=tries, chosen_c=c),
               by_case_rules=dict(verdict=verdict, fail=fail, warn=[]),
               against_openfoam=dict(per_grid=per_grid, tolerance=tol, agree=bool(agree)))
    OUT.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"by the case's rules: {verdict}; fail {fail}; orders profile {order_profile[-1]:.3f}, q {order_q[-1]:.3f}")
    for p in per_grid:
        print(f"gpucfd - OpenFOAM, Ny {p['Ny']:4d}: profile error differs by {p['profile']:.1e}, flow rate by {p['q']:.1e}; "
              f"u_max {p['u_max']:.1e} (OpenFOAM {p['openfoam_u_max']:.1e}), tau {p['tau']:.1e} (OpenFOAM {p['openfoam_tau']:.1e})")
    print(f"agree to {tol:g}: {agree}\nrecord: {OUT}")


if __name__ == "__main__":
    main()
