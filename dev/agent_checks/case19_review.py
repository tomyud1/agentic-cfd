"""Review of case 19 (a shock going round a corner), dev/LOOP.md step 4: recompute the three required
accuracies from the raw fields of the finest run, with the shock's jump conditions written here from scratch.

    PYTHONPATH=. python dev/agent_checks/case19_review.py        (repository root; needs no solver)
"""
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
G = {k: float(v["value"]) for k, v in yaml.safe_load((ROOT / "cases/shock_diffraction_corner/problem.yaml").read_text(encoding="utf-8"))["given"].items()}
g, R, p0, T0, Ms, L, H, Hs, T = G["gamma"], G["R"], G["p_0"], G["T_0"], G["M_s"], G["L"], G["H"], G["H_s"], G["t_end"]
rho0, c0 = p0 / (R * T0), np.sqrt(g * R * T0)
W = Ms * c0
p1 = p0 * (1 + 2 * g / (g + 1) * (Ms ** 2 - 1))
rho1 = rho0 * (g + 1) * Ms ** 2 / ((g - 1) * Ms ** 2 + 2)
u1 = 2 * c0 / (g + 1) * (Ms - 1 / Ms)
E0, E1 = p0 / (g - 1), p1 / (g - 1) + 0.5 * rho1 * u1 ** 2            # energy per volume: still air, behind the shock
assert abs(rho1 * (W - u1) - rho0 * W) < 1e-9 * rho0 * W                # mass through the shock, as a check of these formulas
print(f"exact: shock speed {W:.2f} m/s; behind it p {p1:.0f} Pa, rho {rho1:.5f} kg/m3, u {u1:.2f} m/s; flat shock at x = {W * T:.5f} m at t_end")

runs = ROOT / "results" / "shock_diffraction_corner" / "runs"
for n in (400, 800):
    d, j = np.load(runs / f"N{n}.npz"), json.loads((runs / f"N{n}.json").read_text())
    h = L / n
    P, RHO, U, V = d["map_p"], d["map_rho"], d["map_u"], d["map_v"]
    if P[n // 4, 2] > P[3 * n // 4, 2]:                                 # make the first index y (rows), the second x: high pressure sits at small x above the corner
        P, RHO, U, V = P.T, RHO.T, U.T, V.T
    assert P[3 * n // 4, 2] > 10 * p0 and abs(P[n // 4, -2] / p0 - 1) < 1e-6, "orientation of the stored maps"
    x = (np.arange(n) + 0.5) * h
    top = range(int(0.9 * n), n)      # the tenth of the rows nearest the top wall. The corner's signal has reached y = 0.855 m on the
    #                                   front by t_end (a first try with the top fifth mixed disturbed rows in: -1.6 % in pressure)
    half = 0.5 * (p0 + p1)
    fronts = []
    for r in top:                                                       # where the pressure crosses the mean of the two states, between cell centres
        i = int(np.nonzero(P[r] > half)[0].max())
        fronts.append(x[i] + (P[r, i] - half) / (P[r, i] - P[r, i + 1]) * h)
    fronts = np.array(fronts)
    speed = fronts.mean() / T
    band = (x > W * T - 0.13) & (x < W * T - 0.035)                     # behind the shock, ahead of the start-up waves (which sit at u t and (u - c) t)
    pm, rm, um = (a[top][:, band].mean() for a in (P, RHO, U))
    mass = RHO.sum() * h * h - rho0 * L * H
    energy = (P / (g - 1) + 0.5 * RHO * (U ** 2 + V ** 2)).sum() * h * h - E0 * L * H
    in_mass, in_energy = rho1 * u1 * (H - Hs) * T, (E1 + p1) * u1 * (H - Hs) * T
    print(f"{n} x {n}: shock speed {100 * (speed / W - 1):+.4f} % (shock {((fronts.mean() - W * T) / h):+.2f} cells from exact); behind it p {100 * (pm / p1 - 1):+.3f} %, "
          f"rho {100 * (rm / rho1 - 1):+.3f} %, u {100 * (um / u1 - 1):+.3f} %")
    print(f"      straightness over those rows: {(fronts.max() - fronts.min()) / h:.1e} cells; mass against start + inflow {100 * (mass / in_mass - 1):+.1e} % of the inflow, "
          f"energy {100 * (energy / in_energy - 1):+.1e} %; the agent's record: {json.dumps(j['flat'])[:200]}")
