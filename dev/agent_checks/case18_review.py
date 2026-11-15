"""Review of case 18 (shock tube), dev/LOOP.md step 4: recompute the headline numbers from the raw run data with
an exact solution written here from scratch, independent of the solver agent's theory.py.

    PYTHONPATH=. python dev/agent_checks/case18_review.py        (repository root; needs no solver)

Reads cases/shock_tube_sod/problem.yaml and results/shock_tube_sod/runs/N*.npz|json. Prints each number next
to the one the agent recorded.
"""
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
G = {k: float(v["value"]) for k, v in yaml.safe_load((ROOT / "cases/shock_tube_sod/problem.yaml").read_text(encoding="utf-8"))["given"].items()}
RUNS = ROOT / "results" / "shock_tube_sod" / "runs"
g, pL, rL, pR, rR, L, T = G["gamma"], G["p_L"], G["rho_L"], G["p_R"], G["rho_R"], G["L"], G["t_end"]
cL, cR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)


def side(p, pk, rk, ck):
    """Velocity change across the wave on one side for a middle pressure p (shock if p > pk, else a fan)."""
    if p > pk:
        a, b = 2.0 / ((g + 1.0) * rk), (g - 1.0) / (g + 1.0) * pk
        return (p - pk) * np.sqrt(a / (p + b))
    return 2.0 * ck / (g - 1.0) * ((p / pk) ** ((g - 1.0) / (2.0 * g)) - 1.0)


lo, hi = min(pL, pR) * 1e-6, max(pL, pR)                       # both gases at rest: the middle pressure lies between
for _ in range(200):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if side(mid, pL, rL, cL) + side(mid, pR, rR, cR) < 0.0 else (lo, mid)
ps = 0.5 * (lo + hi)
us = 0.5 * (side(ps, pR, rR, cR) - side(ps, pL, rL, cL))
rsL = rL * (ps / pL) ** (1.0 / g)                               # through the fan: same entropy
csL = cL * (ps / pL) ** ((g - 1.0) / (2.0 * g))
q = ps / pR
rsR = rR * (q + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * q + 1.0)   # through the shock: jump conditions
S = cR * np.sqrt((g + 1.0) / (2.0 * g) * q + (g - 1.0) / (2.0 * g))
head, tail = -cL, us - csL
# the jump conditions at the shock, as a check of this solver itself
assert abs(rR * S - rsR * (S - us)) < 1e-9 * rR * S and abs(pR + rR * S * S - ps - rsR * (S - us) ** 2) < 1e-9 * ps


def exact(xi):
    """Density, velocity and pressure on the rays xi = (x - L/2) / t."""
    rho, u, p = np.full_like(xi, rL), np.zeros_like(xi), np.full_like(xi, pL)
    fan = (xi > head) & (xi <= tail)
    c = (2.0 * cL - (g - 1.0) * xi[fan]) / (g + 1.0)
    u[fan] = 2.0 * (cL + xi[fan]) / (g + 1.0)
    rho[fan] = rL * (c / cL) ** (2.0 / (g - 1.0))
    p[fan] = pL * (c / cL) ** (2.0 * g / (g - 1.0))
    for m, r in (((xi > tail) & (xi <= us), rsL), ((xi > us) & (xi <= S), rsR)):
        rho[m], u[m], p[m] = r, us, ps
    right = xi > S
    rho[right], u[right], p[right] = rR, 0.0, pR
    return rho, u, p


def cell_averages(n, t, sub=64):
    """Exact averages of mass, momentum and energy over each of n cells, and u and p formed from them."""
    xs = (np.arange(n * sub) + 0.5) / (n * sub) * L
    rho, u, p = exact((xs - 0.5 * L) / t)
    avg = lambda a: a.reshape(n, sub).mean(axis=1)                # noqa: E731
    r, m, e = avg(rho), avg(rho * u), avg(p / (g - 1.0) + 0.5 * rho * u * u)
    return r, m / r, (g - 1.0) * (e - 0.5 * m * m / r)


print(f"exact solution, written here: middle pressure {ps:.3f} Pa, middle velocity {us:.4f} m/s, densities {rsL:.6f} and {rsR:.6f} kg/m3")
print(f"   speeds: shock {S:.3f}, border of the two gases {us:.3f}, fan front {head:.3f}, fan back {tail:.3f} m/s; at t_end the shock is at x = {0.5 * L + S * T:.5f} m")

err = {}
for n in (100, 6400, 12800):
    d, j = np.load(RUNS / f"N{n}.npz"), json.loads((RUNS / f"N{n}.json").read_text())
    assert abs(d["times"][-1] - T) < 1e-12 * T and d["rho"].shape[1] == n
    r, u, p = cell_averages(n, T)
    mine = dict(rho=np.mean(np.abs(d["rho"][-1] - r)) / rL, u=np.mean(np.abs(d["u"][-1] - u)) / us, p=np.mean(np.abs(d["p"][-1] - p)) / pL)
    err[n] = mine
    print(f"{n} cells, mean error at t_end, mine | the agent's:  " + "  ".join(f"{k} {100 * mine[k]:.5f} % | {100 * j['errors'][k]:.5f} %" for k in mine))
    if n == 12800:
        x = d["x"]
        mid = (x > 0.5 * L + (us + 0.25 * (S - us)) * T) & (x < 0.5 * L + (us + 0.75 * (S - us)) * T)   # between border and shock
        print(f"   state between the border and the shock: pressure {d['p'][-1][mid].mean():.3f} Pa ({100 * (d['p'][-1][mid].mean() / ps - 1):+.5f} %), "
              f"velocity {d['u'][-1][mid].mean():.4f} m/s ({100 * (d['u'][-1][mid].mean() / us - 1):+.5f} %)")
        print(f"   largest change of total mass {np.abs(d['c_mass'] / d['c_mass'][0] - 1).max():.1e}, of total energy {np.abs(d['c_energy'] / d['c_energy'][0] - 1).max():.1e} (relative)")
print("order of the mean error from 6400 to 12800 cells: " + ", ".join(f"{k} {np.log2(err[6400][k] / err[12800][k]):.3f}" for k in ("rho", "u", "p")))
floor = {"rho": 0.7, "u": 0.85, "p": 0.85}                       # the agent's limits (numerics.yaml, fail_l1_order_floor)
for k, f in floor.items():
    lim = err[100][k] * (100 / 12800) ** f
    print(f"   {k}: limit on 12800 cells {100 * lim:.5f} %, reached {100 * err[12800][k]:.5f} % -> {'inside' if err[12800][k] <= lim else 'OUTSIDE'}")
