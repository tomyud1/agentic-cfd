"""Compressible ladder, rung 1: the GPU kernels against an independent NumPy implementation of the same scheme.

    python -m gpucfd.tests.test_euler_kernels          (inside WSL, from the project root)

One time step of a rough field (a smooth wave, noise, a block at ten times the pressure, parts faster than
sound) is taken by the GPU solver and by gpucfd/tests/euler_ref.py, for every reconstruction, every flux
formula, every boundary type (also one side split into wall and inflow), in 1D, 2D and 3D, with one and with
three stages. Pass: equal to round-off (double 1e-12, float and mixed 1e-5 of each quantity's scale), and the same for
a direction-swapped copy of a 1D problem (the solver treats x, y and z alike).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from gpucfd.euler import Euler
from gpucfd.tests import euler_ref

FAILS = []
REPORT = {}
GAMMA = 1.4
TOL = {"double": 1e-12, "float": 1e-5, "mixed": 1e-5}
IN = dict(rho=1.3, u=0.9, v=-0.4, w=0.3, p=1.6)
FAST = dict(rho=0.8, u=2.5, v=0.2, w=-0.1, p=0.7)          # faster than sound along +x


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def field(n, seed, as_float):
    rng = np.random.default_rng(seed)
    shape = (n[2], n[1], n[0])
    z, y, x = np.meshgrid(*[(np.arange(m) + 0.5) / m for m in shape], indexing="ij")
    rho = 1 + 0.3 * np.sin(2 * np.pi * (x + 2 * y + z)) + 0.05 * rng.random(shape)
    p = 1 + 0.2 * np.cos(2 * np.pi * (x - y)) + 0.05 * rng.random(shape)
    blk = (np.abs(x - 0.5) < 0.2) & ((np.abs(y - 0.5) < 0.2) | (n[1] == 1)) & ((np.abs(z - 0.5) < 0.2) | (n[2] == 1))
    p = np.where(blk, 10 * p, p)
    vel = [1.8 * np.sin(2 * np.pi * (x + 0.3 * k)) * (n[k] > 1) + 0.3 * (rng.random(shape) - 0.5) for k in range(3)]
    U = np.stack([rho, rho * vel[0], rho * vel[1], rho * vel[2], p / (GAMMA - 1) + 0.5 * rho * sum(v * v for v in vel)])
    return U.astype(np.float32).astype(np.float64) if as_float else U


def scales(U):
    rho, vel = U[0], U[1:4] / U[0]
    p = (GAMMA - 1) * (U[4] - 0.5 * rho * (vel ** 2).sum(0))
    m = float(np.max(rho * (np.sqrt((vel ** 2).sum(0)) + np.sqrt(GAMMA * p / rho))))
    return np.array([rho.max(), m, m, m, U[4].max()])


def compare(n, sides, segments, recon, flux, stages, precision, seed=0):
    U0 = field(n, seed, precision == "float")
    h = tuple(1.0 / m for m in n)
    cfg = dict(n=n, h=h, gamma=GAMMA, recon=recon, flux=flux, stages=stages, sensor=0.5, sides=sides, segments=segments,
               weno_eps=1e-20 if precision == "double" else 1e-10)
    rho, vel = U0[0], U0[1:4] / U0[0]
    p = (GAMMA - 1) * (U0[4] - 0.5 * rho * (vel ** 2).sum(0))
    speed = sum((np.abs(vel[d]) + np.sqrt(GAMMA * p / rho)) / h[d] for d in range(3) if n[d] > 1).max()
    dt = 0.3 / speed
    ref = euler_ref.step(U0, dt, cfg)
    s = Euler(n, (1.0, 1.0, 1.0), GAMMA, sides=sides, precision=precision, recon=recon, flux=flux, stages=stages)
    for seg in segments:
        s.segment(*seg)
    s.set_conserved(U0)
    s.run_fixed(dt, 1)
    U1 = s.get_conserved()
    s.close()
    sc = scales(U0).reshape(5, 1, 1, 1)
    return float(np.max(np.abs(U1 - ref) / sc)), float(np.max(np.abs(ref - U0) / sc))


BOUNDARIES = {
    "periodic": ({}, []),
    "walls": ({k: "wall" for k in ("x-", "x+", "y-", "y+", "z-", "z+")}, []),
    "open": ({"x-": ("inflow", IN), "x+": "outflow", "y-": "outflow", "y+": ("inflow", IN), "z-": "wall", "z+": "outflow"}, []),
    "split": ({"x-": "wall", "x+": "outflow", "y-": "wall", "y+": "wall", "z-": "wall", "z+": "wall"},
              [("x-", {"y": (5, 12)}, ("supersonic", FAST)), ("y+", {"x": (0, 6)}, ("inflow", IN)),
               ("x-", {"y": (0, 8)}, "outflow")]),          # the last one overlaps the first: the first wins
}
GRIDS = {"1D": (24, 1, 1), "2D": (20, 16, 1), "3D": (12, 10, 9)}
RECONS = ("first", "minmod", "vanleer", "mc", "central", "weno5")
FLUXES = ("rusanov", "hll", "hllc", "hybrid")


def main():
    for precision in ("double", "float", "mixed"):
        for gname, n in GRIDS.items():
            worst, moved, count, worst_at = 0.0, 1.0, 0, ""
            for bname, recon, flux, stages in itertools.product(BOUNDARIES, RECONS, FLUXES, (1, 3)):
                sides, segs = BOUNDARIES[bname]
                if gname == "1D":
                    segs = [g for g in segs if g[0] == "y+"][:0]          # nothing to split along in 1D
                err, change = compare(n, sides, segs, recon, flux, stages, precision)
                count += 1
                if err > worst:
                    worst, worst_at = err, f"{bname}/{recon}/{flux}/{stages} stages"
                moved = min(moved, change)
            check(f"rung 1: {gname} {precision}, {count} combinations of boundary, reconstruction, flux, stages",
                  worst < TOL[precision] and moved > 1e-3,
                  f"largest difference from the NumPy reference {worst:.1e} of the scale ({worst_at}); "
                  f"smallest change made by the step itself {moved:.1e}")
            REPORT[f"{gname}_{precision}"] = dict(n=n, combinations=count, worst=worst, worst_at=worst_at, smallest_change=moved)

    # the same 1D problem along x, y and z
    U0 = field((24, 1, 1), 3, False)
    res = {}
    for d, n in enumerate(((24, 1, 1), (1, 24, 1), (1, 1, 24))):
        ax = "xyz"[d]
        s = Euler(n, (1.0, 1.0, 1.0), GAMMA, sides={ax + "-": "wall", ax + "+": ("inflow", dict(rho=1.3, p=1.6, **{"uvw"[d]: -0.9}))},
                  precision="double", recon="weno5", flux="hllc")
        line = U0[:, 0, 0, :]
        U = np.zeros((5,) + s.shape)
        U[0], U[1 + d], U[4] = (a.reshape(s.shape) for a in (line[0], line[1], line[4] - 0.5 * (line[2] ** 2 + line[3] ** 2) / line[0]))
        s.set_conserved(U)
        s.run_fixed(1e-3, 20)
        V = s.get_conserved()
        res[ax] = np.stack([V[0].ravel(), V[1 + d].ravel(), V[4].ravel()])
        s.close()
    d = max(float(np.max(np.abs(res[a] - res["x"]))) for a in "yz")
    check("rung 1: a 1D problem along y and along z equals the one along x", d < 1e-13, f"largest difference {d:.1e} after 20 steps")
    REPORT["directions"] = d

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "test_euler_kernels.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
