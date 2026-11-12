"""Verification ladder, rung 7a: walls (no slip, moving wall, body force) against exact solutions.

    python -m gpucfd.tests.test_walls          (inside WSL, from the project root)

1. Couette flow (2D and 3D): lower wall at rest, upper wall moving at U. The linear profile u = U y / H
   is an exact steady solution of the discrete equations too, so it must stay to round-off.
2. Decaying channel mode: u = sin(pi y / H) exp(-nu pi^2 t / H^2) between walls at rest (2D), and the
   same with the flow along z between walls in x (3D, other directions): order 2 in space.
3. Poiseuille flow driven by a body force g: u = g y (H - y) / (2 nu), reached from rest: order 2.
Also: the wall faces carry no flow and the velocity stays divergence-free.
"""
import json
import sys
from pathlib import Path

import numpy as np

from gpucfd import gcfd

FAILS = []
REPORT = {}


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def couette(scheme="rk3"):
    for dims in (2, 3):
        N = 16
        n = (8, N, 1) if dims == 2 else (8, N, 8)
        U = 1.0
        s = gcfd.NavierStokes(n, (1.0, 1.0, 1.0), 0.1, walls=(False, True, False), wall_velocity={"y+": (U, 0, 0)},
                              scheme=scheme)
        X, Y, Z = s.face_coords(0)
        q = [U * Y] + [np.zeros(s.shape)] * (s.ncomp - 1)
        s.set(q)
        h = 1.0 / N
        st = s.run(s.max_dt(U, 0.5), 200)
        out = s.get()
        err = max(np.max(np.abs(out[0] - U * Y)), *(np.max(np.abs(a)) for a in out[1:]))
        check(f"rung 7: Couette {dims}D {scheme} stays exact", err < 1e-12, f"max deviation after 200 steps {err:.1e}; "
              f"max |div u| {st['max_div']:.1e}")
        REPORT[f"couette_{dims}d_{scheme}"] = err
        s.close()


def channel_mode(dims, Ns, nu=0.1, T=0.5, scheme="rk3"):
    """2D: u(y) between walls y = 0, 1. 3D: w(x) between walls x = 0, 1 (y, z periodic)."""
    errs = []
    for N in Ns:
        if dims == 2:
            n, walls, comp, along = (8, N, 1), (False, True, False), 0, 1
        else:
            n, walls, comp, along = (N, 8, 8), (True, False, False), 2, 0
        s = gcfd.NavierStokes(n, (1.0, 1.0, 1.0), nu, walls=walls, scheme=scheme)
        coords = s.face_coords(comp)
        pos = coords[along]
        q = [np.zeros(s.shape) for _ in range(s.ncomp)]
        q[comp] = np.sin(np.pi * pos)
        s.set(q)
        h = 1.0 / N
        steps = int(np.ceil(T / (0.1 * h * h / nu if scheme == "rk3" else 0.5 * h)))
        st = s.run(T / steps, steps)
        out = s.get()
        exact = np.sin(np.pi * pos) * np.exp(-nu * np.pi ** 2 * T)
        errs.append(float(np.max(np.abs(out[comp] - exact))))
        others = max((np.max(np.abs(out[c])) for c in range(s.ncomp) if c != comp), default=0.0)
        s.close()
    orders = [np.log2(errs[q] / errs[q + 1]) for q in range(len(errs) - 1)]
    key = f"channel_mode_{dims}d_{scheme}"
    check(f"rung 7: decaying channel mode {dims}D {scheme}, order", 1.9 <= orders[-1] <= 2.1,
          "errors " + ", ".join(f"{e:.2e}" for e in errs) + " (N=" + ",".join(map(str, Ns)) + "); orders "
          + ", ".join(f"{o:.2f}" for o in orders) + f"; other components stay {others:.0e}")
    REPORT[key] = dict(N=Ns, err=errs, order=orders)


def poiseuille(Ns, nu=1.0, g=8.0, T=2.5, scheme="rk3"):
    errs, divs, walls_flow = [], [], []
    for N in Ns:
        s = gcfd.NavierStokes((8, N, 1), (1.0, 1.0, 1.0), nu, walls=(False, True, False), force=(g, 0, 0),
                              scheme=scheme)
        h = 1.0 / N
        steps = int(np.ceil(T / (0.12 * h * h / nu if scheme == "rk3" else 0.5 * h)))
        st = s.run(T / steps, steps)
        u, v = s.get()
        X, Y, Z = s.face_coords(0)
        exact = g * Y * (1 - Y) / (2 * nu)
        errs.append(float(np.max(np.abs(u - exact))))
        divs.append(st["max_div"])
        walls_flow.append(float(np.max(np.abs(v[:, 0, :]))))      # v on the lower wall faces
        s.close()
    orders = [np.log2(errs[q] / errs[q + 1]) for q in range(len(errs) - 1)]
    check(f"rung 7: Poiseuille flow from rest {scheme}, order", 1.9 <= orders[-1] <= 2.1,
          "errors " + ", ".join(f"{e:.2e}" for e in errs) + "; orders " + ", ".join(f"{o:.2f}" for o in orders))
    check(f"rung 7: no flow through walls, divergence-free {scheme}", max(walls_flow) == 0.0 and max(divs) < 1e-8,
          f"max |v| on wall faces {max(walls_flow):.1e}, max |div u| {max(divs):.1e}")
    REPORT[f"poiseuille_{scheme}"] = dict(N=Ns, err=errs, order=orders)


def steady_same(Ns=(16, 32)):
    """Both time schemes must reach the same steady state (it solves the same discrete equations). This
    caught the non-incremental IMEX projection, whose steady cavity was off by up to 6e-4 U."""
    from gpucfd.compare import cavity
    f = np.linspace(0, 1, 401)
    for N in Ns:
        a = cavity.gpu_cavity(N, ptol=1e-10, settle=1e-10, scheme="rk3")
        b = cavity.gpu_cavity(N, ptol=1e-10, settle=1e-10, scheme="imex")
        d = max(np.abs(cavity.at_ghia(a["line_u"], f) - cavity.at_ghia(b["line_u"], f)).max(),
                np.abs(cavity.at_ghia(a["line_v"], f) - cavity.at_ghia(b["line_v"], f)).max())
        check(f"rung 7: steady cavity N={N}, IMEX = explicit", d < 1e-7,
              f"max centreline difference {d:.1e} U (steady at t = {a['t_steady']:.0f} / {b['t_steady']:.0f})")
        REPORT[f"steady_same_{N}"] = d


if __name__ == "__main__":
    couette()
    channel_mode(2, (16, 32, 64, 128))
    channel_mode(3, (16, 32, 64))
    poiseuille((8, 16, 32))
    couette("imex")
    channel_mode(2, (16, 32, 64, 128), scheme="imex")
    channel_mode(3, (16, 32, 64), scheme="imex")
    poiseuille((8, 16, 32, 64), scheme="imex")
    steady_same()
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "test_walls.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))
