"""Verification ladder, rung 8: reproducibility and "optimisations change nothing".

    python -m gpucfd.tests.test_reproducible          (inside WSL, from the project root)

1. Two identical runs give bit-identical results (deterministic reductions; before they were fixed, an
   atomicAdd made runs differ at 4e-16).
2. CUDA graphs on and off give bit-identical results (a speed optimisation must not change the answer).
   With graphs on, whole solves and time steps run as recorded graphs whose loops the GPU decides;
   with graphs off, the CPU launches every kernel and reads each loop's stop flag.
Checked for the Poisson solver (double and mixed precision) and Navier-Stokes (both time schemes, with
walls and 3D periodic).
"""
import sys

import numpy as np

from gpucfd import gcfd
from gpucfd.tests import test_poisson as tp

FAILS = []


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def poisson(prec, graphs):
    gcfd.use_graphs(graphs)
    n, bc, p, f = tp.fields("mixed", (64, 64, 64), False)
    s = gcfd.Poisson(n, (1.0, 1.0, 1.0), bc, prec)
    x, h, ms = s.solve(f, tol=1e-10, method="pcg")
    s.close()
    return [x]


def cavity(scheme, graphs, N=32, steps=200):
    gcfd.use_graphs(graphs)
    s = gcfd.NavierStokes((N, N, 1), (1.0, 1.0, 1.0), 0.01, walls=(True, True, False),
                          wall_velocity={"y+": (1.0, 0, 0)}, scheme=scheme)
    s.run(s.max_dt(1.0), steps, 1e-6)
    q = s.get()
    s.close()
    return q


def abc3d(scheme, graphs, N=32, steps=20):
    gcfd.use_graphs(graphs)
    L = 2 * np.pi
    s = gcfd.NavierStokes((N, N, N), (L, L, L), 0.01, scheme=scheme)
    q = []
    for c in range(3):
        z, y, x = s.face_coords(c)
        q.append([np.sin(z) + np.cos(y), np.sin(x) + np.cos(z), np.sin(y) + np.cos(x)][c])
    s.set(q)
    s.run(s.max_dt(2.0), steps, 1e-6)
    q = s.get()
    s.close()
    return q


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


if __name__ == "__main__":
    for name, fn in (("Poisson double", lambda g: poisson("double", g)), ("Poisson mixed", lambda g: poisson("mixed", g)),
                     ("cavity rk3", lambda g: cavity("rk3", g)), ("cavity imex", lambda g: cavity("imex", g)),
                     ("3D periodic rk3", lambda g: abc3d("rk3", g)), ("3D periodic imex", lambda g: abc3d("imex", g))):
        off1, off2, on = fn(False), fn(False), fn(True)
        check(f"rung 8: {name}, two runs bit-identical", same(off1, off2), "")
        check(f"rung 8: {name}, CUDA graphs on = off (bit-identical)", same(off1, on), "")
    gcfd.use_graphs(True)
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))
