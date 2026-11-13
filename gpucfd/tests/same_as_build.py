"""Do two builds of the library give bit-identical results? For changes meant to alter speed only.

    GCFD_LIB=<other .so> python -m gpucfd.tests.same_as_build save <file.npz>   (with the other build)
    python -m gpucfd.tests.same_as_build compare <file.npz>                     (with the current build)

Cases: Poisson solves (double, mixed; multigrid and CG), cavity with each time scheme, 3D periodic ABC flow.
"""
import sys

import numpy as np

from gpucfd import gcfd
from gpucfd.tests import test_poisson as tp


def results():
    out = {}
    for prec in ("double", "mixed"):
        for method in ("mg", "pcg"):
            n, bc, p, f = tp.fields("mixed", (64, 64, 64), False)
            s = gcfd.Poisson(n, (1.0, 1.0, 1.0), bc, prec)
            x, h, ms = s.solve(f, tol=1e-10, method=method)
            s.close()
            out[f"poisson_{prec}_{method}"] = x
            out[f"poisson_{prec}_{method}_hist"] = h
    for scheme in ("rk3", "imex"):
        s = gcfd.NavierStokes((32, 32, 1), (1.0,) * 3, 0.01, walls=(True, True, False), wall_velocity={"y+": (1.0, 0, 0)},
                              scheme=scheme)
        s.run(s.max_dt(1.0), 200, 1e-6)
        for c, a in enumerate(s.get()):
            out[f"cavity_{scheme}_{c}"] = a
        s.close()
        L = 2 * np.pi
        s = gcfd.NavierStokes((32, 32, 32), (L,) * 3, 0.01, scheme=scheme)
        q = []
        for c in range(3):
            z, y, x = s.face_coords(c)
            q.append([np.sin(z) + np.cos(y), np.sin(x) + np.cos(z), np.sin(y) + np.cos(x)][c])
        s.set(q)
        s.run(s.max_dt(2.0), 20, 1e-4)
        for c, a in enumerate(s.get()):
            out[f"abc_{scheme}_{c}"] = a
        s.close()
    if hasattr(gcfd.lib(), "gcfd_ns_body"):  # immersed bodies on uniform grids (added 2026-11-13, stage 2)
        from gpucfd import bodies as B
        ctr = (0.5, 0.5, 0.5)  # closed Couette cell between two immersed cylinders
        s = gcfd.NavierStokes((32, 32, 1), (1.0,) * 3, 0.025, walls=(True, True, False),
                              bodies=[B.Cylinder(ctr, 0.2, velocity=B.rotation(ctr, 5.0)), B.Inside(B.Cylinder(ctr, 0.45))])
        s.run(s.max_dt(1.0), 100, 1e-10)
        for c, a in enumerate(s.get()):
            out[f"couette_body_{c}"] = a
        out["couette_body_p"] = s.pressure()
        s.close()

        def inflow(X, Y, Z):  # open channel with a cylinder (DFG geometry)
            return (4 * 0.3 * Y * (0.41 - Y) / 0.41 ** 2, 0 * X, 0 * X)
        s = gcfd.NavierStokes((384, 72, 1), (2.2, 0.41, 1.0), 1e-3, bodies=[B.Cylinder((0.2, 0.2), 0.05)],
                              boundaries={"x-": ("inflow", inflow), "x+": "outflow", "y-": "wall", "y+": "wall"})
        q = []
        for c in range(2):
            X, Y, Z = s.face_coords(c)
            q.append(np.where(s.fluid[c], inflow(X, Y, Z)[c], 0.0))
        s.set(q)
        s.project()
        s.run(s.max_dt(0.45), 60, 1e-8)
        for c, a in enumerate(s.get()):
            out[f"channel_body_{c}"] = a
        out["channel_body_p"] = s.pressure()
        out["channel_body_F"] = s.forces(((0.1, 0.3), (0.1, 0.3), (0, 1)))
        out["channel_body_ps"] = s.surface_pressure([[0.15, 0.2, 0.5], [0.25, 0.2, 0.5]])
        s.close()
    return out


if __name__ == "__main__":
    mode, f = sys.argv[1], sys.argv[2]
    r = results()
    if mode == "save":
        np.savez(f, **r)
        print(f"saved {len(r)} arrays")
        sys.exit(0)
    ref = np.load(f)
    bad = [k for k in r if not np.array_equal(r[k], ref[k])]
    for k in r:
        d = float(np.max(np.abs(r[k] - ref[k]))) if r[k].shape == ref[k].shape else float("nan")
        print(f"{'SAME' if k not in bad else 'DIFF'}  {k}  max diff {d:.3g}")
    print(f"\n{len(bad)} of {len(r)} differ")
    sys.exit(len(bad))
