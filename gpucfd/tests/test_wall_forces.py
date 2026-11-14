"""forces() for a box that reaches a wall (a body that touches the wall, such as a cylinder from wall to wall).

    python -m gpucfd.tests.test_wall_forces            (inside WSL, from the project root)

Both checks are exact statements about the discrete momentum balance, so they hold to round-off or the wall
terms are wrong. Pass rule, fixed before the code was run (dev/compare/dfg_cylinder_3d_re20/plan.yaml): 1e-10.
1. No body, a random field, boxes from wall to wall: the force along the walls is zero. Uniform and stretched
   grids, both ways of giving the boundaries, with a moving wall, before and after two time steps (a
   projected field with a pressure). The component normal to the wall is not computed (NaN).
2. A cylinder from wall to wall in a channel with inflow and outflow: two different boxes around it give the
   same drag and lift. Uniform and x-clustered grids, both time schemes.
"""
import json
import sys
from pathlib import Path

import numpy as np

from gpucfd import bodies as B
from gpucfd import gcfd, grids

FAILS = []
REPORT = {}
RESULTS = Path(__file__).parent / "results"
TOL = 1e-10


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def test_no_body():
    out = {}
    n, length = (16, 24, 8), (1.0, 1.5, 0.5)
    lid = (0.7, -0.4, 0.0)                                         # the wall z+ slides in x and y
    cases = dict(
        uniform_walls=dict(walls=(False, True, True), wall_velocity={"z+": lid}),
        uniform_sides=dict(boundaries={"y-": "wall", "y+": "wall", "z-": "wall", "z+": ("wall", lid)}, scheme="imex"),
        stretched=dict(walls=(False, True, True), wall_velocity={"z+": lid}, scheme="imex",
                       grid=(None, grids.tanh(24, 1.5, 6), grids.tanh(8, 0.5, 4))))
    for label, kw in cases.items():
        s = gcfd.NavierStokes(n, length, 0.03, **kw)
        rng = np.random.default_rng(3)
        s.set([rng.standard_normal(s.shape) for _ in range(3)])
        worst = 0.0
        for box in (((0.2, 0.7), (0.3, 1.2), (0.0, 0.5)),          # z from wall to wall
                    ((0.2, 0.7), (0.0, 1.5), (0.0, 0.5)),          # y and z from wall to wall
                    ((0.2, 0.7), (0.3, 1.2), (0.0, 0.3)),          # the low z wall only
                    ((0.2, 0.7), (0.3, 1.2), (0.2, 0.5))):         # the high z wall only (the moving one)
            F = s.forces(box)
            reach = [box[d][0] <= 0.0 or box[d][1] >= length[d] for d in (1, 2)]
            ok_nan = all(np.isnan(F[d]) == r for d, r in zip((1, 2), reach))
            worst = max(worst, float(np.nanmax(np.abs(F))) if ok_nan else np.inf)
        s.run(1e-4, 2, 1e-12)
        F1 = s.forces(((0.2, 0.7), (0.3, 1.2), (0.0, 0.5)))
        scale = float(np.max(np.abs(s.pressure()))) + 1.0
        s.close()
        out[label] = max(worst, float(np.nanmax(np.abs(F1))) / scale)
    check("forces() with a box at a wall is the exact momentum balance (no body: zero for a random field)",
          max(out.values()) < TOL, ", ".join(f"{k}: {v:.1e}" for k, v in out.items()))
    REPORT["no_body"] = out


def test_cylinder():
    out = {}
    L, H, D, Um, nu = 2.0, 0.5, 0.2, 1.0, 0.02

    def inflow(X, Y, Z):
        return (16 * Um * Y * Z * (H - Y) * (H - Z) / H ** 4, 0 * X, 0 * X)
    bnd = {"x-": ("inflow", inflow), "x+": "outflow", "y-": "wall", "y+": "wall", "z-": "wall", "z+": "wall"}
    for label, n, grid, scheme in (("uniform, rk3", (64, 16, 16), None, "rk3"),
                                   ("clustered in x, imex", (48, 16, 16), (grids.clustered(48, L, (0.4, 1.0), H / 16, (0.2, 0.4)), None, None), "imex")):
        s = gcfd.NavierStokes(n, (L, H, H), nu, boundaries=bnd, bodies=[B.Cylinder((0.6, 0.24), D / 2)], scheme=scheme, grid=grid)
        q = []
        for c in range(3):
            X, Y, Z = s.face_coords(c)
            q.append(np.where(s.fluid[c], inflow(X, Y, Z)[c], 0.0))
        s.set(q)
        s.project()
        s.run(0.5 * s.dt_cfl(0.8), 20, 1e-10)
        Fa = s.forces(((0.3, 1.2), (0.06, 0.44), (0.0, H)))
        Fb = s.forces(((0.42, 0.9), (0.1, 0.4), (0.0, H)))
        try:
            s.forces(((0.3, 1.2), (0.06, 0.44), (0.1, 0.4)))       # a box whose z faces cut the cylinder: still refused
            cut = False
        except ValueError:
            cut = True
        s.close()
        out[label] = dict(drag=float(Fa[0]), lift=float(Fa[1]), boxes_differ=float(np.max(np.abs(Fa[:2] - Fb[:2])) / abs(Fa[0])),
                          cut_refused=cut, normal_is_nan=bool(np.isnan(Fa[2])))
    check("forces() on a cylinder from wall to wall: two boxes give the same drag and lift",
          all(v["boxes_differ"] < TOL and v["cut_refused"] and v["normal_is_nan"] and v["drag"] > 0 for v in out.values()),
          ", ".join(f"{k}: drag {v['drag']:.4g}, boxes differ by {v['boxes_differ']:.1e}" for k, v in out.items()))
    REPORT["cylinder"] = out


PARTS = dict(nobody=test_no_body, cylinder=test_cylinder)

if __name__ == "__main__":
    for name in sys.argv[1:] or PARTS:
        PARTS[name]()
    if not sys.argv[1:]:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "test_wall_forces.json").write_text(json.dumps(dict(report=REPORT, fails=FAILS), indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))
