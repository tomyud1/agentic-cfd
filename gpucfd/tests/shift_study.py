"""One-off study (not part of the ladder; rung 13 reads its results): how much do drag, lift and the surface pressure difference of DFG 2D-1 at D/h 35 move
when the cylinder is shifted along x by a fraction of a cell (a different cut of the surface by the grid)?
Uniform grid (384 x 72 -> 768 x 144, sequenced) and clustered grid (128 x 56 -> 256 x 112, sequenced).
A shift of at most 0.75 cells (2.1 mm of a 2.2 m channel) does not change the flow problem measurably.

    python -m gpucfd.tests.shift_study          (about 8 minutes; writes tests/results/shift_study.json)

Why it exists: rung 10's uniform grid gave the pressure difference to -0.23 % at D/h 35, the clustered grid
of rung 13 to -1.94 %. This study shows the uniform value moves between -0.23 % and -2.39 % with the cut,
so neither number is the grid's accuracy: the band is."""
import json
import sys
from pathlib import Path

import numpy as np

from gpucfd import bodies as B
from gpucfd import gcfd, steady
from gpucfd.tests import test_stretched_bodies as t

DFG = t.DFG
h = DFG["D"] / 35
out = []
kinds = sys.argv[1:] or ["uniform", "clustered"]
for kind in kinds:
    for f in (0.0, 0.25, 0.5, 0.75):
        xc = DFG["xc"] + f * h
        box = ((0.1, 0.3), (0.1, 0.3), (0, 1))
        pts = [[xc - DFG["D"] / 2, DFG["yc"], 0.5], [xc + DFG["D"] / 2, DFG["yc"], 0.5]]

        def make(k):
            if kind == "uniform":
                n, grid = (384 * 2 ** k, 72 * 2 ** k, 1), None
            else:
                n, grid = t.dfg_grid(k)
            bnd = {"x-": ("inflow", t.dfg_inflow), "x+": "outflow", "y-": "wall", "y+": "wall"}
            return gcfd.NavierStokes(n, (DFG["L"], DFG["H"], 1.0), DFG["nu"], boundaries=bnd,
                                     bodies=[B.Cylinder((xc, DFG["yc"]), DFG["D"] / 2)], grid=grid)

        def quantities(s):
            Ubar = 2 * DFG["Um"] / 3
            F = s.forces(box)
            pa, pb = s.surface_pressure(pts)
            return dict(cD=2 * F[0] / (Ubar ** 2 * DFG["D"]), cL=2 * F[1] / (Ubar ** 2 * DFG["D"]), dP=pa - pb, pa=pa, pb=pb)
        res = steady.sequence(make, t.dfg_start, quantities, [0, 1], r=2.0, t_min=30.0, chunk=1.0,
                              dt_of=lambda s: s.max_dt(1.5 * DFG["Um"]), log=None)
        v = res["levels"][-1]["values"]
        ref = t.tb.DFG_REF
        row = dict(kind=kind, shift_cells=f, **v, err={q: v[q] / ref[q] - 1 for q in ref}, gpu_s=sum(x["gpu_s"] for x in res["levels"]))
        out.append(row)
        print(f"{kind} shift {f:.2f} h: cD {v['cD']:.5f} ({100 * row['err']['cD']:+.2f} %) cL {v['cL']:.5f} dP {v['dP']:.5f} "
              f"({100 * row['err']['dP']:+.2f} %) front {v['pa']:.5f} back {v['pb']:.5f} GPU {row['gpu_s']:.0f} s", flush=True)
(Path(__file__).parent / "results" / "shift_study.json").write_text(json.dumps(out, indent=1, default=float))
