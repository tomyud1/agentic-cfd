"""Time per step and solver iterations on the cavity (2D, walls) and the ABC flow (3D periodic).

    python -m gpucfd.bench.bench_step            (inside WSL, from the project root)

Small grids show the fixed cost per step (launches, CPU-GPU syncs); large ones the bandwidth cost.
"""
import json
import sys
from pathlib import Path

import numpy as np

from gpucfd import gcfd

OUT = Path(__file__).parent / "results"


def cavity(N, scheme="imex", warm=2.0, steps=200, ptol=1e-4):
    s = gcfd.NavierStokes((N, N, 1), (1.0,) * 3, 0.01, walls=(True, True, False), wall_velocity={"y+": (1.0, 0, 0)},
                          scheme=scheme)
    dt = s.max_dt(1.0, 0.8)
    s.run(dt, int(warm / dt), ptol)   # developed flow, so iteration counts are typical
    st = s.run(dt, steps, ptol)
    s.close()
    return dict(case=f"cavity {N}^2 {scheme}", ms_step=st["ms"] / steps, p_its=st["poisson_its"], h_its=st["helm_its"])


def abc(N, scheme="imex", steps=20, ptol=1e-4):
    L = 2 * np.pi
    s = gcfd.NavierStokes((N, N, N), (L,) * 3, 0.01, scheme=scheme)
    q = []
    for c in range(3):
        z, y, x = s.face_coords(c)
        q.append([np.sin(z) + np.cos(y), np.sin(x) + np.cos(z), np.sin(y) + np.cos(x)][c])
    s.set(q)
    dt = s.max_dt(2.0, 0.8)
    s.run(dt, 2, ptol)
    st = s.run(dt, steps, ptol)
    s.close()
    return dict(case=f"abc {N}^3 {scheme}", ms_step=st["ms"] / steps, p_its=st["poisson_its"], h_its=st["helm_its"])


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "now"
    rows = [cavity(32), cavity(128), abc(64), abc(128)]
    for r in rows:
        print(f"{r['case']:22s} {r['ms_step']:8.3f} ms/step   p its {r['p_its']:5.2f}   h its {r['h_its']:5.2f}", flush=True)
    OUT.mkdir(exist_ok=True)
    f = OUT / "bench_step.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    d[tag] = rows
    f.write_text(json.dumps(d, indent=1))
