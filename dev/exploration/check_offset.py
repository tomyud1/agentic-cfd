"""Is the FV solution the exact parabola plus a uniform offset? Compare offset to G h^2 / (8 mu)."""
import json
import sys

import numpy as np

raw = json.load(open(sys.argv[1]))
H, G, mu = 0.01, 6.0, 998.2 * 1.004e-6
for r in raw["runs"]:
    y, u = np.array(r["y"]), np.array(r["u"])
    exact = G / (2 * mu) * y * (H - y)
    d = u - exact
    print(f"Ny={r['Ny']:4d}  offset mean={d.mean():.6e}  spread={d.max() - d.min():.1e}  "
          f"G h^2/(8 mu)={G * r['h'] ** 2 / (8 * mu):.6e}")
