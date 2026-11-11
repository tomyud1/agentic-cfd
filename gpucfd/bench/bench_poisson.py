"""Speed of the GPU Poisson solver: time per V-cycle and per solve, cells per second and the fraction of
the measured memory bandwidth achieved (the ceiling for this memory-bound code).

    python -m gpucfd.bench.bench_poisson [quick]       (inside WSL, from the project root)

Writes gpucfd/bench/results/bench_poisson.json. Every solve is checked against the exact solution, so a
fast but wrong configuration shows up as a failed accuracy column, not as a speed-up.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from gpucfd import gcfd
from gpucfd.tests.test_poisson import fields, singular

# Finest-level memory passes per V-cycle (each pass reads or writes one array once): 2 + 2 sweeps of 2
# colours, each reading x and b and writing x (3 passes), residual 3, restriction 1, prolongation 2.
# Coarser levels add 1/8 (3D) or 1/4 (2D) of that per level.
PASSES = 8 * 3 + 3 + 1 + 2
BYTES = {"double": 8, "float": 4, "mixed": 4}   # mixed: the V-cycle runs in single precision


def run(two_d, N, prec, reps=20):
    n, bc, p, f = fields("mixed", (N, N, N), two_d)       # mixed boundary conditions: the general case
    s = gcfd.Poisson(n, (1.0, 1.0, 1.0), bc, prec)
    out = dict(dim="2D" if two_d else "3D", N=N, cells=int(np.prod(n)), precision=prec, levels=len(s.levels()))
    for tol in (1e-6, 1e-10):
        if prec == "float" and tol < 1e-7:
            continue
        best = None
        for _ in range(3):                                  # best of 3 (first includes warm-up)
            x, hist, ms = s.solve(f, tol=tol, maxit=100, method="pcg")
            best = ms if best is None else min(best, ms)
        if singular(bc, n):
            x, p = x - x.mean(), p - p.mean()
        out[f"solve_ms_{tol:.0e}"] = best
        out[f"iterations_{tol:.0e}"] = len(hist) - 1
        out[f"error_vs_exact_{tol:.0e}"] = float(np.max(np.abs(x - p)))
    out["vcycle_ms"] = s.bench_vcycle(reps)
    coarse = 1 / (1 - (0.25 if two_d else 0.125))
    out["model_GB_per_vcycle"] = PASSES * coarse * out["cells"] * BYTES[prec] / 1e9
    out["model_GBs"] = out["model_GB_per_vcycle"] / (out["vcycle_ms"] * 1e-3)
    out["ns_per_cell_vcycle"] = out["vcycle_ms"] * 1e6 / out["cells"]
    s.close()
    return out


if __name__ == "__main__":
    quick = "quick" in sys.argv
    bw = gcfd.bandwidth()
    dev = gcfd.device()
    print(f"{dev['name']}: measured copy bandwidth {bw:.0f} GB/s")
    sizes = [(True, N) for N in ((256, 1024) if quick else (256, 512, 1024, 2048, 4096))] + \
            [(False, N) for N in ((64, 128) if quick else (32, 64, 128, 256))]
    rows = []
    for two_d, N in sizes:
        for prec in ("double", "mixed", "float"):
            t0 = time.time()
            r = run(two_d, N, prec)
            rows.append(r)
            print(f"{r['dim']} {N:5d}^{2 if two_d else 3} {r['cells'] / 1e6:6.2f} M cells {prec:6s} "
                  f"V-cycle {r['vcycle_ms']:7.3f} ms ({r['ns_per_cell_vcycle']:5.2f} ns/cell, "
                  f"{r['model_GBs']:4.0f} GB/s = {100 * r['model_GBs'] / bw:3.0f} % of copy)  "
                  f"solve 1e-6: {r['solve_ms_1e-06']:8.2f} ms ({r['iterations_1e-06']} it)  "
                  + (f"1e-10: {r['solve_ms_1e-10']:8.2f} ms ({r['iterations_1e-10']} it), " if 'solve_ms_1e-10' in r else "")
                  + f"error vs exact {r['error_vs_exact_1e-06']:.1e}  [{time.time() - t0:.0f} s]", flush=True)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "bench_poisson.json").write_text(json.dumps(dict(device=dev, bandwidth_GBs=bw, rows=rows), indent=1))
