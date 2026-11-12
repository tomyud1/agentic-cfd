"""Speed round 2: re-time the comparison problems with the current build (answers must not change).

    python -m gpucfd.bench.round2            (inside WSL, from the project root)

ABC flow (the OpenFOAM comparison, same best settings: CFL 0.8, pressure tolerance 1e-4) and the
lid-driven cavity to steady state (IMEX at CFL 0.8 and 2). Results in bench/results/round2.json.
"""
import json
from pathlib import Path

from gpucfd.compare import abc
from gpucfd.compare.cavity import gpu_cavity

OUT = Path(__file__).parent / "results"

if __name__ == "__main__":
    rows = []
    for N in (32, 64, 128, 256):
        best = None
        for rep in range(2 if N < 256 else 1):
            r = abc.gpu(N, 0.8, 1e-4)
            best = r if best is None or r["loop_s"] < best["loop_s"] else best
        rows.append(dict(case=f"abc {N}^3 rk3", loop_s=best["loop_s"], err=best["err"], steps=best["steps"]))
        print(rows[-1], flush=True)
    for cfl in (0.8, 2.0):
        for N in (32, 64, 128, 256):
            r = gpu_cavity(N, scheme="imex", cfl=cfl)
            rows.append(dict(case=f"cavity {N}^2 imex cfl {cfl:g}", loop_s=r["loop_s"], steps=r["steps"],
                             t_steady=r["t_steady"], u_mid=float(r["line_u"][1][N // 2])))
            print(rows[-1], flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "round2.json").write_text(json.dumps(rows, indent=1))
