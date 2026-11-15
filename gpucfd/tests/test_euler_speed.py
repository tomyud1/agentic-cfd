"""Compressible ladder, rung 5: speed and memory.

    python -m gpucfd.tests.test_euler_speed          (inside WSL, from the project root; nothing else on the GPU)

A smooth vortex in a periodic box (2D and 3D) and a density wave (1D) are stepped with a fixed time step, with
the three-stage scheme, and timed with the GPU finished at both ends. Reported per grid, precision and scheme:
cells times steps per second, nanoseconds per cell and step, and bytes per cell (what the solver allocates).
Best of three timings. Marked busy if an OpenFOAM process is running on the CPU.

Rules fixed before the runs: memory per cell within 5 % of 15 stored numbers per cell (60 bytes in single, 120
in mixed and double precision); on the largest 2D grid mixed precision at least twice as fast as double (this
card runs double-precision arithmetic at 1/64 of the single rate). Everything else is recorded, not judged.
"""
import json
import sys
from pathlib import Path

import numpy as np

from cfdlib import foam
from gpucfd.euler import Euler
from gpucfd.tests.test_euler_smooth import vortex_fields

FAILS = []
REPORT = {}
GAMMA = 1.4


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def timed(n, precision, recon, flux, steps):
    dims = sum(1 for m in n if m > 1)
    L = 20.0
    s = Euler(n, tuple(L if m > 1 else 1.0 for m in n), GAMMA, precision=precision, recon=recon, flux=flux)
    if dims == 1:
        x = s.centres("x")
        s.set(rho=1 + 0.2 * np.sin(2 * np.pi * x / L), u=1.0, p=1.0)
    else:
        rho, u, v, p = vortex_fields(n[0], L, 0.0)
        s.set(rho=rho, u=u, v=v, p=p)                    # in 3D the same vortex in every plane
    dt = 0.4 / s.limits()["speed_over_h"]
    s.run_fixed(dt, 2)                                    # warm up
    ms = min(s.run_fixed(dt, steps)["ms"] for _ in range(3))
    lim = s.limits()
    cells, bytes_ = s.cells, s.bytes
    s.close()
    if not (lim["rho_min"] > 0 and lim["p_min"] > 0):
        raise RuntimeError("the timed run broke")
    rate = cells * steps / (ms / 1e3)
    return dict(n=list(n), cells=cells, precision=precision, recon=recon, flux=flux, steps=steps, ms_per_step=ms / steps,
                cell_steps_per_s=rate, ns_per_cell_step=1e9 / rate, bytes_per_cell=bytes_ / cells)


def main():
    busy = [p[3][:60] for p in foam.running()]
    rows = []
    grids = [(12800, 1, 1), (128, 128, 1), (512, 512, 1), (2048, 2048, 1), (4096, 4096, 1), (128, 128, 128), (256, 256, 256)]
    for n in grids:
        cells = int(np.prod(n))
        for precision in ("float", "mixed", "double"):
            for recon, flux in (("vanleer", "hybrid"), ("weno5", "rusanov")):
                if n[2] > 1 and (recon, flux) != ("vanleer", "hybrid"):
                    continue
                per = 2e8 if precision == "double" else 2e9          # about a second per timing
                steps = int(min(400, max(4, per / cells / 3)))
                r = timed(n, precision, recon, flux, steps)
                rows.append(r)
                print(f"  {'x'.join(str(m) for m in n if m > 1):>12s} {precision:6s} {recon:7s}/{flux:7s}: {r['cell_steps_per_s'] / 1e6:8.1f} million cells x steps / s, "
                      f"{r['ns_per_cell_step']:8.2f} ns per cell and step, {r['ms_per_step']:8.3f} ms per step, {r['bytes_per_cell']:.1f} bytes per cell", flush=True)
    REPORT.update(busy_cpu=bool(busy), busy_processes=busy, rows=rows)
    big = [r for r in rows if r["cells"] >= 10 ** 6]
    want = {"float": 60, "mixed": 120, "double": 120}
    off = max(abs(r["bytes_per_cell"] / want[r["precision"]] - 1) for r in big)
    check("rung 5: memory per cell is 15 stored numbers (60 bytes single, 120 mixed and double)", off < 0.05,
          f"largest difference {100 * off:.1f} % on grids of a million cells or more")
    pick = lambda p: next(r for r in rows if r["n"] == [4096, 4096, 1] and r["precision"] == p and r["recon"] == "vanleer")    # noqa: E731
    f, m, d = pick("float"), pick("mixed"), pick("double")
    check("rung 5: mixed precision at least twice as fast as double (4096 x 4096, van Leer, hybrid flux)",
          m["cell_steps_per_s"] >= 2 * d["cell_steps_per_s"],
          f"million cells x steps per second: single {f['cell_steps_per_s'] / 1e6:.0f}, mixed {m['cell_steps_per_s'] / 1e6:.0f}, double {d['cell_steps_per_s'] / 1e6:.0f}"
          + ("; an OpenFOAM process was running on the CPU" if busy else ""))
    REPORT["largest_2d"] = dict(float=f, mixed=m, double=d)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "test_euler_speed.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
