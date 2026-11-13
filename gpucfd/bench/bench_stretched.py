"""Stretched grids: what they cost and what they buy (DESIGN.md, "Stretched grids, stage 1").

    python -m gpucfd.bench.bench_stretched cost       pressure-solver cost against the cell aspect ratio
    python -m gpucfd.bench.bench_stretched benefit    cells and time to reach a given accuracy, cavity Re 100
    python -m gpucfd.bench.bench_stretched match      two more clustered grids, sized from the first study
    python -m gpucfd.bench.bench_stretched layer      a thin wall layer with an exact solution (Stokes' first problem)

cost: lid-driven cavity (Re 100, IMEX), cells clustered toward all four walls with largest / smallest
cell = 1, 4, 16, 64 (the aspect ratio of the cells at the middle of each wall). The flow is first
developed for 2 time units; then 200 steps are timed. Recorded: pressure iterations per projection,
diffusion iterations per solve, GPU time per step, with line relaxation and with the point smoother.

benefit: the same cavity to steady state (velocity change below 1e-7 U per time unit) on uniform and
clustered grids, IMEX, same CFL number. Error = largest difference of the two centreline profiles from
the uniform 512^2 solution (cubic splines through its points), at each grid's own points.
Results in gpucfd/bench/results/bench_stretched.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from gpucfd import gcfd, grids
from gpucfd.compare import cavity as cv

OUT = Path(__file__).parent / "results" / "bench_stretched.json"


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(d):
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(d, indent=1, default=float))


def cost_row(N, ratio, lines, ptol=1e-8, stretched_path=True):
    gcfd.set_lines(lines)
    try:
        g = grids.tanh(N, 1.0, ratio) if stretched_path else None
        s = gcfd.NavierStokes((N, N, 1), (1.0, 1.0, 1.0), cv.NU, walls=(True, True, False),
                              wall_velocity={"y+": (cv.U, 0, 0)}, scheme="imex", grid=None if g is None else (g, g, None))
        dt = s.max_dt(cv.U)
        t = 0.0
        while t < 2.0:                               # develop the flow
            s.run(dt, 50, ptol)
            t += 50 * dt
            dt = min(s.dt_cfl(0.8), 2 * dt)
        best = None
        for _ in range(3):                           # best of 3 (run-to-run noise about 10 %)
            st = s.run(dt, 200, ptol)
            if best is None or st["ms"] < best["ms"]:
                best = st
        w = s.widths(0)
        s.close()
    finally:
        gcfd.set_lines("xyz")
    return dict(N=N, ratio=ratio, lines=lines, path="stretched" if stretched_path else "uniform", dt=dt,
                smallest_cell=float(w.min()), largest_cell=float(w.max()),
                ms_per_step=best["ms"] / 200, poisson_its=best["poisson_its"], helm_its=best["helm_its"],
                max_div=best["max_div"])


def cost():
    rows = []
    for N in (128, 256):
        rows.append(cost_row(N, 1.0, "xyz", stretched_path=False))
        for ratio in (1.0, 4.0, 16.0, 64.0):
            rows.append(cost_row(N, ratio, "xyz"))
            if ratio > 1:
                rows.append(cost_row(N, ratio, ""))
        for r in rows[-8:]:
            print(f"N={r['N']:4d} ratio {r['ratio']:4.0f} {r['path']:9s} {'lines' if r['lines'] else 'point'}: "
                  f"{r['poisson_its']:6.1f} pressure its/projection, {r['helm_its']:5.1f} diffusion its, "
                  f"{r['ms_per_step']:7.2f} ms/step (dt {r['dt']:.2e})", flush=True)
    d = load()
    d["cost"] = rows
    save(d)


def line_error(r, ref):
    """Largest difference from the reference centrelines at the grid's own points (walls excluded)."""
    e = 0.0
    for key in ("line_u", "line_v"):
        sp = CubicSpline(ref[key][0], ref[key][1])
        x, v = np.asarray(r[key][0])[1:-1], np.asarray(r[key][1])[1:-1]
        e = max(e, float(np.max(np.abs(v - sp(x)))))
    return e


def layer():
    """Stokes' first problem (case 3): a wall set in motion at speed U under fluid at rest; exact solution
    u = U erfc(y / (2 sqrt(nu t))). Run to the time when the layer's 1 % depth is a tenth of the domain
    height, on uniform grids and on grids clustered toward the moving wall. Same time step everywhere
    (4000 steps, IMEX), so the time error is common; one run with 8000 steps measures it."""
    from scipy.special import erfc
    H, U, nu = 1.0, 1.0, 1.0
    delta = 0.1 * H / 3.643                       # erfc(3.643 / 2) = 0.01
    T = delta ** 2 / nu

    def run(N, ratio, steps=4000):
        g = grids.tanh(N, H, ratio, "low") if ratio > 1 else None
        s = gcfd.NavierStokes((8, N, 1), (1.0, H, 1.0), nu, walls=(False, True, False), wall_velocity={"y-": (U, 0, 0)},
                              scheme="imex", grid=None if g is None else (None, g, None))
        st = s.run(T / steps, steps)
        u = s.get()[0]
        Y = s.face_coords(0)[1]
        err = float(np.max(np.abs(u - U * erfc(Y / (2 * np.sqrt(nu * T))))))
        w = s.widths(1)
        s.close()
        return dict(N=N, ratio=ratio, steps=steps, error=err, smallest_cell=float(w.min()), largest_cell=float(w.max()),
                    cells_in_layer=int(np.sum(np.cumsum(w) <= 0.1 * H)), ms_per_step=st["ms"] / steps)
    rows = [run(N, 1.0) for N in (32, 64, 128, 256, 512)]
    rows += [run(N, r) for r in (16.0, 64.0) for N in (16, 32, 64, 128)]
    rows.append(run(128, 16.0, steps=8000))
    for r in rows:
        print(f"N={r['N']:4d} ratio {r['ratio']:3.0f} steps {r['steps']}: error {r['error']:.2e} U, {r['cells_in_layer']:3d} cells in "
              f"the layer, smallest cell {r['smallest_cell']:.1e}", flush=True)
    d = load()
    d["layer"] = dict(H=H, delta99=0.1 * H, T=T, rows=rows)
    save(d)


def benefit(cfl=2.0, cases=None):
    d = load()
    t0 = time.time()
    ref = cv.gpu_cavity(512, scheme="imex", cfl=cfl)
    print(f"reference 512^2: steady at t={ref['t_steady']:.0f}, {ref['steps']} steps, {ref['loop_s']:.1f} GPU-s "
          f"(wall {time.time() - t0:.0f} s)", flush=True)
    rows = [] if cases is None else d["benefit"]["rows"]
    if cases is None:
        cases = [(N, 1.0) for N in (32, 64, 128, 256)] + [(N, r) for r in (2.0, 4.0, 8.0) for N in (32, 64, 128)]
    for N, ratio in cases:
        try:
            r = cv.gpu_cavity(N, scheme="imex", cfl=cfl, ratio=ratio)
        except FloatingPointError:
            rows.append(dict(N=N, cells=N * N, ratio=ratio, cfl=cfl, blew_up=True))
            print(f"N={N:4d} ratio {ratio:3.0f}: blew up at CFL {cfl:g}", flush=True)
            continue
        row = dict(N=N, cells=N * N, ratio=ratio, cfl=cfl, error=line_error(r, ref), t_steady=r["t_steady"],
                   steps=r["steps"], gpu_s=r["loop_s"], wall_s=r["wall_s"], last_change=r["last_change"],
                   poisson_its=r["poisson_its"], helm_its=r["helm_its"],
                   smallest_cell=1.0 / N if ratio == 1 else float(np.diff(grids.tanh(N, 1.0, ratio)).min()))
        rows.append(row)
        print(f"N={N:4d} ratio {ratio:3.0f}: error {row['error']:.2e} U, {row['steps']:6d} steps, {row['gpu_s']:7.1f} GPU-s, "
              f"{1e3 * row['gpu_s'] / row['steps']:.2f} ms/step, pressure its {row['poisson_its']:.1f}"
              + ("" if r["last_change"] <= 1e-7 else f"  NOT STEADY (change {r['last_change']:.1e})"), flush=True)
        d["benefit"] = dict(cfl=cfl, reference=dict(N=512, steps=ref["steps"], gpu_s=ref["loop_s"], t_steady=ref["t_steady"]),
                            rows=rows)
        save(d)
    d["benefit"] = dict(cfl=cfl, reference=dict(N=512, steps=ref["steps"], gpu_s=ref["loop_s"], t_steady=ref["t_steady"]),
                        rows=rows)
    save(d)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "cost"
    if what == "cost":
        cost()
    elif what == "layer":
        layer()
    elif what == "match":   # clustered grids sized to reach the uniform 256^2 error (after the first study)
        benefit(2.0, cases=[(224, 2.0), (240, 2.0)])
    else:
        benefit(float(sys.argv[2]) if len(sys.argv) > 2 else 2.0)
