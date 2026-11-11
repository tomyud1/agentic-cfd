"""Verification ladder, rung 6: Navier-Stokes on a periodic box against exact solutions.

    python -m gpucfd.tests.test_ns          (inside WSL, from the project root)

- 2D Taylor-Green vortex: u = sin x cos y F, v = -cos x sin y F, F = exp(-2 nu t).
- 3D Arnold-Beltrami-Childress flow: u = (A sin z + C cos y, B sin x + A cos z, C sin y + B cos x) exp(-nu t).
  Its convection term is exactly a pressure gradient, so it tests convection, diffusion and the
  projection together in 3D.
Checks: spatial order 2 (tiny time step), temporal order (self-convergence on a fixed grid), divergence
at round-off, and mixed-precision pressure against double.
"""
import json
import sys
from pathlib import Path

import numpy as np

from gpucfd import gcfd

FAILS = []
REPORT = {}
TWO_PI = 2 * np.pi
NU = 0.05
A, B, C = 1.0, 0.7, 0.4


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def exact(flow, s, t):
    """Exact velocity components at the staggered positions of solver s at time t."""
    if flow == "tgv":
        X, Y, _ = s.face_coords(0)
        u = np.sin(X) * np.cos(Y)
        X, Y, _ = s.face_coords(1)
        v = -np.cos(X) * np.sin(Y)
        return [u * np.exp(-2 * NU * t), v * np.exp(-2 * NU * t)]
    X, Y, Z = s.face_coords(0)
    u = A * np.sin(Z) + C * np.cos(Y)
    X, Y, Z = s.face_coords(1)
    v = B * np.sin(X) + A * np.cos(Z)
    X, Y, Z = s.face_coords(2)
    w = C * np.sin(Y) + B * np.cos(X)
    return [q * np.exp(-NU * t) for q in (u, v, w)]


def simulate(flow, N, T, dt, pressure="mixed"):
    n = (N, N, 1) if flow == "tgv" else (N, N, N)
    s = gcfd.NavierStokes(n, (TWO_PI,) * 3, NU, pressure)
    s.set(exact(flow, s, 0.0))
    steps = int(round(T / dt))
    st = s.run(T / steps, steps)
    q = s.get()
    err = max(np.max(np.abs(a - b)) for a, b in zip(q, exact(flow, s, T)))
    umax = max(np.max(np.abs(a)) for a in q)
    s.close()
    return q, err, st, umax


def spatial(flow, Ns, T=1.0):
    errs, divs = [], []
    for N in Ns:
        h = TWO_PI / N
        dt = min(0.05 * h, 0.05 * h * h / NU)          # small: time error far below space error
        _, err, st, umax = simulate(flow, N, T, dt)
        errs.append(err)
        divs.append(st["max_div"] * h / umax)
    orders = [np.log2(errs[q] / errs[q + 1]) for q in range(len(errs) - 1)]
    check(f"rung 6: spatial order {flow}", 1.9 <= orders[-1] <= 2.1,
          "max errors " + ", ".join(f"{e:.2e}" for e in errs) + " (N=" + ",".join(map(str, Ns)) + "); orders "
          + ", ".join(f"{o:.2f}" for o in orders))
    check(f"rung 6: divergence-free {flow}", max(divs) < 1e-8, f"max |div u| h / max|u| = {max(divs):.1e}")
    REPORT[f"spatial_{flow}"] = dict(N=Ns, err=errs, order=orders, div=divs)


def temporal(flow, N, T=1.0):
    h = TWO_PI / N
    dts = [0.8 * h / 2 ** q for q in range(4)]          # Courant number 0.8 down to 0.1
    ref, _, _, _ = simulate(flow, N, T, dts[-1] / 8)
    errs = []
    for dt in dts:
        q, _, _, _ = simulate(flow, N, T, dt)
        errs.append(max(np.max(np.abs(a - b)) for a, b in zip(q, ref)))
    orders = [np.log2(errs[q] / errs[q + 1]) for q in range(len(errs) - 1)]
    check(f"rung 6: temporal order {flow}", min(orders) >= 1.9,
          "differences from dt/8 run " + ", ".join(f"{e:.2e}" for e in errs) + "; orders " + ", ".join(f"{o:.2f}" for o in orders))
    REPORT[f"temporal_{flow}"] = dict(dt=dts, err=errs, order=orders)


def precision(flow, N, T=1.0):
    h = TWO_PI / N
    dt = min(0.4 * h, 0.15 * h * h / NU)                 # convective and diffusive (RK3) stability limits
    qd, errd, std, _ = simulate(flow, N, T, dt, "double")
    qm, errm, stm, _ = simulate(flow, N, T, dt, "mixed")
    d = max(np.max(np.abs(a - b)) for a, b in zip(qd, qm))
    check(f"rung 6: mixed vs double pressure {flow}", d < 1e-3 * errd,
          f"|mixed - double| {d:.1e} vs discretisation error {errd:.1e}; GPU ms double {std['ms']:.0f}, mixed {stm['ms']:.0f}; "
          f"pressure iterations per projection {std['poisson_its']:.1f} / {stm['poisson_its']:.1f}")
    REPORT[f"precision_{flow}"] = dict(diff=d, err=errd, ms_double=std["ms"], ms_mixed=stm["ms"])


if __name__ == "__main__":
    spatial("tgv", (16, 32, 64, 128))
    spatial("abc", (16, 32, 64))
    temporal("tgv", 64)
    temporal("abc", 32)
    precision("tgv", 256)
    precision("abc", 64)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "test_ns.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))
