"""Verification ladder, rungs 0-5, for the GPU Poisson solver (see gpucfd/DESIGN.md).

    python -m gpucfd.tests.test_poisson          (inside WSL, from the project root)

Every check prints PASS/FAIL with its numbers; the exit code is the number of failures. The numpy
reference below is written independently of the CUDA code (ghost-cell padding instead of the
in-stencil boundary handling used on the GPU), so a shared mistake is unlikely.
"""
import json
import sys
from pathlib import Path

import numpy as np

from gpucfd import gcfd

FAILS = []
REPORT = {}


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------ independent numpy reference
def pad(p, axis, lo, hi):
    """One ghost layer along axis (numpy axis order z, y, x) with the boundary condition names."""
    first = np.take(p, [0], axis=axis)
    last = np.take(p, [-1], axis=axis)
    ghost = {"periodic": (last, first), "neumann": (first, last), "dirichlet": (-first, -last)}
    g_lo = ghost[lo][0] if lo != "periodic" else last
    g_hi = ghost[hi][1] if hi != "periodic" else first
    return np.concatenate([g_lo, p, g_hi], axis=axis)


def lap_ref(p, h, bc):
    """Cell-centred 7-point Laplacian. p: (nz, ny, nx); h: (hx, hy, hz); bc: 6 names x-, x+, y-, y+, z-, z+."""
    out = np.zeros_like(p)
    for d, axis in ((0, 2), (1, 1), (2, 0)):              # x is numpy axis 2
        if p.shape[axis] == 1:
            continue
        q = pad(p, axis, bc[2 * d], bc[2 * d + 1])
        n = p.shape[axis]
        out += (np.take(q, range(0, n), axis=axis) - 2 * p + np.take(q, range(2, n + 2), axis=axis)) / h[d] ** 2
    return out


def centres(n, length):
    return [(np.arange(n[d]) + 0.5) * length[d] / n[d] for d in range(3)]


# ------------------------------------------------------------------ manufactured solutions
PI = np.pi
# Exact solutions are products of 1D factors g_d with g_d'' = -k_d^2 g_d, so Laplacian(p) = -sum(k_d^2) p
# over the active directions. Each factor satisfies the boundary conditions of its direction.
CASES = {   # name: (bc, [(g_x, k_x), (g_y, k_y), (g_z, k_z)])
    "neumann": (["neumann"] * 6,
                [(lambda s: np.cos(PI * s), PI), (lambda s: np.cos(2 * PI * s), 2 * PI), (lambda s: np.cos(PI * s), PI)]),
    "dirichlet": (["dirichlet"] * 6,
                  [(lambda s: np.sin(PI * s), PI), (lambda s: np.sin(2 * PI * s), 2 * PI), (lambda s: np.sin(PI * s), PI)]),
    "periodic": (["periodic"] * 6,
                 [(lambda s: np.sin(2 * PI * s), 2 * PI), (lambda s: np.cos(2 * PI * s), 2 * PI),
                  (lambda s: np.sin(2 * PI * s + 0.3), 2 * PI)]),
    # x periodic; y: zero gradient at y = 0, zero value at y = 1; z: zero gradient
    "mixed": (["periodic", "periodic", "neumann", "dirichlet", "neumann", "neumann"],
              [(lambda s: np.sin(2 * PI * s), 2 * PI), (lambda s: np.cos(PI * s / 2), PI / 2), (lambda s: np.cos(PI * s), PI)]),
}


def fields(case, n, two_d):
    bc, fac = CASES[case]
    if two_d:
        n = (n[0], n[1], 1)
        bc = bc[:4] + ["periodic", "periodic"]
    xc, yc, zc = centres(n, (1.0, 1.0, 1.0))
    Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")
    p = fac[0][0](X) * fac[1][0](Y) * (fac[2][0](Z) if not two_d else 1.0)
    k2 = fac[0][1] ** 2 + fac[1][1] ** 2 + (fac[2][1] ** 2 if not two_d else 0.0)
    return n, bc, p, -k2 * p


def singular(bc, n):
    return not any(b == "dirichlet" and n[q // 2] > 1 for q, b in enumerate(bc))


# ------------------------------------------------------------------ rung 1: operator
def rung1():
    rng = np.random.default_rng(1)
    worst = {"double": 0.0, "float": 0.0}
    for case in CASES:
        for two_d in (False, True):
            n = (16, 8, 6) if not two_d else (16, 8, 1)
            n, bc, _, _ = fields(case, n, two_d)
            p = rng.standard_normal((n[2], n[1], n[0]))
            ref = lap_ref(p, [1.0 / n[0], 0.5 / n[1], 2.0 / n[2]], bc)
            for prec in ("double", "float"):
                s = gcfd.Poisson(n, (1.0, 0.5, 2.0), bc, prec)
                err = np.max(np.abs(s.apply(p) - ref)) / np.max(np.abs(ref))
                worst[prec] = max(worst[prec], err)
    check("rung 1: GPU Laplacian = numpy reference (double)", worst["double"] < 1e-12, f"max rel. diff {worst['double']:.1e}")
    check("rung 1: GPU Laplacian = numpy reference (float)", worst["float"] < 1e-5, f"max rel. diff {worst['float']:.1e}")
    REPORT["rung1"] = worst


# ------------------------------------------------------------------ rung 2: solver vs direct solve
def rung2():
    rng = np.random.default_rng(2)
    worst = {}
    for case in CASES:
        for two_d in (False, True):
            n, bc, _, _ = fields(case, (8, 8, 4) if not two_d else (16, 8, 1), two_d)
            N = n[0] * n[1] * n[2]
            h = [1.0 / n[0], 1.0 / n[1], 1.0 / n[2]]
            A = np.column_stack([lap_ref(e.reshape(n[2], n[1], n[0]), h, bc).ravel() for e in np.eye(N)])
            f = rng.standard_normal(N)
            if singular(bc, n):
                f -= f.mean()
                xd = np.linalg.lstsq(A, f, rcond=None)[0]
                xd -= xd.mean()
            else:
                xd = np.linalg.solve(A, f)
            for method in ("mg", "pcg"):
                s = gcfd.Poisson(n, (1.0, 1.0, 1.0), bc, "double")
                x, hist, _ = s.solve(f.reshape(n[2], n[1], n[0]), tol=1e-12, maxit=200, method=method)
                err = np.max(np.abs(x.ravel() - xd)) / np.max(np.abs(xd))
                key = f"{case}{'_2d' if two_d else ''}_{method}"
                worst[key] = (err, len(hist) - 1)
    bad = {k: v for k, v in worst.items() if v[0] > 1e-9}
    w = max(v[0] for v in worst.values())
    check("rung 2: GPU solution = direct solve (all BC sets, 2D/3D, MG and PCG)", not bad,
          f"max rel. diff {w:.1e}" + (f"; failing {bad}" if bad else ""))
    REPORT["rung2"] = {k: v[0] for k, v in worst.items()}


# ------------------------------------------------------------------ rung 3: order of accuracy
def rung3():
    out = {}
    for case in CASES:
        for two_d in (False, True):
            Ns = (16, 32, 64, 128) if two_d else (8, 16, 32, 64)
            errs = []
            for N in Ns:
                n, bc, p, f = fields(case, (N, N, N), two_d)
                s = gcfd.Poisson(n, (1.0, 1.0, 1.0), bc, "double")
                x, hist, _ = s.solve(f, tol=1e-11, maxit=100, method="pcg")
                if singular(bc, n):
                    x, p = x - x.mean(), p - p.mean()
                errs.append(np.max(np.abs(x - p)))
            orders = [np.log2(errs[q] / errs[q + 1]) for q in range(len(errs) - 1)]
            key = f"{case}{'_2d' if two_d else '_3d'}"
            out[key] = dict(N=Ns, err=errs, order=orders)
            check(f"rung 3: order of accuracy {key}", 1.9 <= orders[-1] <= 2.1,
                  "errors " + ", ".join(f"{e:.2e}" for e in errs) + "; orders " + ", ".join(f"{o:.2f}" for o in orders))
    REPORT["rung3"] = out


# ------------------------------------------------------------------ rung 4: multigrid quality
def rung4():
    out = {}
    combos = [(c, m, "double") for c in ("neumann", "mixed") for m in ("mg", "pcg")] + \
             [("neumann", m, "mixed") for m in ("mg", "pcg")]
    for two_d, Ns in ((True, (32, 64, 128, 256, 512, 1024)), (False, (16, 32, 64, 128))):
        for case, method, prec in combos:
            its, rates = [], []
            if True:
                for N in Ns:
                    n, bc, p, f = fields(case, (N, N, N), two_d)
                    s = gcfd.Poisson(n, (1.0, 1.0, 1.0), bc, prec)
                    x, hist, _ = s.solve(f, tol=1e-10, maxit=200, method=method)
                    its.append(len(hist) - 1)
                    rates.append(float((hist[-1] / hist[0]) ** (1.0 / max(1, len(hist) - 1))))
                key = f"{case}_{'2d' if two_d else '3d'}_{method}_{prec}"
                out[key] = dict(N=Ns, iterations=its, rate=rates)
                ok = max(rates) < 0.2 and its[-1] - its[0] <= 3
                check(f"rung 4: multigrid {key}", ok,
                      f"iterations to 1e-10 for N={Ns}: {its}; mean reduction per iteration {max(rates):.3f} (worst)")
    REPORT["rung4"] = out


# ------------------------------------------------------------------ rung 5: single precision
def rung5():
    out = {}
    for case in CASES:
        for two_d, N in ((True, 256), (False, 64)):
            n, bc, p, f = fields(case, (N, N, N), two_d)
            xs, its = {}, {}
            for prec in ("double", "mixed", "float"):
                s = gcfd.Poisson(n, (1.0, 1.0, 1.0), bc, prec)
                xs[prec], hist, _ = s.solve(f, tol=1e-6 if prec == "float" else 1e-11, maxit=100, method="pcg")
                its[prec] = len(hist) - 1
                if singular(bc, n):
                    xs[prec] -= xs[prec].mean()
            if singular(bc, n):
                p = p - p.mean()
            disc = np.max(np.abs(xs["double"] - p))
            md = np.max(np.abs(xs["mixed"] - xs["double"]))
            fd = np.max(np.abs(xs["float"] - xs["double"]))
            key = f"{case}_{'2d' if two_d else '3d'}"
            out[key] = dict(N=N, mixed_minus_double=md, float_minus_double=fd, discretisation_error=disc, iterations=its)
            check(f"rung 5: mixed vs double {key}", md < 0.01 * disc,
                  f"|mixed - double| {md:.1e} vs discretisation error {disc:.1e}; iterations double/mixed "
                  f"{its['double']}/{its['mixed']}; (pure float, informative: {100 * fd / disc:.1f} % of it)")
    REPORT["rung5"] = out


if __name__ == "__main__":
    dev = gcfd.device()
    bw = gcfd.bandwidth()
    print(f"rung 0: {dev['name']} (compute capability {dev['compute_capability']}), measured copy bandwidth {bw:.0f} GB/s")
    REPORT["rung0"] = dict(device=dev, bandwidth_GBs=bw)
    for r in (rung1, rung2, rung3, rung4, rung5):
        r()
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "test_poisson.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))
