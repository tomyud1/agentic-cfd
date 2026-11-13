"""Verification ladder, rung 12: stretched grids (cells of unequal size, clustered toward walls).

    python -m gpucfd.tests.test_stretched [poisson] [ns] [cavity]     (inside WSL, from the project root)

Pressure solver (poisson):
1. The stretched code path given uniform face positions equals the uniform code to round-off.
2. The Laplacian equals an independent numpy reference (written from the cell positions, with ghost
   cells) to round-off, for every boundary set, 2D and 3D.
3. Solutions equal a dense direct solve on small grids.
4. Manufactured solutions converge at order 2.
5. Multigrid quality against the cell aspect ratio at the walls (1, 4, 16, 64): iterations do not grow
   with the grid and stay within twice the uniform count. The point smoother is recorded next to it.
Navier-Stokes (ns):
6. The stretched code path with uniform face positions equals the uniform code (both time schemes).
7. Convection conserves kinetic energy to round-off on a stretched grid (closed box, 2D and 3D).
8. Exact solutions at order 2: plane Poiseuille flow from rest, a decaying channel mode (2D and 3D),
   Kovasznay flow (with convection, clustered in both directions), a channel with inflow and outflow
   (flow out = flow in). Kovasznay and the open channel are studied with IMEX; the explicit scheme must
   reach the same steady state on one grid.
9. Two identical runs are bit-identical, and CUDA graphs on = off.
Cavity (cavity):
10. Lid-driven cavity at Re 100 on clustered grids: both time schemes reach the same steady state; the
    centrelines are within 0.01 U of Ghia et al. and converge to the uniform-grid solution.
"""
import json
import sys
from pathlib import Path

import numpy as np

from gpucfd import gcfd, grids
from gpucfd.tests import test_poisson as tp
from gpucfd.tests import test_open as to

FAILS = []
REPORT = {}
OUT = Path(__file__).parent / "results" / "test_stretched.json"


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def order(errs):
    return [float(np.log2(errs[q] / errs[q + 1])) for q in range(len(errs) - 1)]


# ------------------------------------------------------------------ independent numpy reference
def lap_ref_s(p, faces, bc):
    """Cell-centred Laplacian on a stretched grid, from positions: gradient between neighbouring cell
    centres (ghost centres mirrored beyond a side, values by the boundary rule), then the difference of
    the two face gradients over the cell width. p: (nz, ny, nx); faces: 3 arrays; bc: 6 names."""
    out = np.zeros_like(p)
    for d, axis in ((0, 2), (1, 1), (2, 0)):
        n = p.shape[axis]
        if n == 1:
            continue
        x = np.asarray(faces[d], float)
        xc = 0.5 * (x[:-1] + x[1:])
        q = tp.pad(p, axis, bc[2 * d], bc[2 * d + 1])
        if bc[2 * d] == "periodic":
            xg = np.concatenate(([xc[-1] - (x[-1] - x[0])], xc, [xc[0] + (x[-1] - x[0])]))
        else:
            xg = np.concatenate(([2 * x[0] - xc[0]], xc, [2 * x[-1] - xc[-1]]))
        shp = [1, 1, 1]
        shp[axis] = -1
        grad = np.diff(q, axis=axis) / np.diff(xg).reshape(shp)
        out += np.diff(grad, axis=axis) / np.diff(x).reshape(shp)
    return out


def volumes(faces, n):
    w = [np.diff(faces[d]) if n[d] > 1 else np.ones(1) for d in range(3)]
    return w[2][:, None, None] * w[1][None, :, None] * w[0][None, None, :]


def make_grid(n, length, bc, ratio):
    """Clustered faces in every bounded direction with more than one cell (x both ends, y toward the
    low end, z both ends with half the ratio); periodic directions stay uniform (None)."""
    g = []
    for d in range(3):
        if n[d] == 1 or bc[2 * d] == "periodic":
            g.append(None)
        else:
            g.append(grids.tanh(n[d], length[d], ratio if d < 2 else max(1.0, ratio / 2), "low" if d == 1 else "both"))
    return g


def full_faces(g, n, length):
    return [grids.uniform(n[d], length[d]) if g[d] is None else g[d] for d in range(3)]


def case_fields(case, n, two_d, ratio, length=(1.0, 1.0, 1.0)):
    bc, fac = tp.CASES[case]
    if two_d:
        n = (n[0], n[1], 1)
        bc = bc[:4] + ["periodic", "periodic"]
    g = make_grid(n, length, bc, ratio)
    f = full_faces(g, n, length)
    c = [0.5 * (a[:-1] + a[1:]) for a in f]
    Z, Y, X = np.meshgrid(c[2], c[1], c[0], indexing="ij")
    p = fac[0][0](X) * fac[1][0](Y) * (fac[2][0](Z) if not two_d else 1.0)
    k2 = fac[0][1] ** 2 + fac[1][1] ** 2 + (fac[2][1] ** 2 if not two_d else 0.0)
    return n, bc, g, f, p, -k2 * p


STRETCHED_CASES = ("neumann", "dirichlet", "mixed")   # "periodic" has no bounded direction to stretch


# ------------------------------------------------------------------ 1: uniform faces = uniform code
def p_uniform():
    rng = np.random.default_rng(11)
    worst_a, worst_s = 0.0, 0.0
    for case in STRETCHED_CASES:
        for two_d in (False, True):
            n = (16, 8, 8) if not two_d else (32, 16, 1)
            n, bc, _, _ = tp.fields(case, n, two_d)
            length = (1.0, 0.5, 2.0)
            g = [None if (n[d] == 1 or bc[2 * d] == "periodic") else grids.uniform(n[d], length[d]) for d in range(3)]
            p = rng.standard_normal((n[2], n[1], n[0]))
            a = gcfd.Poisson(n, length, bc, "double")
            b = gcfd.Poisson(n, length, bc, "double", grid=g)
            ra, rb = a.apply(p), b.apply(p)
            worst_a = max(worst_a, np.max(np.abs(ra - rb)) / np.max(np.abs(ra)))
            f = ra if tp.singular(bc, n) else rng.standard_normal(p.shape)   # ra: in the range of the operator
            xa, _, _ = a.solve(f, tol=1e-12, maxit=100)
            xb, hb, _ = b.solve(f, tol=1e-12, maxit=100)
            worst_s = max(worst_s, np.max(np.abs(xa - xb)) / np.max(np.abs(xa)))
    check("rung 12: stretched path with uniform faces = uniform Laplacian", worst_a < 1e-12, f"max rel. diff {worst_a:.1e}")
    check("rung 12: stretched path with uniform faces = uniform solution", worst_s < 1e-9, f"max rel. diff {worst_s:.1e}")
    REPORT["p_uniform"] = dict(apply=worst_a, solve=worst_s)


# ------------------------------------------------------------------ 2: operator
def p_operator():
    rng = np.random.default_rng(12)
    worst = {"double": 0.0, "float": 0.0}
    for case in STRETCHED_CASES:
        for two_d in (False, True):
            n, bc, g, f, _, _ = case_fields(case, (16, 8, 6), two_d, 12.0, (1.0, 0.5, 2.0))
            p = rng.standard_normal((n[2], n[1], n[0]))
            ref = lap_ref_s(p, f, bc)
            for prec in ("double", "float"):
                s = gcfd.Poisson(n, (1.0, 0.5, 2.0), bc, prec, grid=g)
                worst[prec] = max(worst[prec], np.max(np.abs(s.apply(p) - ref)) / np.max(np.abs(ref)))
    check("rung 12: stretched Laplacian = numpy reference (double)", worst["double"] < 1e-12, f"max rel. diff {worst['double']:.1e}")
    check("rung 12: stretched Laplacian = numpy reference (float)", worst["float"] < 1e-5, f"max rel. diff {worst['float']:.1e}")
    REPORT["p_operator"] = worst


# ------------------------------------------------------------------ 3: direct solve
def p_direct():
    rng = np.random.default_rng(13)
    worst = {}
    for case in STRETCHED_CASES:
        for two_d in (False, True):
            n, bc, g, fc, _, _ = case_fields(case, (8, 8, 4) if not two_d else (16, 8, 1), two_d, 10.0)
            N = n[0] * n[1] * n[2]
            A = np.column_stack([lap_ref_s(e.reshape(n[2], n[1], n[0]), fc, bc).ravel() for e in np.eye(N)])
            V = volumes(fc, n).ravel()
            f = rng.standard_normal(N)
            if tp.singular(bc, n):
                f -= np.sum(V * f) / np.sum(V)          # solvable: zero volume-weighted sum
                xd = np.linalg.lstsq(A, f, rcond=None)[0]
                xd -= np.sum(V * xd) / np.sum(V)
            else:
                xd = np.linalg.solve(A, f)
            for method in ("mg", "pcg"):
                for prec in ("double", "mixed"):
                    s = gcfd.Poisson(n, (1.0, 1.0, 1.0), bc, prec, grid=g)
                    x, hist, _ = s.solve(f.reshape(n[2], n[1], n[0]), tol=1e-12, maxit=200, method=method)
                    err = np.max(np.abs(x.ravel() - xd)) / np.max(np.abs(xd))
                    worst[f"{case}{'_2d' if two_d else ''}_{method}_{prec}"] = (err, len(hist) - 1)
    bad = {k: v for k, v in worst.items() if v[0] > 1e-9}
    w = max(v[0] for v in worst.values())
    check("rung 12: stretched solution = direct solve (all BC sets, 2D/3D, MG and PCG, double and mixed)", not bad,
          f"max rel. diff {w:.1e}; iterations {min(v[1] for v in worst.values())}-{max(v[1] for v in worst.values())}"
          + (f"; failing {bad}" if bad else ""))
    REPORT["p_direct"] = {k: dict(err=v[0], its=v[1]) for k, v in worst.items()}


# ------------------------------------------------------------------ 4: order of accuracy
def p_order(ratio=8.0):
    out = {}
    for case in STRETCHED_CASES:
        for two_d in (False, True):
            Ns = (16, 32, 64, 128) if two_d else (8, 16, 32, 64)
            errs = []
            for N in Ns:
                n, bc, g, fc, p, f = case_fields(case, (N, N, N), two_d, ratio)
                V = volumes(fc, n)
                s = gcfd.Poisson(n, (1.0, 1.0, 1.0), bc, "double", grid=g)
                if tp.singular(bc, n):
                    f = f - np.sum(V * f) / np.sum(V)   # the sampled right-hand side sums to zero only to O(h^2)
                x, hist, _ = s.solve(f, tol=1e-11, maxit=100, method="pcg")
                if tp.singular(bc, n):
                    x, p = x - np.sum(V * x) / np.sum(V), p - np.sum(V * p) / np.sum(V)
                errs.append(float(np.max(np.abs(x - p))))
            o = order(errs)
            key = f"{case}{'_2d' if two_d else '_3d'}"
            out[key] = dict(N=Ns, err=errs, order=o)
            check(f"rung 12: order of accuracy on clustered grids (ratio {ratio:g}) {key}", 1.9 <= o[-1] <= 2.1,
                  "errors " + ", ".join(f"{e:.2e}" for e in errs) + "; orders " + ", ".join(f"{v:.2f}" for v in o))
    REPORT["p_order"] = out


# ------------------------------------------------------------------ 5: multigrid quality vs aspect ratio
def mg_run(N, ratio, method, prec, lines="xyz", two_d=True, nu=(1, 1), case="neumann"):
    gcfd.set_lines(lines)
    try:
        n, bc, g, fc, p, f = case_fields(case, (N, N, N), two_d, ratio)
        if ratio > 1:   # cavity-like: both ends in every stretched direction
            g = [None if a is None else grids.tanh(n[d], 1.0, ratio) for d, a in enumerate(g)]
            fc = full_faces(g, n, (1.0, 1.0, 1.0))
            c = [0.5 * (a[:-1] + a[1:]) for a in fc]
            Z, Y, X = np.meshgrid(c[2], c[1], c[0], indexing="ij")
            fac = tp.CASES[case][1]
            f = -(fac[0][1] ** 2 + fac[1][1] ** 2 + (0 if two_d else fac[2][1] ** 2)) * fac[0][0](X) * fac[1][0](Y) * (1 if two_d else fac[2][0](Z))
        V = volumes(fc, n)
        if tp.singular(bc, n):
            f = f - np.sum(V * f) / np.sum(V)
        s = gcfd.Poisson(n, (1.0, 1.0, 1.0), bc, prec, nu=nu if ratio > 1 else (2, 2), grid=g if ratio > 1 else None)
        x, hist, ms = s.solve(f, tol=1e-10, maxit=200, method=method)
        vc = s.bench_vcycle(20)
        s.close()
    finally:
        gcfd.set_lines("xyz")
    its = len(hist) - 1
    return dict(its=its, rate=float((hist[-1] / hist[0]) ** (1.0 / max(1, its))), ms=ms, vcycle_ms=vc)


def p_multigrid():
    out = {}
    ratios = (1, 4, 16, 64)
    Ns = (64, 128, 256, 512)
    for method, prec in (("pcg", "mixed"), ("mg", "mixed"), ("pcg", "double")):
        base = None
        for r in ratios:
            rows = [mg_run(N, float(r), method, prec) for N in Ns]
            its = [q["its"] for q in rows]
            if r == 1:
                base = max(its)
            key = f"2d_{method}_{prec}_ratio{r}"
            out[key] = dict(N=Ns, iterations=its, rate=[q["rate"] for q in rows], vcycle_ms=[q["vcycle_ms"] for q in rows],
                            solve_ms=[q["ms"] for q in rows])
            ok = its[-1] - its[0] <= 3 and max(its) <= 2 * base
            check(f"rung 12: multigrid with line relaxation, 2D, wall aspect ratio {r}, {method} {prec}", ok,
                  f"iterations to 1e-10 for N={Ns}: {its} (uniform: at most {base}); V-cycle {rows[-1]['vcycle_ms']:.2f} ms at N={Ns[-1]}")
    # the point smoother on the same grids: what line relaxation is for (recorded, not judged)
    for r in ratios[1:]:
        rows = [mg_run(N, float(r), "pcg", "mixed", lines="", nu=(2, 2)) for N in (64, 256)]
        out[f"2d_pcg_mixed_ratio{r}_point"] = dict(N=(64, 256), iterations=[q["its"] for q in rows],
                                                   vcycle_ms=[q["vcycle_ms"] for q in rows], solve_ms=[q["ms"] for q in rows])
        print(f"info  point smoother, 2D, wall aspect ratio {r}, pcg mixed: iterations {[q['its'] for q in rows]} for N=(64, 256)"
              " (200 = not converged)", flush=True)
    # 3D: a duct clustered toward all its walls (thin in two directions along the edges)
    for r in (1, 4, 16):
        rows = [mg_run(N, float(r), "pcg", "mixed", two_d=False) for N in (32, 64, 128)]
        its = [q["its"] for q in rows]
        out[f"3d_pcg_mixed_ratio{r}"] = dict(N=(32, 64, 128), iterations=its, vcycle_ms=[q["vcycle_ms"] for q in rows],
                                             solve_ms=[q["ms"] for q in rows])
        print(f"info  line relaxation, 3D box clustered toward all walls, ratio {r}, pcg mixed: iterations {its} for "
              f"N=(32, 64, 128); V-cycle {rows[-1]['vcycle_ms']:.2f} ms at 128^3", flush=True)
    REPORT["p_multigrid"] = out


# ================================================================================================
# Navier-Stokes
# ================================================================================================
def ns_uniform():
    """Same set-ups with and without (uniform) face positions: the new kernels against the old ones."""
    worst = {}
    for scheme in ("rk3", "imex"):
        res = []
        for grid in (None, "uniform"):
            out = []
            N = 32
            g = None if grid is None else (grids.uniform(N, 1.0), grids.uniform(N, 1.0), None)
            s = gcfd.NavierStokes((N, N, 1), (1.0, 1.0, 1.0), 0.01, walls=(True, True, False),
                                  wall_velocity={"y+": (1.0, 0, 0)}, scheme=scheme, grid=g, htol=1e-13)
            s.run(s.max_dt(1.0), 100, 1e-12)
            out += s.get()
            s.close()
            prof = lambda X, Y, Z: (4 * Y * (1 - Y), 0.0 * X, 0.0 * X)
            g = None if grid is None else (grids.uniform(32, 2.0), grids.uniform(16, 1.0), None)
            s = gcfd.NavierStokes((32, 16, 1), (2.0, 1.0, 1.0), 0.1, scheme=scheme, grid=g, htol=1e-13,
                                  boundaries={"x-": ("inflow", prof), "x+": "outflow", "y-": "wall", "y+": "wall"})
            s.run(s.max_dt(1.0), 100, 1e-12)
            out += s.get() + [s.get_boundary("x+", 0)]
            s.close()
            g = None if grid is None else (grids.uniform(16, 1.0), None, grids.uniform(8, 1.0))   # 3D: walls x, z; y periodic
            s = gcfd.NavierStokes((16, 8, 8), (1.0, 1.0, 1.0), 0.05, walls=(True, False, True), force=(0.0, 1.0, 0.0),
                                  wall_velocity={"x+": (0.0, 0.5, 0.3)}, scheme=scheme, grid=g, htol=1e-13)
            q = [0.1 * np.sin(2 * np.pi * Y) * np.sin(np.pi * X) * np.sin(np.pi * Z) for X, Y, Z in (s.face_coords(c) for c in range(3))]
            s.set(q)
            s.project()
            s.run(0.5 * s.max_dt(1.0), 50, 1e-12)
            out += s.get()
            s.close()
            res.append(out)
        worst[scheme] = max(float(np.max(np.abs(a - b))) for a, b in zip(*res))
        check(f"rung 12: stretched path with uniform faces = uniform Navier-Stokes ({scheme}; cavity, open channel, 3D)",
              worst[scheme] < 1e-9, f"max velocity difference {worst[scheme]:.1e}")
    REPORT["ns_uniform"] = worst


def ns_energy():
    """Convection must not create or destroy kinetic energy: for a divergence-free velocity in a closed
    box, sum over all velocity unknowns of u * (convection term) * (box volume) = 0 to round-off."""
    out = {}
    rng = np.random.default_rng(14)
    for dims, n in ((2, (24, 16, 1)), (3, (12, 16, 8))):
        g = (grids.tanh(n[0], 1.0, 9.0), grids.tanh(n[1], 1.0, 20.0, "low"), grids.tanh(n[2], 1.0, 5.0) if dims == 3 else None)
        s = gcfd.NavierStokes(n, (1.0, 1.0, 1.0), 0.0, walls=(True, True, dims == 3), grid=g, pressure="double")
        s.set([rng.standard_normal(s.shape) for _ in range(s.ncomp)])
        s.run(1e-3, 0, 1e-14)                       # sets the projection tolerance
        div = min(s.project() for _ in range(3))
        q = s.get()
        tot, mag = 0.0, 0.0
        for c in range(s.ncomp):
            w = []
            for d in range(3):
                if n[d] == 1:
                    w.append(np.ones(1))
                elif d == c:                        # box of a face: from cell centre to cell centre
                    xc = 0.5 * (g[d][:-1] + g[d][1:])
                    w.append(np.concatenate(([0.0], np.diff(xc))))   # face 0 is the wall (not evolved)
                else:
                    w.append(np.diff(g[d]))
            vol = w[2][:, None, None] * w[1][None, :, None] * w[0][None, None, :]
            t = q[c] * s.rhs(c) * vol
            tot += float(np.sum(t))
            mag += float(np.sum(np.abs(t)))
        s.close()
        out[f"{dims}d"] = dict(relative=abs(tot) / mag, max_div=div)
        check(f"rung 12: convection conserves kinetic energy on a stretched grid ({dims}D)", abs(tot) / mag < 1e-10,
              f"|sum u C(u) V| / sum |u C(u) V| = {abs(tot) / mag:.1e} (max |div u| {div:.1e})")
    REPORT["ns_energy"] = out


def poiseuille(Ns, ratio, scheme, nu=1.0, g=8.0, T=2.5):
    errs, divs = [], []
    for N in Ns:
        y = grids.tanh(N, 1.0, ratio)
        s = gcfd.NavierStokes((8, N, 1), (1.0, 1.0, 1.0), nu, walls=(False, True, False), force=(g, 0, 0),
                              scheme=scheme, grid=(None, y, None))
        hmin = float(np.diff(y).min())
        steps = int(np.ceil(T / (0.12 * hmin ** 2 / nu if scheme == "rk3" else 0.5 / N)))
        st = s.run(T / steps, steps)
        u, v = s.get()
        X, Y, Z = s.face_coords(0)
        errs.append(float(np.max(np.abs(u - g * Y * (1 - Y) / (2 * nu)))))
        divs.append(st["max_div"])
        s.close()
    o = order(errs)
    check(f"rung 12: Poiseuille flow from rest, walls clustered (ratio {ratio:g}), {scheme}, order 2",
          1.9 <= o[-1] <= 2.1 and max(divs) < 1e-8,
          "errors " + ", ".join(f"{e:.2e}" for e in errs) + f" (N={Ns}); orders " + ", ".join(f"{v:.2f}" for v in o)
          + f"; max |div u| {max(divs):.1e}")
    REPORT[f"poiseuille_{scheme}"] = dict(N=Ns, ratio=ratio, err=errs, order=o)


def channel_mode(dims, Ns, ratio, scheme, nu=0.1, T=0.5):
    """u = sin(pi y) exp(-nu pi^2 t) between walls (2D: u(y); 3D: w(x), walls in x)."""
    errs = []
    for N in Ns:
        yg = grids.tanh(N, 1.0, ratio)
        if dims == 2:
            n, walls, comp, along, g = (8, N, 1), (False, True, False), 0, 1, (None, yg, None)
        else:
            n, walls, comp, along, g = (N, 8, 8), (True, False, False), 2, 0, (yg, None, None)
        s = gcfd.NavierStokes(n, (1.0, 1.0, 1.0), nu, walls=walls, scheme=scheme, grid=g)
        pos = s.face_coords(comp)[along]
        q = [np.zeros(s.shape) for _ in range(s.ncomp)]
        q[comp] = np.sin(np.pi * pos)
        s.set(q)
        hmin = float(np.diff(yg).min())
        steps = int(np.ceil(T / (0.1 * hmin ** 2 / nu if scheme == "rk3" else 0.25 / N)))
        s.run(T / steps, steps)
        out = s.get()
        errs.append(float(np.max(np.abs(out[comp] - np.sin(np.pi * pos) * np.exp(-nu * np.pi ** 2 * T)))))
        s.close()
    o = order(errs)
    check(f"rung 12: decaying channel mode {dims}D, walls clustered (ratio {ratio:g}), {scheme}, order 2", 1.9 <= o[-1] <= 2.1,
          "errors " + ", ".join(f"{e:.2e}" for e in errs) + f" (N={Ns}); orders " + ", ".join(f"{v:.2f}" for v in o))
    REPORT[f"channel_mode_{dims}d_{scheme}"] = dict(N=Ns, ratio=ratio, err=errs, order=o)


def kovasznay(N, scheme, ratio, Re=40.0):
    f = to.kovasznay_exact(Re)
    x0, y0, L = -0.5, -0.5, 2.0

    grid = (grids.tanh(N, L, ratio, "low"), grids.tanh(N, L, ratio), None)   # toward the inlet, and toward y-, y+
    # With velocities prescribed on every side, the discrete flow in must equal the flow out exactly. The
    # exact profiles sampled at the face centres of a clustered grid balance only to O(h^2), so the
    # outlet (x+) is shifted by the missing flow (itself O(h^2), like the discretisation error).
    yc, dy = 0.5 * (grid[1][:-1] + grid[1][1:]), np.diff(grid[1])
    xc, dx = 0.5 * (grid[0][:-1] + grid[0][1:]), np.diff(grid[0])
    net = (np.sum((f(L + x0, yc + y0, 0)[0] - f(x0, yc + y0, 0)[0]) * dy)
           + np.sum((f(xc + x0, L + y0, 0)[1] - f(xc + x0, y0, 0)[1]) * dx))

    def g(X, Y, Z):
        u, v, w = f(X + x0, Y + y0, Z)
        return (u - (net / L) * (np.asarray(X) >= L * (1 - 1e-12)), v, w)
    s = gcfd.NavierStokes((N, N, 1), (L, L, 1.0), 1.0 / Re, boundaries={k: ("inflow", g) for k in ("x-", "x+", "y-", "y+")},
                          scheme=scheme, grid=grid)
    exact = [f(X + x0, Y + y0, Z)[c] for c, (X, Y, Z) in enumerate(s.face_coords(c) for c in range(2))]
    s.set(exact)
    s.project()
    dt = s.dt_cfl(0.8)
    st, t, change = to.steady(s, dt, chunk=500, tol=1e-11)
    out = s.get()
    wts = []
    for c in range(2):   # box areas for the L2 norm
        w = [np.diff(grid[d]) if d != c else np.concatenate(([0.0], np.diff(0.5 * (grid[d][:-1] + grid[d][1:])))) for d in range(2)]
        wts.append(w[1][None, :, None] * w[0][None, None, :])
    err = max(float(np.max(np.abs(a - b))) for a, b in zip(out, exact))
    l2 = float(np.sqrt(sum(np.sum(w * (a - b) ** 2) for w, a, b in zip(wts, out, exact)) / sum(np.sum(w) for w in wts)))
    s.close()
    return err, l2, st, t, out


def same_steady(name, a, b):
    """The explicit scheme (diffusion in the momentum kernel) and IMEX (diffusion in its own kernel)
    solve the same discrete equations, so their steady states must agree. This carries the order
    measured with IMEX over to the explicit scheme, whose time step on a clustered grid (0.15 h_min^2 /
    nu) makes a three-grid study slow."""
    d = max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))
    check(f"rung 12: {name}, explicit scheme reaches the IMEX steady state", d < 1e-7, f"max difference {d:.1e}")
    REPORT[f"same_{name.split()[0].lower()}"] = d


def test_kovasznay(scheme, Ns, ratio=4.0):
    rs = [kovasznay(N, scheme, ratio) for N in Ns]
    same_steady("Kovasznay on a clustered grid (N=32)", rs[0][4], kovasznay(32, "rk3", ratio)[4])
    errs, l2 = [r[0] for r in rs], [r[1] for r in rs]
    p, p2 = order(errs), order(l2)
    check(f"rung 12: Kovasznay Re 40 on a grid clustered in x and y (ratio {ratio:g}), {scheme}, order 2",
          1.9 <= p2[-1] <= 2.1 and min(p) > 1.7,
          f"L2 error {', '.join(f'{e:.2e}' for e in l2)}, order {', '.join(f'{v:.2f}' for v in p2)}; max error "
          f"{', '.join(f'{e:.2e}' for e in errs)}, order {', '.join(f'{v:.2f}' for v in p)} on N = {Ns}; "
          f"pressure iterations {rs[-1][2]['poisson_its']:.1f}, max |div u| {rs[-1][2]['max_div']:.1e}")
    REPORT[f"kovasznay_{scheme}"] = dict(N=list(Ns), ratio=ratio, err=errs, order=p, l2=l2, order_l2=p2,
                                         poisson_its=rs[-1][2]["poisson_its"], helm_its=rs[-1][2]["helm_its"])


def open_channel(N, scheme, ratio=8.0, nu=0.1, Um=1.0):
    """Poiseuille channel, parabolic inflow at x-, outflow at x+, walls clustered; x clustered to the inlet."""
    grid = (grids.tanh(2 * N, 2.0, 2.0, "low"), grids.tanh(N, 1.0, ratio), None)
    prof = lambda X, Y, Z: (4 * Um * Y * (1 - Y), 0.0 * X, 0.0 * X)
    s = gcfd.NavierStokes((2 * N, N, 1), (2.0, 1.0, 1.0), nu, scheme=scheme, grid=grid,
                          boundaries={"x-": ("inflow", prof), "x+": "outflow", "y-": "wall", "y+": "wall"})
    exact = [prof(*s.face_coords(c))[c] * np.ones(s.shape) for c in range(2)]
    s.set(exact)
    s.project()
    st, t, change = to.steady(s, s.dt_cfl(0.8))
    out = s.get()
    err = max(float(np.max(np.abs(a - b))) for a, b in zip(out, exact))
    fin, fout = -s.flux("x-"), s.flux("x+")
    s.close()
    return dict(err=err, flux_in=fin, flux_out=fout, max_div=st["max_div"], its=st["poisson_its"], out=out)


def test_open_channel(scheme, Ns):
    rs = [open_channel(N, scheme) for N in Ns]
    errs = [r["err"] for r in rs]
    p = order(errs)
    bal = abs(rs[-1]["flux_out"] - rs[-1]["flux_in"]) / rs[-1]["flux_in"]
    check(f"rung 12: channel with inflow + outflow on a clustered grid, {scheme}, order 2, flow out = flow in",
          to.ordered(p) and bal < 1e-9 and rs[-1]["max_div"] < 1e-8,
          f"max error {', '.join(f'{e:.2e}' for e in errs)} on N = {Ns}; order {', '.join(f'{v:.2f}' for v in p)}; "
          f"in {rs[-1]['flux_in']:.12f}, out {rs[-1]['flux_out']:.12f} (relative {bal:.1e}); max |div u| {rs[-1]['max_div']:.1e}")
    REPORT[f"open_channel_{scheme}"] = dict(N=list(Ns), err=errs, order=p, flux_balance=bal, max_div=rs[-1]["max_div"])
    same_steady("open channel on a clustered grid (N=16)", rs[0]["out"], open_channel(16, "rk3")["out"])


def ns_reproducible():
    def cav(scheme, graphs):
        gcfd.use_graphs(graphs)
        g = (grids.tanh(32, 1.0, 8.0), grids.tanh(32, 1.0, 8.0), None)
        s = gcfd.NavierStokes((32, 32, 1), (1.0, 1.0, 1.0), 0.01, walls=(True, True, False),
                              wall_velocity={"y+": (1.0, 0, 0)}, scheme=scheme, grid=g)
        s.run(s.max_dt(1.0), 100, 1e-6)
        q = s.get()
        s.close()
        return q

    def chan(scheme, graphs):
        gcfd.use_graphs(graphs)
        prof = lambda X, Y, Z: (4 * Y * (1 - Y), 0.0 * X, 0.0 * X)
        s = gcfd.NavierStokes((32, 16, 1), (2.0, 1.0, 1.0), 0.1, scheme=scheme, grid=(None, grids.tanh(16, 1.0, 6.0), None),
                              boundaries={"x-": ("inflow", prof), "x+": "outflow", "y-": "wall", "y+": "wall"})
        s.run(s.max_dt(1.0), 100, 1e-6)
        q = s.get() + [s.get_boundary("x+", 0)]
        s.close()
        return q
    same = lambda a, b: all(np.array_equal(x, y) for x, y in zip(a, b))
    for name, fn in (("cavity", cav), ("channel inflow/outflow", chan)):
        for scheme in ("rk3", "imex"):
            off1, off2, on = fn(scheme, False), fn(scheme, False), fn(scheme, True)
            check(f"rung 12: stretched {name} {scheme}, two runs bit-identical", same(off1, off2), "")
            check(f"rung 12: stretched {name} {scheme}, CUDA graphs on = off (bit-identical)", same(off1, on), "")
    gcfd.use_graphs(True)


# ================================================================================================
# Cavity
# ================================================================================================
def cavity():
    from gpucfd.compare import cavity as cv
    fine = np.linspace(0, 1, 401)
    # both time schemes reach the same steady state on a clustered grid (the explicit and the implicit
    # diffusion are separate kernels)
    a = cv.gpu_cavity(16, ptol=1e-10, settle=1e-10, scheme="rk3", ratio=4.0)
    b = cv.gpu_cavity(16, ptol=1e-10, settle=1e-10, scheme="imex", ratio=4.0)
    d = max(np.abs(cv.at_ghia(a["line_u"], fine) - cv.at_ghia(b["line_u"], fine)).max(),
            np.abs(cv.at_ghia(a["line_v"], fine) - cv.at_ghia(b["line_v"], fine)).max())
    check("rung 12: steady cavity on a clustered grid (N=16, ratio 4), IMEX = explicit", d < 1e-7,
          f"max centreline difference {d:.1e} U")
    REPORT["cavity_schemes"] = d
    # against Ghia and the uniform-grid solution
    gu, gv = cv.ghia()
    iu, iv = gu[1:-1], gv[1:-1]
    ref = cv.gpu_cavity(256, scheme="imex", cfl=2.0)                       # uniform reference
    rows = {}
    for ratio, Ns in ((4.0, (32, 64, 128)), (16.0, (32, 64))):
        diffs = []
        for N in Ns:
            r = cv.gpu_cavity(N, scheme="imex", ratio=ratio)
            du = np.abs(cv.at_ghia(r["line_u"], iu[:, 0]) - iu[:, 1]).max()
            dv = np.abs(cv.at_ghia(r["line_v"], iv[:, 0]) - iv[:, 1]).max()
            dref = max(np.abs(cv.at_ghia(r["line_u"], fine) - cv.at_ghia(ref["line_u"], fine)).max(),
                       np.abs(cv.at_ghia(r["line_v"], fine) - cv.at_ghia(ref["line_v"], fine)).max())
            diffs.append(dref)
            rows[f"ratio{ratio:g}_N{N}"] = dict(vs_ghia_u=float(du), vs_ghia_v=float(dv), vs_uniform256=float(dref),
                                                t_steady=r["t_steady"], steps=r["steps"], loop_s=r["loop_s"],
                                                poisson_its=r["poisson_its"], helm_its=r["helm_its"])
        last = rows[f"ratio{ratio:g}_N{Ns[-1]}"]
        ok = (max(last["vs_ghia_u"], last["vs_ghia_v"]) < 0.01 and all(a > b for a, b in zip(diffs[:-1], diffs[1:]))
              and diffs[-1] < 4e-3 * (64 / Ns[-1]) ** 2)
        check(f"rung 12: cavity Re 100 on clustered grids (ratio {ratio:g}): Ghia within 0.01 U, converges to the uniform solution",
              ok, f"vs Ghia at N={Ns[-1]}: u {last['vs_ghia_u']:.4f}, v {last['vs_ghia_v']:.4f} U; vs uniform 256^2 at N={Ns}: "
              + ", ".join(f"{v:.5f}" for v in diffs) + f" U; pressure iterations {last['poisson_its']:.1f}")
    REPORT["cavity"] = rows


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if a in ("poisson", "ns", "cavity")] or ["poisson", "ns", "cavity"]
    if OUT.exists():
        REPORT.update(json.loads(OUT.read_text()).get("report", {}))
    if "poisson" in which:
        for r in (p_uniform, p_operator, p_direct, p_order, p_multigrid):
            r()
    if "ns" in which:
        ns_uniform()
        ns_energy()
        poiseuille((8, 16, 32, 64), 8.0, "imex")
        poiseuille((8, 16, 32), 4.0, "rk3", T=1.5)
        for scheme in ("imex", "rk3"):
            channel_mode(2, (16, 32, 64), 4.0, scheme)
            channel_mode(3, (16, 32), 4.0, scheme) if scheme == "rk3" else channel_mode(3, (16, 32, 64), 4.0, scheme)
        test_kovasznay("imex", (32, 64, 128))
        test_open_channel("imex", (16, 32, 64))
        ns_reproducible()
    if "cavity" in which:
        cavity()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(dict(report=REPORT, fails=FAILS), indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))
