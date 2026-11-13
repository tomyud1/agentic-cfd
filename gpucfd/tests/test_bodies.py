"""Verification ladder, rung 10: immersed bodies (ghost-cell method) against exact solutions.

    python -m gpucfd.tests.test_bodies          (inside WSL, from the project root)

1. Force balance identity: forces() telescopes the solver's own discrete momentum fluxes, so for ANY
   velocity and pressure field with no body inside the box it must return zero (to round-off).
2. Circular Couette flow between a rotating inner cylinder and a fixed outer one, both immersed (the
   fluid region is closed: the outer cylinder is an Inside() body), exact steady Navier-Stokes solution
   u_theta = A r + B / r, p = int u_theta^2 / r dr. Curved boundaries on both sides, convex and concave,
   a moving surface. Velocity order 2; pressure order measured away from the walls; divergence at
   round-off (closed region: the net flow through the rim faces is removed exactly).
3. The same flow in 3D, extruded along z and along x (periodic along the axis): must equal the 2D
   solution to round-off (checks the trilinear stencils and the third component).
4. Poiseuille flow in a channel tilted at atan(1/2) to the grid (flat walls at a generic angle,
   periodic box, body force along the channel): order 2.
5. Start-up: the Couette flow started from rest reaches the same steady state.
6. DFG benchmark 2D-1 (cylinder in a channel, Re 20, steady): drag, lift and the pressure difference
   across the cylinder against the reference values (Nabh 1998, listed on the FeatFlow benchmark page),
   on three grids; forces from two different boxes must agree to round-off.
Bodies are run with the explicit scheme (IMEX with bodies is not stable yet: DESIGN.md).
"""
import json
import sys
from pathlib import Path

import numpy as np

from gpucfd import bodies as B
from gpucfd import gcfd

FAILS = []
REPORT = {}


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def order(errs):
    return [float(np.log2(errs[q] / errs[q + 1])) for q in range(len(errs) - 1)]


def mean_order(errs):
    """Order over the whole range of grids (for 3 grids doubling: the least-squares slope)."""
    return float(np.log2(errs[0] / errs[-1]) / (len(errs) - 1))


def ib_ordered(errs):
    """Immersed-boundary errors fluctuate with how the surface cuts each grid (the ghost faces' distances
    to it change from grid to grid), so a single pair can read 1.4 or 3: judged on the mean order over the
    range (>= 1.8), with no pair below 1.2."""
    return mean_order(errs) >= 1.8 and all(v >= 1.2 for v in order(errs))


def steady(s, dt, chunk=200, tol=1e-10, tmax=200.0, ptol=1e-13):
    prev = s.get()
    t = 0.0
    while t < tmax:
        st = s.run(dt, chunk, ptol)
        t += dt * chunk
        cur = s.get()
        change = max(np.max(np.abs(a - b)) for a, b in zip(cur, prev)) / (dt * chunk)
        prev = cur
        if change < tol:
            break
    return st, t


def face_errors(s, exact):
    """Max and rms error over the fluid faces of every component."""
    mx, sq, n = 0.0, 0.0, 0
    for c, a in enumerate(s.get()):
        X, Y, Z = s.face_coords(c)
        e = np.abs(a - exact(X, Y, Z)[c])[s.fluid[c]]
        mx, sq, n = max(mx, float(e.max())), sq + float(np.sum(e ** 2)), n + e.size
    return mx, float(np.sqrt(sq / n))


# ------------------------------------------------------------------------------------- 1. identity
def test_identity():
    s = gcfd.NavierStokes((32, 32, 1), (1.0, 1.0, 1.0), 0.03)
    rng = np.random.default_rng(1)
    s.set([rng.standard_normal(s.shape) for _ in range(2)])
    F = s.forces(((0.2, 0.7), (0.3, 0.8), (0, 1)))
    s.close()
    check("rung 10: forces() is the exact discrete momentum balance (no body: zero for a random field)",
          np.max(np.abs(F)) < 1e-12, f"F = ({F[0]:.1e}, {F[1]:.1e}) for a random field of size 1")
    REPORT["identity"] = float(np.max(np.abs(F)))


# --------------------------------------------------------------------------------------- 2. Couette
C0, R1, R2, W1, NU_C = (0.5, 0.5), 0.2, 0.45, 5.0, 0.025
A_C = -W1 * R1 ** 2 / (R2 ** 2 - R1 ** 2)
B_C = W1 * R1 ** 2 * R2 ** 2 / (R2 ** 2 - R1 ** 2)


def couette_exact(axis=2):
    a, b = [d for d in range(3) if d != axis]

    def f(X, Y, Z):
        P = (X, Y, Z)
        x, y = P[a] - 0.5, P[b] - 0.5
        r = np.sqrt(x * x + y * y)
        ut = A_C * r + B_C / r
        out = [0.0 * X, 0.0 * X, 0.0 * X]
        out[a], out[b] = -ut * y / r, ut * x / r
        return out
    return f


def couette_p(r2):
    return A_C ** 2 * r2 / 2 + A_C * B_C * np.log(r2) - B_C ** 2 / (2 * r2)


def couette(N, axis=2, start="exact", dim3=False, image=None):
    """axis 2: 2D (N, N, 1), or 3D (N, N, 4) with dim3; axis 0: (4, N, N). Periodic along the axis."""
    if axis == 2 and not dim3:
        n, length, walls = (N, N, 1), (1.0, 1.0, 1.0), (True, True, False)
    elif axis == 2:
        n, length, walls = (N, N, 4), (1.0, 1.0, 4.0 / N), (True, True, False)
    else:
        n, length, walls = (4, N, N), (4.0 / N, 1.0, 1.0), (False, True, True)
    ctr = [0.5, 0.5, 0.5]
    bod = [B.Cylinder(ctr, R1, axis=axis, velocity=B.rotation(ctr, W1, axis=axis)),
           B.Inside(B.Cylinder(ctr, R2, axis=axis))]
    s = gcfd.NavierStokes(n, length, NU_C, walls=walls, bodies=bod, image_distance=image)
    ex = couette_exact(axis)
    q = []
    for c in range(s.ncomp):
        X, Y, Z = s.face_coords(c)
        q.append(np.where(s.fluid[c], ex(X, Y, Z)[c], 0.0) if start != "rest" else np.zeros(s.shape))
    s.set(q)
    s.project()
    st, t = steady(s, s.max_dt(1.0))
    out = dict(u=s.get(), st=st, t=t)
    out["umax"], out["urms"] = face_errors(s, ex)
    X, Y, Z = s.cell_coords()
    ph, _ = B.phi(bod, X, Y, Z)
    P = (X, Y, Z)
    a, b = [d for d in range(3) if d != axis]
    pe = couette_p((P[a] - 0.5) ** 2 + (P[b] - 0.5) ** 2)
    p = s.pressure()
    hmax = 1.0 / N
    for key, m in (("p_all", ph > 0), ("p_far", ph > 1.5 * hmax)):
        d = (p - p[m].mean()) - (pe - pe[m].mean())
        out[key + "_max"], out[key + "_rms"] = float(np.abs(d[m]).max()), float(np.sqrt(np.mean(d[m] ** 2)))
    s.close()
    return out


def test_couette():
    Ns = (32, 64, 128)
    rs = [couette(N) for N in Ns]
    urms, umax = [r["urms"] for r in rs], [r["umax"] for r in rs]
    pr, pm = [r["p_far_rms"] for r in rs], [r["p_all_max"] for r in rs]
    o, om, op = order(urms), order(umax), order(pr)
    check("rung 10: circular Couette flow between immersed cylinders, velocity order 2 (rms over the fluid)",
          ib_ordered(urms),
          f"rms error {', '.join(f'{e:.2e}' for e in urms)} on N = {Ns}, order {', '.join(f'{v:.2f}' for v in o)} "
          f"(mean {mean_order(urms):.2f}); "
          f"max error {', '.join(f'{e:.2e}' for e in umax)}, order {', '.join(f'{v:.2f}' for v in om)}")
    pav = float(np.log2(pr[0] / pr[-1]) / (len(pr) - 1))
    check("rung 10: Couette pressure converges (rms over cells beyond 1.5 h of the walls, mean order >= 1.5)",
          pav >= 1.5,
          f"rms error {', '.join(f'{e:.2e}' for e in pr)}, order {', '.join(f'{v:.2f}' for v in op)} (mean {pav:.2f}); "
          f"max error over all fluid cells {', '.join(f'{e:.2e}' for e in pm)} (in the cells next to the walls)")
    # the projection stops at an rms divergence of ptol max|u| / h (ptol 1e-13 here), so judge relative to U / h
    dv = max(r["st"]["max_div"] / N for r, N in zip(rs, Ns))
    check("rung 10: Couette, closed fluid region: divergence at the solver tolerance (max |div u| h / U < 1e-11)",
          dv < 1e-11, f"max |div u| h / U {dv:.1e} (absolute {max(r['st']['max_div'] for r in rs):.1e})")
    REPORT["couette"] = dict(N=list(Ns), u_rms=urms, u_max=umax, order_rms=o, order_max=om, p_far_rms=pr, order_p=op,
                             p_all_max=pm, max_div=dv, poisson_its=[r["st"]["poisson_its"] for r in rs])
    return rs


def test_couette_3d():
    # 3D image points sit at (sqrt(3) + 0.05) h, 2D ones at (sqrt(2) + 0.05) h: a different (equally
    # valid) discrete problem, so the 2D reference here uses the 3D distance
    a = couette(32, image=np.sqrt(3) + 0.05)["u"]
    for axis, label in ((2, "extruded along z"), (0, "along x")):
        r = couette(32, axis=axis, dim3=True)
        u = r["u"]
        if axis == 2:  # (4, 32, 32) per component; compare every layer with 2D, w must be 0
            d = max(max(np.max(np.abs(u[c][k] - a[c][0])) for k in range(4)) for c in range(2))
            d = max(d, float(np.max(np.abs(u[2]))))
        else:  # axis x: 2D (x, y) maps to 3D (y, z); 2D u -> 3D v, 2D v -> 3D w; arrays (z, y, x)
            d = max(np.max(np.abs(u[1][:, :, k] - a[0][0])) for k in range(4))
            d = max(d, max(np.max(np.abs(u[2][:, :, k] - a[1][0])) for k in range(4)), float(np.max(np.abs(u[0]))))
        check(f"rung 10: Couette in 3D, axis {label}, equals the 2D solution", d < 1e-9, f"max difference {d:.1e} (N = 32)")
        REPORT[f"couette_3d_{'z' if axis == 2 else 'x'}"] = float(d)


def test_startup(r2d):
    r = couette(32, start="rest")
    d = max(np.max(np.abs(x - y)) for x, y in zip(r["u"], r2d[0]["u"]))
    check("rung 10: Couette started from rest reaches the same steady state", d < 1e-8,
          f"max difference {d:.1e} (reached after t = {r['t']:.1f})")
    REPORT["startup"] = float(d)


# ---------------------------------------------------------------------------------- 4. tilted channel
P_T = 2 / np.sqrt(5)  # period of the strip pattern across the channel (slope 1/2 in a 2 x 1 box)
W_T, NU_T = 0.6 * P_T, 0.1
G_T = 8 * NU_T / W_T ** 2  # centreline velocity 1
T_DIR = np.array([2.0, 1.0]) / np.sqrt(5)


def strip(X, Y):
    return np.mod((-X + 2 * Y) / np.sqrt(5), P_T)


def strip_sdf(X, Y, Z):  # exact signed distance to the periodic walls (fluid: 0 < s < W)
    s = strip(X, Y)
    return np.where(s < W_T, np.minimum(s, W_T - s), -np.minimum(s - W_T, P_T - s))


def tilted_exact(X, Y, Z):
    s = strip(X, Y)
    u = np.where(s < W_T, G_T / (2 * NU_T) * s * (W_T - s), 0.0)
    return (u * T_DIR[0], u * T_DIR[1])


def tilted(N):
    s = gcfd.NavierStokes((2 * N, N, 1), (2.0, 1.0, 1.0), NU_T, force=(G_T * T_DIR[0], G_T * T_DIR[1], 0.0),
                          bodies=[B.SDF(strip_sdf)])
    q = []
    for c in range(2):
        X, Y, Z = s.face_coords(c)
        q.append(np.where(s.fluid[c], tilted_exact(X, Y, Z)[c], 0.0))
    s.set(q)
    s.project()
    st, t = steady(s, s.max_dt(1.0))
    mx, rms = face_errors(s, tilted_exact)
    s.close()
    return mx, rms, st


def test_tilted():
    Ns = (32, 64, 128)
    rs = [tilted(N) for N in Ns]
    rms, mx = [r[1] for r in rs], [r[0] for r in rs]
    o, om = order(rms), order(mx)
    check("rung 10: Poiseuille flow in a channel at atan(1/2) to the grid, order 2 (rms over the fluid)",
          ib_ordered(rms),
          f"rms error {', '.join(f'{e:.2e}' for e in rms)} on N = {Ns}, order {', '.join(f'{v:.2f}' for v in o)} "
          f"(mean {mean_order(rms):.2f}); "
          f"max error {', '.join(f'{e:.2e}' for e in mx)}, order {', '.join(f'{v:.2f}' for v in om)}")
    REPORT["tilted"] = dict(N=list(Ns), u_rms=rms, u_max=mx, order_rms=o, order_max=om)


# ------------------------------------------------------------------------------------------ 6. DFG 2D-1
DFG = dict(H=0.41, L=2.2, D=0.1, xc=0.2, yc=0.2, Um=0.3, nu=1e-3)
DFG_REF = dict(cD=5.57953523384, cL=0.010618948146, dP=0.11752016697)


def dfg(k):
    H, D, Um = DFG["H"], DFG["D"], DFG["Um"]
    Ubar = 2 * Um / 3

    def inflow(X, Y, Z):
        return (4 * Um * Y * (H - Y) / H ** 2, 0 * X, 0 * X)
    n = (384 * 2 ** k, 72 * 2 ** k, 1)  # h_x / h_y = 1.006 (the channel is 2.2 x 0.41; 72 = 9 * 8 for multigrid)
    bnd = {"x-": ("inflow", inflow), "x+": "outflow", "y-": "wall", "y+": "wall"}
    s = gcfd.NavierStokes(n, (DFG["L"], H, 1.0), DFG["nu"], boundaries=bnd,
                          bodies=[B.Cylinder((DFG["xc"], DFG["yc"]), D / 2)])
    q = []
    for c in range(2):
        X, Y, Z = s.face_coords(c)
        q.append(np.where(s.fluid[c], inflow(X, Y, Z)[c], 0.0))
    s.set(q)
    s.project()
    dt = s.max_dt(1.5 * Um)
    chunk, t, prev, ms, steps = int(np.ceil(1.0 / dt)), 0.0, None, 0.0, 0
    box = ((0.1, 0.3), (0.1, 0.3), (0, 1))
    while t < 300:
        st = s.run(dt, chunk, 1e-10)
        t, ms, steps = t + dt * chunk, ms + st["ms"], steps + chunk
        F = s.forces(box)
        pa, pb = s.surface_pressure([[DFG["xc"] - D / 2, DFG["yc"], 0.5], [DFG["xc"] + D / 2, DFG["yc"], 0.5]])
        cur = (2 * F[0] / (Ubar ** 2 * D), 2 * F[1] / (Ubar ** 2 * D), pa - pb)
        if prev is not None and abs(cur[0] - prev[0]) < 1e-6 and abs(cur[1] - prev[1]) < 1e-7 and abs(cur[2] - prev[2]) < 1e-7:
            break
        prev = cur
    F2 = s.forces(((0.08, 0.45), (0.05, 0.37), (0, 1)))
    s.close()
    return dict(n=n, D_h=D / (H / n[1]), cD=cur[0], cL=cur[1], dP=cur[2], box_diff=float(np.max(np.abs(F2 - F))),
                t=t, gpu_s=ms / 1000, ms_step=ms / steps, max_div=st["max_div"])


def test_dfg(ks=(0, 1)):
    """Two grids in the ladder (about 7 minutes); the third (D/h 70) is a one-off run: an hour or more,
    because the explicit scheme's diffusion limit sets the time step there (DESIGN.md)."""
    rs = [dfg(k) for k in ks]
    for r in rs:
        print(f"      D/h {r['D_h']:.1f}: cD {r['cD']:.5f} cL {r['cL']:.5f} dP {r['dP']:.5f} (t {r['t']:.0f}, "
              f"GPU {r['gpu_s']:.0f} s, {r['ms_step']:.2f} ms/step)", flush=True)
    err = {q: [abs(r[q] / DFG_REF[q] - 1) for r in rs] for q in ("cD", "cL", "dP")}
    pd = order(err["cD"])
    check("rung 10: DFG 2D-1 drag converges to the reference (order >= 1.5)", all(v >= 1.5 for v in pd),
          f"cD {', '.join(f'{r['cD']:.4f}' for r in rs)} (reference {DFG_REF['cD']:.4f}): error "
          f"{', '.join(f'{100 * e:.2f}%' for e in err['cD'])}, order {', '.join(f'{v:.2f}' for v in pd)}; "
          f"D/h = {', '.join(f'{r['D_h']:.1f}' for r in rs)}")
    # The pressure at one point of an immersed surface depends on where the grid cuts the body there: with the
    # cylinder shifted by 0, 1/4, 1/2 and 3/4 of a cell at D/h 35 the error is 0.23, 1.90, 2.39 and 0.75 %
    # (tests/shift_study.py, 2026-11-13). The first form of this check (within 0.5 %) passed only because the
    # unshifted position happens to be the best of the four, so the limit is now that measured spread.
    lim = [5e-2 if r["D_h"] < 30 else 3e-2 for r in rs]
    check("rung 10: DFG 2D-1 pressure difference within its spread over cut positions (5 % below D/h 30, 3 % from there)",
          all(e < v for e, v in zip(err["dP"], lim)),
          f"dP {', '.join(f'{r['dP']:.5f}' for r in rs)} (reference {DFG_REF['dP']:.5f}): error "
          f"{', '.join(f'{100 * e:.2f}%' for e in err['dP'])}; one cut position each, measured spread at D/h 35: "
          f"0.23 to 2.39 %")
    # Lift is 0.2 % of the drag here, and at D/h 35 it moves by a factor 2 to 3 with the cut position (0.012 to
    # 0.029 over the four shifts of tests/shift_study.py). Its relative error from one position proves nothing,
    # so the check is on its size against the drag (first form: "lift error falls with refinement").
    check("rung 10: DFG 2D-1 lift stays small (below 2 % of the drag; its value depends on the cut position)",
          all(abs(r["cL"]) < 0.02 * r["cD"] for r in rs),
          f"cL {', '.join(f'{r['cL']:.5f}' for r in rs)} (reference {DFG_REF['cL']:.5f}): "
          f"{', '.join(f'{100 * abs(r['cL']) / r['cD']:.2f}%' for r in rs)} of the drag; error against the reference "
          f"{', '.join(f'{100 * e:.0f}%' for e in err['cL'])}")
    bd = max(r["box_diff"] for r in rs)
    check("rung 10: DFG 2D-1, forces from two different boxes agree", bd < 1e-12, f"max difference {bd:.1e}")
    REPORT["dfg_2d1"] = dict(runs=rs, rel_err=err, ref=DFG_REF)
    return rs, err


if __name__ == "__main__":
    test_identity()
    r2d = test_couette()
    test_couette_3d()
    test_startup(r2d)
    test_tilted()
    test_dfg()
    out = Path(__file__).parent / "results" / "test_bodies.json"
    out.write_text(json.dumps(dict(report=REPORT, fails=FAILS), indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))
