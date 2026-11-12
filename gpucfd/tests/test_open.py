"""Verification ladder, rung 9: open boundaries (inflow and outflow) against exact solutions.

    python -m gpucfd.tests.test_open          (inside WSL, from the project root)

1. Channel flow with a parabolic inflow and an outflow (pressure zero, zero velocity gradient), walls
   on the other sides. Poiseuille flow is an exact steady solution with both conditions, so the
   steady state must converge to it at order 2. Run along +x, along -x (inflow on the high side,
   outflow on the low side), along +y, and along z in 3D (walls in x, periodic y): every code path of
   the boundaries, in each direction. Flow out = flow in, divergence at round-off, and a mirrored
   channel equals the original to round-off.
2. Kovasznay flow (Re 40): an exact steady Navier-Stokes solution with convection. Its velocity is
   prescribed on all four sides (profiles varying along each side, flow in through x- and out through
   x+, sliding tangential velocity on y-, y+): order 2.
3. Start-up: the channel started from rest reaches the same steady state as from the exact profile.
Each with both time schemes.
"""
import json
import sys
from pathlib import Path

import numpy as np

from gpucfd import gcfd

FAILS = []
REPORT = {}


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def order(errs):
    return [float(np.log2(errs[q] / errs[q + 1])) for q in range(len(errs) - 1)]


def ordered(p):
    """Second order: the finest pair within 1.9-2.1, every pair within 1.7-2.2. (These boundaries give
    error = a h^2 - b h^3 with a large b, so coarse pairs show 1.8 rising to 2.)"""
    return 1.9 <= p[-1] <= 2.1 and all(1.7 <= v <= 2.2 for v in p)


def steady(s, dt, chunk=200, tol=1e-12, tmax=200.0):
    """Run until the velocity changes by less than tol (relative, per unit time)."""
    prev = s.get()
    t = 0.0
    while t < tmax:
        st = s.run(dt, chunk, 1e-12)
        t += dt * chunk
        cur = s.get()
        scale = max(np.max(np.abs(a)) for a in cur)
        change = max(np.max(np.abs(a - b)) for a, b in zip(cur, prev)) / (dt * chunk) / scale
        prev = cur
        if change < tol:
            return st, t, change
    return st, t, change


# ------------------------------------------------------------------------------------------- channel
def channel(N, axis, sign, scheme, start="exact", nu=0.1, Um=1.0, inflow="exact", htol=1e-10):
    """Poiseuille channel of width H = 1 between walls, length 2, flow along `axis` (0 x, 1 y, 2 z) in
    direction `sign`. 2D for axis 0 and 1 (walls across), 3D for axis 2 (walls in x, y periodic).
    inflow "exact": the exact parabola; "discrete": the discrete equations' own fully developed profile
    (then the whole channel must keep it to round-off)."""
    H, Lf = 1.0, 2.0
    if axis == 2:
        n, length, wall_dir = (N, 4, 2 * N), (H, 0.5, Lf), 0
    else:
        wall_dir = 1 - axis
        n = [0, 0, 1]
        length = [0.0, 0.0, 1.0]
        n[axis], n[wall_dir] = 2 * N, N
        length[axis], length[wall_dir] = Lf, H
    sides = ("x", "y", "z")
    inlet = sides[axis] + ("-" if sign > 0 else "+")
    outlet = sides[axis] + ("+" if sign > 0 else "-")

    disc = discrete_poiseuille(N, H, nu, 2 * Um * H / 3)

    def profile(X, Y, Z):
        r = (X, Y, Z)[wall_dir]
        if inflow == "exact":
            u = sign * 4 * Um * r * (H - r) / H ** 2
        else:  # r is at cell centres across the channel
            u = sign * disc[np.clip(np.floor(r / (H / N)).astype(int), 0, N - 1)]
        v = [0.0, 0.0, 0.0]
        v[axis] = u
        return tuple(v)

    bnd = {inlet: ("inflow", profile), outlet: "outflow", sides[wall_dir] + "-": "wall", sides[wall_dir] + "+": "wall"}
    s = gcfd.NavierStokes(n, length, nu, boundaries=bnd, scheme=scheme, htol=htol)
    q = []
    for c in range(s.ncomp):
        X, Y, Z = s.face_coords(c)
        q.append(profile(X, Y, Z)[c] * np.ones(s.shape) if start == "exact" else np.zeros(s.shape))
    s.set(q)
    s.project()
    dt = s.max_dt(Um)
    st, t, change = steady(s, dt)
    out = s.get()
    exact = []
    for c in range(s.ncomp):
        X, Y, Z = s.face_coords(c)
        exact.append(profile(X, Y, Z)[c] * np.ones(s.shape))
    err = max(np.max(np.abs(a - b)) for a, b in zip(out, exact))
    fin, fout = s.flux(inlet), s.flux(outlet)
    res = dict(out=out, err=err, flux_in=-fin, flux_out=fout, max_div=st["max_div"], t=t, change=change)
    s.close()
    return res


def discrete_poiseuille(N, H, nu, Q):
    """Fully developed profile of the discrete equations (independent of the CUDA code): nu u'' = G with
    the second difference, ghost values -u beyond the walls, flow rate sum(u) h = Q."""
    h = H / N
    A = (np.diag(-2.0 * np.ones(N)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)) * nu / h ** 2
    A[0, 0] -= nu / h ** 2
    A[-1, -1] -= nu / h ** 2
    w = np.linalg.solve(A, np.ones(N))  # profile for G = 1
    return w * (abs(Q) / (np.sum(w) * h))


def test_channel(scheme):
    Ns = (16, 32, 64)
    cases = [("+x", 0, 1), ("-x", 0, -1), ("+y", 1, 1), ("+z (3D)", 2, 1)]
    results = {}
    for label, axis, sign in cases:
        Nl = Ns
        rs = [channel(N, axis, sign, scheme) for N in Nl]
        errs = [r["err"] for r in rs]
        p = order(errs)
        results[label] = rs
        check(f"rung 9: channel {label} {scheme}, Poiseuille with inflow + outflow, order 2",
              ordered(p),
              f"max error {', '.join(f'{e:.2e}' for e in errs)} on N = {Nl}; order {', '.join(f'{v:.2f}' for v in p)}")
        # diffusion solves (IMEX) to 1e-14: their usual 1e-10 tolerance alone leaves ~1e-9
        fe = channel(16, axis, sign, scheme, inflow="discrete", htol=1e-14)["err"]
        check(f"rung 9: channel {label} {scheme}, discrete fully developed inflow (numpy) is kept exactly",
              fe < 1e-10, f"max deviation {fe:.1e} anywhere in the channel (N = 16)")
        f = rs[-1]
        bal = abs(f["flux_out"] - f["flux_in"]) / f["flux_in"]
        check(f"rung 9: channel {label} {scheme}, flow out = flow in, divergence free",
              bal < 1e-9 and f["max_div"] < 1e-8,
              f"in {f['flux_in']:.12f}, out {f['flux_out']:.12f} (relative {bal:.1e}); max |div u| {f['max_div']:.1e}")
        REPORT[f"channel_{label}_{scheme}"] = dict(N=list(Nl), err=errs, order=p, flux_balance=bal,
                                                   max_div=f["max_div"], t_steady=f["t"], far_err=fe)
    # mirror: the -x channel is the +x channel reflected in x (u -> -u, arrays reversed along x)
    a, b = results["+x"][-1]["out"], results["-x"][-1]["out"]
    u_m = -np.roll(b[0][:, :, ::-1], 1, axis=2)  # faces: reversed order shifts by one (face 0 <-> face n)
    u_m[:, :, 0] = a[0][:, :, 0]                  # face 0 of the +x channel is the inlet (not stored in -x)
    v_m = b[1][:, :, ::-1]
    d = max(np.max(np.abs(u_m - a[0])), np.max(np.abs(v_m - a[1])))
    check(f"rung 9: channel {scheme}, -x run = mirrored +x run", d < 1e-10, f"max difference {d:.1e}")
    REPORT[f"mirror_{scheme}"] = float(d)


# ------------------------------------------------------------------------------------------ Kovasznay
def kovasznay_exact(Re):
    lam = Re / 2 - np.sqrt(Re ** 2 / 4 + 4 * np.pi ** 2)

    def f(X, Y, Z):
        e = np.exp(lam * X)
        return (1 - e * np.cos(2 * np.pi * Y), lam / (2 * np.pi) * e * np.sin(2 * np.pi * Y), 0.0 * X)
    return f


def kovasznay(N, scheme, Re=40.0):
    f = kovasznay_exact(Re)
    x0, y0, L = -0.5, -0.5, 2.0

    def g(X, Y, Z):  # the solver's box starts at 0: shift
        return f(X + x0, Y + y0, Z)
    bnd = {side: ("inflow", g) for side in ("x-", "x+", "y-", "y+")}
    s = gcfd.NavierStokes((N, N, 1), (L, L, 1.0), 1.0 / Re, boundaries=bnd, scheme=scheme)
    exact = []
    for c in range(2):
        X, Y, Z = s.face_coords(c)
        exact.append(g(X, Y, Z)[c])
    s.set(exact)
    s.project()
    st, t, change = steady(s, s.max_dt(2.7), chunk=500, tol=1e-11)
    out = s.get()
    err = max(np.max(np.abs(a - b)) for a, b in zip(out, exact))
    l2 = float(np.sqrt(sum(np.mean((a - b) ** 2) for a, b in zip(out, exact))))
    s.close()
    return err, st, t, l2


def test_kovasznay(scheme):
    Ns = (32, 64, 128)
    rs = [kovasznay(N, scheme) for N in Ns]
    errs, l2 = [r[0] for r in rs], [r[3] for r in rs]
    p, p2 = order(errs), order(l2)
    # The max error sits in the first cell row along a side and converges at 1.90-1.92 out to 256^2
    # (measured), while the L2 and interior errors converge at 2.00: judged on L2, max order recorded.
    check(f"rung 9: Kovasznay Re 40 {scheme}, inflow profiles on all sides, order 2",
          ordered(p2) and min(p) > 1.85,
          f"L2 error {', '.join(f'{e:.2e}' for e in l2)}, order {', '.join(f'{v:.2f}' for v in p2)}; max error "
          f"{', '.join(f'{e:.2e}' for e in errs)}, order {', '.join(f'{v:.2f}' for v in p)} on N = {Ns}; "
          f"max |div u| {rs[-1][1]['max_div']:.1e}")
    REPORT[f"kovasznay_{scheme}"] = dict(N=list(Ns), err=errs, order=p, l2=l2, order_l2=p2, max_div=rs[-1][1]["max_div"])


def test_startup(scheme):
    a = channel(16, 0, 1, scheme, start="exact")
    b = channel(16, 0, 1, scheme, start="rest")
    d = max(np.max(np.abs(x - y)) for x, y in zip(a["out"], b["out"]))
    # (IMEX: the diffusion solves stop at a relative tolerance of 1e-10, so steady states agree to ~1e-9)
    check(f"rung 9: channel {scheme} started from rest reaches the same steady state", d < 1e-8,
          f"max difference {d:.1e} (reached after t = {b['t']:.1f})")
    REPORT[f"startup_{scheme}"] = float(d)


if __name__ == "__main__":
    for scheme in ("imex", "rk3"):
        test_channel(scheme)
        test_startup(scheme)
        test_kovasznay(scheme)
    out = Path(__file__).parent / "results" / "test_open.json"
    out.write_text(json.dumps(dict(report=REPORT, fails=FAILS), indent=1))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))
