"""An independent NumPy implementation of gpucfd's compressible scheme, for rung 1 of its ladder.

Written separately from the CUDA code and built differently on purpose, so that a shared mistake is unlikely:
the CUDA kernel works cell by cell, turns the velocity into the face's direction and applies the boundaries
inside its stencil; this file pads whole arrays with ghost cells, keeps the velocity as a vector, computes one
flux per face and differences the flux arrays. Always in double precision.

    U1 = step(U0, dt, cfg)      U: (5, nz, ny, nx) = rho, rho u, rho v, rho w, E
    cfg = dict(n=(nx, ny, nz), h=(hx, hy, hz), gamma=1.4, recon="vanleer", flux="hllc", stages=3, sensor=0.5,
               sides={"x-": "wall", ...}, segments=[("x-", {"y": (j0, j1)}, ("supersonic", state)), ...])
"""
import numpy as np

G = 3                                    # ghost layers (WENO5 reaches three cells)
KIND = {"periodic": 0, "wall": 1, "inflow": 2, "outflow": 3, "supersonic": 4}
TINY = 1e-300


def side_maps(n, d, sides, segments, high):
    """Boundary type and state number of every grid line that ends on one side (arrays over the two other
    directions, in NumPy's z, y, x order), and the list of states."""
    name = "xyz"[d] + ("+" if high else "-")
    others = [a for a in (2, 1, 0) if a != d]
    shape = tuple(n[a] for a in others)
    table = []

    def code(spec):
        if isinstance(spec, str):
            return KIND[spec], 0
        table.append(spec[1])
        return KIND[spec[0]], len(table) - 1
    kinds, states, taken = np.zeros(shape, int), np.zeros(shape, int), np.zeros(shape, bool)
    kinds[:], states[:] = code(sides.get(name, "periodic"))
    for sname, ranges, spec in segments:
        if sname != name:
            continue
        mask = ~taken                                             # the first segment that covers a line wins
        for pos, a in enumerate(others):
            lo, hi = ranges.get("xyz"[a], (0, n[a]))
            idx = np.arange(n[a])
            mask = mask & ((idx >= lo) & (idx < hi)).reshape([-1 if q == pos else 1 for q in range(2)])
        k, s = code(spec)
        kinds[mask], states[mask] = k, s
        taken |= mask
    return kinds, states, table


def pad(f, lo, hi, key, sign=1.0):
    """f (a, b, n) along the last axis with G ghost cells on each end. lo, hi: (kinds, states, table) of the two
    sides. key(state) is the prescribed value; sign = -1 for the velocity normal to a wall."""
    n = f.shape[-1]
    out = np.empty(f.shape[:-1] + (n + 2 * G,))
    out[..., G:G + n] = f
    for (kinds, states, table), low in ((lo, True), (hi, False)):
        fixed = np.array([key(s) for s in table] or [0.0])[states]
        for k in range(1, G + 1):
            if low:
                cand, at = [f[..., (n - k) % n], sign * f[..., k - 1], f[..., 0]], G - k
            else:
                cand, at = [f[..., (k - 1) % n], sign * f[..., n - k], f[..., n - 1]], G + n - 1 + k
            out[..., at] = np.select([kinds == 0, kinds == 1, kinds == 3], cand, fixed)
    return out


def slope(kind, dm, dp):
    if kind == "central":
        return 0.5 * (dm + dp)
    same = dm * dp > 0
    if kind == "minmod":
        s = np.where(np.abs(dm) < np.abs(dp), dm, dp)
    elif kind == "vanleer":
        s = 2 * dm * dp / np.where(same, dm + dp, 1.0)
    elif kind == "mc":
        s = np.sign(dm) * np.minimum(np.minimum(2 * np.abs(dm), 2 * np.abs(dp)), 0.5 * np.abs(dm + dp))
    else:
        raise ValueError(kind)
    return np.where(same, s, 0.0)


def weno(a, b, c, d, e, weps=1e-20):
    """WENO5-Z value at the face between c and d, seen from c."""
    b0 = 13 / 12 * (a - 2 * b + c) ** 2 + 0.25 * (a - 4 * b + 3 * c) ** 2
    b1 = 13 / 12 * (b - 2 * c + d) ** 2 + 0.25 * (b - d) ** 2
    b2 = 13 / 12 * (c - 2 * d + e) ** 2 + 0.25 * (3 * c - 4 * d + e) ** 2
    eps = weps * (a * a + b * b + c * c + d * d + e * e) + TINY
    tau = np.abs(b0 - b2)
    a0, a1, a2 = 0.1 * (1 + tau / (b0 + eps)), 0.6 * (1 + tau / (b1 + eps)), 0.3 * (1 + tau / (b2 + eps))
    return (a0 * (2 * a - 7 * b + 11 * c) + a1 * (-b + 5 * c + 2 * d) + a2 * (2 * c + 5 * d - e)) / (6 * (a0 + a1 + a2))


def face_values(P, kind, weps=1e-20):
    """Low-face and high-face values of the cells -1 .. n (n + 2 cells) from the padded array P."""
    n = P.shape[-1] - 2 * G
    sh = lambda k: P[..., G - 1 + k:G + n + 1 + k]                 # noqa: E731
    c = sh(0)
    if kind == "first":
        return c.copy(), c.copy()
    if kind == "weno5":
        return weno(sh(2), sh(1), c, sh(-1), sh(-2), weps), weno(sh(-2), sh(-1), c, sh(1), sh(2), weps)
    s = 0.5 * slope(kind, c - sh(-1), sh(1) - c)
    return c - s, c + s


def conserved_and_flux(g, d, r, v, p):
    E = p / (g - 1) + 0.5 * r * (v ** 2).sum(0)
    u = v[d]
    U = np.stack([r, r * v[0], r * v[1], r * v[2], E])
    F = np.stack([r * u, r * u * v[0], r * u * v[1], r * u * v[2], u * (E + p)])
    F[1 + d] += p
    return U, F


def face_flux(kind, g, d, L, R, use_hll):
    (rl, vl, pl), (rr, vr, pr) = L, R
    Ul, Fl = conserved_and_flux(g, d, rl, vl, pl)
    Ur, Fr = conserved_and_flux(g, d, rr, vr, pr)
    ul, ur, cl, cr = vl[d], vr[d], np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)
    if kind == "rusanov":
        s = np.maximum(np.abs(ul) + cl, np.abs(ur) + cr)
        return 0.5 * (Fl + Fr) - 0.5 * s * (Ur - Ul)
    ql, qr = np.sqrt(rl), np.sqrt(rr)
    vt = (ql * vl + qr * vr) / (ql + qr)                          # Roe average
    Ht = (ql * (Ul[4] + pl) / rl + qr * (Ur[4] + pr) / rr) / (ql + qr)
    ct = np.sqrt((g - 1) * (Ht - 0.5 * (vt ** 2).sum(0)))
    SL, SR = np.minimum(ul - cl, vt[d] - ct), np.maximum(ur + cr, vt[d] + ct)
    hll = (SR * Fl - SL * Fr + SL * SR * (Ur - Ul)) / (SR - SL)
    if kind == "hll":
        mid = hll
    else:
        ml, mr = rl * (SL - ul), rr * (SR - ur)
        Ss = (pr - pl + ml * ul - mr * ur) / (ml - mr)
        plr = 0.5 * (pl + pr + ml * (Ss - ul) + mr * (Ss - ur))
        D = np.zeros_like(Ul)
        D[1 + d], D[4] = 1.0, Ss
        hllc = np.where(Ss >= 0, (Ss * (SL * Ul - Fl) + SL * plr * D) / (SL - Ss), (Ss * (SR * Ur - Fr) + SR * plr * D) / (SR - Ss))
        mid = hllc if kind == "hllc" else np.where(use_hll, hll, hllc)
    return np.where(SL >= 0, Fl, np.where(SR <= 0, Fr, mid))


def rate(U, cfg):
    """dU/dt of the scheme."""
    g, n, h = cfg["gamma"], cfg["n"], cfg["h"]
    sides, segments = cfg.get("sides", {}), cfg.get("segments", [])
    rho, vel = U[0], U[1:4] / U[0]
    p = (g - 1) * (U[4] - 0.5 * rho * (vel ** 2).sum(0))
    active = [d for d in range(3) if n[d] > 1]
    key_r, key_p = (lambda s: s["rho"]), (lambda s: s["p"])
    key_v = lambda i: (lambda s: s.get("uvw"[i], 0.0))            # noqa: E731
    maps = {d: (side_maps(n, d, sides, segments, False), side_maps(n, d, sides, segments, True)) for d in active}

    flags = np.zeros(rho.shape, int)
    if cfg["flux"] == "hybrid" and len(active) > 1:
        for d in active:
            pp = pad(np.moveaxis(p, 2 - d, -1), *maps[d], key_p)
            c = pp[..., G:G + n[d]]
            jump = np.zeros(c.shape, bool)
            for nb in (pp[..., G - 1:G + n[d] - 1], pp[..., G + 1:G + n[d] + 1]):
                jump |= np.abs(nb - c) > cfg.get("sensor", 0.5) * np.minimum(nb, c)
            flags |= np.moveaxis(jump, -1, 2 - d).astype(int) << d

    out = np.zeros_like(U)
    for d in active:
        lo, hi = maps[d]
        mv = lambda f: np.moveaxis(f, 2 - d, -1)                  # noqa: E731
        P = [pad(mv(rho), lo, hi, key_r)] + [pad(mv(vel[i]), lo, hi, key_v(i), -1.0 if i == d else 1.0) for i in range(3)] \
            + [pad(mv(p), lo, hi, key_p)]
        fv = [face_values(a, cfg["recon"], cfg.get("weno_eps", 1e-20)) for a in P]
        wm, wp = np.stack([a[0] for a in fv]), np.stack([a[1] for a in fv])
        cell = np.stack([a[..., G - 1:G + n[d] + 1] for a in P])
        bad = ~((wm[0] > 0) & (wm[4] > 0) & (wp[0] > 0) & (wp[4] > 0))
        wm, wp = np.where(bad, cell, wm), np.where(bad, cell, wp)
        L, R = wp[..., :-1], wm[..., 1:]                           # faces 0 .. n: left state from cell f - 1, right from cell f
        use_hll = None
        if cfg["flux"] == "hybrid":
            fp = pad(mv(flags).astype(float), lo, hi, lambda s: 0.0).astype(int)[..., G - 1:G + n[d] + 1]
            use_hll = ((fp[..., :-1] | fp[..., 1:]) & (7 & ~(1 << d))) != 0
        kind = "hllc" if cfg["flux"] == "hybrid" and len(active) == 1 else cfg["flux"]
        F = face_flux(kind, g, d, (L[0], L[1:4], L[4]), (R[0], R[1:4], R[4]), use_hll)
        for (kinds, states, table), at in ((lo, 0), (hi, n[d])):
            for k, s in enumerate(table):                         # supersonic inflow: the state's own flux
                v = np.array([s.get(c, 0.0) for c in "uvw"])
                _, Fs = conserved_and_flux(g, d, s["rho"], v, s["p"])
                sel = (kinds == 4) & (states == k)
                F[:, sel, at] = Fs[:, None]
        out -= np.moveaxis((F[..., 1:] - F[..., :-1]) / h[d], -1, 3 - d)
    return out


def step(U, dt, cfg):
    st = cfg.get("stages", 3)
    U1 = U + dt * rate(U, cfg)
    if st == 1:
        return U1
    if st == 2:
        return 0.5 * U + 0.5 * (U1 + dt * rate(U1, cfg))
    U2 = 0.75 * U + 0.25 * (U1 + dt * rate(U1, cfg))
    return U / 3 + 2 / 3 * (U2 + dt * rate(U2, cfg))
