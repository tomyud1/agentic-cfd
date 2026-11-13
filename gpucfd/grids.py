"""Face positions for stretched grids (see DESIGN.md, "Stretched grids").

A stretched direction of n cells is described by its n + 1 face positions, from 0 to the length. Pass
them per direction to the solvers; None keeps a direction uniform:

    from gpucfd import gcfd, grids
    y = grids.tanh(64, 1.0, ratio=8)                 # small cells at both walls, largest / smallest = 8
    s = gcfd.NavierStokes((32, 64, 1), (2.0, 1.0, 1.0), nu, walls=(False, True, False), grid=(None, y, None))

Any increasing array works. The scheme is second order when neighbouring cells differ only a little
(describe(...)["max_step"] close to 1); a sudden jump in cell size costs accuracy there. Periodic
directions must stay uniform.
"""
import numpy as np


def uniform(n, length):
    return np.linspace(0.0, float(length), int(n) + 1)


def _tanh_unit(n, beta, side):
    xi = np.arange(n + 1) / n
    if side == "both":
        return 0.5 * (1.0 + np.tanh(beta * (2.0 * xi - 1.0)) / np.tanh(beta))
    if side == "low":
        return 1.0 + np.tanh(beta * (xi - 1.0)) / np.tanh(beta)
    if side == "high":
        return np.tanh(beta * xi) / np.tanh(beta)
    raise ValueError(f'side {side!r}: "both", "low" or "high"')


def tanh(n, length, ratio, side="both"):
    """n + 1 face positions on [0, length] with hyperbolic-tangent spacing: cells shrink smoothly toward
    both ends (side="both"), toward 0 ("low") or toward length ("high").
    ratio = largest cell / smallest cell (1: uniform). The smallest cell is at the clustered end(s)."""
    n = int(n)
    if ratio < 1.0:
        raise ValueError("ratio = largest cell / smallest cell must be at least 1")
    if ratio < 1.0 + 1e-9:
        return uniform(n, length)

    def r(beta):
        d = np.diff(_tanh_unit(n, beta, side))
        return d.max() / d.min()
    lo, hi = 1e-4, 1.0
    while r(hi) < ratio:
        hi *= 2.0
        if hi > 64:
            raise ValueError(f"ratio {ratio:g} cannot be reached with {n} cells")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if r(mid) < ratio else (lo, mid)
    x = _tanh_unit(n, 0.5 * (lo + hi), side)
    if side == "both":
        x = 0.5 * (x + (1.0 - x[::-1]))  # exactly symmetric
    x = x * float(length)
    x[0], x[-1] = 0.0, float(length)
    return x


def clustered(n, length, band, h, grow, cap_max=64.0):
    """n + 1 face positions on [0, length] with cells of size h inside band = (lo, hi) and growing away
    from it: at distance d from the band the cell size is h (1 + (c - 1) d / grow), up to c h at distance
    grow and beyond. grow: one length or (below the band, above it). The far-field factor c (the same on
    both sides) is solved so that exactly n cells fill the length; it is returned by describe() as ratio.
    Use it for a body in a large domain: band = the body plus the layers around it, h = the cell size the
    body needs. Doubling n with half h refines every cell by two (the same mapping, sampled twice as
    finely), which is what a grid-convergence study needs.
    Raises ValueError if n cells of size h or larger cannot fill the length (c < 1) or if c > cap_max."""
    n, length, h = int(n), float(length), float(h)
    lo, hi = float(band[0]), float(band[1])
    g = (float(grow), float(grow)) if np.isscalar(grow) else (float(grow[0]), float(grow[1]))
    if not (0.0 <= lo < hi <= length) or h <= 0 or min(g) <= 0:
        raise ValueError("clustered: need 0 <= lo < hi <= length, h > 0 and grow > 0")

    def pieces(c):
        """(x0, x1, kind, g) along x; kind 0: size h; 1: size c h; 2: growing away from x0; 3: shrinking toward x1."""
        p = []
        if lo - g[0] > 0:
            p.append((0.0, lo - g[0], 1, g[0]))
        if lo > 0:
            p.append((max(lo - g[0], 0.0), lo, 3, g[0]))
        p.append((lo, hi, 0, 0.0))
        if hi < length:
            p.append((hi, min(hi + g[1], length), 2, g[1]))
        if hi + g[1] < length:
            p.append((hi + g[1], length, 1, g[1]))
        return p

    def count(c, x0, x1, kind, gg, x=None):
        """Cells between x0 and x (default x1) of one piece: the integral of 1 / (cell size)."""
        x = x1 if x is None else x
        if kind == 0:
            return (x - x0) / h
        if kind == 1:
            return (x - x0) / (c * h)
        k = (c - 1.0) / gg
        if abs(k) * gg < 1e-12:
            return (x - x0) / h
        if kind == 2:  # distance from the band: x - x0
            return np.log1p(k * (x - x0)) / (k * h)
        return (np.log1p(k * (x1 - x0)) - np.log1p(k * (x1 - x))) / (k * h)  # distance: x1 - x

    def total(c):
        return sum(count(c, *p) for p in pieces(c))
    if total(1.0) < n:
        raise ValueError(f"clustered: {n} cells of size {h:g} do not fit in {length:g} (at most {int(total(1.0))})")
    if total(cap_max) > n:
        raise ValueError(f"clustered: {n} cells need a far-field factor above {cap_max:g} (fewest: {total(cap_max):.0f} ... "
                         "use fewer cells, a wider band or a larger h)")
    a, b = 1.0, cap_max
    for _ in range(200):
        m = 0.5 * (a + b)
        a, b = (m, b) if total(m) > n else (a, m)
    c = 0.5 * (a + b)
    ps = pieces(c)
    ends = np.cumsum([count(c, *p) for p in ps])
    starts = np.concatenate(([0.0], ends[:-1]))
    x = np.empty(n + 1)
    for i in range(n + 1):
        q = min(int(np.searchsorted(ends, i, side="left")), len(ps) - 1)
        x0, x1, kind, gg = ps[q]
        s = i - starts[q]  # cells into this piece
        k = (c - 1.0) / gg if kind >= 2 else 0.0
        if kind == 0 or (kind >= 2 and abs(k) * gg < 1e-12):
            x[i] = x0 + s * h
        elif kind == 1:
            x[i] = x0 + s * c * h
        elif kind == 2:
            x[i] = x0 + np.expm1(s * k * h) / k
        else:
            x[i] = x1 - np.expm1(np.log1p(k * (x1 - x0)) - s * k * h) / k
    x[0], x[-1] = 0.0, length
    return x


def describe(faces):
    """Smallest and largest cell, their ratio, and the largest step between neighbouring cells (1 on a
    uniform grid; second-order accuracy wants it close to 1)."""
    d = np.diff(np.asarray(faces, float))
    step = np.maximum(d[1:] / d[:-1], d[:-1] / d[1:]).max() if d.size > 1 else 1.0
    return dict(n=int(d.size), min=float(d.min()), max=float(d.max()), ratio=float(d.max() / d.min()),
                max_step=float(step))
