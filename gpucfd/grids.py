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


def describe(faces):
    """Smallest and largest cell, their ratio, and the largest step between neighbouring cells (1 on a
    uniform grid; second-order accuracy wants it close to 1)."""
    d = np.diff(np.asarray(faces, float))
    step = np.maximum(d[1:] / d[:-1], d[:-1] / d[1:]).max() if d.size > 1 else 1.0
    return dict(n=int(d.size), min=float(d.min()), max=float(d.max()), ratio=float(d.max() / d.min()),
                max_step=float(step))
