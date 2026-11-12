"""Discretisation-error tools: error norms, observed order of accuracy, Richardson, GCI.

GCI follows Celik et al. (2008), "Procedure for Estimation and Reporting of Uncertainty Due
to Discretization in CFD Applications", J. Fluids Eng. 130(7), 078001.
"""
import math

import numpy as np


def relative_l2(sim, exact, scale):
    """RMS error of sim vs exact, divided by a reference scale (e.g. the peak exact value)."""
    sim, exact = np.asarray(sim), np.asarray(exact)
    return float(np.sqrt(np.mean((sim - exact) ** 2)) / abs(scale))


def order_from_errors(h, err):
    """Observed order p between consecutive meshes, from err ~ C h^p (known exact solution)."""
    return [math.log(err[i] / err[i + 1]) / math.log(h[i] / h[i + 1]) for i in range(len(h) - 1)]


def order_three_grids(f_coarse, f_med, f_fine, r=2.0):
    """Observed order p from three solutions with constant refinement ratio r (no exact solution)."""
    return math.log(abs((f_coarse - f_med) / (f_med - f_fine))) / math.log(r)


def richardson(f_med, f_fine, p, r=2.0):
    """Richardson-extrapolated (mesh-independent) estimate."""
    return f_fine + (f_fine - f_med) / (r ** p - 1)


def level_sequence(values, r=2.0, p_min=0.5):
    """Convergence of a sequence of solutions, coarse to fine with ratio r (yours OR a reference's).

    A benchmark that lists several mesh levels is still converging; its finest level is not the exact
    answer (case 4: the reference drag rose level by level, and its own Richardson value sat closer to
    the simulation than its finest level). Returns the observed order and Richardson value from the
    last three levels, whether the changes are monotonic, and the last change (the finest level's
    uncertainty if nothing better is known).
    plausible: False when the observed order is below p_min. The changes then barely shrink, the
    levels are not in the asymptotic range, and Richardson would blow up (it divides by r^p - 1), so
    no Richardson value is given. Found on a pressure probe less than one cell from a wall on the
    coarsest grid: order 0.006, "Richardson" 11 times the exact value."""
    f = [float(v) for v in values]
    if len(f) < 3:
        return dict(order=None, richardson=None, monotonic=None, plausible=None,
                    last_change=abs(f[-1] - f[-2]) if len(f) > 1 else None)
    d1, d2 = f[-2] - f[-3], f[-1] - f[-2]
    mono = d1 * d2 > 0 and abs(d2) < abs(d1)
    p = order_three_grids(*f[-3:], r=r) if mono else None
    ok = p is not None and p >= p_min
    return dict(order=p, richardson=richardson(f[-2], f[-1], p, r) if ok else None, monotonic=mono,
                plausible=ok if mono else None, last_change=abs(d2))


def gci_fine(f_med, f_fine, p, r=2.0, safety=1.25):
    """Grid Convergence Index of the fine solution, as a fraction (0.01 = 1 %)."""
    return safety * abs((f_fine - f_med) / f_fine) / (r ** p - 1)


def iteration_error(values, window=3, rho_min=None):
    """Remaining iteration error of a quantity marching to a steady value, from its history sampled at
    equal intervals (time chunks or iterations).

    Near convergence the slowest decaying mode dominates and each change is about rho times the one
    before, so the changes still to come sum to |last change| * rho / (1 - rho). rho is the largest
    |change ratio| over the last `window` ratios (conservative; also bounds an oscillating approach).
    Returns dict(error, rho, extrapolated, noise): error is inf while the history is too short, not
    contracting or the ratio not settled, except at a noise floor (changes flipping sign without
    shrinking: noise True, error = their largest size; marching longer cannot reduce it); extrapolated (the geometric limit) only for a monotonic approach, for information.

    Limit: a slow mode hidden under a faster one cannot be seen in the history; while the fast mode
    dominates the estimate is too small (11x in the two-mode test). rho_min, the slow rate measured
    elsewhere (the same flow on a coarser grid, same sampling interval), closes that gap: rho is never
    taken below it."""
    v = np.asarray(values, float)
    d = np.diff(v)
    out = dict(error=math.inf, rho=None, extrapolated=None, noise=False)
    if len(d) < window + 1:
        return out
    last = d[-window - 1:]
    if last[-1] == 0.0:
        return dict(error=0.0, rho=0.0, extrapolated=float(v[-1]), noise=False)
    if np.any(last[:-1] == 0.0):
        return out
    ratios = np.abs(last[1:] / last[:-1])
    if rho_min is not None:
        ratios = np.maximum(ratios, rho_min)
    rho = float(np.max(ratios))
    out["rho"] = rho
    # Noise floor: the changes flip sign and no longer shrink (solver tolerances; measured on the
    # Couette pressure probes at about 4e-7 relative). The value then wanders by about the size of the
    # changes, so that is the error estimate. A slow monotonic drift keeps its sign and stays inf.
    unsettled = rho >= 0.999 or (1.0 - float(np.min(ratios))) > 1.5 * (1.0 - rho)
    tail = d[-8:]  # longer window: in pure noise about 2 of 3 consecutive changes flip sign
    flips = int(np.sum(np.sign(tail[1:]) != np.sign(tail[:-1])))
    # and noise does not shrink, while a decaying transient that still oscillates erratically does
    # (DFG 2D-1 lift at t = 9 had flipping changes of 2e-2 falling to 1e-5: not yet a floor)
    level = len(tail) == 8 and np.max(np.abs(tail[4:])) >= 0.2 * np.max(np.abs(tail[:4]))
    if unsettled and level and flips >= 3:
        out.update(error=float(np.max(np.abs(tail))), noise=True)
        return out
    if rho >= 0.999:
        return out
    # Not settled yet: while a fast mode dies out the ratio still rises toward the slow mode's, and the
    # estimate would be too small (5x in a two-mode test). Require 1 - ratio to agree within 1.5x.
    if unsettled:
        return out
    out["error"] = abs(d[-1]) * rho / (1.0 - rho)
    q = d[-1] / d[-2]
    if 0.0 < q < 1.0:
        out["extrapolated"] = float(v[-1] + d[-1] * q / (1.0 - q))
    return out
