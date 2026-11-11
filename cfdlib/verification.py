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


def level_sequence(values, r=2.0):
    """Convergence of a sequence of solutions, coarse to fine with ratio r (yours OR a reference's).

    A benchmark that lists several mesh levels is still converging; its finest level is not the exact
    answer (case 4: the reference drag rose level by level, and its own Richardson value sat closer to
    the simulation than its finest level). Returns the observed order and Richardson value from the
    last three levels, whether the changes are monotonic, and the last change (the finest level's
    uncertainty if nothing better is known)."""
    f = [float(v) for v in values]
    if len(f) < 3:
        return dict(order=None, richardson=None, monotonic=None, last_change=abs(f[-1] - f[-2]) if len(f) > 1 else None)
    d1, d2 = f[-2] - f[-3], f[-1] - f[-2]
    mono = d1 * d2 > 0 and abs(d2) < abs(d1)
    p = order_three_grids(*f[-3:], r=r) if mono else None
    return dict(order=p, richardson=richardson(f[-2], f[-1], p, r) if mono else None, monotonic=mono,
                last_change=abs(d2))


def gci_fine(f_med, f_fine, p, r=2.0, safety=1.25):
    """Grid Convergence Index of the fine solution, as a fraction (0.01 = 1 %)."""
    return safety * abs((f_fine - f_med) / f_fine) / (r ** p - 1)
