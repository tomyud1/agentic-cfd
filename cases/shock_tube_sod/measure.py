"""Reading the requested outputs from a profile along the tube.

Every function takes plain arrays (cell centres x, cell values), so the same rule is applied to a
simulation and to the exact solution's cell averages: the difference between the exact numbers and
what the rule reads from the exact cell averages is the rule's own bias.
"""
import numpy as np


def l1(sim, exact, scale):
    """Mean absolute difference over the cells (equal cells: the L1 norm divided by the tube length),
    as a fraction of `scale`."""
    return float(np.mean(np.abs(np.asarray(sim) - np.asarray(exact))) / abs(scale))


def total_variation(q):
    return float(np.sum(np.abs(np.diff(q))))


def windows(w, L):
    """Three stretches of the tube, one per wave, split half-way between neighbouring waves.
    w: wave positions (head, tail, contact, shock)."""
    a, b = 0.5 * (w["tail"] + w["contact"]), 0.5 * (w["contact"] + w["shock"])
    return dict(fan=(0.0, a), contact=(a, b), shock=(b, L))


def l1_by_wave(x, sim, exact, scale, w, L):
    """The L1 error split over the three stretches (their sum is the whole error)."""
    d = np.abs(np.asarray(sim) - np.asarray(exact)) / (len(x) * abs(scale))
    return {k: float(d[(x >= lo) & (x < hi)].sum()) for k, (lo, hi) in windows(w, L).items()}


def middle(x, q, a, b, frac=0.5):
    """Mean, lowest and highest value over the middle `frac` of the stretch a..b."""
    lo, hi = a + 0.5 * (1 - frac) * (b - a), b - 0.5 * (1 - frac) * (b - a)
    sel = (x >= lo) & (x <= hi)
    if not sel.any():
        return dict(mean=float("nan"), spread=float("nan"), cells=0)
    v = q[sel]
    return dict(mean=float(v.mean()), spread=float(v.max() - v.min()), cells=int(sel.sum()))


def states(x, rho, u, p, w, frac=0.5):
    """The uniform states between the waves: pressure and velocity between the fan and the shock,
    density on each side of the border between the two gases."""
    return dict(p=middle(x, p, w["tail"], w["shock"], frac), u=middle(x, u, w["tail"], w["shock"], frac),
                rho_L=middle(x, rho, w["tail"], w["contact"], frac), rho_R=middle(x, rho, w["contact"], w["shock"], frac))


def crossing(x, q, level, lo, hi):
    """Position inside lo..hi where the profile crosses `level` (straight line between cell centres).
    With several crossings (a wiggle) the first one from the left; nan when there is none."""
    idx = np.nonzero((x >= lo) & (x <= hi))[0]
    if len(idx) < 2:
        return float("nan")
    d = q[idx] - level
    for k in range(len(idx) - 1):
        if d[k] == 0.0:
            return float(x[idx[k]])
        if d[k] * d[k + 1] < 0:
            return float(x[idx[k]] + (x[idx[k + 1]] - x[idx[k]]) * d[k] / (d[k] - d[k + 1]))
    return float("nan")


def wave_positions(x, rho, u, p, st, w, L, p_R, fan_fit=(0.2, 0.8)):
    """Where each wave is. st: the states read by states(); w: the exact positions, used only to
    choose where to look. Shock and border: where the profile is half-way up the jump. Fan: the
    velocity rises along a straight line inside it, so a line fitted to its middle part is extended
    to u = 0 (front) and to u = u* (back)."""
    win = windows(w, L)
    out = dict(shock=crossing(x, p, 0.5 * (st["p"]["mean"] + p_R), *win["shock"]),
               contact=crossing(x, rho, 0.5 * (st["rho_L"]["mean"] + st["rho_R"]["mean"]), *win["contact"]))
    us = st["u"]["mean"]
    sel = (x < win["fan"][1]) & (u > fan_fit[0] * us) & (u < fan_fit[1] * us)
    if sel.sum() >= 3:
        b, a = np.polyfit(u[sel], x[sel], 1)                 # x = a + b u
        out.update(head=float(a), tail=float(a + b * us), fan_cells=int(sel.sum()))
    else:
        out.update(head=float("nan"), tail=float("nan"), fan_cells=int(sel.sum()))
    return out


def jump_widths(x, rho, p, st, w, L, p_R, levels=(0.1, 0.9)):
    """Width of the two jumps in cells: the distance between `levels` of the jump's height.
    Shock: on the pressure. Border between the two gases: on the density."""
    h = x[1] - x[0]
    win = windows(w, L)

    def width(q, low, high, lo, hi):
        a = crossing(x, q, low + levels[0] * (high - low), lo, hi)
        b = crossing(x, q, low + levels[1] * (high - low), lo, hi)
        return float(abs(a - b) / h)
    return dict(shock=width(p, p_R, st["p"]["mean"], *win["shock"]),
                contact=width(rho, st["rho_R"]["mean"], st["rho_L"]["mean"], *win["contact"]))


def read_all(x, rho, u, p, w, L, p_R, ext):
    """states, positions and widths of one profile, with the extraction rules of numerics.yaml."""
    st = states(x, rho, u, p, w, ext["plateau_middle"])
    return dict(states=st, positions=wave_positions(x, rho, u, p, st, w, L, p_R, tuple(ext["fan_fit"])),
                widths=jump_widths(x, rho, p, st, w, L, p_R, tuple(ext["jump_levels"])))


def speeds(times, pos_by_time, t_from):
    """Speed of each wave: slope of a straight line through its positions at the snapshots from
    t_from on. pos_by_time: list of wave_positions() results, one per time."""
    out = {}
    t = np.asarray(times, float)
    for k in ("head", "tail", "contact", "shock"):
        v = np.array([pp[k] for pp in pos_by_time], float)
        sel = (t >= t_from - 1e-15) & np.isfinite(v)
        out[k] = float(np.polyfit(t[sel], v[sel], 1)[0]) if sel.sum() >= 2 else float("nan")
    return out
