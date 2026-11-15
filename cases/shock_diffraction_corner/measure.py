"""Reading rules: how each output is taken from a field. Fields are arrays [row j (y), column i (x)].

Every rule here is fixed in numerics.yaml (`extraction`) before the production runs. `rule_bias`
applies the shock-position rule to the exact flat shock averaged over the cells, to show what
the rule itself adds.
"""
import numpy as np


def centres(N, length):
    return (np.arange(N) + 0.5) * length / N


def front_x(p, level, h):
    """Per row: x where the pressure crosses `level`, coming from the right (the still air). The crossing
    lies between the last cell above the level and its right neighbour (straight line between the two
    cell centres). nan for a row that never exceeds the level, or exceeds it in its last cell."""
    p = np.atleast_2d(p)
    N = p.shape[1]
    above = p > level
    last = N - 1 - np.argmax(above[:, ::-1], axis=1)                # last column above the level
    ok = above.any(axis=1) & (last < N - 1)
    i = np.where(ok, last, 0)
    rows = np.arange(p.shape[0])
    pa, pb = p[rows, i], p[rows, np.minimum(i + 1, N - 1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (i + 0.5) * h + h * (pa - level) / (pa - pb)
    return np.where(ok, x, np.nan)


def wall_shock(pcol, p0, h, start=1.5):
    """Shock on the step face from the pressure in the column of cells next to it (index 0 = floor).
    Coming up from the floor: the first cell above start x p0 opens the front, the first local maximum
    from there on is the level behind the shock, and the shock is where the pressure crosses the middle
    between p0 and that level. Returns (height y of the shock, level behind it) or (nan, nan)."""
    pcol = np.asarray(pcol, float)
    hit = np.nonzero(pcol > start * p0)[0]
    if not len(hit) or hit[0] == 0:
        return np.nan, np.nan
    j = hit[0]
    while j + 1 < len(pcol) and pcol[j + 1] > pcol[j]:
        j += 1
    peak = pcol[j]
    level = 0.5 * (p0 + peak)
    k = np.nonzero(pcol > level)[0][0]                              # first cell above the middle level
    y = (k - 0.5) * h + h * (level - pcol[k - 1]) / (pcol[k] - pcol[k - 1])
    return float(y), float(peak)


def restrict(a):
    """Average over 2 x 2 cells: the field as the grid with cells twice as large would hold it."""
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def restrict1(a):
    return 0.5 * (a[..., 0::2] + a[..., 1::2])


def mean_rel_diff(a, b):
    """Mean absolute difference of two fields as a fraction of their mean size."""
    return float(np.mean(np.abs(a - b)) / np.mean(0.5 * (np.abs(a) + np.abs(b))))


def flat_part(p, rho, u, v, t, fl, ext, h):
    """The flat part of the shock and the uniform state behind it, at one written time.

    Rows: those wholly above the exact height that the corner's signal has reached on the shock, plus a
    margin. Uniform region: the cells above the highest point the signal has reached anywhere, plus the
    margin, and the margin away from the inlet and from the shock."""
    N = p.shape[0]
    y = centres(N, fl.H)
    x = centres(N, fl.L)
    m = ext["margin_cells"] * h
    level = fl.p0 + ext["front_level"] * (fl.p2 - fl.p0)
    rows = y - h / 2 >= fl.flat_from(t) + m
    xs = front_x(p[rows], level, h)
    out = dict(rows=int(rows.sum()), y_from=float(fl.flat_from(t) + m))
    if rows.sum() < 3 or not np.all(np.isfinite(xs)):
        return dict(out, ok=False)
    fit = np.polyfit(y[rows], xs, 1)
    out.update(ok=True, x_mean=float(xs.mean()), dev_mean_cells=float(np.max(np.abs(xs - xs.mean())) / h),
               dev_line_cells=float(np.max(np.abs(xs - np.polyval(fit, y[rows]))) / h),
               tilt_cells=float(abs(fit[0]) * (y[rows][-1] - y[rows][0]) / h), x_top=float(xs[-1]))
    reg_rows = y - h / 2 >= fl.signal_top(t) + m
    reg_cols = (x - h / 2 >= m) & (x + h / 2 <= xs.mean() - m)
    out.update(region_rows=int(reg_rows.sum()), region_cols=int(reg_cols.sum()))
    if reg_rows.sum() >= 2 and reg_cols.sum() >= 2:
        sel = np.ix_(reg_rows, reg_cols)
        for key, f, exact in (("p", p, fl.p2), ("rho", rho, fl.rho2), ("u", u, fl.u2)):
            a = f[sel]
            out[key] = dict(mean=float(a.mean()), err=float(a.mean() / exact - 1), lo=float(a.min() / exact - 1),
                            hi=float(a.max() / exact - 1), ripple=float((a.max() - a.min()) / exact))
        out["v_max"] = float(np.max(np.abs(v[sel])) / fl.u2)
        # the same along the shock only: columns within 3 margins behind it (odd-even patterns show here first)
        near = (x + h / 2 <= xs.mean() - m) & (x - h / 2 >= xs.mean() - 4 * m)
        if near.sum() >= 2:
            a = p[np.ix_(reg_rows, near)]
            out["p_row_spread"] = float(np.max(a.max(axis=0) - a.min(axis=0)) / fl.p2)   # largest change along y in one column
    return out


def rule_bias(fl, ext, N, n=200):
    """The shock-position rule applied to the exact flat shock, averaged over the cells as the solver holds
    it (mass, momentum and energy per cell; the pressure follows from them). Returns the largest and
    smallest (rule - exact) in cells over n positions of the shock inside a cell."""
    h = fl.L / N
    gm = fl.gamma
    level = fl.p0 + ext["front_level"] * (fl.p2 - fl.p0)
    edges = np.arange(N + 1) * h
    out = []
    for f in (np.arange(n) + 0.5) / n:
        xs = (N // 2 + f) * h
        frac = np.clip((xs - edges[:-1]) / h, 0, 1)                # part of each cell behind the shock
        rho = frac * fl.rho2 + (1 - frac) * fl.rho0
        mom = frac * fl.rho2 * fl.u2
        E = frac * fl.E2 + (1 - frac) * fl.E0
        p = (gm - 1) * (E - 0.5 * mom ** 2 / rho)
        out.append((front_x(p, level, h)[0] - xs) / h)
    return float(np.min(out)), float(np.max(out))
