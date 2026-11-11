"""Cycle analysis of the force signals (drag and lift every time step).

A cycle runs from one lift minimum to the next (as in the FeatFlow tables, t0 = time of minimum lift).
Peaks are refined with a parabola through the three samples around each discrete extremum, so the
values do not depend on where the time steps happen to fall.
"""
import numpy as np


def coefficients(ts, given, dz):
    """Drag and lift coefficients c = 2 F / (rho U_mean^2 D) per unit depth (forces scaled by rhoInf = rho)."""
    rho, D, Um = (given[k]["value"] for k in ("rho", "D", "Um"))
    Umean = 2.0 / 3.0 * Um
    F = np.asarray(ts["Fcyl"]) / dz                     # force per unit depth [N/m] (rho = rhoInf in forces)
    q = 0.5 * rho * Umean ** 2 * D
    return np.asarray(ts["t"]), F[:, 0] / q, F[:, 1] / q


def _refine(t, y, i):
    """Parabolic refinement of an extremum at sample i: (time, value)."""
    if i <= 0 or i >= len(y) - 1:
        return t[i], y[i]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2 * y1 + y2
    if den == 0:
        return t[i], y1
    d = 0.5 * (y0 - y2) / den                           # offset in samples, |d| <= 0.5
    dt = t[i + 1] - t[i]
    return t[i] + d * dt, y1 - 0.25 * (y0 - y2) * d


def extrema(t, y, kind):
    """All local minima ('min') or maxima ('max') of a sampled signal, parabola-refined."""
    s = -1 if kind == "min" else 1
    z = s * y
    idx = np.nonzero((z[1:-1] > z[:-2]) & (z[1:-1] >= z[2:]))[0] + 1
    return [(_refine(t, y, i)) for i in idx]


def up_crossings(t, y, level):
    """Times where y crosses `level` upwards (linear interpolation between samples)."""
    z = y - level
    i = np.nonzero((z[:-1] < 0) & (z[1:] >= 0))[0]
    return t[i] - z[i] * (t[i + 1] - t[i]) / (z[i + 1] - z[i])


def cycles(t, cd, cl, window=3.0):
    """Per lift cycle: t0, t1, period, cl max/min, cd max/min (parabola-refined).

    Cycles are delimited by upward crossings of the lift through its mid-range level (of the last
    `window` seconds). The crossings sit on the steep part of the signal, so the period is sharp; lift
    minima are flat and their times jitter by a few time steps (insight 6). Each cycle is one full
    period, so its maximum and minimum are the same as for a cycle from minimum to minimum."""
    late = t > t[-1] - window
    mid = 0.5 * (cl[late].max() + cl[late].min())
    tc = up_crossings(t, cl, mid)
    out = []
    for t0, t1 in zip(tc[:-1], tc[1:]):
        sel = (t >= t0) & (t <= t1)
        tt, l, d = t[sel], cl[sel], cd[sel]
        if len(tt) < 5:
            continue
        lmax = max(extrema(tt, l, "max"), key=lambda e: e[1], default=(tt[np.argmax(l)], l.max()))
        lmin = min(extrema(tt, l, "min"), key=lambda e: e[1], default=(tt[np.argmin(l)], l.min()))
        dmx = max(extrema(tt, d, "max"), key=lambda e: e[1], default=(tt[np.argmax(d)], d.max()))
        dmn = min(extrema(tt, d, "min"), key=lambda e: e[1], default=(tt[np.argmin(d)], d.min()))
        out.append(dict(t0=float(t0), t1=float(t1), period=float(t1 - t0), cl_min=lmin[1], t_clmin=lmin[0],
                        cl_max=lmax[1], t_clmax=lmax[0], cd_max=dmx[1], cd_min=dmn[1],
                        cd_avg=float(np.trapezoid(d, tt) / (tt[-1] - tt[0])),
                        cl_avg=float(np.trapezoid(l, tt) / (tt[-1] - tt[0]))))
    return out


def summary(cyc, n_last):
    """Benchmark numbers from the last n_last cycles (values of the LAST cycle; spread over the n_last)."""
    last = cyc[-n_last:]
    c = last[-1]
    keys = ("cl_max", "cl_min", "cd_max", "cd_min", "period")
    spread = {k: float(max(abs(x[k] - y[k]) for x, y in zip(last[:-1], last[1:]))) for k in keys}
    T = float(np.mean([x["period"] for x in last]))
    res = dict(cd_max=c["cd_max"], cd_min=c["cd_min"], cl_max=c["cl_max"], cl_min=c["cl_min"],
               cd_mean=0.5 * (c["cd_max"] + c["cd_min"]), cd_amp=c["cd_max"] - c["cd_min"],
               cl_mean=0.5 * (c["cl_max"] + c["cl_min"]), cl_amp=c["cl_max"] - c["cl_min"],
               period=T, f=1.0 / T, cd_avg=c["cd_avg"], cl_avg=c["cl_avg"],
               t0=c["t0"], t1=c["t1"], t_clmax=c["t_clmax"], spread=spread, n_cycles=len(last))
    return res


def dominant_frequency(t, y, f_guess=None):
    """Frequency of the strongest spectral peak (zero-padded FFT + parabolic peak refinement)."""
    y = np.asarray(y) - np.mean(y)
    dt = t[1] - t[0]
    n = len(y)
    w = np.hanning(n)
    nfft = 16 * int(2 ** np.ceil(np.log2(n)))
    spec = np.abs(np.fft.rfft(y * w, nfft))
    freqs = np.fft.rfftfreq(nfft, dt)
    k = int(np.argmax(spec[1:]) + 1)
    a, b, c = np.log(spec[k - 1:k + 2] + 1e-300)
    d = 0.5 * (a - c) / (a - 2 * b + c)
    return float(freqs[k] + d * (freqs[1] - freqs[0]))
