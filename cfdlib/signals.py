"""Signals in time: periodic ones (cycles, refined peaks, frequency, a periodicity check) and ones that
fluctuate about a constant level without ever repeating (turbulence: stationary(), settle_time(),
stationary_watch(), at the end of this file).

For any oscillating output (vortex shedding forces, probe velocities, oscillating flows). Built from
case 4 (cases/dfg_cylinder_re100/signals.py), where these lessons were learned:
- Delimit cycles by UPWARD CROSSINGS of the mid-range level, not by minima or maxima: crossings sit on
  the steep part of the signal and are sharp; extrema are flat and their times jitter by a few time
  steps (a cycle-to-cycle "period" from minima jittered 0.4 % on a perfectly periodic signal).
- Refine each extremum with a parabola through the three samples around it, so peak values do not
  depend on where the time steps fall.
- Cross-check the period with an FFT of the signal and, ideally, with an independent record (a probe).

    from cfdlib import signals
    cyc = signals.cycles(t, lift, others={"drag": drag})
    ok, spread = signals.periodic(cyc, ["max", "min", "period"], rtol=5e-4, n_last=5)
    f_fft = signals.dominant_frequency(t[late], lift[late])
"""
import numpy as np


def refine_peak(t, y, i):
    """Parabolic refinement of the extremum at sample i: (time, value)."""
    if i <= 0 or i >= len(y) - 1:
        return float(t[i]), float(y[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2 * y1 + y2
    if den == 0:
        return float(t[i]), float(y1)
    d = 0.5 * (y0 - y2) / den                            # offset in samples, |d| <= 0.5
    return float(t[i] + d * (t[i + 1] - t[i])), float(y1 - 0.25 * (y0 - y2) * d)


def extrema(t, y, kind):
    """All local minima ('min') or maxima ('max') of a sampled signal, parabola-refined: [(t, y), ...]."""
    z = (-1 if kind == "min" else 1) * np.asarray(y)
    idx = np.nonzero((z[1:-1] > z[:-2]) & (z[1:-1] >= z[2:]))[0] + 1
    return [refine_peak(t, y, i) for i in idx]


def up_crossings(t, y, level):
    """Times where y crosses `level` upwards (linear interpolation between samples)."""
    t, z = np.asarray(t), np.asarray(y) - level
    i = np.nonzero((z[:-1] < 0) & (z[1:] >= 0))[0]
    return t[i] - z[i] * (t[i + 1] - t[i]) / (z[i + 1] - z[i])


def _stats(tt, y):
    hi = max(extrema(tt, y, "max"), key=lambda e: e[1], default=(float(tt[np.argmax(y)]), float(y.max())))
    lo = min(extrema(tt, y, "min"), key=lambda e: e[1], default=(float(tt[np.argmin(y)]), float(y.min())))
    return dict(max=hi[1], t_max=hi[0], min=lo[1], t_min=lo[0], amp=hi[1] - lo[1], mid=0.5 * (hi[1] + lo[1]),
                avg=float(np.trapezoid(y, tt) / (tt[-1] - tt[0])))


def cycles(t, y, others=None, window=None):
    """One dict per full cycle of y: t0, t1, period, and max/min/amp/mid/avg (+ times of max/min) of y
    and of every signal in `others` ({name: values}, stored as name_max, name_min, ...).

    Cycles run between consecutive upward crossings of y through its mid-range level, measured over
    the last `window` time units (default: the last third of the record, where the flow is developed)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    others = {k: np.asarray(v, float) for k, v in (others or {}).items()}
    window = window or (t[-1] - t[0]) / 3
    late = t > t[-1] - window
    level = 0.5 * (y[late].max() + y[late].min())
    tc = up_crossings(t, y, level)
    out = []
    for t0, t1 in zip(tc[:-1], tc[1:]):
        sel = (t >= t0) & (t <= t1)
        if sel.sum() < 5:
            continue
        c = dict(t0=float(t0), t1=float(t1), period=float(t1 - t0), **_stats(t[sel], y[sel]))
        for name, v in others.items():
            c.update({f"{name}_{k}": x for k, x in _stats(t[sel], v[sel]).items()})
        out.append(c)
    return out


def periodic(cyc, keys=("max", "min", "period"), rtol=5e-4, n_last=5):
    """Is the signal periodic? Largest cycle-to-cycle change of each key over the last n_last cycles,
    relative to the signal's amplitude (period: relative to the period). Returns (ok, {key: change})."""
    last = cyc[-n_last:]
    if len(last) < 2:
        return False, {}
    amp = abs(last[-1]["amp"]) or 1.0
    spread = {}
    for k in keys:
        scale = last[-1]["period"] if k == "period" else (abs(last[-1].get(k.split("_")[0] + "_amp", amp))
                                                          if "_" in k else amp)
        spread[k] = float(max(abs(a[k] - b[k]) for a, b in zip(last[:-1], last[1:])) / (scale or 1.0))
    return all(v <= rtol for v in spread.values()), spread


def dominant_frequency(t, y):
    """Frequency of the strongest spectral peak (Hann window, zero-padded FFT, parabolic peak refinement).
    Needs a uniform time step; use it on the developed part of the record."""
    t, y = np.asarray(t, float), np.asarray(y, float) - np.mean(y)
    dt = t[1] - t[0]
    nfft = 16 * int(2 ** np.ceil(np.log2(len(y))))
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)), nfft))
    freqs = np.fft.rfftfreq(nfft, dt)
    k = int(np.argmax(spec[1:]) + 1)
    a, b, c = np.log(spec[k - 1:k + 2] + 1e-300)
    return float(freqs[k] + 0.5 * (a - c) / (a - 2 * b + c) * (freqs[1] - freqs[0]))


# ----------------------------------------------------------------- signals that fluctuate about a level
# A turbulent flow never becomes steady or periodic. Its outputs are time averages, and an average over
# a finite time has a statistical error that no grid refinement removes: successive samples resemble
# each other, so 10000 samples may hold only 20 independent ones. Quote every average as mean +- error,
# and compare two results (two grids, or a result and a reference) against that error before reading a
# difference between them as a grid effect.

def _weights(t):
    """Time each sample stands for (trapezoid rule), so unevenly spaced samples average correctly."""
    w = np.empty(len(t))
    w[1:-1] = 0.5 * (t[2:] - t[:-2])
    w[0], w[-1] = 0.5 * (t[1] - t[0]), 0.5 * (t[-1] - t[-2])
    return w


def _block_means(t, y, w, nb):
    """Time-weighted means of y over nb blocks of equal duration (axis 0 = time)."""
    starts = np.searchsorted(t, np.linspace(t[0], t[-1], nb + 1)[:-1], side="left")
    if np.any(np.diff(np.append(starts, len(t))) == 0):
        raise ValueError("a block holds no sample: the sampling is too uneven for this many blocks")
    ww = w.reshape((-1,) + (1,) * (y.ndim - 1))
    return np.add.reduceat(ww * y, starts, axis=0) / np.add.reduceat(ww, starts, axis=0)


def stationary(t, y, t_start=None, min_blocks=8):
    """Time average of a signal that fluctuates about a constant level, with its statistical error.

    t: sample times; y: samples, one number per time or an array per time (axis 0 = time; a profile is
    then averaged point by point). t_start: leave out the samples before it (the start-up transient;
    settle_time() estimates it).
    The error comes from batch means: the record is cut into blocks of equal duration, and the scatter
    of the block means gives the standard error of the overall mean. Blocks shorter than the signal's
    memory resemble each other and give too small an error, so the error is evaluated for min_blocks,
    2 min_blocks, 4 min_blocks ... blocks (at least 4 samples each) and the largest is kept.
    Returns dict(mean, stderr (one standard error), std (of the samples), t_corr (the signal's memory,
    the integral time scale implied by stderr), n_independent (duration / (2 t_corr)), drift_sigma
    (mean of the second half minus mean of the first half, in units of that difference's own standard
    error: beyond about +-2 the level is still moving), t0, t1, samples, reliable).
    reliable is False when the record holds fewer than about 10 independent samples at any point: the
    error estimate is then itself uncertain by more than a third and the average needs a longer record.
    For an array signal mean, stderr, std, t_corr, n_independent and drift_sigma are arrays; reliable
    covers all points."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if t_start is not None:
        keep = t >= t_start
        t, y = t[keep], y[keep]
    n = len(t)
    if n < 4 * min_blocks:
        raise ValueError(f"stationary: {n} samples after t_start, need at least {4 * min_blocks}")
    w = _weights(t)

    def mean_of(yy, ww):
        return np.tensordot(ww / ww.sum(), yy, axes=(0, 0))
    mean = mean_of(y, w)
    std = np.sqrt(mean_of((y - mean) ** 2, w))
    stderr, nb = None, min_blocks
    while n // nb >= 4:
        try:
            e = _block_means(t, y, w, nb).std(axis=0, ddof=1) / np.sqrt(nb)
        except ValueError:      # unevenly spaced samples left a block empty: the coarser levels stand
            if stderr is None:
                raise
            break
        stderr = e if stderr is None else np.maximum(stderr, e)
        nb *= 2
    span = t[-1] - t[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_corr = np.where(std > 0, 0.5 * span * (stderr / np.where(std > 0, std, 1.0)) ** 2, 0.0)
        n_ind = np.where(t_corr > 0, span / (2.0 * np.where(t_corr > 0, t_corr, 1.0)), np.inf)
    half = t <= 0.5 * (t[0] + t[-1])
    k = max(2, min_blocks // 2)
    m1, m2 = mean_of(y[half], w[half]), mean_of(y[~half], w[~half])
    e1 = _block_means(t[half], y[half], w[half], k).std(axis=0, ddof=1) / np.sqrt(k)
    e2 = _block_means(t[~half], y[~half], w[~half], k).std(axis=0, ddof=1) / np.sqrt(k)
    es = np.sqrt(e1 ** 2 + e2 ** 2)
    drift = np.where(es > 0, (m2 - m1) / np.where(es > 0, es, 1.0), 0.0)
    scalar = y.ndim == 1

    def out(v):
        return float(v) if scalar else np.asarray(v)
    return dict(mean=out(mean), stderr=out(stderr), std=out(std), t_corr=out(t_corr), n_independent=out(n_ind),
                drift_sigma=out(drift), t0=float(t[0]), t1=float(t[-1]), samples=int(n),
                reliable=bool(np.min(n_ind) >= 10.0))


def settle_time(t, y, block=5):
    """Estimate where the start-up transient of a fluctuating signal ends (the MSER rule, White 1997).

    The samples are averaged in groups of `block`; the cut d is the one for which the remaining record
    has the smallest naive error of its mean, var(y[d:]) / (n - d). A transient inflates the variance,
    so cutting it lowers this number until the cut starts to waste settled data. The cut is searched in
    the first half only. If it lands at the end of that range the record has not settled (or is too
    short to tell): the returned dict says so. A slow steady drift is not caught here (a ramp that
    doubles the scatter over the record passes as settled); stationary()'s drift_sigma catches it.
    Returns dict(t_start, settled, fraction_cut). An estimate to start from, not a proof: check it with
    stationary(...)["drift_sigma"] and against the flow's own balance (a channel: mean wall shear
    against the driving force)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    m = len(y) // block
    if m < 10:
        raise ValueError("settle_time: record too short")
    z = y[:m * block].reshape(m, block).mean(axis=1)
    best, d_best = np.inf, 0
    for d in range(m // 2 + 1):
        v = z[d:].var() / (m - d)
        if v < best:
            best, d_best = v, d
    return dict(t_start=float(t[d_best * block]), settled=bool(d_best < m // 2), fraction_cut=d_best / m)


def stationary_watch(key, tol, relative=True, t_start=None, min_time=0.0, max_drift=2.0, min_blocks=8):
    """A live check for cfdlib.gpu.run_transient (its `watch`): stop averaging when the mean of the
    sampled series `key` is known well enough.

    Says yes when, over the record after the start-up transient (t_start, or settle_time()'s estimate
    when None), the standard error of the mean is below tol (relative=True: tol times |mean|), the
    estimate is reliable (at least 10 independent samples), the two halves agree (|drift_sigma| <=
    max_drift) and the averaged record is at least min_time long. The physics goes into the choice of
    key and tol: fix them before the run, from the accuracy the comparison needs."""
    def watch(t, d):
        tt, y = d["t"], d[key]
        try:
            t0 = t_start if t_start is not None else settle_time(tt, y)["t_start"]
            st = stationary(tt, y, t_start=t0, min_blocks=min_blocks)
        except ValueError:
            return None
        lim = tol * abs(st["mean"]) if relative else tol
        if (st["reliable"] and st["stderr"] <= lim and abs(st["drift_sigma"]) <= max_drift
                and st["t1"] - st["t0"] >= min_time):
            return (f"average of {key} settled: {st['mean']:.6g} +- {st['stderr']:.2g} over t = {st['t0']:.4g} to "
                    f"{st['t1']:.4g} ({st['n_independent']:.0f} independent samples, drift {st['drift_sigma']:+.1f} sigma)")
        return None
    return watch
