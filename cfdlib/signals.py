"""Periodic signals: cycles, refined peaks, frequency and a periodicity check.

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
