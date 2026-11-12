"""Steady flows with gpucfd: march to steady state, stop on an error estimate, and grid sequencing.

gpucfd reaches a steady state by marching in time. Two things make that slow on fine grids, and this
module addresses both:

1. Where to stop. The approach to steady state is geometric (the slowest decaying mode dominates), so
   the error still left is about |last change| * rho / (1 - rho) (cfdlib.verification.iteration_error).
   That error only has to be small next to the grid's own discretisation error, and on grid k that is
   known while it runs: about |f_k - f_(k-1)| / (r^p - 1). march() stops when the estimate is below a
   fraction (default 1 %) of it, instead of at a fixed, much smaller change per time unit.
2. Where to start. sequence() solves the grids coarse to fine and starts each from the previous grid's
   solution (prolong()), so the fine grid only has to remove the coarse grid's error, not the whole
   start-up transient.

The answers are the same steady states as from rest (to within the stopping tolerance); a test of this
is in tests/test_steady.py.
"""
import math
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cfdlib.verification import iteration_error, level_sequence


def _axes(s, comp):
    """1D face coordinates of component comp per direction (z, y, x order) and the periodic flags."""
    ax, per = [], []
    for d in (2, 1, 0):
        h = s.length[d] / s.n[d]
        ax.append(np.arange(s.n[d]) * h + (0.0 if d == comp else 0.5 * h))
        per.append(s.kinds[2 * d] == 0)
    return ax, per


def prolong(src, dst):
    """Start dst from the velocity of src (same domain and boundaries, any grids): (bi/tri)linear
    interpolation of each component onto dst's faces, then a projection. Faces in bodies get the
    interpolated values too; the ghost-cell rules overwrite them at the projection. Returns max |div u|
    after the projection."""
    if not np.allclose(src.length, dst.length) or src.ncomp != dst.ncomp:
        raise ValueError("prolong: grids must cover the same domain in the same dimension")
    q = src.get()
    out = []
    for c in range(dst.ncomp):
        ax, per = _axes(src, c)
        a = q[c]
        keep = [i for i in range(3) if src.n[2 - i] > 1]  # directions with more than one cell
        pts, data = [], a
        for i in keep:
            x = ax[i]
            if per[i]:  # wrap one face on each side so the interpolation sees the periodic neighbours
                L = src.length[2 - i]
                x = np.concatenate(([x[-1] - L], x, [x[0] + L]))
                data = np.concatenate((np.take(data, [-1], axis=i), data, np.take(data, [0], axis=i)), axis=i)
            pts.append(x)
        data = data.reshape([data.shape[i] for i in keep])
        f = RegularGridInterpolator(pts, data, method="linear", bounds_error=False, fill_value=None)
        X, Y, Z = dst.face_coords(c)
        P = np.stack([(Z, Y, X)[i].ravel() for i in keep], axis=-1)
        out.append(f(P).reshape(dst.shape))
    dst.set(out)
    return dst.project()


def march(s, quantities, dt, tol, chunk=1.0, t_max=math.inf, gpu_max=math.inf, ptol=1e-10, window=3,
          hold=2, t_min=0.0, rho_min=None, log=print, tag=""):
    """March s in chunks of `chunk` time units until every quantity's iteration error estimate is below
    its tolerance, or is at a noise floor (solver tolerances; marching longer cannot reduce it: reported
    in noise_limited when the floor is above the tolerance), on `hold` consecutive chunks (a fast mode dying out can look converged for a moment
    before the slow mode shows), and not before t_min. rho_min: {name: slowest rate per chunk measured
    elsewhere} (see iteration_error; sequence() passes the previous grid's). quantities(s) -> {name: value}. tol: {name: abs tol} or a function
    (name, value) -> abs tol (re-evaluated every chunk). Logs one line per chunk (watch it live).
    Returns dict(values, errors, rho (last settled geometric rate per chunk, for a finer grid's floor),
    noise_limited, extrapolated, t, steps, gpu_s, wall_s, converged, history)."""
    n = max(1, int(math.ceil(chunk / dt)))
    hist, t, steps, ms, t0, good, rate = {}, 0.0, 0, 0.0, time.time(), 0, {}
    while True:
        st = s.run(dt, n, ptol)
        t += n * dt
        steps += n
        ms += st["ms"]
        vals = quantities(s)
        est, ok = {}, True
        for k, v in vals.items():
            hist.setdefault(k, []).append(v)
            e = iteration_error(hist[k], window, (rho_min or {}).get(k))
            lim = tol(k, v) if callable(tol) else tol[k]
            est[k] = (e, lim)
            ok &= e["error"] < lim or e["noise"]
            if math.isfinite(e["error"]) and not e["noise"] and e["rho"] is not None:
                rate[k] = e["rho"]
        if log:
            log(f"{tag}t {t:8.2f} " + "  ".join(
                f"{k} {v:.7g} (err {est[k][0]['error']:.1e}/{est[k][1]:.1e}"
                + (", noise floor)" if est[k][0].get("noise") else
                   f", rho {est[k][0]['rho']:.3f})" if est[k][0]["rho"] is not None else ")")
                for k, v in vals.items()) + f"  div {st['max_div']:.1e} gpu {ms / 1000:.1f}s")
        good = good + 1 if ok and t >= t_min else 0
        if good >= hold or t >= t_max or ms / 1000 >= gpu_max:
            break
    return dict(values=vals, errors={k: e[0]["error"] for k, e in est.items()}, rho=rate,
                noise_limited={k: bool(e[0]["noise"] and e[0]["error"] >= e[1]) for k, e in est.items()},
                extrapolated={k: e[0]["extrapolated"] for k, e in est.items()}, t=t, steps=steps,
                gpu_s=ms / 1000, wall_s=time.time() - t0, converged=bool(ok), history=hist)


def sequence(make, start, quantities, levels, r=2.0, frac=0.01, coarse_rtol=1e-5, rtol_floor=1e-7,
             p_assumed=2.0, chunk=1.0, dt_of=None, log=print, cold=False, target=None, t_min=0.0, **kw):
    """Solve grids levels[0] (coarsest) .. levels[-1] coarse to fine, each started from the previous one.

    make(level) -> NavierStokes; start(s) sets the coarsest grid's initial field (or every grid's, with
    cold=True: the from-rest comparison). quantities(s) -> {name: value}. dt_of(s) -> time step (default
    s.max_dt of the largest |u| at the start). Grid 0 stops at relative error coarse_rtol; grid k > 0 at
    frac times its discretisation error estimate |f_k - f_(k-1)| / (r^p - 1), p the observed order
    once three grids exist (else p_assumed), never below rtol_floor * |f|.
    t_min: minimum marching time on the first grid (or every grid with cold=True): its history alone
    cannot reveal a slow mode hidden under a faster one, so give it a physical time scale (e.g. the
    viscous time L^2 / nu or a few flow-through times). The finer grids use the first grid's measured
    slow rate as a floor instead (rho_min of iteration_error).
    target: {name: relative error}: stop refining once every named quantity's discretisation error
    estimate on the current grid, |f_k - f_(k-1)| / (r^p - 1), is below it (p observed from three
    grids, else p_assumed); the remaining levels are skipped.
    Returns dict(levels: [march results + level, n, prolong_div, disc_err], study: {name:
    level_sequence(...)}, settled: True if the target was met)."""
    res, prev = [], None
    for i, lev in enumerate(levels):
        s = make(lev)
        div0 = None
        if prev is None or cold:
            start(s)
        else:
            div0 = prolong(prev, s)
        if prev is not None:
            prev.close()
        dt = dt_of(s) if dt_of else s.max_dt(max(float(np.abs(a).max()) for a in s.get()))
        if i == 0:
            tol = lambda k, v: coarse_rtol * abs(v)
        else:
            fs = {k: [x["values"][k] for x in res] for k in res[-1]["values"]}

            def tol(k, v, fs=fs):
                f = fs[k] + [v]
                p = p_assumed
                if len(f) >= 3:
                    seq = level_sequence(f[-3:], r)
                    if seq["order"] is not None and 0.5 < seq["order"] < 6:
                        p = seq["order"]
                return max(frac * abs(v - f[-2]) / (r ** p - 1), rtol_floor * abs(v))
        floor = None if (cold or not res) else {k: v for k, v in res[-1]["rho"].items() if v < 0.999}
        out = march(s, quantities, dt, tol, chunk=chunk, log=log, tag=f"[{lev}] ",
                    t_min=t_min if (cold or not res) else 0.0, rho_min=floor, **kw)
        out.update(level=lev, n=s.n, prolong_div=div0)
        res.append(out)
        if log:
            log(f"[{lev}] done: t {out['t']:.1f} steps {out['steps']} gpu {out['gpu_s']:.1f}s converged "
                f"{out['converged']} " + " ".join(f"{k} {v:.7g}" for k, v in out["values"].items()))
        prev = s
        if i > 0:
            de = {}
            for k in out["values"]:
                f = [x["values"][k] for x in res]
                p = p_assumed
                if len(f) >= 3:
                    seq = level_sequence(f[-3:], r)
                    if seq["order"] is not None and 0.5 < seq["order"] < 6:
                        p = seq["order"]
                de[k] = abs(f[-1] - f[-2]) / (r ** p - 1) / max(abs(f[-1]), 1e-300)
            out["disc_err"] = de
            if log:
                log(f"[{lev}] discretisation error estimate (relative): " + " ".join(f"{k} {v:.1e}" for k, v in de.items()))
            if target and all(de[k] < target[k] for k in target):
                settled = True
                break
    else:
        settled = False
    if prev is not None:
        prev.close()
    study = {k: level_sequence([x["values"][k] for x in res], r) for k in res[-1]["values"]}
    return dict(levels=res, study=study, settled=bool(target) and settled)
