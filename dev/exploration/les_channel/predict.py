"""Prediction for turbulence in gpucfd by the eddy-resolving route (LES / DNS), made by measuring.

Kept as the record of the exploration of 2026-11-14. Two things changed afterwards: the published
values below were quoted from memory, and the transcribed reference (Vreman and Kuerten 2014, case 17,
cases/turbulent_channel_re180/reference/) gives a bulk velocity of 15.70, not 15.63; and the helpers
written here became tools (gpucfd.turbulence, gpucfd.grids.tanh_wall, cfdlib.signals.stationary).

    PYTHONPATH=. python dev/exploration/les_channel/predict.py run  [nx ny nz] [t_end]
    PYTHONPATH=. python dev/exploration/les_channel/predict.py time
    (inside WSL, needs the GPU; results in dev/exploration/les_channel/results.json)

The flow: turbulent flow between two parallel plates (the plane channel) at friction Reynolds number
180, the standard first test of an eddy-resolving code (Kim, Moin and Moser 1987). Units: half the
distance between the plates = 1, friction velocity u_tau = 1, so nu = 1/180 and the driving force per
unit mass is 1 (it balances the wall shear stress of both plates exactly). Box 4 pi x 2 x 4 pi / 3,
periodic along the flow (x) and across it (z).

gpucfd as it is today solves this flow with NO turbulence model: on a grid too coarse to resolve every
eddy that is an under-resolved simulation, the baseline an LES model has to improve on.

run:  start from a turbulent mean profile with large disturbances, march to t_end, average over the
      second half. Measures the time step the flow allows, the cost per step, and the answer.
time: cost per step on finer grids, both time schemes, from the developed field of `run` put on them.

Written down BEFORE the first run (2026-11-14):
- Expected: turbulence is sustained on 64^3; the mean speed through the channel (bulk velocity) comes
  out within 10 % of the published 15.6 u_tau, too high rather than too low (coarse grids of this kind
  of scheme lose wall friction). Published values quoted from memory of Kim, Moin and Moser (1987):
  bulk velocity 15.63, centreline 18.20, friction coefficient 8.18e-3. To be transcribed from the
  source before this becomes a case.
- The run ends at t_end (default 60, in units of half-height / u_tau). It is stopped early if the
  flow goes laminar (wall shear below 0.6 of the driving force, averaged over 5 time units, any time
  after t = 20) or the bulk velocity passes 25.
- The average counts as settled if the mean wall shear over the averaging window is within 3 % of the
  driving force (the momentum balance of the whole channel).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from gpucfd import gcfd, grids

HERE = Path(__file__).parent
OUT = HERE / "results.json"
FIELD = Path.home() / "les_channel_field.npz"      # the developed field (large; kept outside the repository)
RE_TAU = 180.0
NU = 1.0 / RE_TAU
LX, LY, LZ = 4.0 * np.pi, 2.0, 4.0 * np.pi / 3.0
PUBLISHED = dict(bulk=15.63, centre=18.20, cf=8.18e-3, source="Kim, Moin & Moser (1987), quoted from memory")


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(d):
    OUT.write_text(json.dumps(d, indent=1, default=float))


def reichardt(yp):
    """A smooth fit of the mean velocity of a turbulent wall layer (Reichardt 1951), in wall units."""
    k = 0.41
    return np.log(1.0 + k * yp) / k + 7.8 * (1.0 - np.exp(-yp / 11.0) - yp / 11.0 * np.exp(-yp / 3.0))


def wall_grid(ny, dy1_plus):
    """Face positions across the channel with the wall cell dy1_plus wall units high (tanh spacing)."""
    lo, hi = 1.0, 500.0
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if np.diff(grids.tanh(ny, LY, mid)).min() * RE_TAU > dy1_plus else (lo, mid)
    return grids.tanh(ny, LY, np.sqrt(lo * hi))


def make(n, scheme, dy1_plus=1.0):
    y = wall_grid(n[1], dy1_plus)
    s = gcfd.NavierStokes(n, (LX, LY, LZ), NU, walls=(False, True, False), force=(1.0, 0.0, 0.0), scheme=scheme,
                          grid=(None, y, None))
    return s, y


def describe(n, y):
    dy = np.diff(y)
    return dict(n=list(n), cells=int(np.prod(n)), dx_plus=LX / n[0] * RE_TAU, dz_plus=LZ / n[2] * RE_TAU,
                dy_plus_wall=float(dy.min() * RE_TAU), dy_plus_centre=float(dy.max() * RE_TAU),
                aspect_ratio=float(LX / n[0] / dy.min()))


def disturbed_profile(s, seed=0):
    """Turbulent mean profile plus streaks, vortices and noise: enough to start turbulence quickly."""
    rng = np.random.default_rng(seed)
    kx, kz, uc = 2.0 * np.pi / LX, 2.0 * np.pi / LZ, 18.0
    out = []
    for c in range(3):
        X, Y, Z = s.face_coords(c)
        env = 1.0 - (Y - 1.0) ** 2
        if c == 0:
            a = reichardt(np.minimum(Y, 2.0 - Y) * RE_TAU) + 0.25 * uc * env * np.cos(3 * kz * Z) * (1.0 + 0.3 * np.sin(2 * kx * X))
        elif c == 1:
            a = 0.10 * uc * env ** 2 * np.cos(3 * kz * Z) * np.cos(kx * X)
        else:
            a = 0.10 * uc * env * np.sin(3 * kz * Z) * np.sin(2 * kx * X)
        out.append(a + 0.05 * uc * env * rng.standard_normal(a.shape))
    s.set(out)
    s.project()


def interp_axis(a, xo, xn, axis, period=None, ends=None):
    """Linear interpolation of a along one axis from points xo to xn. period: periodic direction.
    ends: (x_low, x_high) where the values are zero (walls)."""
    a = np.moveaxis(a, axis, 0)
    if period is not None:
        xo = np.concatenate(([xo[-1] - period], xo, [xo[0] + period]))
        a = np.concatenate((a[-1:], a, a[:1]))
    if ends is not None:
        zero = np.zeros_like(a[:1])
        if ends[0] < xo[0]:
            xo, a = np.concatenate(([ends[0]], xo)), np.concatenate((zero, a))
        if ends[1] > xo[-1]:
            xo, a = np.concatenate((xo, [ends[1]])), np.concatenate((a, zero))
    i = np.clip(np.searchsorted(xo, xn), 1, len(xo) - 1)
    w = ((xn - xo[i - 1]) / (xo[i] - xo[i - 1])).reshape((-1,) + (1,) * (a.ndim - 1))
    return np.moveaxis(a[i - 1] * (1.0 - w) + a[i] * w, 0, axis)


def put_field(s, src):
    """Set s from a stored field of another grid (same box), then make it divergence-free."""
    out = []
    for c in range(3):
        a = src[f"q{c}"]
        for d, axis in ((0, 2), (1, 1), (2, 0)):
            xo, xn = src[f"x{c}{d}"], s._axis(d, d == c)
            if d == 1:
                a = interp_axis(a, xo, xn, axis, ends=(0.0, LY))
            else:
                a = interp_axis(a, xo, xn, axis, period=(LX, LZ)[d // 2])
        out.append(a)
    s.set(out)
    s.project()


def store_field(s, path):
    q = s.get()
    d = {f"q{c}": q[c] for c in range(3)}
    d.update({f"x{c}{dd}": s._axis(dd, dd == c) for c in range(3) for dd in range(3)})
    d["p"] = s.pressure()
    np.savez(path, **d)


def sample(s, q, y):
    """Plane averages of one field: mean velocity U(y), bulk velocity, wall shear at both plates, and the
    fluctuation intensities."""
    dy = np.diff(y)
    U = q[0].mean(axis=(0, 2))
    up = q[0] - U[None, :, None]
    return dict(U=U, bulk=float(np.sum(U * dy) / LY), tau_low=float(NU * U[0] / (0.5 * dy[0])),
                tau_high=float(NU * U[-1] / (0.5 * dy[-1])), uu=(up ** 2).mean(axis=(0, 2)),
                vv=(q[1] ** 2).mean(axis=(0, 2)), ww=((q[2] - q[2].mean(axis=(0, 2))[None, :, None]) ** 2).mean(axis=(0, 2)))


def run(n=(64, 64, 64), t_end=60.0, scheme="imex", ptol=1e-8, chunk=25, start=None, dy1_plus=1.0):
    s, y = make(n, scheme, dy1_plus)
    info = describe(n, y)
    print("grid:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}, flush=True)
    if start is not None and Path(start).exists():
        put_field(s, np.load(start))
        print("started from the stored field", start, flush=True)
    else:
        disturbed_profile(s)
    yc = 0.5 * (y[:-1] + y[1:])
    t, steps, gpu_ms, dt = 0.0, 0, 0.0, None
    series, acc, nacc, its, hits, stopped = [], None, 0, [], [], None
    wall0 = time.perf_counter()
    while t < t_end:
        lim = s.dt_cfl(0.8)
        if dt is None or dt > lim or dt < 0.8 * lim:
            dt = 0.9 * lim
        st = s.run(dt, chunk, ptol)
        t += chunk * dt
        steps += chunk
        gpu_ms += st["ms"]
        its.append(st["poisson_its"])
        hits.append(st["helm_its"])
        m = sample(s, s.get(), y)
        series.append([t, dt, m["bulk"], m["tau_low"], m["tau_high"], st["max_div"]])
        if t > 0.5 * t_end:
            keep = {k: m[k] for k in ("U", "uu", "vv", "ww")}
            acc = keep if acc is None else {k: acc[k] + keep[k] for k in keep}
            nacc += 1
        if len(series) % 40 == 0:
            print(f"t {t:6.2f}  dt {dt:.2e}  bulk {m['bulk']:6.3f}  wall shear {m['tau_low']:.3f} {m['tau_high']:.3f}  "
                  f"pressure its {st['poisson_its']:.1f}  ms/step {st['ms'] / chunk:.1f}", flush=True)
        a = np.array(series)
        recent = a[a[:, 0] > t - 5.0]
        if t > 20.0 and recent[:, 3:5].mean() < 0.6:
            stopped = "laminar: wall shear below 0.6 of the driving force over the last 5 time units"
        if m["bulk"] > 25.0 or not np.isfinite(m["bulk"]):
            stopped = "bulk velocity passed 25"
        if stopped:
            print("STOPPED EARLY:", stopped, flush=True)
            break
    wall = time.perf_counter() - wall0
    a = np.array(series)
    win = a[a[:, 0] > 0.5 * t_end]
    res = dict(grid=info, scheme=scheme, ptol=ptol, t_end=t, steps=steps, gpu_s=gpu_ms / 1e3, wall_s=wall,
               ms_per_step=gpu_ms / steps, dt_mean_window=float(win[:, 1].mean()) if len(win) else None,
               pressure_its=float(np.mean(its)), diffusion_its=float(np.mean(hits)), stopped_early=stopped,
               series=dict(columns=["t", "dt", "bulk", "tau_low", "tau_high", "max_div"], rows=a[::4].tolist()))
    if nacc and not stopped:
        U = acc["U"] / nacc
        bulk = float(win[:, 2].mean())
        tau = float(win[:, 3:5].mean())
        half = len(U) // 2
        Uh = 0.5 * (U[:half] + U[::-1][:half])
        res.update(window=[0.5 * t_end, t], samples=nacc, bulk=bulk, bulk_std=float(win[:, 2].std()),
                   wall_shear_mean=tau, settled=bool(abs(tau - 1.0) < 0.03), centre=float(0.5 * (U[half - 1] + U[half])),
                   cf=2.0 * tau / bulk ** 2, bulk_error_percent=100.0 * (bulk / PUBLISHED["bulk"] - 1.0),
                   y_plus=(yc[:half] * RE_TAU).tolist(), U_plus=Uh.tolist(),
                   u_rms_peak=float(np.sqrt((acc["uu"] / nacc).max())),
                   u_rms_peak_y_plus=float(min(yc[int(np.argmax(acc["uu"]))], 2 - yc[int(np.argmax(acc["uu"]))]) * RE_TAU),
                   v_rms_max=float(np.sqrt((acc["vv"] / nacc).max())), w_rms_max=float(np.sqrt((acc["ww"] / nacc).max())),
                   published=PUBLISHED)
        print(f"\nresult on {n}: bulk velocity {bulk:.3f} +/- {res['bulk_std']:.3f} (published {PUBLISHED['bulk']}: "
              f"{res['bulk_error_percent']:+.1f} %), centre {res['centre']:.2f} ({PUBLISHED['centre']}), "
              f"mean wall shear {tau:.3f} (1 = settled), u rms peak {res['u_rms_peak']:.2f} at y+ {res['u_rms_peak_y_plus']:.0f}")
    print(f"cost: {steps} steps, {gpu_ms / 1e3:.0f} s on the GPU ({wall:.0f} s wall), {gpu_ms / steps:.1f} ms per step, "
          f"pressure iterations {np.mean(its):.1f}, diffusion iterations {np.mean(hits):.1f}", flush=True)
    store_field(s, FIELD.with_name(f"les_channel_field_{n[0]}x{n[1]}x{n[2]}.npz"))
    s.close()
    d = load()
    d[f"run_{n[0]}x{n[1]}x{n[2]}_{scheme}"] = res
    save(d)
    return res


def timing(grids_=((64, 64, 64), (192, 64, 64), (128, 128, 128), (256, 128, 128)), source=(64, 64, 64)):
    """Cost per step on four grids. (192, 64, 64) has cells as long as they are wide (dx = dz), to see how
    much of the pressure solver's work comes from cells three times longer than wide. The looser pressure
    tolerance (1e-6) is timed on the first two grids only; its effect on the answer is not checked here."""
    src = np.load(FIELD.with_name(f"les_channel_field_{source[0]}x{source[1]}x{source[2]}.npz"))
    d = load()
    for n in grids_:
        for scheme in ("imex", "rk3"):
            for ptol in ((1e-8, 1e-6) if scheme == "imex" and np.prod(n) < 1e6 else (1e-8,)):
                s, y = make(n, scheme)
                put_field(s, src)
                dt = 0.9 * s.dt_cfl(0.8)
                s.run(dt, 100, ptol)                          # let the interpolated field adjust
                dt = 0.9 * s.dt_cfl(0.8)
                best = None
                for _ in range(2):
                    st = s.run(dt, 200, ptol)
                    if best is None or st["ms"] < best["ms"]:
                        best = st
                bulk = sample(s, s.get(), y)["bulk"]
                s.close()
                row = dict(grid=describe(n, y), scheme=scheme, ptol=ptol, dt=dt, ms_per_step=best["ms"] / 200,
                           pressure_its=best["poisson_its"], diffusion_its=best["helm_its"], bulk=bulk,
                           steps_per_time_unit=1.0 / dt, gpu_s_per_time_unit=best["ms"] / 200 / 1e3 / dt,
                           million_cell_steps_per_s=np.prod(n) / (best["ms"] / 200 / 1e3) / 1e6)
                d[f"time_{n[0]}x{n[1]}x{n[2]}_{scheme}_{ptol:g}"] = row
                save(d)
                print(f"{n} {scheme} ptol {ptol:g}: dt {dt:.2e}, {row['ms_per_step']:.1f} ms per step, pressure its "
                      f"{best['poisson_its']:.1f}, {row['gpu_s_per_time_unit']:.1f} GPU-s per time unit, "
                      f"{row['million_cell_steps_per_s']:.1f} million cell-steps per second", flush=True)


def summary(t_run=30.0):
    """The comparison table from results.json: per grid, each solver's cost per time unit (half-height /
    u_tau) at the same time step, and the time of a run of t_run time units."""
    d = load()
    print(f"{'grid':>14} {'cells':>9} {'dx+':>5} {'dz+':>5} | gpucfd ms/step (its)  s per unit | OpenFOAM ms/step (cores)  s per unit | "
          f"ratio | run of {t_run:g} units: gpucfd, OpenFOAM")
    for key in sorted((k for k in d if k.startswith("openfoam_")), key=lambda k: d[k]["grid"]["cells"]):
        of, g = d[key], d.get("time_" + key[9:] + "_imex_1e-08")
        if g is None:
            continue
        of_unit = of["best"]["s_per_step"] / of["dt"]
        loose = d.get("time_" + key[9:] + "_imex_1e-06")
        rk = d.get("time_" + key[9:] + "_rk3_1e-08")
        print(f"{key[9:]:>14} {g['grid']['cells']:>9,} {g['grid']['dx_plus']:5.1f} {g['grid']['dz_plus']:5.1f} | "
              f"{g['ms_per_step']:8.1f} ({g['pressure_its']:4.1f})  {g['gpu_s_per_time_unit']:8.1f}   | "
              f"{of['best']['s_per_step'] * 1e3:8.0f} ({of['best']['cores']})  {of_unit:10.0f}     | {of_unit / g['gpu_s_per_time_unit']:5.1f} | "
              f"{g['gpu_s_per_time_unit'] * t_run / 60:6.0f} min, {of_unit * t_run / 3600:5.1f} h"
              + (f"   [explicit scheme: {rk['gpu_s_per_time_unit']:.0f} s per unit]" if rk else "")
              + (f"   [pressure tolerance 1e-6: {loose['gpu_s_per_time_unit']:.0f} s per unit]" if loose else ""))
    for k in sorted(k for k in d if k.startswith("run_")):
        r = d[k]
        if "bulk" in r:
            print(f"{k}: bulk velocity {r['bulk']:.3f} ({r['bulk_error_percent']:+.1f} % against {PUBLISHED['bulk']}), centre "
                  f"{r['centre']:.2f} ({PUBLISHED['centre']}), friction coefficient {r['cf']:.3e} ({100 * (r['cf'] / PUBLISHED['cf'] - 1):+.0f} %), "
                  f"u rms peak {r['u_rms_peak']:.2f} at y+ {r['u_rms_peak_y_plus']:.0f}, wall shear {r['wall_shear_mean']:.3f}; "
                  f"{r['steps']} steps, {r['gpu_s'] / 60:.0f} min on the GPU, mean time step {r['dt_mean_window']:.2e}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "summary":
        summary()
        sys.exit(0)
    print(gcfd.device(), flush=True)
    if cmd == "run":
        n = tuple(int(v) for v in sys.argv[2:5]) if len(sys.argv) >= 5 else (64, 64, 64)
        t_end = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0
        start = sys.argv[6] if len(sys.argv) > 6 else None
        run(n, t_end, start=start)
    elif cmd == "time":
        timing()
