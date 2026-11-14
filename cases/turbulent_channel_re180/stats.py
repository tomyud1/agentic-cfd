"""Numbers of the turbulent channel: time averages of a run with their statistical errors, the reference
on the same footing, and a three-level grid study with unequal refinement ratios.

Used by analyze.py. Everything is in the problem's units; "plus" quantities are in wall units."""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from cfdlib import signals

from . import run as RUN

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "results" / HERE.name / "runs"


def fold_c(a):
    """Cell-centre profiles (last axis: ny cells): mean of the two halves, wall to centre."""
    h = a.shape[-1] // 2
    return 0.5 * (a[..., :h] + a[..., ::-1][..., :h])


def fold_sym(a):
    """Face profiles (ny + 1 faces) of a quantity that is the same in both halves: wall to centre plane."""
    h = (a.shape[-1] - 1) // 2
    return 0.5 * (a[..., :h + 1] + a[..., ::-1][..., :h + 1])


def fold_anti(a):
    """Face profiles of a quantity that changes sign between the halves (shear stress, dU/dy)."""
    h = (a.shape[-1] - 1) // 2
    return 0.5 * (a[..., :h + 1] - a[..., ::-1][..., :h + 1])


def peak(x, y):
    """Largest value of a profile and where: cubic spline through the seven points around the largest sample
    (the extraction of the reference file, setup/parse_vk2014_channel.py). Returns (x_peak, y_peak)."""
    k = int(np.argmax(y))
    k = min(max(k, 3), len(y) - 4)
    sp = CubicSpline(x[k - 3:k + 4], y[k - 3:k + 4])
    roots = [r for r in sp.derivative().roots(extrapolate=False) if x[k - 3] <= r <= x[k + 3]]
    if not roots:
        return float(x[k]), float(y[k])
    r = max(roots, key=lambda v: float(sp(v)))
    return float(r), float(sp(r))


def load_run(name):
    rec = json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))
    with np.load(RUNS / f"{name}.npz") as z:
        d = {k: z[k] for k in z.files}
    return rec, d


def averages(name, g, R):
    """Everything the report needs from one run. g: the given values with u_tau (run.given); R: the stop rule
    (numerics.yaml). The averaging window starts where the stop rule placed the end of the start-up."""
    rec, d = load_run(name)
    t = d["t"]
    f_delta = g["f"] * g["delta"]
    ut, nu, delta = g["u_tau"], g["nu"], g["delta"]
    stop = RUN.stop_state(t[-1], d, R, g)
    t0 = stop["t_start"]

    def S(y):
        return signals.stationary(t, y, t_start=t0)
    y, yf = np.array(rec["y"]), np.array(rec["y_faces"])
    h = len(y) // 2
    out = dict(name=name, rec=rec, stop=stop, t0=t0, t1=float(t[-1]), span=float(t[-1] - t0), t=t, bulk_series=d["bulk"],
               tau_series=0.5 * (d["tau_low"] + d["tau_high"]), tau_low_series=d["tau_low"], tau_high_series=d["tau_high"],
               y_plus=y[:h] * ut / nu, yf_plus=yf[:h + 1] * ut / nu, y=y[:h], yf=yf[:h + 1])
    # what the first stop rule saw (numerics.yaml, changes): cut by settle_time alone, plain block means
    cut1 = signals.settle_time(t, d["bulk"])["t_start"]
    s1 = signals.stationary(t, d["bulk"], t_start=cut1)
    out["first_rule"] = dict(mean=s1["mean"], stderr=s1["stderr"], t0=s1["t0"], n_independent=s1["n_independent"])
    # scalars. The bulk velocity's error knows its slow mode (run.bulk_average); the mean velocity everywhere moves
    # with that mode, so the centre value and the profile get the same relative error added in quadrature.
    out["bulk"] = RUN.bulk_average(t, d, t0, g, R)
    slow_rel = np.sqrt(max(out["bulk"]["stderr"] ** 2 - out["bulk"]["stderr_batch"] ** 2, 0.0)) / out["bulk"]["mean"]
    out["slow_rel"] = float(slow_rel)
    out["tau"], out["tau_low"], out["tau_high"] = S(out["tau_series"]), S(d["tau_low"]), S(d["tau_high"])
    out["tau_diff"] = S(d["tau_low"] - d["tau_high"])
    Uf = fold_c(d["U"])
    d1, d2 = delta - y[h - 1], delta - y[h - 2]                 # even parabola through the two points nearest the centre plane
    out["centre"] = S((Uf[:, h - 1] * d2 ** 2 - Uf[:, h - 2] * d1 ** 2) / (d2 ** 2 - d1 ** 2))
    out["centre"]["stderr_batch"] = out["centre"]["stderr"]
    out["centre"]["stderr"] = float(np.hypot(out["centre"]["stderr"], slow_rel * out["centre"]["mean"]))
    # profiles (mean and standard error at every point)
    Ubar = S(d["U"])["mean"]                                    # time-mean profile, both halves
    out["U"] = S(Uf)
    out["U"]["stderr"] = np.hypot(out["U"]["stderr"], slow_rel * out["U"]["mean"])
    var_U = (d["U"] - Ubar) ** 2                                # variance in time of the plane mean: part of u' about the full mean
    out["uu"] = S(fold_c(d["uu"] + var_U))
    out["uu_plane"] = S(fold_c(d["uu"]))
    out["ww"] = S(fold_c(d["ww"] + d["W"] ** 2))
    out["vv"] = S(fold_sym(d["vv"]))
    out["uv"] = S(fold_anti(d["uv"]))
    out["dUdy"] = S(fold_anti(d["dUdy"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        for q in ("uu", "ww", "vv"):
            r = np.sqrt(out[q]["mean"])
            out[q[0] + "_rms"] = dict(mean=r, stderr=np.where(r > 0, out[q]["stderr"] / (2 * np.where(r > 0, r, 1.0)), 0.0))
    out["shear"] = dict(mean=-out["uv"]["mean"] / ut ** 2, stderr=out["uv"]["stderr"] / ut ** 2)   # -<u'v'> / u_tau^2
    # total shear stress against the exact line, folded and for the whole channel
    visc = nu * out["dUdy"]["mean"]
    exact = f_delta * (1.0 - yf[:h + 1] / delta)
    out["stress"] = dict(viscous=visc, turbulent=-out["uv"]["mean"], total=visc - out["uv"]["mean"], exact=exact,
                         residual=visc - out["uv"]["mean"] - exact)
    full = nu * S(d["dUdy"])["mean"] - S(d["uv"])["mean"] - (f_delta - g["f"] * yf)
    out["stress_full_residual"] = full
    out["stress_max"] = float(np.max(np.abs(full)))
    out["stress_max_y"] = float(yf[int(np.argmax(np.abs(full)))])
    # requested scalars
    ub, ub_e = out["bulk"]["mean"], out["bulk"]["stderr"]
    out["cf"] = 2.0 * ut ** 2 / ub ** 2                         # wall shear exact (tau_w = rho f delta), as in the reference file
    out["cf_err"] = out["cf"] * 2.0 * ub_e / ub
    out["re_b"] = 2.0 * delta * ub / nu
    out["re_b_err"] = 2.0 * delta * ub_e / nu
    yp_pk, u_pk = peak(out["y_plus"], out["u_rms"]["mean"] / ut)
    out["urms_peak"], out["urms_peak_y"] = u_pk, yp_pk
    k = int(np.argmax(out["u_rms"]["mean"]))
    out["urms_peak_err"] = float(out["u_rms"]["stderr"][k] / ut)
    out["urms_plane_peak"] = peak(out["y_plus"], np.sqrt(out["uu_plane"]["mean"]) / ut)[1]
    ys_pk, s_pk = peak(out["yf_plus"], out["shear"]["mean"])
    out["shear_peak"], out["shear_peak_y"] = s_pk, ys_pk
    out["shear_peak_err"] = float(out["shear"]["stderr"][int(np.argmax(out["shear"]["mean"]))])
    # balance of the box over the window: mean wall shear against the driving force, and the change of the bulk velocity
    keep = t >= t0
    out["imbalance"] = out["tau"]["mean"] / f_delta - 1.0
    out["imbalance_low"] = out["tau_low"]["mean"] / f_delta - 1.0
    out["imbalance_high"] = out["tau_high"]["mean"] / f_delta - 1.0
    out["bulk_change_rate"] = float((d["bulk"][keep][-1] - d["bulk"][keep][0]) / (t[keep][-1] - t[keep][0]))
    out["halves_sigma"] = float(abs(out["tau_diff"]["mean"]) / out["tau_diff"]["stderr"]) if out["tau_diff"]["stderr"] > 0 else 0.0
    # the wall cell against the exact wall limit u+ = y+ - y+^2 / (2 Re_tau)
    yp1 = out["y_plus"][0]
    out["wall_exact"] = float(yp1 - yp1 ** 2 / (2 * ut * delta / nu))
    out["wall_sim"] = float(out["U"]["mean"][0] / ut)
    return out


# ------------------------------------------------------------------------------------------ the reference
def reference(ref):
    """The reference on this case's footing. Returns dict(scalars{name: dict(value, codes)}, at(q, y_plus) ->
    mean of the two codes' profiles at y_plus (NaN outside both), codes(q, y_plus) -> the two profiles)."""
    P = ref["profiles"]
    sp, fd = P["spectral"], P["finite_difference"]

    def spl(x, y, sign=1.0):
        return CubicSpline(np.array(x, float), sign * np.array(y, float), extrapolate=False)
    prof = dict(
        U=[spl(sp["y_plus"], sp["mean_velocity"]), spl(fd["y_plus"], fd["mean_velocity"])],
        u_rms=[spl(sp["y_plus"], sp["u_rms"]), spl(fd["y_plus"], fd["u_rms"])],
        w_rms=[spl(sp["y_plus"], sp["w_rms"]), spl(fd["y_plus"], fd["w_rms"])],
        v_rms=[spl(sp["y_plus"], sp["v_rms"]), spl(fd["y_plus_staggered"], fd["v_rms"])],
        uv=[spl(sp["y_plus"], sp["reynolds_shear_stress"], -1.0), spl(fd["y_plus_staggered"], fd["reynolds_shear_stress"], -1.0)])

    def codes(q, y_plus):
        return np.array([c(np.asarray(y_plus, float)) for c in prof[q]])

    def at(q, y_plus):
        v = codes(q, y_plus)
        ok = np.isfinite(v)
        n = ok.sum(axis=0)
        return np.where(n > 0, np.where(ok, v, 0.0).sum(axis=0) / np.maximum(n, 1), np.nan)
    names = dict(bulk="bulk_velocity", centre="centreline_velocity", cf="friction_coefficient", re_b="bulk_reynolds_number",
                 urms_peak="u_rms_peak", urms_peak_y="u_rms_peak_y_plus", shear_peak="reynolds_shear_stress_peak",
                 shear_peak_y="reynolds_shear_stress_peak_y_plus")
    scal = {}
    for k, n in names.items():
        a, b = float(ref["spectral"][n]), float(ref["finite_difference"][n])
        scal[k] = dict(value=0.5 * (a + b), spectral=a, finite_difference=b, half_range=0.5 * abs(a - b))
    return dict(scalars=scal, at=at, codes=codes, raw=ref)


def profile_difference(av, R, q, ut):
    """Largest difference between a run's folded profile and the reference along the profile.
    q: U, u_rms, v_rms, w_rms or uv (shear stress). Returns dict(max_abs, y_plus, diff[], y[], ref[], sim[])."""
    if q in ("v_rms", "uv"):
        yp = av["yf_plus"][1:]                                   # faces; the wall face is zero on both sides
        sim = (av["v_rms"]["mean"][1:] / ut) if q == "v_rms" else av["shear"]["mean"][1:]
        err = (av["v_rms"]["stderr"][1:] / ut) if q == "v_rms" else av["shear"]["stderr"][1:]
    else:
        yp = av["y_plus"]
        sim = (av["U"]["mean"] if q == "U" else av[q]["mean"]) / ut
        err = (av["U"]["stderr"] if q == "U" else av[q]["stderr"]) / ut
    ref = R["at"](q, yp)
    ok = np.isfinite(ref)
    diff = sim[ok] - ref[ok]
    k = int(np.argmax(np.abs(diff)))
    return dict(max_abs=float(np.abs(diff[k])), signed=float(diff[k]), y_plus=float(yp[ok][k]), sigma=float(err[ok][k]), diff=diff, y=yp[ok],
                ref=ref[ok], sim=sim[ok], err=err[ok])


# ------------------------------------------------------------------------------------------ grid study
def three_levels(f, h, safety=1.25, p_formal=2.0, p_min=0.5):
    """Observed order, extrapolated value and fine-grid convergence index from three levels (coarse to fine)
    with cell sizes h (any two ratios). Model: f_k = F + C h_k^p, so
        (f3 - f2) / (f2 - f1) = (h2^p - h3^p) / (h1^p - h2^p),
    solved for p; then F = f3 + (f3 - f2) / ((h2 / h3)^p - 1). For equal ratios this is
    cfdlib.verification.three_grid_study (checked in self_check()). When the three values do not approach a
    limit from one side with a believable order, the index is a placeholder with the formal order (reliable
    False), as in that tool.
    Returns dict(R, R_limit, monotonic, p, p_used, extrap, gci_abs, gci, delta, reliable, basis)."""
    f1, f2, f3 = (float(v) for v in f)
    h1, h2, h3 = (float(v) for v in h)
    R = (f3 - f2) / (f2 - f1) if f2 != f1 else float("nan")
    R0 = np.log(h2 / h3) / np.log(h1 / h2)                       # the ratio of the changes for p -> 0
    mono = bool(np.isfinite(R) and 0.0 < R < R0)
    p = float("nan")
    if mono:
        p = brentq(lambda q: (h2 ** q - h3 ** q) / (h1 ** q - h2 ** q) - R, 1e-6, 60.0)
    reliable = bool(mono and p >= p_min)
    p_use = p if mono else p_formal
    r32 = h2 / h3
    ext = f3 + (f3 - f2) / (r32 ** p_use - 1.0)
    gci_abs = safety * abs(f3 - f2) / (r32 ** p_use - 1.0)
    if not mono:
        basis = f"order {p_formal:g} assumed: the three values do not approach a limit from one side"
    elif p < p_min:
        basis = f"observed order {p:.2f} is below {p_min:g}: the values barely converge"
    else:
        basis = "observed order"
    return dict(R=R, R_limit=R0, monotonic=mono, p=p, p_used=p_use, extrap=ext, gci_abs=gci_abs,
                gci=gci_abs / abs(f3) if f3 else float("nan"), delta=abs(f3 - f2), reliable=reliable, basis=basis)


def three_levels_noise(f, e, h, n=4000, seed=0, **kw):
    """The same study for n sets of values drawn about f with standard errors e (the statistical error of each
    level): how much of the grid study is noise. Returns dict(share_monotonic, p_median, p_lo, p_hi (16 and 84
    per cent), extrap_median, extrap_lo, extrap_hi, extrap_formal_std (spread of the formal-order extrapolation))."""
    rng = np.random.default_rng(seed)
    draws = np.asarray(f, float) + np.asarray(e, float) * rng.standard_normal((n, 3))
    ps, ex, exf = [], [], []
    r32 = h[1] / h[2]
    for row in draws:
        s = three_levels(row, h, **kw)
        exf.append(row[2] + (row[2] - row[1]) / (r32 ** 2 - 1.0))
        if s["reliable"]:
            ps.append(s["p"])
            ex.append(s["extrap"])
    out = dict(share_monotonic=len(ps) / n, extrap_formal_std=float(np.std(exf)))
    if ps:
        out.update(p_median=float(np.median(ps)), p_lo=float(np.percentile(ps, 16)), p_hi=float(np.percentile(ps, 84)),
                   extrap_median=float(np.median(ex)), extrap_lo=float(np.percentile(ex, 16)), extrap_hi=float(np.percentile(ex, 84)))
    return out


def self_check():
    """three_levels against the repository's tool for equal ratios, and against values made with a known order."""
    from cfdlib import verification
    vals = [1.0, 1.3, 1.375]
    a, b = three_levels(vals, [4.0, 2.0, 1.0]), verification.three_grid_study(vals, r=2.0)
    same = all(abs(a[k] - b[kk]) < 1e-10 for k, kk in (("p", "p"), ("extrap", "extrap"), ("gci_abs", "gci_abs")))
    F, C, p = 15.7, -3.0, 1.7
    h = [1 / 96, 1 / 144, 1 / 192]
    made = three_levels([F + C * x ** p for x in h], h)
    known = abs(made["p"] - p) < 1e-8 and abs(made["extrap"] - F) < 1e-9
    return {"three-level study equals cfdlib.verification.three_grid_study for equal ratios": bool(same),
            "three-level study recovers a known order and limit for the ratios 1.5 and 4/3": bool(known)}
