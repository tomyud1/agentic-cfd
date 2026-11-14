"""Optimize stage of case 17, the turbulent channel (dev/LOOP.md step 8; plan and rules: plan.yaml next
to this file; target, budget and stop rule: ROADMAP.md 3b).

Usage (repository root, inside WSL, PYTHONPATH=.):
  python dev/optimize/turbulent_channel_re180/stage.py noise          # the baseline's gaps, and their scatter over short windows
  python dev/optimize/turbulent_channel_re180/stage.py pre <variant>  # adjust, scale the bulk velocity once, settle; saves the state
  python dev/optimize/turbulent_channel_re180/stage.py avg <variant> [time units | stop]
                                                                      # average from the saved state (again: continues it);
                                                                      # "stop": until the stop rule of plan.yaml holds
  python dev/optimize/turbulent_channel_re180/stage.py final <variant>  # that run against the baseline on the case's criteria
  python dev/optimize/turbulent_channel_re180/stage.py table          # every run against the baseline, the directions' shares,
                                                                      # the predicted gaps of the candidate grids, the choice

The solved case's folder is read, never written: its problem, reference, numerics (the limits) and the
baseline's run record and last field. Records of this stage: runs/ next to this file; fields and
checkpoints: ~/gpucfd_runs/turbulent_channel_re180/opt_<variant>.npz (outside git).
The profile gaps are formed exactly as in the case's report (cases/turbulent_channel_re180/stats.py):
largest difference along the folded profile from the mean of the reference's two codes.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cases.turbulent_channel_re180 import run as RUN
from cases.turbulent_channel_re180 import stats as ST
from cfdlib import gpu, signals

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CASE = ROOT / "cases" / "turbulent_channel_re180"
BASE_RUNS = ROOT / "results" / "turbulent_channel_re180" / "runs"
RUNS = HERE / "runs"
LOGS = HERE / "logs"
FIELDS = Path.home() / "gpucfd_runs" / "turbulent_channel_re180"
PLAN = yaml.safe_load((HERE / "plan.yaml").read_text(encoding="utf-8"))
QS = ("U", "u_rms", "v_rms", "w_rms", "uv")
TEST = bool(__import__("os").environ.get("STAGE_TEST"))     # a smoke test of the script: minutes in place of an hour,
if TEST:                                                    # its files apart from the stage's
    PLAN["phases"].update(adjust=0.2, settle=0.1, average=0.3)
    PLAN["numerics"]["store_every"] = 1
    RUNS, LOGS, FIELDS = HERE / "runs_test", HERE / "runs_test", FIELDS / "test"


def setup():
    problem, numerics = RUN.load()
    g = RUN.given(problem)
    ref = ST.reference(yaml.safe_load((CASE / "reference" / "vk2014_chan180.yaml").read_text(encoding="utf-8")))
    return g, ref, numerics["acceptance"]["profile_abs"], numerics


def all_variants():
    return {**PLAN["directional"], **PLAN.get("candidates", {})}


# ------------------------------------------------------------------------------------------ measuring
def gaps(t, d, y, yf, g, ref, t0=None, t1=None):
    """The five profile gaps of a record over the window t0..t1, formed as the case's report forms them
    (stats.averages and stats.profile_difference), plus the bulk velocity and the wall-shear imbalance.
    Returns dict(q: dict(signed, y_plus, sigma), bulk, imbalance, span, urms_peak)."""
    t = np.asarray(t, float)
    keep = np.ones(len(t), bool)
    if t0 is not None:
        keep &= t >= t0
    if t1 is not None:
        keep &= t <= t1
    t = t[keep]
    d = {k: np.asarray(v)[keep] for k, v in d.items() if len(v) == len(keep)}
    ut, nu = g["u_tau"], g["nu"]
    y, yf = np.asarray(y, float), np.asarray(yf, float)
    h = len(y) // 2

    def S(a):
        return signals.stationary(t, a)
    av = dict(y_plus=y[:h] * ut / nu, yf_plus=yf[:h + 1] * ut / nu)
    Ubar = S(d["U"])["mean"]
    av["U"] = S(ST.fold_c(d["U"]))
    uu = S(ST.fold_c(d["uu"] + (d["U"] - Ubar) ** 2))            # about the full mean: plus the variance in time of the plane mean
    ww = S(ST.fold_c(d["ww"] + d["W"] ** 2))
    vv = S(ST.fold_sym(d["vv"]))
    uv = S(ST.fold_anti(d["uv"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, s in (("u_rms", uu), ("w_rms", ww), ("v_rms", vv)):
            r = np.sqrt(s["mean"])
            av[name] = dict(mean=r, stderr=np.where(r > 0, s["stderr"] / (2 * np.where(r > 0, r, 1.0)), 0.0))
    av["shear"] = dict(mean=-uv["mean"] / ut ** 2, stderr=uv["stderr"] / ut ** 2)
    out = {}
    for q in QS:
        pd = ST.profile_difference(av, ref, q, ut)
        out[q] = dict(signed=pd["signed"], y_plus=pd["y_plus"], sigma=pd["sigma"], y=[float(v) for v in pd["y"]],
                      diff=[float(v) for v in pd["diff"]])
    tau = 0.5 * (d["tau_low"] + d["tau_high"])
    out["bulk"] = float(S(d["bulk"])["mean"])
    out["imbalance"] = float(S(tau)["mean"] / (g["f"] * g["delta"]) - 1.0)
    out["span"] = float(t[-1] - t[0])
    out["urms_peak_y"], out["urms_peak"] = (float(v) for v in ST.peak(av["y_plus"], av["u_rms"]["mean"] / ut))
    d1, d2 = g["delta"] - y[h - 1], g["delta"] - y[h - 2]         # even parabola through the two points nearest the centre plane
    out["centre"] = float((av["U"]["mean"][-1] * d2 ** 2 - av["U"]["mean"][-2] * d1 ** 2) / (d2 ** 2 - d1 ** 2) / ut)
    return out


def load_baseline():
    rec = json.loads((BASE_RUNS / f"{PLAN['baseline']['run']}.json").read_text(encoding="utf-8"))
    with np.load(BASE_RUNS / f"{PLAN['baseline']['run']}.npz") as z:
        d = {k: z[k] for k in z.files}
    t0 = float(rec["watch_first"]["text"].split("over t = ")[1].split(" to ")[0])
    return rec, d, t0


def load_variant(name):
    """All averaging parts of a variant joined in time: (records, t, series, y, y_faces), or None."""
    parts = sorted(RUNS.glob(f"{name}_avg*.json"))
    if not parts:
        return None
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in parts]
    ts, series = [], {}
    for k, p in enumerate(parts):
        with np.load(p.with_suffix(".npz")) as z:
            cut = 1 if k else 0                                 # a continued part starts with the last sample of the one before
            ts.append(z["t"][cut:])
            for key in z.files:
                if key not in ("t", "t_all") and not key.endswith("_all"):
                    series.setdefault(key, []).append(z[key][cut:])
    return recs, np.concatenate(ts), {k: np.concatenate(v) for k, v in series.items()}, np.array(recs[0]["y"]), np.array(recs[0]["y_faces"])


def fmt_gaps(G, limits):
    return "  ".join(f"{q} {G[q]['signed']:+.3f}@{G[q]['y_plus']:.0f}" + ("" if abs(G[q]["signed"]) <= limits[q] else "*") for q in QS)


def do_noise():
    """The baseline's gaps over its whole window (must be the report's numbers), and over windows of the
    length a directional run averages: their scatter is what a directional difference has to exceed."""
    g, ref, limits, _ = setup()
    rec, d, t0 = load_baseline()
    y, yf = np.array(rec["y"]), np.array(rec["y_faces"])
    full = gaps(d["t"], d, y, yf, g, ref, t0=t0)
    print(f"baseline {rec['n']}, window {t0:.1f} to {d['t'][-1]:.1f}: bulk {full['bulk']:.4f}  {fmt_gaps(full, limits)}   (* = outside its limit)")
    L = PLAN["phases"]["average"]
    wins = []
    a = t0
    while a + L <= d["t"][-1] + 1e-6:
        wins.append(gaps(d["t"], d, y, yf, g, ref, t0=a, t1=a + L))
        print(f"  window {a:5.1f} to {a + L:5.1f}: bulk {wins[-1]['bulk']:.4f} imbalance {wins[-1]['imbalance']:+.2%}  {fmt_gaps(wins[-1], limits)}")
        a += 0.5 * L                                            # half-overlapping windows: more of them, not independent
    scatter = {q: float(np.std([w[q]["signed"] for w in wins], ddof=1)) for q in QS}
    worst = {q: float(np.max([abs(w[q]["signed"] - full[q]["signed"]) for w in wins])) for q in QS}
    tau = 0.5 * (d["tau_low_all"] + d["tau_high_all"])
    tt = d["t_all"]
    m25 = [float(np.mean(tau[(tt >= a) & (tt < a + 2.5)])) for a in np.arange(t0, tt[-1] - 2.5, 2.5)]
    out = dict(window=L, n_windows=len(wins), full={q: {k: full[q][k] for k in ("signed", "y_plus", "sigma")} for q in QS}, bulk=full["bulk"], scatter=scatter, worst=worst,
               bulk_scatter=float(np.std([w["bulk"] for w in wins], ddof=1)), tau_2p5_std=float(np.std(m25, ddof=1)))
    print(f"scatter of the gaps over {len(wins)} windows of {L:g} time units (standard deviation; largest distance from the full window):")
    print("  " + "  ".join(f"{q} {scatter[q]:.3f} ({worst[q]:.3f})" for q in QS) + f"   bulk {out['bulk_scatter']:.3f}")
    print(f"mean wall shear over 2.5 time units scatters by {out['tau_2p5_std']:.2%}: the bulk velocity it points to by {out['tau_2p5_std'] / 1.75:.2%}")
    RUNS.mkdir(parents=True, exist_ok=True)
    (HERE / "noise.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ running
def make(spec, g):
    from gpucfd import gcfd, grids
    yfaces = grids.tanh_wall(spec["n"][1], 2 * g["delta"], spec["wall_cell_plus"] * g["nu"] / g["u_tau"])
    return gcfd.NavierStokes(spec["n"], (g["Lx"], 2 * g["delta"], g["Lz"]), g["nu"], walls=(False, True, False),
                             force=(g["f"], 0.0, 0.0), scheme=PLAN["numerics"]["scheme"], grid=(None, yfaces, None))


def logger(name):
    LOGS.mkdir(parents=True, exist_ok=True)
    fh = open(LOGS / f"{name}.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    return log


def state_path(name):
    return FIELDS / f"opt_{name}.npz"


def run_args(s, log, tag):
    from gpucfd import turbulence
    N = PLAN["numerics"]
    st = turbulence.PlaneStats(s)
    return dict(sample_dt=None, observe=lambda s_: st.sample(), u_ref=15.0, log=log, tag=tag, log_every=1.0, ptol=N["pressure_tol"],
                adapt_cfl=N["cfl"], sample_steps=N["steps_per_sample"])


def meta(s, spec, g, source):
    N = PLAN["numerics"]
    return dict(level=spec.get("label", ""), cells=int(np.prod(s.n)), n=list(s.n), h=s._hmin(), source=source, pressure_tol=N["pressure_tol"],
                scheme=s.scheme, time_step=dict(adaptive_cfl=N["cfl"], dt_cap=N["dt_cap"], steps_per_sample=N["steps_per_sample"]),
                y=[float(v) for v in s._axis(1, False)], y_faces=[float(v) for v in np.concatenate((s._axis(1, True), [s.length[1]]))],
                dx=s.length[0] / s.n[0], dz=s.length[2] / s.n[2], wall_cell_plus=spec["wall_cell_plus"])


def do_pre(name, source=None):
    """Adjust, scale the bulk velocity once to where the wall shear says it is heading, settle. The state at
    the end is saved for `avg`. source: the variant whose state to start from (default: the baseline's last field)."""
    from gpucfd import steady
    g, ref, limits, _ = setup()
    spec = all_variants()[name]
    P, N = PLAN["phases"], PLAN["numerics"]
    log = logger(name)
    src_spec = PLAN["baseline"] if source is None else all_variants()[source]
    src = make(src_spec, g)
    s = make(spec, g)
    try:
        if source is None:
            f = np.load(Path.home() / "gpucfd_runs" / "turbulent_channel_re180" / f"{PLAN['baseline']['run']}.npz")   # the solver agent's field: u, v, w, p, t
            src.set([f["u"], f["v"], f["w"]])
        else:
            gpu.load_state(src, state_path(source))
        div = steady.prolong(src, s)
        src.close()
        log(f"{name}: grid {list(s.n)} ({int(np.prod(s.n)):,} cells), wall cell {spec['wall_cell_plus']:.2f}, dx+ {180 * s.length[0] / s.n[0]:.1f}, "
            f"dz+ {180 * s.length[2] / s.n[2]:.1f}; field interpolated from {source or 'the baseline'}, max |div u| {div:.1e}")
        a = gpu.run_transient(s, N["dt_cap"], 0.0, P["adjust"], **run_args(s, log, f"{name} adjust: "))
        late = a["t"] >= min(1.5, 0.4 * P["adjust"])
        tau = float(np.average(0.5 * (a["series"]["tau_low"] + a["series"]["tau_high"])[late], weights=signals._weights(a["t"][late])))
        ub = float(np.average(a["series"]["bulk"][late], weights=signals._weights(a["t"][late])))
        factor = float(np.clip(1.0 - (tau / (g["f"] * g["delta"]) - 1.0) / 1.75, 0.97, 1.03))
        q, p = s.get(), s.pressure()
        s.set([c * factor for c in q])
        s.set_pressure(p * factor ** 2)
        log(f"{name}: over t = 1.5 to {P['adjust']:g} the wall shear was {tau:.4f} of the driving force at a bulk velocity of {ub:.4f}; "
            f"field scaled by {factor:.4f} (to {ub * factor:.4f})")
        b = gpu.run_transient(s, N["dt_cap"], P["adjust"], P["adjust"] + P["settle"], **run_args(s, log, f"{name} settle: "))
        gpu.save_state(s, state_path(name), float(b["t"][-1]))
        steps = a["perf"]["steps"] + b["perf"]["steps"]
        wall = a["perf"]["wall_s"] + b["perf"]["wall_s"]
        rec = dict(name=name, n=list(s.n), cells=int(np.prod(s.n)), source=source or PLAN["baseline"]["run"], div_start=div, factor=factor,
                   tau_adjust=tau, bulk_adjust=ub, t_end=float(b["t"][-1]), steps=steps, wall_s=wall,
                   ms_per_step_wall=1e3 * wall / steps, poisson_its=b["perf"]["poisson_its"], wall_per_unit_s=wall / float(b["t"][-1]),
                   health=[a["health"]["status"], b["health"]["status"]],
                   courant=[next(c["value"] for c in r["health"]["checks"] if c["name"] == "Max Courant number") for r in (a, b)],
                   bulk_end=float(b["series"]["bulk"][-1]),
                   tau_settle=float(np.mean(0.5 * (b["series"]["tau_low"] + b["series"]["tau_high"]))),
                   history=dict(t=[float(v) for v in np.concatenate((a["t"], b["t"]))[::20]],
                                bulk=[float(v) for v in np.concatenate((a["series"]["bulk"], b["series"]["bulk"]))[::20]],
                                tau=[float(v) for v in np.concatenate((0.5 * (a["series"]["tau_low"] + a["series"]["tau_high"]),
                                                                       0.5 * (b["series"]["tau_low"] + b["series"]["tau_high"])))[::20]]))
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"{name}_pre.json").write_text(json.dumps(rec, indent=1))
        log(f"{name}: PRE DONE at t = {rec['t_end']:.2f}: health {rec['health']}, Courant {rec['courant']}, {rec['ms_per_step_wall']:.1f} ms per step, "
            f"{rec['poisson_its']:.1f} pressure iterations, {rec['wall_per_unit_s']:.0f} s per time unit; wall shear while settling "
            f"{rec['tau_settle']:.4f}, bulk velocity at the end {rec['bulk_end']:.4f}")
    finally:
        s.close()


def do_avg(name, length):
    """Average from the saved state. length: time units, or "stop": until plan.yaml's stop rule holds (the
    rule sees every averaging part of this variant so far)."""
    from gpucfd import turbulence
    g, ref, limits, _ = setup()
    spec = all_variants()[name]
    N, F = PLAN["numerics"], PLAN["final"]
    log = logger(name)
    part = len(list(RUNS.glob(f"{name}_avg*.json"))) + 1
    pre = json.loads((RUNS / f"{name}_pre.json").read_text())
    if any(h != "ok" for h in pre["health"]):                   # do not average a run whose start was not healthy
        sys.exit(f"{name}: the pre-phase ended with health {pre['health']}; look at it before averaging")
    s = make(spec, g)
    try:
        t0 = gpu.load_state(s, state_path(name))
        args = run_args(s, log, f"{name} avg{part}: ")
        if length == "stop":
            before = load_variant(name)
            rule = turbulence.channel_watch(g["f"], g["delta"], tol=F["stop"]["tol"], min_time=F["stop"]["min_time"], log=log)

            def watch(t, d):                                    # the rule on all parts so far plus this one
                if before is None:
                    return rule(t, d)
                keys = ("bulk", "tau_low", "tau_high")
                joined = {k: np.concatenate((before[2][k + "_all"] if k + "_all" in before[2] else before[2][k], d[k][1:])) for k in keys}
                joined["t"] = np.concatenate((before[1], d["t"][1:]))
                return rule(t, joined)
            done = pre["wall_s"] + sum(r["wall_time"] for r in (before[0] if before else []))     # start-up and earlier parts count
            t_end = t0 + max(F["max_gpu_hours"] * 3600.0 - done, 0.0) / max(pre["wall_per_unit_s"], 1.0)
            res = gpu.run_transient(s, N["dt_cap"], t0, t_end, watch=watch, stop_on_watch=True, watch_every=2.0,
                                    partial=RUNS / f"{name}_avg{part}.partial.npz", checkpoint=state_path(name), checkpoint_every=5.0, **args)
        else:
            res = gpu.run_transient(s, N["dt_cap"], t0, t0 + float(length), partial=RUNS / f"{name}_avg{part}.partial.npz",
                                    checkpoint=state_path(name), checkpoint_every=5.0, **args)
        rec = gpu.write_run(RUNS, f"{name}_avg{part}", res, meta(s, spec, g, f"{name}_pre" if part == 1 else f"{name}_avg{part - 1}"),
                            store_every=N["store_every"], solver=s)
        got = load_variant(name)
        G = gaps(got[1], got[2], got[3], got[4], g, ref)
        log(f"{name}: AVG DONE part {part}, t = {res['t'][0]:.2f} to {res['t'][-1]:.2f}, health {rec['health']['status']}, "
            f"{rec['perf']['ms_per_step_wall']:.1f} ms per step, wall {rec['wall_time']:.0f} s; ended by: "
            f"{(rec['watch_first'] or {}).get('text', 'its length')}")
        log(f"{name}: over {G['span']:.1f} time units: bulk {G['bulk']:.4f}, wall shear {G['imbalance']:+.2%} from the driving force; "
            f"gaps {fmt_gaps(G, limits)}")
    finally:
        s.close()


# ------------------------------------------------------------------------------------------ deciding
def do_table():
    g, ref, limits, _ = setup()
    rec, d, t0 = load_baseline()
    base = gaps(d["t"], d, np.array(rec["y"]), np.array(rec["y_faces"]), g, ref, t0=t0)
    noise = json.loads((HERE / "noise.json").read_text()) if (HERE / "noise.json").exists() else None
    bn = PLAN["baseline"]["n"]
    base_cost = rec["wall_time"] / rec["end_time"]
    print(f"limits: {limits}")
    print(f"{'baseline':9s} {str(bn):15s} bulk {base['bulk']:.4f}  {fmt_gaps(base, limits)}   {base_cost:.0f} s per time unit")
    rows, spent = {}, 0.0
    for name, spec in all_variants().items():
        got = load_variant(name)
        pre = json.loads((RUNS / f"{name}_pre.json").read_text()) if (RUNS / f"{name}_pre.json").exists() else None
        spent += (pre or {}).get("wall_s", 0.0)
        if got is None:
            continue
        recs, t, series, y, yf = got
        spent += sum(r["wall_time"] for r in recs)
        G = gaps(t, series, y, yf, g, ref)
        steps = sum(r["perf"]["steps"] for r in recs)
        G["cost"] = dict(ms_per_step=1e3 * sum(r["perf"]["wall_s"] for r in recs) / steps, steps_per_unit=steps / (t[-1] - t[0]),
                         poisson_its=float(np.mean([r["perf"]["poisson_its"] for r in recs])), cells=recs[0]["cells"])
        G["cost"]["s_per_unit"] = G["cost"]["ms_per_step"] * G["cost"]["steps_per_unit"] / 1e3
        rows[name] = G
        print(f"{name:9s} {str(spec['n']):15s} bulk {G['bulk']:.4f}  {fmt_gaps(G, limits)}   {G['cost']['s_per_unit']:.0f} s per time unit "
              f"({G['cost']['ms_per_step']:.0f} ms x {G['cost']['steps_per_unit']:.0f} steps, {G['cost']['poisson_its']:.1f} its), "
              f"{G['span']:.1f} units, wall shear {G['imbalance']:+.2%}")
    print(f"GPU time spent by the stage so far: {spent / 3600:.2f} h of {PLAN['budget_gpu_hours']:g}")
    dirs = {"x": "x15", "y": "y15", "z": "z15"}
    if not all(v in rows for v in dirs.values()):
        return
    r = 1.5
    share = {q: {k: (rows[v][q]["signed"] - base[q]["signed"]) / (r ** -2 - 1.0) for k, v in dirs.items()} for q in QS}
    print("\neach direction's share of the baseline's gap (second order assumed), and what is left with all three refined without end:")
    for q in QS:
        left = base[q]["signed"] - sum(share[q].values())
        nz = f" (scatter of a short window {noise['scatter'][q]:.3f})" if noise else ""
        print(f"  {q:6s} baseline {base[q]['signed']:+.3f}: x {share[q]['x']:+.3f}  y {share[q]['y']:+.3f}  z {share[q]['z']:+.3f}  -> left {left:+.3f}{nz}")
    # The same model point by point along the profile. The single number above is the largest difference along
    # the profile; its place moves from run to run (and its sign with it), so the shares of that number do not
    # add up. The difference at each height does not have this problem.
    yb = {q: (np.array(base[q]["y"]), np.array(base[q]["diff"])) for q in QS}
    e = {q: {k: (np.interp(yb[q][0], rows[v][q]["y"], rows[v][q]["diff"]) - yb[q][1]) / (r ** -2 - 1.0) for k, v in dirs.items()} for q in QS}
    print()
    print("point by point: what is left with all three refined without end (largest along the profile; a sum of three "
          "shares carries about 3 times the scatter of a short window):")
    left_pt = {}
    for q in QS:
        prof = yb[q][1] - sum(e[q].values())
        j = int(np.argmax(np.abs(prof)))
        left_pt[q] = dict(signed=float(prof[j]), y_plus=float(yb[q][0][j]))
        big = {k: float(np.max(np.abs(e[q][k]))) for k in "xyz"}
        nz = f"  (3 x scatter {3 * noise['scatter'][q]:.3f}; limit {limits[q]})" if noise else ""
        print(f"  {q:6s} left {prof[j]:+.3f}@{yb[q][0][j]:.0f}; largest share along the profile: x {big['x']:.3f}  y {big['y']:.3f}  z {big['z']:.3f}{nz}")
    # cost model from the measured runs: time per cell-step from the nearest measured aspect, steps per unit from the directional runs
    def cost(n):
        fx, fy, fz = n[0] / bn[0], n[1] / bn[1], n[2] / bn[2]
        base_steps = rec["perf"]["steps"] / rec["end_time"]
        sx = rows["x15"]["cost"]["steps_per_unit"] / base_steps
        sy = rows["y15"]["cost"]["steps_per_unit"] / base_steps
        sz = rows["z15"]["cost"]["steps_per_unit"] / base_steps
        steps = base_steps * (1 + (sx - 1) * (fx - 1) / 0.5) * (1 + (sy - 1) * (fy - 1) / 0.5) * (1 + (sz - 1) * (fz - 1) / 0.5)
        us0 = 1e3 * rec["perf"]["ms_per_step_wall"] / rec["cells"]                                   # microseconds per cell-step, baseline
        usx = 1e3 * rows["x15"]["cost"]["ms_per_step"] / rows["x15"]["cost"]["cells"]
        usz = 1e3 * rows["z15"]["cost"]["ms_per_step"] / rows["z15"]["cost"]["cells"]
        usy = 1e3 * rows["y15"]["cost"]["ms_per_step"] / rows["y15"]["cost"]["cells"]
        aspect = fx / fz                                                                           # dz / dx relative to the baseline's square cells
        us = us0 * (usy / us0 if fy > 1 else 1.0) * (max(usx, usz) / us0 if abs(np.log(aspect)) > 0.2 else 1.0)
        return steps * us * 1e-6 * np.prod(n)                                                      # seconds per time unit
    cands = []
    for nx, ny, nz in itertools.product((192, 288, 384), (64, 96, 128), (64, 96, 128, 192)):
        n = (nx, ny, nz)
        if np.prod(n) > float(PLAN["final"]["max_cells"]):
            continue
        pred, where = {}, {}
        for q in QS:
            prof = yb[q][1] + sum(e[q][k] * ((bn[i] / n[i]) ** 2 - 1.0) for i, k in enumerate("xyz"))
            j = int(np.argmax(np.abs(prof)))
            pred[q], where[q] = float(prof[j]), float(yb[q][0][j])
        amp = float(np.sqrt(sum(((1.0 - (bn[i] / n[i]) ** 2) / (1.0 - r ** -2)) ** 2 for i in range(3))))    # of one run's scatter
        ok = all(abs(pred[q]) <= PLAN["final"]["margin"] * limits[q] for q in QS)
        cands.append(dict(n=n, pred=pred, where=where, ok=ok, s_per_unit=cost(n), noise_amplification=amp))
    cands.sort(key=lambda c: c["s_per_unit"])
    print("\ncandidate grids up to 3 million cells, cheapest first (predicted gaps; * = outside 0.8 of its limit):")
    for c in cands[:36]:
        txt = "  ".join(f"{q} {c['pred'][q]:+.3f}@{c['where'][q]:.0f}" + ("" if abs(c["pred"][q]) <= PLAN["final"]["margin"] * limits[q] else "*") for q in QS)
        print(f"  {str(c['n']):16s} {np.prod(c['n']) / 1e6:.2f} M cells, about {c['s_per_unit']:.0f} s per time unit: {txt}   {'PASSES' if c['ok'] else ''}")
    best = next((c for c in cands if c["ok"]), None)
    if best is None:
        print("no candidate passes the rule: by the stage's stop rule, no final run")
    else:
        units = PLAN["final"]["max_gpu_hours"] * 3600 / best["s_per_unit"]
        print(f"the rule picks {best['n']}: about {best['s_per_unit']:.0f} s per time unit, so {PLAN['final']['max_gpu_hours']:g} GPU hours buy {units:.0f} time units")
    (HERE / "table.json").write_text(json.dumps(dict(limits=limits, baseline=base, runs=rows, share=share, left_point_by_point=left_pt,
                                                    candidates=[dict(c, n=list(c["n"])) for c in cands], spent_h=spent / 3600), indent=1, default=float))


def assess(name=None):
    """A run on the case's own criteria (numerics.yaml, acceptance): the four requested outputs, the place of
    the fluctuation peak and the five profile gaps. name: a variant of plan.yaml, or None for the baseline.
    The bulk velocity's error knows its slow settling (gpucfd.turbulence.bulk_average).
    Returns dict(n, cells, bulk, bulk_stderr, span, imbalance, s_per_unit, wall_h, startup_h, steps_per_unit,
    ms_per_step, poisson_its, checks{key: dict(value, limit, ok, unit)}, all_ok, gaps (what gaps() returns))."""
    from gpucfd import turbulence
    g, ref, limits, numerics = setup()
    acc, R = numerics["acceptance"], ref["scalars"]
    if name is None:
        rec, d, t0 = load_baseline()
        t, y, yf, n = d["t"], np.array(rec["y"]), np.array(rec["y_faces"]), rec["n"]
        G = gaps(t, d, y, yf, g, ref, t0=t0)
        wall, startup = rec["wall_time"], rec["wall_time"] * (t0 - 0.0) / rec["end_time"]
        steps_per_unit, ms, its = rec["perf"]["steps"] / rec["end_time"], rec["perf"]["ms_per_step_wall"], rec["perf"]["poisson_its"]
    else:
        recs, t, d, y, yf = load_variant(name)
        pre = json.loads((RUNS / f"{name}_pre.json").read_text())
        t0, n = None, recs[0]["n"]
        G = gaps(t, d, y, yf, g, ref)
        avg_wall = sum(r["wall_time"] for r in recs)
        wall, startup = pre["wall_s"] + avg_wall, pre["wall_s"]
        steps = sum(r["perf"]["steps"] for r in recs)
        steps_per_unit, ms = steps / (t[-1] - t[0]), 1e3 * sum(r["perf"]["wall_s"] for r in recs) / steps
        its = float(np.mean([r["perf"]["poisson_its"] for r in recs]))
    B = turbulence.bulk_average(t, d["bulk"], 0.5 * (d["tau_low"] + d["tau_high"]), g["f"], g["delta"], t_start=t0)
    ub, ue = float(B["mean"]), float(B["stderr"])
    cf = 2.0 * g["u_tau"] ** 2 / ub ** 2
    checks = dict(
        bulk=(100 * (ub / R["bulk"]["value"] - 1), acc["bulk_pct"], "%"),
        centre=(100 * (G["centre"] / R["centre"]["value"] - 1), acc["centre_pct"], "%"),
        cf=(100 * (cf / R["cf"]["value"] - 1), acc["cf_pct"], "%"),
        urms_peak=(100 * (G["urms_peak"] / R["urms_peak"]["value"] - 1), acc["urms_peak_pct"], "%"),
        urms_peak_y=(G["urms_peak_y"] - R["urms_peak_y"]["value"], acc["urms_peak_yplus"], " wall units"))
    for q in QS:
        checks[q] = (G[q]["signed"], limits[q], f"@{G[q]['y_plus']:.0f}")
    checks = {k: dict(value=float(v[0]), limit=float(v[1]), ok=bool(abs(v[0]) <= v[1]), unit=v[2]) for k, v in checks.items()}
    return dict(n=list(n), cells=int(np.prod(n)), bulk=ub, bulk_stderr=ue, bulk_reliable=bool(B.get("reliable", False)), span=G["span"],
                imbalance=G["imbalance"], s_per_unit=float(ms * steps_per_unit / 1e3), wall_h=float(wall) / 3600, startup_h=float(startup) / 3600,
                steps_per_unit=float(steps_per_unit), ms_per_step=float(ms), poisson_its=float(its), cf=cf, centre=G["centre"],
                urms_peak=G["urms_peak"], urms_peak_y=G["urms_peak_y"], checks=checks, all_ok=all(c["ok"] for c in checks.values()), gaps=G)


def do_final(name):
    """The final grid against the baseline on the case's criteria (assess); writes final.json."""
    table = json.loads((HERE / "table.json").read_text()) if (HERE / "table.json").exists() else {}
    out = {}
    for label, key in (("baseline", None), (name, name)):
        A = assess(key)
        out[label] = {k: v for k, v in A.items() if k != "gaps"}
        print(f"{label}: {A['n']} ({A['cells'] / 1e6:.2f} M cells), {A['span']:.1f} time units averaged, wall shear {A['imbalance']:+.2%} from the "
              f"driving force, bulk {A['bulk']:.4f} +- {A['bulk_stderr']:.4f}, {A['s_per_unit']:.0f} s per time unit, {A['wall_h']:.2f} GPU hours")
        print("   " + "  ".join(f"{k} {c['value']:+.3f}{c['unit']} (limit {c['limit']:g})" + ("" if c["ok"] else "*") for k, c in A["checks"].items()))
        bad = [k for k, c in A["checks"].items() if not c["ok"]]
        print(f"   {'every criterion met' if not bad else 'outside its limit: ' + ', '.join(bad)}")
    pred = next((c for c in table.get("candidates", []) if list(c["n"]) == out[name]["n"]), None)
    if pred:
        out["predicted"] = dict(pred=pred["pred"], where=pred["where"])
        print("predicted for this grid from the directional runs: " + "  ".join(f"{q} {pred['pred'][q]:+.3f}@{pred['where'][q]:.0f}" for q in QS))
        print("measured minus predicted:                           " + "  ".join(
            f"{q} {out[name]['checks'][q]['value'] - pred['pred'][q]:+.3f}" for q in QS))
    (HERE / "final.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "noise":
        do_noise()
    elif cmd == "pre":
        do_pre(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "avg":
        arg = sys.argv[3] if len(sys.argv) > 3 else str(PLAN["phases"]["average"])
        do_avg(sys.argv[2], arg if arg == "stop" else float(arg))
    elif cmd == "table":
        do_table()
    elif cmd == "final":
        do_final(sys.argv[2])
    else:
        sys.exit(__doc__)
