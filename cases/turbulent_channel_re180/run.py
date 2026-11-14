"""Turbulent channel at friction Reynolds number 180 on gpucfd, without a turbulence model.

Usage (repository root, inside WSL):
  python -m cases.turbulent_channel_re180.run preflight          # short runs on G1 (from the start field) and G3
                                                                 # (from G1's field): health, cost, projected run times
  python -m cases.turbulent_channel_re180.run tolcheck G1 G3     # which pressure tolerance a level may use
  python -m cases.turbulent_channel_re180.run production G1 G2 G3 G1h   # the runs of numerics.yaml's table, in order

Every run is a row of numerics.yaml (`runs`). A run ends on the stop rule of numerics.yaml (`stop`), applied
live to the record; its t_max is only the longest it may go.
Outputs: results/turbulent_channel_re180/runs/<run>.json + .npz (cfdlib.gpu.write_run), preflight.json,
tolcheck_<level>.json, logs/<run>.log; velocity and pressure fields in ~/gpucfd_runs/turbulent_channel_re180/
(outside git, like OpenFOAM's run directories).

npz keys (gpucfd.turbulence.PlaneStats.sample, one row per sample): t; U, uu, ww, W at the cell centres y; vv, uv,
dUdy at the cell faces; bulk, tau_low, tau_high, umax, courant_rate.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cfdlib import gpu, signals

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS_DIR = OUT / "runs"
LOGS = OUT / "logs"
FIELDS = Path.home() / "gpucfd_runs" / HERE.name


def load():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    return problem, numerics


def given(problem):
    g = {k: v["value"] for k, v in problem["given"].items()}
    g["u_tau"] = float(np.sqrt(g["f"] * g["delta"]))          # the friction velocity the driving force fixes
    return g


def y_faces(level, problem, numerics):
    """The wall-normal faces of a level: tanh stretching with the level's wall cell (in wall units)."""
    from gpucfd import grids
    g = given(problem)
    L = numerics["levels"][level]
    return grids.tanh_wall(L["n"][1], 2 * g["delta"], L["wall_cell_plus"] * g["nu"] / g["u_tau"])


def make(level, problem, numerics):
    """A simulation on a level of the grid family: periodic in x and z, walls in y, driven by the force f."""
    from gpucfd import gcfd
    g = given(problem)
    n = numerics["levels"][level]["n"]
    return gcfd.NavierStokes(n, (g["Lx"], 2 * g["delta"], g["Lz"]), g["nu"], walls=(False, True, False),
                             force=(g["f"], 0.0, 0.0), scheme=numerics["scheme"],
                             grid=(None, y_faces(level, problem, numerics), None))


def observer(s):
    """observe(s) for run_transient: the plane averages of PlaneStats, plus the plane mean of w (W), which
    the fluctuation of w about the full mean needs."""
    from gpucfd import turbulence
    st = turbulence.PlaneStats(s)

    def observe(s_):
        q = s_.get()
        o = st.sample(q)
        o["W"] = q[2].mean(axis=(0, 2))
        return o
    return observe


def field_path(run):
    return FIELDS / f"{run}.npz"


def save_field(s, run, t):
    FIELDS.mkdir(parents=True, exist_ok=True)
    u, v, w = s.get()
    tmp = FIELDS / f"{run}.tmp.npz"
    np.savez(tmp, u=u, v=v, w=w, p=s.pressure(), t=t)
    tmp.replace(field_path(run))


def load_field(s, run):
    """Put a stored field of the same grid into s (velocity and the pressure the scheme carries)."""
    d = np.load(field_path(run))
    s.set([d["u"], d["v"], d["w"]])
    s.set_pressure(d["p"])
    return float(d["t"])


def start(s, run, level, source, source_level, problem, numerics, log):
    """Set the start field of a run. Returns its description."""
    from gpucfd import turbulence, steady
    g = given(problem)
    if source is None:
        div = turbulence.channel_start(s, g["u_tau"])
        log(f"{run}: start field channel_start, max |div u| {div:.1e}")
        return "channel_start"
    if source_level == level:
        load_field(s, source)
        log(f"{run}: continues the last field of {source} (same grid)")
        return source
    src = make(source_level, problem, numerics)
    try:
        d = np.load(field_path(source))
        src.set([d["u"], d["v"], d["w"]])
        div = steady.prolong(src, s)
    finally:
        src.close()
    log(f"{run}: start field interpolated from the last field of {source}, max |div u| {div:.1e}")
    return source


def gpu_memory_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - a missing tool must not end a run
        return None


def settling_time(ub, g, R):
    """T_s = delta U_b / (n u_tau^2): the time in which the bulk velocity forgets a disturbance (theory.py)."""
    return g["delta"] * ub / (R["friction_exponent"] * g["u_tau"] ** 2)


def bulk_average(t, d, t_start, g, R):
    """Mean bulk velocity after t_start with a standard error that knows its slow mode.

    The bulk velocity obeys d U_b / dt = f - tau_w / delta exactly, and the wall shear follows it:
    d U_b / dt = - (U_b - U_eq) / T_s + xi, with T_s the settling time and xi what the eddies add. So the bulk
    velocity remembers for about T_s (9 time units), much longer than blocks of a run of tens of time units,
    and batch means of U_b itself give too small an error (two runs of G1 differed by five of them). xi has a
    short memory, and averaging the equation over the window gives  mean(U_b) - U_eq = T_s (mean(xi) - change of
    U_b / window). So: standard error of the mean = T_s x (standard error of the mean of xi, by batch means),
    and never less than the batch-means error of U_b itself.
    Returns the dict of signals.stationary for U_b with stderr replaced, plus stderr_batch (the plain one),
    xi_independent (independent samples of xi), reliable (at least 10 of them), T_s."""
    tau = 0.5 * (d["tau_low"] + d["tau_high"])
    ub = signals.stationary(t, d["bulk"], t_start=t_start)
    Ts = settling_time(ub["mean"], g, R)
    xi = g["f"] - tau / g["delta"] + (d["bulk"] - ub["mean"]) / Ts
    sx = signals.stationary(t, xi, t_start=t_start)
    out = dict(ub, stderr_batch=ub["stderr"], stderr=max(Ts * sx["stderr"], ub["stderr"]), xi_independent=sx["n_independent"],
               reliable=bool(sx["reliable"]), T_s=Ts)
    return out


def start_up_end(t, d, g, R):
    """Where the averaging may start: the later of settle_time's estimate (the fast start-up) and the time at which
    the bulk velocity's own slow approach has shrunk below R["gap_left"] of its value. The approach is
    exponential with the settling time T_s, from the largest distance seen in the first R["gap_window"] time
    units: t = T_s ln(gap / (gap_left U_b)). Found by a few passes, since the mean depends on the cut."""
    cut = signals.settle_time(t, d[R["key"]])
    t0, early = cut["t_start"], t <= t[0] + R["gap_window"]
    for _ in range(4):
        mean = signals.stationary(t, d["bulk"], t_start=t0)["mean"]
        gap = float(np.max(np.abs(d["bulk"][early] - mean)))
        t_gap = t[0] + settling_time(mean, g, R) * np.log(gap / (R["gap_left"] * abs(mean))) if gap > R["gap_left"] * abs(mean) else t[0]
        t0 = max(cut["t_start"], float(t_gap))
    return dict(t_start=t0, fast=cut["t_start"], slow=float(t_gap), gap=gap, settled=bool(cut["settled"]))


def stop_state(t, d, R, g):
    """The stop rule of numerics.yaml on a record (d: the series, d["t"] the times; g: the given values with
    u_tau). Returns dict(holds, text, t_start, mean, stderr, stderr_batch, drift, n_independent, imbalance, ...)."""
    tt = d["t"]
    try:
        cut = start_up_end(tt, d, g, R)
        t0 = cut["t_start"]
        st = bulk_average(tt, d, t0, g, R)
        tw = signals.stationary(tt, 0.5 * (d["tau_low"] + d["tau_high"]), t_start=t0)
        mid = 0.5 * (st["t0"] + st["t1"])
        first = {k: v[tt <= mid] for k, v in d.items() if k in ("bulk", "tau_low", "tau_high")}
        h1 = bulk_average(tt[tt <= mid], first, t0, g, R)
        h2 = bulk_average(tt, d, mid, g, R)
    except ValueError as e:
        return dict(holds=False, text=f"record too short ({e})")
    drift = (h2["mean"] - h1["mean"]) / np.hypot(h1["stderr"], h2["stderr"])
    imb = tw["mean"] / (g["f"] * g["delta"]) - 1.0
    span = st["t1"] - st["t0"]
    tests = {"error": st["stderr"] <= R["stderr_rel"] * abs(st["mean"]), "reliable": st["reliable"],
             "drift": abs(drift) <= R["max_drift"], "length": span >= R["min_time"],
             "balance": abs(imb) <= R["balance"], "start-up found": cut["settled"]}
    holds = all(tests.values())
    text = (f"{R['key']} {st['mean']:.4f} +- {st['stderr']:.4f} ({st['stderr'] / abs(st['mean']):.2%}; batch means alone "
            f"{st['stderr_batch']:.4f}) over t = {st['t0']:.1f} to {st['t1']:.1f}, {st['xi_independent']:.0f} independent samples, "
            f"drift {drift:+.1f} sigma, wall shear {imb:+.2%} from the driving force")
    if not holds:
        text += "; not yet: " + ", ".join(k for k, ok in tests.items() if not ok)
    return dict(holds=bool(holds), text=text, t_start=st["t0"], mean=st["mean"], stderr=st["stderr"], stderr_batch=st["stderr_batch"],
                drift=float(drift), n_independent=st["xi_independent"], imbalance=imb, T_s=st["T_s"], cut=cut, tests=tests)


def make_watch(s, run, R, g, log, checkpoint_every=10.0):
    """The live check of a run: the stop rule, a line in the log when it does not hold yet, and the field
    saved every checkpoint_every time units (gpucfd has no restart files)."""
    state = dict(next=None)

    def watch(t, d):
        if state["next"] is None:
            state["next"] = d["t"][0] + checkpoint_every
        if t >= state["next"]:
            save_field(s, run, t)
            state["next"] += checkpoint_every
        r = stop_state(t, d, R, g)
        if r["holds"]:
            return "stop rule holds: " + r["text"]
        log(f"{run}: check at t {t:.1f}: {r['text']}")
        return None
    return watch


def logger(name):
    LOGS.mkdir(parents=True, exist_ok=True)
    fh = open(LOGS / f"{name}.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    return log


def thinned(res, keep):
    """The record to store: every keep-th sample of each series (and the last one), so that a run sampled often for
    its step control does not store profiles closer in time than the averages need. The scalar series are also
    kept in full, under <name>_all with their times t_all (the Courant history of the whole run stays checkable)."""
    if keep <= 1:
        return res
    n = len(res["t"])
    idx = np.unique(np.append(np.arange(0, n, keep), n - 1))
    series = {k: v[idx] for k, v in res["series"].items()}
    series["t_all"] = res["t"]
    series.update({f"{k}_all": v for k, v in res["series"].items() if np.ndim(v) == 1})
    return dict(res, t=res["t"][idx], series=series)


def run_one(run, problem, numerics, t_end=None, stop=True, name=None, log=None):
    """One run of the table. t_end: the longest it may go (default: the stop rule's t_max of its level)."""
    g = given(problem)
    T, R = numerics["time"], numerics["stop"]
    spec = numerics["runs"][run]
    level = spec["level"]
    source = spec.get("source")
    name = name or run
    log = log or logger(name)
    cfl = spec.get("cfl", T["cfl"])
    s = make(level, problem, numerics)
    try:
        src = start(s, name, level, source, numerics["runs"][source]["level"] if source else None, problem, numerics, log)
        t0 = 0.0
        t_end = R["t_max"][level] if t_end is None else t_end
        m = spec.get("steps_per_sample", T["steps_per_sample"][level])
        keep = spec.get("store_every", T["store_every"][level]) if stop else 1
        ts = gpu.time_step(s, u_scale=None, cfl=cfl)          # the start field's own limit, for the record
        watch = make_watch(s, name, R, g, log) if stop else None
        res = gpu.run_transient(s, T["dt_cap"], t0, t_end, m * T["dt_cap"], observer(s), u_ref=15.0 * g["u_tau"], log=log,
                                tag=f"{name}: ", log_every=1.0, ptol=T["pressure_tol"], watch=watch, stop_on_watch=stop,
                                watch_every=R["check_every"], partial=RUNS_DIR / f"{name}.partial.npz", adapt_cfl=cfl)
        save_field(s, name, float(res["t"][-1]))             # the field at the record's end
        y = s._axis(1, False)
        meta = dict(level=level, cells=int(np.prod(s.n)), n=list(s.n), h=s._hmin(), source=src, pressure_tol=T["pressure_tol"],
                    scheme=s.scheme, time_step=dict(ts, adaptive_cfl=cfl, dt_cap=T["dt_cap"], steps_per_sample=m, store_every=keep),
                    y=list(map(float, y)), y_faces=list(map(float, np.concatenate((s._axis(1, True), [s.length[1]])))),
                    dx=s.length[0] / s.n[0], dz=s.length[2] / s.n[2], gpu_memory_mib=gpu_memory_mib(),
                    stop_rule=dict(R, t_max=t_end) if stop else None)
        rec = gpu.write_run(RUNS_DIR, name, thinned(res, keep), meta)
        log(f"{name}: done at t = {rec['end_time']:.2f}, health {rec['health']['status']}, {rec['perf']['steps']} steps, "
            f"wall {rec['wall_time']:.0f} s, {rec['perf']['ms_per_step_wall']:.1f} ms per step (wall), "
            f"{rec['perf']['poisson_its']:.1f} pressure iterations per solve; "
            f"ended by: {(rec['watch_first'] or {}).get('text', 't_max reached')}")
        return rec, res
    finally:
        s.close()


def do_preflight(problem, numerics):
    """Short runs on the coarsest and the finest level: health, cost per step, steps per time unit, projection."""
    log = logger("preflight")
    R = numerics["stop"]
    out = dict(runs=[])
    table = dict(numerics["runs"])
    table["pre_G1"] = dict(level="G1", source=None)
    table["pre_G3"] = dict(level="G3", source="pre_G1")
    num = dict(numerics, runs=table)
    for name, length in (("pre_G1", 4.0), ("pre_G3", 1.0)):
        rec, res = run_one(name, problem, num, t_end=length, stop=False, log=log)
        t, rate = res["t"], res["series"]["courant_rate"]
        late = t >= 0.5 * t[-1]                               # the second half: past the start field's own noise
        steps_per_unit = float(np.mean(rate[late])) / rec["time_step"]["adaptive_cfl"]
        level = rec["level"]
        wall_per_unit = steps_per_unit * rec["perf"]["ms_per_step_wall"] / 1000.0
        out["runs"].append(dict(name=name, level=level, cells=rec["cells"], length=length, health=rec["health"],
                                ms_per_step_wall=rec["perf"]["ms_per_step_wall"], ms_per_step_gpu=rec["perf"]["ms_per_step_gpu"],
                                poisson_its=rec["perf"]["poisson_its"], observe_share=rec["perf"]["observe_s"] / rec["perf"]["wall_s"],
                                steps=rec["perf"]["steps"], wall_s=rec["wall_time"], dt_mean=rec["dt"], dt_min=rec["dt_min"],
                                dt_max=rec["dt_max"], courant_rate_late=float(np.mean(rate[late])),
                                steps_per_unit=steps_per_unit, wall_per_unit_s=wall_per_unit,
                                projected_wall_s=wall_per_unit * R["t_max"][level], t_max=R["t_max"][level],
                                bulk_end=float(res["series"]["bulk"][-1]), tau_end=float(0.5 * (res["series"]["tau_low"][-1] + res["series"]["tau_high"][-1])),
                                uu_max_end=float(res["series"]["uu"][-1].max()), gpu_memory_mib=rec["gpu_memory_mib"]))
        r = out["runs"][-1]
        log(f"{name}: {r['ms_per_step_wall']:.1f} ms per step, {steps_per_unit:.0f} steps per time unit, "
            f"{wall_per_unit:.1f} s per time unit, at most {r['projected_wall_s'] / 60:.1f} min for t_max = {r['t_max']:g}")
    (OUT / "preflight.json").write_text(json.dumps(out, indent=1, default=float))
    log("PREFLIGHT DONE")


def do_tolcheck(level, problem, numerics, length=0.3, tols=(1e-10, 1e-9, 1e-8, 1e-7, 1e-6)):
    """Same-answer rule for the pressure tolerance on a level, from the pre-flight field of that level (G1, G3)
    or of the nearest run. Fixed time step (an adaptive one would differ between the runs)."""
    log = logger("tolcheck")
    T = numerics["time"]
    source = {"G1": "pre_G1", "G3": "pre_G3"}.get(level, level)

    def mk():
        s = make(level, problem, numerics)
        load_field(s, source)
        return s
    s = mk()
    dt = 0.8 * T["cfl"] / s.courant_rate()                    # a fixed step with a margin for the flow's changes
    s.close()
    m = T["steps_per_sample"][level]

    def observe(s_):                                          # each tolerance gets a new simulation: one observer per simulation
        if not hasattr(s_, "_observe"):
            s_._observe = observer(s_)
        return s_._observe(s_)
    res = gpu.tolerance_check(mk, dt, 0.0, length, m * dt, observe, 15.0, tols=tols,
                              keys=["bulk", "tau_low", "tau_high"], rule=T["pressure_tol_rule"], log=log)
    res["level"], res["dt"], res["length"], res["source"] = level, dt, length, source
    (OUT / f"tolcheck_{level}.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"tolcheck {level}: loosest tolerance that gives the same answer: {res['chosen']:g}")


if __name__ == "__main__":
    problem, numerics = load()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "preflight":
        do_preflight(problem, numerics)
    elif cmd == "tolcheck":
        for lv in sys.argv[2:]:
            do_tolcheck(lv, problem, numerics)
        print("TOLCHECK DONE", flush=True)
    elif cmd == "production":
        for r in sys.argv[2:]:
            if (RUNS_DIR / f"{r}.json").exists():
                print(f"{r}: record exists, not run again (delete results/.../runs/{r}.json to repeat it)", flush=True)
                continue
            run_one(r, problem, numerics)
        print("PRODUCTION DONE " + " ".join(sys.argv[2:]), flush=True)
    else:
        sys.exit(__doc__)
