"""Tests of the gpucfd run tools in cfdlib/gpu.py (needs the GPU; inside WSL from the project root).

    PYTHONPATH=. python tests/test_gpu_tools.py <empty output folder>

A small lid-driven cavity (64 x 64, Re 100) is enough: the tests are about the run loop, not the flow.
1. run_transient gives the same samples with and without the new options (watch_every, partial).
2. watch_every: the live check is called at its own interval and can stop the run between log lines.
3. partial: the partial record holds exactly what the run had sampled at its last log line, and
   write_run removes it once the full record is saved.
4. Ctrl-C during a run returns the record so far, with a warning in the health checks.
5. tolerance_check: the strictest tolerance passes by definition, a very loose one fails, and the
   chosen tolerance is the loosest that passes.
6. position_spread, check_fit and pick_scheme (no GPU needed).
7. The record: time_step gives the step and the limit that set it, the perf block names each cost by its
   clock, cfdlib.runrecord.read returns the record in the common shape, add_timing gives a spread, and a
   stored tolerance check reads back in one layout (the older layout too).
8. Turbulence resolved in a channel (gpucfd.turbulence): the plane averages give the solver's own balance
   of mean momentum to round-off, the bulk velocity changes by force minus wall shear, and run_transient
   with adapt_cfl holds the Courant number of each cell where a fixed step overshoots.
"""
import sys
from pathlib import Path

import numpy as np

from cfdlib import gpu
from gpucfd import gcfd

OUT = Path(sys.argv[1])
FAILS = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def make():
    s = gcfd.NavierStokes((64, 64, 1), (1.0, 1.0, 1.0), 0.01, walls=(True, True, False), wall_velocity={"y+": (1.0, 0, 0)})
    s.set([np.zeros(s.shape), np.zeros(s.shape)])
    s.project()
    return s


def observe(s):
    u, v = s.get()[:2]
    return dict(ke=float(np.mean(u ** 2 + v ** 2)), umax=float(np.sqrt(np.max(u ** 2) + np.max(v ** 2))), umid=float(u[0, 32, 32]))


s0 = make()
DT = s0.max_dt(1.0)
s0.close()
s0b = make()   # kept for the time-step record of test 7 (closed there)
ARGS = dict(dt=DT, t0=0.0, sample_dt=0.02, observe=observe, u_ref=1.0, log=lambda m: None, ptol=1e-10)

# 1. the options do not change the samples
s = make()
base = gpu.run_transient(s, t_end=1.0, log_every=0.25, **ARGS)
s.close()
s = make()
part = OUT / "runs" / "cav.partial.npz"
opt = gpu.run_transient(s, t_end=1.0, log_every=0.25, watch_every=0.1, watch=lambda t, d: None, partial=part, **ARGS)
s.close()
check("same samples with watch_every and partial as without",
      len(base["t"]) == len(opt["t"]) and all(np.array_equal(base["series"][k], opt["series"][k]) for k in base["series"]),
      f"{len(base['t'])} samples")

# 3. the partial record: what had been sampled at the last log line before the end
p = np.load(part)
n = len(p["t"])
check("partial record holds the samples up to the last log line before the end",
      0.74 < p["t"][-1] < 0.76 + ARGS["sample_dt"] and np.array_equal(p["ke"], base["series"]["ke"][:n]), f"ends at t = {p['t'][-1]:.3f}")
step = gpu.time_step(s0b, 1.0)
rec = gpu.write_run(OUT / "runs", "cav", opt, dict(level="test", cells=64 * 64, h=1.0 / 64, source=None, pressure_tol=1e-10,
                                                  scheme="rk3", time_step=step))
check("write_run saves the record and removes the partial one",
      (OUT / "runs" / "cav.npz").exists() and (OUT / "runs" / "cav.json").exists() and not part.exists())

# 2. the check runs at its own interval and stops the run between log lines
calls = []


def watch(t, d):
    calls.append(t)
    return "settled" if t >= 0.3 - 1e-9 else None


s = make()
r = gpu.run_transient(s, t_end=1.0, log_every=0.5, watch_every=0.1, watch=watch, stop_on_watch=True, **ARGS)
s.close()
check("watch called every watch_every and stops the run at its first answer",
      len(calls) == 3 and abs(r["t"][-1] - calls[-1]) < 1e-12 and 0.3 - 1e-9 <= r["t"][-1] < 0.3 + ARGS["sample_dt"]
      and r["watch_first"]["text"] == "settled", f"calls at {', '.join(f'{c:.3f}' for c in calls)}; run ended at {r['t'][-1]:.3f}")

# 4. Ctrl-C: the record so far comes back
class Interrupting:
    """A simulation whose run() raises KeyboardInterrupt on its 10th call, as Ctrl-C would."""
    def __init__(self, sim):
        self.sim, self.calls = sim, 0

    def __getattr__(self, name):
        return getattr(self.sim, name)

    def run(self, *a):
        self.calls += 1
        if self.calls == 10:
            raise KeyboardInterrupt
        return self.sim.run(*a)


s = make()
r = gpu.run_transient(Interrupting(s), t_end=1.0, log_every=0.25, **ARGS)
s.close()
names = {c["name"]: c["status"] for c in r["health"]["checks"]}
check("Ctrl-C returns the record so far with a warning",
      len(r["t"]) == 10 and names.get("Run not interrupted") == "warn" and r["health"]["status"] == "warn"
      and np.array_equal(r["series"]["ke"], base["series"]["ke"][:10]), f"{len(r['t'])} samples kept")

# 5. tolerance_check


def developed():
    sim = make()
    sim.run(DT, int(round(0.5 / DT)), 1e-10)
    return sim


tc = gpu.tolerance_check(developed, DT, 0.5, 0.3, 0.02, observe, 1.0, tols=(1e-10, 1e-8, 1e-2), keys=["ke", "umid"], rule=1e-9,
                         log=lambda m: print("      " + m, flush=True))
by = {r["tol"]: r for r in tc["runs"]}
check("tolerance_check: strictest passes with zero difference, a very loose one fails, the loosest passing is chosen",
      by[1e-10]["worst"] == 0.0 and by[1e-10]["passes"] and not by[1e-2]["passes"]
      and tc["chosen"] == max(t for t, r in by.items() if r["passes"]) and tc["chosen"] < 1e-2,
      f"chosen {tc['chosen']:g}; differences " + ", ".join(f"{t:g}: {r['worst']:.1e}" for t, r in by.items()))

# 6. no GPU needed
sp = gpu.position_spread({"0": dict(cd=3.1854, cl=0.9775), "1/2": dict(cd=3.1832, cl=0.9745)})
check("position_spread: middle and half range of two positions",
      abs(sp["cd"]["middle"] - 3.1843) < 1e-12 and abs(sp["cd"]["half_range"] - 0.0011) < 1e-12 and set(sp) == {"cd", "cl"})
fit = gpu.check_fit(needs_stretched_grid=True)
check("check_fit: a thin layer is a note, not a blocker, and the note warns about cells against time",
      fit["ok"] and any("accuracy per cell, not time" in n for n in fit["notes"]))

# case 4's clustered levels: cylinder diameter 0.1, nu 1e-3, velocity scale 2.25; time steps as recorded in the runs
ps = {dh: gpu.pick_scheme(0.1 / dh, 1e-3, 2.25) for dh in (35, 70, 140)}
check("pick_scheme: case 4's levels get the scheme that was measured faster, with the recorded time steps",
      [ps[d]["scheme"] for d in (35, 70, 140)] == ["rk3", "rk3", "imex"]
      and abs(ps[140]["dt_rk3"] / 7.653e-5 - 1) < 1e-3 and abs(ps[140]["dt_imex"] / 2.540e-4 - 1) < 1e-3
      and abs(ps[35]["fewer_steps"] - 1) < 1e-12 and abs(ps[140]["fewer_steps"] - 3.318) < 2e-3,
      "; ".join(f"D/h {d}: {p['scheme']}, {p['fewer_steps']:.2f} times fewer steps, gain {p['expected_gain']:.2f}"
                for d, p in ps.items()))

# 7. what the record holds, and the one reader for it
import json  # noqa: E402

from cfdlib import runrecord  # noqa: E402

lim = s0b.dt_limits(1.0)
check("time_step: the step run_transient was given, both limits, and the limit that set it",
      step["dt"] == DT and step["convective"] == lim["convective"] and step["viscous"] == lim["viscous"]
      and step["set_by"] == ("viscous" if lim["viscous"] < lim["convective"] else "convective") and step["scheme"] == "rk3"
      and gpu.time_step(s0b, 1.0, factor=0.5)["dt"] == 0.5 * DT,
      f"dt {step['dt']:.3g}, convective {step['convective']:.3g}, viscous {step['viscous']:.3g}, set by the {step['set_by']} limit")
s0b.run(DT, 200, 1e-10)   # a moving field: the step from the field itself is finite and not below the fixed-scale one
field = gpu.time_step(s0b)
check("time_step without a scale takes the step from the current field",
      field["set_by"] == "current field" and DT <= field["dt"] < float("inf"), f"dt {field['dt']:.3g} against {DT:.3g}")
s0b.close()
p = rec["perf"]
check("the perf block names each cost by its clock, and keeps the older names for the GPU-time values",
      abs(p["us_per_cell_step_wall"] - 1e6 * p["wall_s"] / (p["steps"] * p["cells"])) < 1e-12
      and abs(p["us_per_cell_step_gpu"] - 1e6 * p["gpu_s"] / (p["steps"] * p["cells"])) < 1e-12
      and p["ms_per_step"] == p["ms_per_step_gpu"] and p["us_per_cell_step"] == p["us_per_cell_step_gpu"]
      and p["us_per_cell_step_wall"] > p["us_per_cell_step_gpu"],
      f"per cell and step: wall {p['us_per_cell_step_wall']:.3f}, GPU {p['us_per_cell_step_gpu']:.3f} microseconds")
r = runrecord.read(OUT / "runs" / "cav.json")
check("runrecord.read: the record in the common shape, costs on wall time, the time step's limits",
      r["tool"] == "gpucfd" and r["cells"] == 4096 and r["steps"] == p["steps"] and r["wall_s"] == p["wall_s"]
      and abs(r["us_per_cell_step_wall"] - p["us_per_cell_step_wall"]) < 1e-9 and abs(r["flow_time"] - 1.0) < 1e-9
      and r["time_step"]["set_by"] == step["set_by"] and r["wall_spread"] is None and r["health"] == rec["health"]["status"])
old = dict(name="M", level="M", cells=1000, dt=0.01, start_time=0.0, end_time=1.0, cores=4, wall_time=20.0, h=dict(wall=0.001, box=0.01),
           health=dict(status="ok"), perf=dict(steps=100, wall_s=19.0, us_per_cell_step=190.0, lin_iters=dict(p=5.5)))
ro = runrecord.read(old)
check("runrecord.read: an OpenFOAM record gives the same keys, with nothing on a GPU clock",
      set(ro) == set(r) and ro["tool"] == "openfoam" and ro["wall_s"] == 20.0 and abs(ro["us_per_cell_step_wall"] - 200.0) < 1e-9
      and ro["us_per_cell_step_gpu"] is None and ro["h_min"] == 0.001 and ro["pressure_its"] == 5.5 and ro["cores"] == 4)
r2 = runrecord.add_timing(OUT / "runs" / "cav.json", 1.05 * p["wall_s"], note="test repeat")
check("add_timing: a repeat is kept beside the run and gives the spread of its timings",
      len(r2["walls"]) == 2 and abs(r2["wall_spread"] - 0.05 / 1.025) < 1e-9 and r2["wall_s"] == p["wall_s"],
      f"spread {100 * r2['wall_spread']:.1f} %")
(OUT / "tol_new.json").write_text(json.dumps(tc, default=float))
(OUT / "tol_old.json").write_text(json.dumps([dict(tol=1e-10, d_cd=0.0, d_cl=0.0, health="ok", div="1e-12",
                                                    perf=dict(ms_per_step=3.0, poisson_its=9.0, wall_s=2.0, steps=100)),
                                               dict(tol=1e-8, d_cd=2e-5, d_cl=-4e-5, health="ok", div="1e-10",
                                                    perf=dict(ms_per_step=2.0, poisson_its=6.0, wall_s=1.5, steps=100)),
                                               dict(tol=1e-6, d_cd=3e-3, d_cl=1e-3, health="ok", div="1e-8",
                                                    perf=dict(ms_per_step=1.5, poisson_its=4.0, wall_s=1.2, steps=100))]))
tn, to = gpu.read_tolerance_check(OUT / "tol_new.json"), gpu.read_tolerance_check(OUT / "tol_old.json", rule=1e-4)
check("read_tolerance_check: a stored check reads back, and the older layout comes out in the same one",
      tn["chosen"] == tc["chosen"] and tn["runs"][0]["wall_s"] > 0 and to["chosen"] == 1e-8
      and [x["passes"] for x in to["runs"]] == [True, True, False] and abs(to["runs"][1]["worst"] - 4e-5) < 1e-18
      and set(to["runs"][0]) == set(tn["runs"][0]))


# 8. turbulence resolved in a channel: a small box at friction Reynolds number 180 (far too coarse for the
#    physics; the tests are about the arithmetic)
from cfdlib import signals  # noqa: E402
from gpucfd import grids, turbulence  # noqa: E402


def channel(scheme):
    y = grids.tanh_wall(32, 2.0, 1.0 / 180.0)
    return gcfd.NavierStokes((32, 32, 32), (4 * np.pi, 2.0, 4 * np.pi / 3), 1.0 / 180.0, walls=(False, True, False),
                             force=(1.0, 0.0, 0.0), scheme=scheme, grid=(None, y, None))


for scheme in ("rk3", "imex"):
    sc = channel(scheme)
    turbulence.channel_start(sc, 1.0)
    dtc = 0.5 * sc.dt_cfl(0.8)
    sc.run(dtc, 40, 1e-9)
    ps = turbulence.PlaneStats(sc)
    o = ps.sample()
    solver = sc.rhs(0).mean(axis=(0, 2))                               # the solver's own right-hand side, plane averaged
    mine = 1.0 - np.diff(o["uv"]) / ps.dy + sc.nu * np.diff(o["dUdy"]) / ps.dy
    check(f"PlaneStats ({scheme}): turbulent and viscous stress give the solver's mean-momentum balance to round-off",
          np.abs(solver - mine).max() < 1e-9 * np.abs(solver).max() and np.abs(o["uv"]).max() > 0.01,
          f"largest difference {np.abs(solver - mine).max():.1e}, terms up to {np.abs(solver).max():.2g}")
    tiny, nst = 0.02 * dtc, 4
    sc.run(tiny, nst, 1e-11)
    o2 = ps.sample()
    rate = (o2["bulk"] - o["bulk"]) / (nst * tiny)
    pred = 1.0 - 0.25 * (o["tau_low"] + o2["tau_low"] + o["tau_high"] + o2["tau_high"])
    check(f"PlaneStats ({scheme}): the bulk velocity changes by force minus mean wall shear", abs(rate - pred) < 1e-4 * max(abs(pred), 1e-3),
          f"{rate:.6f} against {pred:.6f}")
    sc.close()
sc = channel("imex")
turbulence.channel_start(sc, 1.0)
sc.run(0.5 * sc.dt_cfl(0.8), 200, 1e-8)
ps = turbulence.PlaneStats(sc)
state = sc.get()
cap = 2.0 * sc.dt_cfl(0.8)
kw = dict(t0=0.0, t_end=0.6, sample_dt=5 * cap, observe=lambda s: ps.sample(), u_ref=18.0, log=lambda m: None, ptol=1e-8)
fixed = gpu.run_transient(sc, cap, **kw)                               # a fixed step of twice the limit: must be reported
sc.set(state)
sc.project()
ad = gpu.run_transient(sc, cap, adapt_cfl=0.8, **kw)
cf_fixed = next(c for c in fixed["health"]["checks"] if c["name"] == "Max Courant number")
cf_ad = next(c for c in ad["health"]["checks"] if c["name"] == "Max Courant number")
check("run_transient: the Courant check uses each cell's own size when observe gives courant_rate",
      cf_fixed["status"] == "warn" and 1.0 < float(cf_fixed["value"]) < 3.0
      and float(fixed["series"]["umax"].max()) * cap / ps.dy.min() > 5.0,
      f"fixed step: {cf_fixed['value']} (largest velocity over smallest cell would give "
      f"{float(fixed['series']['umax'].max()) * cap / ps.dy.min():.1f})")
check("run_transient: adapt_cfl holds the Courant number, ends at t_end and records the range of its steps",
      cf_ad["status"] == "ok" and float(cf_ad["value"]) <= 1.0 and abs(ad["t"][-1] - 0.6) < 1e-9
      and ad["dt_min"] <= ad["dt"] <= ad["dt_max"] <= cap and ad["health"]["steps"] > fixed["health"]["steps"]
      and np.all(np.diff(ad["t"]) > 0) and "dt_min" not in fixed,
      f"Courant {cf_ad['value']}, {ad['health']['steps']} steps of {ad['dt_min']:.2e} to {ad['dt_max']:.2e}")
av = signals.stationary(ad["t"], ad["series"]["U"], min_blocks=2)
check("a series of profiles from run_transient averages point by point", av["mean"].shape == (32,) and av["stderr"].shape == (32,))
reca = gpu.write_run(OUT / "runs", "chan", ad, dict(level="test", cells=32 ** 3, h=float(ps.dy.min()), source=None, pressure_tol=1e-8,
                                                    scheme="imex", time_step=dict(dt=cap, set_by="adapt_cfl 0.8")))
check("write_run: a run with adapt_cfl records its mean, smallest and largest step", reca["dt_min"] == ad["dt_min"] and reca["dt"] == ad["dt"])

# 9. what case 17 asked of the run loop: steps as settings, a step control of its own, restart files,
#    thinned records, a pre-flight, a time-step check, a look into a record
sc.set(state)
sc.project()
ck = OUT / "state" / "chan.npz"
kw9 = {k: v for k, v in kw.items() if k != "sample_dt"}
st9 = gpu.run_transient(sc, None, sample_dt=None, adapt_cfl=0.8, sample_steps=10, cfl_steps=2, checkpoint=ck, checkpoint_every=0.2, **kw9)
cf9 = next(c for c in st9["health"]["checks"] if c["name"] == "Max Courant number")
check("run_transient: sample_steps and cfl_steps (no largest step given) hold the Courant number and end at t_end",
      cf9["status"] == "ok" and float(cf9["value"]) <= float(cf_ad["value"]) + 0.02 and abs(st9["t"][-1] - 0.6) < 1e-9
      and st9["perf"]["sample_steps"] == 10 and st9["health"]["steps"] >= 10 * (len(st9["t"]) - 2),
      f"Courant {cf9['value']} with a look every 2 steps (every 5: {cf_ad['value']}), {st9['health']['steps']} steps, {len(st9['t'])} samples")
end9 = sc.get()
p9 = sc.pressure()
sc2 = channel("imex")
t9 = gpu.load_state(sc2, st9["state"])
check("save_state / load_state: the checkpoint at the end of a run holds its velocity, pressure and time",
      abs(t9 - 0.6) < 1e-9 and all(np.array_equal(a, b) for a, b in zip(end9, sc2.get())) and np.array_equal(p9, sc2.pressure()))
a9, b9 = sc.run(1e-4, 5, 1e-10), sc2.run(1e-4, 5, 1e-10)
check("a run continued from a loaded state goes on like the original",
      max(np.abs(a - b).max() for a, b in zip(sc.get(), sc2.get())) < 1e-9, f"{max(np.abs(a - b).max() for a, b in zip(sc.get(), sc2.get())):.1e}")
sc2.close()
rec9 = gpu.write_run(OUT / "runs", "chan9", st9, dict(level="test", cells=32 ** 3, h=float(ps.dy.min()), source=None, pressure_tol=1e-8,
                                                     scheme="imex", time_step=dict(set_by="adapt_cfl 0.8")), store_every=3, solver=sc)
z9 = np.load(OUT / "runs" / "chan9.npz")
check("write_run(store_every, solver): profiles thinned, scalar series kept in full, solver settings in the record",
      len(z9["t"]) < len(z9["t_all"]) == len(st9["t"]) and z9["U"].shape[0] == len(z9["t"]) and np.array_equal(z9["bulk_all"], st9["series"]["bulk"])
      and z9["t"][-1] == st9["t"][-1] and rec9["store_every"] == 3 and rec9["solver"]["cells"] == [32, 32, 32]
      and rec9["solver"]["scheme"] == "imex" and rec9["state"] == str(ck))
txt9 = gpu.peek(OUT / "runs" / "chan9.npz", windows=3)
check("peek: a record read back as text, one line per scalar series", "bulk" in txt9 and "tau_low" in txt9 and "U " not in txt9, txt9.splitlines()[0])
sc.set(state)
sc.project()
pf9 = gpu.preflight(sc, None, 0.0, 0.3, None, lambda s: ps.sample(), 18.0, time_units=40.0, adapt_cfl=0.8, sample_steps=5,
                    log=lambda m: None, ptol=1e-8)
check("preflight: health, cost per step, steps and seconds per time unit, projected run time",
      pf9["health"]["status"] == "ok" and pf9["steps_per_unit"] > 100 and abs(pf9["projected_wall_s"] - 40 * pf9["wall_per_unit_s"]) < 1e-9
      and 0 < pf9["observe_share"] < 1 and abs(pf9["steps_per_unit"] * pf9["dt_mean"] - 1.0) < 0.3,
      f"{pf9['ms_per_step_wall']:.1f} ms per step, {pf9['steps_per_unit']:.0f} steps and {pf9['wall_per_unit_s']:.1f} s per time unit")


def developed9():
    s_ = channel("imex")
    s_.set(state)
    s_.project()
    return s_


dt9 = 0.5 * cap * 0.4
sk9 = gpu.step_check(developed9, dt9, 0.0, 20 * dt9, 2 * dt9, lambda s: turbulence.PlaneStats(s).sample(), 18.0, ptol=1e-9,
                     keys=["bulk", "tau_low"], log=lambda m: None)
check("step_check: the same stretch with the step and half of it, compared sample by sample",
      set(sk9["diff"]) == {"bulk", "tau_low"} and 0 < sk9["first"]["tau_low"] <= sk9["diff"]["tau_low"] < 1e-2
      and len(sk9["t"]) == 10 and sk9["health"] == ["ok", "ok"],
      f"wall shear differs by {sk9['diff']['tau_low']:.1e} (first sample {sk9['first']['tau_low']:.1e}), bulk velocity by {sk9['diff']['bulk']:.1e}")
# 10. the optimize stage of case 17: a start-up shortened by moving the bulk velocity to where the wall shear points
sc.set(state)
before = ps.sample()
fac = turbulence.rescale_to_balance(sc, 1.035, 1.0)                    # a wall shear 3.5 % above the driving force
after = ps.sample()
check("rescale_to_balance: the bulk velocity moves by 1 - e / 1.75 and the wall shear with it; the factor is limited",
      abs(fac - 0.98) < 1e-12 and abs(after["bulk"] / before["bulk"] - fac) < 1e-12
      and abs(after["tau_low"] / before["tau_low"] - fac) < 1e-9
      and turbulence.balance_factor(2.0, 1.0) == 0.97 and turbulence.balance_factor(1.0, 1.0) == 1.0,
      f"factor {fac:.4f}, bulk {before['bulk']:.4f} to {after['bulk']:.4f}")
sc.close()

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
