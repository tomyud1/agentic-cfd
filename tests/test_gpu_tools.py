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
rec = gpu.write_run(OUT / "runs", "cav", opt, dict(level="test"))
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

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
