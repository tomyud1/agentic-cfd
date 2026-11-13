"""Smoke tests of cfdlib tools added after case 4 (run inside WSL from the project root).

    PYTHONPATH=. python tests/test_tools.py
"""
import json
from pathlib import Path

import numpy as np

from cfdlib import contract, perf, signals, verification

ROOT = Path(__file__).resolve().parents[1]

# 1. signals: must reproduce case 4's own cycle analysis of the finest run
d = np.load(ROOT / "results/dfg_cylinder_re100/runs/M4.npz", allow_pickle=True)
t, F = d["t"], d["Fcyl"]
dz = json.loads((ROOT / "results/dfg_cylinder_re100/runs/M4.json").read_text()).get("dz", 0.01) \
    if (ROOT / "results/dfg_cylinder_re100/runs/M4.json").exists() else 0.01
q = 0.5 * 1.0 * 1.0 ** 2 * 0.1                         # rho Umean^2 D / 2 (Umean = 1)
cd, cl = F[:, 0] / dz / q, F[:, 1] / dz / q
cyc = signals.cycles(t, cl, others={"cd": cd}, window=3.0)
c = cyc[-1]
ok, spread = signals.periodic(cyc, ["max", "min", "period"], rtol=5e-4, n_last=5)
late = t > t[-1] - 3.0
f = signals.dominant_frequency(t[late], cl[late])
print(f"signals: cl max {c['max']:.5f} min {c['min']:.5f} cd max {c['cd_max']:.5f} min {c['cd_min']:.5f} "
      f"St {0.1 / c['period']:.5f} fft St {0.1 * f:.5f} periodic {ok}")
print("  case 4 report: cl max 0.98998, cl min -1.02475, cd min 3.16485, St 0.30150")

# 2. level_sequence on the FeatFlow drag means (levels 3-6, finest time step): order about 1
print("level_sequence:", verification.level_sequence([3.1308, 3.1645, 3.1807, 3.1884]))

# 3. perf: pressure solves by position (v2606 names them all "p")
run = Path("/home/tomyud/OpenFOAM/tomyud-v2606/run/agentic/dfg_cylinder_re100/psolver_M3_GAMG_1e-07_0.01")
if run.exists():
    p = perf.summarize(run, "pimpleFoam", cells=53728)
    print("perf: steps", p["steps"], "p by solve", {k: round(v, 2) for k, v in p["lin_iters_by_solve"].get("p", {}).items()})

# 4. contract: control characters are errors
bad = {"x": {"formula": "a \t b", "svg": "\t ignored"}}
print("control chars:", contract.control_chars(bad))

# 5. monitor.settled: converged, floor (noise above tol), drift (small steps, one direction), and a stop switch
from cfdlib import monitor  # noqa: E402

rng = np.random.default_rng(0)
conv = 3.2 + 0.01 * 0.3 ** np.arange(12)                       # geometric approach: settled
noise = 3.2 + 7e-4 * rng.standard_normal(14)                   # jitter 7e-4 > tol 5e-4: floor
drift = 3.2 + 4e-4 * np.arange(10)                             # each change 4e-4 < tol, but still moving: running
states = [monitor.settled(v, 5e-4)["state"] for v in (conv, noise, drift)]
print("settled states (expect settled, floor, running):", states)
assert states == ["settled", "floor", "running"], states
j = monitor.judge({"a": conv, "b": noise}, {"a": 5e-4, "b": 5e-4})
assert j["stop"] and j["status"] == "warn", j
import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / "system").mkdir()
    (Path(tmp) / "system/controlDict").write_text("application x;\n\nfunctions\n{\n    f1 { type probes; }\n}\n")
    monitor.add_stop_switch(tmp)
    monitor.add_stop_switch(tmp)                               # idempotent
    txt = (Path(tmp) / "system/controlDict").read_text()
    assert txt.count("type            abort;") == 1 and txt.index("liveStop") < txt.index("f1"), txt
print("stop switch: ok")

# 6. monitor.DatTail: a force file grown in pieces (cut mid-line) reads the same as in one go
from cfdlib import foam  # noqa: E402
src = ROOT / "results/dfg_cylinder_re100"
with tempfile.TemporaryDirectory() as tmp:
    d6 = Path(tmp) / "postProcessing/forcesCyl/0"
    d6.mkdir(parents=True)
    lines6 = "# Time  total_x total_y total_z\n" + "".join(
        f"{0.001 * i:.6g}\t({np.sin(i):.8g} {np.cos(i):.8g} 0)\n" for i in range(1, 2001))
    tail = monitor.DatTail(tmp, "forcesCyl", "force.dat")
    for cut in sorted(rng.integers(0, len(lines6), 9).tolist()) + [len(lines6)]:
        (d6 / "force.dat").write_text(lines6[:cut])
        got = tail()
    ref = np.array([monitor._flat(r) for r in foam.parse_dat(lines6) if isinstance(r[0], float)])
    assert got.shape == ref.shape and np.array_equal(got, ref), (got.shape, ref.shape)
print("DatTail: ok")

# 7. Steady stop on the iteration error: replay on case 2's recorded probe histories (SIMPLEC, U 0.9).
#    The estimate must bound the true remaining error (distance to the run's final, residual-converged
#    value); stop rows as first measured with the dev replay (2026-11-12): 171, 521, 1931, 7491 of
#    267, 849, 2997, 10864.
for N, expect in ((32, 171), (64, 521), (128, 1931), (256, 7491)):
    d7 = np.load(ROOT / f"results/lid_driven_cavity_re100/runs/N{N}.npz")
    v7 = d7["probe_vals"]                                  # U = 1, so already in units of U
    r7 = monitor.steady_replay(v7, 1e-6, every=10, hold=2)
    true7 = float(np.max(np.abs(v7[r7["index"]] - v7[-1])))
    print(f"steady stop N{N}: row {r7['index'] + 1} of {len(v7)}, true error {true7:.2g} < 1e-6")
    assert r7["index"] + 1 == expect and true7 < 1e-6, (N, r7["index"], true7)
# live form: the decider on a probes file grown in pieces stops, and not before the replay would
with tempfile.TemporaryDirectory() as tmp:
    d7 = Path(tmp) / "postProcessing/probes/0"
    d7.mkdir(parents=True)
    v7 = np.load(ROOT / "results/lid_driven_cavity_re100/runs/N64.npz")["probe_vals"]
    text7 = "# Probe 0 (0.5 0.5 0.005)\n# Time\n" + "".join(
        f"{i + 1}\t" + "\t".join(f"({v7[i, 2 * j]:.12g} {v7[i, 2 * j + 1]:.12g} 0)" for j in range(4)) + "\n"
        for i in range(len(v7)))
    lines7 = text7.splitlines(keepends=True)
    dec = monitor.steady_decider("probes", "U", 1e-6, columns=[0, 1, 3, 4, 6, 7, 9, 10])
    first = None
    for n in range(100, len(lines7), 100):
        (d7 / "U").write_text("".join(lines7[:n]))
        r = dec(tmp)
        if r and r["stop"]:
            first = r["t"]
            break
    print(f"steady decider (checks every 100 rows): stop at row {first:.0f}; {r['reason']}")
    assert first is not None and first >= 521, first
print("steady stop: ok")

# 8. contract: a dictionary the parser cannot read is reported, not skipped silently
from cfdlib import contract as _contract  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / "system").mkdir()
    (Path(tmp) / "system/controlDict").write_text("application simpleFoam;\n")
    (Path(tmp) / "system/broken").write_text('a "b;\n')             # unterminated string: parse fails
    errs8 = _contract.uncovered_settings([{"settings": ["system/controlDict/*"]}], tmp)
    print("unparsed file reported:", errs8)
    assert len(errs8) == 1 and "could not parse system/broken" in errs8[0], errs8
print("contract unparsed: ok")
