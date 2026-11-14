"""Smoke tests of cfdlib tools added after case 4 (run inside WSL from the project root).

    PYTHONPATH=. python tests/test_tools.py
"""
import json
import math
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
from cfdlib import foam  # noqa: E402
RUN = Path(foam.foam_env("FOAM_RUN")) / "agentic"   # where case runs live on this machine
run = RUN / "dfg_cylinder_re100/psolver_M3_GAMG_1e-07_0.01"
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
    # the decision also says where the rule first held on the history (case 7, insight 7): the replay's row
    print(f"  rule first held at row {r['t_first_held']:.0f} (checks came every 100 rows)")
    assert r["t_first_held"] == 521, r["t_first_held"]
# A solution that flips between two states on every iteration (case 7, insight 3): the recorded trial
# with the turbulence variable relaxed by 0.9. Sampled every 20 rows it is constant, and was stopped
# as settled; the check must now refuse it at every point of the history, for even and odd sampling.
h7 = np.load(ROOT / "results/turbulent_flat_plate_sa/runs/trial_base.npz")["hist"]
tol7 = 1e-5 * np.abs(h7[-1])                                # the run's own tolerance: 1e-5 of each value
for every7 in (20, 25):
    r7 = monitor.steady_check(h7, tol7, every=every7)
    held7 = monitor.steady_replay(h7, tol7, every=every7, hold=2)["index"]
    states7 = sorted({q["state"] for q in r7["quantities"].values()})
    print(f"period-2 cycle, sampled every {every7}: ok {r7['ok']}, states {states7}, replay stop {held7}")
    assert not r7["ok"] and held7 is None, (every7, r7["reason"], held7)
r7 = monitor.steady_check(h7, tol7, every=20)
assert "oscillating" in r7["reason"] and r7["status"] == "warn", r7["reason"]
print(" ", r7["reason"][:150])
# and it must not hold back a run that converges: the main run of the same grid stops where it did before
g7 = np.load(ROOT / "results/turbulent_flat_plate_sa/runs/G1.npz")["hist"]
held7 = monitor.steady_replay(g7, 1e-5 * np.abs(g7[-1]), every=25, hold=2)["index"]
true7 = float(np.max(np.abs(g7[held7] - g7[-1]) / np.abs(g7[-1])))
print(f"converging run of the same grid: rule holds from row {held7 + 1} of {len(g7)}, true error {true7:.1e} of the value")
assert held7 is not None and true7 < 1e-5, (held7, true7)
# synthetic traps: a decaying flip is not a cycle (it settles once it is below the tolerance); a lasting one is
k7 = np.arange(400.0)
decay7 = 1.0 + 1e-2 * (-0.9) ** k7
cycle7 = 1.0 + 1e-3 * (-1.0) ** k7 + 1e-2 * 0.9 ** k7
s7 = [monitor.steady_check(v, 1e-6, every=10)["quantities"]["c0"]["state"] for v in (decay7, cycle7, cycle7[:-1])]
print("decaying flip, lasting flip (two phases):", s7)
assert s7 == ["settled", "oscillating", "oscillating"], s7
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

# 9. Steady stop on rounded output (case 6, insights 9 and 15): its recorded probe histories (column 0 is
#    the iteration) never stopped before the fallback; now every run must stop with the true remaining
#    change (to the run's last row, converged to its printed digits) below the tolerance.
for n6, (rT, rU) in {"L1": (91, 91), "L2": (161, 181), "L3": (341, 441), "L4": (1221, 1611)}.items():
    d9 = np.load(ROOT / f"results/heated_pipe_re200/runs/{n6}.npz")
    for key9, tol9, expect9 in (("probe_T", 1e-6, rT), ("probe_Ux", 1e-7, rU)):
        v9 = d9[key9][:, 1:]
        r9 = monitor.steady_replay(v9, tol9, every=10, hold=2)
        assert r9["index"] is not None, (n6, key9)
        true9 = float(np.max(np.abs(v9[r9["index"]] - v9[-1])))
        print(f"case 6 {n6} {key9}: stop row {r9['index'] + 1} of {len(v9)}, true error {true9:.1e} < {tol9:.0e}")
        assert true9 < tol9 and r9["index"] + 1 == expect9, (n6, key9, r9["index"], true9)
# traps for the fallback: it must not stop on these
k9 = np.arange(4000.0)
two_mode = np.round(1e-3 * 0.97 ** k9 + 2e-5 * 0.9995 ** k9, 12)      # fast mode dies first, slow one stays
r9 = monitor.steady_replay(two_mode, 1e-6, every=10, hold=2)
true9 = abs(two_mode[r9["index"]] - 2e-5 * 0.9995 ** 1e9) if r9["index"] is not None else 0.0
print(f"two modes: stop at {r9['index']}, true remaining {true9:.1e}")
assert r9["index"] is None or true9 < 1e-6, (r9["index"], true9)
# output-precision rule: a one-way drift of one unit of the last digit is refused; flicker is accepted
moved = np.linspace(0.0, 1e-6, 50)                                      # the value has moved earlier
drift = np.concatenate([moved, 1e-6 + 1e-12 * np.arange(1, 31)])
flick = np.concatenate([moved, 1e-6 + 1e-12 * (np.arange(1, 31) % 2)])
e_drift, e_flick = monitor._fallback_error(drift)[0], monitor._fallback_error(flick)
print(f"output precision: one-way drift -> {e_drift}, flicker -> {e_flick[0]:.1e} ({e_flick[1]})")
assert e_drift == math.inf and e_flick[1] == "output precision" and e_flick[0] < 2e-11, (e_drift, e_flick)
late = np.where(k9 < 3000, 5.0, 5.0 + 1e-3 * (1 - 0.99 ** (k9 - 3000)))  # a probe the flow reaches late
r9 = monitor.steady_replay(np.round(late, 12)[:3100], 1e-6, every=10, hold=2)
print("probe reached late (history cut 100 rows after it starts moving): stop at", r9["index"])
assert r9["index"] is None or r9["index"] > 3000, r9
print("steady stop on rounded output: ok")

# 10. foam safety (case 6, insights 3, 7, 10): render_case refuses to delete a case a solver still works in,
#     stop() kills orphans by working directory, ignored_settings flags SIMPLEC for a solver without it,
#     map_fields drops post-processing fields. Needs WSL with OpenFOAM (FOAM_SOLVERS for the source check).
import os, shutil, subprocess, time  # noqa: E401,E402
with tempfile.TemporaryDirectory() as tmp:
    case10 = Path(tmp) / "c"
    (case10 / "system").mkdir(parents=True)
    fake = Path(tmp) / "fakeFoam"                       # an "OpenFOAM solver" by name, sleeping in the case
    shutil.copy("/bin/sleep", fake)                     # a real binary, so ps shows it as fakeFoam
    proc = subprocess.Popen([str(fake), "60"], cwd=case10, start_new_session=True)
    time.sleep(0.5)
    try:
        foam.render_case(Path(tmp) / "c", case10, {})
        raise AssertionError("render_case deleted a live case")
    except RuntimeError as e:
        print("render_case guard:", str(e)[:70], "...")
    killed = foam.stop(case10)
    time.sleep(0.3)
    assert proc.poll() is not None and killed, killed
    print("stop() killed the orphan:", killed)
    (case10 / "system/controlDict").write_text("application buoyantBoussinesqSimpleFoam;\n")
    (case10 / "system/fvSolution").write_text("SIMPLE { consistent yes; }\n")
    w10 = foam.ignored_settings(case10)
    (case10 / "system/controlDict").write_text("application simpleFoam;\n")
    w10b = foam.ignored_settings(case10)
    print("ignored settings: buoyantBoussinesqSimpleFoam ->", len(w10), "warning(s); simpleFoam ->", len(w10b))
    assert len(w10) == 1 and "SIMPLE" in w10[0] and w10b == [], (w10, w10b)
run10 = RUN / "lid_driven_cavity_re100"
if (run10 / "N128").exists() and (run10 / "N256").exists():
    tgt = run10.parent / "_test_map_fields"
    shutil.rmtree(tgt, ignore_errors=True)
    tgt.mkdir()
    for sub in ("system", "constant", "0"):
        shutil.copytree(run10 / "N256" / sub, tgt / sub)
    wall10, removed10 = foam.map_fields(tgt, run10 / "N128")
    left = sorted(f.name for f in (tgt / "0").iterdir())
    print(f"map_fields: kept {left}, removed {removed10}")
    assert left == ["U", "p"], left
    # a fixed boundary value of this case survives the mapping (case 7, insight 16): lid speed 1.25, source 1
    shutil.rmtree(tgt / "0")
    shutil.copytree(run10 / "N256" / "0", tgt / "0")
    lid10 = "movingWall"
    assert foam._fixed_values(tgt / "0/U") == {lid10: "uniform (1.0 0 0)"}, foam._fixed_values(tgt / "0/U")
    foam.run(f"foamDictionary 0/U -entry boundaryField/{lid10}/value -set 'uniform (1.25 0 0)'", tgt, "setLid")
    foam.map_fields(tgt, run10 / "N128", keep_boundary=False)
    lost10 = np.atleast_2d(foam.read_boundary_field(tgt / "0/U", lid10))[:, 0].max()
    shutil.rmtree(tgt / "0")
    shutil.copytree(run10 / "N256" / "0", tgt / "0")
    foam.run(f"foamDictionary 0/U -entry boundaryField/{lid10}/value -set 'uniform (1.25 0 0)'", tgt, "setLid")
    foam.map_fields(tgt, run10 / "N128")
    kept10 = np.atleast_2d(foam.read_boundary_field(tgt / "0/U", lid10))[:, 0].max()
    inner10 = foam.read_internal_field(tgt / "0/U")
    print(f"map_fields, lid speed set to 1.25: plain mapFields leaves {lost10:g} (the source's), "
          f"keep_boundary leaves {kept10:g}; mapped interior {inner10.shape}, max |U| {np.abs(inner10).max():.3f}")
    assert lost10 == 1.0 and kept10 == 1.25 and np.abs(inner10).max() > 0.5, (lost10, kept10)
    shutil.rmtree(tgt)
else:
    print("map_fields: case 2 runs not on this machine, skipped")
with tempfile.TemporaryDirectory() as tmp:        # values on a boundary patch (wall shear stress, y+)
    fld = Path(tmp) / "wallShearStress"
    fld.write_text("""FoamFile { version 2.0; format ascii; class volVectorField; object wallShearStress; }
dimensions [0 2 -2 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
    inlet { type calculated; value uniform (0 0 0); }
    plate
    {
        type            calculated;
        value           nonuniform List<vector> 3
(
(-3.5 1e-12 0)
(-2.25 0 0)
(-1.125 -2e-13 0)
)
;
    }
    plateEnd { type calculated; value uniform (-1 0 0); }
    outlet { type zeroGradient; }
}
""")
    tw = foam.read_boundary_field(fld, "plate")
    assert tw.shape == (3, 3) and list(tw[:, 0]) == [-3.5, -2.25, -1.125], tw
    assert list(foam.read_boundary_field(fld, "plateEnd")) == [-1.0, 0.0, 0.0]      # not the patch whose name it starts with
    for bad in ("outlet", "nowhere"):
        try:
            foam.read_boundary_field(fld, bad)
            raise AssertionError(f"{bad}: expected a ValueError")
        except ValueError as e:
            assert ("stores no value" in str(e)) == (bad == "outlet"), e
    print("read_boundary_field: list, uniform value, and a clear error for a patch without values")
    fld.write_text("""FoamFile { version 2.0; format ascii; class volVectorField; object U; }
internalField uniform (0 0 0);
boundaryField
{
    inlet   { type fixedValue; value uniform (50.15 0 0); }   // the variant's own speed
    plate   { type noSlip; }
    top     { type inletOutlet; inletValue uniform (50 0 0); value uniform (50 0 0); }
    profile { type fixedValue; value nonuniform List<vector> 2((1 0 0) (2 0 0)); }
    outlet  { type zeroGradient; }
}
""")
    fv = foam._fixed_values(fld)
    assert fv == {"inlet": "uniform (50.15 0 0)"}, fv
    print("fixed boundary values found for map_fields to keep:", fv)
# monitor.preflight on a decomposed case (case 7, insight 2): a parallel command without processor folders is
# refused with the reason; with them, the short copy runs on its ranks and is assessed.
run10 = RUN / "turbulent_flat_plate_sa/G1"
if run10.exists():
    src10 = run10.parent / "_test_preflight_mpi"
    for d10 in (src10, src10.parent / (src10.name + "_preflight")):
        shutil.rmtree(d10, ignore_errors=True)
    shutil.copytree(run10, src10, ignore=shutil.ignore_patterns(
        "processor*", "postProcessing", "log.*", "[1-9]*", "*.json", "*.pid", monitor.STOP_FILE))
    (src10 / "system/decomposeParDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\n"
        "numberOfSubdomains 2;\nmethod simple;\ncoeffs { n (2 1 1); }\n")
    cmd10 = "mpirun -np 2 simpleFoam -parallel"
    try:
        monitor.preflight(src10, cmd10, end_time_short=30, end_time_full=1000)
        raise AssertionError("expected a ValueError: no processor folders")
    except ValueError as e:
        assert "decomposePar" in str(e), e
    foam.run("decomposePar -force", src10, "decomposePar")
    pf10 = monitor.preflight(src10, cmd10, end_time_short=30, end_time_full=1000, timeout=120)
    print(f"preflight on 2 ranks: {pf10['steps']} iterations, status {pf10['status']}, "
          f"projected full run {pf10['est_full_runtime']:.1f} s")
    assert pf10["steps"] == 30 and pf10["status"] != "fail" and not pf10["timed_out"], pf10
    for d10 in (src10, src10.parent / (src10.name + "_preflight")):
        shutil.rmtree(d10)
else:
    print("preflight on a decomposed case: case 7 runs not on this machine, skipped")
print("foam safety: ok")

# 11. Checks that came from case 7's report stage (insights 9, 10, 11): a problem statement the report
#     would refuse is caught when it is written; a quoted figure placeholder is not a reference; a
#     placeholder without the word Figure and a sentence in the summary's last column are warned about.
with tempfile.TemporaryDirectory() as tmp:
    pr = Path(tmp) / "problem.yaml"
    base11 = "name: x\ntitle: T\nsetup: ['a']\ngiven: []\nquantities_of_interest: [{key: a, text: b}]\n"
    pr.write_text(base11 + "description: " + "word " * 130 + "\nreference: {file: reference/missing.yaml}\n")
    e11 = contract.check_problem_file(pr)
    pr.write_text(base11 + "description: A short overview.\n")
    print("problem statement check:", [x[:60] for x in e11], "| short one:", contract.check_problem_file(pr))
    assert len(e11) == 2 and "description" in e11[0] and "missing.yaml" in e11[1] and contract.check_problem_file(pr) == []
e11 = contract.check_problem_file(ROOT / "cases/turbulent_flat_plate_sa/problem.yaml")
assert len(e11) == 1 and e11[0].startswith("description: "), e11   # the case that led to the check (kept as given)
for c11 in ("plane_poiseuille", "lid_driven_cavity_re100", "stokes_first_problem", "dfg_cylinder_re100",
            "dfg_cylinder_3d_re20", "heated_pipe_re200"):
    assert contract.check_problem_file(ROOT / f"cases/{c11}/problem.yaml") == [], c11
assert contract.fig_refs("see Figure {fig:a}; the placeholder <code>{fig:key}</code> inserts a number") == ["a"]
w11 = contract.lint({"summary": {"answers": [{"diff": "\\(0.03\\,\\%\\)"}, {"diff": "\\(+0.008\\) (0.95 band)<br>\\(+0.008\\) (0.97 band)<div class='sub'>extrapolated: 0.02</div>"},
                                             {"diff": "the difference between the two values on the finest grid of the family"}]},
                     "x": ["{fig:a}: from the data", "Figures {fig:a} and {fig:b} show it", "(Figure&nbsp;{fig:c})"]})[1]
print("report warnings:", [w[:58] for w in w11])
assert len(w11) == 2 and "bare number" in w11[0] and "answers[2].diff" in w11[1], w11
print("report-stage checks: ok")

# 12. Averages of a signal that fluctuates about a level (turbulence; prepared for the channel case):
#     the statistical error against a signal whose true error is known, the start-up cut, the live check,
#     and the two recorded channel runs of dev/exploration/les_channel (the finer one had not settled).
from cfdlib import gpu as gpu12  # noqa: E402
from gpucfd import grids as grids12  # noqa: E402


def ar1(n, phi, seed):
    e = np.random.default_rng(seed).standard_normal(n)
    y = np.empty(n)
    y[0] = e[0] / np.sqrt(1 - phi ** 2)
    for k in range(1, n):
        y[k] = phi * y[k - 1] + e[k]
    return y


t12 = np.arange(20000) * 0.01
for phi in (0.0, 0.9, 0.98):
    true = np.sqrt(1.0 / (1 - phi ** 2) * (1 + phi) / (1 - phi) / len(t12))   # exact error of the mean of this process
    st12 = [signals.stationary(t12, 5.0 + ar1(len(t12), phi, seed)) for seed in range(12)]
    ratio = np.array([s["stderr"] for s in st12]) / true
    miss = np.array([abs(s["mean"] - 5.0) for s in st12]) / np.array([s["stderr"] for s in st12])
    print(f"stationary, memory {phi}: error estimate / true error = {ratio.mean():.2f} ({ratio.min():.2f} to {ratio.max():.2f}); "
          f"largest miss {miss.max():.1f} standard errors")
    assert 0.9 < ratio.mean() < 1.4 and ratio.min() > 0.7 and ratio.max() < 2.0 and miss.max() < 3.5, (phi, ratio)
    assert all(s["reliable"] and abs(s["drift_sigma"]) < 4 for s in st12)
y12 = 5.0 + ar1(len(t12), 0.9, 3)
prof12 = signals.stationary(t12, np.stack([y12, 2.0 * y12 + 1.0], axis=1))          # an array per time: point by point
one12 = signals.stationary(t12, y12)
assert np.allclose(prof12["mean"], [one12["mean"], 2 * one12["mean"] + 1]) and np.allclose(prof12["stderr"], [one12["stderr"], 2 * one12["stderr"]])
tu12 = np.sort(np.random.default_rng(0).uniform(0, 200, 20000))                      # uneven sampling: weighted by time
assert abs(signals.stationary(tu12, np.where(tu12 < 100, 1.0, 3.0))["mean"] - 2.0) < 1e-3
start12 = y12 + 30.0 * np.exp(-t12 / 4.0)                                            # a start-up transient, down to the size of the fluctuations by t of about 10
cut12 = signals.settle_time(t12, start12)
st_cut = signals.stationary(t12, start12, t_start=cut12["t_start"])
print(f"settle_time: cut at t = {cut12['t_start']:.1f} (settled {cut12['settled']}); mean after it {st_cut['mean']:.3f} +- {st_cut['stderr']:.3f}; "
      f"without the cut {signals.stationary(t12, start12)['mean']:.3f}")
assert cut12["settled"] and 5 <= cut12["t_start"] < 60 and abs(st_cut["mean"] - 5.0) < 4 * st_cut["stderr"]
ramp12 = signals.stationary(t12, y12 + 0.02 * t12)
assert abs(ramp12["drift_sigma"]) > 3 and signals.settle_time(t12, y12 + 0.02 * t12)["settled"]      # a slow ramp: only the drift test sees it
assert not signals.settle_time(t12, y12 + 0.2 * t12)["settled"]                                          # a strong one: both do
w12 = signals.stationary_watch("q", 0.03, min_time=50.0)
said = [w12(t12[k - 1], dict(t=t12[:k], q=start12[:k])) for k in (2000, 6000, 12000, 20000)]
print("stationary_watch on the transient signal at t = 20, 60, 120, 200:", [bool(v) for v in said], "|", said[-1])
assert not said[0] and said[-1] and "+-" in said[-1]
assert signals.stationary_watch("q", 1e-5, min_time=50.0)(t12[-1], dict(t=t12, q=start12)) is None    # a tolerance out of reach
rec12 = json.loads((ROOT / "dev/exploration/les_channel/results.json").read_text())
for key12, drifting in (("run_64x64x64_imex", False), ("run_192x64x64_imex", True)):
    a12 = np.array(rec12[key12]["series"]["rows"])
    s12 = signals.stationary(a12[:, 0], a12[:, 2], t_start=rec12[key12]["window"][0])
    print(f"recorded channel run {key12[4:]}: bulk velocity {s12['mean']:.3f} +- {s12['stderr']:.3f}, drift {s12['drift_sigma']:+.1f} sigma, "
          f"{s12['n_independent']:.0f} independent samples, reliable {s12['reliable']}")
    assert (abs(s12["drift_sigma"]) > 2) == drifting and not s12["reliable"]
yw12 = grids12.tanh_wall(64, 2.0, 1.0 / 180.0)
assert abs(np.diff(yw12).min() * 180.0 - 1.0) < 1e-9 and np.allclose(np.diff(yw12), np.diff(yw12)[::-1]) and len(yw12) == 65
assert np.allclose(grids12.tanh_wall(8, 2.0, 0.3), grids12.uniform(8, 2.0))
f12 = gpu12.check_fit(cell_reynolds=212.0, wall_units=dict(dx=11.8, dy_wall=1.0, dz=11.8))
c12 = gpu12.check_fit(wall_units=dict(dx=35.3, dy_wall=1.0, dz=11.8))
assert f12["ok"] and len(f12["notes"]) == 1 and len(c12["notes"]) == 2 and "several per cent" in c12["notes"][1]
assert not gpu12.check_fit(cell_reynolds=212.0)["ok"] and not gpu12.check_fit(turbulence_model=True)["ok"]
print("fluctuating signals, wall grid, fit check for resolved turbulence: ok")

# 13. What case 17 (the turbulent channel) taught the tools: a slowly settling signal, the error of a
#     channel's bulk velocity from its recorded runs, a grid study with unequal ratios, extra tables in
#     the report, Python files that compile with a warning.
from gpucfd import turbulence as turb13  # noqa: E402


def relaxing13(seed, T=9.0, dt=0.025, span=40.0, level=15.7):
    """y is pulled back to `level` with the time T and pushed by a noise of short memory (0.3)."""
    rng = np.random.default_rng(seed)
    n, a = int(span / dt), np.exp(-dt / 0.3)
    y, rate, x, yy = np.empty(n), np.empty(n), 0.05 * rng.standard_normal(), level + 0.03 * rng.standard_normal()
    for k in range(n):
        x = a * x + np.sqrt(1 - a * a) * 0.05 * rng.standard_normal()
        rate[k] = -(yy - level) / T + x
        y[k] = yy
        yy += dt * rate[k]
    return np.arange(n) * dt, y, rate


sig13 = [relaxing13(seed) for seed in range(120)]
plain13 = [signals.stationary(t_, y_) for t_, y_, _ in sig13]
mem13 = [signals.stationary(t_, y_, memory=9.0) for t_, y_, _ in sig13]
bal13 = [signals.relaxing(t_, y_, r_, 9.0) for t_, y_, r_ in sig13]
true13 = np.std([s["mean"] for s in plain13])
ratio13 = {k: np.mean([s["stderr"] for s in v]) / true13 for k, v in (("plain", plain13), ("memory", mem13), ("balance", bal13))}
print("slowly settling signal (record of 4.4 memories), error estimate / true error:",
      {k: round(v, 2) for k, v in ratio13.items()}, "| plain batch means call themselves reliable in",
      f"{np.mean([s['reliable'] for s in plain13]):.0%} of the records")
assert ratio13["plain"] < 0.6, ratio13                     # the flaw: blocks shorter than the memory
assert 0.8 < ratio13["memory"] < 1.3 and not any(s["reliable"] for s in mem13), ratio13
assert 0.9 < ratio13["balance"] < 1.5 and np.mean([s["reliable"] for s in bal13]) > 0.9, ratio13
assert all(s["stderr_batch"] <= s["stderr"] for s in mem13 + bal13)
w13 = np.random.default_rng(5).standard_normal(8000)
assert abs(signals.stationary(np.arange(8000) * 0.01, w13, memory=0.02)["stderr"] /
           signals.stationary(np.arange(8000) * 0.01, w13)["stderr"] - 1.0) < 0.05      # a memory far below the blocks changes nothing
tt13 = np.arange(4000) * 0.02
slow13 = 15.7 - 0.5 * np.exp(-tt13 / 9.0) + 0.01 * ar1(4000, 0.5, 2)                 # a slow approach under small fluctuations
cut13 = signals.settle_time(tt13, slow13, memory=9.0)       # an approach far above the fluctuations: the fast rule sees it too
assert cut13["t_start"] == max(cut13["fast"], cut13["slow"]) and 20 < cut13["slow"] < 35
said13 = []
assert signals.stationary_watch("q", 1e-6, log=said13.append)(tt13[-1], dict(t=tt13, q=slow13)) is None and "not yet: error" in said13[0]
RUNS13 = ROOT / "results/turbulent_channel_re180/runs"
got13 = {}
for name13 in ("first_G1", "second_G1", "G1", "G1h", "G3"):
    rec13 = json.loads((RUNS13 / f"{name13}.json").read_text())
    d13 = np.load(RUNS13 / f"{name13}.npz")
    s13 = {k[:-4]: d13[k] for k in d13.files if k.endswith("_all")} if "t_all" in d13.files else {k: d13[k] for k in d13.files}
    t0_13 = float(rec13["watch_first"]["text"].split("over t = ")[1].split(" to ")[0])
    got13[name13] = turb13.bulk_average(s13["t"], s13["bulk"], 0.5 * (s13["tau_low"] + s13["tau_high"]), 1.0, 1.0, t_start=t0_13)
    if name13 == "first_G1":     # a real start-up: the fast rule cuts at 4 time units, the slow approach lasts to about 24
        cut13 = signals.settle_time(s13["t"], s13["bulk"], memory=got13[name13]["T_s"])
        print(f"settle_time on a recorded channel run: fast cut {cut13['fast']:.1f}, with the settling time {cut13['slow']:.1f}")
        assert cut13["fast"] < 6 and 20 < cut13["slow"] < 27 and cut13["t_start"] == cut13["slow"]
g1_13 = [got13[k] for k in ("first_G1", "second_G1", "G1", "G1h")]
scatter13 = np.std([s["mean"] for s in g1_13], ddof=1)
print(f"case 17, four runs of the coarsest grid: means scatter by {scatter13:.4f}; error from the balance "
      f"{np.mean([s['stderr'] for s in g1_13]):.4f}, from batch means alone {np.mean([s['stderr_batch'] for s in g1_13]):.4f}")
assert 0.5 < np.mean([s["stderr"] for s in g1_13]) / scatter13 < 2.0 and np.mean([s["stderr_batch"] for s in g1_13]) / scatter13 < 0.4
assert abs(got13["G3"]["mean"] - 15.6457) < 1e-3 and abs(got13["G3"]["stderr"] - 0.0244) < 5e-4 and got13["G3"]["reliable"]   # the report's numbers
assert 8.5 < got13["G3"]["T_s"] < 9.5 and abs(got13["G3"]["imbalance"]) < 1e-3
d13 = np.load(RUNS13 / "G3.npz")
s13 = {k[:-4]: d13[k] for k in d13.files if k.endswith("_all")}
rule13, held13, log13 = turb13.channel_watch(1.0, 1.0, tol=2.5e-3, min_time=36.0), None, []
for te13 in np.arange(10.0, 60.1, 2.0):
    m13 = s13["t"] <= te13
    if rule13(te13, {k: v[m13] for k, v in s13.items()}):
        held13 = te13
        break
turb13.channel_watch(1.0, 1.0, log=log13.append)(30.0, {k: v[s13["t"] <= 30.0] for k, v in s13.items()})
print(f"channel_watch replayed on the finest run: holds at t = {held13} (the run ended on the agent's rule at 58.0) | at 30: {log13[0][-40:]}")
assert held13 == 58.0 and "not yet" in log13[0]
F13, C13, p13, h13 = 15.7, -3.0, 1.7, [1 / 96, 1 / 144, 1 / 192]
u13 = verification.three_grid_study([F13 + C13 * x ** p13 for x in h13], h=h13)
e13, r13 = verification.three_grid_study([1.0, 1.3, 1.375], h=[4.0, 2.0, 1.0]), verification.three_grid_study([1.0, 1.3, 1.375], r=2.0)
n13 = verification.three_grid_study([15.938, 15.441, 15.646], h=h13)                 # case 17's bulk velocity: no trend
assert abs(u13["p"] - p13) < 1e-8 and abs(u13["extrap"] - F13) < 1e-9 and u13["gci_reliable"]
assert all(abs(e13[k] - r13[k]) < 1e-10 for k in ("p", "extrap", "gci_abs")) and not n13["monotonic"] and not n13["gci_reliable"]
assert math.isnan(verification.order_unequal([1.0, 1.3, 1.65], [4.0, 2.0, 1.0]))    # the changes grow: no order
res13 = json.loads((ROOT / "results/plane_poiseuille/results.json").read_text())
res13["comparison"]["tables"] = [dict(title="Repeat runs", head=["run", "value"], rows=[["a", "1"], ["b", "2"]], note="two runs")]
assert not [x for x in contract.check(res13) if "tables" in x]
from cfdlib import report as report13  # noqa: E402
with tempfile.TemporaryDirectory() as tmp13:
    svg13 = {f["key"]: '<svg xmlns="http://www.w3.org/2000/svg" width="100pt" height="50pt"></svg>' for f in res13["figures"]}
    html13 = report13.render(json.loads(json.dumps(res13)), svg13, Path(tmp13) / "report.html").read_text(encoding="utf-8")
assert "<h4>Repeat runs</h4>" in html13 and "<li>two runs</li>" in html13 and html13.count("<h4>Repeat runs</h4>") == 1
res13["comparison"]["tables"][0]["rows"].append(["c"])
assert any("as many cells as the head" in x for x in contract.check(res13))
with tempfile.TemporaryDirectory() as tmp13:
    (Path(tmp13) / "good.py").write_text('x = r"\\(u\\)"\n')
    assert contract.check_code(tmp13) == []
    (Path(tmp13) / "bad.py").write_text('x = "\\(u\\)"\n')
    bad13 = contract.check_code(tmp13)
assert len(bad13) == 1 and "bad.py" in bad13[0] and "escape" in bad13[0], bad13
for case13 in sorted(p.parent for p in (ROOT / "cases").glob("*/problem.yaml")):
    assert contract.check_code(case13) == [], (case13.name, contract.check_code(case13))
assert report13.environment_info(gpu="RTX")[-2] == ["GPU", "RTX"] and len(report13.environment_info()) == 5
print("slow signals, channel bulk velocity, unequal grid ratios, extra tables, code check: ok")

# 14. the one-page report takes gpucfd's column from a record of its runs (cases/briefs.yaml: record)
import yaml  # noqa: E402
from cfdlib import brief as brief14  # noqa: E402
spec14 = {k: v for k, v in yaml.safe_load(brief14.SPEC.read_text(encoding="utf-8"))["plane_poiseuille"].items() if k != "record"}
b14, _ = brief14.from_case("plane_poiseuille", spec14)                      # no record: every gpucfd cell waits
assert all("wait" in r[-1] for r in b14["accuracy"]["rows"] + b14["speed"]["rows"]) and "difference" in b14["accuracy"]["head"][1]
with tempfile.TemporaryDirectory() as tmp14:
    rec14 = dict(answers=dict(b=dict(diff=r"\(1\,\%\)<div class='sub'>small print</div>"), a=dict(diff="", sim="RMS x")),
                 runs=[dict(Ny=8, wall_s=0.5), dict(label="a grid of its own", cells=10, wall_s=12.0)], busy_cpu=True, startup_s=0.2)
    (Path(tmp14) / "gpucfd.json").write_text(json.dumps(rec14))
    b14, _ = brief14.from_case("plane_poiseuille", dict(spec14, record=str(Path(tmp14) / "gpucfd.json")))
    rec14["runs"].append(dict(Ny=7, wall_s=1.0))                            # a run that matches no row of the table
    (Path(tmp14) / "gpucfd.json").write_text(json.dumps(rec14))
    try:
        brief14.from_case("plane_poiseuille", dict(spec14, record=str(Path(tmp14) / "gpucfd.json")))
        raise AssertionError("a gpucfd run without a row was dropped silently")
    except ValueError as e14:
        assert "match no row" in str(e14)
acc14, sp14 = {r[0]: r[-1] for r in b14["accuracy"]["rows"]}, {r[0]: r for r in b14["speed"]["rows"]}
assert acc14["Largest velocity"] == r"\(1\,\%\)" and acc14["Velocity profile across the gap"] == "RMS x" and "wait" in acc14["Flow rate"]
assert sp14["8 cells across the gap"][-1] == "0.50 s on the GPU" and "wait" in sp14["16 cells across the gap"][-1]
assert sp14["a grid of its own"][-2:] == [brief14.DASH, "12 s on the GPU"]
assert "upper bounds" in b14["speed"]["note"] and "0.20 s once" in b14["speed"]["note"] and brief14.check_brief(b14) == []
assert [brief14.seconds(x) for x in (0.336, 3.97, 679.2)] == ["0.34 s", "4.0 s", "679 s"]
print("one-page report with a record of gpucfd's runs: ok")
