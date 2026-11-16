"""Case 21 with gpucfd's joined solver (flow and chemistry on the GPU): the detonation tube exactly as the case's
problem.yaml states it, read by the rules of plan.yaml (rung d), which were written before the runs.

    bash dev/compare/run.sh detonation_tube_h2_air gpucfd_run.py study     the runs  -> gpucfd_study.json
    bash dev/compare/run.sh detonation_tube_h2_air gpucfd_run.py like      the grids of OpenFOAM's solution, added to the study
    bash dev/compare/run.sh detonation_tube_h2_air gpucfd_run.py record    the record -> gpucfd.json
    (or: PYTHONPATH=. python dev/compare/detonation_tube_h2_air/gpucfd_run.py study)

study   d1: the grids 1200, 2400, 4800, 9600 cells (19200 if the answers still move and the run is short),
            and 3200 and 4000 cells, between which the detonation is lost;
        d4: the same grids with a stronger ignition patch (NOT the case's problem; see plan.yaml);
        d2: on 1200 cells, other settings for information (the fixed setting stays the answer);
        d3: the 1200-cell tube stacked 1024 times across, timed, flow and chemistry apart.
        About ten minutes on the development PC. The GPU must be free: the script says what else is running.
record  forms gpucfd.json from the study in the form cfdlib.brief reads (answers, runs, plain, busy_cpu,
        startup_s), next to the numbers of OpenFOAM's trial run that the task statement gave. It runs nothing.
The two are apart so that the study could be run without reading anything of the case's OpenFOAM solution.
"""
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cfdlib import foam
from gpucfd import gcfd, reacting
from gpucfd.reacting import Reacting
from gpucfd.tests import test_react_tube as T

HERE = Path(__file__).resolve().parent
STUDY, OUT = HERE / "gpucfd_study.json", HERE / "gpucfd.json"
GRIDS = (1200, 2400, 4800, 9600)
LIKE_GRIDS = (400, 800, 1600, 3200)                                # the grids of OpenFOAM's solution of the case
BS23 = dict(method="bs23", rtol=1e-6, quiet=0.0)                  # the chemistry kernel's own default
# OpenFOAM's trial run, as the task statement gave it (dev/agent_checks/case21_toolgap.py): not the case's solution
TRIAL = dict(cells=1200, h=1e-4, speed_dev=-0.011, us_per_cell_step=57.0, wall_s=689.0, cores=1, p_peak=1.97e6, p_back=1.56e6, T_back=2920.0, dt=5e-9)


def gpu_free():
    """What else is using the GPU and the CPU right now (for the record; times are only fair on a free machine)."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip()
        util, mem = (float(v) for v in q.split(","))
    except Exception:                                             # noqa: BLE001
        util, mem = float("nan"), float("nan")
    return dict(gpu_utilization_percent=util, gpu_memory_used_MiB=mem, solver_processes=[p[3][:60] for p in foam.running()])


def study():
    plan = yaml.safe_load((HERE / "plan.yaml").read_text(encoding="utf-8"))
    g, mech, moles, cj = T.problem()
    before = gpu_free()
    t0 = time.perf_counter()
    name = gcfd.device()["name"]
    s = Reacting(mech, (64, 1, 1), (1e-3, 1.0, 1.0), sides={"x-": "wall", "x+": "wall"})
    s.set(T=300.0, p=101325.0, moles=moles)
    s.run_fixed(1e-9, 2)
    s.close()
    startup = time.perf_counter() - t0
    print(f"GPU: {name}; start-up {startup:.2f} s; before the runs: {before}", flush=True)
    print(f"flow setting {T.FLOW}; chemistry setting {reacting.CHEMISTRY}", flush=True)
    rec = dict(date=dt.date.today().isoformat(), device=name, startup_s=startup, machine_before=before, flow=T.FLOW, chemistry=dict(reacting.CHEMISTRY), reference=cj,
               given=g, d1=[], d2=[], d3=[])

    def run(tag, N, **kw):
        r = T.tube(g, mech, moles, cj, N, **kw)
        r["tag"] = tag
        print(f"  {tag}: " + T.line(r, cj), flush=True)
        return r

    def save():
        STUDY.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")

    # ---- d1: the grids ----
    fine = None
    for N in GRIDS:
        r = run("fixed setting", N)
        rec["d1"].append(T.slim(r))
        if r["detonation"]:
            fine = r                                              # the finest grid that holds a detonation
    a, b = rec["d1"][-2], rec["d1"][-1]
    if not b["broke"] and not a["broke"]:
        both = a["detonation"] and b["detonation"]
        moved = dict(speed=abs(b["speed"] - a["speed"]) / cj["D"] if both else None, sonic_p=abs(b["sonic"]["p"] / a["sonic"]["p"] - 1) if both and a["sonic"] and b["sonic"] else None,
                     peak=abs(b["p_peak"]["mean"] / a["p_peak"]["mean"] - 1))
        expect_s = 4 * b["wall_s"]                                # twice the cells, twice the steps
        go = (not both or moved["speed"] > 1e-3 or (moved["sonic_p"] or 1.0) > 0.01 or moved["peak"] > 0.02) and expect_s < 900
        rec["finer"] = dict(moved=moved, expected_s=expect_s, run=bool(go),
                            rule="19200 cells are run if between the two finest grids the speed moved by more than 0.1 % of CJ, the pressure at the sonic point by more "
                                 "than 1 % or the highest pressure by more than 2 % (or one of them holds no detonation), and the run is expected to stay below 15 minutes")
        print(f"  between {a['N']} and {b['N']} cells: {moved}; 19200 cells expected to take {expect_s:.0f} s: {'run' if go else 'not run'}", flush=True)
        if go:
            rec["d1"].append(T.slim(run("fixed setting", 2 * GRIDS[-1])))
    # added after the first run of d1 (which found no detonation from 4800 cells on): where is the border?
    rec["d1_border"] = [T.slim(run("fixed setting, to find where the detonation is lost", N)) for N in (3200, 4000)]
    if fine is not None:
        np.savez_compressed(HERE / "gpucfd_profiles.npz", **{k: np.asarray(v) for k, v in fine["final"].items()}, N=fine["N"], t=g["t_end"])
    save()

    # ---- d4 (plan.yaml, added after d1): a stronger patch, which is NOT the case's problem ----
    rec["d4"], rec["d4_given"] = [], dict(T.STRONG)
    for N in GRIDS + (2 * GRIDS[-1],):
        r = run(f"patch at {T.STRONG['p_ign'] / 1e6:.0f} MPa", N, given=T.STRONG)
        rec["d4"].append(T.slim(r))
        if N == GRIDS[-1] and not r["broke"]:
            np.savez_compressed(HERE / "gpucfd_profiles_strong.npz", **{k: np.asarray(v) for k, v in r["final"].items()}, N=N, t=g["t_end"])
    save()

    # ---- d2: other settings on 1200 cells, for information ----
    variants = [("WENO5 with HLLC", dict(recon="weno5", flux="hllc")),
                ("Courant number 0.8", dict(cfl=0.8)),
                ("fixed time step of 5e-9 s (OpenFOAM's trial)", dict(fixed_dt=5e-9)),
                ("splitting: chemistry then flow", dict(chemistry=dict(split="chemistry-flow"))),
                ("splitting: half the chemistry, flow, half the chemistry", dict(chemistry=dict(split="strang"))),
                ("chemistry default bs23, tolerance 1e-6", dict(chemistry=BS23)),
                ("double precision", dict(precision="double"))]
    for tag, kw in variants:
        rec["d2"].append(T.slim(run(tag, 1200, **kw)))
    save()

    # ---- d3: the cost on a full GPU: 1200 x 1024 cells, the same tube in every row ----
    for tag, kw in (("chosen chemistry setting", {}), ("chemistry default bs23, tolerance 1e-6", dict(chemistry=BS23))):
        for timed in (False, True):                               # the run's time first, then the same with flow and chemistry timed apart
            r = run(f"stacked 1200 x 1024, {tag}" + (", parts timed" if timed else ""), 1200, stack=1024, samples=20, timed=timed, **kw)
            rec["d3"].append(T.slim(r))
            if not r["broke"]:
                extra = f"; flow {r['times']['flow_s']:.1f} s, chemistry {r['times']['chemistry_s']:.1f} s" if timed else ""
                print(f"      {r['cells']} cells, {r['steps']} steps: {r['gpu_s']:.1f} s on the GPU, {r['ns_per_cell_step']:.1f} ns per cell and step; rows identical: "
                      f"{r['rows_identical']}{extra}", flush=True)
    rec["machine_after"] = gpu_free()
    rec["plan"] = dict(pass_rules=plan["ladder"]["d"]["pass"], expected=plan["ladder"]["d"]["expected"])
    save()
    print(f"wrote {STUDY.name}", flush=True)


def like():
    """Added after the study, when its runs were finished and OpenFOAM's solution of the case could be read: the
    grids of that solution (400, 800, 1600, 3200 cells), so that the two tools stand on the same grids. Adds
    "like" to gpucfd_study.json; nothing else of the study is run again."""
    g, mech, moles, cj = T.problem()
    rec = json.loads(STUDY.read_text(encoding="utf-8"))
    rec["like_machine_before"] = gpu_free()
    rec["like"] = []
    for N in LIKE_GRIDS:
        r = T.tube(g, mech, moles, cj, N)
        r["tag"] = "fixed setting, a grid of OpenFOAM's solution"
        print("  " + T.line(r, cj), flush=True)
        rec["like"].append(T.slim(r))
        if N == LIKE_GRIDS[-1]:
            np.savez_compressed(HERE / "gpucfd_profiles.npz", **{k: np.asarray(v) for k, v in r["final"].items()}, N=N, t=g["t_end"])
    STUDY.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "study"
    if what == "study":
        study()
    elif what == "like":
        like()
    else:
        from dev.compare.detonation_tube_h2_air.gpucfd_record import record
        record()
