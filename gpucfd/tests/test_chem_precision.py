"""Chemistry ladder, rung 4: what single and mixed precision cost in accuracy, and the default.

    python -m gpucfd.tests.test_chem_precision          (inside WSL, from the project root; about five minutes)

With the wrapper's default integrator and tolerances, in single, mixed and double precision:
  1. case 20's parcel in 100000 steps of 1e-8 s: ignition time and final temperature against double, atoms at
     every step, lowest mass fraction, sub-steps;
  2. the 4096 cells of rung 3 to 0.3 ms at steps of 1e-8 s and 1e-6 s: the worst cell against the kernel at
     tolerance 1e-10 in double, and the atoms of the worst cell;
  3. the same cells at steps of 1e-6 s with relative tolerances from 1e-3 to 1e-8: where rates in single precision
     stop helping (the error estimate then sees round-off, not error);
  4. speed of each precision on a million cells of the mixed population.
Mixed: species and energy stored in double, rates computed in single, the progress of each reaction applied in
double. Single: everything in single.

The rule for the default, fixed before the runs (dev/compare/h2_air_ignition_box/plan.yaml): the fastest
precision that keeps the atoms within 1e-10 over case 20 and moves ignition time and final temperature by less
than a tenth of the required accuracy (0.1 % and 0.01 %) from the double-precision answer at the same tolerances.
Judged: the wrapper's default precision is the one this rule gives; double and mixed keep the atoms to 1e-12.
"""
import json
import sys
from pathlib import Path

import numpy as np

from cfdlib import chem
from gpucfd import chemistry
from gpucfd.tests.test_chem_batch import N, T_HIGH, T_LOW, against, run_pop
from gpucfd.tests.test_chem_parcel import TIGHT, given, run_parcel
from gpucfd.tests.test_chem_speed import states, timed

FAILS = []
REPORT = {}
PRECISIONS = ("float", "mixed", "double")


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main():
    g, moles, t_ref, T_ref = given()
    mech = chem.build()
    REPORT.update(default=dict(chemistry.DEFAULT))

    # ---- 1. case 20, one parcel ----
    parcel = {}
    for precision in PRECISIONS:
        r = run_parcel(mech, g, moles, precision=precision)
        n = chem.atoms(r["Y"], mech)
        present = n[0] > 0
        parcel[precision] = dict(t_ign=r["t_ign"], T_end=r["T_end"], atoms=float(np.max(np.abs(n[:, present] / n[0, present] - 1))), Y_min=float(r["Y"].min()),
                                 sumY=float(np.max(np.abs(r["Y"].sum(1) - 1))), substeps=r["substeps"], rejected=r["rejected"], status=r["status"], wall_s=r["wall_s"],
                                 d_ref_t=r["t_ign"] / t_ref - 1 if r["t_ign"] else None, d_ref_T=r["T_end"] / T_ref - 1)
    for precision in PRECISIONS:
        p, d = parcel[precision], parcel["double"]
        p["d_double_t"] = p["t_ign"] / d["t_ign"] - 1 if p["t_ign"] else float("nan")
        p["d_double_T"] = p["T_end"] / d["T_end"] - 1
        print(f"case 20, {precision:6s}: ignition {1e6 * (p['t_ign'] or float('nan')):.4f} us ({p['d_double_t']:+.1e} from double), final {p['T_end']:.4f} K ({p['d_double_T']:+.1e}), "
              f"atoms {p['atoms']:.1e}, lowest mass fraction {p['Y_min']:.1e}, {p['substeps']} sub-steps, {p['rejected']} rejected, status {p['status']}", flush=True)
    REPORT["parcel"] = parcel

    # ---- 2. the 4096 cells at both steps ----
    T0 = np.linspace(T_LOW, T_HIGH, N)
    pop, refs = [], {}
    for dt in (1e-8, 1e-6):
        refs[dt] = run_pop(mech, g, moles, T0, "double", dt=dt, **TIGHT)
        for precision in PRECISIONS:
            r = run_pop(mech, g, moles, T0, precision, dt=dt)
            e_t, e_T = against(r, refs[dt])
            pop.append(dict(dt=dt, precision=precision, e_t=e_t, e_T=e_T, atoms=r["atoms"], Y_min=r["Y_min"], broke=int((r["status"] > 0).sum()),
                            substeps_per_step=float(r["substeps"].mean() / r["steps"]), rejected_per_step=float(r["rejected"].mean() / r["steps"]), gpu_s=r["ms"] / 1e3))
            x = pop[-1]
            print(f"{N} cells, dt {dt:g}, {precision:6s}: worst cell ignition time {x['e_t']:.1e}, final temperature {x['e_T']:.1e}, atoms {x['atoms']:.1e}, lowest mass "
                  f"fraction {x['Y_min']:.1e}, {x['substeps_per_step']:.2f} sub-steps and {x['rejected_per_step']:.3f} rejected per cell and step, {x['gpu_s']:.1f} s"
                  + (f", {x['broke']} cells broke" if x["broke"] else ""), flush=True)
    REPORT["population"] = pop

    # ---- 3. tolerance sweep at the large step ----
    sweep = []
    for precision in PRECISIONS:
        for rtol in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
            r = run_pop(mech, g, moles, T0, precision, dt=1e-6, rtol=rtol, max_substeps=2000)       # a cell that needs more counts as broken
            e_t, e_T = against(r, refs[1e-6])
            sweep.append(dict(precision=precision, rtol=rtol, e_t=e_t, e_T=e_T, atoms=r["atoms"], broke=int((r["status"] > 0).sum()),
                              substeps_per_step=float(r["substeps"].mean() / r["steps"]), rejected_per_step=float(r["rejected"].mean() / r["steps"]), gpu_s=r["ms"] / 1e3))
            x = sweep[-1]
            print(f"dt 1e-6, {precision:6s}, relative tolerance {rtol:.0e}: ignition time {x['e_t']:.1e}, final temperature {x['e_T']:.1e}, {x['substeps_per_step']:.1f} sub-steps, "
                  f"{x['rejected_per_step']:.2f} rejected per cell and step, {x['gpu_s']:.1f} s" + (f", {x['broke']} cells broke" if x["broke"] else ""), flush=True)
    REPORT["tolerance_sweep"] = sweep

    # ---- 4. speed ----
    _, mixed_pop, _ = states(mech, g, moles)
    speed = {}
    for dt in (1e-8, 1e-6):
        for precision in PRECISIONS:
            r = timed(mech, g, mixed_pop, 1000000, precision, dt)
            speed[f"{precision}_{dt:g}"] = r
            print(f"a million cells of the mixed population, dt {dt:g}, {precision:6s}: {r['cell_steps_per_s'] / 1e6:.2f} million cells x steps per second", flush=True)
    REPORT["speed"] = speed

    # ---- the rule ----
    rate = {p: speed[f"{p}_1e-08"]["cell_steps_per_s"] for p in PRECISIONS}
    allowed = [p for p in PRECISIONS if parcel[p]["status"] == 0 and parcel[p]["atoms"] <= 1e-10 and abs(parcel[p]["d_double_t"]) <= 1e-3 and abs(parcel[p]["d_double_T"]) <= 1e-4]
    best = max(allowed, key=lambda p: rate[p])
    REPORT["rule"] = dict(allowed=allowed, fastest=best, cell_steps_per_s=rate)
    check("rung 4: double and mixed precision keep the atoms within 1e-12 (case 20, every step; and the worst of the 4096 cells)",
          all(parcel[p]["atoms"] <= 1e-12 for p in ("double", "mixed")) and all(x["atoms"] <= 1e-12 for x in pop if x["precision"] != "float"),
          "case 20: " + ", ".join(f"{p} {parcel[p]['atoms']:.1e}" for p in PRECISIONS) + "; worst cell of the batch: "
          + ", ".join(f"{x['precision']} {x['atoms']:.1e} (dt {x['dt']:g})" for x in pop))
    check("rung 4: the wrapper's default precision is the one the rule gives", chemistry.DEFAULT["precision"] == best,
          f"allowed by the rule: {', '.join(allowed)}; million cells x steps per second at 1e-8 s: " + ", ".join(f"{p} {rate[p] / 1e6:.1f}" for p in PRECISIONS)
          + f"; fastest allowed: {best}; default: {chemistry.DEFAULT['precision']}")
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "test_chem_precision.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
