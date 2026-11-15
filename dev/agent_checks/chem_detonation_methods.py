"""A file of the developing agent: which of the chemistry kernel's three methods is cheapest at the state behind a
detonation's shock? The kernel's default (explicit, "bs23") was chosen at 2 atm, where one sub-step per flow step
is enough (gpucfd/CHEMISTRY.md). The review check (chem_kernel_review.py) found about ten sub-steps per step at
28 atm, burnt gas included. This file asks whether another built-in method or a looser tolerance is cheaper there.

    PYTHONPATH=. python dev/agent_checks/chem_detonation_methods.py      (inside WSL; GPU; about ten minutes)

Written before the run.
The run: 250 000 cells, all at the state behind a shock of the Chapman-Jouguet speed in hydrogen and air, burning
at constant volume; 400 steps of 5e-9 s (2 microseconds: the wait, the rise, and burnt gas for most of the time).
Budget: 3 methods x 2 precisions (mixed, double) x 2 relative tolerances (1e-4, 1e-6) = 12 runs, no more.
Accuracy rule (same as the review check, against cfdlib.chem.VolumeParcel): time to 2000 K within 0.1 %, end
temperature within 0.01 %, every cell status 0.
Choice: among the settings that keep the accuracy rule, the least GPU time.
Expected, not required: in double the implicit method ("ros34") is the cheapest; in mixed precision it is not
(its matrix is built from differences of 7-digit rates); the best setting is at most three times cheaper than
the default.

Found (2026-11-15): all 12 settings keep the accuracy rule. Cheapest: mixed "ros34" at tolerance 1e-4, 29 million
cells x steps per second, 1.6 times the default (19 million). In double "ros34" at 1e-4 is cheapest, as expected
(29 s against 44 s for "bs23"). In mixed it is cheapest at 1e-4 but three times worse than the default at 1e-6,
so the expectation about mixed precision was half right. The explicit methods take 9 to 10 sub-steps per step
whatever the tolerance: stability sets their sub-step, not accuracy. The implicit method takes 2 to 3 sub-steps
but each costs 11 rate evaluations, because its matrix is built by differences.
"""
import json
from pathlib import Path

import numpy as np

from cfdlib import chem
from gpucfd.chemistry import DEFAULT, Chemistry

OUT = Path(__file__).with_suffix(".json")
P1, T1, MIX = 101325.0, 300.0, dict(H2=2.0, O2=1.0, N2=3.76)
LEVEL, N, DT, STEPS = 2000.0, 250_000, 5e-9, 400


def main():
    mech = chem.build()
    cj = chem.cj_detonation(mech, P1, T1, MIX)
    vn = chem.shock_frozen(mech, P1, T1, MIX, cj["D"])
    ref = chem.VolumeParcel(mech, vn["p"], vn["T"], MIX).solve(DT * STEPS, T_ign=LEVEL)
    T_ref = float(ref["T"][-1])
    c = Chemistry(mech, 16, constant="volume")                   # start-up of the GPU library, not timed
    c.set(T=vn["T"], p=vn["p"], moles=MIX)
    c.advance(DT, 1)
    c.close()
    rows = []
    for precision in ("mixed", "double"):
        for method in ("bs23", "dp45", "ros34"):
            for rtol in (1e-4, 1e-6):
                c = Chemistry(mech, N, constant="volume", precision=precision, method=method, rtol=rtol, atol=DEFAULT["atol"])
                c.control(level=LEVEL)
                c.set(T=vn["T"], p=vn["p"], moles=MIX)
                out = c.advance(DT, STEPS)
                s, st = c.get(), c.stats()
                c.close()
                row = dict(precision=precision, method=method, rtol=rtol, gpu_s=out["ms"] / 1e3, cell_steps_per_s=N * STEPS / (out["ms"] / 1e3),
                           substeps_per_step=float(st["substeps"].mean() / STEPS), rejected_per_step=float(st["rejected"].mean() / STEPS), status=int(st["status"].max()),
                           d_t=float(st["t_cross"][0] / ref["t_ign"] - 1), d_T=float(s["T"][0] / T_ref - 1))
                row["accurate"] = bool(abs(row["d_t"]) <= 1e-3 and abs(row["d_T"]) <= 1e-4 and row["status"] == 0)
                row["default"] = precision == DEFAULT["precision"] and method == DEFAULT["method"] and rtol == DEFAULT["rtol"]
                rows.append(row)
                print(f"{'ok  ' if row['accurate'] else 'OUT '} {precision:6s} {method:5s} tolerance {rtol:.0e}: {row['gpu_s']:6.1f} s, {row['cell_steps_per_s'] / 1e6:6.1f} million cells x steps "
                      f"per second, {row['substeps_per_step']:5.2f} sub-steps per step ({row['rejected_per_step']:.2f} rejected), time to 2000 K {100 * row['d_t']:+.4f} %, end "
                      f"temperature {100 * row['d_T']:+.5f} %, status {row['status']}{'   <- the default' if row['default'] else ''}", flush=True)
    good = [r for r in rows if r["accurate"]]
    best = min(good, key=lambda r: r["gpu_s"])
    dflt = next(r for r in rows if r["default"])
    print(f"cheapest setting that keeps the accuracy rule: {best['precision']} {best['method']} tolerance {best['rtol']:.0e}, {dflt['gpu_s'] / best['gpu_s']:.2f} times cheaper than the default")
    OUT.write_text(json.dumps(dict(start=dict(p=vn["p"], T=vn["T"]), cells=N, dt=DT, steps=STEPS, reference=dict(t_2000=ref["t_ign"], T_end=T_ref), rows=rows,
                                   best=best, default=dflt, gain=dflt["gpu_s"] / best["gpu_s"]), indent=1, default=float), encoding="utf-8")
    print(f"record: {OUT.name}")


if __name__ == "__main__":
    main()
