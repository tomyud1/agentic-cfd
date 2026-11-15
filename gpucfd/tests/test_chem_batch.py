"""Chemistry ladder, rung 3: many cells at once.

    python -m gpucfd.tests.test_chem_batch           (inside WSL, from the project root; about two minutes)
    python -m gpucfd.tests.test_chem_batch tune      the tuning budget: 48 runs that chose the default integrator

4096 cells with start temperatures spread evenly from 1000 to 1400 K (pressure and mixture of case 20), so that
they ignite at different times, to 0.3 ms. Each cell must give exactly what it gives alone: the kernel couples
nothing between cells. Checked bit for bit, in every precision, on one cell run alone, on sixteen cells run as a
small batch in another order, and on the whole batch in reverse order. Five cells are compared with cfdlib.chem.
The wrapper's default setting is then held against the rule that chose it: every cell within a tenth of case 20's
required accuracy (0.1 % in ignition time, 0.01 % in final temperature) of the same kernel at tolerance 1e-10.

Pass rules, expectations and the tuning budget were written before the runs:
dev/compare/h2_air_ignition_box/plan.yaml. The ignition time of a cell is here the first reach of 1600 K (above
every start temperature), read by case 20's rule: a straight line over the step in which it happens.
"""
import json
import sys
from pathlib import Path

import numpy as np

from cfdlib import chem
from gpucfd import chemistry
from gpucfd.chemistry import Chemistry
from gpucfd.tests.test_chem_parcel import TIGHT, given

FAILS = []
REPORT = {}
N, T_LOW, T_HIGH, T_END, LEVEL = 4096, 1000.0, 1400.0, 3e-4, 1600.0
STAGES = {"bs23": 3, "dp45": 6}                      # rate evaluations per sub-step tried (plus one per step); ros34: see rhs_count


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def run_pop(mech, g, moles, T0, precision="double", dt=1e-7, t_end=T_END, constant="pressure", **control):
    """The cells with start temperatures T0 to t_end in steps of dt. Returns the end state per cell, the ignition
    times, the sub-steps, the largest change of the atoms of any cell, and the GPU time."""
    c = Chemistry(mech, len(T0), constant=constant, precision=precision)
    c.control(level=LEVEL, **control)
    c.set(T=T0, p=g["p"], moles=moles)
    a0 = c.get()["sigma"] @ mech.comp
    out = c.advance(dt, int(round(t_end / dt)))
    s, st = c.get(), c.stats()
    setting = dict(method=c.method, rtol=c.rtol, atol=c.atol)
    c.close()
    a1 = s["sigma"] @ mech.comp
    present = a0[0] > 0
    return dict(precision=precision, dt=dt, steps=out["steps"], ms=out["ms"], T=s["T"], sigma=s["sigma"], t_ign=st["t_cross"], substeps=st["substeps"],
                rejected=st["rejected"], status=st["status"], atoms=float(np.max(np.abs(a1[:, present] / a0[:, present] - 1))),
                Y_min=float(s["Y"].min()), setting=setting)


def rhs_count(r, method):
    """Rate evaluations per cell and step, from the counted sub-steps (the implicit method: one plus one per
    reacting species for the Jacobian per accepted sub-step, two more per sub-step tried)."""
    tried = r["substeps"].sum() + r["rejected"].sum()
    cells_steps = len(r["T"]) * r["steps"]
    if method == "ros34":
        return float((9 * r["substeps"].sum() + 2 * tried) / cells_steps)
    return float(1 + STAGES[method] * tried / cells_steps)


def against(r, ref):
    """Largest differences of any cell from the reference run: ignition time (relative), final temperature (relative)."""
    return float(np.nanmax(np.abs(r["t_ign"] / ref["t_ign"] - 1))) if not np.isnan(r["t_ign"]).any() else float("nan"), float(np.max(np.abs(r["T"] / ref["T"] - 1)))


def tune(mech, g, moles, T0):
    """The tuning budget of plan.yaml: 3 methods x 4 relative x 2 absolute tolerances x 2 steps, double precision."""
    rows = []
    for dt in (1e-8, 1e-6):
        ref = run_pop(mech, g, moles, T0, dt=dt, **TIGHT)
        print(f"step {dt:g} s: reference run (Dormand-Prince, 1e-10) {ref['ms'] / 1e3:.1f} s, ignition times {1e6 * ref['t_ign'].min():.1f} to {1e6 * ref['t_ign'].max():.1f} us", flush=True)
        for method in ("bs23", "dp45", "ros34"):
            for rtol in (1e-3, 1e-4, 1e-6, 1e-8):
                for atol in (1e-10, 1e-14):
                    r = run_pop(mech, g, moles, T0, dt=dt, method=method, rtol=rtol, atol=atol)
                    e_t, e_T = against(r, ref)
                    ok = bool(not r["status"].any() and e_t <= 1e-3 and e_T <= 1e-4)
                    rows.append(dict(dt=dt, method=method, rtol=rtol, atol=atol, gpu_s=r["ms"] / 1e3, e_t=e_t, e_T=e_T, within=ok, broke=int((r["status"] > 0).sum()),
                                     substeps_per_step=float(r["substeps"].mean() / r["steps"]), rejected_per_step=float(r["rejected"].mean() / r["steps"]),
                                     rhs_per_step=rhs_count(r, method), atoms=r["atoms"], Y_min=r["Y_min"]))
                    x = rows[-1]
                    print(f"  {method:5s} rtol {rtol:.0e} atol {atol:.0e}: {x['gpu_s']:7.2f} s, {x['substeps_per_step']:8.2f} sub-steps and {x['rhs_per_step']:8.1f} rate "
                          f"evaluations per cell and step; ignition time {e_t:.1e}, final temperature {e_T:.1e}; {'within' if ok else 'OUTSIDE'} the rule"
                          + (f"; {x['broke']} cells broke" if x["broke"] else ""), flush=True)
    chosen = {}
    for dt in (1e-8, 1e-6):
        ok = [x for x in rows if x["dt"] == dt and x["within"]]
        chosen[f"{dt:g}"] = min(ok, key=lambda x: x["gpu_s"]) if ok else None
        c = chosen[f"{dt:g}"]
        print(f"step {dt:g} s: " + (f"chosen {c['method']} rtol {c['rtol']:.0e} atol {c['atol']:.0e} ({c['gpu_s']:.2f} s)" if c else "no setting within the rule"), flush=True)
    out = Path(__file__).parent / "results" / "test_chem_tune.json"
    out.write_text(json.dumps(dict(cells=len(T0), T0=[T_LOW, T_HIGH], t_end=T_END, level=LEVEL, rule="worst cell within 1e-3 in ignition time and 1e-4 in final temperature; least GPU time",
                                   rows=rows, chosen=chosen), indent=1, default=float))
    print(f"record: {out}")


def main():
    g, moles, _, _ = given()
    mech = chem.build()
    T0 = np.linspace(T_LOW, T_HIGH, N)
    if sys.argv[1:] == ["tune"]:
        return tune(mech, g, moles, T0)
    REPORT.update(cells=N, T0=[T_LOW, T_HIGH], t_end=T_END, level=LEVEL, rows=[])

    # ---- every cell equals its single-cell answer, bit for bit ----
    rng = np.random.default_rng(3)
    pick = rng.choice(N, 16, replace=False)
    keys = ("T", "sigma", "t_ign", "substeps", "rejected")
    same = lambda a, b: all(np.array_equal(a[k], b[k], equal_nan=True) for k in keys)                     # noqa: E731
    for precision in ("double", "mixed", "float"):
        full = run_pop(mech, g, moles, T0, precision)
        small = run_pop(mech, g, moles, T0[pick], precision)
        alone = run_pop(mech, g, moles, T0[pick[:1]], precision)
        rev = run_pop(mech, g, moles, T0[::-1].copy(), precision)
        sub = {k: full[k][pick] for k in keys}
        ok = same(small, sub) and same(alone, {k: full[k][pick[:1]] for k in keys}) and same(rev, {k: full[k][::-1] for k in keys})
        healthy = bool(not full["status"].any() and np.all(np.isfinite(full["T"])) and full["T"].min() > 2000)
        REPORT["rows"].append(dict(precision=precision, setting=full["setting"], identical=bool(ok), gpu_s=full["ms"] / 1e3, atoms=full["atoms"], Y_min=full["Y_min"],
                                   t_ign=[float(full["t_ign"].min()), float(full["t_ign"].max())], T_end=[float(full["T"].min()), float(full["T"].max())],
                                   substeps_per_step=float(full["substeps"].mean() / full["steps"])))
        check(f"rung 3: each cell gives bit for bit what it gives alone, in a batch of 16 in another order, and in the reversed batch ({precision})", ok and healthy,
              f"{N} cells to {1e3 * T_END:g} ms in {full['steps']} steps: ignition between {1e6 * full['t_ign'].min():.1f} and {1e6 * full['t_ign'].max():.1f} us, final "
              f"temperature {full['T'].min():.0f} to {full['T'].max():.0f} K, {full['substeps'].mean() / full['steps']:.2f} sub-steps per cell and step, atoms of the "
              f"worst cell {full['atoms']:.1e}, {full['ms'] / 1e3:.1f} s")

    # ---- five cells against cfdlib.chem (tight tolerances, the same reading of the ignition time) ----
    ref = run_pop(mech, g, moles, T0, "double", **TIGHT)
    t_grid = ref["dt"] * np.arange(ref["steps"] + 1)
    worst_t, worst_T, five = 0.0, 0.0, []
    for k in (0, 1024, 2048, 3072, N - 1):
        ind = chem.Parcel(mech, g["p"], float(T0[k]), moles).solve(T_END, rtol=1e-10, atol=1e-20)
        t_ind = chem.crossing_time(t_grid, ind["sol"](t_grid)[-1], LEVEL)
        five.append(dict(T0=float(T0[k]), t_ign=float(ref["t_ign"][k]), t_ign_independent=t_ind, T_end=float(ref["T"][k]), T_end_independent=float(ind["T"][-1])))
        worst_t, worst_T = max(worst_t, abs(ref["t_ign"][k] / t_ind - 1)), max(worst_T, abs(ref["T"][k] - ind["T"][-1]))
    REPORT["against_cfdlib"] = dict(cells=five, worst_t=worst_t, worst_T=worst_T, atoms=ref["atoms"])
    check("rung 3: five cells of the batch against cfdlib.chem (ignition time within 1e-6, final temperature within 1e-3 K)", worst_t <= 1e-6 and worst_T <= 1e-3,
          "start temperatures " + ", ".join(f"{c['T0']:.0f}" for c in five) + " K: ignition at " + ", ".join(f"{1e6 * c['t_ign']:.2f}" for c in five)
          + f" us; largest difference {worst_t:.1e} in ignition time, {worst_T:.1e} K in final temperature; atoms of the worst of all {N} cells {ref['atoms']:.1e}")

    # ---- the default setting against the rule that chose it ----
    ref8 = run_pop(mech, g, moles, T0, "double", dt=1e-8, **TIGHT)
    for precision in dict.fromkeys(("double", chemistry.DEFAULT["precision"])):
        r = run_pop(mech, g, moles, T0, precision, dt=1e-8)
        e_t, e_T = against(r, ref8)
        REPORT[f"default_{precision}"] = dict(setting=r["setting"], e_t=e_t, e_T=e_T, atoms=r["atoms"], Y_min=r["Y_min"], gpu_s=r["ms"] / 1e3,
                                              substeps_per_step=float(r["substeps"].mean() / r["steps"]), rejected_per_step=float(r["rejected"].mean() / r["steps"]))
        check(f"rung 3: the default setting ({r['setting']['method']}, relative tolerance {r['setting']['rtol']:g}, absolute {r['setting']['atol']:g} mol/kg, {precision}) "
              "keeps every cell within a tenth of the required accuracy at steps of 1e-8 s", not r["status"].any() and e_t <= 1e-3 and e_T <= 1e-4,
              f"worst cell: ignition time {100 * e_t:.4f} % (rule 0.1 %), final temperature {100 * e_T:.5f} % (rule 0.01 %); atoms {r['atoms']:.1e}; "
              f"{r['substeps'].mean() / r['steps']:.2f} sub-steps per cell and step; {r['ms'] / 1e3:.1f} s")

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "test_chem_batch.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
