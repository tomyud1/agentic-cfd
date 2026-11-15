"""Analysis and report for the ignition of the hydrogen mixture.

  python -m cases.h2_air_ignition_box.analyze
Reads results/h2_air_ignition_box/runs, theory.json and preflight.json, applies the rules written in numerics.yaml
before the production runs, draws the figures and writes results.json and report.html (after the contract check).
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np

from cfdlib import contract, report
from cfdlib.report import LETTERS, lines, load_yaml, m, m_fix, m_pct, m_sig, tex_num
from . import figs, numbers, theory
from .run import OUT
from .sketch import sketch

HERE = Path(__file__).resolve().parent
US = r"\,\mathrm{\mu s}"
REF_NAME = "CHEMKIN II (SENKIN)"


def sc(x, sig=2):
    """A number in scientific notation as inline math (zero as 0)."""
    return m(tex_num(x, sig)) if x else m("0")


def sgn(x, dec=3):
    """A signed relative difference as per cent."""
    return m_pct(x, dec, signed=True)


def mus(t, dec=3, unit=True):
    """A time given in seconds, shown in microseconds."""
    return m(f"{1e6 * t:.{dec}f}" + (US if unit else ""))


def kel(T, dec=2):
    return m(f"{T:.{dec}f}" + r"\,\mathrm{K}")


def main():
    c = numbers.compute()
    problem, numerics, ref, g, th = c["problem"], c["numerics"], c["ref"], c["g"], c["th"]
    runs, data, specs, fam, planned, st, hi = c["runs"], c["data"], c["specs"], c["fam"], c["planned"], c["study"], c["history"]
    acc, ext = numerics["acceptance"], numerics["extraction"]
    pf = json.loads((OUT / "preflight.json").read_text())
    ind, ind_of, tref = th["independent"], th["independent_of"], th["reference"]
    eq, bd = th["equilibrium"]["std"], th["bounds"]["std"]
    t_ref, T_ref = c["t_ref"], c["T_ref"]
    R = [runs[n] for n in fam]
    fine, coarse = R[-1], R[0]
    Nf = fine["steps"]
    all_ran = fam == planned and not c["crashed"]
    main10 = c["main10"]
    by = lambda study: [n for n in runs if study in specs[n]["studies"]]
    tol_names = [n for n in by("tolerance") if n != main10["name"]]
    adj_names = by("adjustable")
    met_names = [n for n in by("method") if n != main10["name"]]
    first_fine = json.loads((OUT / "repeat" / "S80000_first.json").read_text()) if (OUT / "repeat" / "S80000_first.json").exists() else None
    inside = lambda r: r["d_ref_t"] is not None and abs(r["d_ref_t"]) <= acc["req_ignition"] and abs(r["d_ref_T"]) <= acc["req_final"]
    dts = c["dts"]

    # ---- the rules written before the production runs ---------------------------------------------------------
    X = []

    def rule(kind, text, expected, found, met, key=None):
        x = dict(kind=kind, text=text, expected=expected, found=found, met=bool(met))
        if key:
            x["key"] = key
        X.append(x)

    req = {q["key"]: q["text"] for q in problem["required_accuracy"]}
    rule("requirement", req["ignition"], "within " + m_pct(acc["req_ignition"], 0) + f" (smallest step, {Nf} steps)",
         mus(fine["t_ign"]) + " against " + mus(t_ref, 2) + ": " + sgn(fine["d_ref_t"]),
         all_ran and abs(fine["d_ref_t"]) <= acc["req_ignition"], key="ignition")
    rule("requirement", req["final"], "within " + m_pct(acc["req_final"], 1) + f" (smallest step, {Nf} steps)",
         kel(fine["T_end"], 3) + " against " + kel(T_ref, 1) + ": " + sgn(fine["d_ref_T"], 4),
         all_ran and abs(fine["d_ref_T"]) <= acc["req_final"], key="final")
    worst_atoms = max(r["atoms_max"] for r in R)
    rule("requirement", req["conserved"], "change of at most " + m_sig(acc["req_conserved"], 1, pct=True, unit=True) + " after every step of every run of the time-step family",
         "largest change of hydrogen, oxygen or nitrogen atoms over every step of the " + str(len(R)) + " runs: " + sc(worst_atoms, 1) + " of the start value",
         all_ran and worst_atoms <= acc["req_conserved"], key="conserved")
    rule("health", "Every run of the time-step family reaches the end of the millisecond without a solver error", f"{len(planned)} runs finished",
         f"{len(R)} of {len(planned)} finished; log checks: " + ", ".join(r["health"]["status"] for r in R),
         all_ran and all(r["health"]["status"] == "ok" for r in R))
    Tcc = th["bounds"]["of"]["T_cc"]
    rule("health", "The temperature stays between the start and the limit of complete burning, and the pressure stays as given",
         "between " + kel(g["T_0"], 0) + " and " + kel(Tcc, 1) + "; pressure unchanged",
         "lowest " + kel(min(r["T_min"] for r in R), 4) + ", highest " + kel(max(r["T_max"] for r in R), 2) + "; largest change of the pressure "
         + sc(max(r["p_dev"] for r in R), 1), all(r["T_min"] > g["T_0"] - 0.01 and r["T_max"] < Tcc and r["p_dev"] < 1e-12 for r in R))
    rule("health", "No mass fraction is negative and the mass fractions add up to one", "none negative; sum within " + sc(1e-9, 1) + " of one",
         "lowest mass fraction " + sc(min(r["Y_min"] for r in R), 1) + "; sum off by at most " + sc(max(r["sumY_dev"] for r in R), 1),
         all(r["Y_min"] >= -1e-12 and r["sumY_dev"] < 1e-9 for r in R))
    p_t, p_T = st["t_ign"]["p"], st["T_end"]["p"]
    rule("expectation", "The error falls in proportion to the time step (first order): observed order from the three smallest steps",
         f"between {acc['exp_order'][0]} and {acc['exp_order'][1]}, for the ignition time and for the final temperature",
         "ignition time " + m_fix(p_t, 3) + ", final temperature " + m_fix(p_T, 3),
         all_ran and all(acc["exp_order"][0] <= p <= acc["exp_order"][1] for p in (p_t, p_T)))
    rule("expectation", "Smallest step against the independent solution with OpenFOAM's constants: ignition time",
         "within " + sc(acc["exp_finest_vs_independent_t"], 1) + " (relative)", sc(abs(fine["d_of_t"]), 2) + " (OpenFOAM earlier)",
         all_ran and abs(fine["d_of_t"]) <= acc["exp_finest_vs_independent_t"])
    rule("expectation", "Smallest step against the independent solution with OpenFOAM's constants: final temperature",
         "within " + m(tex_num(acc["exp_finest_vs_independent_T"], 1) + r"\,\mathrm{K}"), m(tex_num(abs(fine["d_of_T"]), 2) + r"\,\mathrm{K}"),
         all_ran and abs(fine["d_of_T"]) <= acc["exp_finest_vs_independent_T"])
    rule("expectation", "Ignition time extrapolated to a time step of zero, against the independent solution with OpenFOAM's constants",
         "within " + sc(acc["exp_extrapolated_t"], 1) + " (relative)", sc(abs(st["t_ign"]["extrap_vs_of"]), 2),
         all_ran and abs(st["t_ign"]["extrap_vs_of"]) <= acc["exp_extrapolated_t"])
    rule("expectation", "Final temperature extrapolated to a time step of zero, against the same",
         "within " + m(tex_num(acc["exp_extrapolated_T"], 1) + r"\,\mathrm{K}"), m(tex_num(abs(st["T_end"]["extrap_vs_of"]), 2) + r"\,\mathrm{K}"),
         all_ran and abs(st["T_end"]["extrap_vs_of"]) <= acc["exp_extrapolated_T"])
    rule("expectation", "The steep part of the history: times at which the temperature passes " + ", ".join(hi["steep_levels"][:-1]) + " and " + kel(float(hi["steep_levels"][-1]), 0) + ", against the reference",
         "each within " + mus(acc["exp_history_shift"], 2), "largest difference " + mus(hi["shift_max"], 3) + " (OpenFOAM earlier at every level)",
         all_ran and hi["shift_max"] <= acc["exp_history_shift"])
    rule("expectation", "The flat parts of the history (before the rise and above " + kel(2100, 0) + "): temperature at the reference's printed points",
         "within " + kel(acc["exp_history_flat"], 1), "largest difference " + kel(hi["dT_flat_max"], 3) + f" over {hi['n_flat']} points",
         all_ran and hi["dT_flat_max"] <= acc["exp_history_flat"])
    worst_energy = max(r["energy_dev"] for r in R)
    rule("expectation", "Energy balance: the enthalpy of the parcel, rebuilt from temperature and composition, stays at its start value at every step",
         "within " + sc(acc["exp_energy"], 1) + " of the heat released", "largest change " + sc(worst_energy, 1) + " of the heat released",
         all_ran and worst_energy <= acc["exp_energy"])
    rule("expectation", "The atoms are kept to round-off, far inside the requirement", "change of at most " + sc(acc["exp_atoms"], 1),
         "at most " + sc(worst_atoms, 1), all_ran and worst_atoms <= acc["exp_atoms"])
    var_tab = [r for n, r in runs.items() if n not in fam and r.get("atoms_max") is not None]
    worst_var = max(r["atoms_max"] for r in var_tab)
    rule("expectation", "Every other run (other tolerances, other chemistry solvers, adjustable step) keeps the atoms as required of the family",
         "change of at most " + m_sig(acc["exp_variants_conserved"], 1, pct=True, unit=True), f"largest change over {len(var_tab)} runs: " + sc(worst_var, 1),
         worst_var <= acc["exp_variants_conserved"])
    n_tol_in = sum(inside(runs[n]) for n in tol_names)
    tol_spread = max(abs(runs[n]["t_ign"] / main10["t_ign"] - 1) for n in tol_names)
    rule("expectation", "At a fixed step of " + mus(g["t_end"] / main10["steps"], 1) + " every tolerance of the chemistry solver, the tutorial's loose one "
         "included, stays within the two required accuracies", f"{len(tol_names)} of {len(tol_names)} runs",
         f"{n_tol_in} of {len(tol_names)}; their ignition times differ from the tight run's by at most " + sc(tol_spread, 1) + " (relative)", n_tol_in == len(tol_names))
    tight = [n for n in adj_names if specs[n]["rel"] <= acc["exp_adjustable_tight"]]
    n_tight_in = sum(inside(runs[n]) for n in tight)
    rule("expectation", "Step set by the chemistry solver (the tutorial's way): runs with a relative tolerance of " + sc(acc["exp_adjustable_tight"], 1)
         + " or tighter stay within the two required accuracies", f"{len(tight)} of {len(tight)} runs",
         f"{n_tight_in} of {len(tight)}: ignition time " + ", ".join(sgn(runs[n]["d_ref_t"]) for n in tight) + " from the reference", n_tight_in == len(tight))
    shipped = runs[next(n for n in adj_names if specs[n]["rel"] == 0.1)]
    ds = data[shipped["name"]]
    k = int(np.argmax(ds["T"] >= g["T_ign"]))
    rule("expectation", f"The tutorial as shipped (step set by the chemistry solver, relative tolerance 0.1, {shipped['steps']} steps) misses the required "
         + m_pct(acc["req_ignition"], 0) + " on the ignition time: predicted because its steps would be too far apart to read that time",
         "more than " + m_pct(acc["req_ignition"], 0) + " from the reference",
         sgn(shipped["d_ref_t"]) + ": inside the requirement. The prediction was wrong: the solver places its steps closest together where the temperature "
         "rises (" + mus(ds["t"][k] - ds["t"][k - 1], 2) + " apart around the ignition temperature)", abs(shipped["d_ref_t"]) > acc["req_ignition"])
    ode3 = [n for n in met_names if specs[n]["chem"] == "ode"]
    worst_ode = max(abs(runs[n]["t_ign"] / main10["t_ign"] - 1) for n in ode3)
    rule("expectation", "Three other chemistry solvers with error control (" + ", ".join(specs[n]["ode"] for n in ode3) + ") at the same step and tolerances give "
         "the ignition time of the main one", "within " + sc(acc["exp_methods_agree"], 1) + " (relative)", "within " + sc(worst_ode, 1),
         len(ode3) == 3 and worst_ode <= acc["exp_methods_agree"])
    eul = runs[next(n for n in met_names if specs[n]["chem"] == "EulerImplicit")]
    d_eul = eul["t_ign"] / main10["t_ign"] - 1
    rule("expectation", "The chemistry solver without error control (EulerImplicit) is further off, but inside the required " + m_pct(acc["req_ignition"], 0),
         "further from the main solver than " + sc(acc["exp_methods_agree"], 1) + "; within " + m_pct(acc["req_ignition"], 0) + " of the reference",
         sgn(d_eul, 2) + " from the main solver, " + sgn(eul["d_ref_t"], 2) + " from the reference",
         abs(d_eul) > acc["exp_methods_agree"] and abs(eul["d_ref_t"]) <= acc["req_ignition"])
    rule("expectation", "Solver time of the run with the smallest step", "at most " + f"{acc['exp_finest_seconds']} s",
         m_fix(fine["wall_time"], 1) + " s" + (" (" + m_fix(first_fine["wall_time"], 1) + " s the first time, while another job used the machine)" if first_fine else ""),
         all_ran and max(fine["wall_time"], first_fine["wall_time"] if first_fine else 0) <= acc["exp_finest_seconds"])
    verdict = report.verdict_from(X)
    missed = [x for x in X if not x["met"]]

    # ---- numbers used in several places ------------------------------------------------------------------------
    sp = c["species"]
    floor_us = runs[next(n for n in tol_names if specs[n]["rel"] == 0.1 and specs[n]["abs"] == 1e-12)]["us_per_step"]
    loose = runs[next(n for n in tol_names if specs[n]["rel"] == 0.1 and specs[n]["abs"] == 1e-12)]
    peak_us = float(np.max(data[main10["name"]]["cost_us"]))
    ign_s = [r["phase_s"][1] for r in R]
    rkf = runs[next(n for n in met_names if specs[n]["ode"] == "RKF45")]
    notable = c["notable"]
    ind_vs_ref = ind["t_ign"] / t_ref - 1
    of_vs_ref = ind_of["t_ign"] / t_ref - 1
    const_t = th["conv_effect"]["all"]["dt_ign_rel"]
    const_T = th["conv_effect"]["all"]["dT_end"]
    ex_t, ex_T = st["t_ign"]["extrap"], st["T_end"]["extrap"]

    wanted = [dict(letter=LETTERS[i], key=q["key"], text=q["text"]) for i, q in enumerate(problem["quantities_of_interest"])]
    L = {w["key"]: w["letter"] for w in wanted}

    headline = (("Completed, with one difference from what was expected. " if verdict == "WARN" else "Completed as expected. " if verdict == "PASS" else "The experiment failed. ")
                + "OpenFOAM ignites the mixture at " + mus(fine["t_ign"]) + ", " + m_pct(fine["d_ref_t"], 3, size=True) + " before the reference (required: within "
                + m_pct(acc["req_ignition"], 0) + "), and ends at " + kel(fine["T_end"], 2) + ", " + m_pct(fine["d_ref_T"], 4, size=True)
                + " above it (required: " + m_pct(acc["req_final"], 1) + "). The atoms of each element are kept to round-off."
                + (" The one difference: I predicted that the tutorial as shipped would miss the ignition time by more than " + m_pct(acc["req_ignition"], 0)
                   + "; it is off by " + m_pct(shipped["d_ref_t"], 2, size=True) + "." if missed and len(missed) == 1 else ""))

    answers = [
        dict(letter=L["history"], label="Temperature against time",
             theory=f"{hi['n_points']} printed points, to " + kel(ref["uncertainty"]["temperature_resolution_K"], 1) + " (Figure {fig:history})",
             sim="The same curve: flat for " + mus(fine["levels"]["1100"], 0) + ", a rise of " + kel(1000, 0) + " within about " + mus(fine["levels"]["2000"] - fine["levels"]["1100"], 0)
                 + ", then a slow approach to the final temperature. The rise comes " + mus(min(abs(hi["shift"][x]) for x in hi["steep_levels"]), 2, unit=False) + " to "
                 + mus(hi["shift_max"], 2) + " earlier than the reference's",
             diff=lines("rise: " + mus(-hi["shift_max"], 2), "flat parts: " + kel(hi["dT_flat_max"], 2))),
        dict(letter=L["ignition"], label="Ignition time (first reach of " + kel(g["T_ign"], 0) + ")", theory=mus(t_ref, 2), sim=mus(fine["t_ign"]),
             diff=sgn(fine["d_ref_t"])),
        dict(letter=L["final"], label="Temperature at the end", theory=kel(T_ref, 1), sim=kel(fine["T_end"], 3),
             diff=lines(sgn(fine["d_ref_T"], 4), m(f"{fine['T_end'] - T_ref:+.3f}" + r"\,\mathrm{K}"))),
        dict(letter=L["species"], label="Hydrogen, oxygen, water, OH, H and O in time",
             theory="The reference file holds no species. The independent solution (Theory) is the comparison",
             sim="Hydrogen falls from " + m_fix(sp["H2"]["start"], 2) + " to " + m_fix(sp["H2"]["end"], 3) + " mol/kg and oxygen from " + m_fix(sp["O2"]["start"], 2)
                 + " to " + m_fix(sp["O2"]["end"], 2) + " (half is left: the mixture is lean); water rises to " + m_fix(sp["H2O"]["end"], 2)
                 + ". H, O and OH grow from nothing by a factor of ten every " + mus(np.log(10) / ind["growth"], 0) + ", peak during the rise (H at "
                 + m_fix(sp["H"]["peak"], 2) + " mol/kg) and then fall slowly (Figure {fig:species})",
             diff=lines("within " + m_pct(c["species_dmax"], 2), "of each peak")),
        dict(letter=L["conserved"], label="Atoms and energy balance", theory="No change (exact)",
             sim="Yes, both. Atoms of H, O and N: change of at most " + sc(worst_atoms, 1) + ". Enthalpy of the parcel: change of at most " + sc(worst_energy, 1)
                 + " of the heat released. Every step of all " + str(len(R)) + " runs (Figure {fig:conservation})",
             diff=lines("atoms " + sc(worst_atoms, 1), "energy " + sc(worst_energy, 1))),
        dict(letter=L["cost"], label="Effect of the settings, and cost", theory="–",
             sim="Time step: the error is in proportion to it (" + sgn(coarse["d_of_t"]) + " on the ignition time at " + mus(dts[0], 1) + ", " + sgn(fine["d_of_t"], 4)
                 + " at " + mus(dts[-1], 4) + "). Tolerance of the chemistry solver: no effect at a fixed small step (" + sc(tol_spread, 1)
                 + "); when it also sets the step, as in the tutorial, it decides the accuracy (" + sgn(shipped["d_ref_t"], 2) + " at 0.1). Cost: about "
                 + m_fix(floor_us, 0) + " microseconds of one core per step, and up to " + m_fix(peak_us / 1e3, 0) + " milliseconds per step during the rise with tight "
                 "tolerances (Figure {fig:cost})",
             diff="–"),
    ]

    findings = [
        "<b>The answer.</b> Ignition at " + mus(fine["t_ign"]) + " (reference " + mus(t_ref, 2) + ", " + sgn(fine["d_ref_t"]) + "); temperature at the end "
        + kel(fine["T_end"], 2) + " (reference " + kel(T_ref, 1) + "). Both are far inside the required accuracy.",
        "<b>Three programs, one answer.</b> OpenFOAM, extrapolated to a time step of zero, and the independent program written for this report agree on the "
        "ignition time to " + sc(abs(st["t_ign"]["extrap_vs_of"]), 1) + " (relative) and on the final temperature to " + m(tex_num(abs(st["T_end"]["extrap_vs_of"]), 1) + r"\,\mathrm{K}")
        + ", when both use OpenFOAM's constants. So OpenFOAM integrates this reaction scheme correctly.",
        "<b>Most of the small difference from the reference belongs to the reference.</b> The independent program with CHEMKIN's own constants ignites "
        + m_pct(ind_vs_ref, 3, size=True) + " before the reference, and its answer does not move when its tolerances are tightened. The reference states no "
        "tolerances. OpenFOAM's different constants add " + m_pct(const_t, 4, size=True) + " on the ignition time and " + m(f"{const_T:+.2f}" + r"\,\mathrm{K}")
        + " on the final temperature.",
        "<b>The time step matters, to first order.</b> Inside a step OpenFOAM lets the gas react without expanding, and corrects the density only after the step. "
        "The ignition time is early in proportion to the step (observed order " + m_fix(p_t, 2) + ", Figure {fig:convergence}): halve the step, halve the error.",
        "<b>Atoms and energy are kept to round-off</b> at every step, as the method is built to do. The check is made outside OpenFOAM, from the printed "
        "temperature and composition.",
        "<b>What it costs.</b> A step costs about " + m_fix(floor_us, 0) + " microseconds, except during the rise, where a tight tolerance costs " + m_fix(float(np.mean(ign_s)), 1)
        + " s in total whatever the step. The explicit method RKF45 gave the same answer as the tutorial's implicit one in "
        + m_fix(rkf["wall_time"] / main10["wall_time"], 2) + " of the time.",
    ]
    limitations = [
        "One start state only: " + kel(g["T_0"], 0) + ", 2 atm, one lean mixture. Other temperatures, pressures and mixtures were not run.",
        "The reference states no tolerances and prints five digits. Two independent calculations sit " + m_pct(ind_vs_ref, 2, size=True) + " to "
        + m_pct(fine["d_ref_t"], 2, size=True) + " from it, so this comparison cannot show an agreement better than that.",
        "The reaction scheme is taken as given. Whether it describes real hydrogen is not tested here: there is no experiment in this problem.",
        "A parcel at constant pressure with no flow. A detonation burns nearer to constant volume and behind a shock; that is a later step.",
        "The independent solution is this agent's own code. It shares the two data files with OpenFOAM but none of its source.",
        "OpenFOAM's constants were left as they are (the problem says to use the files unchanged), so a difference of " + m_pct(const_t, 3, size=True)
        + " in ignition time and " + m_fix(const_T, 2) + " K in final temperature from a CHEMKIN-exact reading remains.",
        "Costs are for one cell on one processor core with a table row written at every step. Run times repeat only roughly: the finest run took "
        + (m_fix(first_fine["wall_time"], 1) + " s the first time and " if first_fine else "") + m_fix(fine["wall_time"], 1) + " s when repeated alone.",
        "Solved with OpenFOAM only. The GPU solver gpucfd has no chemistry, so there is no second tool to compare speed with.",
    ]

    # ---- problem, assumptions ------------------------------------------------------------------------------------
    worst_cp = max(th["data_jumps"], key=lambda j: j["cp"])
    assumptions = [
        dict(name="Well mixed: one temperature and one composition, no flow", check="Given. The simulation has a single cell, so nothing can vary in space.", status="imposed"),
        dict(name="Constant pressure", check="Given. The solver's pressure changes by " + sc(max(r["p_dev"] for r in runs.values()), 1) + " (relative) at most, in every run.", status="ok"),
        dict(name="No heat exchange", check="Given. The solver has no heat-exchange term. Checked through the energy balance: the enthalpy of the parcel changes by at most "
             + sc(worst_energy, 1) + " of the heat released.", status="ok"),
        dict(name="Ideal gases", check="Given. Density, pressure and temperature printed by the solver satisfy the ideal-gas law to " + sc(max(r["state_dev"] for r in R), 1)
             + ". At 2 atm and 1000 K and above, real-gas effects are far below the required accuracy.", status="ok"),
        dict(name="Heat capacities from the data file", check="Given (<code>therm.dat</code>: two polynomials per species, joined at 1000 K, which is the start temperature). "
             "The largest jump of a heat capacity at the joint is " + sc(worst_cp["cp"], 1) + f" (relative, {worst_cp['species']}): harmless.", status="ok"),
        dict(name="The reaction scheme as given", check=f"Given. OpenFOAM's reading of the file was compared with an independent reading: {pf['scheme']['reactions']} reactions, "
             "largest difference of a rate parameter " + sc(pf["scheme"]["worst"], 1) + " (relative).", status="ok"),
        dict(name="Pure start mixture: hydrogen, oxygen and nitrogen only", check="Given. No reactive species at the start; the first reactions between hydrogen and oxygen "
             "make them. Argon is in the scheme and absent from the mixture (mass fraction zero throughout).", status="imposed"),
        dict(name="Constants of the reference program", check="Not met exactly: OpenFOAM reads the scheme with two constants that differ from CHEMKIN's. Effect computed in Theory: "
             + m_pct(const_t, 4, size=True) + " on the ignition time, " + m_fix(const_T, 2) + " K on the final temperature, both far inside the required accuracy.", status="ok"),
    ]

    # ---- pre-flight ------------------------------------------------------------------------------------------------
    dec = load_yaml(HERE / "decisions.yaml")
    dec_problems = contract.check_decisions(dec) + contract.uncovered_settings(dec, fine["case_dir"])
    sh = pf["short"]
    pre3 = [runs[f"S{n}"] for n in numerics["preflight"]["full_steps"]]
    pre_checks = list(sh[-1]["checks"]) + [
        dict(name="OpenFOAM reads the reaction scheme as the independent program does", status="ok" if pf["scheme"]["ok"] else "fail",
             value=f"{pf['scheme']['reactions']} reactions; largest difference of a rate parameter " + sc(pf["scheme"]["worst"], 1) + ", of a molar mass " + sc(pf["scheme"]["w_dev"], 1),
             meaning="The scheme was converted with OpenFOAM's own tool and compared reaction by reaction: same species, same kinds of reaction, same numbers."),
        dict(name="Every setting belongs to a documented decision", status="ok" if not dec_problems else "fail",
             value="all covered" if not dec_problems else f"{len(dec_problems)} missing", meaning="Checked on the case of the smallest step."),
        dict(name="First-order behaviour on the three largest steps, run in full", status="ok",
             value="ignition time " + ", ".join(m_pct(r["d_of_t"], 4, size=True) for r in pre3) + " before the independent solution on " + ", ".join(str(r["steps"]) for r in pre3) + " steps",
             meaning="The difference halves with the step. This sized the rules for the outcome."),
        dict(name="Run time projected from the short tests", status="warn",
             value="projected " + m_fix(sh[0]["projected"], 2) + " s and " + m_fix(sh[-1]["projected"], 1) + " s for the largest and the smallest step; the full runs took "
                   + m_fix(coarse["wall_time"], 1) + " s and " + m_fix(fine["wall_time"], 1) + " s",
             meaning="The first 5 % of the time holds none of the expensive part: the cost sits in the rise, which the short test never reaches. The three "
                     "full runs of the pre-flight showed it, and the budget was sized from them."),
    ]
    preflight = dict(
        purpose="Before the production runs: does OpenFOAM read the scheme as intended and start from the right state, do the output files hold what the "
                "analysis needs, and how large is the effect of the time step?",
        runs=[
            dict(name="Reading of the reaction scheme", setup="OpenFOAM's converter on the two data files, compared with the independent reading", outcome="identical"),
            dict(name=f"{sh[0]['name'][1:]} and {sh[-1]['name'][1:]} steps, first 5 % of the time", setup=f"{sh[0]['n_steps']} and {sh[-1]['n_steps']} steps, main settings",
                 outcome=f"{sh[0]['status'].upper()} and {sh[-1]['status'].upper()}: every check below"),
            dict(name=", ".join(str(n) for n in numerics["preflight"]["full_steps"]) + " steps, in full", setup="main settings, a few seconds each",
                 outcome="healthy; first order in the time step; their numbers sized the rules for the outcome"),
        ],
        checks=pre_checks,
        conclusion="The setup is right: same scheme, same start state, atoms and energy kept, one output row per step. The time step has a first-order effect that "
                   "the three full runs measured, so the production family was kept at six levels. The run-time projection of the short tests was wrong for "
                   "this kind of problem and was replaced by the times of the full runs.")

    # ---- simulation --------------------------------------------------------------------------------------------
    model = [
        dict(aspect="Gas parcel", real="A well-mixed parcel, the same everywhere", model="One cell (<code>chemFoam</code> builds it itself)"),
        dict(aspect="Chemistry", real="27 reactions between 9 species, rates from the given file", model="The same file, read by OpenFOAM's CHEMKIN reader with OpenFOAM's constants"),
        dict(aspect="Pressure", real="Held constant; the parcel expands", model="Pressure fixed; the density is updated from temperature and composition after each time step, not within it"),
        dict(aspect="Energy", real="No heat exchange: the enthalpy stays constant", model="The enthalpy is stored without its chemical part and the heat of reaction is added each step"),
        dict(aspect="Time", real=r"\(0 \le t \le t_{\mathrm{end}}\), continuous", model=f"{coarse['steps']} to {Nf} equal steps; inside each, the chemistry solver takes its own smaller steps to a set tolerance"),
    ]
    meshes = dict(
        intro="This problem has no mesh: its only discretisation is the time step. The family halves the step five times. All runs use the same chemistry "
              "solver (seulex) with tight tolerances (relative " + sc(numerics["main"]["rel_tol"], 1) + "), so that the family shows the effect of the step alone.",
        head=["Steps", "Time step (µs)", "Steps during the rise (" + m("1010") + " to " + kel(2000, 0) + ")", "Solver time (s)", "Per step (µs)", "Of which during the rise (s)"],
        rows=[[r["steps"], mus(r["dt"], 4, unit=False), r["phase_steps"][1], m_fix(r["wall_time"], 2), m_fix(r["us_per_step"], 0), m_fix(r["phase_s"][1], 2)] for r in R])
    results_tab = dict(
        head=["Steps", "Ignition time (µs)", "Temperature at the end (K)", "Fastest rise (million K/s)", "Time of the fastest rise (µs)", "Reaches " + kel(2000, 0) + " at (µs)"],
        rows=[[r["steps"], mus(r["t_ign"], 4, unit=False), m_fix(r["T_end"], 4), m_fix(r["rise_rate"] / 1e6, 1), mus(r["rise_t"], 2, unit=False), mus(r["levels"]["2000"], 3, unit=False)] for r in R]
             + [["extrapolated to a step of zero", mus(ex_t, 4, unit=False), m_fix(ex_T, 4), "–", "–", "–"],
                ["independent solution, OpenFOAM's constants", mus(ind_of["t_ign"], 4, unit=False), m_fix(ind_of["T_end"], 4), "–", "–", mus(ind_of["levels"]["2000"], 3, unit=False)],
                ["independent solution, CHEMKIN's constants", mus(ind["t_ign"], 4, unit=False), m_fix(ind["T_end"], 4), m_fix(ind["rise"]["rate"] / 1e6, 1), mus(ind["rise"]["t"], 2, unit=False),
                 mus(ind["levels"]["2000"], 3, unit=False)],
                [f"reference ({REF_NAME})", mus(t_ref, 2, unit=False), m_fix(T_ref, 1), "–", "–", mus(tref["levels"]["2000"], 3, unit=False)]])
    health = dict(
        head=["Steps", "Log checks", "Lowest and highest temperature (K)", "Change of the pressure", "Atoms: largest change", "Enthalpy: largest change", "Mass fractions: sum minus one",
              "Lowest mass fraction"],
        rows=[[r["steps"], r["health"]["status"].upper(), m_fix(r["T_min"], 3) + " to " + m_fix(r["T_max"], 2), sc(r["p_dev"], 1), sc(r["atoms_max"], 1), sc(r["energy_dev"], 1),
               sc(r["sumY_dev"], 1), sc(r["Y_min"], 1)] for r in R],
        notes=[
            "A parcel has no residuals to watch. The evidence that each step is right is physical: the atoms of each element and the enthalpy of the parcel must "
            "not change, and they are checked after every step (Figure {fig:conservation}). The enthalpy is rebuilt outside OpenFOAM from the printed temperature and "
            "composition, in units of the heat released by complete burning.",
            "The largest change of the enthalpy, " + sc(worst_energy, 1) + ", appears in the first steps and nowhere else. It is the accuracy to which OpenFOAM finds the "
            "temperature from the enthalpy (its default tolerance is one ten-thousandth of a kelvin), not a loss of energy.",
            "The run stops at the given time. The gas is then still " + m_fix(th["equilibrium"]["of"]["T"] - fine["T_end"], 2) + " K below its final (equilibrium) temperature; reaching "
            "it takes about a second and was not asked.",
            "A smaller step still would cost little and gain nothing that matters: the smallest step is already " + m_pct(fine["d_of_t"], 4, size=True)
            + " from its own limit, while the reference fixes the answer only to about " + m_pct(ind_vs_ref, 2, size=True) + ".",
        ])
    tol_tab = dict(
        title=f"Tolerances of the chemistry solver at a fixed step ({main10['steps']} steps, seulex)",
        head=["Relative tolerance", "Absolute tolerance (kmol/m³)", "Ignition time (µs)", "From the tight run (relative)", "From the reference", "Temperature at the end (K)",
              "Solver time (s)", "Per step (µs)"],
        rows=[[sc(specs[n]["rel"], 1), sc(specs[n]["abs"], 1), mus(runs[n]["t_ign"], 4, unit=False), sc(runs[n]["t_ign"] / main10["t_ign"] - 1, 1) if n != main10["name"] else "the tight run",
               sgn(runs[n]["d_ref_t"]), m_fix(runs[n]["T_end"], 4), m_fix(runs[n]["wall_time"], 2), m_fix(runs[n]["us_per_step"], 0)]
              for n in sorted(tol_names + [main10["name"]], key=lambda n: (-specs[n]["rel"], -specs[n]["abs"]))],
        note="With the step fixed at " + mus(main10["dt"], 1) + " the tolerance hardly matters: so little happens within one step that even the loosest setting "
             "integrates it well. The tight absolute tolerance is what costs time: it makes the solver work hard during the rise.")
    adj_tab = dict(
        title="Step set by the chemistry solver (the tutorial's way)",
        head=["Relative tolerance", "Absolute tolerance (kmol/m³)", "Steps taken", "Smallest step (µs)", "Largest step (µs)", "Ignition time (µs)", "From the reference",
              "Temperature at the end: from the reference", "Within both required accuracies", "Solver time (s)"],
        rows=[[sc(specs[n]["rel"], 1), sc(specs[n]["abs"], 1), runs[n]["steps"], mus(runs[n]["dt_min"], 4, unit=False), mus(runs[n]["dt_max"], 1, unit=False),
               mus(runs[n]["t_ign"], 3, unit=False), sgn(runs[n]["d_ref_t"]), sgn(runs[n]["d_ref_T"], 4), "yes" if inside(runs[n]) else "no", m_fix(runs[n]["wall_time"], 2)]
              for n in sorted(adj_names, key=lambda n: -specs[n]["rel"])],
        note="The first row is the tutorial as shipped. Here each time step is one sub-step of the chemistry solver, so the tolerance sets the step, and through "
             "the step the accuracy. A run time of 0.4 s is the smallest that the timer of the run script can show: these short runs spend it on start-up.")
    met_tab = dict(
        title=f"Other chemistry solvers at the same step and tolerances ({main10['steps']} steps)",
        head=["Chemistry solver", "Kind", "Ignition time (µs)", "From seulex (relative)", "From the reference", "Temperature at the end (K)", "Solver time (s)", "Per step (µs)"],
        rows=[[("seulex (main)" if n == main10["name"] else specs[n]["ode"] if specs[n]["chem"] == "ode" else "EulerImplicit"),
               {"seulex": "implicit, error control", "Rosenbrock34": "implicit, error control", "rodas34": "implicit, error control", "RKF45": "explicit, error control"}.get(
                   specs[n]["ode"] if specs[n]["chem"] == "ode" else "", "implicit, no error control"),
               mus(runs[n]["t_ign"], 4, unit=False), sc(runs[n]["t_ign"] / main10["t_ign"] - 1, 1) if n != main10["name"] else "–", sgn(runs[n]["d_ref_t"]), m_fix(runs[n]["T_end"], 4),
               m_fix(runs[n]["wall_time"], 2), m_fix(runs[n]["us_per_step"], 0)] for n in [main10["name"]] + met_names],
        note="The three other solvers with error control reproduce seulex to nine digits. EulerImplicit takes sub-steps limited by a chemical time scale without "
             "checking their error; it ignites " + m_pct(d_eul, 2, size=True) + " early, still inside the required accuracy.")
    sp_order = [s for s in ("H2", "O2", "H2O", "OH", "H", "O", "HO2", "H2O2") if s in sp]
    sp_tex = {"H2": r"\mathrm{H_2}", "O2": r"\mathrm{O_2}", "H2O": r"\mathrm{H_2O}", "OH": r"\mathrm{OH}", "H": r"\mathrm{H}", "O": r"\mathrm{O}", "HO2": r"\mathrm{HO_2}",
              "H2O2": r"\mathrm{H_2O_2}"}
    sp_tab = dict(
        title=f"Species in moles per kilogram of gas ({Nf} steps)",
        head=["Species", "Start", "Highest value", "Reached at (µs)", "At the end", "Independent solution, at the end", "At equilibrium (never reached in 1 ms)",
              "Largest difference from the independent solution (of the highest value)"],
        rows=[[m(sp_tex[s]), sc(sp[s]["start"], 4) if sp[s]["start"] else m("0"), sc(sp[s]["peak"], 4), mus(sp[s]["t_peak"], 1, unit=False) if sp[s]["t_peak"] > 0 else "start",
               sc(sp[s]["end"], 4), sc(sp[s]["end_ind"], 4), sc(sp[s]["end_eq"], 4), m_pct(sp[s]["dmax"], 3)] for s in sp_order],
        note="The independent solution uses OpenFOAM's constants here. The largest differences occur within the rise, where a shift of a hundredth of a microsecond "
             "already changes a steep curve; they shrink with the time step like the ignition time does. The first six species are the ones asked for.")

    # ---- comparison --------------------------------------------------------------------------------------------
    defs = [
        dict(name="Difference of the ignition time", latex=r"\frac{t_{\mathrm{ign}}}{t_{\mathrm{ign}}^{\mathrm{ref}}} - 1",
             meaning="Negative: ignition before the reference. Read from both histories by the same rule (first step at or above the ignition temperature, straight line to the step before)."),
        dict(name="Difference of the final temperature", latex=r"\frac{T(t_{\mathrm{end}})}{T^{\mathrm{ref}}(t_{\mathrm{end}})} - 1", meaning="Positive: hotter than the reference."),
        dict(name="Observed order", latex=r"p = \log_2\frac{e(2\,\Delta t)}{e(\Delta t)}",
             meaning="How fast the difference \\(e\\) from the independent solution falls when the step is halved. 1: in proportion to the step."),
        dict(name="Value at a step of zero", latex=r"f_0 = f_3 + \frac{f_3 - f_2}{2^{p} - 1}",
             meaning="Richardson extrapolation from the three smallest steps (\\(f_3\\): the smallest), with the order observed on them."),
    ]
    o_t, o_T = st["t_ign"]["orders"], st["T_end"]["orders"]
    comp_rows = [[r["steps"], mus(r["dt"], 4, unit=False), sgn(r["d_ref_t"], 4), sgn(r["d_ref_T"], 5), sgn(r["d_of_t"], 5), m(f"{1e3 * r['d_of_T']:+.3f}"),
                  m_fix(o_t[i - 1], 3) if i else "–", m_fix(o_T[i - 1], 3) if i else "–"] for i, r in enumerate(R)]
    comp_rows += [
        ["step of zero (extrapolated)", m("0"), sgn(st["t_ign"]["extrap_vs_ref"], 4), sgn(st["T_end"]["extrap_vs_ref"], 5), sgn(st["t_ign"]["extrap_vs_of"], 5),
         m(f"{1e3 * st['T_end']['extrap_vs_of']:+.3f}"), "–", "–"],
        ["independent solution, OpenFOAM's constants", "–", sgn(of_vs_ref, 4), sgn(ind_of["T_end"] / T_ref - 1, 5), "–", "–", "–", "–"],
        ["independent solution, CHEMKIN's constants", "–", sgn(ind_vs_ref, 4), sgn(ind["T_end"] / T_ref - 1, 5), sgn(ind["t_ign"] / ind_of["t_ign"] - 1, 5),
         m(f"{1e3 * (ind['T_end'] - ind_of['T_end']):+.1f}"), "–", "–"],
    ]
    levels = [str(int(x)) for x in ext["levels"]]
    sh_lo = min(abs(hi["shift"][x]) for x in hi["steep_levels"])
    sh_ind = max(abs(hi["shift_ind"][x]) for x in hi["steep_levels"])
    ref_behind_ind = float(np.mean([-tref["shift"][x] for x in hi["steep_levels"]]))      # reference minus independent (CHEMKIN's constants)
    ref_behind_of = float(np.mean([-hi["shift"][x] for x in hi["steep_levels"]]))
    lvl_tab = dict(
        title="Times at which the temperature passes a level (µs)",
        head=["Level (K)", f"Reference ({REF_NAME})", f"OpenFOAM, {Nf} steps", "Independent, OpenFOAM's constants", "Independent, CHEMKIN's constants", "OpenFOAM minus reference",
              "OpenFOAM minus independent (same constants)"],
        rows=[[x, mus(tref["levels"][x], 3, unit=False), mus(fine["levels"][x], 3, unit=False), mus(ind_of["levels"][x], 3, unit=False), mus(ind["levels"][x], 3, unit=False),
               m(f"{1e6 * hi['shift'][x]:+.3f}"), m(f"{1e6 * hi['shift_ind'][x]:+.4f}")] for x in levels],
        note="Up to " + kel(2000, 0) + " the whole rise of OpenFOAM is the reference's rise moved earlier by " + mus(sh_lo, 2, unit=False) + " to " + mus(hi["shift_max"], 2)
             + ", and OpenFOAM is within " + mus(sh_ind, 3) + " of the independent solution with the same constants (earlier, and more so as the rise goes on: the "
             "time-step error adds up). At " + kel(2200, 0) + " the curve is almost flat (the reference prints 0.1 K and its points are "
             + mus(ref["uncertainty"]["time_between_printed_points_s"]["largest"], 1) + " apart there), so a crossing time is poorly fixed: that level is shown, not judged. The reference file also lists a fastest rise of "
             + m_fix(tref["fastest_rise"]["rate_K_per_s"] / 1e6, 0) + " million K/s at " + mus(tref["fastest_rise"]["time_s"], 2) + ", against " + m_fix(fine["rise_rate"] / 1e6, 1)
             + " at " + mus(fine["rise_t"], 2) + " here. That number is a slope between two neighbouring printed points, which lie " + mus(ref["uncertainty"]["time_between_printed_points_s"]["smallest"], 2, unit=False)
             + " to " + mus(ref["uncertainty"]["time_between_printed_points_s"]["at_ignition"], 2) + " apart while their times are printed to " + mus(1e-8, 2)
             + ": it can be off by a quarter and more, so it is not compared.")
    ce = th["conv_effect"]
    cv = th["conventions"]
    const_tab = dict(
        title="What the constants of the two programs are worth (independent solution, one constant changed at a time)",
        head=["Constant", "CHEMKIN (reference)", "OpenFOAM", "Effect on the ignition time", "Effect on the final temperature (K)"],
        rows=[["Gas constant used to convert activation energies, cal/(mol K)", m_fix(cv["std"]["R_cal"], 6), m_fix(cv["of"]["R_cal"], 6), sgn(ce["R_cal"]["dt_ign_rel"], 5),
               m(f"{ce['R_cal']['dT_end']:+.4f}")],
              ["Pressure for which the thermodynamic data are tabulated, Pa", m_fix(cv["std"]["p_std"], 0), m_fix(cv["of"]["p_std"], 0), sgn(ce["p_std"]["dt_ign_rel"], 5),
               m(f"{ce['p_std']['dT_end']:+.4f}")],
              ["Gas constant, J/(mol K)", m_fix(cv["std"]["R"], 6), m_fix(cv["of"]["R"], 6), sgn(ce["R"]["dt_ign_rel"], 5), m(f"{ce['R']['dT_end']:+.4f}")],
              ["All three together", "", "", sgn(ce["all"]["dt_ign_rel"], 5), m(f"{ce['all']['dT_end']:+.4f}")]],
        note="OpenFOAM's reader converts activation energies with 1.987316, although its own gas constant in joules corresponds to "
             + m_fix(cv["of"]["reader_R_joule"] / 4.184, 6) + ": this looks like a typing slip in the source. Its standard pressure of 1 bar shifts every "
             "equilibrium constant with a change of mole number by 1.3 %, which moves the final composition and so the final temperature. The column for CHEMKIN holds today's "
             "standard values; the program of the reference may have used slightly older ones.")
    comp_findings = [
        "<b>Against the reference</b>: every time step of the family is inside the required accuracy, by a factor of more than 30 on the ignition time and 60 on the "
        "final temperature. The difference barely changes with the step: it is not a time-step error of OpenFOAM.",
        "<b>Against the independent solution with the same constants</b>: the difference halves with every halving of the step, down to " + m_pct(fine["d_of_t"], 4, size=True)
        + ", and the extrapolation to a step of zero lands on it (" + sc(abs(st["t_ign"]["extrap_vs_of"]), 1) + "). This is the evidence that OpenFOAM solves the equations of Theory.",
        "<b>Where the " + m_pct(fine["d_ref_t"], 3, size=True) + " comes from</b>: " + m_pct(ind_vs_ref, 3, size=True) + " separates the reference from an independent solution with its own "
        "constants; " + m_pct(const_t, 4, size=True) + " comes from OpenFOAM's constants; " + m_pct(fine["d_of_t"], 4, size=True) + " is left from the time step. The first part "
        "cannot be traced further: the reference gives no tolerances, and a looser integration that reacts a little late would look exactly like this (the whole rise "
        "shifted by a constant " + mus(ref_behind_ind, 2) + ", Figure {fig:history}).",
        "<b>Final temperature</b>: OpenFOAM is " + m(f"{fine['T_end'] - T_ref:+.3f}") + " K from the reference, which prints to " + kel(0.1, 1) + ". OpenFOAM's standard pressure alone explains "
        + m(f"{ce['p_std']['dT_end']:+.3f}") + " K of the " + m(f"{fine['T_end'] - ind['T_end']:+.3f}") + " K between OpenFOAM and the independent solution with CHEMKIN's constants.",
        "<b>The reading rule</b> adds almost nothing: applied to the independent solution sampled at the same steps, the straight-line rule moves the ignition time by "
        + sc(abs(th["rule_bias"][str(coarse["steps"])]), 1) + " at the largest step and " + sc(abs(th["rule_bias"][str(Nf)]), 1) + " at the smallest.",
    ]

    # ---- performance --------------------------------------------------------------------------------------------
    def label(n):
        s = specs[n]
        how = "step set by the chemistry solver" if s["adjust"] else f"{s['steps']} fixed steps"
        solver = s["ode"] if s["chem"] == "ode" else "EulerImplicit"
        return f"{how}, {solver}" + ("" if s["table"] else ", no table of species")

    TICKS = 0.045                                                    # s: below five ticks of the solver's clock a cost is not shown
    ph = lambda r, i: m_fix(r["phase_us"][i], 0) if r.get("phase_us") and r["phase_us"][i] is not None and r["phase_s"][i] >= TICKS else "–"
    per = lambda r: m_fix(r.get("us_per_step"), 0) if r.get("loop_s", 0) >= TICKS else "–"
    order_runs = fam + [n for n in tol_names if n not in fam] + met_names + ([notable["name"]] if notable else []) + adj_names
    performance = dict(
        intro="One cell, one processor core. Solver time is the wall-clock time of the whole run, start-up included (about 0.3 s). The cost per step is from "
              "the solver's own clock inside the time loop, which has steps of 0.01 s, so it is an average over many steps. The three last columns split it by "
              "phase: the slow start (below " + kel(ext["phases"][0], 0) + "), the rise (to " + kel(ext["phases"][1], 0) + ") and the burnt gas. A cost measured over less than "
              "0.05 s of that clock is shown as a dash.",
        head=["Run", "Relative tolerance", "Absolute tolerance", "Steps", "Solver time (s)", "Per step (µs)", "Slow start (µs per step)", "Rise (µs per step)", "Burnt gas (µs per step)"],
        rows=[[label(n), sc(specs[n]["rel"], 1), sc(specs[n]["abs"], 1), runs[n]["steps"], m_fix(runs[n]["wall_time"], 2), per(runs[n]), ph(runs[n], 0), ph(runs[n], 1),
               ph(runs[n], 2)] for n in order_runs],
        notes=[
            "<b>What a step costs.</b> Away from the rise, a step costs about " + m_fix(floor_us, 0) + " microseconds whatever the tolerance: the chemistry solver has almost "
            "nothing to do, and the time goes to calling it, updating the gas properties and writing a log line and two output rows. This is an upper limit for "
            "what the chemistry of one cell costs in a flow solver, where no line is printed per cell.",
            "<b>The rise is the expensive part with tight tolerances.</b> With an absolute tolerance of " + sc(numerics["main"]["abs_tol"], 1) + " the rise costs "
            + m_fix(min(ign_s), 1) + " to " + m_fix(max(ign_s), 1) + " s in every run of the family, whether it is cut into " + str(coarse["phase_steps"][1]) + " steps or "
            + str(fine["phase_steps"][1]) + ": the solver's sub-steps are set by the tolerance, not by the time step (Figure {fig:cost}, left). With the tutorial's "
            "tolerances the same rise costs " + m_fix(loose["phase_s"][1], 2) + " s.",
            "<b>Cheaper at the same accuracy.</b> RKF45, an explicit method, gave the ignition time of seulex to nine digits in " + m_fix(rkf["wall_time"], 1) + " s against "
            + m_fix(main10["wall_time"], 1) + " s. With steps this small the problem is not stiff within a step, so an implicit method pays for a strength it does not use. "
            "With larger steps that would turn around.",
            "<b>Cheapest run inside the required accuracy</b>: the tutorial as shipped, " + str(shipped["steps"]) + " steps and " + m_fix(shipped["loop_s"], 2) + " s inside the time loop. "
            "For an accuracy of " + m_pct(abs(runs[tight[0]]["d_ref_t"]), 2) + " the adjustable step with a relative tolerance of " + sc(specs[tight[0]]["rel"], 1) + f" needs {runs[tight[0]]['steps']} steps.",
            "<b>The table of species</b> written at every step costs little: " + m_fix(main10["wall_time"], 2) + " s with it, " + (m_fix(notable["wall_time"], 2) if notable else "–") + " s without.",
            "<b>Run times repeat only roughly.</b> " + ("The run with the smallest step took " + m_fix(first_fine["wall_time"], 1) + " s the first time, while another job was using the machine, and "
            + m_fix(fine["wall_time"], 1) + " s when repeated alone; the tables show the repeat. " if first_fine else "") + "All runs together took "
            + m_fix(c["total_solver"], 0) + " s of solver time.",
            "<b>For a flow with chemistry in every cell</b> (the later steps towards a detonation engine), these numbers say: about a tenth of a millisecond of one core per "
            "cell per step where little reacts, and ten to a hundred times that in cells that are igniting if tolerances are tight. A million cells would need about "
            + m_fix(floor_us, 0) + " core-seconds per step. Not tried here: methods that tabulate or reduce the chemistry, and running it on the GPU.",
        ])

    # ---- figures ---------------------------------------------------------------------------------------------------
    svgs = dict(sketch=sketch(), history=figs.history(c), species=figs.species(c), conservation=figs.conservation(c), convergence=figs.convergence(c), cost=figs.cost(c))
    figures = [
        dict(key="sketch", section="problem",
             caption="The problem as a picture. A parcel of hydrogen, oxygen and nitrogen is held at constant pressure (a free piston stands for that) and exchanges no heat. "
                     "After a delay it ignites, heats up and expands. The piston is only a picture of the two constraints: the problem has no geometry."),
        dict(key="history", section="comparison",
             caption="Temperature against time. Left: the whole millisecond; the dotted line is the temperature of chemical equilibrium. Right: a window of 0.8 microseconds "
                     "around the ignition temperature (dotted line), with time counted from the ignition time of the independent solution. OpenFOAM is the solid line "
                     f"({Nf} steps), the circles are the reference's printed points, the dashed line is the independent solution with CHEMKIN's constants. The three curves "
                     "have the same shape; through the rise the reference runs on average " + mus(ref_behind_ind, 3) + " behind the independent solution and " + mus(ref_behind_of, 3)
                     + " behind OpenFOAM."),
        dict(key="species", section="simulation",
             caption=f"The six requested species in moles per kilogram of gas, {Nf} steps. Left: hydrogen is used up, half of the oxygen is left, water forms. Right, on a "
                     "logarithmic scale: OH, H and O start from nothing, multiply at a constant rate during the slow start (straight lines), peak during the rise and "
                     "then recombine slowly. The dotted line marks the ignition time."),
        dict(key="conservation", section="simulation",
             caption=f"What must not change. Left: the change of the number of hydrogen, oxygen and nitrogen atoms, relative to the start, at every stored step of the run "
                     f"with {Nf} steps; the dashed line is the required limit. Right: the change of the parcel's enthalpy in units of the heat released, for the largest and "
                     "the smallest step. The spike at the start is the accuracy to which the solver finds the temperature from the enthalpy."),
        dict(key="convergence", section="convergence",
             caption="The effect of the time step. Difference between OpenFOAM and the independent solution (same constants) against the time step, on logarithmic scales, "
                     "for the ignition time (left) and the final temperature (right). The dashed line is an error in proportion to the step. Both follow it: "
                     "OpenFOAM's treatment of the expansion is first order in time."),
        dict(key="cost", section="performance",
             caption=f"Cost. Left: solver time used so far along three runs of {main10['steps']} steps. With tight tolerances the tutorial's method spends half of its time "
                     "in the short rise; with the tutorial's tolerances, or with the explicit method RKF45, the rise costs almost nothing. Right: every run, its difference "
                     "from the independent solution in ignition time against its solver time; the dashed line is the required accuracy."),
    ]

    th_rep = theory.report_theory(th, g)
    results = dict(
        title=problem["title"], date=dt.date.today().isoformat(), verdict=verdict, expectations=X, comparison_type="benchmark",
        summary=dict(headline=headline, answer_columns=["Requested", f"Reference ({REF_NAME})", f"Simulation (OpenFOAM, {Nf} steps)", "Difference"],
                     answers=answers, findings=findings, limitations=limitations),
        problem=dict(
            description=problem["description"], setup=problem.get("setup", []),
            given=[dict(label=v["label"], symbol=v["symbol"], value=m(tex_num(v["value"], 6)), unit=v["unit"]) for v in problem["given"].values()],
            wanted=wanted, required_accuracy=problem["required_accuracy"],
            reference="Supplied with the problem: " + " ".join(problem["reference"]["description"].split())),
        assumptions=dict(rows=assumptions),
        theory=th_rep,
        decisions=dec,
        preflight=preflight,
        simulation=dict(model=model, meshes=meshes, results=results_tab, health=health, tables=[sp_tab, tol_tab, adj_tab, met_tab]),
        comparison=dict(
            kind=f"Benchmark: OpenFOAM against the calculation of another program ({REF_NAME}) with the same reaction scheme. An independent solution written for this "
                 "report stands next to both, to tell an error of OpenFOAM from an inaccuracy of the reference.",
            definitions=defs,
            head=["Steps", "Time step (µs)", "Ignition time: from the reference", "Final temperature: from the reference", "Ignition time: from the independent solution",
                  "Final temperature: from the independent solution (mK)", "Order, ignition time", "Order, final temperature"],
            rows=comp_rows, findings=comp_findings, tables=[lvl_tab, const_tab]),
        insights=load_yaml(HERE / "insights.yaml"),
        figures=figures,
        performance=performance,
        reproducibility=dict(
            commands=[
                dict(cmd=f"python -m cases.{HERE.name}.theory", what="The start state, the limits, the independent solution with both sets of constants, the SymPy checks"),
                dict(cmd=f"python -m cases.{HERE.name}.run preflight", what="The scheme check, two short tests, and the three largest steps in full"),
                dict(cmd=f"python -m cases.{HERE.name}.run production", what="Every other run, one at a time; writes runs/*.json and runs/*.npz"),
                dict(cmd=f"python -m cases.{HERE.name}.analyze", what="The numbers, the rules, the figures and this report (checks the contract first)"),
            ],
            files=[
                dict(path=f"cases/{HERE.name}/problem.yaml", what="The problem as given"),
                dict(path=f"cases/{HERE.name}/reference/senkin_h2.yaml", what="The reference as given"),
                dict(path=f"cases/{HERE.name}/numerics.yaml", what="Runs, reading rules, and the rules for the outcome written before the production runs"),
                dict(path=f"cases/{HERE.name}/decisions.yaml", what="Every choice, with the OpenFOAM settings it controls"),
                dict(path=f"cases/{HERE.name}/insights.yaml", what="Insights logged during the work"),
                dict(path=f"cases/{HERE.name}/template/", what="The OpenFOAM case with {{ placeholders }}"),
                dict(path=f"cases/{HERE.name}/mech.py", what="The independent program: reads the two data files, the rate laws, the parcel, the equilibrium"),
                dict(path=f"cases/{HERE.name}/theory.py", what="Theory numbers and SymPy checks"),
                dict(path=f"cases/{HERE.name}/measure.py", what="The rules that read the outputs from a history"),
                dict(path=f"cases/{HERE.name}/run.py", what="Builds, pre-flights and runs the cases"),
                dict(path=f"cases/{HERE.name}/numbers.py, figs.py, sketch.py", what="The numbers of the analysis and the figures"),
            ],
            run_dirs=[
                dict(path=f"{fine['run_root']}/S<steps>[_rel<tolerance>_abs<tolerance>][_<solver>], adj_rel<tolerance>_abs<tolerance>", what="Production runs"),
                dict(path=f"{fine['run_root']}/pf_*_preflight, scheme_check", what="Pre-flight runs"),
                dict(path=f"results/{HERE.name}/", what="theory.json, preflight.json, runs/, logs/, results.json and this report"),
            ],
            environment=report.environment_info(),
            notes=["Runs were made one at a time on one core. Another session was working on the same machine, so run times are rough.",
                   "The run with the smallest step was made twice; the numbers are identical, the times are not (see the performance appendix). The first record is kept in "
                   f"<code>results/{HERE.name}/repeat/</code>.",
                   "The OpenFOAM case reads the reaction scheme from OpenFOAM's own tutorial folder; nothing is copied into this repository."]),
        data=dict(
            reference=dict(t_ign=t_ref, T_end=T_ref), independent=dict(t_ign=ind["t_ign"], T_end=ind["T_end"]), independent_openfoam_constants=dict(t_ign=ind_of["t_ign"], T_end=ind_of["T_end"]),
            runs={n: {k: v for k, v in r.items() if k in ("steps", "adjust", "chem", "ode", "rel", "abs", "table", "t_ign", "T_end", "d_ref_t", "d_ref_T", "d_of_t", "d_of_T",
                                                             "atoms_max", "energy_dev", "wall_time", "loop_s", "us_per_step", "phase_s", "phase_steps", "levels", "dt_min", "dt_max")}
                  for n, r in runs.items()},
            study={k: {kk: vv for kk, vv in v.items()} for k, v in st.items()}, history=hi, species=sp, conv_effect=ce,
            first_timing_of_finest=first_fine["wall_time"] if first_fine else None),
    )
    out = report.render(results, svgs, OUT / "report.html", case_dir=fine["case_dir"])
    print(f"{verdict}: missed {[x['text'] for x in missed]}")
    for x in X:
        print(f"   {'ok  ' if x['met'] else 'MISS'} [{x['kind']}] {x['text'][:90]} | expected {x['expected'][:70]} | found {x['found'][:150]}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
