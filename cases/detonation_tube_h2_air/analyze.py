"""Analysis and report for the detonation tube.

  python -m cases.detonation_tube_h2_air.analyze
Reads results/detonation_tube_h2_air/runs, theory.json and preflight.json, applies the rules written in numerics.yaml
before the production runs, draws the figures and writes results.json and report.html (after the contract check).
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np

from cfdlib import chem, contract, report, verification
from cfdlib.report import LETTERS, lines, load_yaml, m, m_fix, m_pct, m_sig, tex_num
from . import figs, theory
from .run import OUT, RUNS, specs
from .sketch import sketch

HERE = Path(__file__).resolve().parent
PARTS = (0.2, 0.5, 0.75, 1.0)       # of t_end: the front speed is also given over the three stretches between these times


def sc(x, sig=2):
    """A number in scientific notation as inline math (zero as 0)."""
    return m(tex_num(x, sig)) if x else m("0")


def sgn(x, dec=2):
    return m_pct(x, dec, signed=True)


def unit(x, dec, u, scale=1.0):
    return m(f"{scale * x:.{dec}f}" + rf"\,\mathrm{{{u}}}")


mps = lambda v, dec=1: unit(v, dec, "m/s")
mpa = lambda p, dec=3: unit(p, dec, "MPa", 1e-6)
kel = lambda T, dec=0: unit(T, dec, "K")
mmm = lambda x, dec=2: unit(x, dec, "mm", 1e3)
um = lambda x, dec=0: unit(x, dec, r"\mu m", 1e6)
mus = lambda t, dec=1: unit(t, dec, r"\mu s", 1e6)


def compute():
    """Every number of the report that is not already in a run record."""
    problem, numerics, ref = theory.load()
    g, th, tha = theory.given(problem), theory.numbers(), theory.arrays()
    sp = specs(numerics)
    runs, data, missing, crashed = {}, {}, [], []
    for n in sp:
        f = RUNS / f"{n}.json"
        if not f.exists():
            missing.append(n)
            continue
        r = json.loads(f.read_text())
        if r.get("crashed"):
            crashed.append(n)
            continue
        runs[n] = r
        data[n] = dict(np.load(RUNS / f"{n}.npz"))
    planned = [f"N{n}" for n, _ in numerics["runs"]["family"]]
    fam = [n for n in planned if n in runs]
    sg = f"N{numerics['preflight']['passes']['cells']}"
    by = lambda study: [n for n in sp if study in sp[n]["studies"] and n in runs and sp[n]["fraction"] >= 1]
    cj = th["cj"]
    c = dict(problem=problem, numerics=numerics, ref=ref, g=g, th=th, tha=tha, specs=sp, runs=runs, data=data, missing=missing, crashed=crashed,
             planned=planned, fam=fam, study_grid=sg,
             steps=sorted(by("time step"), key=lambda n: sp[n]["steps"]), passes=sorted(by("passes"), key=lambda n: sp[n]["passes"]),
             chemistry=by("chemistry"))
    # grid study of the wave speed (three finest grids, ratio 2)
    if len(fam) >= 3:
        D3 = [runs[n]["speed"]["D"] for n in fam[-3:]]
        c["grid"] = verification.three_grid_study(D3, r=2.0, p_assumed=1.0)
    # how much the speed moves under changes that should not matter: the size below which a difference means nothing
    same = [runs[n]["speed"]["D"] for n in c["chemistry"]]
    c["noise"] = (max(same) - min(same)) / cj["D"] if len(same) > 1 else None
    # one core against 8 ranks, at the time the one-core run stopped
    ser, noc = runs.get(f"{sg}_serial"), runs.get(f"{sg}_serial_nochem")
    if ser and sg in runs:
        d = data[sg]
        k = int(np.argmin(np.abs(d["t"] - ser["t_last"])))
        ks = int(np.argmin(np.abs(d["ts"] - ser["t_last"])))
        c["serial"] = dict(t=ser["t_last"], front=ser["front_last"], front_par=float(d["xp"][k]), diff=float(d["xp"][k] / ser["front_last"] - 1),
                           energy=ser["energy_last"], energy_par=float(d["dE"][ks] / d["q"][ks]), wall=ser["wall_time"])
        if noc:
            ps = ser["passes"]
            c["cost"] = dict(flow=noc["wall_time"] / (noc["steps_done"] * noc["N"] * ps), chem=(ser["wall_time"] - noc["wall_time"]) / (ser["hot_cell_steps"] * ps),
                             share_short=1 - noc["wall_time"] / ser["wall_time"], burnt_share_short=ser["hot_cell_steps"] / (ser["steps_done"] * ser["N"]),
                             wall_serial=ser["wall_time"], wall_nochem=noc["wall_time"])
    for n, r in runs.items():
        if "speed" not in r:
            continue
        d = data[n]                                              # speed of the front over three parts of the time, against the CJ speed
        kk = [int(np.argmin(np.abs(d["t"] - f * g["t_end"]))) for f in PARTS]
        r["parts"] = [float((d["xp"][b] - d["xp"][a]) / (d["t"][b] - d["t"][a]) / cj["D"] - 1) for a, b in zip(kk, kk[1:])]
        r["core_us_per_hot_pass"] = 1e6 * r["wall_time"] * r["cores"] / (r["hot_cell_steps"] * r["passes"])
        r["burnt_share"] = r["hot_cell_steps"] / (r["steps_done"] * r["N"])
        if "cost" in c:
            r["flow_core_s"] = c["cost"]["flow"] * r["N"] * r["steps_done"] * r["passes"]
            r["flow_share"] = r["flow_core_s"] / (r["wall_time"] * r["cores"])
    # the points of the settings figure: label, speed error (the one-pass run is off the scale and is left to the table)
    lab = {}
    main = sp[sg]
    for n in c["steps"]:
        lab[n] = "main" if n == sg else ("step x 2" if sp[n]["steps"] < main["steps"] else "step / 2")
    for n in c["passes"]:
        lab.setdefault(n, f"{sp[n]['passes']} pass" + ("es" if sp[n]["passes"] > 1 else ""))
    for n in c["chemistry"]:
        lab.setdefault(n, "seulex" if sp[n]["ode"] != main["ode"] else "looser tolerance" if sp[n]["rel"] != main["rel"] else "no cut-off")
    c["labels"] = lab
    c["study_points"] = [(lab[n], runs[n]["speed_err"]) for n in lab if abs(runs[n]["speed_err"]) < 0.026]
    # the state behind the front at fixed distances, next to the two pictures of Theory (finest grid, end time)
    if fam:
        fine = runs[fam[-1]]
        order = np.argsort(tha["tw_xi"])
        rows = []
        for a in fine["burnt"]["at"]:
            xi = cj["D"] - a["d"] / g["t_end"]
            rows.append(dict(d=a["d"], p=a["p"], T=a["T"], u=a["u"], H2O=a["H2O"],
                             p_znd=float(np.interp(a["d"], tha["znd_x"], tha["znd_p"])), T_znd=float(np.interp(a["d"], tha["znd_x"], tha["znd_T"])),
                             p_tw=float(np.interp(xi, tha["tw_xi"][order], tha["tw_p"][order])), T_tw=float(np.interp(xi, tha["tw_xi"][order], tha["tw_T"][order]))))
        c["behind"] = rows
        # how the wave starts: its speed early on, and from when the speed over 2 microseconds stays within 1 % of the CJ speed
        d, w = data[fam[-1]], 4
        v = (d["xp"][w:] - d["xp"][:-w]) / (d["t"][w:] - d["t"][:-w]) / cj["D"] - 1
        tm = 0.5 * (d["t"][w:] + d["t"][:-w])
        off = np.nonzero(np.abs(v) > 0.01)[0]
        k2, k4 = int(np.argmin(np.abs(d["t"] - 2e-6))), int(np.argmin(np.abs(d["t"] - 4e-6)))
        c["start"] = dict(early=float((d["xp"][k4] - d["xp"][k2]) / (d["t"][k4] - d["t"][k2]) / cj["D"] - 1),
                          t_settle=float(tm[off[-1] + 1]) if len(off) and off[-1] + 1 < len(tm) else float(tm[0]) if not len(off) else None)
    c["total_wall"] = sum(r["wall_time"] for r in runs.values())
    return c


def main():
    c = compute()
    problem, numerics, ref, g, th = c["problem"], c["numerics"], c["ref"], c["g"], c["th"]
    runs, data, sp, fam, planned, sg = c["runs"], c["data"], c["specs"], c["fam"], c["planned"], c["study_grid"]
    acc, ext, mn = numerics["acceptance"], numerics["extraction"], numerics["main"]
    pf = json.loads((OUT / "preflight.json").read_text())
    cj, fr, zn, tw, st = th["cj"], th["front"], th["znd"], th["taylor"], th["start"]
    R = [runs[n] for n in fam]
    fine, coarse = R[-1], R[0]
    all_ran = fam == planned and not c["crashed"] and not c["missing"]
    full = [r for r in runs.values() if "speed" in r]
    N_of = lambda r: f"{r['N']} cells"
    Rs = runs[sg]

    # ---- the rules written before the production runs ---------------------------------------------------------
    X = []

    def rule(kind, text, expected, found, met, key=None):
        x = dict(kind=kind, text=text, expected=expected, found=found, met=bool(met))
        if key:
            x["key"] = key
        X.append(x)

    req = {q["key"]: q["text"] for q in problem["required_accuracy"]}
    rule("requirement", req["speed"], "within " + m_pct(acc["req_speed"], 0) + f" (finest grid, {N_of(fine)})",
         mps(fine["speed"]["D"]) + " against " + mps(cj["D"], 2) + ": " + sgn(fine["speed_err"]),
         all_ran and abs(fine["speed_err"]) <= acc["req_speed"], key="speed")
    rule("requirement", req["attached"], "at most " + mmm(acc["req_gap"], 0) + f" at every sampled time from {mus(ext['attached_from'], 0)} on (finest grid)",
         "largest distance " + mmm(fine["gap_max"], 3) + "; on the other grids " + ", ".join(mmm(r["gap_max"], 3) for r in R[:-1]),
         all_ran and fine["gap_max"] <= acc["req_gap"], key="attached")
    rule("health", "Every planned run reaches the end time without a solver error, and its log passes the checks",
         f"{len(sp)} runs", f"{len(runs)} of {len(sp)} finished" + (f"; stopped: {', '.join(c['crashed'])}" if c["crashed"] else "")
         + (f"; not run: {', '.join(c['missing'])}" if c["missing"] else "") + "; log checks: " + ", ".join(sorted({r["health"]["status"] for r in runs.values()})),
         not c["crashed"] and not c["missing"] and all(r["health"]["status"] != "fail" for r in runs.values()))
    worst_mass = max(r["mass_dev"] for r in full)
    worst_atoms = max(max(r["atoms_dev"].values()) for r in full)
    rule("health", "The mass in the closed tube and the atoms of each element do not change", "change of at most " + sc(acc["mass"], 1) + " (relative) in every run",
         "largest change of the mass " + sc(worst_mass, 1) + ", of the atoms of an element " + sc(worst_atoms, 1), worst_mass <= acc["mass"] and worst_atoms <= acc["mass"])
    rule("health", "Temperature and pressure never fall below their start values (no undershoot at the front)",
         "not below " + kel(g["T_0"]) + " and " + unit(g["p_0"], 0, "Pa"), "lowest temperature " + kel(min(r["T_min"] for r in full), 2) + ", lowest pressure "
         + unit(min(r["p_min"] for r in full), 1, "Pa"), all(r["T_min"] >= g["T_0"] * (1 - 1e-3) and r["p_min"] >= g["p_0"] * (1 - 1e-3) for r in full))
    worst_rate = max(abs(r["energy"]["rate_late"]) for r in R)
    rule("expectation", "Energy balance: over the second half of the time the total energy changes by a small part of the heat released in that time, on every grid",
         "at most " + m_pct(acc["exp_energy_rate"], 1), "largest " + m_pct(worst_rate, 2) + " (" + ", ".join(sgn(r["energy"]["rate_late"]) for r in R) + ", coarse to fine)",
         all_ran and worst_rate <= acc["exp_energy_rate"])
    worst_speed = max(abs(r["speed_err"]) for r in R)
    rule("expectation", "Every grid of the family, the coarsest included, gives the wave speed within the required accuracy",
         "within " + m_pct(acc["exp_all_grids_speed"], 0) + " on all " + str(len(planned)) + " grids", ", ".join(sgn(r["speed_err"]) for r in R) + " (coarse to fine)",
         all_ran and worst_speed <= acc["exp_all_grids_speed"])
    d_grid = abs(R[-1]["speed"]["D"] - R[-2]["speed"]["D"]) / cj["D"] if len(R) > 1 else float("nan")
    rule("expectation", "The wave speed has settled with the cell size: change from the second finest to the finest grid",
         "at most " + m_pct(acc["exp_grid_change"], 1) + " of the CJ speed", m_pct(d_grid, 2), all_ran and d_grid <= acc["exp_grid_change"])
    ratios = [r["peak"]["ratio"] for r in R]
    rule("expectation", "The highest pressure at the front rises towards the shock pressure of Theory with every refinement, and does not pass it",
         f"rising; on the finest grid between {acc['exp_peak_finest'][0]} and {acc['exp_peak_finest'][1]} of the shock pressure",
         ", ".join(m_fix(x, 3) for x in ratios) + " (coarse to fine)",
         all_ran and all(b > a for a, b in zip(ratios, ratios[1:])) and acc["exp_peak_finest"][0] <= ratios[-1] <= acc["exp_peak_finest"][1])
    b = fine["burnt"]
    rule("expectation", "Burnt state on the finest grid, read where the gas leaves the front at its own speed of sound: temperature and pressure against the CJ state",
         "the point exists; temperature within " + m_pct(acc["exp_sonic_T"], 0) + ", pressure within " + m_pct(acc["exp_sonic_p"], 0),
         ("the point is " + mmm(b["d"]) + " behind the front: temperature " + sgn(b["T"] / cj["T"] - 1) + ", pressure " + sgn(b["p"] / cj["p"] - 1)) if b["found"] else "no such point behind the front",
         all_ran and b["found"] and abs(b["T"] / cj["T"] - 1) <= acc["exp_sonic_T"] and abs(b["p"] / cj["p"] - 1) <= acc["exp_sonic_p"])
    rule("expectation", "Pressure of the gas at rest at the closed end, against the value of the expansion wave of Theory (finest grid)",
         "within " + m_pct(acc["exp_end_p"], 0), mpa(fine["end_p"], 4) + " against " + mpa(tw["p"], 4) + ": " + sgn(fine["end_p"] / tw["p"] - 1),
         all_ran and abs(fine["end_p"] / tw["p"] - 1) <= acc["exp_end_p"])
    steady = max(abs(fine["speed"]["D_min"] / cj["D"] - 1), abs(fine["speed"]["D_max"] / cj["D"] - 1))
    rule("expectation", "The wave is steady over the second half of the time: speed of the front over each half microsecond (finest grid)",
         "within " + m_pct(acc["exp_steady"], 0) + " of the CJ speed", "between " + sgn(fine["speed"]["D_min"] / cj["D"] - 1) + " and " + sgn(fine["speed"]["D_max"] / cj["D"] - 1),
         all_ran and steady <= acc["exp_steady"])
    worst_right = max(r["right_dev"] for r in full)
    rule("expectation", "The gas at the far end is untouched at the end time (the wave has not arrived)", "pressure within " + sc(acc["exp_right"], 1) + " of the start value",
         "largest difference " + (sc(worst_right, 1) if worst_right else m("0")) + ", temperature " + kel(max(r["right_T"] for r in full), 3), worst_right <= acc["exp_right"])
    chem_names = [n for n in c["chemistry"] if n != sg]
    chem_dev = max(abs(runs[n]["speed"]["D"] - Rs["speed"]["D"]) / cj["D"] for n in chem_names) if chem_names else float("nan")
    rule("expectation", f"The chemistry settings do not matter ({Rs['N']} cells): another solver, a looser tolerance and no cut-off temperature give the wave speed of the main settings",
         "within " + m_pct(acc["exp_chemistry"], 1) + " of the CJ speed, 3 runs", f"largest difference {m_pct(chem_dev, 2)} over {len(chem_names)} runs",
         len(chem_names) == 3 and chem_dev <= acc["exp_chemistry"])
    rule("expectation", "Solver time of the run on the finest grid", f"at most {acc['exp_finest_minutes']} minutes", m_fix(fine["wall_time"] / 60, 1) + " minutes",
         all_ran and fine["wall_time"] <= 60 * acc["exp_finest_minutes"])
    verdict = report.verdict_from(X)
    missed = [x for x in X if not x["met"]]

    # ---- numbers used in several places ------------------------------------------------------------------------
    wanted = [dict(letter=LETTERS[i], key=q["key"], text=q["text"]) for i, q in enumerate(problem["quantities_of_interest"])]
    L = {w["key"]: w["letter"] for w in wanted}
    gr = c.get("grid")
    cost, ser = c.get("cost"), c.get("serial")
    p1 = next((runs[n] for n in c["passes"] if sp[n]["passes"] == 1), None)
    half = next((runs[n] for n in c["steps"] if sp[n]["steps"] > sp[sg]["steps"]), None)
    dbl = next((runs[n] for n in c["steps"] if sp[n]["steps"] < sp[sg]["steps"]), None)
    behind = c["behind"]
    seu = next((runs[n] for n in c["chemistry"] if sp[n]["ode"] != sp[sg]["ode"]), None)
    rel = next((runs[n] for n in c["chemistry"] if sp[n]["rel"] != sp[sg]["rel"]), None)
    allc = next((runs[n] for n in c["chemistry"] if sp[n]["t_react"] != sp[sg]["t_react"]), None)
    mc = theory.mech()
    c_frozen_cj = float(chem.sound_speed_frozen(mc, cj["T"], np.array([cj["Y"][s] for s in mc.species]) / mc.W))
    s0 = c["start"]
    start_txt = ("Between 2 and 4 microseconds the wave on the finest grid is " + m_pct(s0["early"], 1, size=True) + (" slower" if s0["early"] < 0 else " faster") + " than the CJ speed"
                 + ("; from " + mus(s0["t_settle"], 0) + " on its speed stays within 1 % of it." if s0["t_settle"] is not None else "; it does not stay within 1 % of it to the end."))
    cells_gap = [r["gap_mean"] / r["dx"] for r in R]
    parts_txt = ", ".join(f"{1e6 * a * g['t_end']:g} to {1e6 * b * g['t_end']:g}" for a, b in zip(PARTS, PARTS[1:])) + " microseconds"
    spread_grid = max(r["speed_err"] for r in R) - min(r["speed_err"] for r in R)
    spread_dt = (max(runs[n]["speed_err"] for n in c["steps"]) - min(runs[n]["speed_err"] for n in c["steps"])) if len(c["steps"]) > 1 else None
    p3 = next((runs[n] for n in c["passes"] if sp[n]["passes"] == 3), None)
    loss_core = (("with three passes per step the loss is " + m_pct(p3["energy"]["rate_late"], 1, size=True) + " of the heat released, against "
                  + m_pct(Rs["energy"]["rate_late"], 1, size=True) + " with five, and the speed is only " + m_pct(Rs["speed_err"] - p3["speed_err"], 2) + " lower.") if p3 else "")
    loss_txt = (" The energy lost by the solver explains little of it: " + loss_core) if p3 else ""
    still_rising = all(r["parts"][2] > r["parts"][1] > r["parts"][0] for r in R)
    faster_coarse = (coarse["parts"][2] - coarse["parts"][0]) > (fine["parts"][2] - fine["parts"][0])
    n_miss = len(missed)
    head_word = ("The experiment failed. " if verdict == "FAIL" else "Completed as expected. " if verdict == "PASS"
                 else f"Completed, with {'one difference' if n_miss == 1 else str(n_miss) + ' differences'} from what was expected. ")
    headline = (head_word + "OpenFOAM's <code>reactingFoam</code> carries a detonation down the tube at " + mps(fine["speed"]["D"]) + ", " + m_pct(fine["speed_err"], 2, size=True)
                + (" below" if fine["speed_err"] < 0 else " above") + " the Chapman-Jouguet speed (required: within " + m_pct(acc["req_speed"], 0)
                + "), with the burning front " + mmm(fine["gap_max"], 2) + " or less behind the pressure front (required: within " + mmm(acc["req_gap"], 0) + ").")

    answers = [
        dict(letter=L["speed"], label="Wave speed over the second half of the time", theory=mps(cj["D"], 2),
             sim=mps(fine["speed"]["D"]) + f" on {N_of(fine)}; " + ", ".join(mps(r["speed"]["D"], 0) for r in R[:-1]) + " on the coarser grids (Figure {fig:xt})",
             diff=sgn(fine["speed_err"])),
        dict(letter=L["attached"], label="Do the two fronts travel together?",
             theory="In the steady wave of Theory half of the water has formed " + mmm(zn["x_half_water"], 2) + " behind the shock",
             sim="Yes: a detonation. From " + mus(ext["attached_from"], 0) + " on the burning front is at most " + mmm(fine["gap_max"], 3) + " behind the pressure front (mean "
                 + mmm(fine["gap_mean"], 3) + "), and on no grid more than " + mmm(max(r["gap_max"] for r in R), 3) + " (Figure {fig:attached})",
             diff=lines("limit " + mmm(acc["req_gap"], 0), "found " + mmm(fine["gap_max"], 2))),
        dict(letter=L["burnt"], label="Burnt gas just behind the wave", theory=lines(mpa(cj["p"]), kel(cj["T"])),
             sim=(("Where the gas leaves the front at its own speed of sound, " + mmm(b["d"]) + " behind it: " + mpa(b["p"]) + " and " + kel(b["T"]) + ". ") if b["found"] else
                  "No point behind the front where the gas leaves at its own speed of sound was found. ")
                 + "The pressure has no flat part behind the front: it falls from the front's peak straight into the expansion (Figure {fig:structure})",
             diff=lines("p " + sgn(b["p"] / cj["p"] - 1, 1), "T " + sgn(b["T"] / cj["T"] - 1, 1)) if b["found"] else "–"),
        dict(letter=L["front"], label="Highest pressure at the front", theory=mpa(fine["peak"]["vn"]) + " behind a shock of the simulated speed without burning",
             sim=mpa(fine["peak"]["mean"]) + f" on {N_of(fine)}, " + m_fix(fine["peak"]["ratio"], 2) + " of the shock pressure. With the cell size: "
                 + ", ".join(m_fix(r["peak"]["ratio"], 2) for r in R) + " (coarse to fine). The cells are too large to show the full peak",
             diff=sgn(fine["peak"]["ratio"] - 1, 0)),
        dict(letter=L["conserved"], label="Mass, atoms and total energy", theory="No change (closed tube)",
             sim="Mass and atoms: kept, change of at most " + sc(max(worst_mass, worst_atoms), 1) + ". Total energy: not kept exactly; over the second half of the time "
                 + f"the run on {N_of(fine)} loses " + m_pct(fine["energy"]["rate_late"], 2, size=True) + " of the heat released (Figure {fig:conservation})",
             diff=lines("mass " + sc(worst_mass, 1), "energy " + sgn(fine["energy"]["rate_late"]))),
        dict(letter=L["cost"], label="Effect of cell size and time step, and cost", theory="–",
             sim="Cell size: " + ", ".join(sgn(r["speed_err"]) for r in R) + " from the CJ speed, coarse to fine. Time step ("
                 + (f"{Rs['N']} cells): " + sgn(dbl["speed_err"]) + ", " + sgn(Rs["speed_err"]) + ", " + sgn(half["speed_err"]) + " at twice, once and half the main step. " if dbl and half else "not run). ")
                 + "Cost: " + m_fix(fine["us_per_cell_step"], 0) + " microseconds per cell and step on the finest grid with 8 cores (Figure {fig:convergence}). "
                 + (("Share of the chemistry: " + m_pct(cost["share_short"], 0) + " of the run time on one core over the first fifth of the time, when only "
                     + m_pct(cost["burnt_share_short"], 0) + " of the cell-steps are burnt cells. Over the full runs the flow takes " + m_pct(min(r["flow_share"] for r in R), 0, unit=False)
                     + " to " + m_pct(max(r["flow_share"] for r in R), 0) + " of the core time; the rest is chemistry and message passing (appendix B)") if cost else "Share of the chemistry: not measured"),
             diff="–"),
    ]

    findings = [
        "<b>It is a detonation, and it travels at the right speed.</b> On every grid the pressure front and the burning front move together, and the speed over the second half "
        "of the time is " + ", ".join(sgn(r["speed_err"]) for r in R) + " from the Chapman-Jouguet speed (" + ", ".join(str(r["N"]) for r in R) + " cells).",
        "<b>The speed does not need a resolved front.</b> None of the grids resolves the stretch in which the gas waits before it burns (" + mmm(zn["x_ind"], 2) + "; the finest grid has "
        + m_fix(fine["cells_per_induction"], 1) + " cells in it). The speed is fixed by conservation and by the heat released, which a conservative run gets right on any grid. "
        "What the coarse grids miss is the pressure peak at the front: " + m_fix(coarse["peak"]["ratio"], 2) + " of the shock pressure on the coarsest grid, " + m_fix(fine["peak"]["ratio"], 2) + " on the finest.",
        "<b>Energy conservation is the setting that decides.</b> With one pass through the equations per time step, as in OpenFOAM's reacting examples, the solver loses energy"
        + (" and the wave runs " + m_pct(p1["speed_err"], 1, size=True) + " too slowly (energy lost over the second half: " + m_pct(p1["energy"]["rate_late"], 1, size=True)
           + " of the heat released)" if p1 else "") + ". Five passes per step bring the loss down to " + m_pct(Rs["energy"]["rate_late"], 2, size=True) + ". The loss does not show in residuals or in the mass.",
        "<b>The burnt state has no plateau.</b> In a closed tube an expansion wave follows the front at once, so the pressure falls continuously. Read where the gas leaves the front "
        "at its own speed of sound" + ((", the state is " + sgn(b["p"] / cj["p"] - 1, 1) + " in pressure and " + sgn(b["T"] / cj["T"] - 1, 1) + " in temperature from the CJ state") if b["found"] else ", no state could be read")
        + ". The gas at rest at the closed end is at " + mpa(fine["end_p"]) + ", " + sgn(fine["end_p"] / tw["p"] - 1) + " from the value Theory gives for the expansion.",
        "<b>The wave is still closing in on the CJ speed when the time ends.</b> On the finest grid the front runs " + ", ".join(sgn(x) for x in fine["parts"]) + " from the CJ speed over "
        + parts_txt + "; on the coarsest " + ", ".join(sgn(x) for x in coarse["parts"]) + ". "
        + ("The deficit shrinks with time on every grid" + (", faster on the coarsest than on the finest" if faster_coarse else "") + ". " if still_rising else "")
        + "So part of the deficit of the answer is a wave that has not finished settling, and the differences between the grids (" + m_pct(spread_grid, 2) + " in all) reflect how far each "
        "one has got; they do not fall with the cell size. How much of the deficit a longer tube would leave is not known from these runs." + loss_txt,
        "<b>What moves the speed, and what does not</b> (" + str(Rs["N"]) + " cells). The time step: " + (m_pct(spread_dt, 2) if spread_dt is not None else "not run") + " between twice and half the main step, "
        "without a trend. The chemistry solver, its tolerance and the cut-off temperature: " + (m_pct(c["noise"], 3) if c["noise"] is not None else "not run") + ". The passes per step: "
        + ", ".join(sgn(runs[n]["speed_err"] - Rs["speed_err"]) + f" with {sp[n]['passes']}" for n in c["passes"] if n != sg) + " against five.",
        "<b>The chemistry is nearly all of the cost.</b> "
        + (("A burnt cell costs about " + m_fix(1e6 * cost["chem"], 0) + " microseconds of one core per pass, a cell without chemistry " + m_fix(1e6 * cost["flow"], 1)
            + f" ({Rs['N']} cells, one core). ") if cost else "")
        + f"The finest run took " + m_fix(fine["wall_time"] / 60, 1) + " minutes on 8 cores; all runs together " + m_fix(c["total_wall"] / 60, 0) + " minutes.",
    ]
    limitations = [
        "One dimension. A real detonation front is not flat: it carries a pattern of cross-waves (cells), which a tube with one row of cells cannot hold. "
        "This case says nothing about that structure.",
        "No grid resolves the burning zone. The finest has " + m_fix(fine["cells_per_induction"], 1) + " cells in the wait before burning and shows " + m_fix(fine["peak"]["ratio"], 2)
        + " of the pressure peak. A plan with 9600 cells was cut for cost before the runs (see the mesh decision).",
        "The wave has not finished settling within the given time" + (": on every grid its speed is still rising in the last quarter" if still_rising else "") + ". The speed asked for is "
        "therefore not a converged property of the grid, no observed order is given, and nothing is extrapolated to zero cell size. A longer tube would be needed to see where each grid ends up.",
        "The time-step, passes and chemistry studies were made on " + str(Rs["N"]) + " cells only; on the finer grids the energy balance of each run stands in for them.",
        "The reference is the ideal steady wave. It says nothing about the first microseconds, in which the wave forms out of the ignition layer, and "
        "the mixture's reaction scheme is taken as given (the reference file puts the spread between published schemes and data at about half a percent of the speed).",
        "Total energy is not conserved by the solver. The remaining loss is " + m_pct(worst_rate, 2) + " of the heat released at most; what it costs in speed was measured only "
        "through the passes study on " + str(Rs["N"]) + " cells.",
        "Solved with OpenFOAM only. The GPU solver gpucfd has no chemistry in its flow solver yet, so there is no second tool to compare with.",
        "Run times are from a shared desktop on 8 cores under WSL, with another session at work; they repeat only roughly.",
    ]

    # ---- assumptions -----------------------------------------------------------------------------------------------
    assumptions = [
        dict(name="One dimension: only the distance along the tube matters", check="Given. One row of cells; the side faces carry no flow. Real fronts are three-dimensional (see the limitations).", status="imposed"),
        dict(name="No friction and no heat conduction", check="Given. The transport file that ships with the scheme sets the viscosity to zero, and with it the conduction and the diffusion "
             "of species in this solver. The front's thickness is then set by the grid.", status="imposed"),
        dict(name="Ideal gases with heat capacities from the data file", check="Given. At the highest pressure of the runs (" + mpa(max(r["p_max"] for r in full), 1) + ", in the ignition layer as it burns) and "
             "temperatures above " + kel(g["T_0"]) + " the gas is far from its liquid state; real-gas effects are well below the required accuracy.", status="standard"),
        dict(name="The reaction scheme and data files as shipped, unchanged", check="The solver reads both files where OpenFOAM ships them. The start state it builds from them agrees with the one computed "
             "without the solver: mass to " + sc(abs(pf["short"][-1]["first"]["mass"]), 1) + ", highest pressure " + mpa(pf["short"][-1]["first"]["p_max"], 4) + " (pre-flight).", status="ok"),
        dict(name="Closed ends: nothing enters or leaves", check="Walls at both ends. The mass in the tube changes by at most " + sc(worst_mass, 1) + " and the atoms of each element by at most "
             + sc(worst_atoms, 1) + " in every run.", status="ok"),
        dict(name="The wave does not reach the far end within the time of interest", check="At the CJ speed the front is at " + mmm(th["x_front_cj"], 1) + " of " + mmm(g["L"], 0) + " at the end. "
             "Found: front at " + mmm(fine["x_front_end"], 1) + "; the pressure at the far wall differs from its start value by " + (sc(worst_right, 1) if worst_right else m("0")) + ".", status="ok"),
        dict(name="The wave has settled by the middle of the time", check="The speed is read from " + mus(g["t_end"] / 2, 0) + " on. Over each half microsecond after that the front speed stays between "
             + sgn(fine["speed"]["D_min"] / cj["D"] - 1) + " and " + sgn(fine["speed"]["D_max"] / cj["D"] - 1) + " of the CJ speed (finest grid; Figure {fig:xt}). It is not fully settled: over the third and the fourth quarter of the time the "
             "front runs " + sgn(fine["parts"][1]) + " and " + sgn(fine["parts"][2]) + " from the CJ speed, still closing in.",
             status="ok" if steady <= acc["exp_steady"] else "violated"),
        dict(name="The burnt gas behind the wave is in chemical equilibrium (assumed by the CJ state)", check=("Where the burnt state is read the water mass fraction is " + m_fix(b["H2O"], 4)
             + " against " + m_fix(cj["Y"]["H2O"], 4) + " at the CJ state." if b["found"] else "Not checked: no burnt state was read."), status="ok" if b["found"] and abs(b["H2O"] / cj["Y"]["H2O"] - 1) < 0.05 else "standard"),
    ]

    # ---- pre-flight ------------------------------------------------------------------------------------------------
    dec = load_yaml(HERE / "decisions.yaml")
    dec_problems = contract.check_decisions(dec) + contract.uncovered_settings(dec, fine["case_dir"])
    sh = pf["short"]
    pre_checks = list(sh[-1]["checks"]) + [
        dict(name="Every setting belongs to a documented decision", status="ok" if not dec_problems else "fail",
             value="all covered" if not dec_problems else f"{len(dec_problems)} missing", meaning="Checked on the case of the finest grid."),
    ]
    if ser:
        pre_checks.append(dict(name="A run on 8 cores gives the answer of a run on one core", status="ok" if abs(ser["diff"]) < 2e-3 else "warn",
                               value="front after " + mus(ser["t"], 0) + ": " + mmm(ser["front"], 3) + " on one core, " + mmm(ser["front_par"], 3) + " on 8 (" + sgn(ser["diff"], 3)
                                     + "); energy balance " + sgn(ser["energy"], 3) + " and " + sgn(ser["energy_par"], 3),
                               meaning="The split of the tube between cores must not change the answer. A difference in the fourth digit remains (insight 12)."))
    if cost:
        pre_checks.append(dict(name="Where the time goes (one core, first fifth of the time)", status="ok",
                               value=m_fix(cost["wall_serial"], 1) + " s with chemistry, " + m_fix(cost["wall_nochem"], 1) + " s without: " + m_fix(1e6 * cost["chem"], 0)
                                     + " microseconds per burnt cell and pass, " + m_fix(1e6 * cost["flow"], 1) + " per cell and pass for the flow",
                               meaning="These two costs sized the run plan (insight 11)."))
    if dbl and half:
        pre_checks.append(dict(name=f"Time step, {Rs['N']} cells, in full", status="ok",
                               value="wave speed " + sgn(dbl["speed_err"]) + ", " + sgn(Rs["speed_err"]) + " and " + sgn(half["speed_err"]) + " from the CJ speed at twice, once and half the main step",
                               meaning="No trend with the step: the three differ by less than the required accuracy by a factor of five or more."))
    if p1:
        pre_checks.append(dict(name=f"Passes per time step, {Rs['N']} cells, in full", status="ok",
                               value="wave speed " + ", ".join(sgn(runs[n]["speed_err"]) + f" with {sp[n]['passes']}" for n in c["passes"]) + " passes; energy lost over the second half "
                                     + ", ".join(m_pct(runs[n]["energy"]["rate_late"], 2, size=True) for n in c["passes"]),
                               meaning="One pass per step fails the requirement on the speed. Three or more meet it; five is the main setting."))
    preflight = dict(
        purpose="Before the production runs: does the solver start from the right state and keep mass and atoms, do the output files hold what the analysis needs, what do the runs cost, "
                "and how large are the effects of the time step and of the passes per step?",
        runs=[
            dict(name=f"{sh[0]['N']} and {sh[-1]['N']} cells, first 4 % of the time", setup=f"{sh[0]['n_steps']} and {sh[-1]['n_steps']} steps of the planned size after the growing first steps, 8 cores",
                 outcome=f"{sh[0]['status'].upper()} and {sh[-1]['status'].upper()}: every check below (shown for {sh[-1]['N']} cells)"),
            dict(name=f"{Rs['N']} cells on one core, first fifth of the time, with and without chemistry", setup="main settings",
                 outcome="same front as on 8 cores to the fourth digit; the two unit costs" if ser else "not run"),
            dict(name=f"{Rs['N']} cells in full: main settings, twice and half the time step, 1, 3 and 7 passes", setup="8 cores, one to two minutes each",
                 outcome="healthy; their numbers sized the rules for the outcome"),
        ],
        checks=pre_checks,
        conclusion="The setup is right: the start state, mass, atoms and the output files all check. The pre-flight changed the plan twice. A first version failed the start-state check "
                   "(an ignition layer of 7.5 cells, and a first time step that made mass), which led to the grid rule and the tiny first step. And its timing run showed that the "
                   "first run plan, up to 9600 cells, would take over an hour; the plan was cut to four grids up to " + str(fine["N"]) + " cells before any full run was kept.")

    # ---- simulation --------------------------------------------------------------------------------------------
    model = [
        dict(aspect="Tube", real="A straight closed tube, the same across its section", model="One row of equal cells between two walls; the side faces are empty (no flow through them)"),
        dict(aspect="Gas", real="Hydrogen, oxygen, nitrogen and what they form: ideal gases, heat capacities from the data file", model="The same data file, read by OpenFOAM in place; 9 species carried, nitrogen as the remainder"),
        dict(aspect="Burning", real="27 reactions with rates from the given file", model="The same file. In every cell behind the front the reactions are integrated over each time step with error control"),
        dict(aspect="Ignition", real="A layer at the left end, hot and at high pressure at the start", model="The cells of the first " + mmm(g["L_ign"], 1) + " start at the given state: a whole number of cells on every grid"),
        dict(aspect="Friction, conduction", real="None (given)", model="Viscosity zero in the scheme's transport file; the front's thickness comes from the grid"),
        dict(aspect="Time", real="From the start to " + mus(g["t_end"], 0), model=f"{coarse['steps']} to {fine['steps']} equal steps after a short ramp; five passes through the equations per step"),
    ]
    meshes = dict(
        intro="Equal cells along the tube, each grid with half the cell size of the one before. The length to compare the cells with is the stretch behind the shock in which the gas waits "
              "before it burns, " + mmm(zn["x_ind"], 3) + " in the steady wave of Theory. The time step is fixed on each grid so that the wave crosses about 0.4 of a cell per step.",
        head=["Cells", "Cell size (µm)", "Cells in the wait before burning", "Time steps", "Time step (ns)", "Cells crossed by the wave per step", "Cores", "Solver time (s)"],
        rows=[[r["N"], m_fix(1e6 * r["dx"], 1), m_fix(r["cells_per_induction"], 1), r["steps"], m_fix(1e9 * r["dt"], 1), m_fix(r["courant_wave"], 2), r["cores"], m_fix(r["wall_time"], 0)] for r in R])
    sonic = lambda r, k: (m_fix(r["burnt"][k] / cj[k], 3) if r["burnt"]["found"] else "–")
    results_tab = dict(
        head=["Cells", "Wave speed (m/s)", "From the CJ speed", "Largest distance between the fronts (µm)", "Highest pressure at the front (MPa)", "Of the shock pressure",
              "Burnt state: distance behind the front (mm)", "Pressure / CJ", "Temperature / CJ", "Pressure at the closed end (MPa)"],
        rows=[[r["N"], m_fix(r["speed"]["D"], 1), sgn(r["speed_err"]), m_fix(1e6 * r["gap_max"], 0), m_fix(r["peak"]["mean"] / 1e6, 3), m_fix(r["peak"]["ratio"], 3),
               m_fix(1e3 * r["burnt"]["d"], 2) if r["burnt"]["found"] else "not found", sonic(r, "p"), sonic(r, "T"), m_fix(r["end_p"] / 1e6, 4)] for r in R]
             + [["Theory", m_fix(cj["D"], 2), "–", "–", m_fix(fr["p"] / 1e6, 3), m("1"), "–", m("1"), m("1"), m_fix(tw["p"] / 1e6, 4)]])
    health = dict(
        head=["Run", "Log checks", "Lowest temperature (K)", "Lowest pressure (Pa)", "Mass: largest change", "Atoms: largest change", "Energy at the end (% of heat released)",
              "Energy, second half (% of heat released there)", "Energy after the first microsecond (% of the layer's surplus)"],
        rows=[[f"{r['N']} cells" + (f", {c['labels'][n]}" if n in c["labels"] and c["labels"][n] != "main" else ""), r["health"]["status"].upper(),
               m_fix(r["T_min"], 2), m_fix(r["p_min"], 1), sc(r["mass_dev"], 1), sc(max(r["atoms_dev"].values()), 1), sgn(r["energy"]["end"], 3).replace(r"\,\%", ""),
               sgn(r["energy"]["rate_late"], 3).replace(r"\,\%", ""), sgn(r["energy_startup"], 2).replace(r"\,\%", "")] for n, r in runs.items() if "speed" in r],
        notes=[
            "A run that marches in time has no final residual to judge. The evidence that each run is sound is physical: the mass and the atoms of each element in the closed tube must not "
            "change, the total energy should not, and nothing may cool below the start state. All are checked after every time step from the solver's own integrals.",
            "Total energy is the sum over the tube of heat content, chemical energy and kinetic energy, minus pressure times volume. The solver does not conserve it by construction "
            "(it solves the energy equation before the pressure). The table gives the change as a share of the heat that the burning had released; Figure {fig:conservation} shows it in time.",
            "The linear solvers reach their tolerance in two to three iterations per equation; no run reached a limit of iterations.",
            "The runs stop at the given end time. Nothing becomes steady in a closed tube, so there is no convergence in time to wait for; the wave speed itself is steady over the "
            "second half (Figure {fig:xt}).",
        ])
    study_rows = lambda names: [[c["labels"][n], runs[n]["steps"], sp[n]["passes"], m_fix(runs[n]["speed"]["D"], 1), sgn(runs[n]["speed_err"]), m_fix(1e6 * runs[n]["gap_max"], 0),
                                 m_fix(runs[n]["peak"]["ratio"], 3), sgn(runs[n]["energy"]["rate_late"], 3), m_fix(runs[n]["wall_time"], 0)] for n in names]
    study_head = ["Run", "Time steps", "Passes per step", "Wave speed (m/s)", "From the CJ speed", "Largest distance between the fronts (µm)", "Highest pressure / shock pressure",
                  "Energy, second half", "Solver time (s)"]
    seen, order_names = set(), []
    for n in c["steps"] + c["passes"] + c["chemistry"]:
        if n not in seen:
            seen.add(n)
            order_names.append(n)
    study_tab = dict(
        title=f"One setting changed at a time, {Rs['N']} cells",
        head=study_head, rows=study_rows(order_names),
        note="Time step and passes were run in the pre-flight, the chemistry settings in production. One pass per step is the setting of OpenFOAM's reacting examples. "
             "The cut-off is the temperature below which the chemistry skips a cell (" + kel(mn["t_react"]) + " in the main settings: only gas the wave has not reached).")
    behind_tab = dict(
        title=f"The state behind the front at the end time, {N_of(fine)}, next to the two pictures of Theory",
        head=["Distance behind the pressure front (mm)", "Pressure / CJ: simulation", "steady wave", "expansion wave", "Temperature / CJ: simulation", "steady wave", "expansion wave",
              "Water mass fraction", "Gas speed (m/s)"],
        rows=[[m_fix(1e3 * a["d"], 1), m_fix(a["p"] / cj["p"], 3), m_fix(a["p_znd"] / cj["p"], 3), m_fix(a["p_tw"] / cj["p"], 3), m_fix(a["T"] / cj["T"], 3), m_fix(a["T_znd"] / cj["T"], 3),
               m_fix(a["T_tw"] / cj["T"], 3), m_fix(a["H2O"], 4), m_fix(a["u"], 0)] for a in behind],
        note="The steady wave is the structure behind a shock of the CJ speed, with the scheme's reaction rates: it describes the first millimetres, where the gas burns and the pressure "
             "falls from the shock value towards the CJ value. The expansion wave is the ideal picture of a front of no thickness followed by the expansion to rest: it describes "
             "what comes after. The real profile passes from one into the other, and the CJ state is the point between them, not a stretch of constant state. CJ water mass fraction: "
             + m_fix(cj["Y"]["H2O"], 4) + ", CJ gas speed " + m_fix(cj["u"], 0) + " m/s.")

    # ---- comparison --------------------------------------------------------------------------------------------
    defs = [
        dict(name="Wave speed", latex=r"D = \frac{x_p(t_{\mathrm{end}}) - x_p(t_{\mathrm{end}}/2)}{t_{\mathrm{end}}/2}",
             meaning="\\(x_p\\): the pressure front, the furthest point where the pressure is halfway between the start pressure and the CJ pressure, on a straight line between two cell centres."),
        dict(name="Distance between the fronts", latex=r"x_p - x_b", meaning="\\(x_b\\): the burning front, the furthest point where the water mass fraction is half its CJ value. "
             "Judged at each of the sampled states (every half microsecond) from " + f"{1e6 * ext['attached_from']:.0f}" + " microseconds on."),
        dict(name="Burnt state", latex=r"u + c = D", meaning="The first point behind the front where the gas (speed \\(u\\), speed of sound \\(c\\) at fixed composition) moves away from the front "
             "at its own speed of sound. Nothing behind that point can reach the front; in the ideal wave it is the CJ state. The CJ state of the reference uses the speed of sound "
             "of gas that stays in chemical equilibrium, " + mps(tw["c_front"], 0) + "; at fixed composition it is " + mps(c_frozen_cj, 0) + " for the same state. With the lower value the point "
             "would lie closer to the front, where the pressure is higher, so this reading gives a pressure on the low side."),
        dict(name="Highest pressure at the front", latex=r"\frac{\overline{p_{\max}}}{p_{\mathrm{s}}(D)}", meaning="The highest pressure in the tube, averaged over every step of the second half of the time, "
             "over the pressure behind a shock of the simulated speed \\(D\\) in the unburnt gas."),
        dict(name="Change of total energy", latex=r"\frac{E(t) - E(0)}{q(t)}", meaning="\\(E\\): total energy in the tube; \\(q\\): heat released by the burning up to time \\(t\\). "
             "For the second half of the time both differences are taken over that half."),
    ]
    comp_rows = [[r["N"], m_fix(1e6 * r["dx"], 1), sgn(r["speed_err"]), sgn((r["speed"]["D"] - R[i - 1]["speed"]["D"]) / cj["D"]) if i else "–",
                  sgn(r["speed"]["D_min"] / cj["D"] - 1, 1) + " to " + sgn(r["speed"]["D_max"] / cj["D"] - 1, 1), m_fix(r["peak"]["ratio"], 3), m_fix(1e6 * r["gap_max"], 0),
                  sgn(r["end_p"] / tw["p"] - 1), sgn(r["energy"]["rate_late"], 3)] for i, r in enumerate(R)]
    noise_txt = m_pct(c["noise"], 3) if c["noise"] is not None else "–"
    ch1, ch2 = ((R[-2]["speed"]["D"] - R[-3]["speed"]["D"]) / cj["D"], (R[-1]["speed"]["D"] - R[-2]["speed"]["D"]) / cj["D"]) if len(R) >= 3 else (0.0, 0.0)
    parts_names = [n for n in fam] + [n for n in c["steps"] if n != sg]
    parts_tab = dict(
        title="Speed of the pressure front over three stretches of the time, as a difference from the CJ speed",
        head=["Run"] + [f"{1e6 * a * g['t_end']:g} to {1e6 * b * g['t_end']:g} µs" for a, b in zip(PARTS, PARTS[1:])] + ["Second half (the asked speed)"],
        rows=[[f"{runs[n]['N']} cells" + (f", {c['labels'][n]}" if n in c["labels"] and c["labels"][n] != "main" else "")] + [sgn(x) for x in runs[n]["parts"]] + [sgn(runs[n]["speed_err"])]
              for n in parts_names],
        note="The wave starts below the CJ speed and closes in on it from below. From the first to the last stretch the deficit shrinks by " + m_pct(coarse["parts"][2] - coarse["parts"][0], 2)
             + " on the coarsest grid and by " + m_pct(fine["parts"][2] - fine["parts"][0], 2) + " on the finest. The last rows are the time-step study: the run with half the step closes in "
             "more slowly than the main run, as a finer grid does.")
    comp_findings = [
        "<b>Speed.</b> Every grid is inside the required " + m_pct(acc["req_speed"], 0) + ": " + ", ".join(sgn(r["speed_err"]) for r in R) + ". The speed does not approach a limit as the cells shrink"
        + ((": the last two changes are " + sgn((R[-2]["speed"]["D"] - R[-3]["speed"]["D"]) / cj["D"]) + " and " + sgn((R[-1]["speed"]["D"] - R[-2]["speed"]["D"]) / cj["D"])
            + (", not in one direction" if ch1 * ch2 < 0 else ", in the same direction and not shrinking" if abs(ch2) >= 0.9 * abs(ch1) else ", shrinking")) if len(R) >= 3 else "")
        + ". The second table shows why: the wave is still speeding up towards the CJ speed at the end of the time, and it does so more slowly on finer grids. What differs between "
        "the grids is how far the settling has got, so no order and no extrapolated value are given. Changing the chemistry settings moves the speed by " + noise_txt + " only.",
        "<b>Why a coarse grid gets the speed right.</b> The CJ speed follows from the three conservation laws and the heat released. A run that conserves mass, momentum and energy and burns "
        "the gas completely must produce it, whatever the front looks like inside. The energy lost by the solver (" + ", ".join(m_pct(r["energy"]["rate_late"], 2, size=True) for r in R)
        + " of the heat released, coarse to fine) costs little speed" + ((": on " + str(Rs["N"]) + " cells, " + loss_core) if p3 else "."),
        "<b>Front structure.</b> The pressure peak is the part that needs resolution: " + ", ".join(m_fix(r["peak"]["ratio"], 2) for r in R) + " of the shock pressure, coarse to fine. "
        "In the steady wave the pressure stays near the shock value for only " + mmm(zn["x_ind"], 2) + ", less than " + m_fix(zn["x_ind"] / fine["dx"], 0) + " cells of the finest grid (Figure {fig:structure}).",
        "<b>Attached.</b> The burning front follows the pressure front at " + ", ".join(um(r["gap_mean"]) for r in R) + " on average, coarse to fine, which is " + ", ".join(m_fix(x, 1) for x in cells_gap)
        + " cells. In the steady wave of Theory the distance is " + um(zn["x_half_water"]) + "; where the cells are larger than that, shock and burning share one smeared front. A second reading, made at every time step and not only at the sampled times, is the "
        "distance between the point of highest pressure and the point of highest heat release: at most " + ", ".join(mmm(r["gap_step_max"], 2) for r in R)
        + ", coarse to fine (" + ", ".join(m_fix(r["gap_step_max"] / r["dx"], 0) for r in R) + " cells). The heat release peaks at the leading edge of the smeared front and the "
        "pressure a few cells behind it, so this reading grows with the cell size; the requirement is judged on the two fronts as defined before the runs.",
        "<b>Behind the wave.</b> The pressure at the closed end is " + sgn(fine["end_p"] / tw["p"] - 1) + " from the value of the expansion wave, and the temperature there is "
        + kel(fine["end_T"]) + " against " + kel(tw["T"]) + ": the gas near the wall is what the ignition layer left behind, which the ideal picture does not contain.",
    ]

    # ---- performance --------------------------------------------------------------------------------------------
    perf_rows = [[f"{r['N']} cells" + (f", {c['labels'][n]}" if n in c["labels"] and c["labels"][n] != "main" else ""), r["cores"], r["steps_done"], sp[n]["passes"], m_fix(r["wall_time"], 0),
                  m_fix(r["us_per_cell_step"], 1), m_pct(r["burnt_share"], 0), m_fix(r["core_us_per_hot_pass"], 0), m_pct(1 - r["flow_share"], 0) if "flow_share" in r else "–",
                  m_fix(r["lin_iters"].get("p"), 1)] for n, r in runs.items() if "speed" in r]
    performance = dict(
        intro="All full runs used 8 cores (MPI ranks) of one desktop under WSL, with the tube cut into 96 short stretches dealt to the ranks in turn. Solver time is wall-clock time. "
              "The chemistry works only in cells the wave has reached (in every cell in the run without the cut-off temperature), so the cost of a step grows as the wave travels; "
              "one column gives the share of those cells in all cell-steps.",
        head=["Run", "Cores", "Steps", "Passes", "Solver time (s)", "Wall µs per cell and step", "Cells the chemistry works on (share of cell-steps)",
              "Core µs per such cell and pass (everything counted)", "Share not spent on the flow", "Pressure solver iterations"],
        rows=perf_rows,
        notes=[
            ("<b>Where the time goes.</b> On one core over the first fifth of the time (" + f"{Rs['N']} cells), the run took " + m_fix(cost["wall_serial"], 1) + " s with chemistry and "
             + m_fix(cost["wall_nochem"], 1) + " s without, although only " + m_pct(cost["burnt_share_short"], 0) + " of the cell-steps were burnt cells. That is " + m_fix(1e6 * cost["chem"], 0)
             + " microseconds per burnt cell and pass for the chemistry, against " + m_fix(1e6 * cost["flow"], 1) + " per cell and pass for the flow. The last column but one applies this flow "
             "cost to each run: what is left is chemistry and, on 8 ranks, the cost of message passing.") if cost else "<b>Where the time goes.</b> The one-core pair was not run.",
            "<b>Why the chemistry is dear.</b> Every burnt cell is integrated again in each of the five passes of a step, with an explicit method that needs many sub-steps in hot gas near "
            "equilibrium. The implicit method is no way out here: " + ((f"the same run with <code>seulex</code> gave the same wave speed (" + sgn(seu["speed_err"]) + ") and took "
            + m_fix(seu["wall_time"] / Rs["wall_time"], 1) + " times as long (" + m_fix(seu["wall_time"], 0) + " s against " + m_fix(Rs["wall_time"], 0) + " s). ") if seu else "it was not run. ")
            + ((f"A looser tolerance of the explicit method saved " + m_pct(1 - rel["wall_time"] / Rs["wall_time"], 0) + " of the time for the same speed (" + sgn(rel["speed_err"]) + "). ") if rel else "")
            + (("Without the cut-off temperature every cell is integrated in every pass, burnt or not: " + m_fix(allc["wall_time"], 0) + " s against " + m_fix(Rs["wall_time"], 0) + " s.") if allc else ""),
            "<b>Parallel runs gain little on small grids.</b> In short trial runs before the plan was fixed (insight 11; not among the runs of this table), a step on 8 ranks cost about "
            "25 milliseconds more than the work it held, on 400 cells as on 1200, so on 1200 cells 8 ranks were only twice as fast as one core. Without the cut into stretches they were "
            "no faster at all, because all the burnt gas sat in the first ranks."
            + ((" In the runs of this table the same shows as the cost per burnt cell and pass: " + m_fix(1e6 * cost["chem"], 0) + " microseconds on one core, "
                + m_fix(Rs["core_us_per_hot_pass"], 0) + f" core-microseconds on 8 ranks ({Rs['N']} cells).") if cost else ""),
            "<b>What would make it faster.</b> Solving the chemistry once per step with the heat release held over the passes (not possible in the stock solver); a density-based solver that "
            "conserves energy in one pass; tabulated or reduced chemistry; the chemistry of all cells on the GPU. None was tried here: the task was to solve with the stock solver.",
            "All runs together took " + m_fix(c["total_wall"] / 60, 1) + " minutes of solver time; the plan first drawn up (finest grid 9600 cells) would have taken over an hour.",
        ])

    # ---- figures ---------------------------------------------------------------------------------------------------
    svgs = dict(sketch=sketch(), xt=figs.xt(c), profiles=figs.profiles(c), structure=figs.structure(c), attached=figs.attached(c), conservation=figs.conservation(c),
                convergence=figs.convergence(c))
    figures = [
        dict(key="sketch", section="problem",
             caption="The problem as a picture (not a result). Top: the closed tube with the wave on its way; the ignition layer at the left end started it. Bottom: the pressure along "
                     "the tube at the same moment, in outline: a jump at the front, a fall while the gas burns, an expansion, and gas at rest again towards the closed end."),
        dict(key="xt", section="simulation",
             caption=f"Left: position of the pressure front (line) and of the burning front (circles) against time on {N_of(fine)}; they cannot be told apart at this scale. The dashed line has "
                     "the slope of the CJ speed; the shaded half is where the wave speed is read. Right: speed of the pressure front over 2 microseconds, as a difference from the CJ speed, on "
                     "every grid; the shaded band is the required accuracy. " + start_txt),
        dict(key="profiles", section="simulation",
             caption=f"Pressure (left) and temperature (right) along the tube at five times, {N_of(fine)}. The wave keeps its height and shape and leaves a growing stretch of gas at rest behind it. "
                     "The dashed line is the ideal wave of Theory at the end time (a front of no thickness followed by the expansion to rest), drawn with its front at the simulated front. "
                     "The hot stretch at the closed end is what is left of the ignition layer."),
        dict(key="structure", section="comparison",
             caption="The state behind the front at the end time on every grid, as pressure (left) and temperature (right) over their CJ values against the distance behind the pressure front. "
                     "The dashed line is the steady wave of Theory with its sharp shock and the wait before burning; the dotted line is the CJ value. Circles mark where the burnt state is read "
                     "(the gas leaves the front at its own speed of sound). The grids smear the shock and the burning into one front of a few cells and cut off the pressure peak."),
        dict(key="attached", section="comparison",
             caption="Left: distance between the pressure front and the burning front against time, on a logarithmic scale; the dashed line is the required limit of 1 mm and the dotted line "
                     "the time from which it applies. Right: the highest pressure in the tube over the pressure behind a shock of the CJ speed (dashed: that pressure; dotted: the CJ pressure). "
                     "The spike at the start is the ignition layer, which is unburnt mixture and burns first, reaching " + mpa(max(r["p_max"] for r in R), 1) + ". Finer grids show more of the peak."),
        dict(key="conservation", section="simulation",
             caption="Change of the total energy in the tube as a percentage of the heat released so far. Left: the four grids with five passes per step. Right: the effect of the number of "
                     f"passes per step on {Rs['N']} cells. " + ("One pass, the setting of OpenFOAM's reacting examples, lies below the frame: it has lost " + m_pct(p1["energy"]["end"], 1, size=True)
                                                                 + " at the end. " if p1 else "") + "The curves start at 2 microseconds, when enough heat has been released for the ratio to mean something."),
        dict(key="convergence", section="convergence",
             caption="Wave speed as a difference from the CJ speed; the shaded band is the required accuracy. Left: against the cell size. Right: one setting changed at a time on "
                     f"{Rs['N']} cells (time step, passes per step, chemistry solver, its tolerance, the cut-off temperature)."
                     + (" The run with one pass per step, at " + sgn(p1["speed_err"], 1) + ", is off the scale and is left out." if p1 and abs(p1["speed_err"]) >= 0.026 else "")),
    ]

    results = dict(
        title=problem["title"], date=dt.date.today().isoformat(), verdict=verdict, expectations=X, comparison_type="verification",
        summary=dict(headline=headline, answer_columns=["Requested", "Theory (Chapman-Jouguet reference)", f"Simulation (OpenFOAM, {N_of(fine)})", "Difference"],
                     answers=answers, findings=findings, limitations=limitations),
        problem=dict(
            description=" ".join(problem["description"].split()), setup=problem.get("setup", []),
            given=[dict(label=v["label"], symbol=v["symbol"], value=m(tex_num(float(v["value"]), 6)), unit=v["unit"]) for v in problem["given"].values()],
            wanted=wanted, required_accuracy=problem["required_accuracy"],
            reference="Supplied with the problem: " + " ".join(problem["reference"]["description"].split())),
        assumptions=dict(rows=assumptions),
        theory=theory.report_theory(th, g, numerics),
        decisions=dec,
        preflight=preflight,
        simulation=dict(model=model, meshes=meshes, results=results_tab, health=health, tables=[study_tab, behind_tab]),
        comparison=dict(
            kind="Verification against theory: the Chapman-Jouguet speed and state of the reference file follow from conservation and the thermodynamic data alone, and the simulated "
                 "wave must settle to them. The steady structure and the expansion wave of Theory stand next to the simulated profiles.",
            definitions=defs,
            head=["Cells", "Cell size (µm)", "Wave speed: from the CJ speed", "Change from the grid before", "Speed over each half microsecond, second half", "Highest pressure / shock pressure",
                  "Largest distance between the fronts (µm)", "Pressure at the closed end: from Theory", "Energy, second half"],
            rows=comp_rows, findings=comp_findings, tables=[parts_tab]),
        insights=load_yaml(HERE / "insights.yaml"),
        figures=figures,
        performance=performance,
        reproducibility=dict(
            commands=[
                dict(cmd=f"python -m cases.{HERE.name}.theory", what="The CJ state, the state behind the shock, the steady structure, the expansion wave, the SymPy checks"),
                dict(cmd=f"python -m cases.{HERE.name}.run preflight", what="Two short tests, the one-core pair, and the studies of time step and passes in full"),
                dict(cmd=f"python -m cases.{HERE.name}.run production", what="The other grids and the chemistry variants, one at a time; writes runs/*.json and runs/*.npz"),
                dict(cmd=f"python -m cases.{HERE.name}.analyze", what="The rules, the numbers, the figures and this report (checks the contract first)"),
            ],
            files=[
                dict(path=f"cases/{HERE.name}/problem.yaml", what="The problem as given"),
                dict(path=f"cases/{HERE.name}/reference/cj.yaml", what="The reference as given"),
                dict(path=f"cases/{HERE.name}/numerics.yaml", what="Runs, reading rules, and the rules for the outcome written before the production runs"),
                dict(path=f"cases/{HERE.name}/decisions.yaml", what="Every choice, with the OpenFOAM settings it controls"),
                dict(path=f"cases/{HERE.name}/insights.yaml", what="Insights logged during the work"),
                dict(path=f"cases/{HERE.name}/template/", what="The OpenFOAM case with {{ placeholders }}"),
                dict(path=f"cases/{HERE.name}/theory.py", what="Theory numbers and SymPy checks"),
                dict(path=f"cases/{HERE.name}/measure.py", what="The rules that read fronts, speed, burnt state and totals from the solver's output"),
                dict(path=f"cases/{HERE.name}/run.py", what="Builds, pre-flights and runs the cases"),
                dict(path=f"cases/{HERE.name}/analyze.py, figs.py, sketch.py", what="The analysis, the figures and the sketch"),
            ],
            run_dirs=[
                dict(path=f"{fine['run_root']}/N<cells>[_s<steps>][_p<passes>][_seulex|_rel<tolerance>|_allcells|_serial[_nochem]]", what="Full runs and the one-core pair"),
                dict(path=f"{fine['run_root']}/pf_N*_preflight", what="Short pre-flight runs"),
                dict(path=f"results/{HERE.name}/", what="theory.json, preflight.json, runs/, logs/, results.json and this report"),
            ],
            environment=report.environment_info(),
            notes=["Runs were made one at a time. Another session was working on the same machine, so run times are rough.",
                   "The OpenFOAM case reads the reaction scheme from OpenFOAM's own tutorial folder; nothing is copied into this repository.",
                   "Earlier trial runs that led to the settings (other limiters, passes, chemistry solvers, a 1200-cell grid) are described in the insights; their folders are named trial_* and ode_*."]),
        data=dict(
            reference=dict(speed=cj["D"], p=cj["p"], T=cj["T"], p_shock=fr["p"], p_end=tw["p"]),
            runs={n: {k: v for k, v in r.items() if k in ("N", "steps", "passes", "ode", "rel", "t_react", "cores", "dx", "dt", "speed", "speed_err", "gap_max", "gap_mean", "gap_step_max",
                                                             "peak", "burnt", "end_p", "end_T", "energy", "energy_startup", "mass_dev", "atoms_dev", "wall_time", "us_per_cell_step",
                                                             "hot_cell_steps", "core_us_per_hot_pass", "flow_share", "front_last", "energy_last")} for n, r in runs.items()},
            grid_study=json.loads(json.dumps(gr, default=float)) if gr else None, noise=c["noise"], serial=ser, cost=cost, behind=behind, start=s0),
    )
    out = report.render(results, svgs, OUT / "report.html", case_dir=fine["case_dir"])
    print(f"{verdict}: missed {[x['text'] for x in missed]}")
    for x in X:
        print(f"   {'ok  ' if x['met'] else 'MISS'} [{x['kind']}] {x['text'][:90]} | expected {x['expected'][:70]} | found {x['found'][:170]}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
