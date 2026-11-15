"""Analysis and report for the shock at the corner.

  python -m cases.shock_diffraction_corner.analyze
Reads results/shock_diffraction_corner/runs and preflight.json, applies the rules written in numerics.yaml
before the production runs, draws the figures and writes results.json and report.html (after the contract check).
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np

from cfdlib import report
from cfdlib.report import LETTERS, lines, load_yaml, m, m_fix, m_pct, m_sig, m_times, tex_num
from . import figs, numbers
from .run import OUT
from .sketch import sketch

HERE = Path(__file__).resolve().parent
REFS = dict(
    quirk="J. J. Quirk, A contribution to the great Riemann solver debate, Int. J. Numer. Methods Fluids 18 (1994) 555",
    cheng="Cheng, Sheng and Wang, Energies 17 (2024) 769, section 4.2",
    whitham="G. B. Whitham, A new approach to problems of shock dynamics, J. Fluid Mech. 2 (1957) 145",
    skews="B. W. Skews, The perturbed region behind a diffracting shock wave, J. Fluid Mech. 29 (1967) 705",
    knp="Kurganov, Noelle and Petrova, SIAM J. Sci. Comput. 23 (2001) 707")


def sc(x, sig=2):
    return m(tex_num(x, sig))


def sgn(x, dec=2):
    """A signed relative difference as per cent."""
    return m_pct(x, dec, signed=True)


def main():
    c = numbers.compute()
    problem, numerics, th, fl = c["problem"], c["numerics"], c["theory"], c["fl"]
    runs, data, fam, pairs, conv, bias = c["runs"], c["data"], c["family"], c["pairs"], c["conv"], c["bias"]
    acc, ext = numerics["acceptance"], numerics["extraction"]
    pf = json.loads((OUT / "preflight.json").read_text())
    R = [runs[n] for n in fam]
    fine, coarse = R[-1], R[0]
    Nf = fine["N"]
    metas = {n: json.loads((OUT / "runs" / f"{n}.json").read_text()) for n in runs}
    t_end = fl.t_end
    sch_names = [n for n in runs if "scheme" in c["specs"][n]["studies"]]
    ts_names = sorted((n for n in c["specs"] if "time step" in c["specs"][n]["studies"]), key=lambda n: c["specs"][n]["spc"])
    stress = [n for n in c["specs"] if "stress" in c["specs"][n]["studies"]]
    main_ts = f"N{numerics['runs']['timestep']['cells']}"
    planned = [f"N{N}" for N in numerics["runs"]["family"]]
    all_ran = fam == planned

    # ---- the rules written before the production runs ---------------------------------------------------------
    X = []

    def rule(kind, text, expected, found, met, key=None):
        x = dict(kind=kind, text=text, expected=expected, found=found, met=bool(met))
        if key:
            x["key"] = key
        X.append(x)

    req = {q["key"]: q["text"] for q in problem["required_accuracy"]}
    st = fine["state"]
    rule("requirement", req["flat_part"],
         "speed within " + m_pct(acc["req_flat_speed"], 1) + "; pressure, density and velocity within " + m_pct(acc["req_flat_state"], 0),
         f"{Nf} cells per side: speed " + sgn(fine["speed_err"], 3) + "; pressure " + sgn(st["p"]["err"], 3) + ", density "
         + sgn(st["rho"]["err"], 3) + ", velocity " + sgn(st["u"]["err"], 3),
         all_ran and abs(fine["speed_err"]) <= acc["req_flat_speed"] and max(abs(st[k]["err"]) for k in st) <= acc["req_flat_state"],
         key="flat_part")
    worst_m, worst_e = max(r["balance"]["mass_vs_inflow"] for r in R), max(r["balance"]["energy_vs_inflow"] for r in R)
    rule("requirement", req["conserved"], "within " + m_sig(acc["req_conserved"], 1, pct=True, unit=True) + " of the inflow so far, after every step of every grid",
         "largest difference over every step of the " + str(len(R)) + " grids: mass " + sc(worst_m, 1) + ", energy " + sc(worst_e, 1)
         + " of the inflow so far", all_ran and max(worst_m, worst_e) <= acc["req_conserved"], key="conserved")
    worst_dev = max(r["dev_cells_max"] for r in R)
    rule("requirement", req["stable"], "largest deviation from a straight line at most " + m_fix(acc["req_straight_cells"], 0) + " cell, every grid, every written time",
         "largest deviation " + sc(worst_dev, 1) + " cells (" + ", ".join(sc(r["dev_cells_max"], 1) for r in R) + " on "
         + ", ".join(str(r["N"]) for r in R) + " cells per side)", all_ran and worst_dev <= acc["req_straight_cells"], key="stable")
    rule("health", "Every run of the grid study ends at the time of interest without a solver error, on a grid that passes checkMesh",
         f"{len(planned)} runs finished", f"{len(R)} of {len(planned)} finished; solver log checks: " + ", ".join(r["health"] for r in R)
         + "; checkMesh: " + ("OK on every grid" if all(metas[n]["mesh_ok"] for n in fam) else "not OK"),
         all_ran and all(r["health"] == "ok" for r in R) and all(metas[n]["mesh_ok"] for n in fam))
    rule("health", "Density, pressure and temperature stay positive at every written time", "all positive",
         "lowest pressure " + ", ".join(m_fix(r["lows"]["p"] / 1e3, 1) for r in R) + " kPa and lowest density "
         + ", ".join(m_fix(r["lows"]["rho"], 3) for r in R) + r" \(\mathrm{kg/m^3}\) on " + ", ".join(str(r["N"]) for r in R) + " cells per side",
         all(min(r["lows"].values()) > 0 for r in R))
    wt = max(max(r["balance"]["mass_vs_total"], r["balance"]["energy_vs_total"]) for r in R)
    rule("expectation", "Mass and energy balance to round-off: the difference from (start + exact inflow), as a fraction of the total in the chamber, after every step",
         "at most " + sc(acc["exp_balance_total"], 1), "at most " + sc(wt, 1), wt <= acc["exp_balance_total"])
    wi = max(r["balance"]["inflow_rate"] for r in R)
    rule("expectation", "The mass flow through the opening is the exact one at every step", "within " + sc(acc["exp_inflow_rate"], 1) + " (relative)",
         "within " + sc(wi, 1), wi <= acc["exp_inflow_rate"])
    wf = max(max(r["far"]["floor"], r["far"]["right"]) for r in R)
    rule("expectation", "No wave has reached the floor or the right side at the time of interest (the flow still has no length of its own)",
         r"pressure there within " + sc(acc["exp_far_walls"], 1) + r" of \(p_0\)", "largest difference " + (sc(wf, 1) if wf > 0 else "exactly zero") + " on every grid",
         wf <= acc["exp_far_walls"])
    wd = max(p["discrete"] for p in pairs)
    rule("expectation", "The method has no length of its own either: the density at half the time on a grid equals the density at the full time on the next coarser grid, cell for cell",
         "largest difference at most " + sc(acc["exp_discrete_similarity"], 1) + r" of \(\rho_0\), every pair of grids",
         ", ".join(sc(p["discrete"], 1) for p in pairs) + " for the pairs " + ", ".join(f"{runs[p['fine']]['N']}/{runs[p['coarse']]['N']}" for p in pairs),
         wd <= acc["exp_discrete_similarity"])
    wtop = max(abs(r["x_top_err_cells"]) for r in R)
    rule("expectation", "The shock on the top wall is where the exact flat shock is", "within " + m_fix(acc["exp_top_cells"], 1) + " cells on every grid",
         ", ".join(m(f"{r['x_top_err_cells']:+.2f}") for r in R) + " cells", wtop <= acc["exp_top_cells"])
    wa = max(r["p_row_spread"] for r in R)
    rule("expectation", "No change of pressure along the shock just behind its flat part (where an instability of the shock would start)",
         "at most " + sc(acc["exp_along_shock"], 1) + r" of \(p_2\)", "at most " + sc(wa, 1), wa <= acc["exp_along_shock"])
    wv = max(r["v_max"] for r in R)
    rule("expectation", "No sideways velocity in the uniform region behind the flat part", r"\(|v|/u_2\) at most " + sc(acc["exp_v_behind"], 1),
         "at most " + sc(wv, 1), wv <= acc["exp_v_behind"])
    wr = max(r["state"]["p"]["ripple"] for r in R)
    rule("expectation", "Ripple of the pressure behind the flat part (highest minus lowest in the uniform region) at the time of interest",
         "at most " + m_pct(acc["exp_ripple_p"], 0) + r" of \(p_2\) on every grid", ", ".join(m_pct(r["state"]["p"]["ripple"], 2) for r in R),
         wr <= acc["exp_ripple_p"])
    dchg = abs(R[-1]["d_wall"] - R[-2]["d_wall"]) / R[-1]["d_wall"] if len(R) > 1 else float("nan")
    rule("expectation", "The distance the bent shock has run down the step face has settled with the grid",
         "change from 400 to 800 cells at most " + m_pct(acc["exp_wall_shock_change"], 1), f"change from {R[-2]['N']} to {R[-1]['N']} cells: " + m_pct(dchg, 2),
         all_ran and dchg <= acc["exp_wall_shock_change"])
    fo = conv.get("field_orders", [])
    rule("expectation", "The change of the density field from one grid to the next falls with refinement", "order at least " + m_fix(acc["exp_field_order"], 1) + " at each refinement",
         "orders " + ", ".join(m_fix(o, 2) for o in fo) + " (changes " + ", ".join(m_pct(p["field_change"], 2) for p in pairs) + ")",
         len(fo) > 0 and min(fo) >= acc["exp_field_order"])
    ratio = fine["sim_wall"] / runs[main_ts]["sim_wall"]
    rule("expectation", "The same picture at two times, density along the step face: the difference shrinks with the grid",
         f"on {Nf} cells at most " + m_fix(acc["exp_two_times_wall"], 1) + " of the difference on " + f"{runs[main_ts]['N']} cells",
         m_pct(fine["sim_wall"], 1) + " against " + m_pct(runs[main_ts]["sim_wall"], 1) + ": a factor " + m_fix(ratio, 2),
         all_ran and ratio <= acc["exp_two_times_wall"])
    rule("expectation", "Solver time of the finest run", "at most " + f"{acc['exp_finest_minutes']} minutes",
         m_fix(fine["wall_time"] / 60, 1) + f" minutes on {fine['cores']} cores", all_ran and fine["wall_time"] / 60 <= acc["exp_finest_minutes"])
    sch_ok = [n for n in sch_names if n != main_ts]
    sch_crashed = [n for n in c["crashed"] if "scheme" in c["specs"][n]["studies"]]
    rule("expectation", "Each of the four other schemes on 200 cells runs to the end, stays positive and keeps the flat part straight to within one cell",
         "4 of 4", f"{sum(1 for n in sch_ok if min(runs[n]['lows'].values()) > 0 and runs[n]['dev_cells_max'] <= 1)} of {len(sch_ok) + len(sch_crashed)}"
         + (" (stopped by the solver: " + ", ".join(c["specs"][n]["flux"] + " with " + c["specs"][n]["limiter"] for n in sch_crashed) + ")" if sch_crashed else ""),
         len(sch_ok) == 4 and not sch_crashed and all(min(runs[n]["lows"].values()) > 0 and runs[n]["dev_cells_max"] <= 1 for n in sch_ok))
    verdict = report.verdict_from(X)
    missed = [x for x in X if not x["met"]]

    # ---- numbers used in several places ------------------------------------------------------------------------
    ex = fine["extra"]
    wall_extr = conv.get("d_wall")
    d_wall, Mw_sim = fine["d_wall"], fine["Mw_dist"]
    whitham_diff = fl.d_wall / d_wall - 1
    us_fine = fine["us_per_cell_step"]
    timing = {r["ranks"]: r for r in pf["timing"]}
    next_cost = 8 * fine["wall_time"]
    crashed_ts = [c["crashed"][n] for n in ts_names if n in c["crashed"]]
    ts_ok = [runs[n] for n in ts_names if n in runs]
    startup = [(fl.u2 - fl.c2) * t_end, fl.u2 * t_end]
    fan_w = [r["extra"]["fan_wiggle"] for r in R]

    # ---- figures ---------------------------------------------------------------------------------------------
    d_f = data[fam[-1]]
    yw = fl.Hs - d_wall
    fr = d_f["front_low"][-1]
    row = lambda y: int(y * Nf / fl.H)
    labels = [
        ("flat shock", (fl.x_top, 0.9), (0.91, 0.9), 0),
        ("bent shock", (float(fr[row(0.3)]), 0.3), (0.82, 0.22), 0),
        ("bent shock\non the step face", (0.0, yw), (0.2, 0.05), 0),
        ("expansion fan", (0.1, 0.5), (0.2, 0.72), 0),
        ("still air", (0.9, 0.55), (0.9, 0.55), 0),
        ("second shock", (0.33, 0.375), (0.66, 0.2), 1),
        ("layer from\nthe corner", (0.07, 0.42), (0.36, 0.045), 1),
        ("inner border", (ex["contact"]["x"], ex["contact"]["y"]), (0.91, 0.62), 1),
        ("start-up\nwaves", (startup[1], 0.955), (0.91, 0.93), 1),
    ]
    zoom_names = fam[-3:]
    svgs = dict(
        sketch=sketch(fl),
        map=figs.flow_map(d_f, fl, Nf, labels),
        front=figs.front_similarity(d_f, fl, Nf),
        corner=figs.corner_zoom(data, fl, zoom_names),
        conservation=figs.conservation(data, runs, fam, fl),
        flat=figs.flat_part(data, runs, fl, fam),
        similar=figs.two_times(d_f, fl, Nf, numerics["snapshots"]),
        convergence=figs.convergence(runs, pairs, fam, fl),
    )
    figures = [
        dict(key="sketch", section="problem",
             caption="The problem. A flat shock comes through the duct and reaches the opening of the chamber at \\(t = 0\\). Above the "
                     "corner it runs on as a flat shock; below, it bends round the corner into the still air. The picture is a sketch "
                     "of an early time, not a result."),
        dict(key="map", section="simulation",
             caption=f"The flow at \\(t_{{\\mathrm{{end}}}}\\) on {Nf} cells per side. Left: density. Right: how fast the density changes from "
                     "cell to cell, which shows every wave as a line (dark = steep). Dashed: the exact edge of the region that the corner "
                     "can have influenced behind the flat shock; outside it the flow must be uniform, and it is, except for the two thin "
                     "vertical lines, which the start of the shock left behind."),
        dict(key="front", section="simulation",
             caption=f"The shock front at the ten written times on {Nf} cells per side, each divided by its time. The ten curves fall on "
                     "one, as they must for a flow without a length of its own; the early ones (lighter) are the same curve drawn "
                     "with fewer cells. Dashed: the exact speed of the flat shock. Dotted: the exact height above which the shock is still flat."),
        dict(key="corner", section="simulation",
             caption="The region below the corner on the three finest grids. The shocks and the fan are in the same place on all three. "
                     "The layer that leaves the corner, across which the speed jumps, gets thinner with every refinement: without "
                     "viscosity it has no thickness of its own, and the grid sets it. The highest Mach number in the fan is "
                     + ", ".join(m_fix(runs[n]["extra"]["mach_max"], 1) for n in zoom_names) + " and the lowest pressure at the time of interest "
                     + ", ".join(m_fix(runs[n]["extra"]["p_min"] / 1e3, 0) for n in zoom_names) + " kPa on the three grids."),
        dict(key="conservation", section="simulation",
             caption="Mass and energy in the chamber against the start plus the exact inflow, after every step, as a fraction of the "
                     "inflow so far. Largest value on any grid: " + sc(max(worst_m, worst_e), 1) + ", at the first step, where the inflow so far is "
                     "smallest; it is round-off of the total. The required limit is the dashed line."),
        dict(key="flat", section="comparison",
             caption=f"Left: pressure, density and velocity along the top row of cells at \\(t_{{\\mathrm{{end}}}}\\), divided by the exact values "
                     f"behind the shock ({Nf} cells per side). They are exact to the plotted accuracy except at two places, the two "
                     "waves that the start of the shock leaves behind; the dotted lines mark where theory puts such waves. Right: the "
                     "position of the shock in every row above the corner, minus the exact flat shock, in cells. Above the dotted "
                     "line the shock must be flat."),
        dict(key="similar", section="comparison",
             caption=f"The same picture at two times ({Nf} cells per side): density along the step face and along the line \\(y = H_s\\), "
                     "against distance divided by time. The mean difference between the two curves is " + m_pct(fine["sim_wall"], 1)
                     + " (step face) and " + m_pct(fine["sim_line"], 1) + " (line) of the mean density."),
        dict(key="convergence", section="convergence",
             caption="Left: how much the density changes from one grid to the next (mean over the cells, as a fraction of the mean "
                     "density), for the whole field and for the two lines of the two-times check. Right: the distance the bent shock "
                     "has run down the step face against the cell size; the labels are the cells per side."),
    ]

    # ---- summary ---------------------------------------------------------------------------------------------
    wanted = [dict(letter=LETTERS[i], key=q["key"], text=q["text"]) for i, q in enumerate(problem["quantities_of_interest"])]
    L = {w["key"]: w["letter"] for w in wanted}
    wr_ = fine["written"]
    answers = [
        dict(letter=L["flat_part"], label="Flat part of the shock: speed and the state behind it",
             theory=lines("speed " + m(f"{fl.W:.1f}") + r" \(\mathrm{m/s}\)", "pressure " + m(f"{fl.p2 / 1e6:.4f}") + r" \(\mathrm{MPa}\)",
                          "density " + m(f"{fl.rho2:.4f}") + r" \(\mathrm{kg/m^3}\)", "velocity " + m(f"{fl.u2:.1f}") + r" \(\mathrm{m/s}\)"),
             sim=lines(m(f"{fine['speed']:.1f}") + r" \(\mathrm{m/s}\)", m(f"{st['p']['mean'] / 1e6:.4f}") + r" \(\mathrm{MPa}\)",
                       m(f"{st['rho']['mean']:.4f}") + r" \(\mathrm{kg/m^3}\)", m(f"{st['u']['mean']:.1f}") + r" \(\mathrm{m/s}\)"),
             diff=lines(sgn(fine["speed_err"], 3), sgn(st["p"]["err"], 3), sgn(st["rho"]["err"], 3), sgn(st["u"]["err"], 3))),
        dict(letter=L["conserved"], label=r"Mass and energy in the chamber at \(t_{\mathrm{end}}\), per metre of depth",
             theory=lines("mass " + m(f"{wr_['mass_exact'][-1]:.6f}") + r" \(\mathrm{kg}\)", "energy " + m(f"{wr_['energy_exact'][-1] / 1e6:.6f}") + r" \(\mathrm{MJ}\)"),
             sim=lines(m(f"{wr_['mass'][-1]:.6f}") + r" \(\mathrm{kg}\)", m(f"{wr_['energy'][-1] / 1e6:.6f}") + r" \(\mathrm{MJ}\)")
                 + "<br>Right: the largest difference at any step of any grid, as a fraction of the inflow so far.",
             diff=lines("mass " + sc(worst_m, 1), "energy " + sc(worst_e, 1))),
        dict(letter=L["stable"], label="Is the flat part straight and free of instability?",
             theory="a straight line; uniform pressure behind it",
             sim="Yes. Largest deviation from a straight line on any grid at any written time: " + sc(worst_dev, 1) + " cells. Pressure "
                 "along the shock changes by " + sc(wa, 1) + r" of \(p_2\) at most. The pressure behind it ripples by " + m_pct(st["p"]["ripple"], 1)
                 + " along the flow, in two thin waves left by the start of the shock.",
             diff=lines(sc(worst_dev, 1) + " cells", "ripple " + m_pct(st["p"]["ripple"], 1))),
        dict(letter=L["similar"], label="The same picture at two times: how much the density differs",
             theory="no difference",
             sim="Mean difference of the density, as a fraction of the mean density along the line: " + m_pct(fine["sim_wall"], 1)
                 + r" along the step face, " + m_pct(fine["sim_line"], 1) + r" along \(y = H_s\). It is the change from the "
                 f"{R[-2]['N']}-cell grid to the {Nf}-cell grid, and falls with refinement along the step face.",
             diff=lines("step face " + m_pct(fine["sim_wall"], 1), "line " + m_pct(fine["sim_line"], 1))),
        dict(letter=L["pattern"], label=r"The flow at \(t_{\mathrm{end}}\): the waves, and how far the shock has run",
             theory=lines("top wall: " + m(f"{fl.x_top:.4f}") + r" \(\mathrm{m}\)", "step face: " + m(f"{fl.d_wall:.3f}") + r" \(\mathrm{m}\)")
                    + "<br>(top wall exact; step face an estimate by Whitham's theory)",
             sim=lines("top wall: " + m(f"{fine['x_top']:.4f}") + r" \(\mathrm{m}\)", "step face: " + m(f"{d_wall:.4f}") + r" \(\mathrm{m}\)")
                 + "<br>(distance below the corner). The waves are named in Figure {fig:map}.",
             diff=lines(m(f"{fine['x_top_err_cells']:+.2f}") + " cells", "estimate " + sgn(whitham_diff, 0))),
        dict(letter=L["grid"], label="Change with the grid, and the cost of a finer answer",
             theory="–",
             sim=f"From {R[-2]['N']} to {Nf} cells per side the density field changes by " + m_pct(pairs[-1]["field_change"], 2)
                 + " on average and the distance down the step face by " + m_pct(dchg, 2) + ". The flat part and the totals are exact on every grid. "
                 f"A grid twice as fine costs 8 times more: about " + m_fix(next_cost / 3600, 1) + f" hours on {fine['cores']} cores of this PC.",
             diff="order " + ", ".join(m_fix(o, 1) for o in fo)),
    ]
    head_tag = {"PASS": "Completed, as expected.", "WARN": "Completed, with differences from what was expected.", "FAIL": "Failed."}[verdict]
    headline = (f"<b>{head_tag}</b> The three requirements are met: on {Nf} cells per side the flat part of the shock moves at the exact "
                "speed to within " + m_pct(fine["speed_err"], 3, size=True) + ", the state behind it is within "
                + m_pct(max(abs(st[k]["err"]) for k in st), 3, size=True) + ", mass and energy match the exact inflow to "
                + sc(max(worst_m, worst_e), 1) + " of it, and the flat part is straight to " + sc(worst_dev, 1) + " cells on every grid."
                + (" Not as expected: " + "; ".join(x["text"][0].lower() + x["text"][1:] for x in missed if x["kind"] == "expectation") + "."
                   if any(x["kind"] == "expectation" for x in missed) else ""))
    findings = [
        "<b>The flat shock is exact.</b> Speed, pressure, density and velocity agree with the jump conditions to "
        + m_pct(max(abs(fine["speed_err"]), max(abs(st[k]["err"]) for k in st)), 2, size=True) + f" on {Nf} cells, and to "
        + m_pct(max(max(abs(coarse["speed_err"]), max(abs(coarse["state"][k]["err"]) for k in st)), 0), 2, size=True) + f" already on {coarse['N']} cells.",
        "<b>Nothing is lost or made.</b> After every step of every run the mass and energy in the chamber equal the start plus the "
        "exact inflow, to round-off (" + sc(wt, 1) + " of the total).",
        "<b>No instability of the shock.</b> The flat part stays straight to " + sc(worst_dev, 1) + " cells, and the pressure does not "
        "change along it. This solver's fluxes smear a little in every direction, which is what prevents the break-up that sharper "
        "methods show on this test.",
        "<b>Two things in the picture are made by the method, not by the flow.</b> The start of the shock leaves two thin waves behind "
        "the flat part (" + m_pct(st["p"]["ripple"], 1) + " in pressure, " + m_pct(st["rho"]["ripple"], 1) + " in density; they get thinner with the grid, "
        "not weaker). And the expansion fan at the corner carries fine stripes: the density along an arc round the corner goes up and "
        "down by " + ", ".join(m_pct(max(w, 0.0), 1) for w in fan_w) + " of its fall on " + ", ".join(str(r["N"]) for r in R) + " cells.",
        "<b>The pattern.</b> A flat shock, a bent shock that has run " + m(f"{d_wall:.3f}") + " m down the step face (Mach number " + m_fix(Mw_sim, 2)
        + " there, against " + m_fix(fl.Ms, 2) + " for the flat part), an expansion fan at the corner in which the gas reaches Mach "
        + m_fix(ex["mach_max"], 1) + " and " + m_fix(ex["p_min"] / 1e3, 0) + " kPa, a second shock that ends the fan, and a thin layer that leaves the "
        "corner, with slowly turning gas between it and the step face.",
        "<b>The time step has a hard limit here.</b> At twice the step used, the solver stopped with a negative pressure in the fan (200 cells); at "
        "1.5 times it ran, with the same answers (200 and 400 cells). The lowest pressure is " + ", ".join(m_fix(r["lows"]["p"] / 1e3, 0) for r in R)
        + " kPa on " + ", ".join(str(r["N"]) for r in R) + " cells.",
        "<b>Whitham's estimate</b> of the bent shock on the step face is " + m_pct(abs(whitham_diff), 0) + " too far (Mach " + m_fix(fl.Mw, 2)
        + " against " + m_fix(Mw_sim, 2) + " found).",
    ]
    limitations = [
        "The layer that leaves the corner is as thin as the grid makes it (Figure {fig:corner}): the gas has no viscosity, so the layer "
        "has no thickness of its own. On finer grids such a layer is known to break up into eddies (from the literature, from memory); "
        "on these grids it does not, so the picture below the corner may still change in kind. The highest Mach number in the fan is "
        "still rising (" + ", ".join(m_fix(r["extra"]["mach_max"], 1) for r in R) + ").",
        "The bent part of the flow has no exact solution and no reference data came with the problem. It is checked for consistency "
        "only: with itself at two times, between grids, and against two approximate theories.",
        "The answer is for a gas with constant specific heats and no viscosity, as given. Air at the " + m(f"{fl.T2:.0f}") + " K behind this "
        "shock does not keep constant specific heats.",
        "One solver and one family of grids. A second method with sharper shocks would show whether the stripes in the fan and the two "
        "start-up waves belong to this scheme.",
        "The sources of the test and of Whitham's theory are cited from memory; none was supplied with the problem.",
    ]

    # ---- assumptions -----------------------------------------------------------------------------------------
    assumptions = [
        dict(name="Ideal gas with a constant ratio of specific heats, no viscosity, no heat conduction", status="imposed",
             check="Given with the problem. It makes the jump conditions exact and removes every length from the flow."),
        dict(name="Two-dimensional flow", status="imposed", check="Given. One layer of cells; nothing can vary across it."),
        dict(name="The air entering through the opening stays in the state behind the flat shock", status="ok",
             check="The gas there moves faster than sound, " + m(rf"M_2 = {fl.M2:.3f}") + ", so nothing can travel back to the opening. In the "
                   "simulation the lowest Mach number in the cells along the opening is " + m_fix(min(r["extra"]["inlet_mach_min"] for r in R), 3)
                   + ", and the mass flow through the opening equals the exact one to " + sc(wi, 1) + " at every step."),
        dict(name="No wave reaches the top wall, the floor or the right side before the time of interest", status="ok",
             check="Top wall and right side: exact (the corner's signal needs " + m_fix(1e3 * fl.t_top, 3) + " ms, the shock "
                   + m_fix(1e3 * fl.t_right, 3) + " ms; the time of interest is " + m_fix(1e3 * t_end, 2) + " ms). Floor: in the simulation "
                   "the bent shock has run " + m(f"{d_wall:.3f}") + " m of the " + m(f"{fl.Hs}") + " m down the step face, and the pressure "
                   "along the floor and the right side is " + ("unchanged to the last digit" if wf == 0 else "within " + sc(wf, 1) + r" of \(p_0\)") + "."),
        dict(name="The flow has no length of its own (the picture only stretches with time)", status="ok",
             check="Follows from the two rows above. The numerical method has none either, with equal square cells and a time step in "
                   "proportion to the cell size: the density at half the time on one grid equals the density at the full time on the "
                   "next coarser grid to " + sc(wd, 1) + r" of \(\rho_0\), cell for cell."),
        dict(name="The time step is small enough", status="ok",
             check=f"On {runs[main_ts]['N']} cells, half the time step changes the density field by "
                   + m_pct(c["variants"][ts_names[-1]]["field_vs_main"], 2) + " on average and the distance down the step face by "
                   + m_pct(abs(c["variants"][ts_names[-1]]["d_wall_vs_main"]), 2) + ". One refinement of the grid changes the field by "
                   + m_pct(next(p for p in pairs if p["fine"] == main_ts)["field_change"], 2) + ". Twice the time step fails (pre-flight)."),
    ]

    # ---- pre-flight --------------------------------------------------------------------------------------------
    short = pf["short"][0]
    tm = [timing[k] for k in sorted(timing)]
    pre_checks = list(short["checks"]) + [
        dict(name=f"Finest grid ({tm[0]['N']} cells per side), first 2 % of the run, on " + ", ".join(str(r["ranks"]) for r in tm) + " cores",
             status="ok" if all(r["status"] == "ok" for r in tm) else "fail",
             value="healthy on every count; " + ", ".join(m_fix(r["s_per_step"], 3) for r in tm) + " seconds per step; the full run projected at "
                   + m_fix(timing[fine["cores"]]["projected"] / 60, 1) + f" minutes on {fine['cores']} cores",
             meaning="The finest run must be healthy and must fit the time budget of about 15 minutes per run."),
        dict(name="Time-step study (200 cells per side): Courant numbers along \\(x\\) of "
                  + ", ".join(m_fix(fl.s_max * fl.t_end / (c["specs"][n]["spc"] * c["specs"][n]["N"]) / (fl.L / c["specs"][n]["N"]), 1) for n in ts_names),
             status="warn" if crashed_ts else "ok",
             value=("the largest step stopped with a negative pressure after " + f"{crashed_ts[0]['steps_done']} of {crashed_ts[0]['steps']} steps "
                    "(the solver's own Courant number had reached " + m_fix(crashed_ts[0]["co_max"], 2) + "); " if crashed_ts else "")
                   + "the other three ran, and the distance down the step face differs by "
                   + m_pct(max(r["d_wall"] for r in ts_ok) / min(r["d_wall"] for r in ts_ok) - 1, 2) + " between them",
             meaning="An explicit Euler step that is too long makes the pressure negative where the gas expands round the corner. "
                     "The main step (0.2) is half the failing one."),
        dict(name=f"Stress run: {runs[stress[0]]['N']} cells per side at 1.5 times the main time step" if stress and stress[0] in runs else "Stress run",
             status="ok" if stress and stress[0] in runs and runs[stress[0]]["health"] == "ok" else "fail",
             value=("ran to the end, healthy; lowest pressure " + m_fix(runs[stress[0]]["lows"]["p"] / 1e3, 1) + " kPa (against "
                    + m_fix(runs[main_ts]["lows"]["p"] / 1e3, 1) + " kPa on 200 cells)") if stress and stress[0] in runs else "stopped by the solver",
             meaning="In the pre-flight the lowest pressure fell from grid to grid, so a step that works on a coarse grid might fail on a fine one. "
                     "This run shows that the main step keeps a margin of 1.5 one level below the finest grid."),
        dict(name="Decisions cover every setting of the case", status="ok" if not pf["decisions"] else "fail",
             value="all covered" if not pf["decisions"] else f"{len(pf['decisions'])} missing", meaning="Every OpenFOAM setting belongs to a documented decision."),
    ]
    preflight = dict(
        purpose="Before any long run: is the setup right (start state, exact inflow, balances, output files), is the finest run "
                "healthy and affordable, how many cores, and how large may the time step be?",
        runs=[
            dict(name=f"{short['N']} cells per side, first 10 % of the time", setup=f"{short['n_steps']} steps, one core",
                 outcome=f"{short['status'].upper()}: every check below"),
            dict(name=f"{tm[0]['N']} cells per side, first 2 % of the time", setup="82 steps on " + ", ".join(str(r["ranks"]) for r in tm) + " cores",
                 outcome="healthy; fastest on " + str(min(tm, key=lambda r: r["s_per_step"])["ranks"]) + " cores"),
            dict(name="100 and 200 cells per side, in full", setup="main time step, one core", outcome="healthy; their numbers sized the rules for the outcome"),
            dict(name="Time-step study, 200 cells per side, in full", setup="Courant numbers 0.4, 0.3, 0.2 and 0.1 along \\(x\\)",
                 outcome="0.4 stopped with a negative pressure; 0.3, 0.2 and 0.1 agree"),
            dict(name="Stress run, 400 cells per side, in full", setup="Courant number 0.3 (1.5 times the main step)", outcome="healthy"),
        ],
        checks=pre_checks,
        conclusion="The setup is right: the opening delivers the exact flow and the balances close to round-off. The main time step "
                   "is kept, with a measured margin. The finest run fits the budget on 8 cores. The first pre-flight had planned "
                   "three time steps; the fourth and the stress run were added when the largest step failed.")

    # ---- simulation --------------------------------------------------------------------------------------------
    model = [
        dict(aspect="Gas", real="Air as an ideal gas with constant specific heats, no viscosity", model="The same (Euler equations)"),
        dict(aspect="Duct", real="A duct through which the shock arrives", model="Not computed: its opening carries the exact state behind the shock"),
        dict(aspect="Chamber", real="1 m by 1 m, walls on all other sides", model=f"{coarse['N']} to {Nf} equal square cells per side; frictionless walls"),
        dict(aspect="Shock", real="A jump of no thickness", model="Captured: spread over two to three cells, moving at the exact speed"),
        dict(aspect="Third direction", real="None (two-dimensional)", model="One layer of cells, 1 m thick"),
        dict(aspect="Time", real=r"\(0 \le t \le t_{\mathrm{end}}\)", model="Explicit steps of fixed size, 5.1 per cell crossed by the fastest signal at Courant 0.2"),
    ]
    meshes = dict(
        intro="Equal square cells; each level halves the cell size and the time step. All runs use the same schemes.",
        head=["Cells per side", "Cells", "Cell size (mm)", "Steps", "Time step (s)", "Cores", "Solver time (s)"],
        rows=[[r["N"], f"{r['N'] ** 2:,}", m_fix(1e3 * r["h"], 2), r["steps"], sc(t_end / r["steps"], 3), r["cores"], m_fix(r["wall_time"], 1)] for r in R])
    results_tab = dict(
        head=["Cells per side", "Shock on the top wall (m)", "Speed of the flat part (m/s)", "Down the step face (m)", "Mach number on the step face",
              "Lowest pressure at the time of interest (kPa)", "Highest Mach number"],
        rows=[[r["N"], m_fix(r["x_top"], 4), m_fix(r["speed"], 1), m_fix(r["d_wall"], 4), m_fix(r["Mw_dist"], 3), m_fix(r["extra"]["p_min"] / 1e3, 1),
               m_fix(r["extra"]["mach_max"], 2)] for r in R]
             + [["exact or estimate", m_fix(fl.x_top, 4), m_fix(fl.W, 1), "about " + m_fix(fl.d_wall, 3), "about " + m_fix(fl.Mw, 2), "–", "–"]])
    health = dict(
        head=["Cells per side", "Log checks", "Solver's Courant number", "Mass balance", "Energy balance", "Lowest density (kg/m³)",
              "Lowest pressure (kPa)", "Floor and right side"],
        rows=[[r["N"], r["health"].upper(), m_fix(r["co_max"], 3), sc(r["balance"]["mass_vs_inflow"], 1), sc(r["balance"]["energy_vs_inflow"], 1),
               m_fix(r["lows"]["rho"], 3), m_fix(r["lows"]["p"] / 1e3, 1), "untouched" if max(r["far"]["floor"], r["far"]["right"]) == 0 else sc(max(r["far"]["floor"], r["far"]["right"]), 1)] for r in R],
        notes=[
            "An explicit solver has no residuals. The evidence that each step is right is physical: the mass and the energy in the chamber "
            "must equal the start plus the exact inflow. Both are recorded after every step. Columns: the largest difference during the "
            "run as a fraction of the inflow so far (Figure {fig:conservation}); required: " + sc(acc["req_conserved"], 1) + ".",
            "Log checks (cfdlib.monitor): no crash, no NaN, the Courant number and its trend. The solver prints its own Courant number, "
            "which adds both directions; the usual one along \\(x\\) behind the shock is " + m_fix(fine["co_x"], 2) + " on every grid.",
            "The lowest pressure and density are in the expansion at the corner, just before the second shock. The lower they are, the "
            "closer a run is to a negative pressure (pre-flight). From the coarsest grid to the finest the lowest pressure goes "
            + ", ".join(m_fix(r["lows"]["p"] / 1e3, 1) for r in R) + " kPa: the change in the last refinement is "
            + m_pct(abs(R[-1]["lows"]["p"] / R[-2]["lows"]["p"] - 1), 0) + ", against " + m_pct(abs(R[1]["lows"]["p"] / R[0]["lows"]["p"] - 1), 0) + " in the first.",
            "Floor and right side: the pressure in the cells along them at the time of interest, against \\(p_0\\). Untouched means equal to the last digit.",
            "The grid of every run passed checkMesh." if all(metas[n]["mesh_ok"] for n in fam) else "A grid did not pass checkMesh.",
        ])
    co_of = lambda n: fl.s_max * fl.t_end / (c["specs"][n]["spc"] * c["specs"][n]["N"]) / (fl.L / c["specs"][n]["N"])
    ts_rows = []
    for n in ts_names:
        if n in c["crashed"]:
            k = c["crashed"][n]
            ts_rows.append([m_fix(co_of(n), 1), m_fix(k["co_max"], 2), f"{k['steps_done']} of {k['steps']}", "stopped: negative pressure", "–", "–", "–", "–", "–"])
        else:
            r = runs[n]
            ts_rows.append([m_fix(co_of(n), 1) + (" (main)" if n == main_ts else ""), m_fix(r["co_max"], 2), r["steps"], m_fix(r["d_wall"], 4),
                            sgn(r["speed_err"], 2), m_pct(r["state"]["p"]["ripple"], 2), m_pct(max(r["extra"]["fan_wiggle"], 0.0), 1),
                            m_fix(r["lows"]["p"] / 1e3, 1), m_pct(c["variants"][n]["field_vs_main"], 2) if n in c["variants"] else "–"])
    ts_tab = dict(
        title=f"Time-step study, {runs[main_ts]['N']} cells per side",
        head=["Courant number along x", "The solver's own", "Steps", "Down the step face (m)", "Speed of the flat part", "Ripple behind the flat part",
              "Stripes in the fan", "Lowest pressure (kPa)", "Density field against the main step"],
        rows=ts_rows,
        note="Courant number along x: the fastest signal behind the flat shock times the time step over the cell size. The solver's own number adds "
             "both directions. Stripes in the fan: the up-and-down of the density along an arc round the corner, as a fraction of its fall. "
             "Last column: mean difference of the density field at the time of interest from the run at the main step.")
    sch_rows = []
    for n in sch_names + sch_crashed:
        s = c["specs"][n]
        lab = [s["flux"], s["limiter"] + (" (main)" if n == main_ts else "")]
        if n in c["crashed"]:
            k = c["crashed"][n]
            sch_rows.append(lab + [f"stopped at step {k['steps_done']} of {k['steps']}", "–", "–", "–", "–", "–", "–"])
        else:
            r = runs[n]
            sch_rows.append(lab + [m_fix(r["d_wall"], 4), sgn(r["speed_err"], 2), sc(r["dev_cells_max"], 1), m_pct(r["state"]["p"]["ripple"], 2),
                                   m_pct(max(r["extra"]["fan_wiggle"], 0.0), 1), m_fix(r["lows"]["p"] / 1e3, 1),
                                   m_pct(c["variants"][n]["field_vs_main"], 2) if n in c["variants"] else "–"])
    sch_tab = dict(
        title=f"Flux scheme and limiter, {runs[main_ts]['N']} cells per side, same time step",
        head=["Flux", "Limiter", "Down the step face (m)", "Speed of the flat part", "Flat part: deviation (cells)", "Ripple behind the flat part",
              "Stripes in the fan", "Lowest pressure (kPa)", "Density field against the main scheme"],
        rows=sch_rows,
        note="upwind is the first-order scheme: no slope inside a cell. Minmod is the most cautious limiter; van Leer and van Albada are "
             "steeper. Tadmor is the more smearing of the two fluxes.")

    # ---- comparison ------------------------------------------------------------------------------------------
    defs = [
        dict(name="Position of the shock in a row of cells", latex=r"p(x_s) = p_0 + \tfrac12 (p_2 - p_0)",
             meaning="Where the pressure crosses the middle between the still air and the exact state behind the shock, on a straight line "
                     "between two cell centres. Applied to the exact shock averaged over the cells, the rule reads "
                     + m(f"{bias[Nf][0]:+.2f}") + " to " + m(f"{bias[Nf][1]:+.2f}") + " cells off, depending on where in a cell the shock is."),
        dict(name="Flat part and uniform region", latex=r"\begin{aligned} y &\ge H_s + v_{\mathrm{flat}}\,t + 6h &&\text{(rows of the flat part)}\\ "
                                                         r"y &\ge H_s + c_2 t + 6h &&\text{(uniform region)}\end{aligned}",
             meaning="The exact heights from Theory plus six cells. The uniform region also stays six cells clear of the shock and of the opening."),
        dict(name="Deviation from a straight line", latex=r"\max_{\text{rows}} \frac{|x_s - \overline{x_s}|}{h}",
             meaning="Largest distance of a row's shock position from the mean over the rows of the flat part, in cells. A tilt counts too."),
        dict(name="Ripple behind the flat part", latex=r"\frac{\max p - \min p}{p_2}\quad\text{over the uniform region}",
             meaning="The exact pressure there is uniform, so any spread is made by the method."),
        dict(name="Difference between two fields", latex=r"\frac{\sum_i |a_i - b_i|}{\sum_i \tfrac12 (|a_i| + |b_i|)}",
             meaning="Mean absolute difference as a fraction of the mean. Used for the two times (the later picture averaged over pairs of "
                     "cells, so that both are on the same points of distance over time) and for two grids (the finer averaged over "
                     "2 by 2 cells). The largest difference is not used: next to a jump it never falls."),
    ]
    comp_rows = [[r["N"], sgn(r["speed_err"], 3), sgn(r["state"]["p"]["err"], 3), sgn(r["state"]["rho"]["err"], 3), sgn(r["state"]["u"]["err"], 3),
                  m(f"{r['x_top_err_cells']:+.2f}"), sc(r["dev_cells_max"], 1), m_pct(r["state"]["p"]["ripple"], 2), m_pct(r["state"]["rho"]["ripple"], 2),
                  sc(r["p_row_spread"], 1)] for r in R]
    pair_tab = dict(
        title="Two times and two grids",
        head=["Finer grid", "Coarser grid", r"Half time on the finer against full time on the coarser (largest, of \(\rho_0\))", "Density field, coarser to finer",
              "Order", "Two times, step face", r"Two times, line \(y = H_s\)"],
        rows=[[runs[p["fine"]]["N"], runs[p["coarse"]]["N"], sc(p["discrete"], 1), m_pct(p["field_change"], 2), m_fix(fo[i - 1], 2) if i else "–",
               m_pct(runs[p["fine"]]["sim_wall"], 1), m_pct(runs[p["fine"]]["sim_line"], 1)] for i, p in enumerate(pairs)],
        note="Third column: the check that the method has no length of its own; the two fields should be equal and are, to that number. "
             "It follows that the two-times difference on a grid (last two columns) is the change from the next coarser grid to this one, "
             "along that line. Order: by how much the change of the whole field falls per refinement (1 = it halves).")
    wall_tab = dict(
        title="The bent shock on the step face",
        head=["Cells per side", "Distance below the corner (m)", "Change from the coarser grid", "Mach number from the distance", "Mach number from the pressure behind it",
              "Pressure behind it (kPa)"],
        rows=[[r["N"], m_fix(r["d_wall"], 4), sgn(r["d_wall"] / R[i - 1]["d_wall"] - 1, 2) if i else "–", m_fix(r["Mw_dist"], 3), m_fix(r["Mw_peak"], 3),
               m_fix(r["wall_peak"] / 1e3, 0)] for i, r in enumerate(R)],
        note="Mach number from the distance: distance over time over the speed of sound of the still air (the flow only stretches, so the shock's "
             "speed is its distance over the time). From the pressure: the jump condition applied to the first pressure maximum behind the shock. "
             "Whitham's estimate: Mach " + m_fix(fl.Mw, 2) + ", distance " + m_fix(fl.d_wall, 3) + " m."
             + (" The changes fall steadily. The order observed on the last three grids, " + m_fix(wall_extr["order"], 1) + ", is higher than the "
                "method can have, so the last change is the better measure of what is left; the value extrapolated with that order, "
                + m_fix(wall_extr["richardson"], 4) + " m, differs from the finest grid by less than that." if wall_extr and wall_extr.get("monotonic") and wall_extr.get("plausible")
                else " The changes do not fall steadily, so no value extrapolated to an infinitely fine grid is given."))
    cons_tab = dict(
        title=f"Mass and energy at the ten written times, {Nf} cells per side (per metre of depth)",
        head=["Time (ms)", "Mass (kg)", "Start + exact inflow (kg)", "Difference / inflow", "Energy (MJ)", "Start + exact inflow (MJ)", "Difference / inflow"],
        rows=[[m_fix(1e3 * wr_["t"][k], 3), m_fix(wr_["mass"][k], 6), m_fix(wr_["mass_exact"][k], 6), sc(wr_["mass_err"][k], 1),
               m_fix(wr_["energy"][k] / 1e6, 6), m_fix(wr_["energy_exact"][k] / 1e6, 6), sc(wr_["energy_err"][k], 1)] for k in range(len(wr_["t"]))],
        note="The request is for the written times; the check was made after every step (Figure {fig:conservation}).")
    cq = ex["contact"]
    comp_findings = [
        "<b>The flat part (request " + L["flat_part"] + ").</b> A method that balances mass, momentum and energy over every cell moves a "
        "shock at the exact speed and leaves the exact state behind it; the errors in the table are what the reading rules and the "
        "two start-up waves add. They fall with the grid.",
        "<b>The two start-up waves.</b> At \\(t = 0\\) the shock is a sharp jump at the opening. In its first steps the method spreads it "
        "over two to three cells, and that adjustment sends off a sound wave and leaves a spot of wrong density in the gas. Theory says "
        "where such waves must be at the time of interest: the sound wave at \\((u_2 - c_2)\\,t = " + f"{startup[0]:.3f}" + "\\,\\mathrm{m}\\), "
        "the spot at \\(u_2 t = " + f"{startup[1]:.3f}" + "\\,\\mathrm{m}\\). That is where they are (Figure {fig:flat}, left). Their height does "
        "not fall with the grid (pressure ripple " + ", ".join(m_pct(r["state"]["p"]["ripple"], 1) for r in R) + "), nor with the time step; their "
        "width is a fixed number of cells.",
        "<b>Straightness (request " + L["stable"] + ").</b> The rows of the flat part hold the same numbers to "
        + sc(worst_dev, 1) + " cells: nothing disturbs the shock there, and the method does not amplify what little arrives from the corner. "
        "Below the exact end of the flat part the shock bends smoothly (Figure {fig:flat}, right).",
        "<b>The same picture at two times (request " + L["similar"] + ").</b> Because the method has no length of its own, the earlier picture "
        "on a grid is the later picture on a grid twice as coarse (checked to " + sc(wd, 1) + r" of \(\rho_0\)). So this request and the grid "
        "study measure the same thing. Along the step face the difference falls from " + m_pct(R[1]["sim_wall"], 1) + " to " + m_pct(fine["sim_wall"], 1)
        + f" between {R[1]['N']} and {Nf} cells. Along \\(y = H_s\\) it falls too: " + ", ".join(m_pct(r["sim_line"], 2) for r in R[1:])
        + ". Before the runs I expected no clean fall there, because the layer from the corner starts on this line. It leaves the line at "
        "once, downwards (Figure {fig:corner}), so the line crosses the fan and the bent shock, which do settle.",
        "<b>The inner border.</b> Inside the bent shock runs a line across which the density changes and the pressure hardly does (ratios "
        + m_fix(cq["rho_ratio"], 3) + " and " + m_fix(cq["p_ratio"], 3) + " across eight cells where it crosses \\(y = 0.6\\,\\mathrm{m}\\)). I read it as "
        "the border between gas that crossed the flat shock and gas that crossed the weaker bent shock; the literature calls it the contact "
        "surface. The density ratio across a fixed number of cells falls as the cells get smaller ("
        + ", ".join(m_fix(r["extra"]["contact"]["rho_ratio"], 2) for r in R) + "), so on these grids it looks like a steep smooth change, not a jump.",
        "<b>Whitham's estimate.</b> The bent shock on the step face has Mach number " + m_fix(Mw_sim, 2) + " (from its distance) or "
        + m_fix(fine["Mw_peak"], 2) + " (from the pressure behind it); Whitham's theory gives " + m_fix(fl.Mw, 2) + ". The theory ignores what happens "
        "behind the shock, and here a strong expansion sits right behind it. The simulation's value changes by " + m_pct(dchg, 2)
        + " in the last refinement, so the difference is not a grid effect.",
        "<b>Stripes in the fan.</b> Fine stripes fan out from the corner in the density (Figure {fig:map}, right). A smooth fan has none. "
        "Measured as the up-and-down of the density along an arc round the corner, as a fraction of its fall, they are "
        + ", ".join(m_pct(max(r["extra"]["fan_wiggle"], 0.0), 1) for r in R) + " on " + ", ".join(str(r["N"]) for r in R) + " cells per side: absent on the coarse "
        "grids, present on the fine ones."
        + (f" On {runs[stress[0]]['N']} cells they are larger at the larger time step (" + m_pct(runs[stress[0]]["extra"]["fan_wiggle"], 1) + " at 1.5 times the "
           "main step against " + m_pct(runs[f"N{runs[stress[0]]['N']}"]["extra"]["fan_wiggle"], 1) + "), which points at the explicit Euler step, as with the "
           "ringing in case 18. A smaller step on a fine grid was not run, so this is not settled."
           if stress and stress[0] in runs and f"N{runs[stress[0]]['N']}" in runs else ""),
    ]

    # ---- performance -------------------------------------------------------------------------------------------
    order_names = fam + [n for n in ts_names if n in runs and n != main_ts] + [n for n in stress if n in runs] + [n for n in sch_names if n != main_ts]
    perf_rows = [[n, f"{runs[n]['N'] ** 2:,}", runs[n]["cores"], runs[n]["steps"], m_fix(runs[n]["wall_time"], 1), m_fix(runs[n]["us_per_cell_step"], 2),
                  m_pct(runs[n]["cpu_efficiency"], 0) if runs[n].get("cpu_efficiency") else "–",
                  m_fix(runs[n]["peak_mem_mb"], 0) if runs[n].get("peak_mem_mb") else "–"] for n in order_names]
    total_solver = sum(runs[n]["wall_time"] for n in runs)
    performance = dict(
        intro="The solver is explicit, so a run costs cells times steps, and the steps grow with the cells per side: each refinement costs "
              "8 times the one before. All runs were made one at a time.",
        head=["Run", "Cells", "Cores", "Steps", "Solver time (s)", "Microseconds per cell and step (wall)", "CPU busy", "Peak memory of one process (MB)"],
        rows=perf_rows,
        notes=[
            f"Cores, measured on the first 82 steps of the {tm[0]['N']}-cell grid after start-up: " + ", ".join(m_fix(r["s_per_step"], 3) for r in tm)
            + " seconds per step on " + ", ".join(str(r["ranks"]) for r in tm) + " cores, so " + ", ".join(m_times(tm[0]["s_per_step"] / r["s_per_step"], 1) for r in tm[1:])
            + " faster than one core. Four cores give most of the gain and sixteen are slower than eight: the work is moving memory, which all cores share.",
            "On one core a cell costs about " + m_fix(runs[fam[-2]]["us_per_cell_step"], 2) + " microseconds per step, twice the one-dimensional case 18 "
            "(two directions of fluxes).",
            "What a finer answer would cost: 1600 cells per side is 8 times the finest run, about " + m_fix(next_cost / 3600, 1) + " hours here. With the "
            "observed order of about " + m_fix(float(np.mean(fo)), 1) + " that would cut the change between grids by about "
            + m_times(2 ** float(np.mean(fo)), 1) + ". The flat part and the totals would gain nothing: they are exact already.",
            "The time step cannot be raised much to save time: twice the step fails on 200 cells; 1.5 times was tested on 200 and 400 cells and ran.",
            "All runs together took " + m_fix(total_solver / 60, 1) + " minutes of solver time; the finest " + m_fix(fine["wall_time"] / 60, 1) + " minutes.",
        ])

    results = dict(
        title=problem["title"], date=dt.date.today().isoformat(), verdict=verdict, expectations=X, comparison_type="verification",
        summary=dict(headline=headline, answer_columns=["Requested", "Exact value, or estimate", f"Simulation ({Nf} cells per side)", "Difference"],
                     answers=answers, findings=findings, limitations=limitations),
        problem=dict(
            description=problem["description"], setup=problem.get("setup", []),
            given=[dict(label=v["label"], symbol=v["symbol"], value=m(tex_num(v["value"], 6)), unit=v["unit"]) for v in problem["given"].values()],
            wanted=wanted, required_accuracy=problem["required_accuracy"],
            reference="No reference data came with the problem. The flat shock and the inflow are compared with the exact jump conditions "
                      "(derived and checked in Theory); the rest is checked for consistency. The test is that of " + REFS["quirk"] + " and " + REFS["cheng"] + "."),
        assumptions=dict(rows=assumptions),
        theory=th,
        decisions=load_yaml(HERE / "decisions.yaml"),
        preflight=preflight,
        simulation=dict(model=model, meshes=meshes, results=results_tab, health=health, tables=[ts_tab, sch_tab]),
        comparison=dict(
            kind="Verification of what is known exactly (the flat shock, the inflow, the totals), and a consistency check of the rest "
                 "(two times, four grids, two approximate theories).",
            definitions=defs,
            head=["Cells per side", "Speed", "Pressure", "Density", "Velocity", "Top wall (cells)", "Deviation from straight (cells)", "Ripple, pressure",
                  "Ripple, density", "Pressure along the shock"],
            rows=comp_rows, findings=comp_findings, tables=[pair_tab, wall_tab, cons_tab]),
        insights=load_yaml(HERE / "insights.yaml"),
        figures=figures,
        performance=performance,
        reproducibility=dict(
            commands=[
                dict(cmd=f"python -m cases.{HERE.name}.run preflight", what="Short tests, timing on 1, 8 and 16 cores, the two coarsest grids and the time-step study in full"),
                dict(cmd=f"python -m cases.{HERE.name}.run preflight full N200_spc3.4 N400_spc3.4", what="The time step added to the study and the stress run"),
                dict(cmd=f"python -m cases.{HERE.name}.run preflight timing 4", what="The timing test on 4 cores"),
                dict(cmd=f"python -m cases.{HERE.name}.run production", what="Every other run, one at a time; writes runs/*.json and runs/*.npz"),
                dict(cmd=f"python -m cases.{HERE.name}.analyze", what="The numbers, the rules, the figures and this report (checks the contract first)"),
            ],
            files=[
                dict(path=f"cases/{HERE.name}/problem.yaml", what="The problem as given"),
                dict(path=f"cases/{HERE.name}/numerics.yaml", what="Runs, reading rules, and the rules for the outcome written before the production runs"),
                dict(path=f"cases/{HERE.name}/decisions.yaml", what="Every choice, with the OpenFOAM settings it controls"),
                dict(path=f"cases/{HERE.name}/insights.yaml", what="Insights logged during the work"),
                dict(path=f"cases/{HERE.name}/template/", what="The OpenFOAM case with {{ placeholders }}"),
                dict(path=f"cases/{HERE.name}/theory.py", what="The jump conditions, the inflow, the corner's signal and the two estimates, with SymPy checks"),
                dict(path=f"cases/{HERE.name}/measure.py", what="The rules that read the outputs from the fields"),
                dict(path=f"cases/{HERE.name}/run.py", what="Builds, pre-flights and runs the cases"),
                dict(path=f"cases/{HERE.name}/numbers.py, figs.py, sketch.py", what="The numbers of the analysis and the figures"),
            ],
            run_dirs=[
                dict(path=f"{metas[fam[-1]]['run_root']}/N<cells per side>[_spc<steps per cell>][_<flux>_<limiter>]", what="Production runs"),
                dict(path=f"{metas[fam[-1]]['run_root']}/pf_*_preflight", what="Pre-flight runs"),
                dict(path=f"results/{HERE.name}/", what="preflight.json, runs/, logs/, results.json and this report"),
            ],
            environment=report.environment_info(),
            notes=["Runs were made one at a time. The finest grid ran on 8 cores (MPI), every other run on one core.",
                   "Sources cited from memory, not supplied with the problem: " + "; ".join(REFS.values()) + "."]),
        data=dict(exact=dict(W=fl.W, p2=fl.p2, rho2=fl.rho2, u2=fl.u2, c2=fl.c2, M2=fl.M2, mdot=fl.mdot, edot=fl.edot, x_top=fl.x_top,
                             flat_from=fl.flat_from(t_end), Mw_whitham=fl.Mw, d_wall_whitham=fl.d_wall),
                  runs={n: {k: v for k, v in r.items() if k not in ("extra", "written")} | {"extra": {k: v for k, v in r["extra"].items() if k != "fan_arc"}}
                        for n, r in runs.items()},
                  stopped={n: dict(steps_done=k["steps_done"], steps=k["steps"], t_stop=k["t_stop"], co_max=k["co_max"]) for n, k in c["crashed"].items()},
                  pairs=pairs, conv={k: v for k, v in conv.items() if k.endswith("orders")}, variants=c["variants"],
                  rule_bias={str(k): v for k, v in bias.items()}),
    )
    out = report.render(results, svgs, OUT / "report.html", case_dir=metas[fam[-1]]["case_dir"])
    print(f"{verdict}: missed {[x['text'] for x in missed]}")
    for x in X:
        print(f"   {'ok  ' if x['met'] else 'MISS'} [{x['kind']}] {x['text'][:90]} | expected {x['expected'][:70]} | found {x['found'][:150]}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
