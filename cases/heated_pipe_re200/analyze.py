"""Compare the heated-tube simulations with the exact solution; build results.json and the report.

Usage (repo root, inside WSL):  python -m cases.heated_pipe_re200.analyze
Reads results/heated_pipe_re200/runs/*.json|npz, preflight.json and the case's YAML files.
"""
import datetime as dt
import json
import math
import re
from pathlib import Path

import numpy as np

from cfdlib import insights, plots, report, verification as V
from cfdlib.report import LETTERS, load_yaml, m, tex_num as tn, tex_pct as tp, tex_fix as tf
from . import metrics as M
from . import theory
from .sketch import sketch

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"
C_WATER = 1482.0              # m/s, speed of sound in water at 20 °C (approximate standard value)
RE_CRIT_PIPE = 2300           # commonly quoted lower limit of transition in pipe flow (approximate)
SIEDER_TATE_EXP = 0.14        # exponent of the viscosity-ratio correction, Sieder & Tate (1936), Ind. Eng. Chem. 28 (approximate use)


def sgn_pct(x, sig=2):
    return ("+" if x >= 0 else "-") + tp(abs(x), sig)


def main():
    problem = load_yaml(HERE / "problem.yaml")
    num = load_yaml(HERE / "numerics.yaml")
    acc = num["acceptance"]
    G = {k: v["value"] for k, v in problem["given"].items()}
    D, a = G["D"], G["D"] / 2
    th = theory.evaluate(problem["given"])
    ex, nb = th["exact"], th["numbers"]
    half = num["geometry"]["wedge_half_angle_deg"]
    levels = [lv for lv in num["mesh"]["levels"] if (RUNS / f"{lv}.json").exists()]   # all four in the final report
    draft = len(levels) < len(num["mesh"]["levels"])

    # ---- per-level metrics -------------------------------------------------------------------------
    runs, mets = [], []
    for lv in levels:
        rec, arr = M.load(lv)
        runs.append((rec, arr))
        mets.append(M.wedge(rec, arr, ex, G, half))
    fine, finerec = mets[-1], runs[-1][0]
    h = [D / 2 / n for n in num["mesh"]["Nr"][:len(levels)]]       # representative size: a / N_r (ratio 2)
    order_rms = V.order_from_errors(h, [x["e_rms"] for x in mets])
    seq = {k: V.level_sequence([x[k] for x in mets[-3:]]) for k in ("nu_out", "nu_fd", "l_th", "Tw_out")}
    useq = V.level_sequence([x["u_c0"] for x in mets[-3:]])
    useq_val = useq["richardson"]
    gci = {}
    for k, s in seq.items():
        if s["order"] and s["plausible"]:
            gci[k] = V.gci_fine(mets[-2][k], mets[-1][k], s["order"])
        else:
            gci[k] = None

    # ---- buoyancy check (3D half tube) ------------------------------------------------------------
    b3 = {}
    for tag in ("g0", "g1"):
        p = RUNS / f"B3D_{tag}.json"
        if p.exists():
            rec, arr = M.load(f"B3D_{tag}")
            b3[tag] = (rec, M.half3d(rec, arr, G))
    buoy = None
    if len(b3) == 2:
        g0, g1 = b3["g0"][1], b3["g1"][1]
        w = g0["x"] >= M.X_MIN_D * D
        d_nu = g1["nu"] / g0["nu"] - 1
        buoy = dict(x=g0["x"], d_nu=d_nu, max=float(np.max(np.abs(d_nu[w]))), out=float(d_nu[-1]),
                    at=float(g0["x"][w][np.argmax(np.abs(d_nu[w]))]),
                    topbot=float(abs(g1["Tw_top"][-1] - g1["Tw_bot"][-1])) / fine["dT_fd"],
                    sec=float(np.max(g1["sec_max"])) / G["U"], nu0_out=float(g0["nu"][-1]), nu1_out=float(g1["nu"][-1]),
                    nu0_rel_2d=float(g0["nu"][-1] / ex.nu(g0["x"][-1])[0] - 1),
                    wall0=b3["g0"][0]["wall_time"], wall1=b3["g1"][0]["wall_time"],
                    it0=b3["g0"][0]["iterations"], it1=b3["g1"][0]["iterations"], cells=b3["g0"][0]["cells"])

    # ---- assumptions ------------------------------------------------------------------------------
    Ma = 2 * G["U"] / C_WATER
    nu_eff = nb["mu_change_wb"] * SIEDER_TATE_EXP
    if buoy is None:
        b_status, b_text = "standard", "Not measured (3D runs missing)."
    else:
        b_status = "ok" if buoy["max"] < acc["buoyancy_ok"] else "violated"
        b_text = ("Measured: the same 3D half-tube mesh run with and without gravity. Gravity changes the peripherally "
                  "averaged " + m("Nu") + " by at most " + m(tp(buoy["max"], 2)) + " (at " + m(rf"x = {buoy['at'] / D:.1f}\,D")
                  + "), " + m(sgn_pct(buoy["out"])) + " at the outlet; the secondary flow peaks at " + m(tn(buoy["sec"], 2) + r"\,U")
                  + ", and the top and bottom of the wall differ by up to " + m(tp(buoy["topbot"], 2)) + " of "
                  + m(r"\Delta T_{fd}") + ". Indicators: " + m(rf"Gr = g\beta\Delta T_{{fd}}D^3/\nu^2 = {tn(nb['Gr'], 3)}") + ", "
                  + m(rf"Ra = {tn(nb['Ra'], 3)}") + ", " + m(rf"Gr/Re^2 = {tn(nb['Ri'], 2)}") + ". Rule (fixed before the runs): "
                  + m(rf"< {tp(acc['buoyancy_ok'], 1)}") + " for the no-buoyancy idealisation to hold.")
    assumptions = [
        dict(name="Laminar flow", status="ok" if nb["Re"] < RE_CRIT_PIPE else "violated",
             check=m(rf"Re = UD/\nu = {tn(nb['Re'], 4)}") + f", far below the usual pipe transition value of about {RE_CRIT_PIPE} (approximate)."),
        dict(name="Incompressible", status="ok",
             check=m(rf"Ma = u_{{\max}}/c = {tn(Ma, 2)}") + f" with the speed of sound in water about {C_WATER:.0f} m/s (approximate); "
                   "density changes by " + m(rf"\beta\,\Delta T_{{\max}} = {tn(G['beta'] * nb['T_w_max_rise'], 2)}") + "."),
        dict(name="Constant properties", status="ok",
             check="From the two given viscosities, " + m(rf"d\mu/dT \approx {tn(nb['dmu_dT'], 3)}\,\mathrm{{Pa\,s/K}}") + ". The hottest "
                   "water (outlet wall, " + m(rf"{tn(nb['T_w_max_rise'], 3)}\,\mathrm{{K}}") + " above the inlet) is "
                   + m(tp(nb["mu_change_wall"], 2)) + " less viscous; wall and bulk differ by " + m(tp(nb["mu_change_wb"], 2))
                   + ". With the Sieder-Tate exponent " + m(f"{SIEDER_TATE_EXP}") + " (approximate) that changes " + m("Nu")
                   + " by about " + m(tp(nu_eff, 1)) + ", below the mesh error."),
        dict(name="No buoyancy (gravity does not change the flow)", status=b_status, check=b_text),
        dict(name="No viscous heating", status="ok",
             check=m(rf"Br = \mu U^2/(q_w D) = {tn(nb['Br'], 2)}") + ": the friction heat is negligible against the wall flux."),
        dict(name="Axial heat conduction negligible (exact solution only)", status="ok",
             check=m(rf"Pe = {tn(nb['Pe'], 4)} \gg 1") + ". The simulation keeps axial conduction; the exact solution drops it. "
                   "The difference is largest just after " + m("x = 0") + " and is part of what the comparison measures."),
        dict(name="Fully developed velocity at the start of heating (exact solution only)",
             status="ok" if abs(fine["e_uc0"]) < acc["velocity_x0_ok"] else "violated",
             check="The simulation develops the velocity from a uniform inlet over " + m(rf"L_0 = 20\,D = {tn(nb['L0_over_ReD'], 2)}\,Re\,D")
                   + ". At " + m("x = 0") + " its centreline velocity is " + m(sgn_pct(fine["e_uc0"])) + " from the parabola's "
                   + m("2U") + " and the largest difference across the profile is " + m(tp(fine["e_u0"], 2)) + " of " + m("2U")
                   + " (finest mesh, Figure {fig:velocity}). On the meshes L1 to " + levels[-1] + " the centreline difference is "
                   + ", ".join(m(sgn_pct(x["e_uc0"], 2)) for x in mets) + ": it shrinks with each refinement (at first order "
                   "on the coarse meshes, set by the singular corner where the uniform inlet meets the no-slip wall), so most "
                   "of it is mesh error. Extrapolated to zero cell size (last three meshes) it is "
                   + m(sgn_pct(useq_val / (2 * G["U"]) - 1, 1) if useq_val is not None else r"\text{n/a}")
                   + ": a small real deficit remains after 20 diameters, far below the 1 % rule."),
        dict(name="Steady, axisymmetric", status="imposed",
             check="Steady solver; without gravity nothing varies around the axis, so a " + m(rf"{2 * half:g}^\circ") + " wedge is exact. "
                   "The wedge's flat wall faces change the perimeter-to-area ratio by " + m(tn(1 / math.cos(math.radians(half)) - 1, 2)) + "."),
    ]

    # ---- verdict (rules fixed in numerics.yaml before production) ----------------------------------
    fails, warns = [], []

    def rule(name, err, tol, warn_factor=acc["warn_factor"]):
        e = abs(err)
        if e <= tol:
            return "ok"
        if e <= warn_factor * tol:
            warns.append(f"{name}: {100 * e:.3g} % > {100 * tol:.3g} %")
            return "warn"
        fails.append(f"{name}: {100 * e:.3g} % > {warn_factor} x {100 * tol:.3g} %")
        return "fail"
    st = dict(
        rms=rule("RMS error of Nu(x), finest mesh", fine["e_rms"], acc["nu_rms"]),
        nu_out=rule("Nu at the outlet", fine["e_nu_out"], acc["nu_out"]),
        nu_fd=rule("fully developed Nu (tail fit)", fine["e_nu_fd"], acc["nu_fd"]),
        l_th=rule("thermal entrance length", fine["e_l_th"], acc["l_th"]),
        tw=rule("wall temperature (max over the heated section, of the wall-bulk difference)", fine["e_Tw"], acc["t_wall"]),
    )
    if max(abs(x["energy"]) for x in mets) > acc["energy_balance"]:
        fails.append("energy balance not satisfied")
    if any(r[0]["health"]["status"] == "fail" for r in runs):
        fails.append("a production run failed a health check")
    if not all(r[0]["mesh_ok"] for r in runs):
        fails.append("checkMesh reported a problem")
    stopped = [bool((r[0]["live_stop"] or {}).get("stopped")) for r in runs]
    if not all(stopped):
        warns.append("a run reached its iteration cap before the live stop held")
    p_last = order_rms[-1]
    if not acc["order_range"][0] <= p_last <= acc["order_range"][1]:
        warns.append(f"observed order of the Nu(x) error {p_last:.2f} outside {acc['order_range']}")
    if buoy is not None and buoy["max"] >= acc["buoyancy_ok"]:
        warns.append(f"buoyancy changes Nu by {100 * buoy['max']:.2g} % in the real (horizontal) tube: the exact solution "
                     "describes it only approximately")
    if draft:
        warns.append(f"DRAFT: only {', '.join(levels)} available")
    verdict = "FAIL" if fails else "WARN" if warns else "PASS"

    # ---- lettered answers --------------------------------------------------------------------------
    wanted = [dict(letter=LETTERS[i], key=w["key"], text=w["text"]) for i, w in enumerate(problem["quantities_of_interest"])]
    sub = lambda s: f"<div class='sub'>{s}</div>"
    ans = {
        "nu_x": dict(theory="Series solution, Figure {fig:nu}", sim="RMS difference " + m(tp(fine["e_rms"], 2))
                     + sub("over " + m(r"0.5\,D \le x \le 100\,D") + ", length-weighted; largest " + m(tp(fine["e_max"], 2))
                           + " near " + m("x = 0")), diff=m(tp(fine["e_rms"], 2))),
        "nu_fd": dict(theory=m(rf"Nu_{{fd}} = 48/11 = {tn(ex.nu_fd, 6)}") + sub("at the outlet the exact value is still "
                      + m(tn(fine["nu_out_ex"], 6))),
                      sim=m(tn(fine["nu_fd"], 6)) + sub("tail fit " + m(r"60\text{–}100\,D") + "; at the outlet " + m(tn(fine["nu_out"], 6))
                                                       + " (" + m(sgn_pct(fine["e_nu_out"])) + ")"),
                      diff=m(sgn_pct(fine["e_nu_fd"]))),
        "l_th": dict(theory=m(rf"{tn(th['l_th'] / D, 4)}\,D = {tn(th['l_th'], 4)}\,\mathrm{{m}}") + sub(m(rf"x^+ = {tn(ex.X(th['l_th']) / 4, 3)}")),
                     sim=m(rf"{tn(fine['l_th'] / D, 4)}\,D") + sub("crossing of " + m(r"1.05\,Nu_{fd}") + " with its own " + m("Nu_{fd}")),
                     diff=m(sgn_pct(fine["e_l_th"]))),
        "t_wall": dict(theory=m(rf"T_w(L_h) = {tf(fine['Tw_out_ex'], 4)}") + ", " + m(rf"T_b(L_h) = {tf(fine['Tb_out_ex'], 4)}\,\mathrm{{°C}}")
                       + sub("profiles: Figure {fig:temps}"),
                       sim=m(rf"T_w = {tf(fine['Tw_out'], 4)}") + ", " + m(rf"T_b = {tf(fine['Tb_out'], 4)}\,\mathrm{{°C}}")
                       + sub("largest " + m("T_w") + " difference along the tube " + m(tp(fine["e_Tw"], 2)) + " of " + m(r"\Delta T_{fd}")),
                       diff=m(rf"{tn(fine['Tw_out'] - fine['Tw_out_ex'], 2)}") + ", " + m(rf"{tn(fine['Tb_out'] - fine['Tb_out_ex'], 2)}\,\mathrm{{K}}")),
    }
    answers = [dict(letter=w["letter"], label=w["text"], **ans[w["key"]]) for w in wanted]

    # ---- findings ----------------------------------------------------------------------------------
    rs = seq["nu_out"]
    findings = [
        f"<b>Verification {'passed' if verdict == 'PASS' else 'gave ' + verdict}.</b> On the finest mesh ({fine['x'].size} cells along the "
        f"heated section, {num['mesh']['Nr'][-1]} across the radius) the local Nusselt number follows the exact solution with an "
        "RMS difference of " + m(tp(fine["e_rms"], 2)) + "; the error falls with observed order " + m(f"p = {p_last:.2f}")
        + " (second-order schemes: 2 expected). By region, from " + levels[-2] + " to " + levels[-1] + ": order "
        + m(f"{math.log(mets[-2]['e_rms_entr'] / fine['e_rms_entr'], 2):.2f}") + " near the start of heating ("
        + m(r"0.5\text{–}10\,D") + ") and " + m(f"{math.log(mets[-2]['e_rms_down'] / fine['e_rms_down'], 2):.2f}")
        + " downstream (" + m(r"40\text{–}100\,D") + "). Near " + m("x = 0") + " the convergence slows, probably because part "
        "of the difference there does not shrink with the mesh: the simulated velocity at " + m("x = 0") + " is slightly "
        "flatter than the parabola (Assumptions; a flatter profile has a larger wall shear and so a higher " + m("Nu")
        + ", the sign seen), and the simulation conducts heat along the axis, which the exact solution neglects. The two "
        "were not separated by a run.",
        "Fully developed Nusselt number: the simulation's own tail gives " + m(tn(fine["nu_fd"], 5)) + " against " + m(r"48/11 = 4.3636")
        + " (" + m(sgn_pct(fine["e_nu_fd"])) + "). The tube is not quite long enough to be fully developed: at the outlet the exact "
        + m("Nu") + " is still " + m(tp(fine["nu_out_ex"] / ex.nu_fd - 1, 2)) + " above " + m("Nu_{fd}") + ". Of the "
        + m(sgn_pct(fine["e_nu_fd"])) + ", " + m(sgn_pct(fine["pp_nu_fd"])) + " is the bias of the tail fit itself (the same fit "
        "on the exact solution); the entrance length carries a fit bias of " + m(sgn_pct(fine["pp_l_th"])) + ".",
        "Energy is conserved to " + m(tn(max(abs(x["energy"]) for x in mets), 2)) + " on every mesh, so the outlet bulk temperature is exact "
        "up to the iteration and round-off error, as it must be.",
        "Thermal entrance length (" + m(r"Nu \le 1.05\,Nu_{fd}") + "): " + m(rf"{tn(fine['l_th'] / D, 4)}\,D") + " simulated, "
        + m(rf"{tn(th['l_th'] / D, 4)}\,D") + " exact. Heat diffuses about 7 times more slowly than momentum ("
        + m(rf"Pr = {tn(nb['Pr'], 3)}") + "), so the temperature profile takes far longer to develop than the velocity "
        "profile, which is already within " + m(tp(abs(fine["e_uc0"]), 1)) + " of the parabola at the end of the 20-diameter "
        "unheated section.",
    ]
    if buoy is not None:
        findings.append("Gravity also makes the wall temperature vary around the tube: at the outlet the top of the wall is "
                        + m(rf"{tn(buoy['topbot'] * fine['dT_fd'], 2)}\,\mathrm{{K}}") + " warmer than the bottom ("
                        + m(tp(buoy["topbot"], 2)) + " of " + m(r"\Delta T_{fd}") + "), while the peripheral mean and "
                        + m("Nu") + " barely change. The exact solution's single " + m("T_w(x)") + " is the peripheral mean.")
        findings.append("Gravity (horizontal tube): measured effect on " + m("Nu") + " at most " + m(tp(buoy["max"], 2))
                        + (", below the " + m(tp(acc["buoyancy_ok"], 1)) + " rule: for " + m("Nu") + " and the peripheral-mean "
                           "wall temperature the no-buoyancy exact solution describes the real tube (up to " + m(r"120\,D")
                           + "; the effect still grows toward the outlet)."
                           if buoy["max"] < acc["buoyancy_ok"] else
                           ", above the " + m(tp(acc["buoyancy_ok"], 1)) + " rule: the real horizontal tube departs from the exact "
                           "solution by more than the simulation's own error."))
    findings += [f"<b>Warning:</b> {w}." for w in warns] + [f"<b>Failed:</b> {f}." for f in fails]

    # ---- figures -----------------------------------------------------------------------------------
    svgs = {"sketch": sketch()}
    xx = np.geomspace(0.5 * D, G["Lh"], 400)
    fig, axs = plots.wide(2, h=3.9)
    ax = axs[0]
    ax.semilogx(xx / D, ex.nu(xx), color=plots.INK, lw=1.5, label="Exact solution", zorder=3)
    for k, (lv, mt) in enumerate(zip(levels, mets)):
        if lv in (levels[1], levels[-1]):
            sel = mt["x"] >= 0.5 * D
            step = max(1, sel.sum() // 45)
            ax.semilogx(mt["x"][sel][::step] / D, mt["nu"][sel][::step], "o", ms=4 if lv == levels[-1] else 6,
                        color=plots.SERIES[0 if lv == levels[-1] else 1], mec=plots.SURFACE, mew=0.8, label=f"Simulation {lv}", zorder=4)
    ax.axhline(ex.nu_fd, color=plots.MUTED, lw=0.8, ls="--")
    ax.annotate(r"$Nu_{fd} = 48/11$", (2.0, ex.nu_fd), xytext=(0, -14), textcoords="offset points", color=plots.MUTED, fontsize=9)
    ax.plot([th["l_th"] / D], [1.05 * ex.nu_fd], "v", color=plots.INK, ms=6, zorder=5)
    ax.annotate(r"$L_{th}$: $Nu = 1.05\,Nu_{fd}$", (th["l_th"] / D, 1.05 * ex.nu_fd), xytext=(0, -38), textcoords="offset points", va="top",
                ha="center", color=plots.INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=plots.MUTED, lw=0.8))
    ax.set_xlabel(r"Distance from the start of heating $x/D$")
    ax.set_ylabel(r"Local Nusselt number $Nu(x)$")
    ax.set_ylim(0, 19)
    ax.legend(loc="upper right")
    ax = axs[1]
    for k, (lv, mt) in enumerate(zip(levels, mets)):
        sel = mt["x"] >= 0.5 * D
        ax.loglog(mt["x"][sel] / D, np.abs(mt["rel"][sel]), color=plots.SERIES[k % 4], lw=1.4, label=lv)
    ax.axvspan(M.X_MAX_D, G["Lh"] / D * 1.08, color=plots.GRID, alpha=0.6, lw=0, zorder=0)
    ax.set_xlim(0.4, G["Lh"] / D * 1.08)
    ax.set_xlabel(r"$x/D$  (shaded: outlet cells, not compared)")
    ax.set_ylabel(r"$|Nu_{sim}/Nu_{exact} - 1|$")
    ax.legend(loc="lower left", ncol=2)
    svgs["nu"] = plots.to_svg(fig)

    fig, ax = plots.figure(6.4, 3.8)
    rec4, arr4 = runs[-1]
    xa = arr4["x"]
    ax.plot(xa / D, arr4["Tw"], color=plots.SERIES[1], lw=2, label=r"$T_w$ simulation (finest)")
    ax.plot(xa / D, arr4["Tb"], color=plots.SERIES[0], lw=2, label=r"$T_b$ simulation (finest)")
    xe = np.linspace(0.5 * D, G["Lh"], 300)
    ax.plot(xe / D, ex.T_w(xe), "--", color=plots.INK, lw=1.2, label="Exact")
    ax.plot(xe / D, ex.T_b(xe), "--", color=plots.INK, lw=1.2)
    ax.set_xlabel(r"$x/D$")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper left")
    svgs["temps"] = plots.to_svg(fig)

    fig, ax = plots.figure(7.4, 4.0)
    hh = np.array(h) / a
    series = [("e_rms", r"$Nu(x)$, RMS"), ("e_nu_out", r"$Nu(L_h)$"), ("e_l_th", r"$L_{th}$"), ("e_Tw", r"$T_w$, max / $\Delta T_{fd}$")]
    for k, (key, lab) in enumerate(series):
        ax.loglog(hh, [abs(x[key]) for x in mets], "o-", color=plots.SERIES[k], ms=5, mec=plots.SURFACE, mew=1, label=lab)
    ref = abs(mets[0]["e_rms"]) * 0.4 * (hh / hh[0]) ** 2
    ax.loglog(hh, ref, "--", color=plots.MUTED, lw=1, label="slope 2")
    ax.set_xlabel(r"Mesh size $1/N_r$ (L1 to L4)")
    ax.set_ylabel("Relative error")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    svgs["conv"] = plots.to_svg(fig)

    fig, ax = plots.figure(5.6, 3.6)
    rr = np.linspace(0, a, 200)
    ax.plot(2 * G["U"] * (1 - (rr / a) ** 2), rr / a, color=plots.INK, lw=1.4, label="Parabola (exact solution's assumption)")
    ax.plot(fine["u0"], fine["r"] / a, "o", color=plots.SERIES[0], ms=3.5, label=r"Simulation at $x = 0$ (finest)")
    ax.set_xlabel(r"Axial velocity $u$ (m/s)")
    ax.set_ylabel(r"$r/a$")
    ax.set_xlim(0, 0.11)
    ax.legend(loc="lower left", fontsize=9)
    svgs["velocity"] = plots.to_svg(fig)

    fig, ax = plots.figure(6.0, 3.6)
    for k, f in enumerate(("Ux", "p_rgh", "T")):
        rv = arr4.get(f"res_{f}")
        if rv is not None and len(rv):
            ax.semilogy(np.arange(1, len(rv) + 1), rv, color=plots.SERIES[k], lw=1.2, label=f)
    ax.set_xlabel(f"Iteration ({levels[-1]}, started from {levels[-2]})")
    ax.set_ylabel("Initial residual")
    ax.legend(loc="upper right")
    svgs["residuals"] = plots.to_svg(fig)

    figures = [
        dict(key="sketch", section="problem", caption="The tube, the two wall sections and the coordinates (not to scale)."),
        dict(key="velocity", section="simulation",
             caption="Velocity profile where heating starts. The simulation develops it from a uniform inlet over 20 diameters; "
                     "the exact solution assumes the parabola. Centreline difference " + m(sgn_pct(fine["e_uc0"])) + "."),
        dict(key="residuals", section="convergence",
             caption="Initial residuals on the finest mesh. The run was ended by the live stop on the probe values, not on residuals."),
        dict(key="nu", section="comparison",
             caption="Left: local Nusselt number, exact (line) and simulated on L2 and L4 (dots, every few cells). Right: relative "
                     "difference on each mesh. Per refinement it falls by a factor "
                     + ", ".join(m(tn(mets[i]["e_rms_down"] / mets[i + 1]["e_rms_down"], 2)) for i in range(len(mets) - 1))
                     + " between " + m(r"40") + " and " + m(r"100\,D") + " and by "
                     + ", ".join(m(tn(mets[i]["e_rms_entr"] / mets[i + 1]["e_rms_entr"], 2)) for i in range(len(mets) - 1))
                     + " near the start of heating (second order would be 4)."),
        dict(key="temps", section="comparison",
             caption="Wall and bulk temperature along the tube, including the insulated section " + m("x < 0") + ". Dashed: exact "
                     "solution. The wall-to-bulk gap settles to " + m(rf"\Delta T_{{fd}} = {tn(fine['dT_fd'], 3)}\,\mathrm{{K}}") + "."),
        dict(key="conv", section="comparison",
             caption="Error of each requested quantity against mesh size. Dashed: slope 2. The entrance-length error does not "
                     "fall: on the finer meshes it is dominated by the bias of the extraction itself (" + m(sgn_pct(fine["pp_l_th"]))
                     + " when applied to the exact solution, Comparison findings)."),
    ]
    if buoy is not None:
        fig, axs = plots.wide(2, h=3.6)
        sel = g0["x"] >= M.X_MIN_D * D
        ax = axs[0]
        ax.semilogx(g0["x"][sel] / D, 100 * buoy["d_nu"][sel], color=plots.SERIES[1], lw=2)
        ax.axhline(100 * acc["buoyancy_ok"], color=plots.MUTED, lw=0.8, ls="--")
        ax.annotate("rule: 1 %", (1.0, 100 * acc["buoyancy_ok"]), xytext=(0, -14), textcoords="offset points",
                    color=plots.MUTED, fontsize=9)
        ax.set_xlabel(r"$x/D$")
        ax.set_ylabel(r"$Nu_{gravity}/Nu_{g=0} - 1$ (%)")
        ax.set_ylim(-0.05, 1.1)
        ax = axs[1]
        ax.semilogx(g1["x"][sel] / D, (g1["Tw_top"] - g1["Tw_bot"])[sel] / fine["dT_fd"], color=plots.SERIES[1], lw=2, label="with gravity")
        ax.semilogx(g0["x"][sel] / D, (g0["Tw_top"] - g0["Tw_bot"])[sel] / fine["dT_fd"], color=plots.SERIES[0], lw=2, label=r"$g = 0$")
        ax.set_xlabel(r"$x/D$")
        ax.set_ylabel(r"$(T_{w,top} - T_{w,bottom}) / \Delta T_{fd}$")
        ax.legend(loc="upper left")
        svgs["buoyancy"] = plots.to_svg(fig)
        figures.append(dict(key="buoyancy", section="simulation",
                            caption="Buoyancy check on the 3D half tube, same mesh with and without gravity. Left: change of the "
                                    "peripherally averaged Nusselt number (largest " + m(tp(buoy["max"], 2)) + ", dashed: the 1 % rule). "
                                    "Right: wall temperature difference between the top and the bottom of the tube."))

    # ---- pre-flight --------------------------------------------------------------------------------
    pfs = json.loads((OUT / "preflight.json").read_text())
    pf_runs, pf_checks = [], []
    for pf in pfs:
        nm = f"{pf['level']} ({pf['cells']} cells), " + ("GAMG" if "GAMG" in pf["p_solver"] else "PCG/DIC") + " pressure solver"
        pf_runs.append(dict(name=nm, setup=f"{pf['end_time_short']} iterations from rest, " + m(r"2^\circ") + " wedge",
                            outcome=f"{pf['status']}; {pf['seconds_per_iteration']:.3g} s per iteration"))
        for c in pf["checks"]:
            pf_checks.append(dict(c, name=f"{c['name']} ({pf['level']}, {'GAMG' if 'GAMG' in pf['p_solver'] else 'PCG'})"))
    pf1rec, pf1arr = M.load("pfL1full")
    pf1 = M.wedge(pf1rec, pf1arr, ex, G, half)
    pf_runs.append(dict(name="L1 to convergence (PCG pressure solver)",
                        setup="live stop rule, from rest, " + m(r"2^\circ") + " wedge",
                        outcome=f"{pf1rec['iterations']} iterations; error of " + m("Nu") + f" (RMS) {100 * pf1['e_rms']:.2f} %, of "
                                + m("Nu(L_h)") + " " + m(sgn_pct(pf1["e_nu_out"])) + ", of " + m("L_{th}") + " " + m(sgn_pct(pf1["e_l_th"]))
                                + " (outlet values from the fit as in production)"))
    pf4 = [p for p in pfs if p["level"] == levels[-1]]
    spi = {p["variant"]: p["seconds_per_iteration"] for p in pf4}

    # ---- performance -------------------------------------------------------------------------------
    perf_rows, total_wall = [], 0.0
    all_runs = [(r[0]["name"], r[0]) for r in runs] + [(f"B3D_{t}", b3[t][0]) for t in b3]
    for nm, r in all_runs:
        pr = r["perf"]
        total_wall += r["wall_time"]
        perf_rows.append([nm, f"{r['cells']:,}", r["cores"], pr["steps"], f"{r['wall_time']:.0f}",
                          f"{pr['us_per_cell_step']:.2f}" if pr["us_per_cell_step"] else "–",
                          f"{pr['lin_iters'].get('p_rgh', float('nan')):.0f}", f"{pr['lin_iters'].get('T', float('nan')):.0f}"])
    pf_wall = sum(p.get("wall_time") or 0 for p in pfs) + pf1rec["wall_time"]

    # ---- results.json ------------------------------------------------------------------------------
    lvrow = lambda i, mt: f"{levels[i]}"
    results = dict(
        title=problem["title"], date=dt.date.today().isoformat(), verdict=verdict, comparison_type="verification",
        summary=dict(
            headline=(f"<b>{verdict}.</b> The simulated local Nusselt number matches the exact solution to "
                      + m(tp(fine["e_rms"], 2)) + " (RMS, finest mesh), the fully developed value to " + m(sgn_pct(fine["e_nu_fd"]))
                      + " and the 5 % entrance length " + m(rf"{tn(th['l_th'] / D, 3)}\,D") + " to " + m(sgn_pct(fine["e_l_th"])) + "."),
            answer_columns=["Requested", "Exact solution", f"Simulation ({levels[-1]})", "Difference"],
            answers=answers, findings=findings,
            limitations=[
                "The exact solution assumes a parabolic velocity at the start of heating and no axial conduction; the simulation "
                "keeps both effects of the real tube. Their effect is part of the measured difference, largest near " + m("x = 0") + ".",
                "Constant properties: the viscosity change (" + m(tp(nb["mu_change_wb"], 2)) + " between wall and bulk) is estimated, "
                "not simulated.",
                ("Buoyancy was checked on one 3D mesh only (the effect is a difference of two runs on the same mesh, so most of "
                 "the mesh error cancels); a 3D mesh study was not made." if buoy is not None else "Buoyancy was not measured."),
                "The outlet is at " + m(r"120\,D") + ": the flow there is " + m(tp(fine["nu_out_ex"] / ex.nu_fd - 1, 2))
                + " short of fully developed, so " + m("Nu_{fd}") + " comes from a fit of the simulated tail, not from a direct reading.",
            ]),
        problem=dict(description=problem["description"], setup=problem["setup"],
                     given=[dict(label=re.sub(r"\((\d+) D\)", r"(\\(\1\\,D\\))", v["label"]), symbol=v["symbol"], value=m(tn(v["value"])), unit=v["unit"]) for v in problem["given"].values()],
                     wanted=wanted, reference=None),
        assumptions=dict(rows=assumptions),
        theory=dict(
            intro=("This problem has an exact solution: the extended Graetz problem for a uniform wall heat flux (Siegel, Sparrow and "
                   "Hallman 1958), for a parabolic velocity profile, constant properties and no axial conduction. Its fully developed "
                   "part is derived and checked by SymPy; the entrance region is an eigenfunction series (Kummer functions) evaluated "
                   "with mpmath and checked against an independent finite-volume solution of the same equation. Nothing is taken "
                   "from tables."),
            steps=th["steps"], derived=th["derived"],
            self_checks=[dict(name=k, ok=bool(v)) for k, v in th["self_checks"].items()]),
        decisions=load_yaml(HERE / "decisions.yaml"),
        preflight=dict(
            purpose=("Short runs from rest on the coarsest and the finest mesh check that the setup runs and is stable and give the "
                     "time per iteration; the finest mesh is run with both pressure solvers. The coarsest mesh is also run to "
                     "convergence: its error against the exact solution, with the expected order 2, sized the acceptance tolerances "
                     "before any production run."),
            runs=pf_runs, checks=pf_checks,
            conclusion=(f"After the first pre-flight failed (SIMPLEC is not available in this solver, see Insights), all checks pass. "
                        f"On {levels[-1]} the PCG pressure solver took {spi.get('chosen', float('nan')):.3g} s per iteration and GAMG "
                        f"{spi.get('gamg', float('nan')):.3g} s, so production uses GAMG (on L3 the ranking had been reversed, "
                        f"Insights). The converged L1 errors ({100 * pf1['e_rms']:.2f} % RMS in " + m("Nu") + ") set the "
                        f"finest-mesh tolerances at one sixteenth of them (order 2 over three refinements would give one "
                        f"sixty-fourth). The pre-flight used a " + m(r"2^\circ") + " wedge; it was cut to " + m(r"0.5^\circ")
                        + " before production because its geometric bias was close to those tolerances.")),
        simulation=dict(
            model=[
                dict(aspect="Geometry", real="Round tube, 140 diameters long",
                     model=m(rf"{2 * half:g}^\circ") + " axisymmetric wedge, one cell around, from the inlet to the outlet"),
                dict(aspect="Heated wall", real=m(rf"q_w = {tn(G['q_w'])}\,\mathrm{{W/m^2}}"),
                     model="Fixed temperature gradient " + m(rf"\partial T/\partial n = q_w/k = {tn(G['q_w'] / G['k'], 5)}\,\mathrm{{K/m}}")),
                dict(aspect="Insulated wall", real="No heat flux", model="Zero temperature gradient"),
                dict(aspect="Water", real="20 °C", model=m(rf"\nu = {tn(G['nu'])}\,\mathrm{{m^2/s}},\ Pr = {tn(nb['Pr'], 6)}") + ", constant"),
                dict(aspect="Gravity", real="Across the axis", model="Off in the main study; on in the 3D buoyancy check"),
            ],
            meshes=dict(
                intro=("Four meshes; each doubles every cell count. Cells are concentrated where heating starts and toward the wall."),
                head=["Mesh", m("N_r"), "Cells along " + m("x") + " (unheated + heated)", "Cells", "First heated cell " + m(r"\Delta x/D"),
                      "Last heated cell " + m(r"\Delta x/D")],
                rows=[[lv, num["mesh"]["Nr"][i], f"{num['mesh']['Nx_unheated'][i]} + {num['mesh']['Nx_heated'][i]}",
                       f"{runs[i][0]['cells']:,}", m(tn(runs[i][0]['axial']['dx0'] / D, 3)), m(tn(runs[i][0]['axial']['dx_out'] / D, 3))]
                      for i, lv in enumerate(levels)]),
            results=dict(
                head=["Mesh", m("Nu(L_h)"), m("Nu_{fd}") + " (tail fit)", m("L_{th}/D"), m(r"T_w(L_h)") + " (°C)", m(r"T_b(L_h)") + " (°C)"],
                rows=[[lv, m(tf(mt["nu_out"], 4)), m(tf(mt["nu_fd"], 4)), m(tf(mt["l_th"] / D, 2)), m(tf(mt["Tw_out"], 5)),
                       m(tf(mt["Tb_out"], 6))] for lv, mt in zip(levels, mets)]
                     + [["Exact", m(tf(fine["nu_out_ex"], 4)), m(tf(ex.nu_fd, 4)), m(tf(th["l_th"] / D, 2)), m(tf(fine["Tw_out_ex"], 5)),
                         m(tf(fine["Tb_out_ex"], 6))]]),
            health=dict(
                head=["Mesh", "Iterations", "Live stop", "Energy balance", "Mass balance", "Log checks", "Time (s)"],
                rows=[[lv, r[0]["iterations"], ("✓ " if r[0]["live_stop"]["stopped"] else "✗ ") + (r[0]["live_stop"]["decision"] or {}).get("status", "–"),
                       m(tn(mt["energy"], 2)), m(tn(mt["mass"], 2)), report.CHECK_LEGEND[r[0]["health"]["status"]], f"{r[0]['wall_time']:.0f}"]
                      for lv, r, mt in zip(levels, runs, mets)],
                notes=[
                    "Each run stopped live once the estimated iteration error of all 9 temperature and velocity probes was below "
                    + m(rf"{tn(num['convergence']['tol_T'])}\,\mathrm{{K}}") + " and " + m(rf"{tn(num['convergence']['tol_U'])}\,\mathrm{{m/s}}")
                    + " (or a probe stayed unchanged in its 12 printed digits, Insights), held for 2 checks. Each finer mesh started from the coarser answer.",
                    "Energy balance: " + m(r"\rho c_p \sum \phi T|_{\mathrm{out}+\mathrm{in}} / (q_w A_h) - 1") + "; mass balance: net volume flux / inflow.",
                    "The residual of " + m("U_z") + " stays near " + m(r"10^{-4}") + ": the wedge's out-of-plane velocity is zero, "
                    "so its normalised residual is round-off (primer pitfall 3).",
                ])),
        comparison=dict(
            kind="Verification: the simulation is compared with the exact solution of the idealised problem.",
            definitions=[
                dict(name="RMS error of Nu(x)", latex=r"\varepsilon_{rms} = \sqrt{\frac{\int (Nu_{sim}/Nu_{ex} - 1)^2\,dx}{\int dx}}",
                     meaning="Typical relative difference along the heated section, weighted by cell length, over "
                             + m(r"0.5\,D \le x \le 100\,D") + " (the last cells before the outlet are excluded, see Insights)."),
                dict(name="Observed order", latex=r"p = \frac{\ln(\varepsilon_{coarse}/\varepsilon_{fine})}{\ln 2}",
                     meaning=m(r"\varepsilon \propto h^p") + "; 2 for these schemes."),
                dict(name="Fully developed Nu of the simulation", latex=r"T_w - T_b \approx A + B\,e^{-c x},\quad Nu_{fd} = \frac{q_w D}{k A}",
                     meaning="A three-parameter least-squares fit of the simulated tail, " + m(r"60\,D \le x \le 100\,D")
                             + "; no input from the exact solution. " + m("Nu(L_h)") + " is the same fit evaluated at the outlet."),
                dict(name="Wall temperature error", latex=r"\max_x |T_{w,sim} - T_{w,ex}| / \Delta T_{fd}",
                     meaning="Largest wall-temperature difference, as a fraction of the wall-to-bulk difference that sets " + m("Nu") + "."),
            ],
            head=["Mesh", m(r"\varepsilon_{rms}"), m("p"), "max, " + m(r"x \ge 5\,D"), m("Nu(L_h)"), m("Nu_{fd}"), m("L_{th}"), m("T_w")],
            rows=[[lv, m(tp(mt["e_rms"], 2)), m(f"{order_rms[i - 1]:.2f}") if i else "–", m(tp(mt["e_max5"], 2)),
                   m(sgn_pct(mt["e_nu_out"])), m(sgn_pct(mt["e_nu_fd"])), m(sgn_pct(mt["e_l_th"])), m(tp(mt["e_Tw"], 2))]
                  for i, (lv, mt) in enumerate(zip(levels, mets))],
            findings=_cmp_findings(seq, gci, mets, ex, th, fine, D, acc)),
        insights=insights.load(HERE.name),
        figures=figures,
        performance=dict(
            intro=("Run cost per production run. The 2D runs (one core each) ran one after another; the 3D pair (4 MPI ranks) ran "
                   "at the same time as the 2D sequence, so both shared the machine."),
            head=["Run", "Cells", "Cores", "Iterations", "Wall (s)", "µs per cell per iteration", "p iterations per solve", "T iterations per solve"],
            rows=perf_rows,
            notes=[f"Total wall time of the production runs: {total_wall / 60:.1f} min; pre-flight runs: {pf_wall / 60:.1f} min. "
                   "Not counted: the relaxation and linear-solver trials and the production attempts discarded after the problems "
                   "in Insights (wrong pressure tolerance, never-ending live stop, orphaned solvers).",
                   "Cells up to 300 times longer than wide slow the linear solvers: on L4 the temperature solve (PBiCGStab, about 140 "
                   "iterations) costs more than the pressure solve. For the pressure, GAMG beat PCG/DIC on L4 (0.62 against 0.74 s "
                   "per iteration in the pre-flight), the reverse of the L3 trial (Insights).",
                   "Next: SIMPLEC with simpleFoam plus a scalarTransport function object for the main study (the buoyant solver has "
                   "no SIMPLEC); GAMG or relaxation for the temperature equation; fewer, less stretched cells near the outlet."]),
        reproducibility=dict(
            commands=[
                dict(cmd=f"python -m cases.{HERE.name}.theory", what="Exact solution and its self-checks (prints the key numbers)"),
                dict(cmd=f"python -m cases.{HERE.name}.run preflight", what="Pre-flight runs (L1, L4 with two pressure solvers) and L1 to convergence"),
                dict(cmd=f"python -m cases.{HERE.name}.run production", what="L1 to L4, live stop, each from the coarser answer"),
                dict(cmd=f"python -m cases.{HERE.name}.run buoyancy", what="3D half tube with g = 0, then with gravity (4 MPI ranks)"),
                dict(cmd=f"python -m cases.{HERE.name}.analyze", what="Comparison, figures, this report"),
            ],
            files=[dict(path=f"cases/{HERE.name}/{f}", what=w) for f, w in (
                ("problem.yaml", "The problem as given"), ("numerics.yaml", "Meshes, solver settings, stopping rule, acceptance rules"),
                ("decisions.yaml", "Every choice with the settings it controls"), ("insights.yaml", "Live workflow log"),
                ("theory.py", "Exact solution (SymPy + eigenfunction series) and self-checks"), ("run.py", "Meshes, runs, post-processing"),
                ("metrics.py", "Nu, temperatures, balances per run"), ("template/", "OpenFOAM case with placeholders"))],
            run_dirs=[dict(path=str(Path(finerec["case_dir"]).parent) + "/<L1..L4>", what="Production runs"),
                      dict(path=str(Path(finerec["case_dir"]).parent) + "/B3D_g0, B3D_g1", what="3D buoyancy check"),
                      dict(path=str(Path(finerec["case_dir"]).parent) + "/pf*", what="Pre-flight runs"),
                      dict(path=f"results/{HERE.name}/", what="runs/*.json|npz, preflight.json, results.json, report.html")],
            environment=report.environment_info(),
            notes=["2D runs: one core each, one after another. 3D runs: 4 MPI ranks, one after another, but at the same time as "
                   "the 2D sequence (wall times are inflated by the sharing).",
                   "Relaxation and pressure-solver trials ran side by side (2 to 4 at once), which inflates their times.",
                   "Run the commands from a state with no solver running (<code>python -m cfdlib.foam ps</code>): a leftover solver "
                   "writing into a re-created case directory corrupts it (Insights)."]),
        data=dict(levels=levels, order_rms=order_rms, seq=seq, gci=gci, buoyancy={k: v for k, v in (buoy or {}).items() if np.isscalar(v)},
                  metrics=[{k: v for k, v in mt.items() if np.isscalar(v)} for mt in mets],
                  pf1={k: v for k, v in pf1.items() if np.isscalar(v)}, total_wall_s=total_wall, preflight_wall_s=pf_wall),
    )
    out = report.render(results, svgs, OUT / "report.html", case_dir=finerec["case_dir"])
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"{verdict}: Nu rms {100 * fine['e_rms']:.3f} %, order {p_last:.2f}; warns {warns}; fails {fails}. Report: {out}")


def _cmp_findings(seq, gci, mets, ex, th, fine, D, acc):
    out = ["<b>Post-processing bias.</b> " + m("Nu_{fd}") + ", " + m("Nu(L_h)") + " and " + m("L_{th}") + " are extracted by a "
           "tail fit and a threshold crossing. The same extraction applied to the exact solution, sampled at the "
           + mets[-1]["x"].size.__str__() + " cell positions of the finest mesh, is off by " + m(sgn_pct(fine["pp_nu_fd"], 2)) + ", "
           + m(sgn_pct(fine["pp_nu_out"], 2)) + " and " + m(sgn_pct(fine["pp_l_th"], 2)) + ": that part of the differences belongs "
           "to the method, not to the simulation. The simulation's own share (difference minus bias) on the finest mesh: "
           + m(sgn_pct(fine["e_nu_fd"] - fine["pp_nu_fd"], 2)) + ", " + m(sgn_pct(fine["e_nu_out"] - fine["pp_nu_out"], 2)) + ", "
           + m(sgn_pct(fine["e_l_th"] - fine["pp_l_th"], 2)) + ". The verdict uses the rules as fixed before the runs, "
           "bias included."]
    for k, lab, exact, bias in (("nu_out", "Nu(L_h)", fine["nu_out_ex"], fine["pp_nu_out"]), ("nu_fd", "Nu_{fd}", ex.nu_fd, fine["pp_nu_fd"]),
                                ("l_th", r"L_{th}", th["l_th"], fine["pp_l_th"]), ("Tw_out", r"T_w(L_h)", fine["Tw_out_ex"], None)):
        s = seq[k]
        if s["order"] and s["plausible"] and gci[k] is not None:
            band = gci[k] * abs(mets[-1][k])
            ok = abs(s["richardson"] - exact) <= band + 1e-12 * abs(exact) or abs(mets[-1][k] - exact) <= band
            txt = (m(lab) + ": observed order " + m(f"{s['order']:.2f}") + ", Richardson value " + m(tn(s["richardson"], 6))
                   + ", finest-mesh GCI " + m(tp(gci[k], 2)) + "; the exact value " + m(tn(exact, 6))
                   + (" lies within the band." if ok else " lies OUTSIDE the band: a difference that refinement does not remove."))
            if not ok and bias is not None:
                target = exact * (1 + bias)
                ok2 = abs(s["richardson"] - target) <= band
                txt += (" The exact solution passed through the same extraction gives " + m(tn(target, 6)) + ", "
                        + (m(tp(abs(s["richardson"] / target - 1), 1)) + " from the Richardson value"
                           + (", within the band: the remaining difference is the extraction's bias, not the simulation's."
                              if ok2 else ", still outside the band (" + m(tp(band / abs(target), 2)) + "): part of the "
                              "difference is neither mesh error nor extraction bias; the candidates are the exact solution's "
                              "idealisations (axial conduction, the slightly undeveloped velocity at " + m("x = 0") + ").")))
            out.append(txt)
        else:
            out.append(m(lab) + ": the last three meshes do not converge monotonically (changes "
                       + ", ".join(m(tn(mets[i][k] - mets[i - 1][k], 2)) for i in range(len(mets) - 2, len(mets)))
                       + "), so no Richardson value; the finest-mesh difference from the exact value is "
                       + m(sgn_pct(mets[-1][k] / exact - 1)) + ".")
    return out


if __name__ == "__main__":
    main()
