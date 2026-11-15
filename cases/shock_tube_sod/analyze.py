"""Compare the shock-tube runs with the exact solution; build results.json and the report.

Usage (repository root, inside WSL):  python -m cases.shock_tube_sod.analyze
Reads results/shock_tube_sod/{preflight.json, gas_constant.json, runs/*.json, runs/*.npz} and the case YAML files.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np

from cfdlib import plots, report, verification as V
from cfdlib.report import LETTERS, load_yaml, m, m_fix, m_pct, tex_fix as tf, tex_num as tn
from . import measure, theory
from .run import specs
from .sketch import sketch

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"
CHECK = {"ok": "✓ OK", "warn": "⚠ warning", "fail": "✗ fail"}
WAVES = ("head", "tail", "contact", "shock")
WAVE = dict(head="front of the fan", tail="back of the fan", contact="border of the two gases", shock="shock")
QTY = (("rho", r"\rho", "density"), ("u", "u", "velocity"), ("p", "p", "pressure"))
REFS = dict(
    sod="G. A. Sod, J. Comput. Phys. 27 (1978) 1–31",
    knp="A. Kurganov, S. Noelle and G. Petrova, SIAM J. Sci. Comput. 23 (2001) 707–740",
    green="C. J. Greenshields, H. G. Weller, L. Gasparini and J. M. Reese, Int. J. Numer. Meth. Fluids 63 (2010) 1–21",
    banks="J. W. Banks, T. Aslam and W. J. Rider, J. Comput. Phys. 227 (2008) 6985–7002")


def sc(x, sig=2):
    """A small number as math, e.g. 2.4 x 10^-14."""
    return m(tn(x, sig))


def pct(x, sig=2):
    """A fraction as per cent with `sig` significant digits and no exponent (0.000068 -> 0.0068 %)."""
    return report.m_sig(x, sig, pct=True, unit=True)


def analyse(meta, d, ex, g, ext):
    """Everything the report needs from one run, at the time of interest unless said otherwise."""
    N, x, L, tend = meta["N"], d["x"], g["L"], g["t_end"]
    w = ex.positions(tend)
    exc = ex.cell_averages(np.linspace(0, L, N + 1), tend)
    sim = (d["rho"][-1], d["u"][-1], d["p"][-1])
    scales = (g["rho_L"], ex.u_star, g["p_L"])
    r = {k: meta[k] for k in ("name", "N", "spc", "flux", "limiter", "studies", "steps", "dt", "h", "wall_time",
                              "mesh_ok", "case_dir", "run_root", "perf")}
    h = meta["health"]
    r.update(health=h["status"], mass_drift=h["mass_drift"], energy_drift=h["energy_drift"], momentum_err=h["momentum_err"],
             mass_start=h["mass_start"], energy_start=h["energy_start"], ends=h["ends"],
             co_solver=float(next(c["value"] for c in h["checks"] if c["name"] == "Max Courant number")),
             courant=ex.s_max * meta["dt"] / meta["h"])
    r["e"] = {k: measure.l1(sim[i], exc[i], scales[i]) for i, (k, _, _) in enumerate(QTY)}
    r["by_wave"] = {k: measure.l1_by_wave(x, sim[i], exc[i], scales[i], w, L) for i, (k, _, _) in enumerate(QTY)}
    r["wiggle"] = dict(rho=measure.total_variation(sim[0]) / (g["rho_L"] - g["rho_R"]) - 1,
                       p=measure.total_variation(sim[2]) / (g["p_L"] - g["p_R"]) - 1)
    r["read"] = measure.read_all(x, *sim, w, L, g["p_R"], ext)
    r["read_exact"] = measure.read_all(x, *exc, w, L, g["p_R"], ext)
    st_exact = dict(p=ex.p_star, u=ex.u_star, rho_L=ex.rho_starL, rho_R=ex.rho_starR)
    r["state"] = {k: r["read"]["states"][k]["mean"] for k in st_exact}
    r["state_err"] = {k: r["state"][k] / st_exact[k] - 1 for k in st_exact}
    r["ripple"] = {k: r["read"]["states"][k]["spread"] / abs(r["state"][k]) for k in ("p", "u")}
    r["pos"] = {k: r["read"]["positions"][k] for k in WAVES}
    r["pos_err"] = {k: r["pos"][k] - w[k] for k in WAVES}                      # metres
    r["pos_bias"] = {k: r["read_exact"]["positions"][k] - w[k] for k in WAVES}
    r["width"] = r["read"]["widths"]
    r["width_exact"] = r["read_exact"]["widths"]
    r["pos_t"] = [measure.read_all(x, d["rho"][k], d["u"][k], d["p"][k], ex.positions(float(t)), L, g["p_R"], ext)["positions"]
                  for k, t in enumerate(d["times"])]
    r["speed"] = measure.speeds(d["times"], r["pos_t"], ext["speed_from"] * tend)
    r["speed_err"] = {k: r["speed"][k] - ex.speeds[k] for k in WAVES}          # m/s
    r["speed_last"] = {k: (r["pos"][k] - ex.x0) / tend for k in WAVES}
    r.update(x=x, sim=sim, exc=exc, times=d["times"], c_t=d["c_t"], c_mass=d["c_mass"], c_energy=d["c_energy"], c_mom=d["c_mom"],
             rho_t=d["rho"])
    return r


def main():
    problem = load_yaml(HERE / "problem.yaml")
    numerics = load_yaml(HERE / "numerics.yaml")
    g = {k: v["value"] for k, v in problem["given"].items()}
    L, tend = g["L"], g["t_end"]
    acc, ext, mainc = numerics["acceptance"], numerics["extraction"], numerics["main"]
    th, ex = theory.evaluate(problem["given"])
    dmap = {x["key"]: x for x in th["derived"]}
    w = ex.positions(tend)
    pf = json.loads((OUT / "preflight.json").read_text())
    gc = json.loads((OUT / "gas_constant.json").read_text())
    sp_all = specs(numerics)

    R = {}
    for name in sp_all:
        meta = json.loads((RUNS / f"{name}.json").read_text())
        R[name] = analyse(meta, np.load(RUNS / f"{name}.npz"), ex, g, ext)
    fam = [R[f"N{N}"] for N in numerics["runs"]["family"]]
    coarse, fine = fam[0], fam[-1]
    ts = sorted((r for r in R.values() if "time step" in r["studies"]), key=lambda r: r["spc"])      # largest step first
    sch = [r for r in R.values() if "scheme" in r["studies"]]
    main_ts = next(r for r in ts if r["spc"] == mainc["steps_per_cell"])
    hs = [r["h"] for r in fam]
    Ns = [r["N"] for r in fam]

    # ---- orders -----------------------------------------------------------------------------------
    order = {k: V.order_from_errors(hs, [r["e"][k] for r in fam]) for k, _, _ in QTY}
    overall = {k: float(-np.polyfit(np.log(Ns), np.log([r["e"][k] for r in fam]), 1)[0]) for k, _, _ in QTY}
    wave_overall = {k: {z: float(-np.polyfit(np.log(Ns), np.log([r["by_wave"][k][z] for r in fam]), 1)[0])
                        for z in ("fan", "contact", "shock")} for k, _, _ in QTY}
    share = {k: {z: fine["by_wave"][k][z] / fine["e"][k] for z in ("fan", "contact", "shock")} for k, _, _ in QTY}
    width_exp = {z: float(np.polyfit(np.log(Ns), np.log([r["width"][z] for r in fam]), 1)[0]) for z in ("shock", "contact")}
    wiggle_exp = float(np.polyfit(np.log(Ns), np.log([r["wiggle"]["p"] for r in fam]), 1)[0])
    shock_w = [r["width"]["shock"] for r in fam]
    # time-step study: change of the density profile each time the step is halved, and its order
    ts_change = [measure.l1(b["sim"][0], a["sim"][0], g["rho_L"]) for a, b in zip(ts[:-1], ts[1:])]
    ts_order = [float(np.log2(ts_change[i] / ts_change[i + 1])) for i in range(len(ts_change) - 1)]
    ts_main_vs_small = measure.l1(main_ts["sim"][0], ts[-1]["sim"][0], g["rho_L"])

    # ---- verdict: the rules of numerics.yaml, written before the production runs ---------------------
    fails, warns, rules = [], [], []

    def rule(kind, text, value, limit, ok, shown=None):
        rules.append(dict(kind=kind, text=text, value=value, limit=limit, ok=bool(ok), shown=shown))
        if not ok:
            (fails if kind == "fail" else warns).append(text)

    allr = list(R.values())
    rule("fail", "every grid passes checkMesh and every run ends without an error", "", "", all(r["mesh_ok"] for r in allr)
         and all(r["health"] != "fail" for r in allr), "all runs")
    worst = {k: max(r[k] for r in allr) for k in ("mass_drift", "energy_drift", "momentum_err", "ends")}
    rule("fail", "total mass does not change during a run", sc(worst["mass_drift"]), sc(acc["fail_mass_drift"], 1),
         worst["mass_drift"] <= acc["fail_mass_drift"], "largest of all runs")
    rule("fail", "total energy does not change during a run", sc(worst["energy_drift"]), sc(acc["fail_energy_drift"], 1),
         worst["energy_drift"] <= acc["fail_energy_drift"], "largest of all runs")
    rule("fail", "total momentum equals the push of the end walls", sc(worst["momentum_err"]), sc(acc["fail_momentum"], 1),
         worst["momentum_err"] <= acc["fail_momentum"], "largest of all runs")
    rule("fail", "the two ends are still undisturbed at the time of interest", sc(worst["ends"]), sc(acc["fail_ends"], 1),
         worst["ends"] <= acc["fail_ends"], "largest of all runs")
    l1_limit = {k: coarse["e"][k] * (fine["h"] / coarse["h"]) ** acc["fail_l1_order_floor"][k] for k, _, _ in QTY}
    for k, sym, word in QTY:
        rule("fail", f"mean {word} error on the finest grid", pct(fine["e"][k]), pct(l1_limit[k]), fine["e"][k] <= l1_limit[k],
             f"{fine['N']} cells")
    worst_state = max(abs(v) for v in fine["state_err"].values())
    rule("fail", "uniform states between the waves on the finest grid", pct(worst_state), pct(acc["fail_state"], 1),
         worst_state <= acc["fail_state"], "largest of the four")
    worst_jump = max(abs(fine["pos_err"][k]) for k in ("contact", "shock")) / L
    worst_fan = max(abs(fine["pos_err"][k]) for k in ("head", "tail")) / L
    rule("fail", "position of the shock and of the border of the two gases on the finest grid", sc(worst_jump) + " of " + m("L"),
         sc(acc["fail_position_jump"], 1), worst_jump <= acc["fail_position_jump"], "larger of the two")
    rule("fail", "position of the two ends of the fan on the finest grid", sc(worst_fan) + " of " + m("L"),
         sc(acc["fail_position_fan"], 1), worst_fan <= acc["fail_position_fan"], "larger of the two")
    worst_speed = max(abs(fine["speed_err"][k]) for k in WAVES) / ex.S
    rule("fail", "speed of the four waves on the finest grid", sc(worst_speed) + " of " + m("S"), sc(acc["fail_speed"], 1),
         worst_speed <= acc["fail_speed"], "largest of the four")
    for k, sym, word in QTY:
        lo, hi = acc["warn_order"][k]
        rule("warn", f"observed order of the {word} error at the last refinement", m_fix(order[k][-1], 2), f"{lo} to {hi}",
             lo <= order[k][-1] <= hi, f"{fam[-2]['N']} to {fine['N']} cells")
    worst_wiggle = max(max(r["wiggle"].values()) for r in fam)
    rule("warn", "extra up-and-down of density and pressure (wiggles), every grid of the family", m_pct(worst_wiggle, 1),
         m_pct(acc["warn_wiggle"], 0), worst_wiggle <= acc["warn_wiggle"], "largest of the family")
    worst_ripple = max(fine["ripple"].values())
    rule("warn", "ripple of pressure and velocity between the fan and the shock, finest grid", m_pct(worst_ripple, 2),
         m_pct(acc["warn_plateau_ripple"], 0), worst_ripple <= acc["warn_plateau_ripple"], "larger of the two")
    rule("warn", "width of the shock on every grid", m_fix(max(shock_w), 2) + " cells", m_fix(acc["warn_shock_cells"], 0) + " cells",
         max(shock_w) <= acc["warn_shock_cells"], "widest of the family")
    verdict = "FAIL" if fails else "WARN" if warns else "PASS"
    n_fail_rules = sum(1 for x in rules if x["kind"] == "fail")
    n_warn_rules = len(rules) - n_fail_rules

    # ---- assumptions ---------------------------------------------------------------------------------
    T_all = [g["p_L"] / (g["rho_L"] * g["R"]), g["p_R"] / (g["rho_R"] * g["R"]), ex.p_star / (ex.rho_starL * g["R"]),
             ex.p_star / (ex.rho_starR * g["R"])]
    mach = (ex.u_star / ex.c_starL, ex.u_star / ex.c_starR)
    assumptions = [
        dict(name="The flow depends on the position along the tube and on time only", status="imposed",
             check="One cell across the tube, with empty sides. A real tube has a thin layer of slowed gas at its wall; "
                   "the problem excludes it by making the wall frictionless."),
        dict(name="No viscosity, no heat conduction (the Euler equations)", status="imposed",
             check="Given in the problem, and what the exact solution assumes. The solver's viscosity is set to zero, so it "
                   "skips its viscous steps."),
        dict(name="Ideal gas with a constant ratio of specific heats", status="standard",
             check="Given in the problem. The temperatures in the exact solution lie between "
                   + m(rf"{tf(min(T_all), 0)}") + " and " + m(rf"{tf(max(T_all), 0)}\,\mathrm{{K}}")
                   + r" (from \(T = p/(\rho R)\) in the four uniform states), a range in which the specific heats of air "
                   "change little; that last statement is textbook knowledge and was not checked here."),
        dict(name="Compressibility matters", status="ok" if max(mach) > 0.3 else "violated",
             check="The gas between the waves moves at " + m(rf"{tf(mach[0], 2)}") + " and " + m(rf"{tf(mach[1], 2)}")
                   + " times the local speed of sound (left and right of the border of the two gases), and the shock at "
                   + m(rf"M_s = {tf(ex.S / ex.cR, 3)}") + ". Far above 0.3: a constant-density solver cannot be used."),
        dict(name="No wave reaches an end of the tube before the time of interest", status="ok" if worst["ends"] <= acc["fail_ends"] else "violated",
             check="Exact solution: the front of the fan is at " + m(rf"x = {tf(w['head'], 3)}\,\mathrm{{m}}") + " and the shock at "
                   + m(rf"x = {tf(w['shock'], 3)}\,\mathrm{{m}}") + " in a tube of " + m(rf"{tn(L)}\,\mathrm{{m}}")
                   + ". Simulation: in the two end cells of every run, pressure and velocity differ from the start state by "
                   "at most " + sc(worst["ends"]) + " (relative)."),
        dict(name="The wall is removed instantly", status="imposed",
             check="The start state has the jump on one cell face. A real diaphragm takes time to burst and leaves a "
                   "curved, three-dimensional front at first."),
        dict(name="Closed ends, frictionless and without heat flow", status="imposed",
             check="No flow through the ends, no temperature gradient at them. They only close the tube here: nothing "
                   "reaches them, and the conservation checks confirm that nothing leaks."),
    ]

    # ---- answers -------------------------------------------------------------------------------------
    wanted = [dict(letter=LETTERS[i], key=q["key"], text=q["text"]) for i, q in enumerate(problem["quantities_of_interest"])]
    sub = lambda s: f"<div class='sub'>{s}</div>"
    st_tex = dict(p=("p^*", r"\mathrm{Pa}", 1), u=("u^*", r"\mathrm{m/s}", 3), rho_L=(r"\rho^*_L", r"\mathrm{kg/m^3}", 5),
                  rho_R=(r"\rho^*_R", r"\mathrm{kg/m^3}", 5))
    st_exact = dict(p=ex.p_star, u=ex.u_star, rho_L=ex.rho_starL, rho_R=ex.rho_starR)
    mm = lambda v: m(rf"{1e3 * v:+.2f}\,\mathrm{{mm}}")
    answers = []
    for q in wanted:
        k = q["key"]
        if k == "profiles":
            answers.append(dict(letter=q["letter"], label=q["text"],
                                theory="the exact solution, averaged over each cell" + sub("Figure {fig:profiles}"),
                                sim=f"mean difference along the tube on {fine['N']} cells" + sub(
                                    "as a fraction of " + m(r"\rho_L") + ", " + m("u^*") + " and " + m("p_L")),
                                diff="<br>".join(m(sym) + ": " + pct(fine["e"][kk]) for kk, sym, _ in QTY)))
        elif k == "states":
            answers.append(dict(letter=q["letter"], label=q["text"],
                                theory="<br>".join(m(rf"{st_tex[s][0]} = {tf(st_exact[s], st_tex[s][2])}\ {st_tex[s][1]}") for s in st_tex),
                                sim="<br>".join(m(rf"{tf(fine['state'][s], st_tex[s][2])}\ {st_tex[s][1]}") for s in st_tex),
                                diff="<br>".join(pct(fine["state_err"][s]) for s in st_tex)))
        elif k == "positions":
            short = dict(head="fan front", tail="fan back", contact="border", shock="shock")
            line = lambda s: f"<span style='white-space:nowrap'>{s}</span>"
            answers.append(dict(letter=q["letter"], label=q["text"],
                                theory="<br>".join(line(f"{short[z]}: " + m(rf"{tf(w[z], 5)}\,\mathrm{{m}},\ {tf(ex.speeds[z], 2)}\,\mathrm{{m/s}}"))
                                                   for z in WAVES)
                                       + sub("position at " + m(r"t_{\mathrm{end}}") + " and speed; border: of the two gases"),
                                sim="<br>".join(line(m(rf"{tf(fine['pos'][z], 5)}\,\mathrm{{m}},\ {tf(fine['speed'][z], 2)}\,\mathrm{{m/s}}"))
                                                for z in WAVES) + sub("Figure {fig:paths}"),
                                diff="<br>".join(m(rf"{1e3 * fine['pos_err'][z]:+.2f}\,\mathrm{{mm}},\ {fine['speed_err'][z]:+.2f}\,\mathrm{{m/s}}")
                                                 for z in WAVES)))
        elif k == "conserved":
            answers.append(dict(letter=q["letter"], label=q["text"],
                                theory="constant: " + m(rf"M = {dmap['mass']['value_tex']}\ \mathrm{{kg/m^2}}") + ", "
                                       + m(rf"E_{{\mathrm{{tot}}}} = {tn(dmap['energy']['value'], 6)}\ \mathrm{{J/m^2}}")
                                       + sub("per unit of cross-section"),
                                sim="constant. Largest change during any of the " + str(len(allr)) + " runs: mass "
                                    + sc(worst["mass_drift"]) + ", energy " + sc(worst["energy_drift"]) + " (relative)"
                                    + sub("Figure {fig:conservation}"),
                                diff="round-off"))
        elif k == "sharpness":
            answers.append(dict(letter=q["letter"], label=q["text"],
                                theory="An exact jump has no width. A jump spread over a few cells leaves an error in "
                                       "proportion to the cell size at best (order 1).",
                                sim="shock: " + m_fix(min(shock_w), 1) + " to " + m_fix(max(shock_w), 1) + " cells on every grid<br>"
                                    "border of the two gases: " + m_fix(coarse["width"]["contact"], 1) + " cells on "
                                    + f"{coarse['N']} cells, " + m_fix(fine["width"]["contact"], 1) + f" on {fine['N']}<br>"
                                    "error against cell size, all eight grids: order "
                                    + ", ".join(m_fix(overall[kk], 2) + " (" + m(sym) + ")" for kk, sym, _ in QTY)
                                    + sub("Figures {fig:sharpness} and {fig:convergence}"),
                                diff="–"))

    # ---- findings ------------------------------------------------------------------------------------
    best_sch = min(sch, key=lambda r: r["e"]["rho"])
    upw = next(r for r in sch if r["limiter"] == "upwind")
    sbee = next(r for r in sch if r["limiter"] == "SuperBee")
    ok_rules = [x for x in rules if x["ok"]]
    findings = [
        f"<b>Verdict by the rules written before the runs: {verdict}.</b> {len(ok_rules)} of {len(rules)} rules hold. "
        + ("Broken, FAIL: " + "; ".join(fails) + ". " if fails else "")
        + ("Broken, WARN: " + "; ".join(warns) + "." if warns else ""),
        "<b>The simulation converges to the exact solution.</b> From " + f"{coarse['N']} to {fine['N']} cells the mean error falls by "
        + ", ".join(report.m_times(coarse["e"][k] / fine["e"][k], 0) + " (" + m(sym) + ")" for k, sym, _ in QTY)
        + ". On the finest grid the uniform states are within " + pct(worst_state) + ", the shock within "
        + m_fix(abs(fine["pos_err"]["shock"]) / fine["h"], 2) + " cells and the border of the two gases within "
        + m_fix(abs(fine["pos_err"]["contact"]) / fine["h"], 2) + " cells of their exact places.",
        "<b>It converges more slowly than the pressure limit assumed.</b> The limit was sized for an order of "
        + m_fix(acc["fail_l1_order_floor"]["p"], 2) + "; the measured order over the eight grids is " + m_fix(overall["p"], 2)
        + ". The pressure error on the finest grid is " + pct(fine["e"]["p"]) + " of " + m("p_L") + ", the limit "
        + pct(l1_limit["p"]) + ". The error sits in the expansion fan: " + m_pct(share["p"]["fan"], 0)
        + " of the pressure error and " + m_pct(share["rho"]["fan"], 0) + " of the density error on the finest grid, falling with order "
        + m_fix(wave_overall["p"]["fan"], 2) + " there. A smaller time step does not reduce it (see the time-step study).",
        "<b>Mass, momentum and energy are kept to round-off</b> on every step of every run (largest relative change "
        + sc(max(worst["mass_drift"], worst["energy_drift"])) + "). This is why the shock sits in the right place on every grid.",
        "<b>Sharpness.</b> The shock is " + m_fix(min(shock_w), 1) + " to " + m_fix(max(shock_w), 1)
        + " cells wide on every grid, so its width in metres halves with the cell size. The border of the two gases is wider, "
        + m_fix(coarse["width"]["contact"], 1) + " to " + m_fix(fine["width"]["contact"], 1)
        + " cells: nothing in the flow pushes it back together once it is smeared.",
        "<b>The time step matters more than the tutorial suggests.</b> At the tutorial's limit (twice the step used here) "
        "pressure and velocity between the fan and the shock ripple by " + m_pct(ts[0]["ripple"]["p"], 1) + " on "
        + f"{ts[0]['N']} cells; at the step used here by " + m_pct(main_ts["ripple"]["p"], 1)
        + ". The ripple does not go away with a finer grid: it is " + m_pct(fine["ripple"]["p"], 1) + f" on {fine['N']} cells.",
        "<b>Choice of limiter</b> (" + f"{sch[0]['N']} cells): the tutorial's van Leer limiter has the lowest density error of the six "
        "combinations tried" if best_sch["limiter"] == mainc["limiter"] and best_sch["flux"] == mainc["flux"] else
        "<b>Choice of limiter</b> (" + f"{sch[0]['N']} cells): the lowest density error of the six combinations is "
        + f"{best_sch['flux']} with {best_sch['limiter']}",
    ]
    findings[-1] += (". The first-order scheme smears the border of the two gases over " + m_fix(upw["width"]["contact"], 0)
                     + " cells; SuperBee keeps it to " + m_fix(sbee["width"]["contact"], 1) + " cells but doubles the error in the fan.")

    limitations = [
        "One of the limits set before the runs is not met (pressure on the finest grid), so the verdict is FAIL. "
        "The limit rested on my expectation that pressure would converge with an order of at least "
        + m_fix(acc["fail_l1_order_floor"]["p"], 2) + ". Whether the measured " + m_fix(overall["p"], 2)
        + " is a property of this solver's scheme or of any method of this kind was not settled: only one solver was run. "
        "A second, independent solver on the same grids would tell.",
        "The time step is not converged. On " + f"{main_ts['N']} cells the density profile changes by " + pct(ts_main_vs_small)
        + " of " + m(r"\rho_L") + " when the step is cut to a quarter, against a total error of " + pct(main_ts["e"]["rho"])
        + ". The error against the exact solution is lowest at the step used here because the time error and the space "
        "error partly cancel, not because the time error is small.",
        "The solver's only time scheme (explicit Euler) leaves a ripple of about " + m_pct(fine["ripple"]["p"], 1)
        + " behind the shock at this time step, and the extra up-and-down of the profiles grows slowly with the number of cells ("
        + m_pct(coarse["wiggle"]["p"], 1) + " to " + m_pct(fine["wiggle"]["p"], 1) + ").",
        "The grid study was made at one Courant number. The order of convergence at a smaller time step was not measured.",
        "The problem is one-dimensional, without viscosity, and ends before any wave reaches a wall. Reflections, "
        "boundary layers and the bursting of a real diaphragm are outside it.",
        "With an error that falls like the cell size to the power " + m_fix(overall["rho"], 1)
        + ", accuracy is expensive: ten times less error needs about " + m_fix(10 ** (1 / overall["rho"]), 0)
        + " times more cells, and in one dimension the time steps grow with the cells, so about "
        + report.m_sig(10 ** (2 / overall["rho"]), 1) + " times more work.",
        "Statements about published methods (the flux scheme, the expected order at a jump) are cited from memory, not "
        "from references supplied with the problem: " + "; ".join(REFS.values()) + ".",
    ]

    # ---- figures -------------------------------------------------------------------------------------
    svgs = {"sketch": sketch(w, L)}
    col = plots.SERIES
    xx = np.linspace(0, L, 3001)
    exact_pt = ex.primitive(xx, tend)
    mid = fam[3]                                              # 800 cells
    fig, axs = plots.wide(3, 3.5)
    ylab = (r"Density $\rho$ (kg/m$^3$)", r"Velocity $u$ (m/s)", r"Pressure $p$ (kPa)")
    fac = (1.0, 1.0, 1e-3)
    for i, ax in enumerate(axs):
        ax.plot(xx, fac[i] * exact_pt[i], color=plots.INK, lw=1.3, label="exact")
        ax.plot(coarse["x"], fac[i] * coarse["sim"][i], "o", ms=3.2, color=col[1], mec=plots.SURFACE, mew=0.4,
                label=f"{coarse['N']} cells")
        ax.plot(mid["x"], fac[i] * mid["sim"][i], "-", lw=1.0, color=col[0], label=f"{mid['N']} cells")
        ax.set_xlabel(r"Position $x$ (m)")
        ax.set_ylabel(ylab[i])
    axs[0].legend(loc="upper right", fontsize=8.5)
    svgs["profiles"] = plots.to_svg(fig, name="profiles")

    wave_col = dict(head=col[0], tail=col[2], contact=col[3], shock=col[1])
    xt = R[f"N{numerics['runs']['timestep']['cells']}"]
    fig, ax = plots.figure(6.8, 3.8)
    for z in WAVES:
        ax.plot([ex.x0, w[z]], [0, 1e3 * tend], color=wave_col[z], lw=1.5, label=f"{WAVE[z]}, exact")
        ax.plot([pp[z] for pp in xt["pos_t"]], 1e3 * xt["times"], "o", ms=4, color=wave_col[z], mec=plots.INK, mew=0.5,
                label="_nolegend_")
    ax.plot([], [], "o", ms=4, color=plots.MUTED, mec=plots.INK, mew=0.5, label=f"simulation, {xt['N']} cells")
    ax.set_xlabel(r"Position $x$ (m)")
    ax.set_ylabel(r"Time $t$ (ms)")
    ax.set_xlim(0, L)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    svgs["paths"] = plots.to_svg(fig, name="paths")

    show = [fam[0], fam[3], fam[-1]]
    fig, (a1, a2) = plots.wide(2, 3.7)
    for k, r in enumerate(show):
        xi = (r["x"] - w["shock"]) / r["h"]
        s = np.abs(xi) <= 7
        a1.plot(xi[s], 1e-3 * r["sim"][2][s], "o-", ms=3.5, lw=1.0, color=col[k], mec=plots.SURFACE, mew=0.4, label=f"{r['N']} cells")
        xi = (r["x"] - w["contact"]) / r["h"]
        s = np.abs(xi) <= 11                       # on the coarsest grid the shock is 13 cells from the border
        a2.plot(xi[s], r["sim"][0][s], "o-", ms=3.0, lw=1.0, color=col[k], mec=plots.SURFACE, mew=0.4, label=f"{r['N']} cells")
    a1.set_xlabel("Cells from the exact shock position")
    a1.set_ylabel(r"Pressure $p$ (kPa)")
    a1.set_title("Shock", loc="left")
    a1.legend(loc="upper right", fontsize=8.5)
    a2.set_xlabel("Cells from the exact position of the border")
    a2.set_ylabel(r"Density $\rho$ (kg/m$^3$)")
    a2.set_title("Border of the two gases", loc="left")
    a2.legend(loc="upper right", fontsize=8.5)
    svgs["sharpness"] = plots.to_svg(fig, name="sharpness")

    fig, ax = plots.figure(6.8, 3.4)
    for k, r in enumerate((coarse, fine)):
        st = max(1, len(r["c_t"]) // 400)
        ax.plot(1e3 * r["c_t"][::st], 1e14 * (r["c_mass"][::st] - r["c_mass"][0]) / dmap["mass"]["value"], color=col[k], lw=1.3,
                label=f"mass, {r['N']} cells")
        ax.plot(1e3 * r["c_t"][::st], 1e14 * (r["c_energy"][::st] - r["c_energy"][0]) / dmap["energy"]["value"], color=col[k],
                lw=1.3, ls="--", label=f"energy, {r['N']} cells")
    ax.set_xlabel(r"Time $t$ (ms)")
    ax.set_ylabel("Change since the first step\n" + r"(parts in $10^{14}$)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    svgs["conservation"] = plots.to_svg(fig, name="conservation")

    fig, (c1, c2) = plots.wide(2, 3.8)
    Na = np.array(Ns, float)
    for k, (kk, sym, word) in enumerate(QTY):
        c1.loglog(Na, [100 * r["e"][kk] for r in fam], "o-", ms=4.5, lw=1.4, color=col[k], mec=plots.SURFACE, mew=0.5, label=word)
    ref = 100 * fam[0]["e"]["p"] * 0.45
    c1.loglog(Na, ref * (Na / Na[0]) ** -1.0, "--", color=plots.MUTED, lw=1, label="order 1")
    c1.loglog(Na, ref * (Na / Na[0]) ** (-2 / 3), ":", color=plots.MUTED, lw=1.2, label="order 2/3")
    c1.plot([fine["N"]], [100 * l1_limit["p"]], "_", ms=14, mew=2, color=col[2], label="limit for pressure")
    c1.set_xlabel("Cells along the tube")
    c1.set_ylabel("Mean error (%)")
    c1.set_title("Whole tube", loc="left")
    c1.legend(loc="lower left", fontsize=8)
    for k, z in enumerate(("fan", "contact", "shock")):
        c2.loglog(Na, [100 * r["by_wave"]["rho"][z] for r in fam], "o-", ms=4.5, lw=1.4, color=col[k], mec=plots.SURFACE, mew=0.5,
                  label={"fan": "fan", "contact": "border of the two gases", "shock": "shock"}[z])
    c2.loglog(Na, 100 * fam[0]["by_wave"]["rho"]["shock"] * 0.5 * (Na / Na[0]) ** -1.0, "--", color=plots.MUTED, lw=1, label="order 1")
    c2.set_xlabel("Cells along the tube")
    c2.set_ylabel(r"Density error, part of each wave (%)")
    c2.set_title("Density error by wave", loc="left")
    c2.legend(loc="lower left", fontsize=8)
    svgs["convergence"] = plots.to_svg(fig, name="convergence")

    fig, (t1, t2) = plots.wide(2, 3.7)
    s = (xx > w["contact"] - 0.05) & (xx < w["shock"] + 0.012)
    t1.plot(xx[s], 1e-3 * exact_pt[2][s], color=plots.INK, lw=1.2, label="exact")
    for k, r in enumerate(ts):
        s = (r["x"] > w["contact"] - 0.05) & (r["x"] < w["shock"] + 0.012)
        t1.plot(r["x"][s], 1e-3 * r["sim"][2][s], "-", lw=1.1, color=col[k], label=f"Courant {r['courant']:.2f}")
    t1.set_ylim(1e-3 * ex.p_star * 0.95, 1e-3 * ex.p_star * 1.035)
    t1.set_xlabel(r"Position $x$ (m)")
    t1.set_ylabel(r"Pressure $p$ (kPa)")
    t1.set_title(f"Pressure behind the shock, {ts[0]['N']} cells", loc="left")
    t1.legend(loc="lower left", fontsize=8.5)
    co = [r["courant"] for r in ts]
    for k, (kk, sym, word) in enumerate(QTY):
        t2.semilogx(co, [100 * r["e"][kk] for r in ts], "o-", ms=4.5, lw=1.4, color=col[k], mec=plots.SURFACE, mew=0.5, label=word)
    t2.set_xlabel("Courant number (smaller: shorter time step)")
    t2.set_ylabel("Mean error (%)")
    t2.set_ylim(0, None)
    t2.set_title("Error against the exact solution", loc="left")
    t2.legend(loc="upper left", fontsize=8.5)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    t2.xaxis.set_major_locator(FixedLocator(co))
    t2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    t2.xaxis.set_minor_formatter(NullFormatter())
    svgs["timestep"] = plots.to_svg(fig, name="timestep")

    sch_cols = [col[0], col[1], col[2], col[3], plots.INK, plots.MUTED]
    fig, (s1, s2) = plots.wide(2, 3.9)
    handles = []
    for k, r in enumerate(sch):
        xi = (r["x"] - w["contact"]) / r["h"]
        sel = np.abs(xi) <= 18
        hnd, = s1.plot(xi[sel], r["sim"][0][sel], "-", lw=1.3, color=sch_cols[k], label=f"{r['flux']}, {r['limiter']}")
        handles.append(hnd)
        xi = (r["x"] - w["shock"]) / r["h"]
        sel = (xi >= -40) & (xi <= 6)
        s2.plot(xi[sel], 1e-3 * r["sim"][2][sel], "-", lw=1.3, color=sch_cols[k])
    s1.set_xlabel("Cells from the exact position of the border")
    s1.set_ylabel(r"Density $\rho$ (kg/m$^3$)")
    s1.set_title("Border of the two gases", loc="left")
    s2.set_xlabel("Cells from the exact shock position")
    s2.set_ylabel(r"Pressure $p$ (kPa)")
    s2.set_ylim(1e-3 * ex.p_star * 0.9, 1e-3 * ex.p_star * 1.04)
    s2.set_title("Pressure behind the shock", loc="left")
    fig.legend(handles, [h_.get_label() for h_ in handles], loc="outside lower center", ncol=3, fontsize=8.5)
    svgs["schemes"] = plots.to_svg(fig, name="schemes")

    n_show = ", ".join(str(r["N"]) for r in show[:-1]) + f" and {show[-1]['N']}"
    near = (fine["x"] > w["contact"]) & (fine["x"] < w["contact"] + 11 * fine["h"])
    dip = float(1 - fine["sim"][0][near].min() / ex.rho_starR)
    dip_t = []                                    # the same dip at every written time: does it travel with the border?
    for k, t in enumerate(fine["times"]):
        xc = ex.positions(float(t))["contact"]
        sel = (fine["x"] > xc) & (fine["x"] < xc + 11 * fine["h"])
        dip_t.append(float(1 - fine["rho_t"][k][sel].min() / ex.rho_starR))
    figures = [
        dict(key="sketch", section="problem",
             caption="Top: the tube before the wall is removed. Bottom: the tube at the time of interest, with the three "
                     "waves at their exact positions (to scale along the tube). The arrows show which way each wave moves; "
                     "the two dotted lines are the front and the back of the expansion fan."),
        dict(key="profiles", section="simulation",
             caption="Density, velocity and pressure along the tube at the time of interest. Black: exact solution. Dots: "
                     f"{coarse['N']} cells (mean density error " + pct(coarse["e"]["rho"]) + "). Blue line: "
                     f"{mid['N']} cells (" + pct(mid["e"]["rho"]) + "). On the finest grid (" + pct(fine["e"]["rho"])
                     + ") the simulation cannot be told from the exact solution at this scale."),
        dict(key="paths", section="simulation",
             caption="Where each wave is at the ten written times. Lines: exact (each wave leaves the wall position at a "
                     f"constant speed). Dots: read from the simulation on {xt['N']} cells with the rules of this report. "
                     "The slope of the dots over the second half of the time gives the wave speeds; on this grid they differ "
                     "from the exact speeds by at most " + m_pct(max(abs(xt["speed_err"][z]) for z in WAVES) / ex.S, 2)
                     + " of the shock speed."),
        dict(key="sharpness", section="simulation",
             caption=f"The two jumps on {n_show} cells, plotted against the distance from the exact position counted in cells. "
                     "Left: the shock keeps the same shape in cells on every grid (" + m_fix(min(shock_w), 1) + " to "
                     + m_fix(max(shock_w), 1) + " cells between 10 and 90 % of its height). Right: the border of the two gases "
                     "is wider and widens a little with refinement (" + m_fix(coarse["width"]["contact"], 1) + " to "
                     + m_fix(fine["width"]["contact"], 1) + " cells). Just right of it the density on the finest grid dips "
                     + m_pct(dip, 1) + " below the exact value. The dip is there at every written time, next to the border ("
                     + m_pct(min(dip_t), 1) + " to " + m_pct(max(dip_t), 1) + "), so it travels with the gas; its origin was not "
                     "traced further."),
        dict(key="conservation", section="convergence",
             caption="Change of the total mass (solid) and total energy (dashed) since the first step, in parts in "
                     + m("10^{14}") + ", on the coarsest and the finest grid. The largest change in any run is "
                     + sc(max(worst["mass_drift"], worst["energy_drift"])) + ": the last digits a computer carries."),
        dict(key="convergence", section="comparison",
             caption="Left: mean error against the number of cells, with lines of order 1 and 2/3 for comparison. "
                     "Fitted over the eight grids the orders are "
                     + ", ".join(m_fix(overall[kk], 2) + f" ({word})" for kk, _, word in QTY)
                     + ". The green bar is the limit set for the pressure on the finest grid; the pressure error ends "
                     + ("above" if fine["e"]["p"] > l1_limit["p"] else "below") + " it. Right: the density error split by wave. "
                     "On the finest grid the fan holds " + m_pct(share["rho"]["fan"], 0) + " of it, the border of the two gases "
                     + m_pct(share["rho"]["contact"], 0) + " and the shock " + m_pct(share["rho"]["shock"], 0)
                     + "; the shock's part jumps up and down because it depends on where the shock sits inside its cell."),
        dict(key="timestep", section="comparison",
             caption=f"Time-step study on {ts[0]['N']} cells. Left: pressure between the border of the two gases and the shock "
                     "for four time steps; the exact value is the black line. The ripple is "
                     + ", ".join(m_pct(r["ripple"]["p"], 1) for r in ts) + " of the pressure at Courant numbers "
                     + ", ".join(f"{r['courant']:.2f}" for r in ts) + ". Right: mean error against the exact solution. It is "
                     + ("lowest at the step used for the grid study (Courant " + f"{main_ts['courant']:.2f})"
                        if min(ts, key=lambda r: r["e"]["rho"]) is main_ts else "not lowest at the step used for the grid study")
                     + " and rises again for smaller steps."),
        dict(key="schemes", section="comparison",
             caption=f"Six ways of building the face values, same grid ({sch[0]['N']} cells) and time step. Left: density across "
                     "the border of the two gases. Right: pressure behind the shock (the shock is at zero). Widths and errors "
                     "are in the table below."),
    ]

    # ---- pre-flight ----------------------------------------------------------------------------------
    pf_runs, pf_checks = [], []
    for p in pf["short"]:
        sec = lambda v: m((f"{v:.2f}" if v < 10 else f"{v:.0f}") + r"\,\mathrm{s}")
        pf_runs.append(dict(name=f"{p['N']} cells, short", setup=f"first {p['n_steps']} of {round(p['n_steps'] / numerics['preflight']['fraction'])} steps",
                            outcome=f"{CHECK[p['status']]}; " + sec(p["wall_time"]) + " of solver time; projected full run "
                                    + sec(p["est_full_runtime"])))
        for c in p["checks"]:
            pf_checks.append(dict(c, name=f"{c['name']} ({p['N']} cells)"))
    for r_ in pf["timestep"]:
        e = r_["errors"]
        pf_runs.append(dict(name=f"{numerics['runs']['timestep']['cells']} cells, Courant {r_['courant']:.2f}, full",
                            setup="time-step study: the whole run, time step halved from one run to the next",
                            outcome="mean errors " + ", ".join(m(sym) + " " + pct(e[kk]) for kk, sym, _ in QTY)
                                    + "; extra up-and-down of the pressure " + m_pct(e["wiggle_p"], 1)))
    cz = pf["sizing"][f"N{coarse['N']}"]
    pf_runs.append(dict(name=f"{coarse['N']} cells, full", setup="the coarsest grid in full, to size the limits",
                        outcome="mean errors " + ", ".join(m(sym) + " " + pct(cz["errors"][kk]) for kk, sym, _ in QTY)))
    pf_fine = pf["short"][-1]
    pf_checks.append(dict(name="Every setting belongs to a decision", status="ok" if not pf["decisions"] else "fail",
                          value="all covered" if not pf["decisions"] else f"{len(pf['decisions'])} missing",
                          meaning="The checker compares the finest case's files with decisions.yaml."))
    preflight = dict(
        purpose="Before the production runs: measure OpenFOAM's gas constant; run the first 10 % of the coarsest and the "
                "finest grid to see that the case runs, keeps mass and energy, records its outputs and to project the run "
                "time; run the time-step study on one grid to choose the time step; run the coarsest grid in full to size "
                "the pass and fail limits.",
        runs=pf_runs, checks=pf_checks,
        conclusion="All pre-flight checks passed. What the pre-flight decided: (1) the molar mass, from the measured gas "
                   "constant " + m(rf"{tn(gc['RR'], 9)}\ \mathrm{{J/(kmol\,K)}}") + "; (2) the time step: half the tutorial's, because at the "
                   "tutorial's step the pressure behind the shock rippled by " + m_pct(ts[0]["ripple"]["p"], 1)
                   + " (the solver prints a Courant number that is half the usual one, so the tutorial's limit of 0.2 is a usual "
                   + m_fix(ts[0]["courant"], 2) + "); (3) the limits in numerics.yaml, written before any finer grid was run. "
                   "Projected time of the finest run: " + m(rf"{pf_fine['est_full_runtime']:.0f}\,\mathrm{{s}}") + "; measured: "
                   + m(rf"{fine['wall_time']:.0f}\,\mathrm{{s}}") + ".")

    # ---- simulation tables ---------------------------------------------------------------------------
    order_of = lambda r: (0 if "grid" in r["studies"] else 1 if "time step" in r["studies"] else 2, r["N"], -r["spc"])
    rows_sorted = sorted(allr, key=order_of)
    model = [
        dict(aspect="Tube", real="A straight tube with closed ends, flow along it only",
             model=f"{coarse['N']} to {fine['N']} equal cells along " + m("x") + ", one cell across with empty sides; "
                   "cross-section " + m(r"1\,\mathrm{m^2}") + ", so that totals are per unit of cross-section."),
        dict(aspect="Start", real="Two uniform states at rest, separated by a thin wall that is removed at once",
             model="The jump lies on the cell face at " + m("x = L/2") + ". Pressure and temperature are set cell by cell; "
                   "the solver's start densities equal the given ones to 12 digits (pre-flight)."),
        dict(aspect="Gas", real="Air as an ideal gas with constant " + m(r"\gamma") + ", no viscosity, no heat conduction",
             model=m(rf"c_p = \gamma R/(\gamma-1) = {tn(g['gamma'] * g['R'] / (g['gamma'] - 1), 5)}\ \mathrm{{J/(kg\,K)}}")
                   + ", molar mass " + m(tn(gc["mol_weight"], 8)) + " (from OpenFOAM's measured gas constant), viscosity zero."),
        dict(aspect="Ends", real="Closed, frictionless, no heat flow",
             model="No flow through them; zero gradient of pressure and temperature. Nothing reaches them before the time of interest."),
        dict(aspect="Time", real="From the removal of the wall to " + m(rf"t_{{\mathrm{{end}}}} = {tn(tend)}\,\mathrm{{s}}"),
             model="Explicit steps of fixed size, " + m_fix(mainc["steps_per_cell"], 1) + " steps per cell: Courant number "
                   + m_fix(fine["courant"], 2) + " (the solver prints " + m_fix(fine["co_solver"], 3) + ", it counts half)."),
        dict(aspect="Outputs", real="Profiles, states, wave positions and speeds, totals",
             model="Fields at ten times; total mass, momentum and energy after every step. States, positions and widths "
                   "are read from the profiles by fixed rules (numerics.yaml, extraction)."),
    ]
    meshes = dict(
        intro=f"{len(allr)} runs in three studies. <b>Grid</b>: cells doubled seven times, the time step halved with the cell "
              "size. <b>Time step</b>: one grid, the step halved three times. <b>Scheme</b>: one grid and step, six ways of "
              "building the face values. Every run covers the whole time of interest.",
        head=["Run", "Cells", "Cell size (mm)", "Time steps", "Courant number", "Flux, limiter", "Studies"],
        rows=[[r["name"], r["N"], m(tn(1e3 * r["h"], 3)), r["steps"], m_fix(r["courant"], 3), f"{r['flux']}, {r['limiter']}",
               ", ".join(r["studies"])] for r in rows_sorted])
    results_tab = dict(
        head=["Cells", m("p^*") + " (Pa)", m("u^*") + " (m/s)", m(r"\rho^*_L") + " (kg/m³)", m(r"\rho^*_R") + " (kg/m³)",
              "Shock width (cells)", "Border width (cells)"],
        rows=[["exact", m(tf(ex.p_star, 1)), m(tf(ex.u_star, 3)), m(tf(ex.rho_starL, 5)), m(tf(ex.rho_starR, 5)), m("0"), m("0")]]
             + [[f"{r['N']}", m(tf(r["state"]["p"], 1)), m(tf(r["state"]["u"], 3)), m(tf(r["state"]["rho_L"], 5)),
                 m(tf(r["state"]["rho_R"], 5)), m_fix(r["width"]["shock"], 2), m_fix(r["width"]["contact"], 2)] for r in fam])
    pos_tab = dict(
        title="Wave positions at the time of interest and wave speeds, grid study",
        head=["Cells"] + [f"{WAVE[z]}: " + m("x") + " (m)" for z in WAVES] + [f"{WAVE[z]}: speed (m/s)" for z in ("contact", "shock")],
        rows=[["exact"] + [m(tf(w[z], 5)) for z in WAVES] + [m(tf(ex.speeds[z], 2)) for z in ("contact", "shock")]]
             + [[f"{r['N']}"] + [m(tf(r["pos"][z], 5)) for z in WAVES] + [m(tf(r["speed"][z], 2)) for z in ("contact", "shock")]
                for r in fam],
        note="Positions: shock and border where the profile is half-way up the jump; the two ends of the fan from a straight "
             "line fitted to the velocity inside it. Speeds: slope of position against time over the written times from "
             + m(r"t_{\mathrm{end}}/2") + " on. Speeds of the two ends of the fan on the finest grid: "
             + m(rf"{tf(fine['speed']['head'], 2)}") + " and " + m(rf"{tf(fine['speed']['tail'], 2)}\,\mathrm{{m/s}}") + " (exact "
             + m(rf"{tf(ex.speeds['head'], 2)}") + " and " + m(rf"{tf(ex.speeds['tail'], 2)}") + ").")
    health = dict(
        head=["Run", "Log checks", "Mass change", "Energy change", "Momentum", "Ends", "Wiggles " + m(r"\rho") + ", " + m("p"),
              "Solver time (s)"],
        rows=[[r["name"], CHECK[r["health"]], sc(r["mass_drift"], 1), sc(r["energy_drift"], 1), sc(r["momentum_err"], 1),
               sc(r["ends"], 1), m_pct(r["wiggle"]["rho"], 1) + ", " + m_pct(r["wiggle"]["p"], 1), m_fix(r["wall_time"], 1)]
              for r in rows_sorted],
        notes=[
            "An explicit solver has no residuals to watch. The evidence that each step is right is physical: the total mass "
            "and the total energy of the air must not change (closed ends), and the total momentum must equal the push of "
            "the two end walls, " + m(r"(p_L - p_R)\,t") + ". All three are recorded after every step. Columns: the largest "
            "change during the run, relative (Figure {fig:conservation}). Limits: " + sc(acc["fail_mass_drift"], 1) + ".",
            "Ends: the largest relative difference from the start state in the two end cells at the time of interest "
            "(limit " + sc(acc["fail_ends"], 1) + "). It confirms that no wave has reached an end.",
            "Wiggles: the exact density and pressure only fall from left to right. The column is the simulation's total "
            "up-and-down above the exact one, so zero means no wiggle at all (limit for the grid study: "
            + m_pct(acc["warn_wiggle"], 0) + "). It grows slowly with the number of cells, about like the number of cells to the power "
            + m_fix(wiggle_exp, 2) + ": the ripple behind the shock keeps its size and covers more cells.",
            "Log checks (cfdlib.monitor): no crash, no NaN, Courant number and its trend. Density and pressure stayed positive in every run.",
            "The grid of every run passed checkMesh." if all(r["mesh_ok"] for r in allr) else "A grid did not pass checkMesh.",
        ])

    # ---- comparison ----------------------------------------------------------------------------------
    defs = [
        dict(name="Mean error (the L1 norm)", latex=r"e_q = \frac{1}{N}\sum_{i=1}^{N}\frac{|q_i - \bar q_i^{\,\mathrm{exact}}|}{q_{\mathrm{ref}}},\qquad "
                                                    r"q_{\mathrm{ref}} = \rho_L,\ u^*,\ p_L",
             meaning="The mean absolute difference over the cells between the simulation and the exact solution averaged "
                     "over the same cell. This is the measure to use with jumps: the largest difference never falls, because "
                     "a cell inside a smeared jump is always wrong by a part of the jump."),
        dict(name="Observed order", latex=r"p = \frac{\ln(e_{\mathrm{coarse}}/e_{\mathrm{fine}})}{\ln 2}",
             meaning=m("p = 1") + ": halving the cells halves the error. " + m("p = 2") + " would cut it by four."),
        dict(name="Width of a jump", latex=r"\frac{|x_{90\%} - x_{10\%}|}{\Delta x}",
             meaning="Distance between the points where the profile has made 10 % and 90 % of the jump, in cells. The same "
                     "rule applied to the exact cell averages gives " + m_fix(min(r["width_exact"]["shock"] for r in fam), 1) + " to "
                     + m_fix(max(max(r["width_exact"].values()) for r in fam), 1) + " cells: that is the rule's own floor."),
        dict(name="Wiggles", latex=r"\frac{\sum_i |q_{i+1} - q_i|}{q_L - q_R} - 1,\qquad q = \rho,\ p",
             meaning="Zero for a profile that only falls from left to right, as the exact density and pressure do."),
        dict(name="Ripple between the fan and the shock", latex=r"\frac{\max q - \min q}{\mathrm{mean}\ q},\qquad q = p,\ u",
             meaning="Over the middle half of the stretch between the back of the fan and the shock, where the exact "
                     "pressure and velocity are constant."),
    ]
    comp_rows = []
    for i, r in enumerate(fam):
        row = [f"{r['N']}"]
        for k, _, _ in QTY:
            row += [pct(r["e"][k]), m_fix(order[k][i - 1], 2) if i else "–"]
        comp_rows.append(row)
    wave_tab = dict(
        title="Where the error is: each wave's part of the mean error on the finest grid, and its order over the eight grids",
        head=["Quantity", "Fan: part", "Fan: order", "Border: part", "Border: order", "Shock: part", "Shock: order"],
        rows=[[word] + [c for z in ("fan", "contact", "shock") for c in (m_pct(share[k][z], 0), m_fix(wave_overall[k][z], 2))]
              for k, _, word in QTY],
        note="The tube is split half-way between neighbouring waves. Order: slope of a straight-line fit of the error against "
             "the number of cells on logarithmic axes.")
    ts_tab = dict(
        title=f"Time-step study, {ts[0]['N']} cells",
        head=["Courant number", "The solver prints", "Steps", m("e_\\rho"), m("e_u"), m("e_p"), "Error in the fan (" + m("p") + ")",
              "Ripple " + m("p"), "Wiggles " + m("p"), "Change of " + m(r"\rho") + " from the larger step"],
        rows=[[m_fix(r["courant"], 3), m_fix(r["co_solver"], 3), r["steps"], pct(r["e"]["rho"]), pct(r["e"]["u"]), pct(r["e"]["p"]),
               pct(r["by_wave"]["p"]["fan"]), m_pct(r["ripple"]["p"], 2), m_pct(r["wiggle"]["p"], 1),
               pct(ts_change[i - 1]) if i else "–"] for i, r in enumerate(ts)],
        note="The second row is the time step of the grid study; the first is the tutorial's limit. Last column: mean "
             "difference between the density profiles of two neighbouring rows. It falls by "
             + " and ".join(report.m_times(ts_change[i] / ts_change[i + 1], 1) for i in range(len(ts_change) - 1))
             + " per halving (orders " + " and ".join(m_fix(o, 2) for o in ts_order) + "), heading for the factor 2 of a first-order time scheme.")
    sch_tab = dict(
        title=f"Flux scheme and limiter, {sch[0]['N']} cells, same time step",
        head=["Flux", "Limiter", m("e_\\rho"), m("e_u"), m("e_p"), "Shock width (cells)", "Border width (cells)", "Ripple " + m("p"),
              "Wiggles " + m("p")],
        rows=[[r["flux"], r["limiter"] + (" (main)" if (r["flux"], r["limiter"]) == (mainc["flux"], mainc["limiter"]) else ""),
               pct(r["e"]["rho"]), pct(r["e"]["u"]), pct(r["e"]["p"]), m_fix(r["width"]["shock"], 2), m_fix(r["width"]["contact"], 2),
               m_pct(r["ripple"]["p"], 2), m_pct(r["wiggle"]["p"], 1)] for r in sch],
        note="upwind is the first-order scheme: no slope inside a cell. Minmod is the most cautious limiter, SuperBee the "
             "steepest; van Leer and van Albada lie between.")
    bias_tab = dict(
        title="The reading rules applied to the exact solution: their own bias",
        head=["Cells"] + [f"{WAVE[z]} (cells)" for z in WAVES] + ["Shock width (cells)", "Border width (cells)"],
        rows=[[f"{r['N']}"] + [m(rf"{r['pos_bias'][z] / r['h']:+.3f}") for z in WAVES]
              + [m_fix(r["width_exact"]["shock"], 2), m_fix(r["width_exact"]["contact"], 2)] for r in (coarse, mid, fine)],
        note="Each rule reads the exact cell averages on the same grid; the table shows what it returns minus the exact "
             "position, in cells. A jump that is exact but averaged over cells is read up to "
             + m_fix(max(abs(r["pos_bias"][z]) / r["h"] for r in fam for z in ("contact", "shock")), 2)
             + " cells off and about one and a half cells wide. The uniform states are read exactly (to "
             + sc(max(abs(r["read_exact"]["states"][s]["mean"] / st_exact[s] - 1) for r in fam for s in st_exact), 1) + ").")
    rules_tab = dict(
        title="The rules, written before the production runs, and what was measured",
        head=["Rule", "Kind", "Measured", "Limit", "Where", "Holds"],
        rows=[[x["text"], x["kind"].upper(), x["value"] or "–", x["limit"] or "–", x["shown"] or "–", "✓ yes" if x["ok"] else "✗ no"]
              for x in rules],
        note="The limits are in numerics.yaml (acceptance), with how each was sized from the pre-flight. A broken FAIL rule "
             "gives the verdict FAIL, a broken WARN rule the verdict WARN.")
    fan_ts = [r["by_wave"]["p"]["fan"] for r in ts]
    comp_findings = [
        "<b>Order of convergence.</b> Halving the cells cuts the mean error by about "
        + report.m_times(2 ** overall["rho"], 2) + ", not by 2 or 4: orders "
        + ", ".join(m_fix(overall[k], 2) + f" ({word})" for k, _, word in QTY)
        + " over the eight grids (Figure {fig:convergence}). Second order is out of reach with jumps in the solution: a jump "
        "spread over a fixed number of cells leaves an error in proportion to the cell size. The surprise is that the order "
        "stays below 1.",
        "<b>Where the error is.</b> Not at the shock. On the finest grid the expansion fan holds " + m_pct(share["p"]["fan"], 0)
        + " of the pressure error, " + m_pct(share["u"]["fan"], 0) + " of the velocity error and " + m_pct(share["rho"]["fan"], 0)
        + " of the density error, and its part falls with order " + m_fix(wave_overall["p"]["fan"], 2) + " (pressure), steadily over all "
        "grids. The fan is smooth inside, but its two ends are corners, and the scheme rounds them. The limit for the "
        "pressure was sized on the assumption that the shock sets the pressure error (order 1 expected, 0.85 allowed); "
        "the fan sets it. The shock's own part does fall with order " + m_fix(wave_overall["p"]["shock"], 2)
        + ". The slowest part is the stretch around the border of the two gases, where the ripple is: order "
        + m_fix(wave_overall["p"]["contact"], 2) + " for the pressure. It is " + m_pct(share["p"]["contact"], 0)
        + " of the pressure error on the finest grid and would take over on still finer ones.",
        "<b>The time step is not the cause.</b> In the time-step study the fan's part of the pressure error is "
        + ", ".join(pct(v) for v in fan_ts) + " at Courant numbers " + ", ".join(f"{r['courant']:.2f}" for r in ts)
        + ": below the step of the grid study it does not fall, it rises a little. I first suspected the ripple behind the shock; "
        "the split by wave shows that the ripple is a small part of the error.",
        "<b>The limiter is the cause, in part.</b> With the same grid and time step the density error in the fan is "
        + ", ".join(pct(r["by_wave"]["rho"]["fan"]) + f" ({r['limiter']}" + (", Tadmor flux" if r["flux"] == "Tadmor" else "") + ")" for r in sch)
        + ". No combination that OpenFOAM offers here does clearly better than the tutorial's.",
        "<b>Time-step error.</b> The error against the exact solution is lowest at the step used for the grid study and "
        "rises for smaller steps (Figure {fig:timestep}). So the step is not converged: at a quarter of it the density "
        "profile differs by " + pct(ts_main_vs_small) + " of " + m(r"\rho_L") + ", while the total error is "
        + pct(main_ts["e"]["rho"]) + ". Time error and space error partly cancel at this step. Because the step is halved with "
        "the cell size, both shrink together in the grid study.",
        "<b>Wave positions.</b> On the finest grid the shock is " + m_fix(abs(fine["pos_err"]["shock"]) / fine["h"], 2)
        + " cells and the border of the two gases " + m_fix(abs(fine["pos_err"]["contact"]) / fine["h"], 2)
        + " cells from their exact places. The two ends of the fan, read from the straight-line fit, are "
        + m_fix(abs(fine["pos_err"]["head"]) / fine["h"], 1) + " and " + m_fix(abs(fine["pos_err"]["tail"]) / fine["h"], 1)
        + " cells off; the back of the fan is about one cell too far right on every grid ("
        + m_fix(min(r["pos_err"]["tail"] / r["h"] for r in fam), 2) + " to " + m_fix(max(r["pos_err"]["tail"] / r["h"] for r in fam), 2)
        + " cells), so that error halves with the cell size.",
    ]

    perf_rows = [[r["name"], r["N"], r["steps"], m_fix(r["wall_time"], 1), m_fix(r["perf"]["us_per_cell_step"], 2),
                  m_pct(r["perf"]["cpu_efficiency"], 0) if r["perf"].get("cpu_efficiency") else "–"] for r in fam]
    big = fam[-1]["perf"]["us_per_cell_step"]

    results = dict(
        title=problem["title"], date=dt.date.today().isoformat(), verdict=verdict, comparison_type="verification",
        summary=dict(
            headline=("<b>Verification passed.</b> " if verdict == "PASS" else "<b>Verification with warnings.</b> " if verdict == "WARN"
                      else "<b>Verification FAILED on one of its own limits.</b> " if len(fails) == 1 else "<b>Verification FAILED.</b> ")
                     + "OpenFOAM's shock solver converges to the exact solution of the shock tube: on " + f"{fine['N']} cells the mean "
                     "errors are " + ", ".join(pct(fine["e"][k]) + f" ({word})" for k, _, word in QTY) + ", the states between the waves "
                     "are within " + pct(worst_state) + " and mass and energy are kept to round-off. "
                     + ("But the pressure error falls more slowly than the limit set before the runs assumed (order "
                        + m_fix(overall["p"], 2) + " against " + m_fix(acc["fail_l1_order_floor"]["p"], 2) + "): " + pct(fine["e"]["p"])
                        + " against a limit of " + pct(l1_limit["p"]) + "." if fine["e"]["p"] > l1_limit["p"] else ""),
            answer_columns=["Requested", "Exact solution", f"Simulation ({fine['N']} cells)", "Difference"],
            answers=answers, findings=findings, limitations=limitations),
        problem=dict(
            description=problem["description"], setup=problem.get("setup", []),
            given=[dict(label=v["label"], symbol=v["symbol"], value=m(tn(v["value"], 6)), unit=v["unit"]) for v in problem["given"].values()],
            wanted=wanted,
            reference="The exact solution of this problem (derived and checked in Theory). The problem is the test of " + REFS["sod"] + "."),
        assumptions=dict(rows=assumptions),
        theory=th,
        decisions=load_yaml(HERE / "decisions.yaml"),
        preflight=preflight,
        simulation=dict(model=model, meshes=meshes, results=results_tab, health=health, tables=[pos_tab]),
        comparison=dict(
            kind="Verification: every run is compared with the exact solution of the same equations, averaged over the run's own cells.",
            definitions=defs,
            head=["Cells", m("e_\\rho"), "order", m("e_u"), "order", m("e_p"), "order"],
            rows=comp_rows, findings=comp_findings, tables=[rules_tab, wave_tab, ts_tab, sch_tab, bias_tab]),
        insights=load_yaml(HERE / "insights.yaml"),
        figures=figures,
        performance=dict(
            intro="One core, one run at a time. The solver is explicit, so the cost of a run is cells times steps, and the "
                  "steps grow with the cells: each refinement costs four times the one before.",
            head=["Run", "Cells", "Steps", "Solver time (s)", "Microseconds per cell and step", "CPU busy"],
            rows=perf_rows,
            notes=["On the larger grids a cell costs about " + m_fix(big, 2) + " microseconds per step. The small runs are dominated by "
                   "start-up and by writing the totals after every step (" + m_fix(fam[0]["perf"]["us_per_cell_step"], 0)
                   + " microseconds per cell and step on " + f"{fam[0]['N']} cells).",
                   "What limits speed is the time step, not the work per step: the Courant number of " + m_fix(fine["courant"], 2)
                   + " is set by the ripple the Euler step makes: at twice the step the run still holds together but ripples by "
                   + m_pct(ts[0]["ripple"]["p"], 0) + ". A time scheme with several stages per step is the usual way to take "
                   "larger steps without that ripple; OpenFOAM's solver does not offer one, and none was tested here.",
                   "All " + str(len(allr)) + " production runs together took " + m_fix(sum(r["wall_time"] for r in allr), 0)
                   + " seconds of solver time; the finest run " + m_fix(fine["wall_time"], 0) + " seconds.",
                   "In two and three dimensions the same order of convergence makes accuracy much dearer (cells times steps "
                   "grows like the third and fourth power of the refinement). This is the number to carry to the next steps of the ladder."]),
        reproducibility=dict(
            commands=[
                dict(cmd=f"python -m cases.{HERE.name}.run preflight", what="Gas constant, short tests, time-step study, coarsest grid; writes preflight.json"),
                dict(cmd=f"python -m cases.{HERE.name}.run production", what="Every other run, one at a time; writes runs/*.json and runs/*.npz"),
                dict(cmd=f"python -m cases.{HERE.name}.analyze", what="Exact solution, comparison, figures and this report (checks the contract first)"),
            ],
            files=[
                dict(path=f"cases/{HERE.name}/problem.yaml", what="The problem as given"),
                dict(path=f"cases/{HERE.name}/numerics.yaml", what="Runs, reading rules and the pass and fail limits"),
                dict(path=f"cases/{HERE.name}/decisions.yaml", what="Every choice, with the OpenFOAM settings it controls"),
                dict(path=f"cases/{HERE.name}/insights.yaml", what="Insights logged during the work"),
                dict(path=f"cases/{HERE.name}/template/", what="The OpenFOAM case with {{ placeholders }}"),
                dict(path=f"cases/{HERE.name}/theory.py", what="The exact solution, derived and checked with SymPy; exact cell averages"),
                dict(path=f"cases/{HERE.name}/measure.py", what="The rules that read states, positions and widths from a profile"),
                dict(path=f"cases/{HERE.name}/run.py", what="Builds, pre-flights and runs the cases"),
            ],
            run_dirs=[
                dict(path=f"{fine['run_root']}/N<cells>[_spc<steps per cell>][_<flux>_<limiter>]", what="Production runs"),
                dict(path=f"{fine['run_root']}/pf_N<cells>_preflight", what="Pre-flight runs"),
                dict(path=f"results/{HERE.name}/", what="gas_constant.json, preflight.json, runs/, results.json and this report"),
            ],
            environment=report.environment_info(),
            notes=["Runs were made one at a time on one core, so the solver times can be compared.",
                   "Sources cited from memory, not supplied with the problem: " + "; ".join(REFS.values()) + "."]),
        data=dict(fails=fails, warns=warns, order=order, overall=overall, wave_overall=wave_overall, share=share, width_exp=width_exp,
                  l1_limit=l1_limit, ts_change=ts_change, ts_order=ts_order, ts_main_vs_small=ts_main_vs_small,
                  exact=dict(p_star=ex.p_star, u_star=ex.u_star, rho_starL=ex.rho_starL, rho_starR=ex.rho_starR, speeds=ex.speeds, positions=w),
                  runs={r["name"]: dict(N=r["N"], steps=r["steps"], courant=r["courant"], e=r["e"], by_wave=r["by_wave"], state=r["state"],
                                        state_err=r["state_err"], pos=r["pos"], pos_err=r["pos_err"], speed=r["speed"], width=r["width"],
                                        wiggle=r["wiggle"], ripple=r["ripple"], mass_drift=r["mass_drift"], energy_drift=r["energy_drift"],
                                        momentum_err=r["momentum_err"], ends=r["ends"], wall_time=r["wall_time"]) for r in rows_sorted}),
    )
    out = report.render(results, svgs, OUT / "report.html", case_dir=fine["case_dir"])
    print(f"{verdict}: fails {fails}; warns {warns}")
    print(f"finest: e {fine['e']}, limits {l1_limit}; orders overall {overall}; last {[order[k][-1] for k, _, _ in QTY]}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
