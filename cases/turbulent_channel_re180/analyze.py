"""Analysis and report of the turbulent channel at friction Reynolds number 180 (gpucfd, no model).

  python -m cases.turbulent_channel_re180.analyze        (repository root, inside WSL)

Reads results/turbulent_channel_re180/runs/*.json|npz, preflight.json and tolcheck_*.json, compares with the
reference of Vreman and Kuerten (2014), applies the pass / fail limits of numerics.yaml and writes results.json
and report.html. Every number in the report is computed here. With runs still missing it writes a draft
(verdict WARN, marked as a draft) so that the layout can be checked while the finest grid runs.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np

from cfdlib import gpu, insights, plots, report
from cfdlib.report import load_yaml, m, m_fix, m_pct, tex_fix as tf, tex_num as tn

from . import run as RUN
from . import stats as ST
from . import theory as TH
from .sketch import sketch

NAME = "turbulent_channel_re180"
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / NAME
RUNS = OUT / "runs"
LEVELS = ("G1", "G2", "G3")
REPEATS = ("first_G1", "second_G1")          # complete runs of G1 made under earlier settings (numerics.yaml, changes)
KAPPA, B_LOG = 0.41, 5.2                     # law of the wall, textbook values (approximate; drawn for orientation only)


def pm(x, e, d=3):
    return m(rf"{x:.{d}f} \pm {e:.{d}f}")


def html_table(head, rows):
    th = "".join(f'<th class="{"" if i == 0 else "num"}">{h}</th>' for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td class="{"" if i == 0 else "num"}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="table-wrap"><table class="full"><tr>{th}</tr>{body}</table></div>'


def have(name):
    return (RUNS / f"{name}.json").exists()


def main():
    problem, num = RUN.load()
    g = RUN.given(problem)
    ut, nu, delta = g["u_tau"], g["nu"], g["delta"]
    R, T, A = num["stop"], num["time"], num["acceptance"]
    ref_yaml = load_yaml(HERE / problem["reference"]["file"])
    REF = ST.reference(ref_yaml)
    rs = REF["scalars"]
    unc_mean, unc_rms = num["reference_uncertainty_pct"]["mean_velocity"] / 100, num["reference_uncertainty_pct"]["rms"] / 100
    paper = ref_yaml["uncertainty"]["stated_in_paper"]

    levels = [lv for lv in LEVELS if have(lv)]
    draft = len(levels) < 3 or not have("G1h")
    AV = {lv: ST.averages(lv, g, R) for lv in levels}
    extra = {n: ST.averages(n, g, R) for n in ("G1h",) + REPEATS if have(n)}
    fin = levels[-1]
    F = AV[fin]
    th = TH.evaluate(problem["given"], rs["bulk"]["value"], R["balance"])
    checks_code = ST.self_check()
    n_exp = R["friction_exponent"]

    # ------------------------------------------------------------------ differences from the reference
    def rel(x, key):
        return x / rs[key]["value"] - 1.0
    PD = {lv: {q: ST.profile_difference(AV[lv], REF, q, ut) for q in ("U", "u_rms", "v_rms", "w_rms", "uv")} for lv in levels}
    d_bulk, d_centre, d_cf = rel(F["bulk"]["mean"], "bulk"), rel(F["centre"]["mean"], "centre"), rel(F["cf"], "cf")
    d_urms, d_urms_y = rel(F["urms_peak"], "urms_peak"), F["urms_peak_y"] - rs["urms_peak_y"]["value"]
    d_shear = rel(F["shear_peak"], "shear_peak")
    s_bulk, s_centre = F["bulk"]["stderr"] / F["bulk"]["mean"], F["centre"]["stderr"] / F["centre"]["mean"]
    s_urms = F["urms_peak_err"] / F["urms_peak"]

    # ------------------------------------------------------------------ grid study (three levels, unequal ratios)
    study, noise, study_c, study_u = None, None, None, None
    h_rel = {lv: AV[lv]["rec"]["dx"] / AV[fin]["rec"]["dx"] for lv in levels}       # cell size relative to the finest level
    if len(levels) == 3:
        hs = [h_rel[lv] for lv in levels]
        study = ST.three_levels([AV[lv]["bulk"]["mean"] for lv in levels], hs)
        noise = ST.three_levels_noise([AV[lv]["bulk"]["mean"] for lv in levels], [AV[lv]["bulk"]["stderr"] for lv in levels], hs)
        study_c = ST.three_levels([AV[lv]["centre"]["mean"] for lv in levels], hs)
        study_u = ST.three_levels([AV[lv]["urms_peak"] for lv in levels], hs)
        grid_unc = study["gci_abs"] if study["reliable"] else study["delta"]          # the last change when no index can be trusted
        target = study["extrap"] if study["reliable"] else F["bulk"]["mean"]
        band = grid_unc + 2 * F["bulk"]["stderr"] + unc_mean * rs["bulk"]["value"]
        in_band = abs(target - rs["bulk"]["value"]) <= band

    # ------------------------------------------------------------------ time-step check and repeat runs (all on G1)
    g1_runs = [n for n in ("G1", "G1h") + REPEATS if n in AV or n in extra]
    G1all = {n: (AV[n] if n in AV else extra[n]) for n in g1_runs}
    dt_check = None
    if "G1" in AV and "G1h" in extra:
        a, b = AV["G1"], extra["G1h"]
        sig = float(np.hypot(a["bulk"]["stderr"], b["bulk"]["stderr"]))
        dt_check = dict(diff=b["bulk"]["mean"] - a["bulk"]["mean"], sigma=sig,
                        n_sigma=abs(b["bulk"]["mean"] - a["bulk"]["mean"]) / sig,
                        urms=b["urms_peak"] - a["urms_peak"], urms_sigma=float(np.hypot(a["urms_peak_err"], b["urms_peak_err"])),
                        centre=b["centre"]["mean"] - a["centre"]["mean"],
                        centre_sigma=float(np.hypot(a["centre"]["stderr"], b["centre"]["stderr"])))
    rep = None
    if len(G1all) >= 2:
        vals = np.array([G1all[n]["bulk"]["mean"] for n in G1all])
        errs = np.array([G1all[n]["bulk"]["stderr"] for n in G1all])
        batch = np.array([G1all[n]["bulk"]["stderr_batch"] for n in G1all])
        wmean = float(np.sum(vals / errs ** 2) / np.sum(1 / errs ** 2))
        rep = dict(names=list(G1all), values=vals, errs=errs, batch=batch, mean=wmean, spread=float(vals.max() - vals.min()),
                   std=float(vals.std(ddof=1)), chi2=float(np.sum(((vals - wmean) / errs) ** 2)),
                   chi2_batch=float(np.sum(((vals - np.sum(vals / batch ** 2) / np.sum(1 / batch ** 2)) / batch) ** 2)), dof=len(vals) - 1)

    # ------------------------------------------------------------------ rules (limits fixed in numerics.yaml before the runs)
    fails, warns, passed, missed = [], [], [], []

    def rule(ok, text_ok, text_bad, fail=False, short=None):
        (passed if ok else (fails if fail else warns)).append(text_ok if ok else text_bad)
        if not ok and fail and short:
            missed.append(short)
        return ok

    def straddles(diff, sigma2, limit):
        return abs(abs(diff) - limit) < sigma2

    names_run = [n for n in levels + ["G1h"] if n in AV or n in extra]
    allrun = {n: (AV[n] if n in AV else extra[n]) for n in names_run}
    for key, lab, dv, sv, lim in (("bulk", "Bulk velocity", d_bulk, s_bulk, A["bulk_pct"] / 100),
                                  ("centre", "Centre-plane velocity", d_centre, s_centre, A["centre_pct"] / 100),
                                  ("cf", "Friction coefficient", d_cf, 2 * s_bulk, A["cf_pct"] / 100),
                                  ("urms", "Peak of \\(u'_{rms}\\)", d_urms, s_urms, A["urms_peak_pct"] / 100)):
        txt = f"{lab} on {fin}: {m_pct(dv, 2, signed=True)} from the reference (limit {m_pct(lim, 1)}, statistical error {m_pct(sv, 2)})."
        rule(abs(dv) <= lim, txt, txt, fail=True, short=lab.lower() if key != "urms" else "peak of \\(u'_{rms}\\)")
        if straddles(dv, 2 * sv, lim):
            warns.append(f"{lab}: the difference is within two statistical errors of its limit, so a longer average could change "
                         f"this rule's outcome ({m_pct(dv, 2, signed=True)} against {m_pct(lim, 1)}, error {m_pct(sv, 2)}).")
    txt = (f"Position of the peak of \\(u'_{{rms}}\\) on {fin}: {m(tf(F['urms_peak_y'], 1))} against {m(tf(rs['urms_peak_y']['value'], 1))} "
           f"wall units (limit {m(tf(A['urms_peak_yplus'], 1))}).")
    rule(abs(d_urms_y) <= A["urms_peak_yplus"], txt, txt, fail=True, short="position of that peak")
    qlab = {"U": "mean velocity", "u_rms": "\\(u'_{rms}\\)", "v_rms": "\\(v'_{rms}\\)", "w_rms": "\\(w'_{rms}\\)", "uv": "turbulent shear stress"}
    for q, lim in A["profile_abs"].items():
        p = PD[fin][q]
        txt = (f"Profile of {qlab[q]} on {fin}: largest difference {m(tf(p['signed'], 3))} at \\(y^+ = {tf(p['y_plus'], 0)}\\) "
               f"(limit {m(tf(lim, 2))}).")
        rule(p["max_abs"] <= lim, txt, txt, fail=True, short=f"profile of {qlab[q]}")
        if straddles(p["signed"], 2 * p["sigma"], lim):
            warns.append(f"Profile of {qlab[q]}: the largest difference is within two statistical errors of its limit, so a longer average could change "
                         f"this rule's outcome ({m(tf(p['signed'], 3))} against {m(tf(lim, 2))}, error {m(tf(p['sigma'], 3))}).")
    n_output_passed = len(passed)                              # the rules above are the ones on the requested outputs
    for n in names_run:
        a = allrun[n]
        hs_ = a["rec"]["health"]["status"]
        txt = f"Run {n}: health {hs_}."
        if hs_ == "fail":
            fails.append(txt)
        else:
            rule(hs_ == "ok", txt, txt + " " + "; ".join(f"{c['name']}: {c['value']}" for c in a["rec"]["health"]["checks"] if c["status"] != "ok") + ".")
        held = a["rec"]["watch_first"] is not None
        rule(held, f"Run {n} ended on the stop rule at \\(t = {tf(a['t1'], 1)}\\).",
             f"Run {n} reached its longest time (\\(t = {tf(a['t1'], 1)}\\)) without the stop rule holding: "
             + ", ".join(k for k, ok in a["stop"]["tests"].items() if not ok) + ".")
        rule(a["stress_max"] <= A["total_stress_abs"],
             f"Run {n}: total shear stress within {m(tf(a['stress_max'], 4))} of the exact line (limit {m(tf(A['total_stress_abs'], 2))}).",
             f"Run {n}: total shear stress up to {m(tf(a['stress_max'], 4))} from the exact line (limit {m(tf(A['total_stress_abs'], 2))}).")
        rule(a["halves_sigma"] <= A["halves_sigma"],
             f"Run {n}: the two plates' mean wall shear agree within {m(tf(a['halves_sigma'], 1))} standard errors.",
             f"Run {n}: the two plates' mean wall shear differ by {m(tf(a['halves_sigma'], 1))} standard errors (limit {m(tf(A['halves_sigma'], 0))}).")
    if study:
        lo, hi = A["order_range"]
        ok = study["reliable"] and lo <= study["p"] <= hi
        rule(ok, f"Bulk velocity over the three grids: one-sided convergence with observed order {m(tf(study['p'], 2))} (accepted range {lo:g} to {hi:g}).",
             ("Bulk velocity over the three grids: " + (f"observed order {m(tf(study['p'], 2))}, outside the accepted range {lo:g} to {hi:g}"
                                                      if study["monotonic"] else "the three values do not approach a limit from one side, so no order can be read")
              + f". With the statistical error of each level, {m_pct(noise['share_monotonic'], 0)} of resampled sets give a usable order."))
        txt = (("Grid-extrapolated" if study["reliable"] else "Finest-grid") + f" bulk velocity {m(tf(target, 3))} against the reference "
               f"{m(tf(rs['bulk']['value'], 3))}: difference {m(tf(target - rs['bulk']['value'], 3))}, band {m(tf(band, 3))} "
               "(grid part, two statistical errors, reference uncertainty).")
        rule(in_band, txt, txt + " Outside the band.")
    else:
        warns.append("DRAFT: not all three grid levels are available yet; no grid study.")
    if dt_check:
        txt = (f"Half the time step on G1: bulk velocity changes by {m(tf(dt_check['diff'], 3))}, {m(tf(dt_check['n_sigma'], 1))} combined "
               f"standard errors (limit {m(tf(A['time_step_sigma'], 0))}).")
        rule(dt_check["n_sigma"] <= A["time_step_sigma"], txt, txt)
    else:
        warns.append("DRAFT: the time-step check has not run yet.")
    verdict = "FAIL" if fails else ("WARN" if warns else "PASS")

    # ------------------------------------------------------------------ error budget of the bulk velocity (request h)
    settle_unc = abs(F["imbalance"]) / n_exp                               # theory: relative gap that this imbalance allows
    budget = dict(difference=d_bulk, statistical=s_bulk, settling=settle_unc, reference=unc_mean,
                  codes=rs["bulk"]["half_range"] * 2 / rs["bulk"]["value"],
                  grid=(F["bulk"]["mean"] - study["extrap"]) / rs["bulk"]["value"] if study and study["reliable"] else None,
                  grid_last=(study["delta"] / rs["bulk"]["value"]) if study else None,
                  time_step=(dt_check["diff"] / G1all["G1"]["bulk"]["mean"], dt_check["sigma"] / G1all["G1"]["bulk"]["mean"]) if dt_check else None)
    explained = abs(d_bulk) <= 2 * s_bulk + unc_mean

    # ------------------------------------------------------------------ figures
    svgs, figures = {}, []

    def add(key, section, svg, caption):
        svgs[key] = svg
        figures.append(dict(key=key, section=section, caption=caption))

    from gpucfd import turbulence
    uc0 = float(turbulence.reichardt(delta * ut / nu))
    add("sketch", "problem", sketch(lambda yy: turbulence.reichardt(np.minimum(yy, 2.0 - yy) * delta * ut / nu) / uc0),
        "The problem: fluid between two plates at rest, driven along \\(x\\) by a constant force. The flow is turbulent; the box "
        "shown repeats along \\(x\\) and \\(z\\). The profile is a sketch, not a result.")
    add("history", "simulation", fig_history(AV, extra, levels, g),
        "Left: bulk velocity of each run against time (time unit \\(\\delta/u_\\tau\\)); the dotted line of the same colour marks "
        "where its average starts. The slow wandering over ten and more time units is the bulk velocity's long memory "
        f"(Section 3). Right: wall shear stress on the two plates in run {fin}; its average must equal the driving force (the line at 1).")
    if len(levels) >= 2:
        add("grid", "convergence", fig_grid(AV, levels, h_rel, REF, PD, study, unc_mean),
            "Left: bulk velocity on each grid against its cell size (relative to the finest grid), with one statistical standard "
            "error; the band is the reference with its own uncertainty"
            + (", the open marker the value extrapolated to zero cell size with the observed order" if study and study["reliable"] else "")
            + ". Right: largest difference from the reference along each profile, in wall units.")
    add("profile", "comparison", fig_profile(AV, levels, REF, PD, A, ut),
        "Left: mean velocity in wall units on the three grids against the reference, with the law of the wall (\\(u^+ = y^+\\) at "
        "the wall; the logarithmic law with textbook constants further out, for orientation). Right: difference from the "
        f"reference; the dashed lines are the limit set before the runs, the shading two statistical errors of {fin}.")
    add("rms", "comparison", fig_rms(AV, levels, REF, ut, fin),
        f"Left: the three fluctuation intensities on {fin} (markers) against the reference (lines), in wall units. Right: "
        "\\(u'_{rms}\\) near its peak on the three grids.")
    add("stress", "comparison", fig_stress(AV, levels, REF, fin, g),
        f"Left: shear stresses on {fin} across half the channel: the part carried by the eddies, the viscous part, their sum, and "
        "the exact straight line of the mean momentum balance. Right: what is left of that balance after averaging (sum minus "
        "exact line), over the whole channel, for each run.")

    # ------------------------------------------------------------------ tables
    def cells_row(lv):
        a = AV[lv]
        r = a["rec"]
        dy = np.diff(np.array(r["y_faces"])) * ut / nu
        return [lv, m(rf"{r['n'][0]} \times {r['n'][1]} \times {r['n'][2]}"), f"{r['cells']:,}", m_fix(r["dx"] * ut / nu, 1),
                m_fix(r["dz"] * ut / nu, 1), m_fix(float(dy.min()), 2), m_fix(float(dy.max()), 1), m_fix(float(np.max(dy[1:] / dy[:-1])), 2),
                m_fix(h_rel[lv], 2)]
    cs = paper["largest_cell_sizes_in_wall_units"]
    mesh_rows = [cells_row(lv) for lv in levels]
    for code, nm in (("finite_difference", "Reference, finite differences"), ("spectral", "Reference, spectral")):
        res_ = ref_yaml["setup"][code]["resolution"].replace(" x ", r" \times ")
        mesh_rows.append([nm, m(res_), "", m_fix(cs[code]["x"], 1), m_fix(cs[code]["z"], 1), "", m_fix(cs[code]["y"], 1), "", ""])

    def res_row(n, a):
        return [n, m(rf"{a['t0']:.1f} \ldots {a['t1']:.1f}"), pm(a["bulk"]["mean"], a["bulk"]["stderr"]), pm(a["centre"]["mean"], a["centre"]["stderr"]),
                m(tf(a["cf"], 6)), m(f"{a['re_b']:.0f}"), pm(a["urms_peak"], a["urms_peak_err"]), m_fix(a["urms_peak_y"], 1),
                m_fix(a["shear_peak"], 3)]
    res_rows = [res_row(n, allrun[n]) for n in names_run] + [res_row(n, extra[n]) for n in REPEATS if n in extra]
    res_rows.append(["Reference", "200 time units", m(tf(rs["bulk"]["value"], 3)), m(tf(rs["centre"]["value"], 3)), m(tf(rs["cf"]["value"], 6)),
                     m(f"{rs['re_b']['value']:.0f}"), m(tf(rs["urms_peak"]["value"], 3)), m_fix(rs["urms_peak_y"]["value"], 1),
                     m_fix(rs["shear_peak"]["value"], 3)])

    def health_row(n, a):
        r = a["rec"]
        c = {x["name"]: x for x in r["health"]["checks"]}
        return [n, r["health"]["status"], f"{r['perf']['steps']:,}", m_fix(r["dt"] * ut ** 2 / nu, 2), c["Max Courant number"]["value"],
                m(tn(float(c["Divergence after projection"]["value"]), 2)), "stop rule" if r["watch_first"] else "longest time",
                m_fix(a["bulk"]["xi_independent"], 0), m_fix(a["stop"]["drift"], 1), m_pct(a["imbalance"], 2, signed=True),
                m_fix(a["halves_sigma"], 1), m(tf(a["stress_max"], 4))]
    health_rows = [health_row(n, allrun[n]) for n in names_run] + [health_row(n, extra[n]) for n in REPEATS if n in extra]

    def perf_row(n, a):
        r = a["rec"]
        p = r["perf"]
        return [n, f"{r['cells']:,}", f"{p['steps']:,}", m_fix(r["end_time"], 1), m_fix(p["wall_s"] / 60, 1), m_fix(p["ms_per_step_wall"], 1),
                m_fix(p["us_per_cell_step_wall"], 3), m_fix(p["poisson_its"], 1), m_pct(p["observe_s"] / p["wall_s"], 1),
                m_fix(p["wall_s"] / max(r["end_time"] - r["start_time"], 1e-9), 1), m(f"{r['gpu_memory_mib']:.0f}") if r.get("gpu_memory_mib") else "–"]
    perf_rows = [perf_row(n, allrun[n]) for n in names_run] + [perf_row(n, extra[n]) for n in REPEATS if n in extra]
    total_wall = sum(a["rec"]["perf"]["wall_s"] for a in list(allrun.values()) + [extra[n] for n in REPEATS if n in extra])

    # ------------------------------------------------------------------ pre-flight
    pf = json.loads((OUT / "preflight.json").read_text(encoding="utf-8"))
    tolc = {lv: json.loads((OUT / f"tolcheck_{lv}.json").read_text(encoding="utf-8")) for lv in ("G1", "G3") if (OUT / f"tolcheck_{lv}.json").exists()}
    pf_runs, pf_checks = [], []
    for r_ in pf["runs"]:
        pf_runs.append(dict(
            name=r_["name"],
            setup=(f"{r_['level']}, {r_['cells']:,} cells, {m(tf(r_['length'], 0))} time units from "
                   + ("the artificial start field" if r_["name"] == "pre_G1" else "the coarse run's last field, interpolated")
                   + " (first plan: Courant target 0.8, pressure tolerance \\(10^{-8}\\))"),
            outcome=(f"health {r_['health']['status']}; {m(tf(r_['ms_per_step_wall'], 1))} ms per step, {m(tf(r_['steps_per_unit'], 0))} steps and "
                     f"{m(tf(r_['wall_per_unit_s'], 1))} s per time unit; wall shear {m(tf(r_['tau_end'], 2))} of the driving force at the end")))
        for c in r_["health"]["checks"]:
            val = m(tn(float(c["value"]), 2)) if "e-" in c["value"] else c["value"]          # 3.6e-12 as typeset math
            pf_checks.append(dict(name=f"{r_['name']}: {c['name']}", value=val, status=c["status"], meaning=c["meaning"]))
    for lv, tc in tolc.items():
        loose = tc["runs"][-1]
        mid = next(x for x in tc["runs"] if abs(x["tol"] - 1e-8) < 1e-12)
        pf_runs.append(dict(
            name=f"tolerance check {lv}",
            setup=(f"The same {m(tf(tc['length'], 1))} time units of flow with pressure tolerances "
                   + ", ".join(report.m_pow10(x["tol"]) for x in tc["runs"]) + " (fixed time step)"),
            outcome=(f"largest change of bulk velocity and wall shear with {report.m_pow10(loose['tol'])}: {m(tn(loose['worst'], 2))} "
                     f"(rule {m(tn(tc['rule'], 1))}); {m(tf(loose['ms_per_step'], 1))} ms per step against {m(tf(mid['ms_per_step'], 1))} with "
                     f"{report.m_pow10(mid['tol'])}")))
        pf_checks.append(dict(name=f"{lv}: same answer with pressure tolerance {report.m_pow10(tc['chosen'])}", value=m(tn(loose["worst"], 2)),
                              status="ok" if loose["passes"] else "fail",
                              meaning="Largest difference of bulk velocity and wall shear from the run with the strictest tolerance. "
                                      "All five tolerances pass by many orders: for a flow without bodies these outputs barely feel "
                                      "the pressure tolerance, so the check supports the choice but does not discriminate (insight)."))
    for n in REPEATS:
        if n in extra:
            a = extra[n]
            c = {x["name"]: x for x in a["rec"]["health"]["checks"]}
            first = n == "first_G1"
            pf_runs.append(dict(
                name=n + (" (first complete run of G1)" if first else " (second complete run of G1)"),
                setup=("Courant target 0.8, step re-chosen every 10 steps, first stop rule" if first else
                       "Courant target 0.75, step re-chosen every 5 steps, first stop rule"),
                outcome=(f"health {a['rec']['health']['status']} (largest Courant number {c['Max Courant number']['value']}); ended on the first stop "
                         f"rule at \\(t = {tf(a['t1'], 1)}\\); bulk velocity over the window it used then: "
                         + m(tf(a["first_rule"]["mean"], 3)) + f" with a block estimate of {m(tf(a['first_rule']['stderr'], 3))}")))
    if rep:
        pf_checks.append(dict(
            name="Repeat runs of G1 agree within their statistical errors", value=(
                f"spread {m(tf(rep['spread'], 3))}; \\(\\chi^2 = {tf(rep['chi2'], 1)}\\) for {rep['dof']} degrees of freedom with the error "
                f"used; {m(tf(rep['chi2_batch'], 0))} with plain block means"),
            status="ok" if rep["chi2"] <= 3 * rep["dof"] else "warn",
            meaning="The runs of the coarsest grid (the two early ones, the final one and the one with half the time step) are "
                    "independent samples of the same flow. Their scatter tests the error estimate directly: plain block means of the "
                    "bulk velocity are far too small, the estimate that knows the slow mode (Section 3) is consistent."))
    pf_checks.append(dict(name="Projected time of the finest run", value=m(tf(pf["runs"][-1]["wall_per_unit_s"], 0)) + " s per time unit",
                          status="warn", meaning="About an hour for the 60 time units the stop rule needs: above the machine's usual "
                                                 "15 minutes per run. Accepted, because the length is set by the flow (Decisions, run budget)."))

    # ------------------------------------------------------------------ text: answers
    refname = "Vreman and Kuerten (2014), mean of two codes"
    wanted = [dict(letter=report.LETTERS[i], key=q["key"], text=q["text"]) for i, q in enumerate(problem["quantities_of_interest"])]
    L = {w["key"]: w["letter"] for w in wanted}
    answers = [
        dict(letter=L["bulk"], label="Bulk velocity and bulk Reynolds number",
             theory=f"\\(U_b/u_\\tau = {tf(rs['bulk']['value'], 3)}\\), \\(Re_b = {rs['re_b']['value']:.0f}\\) (own uncertainty {m_pct(unc_mean, 1)})",
             sim=f"\\(U_b/u_\\tau = {tf(F['bulk']['mean'] / ut, 3)} \\pm {tf(F['bulk']['stderr'] / ut, 3)}\\), \\(Re_b = {F['re_b']:.0f} \\pm {F['re_b_err']:.0f}\\) "
                 f"(one statistical standard error, {m_pct(s_bulk, 2)})",
             diff=m_pct(d_bulk, 2, signed=True)),
        dict(letter=L["cf"], label="Friction coefficient", theory=m(tn(rs["cf"]["value"], 4)),
             sim=m(rf"{tn(F['cf'], 4)} \pm {tn(F['cf_err'], 2)}"), diff=m_pct(d_cf, 2, signed=True)),
        dict(letter=L["centre"], label="Centre-plane velocity", theory=f"\\(U_c/u_\\tau = {tf(rs['centre']['value'], 3)}\\)",
             sim=f"\\(U_c/u_\\tau = {tf(F['centre']['mean'] / ut, 3)} \\pm {tf(F['centre']['stderr'] / ut, 3)}\\)", diff=m_pct(d_centre, 2, signed=True)),
        dict(letter=L["profile"], label="Mean velocity profile", theory="Profiles of both codes; at the wall the exact limit \\(u^+ = y^+ - (y^+)^2/(2Re_\\tau)\\)",
             sim=(f"Figure {{fig:profile}}. Largest difference from the reference {m(tf(PD[fin]['U']['signed'], 3))} at \\(y^+ = {tf(PD[fin]['U']['y_plus'], 0)}\\); "
                  f"wall cell {m_pct(F['wall_sim'] / F['wall_exact'] - 1, 2, signed=True)} from the exact wall limit"),
             diff=m(tf(PD[fin]["U"]["signed"], 3)) + "<br>(in \\(u_\\tau\\))"),
        dict(letter=L["rms"], label="Velocity fluctuations",
             theory=f"peak \\(u'_{{rms}} = {tf(rs['urms_peak']['value'], 3)}\\) at \\(y^+ = {tf(rs['urms_peak_y']['value'], 1)}\\)",
             sim=(f"Figure {{fig:rms}}. Peak \\(u'_{{rms}} = {tf(F['urms_peak'], 3)} \\pm {tf(F['urms_peak_err'], 3)}\\) at \\(y^+ = {tf(F['urms_peak_y'], 1)}\\). Largest "
                  f"differences along the profiles: \\(u'\\) {m(tf(PD[fin]['u_rms']['signed'], 3))}, \\(v'\\) {m(tf(PD[fin]['v_rms']['signed'], 3))}, "
                  f"\\(w'\\) {m(tf(PD[fin]['w_rms']['signed'], 3))}"),
             diff=m_pct(d_urms, 2, signed=True) + "<br>(peak)"),
        dict(letter=L["shear"], label="Shear stress",
             theory=f"peak \\(-\\langle u'v'\\rangle/u_\\tau^2 = {tf(rs['shear_peak']['value'], 4)}\\) at \\(y^+ = {tf(rs['shear_peak_y']['value'], 1)}\\); total stress: the exact line",
             sim=(f"Figure {{fig:stress}}. Peak {m(tf(F['shear_peak'], 4))} at \\(y^+ = {tf(F['shear_peak_y'], 1)}\\); total stress within "
                  f"{m(tf(F['stress_max'], 4))} of the exact line across the channel"),
             diff=m_pct(d_shear, 2, signed=True) + "<br>(peak)"),
        dict(letter=L["balance"], label="Wall shear against driving force", theory="\\(\\tau_w = \\rho f\\delta\\) exactly, for a settled average",
             sim=(f"mean wall shear {m(tf(F['tau']['mean'] / (g['f'] * delta), 4))} of \\(\\rho f\\delta\\) over {m(tf(F['span'], 0))} time units "
                  f"(lower plate {m_pct(F['imbalance_low'], 2, signed=True)}, upper {m_pct(F['imbalance_high'], 2, signed=True)}); by the "
                  f"balance of the box this allows the bulk velocity to be up to {m_pct(settle_unc, 2)} from settled"),
             diff=m_pct(F["imbalance"], 2, signed=True)),
        dict(letter=L["resolved"], label="Is the turbulence resolved?", theory="–",
             sim=resolved_text(fin, F, d_bulk, s_bulk, d_urms, PD, A, budget, study, noise, unc_mean, fails, d_centre, d_urms_y),
             diff=m_pct(d_bulk, 2, signed=True) + "<br>(bulk)"),
    ]

    # ------------------------------------------------------------------ findings, limitations
    sign = "above" if d_bulk > 0 else "below"
    sp_lo, sp_hi = round(min(a["span"] for a in AV.values())), round(max(a["span"] for a in AV.values()))
    span_txt = f"about {m(str(sp_lo))}" if sp_lo == sp_hi else f"{m(str(sp_lo))} to {m(str(sp_hi))}"
    mean_ok = abs(d_bulk) <= A["bulk_pct"] / 100 and abs(d_centre) <= A["centre_pct"] / 100 and PD[fin]["U"]["max_abs"] <= A["profile_abs"]["U"]
    fluct_ok = (abs(d_urms) <= A["urms_peak_pct"] / 100 and abs(d_urms_y) <= A["urms_peak_yplus"]
                and all(PD[fin][q]["max_abs"] <= A["profile_abs"][q] for q in ("u_rms", "v_rms", "w_rms", "uv")))
    findings = [
        f"On the finest grid ({m(AV[fin]['rec']['n'][0])} by {m(AV[fin]['rec']['n'][1])} by {m(AV[fin]['rec']['n'][2])} cells, {m(tf(AV[fin]['rec']['dx'] * ut / nu, 1))} wall "
        f"units along and across the flow) the bulk velocity is {m_pct(abs(d_bulk), 2)} {sign} the reference, with a statistical error of "
        f"{m_pct(s_bulk, 2)} from {m(tf(F['span'], 0))} time units of averaging and a reference uncertainty of {m_pct(unc_mean, 1)}.",
        f"Fluctuations on the finest grid: the peak of \\(u'_{{rms}}\\) is {m_pct(d_urms, 1, signed=True)} from the reference and sits at \\(y^+ = {tf(F['urms_peak_y'], 1)}\\) "
        f"against {m(tf(rs['urms_peak_y']['value'], 1))}; along the profiles \\(u'_{{rms}}\\) differs by up to {m(tf(PD[fin]['u_rms']['signed'], 3))} (at \\(y^+ = "
        f"{tf(PD[fin]['u_rms']['y_plus'], 0)}\\)), \\(v'_{{rms}}\\) by {m(tf(PD[fin]['v_rms']['signed'], 3))} and \\(w'_{{rms}}\\) by {m(tf(PD[fin]['w_rms']['signed'], 3))} "
        "(in \\(u_\\tau\\))."
        + (" Too much energy along the flow close to the wall and too little across it is what cells too large for the streaks at the wall would give; "
           "that reading was not tested by refining one direction alone."
           if PD[fin]["u_rms"]["signed"] > 0 and PD[fin]["v_rms"]["signed"] < 0 and PD[fin]["w_rms"]["signed"] < 0 else ""),
    ]
    pu = PD[fin]["U"]
    lo_i, hi_i = int(np.argmin(pu["diff"])), int(np.argmax(pu["diff"]))
    if pu["diff"][lo_i] < -2 * pu["err"][lo_i] and abs(pu["diff"][lo_i]) > 3 * abs(d_bulk) * F["bulk"]["mean"] / ut:
        findings.append(
            f"The close agreement of the bulk velocity hides a larger error of the profile: the mean velocity on {fin} is {m(tf(abs(pu['diff'][lo_i]), 2))} "
            f"\\(u_\\tau\\) below the reference at \\(y^+ = {tf(pu['y'][lo_i], 0)}\\) (many statistical errors) and {m(tf(pu['diff'][-1], 2))} from it at the "
            f"centre, while the bulk velocity, the average of the profile over the channel, is only {m(tf(abs(d_bulk) * rs['bulk']['value'], 2))} off. "
            "Differences of opposite tendency in the wall layer and in the core partly cancel in it"
            + (", and on the three grids the bulk velocity does not move toward the reference from one side." if study and not study["monotonic"] else "."))
    if study:
        seq = ", ".join(m(tf(AV[lv]["bulk"]["mean"], 3)) for lv in levels)
        findings.append(
            f"Grid study: the bulk velocity on the three grids is {seq}. "
            + (f"It approaches a limit from one side with observed order {m(tf(study['p'], 2))}; the extrapolated value is {m(tf(study['extrap'], 3))}, "
               f"{m_pct(study['extrap'] / rs['bulk']['value'] - 1, 2, signed=True)} from the reference. " if study["reliable"] else
               "The three values do not approach a limit from one side, so no order of accuracy and no extrapolated value can be given; the last "
               f"change, {m(tf(study['delta'], 3))}, stands for the grid uncertainty. ")
            + f"With the statistical error of each level, {m_pct(noise['share_monotonic'], 0)} of resampled sets of the three values give a usable order"
            + (f" (16 to 84 per cent range of the order: {m(tf(noise['p_lo'], 1))} to {m(tf(noise['p_hi'], 1))})." if "p_lo" in noise else "."))
    if rep:
        findings.append(
            f"Statistical error: {len(rep['names'])} independent runs of the coarsest grid give bulk velocities between {m(tf(rep['values'].min(), 3))} and "
            f"{m(tf(rep['values'].max(), 3))}. Plain block means claim errors of {m(tf(rep['batch'].min(), 3))} to {m(tf(rep['batch'].max(), 3))}, which the "
            f"scatter contradicts; the estimate from the balance of the box gives {m(tf(rep['errs'].min(), 3))} to {m(tf(rep['errs'].max(), 3))} and is consistent "
            f"with it. The bulk velocity remembers for about {m(tf(th['values']['t_settle'], 0))} time units, so tens of time units fix it to a few tenths of a per cent, not better.")
    if dt_check:
        findings.append(f"Time step: with half the step on the coarsest grid the bulk velocity changes by {m(tf(dt_check['diff'], 3))} "
                        f"\\(\\pm\\) {m(tf(dt_check['sigma'], 3))} and the peak of \\(u'_{{rms}}\\) by {m(tf(dt_check['urms'], 3))} \\(\\pm\\) {m(tf(dt_check['urms_sigma'], 3))}: "
                        "no effect of the time step can be seen above the statistical error.")
    findings.append(f"Checks of the averages themselves on {fin}: mean wall shear {m_pct(F['imbalance'], 2, signed=True)} from the driving force, total shear "
                    f"stress within {m(tf(F['stress_max'], 4))} of the exact line, the two plates alike within {m(tf(F['halves_sigma'], 1))} standard errors.")
    if fails:
        findings.append("<b>Limits missed (set before the runs):</b> " + " ".join(fails))
    if warns:
        findings.append("<b>Checks of the evidence that did not pass:</b> " + " ".join(warns))
    limitations = [
        f"One finest grid of {AV[fin]['rec']['cells']:,} cells: the reference used cells two to four times smaller in each direction. A finer level (ratio 1.5: 2.7 million cells) "
        "would cost about four hours for the same averaging time on this computer and was not run.",
        f"Averaging times of {span_txt} time units against the reference's 200: the "
        f"statistical error of the bulk velocity is {m_pct(s_bulk, 2)}, and the error estimate itself rests on a handful of independent samples and on a settling time "
        "from an empirical friction law.",
        "The time-step check and the repeat runs are on the coarsest grid only; the finer grids use smaller steps.",
        "The grid family refines all three directions together. Which direction limits the accuracy (cells along the flow, across it, or normal to the wall) was not separated.",
        "Start-up cut, statistical error model and stop rule were revised during the work, after the first two runs of the coarsest grid (Section 5 and the insights); the "
        "pass and fail limits on the outputs were not changed.",
        "The logarithmic law drawn in the profile figure uses textbook constants quoted from memory; it is for orientation and enters no number.",
    ]
    if draft:
        limitations.insert(0, "<b>DRAFT:</b> not all runs are finished; numbers and verdict will change.")

    headline = (f"<b>{verdict}.</b> " + ("DRAFT. " if draft else "")
                + f"Computed without a turbulence model on {AV[fin]['rec']['cells']:,} cells, the bulk velocity is \\(U_b/u_\\tau = {tf(F['bulk']['mean'] / ut, 2)} "
                  f"\\pm {tf(F['bulk']['stderr'] / ut, 2)}\\), {m_pct(d_bulk, 1, signed=True)} from the reference ({m(tf(rs['bulk']['value'], 2))}); "
                + (f"{len(fails)} of the {len(fails) + n_output_passed} limits set before the runs on the requested outputs "
                   + ("is" if len(fails) == 1 else "are") + " missed: " + ", ".join(missed) + "." if fails else
                   "every limit on the requested outputs is met" + (", with caveats on the evidence." if warns else ".")))

    # ------------------------------------------------------------------ comparison table and its findings
    def crow(label, refv, simv, diff, limit, ok):
        return [label, refv, simv, diff, limit, "pass" if ok else "MISS"]
    comp_rows = [
        crow("Bulk velocity \\(U_b/u_\\tau\\)", m(tf(rs["bulk"]["value"], 3)), pm(F["bulk"]["mean"], F["bulk"]["stderr"]), m_pct(d_bulk, 2, signed=True),
             m_pct(A["bulk_pct"] / 100, 1), abs(d_bulk) <= A["bulk_pct"] / 100),
        crow("Centre-plane velocity \\(U_c/u_\\tau\\)", m(tf(rs["centre"]["value"], 3)), pm(F["centre"]["mean"], F["centre"]["stderr"]),
             m_pct(d_centre, 2, signed=True), m_pct(A["centre_pct"] / 100, 1), abs(d_centre) <= A["centre_pct"] / 100),
        crow("Friction coefficient \\(C_f\\)", m(tn(rs["cf"]["value"], 4)), m(rf"{tn(F['cf'], 4)} \pm {tn(F['cf_err'], 2)}"), m_pct(d_cf, 2, signed=True),
             m_pct(A["cf_pct"] / 100, 1), abs(d_cf) <= A["cf_pct"] / 100),
        crow("Peak of \\(u'_{rms}/u_\\tau\\)", m(tf(rs["urms_peak"]["value"], 3)), pm(F["urms_peak"], F["urms_peak_err"]), m_pct(d_urms, 2, signed=True),
             m_pct(A["urms_peak_pct"] / 100, 1), abs(d_urms) <= A["urms_peak_pct"] / 100),
        crow("Its position \\(y^+\\)", m(tf(rs["urms_peak_y"]["value"], 1)), m(tf(F["urms_peak_y"], 1)), m(tf(d_urms_y, 1)), m(tf(A["urms_peak_yplus"], 1)),
             abs(d_urms_y) <= A["urms_peak_yplus"]),
        crow("Peak of \\(-\\langle u'v'\\rangle/u_\\tau^2\\)", m(tf(rs["shear_peak"]["value"], 4)), pm(F["shear_peak"], F["shear_peak_err"], 4),
             m_pct(d_shear, 2, signed=True), "–", True),
    ]
    plabel = {"U": "Mean velocity profile, largest difference", "u_rms": "\\(u'_{rms}\\) profile, largest difference",
              "v_rms": "\\(v'_{rms}\\) profile, largest difference", "w_rms": "\\(w'_{rms}\\) profile, largest difference",
              "uv": "Turbulent shear stress profile, largest difference"}
    for q, lim in A["profile_abs"].items():
        p = PD[fin][q]
        comp_rows.append(crow(plabel[q], "profile", f"at \\(y^+ = {tf(p['y_plus'], 0)}\\)", m(tf(p["signed"], 3)), m(tf(lim, 2)), p["max_abs"] <= lim))
    comp_findings = []
    if len(levels) == 3:
        rows = []
        for lv in levels:
            a = AV[lv]
            rows.append([lv, m_fix(h_rel[lv], 2), pm(a["bulk"]["mean"], a["bulk"]["stderr"]), m_pct(rel(a["bulk"]["mean"], "bulk"), 2, signed=True),
                         pm(a["centre"]["mean"], a["centre"]["stderr"]), m_pct(rel(a["centre"]["mean"], "centre"), 2, signed=True),
                         pm(a["urms_peak"], a["urms_peak_err"]), m_pct(rel(a["urms_peak"], "urms_peak"), 2, signed=True)])
        prow = [[lv] + [f"{m(tf(PD[lv][q]['signed'], 3))} at {m(tf(PD[lv][q]['y_plus'], 0))}" for q in ("U", "u_rms", "v_rms", "w_rms", "uv")] for lv in levels]
        prow.append(["Limit"] + [m(tf(A["profile_abs"][q], 2)) for q in ("U", "u_rms", "v_rms", "w_rms", "uv")])
        ex = ["Extrapolated", "0", (m(tf(study["extrap"], 3)) if study["reliable"] else "–"),
              (m_pct(rel(study["extrap"], "bulk"), 2, signed=True) if study["reliable"] else "–"),
              (m(tf(study_c["extrap"], 3)) if study_c["reliable"] else "–"), (m_pct(rel(study_c["extrap"], "centre"), 2, signed=True) if study_c["reliable"] else "–"),
              (m(tf(study_u["extrap"], 3)) if study_u["reliable"] else "–"), (m_pct(rel(study_u["extrap"], "urms_peak"), 2, signed=True) if study_u["reliable"] else "–")]
        od = ["Observed order", "", (m(tf(study["p"], 2)) if study["monotonic"] else "not one-sided"), "",
              (m(tf(study_c["p"], 2)) if study_c["monotonic"] else "not one-sided"), "", (m(tf(study_u["p"], 2)) if study_u["monotonic"] else "not one-sided"), ""]
        comp_findings.append(
            "<b>Grid study.</b> The three levels, each with one statistical standard error, and their differences from the reference. Where three values approach a "
            "limit from one side, the observed order and the value extrapolated to zero cell size are given (Figure {fig:grid})."
            + html_table(["Grid", "Cell size", "\\(U_b/u_\\tau\\)", "from ref.", "\\(U_c/u_\\tau\\)", "from ref.", "peak \\(u'_{rms}/u_\\tau\\)", "from ref."], rows + [ex, od])
            + "Largest difference from the reference along each profile, in wall units, with the \\(y^+\\) where it occurs:"
            + html_table(["Grid", "\\(U\\)", "\\(u'_{rms}\\)", "\\(v'_{rms}\\)", "\\(w'_{rms}\\)", "\\(-\\langle u'v'\\rangle\\)"], prow)
            + (f"The index of the finest grid for the bulk velocity is {m_pct(study['gci'], 2)} (factor 1.25 on the observed order). " if study["reliable"] else
               f"No convergence index is given for the bulk velocity ({study['basis']}). ")
            + (f"The peak of \\(u'_{{rms}}\\) does fall from one side, but its extrapolated value is further from the reference than the value on {fin}: "
               "three levels this coarse are not in the range where an extrapolation can be trusted. "
               if study_u["reliable"] and abs(rel(study_u["extrap"], "urms_peak")) > abs(d_urms) else "")
            + f"Resampling the three bulk velocities with their statistical errors, {m_pct(noise['share_monotonic'], 0)} of the sets give a usable order"
            + (f", with the extrapolated value between {m(tf(noise['extrap_lo'], 2))} and {m(tf(noise['extrap_hi'], 2))} (16 to 84 per cent)." if "extrap_lo" in noise else "."))
    b = budget
    brow = [["Difference from the reference on " + fin, m_pct(b["difference"], 2, signed=True), "measured"],
            ["Statistical error of the average (one standard error)", "\\(\\pm\\)" + m_pct(b["statistical"], 2), f"from the balance of the box, {m(tf(F['span'], 0))} time units"],
            ["Not fully settled (allowed by the wall-shear imbalance)", "\\(\\pm\\)" + m_pct(b["settling"], 2), "imbalance over the friction-law exponent"],
            ["Reference: authors' bound on its error", "\\(\\pm\\)" + m_pct(b["reference"], 1), f"its two codes differ by {m_pct(b['codes'], 2)}"]]
    if b["grid"] is not None:
        brow.insert(1, ["Grid error (finest grid minus extrapolated value)", m_pct(b["grid"], 2, signed=True), f"observed order {m(tf(study['p'], 2))}"])
    elif b["grid_last"] is not None:
        brow.insert(1, ["Grid: change from the second finest to the finest grid", "\\(\\pm\\)" + m_pct(b["grid_last"], 2), "no one-sided convergence, so no error estimate"])
    if b["time_step"]:
        brow.append(["Time step (half the step, coarsest grid)", m_pct(b["time_step"][0], 2, signed=True) + " \\(\\pm\\) " + m_pct(b["time_step"][1], 2), "not distinguishable from zero"
                     if abs(b["time_step"][0]) <= 2 * b["time_step"][1] else "larger than two standard errors"])
    comp_findings.append("<b>What the remaining difference of the bulk velocity is made of.</b>" + html_table(["Part", "Size", "From"], brow)
                         + ("The difference is no larger than two statistical errors plus the reference's uncertainty: on this evidence the mean flow rate "
                            "cannot be told apart from the reference." if explained else
                            "The difference is larger than two statistical errors plus the reference's uncertainty: a real error of the grid remains in the mean flow rate."))
    if rep:
        rrow = [[n, pm(G1all[n]["bulk"]["mean"], G1all[n]["bulk"]["stderr"]), m(tf(G1all[n]["bulk"]["stderr_batch"], 3)), m(tf(G1all[n]["span"], 0)),
                 m(tf(G1all[n]["rec"]["time_step"]["adaptive_cfl"], 3))] for n in rep["names"]]
        comp_findings.append(
            "<b>Repeat runs of the coarsest grid: a direct test of the statistical error.</b> Independent runs of the same grid, each averaged after its own start-up."
            + html_table(["Run", "\\(U_b\\) with the error used", "Plain block-means error", "Time units averaged", "Courant target"], rrow)
            + f"The values spread over {m(tf(rep['spread'], 3))}. Against the error used, \\(\\chi^2 = {tf(rep['chi2'], 1)}\\) for {rep['dof']} degrees of freedom "
              f"(consistent when about equal); against plain block means it is {m(tf(rep['chi2_batch'], 0))}.")
    comp_findings.append("<b>Rules that passed.</b> " + " ".join(passed))
    if fails or warns:
        comp_findings.append("<b>Rules that did not pass.</b> " + " ".join(fails + warns))

    # ------------------------------------------------------------------ assumptions
    assumptions = [
        dict(name="Incompressible flow, constant density and viscosity", check="Given in the problem; the reference solves the same equations.", status="imposed"),
        dict(name="The periodic box stands for a very long and wide channel", status="standard",
             check="Box and periodicity as given and as in the reference, so any effect of the box size is the same on both sides of the comparison. "
                   f"The mean flow crosses the box once in {m(tf(th['values']['t_flow'], 2))} time units."),
        dict(name="The flow is turbulent, not the laminar solution", status="ok",
             check=f"Bulk velocity {m(tf(F['bulk']['mean'], 2))} against the laminar {m(tf(th['values']['lam_bulk'], 0))}; fluctuations of the size of the "
                   f"reference's (peak \\(u'_{{rms}}\\) {m(tf(F['urms_peak'], 2))})."),
        dict(name="The averages are those of a settled flow", status="ok" if abs(F["imbalance"]) <= R["balance"] and abs(F["stop"]["drift"]) <= R["max_drift"] else "violated",
             check=f"On {fin}: mean wall shear {m_pct(F['imbalance'], 2, signed=True)} from the driving force (limit {m_pct(R['balance'], 1)}); second half of the "
                   f"record against the first: {m(tf(F['stop']['drift'], 1))} standard errors; start-up of {m(tf(F['t0'], 1))} time units cut."),
        dict(name="The two halves of the channel are statistically alike", status="ok" if F["halves_sigma"] <= A["halves_sigma"] else "violated",
             check=f"Mean wall shear of the two plates on {fin}: {m_pct(F['imbalance_low'], 2, signed=True)} and {m_pct(F['imbalance_high'], 2, signed=True)} from the "
                   f"driving force, {m(tf(F['halves_sigma'], 1))} standard errors apart. The profiles are averaged over both halves."),
        dict(name="All eddies are resolved by the grid (no model)", status="violated" if fails else "ok",
             check=f"Tested by the grid study and against the reference. Cells of {m(tf(AV[fin]['rec']['dx'] * ut / nu, 1))} wall units in the wall plane on {fin} carry the "
                   f"bulk velocity to {m_pct(abs(d_bulk), 1)}; the mean flow " + ("meets" if mean_ok else "misses") + " its limits and the fluctuations "
                   + ("meet" if fluct_ok else "miss") + " theirs (Summary, request " + L["resolved"] + ")."),
        dict(name="The time step is small enough", status="ok" if dt_check and dt_check["n_sigma"] <= A["time_step_sigma"] else "standard",
             check=(f"Half the step on the coarsest grid changes the bulk velocity by {m(tf(dt_check['n_sigma'], 1))} combined standard errors. " if dt_check else "")
                   + f"Mean step on {fin}: {m(tf(AV[fin]['rec']['dt'] * ut ** 2 / nu, 2))} viscous time units \\(\\nu/u_\\tau^2\\)."),
    ]

    # ------------------------------------------------------------------ decisions with their numbers filled in
    fit = gpu.check_fit(wall_units=dict(dx=AV[fin]["rec"]["dx"] * ut / nu, dy_wall=num["levels"][fin]["wall_cell_plus"], dz=AV[fin]["rec"]["dz"] * ut / nu))
    t3 = tolc.get("G3") or tolc.get("G1")
    loose3, mid3 = t3["runs"][-1], next(x for x in t3["runs"] if abs(x["tol"] - 1e-8) < 1e-12)
    tokens = {
        "fit": "fits, " + ("no blockers" if fit["ok"] else "; ".join(fit["blockers"])),
        "grid_list": ", ".join(m(rf"{v['n'][0]} \times {v['n'][1]} \times {v['n'][2]}") for v in num["levels"].values()),
        "wall_cells": ", ".join(m(tf(v["wall_cell_plus"], 2)) for v in num["levels"].values()),
        "steps_per_sample": m(str(T["steps_per_sample"]["G3"])), "cfl": m(tf(T["cfl"], 2)), "ptol": report.m_pow10(T["pressure_tol"]),
        "tol_diff_G3": m(tn(loose3["worst"], 2)), "tol_rule": m(tn(t3["rule"], 1)), "tol_div_G3": m(tn(float(loose3["divergence"]), 2)),
        "tol_gain_G3": m_pct(1 - loose3["ms_per_step"] / mid3["ms_per_step"], 0),
        "stop_stderr": m_pct(R["stderr_rel"], 2), "stop_min_time": m(tf(R["min_time"], 0)), "stop_drift": m(tf(R["max_drift"], 0)),
        "stop_balance": m_pct(R["balance"], 1), "stop_gap": m_pct(R["gap_left"], 1),
        "ref_unc_mean": m_pct(unc_mean, 1), "ref_unc_rms": m_pct(unc_rms, 1), "ref_between_bulk": m_pct(budget["codes"], 2),
        "acc_bulk": m_pct(A["bulk_pct"] / 100, 0), "acc_cf": m_pct(A["cf_pct"] / 100, 0), "acc_urms": m_pct(A["urms_peak_pct"] / 100, 0),
        "tmax_G3": m(tf(R["t_max"]["G3"], 0)),
    }
    decisions = load_yaml(HERE / "decisions.yaml")

    def fill(s):
        if isinstance(s, str):
            for k, v in tokens.items():
                s = s.replace(f"[[{k}]]", v)
            if "[[" in s:
                raise ValueError(f"decisions.yaml: unknown token in: {s[:120]}")
        return s
    decisions = [{k: ([fill(x) for x in v] if isinstance(v, list) else fill(v)) for k, v in d_.items()} for d_ in decisions]

    # ------------------------------------------------------------------ the report
    env = report.environment_info() + [["GPU", AV[fin]["rec"]["gpu"]], ["Flow solver", "gpucfd (this repository, gpucfd/), IMEX scheme, mixed-precision multigrid"]]
    results = dict(
        title=problem["title"], date=dt.date.today().isoformat(), verdict=verdict, comparison_type="benchmark",
        summary=dict(headline=headline, answer_columns=["Requested", f"Reference ({refname})", f"This simulation (finest grid, {fin})", "Difference"],
                     answers=answers, findings=findings, limitations=limitations),
        problem=dict(description=problem["description"].strip(), setup=list(problem["setup"]),
                     given=[dict(label=v["label"], symbol=v["symbol"], value=m(tn(v["value"], 10)), unit=v["unit"]) for v in problem["given"].values()],
                     wanted=wanted, reference=problem["reference"]["description"].strip()),
        assumptions=dict(rows=assumptions),
        theory=dict(
            intro=("A turbulent flow has no exact solution: it is chaotic, and only its averages are reproducible. Theory still fixes several things exactly, "
                   "and they are what the simulation is checked against apart from the reference: the wall shear stress is set by the driving force, the total "
                   "shear stress is a straight line, the profile at the wall is known, and the balance of the whole box says how slowly the bulk velocity settles "
                   "and how large the statistical error of its average is. SymPy derives and checks each of them."),
            steps=th["steps"], derived=th["derived"],
            self_checks=[dict(name=k, ok=bool(v)) for k, v in list(th["self_checks"].items()) + list(checks_code.items())]),
        decisions=decisions,
        preflight=dict(
            purpose=("Before the long runs: a short run on the coarsest grid from the artificial start field (does it turn turbulent, is it healthy, what does a step "
                     "cost) and one on the finest grid from the coarse field (cost per time unit), then the check of the pressure tolerance. Two complete runs of "
                     "the coarsest grid followed before the settings were final; each exposed a problem, and both are kept here as part of the test."),
            runs=pf_runs, checks=pf_checks,
            conclusion=("The solver is healthy and the start field turns turbulent within two time units. Three things were changed before the runs of the report: "
                        "the time step is re-chosen every 5 steps with a Courant target of 0.75 (the first complete run passed a Courant number of 1 at a few moments), "
                        "the pressure tolerance was relaxed to the value the same-answer check supports, and the statistical error of the bulk velocity is taken from "
                        "the balance of the box (two complete runs disagreed by five times the plain block estimate). The pass and fail limits on the outputs stayed as set.")),
        simulation=dict(
            model=[
                dict(aspect="Fluid", real="Constant density and viscosity", model="The same; \\(\\nu = 1/180\\) in units of \\(u_\\tau\\delta\\)"),
                dict(aspect="Channel", real="Very long and wide, two smooth plates", model="Periodic box \\(4\\pi\\delta \\times 2\\delta \\times \\tfrac{4}{3}\\pi\\delta\\), no slip on the plates"),
                dict(aspect="Driving", real="Constant mean pressure gradient", model="Constant force per unit mass along \\(x\\); the flow rate is free"),
                dict(aspect="Turbulence", real="Eddies from the channel size down to a few wall units", model="Computed on the grid, no model; eddies smaller than the cells are lost"),
                dict(aspect="Time", real="Statistically steady", model="Run in time from a start field; start-up cut; averages over tens of time units with their statistical error"),
            ],
            meshes=dict(
                intro=("One family of grids, refined together in all three directions; cell sizes in wall units \\(\\nu/u_\\tau\\). Along \\(y\\) the cells grow smoothly "
                       "from the plate to the centre plane (tanh law). The reference's grids are listed for scale."),
                head=["Grid", "Cells", "Total", "\\(\\Delta x^+\\)", "\\(\\Delta z^+\\)", "Wall cell \\(\\Delta y^+\\)", "Largest \\(\\Delta y^+\\)", "Largest ratio of neighbours",
                      "Cell size / finest"],
                rows=mesh_rows),
            results=dict(
                head=["Run", "Averaged over \\(t\\)", "\\(U_b/u_\\tau\\)", "\\(U_c/u_\\tau\\)", "\\(C_f\\)", "\\(Re_b\\)", "Peak \\(u'_{rms}/u_\\tau\\)", "at \\(y^+\\)",
                      "Peak \\(-\\langle u'v'\\rangle/u_\\tau^2\\)"],
                rows=res_rows),
            health=dict(
                head=["Run", "Health", "Steps", "Mean step \\(\\Delta t^+\\)", "Largest Courant number", "Divergence left", "Ended by", "Independent samples", "Drift (std. errors)",
                      "Wall shear minus driving force", "Plates apart (std. errors)", "Total stress off the line"],
                rows=health_rows,
                notes=[
                    f"Each run ended itself when the stop rule held: start-up cut, at least {m(tf(R['min_time'], 0))} time units averaged, statistical error of the bulk velocity below "
                    f"{m_pct(R['stderr_rel'], 2)}, second half of the record within {m(tf(R['max_drift'], 0))} standard errors of the first, mean wall shear within "
                    f"{m_pct(R['balance'], 1)} of the driving force (Figure {{fig:history}}).",
                    "Independent samples: of the short-memory signal from which the error of the bulk velocity is formed (Section 3). Drift: second half minus first half of the "
                    "averaged record, in its own standard errors. Total stress off the line: largest distance of the averaged total shear stress from the exact straight line, "
                    "in \\(u_\\tau^2\\), anywhere in the channel.",
                    "G1h is the coarsest grid continued with half the time step. first_G1 and second_G1 are the two complete runs made before the settings were final "
                    "(Section 5); they are shown with the final start-up cut and error estimate and enter the report only as repeat runs.",
                    f"Runs G2 and G3 started from the last field of the grid below; the start-up cut waits until the bulk velocity's slow approach has decayed to {m_pct(R['gap_left'], 1)}.",
                ])),
        comparison=dict(
            kind=("Benchmark: comparison with the direct numerical simulations of Vreman and Kuerten (2014) of the same flow in the same box, two independent codes "
                  "on far finer grids, averaged over 200 time units. Every difference is judged against limits fixed before the runs and is shown with the statistical "
                  "error of this simulation's average."),
            definitions=[
                dict(name="Difference", latex=r"\Delta q = \frac{q_{sim} - q_{ref}}{q_{ref}},\quad q_{ref} = \frac{q_{S} + q_{FD}}{2}",
                     meaning="Relative difference of a number from the mean of the reference's two codes (S: spectral, FD: finite differences). For profiles: the largest absolute "
                             "difference along the profile, in wall units."),
                dict(name="Statistical error", latex=r"\sigma_{\overline{U_b}} = T_s\,\sigma_{\overline{\xi}},\quad \sigma_{\overline{q}} = \frac{\sigma_{blocks}}{\sqrt{N_{blocks}}}",
                     meaning="One standard error of a time average. For the bulk velocity (and with it the mean velocity everywhere) from the balance of the box; for the other outputs "
                             "from the scatter of block means."),
                dict(name="Grid study", latex=r"q_k = q_0 + C\,h_k^{\,p}\ \ (k = 1, 2, 3)\quad\Rightarrow\quad p,\ q_0",
                     meaning="Three grids with cell sizes \\(h_k\\) give the observed order \\(p\\) and the value \\(q_0\\) extrapolated to zero cell size, when the three values approach a limit from one side."),
                dict(name="Fluctuation", latex=r"u'_{rms} = \sqrt{\langle (u - \langle u\rangle)^2\rangle},\qquad \langle\cdot\rangle:\ x,\ z,\ t,\ \text{both halves}",
                     meaning="About the full mean, as in the reference. The same for \\(v\\) and \\(w\\)."),
            ],
            head=["Output (finest grid)", "Reference", "Simulation", "Difference", "Limit", ""],
            rows=comp_rows, findings=comp_findings),
        insights=insights.load(NAME),
        figures=figures,
        performance=dict(
            intro="Cost of each run on one RTX 4070, one run at a time. Wall time includes sampling the plane averages.",
            head=["Run", "Cells", "Steps", "Time units", "Wall (min)", "ms per step", "µs per cell and step", "Pressure iterations per solve", "Share spent sampling",
                  "Seconds per time unit", "GPU memory (MiB)"],
            rows=perf_rows,
            notes=[f"Wall time of all runs in the table: {total_wall / 60:.0f} min. Not counted: the pre-flight, the tolerance checks and the part of a half-step run that was stopped (a few minutes).",
                   "What sets the cost is the number of time units, not the grid alone: the bulk velocity needs about 15 to 25 time units of start-up and 40 of averaging for an error "
                   "of a quarter of a per cent, and four times the averaging for half of that.",
                   f"The pressure tolerance chosen by the same-answer check made a step {tokens['tol_gain_G3']} cheaper on the finest grid than the tolerance first planned.",
                   "Next to try: a start field for the finer grids whose bulk velocity is already near its settled value (it would save most of the start-up), cells twice as long "
                   "as wide, and several shorter independent runs in place of one long one (the same statistics, and a direct error estimate)."]),
        reproducibility=dict(
            commands=[
                dict(cmd=f"python -m cases.{NAME}.theory", what="Theory self-checks and derived numbers"),
                dict(cmd=f"python -m cases.{NAME}.run preflight", what="Short runs on G1 and G3: health, cost, projection"),
                dict(cmd=f"python -m cases.{NAME}.run tolcheck G1 G3", what="Pressure tolerance, same-answer check"),
                dict(cmd=f"python -m cases.{NAME}.run production G1 G2 G3 G1h", what="The four runs of the table, each ending on the stop rule"),
                dict(cmd=f"python -m cases.{NAME}.analyze", what="Averages, comparison, figures, this report"),
            ],
            files=[dict(path=f"cases/{NAME}/{f_}", what=w) for f_, w in (
                ("problem.yaml", "The problem as given"), ("reference/", "Reference data of Vreman and Kuerten (2014), as published and parsed"),
                ("numerics.yaml", "Grids, time step, stop rule, pass and fail limits, and the list of changes made on the way"),
                ("decisions.yaml", "Every choice"), ("insights.yaml", "Live workflow log"), ("theory.py", "Exact results and derived numbers (SymPy)"),
                ("run.py", "Runs, start fields, stop rule, error of the bulk velocity"), ("stats.py", "Averages, reference, grid study"),
                ("analyze.py", "Rules, figures, report"), ("sketch.py", "The problem sketch"), ("run.sh, stop_runs.sh", "Launch and stop a run from Windows"))],
            run_dirs=[dict(path="~/gpucfd_runs/turbulent_channel_re180/", what="Last velocity and pressure field of each run (outside git)"),
                      dict(path=f"results/{NAME}/runs/", what="Record of each run: <run>.json (settings, health, cost) and <run>.npz (the sampled plane averages)"),
                      dict(path=f"results/{NAME}/", what="preflight.json, tolcheck_G1.json, tolcheck_G3.json, logs/, results.json, report.html")],
            environment=env,
            notes=["One run at a time on the GPU. A turbulent run is chaotic: a repeat gives the same averages within the statistical error, not the same numbers.",
                   "first_G1 and second_G1 were made with earlier settings (numerics.yaml, changes) and cannot be reproduced by the commands above as they stand; their records are kept. "
                   "The same holds for the recorded pre-flight and tolerance checks: they ran with the first plan's Courant target and sampling, and the commands now use the final ones.",
                   "The records keep every 2nd to 4th sampled profile (about 0.025 time units apart) and every sample of the scalar series."]),
        data=dict(levels=levels, draft=draft, fails=fails, warns=warns, passed=passed,
                  bulk={n: dict(mean=a["bulk"]["mean"], stderr=a["bulk"]["stderr"], stderr_batch=a["bulk"]["stderr_batch"], t0=a["t0"], t1=a["t1"],
                                centre=a["centre"]["mean"], centre_err=a["centre"]["stderr"], cf=a["cf"], re_b=a["re_b"], urms_peak=a["urms_peak"],
                                urms_peak_y=a["urms_peak_y"], urms_peak_err=a["urms_peak_err"], shear_peak=a["shear_peak"], imbalance=a["imbalance"],
                                stress_max=a["stress_max"], halves_sigma=a["halves_sigma"], drift=a["stop"]["drift"])
                        for n, a in list(allrun.items()) + [(k, v) for k, v in extra.items() if k in REPEATS]},
                  reference={k: v for k, v in rs.items()}, study=study, study_noise=noise, study_centre=study_c, study_urms=study_u,
                  profile_differences={lv: {q: dict(max_abs=p["max_abs"], signed=p["signed"], y_plus=p["y_plus"]) for q, p in PD[lv].items()} for lv in levels},
                  budget=budget, time_step_check=dt_check,
                  repeat=dict(names=rep["names"], values=list(map(float, rep["values"])), errs=list(map(float, rep["errs"])),
                              batch=list(map(float, rep["batch"])), chi2=rep["chi2"], chi2_batch=rep["chi2_batch"], dof=rep["dof"]) if rep else None,
                  total_wall_s=total_wall),
    )
    out = report.render(results, svgs, OUT / "report.html", case_dir=None)
    print(f"{verdict}{' (DRAFT)' if draft else ''}: U_b {F['bulk']['mean']:.4f} +- {F['bulk']['stderr']:.4f} ({100 * d_bulk:+.2f} %), U_c {F['centre']['mean']:.3f} "
          f"({100 * d_centre:+.2f} %), u_rms peak {F['urms_peak']:.3f} ({100 * d_urms:+.2f} %) on {fin}")
    print("fails:", *fails, sep="\n  ")
    print("warns:", *warns, sep="\n  ")
    print("report:", out)


def resolved_text(fin, F, d_bulk, s_bulk, d_urms, PD, A, budget, study, noise, unc_mean, fails, d_centre, d_urms_y):
    parts = []
    mean_ok = abs(d_bulk) <= A["bulk_pct"] / 100 and abs(d_centre) <= A["centre_pct"] / 100 and PD[fin]["U"]["max_abs"] <= A["profile_abs"]["U"]
    fluct_ok = (abs(d_urms) <= A["urms_peak_pct"] / 100 and abs(d_urms_y) <= A["urms_peak_yplus"]
                and all(PD[fin][q]["max_abs"] <= A["profile_abs"][q] for q in ("u_rms", "v_rms", "w_rms", "uv")))
    mean_txt = (f"bulk velocity {m_pct(d_bulk, 2, signed=True)}, centre velocity {m_pct(d_centre, 2, signed=True)}, profile within "
                f"{m(tf(PD[fin]['U']['max_abs'], 2))} \\(u_\\tau\\) of the reference")
    fl_txt = (f"peak of \\(u'_{{rms}}\\) {m_pct(d_urms, 1, signed=True)} and {m(tf(abs(d_urms_y), 1))} wall units from its place; profiles of \\(u'\\), \\(v'\\), "
              f"\\(w'\\) off by up to {m(tf(PD[fin]['u_rms']['max_abs'], 2))}, {m(tf(PD[fin]['v_rms']['max_abs'], 2))}, {m(tf(PD[fin]['w_rms']['max_abs'], 2))} \\(u_\\tau\\)")
    if mean_ok and fluct_ok:
        parts.append(f"<b>Yes, within the limits set before the runs.</b> Mean flow: {mean_txt}. Fluctuations: {fl_txt}.")
    elif mean_ok:
        parts.append(f"<b>Not fully.</b> The mean flow meets its limits ({mean_txt}), but the fluctuations do not ({fl_txt}): the cells are too large "
                     "for the smallest eddies at the wall.")
    elif fluct_ok:
        parts.append(f"<b>Not fully.</b> The fluctuations meet their limits ({fl_txt}), but the mean flow does not ({mean_txt}).")
    else:
        parts.append(f"<b>No, not within the limits set before the runs.</b> Mean flow: {mean_txt}. Fluctuations: {fl_txt}. The cells are too large for the "
                     "smallest eddies at the wall.")
    grid = (f"grid {m_pct(budget['grid'], 2, signed=True)} (finest grid minus extrapolated value)" if budget["grid"] is not None else
            (f"grid: no error estimate, the last refinement changed the value by {m_pct(budget['grid_last'], 2)}" if budget["grid_last"] is not None else "grid: study not finished"))
    parts.append(f"The difference of the bulk velocity is made of: {grid}; averaging time \\(\\pm\\){m_pct(s_bulk, 2)} (one standard error) and up to "
                 f"{m_pct(budget['settling'], 2)} for not being fully settled; reference \\(\\pm\\){m_pct(unc_mean, 1)}.")
    return " ".join(parts)


# ============================================================================================ figures
COL = {"G1": plots.SERIES[3], "G2": plots.SERIES[1], "G3": plots.SERIES[0], "G1h": plots.SERIES[2]}


def fig_history(AV, extra, levels, g):
    fig, (a1, a2) = plots.wide(2)
    runs = [(n, AV[n]) for n in levels] + ([("G1h", extra["G1h"])] if "G1h" in extra else [])
    for n, a in runs:
        a1.plot(a["t"], a["bulk_series"] / g["u_tau"], color=COL[n], lw=1.1, label=n)
        a1.axvline(a["t0"], color=COL[n], ls=":", lw=1.2)
    a1.set_xlabel(r"time $t\,u_\tau/\delta$ (each run from its own start)")
    a1.set_ylabel(r"bulk velocity $U_b/u_\tau$")
    fin = levels[-1]
    a = AV[fin]
    fd = g["f"] * g["delta"]
    a2.plot(a["t"], a["tau_low_series"] / fd, color=plots.INK2, lw=0.7, label="lower plate")
    a2.plot(a["t"], a["tau_high_series"] / fd, color="#b9a57a", lw=0.7, label="upper plate")
    a2.axhline(1.0, color=plots.INK, lw=1.0)
    a2.axvline(a["t0"], color=plots.INK2, ls=":", lw=1.2)
    a2.set_xlabel(rf"time $t\,u_\tau/\delta$ (run {fin})")
    a2.set_ylabel(r"wall shear stress $\tau_w/(\rho f\delta)$")
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=len(l1) + len(l2), fontsize=9)
    return plots.to_svg(fig, "history")


def fig_grid(AV, levels, h_rel, REF, PD, study, unc_mean):
    fig, (a1, a2) = plots.wide(2)
    rv = REF["scalars"]["bulk"]["value"]
    hs = [h_rel[lv] for lv in levels]
    a1.axhspan(rv * (1 - unc_mean), rv * (1 + unc_mean), color=plots.GRID, zorder=0, label="reference with its uncertainty")
    a1.axhline(rv, color=plots.INK, lw=1.0)
    a1.errorbar(hs, [AV[lv]["bulk"]["mean"] for lv in levels], yerr=[AV[lv]["bulk"]["stderr"] for lv in levels], fmt="o-", color=plots.SERIES[0],
                capsize=3, lw=1.2, label="simulation, one standard error")
    if study and study["reliable"]:
        a1.plot([0], [study["extrap"]], "o", mfc="white", color=plots.SERIES[0], label="extrapolated")
    a1.set_xlim(-0.08, max(hs) * 1.08)
    a1.set_xlabel("cell size relative to the finest grid")
    a1.set_ylabel(r"bulk velocity $U_b/u_\tau$")
    lab = {"U": r"$U$", "u_rms": r"$u'_{rms}$", "v_rms": r"$v'_{rms}$", "w_rms": r"$w'_{rms}$", "uv": r"$-\langle u'v'\rangle$"}
    mk = {"U": "o", "u_rms": "s", "v_rms": "^", "w_rms": "v", "uv": "D"}
    col = {"U": plots.INK, "u_rms": plots.SERIES[0], "v_rms": plots.SERIES[1], "w_rms": plots.SERIES[2], "uv": plots.SERIES[3]}
    for q in lab:
        a2.plot(hs, [PD[lv][q]["max_abs"] for lv in levels], marker=mk[q], color=col[q], lw=1.1, label=lab[q])
    a2.set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a2.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "125" else ""))
    a2.set_xlabel("cell size relative to the finest grid")
    a2.set_ylabel("largest difference from reference")
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=4, fontsize=8)
    return plots.to_svg(fig, "grid")


def fig_profile(AV, levels, REF, PD, A, ut):
    fig, (a1, a2) = plots.wide(2)
    yp = np.geomspace(0.3, 180.0, 300)
    a1.plot(yp, REF["at"]("U", yp), color=plots.INK, lw=1.6, label="reference")
    yv = np.geomspace(0.3, 11.0, 50)
    a1.plot(yv, yv, color=plots.MUTED, ls="--", lw=1.0, label=r"$u^+ = y^+$")
    yl = np.geomspace(12.0, 180.0, 50)
    a1.plot(yl, np.log(yl) / KAPPA + B_LOG, color=plots.MUTED, ls="-.", lw=1.0, label=r"$u^+ = \ln(y^+)/0.41 + 5.2$")
    for lv in levels:
        a = AV[lv]
        a1.plot(a["y_plus"], a["U"]["mean"] / ut, marker="o", ms=3, lw=0.8, color=COL[lv], label=lv)
        a2.plot(PD[lv]["U"]["y"], PD[lv]["U"]["diff"], marker="o", ms=3, lw=0.9, color=COL[lv])
    fin = levels[-1]
    a = AV[fin]
    ok = np.isfinite(REF["at"]("U", a["y_plus"]))
    a2.fill_between(a["y_plus"][ok], PD[fin]["U"]["diff"] - 2 * a["U"]["stderr"][ok] / ut, PD[fin]["U"]["diff"] + 2 * a["U"]["stderr"][ok] / ut,
                    color=COL[fin], alpha=0.18, lw=0)
    for s_ in (-1, 1):
        a2.axhline(s_ * A["profile_abs"]["U"], color=plots.INK2, ls="--", lw=0.9)
    a2.axhline(0, color=plots.INK, lw=0.8)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlabel(r"distance from the plate $y^+$")
    a1.set_ylabel(r"mean velocity $u^+$")
    a2.set_ylabel(r"$u^+$ minus reference")
    fig.legend(*a1.get_legend_handles_labels(), loc="outside lower center", ncol=6, fontsize=9)
    return plots.to_svg(fig, "profile")


def fig_rms(AV, levels, REF, ut, fin):
    fig, (a1, a2) = plots.wide(2)
    yp = np.linspace(0.5, 179.5, 400)
    a = AV[fin]
    comps = (("u_rms", r"$u'_{rms}$", plots.SERIES[0], a["y_plus"], a["u_rms"]["mean"] / ut),
             ("w_rms", r"$w'_{rms}$", plots.SERIES[2], a["y_plus"], a["w_rms"]["mean"] / ut),
             ("v_rms", r"$v'_{rms}$", plots.SERIES[1], a["yf_plus"], a["v_rms"]["mean"] / ut))
    for q, lab, c, x, yv in comps:
        a1.plot(yp, REF["at"](q, yp), color=c, lw=1.5)
        a1.plot(x, yv, "o", ms=3.2, color=c, label=lab + f" ({fin})")
    a1.plot([], [], color=plots.INK2, lw=1.5, label="reference (lines)")
    a1.set_xlabel(r"distance from the plate $y^+$")
    a1.set_ylabel(r"fluctuation intensity (in $u_\tau$)")
    yq = np.linspace(0.5, 60.0, 200)
    a2.plot(yq, REF["at"]("u_rms", yq), color=plots.INK, lw=1.6, label="reference")
    for lv in levels:
        b = AV[lv]
        k = b["y_plus"] <= 60
        a2.plot(b["y_plus"][k], b["u_rms"]["mean"][k] / ut, marker="o", ms=3, lw=0.9, color=COL[lv], label=lv)
    a2.set_xlabel(r"distance from the plate $y^+$")
    a2.set_ylabel(r"$u'_{rms}/u_\tau$")
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=4, fontsize=9)
    return plots.to_svg(fig, "rms")


def fig_stress(AV, levels, REF, fin, g):
    fig, (a1, a2) = plots.wide(2)
    a = AV[fin]
    fd = g["f"] * g["delta"]
    x = a["yf"] / g["delta"]
    st = a["stress"]
    yp = np.linspace(0.5, 179.9, 300)
    a1.plot(yp * g["nu"] / (g["u_tau"] * g["delta"]), REF["at"]("uv", yp), color=plots.INK, lw=1.5, label=r"reference $-\langle u'v'\rangle$")
    a1.plot(x, st["turbulent"] / fd, "o", ms=3.2, color=plots.SERIES[0], label="carried by the eddies")
    a1.plot(x, st["viscous"] / fd, "s", ms=3.2, color=plots.SERIES[1], label="viscous")
    a1.plot(x, st["total"] / fd, "^", ms=3.2, color=plots.SERIES[2], label="sum")
    a1.plot(x, st["exact"] / fd, color=plots.MUTED, ls="--", lw=1.2, label="exact line")
    a1.set_xlabel(r"distance from the plate $y/\delta$")
    a1.set_ylabel(r"shear stress (in $\rho u_\tau^2$)")
    for lv in levels:
        b = AV[lv]
        a2.plot(np.array(b["rec"]["y_faces"]) / g["delta"], b["stress_full_residual"] / fd, lw=1.1, color=COL[lv], label=f"{lv}: sum minus exact line")
    a2.axhline(0, color=plots.INK, lw=0.8)
    a2.set_xlabel(r"position across the channel $y/\delta$")
    a2.set_ylabel(r"total stress minus exact line (in $\rho u_\tau^2$)")
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=4, fontsize=8)
    return plots.to_svg(fig, "stress")


if __name__ == "__main__":
    main()
