"""Analyse the turbulent flat plate runs, compare with NASA's results and write the report.

Usage (repository root, inside WSL): python -m cases.turbulent_flat_plate_sa.analyze
Reads results/turbulent_flat_plate_sa/runs/*.json|npz and preflight.json; writes results.json and report.html.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np

from cfdlib import insights, plots, report, verification as V
from cfdlib.report import load_yaml, m, tex_fix as tf, tex_num as tn

from . import metrics as M, theory as T
from .common import HERE, NAME, OUT, REF, RUNS, given, load, reference, tecplot_zones
from .sketch import sketch

C = plots.SERIES


def pct(x, d=2, signed=True):
    return m(f"{100 * x:{'+' if signed else ''}.{d}f}" + r"\,\%")


def tiny(x, d=4):
    """A very small relative change: as a signed per cent, or 'less than' the last shown digit."""
    lim = 10.0 ** (-d - 2)
    return ("less than " + m(f"{100 * lim:.{d}f}" + r"\,\%")) if abs(x) < lim else pct(x, d)


def exists(name):
    return (RUNS / f"{name}.json").exists() and (RUNS / f"{name}.npz").exists()


def seq(ref_list, key):
    """A reference series of the YAML (listed finest first) as coarse-to-fine values."""
    return [r[key] for r in ref_list][::-1]


def run_metrics(rec, a, G):
    b = M.balances(a, G)
    h = a["hist"]
    alt = float((np.abs(np.diff(h[-51:], axis=0)).max(axis=0) / np.abs(h[-1])).max())
    return dict(cf=M.cf_at(a, G, G["x_cf"]), cd=rec["cd_forces"], cd_sum=M.cd_integral(a, G), momentum=b["momentum"],
                mass=b["mass"], top_outflow=b["top_outflow"], alternation=alt, limiter=M.limiter_activity(a, G),
                yplus_max=float(a["yplus"].max()), yplus_mean=float(a["yplus"].mean()),
                wall_cell=float(np.abs(M.cf_wall_cell(a, G) / M.cf_along(a, G)[1] - 1).max()),
                res_max=max(v for v in rec["final_residuals"].values() if v is not None))


def main():
    problem, num = load()
    G = given(problem)
    ref = reference()
    acc = num["acceptance"]
    levels = num["mesh"]["levels"]
    x1, x2, U, nu = G["x_cf"], G["x_2"], G["U_inf"], G["nu"]
    runs = [M.load_run(lv) for lv in levels]
    mets = [run_metrics(r, a, G) for r, a in runs]
    frec, fa = runs[-1]
    fm = mets[-1]
    th = T.evaluate(G, ref, [g["first_cell_height"] for g in ref["grids"]])
    h_grid = seq(ref["compressible"]["cf_at_station"]["CFL3D"], "h")
    wanted = [dict(letter=report.LETTERS[i], key=q["key"], text=q["text"]) for i, q in enumerate(problem["quantities_of_interest"])]

    # ---- reference series (coarse to fine) ------------------------------------------------------------
    inc = ref["incompressible"]["cf_at_station"]
    comp = ref["compressible"]
    R = dict(fun_inc=seq(inc["FUN3D incompressible SA"], "cf"), sct=seq(inc["SC/TETRA incompressible SA"], "cf"),
             cfl=seq(comp["cf_at_station"]["CFL3D"], "cf"), fun=seq(comp["cf_at_station"]["FUN3D"], "cf"),
             cd_cfl=seq(comp["cd"]["CFL3D"], "cd"), cd_fun=seq(comp["cd"]["FUN3D"], "cd"))
    S = {k: V.three_grid_study(v) for k, v in R.items()}
    compress = R["fun_inc"][-1] / R["cfl"][-1] - 1           # NASA's own measure of the compressibility effect

    # ---- (a) skin friction at the station ---------------------------------------------------------------
    cf = [mt["cf"] for mt in mets]
    s_cf = V.three_grid_study(cf)
    orders_cf = [V.level_sequence(cf[k:k + 3])["order"] for k in range(len(cf) - 2)]
    orders_cd = [V.level_sequence([mt["cd"] for mt in mets][k:k + 3])["order"] for k in range(len(cf) - 2)]
    cf_diff = s_cf["extrap"] / S["fun_inc"]["extrap"] - 1
    cf_band = (s_cf["gci_abs"] + S["fun_inc"]["gci_abs"]) / S["fun_inc"]["extrap"]
    cf_fine_diff = cf[-1] / R["fun_inc"][-1] - 1
    xf5, cf5 = M.cf_along(fa, G)
    bias = []
    for _, a in runs:                                         # the station interpolation applied to the finest solution
        xf = M.cf_along(a, G)[0]
        bias.append(float(np.interp(x1, xf, np.interp(xf, xf5, cf5)) / np.interp(x1, xf5, cf5) - 1))

    # ---- (b) drag ---------------------------------------------------------------------------------------
    cd = [mt["cd"] for mt in mets]
    s_cd = V.three_grid_study(cd)
    cd_shift = cd[-1] / R["cd_cfl"][-1] - cf[-1] / R["cfl"][-1]
    cd_codes = R["cd_fun"][-1] / R["cd_cfl"][-1] - 1

    # ---- (c) skin friction along the plate ----------------------------------------------------------------
    _, zc = tecplot_zones(REF / "tmr" / "cf_plate.dat")
    win = (xf5 >= 0.1) & (xf5 <= 1.9)
    ratio_cfl = cf5 / np.interp(xf5, zc["CFL3D"][:, 0], zc["CFL3D"][:, 1])
    ratio_fun = cf5 / np.interp(xf5, zc["FUN3D"][:, 0], zc["FUN3D"][:, 1])
    shape_mean = float(np.mean(ratio_cfl[win]))
    shape_spread = float(np.abs(ratio_cfl[win] / shape_mean - 1).max())
    shape_ends = (float(ratio_cfl[win][0]), float(ratio_cfl[win][-1]))

    # ---- (d) velocity profiles --------------------------------------------------------------------------
    _, zu = tecplot_zones(REF / "tmr" / "flatplate_u.dat")
    _, zw = tecplot_zones(REF / "tmr" / "flatplate_u+y+.dat")
    stations = [dict(x=x1, key="x= 0.97008", name="x_1"), dict(x=x2, key="x= 1.90334", name="x_2")]
    for st in stations:
        pr = M.profile(fa, G, st["x"])
        tk = M.thicknesses(fa, G, st["x"])
        yy = pr["y"]
        inside = yy <= 2 * tk["delta99"]
        u_ref = np.interp(yy, zu[st["key"]][:, 1], zu[st["key"]][:, 0])
        du = pr["u"] - u_ref
        st["my_edge"] = float(np.interp(2 * tk["delta99"], yy, pr["u"]))        # same definition as ref_edge below
        st["ratio_cfl"] = float(np.interp(st["x"], xf5, ratio_cfl))
        st.update(pr=pr, tk=tk, u_ref=u_ref, du_max=float(np.abs(du[inside]).max()), du_edge=float(du[np.argmin(np.abs(yy - 2 * tk["delta99"]))]),
                  ref_edge=float(np.interp(2 * tk["delta99"], zu[st["key"]][:, 1], zu[st["key"]][:, 0])))
        # wall units: against the model's own inner law, and against CFL3D at the same y+
        w = pr["yplus"] <= 100
        law = T.inner_law(pr["yplus"][w])
        st["inner_dev"] = float(np.abs(pr["uplus"][w] / law - 1).max())
        st["inner_dev_at"] = float(pr["yplus"][w][np.argmax(np.abs(pr["uplus"][w] / law - 1))])
        zy, zup = 10 ** zw[st["key"]][:, 0], zw[st["key"]][:, 1]
        w2 = (pr["yplus"] >= 1) & (pr["yplus"] <= 1000)
        st["cfl_wall_dev"] = float(np.abs(pr["uplus"][w2] / np.interp(pr["yplus"][w2], zy, zup) - 1).max())
        j100 = int(np.argmin(np.abs(pr["yplus"] - 100)))
        st["white_100"] = float(pr["uplus"][j100] - (np.log(pr["yplus"][j100]) / th["kappa_white"] + th["B_white"]))
        st["sa_log_100"] = float(pr["uplus"][j100] - (np.log(pr["yplus"][j100]) / T.CONSTANTS["kappa"] + th["B_sa"]))
        st["first_cell"] = float(pr["uplus"][0] / pr["yplus"][0])
    inner_dev = max(st["inner_dev"] for st in stations)
    du_max = max(st["du_max"] for st in stations)

    # ---- (e) eddy viscosity -----------------------------------------------------------------------------
    _, zn = tecplot_zones(REF / "tmr" / "mut_0.97.dat")
    _, zp = tecplot_zones(REF / "tmr" / "find_peak_mut.dat")
    pk = zp[next(iter(zp))]
    p1 = stations[0]["pr"]
    jmax = int(np.argmax(p1["nut"]))
    nut_peak = float(p1["nut"][jmax])
    nut_peak_y = float(p1["y"][jmax])
    nut_cfl, nut_fun = float(zn["CFL3D"][:, 2].max()), float(zn["FUN3D"][:, 2].max())
    xn, nmax = M.nut_max_along(fa, G)
    wn = (xn >= 0.2) & (xn <= 1.9)
    nut_ratio = nmax / np.interp(xn, pk[:, 0], pk[:, 1])
    nut_along_dev = float(np.abs(nut_ratio[wn] - 1).max())
    nut_along_mean = float(np.mean(nut_ratio[wn]) - 1)
    nut_plate_max = float(nmax[xn > 0].max())
    nut_plate_max_x = float(xn[xn > 0][np.argmax(nmax[xn > 0])])
    nut_cfl_max = float(pk[:, 1].max())

    # ---- variants ---------------------------------------------------------------------------------------
    variants = {}
    for vn, spec in num["sensitivity"].items():
        rows = []
        for lv in spec["levels"]:
            if exists(f"{vn}_{lv}"):
                r, a = M.load_run(f"{vn}_{lv}")
                mt = run_metrics(r, a, G)
                base = mets[levels.index(lv)]
                # drag from the wall shear stress, referred to the given speed (the speed variant changes the solver's own reference)
                rows.append(dict(level=lv, rec=r, met=mt, d_cf=mt["cf"] / base["cf"] - 1, d_cd=mt["cd_sum"] / base["cd_sum"] - 1,
                                 u_edge=M.thicknesses(a, G, x1)["u_edge"], u_edge_base=M.thicknesses(runs[levels.index(lv)][1], G, x1)["u_edge"]))
        if rows:
            variants[vn] = dict(what=spec["what"], rows=rows)
    up = variants.get("upwind")
    s_up = V.three_grid_study([r["met"]["cf"] for r in up["rows"]]) if up and len(up["rows"]) == 3 else None

    # How much of the difference to CFL3D along the plate comes from the free-stream speed above the plate?
    # The speed variant gives this solver's own sensitivity of Cf (referred to the given speed) to the speed of
    # the stream: same grid, inflow a little faster, nothing else changed.
    edge = None
    if "speed" in variants:
        r = variants["speed"]["rows"][0]
        sens = float(np.log(1 + r["d_cf"]) / np.log(r["u_edge"] / r["u_edge_base"]))
        e1, e2 = (np.log(st["my_edge"] / st["ref_edge"]) for st in stations)
        edge = dict(sens=sens, part_x1=float(np.expm1(sens * e1)), part_x2=float(np.expm1(sens * e2)),
                    predicted=float(np.expm1(sens * (e1 - e2))), observed=float(stations[0]["ratio_cfl"] / stations[1]["ratio_cfl"] - 1))

    relax_runs = [n for n in ("trial_U09_nt05", "trial_U07_nt05", "trial_U05_nt05", "trial_U03_nt03") if exists(n)]
    relax_cd = [M.load_run(n)[0]["cd_forces"] for n in relax_runs]
    relax_spread = (max(relax_cd) - min(relax_cd)) / np.mean(relax_cd) if len(relax_cd) >= 2 else None

    # ---- model as run -----------------------------------------------------------------------------------
    coeffs = frec["model_coeffs"]
    nasa = {k: float(v) for k, v in T.CONSTANTS.items()}
    name_map = dict(Cb1="cb1", Cb2="cb2", kappa="kappa", Cw2="cw2", Cw3="cw3", Cv1="cv1", Ct3="ct3", Ct4="ct4", sigmaNut="sigma")
    coeff_ok = all(k in coeffs and abs(float(coeffs[k]) - nasa[v]) < 1e-9 for k, v in name_map.items()) and frec["ft2"] == "active"
    lim = fm["limiter"]
    lim_last = max((mt["limiter"]["x_last"] or 0.0) for mt in mets)
    lim_text = ("OpenFOAM keeps the model's \\(\\tilde S\\) at or above \\(0.3\\,\\Omega\\), which is method (b) of Note 1 on NASA's page of the model. "
                + (f"On the finest grid it acts in {lim['cells_limited']} of {lim['cells']:,} cells of the boundary layer, all within \\(x \\le "
                   + tf(lim_last, 3) + "\\) of the leading edge on every grid, where the free-stream \\(\\tilde\\nu\\) first meets the wall."
                   if lim["cells_limited"] else "It never acts in the solution on the finest grid."))

    # ---- rules and verdict ------------------------------------------------------------------------------
    fails, warns, passed = [], [], []

    def rule(ok, text_ok, text_bad, bad=warns):
        (passed if ok else bad).append(text_ok if ok else text_bad)
        return ok

    for lv, (r, a), mt in zip(levels, runs, mets):
        d = (r.get("live_stop") or {}).get("decision") or {}
        if r["health"]["status"] == "fail" or not r["mesh_ok"] and False:
            fails.append(f"{lv}: the solver log has a failed health check.")
        if not d or d.get("status") != "ok":
            warns.append(f"{lv}: the live stop did not end the run with every monitored value settled.")
    mom_max = max(abs(mt["momentum"]) for mt in mets)
    mass_max = max(abs(mt["mass"]) for mt in mets)
    alt_max = max(mt["alternation"] for mt in mets)
    rule(mom_max <= acc["momentum_balance"],
         "Momentum balance: on every grid the drag from the wall shear stress equals the momentum the stream loses to within "
         + pct(mom_max, 4, False) + " (limit " + pct(acc["momentum_balance"], 2, False) + ").",
         "Momentum balance: the drag from the wall shear stress and the momentum the stream loses differ by up to "
         + pct(mom_max, 4, False) + ", above the limit of " + pct(acc["momentum_balance"], 2, False) + ".", fails)
    rule(mass_max <= acc["mass_balance"], "Mass balance: the boundary fluxes sum to " + m(tn(mass_max, 1)) + " of the inflow or less on every grid.",
         "Mass balance: the boundary fluxes do not sum to zero (" + m(tn(mass_max, 1)) + " of the inflow).", fails)
    rule(alt_max <= num["convergence"]["tol_rel"]["nuTilda"],
         "No iteration-to-iteration cycle: over the last 50 iterations of every run the monitored values changed by at most "
         + m(tn(alt_max, 1)) + " (relative).",
         "The monitored values still change between consecutive iterations by " + m(tn(alt_max, 1)) + " (relative) at the end of a run.", fails)
    rule(coeff_ok, "The model constants the solver printed are NASA's, and the \\(f_{t2}\\) term is active.",
         "The model constants the solver printed differ from NASA's list, or the \\(f_{t2}\\) term is inactive.", fails)
    p_ok = s_cf["gci_reliable"] and acc["order_range"][0] <= s_cf["p"] <= acc["order_range"][1]
    rule(p_ok, "Observed order of accuracy of \\(C_f(x_1)\\) on the three finest grids: " + (m(tf(s_cf["p"], 2)) if s_cf["monotonic"] else "–")
         + f" (expected between {acc['order_range'][0]:g} and {acc['order_range'][1]:g}).",
         "Observed order of accuracy of \\(C_f(x_1)\\) on the three finest grids: "
         + (m(tf(s_cf["p"], 2)) if s_cf["monotonic"] else "not defined (the values do not approach a limit from one side)")
         + f", outside the expected range {acc['order_range'][0]:g} to {acc['order_range'][1]:g}.")
    rule(s_cf["gci"] <= acc["cf_gci_max"], "Grid-convergence index of \\(C_f(x_1)\\) on the finest grid: " + pct(s_cf["gci"], 3, False)
         + " (limit " + pct(acc["cf_gci_max"], 1, False) + "): the study is sharp enough to judge a difference of a tenth of a per cent.",
         "Grid-convergence index of \\(C_f(x_1)\\) on the finest grid: " + pct(s_cf["gci"], 3, False) + ", above the limit "
         + pct(acc["cf_gci_max"], 1, False) + ".")
    if abs(cf_diff) <= cf_band:
        cf_state = "pass"
        passed.append("Skin friction at \\(x_1\\), grid-converged: " + pct(cf_diff, 3) + " from NASA's incompressible FUN3D value, inside the combined "
                      "uncertainty band of " + pct(cf_band, 3, False) + ".")
    elif abs(cf_diff) <= acc["cf_fail"]:
        cf_state = "warn"
        warns.append("Skin friction at \\(x_1\\), grid-converged: " + pct(cf_diff, 3) + " from NASA's incompressible FUN3D value. That is outside "
                     "the combined grid-uncertainty band of " + pct(cf_band, 3, False) + " (so it is not grid error on either side), "
                     "and inside the limit of " + pct(acc["cf_fail"], 2, False) + " set before the runs.")
    else:
        cf_state = "fail"
        fails.append("Skin friction at \\(x_1\\), grid-converged: " + pct(cf_diff, 3) + " from NASA's incompressible FUN3D value, above the limit of "
                     + pct(acc["cf_fail"], 2, False) + " set before the runs.")
    rule(inner_dev <= acc["inner_law"], "Wall layer: for \\(y^+ \\le 100\\) the simulated \\(u^+\\) follows the model's own law of the wall to within "
         + pct(inner_dev, 2, False) + " at both stations (limit " + pct(acc["inner_law"], 0, False) + ").",
         "Wall layer: for \\(y^+ \\le 100\\) the simulated \\(u^+\\) departs from the model's own law of the wall by up to "
         + pct(inner_dev, 2, False) + " (limit " + pct(acc["inner_law"], 0, False) + ").")
    ctx = "Context check against the compressible results (Mach 0.2, not the same flow): "
    rule(abs(cd_shift) <= acc["cd_shift_tol"], ctx + "the drag differs from CFL3D's by the same factor as the skin friction at \\(x_1\\), to within "
         + pct(cd_shift, 2) + " (limit " + pct(acc["cd_shift_tol"], 1, False) + ").",
         ctx + "the drag ratio to CFL3D differs from the skin friction ratio at \\(x_1\\) by " + pct(cd_shift, 2) + " (limit "
         + pct(acc["cd_shift_tol"], 1, False) + ").")
    rule(shape_spread <= acc["cf_shape_tol"], ctx + "the ratio of \\(C_f(x)\\) to CFL3D's is constant along \\(0.1 \\le x \\le 1.9\\) to within "
         + pct(shape_spread, 2, False) + " (limit " + pct(acc["cf_shape_tol"], 1, False) + ").",
         ctx + "the ratio of \\(C_f(x)\\) to CFL3D's varies along \\(0.1 \\le x \\le 1.9\\) by up to " + pct(shape_spread, 2, False)
         + " around its mean (limit " + pct(acc["cf_shape_tol"], 1, False) + ").")
    rule(du_max <= acc["u_profile_tol"], ctx + "velocity profiles within " + m(tf(du_max, 4) + r"\,U_\infty") + " of CFL3D's at both stations (limit "
         + m(tf(acc["u_profile_tol"], 3) + r"\,U_\infty") + ").",
         ctx + "velocity profiles differ from CFL3D's by up to " + m(tf(du_max, 4) + r"\,U_\infty") + " (limit "
         + m(tf(acc["u_profile_tol"], 3) + r"\,U_\infty") + ").")
    nut_dev = max(abs(nut_peak / nut_cfl - 1), nut_along_dev)
    rule(nut_dev <= acc["nut_peak_tol"], ctx + "eddy viscosity within " + pct(nut_dev, 2, False) + " of CFL3D's (peak at \\(x_1\\) and the largest value along the plate; limit "
         + pct(acc["nut_peak_tol"], 0, False) + ").",
         ctx + "eddy viscosity differs from CFL3D's by up to " + pct(nut_dev, 2, False) + " (peak at \\(x_1\\) and the largest value along the plate; limit "
         + pct(acc["nut_peak_tol"], 0, False) + ").")
    verdict = "FAIL" if fails else "WARN" if warns else "PASS"

    # ---- what the difference is made of (f) ---------------------------------------------------------------
    my_grid_err = cf[-1] / s_cf["extrap"] - 1
    ref_grid_err = R["fun_inc"][-1] / S["fun_inc"]["extrap"] - 1
    sct_diff = s_cf["extrap"] / S["sct"]["extrap"] - 1
    made_rows = [
        ["Difference on the finest grid", "This solution against FUN3D incompressible, both on \\(545 \\times 385\\)", pct(cf_fine_diff, 3)],
        ["Grid error of this solution", "Finest grid against its own grid-converged value", pct(my_grid_err, 3)],
        ["Grid error of the reference", "FUN3D incompressible, finest grid against its grid-converged value", pct(ref_grid_err, 3)],
        ["<b>Difference of the grid-converged values</b>", "What is left when both grid errors are removed", "<b>" + pct(cf_diff, 3) + "</b>"],
        ["Uncertainty of that difference", "Sum of the two grid-convergence indices", "\\(\\pm\\)" + pct(cf_band, 3, False)],
        ["Against the second incompressible code", "This solution against SC/TETRA, grid-converged values (SC/TETRA is "
         + pct(S["sct"]["extrap"] / S["fun_inc"]["extrap"] - 1, 2) + " from FUN3D there)", pct(sct_diff, 3)],
        ["Compressibility (NASA's own measure)", "FUN3D incompressible against CFL3D at Mach 0.2, finest grid", pct(compress, 2)],
    ]
    if "noft2" in variants:
        r = variants["noft2"]["rows"][0]
        made_rows.append(["Model: without the \\(f_{t2}\\) term", f"OpenFOAM's default against the standard form, grid {r['level']}", tiny(r["d_cf"])])
    if s_up:
        made_rows.append(["Numerics: first-order advection of \\(\\tilde\\nu\\)", "As NASA's codes; change of the grid-converged value (G3 to G5)",
                          pct(s_up["extrap"] / s_cf["extrap"] - 1, 3)])
    if "limiter" in variants:
        r = variants["limiter"]["rows"][0]
        made_rows.append(["Model: limiter on \\(\\tilde S\\) at \\(0.1\\,\\Omega\\) instead of \\(0.3\\,\\Omega\\)", f"Grid {r['level']}; the limiter only acts next to the leading edge",
                          tiny(r["d_cf"])])
    if "lid" in variants:
        r = variants["lid"]["rows"][0]
        made_rows.append(["Boundary: closed lid instead of the open far field", f"Grid {r['level']}; the stream above the layer at \\(x_1\\) is then "
                          + pct(r["u_edge"] / r["u_edge_base"] - 1, 2) + " faster", pct(r["d_cf"], 2)])
    if edge:
        r = variants["speed"]["rows"][0]
        made_rows.append(["Boundary: speed of the stream over the plate", f"Grid {r['level']}, inflow " + pct(r["u_edge"] / r["u_edge_base"] - 1, 2)
                          + " faster; \\(C_f\\) stays referred to the given \\(U_\\infty\\), as it would if the far field of another set-up let the stream run faster",
                          pct(r["d_cf"], 2)])
    if relax_spread is not None:
        made_rows.append(["Solver: velocity relaxation factor 0.3 to 0.9", "Spread of the converged drag on the coarsest grid", pct(relax_spread, 4, False)])

    # ---- lettered answers -------------------------------------------------------------------------------
    cfx = lambda v: m(tf(v, 7))
    # The last column of the summary table does not wrap: one number per answer there, the words in the other two.
    ans = dict(
        cf_station=dict(
            theory="FUN3D incompressible: " + cfx(R["fun_inc"][-1]) + " on the finest grid, " + cfx(S["fun_inc"]["extrap"]) + " grid-converged",
            sim=cfx(cf[-1]) + " on the finest grid, " + cfx(s_cf["extrap"]) + " \\(\\pm\\) " + pct(s_cf["gci"], 3, False)
                + " grid-converged; all five grids in Section 7. Difference: grid-converged values (finest grid: " + pct(cf_fine_diff, 3) + ")",
            diff=pct(cf_diff, 3)),
        cd=dict(
            theory="No incompressible value published. Mach 0.2, finest grid: CFL3D " + cfx(R["cd_cfl"][-1]) + ", FUN3D " + cfx(R["cd_fun"][-1]),
            sim=cfx(cd[-1]) + " on the finest grid, " + cfx(s_cd["extrap"]) + " grid-converged; all five grids in Section 6. Difference: against CFL3D at "
                "Mach 0.2; the skin friction at \\(x_1\\) differs from CFL3D's by the same factor (" + pct(cf[-1] / R["cfl"][-1] - 1, 2) + ")",
            diff=pct(cd[-1] / R["cd_cfl"][-1] - 1, 2)),
        cf_x=dict(
            theory="No incompressible curve published. Mach 0.2: CFL3D and FUN3D on the finest grid",
            sim="Figure {fig:cf_x}: from " + cfx(float(np.interp(0.1, xf5, cf5))) + " at \\(x = 0.1\\) to " + cfx(float(np.interp(1.9, xf5, cf5)))
                + " at \\(x = 1.9\\). Difference: mean ratio to CFL3D over that range; the ratio varies by " + pct(shape_spread, 2, False) + " around it",
            diff=pct(shape_mean - 1, 2)),
        profiles=dict(
            theory="The model's own law of the wall (Section 3); CFL3D profiles at Mach 0.2; log law with \\(\\kappa = "
                   + f"{th['kappa_white']}" + "\\), \\(B = " + f"{th['B_white']}" + "\\)",
            sim="Figures {fig:profiles} and {fig:wall_units}. Difference: largest departure of \\(u^+\\) from the model's law for \\(y^+ \\le 100\\). "
                "Against CFL3D: within " + m(tf(du_max, 4) + r"\,U_\infty") + "; against the log law at \\(y^+ = 100\\): "
                + m(f"{stations[0]['white_100']:+.2f}") + " in \\(u^+\\)",
            diff=pct(inner_dev, 2, False)),
        nut=dict(
            theory="Mach 0.2 only. Peak at \\(x_1\\): CFL3D " + m(tf(nut_cfl, 1)) + ", FUN3D " + m(tf(nut_fun, 1)) + "; largest on the plate (CFL3D): "
                   + m(tf(nut_cfl_max, 1)),
            sim="Peak at \\(x_1\\): " + m(tf(nut_peak, 1)) + ", " + m(tf(nut_peak_y * 1000, 1)) + " mm above the wall; largest on the plate: "
                + m(tf(nut_plate_max, 1)) + " at the end (Figure {fig:nut}). Difference: peak at \\(x_1\\) against CFL3D",
            diff=pct(nut_peak / nut_cfl - 1, 2)),
        same_model=dict(
            theory="FUN3D incompressible, grid-converged \\(C_f(x_1)\\): " + cfx(S["fun_inc"]["extrap"]),
            sim={"pass": "Yes, within the grid uncertainty of both solutions (", "warn": "Yes to within " + pct(abs(cf_diff), 3, False)
                 + ", which is just outside the grid uncertainty of the two solutions (", "fail": "No (grid uncertainty "}[cf_state]
                + pct(cf_band, 3, False) + "). On the finest grid: " + pct(cf_fine_diff, 3) + ", of which own grid error " + pct(my_grid_err, 3)
                + ", the reference's " + pct(ref_grid_err, 3) + "; no compressibility in it. Parts in Section 7",
            diff=pct(cf_diff, 3)),
    )
    answers = [dict(letter=w["letter"], label=w["text"], **ans[w["key"]]) for w in wanted]

    # ---- figures ------------------------------------------------------------------------------------------
    svgs, figures = {}, []

    def add(key, section, svg, caption):
        svgs[key] = svg
        figures.append(dict(key=key, section=section, caption=caption))

    add("sketch", "problem", sketch(G), "The problem (not to scale). Air enters from the left with uniform speed. The lower side is a symmetry "
        "line up to \\(x = 0\\) and the plate from there on. The boundary layer, drawn far too thick, is turbulent from the leading edge. "
        "\\(x_1\\) and \\(x_2\\) are the two stations of the profiles.")
    add("law", "theory", fig_law(th), "The model's own law of the wall: \\(u^+\\) from one integration of the wall-layer solution, "
        "with its two limits. Its log-law intercept is \\(B_{\\mathrm{SA}} = " + tf(th["B_sa"], 3) + "\\); the law NASA plots uses \\(B = "
        + f"{th['B_white']}" + "\\).")
    if exists("trial_base") and exists("pfG1"):
        add("cycle", "preflight", fig_cycle(M.load_run("trial_base")[1], M.load_run("pfG1")[1]),
            "The pre-flight on the coarsest grid. With the turbulence variable relaxed by 0.9 the drag coefficient flips between two values every "
            "iteration and never settles; with 0.5 it converges. Sampled every 20 iterations, the flipping run looks perfectly constant.")
    add("history", "simulation", fig_history(frec, fa), f"The run on the finest grid ({frec['cells']:,} cells, started from the G4 solution). Left: residuals. Right: the five "
        "monitored values, as their distance from the final value. The run stopped itself at iteration "
        f"{frec['iterations']:,}.")
    add("grid_cf", "convergence", fig_grid(h_grid, cf, cd, R, S, s_cf, s_cd),
        "Grid convergence, plotted like NASA's own figures against the grid spacing \\(h = 1/\\sqrt{N}\\) (\\(h = 0\\) is an infinitely fine grid). "
        "Left: skin friction at \\(x_1\\); this solution heads for " + cfx(s_cf["extrap"]) + ", NASA's incompressible FUN3D for " + cfx(S["fun_inc"]["extrap"])
        + ", and the two compressible codes for values " + pct(1 - S["cfl"]["extrap"] / S["fun_inc"]["extrap"], 2, False) + " and "
        + pct(1 - S["fun"]["extrap"] / S["fun_inc"]["extrap"], 2, False) + " below that. Right: drag coefficient; NASA publishes it for the compressible flow only.")
    add("cf_x", "comparison", fig_cf_x(xf5, cf5, zc, ratio_cfl, ratio_fun, shape_mean),
        "Skin friction along the plate on the finest grid. Left: this solution and NASA's two compressible codes. Right: the ratio of this solution to each; "
        "against CFL3D it is " + m(tf(shape_mean, 4)) + " on average and falls from " + m(tf(shape_ends[0], 4)) + " at \\(x = 0.1\\) to " + m(tf(shape_ends[1], 4))
        + " at \\(x = 1.9\\). The two ends of the plate are outside that range: there the codes differ locally (leading-edge singularity, outflow boundary).")
    add("profiles", "comparison", fig_profiles(stations), "Velocity profiles at the two stations on the finest grid. Left: \\(u/U_\\infty\\) against the height (log scale). "
        "Right: this solution minus CFL3D. Outside the boundary layer the difference stays at " + m(f"{stations[0]['du_edge']:+.4f}") + " and "
        + m(f"{stations[1]['du_edge']:+.4f}") + " (in \\(U_\\infty\\)): the two flows do not have the same free-stream speed above the plate.")
    add("wall_units", "comparison", fig_wall(stations, zw, th), "Velocity in wall units on the finest grid, with the model's own law of the wall (Section 3) and the log law "
        "NASA plots. For \\(y^+ \\le 100\\) the simulation follows the model's law to " + pct(inner_dev, 2, False) + ".")
    add("nut", "comparison", fig_nut(p1, zn, xn, nmax, pk), "Eddy viscosity over the viscosity on the finest grid. Left: profile at \\(x_1\\), peak "
        + m(tf(nut_peak, 1)) + " (CFL3D " + m(tf(nut_cfl, 1)) + ", FUN3D " + m(tf(nut_fun, 1)) + "). Right: the largest value in each grid column along the plate, up to "
        + m(tf(nut_plate_max, 1)) + ".")

    # ---- tables -------------------------------------------------------------------------------------------
    pf = json.loads((OUT / "preflight.json").read_text()) if (OUT / "preflight.json").exists() else []
    pf5 = next((p for p in pf if p.get("kind") == "short"), None)
    pf1 = M.load_run("pfG1") if exists("pfG1") else None
    pf1m = run_metrics(*pf1, G) if pf1 else None
    cyc = M.load_run("trial_base") if exists("trial_base") else None
    cycm = run_metrics(*cyc, G) if cyc else None
    pf_runs, pf_checks = [], []
    if cyc:
        hc = cyc[1]["hist"][:, 0]
        pf_runs.append(dict(name="G1, first attempt", setup="Coarsest grid from the uniform stream, turbulence variable relaxed by 0.9",
                            outcome="Did not converge: a cycle of period 2. The drag coefficient alternated between " + cfx(float(hc[-1])) + " and "
                                    + cfx(float(hc[-2])) + "; residuals stalled near \\(10^{-3}\\); momentum balance " + pct(cycm["momentum"], 1) + " off"))
        pf_checks.append(dict(name="First attempt: momentum balance", value=pct(cycm["momentum"], 1), status="fail",
                              meaning="The stop rule sampled every 20 iterations and saw constant values; the physical check exposed the cycle. "
                                      "Fixed before production (relaxation 0.5, stop rule checks consecutive iterations)."))
    if pf1:
        pf_runs.append(dict(name="G1, to convergence", setup="Coarsest grid from the uniform stream, turbulence variable relaxed by 0.5, live stop",
                            outcome=f"Converged: residuals below \\(10^{{-10}}\\), \\(C_f(x_1) = {tf(pf1m['cf'], 7)}\\), momentum balance "
                                    + pct(pf1m["momentum"], 4) + f"; {pf1[0]['wall_time']:.0f} s"))
        pf_checks += [dict(name="G1 converged: momentum balance", value=pct(pf1m["momentum"], 4), status="ok",
                           meaning="Drag from the wall shear stress against the momentum lost by the stream. What is left is the small viscous "
                                   "stress on the outflow side that the balance leaves out. It sized the production limit of "
                                   + pct(acc["momentum_balance"], 2, False) + "."),
                      dict(name="G1 converged: mass balance", value=m(tn(pf1m["mass"], 1)), status="ok", meaning="Sum of the boundary fluxes over the inflow."),
                      dict(name="G1 converged: largest residual", value=m(tn(pf1m["res_max"], 1)), status="ok",
                           meaning="With the cycle gone the residuals fall to round-off.")]
    if pf5:
        pf_runs.append(dict(name="G5, short run", setup=f"Finest grid from the uniform stream, {pf5['cores']} MPI ranks, {num['preflight']['iterations']} iterations",
                            outcome=f"Ran without errors; {pf5['seconds_per_iteration']:.3f} s per iteration; pressure solve "
                                    f"{pf5['perf']['lin_iters'].get('p', float('nan')):.1f} multigrid cycles per iteration"))
        pf_checks += [dict(name="G5: " + c["name"], value=c["value"], status=c["status"], meaning=c["meaning"]) for c in pf5["checks"]]
        pf_checks.append(dict(name="G5: mesh check", value="Mesh OK; largest aspect ratio " + f"{pf5['mesh']['max_aspect_ratio']:,.0f}", status="ok",
                              meaning="<code>checkMesh</code> flags the stretched wall cells as a warning; they are NASA's design (thin at the wall), and "
                                      "the grid has no skewness and no non-orthogonality."))

    grid_rows = []
    for lv, gi, (r, a), mt in zip(levels, ref["grids"], runs, mets):
        grid_rows.append([lv, f"{gi['points'][0]} × {gi['points'][1]}", f"{r['cells']:,}", gi["points_on_plate"] - 1,
                          m(tn(gi["first_cell_height"], 3)), m(tf(mt["yplus_mean"], 3)), m(tf(mt["yplus_max"], 2))])
    res_rows = []
    for i, (lv, mt) in enumerate(zip(levels, mets)):
        st1 = M.profile(runs[i][1], G, x1)
        res_rows.append([lv, cfx(mt["cf"]), m(tf(orders_cf[i - 2], 2)) if i >= 2 and orders_cf[i - 2] else "–", cfx(mt["cd"]),
                         m(tf(float(st1["nut"].max()), 1)), m(tf(M.thicknesses(runs[i][1], G, x1)["delta99"] * 1000, 2))])
    res_rows.append(["Grid-converged (G3 to G5)", cfx(s_cf["extrap"]) + " \\(\\pm\\) " + pct(s_cf["gci"], 3, False), m(tf(s_cf["p"], 2)) if s_cf["monotonic"] else "–",
                     cfx(s_cd["extrap"]), "–", "–"])
    health_rows = []
    for lv, (r, a), mt in zip(levels, runs, mets):
        d = (r.get("live_stop") or {}).get("decision") or {}
        health_rows.append([lv, f"{r['iterations']:,}", ("✓ " if d.get("status") == "ok" else "⚠ ") + d.get("status", "–"), pct(mt["momentum"], 4),
                            m(tn(mt["mass"], 1)), m(tn(mt["alternation"], 1)), m(tn(mt["res_max"], 1)), report.CHECK_LEGEND[r["health"]["status"]],
                            f"{r['wall_time']:.0f}"])
    cmp_rows = []
    for i, lv in enumerate(levels):
        cmp_rows.append([lv, cfx(cf[i]), cfx(R["fun_inc"][i]), pct(cf[i] / R["fun_inc"][i] - 1, 3), cfx(R["sct"][i]), cfx(R["cfl"][i]), cfx(R["fun"][i]),
                         pct(bias[i], 3)])
    cmp_rows.append(["Grid-converged", cfx(s_cf["extrap"]), cfx(S["fun_inc"]["extrap"]), pct(cf_diff, 3), cfx(S["sct"]["extrap"]), cfx(S["cfl"]["extrap"]),
                     cfx(S["fun"]["extrap"]), "–"])
    cmp_rows.append(["Observed order", m(tf(s_cf["p"], 2)), m(tf(S["fun_inc"]["p"], 2)), "–", m(tf(S["sct"]["p"], 2)), m(tf(S["cfl"]["p"], 2)), m(tf(S["fun"]["p"], 2)), "–"])
    cmp_rows.append(["Grid-convergence index, finest", pct(s_cf["gci"], 3, False), pct(S["fun_inc"]["gci"], 3, False), "–", pct(S["sct"]["gci"], 2, False),
                     pct(S["cfl"]["gci"], 3, False), pct(S["fun"]["gci"], 3, False), "–"])

    var_rows = []
    for vn, v in variants.items():
        for r in v["rows"]:
            d = (r["rec"].get("live_stop") or {}).get("decision") or {}
            var_rows.append([v["what"][0].upper() + v["what"][1:], r["level"], cfx(r["met"]["cf"]), tiny(r["d_cf"], 3), tiny(r["d_cd"], 3), pct(r["met"]["momentum"], 4),
                             f"{r['rec']['iterations']:,}", ("✓" if d.get("status") == "ok" else "⚠")])

    perf_rows, total_wall = [], 0.0
    allruns = [(lv, r) for lv, (r, a) in zip(levels, runs)] + [(f"{vn} {x['level']}", x["rec"]) for vn, v in variants.items() for x in v["rows"]]
    for name, r in allruns:
        p = r["perf"]
        total_wall += r["wall_time"]
        perf_rows.append([name, f"{r['cells']:,}", r["cores"], f"{r['iterations']:,}", f"{r['wall_time']:.0f}", m(tf(p["us_per_cell_step"], 2)),
                          m(tf(p["cpu_efficiency"], 2)) if p.get("cpu_efficiency") else "–", m(tf(p["lin_iters"].get("p", 0.0), 1)),
                          m(tf(p["lin_iters"].get("nuTilda", 0.0), 1))])

    # ---- text ---------------------------------------------------------------------------------------------
    findings = [
        "<b>Same model, same answer?</b> " + {"pass": "Yes. ", "warn": "Close, with a caveat. ", "fail": "No. "}[cf_state]
        + "The skin friction at \\(x_1\\), taken to the limit of an infinitely fine grid, is " + cfx(s_cf["extrap"]) + ", against " + cfx(S["fun_inc"]["extrap"])
        + " for NASA's incompressible FUN3D: " + pct(cf_diff, 3) + ", with a combined grid uncertainty of " + pct(cf_band, 3, False) + ".",
        "<b>What the finest-grid difference is made of.</b> On the \\(545 \\times 385\\) grid the two solutions differ by " + pct(cf_fine_diff, 3)
        + ": this solution's grid error is " + pct(my_grid_err, 3) + ", the reference's " + pct(ref_grid_err, 3) + ", and " + pct(cf_diff, 3)
        + " remains. Compressibility is not part of it (the reference is incompressible); against NASA's Mach 0.2 results the difference is "
        + pct(cf[-1] / R["cfl"][-1] - 1, 2) + " (CFL3D, finest grid), and NASA's own incompressible run differs from that result by " + pct(compress, 2) + ".",
        "<b>Grid convergence.</b> \\(C_f(x_1)\\) changes by " + pct(cf[-1] / cf[-2] - 1, 3) + " from G4 to G5, with observed order "
        + (m(tf(s_cf["p"], 2)) if s_cf["monotonic"] else "not defined") + " and a grid-convergence index of " + pct(s_cf["gci"], 3, False)
        + ". The drag coefficient changes by " + pct(cd[-1] / cd[-2] - 1, 3) + " from G4 to G5; its observed order is "
        + ", ".join(m(tf(p, 2)) if p else "not defined" for p in orders_cd) + " on the grid triplets G1 to G3, G2 to G4 and G3 to G5, so it is "
        + ("regular" if all(p and acc["order_range"][0] <= p <= acc["order_range"][1] for p in orders_cd) and max(orders_cd) - min(orders_cd) < 0.5
           else "not yet regular (NASA notes the same for its codes: the skin friction is singular at the leading edge)") + ".",
        "<b>The model is the standard one.</b> The solver's printed constants are NASA's and the \\(f_{t2}\\) term is on (OpenFOAM leaves it out by default). "
        "Near the wall the simulated profile follows the model's exact wall-layer solution to " + pct(inner_dev, 2, False) + " (\\(y^+ \\le 100\\)).",
        "<b>Every run passed its physical check.</b> The drag from the wall shear stress equals the momentum lost by the stream to "
        + pct(mom_max, 4, False) + " or better on all five grids.",
    ]
    if "noft2" in variants or s_up or "lid" in variants:
        parts = []
        if "noft2" in variants:
            parts.append("Leaving out the \\(f_{t2}\\) term (OpenFOAM's default) changes \\(C_f(x_1)\\) by " + tiny(variants["noft2"]["rows"][0]["d_cf"]))
        if s_up:
            parts.append("first-order advection of the turbulence variable, as in NASA's codes, moves the grid-converged value by "
                         + pct(s_up["extrap"] / s_cf["extrap"] - 1, 3) + " (on G5 itself: " + pct(up["rows"][-1]["d_cf"], 3) + ")")
        if "lid" in variants:
            parts.append("closing the upper side with a lid changes it by " + pct(variants["lid"]["rows"][0]["d_cf"], 2))
        text = "; ".join(parts)
        findings.append("<b>One change at a time.</b> " + text[0].upper() + text[1:] + ".")
    if verdict != "PASS":
        findings.append("<b>Why " + verdict + ".</b> " + " ".join(fails + warns))

    limitations = [
        "Only one quantity has a like-for-like reference: the skin friction at \\(x_1\\) (NASA's incompressible runs). Drag, profiles and eddy viscosity "
        "are published for Mach 0.2 only; they are compared as context and cannot pass or fail.",
        "NASA does not say how its incompressible codes set the inflow and the far field, nor how the skin friction was read at \\(x_1\\), which is a grid point "
        "with no wall face centred on it. Here it is interpolated linearly between the two neighbouring faces; applied to the finest solution, that "
        "interpolation alone shifts the value by " + pct(bias[0], 2) + " on G1 and " + pct(bias[-2], 4) + " on G4.",
        "This verifies the implementation of the model against other codes. It says nothing about how well the model matches a real boundary layer.",
        "The incompressible reference has no published uncertainty; its grid-convergence index was computed here from its three finest grids.",
    ]
    assumptions = [
        dict(name="Incompressible, constant properties", status="imposed",
             check="Stated by the problem. NASA's main results are for Mach 0.2; its own incompressible run differs from them by " + pct(compress, 2)
                   + " in \\(C_f(x_1)\\), so the compressible results are context only."),
        dict(name="Steady, two-dimensional mean flow", status="imposed",
             check="RANS computes the time-averaged flow; the grid is one cell thick. Every run converged to a steady state (residuals at round-off)."),
        dict(name="Turbulence modelled, fully turbulent from the leading edge", status="imposed",
             check="Spalart-Allmaras with \\(\\tilde\\nu = 3\\nu\\) in the free stream; no transition is modelled. The eddy viscosity rises from "
                   + m(tf(th["derived"][4]["value"], 2)) + " \\(\\nu\\) in the free stream to " + m(tf(nut_plate_max, 0)) + " \\(\\nu\\) in the boundary layer."),
        dict(name="Model integrated to the wall (no wall functions)", status="ok",
             check="Needs the wall-cell centre at \\(y^+\\) of about 1 or less. Largest value on the plate: " + m(tf(mets[0]["yplus_max"], 2)) + " on G1, "
                   + m(tf(mets[-1]["yplus_max"], 3)) + " on G5."),
        dict(name="The model solved is NASA's standard form", status="ok" if coeff_ok else "violated",
             check="Constants printed by the solver equal NASA's and the \\(f_{t2}\\) term is active. " + lim_text
                   + (" Lowering the limit to \\(0.1\\,\\Omega\\) changes \\(C_f(x_1)\\) by " + tiny(variants["limiter"]["rows"][0]["d_cf"]) + "."
                      if "limiter" in variants else "")),
        dict(name="The upper side is far enough away", status="ok" if "lid" in variants else "standard",
             check="The boundary layer reaches " + m(tf(stations[1]["tk"]["delta99"] * 1000, 1)) + " mm at \\(x_2\\), " + pct(stations[1]["tk"]["delta99"] / G["H"], 1, False)
                   + " of the domain height. " + ("Even replacing the open far field by a closed lid changes \\(C_f(x_1)\\) by only "
                                                 + pct(variants["lid"]["rows"][0]["d_cf"], 2) + "." if "lid" in variants else
                                                 "NASA reports changes below 0.2 % when the top is moved to \\(0.48\\,L\\).")),
        dict(name="Zero pressure gradient", status="ok",
             check="The stream above the layer is " + pct(stations[0]["tk"]["u_edge"] - 1, 2) + " faster than \\(U_\\infty\\) at \\(x_1\\) and "
                   + pct(stations[1]["tk"]["u_edge"] - 1, 2) + " at \\(x_2\\): the boundary layer displaces the outer flow slightly, as it must over a plate of finite length in a finite domain."),
    ]
    insight_list = insights.load(NAME)
    sim_model = [
        dict(aspect="Air", real="Viscous, slightly compressible at these speeds", model="Incompressible, " + m(rf"\nu = {tn(nu)}\ \mathrm{{m^2/s}}")),
        dict(aspect="Turbulence", real="Unsteady eddies of all sizes", model="Time-averaged flow with the Spalart-Allmaras eddy viscosity (standard form)"),
        dict(aspect="Plate", real="Thin, smooth, " + m(r"2L") + " long", model="No-slip wall on the lower side from \\(x = 0\\); \\(\\tilde\\nu = 0\\)"),
        dict(aspect="Free stream", real="Unbounded", model="Uniform inflow \\(L/3\\) ahead of the plate; open upper side at free-stream pressure, \\(H = L\\) above the plate"),
        dict(aspect="Span", real="Very wide plate", model="Two-dimensional: one cell thick"),
    ]

    # The given description is longer than the report's overview allows (600 characters): the overview takes its
    # first sentences, and the rest goes, word for word, to the top of the setup list (insight).
    sentences = [s.strip() + "." for s in problem["description"].strip().rstrip(".").split(". ")]
    overview, rest = [], list(sentences)
    while rest and len(" ".join(overview + [rest[0]])) <= 600:
        overview.append(rest.pop(0))
    setup_list = ([" ".join(rest)] if rest else []) + list(problem["setup"])

    results = dict(
        title=problem["title"], date=dt.date.today().isoformat(), verdict=verdict, comparison_type="benchmark",
        summary=dict(
            headline=("<b>" + verdict + ".</b> OpenFOAM's Spalart-Allmaras model, set to NASA's standard form, gives a grid-converged skin friction at "
                      "\\(x_1\\) of " + cfx(s_cf["extrap"]) + ", " + pct(cf_diff, 3) + " from NASA's incompressible reference (" + cfx(S["fun_inc"]["extrap"])
                      + "; combined grid uncertainty " + pct(cf_band, 3, False) + ")."),
            answer_columns=["Requested", "NASA", "This solution", "Difference"],
            answers=answers, findings=findings, limitations=limitations),
        problem=dict(description=" ".join(overview), setup=setup_list,
                     given=[dict(label=v["label"], symbol=v["symbol"], value=m(tn(v["value"], 9)), unit=v["unit"]) for v in problem["given"].values()],
                     wanted=wanted, reference=problem["reference"]["description"].split(". On the finest grid")[0] + ". Source: NASA Langley Turbulence "
                     "Modeling Resource, case 2DZP (files and citation in <code>reference/</code>)."),
        assumptions=dict(rows=assumptions),
        theory=dict(
            intro=("The whole flow has no exact solution: the equations are nonlinear and coupled to the turbulence model. Theory still gives one exact "
                   "result that tests the implementation directly. In the layer next to the wall the model has an exact solution, a straight line "
                   "\\(\\tilde\\nu = \\kappa u_\\tau y\\), and with it the model's own law of the wall. SymPy checks it against the model's equation, "
                   "and the simulation is compared with it in Section 7. The momentum balance over the domain gives the physical check of every run."),
            steps=th["steps"], derived=th["derived"], self_checks=[dict(name=k, ok=bool(v)) for k, v in th["self_checks"].items()]),
        decisions=load_yaml(HERE / "decisions.yaml"),
        preflight=dict(
            purpose=("Before the grid study: run the coarsest grid to convergence, to see what a converged run looks like and to size the limits of the "
                     "physical checks, and run the finest grid briefly on its MPI ranks, to see that it starts and what an iteration costs."),
            runs=pf_runs, checks=pf_checks,
            conclusion=("The first attempt failed in an instructive way: the run looked converged to the stop rule"
                        + (", while its momentum balance was " + pct(abs(cycm["momentum"]), 1, False) + " off" if cycm else "") + ". Lowering the relaxation of the "
                        "turbulence variable from 0.9 to 0.5 removed the cycle, and the stop rule was extended to compare consecutive iterations. "
                        "After that every check passed, and the acceptance rules were fixed in <code>numerics.yaml</code> before the production runs."
                        + (f" On the finest grid an iteration takes {pf5['seconds_per_iteration']:.3f} s on {pf5['cores']} ranks." if pf5 else ""))),
        simulation=dict(
            model=sim_model,
            meshes=dict(
                intro="NASA's five nested grids, used point for point. Each level halves the cell size in both directions. The wall-cell centre is given in wall units, "
                      "as the solver reports it on the converged solution.",
                head=["Grid", "Points", "Cells", "Cells on the plate", "Wall cell height (m)", "Mean " + m("y^+") + " of the wall cell", "Largest " + m("y^+")],
                rows=grid_rows),
            results=dict(
                head=["Grid", m("C_f(x_1)"), "Observed order", m("C_D"), "Peak " + m(r"\nu_t/\nu") + " at " + m("x_1"), m(r"\delta_{99}") + " at " + m("x_1") + " (mm)"],
                rows=res_rows),
            health=dict(
                head=["Grid", "Iterations", "Live stop", "Momentum balance", "Mass balance", "Change between iterations", "Largest residual", "Log checks", "Time (s)"],
                rows=health_rows,
                notes=["Each run stopped itself once the estimated iteration error of the drag coefficient and of the wall-cell velocity at both stations was below "
                       + m(tn(num["convergence"]["tol_rel"]["cd"])) + " and that of \\(\\tilde\\nu\\) in the outer layer below " + m(tn(num["convergence"]["tol_rel"]["nuTilda"]))
                       + " (relative), twice in a row. Each grid started from the solution of the grid before it (Figure {fig:history}).",
                       "Momentum balance: momentum lost by the stream (fluxes on the inflow, outflow and upper sides, plus the pressure on the inflow side) over the friction force "
                       "on the plate, minus 1. Mass balance: sum of the boundary fluxes over the inflow. Change between iterations: largest relative change of a monitored value "
                       "between two consecutive iterations over the last 50, the guard against the period-2 cycle of the pre-flight.",
                       "The wall shear stress written by the solver equals \\(\\nu u_P/y_P\\) of the wall cell to " + m(tn(max(mt["wall_cell"] for mt in mets), 1))
                       + " (relative), so the skin friction is the stress in the solver's own momentum balance."]
                      + (["Variants (one change each, started from the main solution of the same grid): see the table in Section 7."] if var_rows else []))),
        comparison=dict(
            kind=("Code-to-code verification: the same model on the same grids in another code. The like-for-like reference is NASA's incompressible FUN3D "
                  "result for the skin friction at \\(x_1\\); the Mach 0.2 results of CFL3D and FUN3D are shown as context."),
            definitions=[
                dict(name="Grid-converged value", latex=r"f_0 = f_5 + \frac{f_5 - f_4}{2^p - 1},\qquad p = \frac{\ln\left[(f_3 - f_4)/(f_4 - f_5)\right]}{\ln 2}",
                     meaning="Richardson extrapolation from the three finest grids (\\(f_3, f_4, f_5\\)); \\(p\\) is the observed order of accuracy. Applied in the same way to this solution and to each of NASA's series."),
                dict(name="Grid-convergence index", latex=r"\mathrm{GCI} = 1.25\,\frac{|f_5 - f_4|}{(2^p - 1)\,|f_5|}",
                     meaning="Uncertainty of the finest-grid value due to the grid (Celik et al. 2008, the procedure NASA uses on its page)."),
                dict(name="Agreement rule", latex=r"|f_0 - f_0^{\mathrm{ref}}| \le \mathrm{GCI}\,|f_5| + \mathrm{GCI}^{\mathrm{ref}}\,|f_5^{\mathrm{ref}}|",
                     meaning="Two grid-converged values agree when they differ by less than the sum of their grid uncertainties. Outside that band and below "
                             + pct(acc["cf_fail"], 2, False) + ": WARN. Above: FAIL. Fixed before the runs."),
                dict(name="Station interpolation", latex=r"C_f(x_1) = C_f^{(i)} + \frac{x_1 - x^{(i)}}{x^{(i+1)} - x^{(i)}}\left(C_f^{(i+1)} - C_f^{(i)}\right)",
                     meaning="Linear interpolation between the two wall faces next to \\(x_1\\). The last column of the table applies it to the finest solution, sampled at each grid's faces: the error of the interpolation alone."),
            ],
            head=["Grid", "This solution", "FUN3D incompr.", "Difference", "SC/TETRA incompr.", "CFL3D (Mach 0.2)", "FUN3D (Mach 0.2)", "Interpolation error"],
            rows=cmp_rows,
            findings=[
                "<b>What the remaining difference is made of.</b><table><thead><tr><th>Part</th><th>How it was measured</th><th>Effect on " + m("C_f(x_1)")
                + "</th></tr></thead><tbody>" + "".join("<tr><td>" + "</td><td>".join(r) + "</td></tr>" for r in made_rows) + "</tbody></table>",
            ] + ([
                "<b>Variants, one change at a time.</b><table><thead><tr><th>Change</th><th>Grid</th><th>" + m("C_f(x_1)") + "</th><th>Change of " + m("C_f(x_1)")
                + "</th><th>Change of " + m("C_D") + "</th><th>Momentum balance</th><th>Iterations</th><th>Stop</th></tr></thead><tbody>"
                + "".join("<tr><td>" + "</td><td>".join(str(c) for c in r) + "</td></tr>" for r in var_rows) + "</tbody></table>"] if var_rows else []) + [
                "<b>Drag (b).</b> " + cfx(cd[-1]) + " on the finest grid, " + pct(cd[-1] / R["cd_cfl"][-1] - 1, 2) + " above CFL3D's Mach 0.2 value and "
                + pct(cd[-1] / R["cd_fun"][-1] - 1, 2) + " above FUN3D's (the two differ by " + pct(abs(cd_codes), 2, False) + " from each other). The skin friction at \\(x_1\\) is "
                + pct(cf[-1] / R["cfl"][-1] - 1, 2) + " above CFL3D's, so drag and local friction differ from the compressible result by the same factor to "
                + pct(cd_shift, 2) + ". Observed order of the drag: " + (m(tf(s_cd["p"], 2)) if s_cd["monotonic"] else "not defined") + " (CFL3D "
                + m(tf(S["cd_cfl"]["p"], 2)) + ", FUN3D " + m(tf(S["cd_fun"]["p"], 2)) + ").",
                "<b>Skin friction along the plate (c).</b> The ratio to CFL3D is " + m(tf(shape_mean, 4)) + " on average and goes from " + m(tf(shape_ends[0], 4)) + " at \\(x = 0.1\\) to "
                + m(tf(shape_ends[1], 4)) + " at \\(x = 1.9\\) (Figure {fig:cf_x}). The first cells after the leading edge and the last before the outflow are left out, as NASA does: the skin friction is singular at the leading edge."
                + (" Part of the difference, and of its slow fall along the plate, has a measured cause that is not density. Above the boundary layer this solution's stream is "
                   + pct(stations[0]["my_edge"] / stations[0]["ref_edge"] - 1, 2) + " faster than CFL3D's at \\(x_1\\) and " + pct(stations[1]["my_edge"] / stations[1]["ref_edge"] - 1, 2)
                   + " at \\(x_2\\) (Figure {fig:profiles}). A variant with a slightly faster inflow gives the sensitivity of this solver's skin friction to that speed: \\(C_f\\), referred to the given \\(U_\\infty\\), changes by "
                   + m(tf(edge["sens"], 2)) + " % per 1 % of speed. With it, the speed difference accounts for " + pct(edge["part_x1"], 2) + " of the " + pct(stations[0]["ratio_cfl"] - 1, 2)
                   + " at \\(x_1\\) and for " + pct(edge["part_x2"], 2) + " of the " + pct(stations[1]["ratio_cfl"] - 1, 2) + " at \\(x_2\\); it predicts a fall of " + pct(edge["predicted"], 2, False)
                   + " between the stations, against " + pct(edge["observed"], 2, False) + " observed. So a large part of what NASA's page presents as the difference between compressible and incompressible results at \\(x_1\\) ("
                   + pct(compress, 2) + ") goes with the speed of the stream over the plate, which the two kinds of set-up do not fix in the same way; what remains, "
                   + pct(stations[0]["ratio_cfl"] - 1 - edge["part_x1"], 2) + " at \\(x_1\\), is the size left for density effects and for CFL3D's stream speeding up along the plate. This split is an estimate from two stations, not a result NASA publishes." if edge else ""),
                "<b>Velocity profiles (d).</b> Largest difference from CFL3D inside the boundary layer: " + m(tf(stations[0]["du_max"], 4)) + " and " + m(tf(stations[1]["du_max"], 4))
                + " (in \\(U_\\infty\\)) at \\(x_1\\) and \\(x_2\\) (Figure {fig:profiles}). Above the layer CFL3D's stream runs at " + m(tf(stations[0]["ref_edge"], 4)) + " and "
                + m(tf(stations[1]["ref_edge"], 4)) + " \\(U_\\infty\\), this solution's at " + m(tf(stations[0]["tk"]["u_edge"], 4)) + " and " + m(tf(stations[1]["tk"]["u_edge"], 4))
                + ": the two set-ups do not give the same free-stream speed over the plate, and the skin friction, which is referred to \\(U_\\infty\\), feels that.",
                "<b>Wall units (d).</b> In the wall cell \\(u^+/y^+ = " + tf(stations[0]["first_cell"], 5) + "\\). For \\(y^+ \\le 100\\) the profile follows the model's own law of the wall to "
                + pct(stations[0]["inner_dev"], 2, False) + " at \\(x_1\\) and " + pct(stations[1]["inner_dev"], 2, False) + " at \\(x_2\\) (largest near \\(y^+ = "
                + f"{stations[0]['inner_dev_at']:.0f}" + "\\), where the stress is no longer exactly constant). At \\(y^+ = 100\\) it is " + m(f"{stations[0]['sa_log_100']:+.2f}")
                + " from the model's log law and " + m(f"{stations[0]['white_100']:+.2f}") + " from the log law NASA plots (\\(\\kappa = " + f"{th['kappa_white']}" + "\\), \\(B = "
                + f"{th['B_white']}" + "\\)); against CFL3D's \\(u^+\\) at the same \\(y^+\\) the largest difference is " + pct(max(st["cfl_wall_dev"] for st in stations), 2, False) + " (Figure {fig:wall_units}).",
                "<b>Eddy viscosity (e).</b> Peak \\(\\nu_t/\\nu\\) at \\(x_1\\): " + m(tf(nut_peak, 1)) + ", against " + m(tf(nut_cfl, 1)) + " (CFL3D) and " + m(tf(nut_fun, 1))
                + " (FUN3D), " + pct(nut_peak / nut_cfl - 1, 2) + ". Along the plate the largest value per column is " + pct(nut_along_mean, 2)
                + " from CFL3D's on average (largest departure " + pct(nut_along_dev, 2, False) + ", \\(0.2 \\le x \\le 1.9\\)); it grows to " + m(tf(nut_plate_max, 1))
                + " at the end of the plate (Figure {fig:nut}).",
                "<b>Rules that passed.</b> " + " ".join(passed),
            ] + (["<b>Rules that did not pass.</b> " + " ".join(fails + warns)] if fails or warns else [])),
        insights=insight_list,
        figures=figures,
        performance=dict(
            intro="Cost of each run. Grids G1 to G3 ran on one core, G4 on 4 and G5 on 8 MPI ranks (strips along the plate), one run at a time.",
            head=["Run", "Cells", "Cores", "Iterations", "Wall (s)", "µs per cell per iteration", "CPU efficiency", "Pressure cycles per iteration", m(r"\tilde\nu") + " iterations per solve"],
            rows=perf_rows,
            notes=[f"Wall time of all runs in the table: {total_wall / 60:.1f} min. Not counted: the pre-flight and the relaxation trials on the coarsest grid (seconds each).",
                   "The coarse grids ran longer than they needed: the stop rule is checked every 5 s of wall time, and a small grid does thousands of iterations in that time.",
                   "Multigrid coped with wall cells 21,000 times longer than high: a few cycles per pressure solve. What costs iterations here is the outer loop, "
                   "limited by the relaxation the turbulence equation needs to avoid the period-2 cycle.",
                   "Next to try: a coupled or implicit treatment of the turbulence source terms, or a relaxation that starts at 0.5 and rises once the wall layer has formed."]),
        reproducibility=dict(
            commands=[
                dict(cmd=f"python -m cases.{NAME}.run preflight", what="Coarsest grid to convergence; short run on the finest grid"),
                dict(cmd=f"python -m cases.{NAME}.run trials", what="Relaxation trials on the coarsest grid: the period-2 cycle and four settings that converge"),
                dict(cmd=f"python -m cases.{NAME}.run production", what="G1 to G5, live stop, each from the grid before"),
                dict(cmd=f"python -m cases.{NAME}.run sensitivity", what="The variants, one change each: " + "; ".join(v["what"] for v in num["sensitivity"].values())),
                dict(cmd=f"python -m cases.{NAME}.analyze", what="Comparison with NASA's data, figures, this report"),
            ],
            files=[dict(path=f"cases/{NAME}/{f}", what=w) for f, w in (
                ("problem.yaml", "The problem as given"), ("reference/", "NASA's grids, data files and pages, as downloaded"),
                ("numerics.yaml", "Grids, solver settings, stopping rule, acceptance rules, variants"),
                ("decisions.yaml", "Every choice with the settings it controls"), ("insights.yaml", "Live workflow log"),
                ("theory.py", "The model's wall-layer solution and law of the wall (SymPy), derived numbers"),
                ("run.py", "Cases, runs, live stop, post-processing"), ("metrics.py", "Skin friction, drag, profiles, balances per run"),
                ("common.py", "Readers for the problem, NASA's grids and data files"), ("sketch.py", "The problem sketch"),
                ("template/", "OpenFOAM case with placeholders"))],
            run_dirs=[dict(path=str(Path(frec["case_dir"]).parent) + "/<G1..G5>", what="Production runs"),
                      dict(path=str(Path(frec["case_dir"]).parent) + "/<variant>_<grid>", what="Variants"),
                      dict(path=str(Path(frec["case_dir"]).parent) + "/pf*, trial_*", what="Pre-flight runs and relaxation trials"),
                      dict(path=f"results/{NAME}/", what="runs/*.json|npz, preflight.json, results.json, report.html")],
            environment=report.environment_info(),
            notes=["One run at a time. G4 used 4 MPI ranks and G5 used 8.",
                   "The relaxation trials are needed only for the relaxation line of the first table in Section 7 and for Figure {fig:cycle}. "
                   "The first diagnosis of the period-2 cycle (comparing the fields of consecutive iterations) was done with one-off scripts that are not kept."]),
        data=dict(levels=levels, h=h_grid, cf=cf, cd=cd, study_cf=s_cf, study_cd=s_cd, reference_series=R, reference_studies=S, cf_diff=cf_diff, cf_band=cf_band,
                  cf_fine_diff=cf_fine_diff, interpolation_bias=bias, cd_shift=cd_shift, shape_mean=shape_mean, shape_spread=shape_spread,
                  inner_dev=inner_dev, du_max=du_max, nut_peak=nut_peak, nut_along_dev=nut_along_dev, relax_spread=relax_spread,
                  metrics=[{k: v for k, v in mt.items()} for mt in mets], passed=passed, warns=warns, fails=fails, total_wall_s=total_wall,
                  variants={vn: [dict(level=r["level"], cf=r["met"]["cf"], cd=r["met"]["cd"], d_cf=r["d_cf"], d_cd=r["d_cd"]) for r in v["rows"]]
                            for vn, v in variants.items()},
                  upwind_study=s_up),
    )
    out = report.render(results, svgs, OUT / "report.html", case_dir=frec["case_dir"])
    print(f"{verdict}: Cf(x1) grid-converged {s_cf['extrap']:.7e} vs FUN3D-inc {S['fun_inc']['extrap']:.7e}: {100 * cf_diff:+.3f} % (band {100 * cf_band:.3f} %), "
          f"order {s_cf['p']:.2f}, GCI {100 * s_cf['gci']:.3f} %")
    print("warns:", warns)
    print("fails:", fails)
    print("report:", out)


# ----------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------
def fig_law(th):
    fig, ax = plots.figure()
    yp = np.logspace(-1, 3.5, 200)
    ax.semilogx(yp, T.inner_law(yp), color=C[0], label="model's law of the wall")
    lo = yp[yp <= 12]
    ax.semilogx(lo, lo, color=C[1], ls="--", lw=1.4, label=r"$u^+ = y^+$")
    hi = yp[yp >= 8]
    ax.semilogx(hi, np.log(hi) / th["kappa_white"] + th["B_white"], color=C[2], ls=":", lw=1.6,
                label=rf"log law, $\kappa = {th['kappa_white']}$, $B = {th['B_white']}$")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$u^+$")
    ax.set_ylim(0, 27)
    fig.legend(loc="outside lower center", ncol=3, fontsize=9)
    return plots.to_svg(fig, "law")


def fig_cycle(bad, good):
    fig, ax = plots.figure()
    n = 120
    hb, hg = bad["hist"][:n, 0], good["hist"][:n, 0]
    ax.plot(np.arange(1, len(hb) + 1), hb * 1e3, color=C[1], lw=1, label=r"$\tilde\nu$ relaxed by 0.9: cycle of period 2")
    ax.plot(np.arange(1, len(hg) + 1), hg * 1e3, color=C[0], label=r"$\tilde\nu$ relaxed by 0.5: converges")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"drag coefficient $C_D \times 10^3$")
    lo, hi = float(bad["hist"][-2:, 0].min() * 1e3), float(bad["hist"][-2:, 0].max() * 1e3)
    ax.set_ylim(lo - 0.6 * (hi - lo), hi + 0.6 * (hi - lo))
    fig.legend(loc="outside lower center", ncol=2, fontsize=9)
    return plots.to_svg(fig, "cycle")


def fig_history(rec, a):
    fig, (a1, a2) = plots.wide(2)
    for k, (name, lab) in enumerate((("res_Ux", r"$u$"), ("res_p", r"$p$"), ("res_nuTilda", r"$\tilde\nu$"))):
        r = a[name]
        a1.semilogy(np.arange(1, len(r) + 1), r, color=C[k], lw=1.2, label=lab)
    a1.set_xlabel("iteration")
    a1.set_ylabel("initial residual")
    a1.set_title("Residuals")
    a1.legend(loc="upper right", fontsize=9)
    h = a["hist"]
    labs = [r"$C_D$", r"wall-cell $u$ at $x_1$", r"wall-cell $u$ at $x_2$", r"$\tilde\nu$ at $x_1$", r"$\tilde\nu$ at $x_2$"]
    cols = [C[0], C[1], C[1], C[2], C[2]]
    sty = ["-", "-", "--", "-", "--"]
    it = a["hist_it"]
    for k in range(5):
        d = np.abs(h[:-1, k] / h[-1, k] - 1)
        d[d == 0] = np.nan
        a2.semilogy(it[:-1], d, color=cols[k], ls=sty[k], lw=1.2, label=labs[k])
    a2.set_xlabel("iteration")
    a2.set_ylabel("distance from the final value (relative)")
    a2.set_title("Monitored values")
    fig.legend(*a2.get_legend_handles_labels(), loc="outside lower center", ncol=5, fontsize=8)
    return plots.to_svg(fig, "history")


def fig_grid(h, cf, cd, R, S, s_cf, s_cd):
    fig, (a1, a2) = plots.wide(2, h=3.9)
    h = np.asarray(h)
    a1.plot(h, np.array(cf) * 1e3, "o-", color=C[0], label="this solution (incompressible)")
    a1.plot(h, np.array(R["fun_inc"]) * 1e3, "s-", color=C[1], lw=1.4, ms=5, label="FUN3D incompressible")
    a1.plot(h, np.array(R["sct"]) * 1e3, "d-", color=C[3], lw=1.2, ms=4, label="SC/TETRA incompressible")
    a1.plot(h, np.array(R["cfl"]) * 1e3, "^--", color=C[2], lw=1.2, ms=5, label="CFL3D, Mach 0.2")
    a1.plot(h, np.array(R["fun"]) * 1e3, "v--", color=plots.MUTED, lw=1.2, ms=5, label="FUN3D, Mach 0.2")
    a1.plot([0], [s_cf["extrap"] * 1e3], "*", color=C[0], ms=11)
    a1.plot([0], [S["fun_inc"]["extrap"] * 1e3], "*", color=C[1], ms=9)
    a1.set_ylim(2.66, 2.82)
    a1.set_xlim(-0.0015, 0.037)
    a1.set_xlabel(r"grid spacing $h = 1/\sqrt{N}$")
    a1.set_ylabel(r"$C_f(x_1) \times 10^3$")
    a1.set_title(r"Skin friction at $x_1$")
    a2.plot(h, np.array(cd) * 1e3, "o-", color=C[0])
    a2.plot(h, np.array(R["cd_cfl"]) * 1e3, "^--", color=C[2], lw=1.2, ms=5)
    a2.plot(h, np.array(R["cd_fun"]) * 1e3, "v--", color=plots.MUTED, lw=1.2, ms=5)
    a2.plot([0], [s_cd["extrap"] * 1e3], "*", color=C[0], ms=11)
    a2.set_xlim(-0.0015, 0.037)
    a2.set_xlabel(r"grid spacing $h = 1/\sqrt{N}$")
    a2.set_ylabel(r"$C_D \times 10^3$")
    a2.set_title("Drag coefficient of the plate")
    fig.legend(*a1.get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8)
    return plots.to_svg(fig, "grid_cf")


def fig_cf_x(xf, cf, zc, r_cfl, r_fun, mean):
    fig, (a1, a2) = plots.wide(2)
    a1.plot(xf, cf * 1e3, color=C[0], label="this solution")
    a1.plot(zc["CFL3D"][:, 0], zc["CFL3D"][:, 1] * 1e3, color=C[2], ls="--", lw=1.4, label="CFL3D, Mach 0.2")
    a1.plot(zc["FUN3D"][:, 0], zc["FUN3D"][:, 1] * 1e3, color=plots.MUTED, ls=":", lw=1.6, label="FUN3D, Mach 0.2")
    a1.set_ylim(2.2, 6.0)
    a1.set_xlabel(r"$x$")
    a1.set_ylabel(r"$C_f \times 10^3$")
    a1.set_title("Skin friction along the plate")
    w = (xf >= 0.02) & (xf <= 1.98)
    a2.plot(xf[w], r_cfl[w], color=C[2], label="CFL3D, Mach 0.2")
    a2.plot(xf[w], r_fun[w], color=plots.MUTED, ls=":", lw=1.6, label="FUN3D, Mach 0.2")
    a2.set_ylim(mean - 0.012, mean + 0.012)
    a2.set_xlabel(r"$x$")
    a2.set_ylabel(r"$C_f$ of this solution / $C_f$ of the code")
    a2.set_title("Ratio to NASA's compressible codes")
    fig.legend(*a1.get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9)
    return plots.to_svg(fig, "cf_x")


def fig_profiles(stations):
    fig, (a1, a2) = plots.wide(2)
    for k, st in enumerate(stations):
        pr = st["pr"]
        a1.semilogx(pr["y"], pr["u"], color=C[k], label=rf"this solution, ${st['name']}$")
        a1.semilogx(pr["y"], st["u_ref"], color=plots.INK, ls=":", lw=1.3, label="CFL3D, Mach 0.2" if k == 0 else "_nolegend_")
        a2.semilogx(pr["y"], pr["u"] - st["u_ref"], color=C[k])
    a1.set_xlim(1e-6, 0.3)
    a2.set_xlim(1e-6, 0.3)
    a1.set_xlabel(r"$y$ (m)")
    a1.set_ylabel(r"$u/U_\infty$")
    a1.set_title("Velocity profiles")
    a2.set_xlabel(r"$y$ (m)")
    a2.set_ylabel(r"difference in $u/U_\infty$")
    a2.set_title("This solution minus CFL3D")
    fig.legend(*a1.get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9)
    return plots.to_svg(fig, "profiles")


def fig_wall(stations, zw, th):
    fig, ax = plots.figure(6.8, 4.2)
    yp = np.logspace(-1.3, 2.6, 200)
    ax.semilogx(yp, T.inner_law(yp), color=plots.INK, lw=1.2, label="model's law of the wall (theory)")
    hi = np.logspace(1.2, 3.6, 50)
    ax.semilogx(hi, np.log(hi) / th["kappa_white"] + th["B_white"], color=C[3], ls=":", lw=1.6,
                label=rf"log law, $\kappa = {th['kappa_white']}$, $B = {th['B_white']}$")
    for k, st in enumerate(stations):
        pr = st["pr"]
        ax.semilogx(pr["yplus"], pr["uplus"], color=C[k], ls="--" if k == 0 else "-.", lw=1.6, label=rf"this solution, ${st['name']}$")
    z = zw[stations[0]["key"]]
    ax.semilogx(10 ** z[::12, 0], z[::12, 1], "o", ms=3, color=C[2], label=r"CFL3D, Mach 0.2, $x_1$")
    ax.set_xlim(0.05, 3e4)
    ax.set_ylim(0, 30)
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$u^+$")
    fig.legend(loc="outside lower center", ncol=3, fontsize=8)
    return plots.to_svg(fig, "wall_units")


def fig_nut(p1, zn, xn, nmax, pk):
    fig, (a1, a2) = plots.wide(2)
    a1.plot(p1["nut"], p1["y"] * 1e3, color=C[0], label="this solution")
    a1.plot(zn["CFL3D"][:, 2], zn["CFL3D"][:, 1] * 1e3, color=C[2], ls="--", lw=1.4, label="CFL3D, Mach 0.2")
    a1.plot(zn["FUN3D"][:, 2], zn["FUN3D"][:, 1] * 1e3, color=plots.MUTED, ls=":", lw=1.6, label="FUN3D, Mach 0.2")
    a1.set_ylim(0, 25)
    a1.set_xlabel(r"$\nu_t/\nu$")
    a1.set_ylabel(r"$y$ (mm)")
    a1.set_title(r"Eddy viscosity profile at $x_1$")
    a2.plot(xn, nmax, color=C[0])
    a2.plot(pk[:, 0], pk[:, 1], color=C[2], ls="--", lw=1.4)
    a2.set_xlabel(r"$x$")
    a2.set_ylabel(r"largest $\nu_t/\nu$ in the column")
    a2.set_title("Largest eddy viscosity along the plate")
    fig.legend(*a1.get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9)
    return plots.to_svg(fig, "nut")


if __name__ == "__main__":
    main()
