"""Compare the 3D cylinder simulations (DFG 3D-1Z, Re = 20) with the benchmark; mesh study; results.json + report.

Usage (repo root, inside WSL):  python -m cases.dfg_cylinder_3d_re20.analyze [--report]
Reads results/dfg_cylinder_3d_re20/{preflight.json, runs/*.json|npz}, the case YAML files and the reference data
in reference/. Every number in the report is computed here.
"""
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

from cfdlib import plots, report, verification as V
from cfdlib.report import LETTERS, load_yaml, m, tex_num as tn, tex_fix as tf
from . import theory
from .run import mesh_spec
from .sketch import sketch

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"
QK = ["cd", "cl"]
SYM = dict(cd="c_D", cl="c_L")
DEC = dict(cd=4, cl=5)
ST = {"ok": "✓ OK", "warn": "⚠ warning", "fail": "✗ fail"}
# Sensitivity runs: name -> what changed (run.py variant / scratch r2.sh)
SENS = {"L3_LU": "L3, linear-upwind convection", "L3_R2": "L3, O-grid radial ratio 2", "L3_R8": "L3, O-grid radial ratio 8",
        "L4_R2": "L4, O-grid radial ratio 2"}
SOFT = {"OF": "OpenFOAM", "CFX": "CFX", "FeatFlow": "FeatFlow"}


# ---------------------------------------------------------------------------------------------
# Reference: Braack and Richter (2006) values + the paper's level 1-4 results of three codes
# ---------------------------------------------------------------------------------------------
def reference(ref):
    val = {k: float(ref["reference_values"][k]) for k in QK}
    by = {}
    for r in ref["results"]:
        by.setdefault(r["software"], {})[r["level"]] = dict(cd=float(r["cd"]), cl=float(r["cl"]), cells=int(r["cells"]),
                                                            err_cd=float(r["err_cd_pct"]), err_cl=float(r["err_cl_pct"]))
    top = max(by["FeatFlow"])
    # Reference uncertainty: distance to the finest FeatFlow result (an independent, high-order solution);
    # the printed digits alone would allow only half a unit in the last digit.
    digits = {k: 0.5 * 10 ** (-len(str(ref["reference_values"][k]).split(".")[1])) for k in QK}
    unc = {k: max(abs(val[k] - by["FeatFlow"][top][k]), digits[k]) for k in QK}
    seq = {s: {k: V.level_sequence([by[s][lv][k] for lv in sorted(by[s])]) for k in QK} for s in by}
    return dict(val=val, by=by, top=top, unc=unc, digits=digits, seq=seq)


# ---------------------------------------------------------------------------------------------
# One converged run: coefficients, convergence check, balances
# ---------------------------------------------------------------------------------------------
def analyse_run(name, g, numerics):
    meta = json.loads((RUNS / f"{name}.json").read_text())
    d = np.load(RUNS / f"{name}.npz")
    D, H, rho = g["D"]["value"], g["H"]["value"], g["rho"]["value"]
    Ubar = th_Ubar(g)
    F0 = 0.5 * rho * Ubar ** 2 * D * H
    F = d["Fcyl"]
    cd, cl = F[:, 0] / F0, F[:, 1] / F0
    cv = numerics["convergence"]
    k = int((1 - cv["window"]) * len(cd))
    dcd, dcl = float(np.ptp(cd[k:])), float(np.ptp(cl[k:]))
    Fx = float(F[-1, 0])
    # steady x-momentum balance (kinematic): F_cyl + F_wall = P_in - P_out - (MF_in + MF_out)
    lhs = Fx + float(d["Fwall"][-1, 0])
    rhs = float(d["p_in"][-1] - d["p_out"][-1] - (d["mom_in"][-1][0] + d["mom_out"][-1][0]))
    q_in, q_out = float(d["q_in"][-1]), float(d["q_out"][-1])
    iv = RUNS / "inlet_visc.json"
    iv = json.loads(iv.read_text()).get(name) if iv.exists() else None
    bal_visc = (lhs - rhs - iv["F_visc_in"]) / Fx if iv else float("nan")
    return dict(name=name, meta=meta, it=d["it"], cdh=cd, clh=cl, cd=float(cd[-1]), cl=float(cl[-1]),
                balance_visc=bal_visc, F_visc_in=iv["F_visc_in"] / Fx if iv else float("nan"),
                cd_p=float(d["Fcyl_p"][-1, 0] / F0), cd_v=float(d["Fcyl_v"][-1, 0] / F0),
                cl_p=float(d["Fcyl_p"][-1, 1] / F0), cl_v=float(d["Fcyl_v"][-1, 1] / F0),
                dcd=dcd, dcl=dcl, conv_ok=dcd < cv["cd_change"] and dcl < cv["cl_change"],
                side=float(abs(F[-1, 2]) / Fx), balance=(lhs - rhs) / Fx, mass=abs(q_in + q_out) / abs(q_in),
                inflow_excess=-q_in / (Ubar * H * H) - 1, F0=F0)


def th_Ubar(g):
    return 4.0 / 9.0 * g["Um"]["value"]


def study(values, r=2.0):
    """Observed order, Richardson value and fine-mesh GCI from three solutions (coarse, medium, fine)."""
    f1, f2, f3 = values[-3:]
    R = (f2 - f3) / (f1 - f2) if f1 != f2 else float("nan")
    monotonic = bool(0 < R < 1)
    p = V.order_three_grids(f1, f2, f3, r) if monotonic else float("nan")
    p_use = p if monotonic else 2.0
    ext = V.richardson(f2, f3, p_use, r)
    gci_abs = 1.25 * abs(f3 - f2) / (r ** p_use - 1)
    return dict(R=R, monotonic=monotonic, p=p, p_used=p_use, extrap=ext, gci_abs=gci_abs, gci=gci_abs / abs(f3))


def wake(g):
    """Mid-plane picture data of the finest run: reversed-flow region and its length on the cylinder's centreline."""
    lv = sorted(RUNS.glob("*_mid.npz"))
    if not lv:
        return None
    w = np.load(lv[-1])
    import matplotlib.tri as mtri
    D, xc, yc = (g[k]["value"] for k in ("D", "xc", "yc"))
    sel = (w["x"] > xc - 0.25) & (w["x"] < xc + 1.0)
    tri = mtri.Triangulation(w["x"][sel], w["y"][sel])
    tx, ty = w["x"][sel][tri.triangles].mean(axis=1), w["y"][sel][tri.triangles].mean(axis=1)
    tri.set_mask(np.hypot(tx - xc, ty - yc) < 0.5 * D)
    # the Delaunay triangulation of a structured grid has degenerate triangles (fine for plotting, not for a
    # trifinder): interpolate along lines with scipy instead
    from scipy.interpolate import LinearNDInterpolator
    ip = LinearNDInterpolator(np.c_[w["x"][sel], w["y"][sel]], w["u"][sel])
    xs = np.linspace(xc + 0.5 * D + 1e-4, xc + 4 * D, 4000)
    # line of the wake: through the most downstream cell centre with reversed flow (u < 0)
    neg_c = np.nonzero((w["u"] < 0) & (w["x"] > xc))[0]
    y_wake = float(w["y"][neg_c[np.argmax(w["x"][neg_c])]])
    u_line = ip(xs, np.full_like(xs, y_wake))
    neg = np.nonzero(u_line < 0)[0]
    x_end = float(xs[neg[-1]]) if len(neg) else float("nan")
    if len(neg) and neg[-1] + 1 < len(xs):                       # interpolate the zero crossing
        i = neg[-1]
        x_end = float(xs[i] - u_line[i] * (xs[i + 1] - xs[i]) / (u_line[i + 1] - u_line[i]))
    return dict(w=w, tri=tri, sel=sel, y_wake=y_wake, xs=xs, u_line=u_line, x_end=x_end,
                Lr=(x_end - xc - 0.5 * D) / D, u_min=float(np.min(w["u"])), w_max=float(np.max(np.abs(w["w"]))),
                name=lv[-1].name.split("_mid")[0])


def main():
    problem = load_yaml(HERE / "problem.yaml")
    numerics = load_yaml(HERE / "numerics.yaml")
    acc = numerics["acceptance"]
    cv = numerics["convergence"]
    g = problem["given"]
    ref = reference(load_yaml(HERE / problem["reference"]["file"]))
    th = theory.evaluate(g)
    levels = numerics["mesh"]["levels"]
    have = [L for L in levels if (RUNS / f"{L}.json").exists()]
    missing = [L for L in levels if L not in have]
    runs = {L: analyse_run(L, g, numerics) for L in have}
    # sensitivity runs (same solver, one change each): convection scheme and O-grid radial grading
    sens = {n: analyse_run(n, g, numerics) for n in SENS if (RUNS / f"{n}.json").exists()}
    fine = runs[have[-1]]
    ms = {k: study([runs[L][k] for L in have]) for k in QK} if len(have) >= 3 else {}
    ms_coarse = {k: study([runs[L][k] for L in have[:3]]) for k in QK} if len(have) >= 4 else {}
    comp = {}
    for k in QK:
        sim, r0 = fine[k], ref["val"][k]
        u_sim = ms[k]["gci_abs"] if ms else float("nan")
        band = u_sim + ref["unc"][k]
        diff = sim - r0
        status = "ok" if abs(diff) <= band else ("warn" if abs(diff) <= 2 * band else "fail")
        of4 = ref["by"]["OF"][ref["top"]][k]
        comp[k] = dict(sim=sim, ref=r0, u_ref=ref["unc"][k], u_sim=u_sim, band=band, diff=diff, rel=diff / abs(r0),
                       ratio=abs(diff) / band if band else float("inf"), status=status,
                       extrap=ms[k]["extrap"] if ms else float("nan"),
                       diff_ext=(ms[k]["extrap"] - r0) if ms else float("nan"),
                       of4=of4, of4_rel=(of4 - r0) / abs(r0))
    # ---- verdict (rules fixed in numerics.yaml before the production results were seen) --------------
    fail, warn = [], []
    if missing:
        fail.append(f"planned meshes not run: {missing}")
    for nm, r in runs.items():
        if r["meta"]["mesh_ok"] is False:
            fail.append(f"checkMesh reported a problem ({nm})")
        if r["meta"]["health"]["status"] == "fail":
            fail.append(f"run {nm} failed a log health check")
        if not r["meta"]["converged"]:
            fail.append(f"run {nm} stopped before its residual targets")
        if not r["conv_ok"]:
            fail.append(f"run {nm}: the force coefficients still changed over the last iterations (physical stopping check)")
        if r["side"] > cv["side_force_rel"]:
            warn.append(f"run {nm}: side force not negligible (mid-plane symmetry)")
    if abs(fine["balance"]) > cv["momentum_balance_rel"]:
        fail.append("steady momentum balance not closed on the finest mesh")
    for k in QK:
        if comp[k]["status"] == "fail":
            fail.append(f"{m(SYM[k])} differs from the reference by more than twice the combined uncertainty")
        elif comp[k]["status"] == "warn":
            warn.append(f"{m(SYM[k])} differs from the reference by more than the combined uncertainty (but less than twice)")
        if ms and not ms[k]["monotonic"]:
            warn.append(f"{m(SYM[k])}: convergence on the three finest meshes not monotonic (order undefined)")
        elif ms and not acc["order_range"][0] <= ms[k]["p"] <= acc["order_range"][1]:
            warn.append(f"{m(SYM[k])}: observed order {m(tf(ms[k]['p'], 2))} outside "
                        + m(tn(acc['order_range'][0]) + r' \le p \le ' + tn(acc['order_range'][1])))
    verdict = "FAIL" if fail else "WARN" if warn else "PASS"
    wk = wake(g)
    data = dict(fail=fail, warn=warn, comp=comp, ms=ms, ms_coarse=ms_coarse,
                ref=dict(val=ref["val"], unc=ref["unc"], seq=ref["seq"]),
                runs={nm: {k: v for k, v in r.items() if k not in ("cdh", "clh", "it", "meta")} | dict(
                    iterations=r["meta"]["iterations"], cells=r["meta"]["cells"], wall=r["meta"]["wall_time"])
                      for nm, r in runs.items()},
                wake={k: wk[k] for k in ("y_wake", "x_end", "Lr", "u_min", "w_max", "name")} if wk else None,
                sens={n: dict(cd=r["cd"], cl=r["cl"], iterations=r["meta"]["iterations"]) for n, r in sens.items()})
    (OUT / "analysis.json").write_text(json.dumps(data, indent=1, default=float))
    print("verdict", verdict, "fail", fail, "warn", warn)
    for k in QK:
        c = comp[k]
        print(f"{k}: sim {c['sim']:.6f} ref {c['ref']:.6f} diff {c['diff']:+.6f} ({100 * c['rel']:+.3f} %) band {c['band']:.2e} "
              f"{c['status']}; " + (f"p {ms[k]['p']:.2f} ext {ms[k]['extrap']:.6f} ({100 * c['diff_ext'] / c['ref']:+.3f} %) GCI {ms[k]['gci_abs']:.2e}" if ms else "")
              + f"; OF L4 {c['of4']:.6f} ({100 * c['of4_rel']:+.3f} %)")
    for nm, r in runs.items():
        print(f"{nm}: cd {r['cd']:.6f} cl {r['cl']:.7f} it {r['meta']['iterations']} dcd {r['dcd']:.1e} dcl {r['dcl']:.1e} "
              f"side {r['side']:.1e} balance {r['balance']:+.2e} inflow {r['inflow_excess']:.2e}")
    for nm, r in sens.items():
        print(f"{nm}: cd {r['cd']:.6f} cl {r['cl']:.7f} ({100 * (r['cl'] / ref['val']['cl'] - 1):+.2f} %) it {r['meta']['iterations']}")
    if wk:
        print("wake", {k: wk[k] for k in ("y_wake", "x_end", "Lr", "u_min", "w_max")})
    if "--report" in sys.argv:
        build_report(problem, numerics, g, ref, th, runs, have, fine, ms, ms_coarse, comp, fail, warn, verdict, wk, data, sens)


def build_report(problem, numerics, g, ref, th, runs, have, fine, ms, ms_coarse, comp, fail, warn, verdict, wk, data, sens):
    acc, cv = numerics["acceptance"], numerics["convergence"]
    D, H, Um = g["D"]["value"], g["H"]["value"], g["Um"]["value"]
    dv = {d["key"]: d["value"] for d in th["derived"]}
    fl = have[-1]
    fm = fine["meta"]
    top = ref["top"]
    wanted = [dict(letter=LETTERS[i], key=w["key"], text=w["text"]) for i, w in enumerate(problem["quantities_of_interest"])]
    L_of = {w["key"]: w["letter"] for w in wanted}
    spec = {L: mesh_spec(L, problem, numerics) for L in have}
    preflight = json.loads((OUT / "preflight.json").read_text())
    gpu = load_gpu_summary()

    def q(k, v=None):
        return tf(comp[k]["sim"] if v is None else v, DEC[k])

    ans = {}
    for k, key in (("cd", "drag"), ("cl", "lift")):
        c = comp[k]
        ans[key] = dict(
            theory=m(rf"{SYM[k]} = {q(k, c['ref'])} \pm {tn(c['u_ref'], 2)}") + "<div class='sub'>OpenFOAM in the paper, level "
                   + str(top) + ": " + m(q(k, c["of4"])) + "</div>",
            sim=m(rf"{SYM[k]} = {q(k)}") + "<div class='sub'>extrapolated: " + m(q(k, c["extrap"])) + "</div>",
            diff=m(f"{c['diff']:+.{DEC[k]}f}") + " (" + m(tn(100 * c["rel"], 2) + r"\,\%") + ")<br>"
                 + f"{tf(c['ratio'], 2)} band " + ST[c["status"]][0] + "<div class='sub'>extrapolated: "
                 + m(tn(100 * c["diff_ext"] / c["ref"], 2) + r"\,\%") + "</div>")
    if wk:
        ans["flow"] = dict(theory="Steady flow; a closed region of reversed flow behind the cylinder (Theory, step 1)",
                           sim="Figure {fig:flow}: reversed flow up to " + m(rf"{tf(wk['Lr'], 2)}\,D") + " behind the cylinder",
                           diff="no reference value<div class='sub'>symmetry check: "
                                + m(rf"|w|_{{\max}} = {tn(wk['w_max'], 2)}\,\mathrm{{m/s}}") + " on the mid-plane</div>")
    else:
        ans["flow"] = dict(theory="Steady flow, closed reversed-flow region", sim="picture not produced", diff="–")
    answers = [dict(letter=w["letter"], label=w["text"], **ans[w["key"]]) for w in wanted]

    # ---- figures --------------------------------------------------------------------------------
    svgs = {"sketch": sketch(g), "iters": _iters_fig(runs, have, cv), "levels": _levels_fig(runs, have, ref, ms, sens)}
    figures = [
        dict(key="sketch", section="problem", caption="The channel and the cylinder to scale: side view in the mid-plane "
             + m("z = H/2") + " (left) and the square cross-section seen from the inlet (right), where the cylinder spans the "
             "channel from wall to wall. The inflow profile is a parabola in both " + m("y") + " and " + m("z") + ". Orange: the "
             "requested forces (a), (b); dotted: the steady region of reversed flow shown in (c)."),
        dict(key="iters", section="convergence", caption="Physical stopping check: change of the drag (left) and lift (right) "
             "coefficient from its final value, against the iteration, on each mesh (finer meshes start from the converged "
             "coarser flow). Every run stops on its residual targets; the dashed lines are the tolerances fixed beforehand for "
             "the change over the last " + m(tn(100 * cv["window"]) + r"\,\%") + " of the iterations. All runs end "
             + ("below them." if all(r["conv_ok"] for r in runs.values()) else "– not all below them.")),
        dict(key="levels", section="comparison", caption="Drag and lift against the number of cells: this simulation (blue, "
             "levels L1-L4), the benchmark paper's OpenFOAM, CFX and FeatFlow results on its levels 1-4, and the reference value "
             "(dotted). Open circles: the same setup with a different radial cell distribution in the O-grid (ratio 2 instead "
             "of 4). The open diamond is the Richardson extrapolation of L2-L4 with its GCI bar. Right: the lift as its error "
             "relative to the reference, on a scale that is linear between " + m(r"-1") + " and " + m(r"1\,\%") + " and "
             "logarithmic outside, because the coarse levels are far off."),
    ]
    if wk:
        svgs["flow"] = _flow_fig(wk, g)
        figures.append(dict(key="flow", section="simulation", caption="Output (" + L_of["flow"] + "): streamwise velocity "
                            + m("u") + " in the mid-plane " + m("z = H/2") + " on the finest mesh (" + wk["name"] + "). The black "
                            "line is " + m("u = 0") + ": inside it (blue) the fluid flows backwards, towards the cylinder; the "
                            "colour scale is stretched below zero so that this weak backflow (at most "
                            + m(rf"{tf(-wk['u_min'], 3)}\,\mathrm{{m/s}}") + ") is visible. "
                            "Right: " + m("u") + " along the line " + m(rf"y = {tf(wk['y_wake'], 3)}\,\mathrm{{m}}")
                            + " through the reversed-flow region; it is negative up to " + m(rf"x = {tf(wk['x_end'], 3)}\,\mathrm{{m}}")
                            + ", " + m(rf"{tf(wk['Lr'], 2)}\,D") + " behind the cylinder surface. The fast fluid squeezed "
                            "between cylinder and walls is red."))

    # ---- tables --------------------------------------------------------------------------------
    delta = dv["delta"]
    mesh_rows = []
    for L in have:
        s_, r = spec[L], runs[L]
        mesh_rows.append([L, f"{s_['cells']:,}", str(4 * s_["nc"]), str(s_["nz"]), m(tn(s_["h"]["wall"] * 1000, 3) + r"\,\mathrm{mm}"),
                          m(tn(delta / s_["h"]["wall"], 3)), m(tn(s_["h"]["z_wall"] * 1000, 3) + r"\,\mathrm{mm}"),
                          m(tn(r["meta"]["checkmesh"]["max_nonorth"], 3)) + "°", f"{r['meta']['iterations']}"])
    res_rows = []
    of, ff = ref["by"]["OF"], ref["by"]["FeatFlow"]
    for i, L in enumerate(have):
        r = runs[L]
        lv = i + 1
        res_rows.append([L, f"{r['meta']['cells']:,}", m(q("cd", r["cd"])), m(q("cl", r["cl"])),
                         m(tf(r["cd_p"], 4)) + " / " + m(tf(r["cd_v"], 4)),
                         f"{of[lv]['cells']:,}" if lv in of else "–", m(q("cd", of[lv]["cd"])) if lv in of else "–",
                         m(q("cl", of[lv]["cl"])) if lv in of else "–"])
    if ms:
        res_rows += [
            ["Observed order " + m("p") + " (L2-L4)", "", m(tf(ms["cd"]["p"], 2)) if ms["cd"]["monotonic"] else "–",
             m(tf(ms["cl"]["p"], 2)) if ms["cl"]["monotonic"] else "–", "", "",
             m(tf(ref["seq"]["OF"]["cd"]["order"], 2)) if ref["seq"]["OF"]["cd"]["order"] else "–",
             m(tf(ref["seq"]["OF"]["cl"]["order"], 2)) if ref["seq"]["OF"]["cl"]["order"] else "–"],
            ["Richardson extrapolation", "", m(q("cd", ms["cd"]["extrap"])), m(q("cl", ms["cl"]["extrap"])), "", "",
             m(q("cd", ref["seq"]["OF"]["cd"]["richardson"])) if ref["seq"]["OF"]["cd"]["richardson"] else "–",
             m(q("cl", ref["seq"]["OF"]["cl"]["richardson"])) if ref["seq"]["OF"]["cl"]["richardson"] else "–"],
            ["GCI of " + fl + " (absolute)", "", m(tn(ms["cd"]["gci_abs"], 2)), m(tn(ms["cl"]["gci_abs"], 2)), "", "", "", ""],
            ["Reference (Braack and Richter)", "", m(q("cd", ref["val"]["cd"])), m(q("cl", ref["val"]["cl"])), "", "", "", ""],
        ]
    for n in SENS:
        if n in sens:
            r = sens[n]
            res_rows.append([n + " (" + SENS[n].split(", ", 1)[1] + ")", f"{r['meta']['cells']:,}", m(q("cd", r["cd"])),
                             m(q("cl", r["cl"])), m(tf(r["cd_p"], 4)) + " / " + m(tf(r["cd_v"], 4)), "", "", ""])
    health_rows = []
    for L in list(have) + [n for n in SENS if n in sens]:
        r = runs.get(L) or sens[L]
        health_rows.append([L, f"{r['meta']['iterations']}", ST[r["meta"]["health"]["status"]],
                            m(tn(r["dcd"], 1)) + "<br>" + m(tn(r["dcl"], 1)), "✓" if r["conv_ok"] else "✗",
                            m(tn(100 * r["balance"], 2) + r"\,\%"),
                            m(tn(100 * r["balance_visc"], 2) + r"\,\%") if np.isfinite(r["balance_visc"]) else "–",
                            m(tn(r["side"], 1)), m(tn(r["inflow_excess"], 2)), f"{r['meta']['wall_time']:.0f} ({r['meta']['cores']})"])
    comp_rows = []
    for k in QK:
        c = comp[k]
        comp_rows.append([m(SYM[k]), m(q(k, c["ref"])), m(tn(c["u_ref"], 2)), m(q(k)), m(tn(c["u_sim"], 2)),
                          m(f"{c['diff']:+.{DEC[k]}f}") + "<br>" + m(tn(100 * c["rel"], 2) + r"\,\%"),
                          m(tf(c["ratio"], 2)) + "<br>" + ST[c["status"]],
                          m(q(k, c["extrap"])) + "<br>" + m(tn(100 * c["diff_ext"] / c["ref"], 2) + r"\,\%"),
                          m(q(k, c["of4"])) + "<br>" + m(tn(100 * c["of4_rel"], 2) + r"\,\%")])

    # ---- findings ------------------------------------------------------------------------------
    cdd = [runs[L]["cd"] for L in have]
    cld = [runs[L]["cl"] for L in have]
    findings = [
        f"<b>Drag ({L_of['drag']}).</b> " + m(rf"c_D = {q('cd')}") + f" on {fl} ({fm['cells']:,} cells), "
        + m(tn(100 * comp["cd"]["rel"], 2) + r"\,\%") + " from the reference " + m(q("cd", ref["val"]["cd"]))
        + "; Richardson extrapolation of L2-L4 gives " + m(q("cd", comp["cd"]["extrap"])) + " ("
        + m(tn(100 * comp["cd"]["diff_ext"] / comp["cd"]["ref"], 2) + r"\,\%") + "). The paper's OpenFOAM result on its level "
        + str(top) + " (" + f"{of[top]['cells']:,}" + " cells) is " + m(q("cd", comp["cd"]["of4"])) + " ("
        + m(tn(100 * comp["cd"]["of4_rel"], 2) + r"\,\%") + ").",
        f"<b>Lift ({L_of['lift']}).</b> " + m(rf"c_L = {q('cl')}") + ", " + m(tn(100 * comp["cl"]["rel"], 2) + r"\,\%")
        + " from the reference " + m(q("cl", ref["val"]["cl"])) + "; extrapolated " + m(q("cl", comp["cl"]["extrap"])) + " ("
        + m(tn(100 * comp["cl"]["diff_ext"] / comp["cl"]["ref"], 2) + r"\,\%") + "). The paper's OpenFOAM level " + str(top)
        + " is " + m(tn(100 * comp["cl"]["of4_rel"], 2) + r"\,\%") + " off. The lift is a small difference of pressure and "
        "friction forces (" + m(tf(fine["cl_p"], 5)) + " and " + m(tf(fine["cl_v"], 5)) + ") and converges from far above: "
        + ", ".join(m(tf(v, 4)) for v in cld) + " on L1-" + fl + ".",
        "<b>Mesh convergence.</b> " + ", ".join(m(SYM[k]) + ": observed order " + (m(tf(ms[k]["p"], 2)) if ms[k]["monotonic"] else "not monotonic")
                                                for k in QK) + " on L2-L4 (design order 2); GCI of " + fl + " "
        + m(tn(100 * ms["cd"]["gci"], 2) + r"\,\%") + " (drag) and " + m(tn(100 * ms["cl"]["gci"], 2) + r"\,\%") + " (lift).",
        f"<b>Flow ({L_of['flow']}).</b> " + ("Behind the cylinder the fluid turns back over " + m(rf"{tf(wk['Lr'], 2)}\,D")
                                            + " (mid-plane, Figure {fig:flow}); the fastest backflow is "
                                            + m(rf"{tf(-wk['u_min'], 3)}\,\mathrm{{m/s}}") + ", " + m(tn(-wk["u_min"] / th["Ubar"] * 100, 2) + r"\,\%")
                                            + " of the mean inflow." if wk else "picture not produced."),
        "<b>Converged and conserving.</b> Over the last " + m(tn(100 * cv["window"]) + r"\,\%") + " of the iterations "
        + m(SYM["cd"]) + " changed by at most " + m(tn(max(r["dcd"] for r in runs.values()), 2)) + " and " + m(SYM["cl"])
        + " by " + m(tn(max(r["dcl"] for r in runs.values()), 2)) + "; the momentum balance of the channel closes to "
        + m(tn(abs(100 * fine["balance"]), 2) + r"\,\%") + " of the drag on " + fl + " ("
        + m(tn(abs(100 * fine["balance_visc"]), 2) + r"\,\%") + " with the inlet friction it neglects); the side force is "
        + m(tn(max(r["side"] for r in runs.values()), 2)) + " of the drag (mid-plane symmetry).",
        "<b>GPU question.</b> " + (gpu["headline"] if gpu else "See the separate GPU study.")
        + " Details: <a href='gpu_study.html'>GPU study report</a>.",
    ] + [f"<b>Rule failed:</b> {x}." for x in fail] + [f"<b>Caveat:</b> {x}." for x in warn]

    ofs, ffs = ref["seq"]["OF"], ref["seq"]["FeatFlow"]
    comp_findings = [
        "<b>The reference.</b> Braack and Richter give " + m(rf"c_D = {q('cd', ref['val']['cd'])}") + " and "
        + m(rf"c_L = {q('cl', ref['val']['cl'])}") + " without an uncertainty (printed digits: " + m(tn(ref["digits"]["cd"], 1))
        + " and " + m(tn(ref["digits"]["cl"], 1)) + "). As its uncertainty this report uses the distance to the finest "
        "result of an independent high-order code, FeatFlow level " + str(top) + ": " + m(tn(ref["unc"]["cd"], 2)) + " and "
        + m(tn(ref["unc"]["cl"], 2)) + ".",
        "<b>Level by level against the paper's OpenFOAM.</b> Same method family (cell-centred finite volumes, central "
        "differences), similar cell counts (Figure {fig:levels}). Drag error on L1-" + fl + ": "
        + ", ".join(m(tn(100 * (runs[L]["cd"] - ref["val"]["cd"]) / ref["val"]["cd"], 2) + r"\,\%") for L in have)
        + "; paper's OpenFOAM: " + ", ".join(m(tn(100 * (of[lv]["cd"] - ref["val"]["cd"]) / ref["val"]["cd"], 2) + r"\,\%")
                                            for lv in sorted(of)) + ". Lift error: "
        + ", ".join(m(tn(100 * (runs[L]["cl"] - ref["val"]["cl"]) / ref["val"]["cl"], 2) + r"\,\%") for L in have)
        + " against " + ", ".join(m(tn(100 * (of[lv]["cl"] - ref["val"]["cl"]) / ref["val"]["cl"], 2) + r"\,\%") for lv in sorted(of)) + ".",
        "<b>The paper's OpenFOAM drag did not converge cleanly.</b> Its changes from level 2 to 3 and 3 to 4 are "
        + m(f"{of[3]['cd'] - of[2]['cd']:+.4f}") + " and " + m(f"{of[4]['cd'] - of[3]['cd']:+.4f}")
        + (": they do not shrink, so it cannot be extrapolated" if not ofs["cd"]["monotonic"] else
           " (observed order " + m(tf(ofs["cd"]["order"], 2)) + ")")
        + "; its lift converges with observed order " + (m(tf(ofs["cl"]["order"], 2)) if ofs["cl"]["order"] else "undefined")
        + ". This simulation converges monotonically in both on L2-L4. FeatFlow (finite elements, more unknowns per cell) "
        "is the most accurate of the three on every level.",
    ]
    if ms_coarse:
        dx = {k: abs(ms[k]["extrap"] - ms_coarse[k]["extrap"]) for k in QK}
        comp_findings.append(
            "<b>How stable is the extrapolation?</b> Richardson from L1-L3 gives " + m(q("cd", ms_coarse["cd"]["extrap"])) + " and "
            + m(q("cl", ms_coarse["cl"]["extrap"])) + "; from L2-L4 " + m(q("cd", ms["cd"]["extrap"])) + " and "
            + m(q("cl", ms["cl"]["extrap"])) + ". " + " ".join(
                ("For " + m(SYM[k]) + " the two differ by " + m(tn(dx[k], 2)) + (", less than the GCI of " + fl + ": stable."
                 if dx[k] < ms[k]["gci_abs"] else ", more than the GCI of " + fl + " (" + m(tn(ms[k]["gci_abs"], 2))
                 + "): the lift is not yet in its asymptotic range, and its GCI understates the error."))
                for k in QK))
    if sens:
        b3 = runs.get("L3")
        rows_ = [(n, sens[n]) for n in SENS if n in sens]
        l3v = [r for n, r in rows_ if n.startswith("L3")]
        spread3 = (max(r["cl"] for r in l3v + [b3]) - min(r["cl"] for r in l3v + [b3])) if b3 and l3v else float("nan")
        comp_findings.append(
            "<b>Robustness of the lift (sensitivity runs).</b> Same solver, one change at a time: "
            + "; ".join(SENS[n] + ": " + m(rf"c_D = {q('cd', r['cd'])}") + ", " + m(rf"c_L = {q('cl', r['cl'])}") + " ("
                        + m(tn(100 * (r["cl"] / ref["val"]["cl"] - 1), 2) + r"\,\%") + ")" for n, r in rows_)
            + ". On L3 the lift spans " + m(tn(100 * spread3 / ref["val"]["cl"], 2) + r"\,\%") + " of the reference between "
            "these meshes of the same size while the drag moves by at most "
            + m(tn(100 * max(abs(r["cd"] - b3["cd"]) for r in l3v) / b3["cd"], 2) + r"\,\%")
            + ": the lift depends on how the cells are distributed in the O-grid (the flow through the gaps between cylinder "
            "and walls), not only on their number. The convection scheme matters little.")
        if "L4_R2" in sens:
            r2 = sens["L4_R2"]
            comp_findings.append(
                "<b>A second mesh family.</b> With radial ratio 2 the lift goes from " + m(q("cl", sens["L3_R2"]["cl"])) + " (L3) to "
                + m(q("cl", r2["cl"])) + " (L4), " + m(tn(100 * (r2["cl"] / ref["val"]["cl"] - 1), 2) + r"\,\%")
                + " from the reference, while the production family (ratio 4) gives " + m(q("cl")) + ". The two families "
                "approach the reference from " + ("opposite sides" if (r2["cl"] - ref["val"]["cl"]) * comp["cl"]["diff"] < 0 else "the same side")
                + "; their difference on L4, " + m(tn(abs(r2["cl"] - comp["cl"]["sim"]), 2)) + ", is a more honest measure of "
                "the lift's mesh uncertainty than the GCI of one family (" + m(tn(ms["cl"]["gci_abs"], 2)) + "). The drag of the "
                "two families differs by only " + m(tn(abs(r2["cd"] - comp["cd"]["sim"]), 2)) + ".")
    comp_findings.append(
        "<b>Pressure and friction.</b> On " + fl + " the pressure gives " + m(tn(100 * fine["cd_p"] / fine["cd"], 3) + r"\,\%")
        + " of the drag and friction the rest: at " + m("Re = 20") + " friction is a large part of the drag, unlike at high "
        "Reynolds numbers.")

    # ---- pre-flight -------------------------------------------------------------------------------
    pf_runs, pf_checks = [], []
    for pf in preflight:
        pf_runs.append(dict(name=f"{pf['name']} ({pf['cells']:,} cells, {pf['cores']} rank{'s' if pf['cores'] > 1 else ''})",
                            setup=f"{pf['iterations']} SIMPLE iterations from rest",
                            outcome=f"{ST[pf['status']]}; {pf['wall']:.1f} s, " + m(tf(pf['s_per_iter'], 3) + r"\,\mathrm{s}")
                                    + " per iteration"))
        for c in pf["checks"]:
            val = c["value"]
            if isinstance(val, str) and val.startswith("≥"):      # cfdlib.monitor writes a plain '≥' (insight: tooling)
                val = m(r"\ge " + val[1:].strip())
            pf_checks.append(dict(c, name=f"{c['name']} ({pf['name']})", value=val))
        exc = -pf["q_in"] / (th["Ubar"] * H * H) - 1
        pf_checks.append(dict(name=f"Inflow rate ({pf['name']})", status="ok" if abs(exc) < 1e-2 else "warn",
                              value=m(rf"{tn(exc, 2)}") + " relative to " + m(r"\bar U H^2"),
                              meaning="The inlet must deliver the benchmark's flow rate; face-centre values of the profile give a "
                                      "small excess that shrinks with the square of the cell size (4 times per level)."))
        pf_checks.append(dict(name=f"Forces recorded every iteration ({pf['name']})",
                              status="ok" if pf["n_force_rows"] >= pf["iterations"] else "fail",
                              value=f"{pf['n_force_rows']} rows for {pf['iterations']} iterations",
                              meaning="Drag and lift at every iteration are needed for the stopping check."))
        prof = pf.get("profiling", {})
        if "fvMatrix::solve.p" in prof:
            run_t = next(v for k_, v in prof.items() if k_.startswith("time.run()"))["total"]
            pf_checks.append(dict(name=f"Pressure-solve share of the run ({pf['name']})", status="ok",
                                  value=m(tn(100 * prof["fvMatrix::solve.p"]["total"] / run_t, 3) + r"\,\%"),
                                  meaning="Built-in profiling works and splits the time; the pressure solve is the part a GPU "
                                          "solver would take over."))
    pf4 = preflight[-1]
    est4 = pf4["s_per_iter"] * fine["meta"]["iterations"]

    # ---- performance appendix --------------------------------------------------------------------
    perf_rows = []
    for L in have:
        p = runs[L]["meta"]["perf"]
        pr = p.get("profiling", {})
        run_t = next((v["total"] for k_, v in pr.items() if k_.startswith("time.run()")), None)
        share = pr.get("fvMatrix::solve.p", {}).get("total", 0) / run_t if run_t else None
        perf_rows.append([L, f"{p['cells']:,}", str(p["cores"]), f"{p['steps']:,}", f"{p['wall_s']:.0f}",
                          m(tf(1e6 * p["loop_s_per_iter"] / p["cells"], 3)) if p.get("loop_s_per_iter") else "–",
                          m(tf(p["lin_iters"].get("p", float("nan")), 1)),
                          m(tf(100 * share, 1) + r"\,\%") if share else "–", m(tf(p["cpu_efficiency"], 2)) if p["cpu_efficiency"] else "–"])
    perf_notes = [
        "<b>Accuracy runs ran next to the GPU tuning runs</b> (L3 on 8 and L4 on 12 ranks, the tuning runs on 4 ranks plus the "
        "GPU): their wall times are inflated and are not used for any speed comparison. All timing comparisons are in the "
        "<a href='gpu_study.html'>GPU study</a>, measured with nothing else running.",
        "<b>Where the time goes on the CPU.</b> Pressure share from OpenFOAM's built-in profiling (" + m(r"\texttt{fvMatrix::solve.p}")
        + " over the whole iteration loop); the rest is the momentum solve, matrix assembly and the force output every iteration.",
    ] + ((gpu["notes"]) if gpu else [])

    # ---- assumptions ------------------------------------------------------------------------------
    assumptions = [
        dict(name="Steady flow", status="standard",
             check="The problem states that the flow is steady at " + m(rf"Re = {tn(dv['Re'], 3)}") + "; a steady solver cannot "
                   "show an instability. Each run converged to a fixed point (forces change by less than "
                   + m(tn(max(r["dcd"] for r in runs.values()), 1)) + " over its last iterations), which a steady solver would not "
                   "reach if the steady solution were strongly unstable."),
        dict(name="Laminar, incompressible, Newtonian", status="standard",
             check=m(rf"Re = \bar U D/\nu = {tn(dv['Re'], 3)}") + ": laminar. Constant density and viscosity are given; "
                   + m(rf"U_m = {tn(Um)}\,\mathrm{{m/s}}") + " is far below the speed of sound of any liquid or gas."),
        dict(name="Mid-plane symmetry", status="ok",
             check="Not imposed (the whole channel is computed). The side force is at most " + m(tn(max(r["side"] for r in runs.values()), 2))
                   + " of the drag" + (", and the velocity across the mid-plane at most " + m(rf"{tn(wk['w_max'], 2)}\,\mathrm{{m/s}}") if wk else "")
                   + ": round-off level, as theory requires."),
        dict(name="Outlet far enough downstream", status="standard",
             check="The outlet is " + m(rf"{tn((g['L']['value'] - g['xc']['value']) / D, 3)}\,D") + " behind the cylinder, as in the "
                   "benchmark; the reversed flow ends " + (m(rf"{tf(wk['Lr'], 2)}\,D") if wk else "close") + " behind it. "
                   "Zero-gradient velocity with zero pressure equals the stress-free outflow for parallel outflow."),
        dict(name="Cylinder surface as straight faces", status="imposed",
             check="The circle is represented by " + str(4 * spec[fl]["nc"]) + " straight faces on the finest mesh (points on the "
                   "exact circle); the lost area is a second-order error included in the mesh study."),
        dict(name="Inflow profile from face centres", status="ok",
             check="The inflow rate is " + m(tn(fine["inflow_excess"], 2)) + " (relative) above " + m(r"\bar U H^2") + " on " + fl
                   + ", shrinking about 4 times per level (" + ", ".join(m(tn(runs[L]["inflow_excess"], 2)) for L in have) + ")."),
    ]

    ok_word = {"PASS": "passed", "WARN": "passed with caveats", "FAIL": "FAILED"}[verdict]
    results = dict(
        title=problem["title"], date=dt.date.today().isoformat(), verdict=verdict, comparison_type="benchmark",
        summary=dict(
            headline=(f"<b>Benchmark comparison {ok_word}.</b> On the finest mesh ({fl}, {fm['cells']:,} cells) the drag is "
                      + m(q("cd")) + " (reference " + m(q("cd", ref["val"]["cd"])) + ", " + m(tn(100 * comp["cd"]["rel"], 2) + r"\,\%")
                      + ") and the lift " + m(q("cl")) + " (" + m(q("cl", ref["val"]["cl"])) + ", " + m(tn(100 * comp["cl"]["rel"], 2) + r"\,\%")
                      + "); mesh-extrapolated: " + m(tn(100 * comp["cd"]["diff_ext"] / comp["cd"]["ref"], 2) + r"\,\%")
                      + " and " + m(tn(100 * comp["cl"]["diff_ext"] / comp["cl"]["ref"], 2) + r"\,\%") + ". "
                      + " ".join(("The drag" if k == "cd" else "The lift") + (" passes" if comp[k]["status"] == "ok" else
                                 " misses" if comp[k]["status"] == "fail" else " is near")
                                 + " the band fixed beforehand (" + m(tf(comp[k]["ratio"], 2)) + " times its width)." for k in QK)
                      + (" The lift is very sensitive to the cell distribution around the cylinder (sensitivity runs); "
                         "the paper's OpenFOAM was " + m(tn(100 * comp["cl"]["of4_rel"], 2) + r"\,\%") + " off on its finest level."
                         if comp["cl"]["status"] != "ok" else "")),
            answer_columns=["Requested", "Reference (Braack and Richter 2006)", f"Simulation ({fl})", "Difference (÷ band)"],
            answers=answers, findings=findings,
            limitations=[
                "The reference is itself a numerical solution (adaptive finite elements), not an experiment; its uncertainty is "
                "estimated from a second code (FeatFlow level " + str(top) + ").",
                "Four meshes up to " + f"{fm['cells']:,}" + " cells; a fifth level (about " + f"{8 * fm['cells'] / 1e6:.0f}"
                + " million cells) would need about " + f"{8 * 1.4:.0f}" + " GB of memory and hours per run on this machine.",
                "The flow picture (c) has no reference to compare with; the length of the reversed-flow region is reported "
                "for information only.",
                "Steady laminar model as the benchmark; whether this steady state is stable (would be reached by a real flow) is "
                "not tested by a steady solver.",
                "Wall times of the accuracy runs are inflated (they shared the machine with GPU tuning runs); speed numbers are "
                "in the GPU study.",
            ]),
        problem=dict(
            description=problem["description"].strip(), setup=problem.get("setup", []),
            given=[dict(label=v["label"], symbol=v["symbol"], value=m(tn(v["value"])), unit=v["unit"]) for v in g.values()],
            wanted=wanted,
            reference="Braack and Richter (2006) drag and lift, and the level 1-4 results of OpenFOAM, CFX and FeatFlow of "
                      "Bayraktar, Mierka and Turek (2012), parsed from the FeatFlow benchmark pages into <code>"
                      + problem["reference"]["file"] + "</code>."),
        assumptions=dict(rows=assumptions),
        theory=dict(
            intro="Flow around a cylinder in a duct has <b>no exact solution</b>: the convection term makes the equations "
                  "nonlinear and the geometry is three-dimensional. The comparison therefore uses the benchmark's reference "
                  "values. Theory still fixes the scales (one number, " + m("Re") + "), the inflow (its mean and how far it is "
                  "from developed duct flow), what must hold by symmetry (no side force), and a conservation law used as a check "
                  "(the momentum balance of the channel). The algebra marked below is checked by SymPy.",
            steps=th["steps"], derived=th["derived"],
            self_checks=[dict(name=k, ok=bool(v)) for k, v in th["self_checks"].items()]),
        decisions=load_yaml(HERE / "decisions.yaml"),
        preflight=dict(
            purpose="Before the production runs, short runs on the coarsest and the finest mesh check that the setup runs, "
                    "stays stable, records every output, delivers the right inflow and fits the time budget.",
            runs=pf_runs, checks=pf_checks,
            conclusion=("All pre-flight checks passed" if all(p["status"] != "fail" for p in preflight) else "A pre-flight check FAILED")
                       + ". The " + pf4["name"] + " pre-flight projected at least " + m(tn(pf4["est_full_runtime"] / 60, 2) + r"\,\mathrm{min}")
                       + " to the residual targets (a lower bound from the early residual decay); started from the converged L3 "
                       "flow, the production run needed " + str(fine["meta"]["iterations"]) + " iterations, about "
                       + m(tn(est4 / 60, 2) + r"\,\mathrm{min}") + " at the pre-flight speed on 16 ranks."),
        simulation=dict(
            model=[
                dict(aspect="Geometry", real="Square channel " + m(r"2.5 \times 0.41 \times 0.41\,\mathrm{m}") + ", cylinder "
                     + m(r"D = 0.1\,\mathrm{m}") + " from wall to wall",
                     model="The same, meshed in full (no symmetry plane); the circle as straight faces between points on it."),
                dict(aspect="Inflow", real="Product of two parabolas, maximum " + m(rf"U_m = {tn(Um)}\,\mathrm{{m/s}}"),
                     model="The profile evaluated at each inlet face centre; flow rate within " + m(tn(abs(fine["inflow_excess"]), 2))
                           + " (relative) of " + m(r"\bar U H^2") + " on " + fl + "."),
                dict(aspect="Outflow", real="Free (stress-free) outflow", model="Zero normal velocity gradient, zero pressure."),
                dict(aspect="Steady state", real="The flow after all transients have died out",
                     model="Steady SIMPLEC iterations until the residuals and the forces stop changing (physical check)."),
                dict(aspect="Forces", real="Pressure and friction on the cylinder",
                     model="OpenFOAM <code>forces</code> on the cylinder patch every iteration; coefficients with "
                           + m(r"\tfrac12\rho\bar U^2 D H") + " computed in the analysis."),
            ],
            meshes=dict(
                intro="Four block-structured O-grid meshes; every cell count doubles from one level to the next in all three "
                      "directions (8 times the cells). " + m("h_w") + ": radial size of the cells on the cylinder; "
                      + m(r"\delta = D/\sqrt{Re}") + " is the boundary-layer scale; " + m(r"h_z") + ": cell size at the side walls.",
                head=["Mesh", "Cells", "Faces around cylinder", "Cells across " + m("z"), "Wall cell " + m("h_w"), m(r"\delta/h_w"),
                      m(r"h_z") + " at side wall", "Max non-orth.", "Iterations"],
                rows=mesh_rows),
            results=dict(head=["Mesh", "Cells", m("c_D"), m("c_L"), m("c_D") + " pressure / friction",
                               "Paper OpenFOAM cells", "Paper OF " + m("c_D"), "Paper OF " + m("c_L")],
                         rows=res_rows),
            health=dict(
                head=["Run", "Iterations", "Log", m(r"\Delta c_D") + " / " + m(r"\Delta c_L"), "Converged", "Momentum",
                      "Momentum + inlet friction", "Side force", "Inflow excess", "Wall s (ranks)"],
                rows=health_rows,
                notes=[
                    "Log: automatic log checks (no crash, no NaN, residual and continuity-error trends; cfdlib.monitor).",
                    m(r"\Delta c_D") + ", " + m(r"\Delta c_L") + ": largest change over the last " + m(tn(100 * cv["window"]) + r"\,\%")
                    + " of the iterations; tolerances " + m(tn(cv["cd_change"])) + " and " + m(tn(cv["cl_change"])) + ", fixed before "
                    "the runs. Every run stopped on its residual targets (" + m(tn(numerics["simple"]["residual_p"])) + " pressure, "
                    + m(tn(numerics["simple"]["residual_U"])) + " velocity).",
                    "Momentum: imbalance of the steady x-momentum balance (Theory, step 5) divided by the drag force; tolerance "
                    + m(tn(100 * cv["momentum_balance_rel"]) + r"\,\%") + " on the finest mesh. The next column adds the "
                    "viscous force through the inlet, which the balance neglects: zero for the continuous problem, but not for "
                    "the discrete one (one-sided gradient at an inlet whose profile is not developed duct flow; computed from the "
                    "fields, Insights). Side force: " + m(r"|F_z|/F_D") + ".",
                    "Runs L3_*, L4_R2: sensitivity runs (one change each, see the Comparison).",
                    "Wall times include waiting for cores shared with the GPU tuning runs (Performance appendix).",
                ])),
        comparison=dict(
            kind="Benchmark: the simulation is compared with the reference values of Braack and Richter (adaptive finite "
                 "elements). Agreement is judged against a band: the GCI of the finest mesh plus the reference uncertainty "
                 "(distance to FeatFlow level " + str(top) + "). The benchmark paper's own OpenFOAM results are shown for comparison.",
            definitions=[
                dict(name="Force coefficients", latex=r"c_D = \frac{2F_D}{\rho\,\bar U^2 D H},\qquad c_L = \frac{2F_L}{\rho\,\bar U^2 D H}",
                     meaning=m(r"\bar U = \tfrac49 U_m") + ", " + m("H") + " the cylinder length; forces are the converged values."),
                dict(name="Reference uncertainty", latex=r"U_{\mathrm{ref}} = |q_{\mathrm{ref}} - q_{\mathrm{FeatFlow},4}|",
                     meaning="Distance to an independent high-order solution on its finest level."),
                dict(name="Simulation uncertainty", latex=r"U_{\mathrm{sim}} = 1.25\,\frac{|q_{L4} - q_{L3}|}{2^p - 1}",
                     meaning="GCI of the finest mesh (Celik et al. 2008) from L2, L3, L4 with their observed order " + m("p") + "."),
                dict(name="Verdict per number", latex=r"|q_{L4} - q_{\mathrm{ref}}| \le U_{\mathrm{sim}} + U_{\mathrm{ref}}",
                     meaning="Inside the band: ok; up to twice the band: warning; beyond: fail."),
            ],
            head=["Number", "Reference", m(r"U_{\mathrm{ref}}"), f"Simulation ({fl})", m(r"U_{\mathrm{sim}}"), "Difference",
                  "÷ band, status", "Richardson (rel. diff.)", f"Paper OpenFOAM L{top} (rel. diff.)"],
            rows=comp_rows, findings=comp_findings),
        insights=load_yaml(HERE / "insights.yaml"),
        figures=figures,
        reproducibility=dict(
            commands=[
                dict(cmd=f"python -m cases.{HERE.name}.run preflight", what="Short test runs on L1 (1 core) and L4 (16 ranks)"),
                dict(cmd=f"python -m cases.{HERE.name}.run production L1 L2 L3", what="Converged runs, each from the converged coarser flow"),
                dict(cmd=f"python -m cases.{HERE.name}.run production L4", what="Finest mesh, 12 MPI ranks"),
                dict(cmd=f"python -m cases.{HERE.name}.run flow L4", what="Mid-plane velocity for the picture"),
                dict(cmd=f"python -m cases.{HERE.name}.run inletvisc L1 L2 L3 L4", what="Viscous force through the inlet (balance diagnostic)"),
                dict(cmd=f"python -m cases.{HERE.name}.run variant L3_LU L3 \"schemes.div_U=bounded Gauss linearUpwind grad(U)\"",
                     what="Sensitivity: convection scheme (8 ranks)"),
                dict(cmd=f"python -m cases.{HERE.name}.run variant L3_R2 L3 mesh.radial_ratio=2", what="Sensitivity: O-grid radial ratio 2 (also L3_R8 with 8)"),
                dict(cmd=f"python -m cases.{HERE.name}.run variant L4_R2 L4 from=L3_R2 ranks=12 mesh.radial_ratio=2",
                     what="Second mesh family on L4, started from L3_R2"),
                dict(cmd=f"python -m cases.{HERE.name}.analyze --report", what="Analysis, figures and this report"),
                dict(cmd=f"python -m cases.{HERE.name}.gpu_study", what="GPU study report (its own reproducibility section lists the runs)"),
            ],
            files=[
                dict(path=f"cases/{HERE.name}/problem.yaml", what="The problem as given"),
                dict(path=f"cases/{HERE.name}/reference/", what="Benchmark values (parsed) and copies of the benchmark pages"),
                dict(path=f"cases/{HERE.name}/numerics.yaml", what="Meshes, solver settings, stopping and acceptance rules"),
                dict(path=f"cases/{HERE.name}/decisions.yaml", what="Every choice, with the OpenFOAM settings it controls"),
                dict(path=f"cases/{HERE.name}/insights.yaml", what="Workflow insights logged during the work"),
                dict(path=f"cases/{HERE.name}/template/", what="OpenFOAM case; blockMeshDict filled from run.mesh_spec"),
                dict(path=f"cases/{HERE.name}/run.py", what="Mesh generator, runs, mapping, MPI, GPU configurations, timing"),
                dict(path=f"cases/{HERE.name}/theory.py", what="What theory can say (SymPy)"),
            ],
            run_dirs=[
                dict(path=f"{fm['run_root']}/<level>", what="One directory per level (L1-L4)"),
                dict(path=f"{fm['run_root']}/pf_<level>", what="Pre-flight runs"),
                dict(path=f"results/{HERE.name}/", what="preflight.json, runs/, perf/, timing/, analysis.json, results.json, both reports"),
            ],
            environment=report.environment_info(),
            notes=[
                "L3 (8 ranks) and L4 (12 ranks) ran at the same time as GPU tuning runs (4 ranks and the GPU); L1 and L2 ran alone.",
                "MPI runs: <code>decomposePar</code> (scotch), <code>mpirun -np N simpleFoam -parallel</code>, "
                "<code>reconstructPar -latestTime</code>; the finer levels start from <code>mapFields</code> of the coarser one.",
            ],
        ),
        performance=dict(
            intro="Run-time data of the accuracy runs. Wall microseconds per cell per iteration inside the iteration loop; "
                  "pressure iterations are the mean GAMG iterations per solve; pressure share is the fraction of the loop time "
                  "spent in the pressure solve (built-in profiling).",
            head=["Run", "Cells", "Ranks", "Iterations", "Wall (s)", m(r"\mu s") + "/cell/iter", m("p") + " iters", "Pressure share",
                  "CPU eff."],
            rows=perf_rows, notes=perf_notes),
        data=dict(analysis="analysis.json", verdict_rules=dict(fail=fail, warn=warn)),
    )
    out = report.render(results, svgs, OUT / "report.html", case_dir=fm["case_dir"])
    print("Report:", out)


def load_gpu_summary():
    """Headline and notes written by gpu_study.py (if it has run)."""
    p = OUT / "gpu_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
COL = dict(L1=plots.SERIES[3], L2=plots.SERIES[2], L3=plots.SERIES[1], L4=plots.SERIES[0])


def _iters_fig(runs, have, cv):
    fig, (ax, bx) = plots.wide(2, 3.3)
    for L in have:
        r = runs[L]
        for a, y in ((ax, r["cdh"]), (bx, r["clh"])):
            dv = np.abs(y - y[-1])
            dv[dv == 0] = np.nan
            a.semilogy(r["it"], dv, color=COL[L], lw=1.3, label=L)
    ax.axhline(cv["cd_change"], color=plots.INK2, ls="--", lw=0.9)
    bx.axhline(cv["cl_change"], color=plots.INK2, ls="--", lw=0.9)
    ax.set_ylabel(r"$|c_D - c_{D,\mathrm{final}}|$")
    bx.set_ylabel(r"$|c_L - c_{L,\mathrm{final}}|$")
    for a in (ax, bx):
        a.set_xlabel("Iteration")
    bx.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return plots.to_svg(fig, "iters")


def _levels_fig(runs, have, ref, ms, sens=None):
    fig, (ax, bx) = plots.wide(2, 3.6)
    marks = {"OF": ("s", plots.SERIES[1]), "CFX": ("^", plots.SERIES[2]), "FeatFlow": ("D", plots.SERIES[3])}
    cells = [runs[L]["meta"]["cells"] for L in have]
    fam2 = [n for n in ("L3_R2", "L4_R2") if sens and n in sens]
    for a, k in ((ax, "cd"), (bx, "cl")):
        r0 = ref["val"][k]
        tr = (lambda v: v) if k == "cd" else (lambda v: 100 * (v - r0) / r0)     # lift: signed error in %
        for s, (mk, c) in marks.items():
            lv = sorted(ref["by"][s])
            a.plot([ref["by"][s][i]["cells"] for i in lv], [tr(ref["by"][s][i][k]) for i in lv], mk + "--", color=c, ms=4.5,
                   lw=1, label=f"paper: {SOFT[s]}")
        a.plot(cells, [tr(runs[L][k]) for L in have], "o-", color=plots.SERIES[0], ms=5.5, lw=1.8, label="this simulation")
        if len(fam2) >= 1:
            a.plot([sens[n]["meta"]["cells"] for n in fam2], [tr(sens[n][k]) for n in fam2], "o:", mfc="none",
                   color=plots.SERIES[0], ms=6, lw=1.2, label="same, radial ratio 2")
        a.set_xscale("log")
        a.set_xlabel("Cells")
        if ms:
            if k == "cd":
                a.errorbar([cells[-1] * 3.0], [ms[k]["extrap"]], yerr=[ms[k]["gci_abs"]], fmt="D", mfc="none",
                           color=plots.SERIES[0], ms=6, capsize=3, label="Richardson (L2-L4)")
            else:
                a.errorbar([cells[-1] * 3.0], [tr(ms[k]["extrap"])], yerr=[100 * ms[k]["gci_abs"] / r0], fmt="D", mfc="none",
                           color=plots.SERIES[0], ms=6, capsize=3, label="Richardson (L2-L4)")
        a.axhline(tr(r0), color=plots.INK2, ls=":", lw=1, label="reference")
        if k == "cd":
            a.set_ylabel(r"$c_D$")
            a.set_ylim(min(5.6, r0 - 0.6), r0 + 0.15)
        else:
            a.set_yscale("symlog", linthresh=1.0)
            a.set_yticks([-10, -1, 0, 1, 10, 100])
            a.set_yticklabels(["-10", "-1", "0", "1", "10", "100"])
            a.set_ylabel(r"$c_L$ error (%)")
    bx.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return plots.to_svg(fig, "levels")


def _flow_fig(wk, g):
    from matplotlib.patches import Circle
    D, xc, yc, H = (g[k]["value"] for k in ("D", "xc", "yc", "H"))
    w, tri, sel = wk["w"], wk["tri"], wk["sel"]
    fig, (ax, bx) = plots.wide(2, 3.0, gridspec_kw=dict(width_ratios=[2.6, 1]))
    ax.grid(False)
    u = w["u"][sel]
    vmax = float(np.max(u))
    vneg = 1.3 * max(-float(np.min(u)), 1e-3)                # stretch the small negative range so the backflow shows
    lev = np.concatenate([np.linspace(-vneg, 0, 6), np.linspace(0, vmax, 18)[1:]])
    from matplotlib.colors import TwoSlopeNorm
    cs = ax.tricontourf(tri, u, levels=lev, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vneg, vcenter=0.0, vmax=vmax), extend="both")
    ax.tricontour(tri, u, levels=[0.0], colors=plots.INK, linewidths=1.0)
    ax.add_patch(Circle((xc, yc), D / 2, facecolor="#c3c2b7", edgecolor=plots.INK, lw=0.8))
    ax.set_xlim(xc - 0.25, xc + 0.75)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    cb = fig.colorbar(cs, ax=ax, shrink=0.85, ticks=[round(-vneg / 1.3, 3), 0, 0.2, 0.4, 0.6])
    cb.set_label(r"$u$ (m/s)")
    bx.plot(wk["xs"] - xc, wk["u_line"], color=plots.SERIES[0], lw=1.6)
    bx.axhline(0, color=plots.MUTED, lw=0.8)
    bx.axvline(wk["x_end"] - xc, color=plots.INK2, ls="--", lw=0.9)
    bx.set_xlabel(r"$x - x_c$ (m)")
    bx.set_ylabel(r"$u$ (m/s) on the wake line")
    return plots.to_svg(fig, "flow")


if __name__ == "__main__":
    main()
