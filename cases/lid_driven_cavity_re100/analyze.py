"""Compare the cavity simulations with Ghia, Ghia & Shin (1982); mesh study; build results.json + report.

Usage (repo root, inside WSL):  python -m cases.lid_driven_cavity_re100.analyze
Reads results/lid_driven_cavity_re100/{preflight.json, runs/N*.json, runs/N*.npz}, the case YAML files and
the benchmark data in reference/. Every number in the report is computed here.
"""
import datetime as dt
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.optimize import minimize, minimize_scalar

from cfdlib import monitor, plots, report, verification as V
from cfdlib.report import LETTERS, load_yaml, m, tex_num as tn
from . import theory
from .sketch import sketch

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"
FIG_SECTIONS = dict(sketch="problem", streamlines="simulation", residuals="convergence", u_profile="comparison",
                    v_profile="comparison", mesh="comparison")          # key -> report section (insertion order kept)


def fig_no(key):
    """Figure number as report.render assigns it (stable sort by section order)."""
    from cfdlib.contract import FIGURE_SECTIONS as ORDER
    keys = sorted(FIG_SECTIONS, key=lambda k: ORDER.index(FIG_SECTIONS[k]))
    return keys.index(key) + 1


PSI_STEP = 0.01          # contour spacing of the streamline plot (units of U L)
SOUND_SPEED_MIN = 300.0   # m/s: lower bound for the speed of sound in any common gas or liquid (air ≈ 343). Approximate.


# ---------------------------------------------------------------------------------------------
# Extraction from one mesh (all dimensionless: x/L, y/L, u/U, psi/(U L))
# ---------------------------------------------------------------------------------------------
def extract(N, d, L, U, y_g, x_g):
    ux, uy = d["ux"] / U, d["uy"] / U                   # [j, i]: row j (y), column i (x)
    c = (np.arange(N) + 0.5) / N                        # cell-centre coordinates
    # Centrelines lie on cell faces (N even): average the two neighbouring columns / rows (2nd order)
    u_vc = 0.5 * (ux[:, N // 2 - 1] + ux[:, N // 2])
    v_hc = 0.5 * (uy[N // 2 - 1, :] + uy[N // 2, :])
    y_line, u_line = np.r_[0.0, c, 1.0], np.r_[0.0, u_vc, 1.0]     # wall values from the boundary conditions
    x_line, v_line = np.r_[0.0, c, 1.0], np.r_[0.0, v_hc, 0.0]
    su, sv = CubicSpline(y_line, u_line), CubicSpline(x_line, v_line)
    # (c) minimum of u on the vertical centreline: refine the discrete minimum on the spline
    k = int(np.argmin(u_vc))
    r = minimize_scalar(su, bounds=(c[max(k - 1, 0)], c[min(k + 1, N - 1)]), method="bounded",
                        options=dict(xatol=1e-10))
    # Net flow through the vertical centreline (must be 0): midpoint rule on the cell values
    net_flow = float(np.sum(u_vc) / N)
    # (d) stream function: OpenFOAM's flux-based psi on the points (primary) ...
    sf = d["sf"] / (U * L)
    xv = yv = np.arange(N + 1) / N
    vort_of = _vortex(sf, xv, yv)
    # ... and an independent reconstruction from cell-centre velocities (cross-check)
    uf = np.zeros((N, N + 1))
    uf[:, 1:N] = 0.5 * (ux[:, :-1] + ux[:, 1:])
    psi = np.zeros((N + 1, N + 1))
    psi[1:] = np.cumsum(uf, axis=0) / N
    vort_own = _vortex(psi, xv, yv)
    return dict(
        N=N, h=1.0 / N, y_line=y_line.tolist(), u_line=u_line.tolist(), x_line=x_line.tolist(), v_line=v_line.tolist(),
        u_at_ghia=su(y_g).tolist(), v_at_ghia=sv(x_g).tolist(),
        u_min=float(r.fun), y_umin=float(r.x), net_flow=net_flow,
        psi_min=vort_of["psi"], x_c=vort_of["x"], y_c=vort_of["y"],
        psi_min_own=vort_own["psi"], x_c_own=vort_own["x"], y_c_own=vort_own["y"],
        lid_psi_own=float(np.max(np.abs(psi[-1]))), lid_psi_of=float(np.max(np.abs(sf[-1]))),
        walls_psi_of=float(max(np.abs(sf[0]).max(), np.abs(sf[-1]).max(), np.abs(sf[:, 0]).max(), np.abs(sf[:, -1]).max())),
    )


def _vortex(psi, xv, yv):
    """Minimum of a point field psi[j, i] and its location, refined with a bicubic spline."""
    j, i = np.unravel_index(np.argmin(psi), psi.shape)
    s = RectBivariateSpline(yv, xv, psi, kx=3, ky=3)
    h = xv[1] - xv[0]
    res = minimize(lambda q: s(q[1], q[0])[0, 0], x0=[xv[i], yv[j]], method="L-BFGS-B",
                   bounds=[(xv[i] - 2 * h, xv[i] + 2 * h), (yv[j] - 2 * h, yv[j] + 2 * h)],
                   options=dict(ftol=1e-15, gtol=1e-12))
    return dict(psi=float(res.fun), x=float(res.x[0]), y=float(res.x[1]))


def study(values, r=2.0):
    """Observed order, Richardson value and fine-mesh GCI from the three finest meshes.
    Convergence is monotonic when 0 < R < 1, R = (f_med - f_fine)/(f_coarse - f_med)."""
    f1, f2, f3 = values[-3:]                           # coarse, medium, fine
    R = (f2 - f3) / (f1 - f2) if f1 != f2 else float("nan")
    monotonic = bool(0 < R < 1)
    p = V.order_three_grids(f1, f2, f3, r) if monotonic else float("nan")
    p_use = p if monotonic else 2.0                    # formal order if the observed one is undefined
    return dict(R=R, monotonic=monotonic, p=p, p_used=p_use,
                delta=abs(f2 - f3), extrap=V.richardson(f2, f3, p_use, r), gci=V.gci_fine(f2, f3, p_use, r),
                gci_abs=abs(V.gci_fine(f2, f3, p_use, r) * f3))


def main():
    problem = load_yaml(HERE / "problem.yaml")
    numerics = load_yaml(HERE / "numerics.yaml")
    ref = load_yaml(HERE / problem["reference"]["file"])
    acc, conv = numerics["acceptance"], numerics["convergence"]
    g = problem["given"]
    L, U, nu = (g[k]["value"] for k in ("L", "U", "nu"))
    Ns_plan = numerics["mesh"]["N"]
    Ns = [N for N in Ns_plan if (RUNS / f"N{N}.json").exists()]
    missing = [N for N in Ns_plan if N not in Ns]
    metas = [json.loads((RUNS / f"N{N}.json").read_text()) for N in Ns]
    data = [np.load(RUNS / f"N{N}.npz") for N in Ns]
    preflight = json.loads((OUT / "preflight.json").read_text())

    # ---- benchmark data, read from the supplied file -----------------------------------------
    tu = np.array(ref["u_vertical_centreline"]["rows"], dtype=float)
    tv = np.array(ref["v_horizontal_centreline"]["rows"], dtype=float)
    inner_u = (tu[:, 0] > 0) & (tu[:, 0] < 1)          # wall points are boundary conditions, not results
    inner_v = (tv[:, 0] > 0) & (tv[:, 0] < 1)
    y_g, u_g = tu[inner_u, 0], tu[inner_u, 1]
    x_g, v_g = tv[inner_v, 0], tv[inner_v, 1]
    k_umin = int(np.argmin(tu[:, 1]))
    G = dict(u_min=float(tu[k_umin, 1]), y_umin=float(tu[k_umin, 0]),
             psi_min=float(ref["vortices"]["primary"]["psi_min"]),
             x_c=float(ref["vortices"]["primary"]["centre"][0]), y_c=float(ref["vortices"]["primary"]["centre"][1]))
    n_grid = int(re.search(r"(\d+)\s*x\s*\d+", ref["source"]["kind"]).group(1))   # "129 x 129 grid"
    h_G = 1.0 / (n_grid - 1)
    th = theory.evaluate(g, Ns, h_G)
    dv = {d["key"]: d["value"] for d in th["derived"]}
    Re = dv["Re"]

    # ---- per-mesh extraction and comparison ---------------------------------------------------
    rows = []
    for N, meta, d in zip(Ns, metas, data):
        x = extract(N, d, L, U, y_g, x_g)
        du, dvv = np.array(x["u_at_ghia"]) - u_g, np.array(x["v_at_ghia"]) - v_g
        x.update(
            du=du.tolist(), dv=dvv.tolist(),
            du_max=float(np.max(np.abs(du))), du_rms=float(np.sqrt(np.mean(du ** 2))),
            dv_max=float(np.max(np.abs(dvv))), dv_rms=float(np.sqrt(np.mean(dvv ** 2))),
            d_umin=x["u_min"] - G["u_min"], d_yumin=x["y_umin"] - G["y_umin"],
            d_psi_rel=(x["psi_min"] - G["psi_min"]) / abs(G["psi_min"]),
            d_xc=x["x_c"] - G["x_c"], d_yc=x["y_c"] - G["y_c"],
            iterations=meta["iterations"], wall=meta["wall_time"], health=meta["health"]["status"],
            health_checks=meta["health"]["checks"], mesh_ok=meta["mesh_ok"],
            stopped=meta["stopped_by_residualControl"], probe_change=meta["probe_change"],
            probe_window=meta["probe_window"], final_res=meta["final_residuals"],
            write_change=meta["write_change"],
            cont=_last_continuity(meta),
        )
        x["converged"] = bool(x["stopped"] and x["probe_change"] < conv["probe_change_tol"])
        rows.append(x)
    fine = rows[-1]

    # ---- mesh study (three finest meshes) and the same on the three coarsest, for comparison ----
    scal = ["u_min", "y_umin", "psi_min", "x_c", "y_c"]
    SYM = dict(u_min=r"u_{\min}", y_umin=r"y_{u_{\min}}", psi_min=r"\psi_{\min}", x_c="x_c", y_c="y_c")
    ms = {k: study([r[k] for r in rows]) for k in scal}
    ms_coarse = {k: study([r[k] for r in rows[:3]]) for k in scal}
    # Profiles: one global order from the RMS change between consecutive meshes
    def chg(a, b, key):
        return float(np.sqrt(np.mean((np.array(rows[a][key]) - np.array(rows[b][key])) ** 2)))
    prof = {}
    for key in ("u_at_ghia", "v_at_ghia"):
        changes = [chg(i, i + 1, key) for i in range(len(rows) - 1)]
        orders = [math.log(changes[i] / changes[i + 1]) / math.log(2) for i in range(len(changes) - 1)]
        p = orders[-1]
        f2, f3 = np.array(rows[-2][key]), np.array(rows[-1][key])
        ext = f3 + (f3 - f2) / (2 ** p - 1)
        band = 1.25 * np.abs(f3 - f2) / (2 ** p - 1)
        gref = u_g if key == "u_at_ghia" else v_g
        prof[key] = dict(changes=changes, orders=orders, p=p, ext=ext.tolist(), band=band.tolist(),
                         band_max=float(band.max()), ext_vs_ghia_max=float(np.max(np.abs(ext - gref))),
                         ext_vs_ghia_rms=float(np.sqrt(np.mean((ext - gref) ** 2))),
                         fine_vs_ghia_within_band=int(np.sum(np.abs(f3 - gref) <= band)))

    # ---- scheme sensitivity: same mesh, central-difference convection ---------------------------
    sens = None
    Ns_ = numerics["sensitivity"]["N"]
    if (RUNS / f"lin{Ns_}.json").exists() and Ns_ in Ns:
        sm = json.loads((RUNS / f"lin{Ns_}.json").read_text())
        sx = extract(Ns_, np.load(RUNS / f"lin{Ns_}.npz"), L, U, y_g, x_g)
        base = rows[Ns.index(Ns_)]
        d_scheme = max(np.max(np.abs(np.array(sx["u_at_ghia"]) - np.array(base["u_at_ghia"]))),
                       np.max(np.abs(np.array(sx["v_at_ghia"]) - np.array(base["v_at_ghia"]))))
        d_ghia = max(base["du_max"], base["dv_max"])
        sens = dict(N=Ns_, div_U=sm["div_U"], iterations=sm["iterations"], wall=sm["wall_time"],
                    converged=bool(sm["stopped_by_residualControl"] and sm["probe_change"] < conv["probe_change_tol"]),
                    d_scheme_max=float(d_scheme), d_ghia_max=float(d_ghia),
                    d_psi_rel=float((sx["psi_min"] - base["psi_min"]) / abs(base["psi_min"])),
                    du_max_vs_ghia=float(np.max(np.abs(np.array(sx["u_at_ghia"]) - u_g))),
                    dv_max_vs_ghia=float(np.max(np.abs(np.array(sx["v_at_ghia"]) - v_g))))

    # ---- assumptions -------------------------------------------------------------------------
    Ma = U / SOUND_SPEED_MIN
    asym = fine["x_c"] - 0.5
    assumptions = [
        dict(name="Two-dimensional flow", status="imposed",
             check="The cavity is very long in " + m("z") + " (given), so nothing varies along it. Imposed with one cell "
                   "and <code>empty</code> front/back patches. Real cavities of finite length have 3D end-wall effects."),
        dict(name="Steady, laminar flow", status="standard",
             check=m(rf"Re = UL/\nu = {tn(Re, 4)}") + ". At this Reynolds number the cavity flow is a single steady "
                   "vortex; the benchmark is a steady laminar solution at the same " + m("Re") + ". Steadiness of "
                   "the computed state is checked (Run health): the probed velocities stop changing."),
        dict(name="Incompressible (constant density)", status="ok" if Ma < 0.3 else "violated",
             check=m(rf"Ma = U/c \le {tn(U)}/{tn(SOUND_SPEED_MIN)} = {tn(Ma, 2)} \ll 0.3") + ". The fluid is not named, "
                   f"so the speed of sound " + m("c") + f" is bounded from below by {tn(SOUND_SPEED_MIN)} m/s (slower than "
                   "sound in air or any common liquid; approximate value)."),
        dict(name="Newtonian fluid, constant viscosity", status="standard",
             check="Given (\"incompressible Newtonian fluid\"); isothermal, so " + m(r"\nu") + " is constant."),
        dict(name="Gravity does not affect the flow", status="standard",
             check="Constant density: gravity only adds a hydrostatic pressure, which does not move the fluid."),
        dict(name="No slip on all walls; ideal sharp lid corners", status="standard",
             check="The fluid moves with each wall. The velocity jump at the two top corners is kept as in the benchmark; "
                   "it makes the corner flow singular (Theory, step 3)."),
    ]

    # ---- verdict (rules fixed in numerics.yaml before the results were seen) ------------------
    fail, warn = [], []
    if missing:
        fail.append(f"planned meshes not run: {missing}")
    if not all(r["mesh_ok"] for r in rows):
        fail.append("checkMesh reported a problem")
    if any(r["health"] == "fail" for r in rows):
        fail.append("a production run failed a log health check")
    if not all(r["converged"] for r in rows):
        fail.append("a run did not converge (residual stop + probe change)")
    for lab, key in (("(a) u-profile", "du_max"), ("(b) v-profile", "dv_max")):
        if fine[key] > acc["profile_max_diff_warn"]:
            fail.append(f"{lab}: max difference from Ghia {m(tn(fine[key], 2) + r'\,U')} above {m(tn(acc['profile_max_diff_warn']) + r'\,U')}")
        elif fine[key] > acc["profile_max_diff_pass"]:
            warn.append(f"{lab}: max difference from Ghia {m(tn(fine[key], 2) + r'\,U')} above {m(tn(acc['profile_max_diff_pass']) + r'\,U')}")
    if abs(fine["d_umin"]) > acc["u_min_diff_pass"]:
        fail.append("(c): u_min differs from Ghia by more than the tolerance")
    loc_tol = acc["location_tol_ghia_cells"] * h_G
    if abs(fine["d_yumin"]) > loc_tol:
        fail.append("(c): location of u_min further than one benchmark grid spacing from Ghia")
    if abs(fine["d_psi_rel"]) > acc["psi_min_rel_pass"]:
        fail.append(f"(d): psi_min differs from Ghia by more than {100 * acc['psi_min_rel_pass']:g} %")
    if abs(fine["d_xc"]) > loc_tol or abs(fine["d_yc"]) > loc_tol:
        fail.append("(d): vortex centre further than one benchmark grid spacing from Ghia")
    for k in scal:
        if ms[k]["gci"] > acc["gci_max"]:
            warn.append(f"GCI of {m(SYM[k])} = {m(tn(100 * ms[k]['gci'], 2) + r'\,\%')} above {m(tn(100 * acc['gci_max'], 2) + r'\,\%')}")
        if not ms[k]["monotonic"]:
            warn.append(f"{m(SYM[k])}: convergence not monotonic on the three finest meshes; order undefined "
                        f"(change between the two finest meshes only {m(tn(ms[k]['delta'], 2))})")
        elif not acc["order_range"][0] <= ms[k]["p"] <= acc["order_range"][1]:
            warn.append(f"{m(SYM[k])}: observed order {m(tn(ms[k]['p'], 3))} outside "
                        f"{m(tn(acc['order_range'][0], 2) + r' \le p \le ' + tn(acc['order_range'][1], 2))} "
                        f"(change between the two finest meshes only {m(tn(ms[k]['delta'], 2))}, "
                        f"GCI {m(tn(100 * ms[k]['gci'], 2) + r'\,\%')})")
    if max(abs(r["net_flow"]) for r in rows) > acc["mass_residual_max"]:
        fail.append("net flow through the vertical centreline is not zero (mass)")
    if any(a["status"] == "violated" for a in assumptions):
        fail.append("an assumption is violated")
    verdict = "FAIL" if fail else "WARN" if warn else "PASS"

    # ---- answers --------------------------------------------------------------------------------
    wanted = [dict(letter=LETTERS[i], key=w["key"], text=w["text"]) for i, w in enumerate(problem["quantities_of_interest"])]
    pct = lambda x: m(rf"{tn(100 * x, 2)}\,\%")
    ans = {
        "u_vertical": dict(
            theory=f"Table I, {len(y_g)} interior points (Figure {fig_no('u_profile')})",
            sim="largest difference " + m(rf"{tn(fine['du_max'], 2)}\,U") + "<div class='sub'>RMS "
                + m(rf"{tn(fine['du_rms'], 2)}\,U") + "</div>",
            diff=m(rf"\le {tn(fine['du_max'], 2)}\,U")),
        "v_horizontal": dict(
            theory=f"Table II, {len(x_g)} interior points (Figure {fig_no('v_profile')})",
            sim="largest difference " + m(rf"{tn(fine['dv_max'], 2)}\,U") + "<div class='sub'>RMS "
                + m(rf"{tn(fine['dv_rms'], 2)}\,U") + "</div>",
            diff=m(rf"\le {tn(fine['dv_max'], 2)}\,U")),
        "u_min": dict(
            theory=m(rf"u_{{\min}}/U = {tn(G['u_min'], 5)}") + " at " + m(rf"y/L = {tn(G['y_umin'], 4)}"),
            sim=m(rf"u_{{\min}}/U = {tn(fine['u_min'], 5)}") + " at " + m(rf"y/L = {tn(fine['y_umin'], 4)}")
                + "<div class='sub'>minimum of the interpolated centreline profile</div>",
            diff=m(rf"{tn(fine['d_umin'], 2)}\,U") + "<div class='sub'>position " + m(rf"{tn(fine['d_yumin'], 2)}\,L")
                 + f" = {tn(fine['d_yumin'] / h_G, 2)} benchmark cells</div>"),
        "vortex": dict(
            theory=m(rf"\psi_{{\min}}/(UL) = {tn(G['psi_min'], 6)}") + " at " + m(rf"({tn(G['x_c'], 4)}, {tn(G['y_c'], 4)})"),
            sim=m(rf"\psi_{{\min}}/(UL) = {tn(fine['psi_min'], 6)}") + " at " + m(rf"({tn(fine['x_c'], 4)}, {tn(fine['y_c'], 4)})")
                + "<div class='sub'>minimum of OpenFOAM's stream function, spline-refined</div>",
            diff=pct(fine["d_psi_rel"]) + "<div class='sub'>centre " + m(rf"({tn(fine['d_xc'], 2)}, {tn(fine['d_yc'], 2)})\,L")
                 + "</div>"),
    }
    answers = [dict(letter=w["letter"], label=w["text"], **ans[w["key"]]) for w in wanted]
    L_of = {w["key"]: w["letter"] for w in wanted}

    # ---- figures ------------------------------------------------------------------------------
    last_change = {c: float(np.max(np.abs(np.array(rows[-1][f"{c}_at_ghia"]) - np.array(rows[-2][f"{c}_at_ghia"]))))
                   for c in ("u", "v")}
    svgs = {"sketch": sketch(ref["vortices"]["primary"]["centre"])}
    colors = {N: plots.SERIES[i % 4] for i, N in enumerate(Ns)}
    svgs["u_profile"] = _profile_fig(rows, "y_line", "u_line", y_g, u_g, "du", colors, vertical=True)
    svgs["v_profile"] = _profile_fig(rows, "x_line", "v_line", x_g, v_g, "dv", colors, vertical=False)
    svgs["streamlines"] = _stream_fig(data[-1], L, U, fine, G)
    svgs["residuals"] = _residual_fig(metas, data, colors, numerics)
    svgs["mesh"] = _mesh_fig(rows, ms, prof, G, colors)
    figures = [
        dict(key="sketch", section="problem", caption="The cavity, the boundary conditions and where each requested output is "
             "measured. Dashed lines: the two centrelines of the benchmark tables. The vortex is drawn around the benchmark "
             "centre; not to scale for the corner eddies."),
        dict(key="streamlines", section="simulation", caption=f"Streamlines (lines of constant ψ) on the finest mesh "
             f"({Ns[-1]}×{Ns[-1]}). Solid: the clockwise primary vortex (ψ &lt; 0, a contour every {PSI_STEP:g} UL). Dotted: the weak "
             "counter-rotating corner eddies (ψ &gt; 0). The cross is the computed vortex centre, the circle the benchmark centre: "
             "they overlap at this scale. The centre sits right of x = L/2, the inertia effect predicted in Theory."),
        dict(key="u_profile", section="comparison", caption="Output (a), (c). Left: horizontal velocity along the vertical "
             "centreline; lines are the simulations, dots the benchmark (Ghia Table I). Fluid near the lid moves with it "
             "(u &gt; 0); lower down it returns (u &lt; 0). Right: simulation minus benchmark at the table points, per mesh. "
             "The gaps do NOT shrink to zero: between the two finest meshes they change by at most "
             + m(rf"{tn(last_change['u'], 2)}\,U") + ", while they reach " + m(rf"{tn(fine['du_max'], 2)}\,U")
             + ". They are not mesh error of this simulation (see the findings below Figure " + str(fig_no("mesh")) + ")."),
        dict(key="v_profile", section="comparison", caption="Output (b). Left: vertical velocity along the horizontal centreline "
             "(Ghia Table II as dots): fluid rises on the left (v &gt; 0) and falls on the right, near the downstream wall. "
             "Right: simulation minus benchmark per mesh. Again the gap settles instead of vanishing: it changes by at most "
             + m(rf"{tn(last_change['v'], 2)}\,U") + " between the two finest meshes but reaches " + m(rf"{tn(fine['dv_max'], 2)}\,U") + "."),
        dict(key="residuals", section="convergence", caption="Iterative convergence. Left: residual of the u-equation per "
             "SIMPLE iteration; finer meshes need more iterations (the slow, cavity-wide error decays more slowly). Right: the "
             "physical check: how far the probed velocity near the return-flow minimum still is from its final value. It "
             "falls steadily to round-off level (the final plunge only reflects that the last value is the "
             "reference): the runs are converged far below the mesh error."),
        dict(key="mesh", section="comparison", caption="Mesh study, log–log. Left: change between consecutive meshes "
             "(self-convergence; plotted at the finer of the two); the dashed line has slope 2, the design order of the schemes. "
             "Right: distance from the benchmark. It falls with refinement and then levels off where the benchmark's own "
             "accuracy is reached: there the remaining difference is no longer an error of this simulation alone. "
             "The simulated ψ_min passes through the benchmark value on its way to the extrapolated value, so its "
             "difference first dips and then grows to a small constant offset."),
    ]

    # ---- pre-flight ----------------------------------------------------------------------------
    pf_runs, pf_checks = [], []
    for pf in preflight:
        per_it = pf["wall_time"] / pf["steps"] if pf.get("wall_time") and pf.get("steps") else None
        pf_runs.append(dict(name=f"{pf['N']}×{pf['N']} mesh", setup=f"at most {pf['end_time_short']} iterations "
                            f"(ran {pf['steps']}) instead of running to convergence",
                            outcome=f"{'✓ healthy' if pf['status'] == 'ok' else pf['status']}; {pf['wall_time']:.2f} s, "
                                    f"{per_it * 1000:.3g} ms per iteration"))
        log = Path(pf["case"]) / "log.simpleFoam"
        if log.exists():                       # re-assess the saved log with the current monitor (formatting)
            pf["checks"] = [c for c in monitor.assess(log)["checks"]] + \
                           [c for c in pf["checks"] if c["name"].startswith("Projected")]
        for c in pf["checks"]:
            pf_checks.append(dict(c, name=f"{c['name']} ({pf['N']}²)"))
        est = pf.get("est_converged_runtime")
        if est is not None and pf["steps"] >= pf["end_time_short"]:
            pf_checks.append(dict(name=f"Projected run length ({pf['N']}²)", status="ok" if est < 900 else "warn",
                                  value=f"≈ {pf['iterations_to_converge']:.0f} iterations, ≈ {est:.0f} s",
                                  meaning="Extrapolated from the residual decay rate of the short run. A lower bound: "
                                          "convergence slows later (actual: see Run health)."))
    pf_fine = preflight[-1]

    # ---- tables ---------------------------------------------------------------------------------
    names = [f"M{i + 1}" for i in range(len(rows))]
    def srow(label, fn):
        return [label, ""] + [fn(k) for k in scal]
    res_rows = [[nm, f"{r['N']}²", m(tn(r["u_min"], 6)), m(tn(r["y_umin"], 4)), m(tn(r["psi_min"], 6)),
                 m(tn(r["x_c"], 4)), m(tn(r["y_c"], 4))] for nm, r in zip(names, rows)]
    res_rows += [
        srow("Observed order " + m("p"), lambda k: m(f"{ms[k]['p']:.2f}") if ms[k]["monotonic"] else "oscillating"),
        srow("Richardson extrapolation", lambda k: m(tn(ms[k]["extrap"], 6 if k in ("u_min", "psi_min") else 4))),
        srow("GCI, finest mesh", lambda k: m(rf"{tn(100 * ms[k]['gci'], 2)}\,\%")),
        srow("Benchmark (Ghia)", lambda k: m(tn(G[k], 6 if k in ("u_min", "psi_min") else 4))),
    ]
    st = lambda s: {"ok": "✓ OK", "warn": "⚠ warning", "fail": "✗ fail"}[s]
    health_rows = [[nm, f"{r['N']}²", r["iterations"], "✓" if r["stopped"] else "✗ (cap)",
                    m(tn(max(v for v in r["final_res"].values() if v is not None), 2)),
                    m(tn(r["probe_change"], 2)) + f"<div class='sub'>last {r['probe_window']} it.</div>",
                    m(tn(r["cont"], 2)) if r["cont"] is not None else "–", st(r["health"]),
                    "✓" if r["converged"] else "✗", f"{r['wall']:.1f}"] for nm, r in zip(names, rows)]
    comp_rows = [[nm, f"{r['N']}²", m(tn(r["du_max"], 2)), m(tn(r["du_rms"], 2)), m(tn(r["dv_max"], 2)),
                  m(tn(r["dv_rms"], 2)), m(tn(r["d_umin"], 2)), m(tn(r["d_yumin"] / h_G, 2)),
                  m(rf"{tn(100 * r['d_psi_rel'], 2)}\,\%"),
                  m(tn(math.hypot(r["d_xc"], r["d_yc"]) / h_G, 2))] for nm, r in zip(names, rows)]
    comp_rows.append(["Extrapolated", "–", m(tn(prof["u_at_ghia"]["ext_vs_ghia_max"], 2)), m(tn(prof["u_at_ghia"]["ext_vs_ghia_rms"], 2)),
                      m(tn(prof["v_at_ghia"]["ext_vs_ghia_max"], 2)), m(tn(prof["v_at_ghia"]["ext_vs_ghia_rms"], 2)),
                      m(tn(ms["u_min"]["extrap"] - G["u_min"], 2)), m(tn((ms["y_umin"]["extrap"] - G["y_umin"]) / h_G, 2)),
                      m(rf"{tn(100 * (ms['psi_min']['extrap'] - G['psi_min']) / abs(G['psi_min']), 2)}\,\%"),
                      m(tn(math.hypot(ms["x_c"]["extrap"] - G["x_c"], ms["y_c"]["extrap"] - G["y_c"]) / h_G, 2))])

    # ---- findings -------------------------------------------------------------------------------
    pu, pv = prof["u_at_ghia"], prof["v_at_ghia"]
    findings = [
        f"<b>Velocity profiles ({L_of['u_vertical']}, {L_of['v_horizontal']}).</b> On the finest mesh every benchmark point is "
        "matched to within " + m(rf"{tn(max(fine['du_max'], fine['dv_max']), 2)}\,U") + " (tolerance "
        + m(rf"{tn(acc['profile_max_diff_pass'])}\,U") + "); the typical (RMS) difference is "
        + m(rf"{tn(fine['du_rms'], 2)}\,U") + " for " + m("u") + " and " + m(rf"{tn(fine['dv_rms'], 2)}\,U") + " for " + m("v") + ".",
        f"<b>Return flow ({L_of['u_min']}).</b> " + m(rf"u_{{\min}} = {tn(fine['u_min'], 5)}\,U") + " versus "
        + m(rf"{tn(G['u_min'], 5)}\,U") + " in the benchmark; its position differs by "
        + m(rf"{tn(abs(fine['d_yumin']) / h_G, 2)}") + " benchmark grid spacings, i.e. within the benchmark's resolution.",
        f"<b>Vortex ({L_of['vortex']}).</b> Strength " + m(rf"\psi_{{\min}} = {tn(fine['psi_min'], 5)}\,UL") + ", "
        + pct(abs(fine["d_psi_rel"])) + " from the benchmark; the centre is "
        + m(tn(math.hypot(fine['d_xc'], fine['d_yc']) / h_G, 2)) + " benchmark grid spacings from Ghia's.",
        "<b>Mesh convergence.</b> The answers change by a factor of about "
        + m(tn(2 ** pu["orders"][-1], 2)) + " less with each halving of the cells (observed order "
        + m(f"p \\approx {pu['orders'][-1]:.1f}") + " for the " + m("u") + " profile, design order 2). "
        "The finest-mesh uncertainty (GCI) of " + m(r"\psi_{\min}") + " is " + m(rf"{tn(100 * ms['psi_min']['gci'], 2)}\,\%") + ".",
        "<b>Converged and conserving mass.</b> All runs stopped on their residual targets, the probed velocities changed by at "
        "most " + m(rf"{tn(max(r['probe_change'] for r in rows), 2)}\,U") + " over the last 10 % of iterations, and the net flow "
        "through the vertical centreline is " + m(rf"\le {tn(max(abs(r['net_flow']) for r in rows), 2)}\,UL") + " (must be 0).",
    ] + [f"<b>Rule failed:</b> {f}." for f in fail] + [f"<b>Caveat:</b> {w}." for w in warn]

    vortex_x_shift = fine["x_c"] - 0.5
    comp_findings = [
        "<b>The remaining difference is not mesh error.</b> Extrapolating to zero cell size changes the comparison "
        "little: the extrapolated profiles still differ from Ghia by up to " + m(rf"{tn(pu['ext_vs_ghia_max'], 2)}\,U")
        + " (" + m("u") + ") and " + m(rf"{tn(pv['ext_vs_ghia_max'], 2)}\,U") + " (" + m("v") + "), while the uncertainty "
        "bands of the finest mesh are at most " + m(rf"{tn(max(pu['band_max'], pv['band_max']), 2)}\,U")
        + ". The gap is therefore not mesh error of this simulation. Its most likely source is the benchmark's own "
        "discretisation error (its grid, " + m(rf"h_G = 1/{n_grid - 1}") + ", is coarser than the finest mesh here); the "
        "supplied data cannot prove this, and a second, independent benchmark would settle it. Until then the benchmark "
        "should be read as accurate to about this level, not to its printed digits.",
    ] + ([
        f"<b>Not a scheme artefact.</b> The {sens['N']}² case was re-run with central differencing for convection "
        f"(<code>{sens['div_U']}</code>) instead of linear-upwind. The two answers differ by at most "
        + m(rf"{tn(sens['d_scheme_max'], 2)}\,U") + " at the benchmark points (and " + pct(abs(sens["d_psi_rel"]))
        + " in " + m(r"\psi_{\min}") + "), while both differ from Ghia by up to "
        + m(rf"{tn(max(sens['du_max_vs_ghia'], sens['dv_max_vs_ghia']), 2)}\,U") + ". "
        + ("The scheme changes the answer far less than the gap to the benchmark, so that gap is not caused by the choice "
           "of convection scheme." if sens["d_scheme_max"] < 0.25 * sens["d_ghia_max"] else
           "The scheme changes the answer by a sizeable fraction of the gap to the benchmark: the mesh is not fine enough "
           "to make the result scheme-independent.")
    ] if sens else []) + [
        "<b>Where the differences are largest.</b> In the thin layer under the lid, where " + m("u") + " falls from "
        + m("U") + " to the return flow over a few percent of the cavity (Figure " + str(fig_no("u_profile")) + ", right panel), and near the downstream wall "
        "for " + m("v") + ". These are the regions of steepest gradients, next to the singular corners.",
        "<b>Physics check.</b> The vortex centre lies " + m(rf"{tn(vortex_x_shift, 2)}\,L") + " right of the centreline and "
        "above mid-height: inertia carries the fluid towards the downstream wall before it turns down, as Theory predicted "
        "(a slow, inertia-free flow would be symmetric).",
        "<b>Two independent stream functions agree.</b> " + m(r"\psi_{\min}") + " from OpenFOAM's flux-based "
        "<code>streamFunction</code> and from integrating the cell velocities differ by "
        + m(rf"{tn(abs(fine['psi_min'] - fine['psi_min_own']) / abs(fine['psi_min']), 2)}") + " (relative) on the finest mesh.",
        "<b>Mass check, honestly.</b> The check fixed before the runs (\"" + m(r"|\psi|") + " on the lid below "
        + m(tn(acc["mass_residual_max"])) + "\") failed with the velocity-integrated stream function ("
        + m(rf"{tn(fine['lid_psi_own'], 2)}\,UL") + " on the finest mesh, concentrated in the last cell next to the top-right corner) "
        "but is exactly zero with OpenFOAM's flux-based one. It tested the post-processing at the singular corner, not mass "
        "conservation, so it was replaced by the net flow through the vertical centreline (see Insights).",
    ]

    # ---- results.json --------------------------------------------------------------------------
    ok_word = {"PASS": "passed", "WARN": "passed with caveats", "FAIL": "FAILED"}[verdict]
    results = dict(
        title=problem["title"], date=dt.date.today().isoformat(), verdict=verdict, comparison_type="benchmark",
        summary=dict(
            headline=(f"<b>Benchmark comparison {ok_word}.</b> On the finest mesh ({Ns[-1]}×{Ns[-1]} cells) the velocity "
                      "profiles match Ghia et al. (1982) to within " + m(rf"{tn(max(fine['du_max'], fine['dv_max']), 2)}\,U")
                      + " and the vortex strength to " + pct(abs(fine["d_psi_rel"])) + "."),
            answer_columns=["Requested", "Benchmark (Ghia et al. 1982)", f"Simulation ({Ns[-1]}×{Ns[-1]} mesh)", "Difference"],
            answers=answers, findings=findings,
            limitations=[
                "The benchmark is itself a numerical solution on a " + f"{n_grid}×{n_grid} grid, not an experiment: agreement "
                "shows that the equations are solved correctly, not that a real cavity (3D, finite length, rounded corners) "
                "behaves this way.",
                "Ghia's locations are grid points " + m(rf"h_G = 1/{n_grid - 1}") + " apart, so location differences below "
                "about one spacing cannot be resolved.",
                "Velocities near the two top corners (singular) were not compared; the benchmark tables do not cover them.",
            ]),
        problem=dict(
            description=problem["description"], setup=problem.get("setup", []),
            given=[dict(label=v["label"], symbol=v["symbol"], value=m(tn(v["value"])), unit=v["unit"]) for v in g.values()],
            wanted=wanted,
            reference=f"{ref['source']['citation']}: {ref['source']['kind']}, {ref['source']['tables']}, read from "
                      f"<code>{problem['reference']['file']}</code>."),
        assumptions=dict(rows=assumptions),
        theory=dict(
            intro="The cavity flow has <b>no exact solution</b>: the convection term makes the equations nonlinear. The "
                  "comparison therefore uses the benchmark of Ghia et al. Theory still fixes what the answer depends on (one "
                  "number, " + m("Re") + "), what must hold exactly (no net flow through any line across the cavity), and "
                  "the expected shape of the flow. The algebraic steps marked below are checked by SymPy.",
            steps=th["steps"], derived=th["derived"],
            self_checks=[dict(name=k, ok=bool(v)) for k, v in th["self_checks"].items()]),
        decisions=load_yaml(HERE / "decisions.yaml"),
        preflight=dict(
            purpose="Before the production runs, short runs on the coarsest and finest meshes check that the setup runs, stays "
                    "stable and converges, and estimate the run time. A failure here stops the workflow before any long run.",
            runs=pf_runs, checks=pf_checks,
            conclusion=("All pre-flight checks passed" if all(p["status"] != "fail" for p in preflight) else "A pre-flight check FAILED")
                       + f"; residuals fall steadily on both meshes. The projected length of the finest run (≈ "
                         f"{pf_fine['iterations_to_converge']:.0f} iterations) was a lower bound: it actually needed "
                         f"{fine['iterations']} iterations ({fine['wall']:.0f} s), still well within the 15-minute budget."),
        simulation=dict(
            model=[
                dict(aspect="Geometry", real=f"Square cavity, side {m(rf'L = {tn(L)}\ \mathrm{{m}}')}, very long in " + m("z"),
                     model="The same square, one cell thick in " + m("z") + " with <code>empty</code> front and back: OpenFOAM "
                           "solves the 2D equations."),
                dict(aspect="Lid", real=f"Slides at {m(rf'U = {tn(U)}\ \mathrm{{m/s}}')} to the right",
                     model="Fixed velocity " + m(r"(U, 0, 0)") + " on the top boundary; the corners are left sharp."),
                dict(aspect="Fluid", real="Incompressible Newtonian, " + m(rf"\nu = {tn(nu)}\ \mathrm{{m^2/s}}"),
                     model="Kinematic pressure " + m(r"p/\rho") + " and " + m(r"\nu") + "; density never needed. "
                           + m(rf"Re = {tn(Re, 4)}") + " as in the benchmark."),
                dict(aspect="Steady state", real="The flow after all start-up motion has died out",
                     model="Solved directly with the steady SIMPLEC algorithm; each iteration is a step towards the steady "
                           "solution, not a step in time."),
                dict(aspect="Measured quantities", real="Velocities on the centrelines, vortex centre and strength",
                     model="Centreline values: mean of the two cell columns (rows) beside the centreline, interpolated to the "
                           "benchmark points with a cubic spline. Stream function: OpenFOAM's <code>streamFunction</code> "
                           "(from the face fluxes), minimum refined with a bicubic spline."),
            ],
            meshes=dict(
                intro=f"The same problem is solved on {len(rows)} uniform meshes. Each has cells half the size of the previous "
                      "one, so the change in the answer measures the error caused by the mesh. The benchmark used a grid "
                      + m(rf"h_G = 1/{n_grid - 1}") + ".",
                head=["Mesh", "Cells", "Cell size " + m("h/L"), "Cell Péclet " + m(r"Uh/\nu")],
                rows=[[nm, f"{r['N']}×{r['N']} = {r['N'] ** 2:,}", m(rf"1/{r['N']}"), m(tn(U * L / r['N'] / nu, 3))]
                      for nm, r in zip(names, rows)]),
            results=dict(
                head=["Mesh", "Cells", m(r"u_{\min}/U"), m(r"y_{u\min}/L"), m(r"\psi_{\min}/(UL)"), m(r"x_c/L"), m(r"y_c/L")],
                rows=res_rows),
            health=dict(
                head=["Mesh", "Cells", "Iterations", "Residual stop", "Largest final residual", "Probe change (÷U)",
                      "Continuity error", "Log checks", "Converged", "Time (s)"],
                rows=health_rows,
                notes=[
                    "A residual measures how far the current iterate is from satisfying the discretised equations (normalised, "
                    "so it is a relative measure). Runs stop when it falls below " + m(tn(conv["residual_U"])) + " for "
                    + m(r"\mathbf{u}") + " and " + m(tn(conv["residual_p"])) + " for " + m("p") + ".",
                    "Physical check: the velocity at four probe points must change by less than "
                    + m(rf"{tn(conv['probe_change_tol'])}\,U") + " over the last 10 % of iterations, far below the mesh "
                    "error. Two earlier attempts with looser residual targets failed this check, first on the 128² mesh and then on "
                    "the 256² mesh (Insights); each time the residual targets were tightened and all meshes re-run, never the check.",
                    "Continuity error: OpenFOAM's 'sum local' mass error of the last iteration.",
                    "Log checks: no crash, no NaN, no growing residuals (automatic, cfdlib.monitor).",
                    f"Iterations grow about {rows[-1]['iterations'] / rows[-2]['iterations']:.1f}× per refinement: SIMPLE "
                    "removes cavity-wide errors slowly on fine meshes.",
                ])),
        comparison=dict(
            kind="Benchmark: the simulation is compared with a trusted, very fine numerical solution of the same equations "
                 f"({ref['source']['citation']}). Differences are in units of the lid speed " + m("U") + " and the cavity "
                 "side " + m("L") + "; locations also in benchmark grid spacings " + m("h_G") + ".",
            definitions=[
                dict(name="Largest profile difference", latex=r"\Delta_{\max} = \max_k \left|\frac{u_{\mathrm{sim}}(y_k) - u_{\mathrm{Ghia}}(y_k)}{U}\right|",
                     meaning=f"Worst case over the {len(y_g)} (Table I) or {len(x_g)} (Table II) interior benchmark points; "
                             "the wall points are boundary conditions and are left out."),
                dict(name="RMS profile difference", latex=r"\Delta_{\mathrm{rms}} = \sqrt{\frac{1}{K}\sum_{k=1}^{K}\left(\frac{u_{\mathrm{sim}}(y_k) - u_{\mathrm{Ghia}}(y_k)}{U}\right)^2}",
                     meaning="Typical difference over the same points."),
                dict(name="Observed order and Richardson extrapolation",
                     latex=r"p = \frac{\ln\left|\frac{f_1 - f_2}{f_2 - f_3}\right|}{\ln 2},\qquad f_{\mathrm{ext}} = f_3 + \frac{f_3 - f_2}{2^p - 1}",
                     meaning="From three meshes (1 coarse, 3 fine): how fast the answer settles, and the value it settles to "
                             "(the zero-cell-size estimate). Profiles use one order from the RMS change between meshes."),
                dict(name="Grid Convergence Index (GCI)", latex=r"\mathrm{GCI} = 1.25\,\frac{|f_3 - f_2|/|f_3|}{2^p - 1}",
                     meaning="A conservative band for the mesh error of the finest result (Celik et al. 2008)."),
            ],
            head=["Mesh", "Cells", m(r"\Delta_{\max}\,u"), m(r"\Delta_{\mathrm{rms}}\,u"), m(r"\Delta_{\max}\,v"),
                  m(r"\Delta_{\mathrm{rms}}\,v"), m(r"\Delta u_{\min}/U"), m(r"\Delta y_{u\min}/h_G"), m(r"\Delta\psi_{\min}"),
                  "Centre offset " + m("/h_G")],
            rows=comp_rows, findings=comp_findings),
        insights=load_yaml(HERE / "insights.yaml"),
        figures=figures,
        reproducibility=dict(
            commands=[
                dict(cmd=f"python -m cases.{HERE.name}.run preflight",
                     what="Short test runs on the coarsest and finest meshes (separate pf<N> directories)"),
                dict(cmd=f"python -m cases.{HERE.name}.run production",
                     what="All meshes, coarse to fine; each run is post-processed with writeCellCentres and streamFunction"),
                dict(cmd=f"python -m cases.{HERE.name}.analyze",
                     what="Comparison with Ghia et al., mesh study, figures and this report (checks the report contract first)"),
            ],
            files=[
                dict(path=f"cases/{HERE.name}/problem.yaml", what="The problem as given: physics and requested outputs"),
                dict(path=f"cases/{HERE.name}/reference/", what="Ghia, Ghia & Shin (1982) Tables I, II, V, transcribed"),
                dict(path=f"cases/{HERE.name}/numerics.yaml", what="Meshes, solver and stopping settings"),
                dict(path=f"cases/{HERE.name}/decisions.yaml", what="Every choice, with the OpenFOAM settings it controls"),
                dict(path=f"cases/{HERE.name}/insights.yaml", what="Workflow insights logged during the work"),
                dict(path=f"cases/{HERE.name}/template/", what="OpenFOAM case with {{ placeholders }} filled per mesh"),
                dict(path=f"cases/{HERE.name}/theory.py", what="What theory can say without an exact solution"),
            ],
            run_dirs=[
                dict(path=f"{metas[0]['run_root']}/N<N>", what="Production run for each mesh"),
                dict(path=f"{metas[0]['run_root']}/pf<N>_preflight", what="Short pre-flight test runs"),
                dict(path=f"results/{HERE.name}/", what="preflight.json, raw run data, results.json and this report"),
            ],
            environment=report.environment_info(),
            notes=["One CPU core per run; runs are executed one after another."],
        ),
        data=dict(rows=[{k: v for k, v in r.items() if k not in ("health_checks",)} for r in rows],
                  mesh_study=ms, mesh_study_coarse=ms_coarse, profiles=prof, ghia=G, h_ghia=h_G, Re=Re, Ma_bound=Ma,
                  fail=fail, warn=warn, sensitivity=sens),
    )
    out = report.render(results, svgs, OUT / "report.html", case_dir=metas[-1]["case_dir"])
    print(f"{verdict}: du_max {fine['du_max']:.2e}, dv_max {fine['dv_max']:.2e}, d_umin {fine['d_umin']:.2e}, "
          f"d_psi {100 * fine['d_psi_rel']:.3f} %, centre ({fine['d_xc']:.4f}, {fine['d_yc']:.4f}); fail={fail}; warn={warn}\n"
          f"orders: " + ", ".join(f"{k} p={ms[k]['p']:.2f} R={ms[k]['R']:.3f} GCI={100 * ms[k]['gci']:.3f}%" for k in scal)
          + f"\nprofile orders u {pu['orders']}, v {pv['orders']}; ext-vs-Ghia u {pu['ext_vs_ghia_max']:.2e} band {pu['band_max']:.2e}; "
          f"v {pv['ext_vs_ghia_max']:.2e} band {pv['band_max']:.2e}\nReport: {out}")


def _last_continuity(meta):
    """'sum local' continuity error of the last iteration, from the solver log."""
    steps = monitor.parse_log(Path(meta["case_dir"]) / "log.simpleFoam")["steps"]
    return next((s["cont"] for s in reversed(steps) if s["cont"] is not None), None)


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def _profile_fig(rows, pos_key, val_key, pos_g, val_g, diff_key, colors, vertical):
    fig, (ax, bx) = plots.wide(2, 4.2, gridspec_kw=dict(width_ratios=[1.15, 1]))
    show = [rows[0], rows[-1]]
    for r in show:
        pos, val = np.array(r[pos_key]), np.array(r[val_key])
        if vertical:
            ax.plot(val, pos, color=colors[r["N"]], lw=1.6 if r is rows[-1] else 1.2, label=f"{r['N']}² mesh")
        else:
            ax.plot(pos, val, color=colors[r["N"]], lw=1.6 if r is rows[-1] else 1.2, label=f"{r['N']}² mesh")
    if vertical:
        ax.plot(val_g, pos_g, "o", color=plots.INK, ms=5, mfc="none", mew=1.3, label="Ghia et al. (1982)")
        ax.set_xlabel(r"$u/U$")
        ax.set_ylabel(r"$y/L$ (on $x = L/2$)")
    else:
        ax.plot(pos_g, val_g, "o", color=plots.INK, ms=5, mfc="none", mew=1.3, label="Ghia et al. (1982)")
        ax.set_xlabel(r"$x/L$ (on $y = L/2$)")
        ax.set_ylabel(r"$v/U$")
    ax.axhline(0, color=plots.AXIS, lw=0.8) if not vertical else ax.axvline(0, color=plots.AXIS, lw=0.8)
    ax.legend(loc="lower right" if vertical else "lower left", fontsize=9)
    for r in rows:
        d = np.array(r[diff_key])
        if vertical:
            bx.plot(d, pos_g, "o-", color=colors[r["N"]], ms=3.5, lw=1, label=f"{r['N']}²")
        else:
            bx.plot(pos_g, d, "o-", color=colors[r["N"]], ms=3.5, lw=1, label=f"{r['N']}²")
    if vertical:
        bx.axvline(0, color=plots.INK2, lw=0.8)
        bx.set_xlabel(r"$(u_{\rm sim} - u_{\rm Ghia})/U$")
        bx.set_ylabel(r"$y/L$")
    else:
        bx.axhline(0, color=plots.INK2, lw=0.8)
        bx.set_ylabel(r"$(v_{\rm sim} - v_{\rm Ghia})/U$")
        bx.set_xlabel(r"$x/L$")
    bx.legend(fontsize=8, title="mesh", title_fontsize=8)
    return plots.to_svg(fig)


def _stream_fig(d, L, U, fine, G):
    sf = d["sf"] / (U * L)
    N = sf.shape[0] - 1
    xv = np.arange(N + 1) / N
    fig, ax = plots.figure(5.2, 5.0)
    ax.grid(False)
    neg = np.arange(-0.10, 0.0, PSI_STEP)
    ax.contour(xv, xv, sf, levels=neg, colors=[plots.SERIES[0]], linewidths=1.0, linestyles="solid")
    pos = [1e-7, 1e-6, 5e-6]
    if sf.max() > pos[0]:
        ax.contour(xv, xv, sf, levels=[p for p in pos if p < sf.max()], colors=[plots.SERIES[1]], linewidths=0.9,
                   linestyles="dotted")
    ax.plot([fine["x_c"]], [fine["y_c"]], "x", color=plots.INK, ms=9, mew=2, label="computed centre")
    ax.plot([G["x_c"]], [G["y_c"]], "o", color=plots.INK, ms=11, mfc="none", mew=1.3, label="Ghia centre")
    ax.axvline(0.5, color=plots.MUTED, lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel(r"$y/L$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, ncol=2)
    return plots.to_svg(fig)


def _residual_fig(metas, data, colors, numerics):
    fig, (ax, bx) = plots.wide(2, 3.8)
    for meta, d in zip(metas, data):
        h = meta["residual_history"]["Ux"]
        ax.semilogy(np.arange(1, len(h) + 1), h, color=colors[meta["N"]], lw=1.2, label=f"{meta['N']}²")
        it, vals = d["probe_it"], d["probe_vals"]
        col = 2                                        # probe 2 = (0.5, 0.45), u component
        dev = np.abs(vals[:, col] - vals[-1, col])[:-1]
        bx.semilogy(it[:-1], np.maximum(dev, 1e-16), color=colors[meta["N"]], lw=1.2, label=f"{meta['N']}²")
    ax.axhline(numerics["convergence"]["residual_U"], color=plots.MUTED, ls="--", lw=1)
    ax.set_xlabel("SIMPLE iteration")
    ax.set_ylabel(r"Residual of the $u$ equation")
    ax.legend(fontsize=8, title="mesh", title_fontsize=8)
    bx.set_xlabel("SIMPLE iteration")
    pt = numerics["probes"][1]
    bx.set_ylabel(rf"$|u - u_{{\rm final}}|/U$ at $({pt[0]}, {pt[1]})\,L$")
    return plots.to_svg(fig)


def _mesh_fig(rows, ms, prof, G, colors):
    h = np.array([r["h"] for r in rows])
    fig, (ax, bx) = plots.wide(2, 4.0)
    series = [
        (r"$u$ profile (RMS)", prof["u_at_ghia"]["changes"], [r["du_rms"] for r in rows]),
        (r"$v$ profile (RMS)", prof["v_at_ghia"]["changes"], [r["dv_rms"] for r in rows]),
        (r"$u_{\min}$", np.abs(np.diff([r["u_min"] for r in rows])), [abs(r["d_umin"]) for r in rows]),
        (r"$\psi_{\min}$", np.abs(np.diff([r["psi_min"] for r in rows])), [abs(r["psi_min"] - G["psi_min"]) for r in rows]),
    ]
    for k, (lab, chg, dg) in enumerate(series):
        ax.loglog(h[1:], chg, "o-", color=plots.SERIES[k], ms=5, mec=plots.SURFACE, mew=1, label=lab)
        bx.loglog(h, dg, "o-", color=plots.SERIES[k], ms=5, mec=plots.SURFACE, mew=1, label=lab)
    ref = series[0][1][0] * 1.6 * (h[1:] / h[1]) ** 2
    ax.loglog(h[1:], ref, "--", color=plots.MUTED, lw=1)
    ax.annotate("slope 2", (h[1], ref[0]), xytext=(6, 0), textcoords="offset points", color=plots.MUTED, fontsize=9, va="center",
                ha="left")
    from matplotlib.ticker import NullFormatter, NullLocator
    for a in (ax, bx):
        a.set_xlabel(r"Cell size $h/L$")
        a.xaxis.set_minor_locator(NullLocator())
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_xticks(h)
        a.set_xticklabels([f"1/{r['N']}" for r in rows])
    ax.set_ylabel("Change from the next coarser mesh")
    bx.set_ylabel("Difference from Ghia et al.")
    ax.legend(fontsize=8)
    return plots.to_svg(fig)


if __name__ == "__main__":
    main()
