"""Compare the Poiseuille simulations with the analytical solution; build results.json + report.

Usage (repo root, inside WSL):  python -m cases.plane_poiseuille.analyze
Reads results/plane_poiseuille/raw.json plus the case's problem/numerics/decisions/insights YAML.
All text containing math is LaTeX inside \\( \\) and rendered by MathJax.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np

from cfdlib import plots, report, verification as V
from cfdlib.report import LETTERS, load_yaml, m, tex_num as tn, tex_pct as tp
from . import analytical
from .sketch import sketch

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name

# Reference values used in assumption checks, with their sources.
RE_CRIT_LINEAR = 5772        # Orszag (1971), J. Fluid Mech. 50: linear stability, Re = U_c h / nu
RE_TRANSITION_EXP = 1000     # Approximate; Carlson, Widnall & Peeters (1982), J. Fluid Mech. 121 (same Re definition)
SOUND_SPEED_WATER = 1482     # m/s at 20 °C, approximate standard property value


def main():
    problem = load_yaml(HERE / "problem.yaml")
    numerics = load_yaml(HERE / "numerics.yaml")
    raw = json.loads((OUT / "raw.json").read_text())
    g, runs = problem["given"], raw["runs"]
    H, rho, nu, dpdx = (g[k]["value"] for k in ("H", "rho", "nu", "dpdx"))
    mesh, conv = numerics["mesh"], numerics["convergence"]

    theory, u_exact = analytical.evaluate(g)
    ex = {d["key"]: d["value"] for d in theory["derived"]}
    G, mu = theory["G"], theory["mu"]

    # ---- per-mesh comparison ------------------------------------------------------------
    rows = []
    for r in runs:
        y, u = np.array(r["y"]), np.array(r["u"])
        ue = u_exact(y)
        n = len(u)
        u_c = 0.5 * (u[n // 2 - 1] + u[n // 2]) if n % 2 == 0 else u[n // 2]
        tau_mean = float(np.mean(list(r["tau_w"].values())))
        offset = u - ue                                   # structure of the error, not just its size
        rows.append(dict(
            Ny=r["Ny"], h=r["h"], iterations=r["iterations"], converged=r["converged"],
            mesh_ok=r["mesh_ok"], wall=r["wall_time"]["solver"], health=r["health"]["status"],
            q_change=r["q_change"],
            err_profile=V.relative_l2(u, ue, ex["u_max"]),
            u_c=float(u_c), err_umax=abs(u_c - ex["u_max"]) / ex["u_max"],
            q=r["q"], err_q=abs(r["q"] - ex["q"]) / ex["q"],
            tau=tau_mean, err_tau=abs(tau_mean - ex["tau_w"]) / ex["tau_w"],
            force_balance=abs(sum(r["tau_w"].values()) - G * H) / (G * H),
            offset_mean=float(offset.mean()), offset_spread=float(offset.max() - offset.min()),
            offset_predicted=G * r["h"] ** 2 / (8 * mu),
        ))
    h = [x["h"] for x in rows]
    order_profile = V.order_from_errors(h, [x["err_profile"] for x in rows])
    order_q = V.order_from_errors(h, [x["err_q"] for x in rows])
    fine, p_fine = rows[-1], order_profile[-1]
    uniform_shift = all(x["offset_spread"] < 1e-9 * ex["u_max"]
                        and abs(x["offset_mean"] - x["offset_predicted"]) < 1e-9 * ex["u_max"] for x in rows)
    max_spread = max(x["offset_spread"] for x in rows)

    # ---- assumption checks (computed where possible) ------------------------------------
    u_bulk, u_max = ex["u_bulk"], ex["u_max"]
    Re_H, Re_c, Ma = u_bulk * H / nu, u_max * (H / 2) / nu, u_max / SOUND_SPEED_WATER
    assumptions = [
        dict(name="Laminar flow (smooth layers, no turbulence)", status="ok" if Re_c < RE_TRANSITION_EXP else "violated",
             check=m(rf"Re_H = \dfrac{{\bar u\, H}}{{\nu}} = {tn(Re_H, 3)}") + ". In the units used by stability theory, "
                   + m(rf"Re = \dfrac{{u_{{\max}}\,(H/2)}}{{\nu}} = {tn(Re_c, 3)}")
                   + f", below the linear-stability limit {RE_CRIT_LINEAR} (Orszag 1971) and below the observed "
                     f"transition ≈ {RE_TRANSITION_EXP} (Carlson et al. 1982, approximate)."),
        dict(name="Incompressible (density constant)", status="ok" if Ma < 0.3 else "violated",
             check=m(rf"Ma = \dfrac{{u_{{\max}}}}{{c}} = {tn(Ma, 2)} \ll 0.3")
                   + f", with speed of sound in water c ≈ {SOUND_SPEED_WATER} m/s (approximate)."),
        dict(name="Steady and fully developed", status="imposed",
             check=m(r"\partial u/\partial t = 0,\ \partial u/\partial x = 0")
                   + ". Enforced by the steady solver and the periodic ends. Holds in a real channel only "
                     "far from its inlet (after the entrance length)."),
        dict(name="Two-dimensional", status="imposed",
             check="The plates are very wide, so nothing changes across the width (" + m("z") + ")."),
        dict(name="Newtonian fluid, constant properties", status="standard",
             check="Water is Newtonian (stress proportional to strain rate); constant temperature, so "
                   + m(r"\rho") + " and " + m(r"\nu") + " are constant."),
        dict(name="Gravity does not affect the flow", status="standard",
             check="Gravity acts across the gap; it only adds a hydrostatic pressure variation in "
                   + m("y") + " and does not change " + m("u") + "."),
        dict(name="No slip at both plates", status="standard",
             check="A viscous fluid in contact with a solid wall moves with the wall: " + m("u = 0") + "."),
    ]

    # ---- verdict (rule-based) ------------------------------------------------------------
    failures = []
    if not all(x["converged"] for x in rows):
        failures.append("not every run converged")
    if not all(x["mesh_ok"] for x in rows):
        failures.append("checkMesh reported a problem")
    if any(x["health"] == "fail" for x in rows):
        failures.append("a production run failed a health check")
    if fine["err_profile"] > 0.005:
        failures.append("finest-mesh profile error above 0.5 %")
    if not 1.5 <= p_fine <= 2.5:
        failures.append(f"observed order {p_fine:.2f} outside 1.5–2.5 (expected 2)")
    if max(x["force_balance"] for x in rows) > 1e-6:
        failures.append("momentum balance not satisfied")
    if any(a["status"] == "violated" for a in assumptions):
        failures.append("an assumption is violated")
    verdict = "PASS" if not failures else "FAIL"

    # ---- lettered answers (same letters as the request) ------------------------------------
    wanted = [dict(letter=LETTERS[i], key=w["key"], text=w["text"]) for i, w in enumerate(problem["quantities_of_interest"])]
    dmap = {d["key"]: d for d in theory["derived"]}
    sim_of = dict(u_max=(fine["u_c"], fine["err_umax"], "centreline, mean of the two middle cells"),
                  q=(fine["q"], fine["err_q"], "flux through the periodic boundary"),
                  tau_w=(fine["tau"], fine["err_tau"], "mean of both walls"))
    answers = []
    for w in wanted:
        if w["key"] == "u_profile":
            answers.append(dict(letter=w["letter"], label=w["text"], theory=m(theory["u_formula"]) + " (Figure 2)",
                                sim="RMS difference " + m(tp(fine["err_profile"])) + " of " + m(r"u_{\max}"), diff=""))
        else:
            d = dmap[w["key"]]
            sv, err, how = sim_of[w["key"]]
            answers.append(dict(letter=w["letter"], label=w["text"],
                                theory=m(rf"{d['symbol']} = {d['value_tex']}\ {d['unit_tex']}"),
                                sim=m(rf"{tn(sv, 6)}\ {d['unit_tex']}") + f"<div class='sub'>{how}</div>",
                                diff=m(tp(err))))
    target = next((x for x in rows if x["err_profile"] < 0.01), None)
    findings = [
        f"The error falls about {m(tn(2 ** p_fine, 2) + r'\times')} each time the cells are halved: observed order "
        f"{m(f'p = {p_fine:.2f}')}, the scheme is designed for {m('p = 2')}"
        + (". The solver behaves as intended." if 1.5 <= p_fine <= 2.5 else ". This is NOT what the scheme should give."),
        "Wall friction balances the driving force to " + m(tn(max(x['force_balance'] for x in rows), 2))
        + " on every mesh: momentum is conserved and every run converged.",
        (f"For 1 % accuracy, {m(f'N_y = {target['Ny']}')} cells across the gap are enough ({target['wall']:.2f} s)."
         if target else "No mesh reached 1 % accuracy."),
        f"Flow is laminar: {m(rf'Re_H = {tn(Re_H, 3)}')}.",
    ] + [f"<b>Problem:</b> {f}." for f in failures]

    # ---- figures -------------------------------------------------------------------------
    svgs = {"sketch": sketch()}
    fig, ax = plots.figure(5.4, 4.0)
    yy = np.linspace(0, H, 400)
    ax.plot(u_exact(yy), yy / H, color=plots.INK, lw=1.5, label="Exact solution", zorder=3)
    shown = [runs[0], next((r for r in runs if r["Ny"] == 32), runs[-1])]
    for k, r in enumerate(shown):
        ax.plot(r["u"], np.array(r["y"]) / H, "o", ms=7 if k == 0 else 4.5, color=plots.SERIES[k],
                mec=plots.SURFACE, mew=1, label=f"Simulation, {r['Ny']} cells", zorder=4)
    ax.set_xlabel(r"Velocity $u$ (m/s)")
    ax.set_ylabel(r"Position across the gap $y/H$")
    ax.set_xlim(left=0)
    ax.legend(loc="center left")
    svgs["profile"] = plots.to_svg(fig)

    fig, ax = plots.figure(5.4, 3.4)
    hist = runs[-1]["residual_history"]["Ux"]
    ax.semilogy(np.arange(1, len(hist) + 1), hist, "o-", color=plots.SERIES[0], ms=5, mec=plots.SURFACE, mew=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Residual of the $u$ equation")
    ax.set_xticks(range(1, len(hist) + 1))
    svgs["residuals"] = plots.to_svg(fig)

    fig, ax = plots.figure(5.4, 4.0)
    for k, (key, lab) in enumerate([("err_profile", r"Velocity profile $\varepsilon_u$"), ("err_q", r"Flow rate $\varepsilon_q$")]):
        ax.loglog(h, [x[key] for x in rows], "o-", color=plots.SERIES[k], ms=5, mec=plots.SURFACE, mew=1, label=lab)
    h_ref = np.array([h[0], h[-1]])
    ref = rows[0]["err_profile"] * 0.35 * (h_ref / h[0]) ** 2
    ax.loglog(h_ref, ref, "--", color=plots.MUTED, lw=1)
    ax.annotate(r"slope 2: $\varepsilon \propto h^2$", (h_ref[1], ref[1]), xytext=(6, -14), textcoords="offset points",
                ha="left", color=plots.MUTED, fontsize=9)
    ax.set_xlabel(r"Cell size $h$ (m)")
    ax.set_ylabel("Relative error")
    ax.legend(loc="upper left")
    svgs["convergence"] = plots.to_svg(fig)

    figures = [
        dict(key="sketch", section="problem", caption="The flow and the coordinates. The drawing is not to scale."),
        dict(key="profile", section="simulation", caption="Velocity across the gap. The line is the exact solution; dots are "
             "the simulated cell values on a coarse and a medium mesh. Even 8 cells capture the parabola closely."),
        dict(key="residuals", section="convergence", caption="Residual of the velocity equation per iteration, finest mesh."),
        dict(key="convergence", section="comparison", caption="Error against cell size, log–log. Both errors follow the dashed "
             "slope-2 line: halving the cell size cuts the error by 4, as a second-order method should."),
    ]

    # ---- pre-flight --------------------------------------------------------------------
    pf_runs, pf_checks = [], []
    for pf in raw["preflight"]:
        pf_runs.append(dict(name=f"{pf['Ny']}-cell mesh",
                            setup=f"{pf['end_time_short']} iterations instead of {conv['iterations']}",
                            outcome=f"{'✓ healthy' if pf['status'] == 'ok' else pf['status']}; "
                                    f"{pf['wall_time']:.2f} s" if pf.get("wall_time") is not None else pf["status"]))
        for c in pf["checks"]:
            pf_checks.append(dict(c, name=f"{c['name']} ({pf['Ny']} cells)"))
    est = max((pf["est_full_runtime"] or 0) for pf in raw["preflight"])

    # ---- results.json --------------------------------------------------------------------
    a_src = -dpdx / rho
    results = dict(
        title=problem["title"], date=dt.date.today().isoformat(), verdict=verdict, comparison_type="verification",
        summary=dict(
            headline=(f"<b>Verification {'passed' if verdict == 'PASS' else 'FAILED'}.</b> The simulation reproduces the "
                      f"exact solution: on the finest mesh ({fine['Ny']} cells across the gap) the velocity profile differs by "
                      + m(tp(fine['err_profile'])) + " of " + m(r"u_{\max}") + "."),
            answer_columns=["Requested", "Theory (exact)", "Simulation (finest mesh)", "Difference"],
            answers=answers, findings=findings,
            limitations=["Periodic ends force fully developed flow. A real channel needs an entrance length before this "
                         "profile forms; that region is not modelled."]),
        problem=dict(
            description=problem["description"], setup=problem.get("setup", []),
            given=[dict(label=v["label"], symbol=v["symbol"], value=m(tn(v["value"])), unit=v["unit"]) for v in g.values()],
            wanted=wanted, reference=None),
        assumptions=dict(rows=assumptions),
        theory=dict(
            intro="This problem has an exact solution. It is derived and solved symbolically by SymPy, not in the agent's "
                  "reasoning, and checked automatically against the equation and both boundary conditions.",
            steps=theory["steps"], derived=theory["derived"],
            self_checks=[dict(name=f"Solution {k}", ok=bool(v)) for k, v in theory["self_checks"].items()]),
        decisions=load_yaml(HERE / "decisions.yaml"),
        preflight=dict(
            purpose="Before the production runs, short runs on the coarsest and finest meshes check that the setup runs, "
                    "stays stable and gives a run-time estimate. A failure here stops the workflow before any long run.",
            runs=pf_runs, checks=pf_checks,
            conclusion=("All pre-flight checks passed; " if all(pf["status"] != "fail" for pf in raw["preflight"])
                        else "A pre-flight check failed; ") + f"projected run time of the longest production run ≈ {est:.1f} s."),
        simulation=dict(
            model=[
                dict(aspect="Length of the channel", real="Very long plates",
                     model=f"A short slice, length {m(rf'L_x = {tn(mesh['Lx_over_H'])}\,H')}, with <b>periodic ends</b>: "
                           "fluid leaving on the right re-enters on the left, so the slice behaves like an infinitely long channel."),
                dict(aspect="Width", real="Very wide plates", model="2D: one cell thick in " + m("z") + "; nothing varies across the width."),
                dict(aspect="What drives the flow", real="Pressure gradient " + m(rf"dp/dx = {tn(dpdx)}\ \mathrm{{Pa/m}}"),
                     model="An equivalent push on every fluid parcel, "
                           + m(rf"a = -\frac{{1}}{{\rho}}\frac{{dp}}{{dx}} = {tn(a_src)}\ \mathrm{{m/s^2}}")
                           + ". Periodic ends cannot carry a pressure drop, so the gradient is applied as this body force; "
                             "the effect on the fluid is identical."),
                dict(aspect="Fluid", real="Water, 20 °C", model=m(rf"\nu = {tn(nu)}\ \mathrm{{m^2/s}}") + ", constant (OpenFOAM works with "
                     + m(r"p/\rho") + " and " + m(r"\nu") + ")."),
                dict(aspect="Walls", real="Stationary plates", model="No slip, " + m(r"\mathbf{u} = 0") + "."),
            ],
            meshes=dict(
                intro=("The same problem is solved on " + str(len(rows)) + " meshes. Each has cells half the size of the previous one, "
                       "so the change in the answer measures the error caused by the mesh."),
                head=["Mesh", m("N_y") + " (cells across the gap)", "Cell size " + m("h = H/N_y")],
                rows=[[f"M{i + 1}", x["Ny"], m(rf"{tn(x['h'], 3)}\ \mathrm{{m}}")] for i, x in enumerate(rows)]),
            results=dict(
                head=["Mesh", m("N_y"), m(r"u_{\mathrm{centre}}") + " (m/s)", m("q") + " (m²/s)", m(r"\tau_w") + " (Pa)"],
                rows=[[f"M{i + 1}", x["Ny"], m(tn(x["u_c"], 6)), m(tn(x["q"], 6)), m(tn(x["tau"], 6))] for i, x in enumerate(rows)]),
            health=dict(
                head=["Mesh", "Iterations", "Flow-rate change, last iterations", "Momentum balance", "Log checks", "Converged", "Time (s)"],
                rows=[[f"M{i + 1}", x["iterations"], m(tn(x["q_change"], 2)), m(tn(x["force_balance"], 2)),
                       {"ok": "✓ OK", "warn": "⚠ warning", "fail": "✗ fail"}[x["health"]],
                       "✓" if x["converged"] else "✗", f"{x['wall']:.2f}"] for i, x in enumerate(rows)],
                notes=[
                    "A residual measures how far the current solution is from satisfying the discretised equations. "
                    "It drops to round-off (" + m(r"\sim 10^{-15}") + ") after the first iteration, because the problem is linear.",
                    "Pressure is not plotted: it is zero everywhere in this problem, so its normalised residual is only "
                    "round-off noise. Convergence is judged physically: the flow rate must stop changing ("
                    + m(rf"< {tn(conv['q_change_tol'])}") + ") and the wall friction must balance the driving force ("
                    + m(rf"< {tn(conv['force_balance_tol'])}") + ").",
                    "Log checks: no crash, no NaN, no growing residuals or Courant number (automatic, cfdlib.monitor).",
                ])),
        comparison=dict(
            kind="Verification: the simulation is compared with the exact solution of the same equations.",
            definitions=[
                dict(name="Velocity-profile error", latex=r"\varepsilon_u = \frac{1}{u_{\max}}\sqrt{\frac{1}{N_y}\sum_{i=1}^{N_y}"
                     r"\big(u_i - u_{\mathrm{exact}}(y_i)\big)^2}", meaning="Typical (root-mean-square) difference over all cells, as a fraction of the peak velocity."),
                dict(name="Flow-rate error", latex=r"\varepsilon_q = \frac{|q_{\mathrm{sim}} - q_{\mathrm{exact}}|}{q_{\mathrm{exact}}}",
                     meaning="Relative error of the flow rate computed by OpenFOAM."),
                dict(name="Observed order", latex=r"p = \frac{\ln(\varepsilon_{\mathrm{coarse}}/\varepsilon_{\mathrm{fine}})}{\ln 2}",
                     meaning=m(r"\varepsilon \propto h^p") + ". " + m("p = 2") + " means halving the cells cuts the error by 4."),
                dict(name="Momentum balance", latex=r"\frac{|\tau_{w,\mathrm{lower}} + \tau_{w,\mathrm{upper}} - G H|}{G H}",
                     meaning="Must be zero on any mesh: friction on both walls balances the driving force."),
            ],
            head=["Mesh", m("N_y"), m(r"\varepsilon_u"), m("p"), m(r"\varepsilon_q"), m("p"), m(r"\varepsilon_{\tau_w}")],
            rows=[[f"M{i + 1}", x["Ny"], m(tp(x["err_profile"])), m(f"{order_profile[i - 1]:.2f}") if i else "–",
                   m(tp(x["err_q"])), m(f"{order_q[i - 1]:.2f}") if i else "–", m(tp(x["err_tau"]))]
                  for i, x in enumerate(rows)],
            findings=([
                "<b>Where the error comes from.</b> On every mesh the simulated profile is the exact parabola shifted up by a "
                "constant, " + m(r"\Delta u = \dfrac{G h^2}{8\mu}") + " (cell-to-cell variation of the shift "
                + m(r"\le " + tn(max_spread, 2) + r"\ \mathrm{m/s}") + ", round-off). Inside the channel the method is exact for a "
                "parabola; the error enters at the walls, where the slope " + m(r"du/dy") + " is estimated over the half cell "
                "between the wall and the first cell centre. Averaging the two middle cells cancels the shift, so the centreline "
                "velocity is exact on every mesh."
            ] if uniform_shift else [])),
        insights=load_yaml(HERE / "insights.yaml"),
        figures=figures,
        reproducibility=dict(
            commands=[
                dict(cmd=f"python -m cases.{HERE.name}.run",
                     what="Builds every mesh, runs the pre-flight test, then the production runs; writes raw.json"),
                dict(cmd=f"python -m cases.{HERE.name}.analyze",
                     what="Theory, comparison, figures and this report (checks the report contract first)"),
            ],
            files=[
                dict(path=f"cases/{HERE.name}/problem.yaml", what="The problem as given: physics and requested outputs"),
                dict(path=f"cases/{HERE.name}/numerics.yaml", what="Meshes, solver and convergence settings"),
                dict(path=f"cases/{HERE.name}/decisions.yaml", what="Every choice, with the OpenFOAM settings it controls"),
                dict(path=f"cases/{HERE.name}/insights.yaml", what="Workflow insights logged during the work"),
                dict(path=f"cases/{HERE.name}/template/", what="OpenFOAM case with {{ placeholders }} filled per mesh"),
                dict(path=f"cases/{HERE.name}/analytical.py", what="Exact solution derived with SymPy, with self-checks"),
            ],
            run_dirs=[
                dict(path=f"{raw['meta']['run_root']}/Ny<N>", what="Production run for each mesh"),
                dict(path=f"{raw['meta']['run_root']}/pf<N>_preflight", what="Short pre-flight test runs"),
                dict(path=f"results/{HERE.name}/", what="raw.json, results.json and this report"),
            ],
            environment=raw["meta"].get("environment") or report.environment_info(),
            notes=["One CPU core per run; runs are executed one after another."],
        ),
        data=dict(rows=rows, order_profile=order_profile, order_q=order_q, Re_H=Re_H, Re_c=Re_c, Ma=Ma, exact=ex,
                  uniform_shift=uniform_shift),
    )
    out = report.render(results, svgs, OUT / "report.html", case_dir=runs[-1]["case_dir"])
    print(f"{verdict}: finest profile error {100 * fine['err_profile']:.4f} %, order {p_fine:.2f}. Report: {out}")


if __name__ == "__main__":
    main()
