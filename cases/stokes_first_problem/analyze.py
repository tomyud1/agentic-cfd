"""Compare the Stokes-first-problem simulations with the exact solution; build results.json + report.

Usage (repo root, inside WSL):  python -m cases.stokes_first_problem.analyze
Reads results/stokes_first_problem/{preflight.json, runs/*.json, runs/*.npz} and the case YAML files.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np

from cfdlib import plots, report, verification as V
from cfdlib.report import LETTERS, load_yaml, m, tex_num as tn, tex_pct as tp
from . import analytical
from .run import name_of
from .sketch import sketch

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"
SOUND_SPEED_WATER = 1482     # m/s near 20 °C, approximate standard property value (only used for Ma)
CHECK = {"ok": "✓ OK", "warn": "⚠ warning", "fail": "✗ fail"}


def depth(y, u, target):
    """Height where the (monotonically falling) profile crosses `target`, linear between cell centres."""
    i = int(np.argmax(u < target))
    if i == 0:
        return float("nan")
    return float(y[i - 1] + (target - u[i - 1]) * (y[i] - y[i - 1]) / (u[i] - u[i - 1]))


def momentum_balance(mom, tau, dt_):
    """|discrete d(int u dy)/dt - tau/rho| per step, with the time scheme the solver applied:
    OpenFOAM's backward (BDF2) uses Euler on the first step (no older field), BDF2 afterwards."""
    M = np.concatenate([[0.0], mom])
    ddt = np.empty(len(mom))
    ddt[0] = (M[1] - M[0]) / dt_
    ddt[1:] = (1.5 * M[2:] - 2.0 * M[1:-1] + 0.5 * M[:-2]) / dt_
    return np.abs(ddt - tau)


def analyse_run(meta, d, ex, U, nu, rho, times_req, frac):
    y, times, u = d["y"], d["times"], d["u"]
    r = dict(name=meta["name"], Ny=meta["Ny"], dt=meta["dt"], h=meta["h"], steps=meta["steps"],
             wall=meta["wall_time"], health=meta["health"]["status"], mesh_ok=meta["mesh_ok"],
             case_dir=meta["case_dir"], eu={}, delta={}, e_delta={}, prof={})
    for tt in times_req:
        i = int(np.argmin(np.abs(times - tt)))
        assert abs(times[i] - tt) < 1e-9, f"{meta['name']}: no field written at t = {tt}"
        r["eu"][tt] = float(np.max(np.abs(u[i] - ex["u"](y, tt))) / U)
        r["delta"][tt] = depth(y, u[i], frac * U)
        r["e_delta"][tt] = abs(r["delta"][tt] - float(ex["delta"](tt))) / float(ex["delta"](tt))
        r["prof"][tt] = u[i]
    tau_t, tau = d["tau_t"], rho * d["tau_x"]
    tau_ex = ex["tau"](tau_t)
    rel = np.abs(tau - tau_ex) / tau_ex
    r.update(tau_t=tau_t, tau=tau, tau_rel=rel, tau_end=float(tau[-1]), e_tau=float(rel[-1]),
             tau_sign=float(np.sign(tau[-1] - tau_ex[-1])))
    bad = np.nonzero(rel > 0.01)[0]
    r["t_1pct"] = float(tau_t[bad[-1] + 1]) if len(bad) else float(tau_t[0])
    i1 = int(np.argmin(np.abs(tau_t - 1.0)))
    r["tau_1s"] = float(tau[i1])
    P = rho * d["mom_x"]
    r.update(P_t=d["mom_t"], P=P, P_end=float(P[-1]),
             e_P=float(abs(P[-1] - ex["P"](d["mom_t"][-1])) / ex["P"](d["mom_t"][-1])))
    bal = momentum_balance(d["mom_x"], d["tau_x"], meta["dt"]) / d["tau_x"][-1]
    r.update(balance=bal, balance_max=float(bal.max()))
    r["far"] = float(np.max(np.abs(u[-1, -5:])) / U)                 # top five cells at t_end
    r["y"], r["times"] = y, times
    r["eu_max"] = max(r["eu"].values())
    r["e_delta_max"] = max(r["e_delta"].values())
    return r


def main():
    problem = load_yaml(HERE / "problem.yaml")
    numerics = load_yaml(HERE / "numerics.yaml")
    g = problem["given"]
    U, nu, rho, tend = (g[k]["value"] for k in ("U", "nu", "rho", "t_end"))
    H = numerics["domain"]["H"]
    acc = numerics["acceptance"]
    T = [float(t) for t in numerics["profile_times"]]
    frac = float(analytical.FRACTION)
    theory, ex = analytical.evaluate(g, H)
    dmap = {x["key"]: x for x in theory["derived"]}
    pf_raw = json.loads((OUT / "preflight.json").read_text())

    R = {}
    for f in RUNS.glob("*.json"):
        meta = json.loads(f.read_text())
        R[meta["name"]] = analyse_run(meta, np.load(f.with_suffix(".npz")), ex, U, nu, rho, T, frac)
    study = {k: [R[name_of(Ny, dt_)] for Ny, dt_ in numerics["runs"][k]] for k in ("family", "mesh", "timestep")}
    fam = study["family"]
    fine = fam[-1]
    mesh_s = study["mesh"] + [fine]                     # dt fixed (finest), h halved
    time_s = study["timestep"] + [fine]                 # h fixed (finest), dt halved
    hs = [x["h"] for x in fam]
    order = {k: V.order_from_errors(hs, [x[k] if not isinstance(k, tuple) else x[k[0]][k[1]] for x in fam])
             for k in ("e_tau", "e_P")}
    for tt in T:
        order[("eu", tt)] = V.order_from_errors(hs, [x["eu"][tt] for x in fam])
    p_main = order[("eu", T[-1])][-1]
    order_mesh = V.order_from_errors([x["h"] for x in mesh_s], [x["eu"][T[-1]] for x in mesh_s])
    # Time-step study on one mesh: successive differences of the t = 1 s profile isolate the time error.
    d_time = [float(np.max(np.abs(a["prof"][T[0]] - b["prof"][T[0]])) / U) for a, b in zip(time_s[:-1], time_s[1:])]
    order_time = [np.log(d_time[i] / d_time[i + 1]) / np.log(2) for i in range(len(d_time) - 1)]
    all_orders = [v for vals in order.values() for v in vals]
    min_p, max_p = min(all_orders), max(all_orders)
    time_err_fine = d_time[-1] / (2 ** order_time[-1] - 1)          # Richardson estimate of the finest-dt time error
    # tau error versus the wall-gradient estimate h^2 / (48 nu t): ratio per family run at t_end
    c_tau = [x["e_tau"] / (x["h"] ** 2 / (48 * nu * tend)) for x in fam]

    # ---- assumption checks -----------------------------------------------------------------------
    Ma = U / SOUND_SPEED_WATER
    Re_d = theory["Re_delta"]
    far_ex = dmap["far"]["value"]
    assumptions = [
        dict(name="Parallel flow: u depends only on y and t (very large plate)", status="imposed",
             check="One cell along " + m("x") + " with periodic ends and an empty " + m("z") + " direction. "
                   "Edge effects of a real, finite plate are excluded by construction."),
        dict(name="Very deep water (at rest far above)", status="ok" if max(x["far"] for x in R.values()) < acc["far_field_max"] else "violated",
             check="Exact solution at the top of the simulated layer " + m(rf"H = {tn(H)}\,\mathrm{{m}}") + " at "
                   + m(rf"t = {tn(tend)}\,\mathrm{{s}}") + ": " + m(rf"u(H)/U = {tn(far_ex, 2)}") + "; simulated, largest over all runs: "
                   + m(rf"{tn(max(x['far'] for x in R.values()), 2)}") + ". The layer is "
                   + m(tn(H / dmap['delta_20']['value'], 3)) + " times the 1 % depth at the end."),
        dict(name="Laminar flow", status="standard",
             check="The exact solution is laminar and the problem asks for it. The convection terms are identically zero in "
                   "this flow, so nothing in the model could become unstable. A Reynolds number on the 1 % depth at the end, "
                   + m(rf"Re_\delta = U\delta_{{1\%}}/\nu = {tn(Re_d, 3)}") + ", is given for orientation; no stability or "
                   "transition reference for this flow was supplied, so laminarity is assumed, not checked."),
        dict(name="Incompressible (density constant)", status="ok" if Ma < 0.3 else "violated",
             check=m(rf"Ma = U/c = {tn(Ma, 2)} \ll 0.3") + f", with speed of sound in water c ≈ {SOUND_SPEED_WATER} m/s (approximate)."),
        dict(name="Newtonian water, constant properties", status="standard",
             check="Water is Newtonian; one temperature, so " + m(r"\rho") + " and " + m(r"\nu") + " are constant."),
        dict(name="Gravity does not affect the motion", status="standard",
             check="Gravity is normal to the plate; it only adds hydrostatic pressure in " + m("y") + " and no force along " + m("x") + "."),
        dict(name="No slip on the plate; impulsive start", status="imposed",
             check="The water touching the plate moves with it. The jump from rest to " + m("U") + " is instantaneous, "
                   "as stated; a real plate needs a short time to accelerate, so the infinite wall stress at "
                   + m("t = 0") + " is an idealisation."),
    ]

    # ---- verdict (rule-based, rules from numerics.yaml) ---------------------------------------------
    failures, warns = [], []
    if not all(x["mesh_ok"] for x in R.values()):
        failures.append("checkMesh reported a problem")
    if any(x["health"] == "fail" for x in R.values()):
        failures.append("a production run failed a health check")
    if fine["eu_max"] > acc["profile_max_err"]:
        failures.append(f"finest-run profile error above {100 * acc['profile_max_err']:g} % of U")
    if fine["e_tau"] > acc["tau_err_tend"]:
        failures.append("finest-run wall shear stress error at the end above tolerance")
    if fine["e_delta_max"] > acc["delta_err"]:
        failures.append("finest-run penetration-depth error above tolerance")
    if fine["e_P"] > acc["momentum_err"]:
        failures.append("finest-run momentum error above tolerance")
    if max(x["balance_max"] for x in R.values()) > acc["momentum_balance_max"]:
        failures.append("discrete momentum balance not satisfied")
    if max(x["far"] for x in R.values()) > acc["far_field_max"]:
        failures.append("water at the top of the layer is not at rest")
    if any(a["status"] == "violated" for a in assumptions):
        failures.append("an assumption is violated")
    lo, hi = acc["order_range"]
    if not lo <= p_main <= hi:
        warns.append(f"observed order {p_main:.2f} outside {lo}–{hi}")
    verdict = "FAIL" if failures else "WARN" if warns else "PASS"

    # ---- lettered answers ------------------------------------------------------------------------
    wanted = [dict(letter=LETTERS[i], key=w["key"], text=w["text"]) for i, w in enumerate(problem["quantities_of_interest"])]
    mm = lambda v: tf(1e3 * v, 5)
    sub = lambda s: f"<div class='sub'>{s}</div>"
    tlist = ", ".join(f"{t:g}" for t in T)
    answers = []
    for w in wanted:
        k = w["key"]
        if k == "u_profiles":
            answers.append(dict(letter=w["letter"], label=w["text"],
                                theory=m(theory["u_formula"]) + sub("Figure {fig:profiles}"),
                                sim="largest difference over all cells: " + ", ".join(
                                    m(rf"{tp(fine['eu'][t])}") + f" at {t:g} s" for t in T) + sub("as a fraction of " + m("U")),
                                diff=m(r"\le " + tp(fine["eu_max"]))))
        elif k == "tau_w":
            d = dmap["tau_w"]
            answers.append(dict(letter=w["letter"], label=w["text"],
                                theory=m(rf"|\tau_w| = \dfrac{{\mu U}}{{\sqrt{{\pi\nu t}}}}") + "; at " + m(rf"t = {tn(tend)}\,\mathrm{{s}}")
                                       + ": " + m(rf"{d['value_tex']}\ \mathrm{{Pa}}") + sub("acting against the plate motion (Figure {fig:tau})"),
                                sim=m(rf"{tn(fine['tau_end'], 6)}\ \mathrm{{Pa}}") + sub(
                                    "within 1 % of the exact value for " + m(rf"t \ge {tn(fine['t_1pct'], 2)}\,\mathrm{{s}}")),
                                diff=m(tp(fine["e_tau"]))))
        elif k == "delta99":
            answers.append(dict(letter=w["letter"], label=w["text"],
                                theory=", ".join(m(rf"{mm(dmap[key]['value'])}") for key in ("delta_1", "delta_5", "delta_20"))
                                       + r" \(\mathrm{mm}\)" + sub(m(r"\delta_{1\%} = 2\eta_1\sqrt{\nu t}") + f" at {tlist} s"),
                                sim=", ".join(m(mm(fine["delta"][t])) for t in T) + r" \(\mathrm{mm}\)",
                                diff=m(r"\le " + tp(fine["e_delta_max"]))))
        elif k == "momentum":
            d = dmap["P"]
            answers.append(dict(letter=w["letter"], label=w["text"],
                                theory=m(rf"P = {d['value_tex']}\ \mathrm{{kg/(m\,s)}}") + sub("i.e. " + m(r"\mathrm{N\,s/m^2}") + ", per square metre of plate"),
                                sim=m(rf"{tn(fine['P_end'], 6)}\ \mathrm{{kg/(m\,s)}}"),
                                diff=m(tp(fine["e_P"]))))

    c_lo, c_hi = min(c_tau), max(c_tau)
    findings = [
        f"Every requested quantity matches the exact solution on the finest run ({fine['Ny']} cells, "
        + m(rf"\Delta t = {tn(fine['dt'])}\,\mathrm{{s}}") + "): velocity within " + m(tp(fine["eu_max"])) + " of " + m("U")
        + ", wall shear stress, depth and momentum within " + m(tp(max(fine["e_tau"], fine["e_delta_max"], fine["e_P"]))) + ".",
        "Halving the cell size and the time step together cuts the error by "
        + m(tn(2 ** p_main, 3) + r"\times") + ": observed order " + m(f"p = {p_main:.2f}") + " (design order 2).",
        "The error is almost entirely spatial: on the finest mesh, refining the time step from "
        + m(rf"{tn(time_s[0]['dt'])}") + " to " + m(rf"{tn(fine['dt'])}\,\mathrm{{s}}") + " changes the "
        + m(rf"t = {tn(T[0])}\,\mathrm{{s}}") + " profile by at most "
        + m(tp(float(np.max(np.abs(time_s[0]["prof"][T[0]] - fine["prof"][T[0]])) / U)))
        + " of " + m("U") + ".",
        "The wall shear stress cannot be matched at the very start (the exact value is infinite at " + m("t = 0")
        + "). It is within 1 % of the exact value after " + m(rf"{tn(fine['t_1pct'], 2)}\,\mathrm{{s}}")
        + " on the finest run and after " + m(rf"{tn(fam[0]['t_1pct'], 3)}\,\mathrm{{s}}") + " on the coarsest.",
        "Momentum is conserved exactly: on every step of every run, the change of the water's momentum equals the plate's "
        "drag to " + m(tn(max(x["balance_max"] for x in R.values()), 2)) + " of the final drag.",
    ] + [f"<b>Problem:</b> {f}." for f in failures] + [f"<b>Warning:</b> {w}." for w in warns]

    # ---- figures ----------------------------------------------------------------------------------
    svgs = {"sketch": sketch()}
    coarse = fam[0]
    cols = {t: plots.SERIES[i] for i, t in enumerate(T)}
    fig, (a1, a2) = plots.wide(2, 3.9)
    yy = np.linspace(0, 1.5 * dmap["delta_20"]["value"], 400)
    for t in T:
        a1.plot(1e3 * yy, ex["u"](yy, t) / U, color=cols[t], lw=1.6, label=f"exact, t = {t:g} s")
        sel = coarse["y"] <= yy[-1]
        a1.plot(1e3 * coarse["y"][sel], coarse["prof"][t][sel] / U, "o", ms=3.6, color=cols[t], mec=plots.SURFACE, mew=0.6,
                label="_nolegend_")
    a1.plot([], [], "o", ms=4, color=plots.MUTED, label=f"simulation, {coarse['Ny']} cells")
    a1.set_xlabel(r"Height above the plate $y$ (mm)")
    a1.set_ylabel(r"Velocity $u/U$")
    a1.set_title("Profiles at three times", loc="left")
    a1.legend(loc="upper right", fontsize=8.5)
    ee = np.linspace(0, 3, 300)
    a2.plot(ee, ex["u"](2 * np.sqrt(nu * 1.0) * ee, 1.0) / U, color=plots.INK, lw=1.4, label=r"exact: $\mathrm{erfc}\,\eta$")
    for k, t in enumerate(T):
        eta_c = fine["y"] / (2 * np.sqrt(nu * t))
        sel = eta_c <= 3
        stride = max(1, int(sel.sum() // 18))
        a2.plot(eta_c[sel][k::stride], fine["prof"][t][sel][k::stride] / U, "os^"[k], ms=4.5, color=cols[t],
                mec=plots.SURFACE, mew=0.6, label=f"simulation, t = {t:g} s")
    a2.set_xlabel(r"Similarity coordinate $\eta = y/(2\sqrt{\nu t})$")
    a2.set_ylabel(r"$u/U$")
    a2.set_title(f"Collapse onto one curve ({fine['Ny']} cells)", loc="left")
    a2.legend(loc="upper right", fontsize=8.5)
    svgs["profiles"] = plots.to_svg(fig, name="profiles")

    fig, (b1, b2) = plots.wide(2, 3.9)
    tt = np.logspace(np.log10(fine["tau_t"][0]), np.log10(tend), 300)
    b1.loglog(tt, ex["tau"](tt), color=plots.INK, lw=1.4, label="exact")
    # finest first (small, filled), coarsest on top (larger, hollow) so both stay visible
    for k, x in ((1, fine), (0, coarse)):
        st = np.unique(np.round(np.logspace(0, np.log10(len(x["tau_t"]) - 1), 22)).astype(int))   # even on a log axis
        b1.loglog(x["tau_t"][st], x["tau"][st], "o", **_mk(k), label=f"{x['Ny']} cells, " + rf"$\Delta t = {x['dt']:g}$ s")
    b1.set_xlabel(r"Time $t$ (s)")
    b1.set_ylabel(r"$|\tau_w|$ (Pa)")
    b1.set_title("Wall shear stress", loc="left")
    b1.legend(loc="lower left", fontsize=8.5)
    for k, x in enumerate(fam):
        b2.loglog(x["tau_t"], x["tau_rel"], color=plt_color(k), lw=1.3, label=f"{x['Ny']} cells")
    b2.axhline(0.01, color=plots.MUTED, lw=1, ls="--")
    b2.annotate("1 %", (fine["tau_t"][0], 0.01), xytext=(4, 4), textcoords="offset points", color=plots.MUTED, fontsize=9)
    b2.set_xlabel(r"Time $t$ (s)")
    b2.set_ylabel("Relative error of " + r"$|\tau_w|$")
    b2.set_title("Error falls with time and refinement", loc="left")
    b2.legend(loc="lower left", fontsize=8, ncols=2)
    svgs["tau"] = plots.to_svg(fig, name="tau")

    fig, (c1, c2) = plots.wide(2, 3.6)
    tw = np.linspace(0.05, tend, 300)
    c1.plot(tw, 1e3 * ex["delta"](tw), color=plots.INK, lw=1.4, label="exact")
    for k, x in ((1, fine), (0, coarse)):
        dd = [depth(x["y"], x_u, frac * U) for x_u in np.load(RUNS / f"{x['name']}.npz")["u"]]
        c1.plot(x["times"], 1e3 * np.array(dd), "o", **_mk(k), label=f"{x['Ny']} cells")
    c1.set_xlabel(r"Time $t$ (s)")
    c1.set_ylabel(r"$\delta_{1\%}$ (mm)")
    c1.set_title("Penetration depth", loc="left")
    c1.legend(loc="lower right", fontsize=8.5)
    c2.plot(tw, ex["P"](tw), color=plots.INK, lw=1.4, label="exact")
    for k, x in ((1, fine), (0, coarse)):
        st = max(1, len(x["P_t"]) // 20)
        c2.plot(x["P_t"][st - 1::st], x["P"][st - 1::st], "o", **_mk(k), label=f"{x['Ny']} cells")
    c2.set_xlabel(r"Time $t$ (s)")
    c2.set_ylabel(r"$P$ (kg/(m s))")
    c2.set_title("Momentum given to the water", loc="left")
    c2.legend(loc="lower right", fontsize=8.5)
    svgs["growth"] = plots.to_svg(fig, name="growth")

    fig, ax = plots.figure(6.4, 3.4)
    edges = np.arange(0.0, tend + 1e-9, 0.5)
    for k, x in enumerate(fam):                     # envelope: largest mismatch in each half-second window
        idx = np.digitize(x["tau_t"], edges[1:-1])
        env = [x["balance"][idx == i].max() for i in range(len(edges) - 1)]
        ax.semilogy(0.5 * (edges[1:] + edges[:-1]), env, "o-", ms=3, color=plt_color(k), lw=1.2, label=f"{x['Ny']} cells")
    ax.axhline(acc["momentum_balance_max"], color=plots.MUTED, lw=1, ls="--")
    ax.annotate("acceptance limit", (tend, acc["momentum_balance_max"]), xytext=(-4, -12), textcoords="offset points",
                ha="right", color=plots.MUTED, fontsize=9)
    ax.set_xlabel(r"Time $t$ (s)")
    ax.set_ylabel("Momentum-balance error\n" + r"(fraction of final $|\tau_w|$)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    svgs["balance"] = plots.to_svg(fig, name="balance")

    fig, (d1, d2) = plots.wide(2, 3.9)
    hmm = 1e3 * np.array(hs)
    for k, (key, lab) in enumerate([(("eu", T[0]), r"profile, $t = 1$ s"), (("eu", T[-1]), r"profile, $t = 20$ s"),
                                    ("e_tau", r"$|\tau_w|$, $t = 20$ s"), ("e_P", r"$P$, $t = 20$ s")]):
        vals = [x[key[0]][key[1]] if isinstance(key, tuple) else x[key] for x in fam]
        d1.loglog(hmm, vals, "o-", color=plots.SERIES[k], ms=4.5, mec=plots.SURFACE, mew=0.6, lw=1.4, label=lab)
    ref = fam[0]["eu"][T[-1]] * 0.3 * (hmm / hmm[0]) ** 2
    d1.loglog(hmm, ref, "--", color=plots.MUTED, lw=1)
    d1.annotate("slope 2", (hmm[-2], ref[-2]), xytext=(4, -12), textcoords="offset points", color=plots.MUTED, fontsize=9)
    d1.set_xlabel(r"Cell size $h$ (mm), with $\Delta t \propto h$")
    d1.set_ylabel("Relative error")
    d1.set_title("Combined refinement", loc="left")
    d1.legend(loc="upper left", fontsize=8)
    ratio = [1, 2, 4, 8]
    sp_err = [x["eu"][T[-1]] for x in mesh_s][::-1]                     # finest first
    d2.loglog(ratio, sp_err, "o-", color=plots.SERIES[0], ms=4.5, mec=plots.SURFACE, mew=0.6, lw=1.4,
              label="space: profile error, 20 s")
    d2.loglog(ratio[:len(d_time)], d_time[::-1], "s-", color=plots.SERIES[1], ms=4.5, mec=plots.SURFACE, mew=0.6, lw=1.4,
              label="time: change of 1 s profile")
    ref2 = sp_err[0] * 0.4 * np.array(ratio, float) ** 2
    d2.loglog(ratio, ref2, "--", color=plots.MUTED, lw=1)
    d2.annotate("slope 2", (ratio[2], ref2[2]), xytext=(6, -10), textcoords="offset points", color=plots.MUTED, fontsize=9)
    d2.set_xlabel(r"$h/h_{\min}$  or  $\Delta t/\Delta t_{\min}$")
    d2.set_ylabel(r"Error measure (fraction of $U$)")
    d2.set_title("Space and time errors separately", loc="left")
    d2.legend(loc="lower right", fontsize=8)
    log_ticks(d1, hmm, "x")
    log_ticks(d2, ratio, "x")
    svgs["order"] = plots.to_svg(fig, name="order")

    figures = [
        dict(key="sketch", section="problem", caption="The plate, the water above it and the velocity profiles at three "
             "times (drawing not to scale). The moving layer thickens with time."),
        dict(key="profiles", section="simulation",
             caption=f"Left: velocity profiles at {tlist} s; lines are the exact solution, dots the coarsest run "
                     f"({coarse['Ny']} cells, largest difference " + m(tp(coarse['eu_max'])) + " of " + m("U")
                     + f"). Right: the finest run's profiles at the three times, plotted against the similarity "
                     "coordinate, fall on the single exact curve."),
        dict(key="tau", section="simulation",
             caption="Left: wall shear stress magnitude against time (log–log; the exact solution falls like "
                     + m(r"t^{-1/2}") + "). Right: its relative error for the combined-refinement runs; the dashed line "
                     "is 1 %. The error is large only in the first few time steps, where the exact stress is infinite."),
        dict(key="growth", section="simulation",
             caption="Left: 1 % penetration depth from the written fields (every second). Right: x-momentum per unit plate "
                     "area. Lines exact, markers simulation (coarsest and finest runs)."),
        dict(key="balance", section="convergence",
             caption="Per-step momentum balance: the change of the water's momentum per step minus the plate's drag, as a "
                     "fraction of the final drag; each point is the largest value in a half-second window. It stays at "
                     "round-off level (largest "
                     + m(tn(max(x["balance_max"] for x in fam), 2)) + "), so each step is solved completely."),
        dict(key="order", section="comparison",
             caption="Left: errors of the combined-refinement runs against cell size; "
                     + ("all follow the dashed slope-2 line (every observed order between "
                        + m(f"{min_p:.2f}") + " and " + m(f"{max_p:.2f}") + ")" if 1.8 <= min_p and max_p <= 2.2 else
                        "observed orders range from " + m(f"{min_p:.2f}") + " to " + m(f"{max_p:.2f}"))
                     + ". Right: the space error alone (mesh halved at the finest time step, order "
                     + m(f"{order_mesh[-1]:.2f}") + ") and the time error alone (change of the 1 s profile on the "
                     f"finest mesh when the time step is halved, order " + m(f"{order_time[-1]:.2f}") + "), both against "
                     "the refinement factor relative to the finest run."),
    ]

    # ---- pre-flight ------------------------------------------------------------------------------
    pf_runs, pf_checks = [], []
    for pf in pf_raw:
        pf_runs.append(dict(name=f"{pf['Ny']} cells, " + m(rf"\Delta t = {tn(pf['dt'])}\,\mathrm{{s}}"),
                            setup=f"first {pf['end_time_short']:g} s of {tend:g} s ({pf['n_steps']} steps)",
                            outcome=f"{CHECK[pf['status']]}; {pf['wall_time']:.2f} s; projected full run "
                                    f"{pf['est_full_runtime']:.1f} s"))
        for c in pf["checks"]:
            pf_checks.append(dict(c, name=f"{c['name']} ({pf['Ny']} cells)"))
    est = max(pf["est_full_runtime"] for pf in pf_raw)
    fine_pf = pf_raw[-1]

    # ---- tables ----------------------------------------------------------------------------------
    names = [name_of(*p) for p in _all(numerics)]
    member = {n: [] for n in names}
    for k, lab in (("family", "combined"), ("mesh", "space"), ("timestep", "time")):
        for Ny, dt_ in numerics["runs"][k]:
            member[name_of(Ny, dt_)].append(lab)
    member[fine["name"]] += ["space", "time"]
    rows_sorted = sorted(R.values(), key=lambda x: (x["Ny"], -x["dt"]))
    model = [
        dict(aspect="Plate", real="Very large flat plate, suddenly moving at " + m("U"),
             model="A wall patch with velocity fixed at " + m(rf"({tn(U)},\,0,\,0)\ \mathrm{{m/s}}") + " from the first step; "
                   "one cell along " + m("x") + " with periodic ends (an infinitely long plate) and an empty " + m("z")
                   + " direction: the model is 1D in " + m("y") + "."),
        dict(aspect="Water depth", real="Very deep",
             model="A layer " + m(rf"H = {tn(H)}\,\mathrm{{m}}") + " deep (" + m(tn(H / dmap['delta_20']['value'], 3))
                   + " times the 1 % depth at 20 s) with a shear-free top (symmetry plane): no momentum can leave."),
        dict(aspect="Start", real="Water at rest, plate jumps to " + m("U") + " at " + m("t = 0"),
             model="Velocity zero everywhere at " + m("t = 0") + "; the plate value applies from the first step."),
        dict(aspect="Fluid", real="Water", model=m(rf"\nu = {tn(nu)}\,\mathrm{{m^2/s}}") + "; OpenFOAM solves for "
             + m(r"p/\rho") + ", and stresses and momentum are multiplied by " + m(rf"\rho = {tn(rho)}\,\mathrm{{kg/m^3}}") + "."),
        dict(aspect="Outputs", real="Profiles, wall stress, depth, momentum",
             model="Fields every second; plate-averaged wall shear stress and the integral of " + m("u") + " over the layer "
                   "every time step (OpenFOAM function objects)."),
    ]
    meshes = dict(
        intro=f"{len(R)} runs in three studies. <b>Combined</b>: cell size and time step halved together "
              f"({len(fam)} levels). <b>Space</b>: cell size halved at the finest time step. <b>Time</b>: time step halved on "
              "the finest mesh. Every run covers the full " + m(rf"{tn(tend)}\,\mathrm{{s}}") + ".",
        head=["Run", m("N_y"), "Cell size " + m("h") + " (mm)", m(r"\Delta t") + " (s)", "Time steps", "Studies"],
        rows=[[x["name"], x["Ny"], m(tn(1e3 * x["h"], 3)), m(tn(x["dt"])), x["steps"], ", ".join(member[x["name"]])]
              for x in rows_sorted])
    results_tab = dict(
        head=["Run", m(r"|\tau_w|") + " at 20 s (Pa)", m("P") + " at 20 s (kg/(m s))"]
             + [m(r"\delta_{1\%}") + f" at {t:g} s (mm)" for t in T],
        rows=[["exact", m(tf(dmap["tau_w"]["value"], 6)), m(tf(dmap["P"]["value"], 6))]
              + [m(tf(1e3 * float(ex["delta"](t)), 5)) for t in T]]
             + [[x["name"], m(tf(x["tau_end"], 6)), m(tf(x["P_end"], 6))] + [m(tf(1e3 * x["delta"][t], 5)) for t in T]
                for x in rows_sorted])
    health = dict(
        head=["Run", "Log checks", "Momentum balance (max)", m("|u|/U") + " at the top", "Solver time (s)"],
        rows=[[x["name"], CHECK[x["health"]], m(tn(x["balance_max"], 2)), m(tn(x["far"], 2)), f"{x['wall']:.1f}"]
              for x in rows_sorted],
        notes=[
            "Each time step is a linear problem solved to round-off (velocity tolerance "
            + m(tn(numerics["linear_solvers"]["U_tol"])) + "). The convergence evidence is physical: the momentum of the "
            "water must change, step by step, by exactly the plate's drag (Figure {fig:balance}); the largest mismatch on "
            "any run is " + m(tn(max(x["balance_max"] for x in R.values()), 2)) + " of the final drag (limit "
            + m(tn(acc["momentum_balance_max"])) + ").",
            "Log checks (cfdlib.monitor): no crash, no NaN, Courant number and residual trends. The continuity error and "
            + m("v") + " are exactly zero in this flow; their trend checks only see round-off and are judged by their "
            "absolute level instead (see Insights).",
            "Top of the layer: the water there must stay at rest; largest " + m("|u|/U") + " in the top five cells at 20 s "
            "(limit " + m(tn(acc["far_field_max"])) + ").",
        ])

    defs = [
        dict(name="Profile error", latex=r"\varepsilon_u(t) = \max_i \frac{|u_i - u_{\mathrm{exact}}(y_i, t)|}{U}",
             meaning="Largest difference over all cells, as a fraction of the plate speed."),
        dict(name="Wall-stress error",
             latex=r"\varepsilon_\tau = \frac{\big||\tau_{w}| - |\tau_{w,\mathrm{exact}}|\big|}{|\tau_{w,\mathrm{exact}}|}",
             meaning="Relative error of the plate-averaged wall shear stress from OpenFOAM's wallShearStress function object."),
        dict(name="Momentum and depth errors",
             latex=r"\varepsilon_P = \frac{|P - P_{\mathrm{exact}}|}{P_{\mathrm{exact}}},\qquad "
                   r"\varepsilon_\delta = \frac{|\delta_{1\%} - \delta_{1\%,\mathrm{exact}}|}{\delta_{1\%,\mathrm{exact}}}",
             meaning="Relative errors; the depth is read from the profile by linear interpolation between cell centres."),
        dict(name="Observed order", latex=r"p = \frac{\ln(\varepsilon_{\mathrm{coarse}}/\varepsilon_{\mathrm{fine}})}{\ln 2}",
             meaning=m("p = 2") + " means halving cell size and time step cuts the error by 4."),
        dict(name="Momentum balance", latex=r"\frac{\big|\,\mathrm{d}_t\!\int u\,dy - \tau_w/\rho\,\big|}{|\tau_w(t_{\mathrm{end}})|/\rho}",
             meaning="Per time step, with the solver's own time derivative (Euler on the first step, BDF2 afterwards). "
                     "Zero up to round-off if every step is solved completely and no momentum is lost."),
    ]
    comp_rows = []
    for i, x in enumerate(fam):
        o = lambda key: m(f"{order[key][i - 1]:.2f}") if i else "–"
        comp_rows.append([x["name"], m(tp(x["eu"][T[0]])), m(tp(x["eu"][T[-1]])), o(("eu", T[-1])), m(tp(x["e_tau"])),
                          o("e_tau"), m(tp(x["e_delta_max"])), m(tp(x["e_P"]))])
    comp_findings = [
        "<b>How the wall-stress error scales.</b> The solver evaluates the stress from the plate to the first cell centre. "
        "Theory (last step) gives the truncation error of that two-point difference alone as "
        + m(r"-h^2/(48\,\nu t)") + " (it would make the stress too low). The measured error at 20 s is "
        + m(rf"{tn(c_lo, 3)}") + "–" + m(rf"{tn(c_hi, 3)}") + " times the size of that term on every level, so it follows the same "
        + m(r"h^2/(\nu t)") + " law, about " + m(rf"h^2/({tn(48 / np.mean(c_tau), 3)}\,\nu t)") + ", but the simulated stress is "
        + ("lower" if fine["tau_sign"] < 0 else "higher") + " than exact"
        + (": the error is dominated by the error of the near-wall cell values, not by the gradient formula. "
           if fine["tau_sign"] > 0 else ". ")
        + "Because it depends on how many cells span the diffusion length " + m(r"\sqrt{\nu t}")
        + ", it is large in the first steps and falls with time.",
        "<b>Space versus time.</b> With the time step fixed at its finest value, halving the cells gives order "
        + m(f"{order_mesh[-1]:.2f}") + ". On the finest mesh, halving the time step changes the 1 s profile by "
        + ", ".join(m(tp(v)) for v in d_time) + " of " + m("U") + " (order " + m(f"{order_time[-1]:.2f}")
        + "); the time error left in the finest run is about " + m(tp(time_err_fine)) + " of " + m("U")
        + ", against a total error of " + m(tp(fine["eu"][T[0]])) + " at 1 s.",
        "<b>Momentum.</b> The simulated momentum and the time integral of the simulated drag are the same number to round-off "
        "(Figure {fig:balance}), so the momentum error " + m(tp(fine["e_P"])) + " is the accumulated wall-stress error, "
        "not a conservation error.",
    ]

    results = dict(
        title=problem["title"], date=dt.date.today().isoformat(), verdict=verdict, comparison_type="verification",
        summary=dict(
            headline=(f"<b>Verification {'passed' if verdict == 'PASS' else 'with warnings' if verdict == 'WARN' else 'FAILED'}.</b> "
                      "OpenFOAM reproduces the exact solution of Stokes' first problem: on the finest run the velocity differs by "
                      "at most " + m(tp(fine["eu_max"])) + " of " + m("U") + ", and the wall shear stress at 20 s, "
                      + m(rf"{tn(fine['tau_end'], 4)}\ \mathrm{{Pa}}") + ", by " + m(tp(fine["e_tau"])) + "."),
            answer_columns=["Requested", "Theory (exact)", "Simulation (finest run)", "Difference"],
            answers=answers, findings=findings,
            limitations=[
                "The plate is infinite and the start instantaneous, as stated. The infinite wall stress at "
                + m("t = 0") + " cannot be reproduced by any finite mesh and time step; the first "
                + m(rf"{tn(fine['t_1pct'], 2)}\,\mathrm{{s}}") + " of the stress history are less accurate than 1 %.",
                "Laminar flow is assumed (the exact solution is laminar); whether a real layer at "
                + m(rf"Re_\delta \approx {tn(Re_d, 3)}") + " stays undisturbed was not checked (no reference supplied).",
                "The very deep water is a " + m(rf"{tn(1e3 * H, 3)}\,\mathrm{{mm}}") + " layer with a shear-free top; valid "
                "while the motion has not reached it, i.e. well beyond the 20 s asked for.",
            ]),
        problem=dict(
            description=problem["description"], setup=problem.get("setup", []),
            given=[dict(label=v["label"], symbol=v["symbol"], value=m(tn(v["value"])), unit=v["unit"]) for v in g.values()],
            wanted=wanted, reference=None),
        assumptions=dict(rows=assumptions),
        theory=dict(
            intro="This problem has an exact solution. SymPy reduces the equation to an ordinary differential equation with "
                  "the similarity variable, solves it, and checks the result against the equation, the plate condition, the "
                  "far field and the initial state. The requested quantities follow from it symbolically.",
            steps=theory["steps"], derived=theory["derived"],
            self_checks=[dict(name=k, ok=bool(v)) for k, v in theory["self_checks"].items()]),
        decisions=load_yaml(HERE / "decisions.yaml"),
        preflight=dict(
            purpose="Before production, the first " + m(rf"{tn(pf_raw[0]['end_time_short'])}\,\mathrm{{s}}") + " (5 %) of the "
                    "coarsest and finest combined-refinement runs check that the setup runs, stays healthy, records the "
                    "outputs every step, and give a run-time projection (reliable for a fixed-step transient run).",
            runs=pf_runs, checks=pf_checks,
            conclusion=("All pre-flight checks passed" if all(p["status"] == "ok" for p in pf_raw) else
                        "A pre-flight check did not pass") + f"; projected solver time of the longest run ≈ {est:.0f} s "
                        f"(measured in production: {fine['wall']:.0f} s, run alongside others). The first pre-flight "
                        "attempt was flagged by a round-off continuity-error trend; see Insights."),
        simulation=dict(model=model, meshes=meshes, results=results_tab, health=health),
        comparison=dict(
            kind="Verification: every run is compared with the exact solution of the same equations, at its own cell centres "
                 "and time steps.",
            definitions=defs,
            head=["Run", m(r"\varepsilon_u") + " 1 s", m(r"\varepsilon_u") + " 20 s", m("p"), m(r"\varepsilon_\tau") + " 20 s",
                  m("p"), m(r"\varepsilon_\delta") + " (max)", m(r"\varepsilon_P") + " 20 s"],
            rows=comp_rows, findings=comp_findings),
        insights=load_yaml(HERE / "insights.yaml"),
        figures=figures,
        reproducibility=dict(
            commands=[
                dict(cmd=f"python -m cases.{HERE.name}.run preflight", what="Short runs on the coarsest and finest levels; writes preflight.json"),
                dict(cmd=f"python -m cases.{HERE.name}.run production --jobs 4", what="All production runs, four at a time; writes runs/*.json, runs/*.npz"),
                dict(cmd=f"python -m cases.{HERE.name}.analyze", what="Theory, comparison, figures and this report (checks the contract first)"),
            ],
            files=[
                dict(path=f"cases/{HERE.name}/problem.yaml", what="The problem as given"),
                dict(path=f"cases/{HERE.name}/numerics.yaml", what="Domain, runs, tolerances and acceptance rules"),
                dict(path=f"cases/{HERE.name}/decisions.yaml", what="Every choice, with the OpenFOAM settings it controls"),
                dict(path=f"cases/{HERE.name}/insights.yaml", what="Workflow insights logged during the work"),
                dict(path=f"cases/{HERE.name}/template/", what="OpenFOAM case with {{ placeholders }}"),
                dict(path=f"cases/{HERE.name}/analytical.py", what="Exact solution derived and self-checked with SymPy"),
                dict(path=f"cases/{HERE.name}/run.py", what="Builds, pre-flights and runs the cases"),
            ],
            run_dirs=[
                dict(path=f"{json.loads((RUNS / (fine['name'] + '.json')).read_text())['run_root']}/N<cells>_dt<step>ms", what="Production runs"),
                dict(path=f"{json.loads((RUNS / (fine['name'] + '.json')).read_text())['run_root']}/pf_N<cells>_dt<step>ms_preflight", what="Pre-flight runs"),
                dict(path=f"results/{HERE.name}/", what="preflight.json, runs/, results.json and this report"),
            ],
            environment=report.environment_info(),
            notes=["Production runs were executed four at a time (one core each), which inflates the per-run solver times "
                   f"a little. The {len(R)} runs used {sum(x['wall'] for x in R.values()):.0f} s of solver time in total; "
                   f"the longest single run took {max(x['wall'] for x in R.values()):.0f} s."]),
        data=dict(order={str(k): v for k, v in order.items()}, order_mesh=order_mesh, order_time=order_time,
                  d_time=d_time, c_tau=c_tau, failures=failures, warns=warns,
                  runs={x["name"]: dict(eu={str(k): v for k, v in x["eu"].items()}, e_tau=x["e_tau"], e_P=x["e_P"],
                                        e_delta={str(k): v for k, v in x["e_delta"].items()}, t_1pct=x["t_1pct"],
                                        balance_max=x["balance_max"], far=x["far"]) for x in rows_sorted}),
    )
    out = report.render(results, svgs, OUT / "report.html", case_dir=fine["case_dir"])
    print(f"{verdict}: finest profile error {100 * fine['eu_max']:.4f} % of U, order {p_main:.3f}. Report: {out}")


def log_ticks(ax, vals, which):
    """Ticks exactly at the given values on a log axis, plain numbers, no minor-tick labels."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(FixedLocator(list(vals)))
    axis.set_major_formatter(plt_formatter())
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def plt_formatter():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _: f"{v:g}")


def tf(x, sig):
    """Plain number with a FIXED count of significant digits (keeps trailing zeros, unlike tex_num),
    so table columns line up. Only for values between 1e-3 and 1e5 (no exponent)."""
    assert 1e-3 <= abs(x) < 1e5, x
    return f"{x:#.{sig}g}".rstrip(".")


def _mk(k):
    """Marker style: k = 1 finest run (small, filled), k = 0 coarsest (larger, hollow, drawn on top)."""
    if k:
        return dict(ms=3.5, color=plots.SERIES[1], mec=plots.SURFACE, mew=0.5)
    return dict(ms=7, color=plots.SERIES[0], mfc="none", mew=1.2)


def plt_color(k):
    """Series colour k; beyond the 4 categorical slots, fall back to ink."""
    return plots.SERIES[k] if k < len(plots.SERIES) else plots.INK


def _all(numerics):
    from .run import all_runs
    return all_runs(numerics)


if __name__ == "__main__":
    main()
