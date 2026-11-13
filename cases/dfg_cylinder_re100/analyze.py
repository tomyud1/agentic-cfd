"""Compare the cylinder simulations with the FeatFlow benchmark; mesh and time-step studies; results.json + report.

Usage (repo root, inside WSL):  python -m cases.dfg_cylinder_re100.analyze
Reads results/dfg_cylinder_re100/{preflight.json, runs/*.json|npz, perf/*.json}, the case YAML files and the
reference data in reference/. Every number in the report is computed here.
"""
import datetime as dt
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from cfdlib import plots, report, verification as V
from cfdlib.report import LETTERS, load_yaml, m, tex_num as tn, tex_fix as tf
from . import signals as S
from . import theory
from .run import mesh_spec
from .sketch import sketch

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"
PERF = OUT / "perf"
QKEYS = ["cd_max", "cd_min", "cd_mean", "cd_amp", "cl_max", "cl_min", "cl_mean", "cl_amp", "St"]
SYM = dict(cd_max=r"c_{D,\max}", cd_min=r"c_{D,\min}", cd_mean=r"\bar c_D", cd_amp=r"\Delta c_D",
           cl_max=r"c_{L,\max}", cl_min=r"c_{L,\min}", cl_mean=r"\bar c_L", cl_amp=r"\Delta c_L", St="St")
DEC = dict(cd_max=4, cd_min=4, cd_mean=4, cd_amp=4, cl_max=4, cl_min=4, cl_mean=4, cl_amp=4, St=4)
PISO_RUNS = ["piso_M2_dtx2", "piso_M2", "piso_M2_dtx0.5", "X_M2_o1_dt025"]
ST = {"ok": "✓ OK", "warn": "⚠ warning", "fail": "✗ fail"}


# ---------------------------------------------------------------------------------------------
# Reference: FeatFlow tables (levels 3-6, three time steps)
# ---------------------------------------------------------------------------------------------
def reference(ref):
    rows = {}
    for tab, keys in (("drag", ("cd_min", "cd_max", "cd_mean", "cd_amp")), ("lift", ("cl_min", "cl_max", "cl_mean", "cl_amp")),
                      ("period", ("St", "period"))):
        for r in ref[tab]:
            key = (r["level"], r["dt"])
            rows.setdefault(key, dict(level=r["level"], dt=r["dt"]))
            for k in keys:
                rows[key][k] = float(r[k])
    rows = list(rows.values())
    top = max(r["level"] for r in rows)
    best = [r for r in rows if r["level"] == top]
    assert len(best) == 1, "expected one row on the finest reference level"
    best = best[0]
    below = [r for r in rows if r["level"] == top - 1]
    val = {k: best[k] for k in QKEYS}
    unc = {k: max(abs(best[k] - r[k]) for r in below) for k in QKEYS}
    # The reference's own mesh convergence: levels top-2, top-1, top at the finest-level time step
    seq = {}
    for k in QKEYS:
        f = [next(r[k] for r in rows if r["level"] == lv and r["dt"] == best["dt"]) for lv in (top - 2, top - 1, top)]
        R = (f[1] - f[2]) / (f[0] - f[1]) if f[0] != f[1] else float("nan")
        seq[k] = dict(values=f, R=R, p=V.order_three_grids(*f) if 0 < R < 1 else float("nan"),
                      extrap=V.richardson(f[1], f[2], V.order_three_grids(*f)) if 0 < R < 1 else float("nan"))
    return dict(rows=rows, top=top, best=best, below=below, value=val, unc=unc, seq=seq, dt=best["dt"])


# ---------------------------------------------------------------------------------------------
# One run: coefficients, cycles, frequencies, momentum balance
# ---------------------------------------------------------------------------------------------
def analyse_run(name, given, numerics):
    meta = json.loads((RUNS / f"{name}.json").read_text())
    d = np.load(RUNS / f"{name}.npz")
    dz = numerics["mesh"]["dz"]
    D, Um = given["D"]["value"], given["Um"]["value"]
    Umean = 2 * Um / 3
    t, cd, cl = S.coefficients(d, given, dz)
    cyc = S.cycles(t, cd, cl)
    n = numerics["time"]["cycles_measured"]
    sm = S.summary(cyc, n)
    sm["St"] = sm["f"] * D / Umean
    win = (t >= cyc[-n]["t0"]) & (t <= cyc[-1]["t1"])
    tw = t[win]
    f_fft = S.dominant_frequency(tw, cl[win])
    fD_fft = S.dominant_frequency(tw, cd[win])
    tp, pv = d["probe_t"], d["probe_U"][:, 1]
    sel = (tp >= tw[0]) & (tp <= tw[-1])
    f_probe = S.dominant_frequency(tp[sel], pv[sel])
    # Time-averaged x-momentum balance over the measured cycles (kinematic, per unit depth)
    T0, T1 = cyc[-n]["t0"], cyc[-1]["t1"]

    def avg(key, comp=None, tkey="t"):
        tt = d[tkey]
        y = d[key] if comp is None else d[key][:, comp]
        s_ = (tt >= T0) & (tt <= T1)
        return float(np.trapezoid(y[s_], tt[s_]) / (tt[s_][-1] - tt[s_][0]))

    Fc = avg("Fcyl", 0) / dz
    Fw = avg("Fwall", 0) / dz
    Pin, Pout = avg("p_in", tkey="p_in_t") / dz, avg("p_out", tkey="p_out_t") / dz
    MFin = avg("mom_in", 0, "mom_in_t") / dz
    MFout = avg("mom_out", 0, "mom_out_t") / dz
    mt, mx = d["momentum_t"], d["momentum"][:, 0] / dz
    dM = float(np.interp(T1, mt, mx) - np.interp(T0, mt, mx)) / (T1 - T0)
    rhs = Pin - Pout - (MFin + MFout) - dM
    balance = dict(F_cyl=Fc, F_wall=Fw, P_in=Pin, P_out=Pout, MF_in=MFin, MF_out=MFout, dMdt=dM,
                   imbalance=Fc + Fw - rhs, rel=(Fc + Fw - rhs) / Fc)
    qin, qout = d["q_in"], d["q_out"]
    mass = float(np.max(np.abs(qin + qout)) / np.max(np.abs(qin)))
    inflow_excess = float(-np.mean(qin) / (Umean * given["H"]["value"] * dz) - 1)
    return dict(name=name, meta=meta, t=t, cd=cd, cl=cl, cycles=cyc, sm=sm, f_fft=f_fft, fD_fft=fD_fft, f_probe=f_probe,
                ratio_fD=fD_fft / f_fft, balance=balance, mass=mass, inflow_excess=inflow_excess, T0=T0, T1=T1)


def study(values, r=2.0):
    """Observed order, Richardson value and fine-mesh GCI from three solutions (coarse, medium, fine).
    When the three do not converge monotonically, order 2 is assumed and `gci_reliable` is False: the index
    is then a placeholder, not an estimate (cfdlib.verification.three_grid_study)."""
    return V.three_grid_study(values, r)


def main():
    problem = load_yaml(HERE / "problem.yaml")
    numerics = load_yaml(HERE / "numerics.yaml")
    acc = numerics["acceptance"]
    g = problem["given"]
    ref = reference(load_yaml(HERE / problem["reference"]["file"]))
    th = theory.evaluate(g)
    dv = {d["key"]: d["value"] for d in th["derived"]}
    Umean, D = th["Umean"], g["D"]["value"]
    levels = numerics["mesh"]["levels"]
    have = [L for L in levels if (RUNS / f"{L}.json").exists()]
    missing = [L for L in levels if L not in have]
    runs = {L: analyse_run(L, g, numerics) for L in have}
    ts_names = [f"{numerics['timestep_study']['mesh']}_dtx{f:g}" for f in numerics["timestep_study"]["factors"]]
    for nm in ts_names + [f"{numerics['sensitivity']['mesh']}_LU"]:
        if (RUNS / f"{nm}.json").exists():
            runs[nm] = analyse_run(nm, g, numerics)
    fine = runs[have[-1]]
    # First attempt (PISO, one outer corrector), kept as evidence of its first-order time error (insight 10)
    piso = {n: analyse_run(n, g, numerics) for n in PISO_RUNS if (RUNS / f"{n}.json").exists()}
    preflight = json.loads((OUT / "preflight.json").read_text())

    # ---- mesh study (family: h and dt halved together) on the three finest meshes ---------------
    vals = {k: [runs[L]["sm"][k] for L in have] for k in QKEYS}
    ms = {k: study(vals[k]) for k in QKEYS} if len(have) >= 3 else {}
    ms_coarse = {k: study(vals[k][:3]) for k in QKEYS} if len(have) >= 3 else {}

    # ---- time-step study on M2 ------------------------------------------------------------------
    tsm = numerics["timestep_study"]["mesh"]
    tsr = sorted([runs[n] for n in ts_names if n in runs] + [runs[tsm]], key=lambda r: -r["meta"]["dt"])
    tstudy, u_time = {}, {k: 0.0 for k in QKEYS}
    if len(tsr) == 3:
        for k in QKEYS:
            st = study([r["sm"][k] for r in tsr])
            # Time error of the baseline (largest dt = the production setting) projected to the finest run's dt.
            # Monotonic: Richardson in dt. Otherwise the changes are not a clean dt-power law (e.g. at the level of
            # the residual start-up transient): take the largest change seen, unscaled (conservative).
            base = runs[tsm]["sm"][k]
            if st["monotonic"]:
                e_base = abs(base - st["extrap"])
                scale = (fine["meta"]["dt"] / runs[tsm]["meta"]["dt"]) ** st["p"]
            else:
                e_base = max(abs(r["sm"][k] - base) for r in tsr)
                scale = 1.0
            st.update(e_base=e_base, e_fine=e_base * scale, p_t=st["p"])
            tstudy[k] = st
            u_time[k] = e_base * scale

    # ---- comparison with the reference ----------------------------------------------------------
    comp = {}
    for k in QKEYS:
        sim = fine["sm"][k]
        u_sim = (ms[k]["gci_abs"] if ms else float("nan")) + u_time[k]
        band = u_sim + ref["unc"][k]
        diff = sim - ref["value"][k]
        status = "ok" if abs(diff) <= band else ("warn" if abs(diff) <= 2 * band else "fail")
        ext = ms[k]["extrap"] if ms else float("nan")
        comp[k] = dict(sim=sim, ref=ref["value"][k], u_ref=ref["unc"][k], u_sim=u_sim, u_time=u_time[k], band=band,
                       diff=diff, rel=diff / abs(ref["value"][k]), ratio=abs(diff) / band if band else float("inf"),
                       status=status, extrap=ext, diff_ext=ext - ref["value"][k])

    # ---- verdict (rules fixed in numerics.yaml before the results were seen) ----------------------
    fail, warn = [], []
    if missing:
        fail.append(f"planned meshes not run: {missing}")
    for nm, r in runs.items():
        if not r["meta"]["mesh_ok"]:
            fail.append(f"checkMesh reported a problem ({nm})")
        if r["meta"]["health"]["status"] == "fail":
            fail.append(f"run {nm} failed a log health check")
    per = {}
    for nm, r in runs.items():
        sp = r["sm"]["spread"]
        ok = (max(sp["cl_max"], sp["cl_min"]) <= acc["periodic_cl_peak"] and max(sp["cd_max"], sp["cd_min"]) <= acc["periodic_cd_peak"]
              and sp["period"] / r["sm"]["period"] <= acc["periodic_period_rel"])
        per[nm] = ok
        if not ok:
            (fail if nm == have[-1] else warn).append(f"run {nm} not periodic to the set tolerance over its last "
                                                       f"{numerics['time']['cycles_measured']} cycles")
    if abs(fine["balance"]["rel"]) > acc["momentum_balance_rel"]:
        fail.append("time-averaged momentum balance not closed on the finest mesh")
    for k in acc["primary"]:
        c = comp[k]
        if c["status"] == "fail":
            fail.append(f"{m(SYM[k])} differs from the reference by more than twice the combined uncertainty")
        elif c["status"] == "warn":
            warn.append(f"{m(SYM[k])} differs from the reference by more than the combined uncertainty (but less than twice)")
    for k in acc["secondary"]:
        if comp[k]["status"] == "fail":
            warn.append(f"{m(SYM[k])} (secondary) differs from the reference by more than twice the combined uncertainty")
    if ms:
        for k in acc["primary"]:
            if not ms[k]["monotonic"]:
                warn.append(f"{m(SYM[k])}: convergence on the three finest meshes not monotonic (order undefined)")
            elif not acc["order_range"][0] <= ms[k]["p"] <= acc["order_range"][1]:
                warn.append(f"{m(SYM[k])}: observed order {m(tf(ms[k]['p'], 2))} outside "
                            f"{m(tn(acc['order_range'][0]) + r' \le p \le ' + tn(acc['order_range'][1]))}")
    if abs(fine["ratio_fD"] / 2 - 1) > acc["drag_freq_ratio_tol"]:
        warn.append("drag frequency is not twice the lift frequency")
    verdict = "FAIL" if fail else "WARN" if warn else "PASS"

    data = dict(fail=fail, warn=warn, comp=comp, ms=ms, ms_coarse=ms_coarse, tstudy=tstudy, ref=ref,
                periodic=per, runs={nm: dict(sm=r["sm"], balance=r["balance"], f_fft=r["f_fft"], fD_fft=r["fD_fft"],
                                              f_probe=r["f_probe"], mass=r["mass"], inflow_excess=r["inflow_excess"],
                                              meta={k: v for k, v in r["meta"].items() if k not in ("health",)})
                                    for nm, r in runs.items()})
    (OUT / "analysis.json").write_text(json.dumps(data, indent=1, default=float))
    print("verdict", verdict, "fail", fail, "warn", warn)
    for k in QKEYS:
        c = comp[k]
        print(f"{k:8s} sim {c['sim']:.5f} ref {c['ref']:.5f} diff {c['diff']:+.5f} band {c['band']:.5f} "
              f"(sim {c['u_sim']:.5f} time {c['u_time']:.5f} ref {c['u_ref']:.5f}) {c['status']}; "
              + (f"p {ms[k]['p']:.2f} R {ms[k]['R']:.3f} ext {ms[k]['extrap']:.5f}" if ms else ""))
    for nm, r in runs.items():
        b = r["balance"]
        print(f"{nm}: St {r['sm']['St']:.5f} fft {r['f_fft'] * D / Umean:.5f} probe {r['f_probe'] * D / Umean:.5f} fD/fL "
              f"{r['ratio_fD']:.5f} balance {b['rel']:+.2e} mass {r['mass']:.1e} periodic {per[nm]} spread {r['sm']['spread']}")
    if "--report" in __import__("sys").argv:
        build_report(problem, numerics, g, ref, th, runs, have, fine, ms, ms_coarse, tstudy, comp, per, fail, warn, verdict, piso,
                     preflight, data)


def _co_max(meta):
    c = next((c for c in meta["health"]["checks"] if c["name"] == "Max Courant number"), None)
    return float(c["value"]) if c else float("nan")


def build_report(problem, numerics, g, ref, th, runs, have, fine, ms, ms_coarse, tstudy, comp, per, fail, warn, verdict, piso,
                 preflight, data):
    acc = numerics["acceptance"]
    D, Um, H = g["D"]["value"], g["Um"]["value"], g["H"]["value"]
    Umean = th["Umean"]
    dv = {d["key"]: d["value"] for d in th["derived"]}
    fl = have[-1]
    fm = fine["meta"]
    nmeas = numerics["time"]["cycles_measured"]
    wanted = [dict(letter=LETTERS[i], key=w["key"], text=w["text"]) for i, w in enumerate(problem["quantities_of_interest"])]
    L_of = {w["key"]: w["letter"] for w in wanted}
    wake = np.load(RUNS / f"{fl}_wake.npz") if (RUNS / f"{fl}_wake.npz").exists() else None
    ref_lab = f"FeatFlow level {ref['top']}, " + m(rf"\Delta t = {ref['dt']}\,\mathrm{{s}}")

    def q(k, v=None, dec=None):
        return tf(comp[k]["sim"] if v is None else v, dec or DEC[k])

    def refcell(keys):
        return "<br>".join(m(rf"{SYM[k]} = {tf(ref['value'][k], DEC[k])} \pm {tn(ref['unc'][k], 2)}") for k in keys)

    def simcell(keys):
        return "<br>".join(m(rf"{SYM[k]} = {q(k)}") for k in keys)

    def diffcell(keys):
        return "<br>".join(m(rf"{comp[k]['diff']:+.4f}") + f" ({tf(comp[k]['ratio'], 2)} band) {ST[comp[k]['status']][0]}"
                           for k in keys)

    sm = fine["sm"]
    f_ref = ref["value"]["St"] * Umean / D
    ans = {
        "drag": dict(theory=refcell(["cd_max", "cd_min", "cd_mean", "cd_amp"]), sim=simcell(["cd_max", "cd_min", "cd_mean", "cd_amp"]),
                     diff=diffcell(["cd_max", "cd_min", "cd_mean", "cd_amp"])),
        "lift": dict(theory=refcell(["cl_max", "cl_min", "cl_mean", "cl_amp"]), sim=simcell(["cl_max", "cl_min", "cl_mean", "cl_amp"]),
                     diff=diffcell(["cl_max", "cl_min", "cl_mean", "cl_amp"])),
        "strouhal": dict(theory=m(rf"f = {tf(f_ref, 3)}\,\mathrm{{Hz}}") + "<br>" + refcell(["St"]),
                         sim=m(rf"f = {tf(sm['f'], 3)}\,\mathrm{{Hz}}") + "<br>" + simcell(["St"]),
                         diff=diffcell(["St"])),
        "wake": dict(theory="Alternating vortices of opposite sign shed from the two sides (Theory, step 4)",
                     sim=(("Vorticity at the lift maximum, Figure {fig:wake}" if wake is not None else "picture not produced") + (
                         "<div class='sub'>" + m(rf"t = {tn(float(wake['t']), 6)}\,\mathrm{{s}}") + ", "
                         + m(rf"c_L = {tf(float(wake['cl']), 4)}") + "</div>" if wake is not None else "")),
                     diff=(m(rf"{tn(abs(float(wake['t']) - float(wake['t_clmax'])) / sm['period'] * 100, 2)}\,\%") +
                           " of a period from the exact maximum" if wake is not None else "–")),
    }
    answers = [dict(letter=w["letter"], label=w["text"], **ans[w["key"]]) for w in wanted]

    # ---- figures ------------------------------------------------------------------------------
    pv = [piso[n]["sm"]["cl_amp"] for n in sorted(piso, key=lambda n: -piso[n]["meta"]["dt"])]
    piso_p = V.order_three_grids(*pv[-3:]) if len(pv) >= 3 else float("nan")
    tsn = [n for n in runs if n == numerics["timestep_study"]["mesh"] or n.startswith(numerics["timestep_study"]["mesh"] + "_dtx")]
    pimple_dmax = max(runs[n]["sm"]["cl_amp"] for n in tsn) - min(runs[n]["sm"]["cl_amp"] for n in tsn)
    svgs = {"sketch": sketch(g, numerics["probes"][0])}
    svgs["forces"] = _forces_fig(fine, nmeas)
    if wake is not None:
        svgs["wake"] = _wake_fig(wake, g)
    svgs["periodic"] = _periodic_fig(runs, have, acc["periodic_cl_peak"])
    svgs["cycle"] = _cycle_fig(runs, have, ref)
    svgs["conv"] = _conv_fig(runs, have, ms, comp, ref)
    if tstudy:
        svgs["timestep"] = _timestep_fig(runs, numerics, piso, ref)
    scaling = json.loads((PERF / "scaling.json").read_text()) if (PERF / "scaling.json").exists() else []
    if scaling:
        svgs["scaling"] = _scaling_fig(scaling)
    late_change = {L: float(np.max(np.abs(np.diff([c["cl_max"] for c in runs[L]["cycles"][-nmeas:]])))) for L in have}
    figures = [
        dict(key="sketch", section="problem", caption="The channel with the cylinder (to scale), the parabolic inflow and the "
             "free outflow. The orange arrows are the requested forces, the dotted circles a sketch of the vortex street; the "
             "square marks the wake probe used as a second frequency record."),
        dict(key="forces", section="simulation", caption=f"Drag and lift coefficients on the finest mesh ({fl}). The run starts "
             f"from the developed flow of the next coarser mesh, interpolated onto this mesh; the jolt at the start (drag axis clipped) is that "
             f"interpolation settling. The shaded band is the {nmeas} cycles used for the answers. Drag oscillates at twice "
             f"the lift frequency (measured ratio " + m(tf(fine['ratio_fD'], 4)) + ")."),
    ]
    if wake is not None:
        figures.append(dict(key="wake", section="simulation", caption="Output (" + L_of["wake"] + "): vorticity "
                            + m(r"\omega_z = \partial v/\partial x - \partial u/\partial y") + f" on the finest mesh at the "
                            "moment of maximum lift (written field closest to it). Red: anticlockwise rotation, blue: clockwise. "
                            "Vortices of alternating sign leave the upper and lower sides of the cylinder and are carried "
                            "downstream as a staggered street, which slowly diffuses on the way to the outlet."))
    figures += [
        dict(key="periodic", section="convergence", caption="Physical check that each run reached the periodic state: "
             "change of the lift maximum from one cycle to the next. After the start (from rest on M1, from the "
             "interpolated coarser flow on the others) it falls "
             + ("below" if all(v <= acc["periodic_cl_peak"] for v in late_change.values()) else "towards")
             + " the tolerance fixed before the runs (dashed). "
             + " ".join(f"{L} ends at " + m(tn(late_change[L], 2)) + "." for L in have)
             + " The answers use the last " + str(nmeas) + " cycles of each run."),
        dict(key="cycle", section="comparison", caption="One shedding cycle on each mesh (time from an upward zero crossing of "
             "the lift, in units of the period). Dashed: reference maxima and minima (" + ref_lab + "). The coarse meshes "
             "overpredict the lift amplitude; refinement moves it towards the reference."),
        dict(key="conv", section="comparison", caption="Mesh study for three of the requested numbers, against the cell size "
             "relative to the finest mesh (the time step is halved with the cell size). Dots: the four meshes; open diamond: "
             "Richardson extrapolation from the three finest, with its GCI bar; shaded: the reference value with its own "
             "spread (level 5 versus level 6)."),
    ]
    if tstudy:
        figures.append(dict(key="timestep", section="comparison", caption=f"Time-step study on {numerics['timestep_study']['mesh']} "
                            "(same mesh, each run started from the same developed flow). Orange: the first plan, PISO with one "
                            "outer corrector: the lift amplitude converges with observed order " + m(tf(piso_p, 2))
                            + " in the time step and is far from converged at the planned step. Blue: PIMPLE with two outer "
                            "correctors, used for all production runs: the lift amplitude varies by at most " + m(tn(pimple_dmax, 2))
                            + " over a four-fold range of time steps. Dotted: the reference."))
    if scaling:
        figures.append(dict(key="scaling", section="performance", caption="MPI scaling on the finest mesh (" + f"{scaling[0]['cells']:,}"
                            + " cells, the same 200 time steps): speed-up over one core (left) and wall time per cell and "
                            "time step inside the time loop (right). The dashed line is ideal scaling."))

    # ---- tables -------------------------------------------------------------------------------
    names_all = have + [n for n in runs if n not in have]
    spec = {L: mesh_spec(L, problem, numerics) for L in have}
    delta = dv["delta"]
    mesh_rows = []
    for L in have:
        s_, r = spec[L], runs[L]
        mesh_rows.append([L, f"{s_['cells']:,}", str(4 * s_["nc"]), m(tn(s_["h"]["wall"] * 1000, 3) + r"\,\mathrm{mm}"),
                          m(tn(delta / s_["h"]["wall"], 3)), m(tn(r["meta"]["dt"], 3) + r"\,\mathrm{s}"),
                          m(tn(_co_max(r["meta"]), 2)), f"{r['sm']['period'] / r['meta']['dt']:.0f}"])
    res_rows = []
    for nm in names_all:
        r = runs[nm]
        res_rows.append([nm] + [m(q(k, r["sm"][k])) for k in QKEYS])
    if ms:
        res_rows += [
            ["Observed order " + m("p") + " (– not monotonic)"] +[m(tf(ms[k]["p"], 2)) if ms[k]["monotonic"] else "–" for k in QKEYS],
            ["Richardson extrapolation"] + [m(q(k, ms[k]["extrap"])) for k in QKEYS],
            ["GCI, finest mesh (absolute)"] + [m(tn(ms[k]["gci_abs"], 2)) for k in QKEYS],
            [f"Reference (level {ref['top']})"] + [m(q(k, ref["value"][k])) for k in QKEYS],
            ["Reference spread " + m("U_{\\mathrm{ref}}")] + [m(tn(ref["unc"][k], 2)) for k in QKEYS],
        ]
    health_rows = []
    for nm in names_all:
        r = runs[nm]
        sp = r["sm"]["spread"]
        health_rows.append([nm, f"{r['meta']['perf']['steps']}", m(tn(_co_max(r["meta"]), 2)),
                            ST[r["meta"]["health"]["status"]],
                            m(tn(max(sp["cl_max"], sp["cl_min"]), 2)), m(tn(max(sp["cd_max"], sp["cd_min"]), 2)),
                            m(tn(sp["period"] / r["sm"]["period"], 2)), "✓" if per[nm] else "✗",
                            m(tn(r["balance"]["rel"], 2)), m(tn(r["mass"], 2)), f"{r['meta']['wall_time']:.0f}"])
    comp_rows = []
    for k in QKEYS:
        c = comp[k]
        comp_rows.append([m(SYM[k]), m(q(k, c["ref"])), m(tn(c["u_ref"], 2)), m(q(k)), m(tn(c["u_sim"], 2)),
                          m(f"{c['diff']:+.4f}"), m(tn(100 * c["rel"], 2) + r"\,\%"), m(tf(c["ratio"], 2)),
                          ST[c["status"]] + ("" if k in acc["primary"] else " (secondary)"),
                          m(q(k, c["extrap"])) if ms else "–"])

    # ---- findings -----------------------------------------------------------------------------
    worst = max(acc["primary"], key=lambda k: comp[k]["ratio"])
    b = fine["balance"]
    findings = [
        f"<b>Drag ({L_of['drag']}).</b> " + m(rf"\bar c_D = {q('cd_mean')}") + " against " + m(tf(ref["value"]["cd_mean"], 4))
        + " (" + m(tn(100 * comp["cd_mean"]["rel"], 2) + r"\,\%") + "); maximum " + m(q("cd_max")) + " against "
        + m(tf(ref["value"]["cd_max"], 4)) + ". The drag amplitude " + m(rf"\Delta c_D = {q('cd_amp')}") + " is "
        + m(tn(100 * comp["cd_amp"]["rel"], 2) + r"\,\%") + " off the reference.",
        f"<b>Lift ({L_of['lift']}).</b> " + m(rf"c_{{L,\max}} = {q('cl_max')}") + ", " + m(rf"c_{{L,\min}} = {q('cl_min')}")
        + ", amplitude " + m(q("cl_amp")) + " against " + m(tf(ref["value"]["cl_amp"], 4)) + " ("
        + m(tn(100 * comp["cl_amp"]["rel"], 2) + r"\,\%") + "). The small negative mean lift "
        + m(rf"\bar c_L = {q('cl_mean')}") + " comes from the cylinder sitting slightly below the centreline.",
        f"<b>Frequency ({L_of['strouhal']}).</b> " + m(rf"f = {tf(sm['f'], 4)}\,\mathrm{{Hz}}") + ", "
        + m(rf"St = {q('St')}") + " against " + m(tf(ref["value"]["St"], 4)) + " (" + m(tn(100 * comp["St"]["rel"], 2) + r"\,\%")
        + "). Lift zero crossings, the FFT of the lift and the wake probe agree to "
        + m(tn(max(abs(fine["f_fft"] - sm["f"]), abs(fine["f_probe"] - sm["f"])) / sm["f"], 2)) + " (relative).",
        "<b>Largest disagreement</b> relative to the combined uncertainty band: " + m(SYM[worst]) + ", "
        + m(tf(comp[worst]["ratio"], 2)) + " times the band (simulation GCI plus time error plus reference spread).",
        "<b>Mesh convergence.</b> " + (", ".join(m(SYM[k]) + " " + (m(f"p = {ms[k]['p']:.2f}") if ms[k]["monotonic"] else "not monotonic")
                                            for k in ("cd_mean", "cl_amp", "St")) if ms else "fewer than three meshes")
        + " on the three finest meshes (time step halved with the cell size; design order 2)."
        + ((" The mean drag changes by at most " + m(tn(max(abs(np.diff([runs[L]["sm"]["cd_mean"] for L in have[-3:]]))), 2))
            + " between them, too little to give an order.") if ms and not ms["cd_mean"]["monotonic"] else ""),
        "<b>Periodic and conserving.</b> On every run the peaks change by at most "
        + m(tn(max(max(r["sm"]["spread"]["cl_max"], r["sm"]["spread"]["cl_min"]) for r in runs.values()), 2))
        + " (lift) between the last cycles; the time-averaged momentum balance of the channel closes to "
        + m(tn(abs(b["rel"]) * 100, 2) + r"\,\%") + " of the drag on the finest mesh.",
        "<b>The same case on the project's GPU solver.</b> gpucfd ran this problem on grids of equal cells and of cells "
        "clustered around the cylinder. Accuracy, cost and limits next to the OpenFOAM runs of this report: "
        "<a href='gpu_study.html'>gpucfd study report</a> (cases/dfg_cylinder_re100/gpu_study.py).",
    ] + [f"<b>Rule failed:</b> {f}." for f in fail] + [f"<b>Caveat:</b> {w}." for w in warn]

    # comparison findings, computed
    ref_seq = ref["seq"]
    comp_findings = [
        "<b>The reference has its own spread.</b> From level " + str(ref["top"] - 1) + " to " + str(ref["top"])
        + " FeatFlow's " + m(SYM["cd_max"]) + " still changes by " + m(tn(abs(ref_seq["cd_max"]["values"][2] - ref_seq["cd_max"]["values"][1]), 2))
        + ((", and its levels " + str(ref["top"] - 2) + "-" + str(ref["top"]) + " converge with order " + m(tf(ref_seq["cd_max"]["p"], 2))
            + " towards " + m(tf(ref_seq["cd_max"]["extrap"], 4))) if not math.isnan(ref_seq["cd_max"]["p"]) else "")
        + ". The level-" + str(ref["top"]) + " value is therefore used with an uncertainty equal to its largest "
        "difference from the three level-" + str(ref["top"] - 1) + " results (all three time steps); for the Strouhal number "
        "this also covers the period being read only to about one time step.",
    ]
    if not math.isnan(ref_seq["cd_mean"]["extrap"]):
        comp_findings.append(
            "<b>Drag against the reference's own extrapolation.</b> The simulated " + m(SYM["cd_mean"]) + " lies "
            + m(f"{comp['cd_mean']['diff']:+.4f}") + " from the level-" + str(ref["top"]) + " value but only "
            + m(f"{comp['cd_mean']['sim'] - ref_seq['cd_mean']['extrap']:+.4f}") + " from the value the reference's levels "
            + str(ref["top"] - 2) + "-" + str(ref["top"]) + " extrapolate to (" + m(tf(ref_seq["cd_mean"]["extrap"], 4))
            + ", observed order " + m(tf(ref_seq["cd_mean"]["p"], 2)) + ")."
            + (" The drag difference is therefore mostly the reference's remaining discretisation error, not the "
               "simulation's." if abs(comp["cd_mean"]["sim"] - ref_seq["cd_mean"]["extrap"]) < 0.5 * abs(comp["cd_mean"]["diff"])
               else " The extrapolation does not explain the difference.")
            + " The verdict still uses the level-" + str(ref["top"]) + " value and the spread rule fixed beforehand.")
    if ms:
        dd = [k for k in acc["primary"] if abs(comp[k]["diff_ext"]) < abs(comp[k]["diff"])]
        comp_findings.append(
            "<b>Extrapolation.</b> Richardson extrapolation of the three finest meshes moves "
            + str(len(dd)) + " of " + str(len(acc["primary"])) + " primary numbers closer to the reference "
            "(table, last column). " + ", ".join(m(SYM[k]) + ": " + m(f"{comp[k]['diff']:+.4f}") + r" \(\to\) "
                                                    + m(f"{comp[k]['diff_ext']:+.4f}") for k in ("cd_max", "cl_amp", "St")) + ".")
    if tstudy:
        tsm_ = numerics["timestep_study"]["mesh"]
        d_t = {k: max(abs(runs[n]["sm"][k] - runs[tsm_]["sm"][k]) for n in ts_names_of(numerics) if n in runs) for k in ("cl_amp", "St")}
        d_h = {k: abs(runs[have[-1]]["sm"][k] - runs[have[-2]]["sm"][k]) for k in ("cl_amp", "St")}
        small = all(d_t[k] < 0.5 * d_h[k] for k in d_t)
        comp_findings.append(
            ("<b>Time-step error is small.</b> " if small else "<b>Time-step error is not negligible.</b> ")
            + "On " + tsm_ + ", reducing the time step up to four times changes " + m(SYM["cl_amp"]) + " by at most "
            + m(tn(d_t["cl_amp"], 2)) + " and " + m("St") + " by " + m(tn(d_t["St"], 2)) + ", against "
            + m(tn(d_h["cl_amp"], 2)) + " and " + m(tn(d_h["St"], 2)) + " between the two finest meshes "
            + have[-2] + r" \(\to\) " + have[-1] + (": the mesh, not the time step, dominates the remaining error." if small else "."))
    if piso:
        pn = sorted(piso, key=lambda n: -piso[n]["meta"]["dt"])
        comp_findings.append(
            "<b>The first plan was first order in time.</b> With PISO (one outer corrector) the M2 lift amplitude was "
            + ", ".join(m(tf(piso[n]["sm"]["cl_amp"], 3)) for n in pn) + " for "
            + ", ".join(m(tn(piso[n]["meta"]["dt"] * 1e3, 2)) for n in pn) + r" \(\mathrm{ms}\) (observed order "
            + m(tf(piso_p, 2)) + "): the face flux that convects the momentum lags one step, which adds an error proportional "
            "to the time step. That family of runs looked like slow mesh convergence, because the time step was refined "
            "together with the mesh. With two outer correctors the same mesh gives " + m(tf(runs[numerics['timestep_study']['mesh']]["sm"]["cl_amp"], 3))
            + " at " + m(tn(runs[numerics['timestep_study']['mesh']]["meta"]["dt"] * 1e3, 2) + r"\,\mathrm{ms}") + ". All production runs were repeated with two outer correctors (Insights).")
    sname = f"{numerics['sensitivity']['mesh']}_LU"
    if sname in runs:
        base = runs[numerics["sensitivity"]["mesh"]]["sm"]
        lu = runs[sname]["sm"]
        comp_findings.append(
            "<b>Convection scheme.</b> " + numerics["sensitivity"]["mesh"] + " with linear-upwind instead of central convection: "
            + ", ".join(m(SYM[k]) + " " + m(f"{lu[k] - base[k]:+.4f}") for k in ("cd_mean", "cl_amp", "St"))
            + ". Compare with the mesh change " + have[-2] + r" \(\to\) " + have[-1] + ": "
            + ", ".join(m(f"{runs[have[-1]]['sm'][k] - runs[have[-2]]['sm'][k]:+.4f}") for k in ("cd_mean", "cl_amp", "St")) + ".")
    comp_findings.append(
        "<b>Symmetry check.</b> The drag frequency is " + m(tf(fine["ratio_fD"], 4)) + " times the lift frequency (theory for a "
        "centred cylinder: exactly 2).")

    # ---- pre-flight -------------------------------------------------------------------------------
    pf_runs, pf_checks = [], []
    for pf in preflight:
        pf_runs.append(dict(name=f"{pf['name']} ({pf['cells']:,} cells, {pf['cores']} core{'s' if pf['cores'] > 1 else ''})",
                            setup=f"{pf['end_time_short']} s of simulated time from rest ({pf['n_steps']} steps) instead of "
                                  f"{pf['full_length']} s",
                            outcome=f"{ST[pf['status']]}; {pf['wall']:.1f} s; projected {pf['est_full'] / 60:.1f} min for the full run"))
        for c in pf["checks"]:
            pf_checks.append(dict(c, name=f"{c['name']} ({pf['name']})"))
        exact_q = Umean * H * numerics["mesh"]["dz"]
        exc = -pf["q_in"] / exact_q - 1
        pf_checks.append(dict(name=f"Inflow rate ({pf['name']})", status="ok" if abs(exc) < 1e-3 else "warn",
                              value=m(rf"{tn(exc, 2)}") + " relative to " + m(r"U_{\mathrm{mean}} H"),
                              meaning="The parabolic inlet must deliver the mean velocity of the benchmark; face-centre values "
                                      "give a small second-order excess " + m(r"\Delta y^2/(2H^2)") + "."))
        pf_checks.append(dict(name=f"Forces recorded every step ({pf['name']})",
                              status="ok" if pf["n_force_rows"] >= pf["n_steps"] else "fail",
                              value=f"{pf['n_force_rows']} rows for {pf['n_steps']} steps",
                              meaning="Drag and lift are needed at every step for the cycle analysis."))
        pf_checks.append(dict(name=f"Projected run time ({pf['name']})", status="ok" if pf["est_full"] < 1800 else "warn",
                              value=f"{pf['est_full'] / 60:.1f} min", meaning="Budget: 30 minutes per run."))

    # ---- performance appendix -----------------------------------------------------------------
    perf_rows = []
    allr = dict(runs, **piso)
    for nm in names_all + list(piso):
        p = allr[nm]["meta"]["perf"]
        perf_rows.append([nm, f"{p['cells']:,}", str(p["cores"]), f"{p['steps']:,}", f"{p['wall_s']:.0f}",
                          m(tf(p["us_per_cell_step"], 2)), m(tf(p["us_per_cell_step"] * p["cores"], 2)),
                          m(tf(p["cpu_efficiency"], 2)) if p["cpu_efficiency"] else "–",
                          m(tf(p["lin_iters"].get("p", float("nan")), 1)),
                          wshare(p["write_share"])])
    for s_ in scaling:
        perf_rows.append([f"scaling, {s_['cores']} rank{'s' if s_['cores'] > 1 else ''}", f"{s_['cells']:,}", str(s_["cores"]),
                          f"{s_['steps']:,}", f"{s_['wall_s']:.0f}", m(tf(s_["us_per_cell_step"], 2)),
                          m(tf(s_["us_per_cell_step"] * s_["cores"], 2)), m(tf(s_["cpu_efficiency"], 2)),
                          m(tf(s_["lin_iters"].get("p", float("nan")), 1)),
                          wshare(s_["write_share"])])
    ps = json.loads((PERF / "psolver_M3.json").read_text()) if (PERF / "psolver_M3.json").exists() else []
    for s_ in ps:
        v = s_["variant"]
        perf_rows.append([f"p solver {v['solver']}, " + m(rf"\mathrm{{tol}}\ {tn(v['tol'])}") + ", " + m(rf"\mathrm{{relTol}}\ {tn(v['relTol'])}"),
                          f"{s_['cells']:,}", "1", f"{s_['steps']:,}", f"{s_['wall_s']:.0f}", m(tf(s_["us_per_cell_step"], 2)),
                          m(tf(s_["us_per_cell_step"], 2)), m(tf(s_["cpu_efficiency"], 2)),
                          m(tf(s_["lin_iters"].get("p", float("nan")), 1)),
                          wshare(s_["write_share"])])
    perf_notes = performance_notes(scaling, ps, runs, have, numerics)

    # ---- assumptions ----------------------------------------------------------------------------
    assumptions = [
        dict(name="Two-dimensional flow", status="imposed",
             check="The benchmark is posed in 2D. Imposed with one cell and <code>empty</code> front/back. A real cylinder at "
                   + m("Re = 100") + " also sheds a 2D street, but end walls of a real channel would add 3D effects."),
        dict(name="Laminar, incompressible, Newtonian", status="standard",
             check=m(rf"Re = U_{{\mathrm{{mean}}}} D/\nu = {tn(dv['Re'], 4)}") + ": laminar; constant density and viscosity are "
                   "given. The fluid is not named, so compressibility is judged by the velocity: "
                   + m(rf"U_m = {tn(Um)}\,\mathrm{{m/s}}") + " is far below the speed of sound of any common fluid."),
        dict(name="Periodic state reached", status="ok" if per[fl] else "violated",
             check="Checked per run: over the last " + str(nmeas) + " cycles the lift peaks change by less than "
                   + m(tn(acc["periodic_cl_peak"])) + ", the drag peaks by less than " + m(tn(acc["periodic_cd_peak"]))
                   + " and the period by less than " + m(tn(acc["periodic_period_rel"])) + " (relative); finest run: "
                   + m(tn(max(sm["spread"]["cl_max"], sm["spread"]["cl_min"]), 2)) + ", "
                   + m(tn(max(sm["spread"]["cd_max"], sm["spread"]["cd_min"]), 2)) + ", "
                   + m(tn(sm["spread"]["period"] / sm["period"], 2)) + " ({fig:periodic})."),
        dict(name="Outlet far enough downstream", status="standard",
             check="The outlet is " + m(rf"{tn((g['L']['value'] - g['xc']['value']) / D, 3)}\,D") + " behind the cylinder, as in "
                   "the benchmark. The model's zero-gradient/zero-pressure outlet equals the benchmark's stress-free outflow "
                   "for parallel outflow; the time-averaged momentum balance closes to "
                   + m(tn(abs(b["rel"]) * 100, 2) + r"\,\%") + " of the drag."),
        dict(name="Cylinder surface as straight faces", status="imposed",
             check="The circle is represented by " + str(4 * spec[fl]["nc"]) + " straight faces on the finest mesh (points on "
                   "the exact circle). The area lost is a second-order error that shrinks with the mesh and is part of the mesh study."),
    ]

    ok_word = {"PASS": "passed", "WARN": "passed with caveats", "FAIL": "FAILED"}[verdict]
    results = dict(
        title=problem["title"], date=dt.date.today().isoformat(), verdict=verdict, comparison_type="benchmark",
        summary=dict(
            headline=(f"<b>Benchmark comparison {ok_word}.</b> On the finest mesh ({fl}, {fm['cells']:,} cells) the mean drag "
                      "is " + m(q("cd_mean")) + " (reference " + m(tf(ref["value"]["cd_mean"], 4)) + "), the lift amplitude "
                      + m(q("cl_amp")) + " (" + m(tf(ref["value"]["cl_amp"], 4)) + ") and the Strouhal number " + m(q("St"))
                      + " (" + m(tf(ref["value"]["St"], 4)) + ")."),
            answer_columns=["Requested", f"Reference ({ref_lab})", f"Simulation ({fl}, last cycle)", "Difference (÷ band)"],
            answers=answers, findings=findings,
            limitations=[
                "The reference is itself a numerical solution (finite elements, " + ref_lab + "), not an experiment; its own "
                "spread between levels is used as its uncertainty.",
                "Only four meshes and one time-step study (on " + numerics["timestep_study"]["mesh"] + ") fit the time budget; a fifth "
                "mesh (about " + f"{4 * fm['cells']:,}" + " cells, 8 times the cost of " + fl + ") and a time-step study on the finest mesh were left out.",
                "Each finer mesh ran only " + m(tn(fm["end_time"] - fm["start_time"], 3) + r"\,\mathrm{s}") + " (about "
                + f"{(fm['end_time'] - fm['start_time']) / sm['period']:.0f}" + " cycles) from the interpolated coarser flow, not "
                "25-30 s from rest as the reference; the answers use the last " + str(nmeas) + " cycles, and the periodicity "
                "check shows they were settled.",
                "Two-dimensional, laminar model, as the benchmark: no statement about a real three-dimensional channel.",
            ] + ([f"The finest run took {fm['wall_time'] / 60:.0f} min, over the 30-minute budget per run: the pre-flight "
                  f"projection ({preflight[-1]['est_full'] / 60:.0f} min) was made from a start from rest, where the pressure "
                  "solver needs fewer iterations than in the developed shedding flow (Insights). It was let run to the planned "
                  "length rather than cut short, so that the periodicity check and the measured cycles are not compromised."]
                 if fm["wall_time"] > 1800 else [])),
        problem=dict(
            description=problem["description"], setup=problem.get("setup", []),
            given=[dict(label=v["label"], symbol=v["symbol"], value=m(tn(v["value"])), unit=v["unit"]) for v in g.values()],
            wanted=wanted,
            reference="FeatFlow group (TU Dortmund), DFG benchmark 2D-2 (Schäfer and Turek 1996): drag, lift and period on "
                      "mesh levels 3-6 and three time steps, parsed from the benchmark page into <code>"
                      + problem["reference"]["file"] + "</code>."),
        assumptions=dict(rows=assumptions),
        theory=dict(
            intro="Vortex shedding behind a cylinder has <b>no exact solution</b>: the convection term makes the equations "
                  "nonlinear, and the steady solution is unstable at this Reynolds number. The comparison therefore uses the "
                  "FeatFlow benchmark. Theory still fixes the scales (one number, " + m("Re") + "), the inflow (an exact "
                  "channel-flow solution), what the shedding must look like (symmetry: drag at twice the lift frequency) and a "
                  "conservation law used as a check (the momentum balance of the channel). The algebra marked below is checked by SymPy.",
            steps=th["steps"], derived=th["derived"],
            self_checks=[dict(name=k, ok=bool(v)) for k, v in th["self_checks"].items()]),
        decisions=load_yaml(HERE / "decisions.yaml"),
        preflight=dict(
            purpose="Before the production runs, short runs on the coarsest and the finest mesh check that the setup runs, "
                    "stays stable, records every output and fits the time budget.",
            runs=pf_runs, checks=pf_checks,
            conclusion=("All pre-flight checks passed" if all(p["status"] != "fail" for p in preflight) else "A pre-flight check FAILED")
                       + f" after one fix (the inlet condition's keyword, Insights). The finest-mesh projection "
                       f"({preflight[-1]['est_full'] / 60:.0f} min for {preflight[-1]['full_length']:g} s on {preflight[-1]['cores']} "
                       f"cores) exceeded the 30-minute budget; the scaling test then showed {fm['cores']} ranks to be "
                       f"faster, and the {fm['end_time'] - fm['start_time']:g} s production run took {fm['wall_time'] / 60:.1f} min."),
        simulation=dict(
            model=[
                dict(aspect="Geometry", real="Channel " + m(r"2.2 \times 0.41\,\mathrm{m}") + ", cylinder " + m(r"D = 0.1\,\mathrm{m}")
                     + " at " + m(r"(0.2, 0.2)\,\mathrm{m}"),
                     model="The same, one cell thick in " + m("z") + " with <code>empty</code> front and back; the circle as straight "
                           "faces between points on it."),
                dict(aspect="Inflow", real="Parabola with maximum " + m(rf"U_m = {tn(Um)}\,\mathrm{{m/s}}"),
                     model="The parabola evaluated at each inlet face centre; inflow rate within "
                           + m(tn(abs(fine["inflow_excess"]), 2)) + " (relative) of " + m(r"U_{\mathrm{mean}} H") + " on " + fl + "."),
                dict(aspect="Outflow", real="Free (stress-free) outflow",
                     model="Zero normal velocity gradient, zero pressure."),
                dict(aspect="Fully developed periodic flow", real="The flow long after any start-up",
                     model="M1 from rest for " + m(tn(numerics["time"]["develop_end"]) + r"\,\mathrm{s}") + "; each finer mesh from "
                           "the developed coarser flow; answers from the last " + str(nmeas) + " cycles after a periodicity check."),
                dict(aspect="Forces", real="Pressure and friction on the cylinder per unit length",
                     model="OpenFOAM <code>forces</code> on the cylinder patch every step, divided by the slab thickness; "
                           "coefficients and cycle values computed in the analysis."),
            ],
            meshes=dict(
                intro=f"Four block-structured O-grid meshes; every cell count doubles from one to the next (ratio 2), and the time "
                      "step is halved with the cell size. " + m(r"h_w") + ": radial size of the cells on the cylinder; "
                      + m(r"\delta = D/\sqrt{Re}") + " is the boundary-layer scale.",
                head=["Mesh", "Cells", "Faces on cylinder", "Wall cell " + m("h_w"), m(r"\delta/h_w"), "Time step",
                      "Max Courant", "Steps per period"],
                rows=mesh_rows),
            results=dict(head=["Run"] + [m(SYM[k]) for k in QKEYS], rows=res_rows),
            health=dict(
                head=["Run", "Steps", "Max Co", "Log", m("c_L") + " peaks", m("c_D") + " peaks", "Period",
                      "Periodic", "Momentum", "Mass", "Wall (s)"],
                rows=health_rows,
                notes=[
                    "Log: automatic log checks (no crash, no NaN, Courant number up to " + m(tn(2.0)) + ", residual and "
                    "continuity-error trends; cfdlib.monitor).",
                    m("c_L") + " peaks, " + m("c_D") + " peaks, Period (relative): largest change between consecutive cycles "
                    "over the last " + str(nmeas) + " cycles; tolerances "
                    + m(tn(acc["periodic_cl_peak"])) + " (lift peaks), " + m(tn(acc["periodic_cd_peak"])) + " (drag peaks), "
                    + m(tn(acc["periodic_period_rel"])) + " (period), fixed before the runs.",
                    "Momentum: imbalance of the time average over the measured cycles of drag + wall friction minus the inlet/outlet "
                    "pressure and momentum fluxes and the change of stored momentum (Theory, step 5), divided by the mean drag "
                    "force; tolerance " + m(tn(100 * acc["momentum_balance_rel"]) + r"\,\%") + " on the finest mesh.",
                    "Mass: largest " + m(r"|Q_{\mathrm{in}} + Q_{\mathrm{out}}|/Q_{\mathrm{in}}") + " over all steps.",
                    "Runs named M2_dtx* are the time-step study, " + f"{numerics['sensitivity']['mesh']}_LU" + " the convection-scheme test.",
                ])),
        comparison=dict(
            kind="Benchmark: the simulation is compared with the FeatFlow finite-element solution of the same equations ("
                 + ref_lab + "). Agreement is judged against a band: the simulation's own uncertainty (GCI of the three finest "
                 "meshes plus the estimated time-step error) plus the reference's spread.",
            definitions=[
                dict(name="Cycle values", latex=r"\bar c = \tfrac12(c_{\max} + c_{\min}),\qquad \Delta c = c_{\max} - c_{\min}",
                     meaning="Over the last complete lift cycle, as the reference defines them."),
                dict(name="Strouhal number", latex=r"St = \frac{D}{U_{\mathrm{mean}}\,T}",
                     meaning="The period " + m("T") + " is the mean interval between upward zero crossings of the lift over "
                             "the last " + str(nmeas) + " cycles."),
                dict(name="Reference uncertainty", latex=r"U_{\mathrm{ref}} = \max_{\Delta t}\left|q_{6} - q_{5}(\Delta t)\right|",
                     meaning="Largest difference between the finest reference level and the three results one level coarser."),
                dict(name="Simulation uncertainty", latex=r"U_{\mathrm{sim}} = 1.25\,\frac{|q_4 - q_3|}{2^p - 1} + e_t",
                     meaning="GCI of the finest mesh (Celik et al. 2008) from M2, M3, M4 with their observed order " + m("p")
                             + ", plus the time-step error " + m("e_t") + " from the M2 study: Richardson in "
                             + m(r"\Delta t") + " scaled to the finest time step when the three M2 results change "
                             "monotonically, otherwise the largest change among them (no scaling)."),
                dict(name="Verdict per number", latex=r"|q_{\mathrm{sim}} - q_{\mathrm{ref}}| \le U_{\mathrm{sim}} + U_{\mathrm{ref}}",
                     meaning="Inside the band: ok; up to twice the band: warning; beyond: fail (warning only for the "
                             "secondary numbers " + m(SYM["cd_amp"]) + " and " + m(SYM["cl_mean"]) + ")."),
            ],
            head=["Number", "Reference", m("U_{\\mathrm{ref}}"), f"Simulation ({fl})", m("U_{\\mathrm{sim}}"), "Difference", "Relative",
                  "÷ band", "Status", "Richardson"],
            rows=comp_rows, findings=comp_findings),
        insights=load_yaml(HERE / "insights.yaml"),
        figures=figures,
        reproducibility=dict(
            commands=[
                dict(cmd=f"python -m cases.{HERE.name}.run preflight", what="Short test runs on M1 (1 core) and M4 (16 ranks)"),
                dict(cmd=f"python -m cases.{HERE.name}.run production M1 M2", what="M1 from rest (0 to 15 s), M2 from the developed M1 flow"),
                dict(cmd=f"python -m cases.{HERE.name}.run production M3 M2_dtx0.5 M2_dtx0.25 --jobs 3",
                     what="M3 (4 ranks) and the two time-step-study runs, at the same time"),
                dict(cmd=f"python -m cases.{HERE.name}.run production M4", what="Finest mesh, 16 MPI ranks (hardware threads)"),
                dict(cmd=f"python -m cases.{HERE.name}.run wake M4", what="Vorticity at the lift maximum for the picture"),
                dict(cmd=f"python -m cases.{HERE.name}.run production M3_LU", what="Convection-scheme test on M3"),
                dict(cmd=f"python -m cases.{HERE.name}.run scaling 16 8 4 2 1", what="MPI scaling test on M4 (200 steps each)"),
                dict(cmd=f"python -m cases.{HERE.name}.run psolver", what="Pressure linear-solver test on M3 (200 steps each; was run with nOuterCorrectors 1)"),
                dict(cmd=f"python -m cases.{HERE.name}.run exp X_M2_o2_dt1 M2 1 2 M2 -- X_M2_o1_dt025 M2 0.25 1 M2",
                     what="Time-integration experiments on M2 (outer correctors and time step), started from the M2 run"),
                dict(cmd=f"python -m cases.{HERE.name}.analyze --report", what="Analysis, figures and this report"),
            ],
            files=[
                dict(path=f"cases/{HERE.name}/problem.yaml", what="The problem as given"),
                dict(path=f"cases/{HERE.name}/reference/", what="FeatFlow tables (parsed) and a copy of the benchmark page"),
                dict(path=f"cases/{HERE.name}/numerics.yaml", what="Meshes, time steps, run lengths, acceptance rules"),
                dict(path=f"cases/{HERE.name}/decisions.yaml", what="Every choice, with the OpenFOAM settings it controls"),
                dict(path=f"cases/{HERE.name}/insights.yaml", what="Workflow insights logged during the work"),
                dict(path=f"cases/{HERE.name}/template/", what="OpenFOAM case; blockMeshDict filled from run.mesh_spec"),
                dict(path=f"cases/{HERE.name}/run.py", what="Mesh generator, runs, mapping, MPI, performance tests"),
                dict(path=f"cases/{HERE.name}/signals.py", what="Cycle analysis of the force signals"),
                dict(path=f"cases/{HERE.name}/theory.py", what="What theory can say (SymPy)"),
            ],
            run_dirs=[
                dict(path=f"{fm['run_root']}/<run>", what="One directory per run (M1-M4, M2_dtx*, M3_LU)"),
                dict(path=f"{fm['run_root']}/pf_<mesh>", what="Pre-flight runs"),
                dict(path=f"{fm['run_root']}/scale_M4_np<N>, psolver_*", what="Performance tests"),
                dict(path=f"results/{HERE.name}/", what="preflight.json, runs/, perf/, analysis.json, results.json, this report"),
            ],
            environment=report.environment_info(),
            notes=[
                "M3 (4 MPI ranks) and the two time-step-study runs (1 core each) ran at the same time; all other production runs "
                "ran alone. Per-run wall times of concurrent runs are inflated by shared memory bandwidth.",
                "The piso_* runs are the first plan (nOuterCorrectors 1, M1 time step 2 ms), made with the same commands "
                "before numerics.yaml was changed and then renamed; X_* runs come from the <code>exp</code> command.",
                "MPI runs: <code>decomposePar</code> (scotch), <code>mpirun -np N pimpleFoam -parallel</code>, <code>reconstructPar</code>.",
            ],
        ),
        performance=dict(
            intro="Run-time data for optimising future cases. Wall microseconds per cell per time step: lower is better; "
                  "multiplied by the core count it gives the cost in core-microseconds (constant for ideal scaling). "
                  "Pressure iterations are the mean GAMG/PCG iterations per solve (first corrector " + m("p") + ", final "
                  + m(r"p_{\mathrm{final}}") + "): this is the part a GPU linear solver would take over. Outside loop: share "
                  "of wall time spent in start-up, decomposition overhead and writing.",
            head=["Run", "Cells", "Cores", "Steps", "Wall (s)", m(r"\mu s") + "/cell/step", "core-" + m(r"\mu s") + "/cell/step",
                  "CPU eff.", m("p") + " iters", "Outside loop"],
            rows=perf_rows, notes=perf_notes),
        data=dict(analysis="analysis.json", verdict_rules=dict(fail=fail, warn=warn)),
    )
    out = report.render(results, svgs, OUT / "report.html", case_dir=fm["case_dir"])
    print("Report:", out)


def ts_names_of(numerics):
    return [_tsname(numerics, f) for f in numerics["timestep_study"]["factors"]]


def wshare(x):
    """Share of wall time outside the time loop, as report math (tiny or slightly negative values: clock rounding)."""
    if x is None:
        return "–"
    return m(r"< 0.1\,\%") if x < 1e-3 else m(tf(100 * x, 1) + r"\,\%")


def _tsname(numerics, f):
    return f"{numerics['timestep_study']['mesh']}_dtx{f:g}"


def performance_notes(scaling, ps, runs, have, numerics):
    notes = []
    if scaling:
        one = next((s for s in scaling if s["cores"] == 1), None)
        best = min(scaling, key=lambda s: s["loop_us_per_cell_step"])
        if one:
            sp = {s["cores"]: one["loop_us_per_cell_step"] / s["loop_us_per_cell_step"] for s in scaling}
            notes.append("<b>Scaling (finest mesh, inside the time loop).</b> Speed-up over one core: "
                         + ", ".join(f"{c} ranks " + m(tf(sp[c], 2)) for c in sorted(sp)) + ". Fastest: "
                         + f"{best['cores']} ranks. Parallel efficiency at 8 ranks "
                         + (m(tf(sp.get(8, float('nan')) / 8, 2)) if 8 in sp else "–") + ".")
        ref_F = scaling[0]["Fcyl_last"][0]
        dev = max(abs(s["Fcyl_last"][0] - ref_F) / abs(ref_F) for s in scaling)
        notes.append("<b>Same answer.</b> After 200 steps the drag force differs between decompositions by at most "
                     + m(tn(dev, 2)) + " (relative): round-off and solver-tolerance level, far below the mesh error.")
        it = {s["cores"]: s["lin_iters"].get("p") for s in scaling}
        notes.append("<b>What limits speed.</b> GAMG pressure iterations per solve change with the decomposition ("
                     + ", ".join(f"{c} ranks " + m(tf(v, 1)) for c, v in sorted(it.items()) if v) + "), and the cost per "
                     "core grows with the rank count: the 2D problem has few cells per rank (about "
                     + f"{scaling[0]['cells'] // 8:,}" + " at 8 ranks) and is limited by memory bandwidth and communication.")
    if ps:
        base = ps[0]
        Fb = np.array(base["Fcyl"])[:, 0]
        parts = []
        for s in ps[1:]:
            F = np.array(s["Fcyl"])[:, 0]
            parts.append(f"{s['variant']['solver']} tol " + m(tn(s["variant"]["tol"])) + " relTol " + m(tn(s["variant"]["relTol"]))
                         + ": " + m(tf(s["loop_us_per_cell_step"] / base["loop_us_per_cell_step"], 2)) + r" \(\times\) time, drag "
                         + m(tn(float(np.max(np.abs(F - Fb)) / np.max(np.abs(Fb))), 2)))
        notes.append("<b>Pressure solver (M3, 1 core, 200 steps, relative to GAMG tol " + m(tn(base["variant"]["tol"]))
                     + " relTol " + m(tn(base["variant"]["relTol"])) + ").</b> " + "; ".join(parts)
                     + ". (Run with one outer corrector, before the switch to two; the comparison between solvers carries over.)")
    old = PERF / "scaling_piso.json"
    if old.exists():
        sp_ = json.loads(old.read_text())
        one = next((s for s in sp_ if s["cores"] == 1), None)
        if one:
            notes.append("<b>Same test with one outer corrector</b> (the first plan): speed-up "
                         + ", ".join(f"{s['cores']} ranks " + m(tf(one['loop_us_per_cell_step'] / s['loop_us_per_cell_step'], 2))
                                     for s in sorted(sp_, key=lambda s: s["cores"])) + "; pressure iterations per solve "
                         + ", ".join(m(tf(s["lin_iters"].get("p", float("nan")), 1)) for s in sorted(sp_, key=lambda s: s["cores"]))
                         + ".")
    if runs:
        tot = sum(r["meta"]["wall_time"] for r in runs.values())
        fin = runs[have[-1]]["meta"]
        notes.append("<b>Where the time went.</b> All production runs together: " + m(tf(tot / 60, 1)) + r" \(\mathrm{min}\) of wall "
                     "time, of which the finest mesh alone " + m(tf(fin["wall_time"] / 60, 1)) + r" \(\mathrm{min}\) ("
                     + m(tf(100 * fin["wall_time"] / tot, 0)) + r"\(\,\%\)). Per time step it cost "
                     + m(tf(fin["perf"]["us_per_cell_step"], 2)) + r" wall \(\mu\mathrm{s}\) per cell with "
                     + str(fin["cores"]) + " ranks; every step has two outer correctors with two pressure solves each.")
    notes.append("<b>What to try next.</b> (1) Run the time-step and scheme tests at the same time as the finest mesh only if "
                 "they fit in the cores left over (16 hardware threads in total). (2) For a GPU linear solver, the number "
                 "to beat is the GAMG pressure solve on 16 CPU ranks; measure time per pressure solve and iterations on M3 "
                 "and M4, the host-device copy per solve, and check the drag after 200 steps against the CPU run (as in the "
                 "scaling test). A 2D case of " + f"{fin['cells']:,}" + " cells is small for a GPU; a 3D mesh of a few million "
                 "cells would show the benefit better. (3) Write fewer, binary fields on long runs (the wake picture needs "
                 "only one cycle of output).")
    return notes


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
COL = dict(M1=plots.SERIES[3], M2=plots.SERIES[2], M3=plots.SERIES[1], M4=plots.SERIES[0])


def _forces_fig(r, nmeas):
    fig, (ax, bx) = plots.wide(2, 3.4)
    t, cd, cl = r["t"], r["cd"], r["cl"]
    for a, y, lab in ((ax, cd, r"$c_D$"), (bx, cl, r"$c_L$")):
        a.plot(t, y, color=plots.SERIES[0], lw=0.9)
        a.axvspan(r["T0"], r["T1"], color=plots.GRID, alpha=0.7, lw=0)
        a.set_xlabel(r"$t$ (s)")
        a.set_ylabel(lab)
    lo, hi = np.percentile(cd[t > r["T0"]], [0, 100])
    pad = 0.8 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    return plots.to_svg(fig, "forces")


def _wake_fig(w, g):
    import matplotlib.tri as mtri
    D, xc, yc, H = (g[k]["value"] for k in ("D", "xc", "yc", "H"))
    x, y, wz = w["x"], w["y"], w["wz"]
    sel = x < 1.6
    tri = mtri.Triangulation(x[sel], y[sel])
    tx, ty = x[sel][tri.triangles].mean(axis=1), y[sel][tri.triangles].mean(axis=1)
    tri.set_mask(np.hypot(tx - xc, ty - yc) < 0.5 * D * 1.02)
    fig, ax = plots.wide(1, 2.9)
    ax.grid(False)
    lim = 30.0
    lev = np.linspace(-lim, lim, 25)
    cs = ax.tricontourf(tri, np.clip(wz[sel], -lim, lim), levels=lev, cmap="RdBu_r", extend="both")
    from matplotlib.patches import Circle
    ax.add_patch(Circle((xc, yc), D / 2, facecolor="#c3c2b7", edgecolor=plots.INK, lw=0.8))
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    cb = fig.colorbar(cs, ax=ax, shrink=0.9, ticks=[-30, -15, 0, 15, 30])
    cb.set_label(r"$\omega_z$ (1/s)")
    return plots.to_svg(fig, "wake")


def _periodic_fig(runs, have, tol):
    fig, ax = plots.figure(6.4, 3.4)
    for L in have:
        cyc = runs[L]["cycles"]
        v = np.array([c["cl_max"] for c in cyc])
        n = np.arange(1, len(v))
        ax.semilogy(n, np.maximum(np.abs(np.diff(v)), 1e-7), "o-", ms=3, lw=1, color=COL[L], label=L)
    ax.axhline(tol, color=plots.INK2, ls="--", lw=0.9)
    ax.set_xlabel("Cycle number in the run")
    ax.set_ylabel(r"$|c_{L,\max,i} - c_{L,\max,i-1}|$")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return plots.to_svg(fig, "periodic")


def _cycle_fig(runs, have, ref):
    fig, (ax, bx) = plots.wide(2, 3.4)
    for L in have:
        r = runs[L]
        c = r["cycles"][-1]
        s = (r["t"] >= c["t0"]) & (r["t"] <= c["t1"])
        ph = (r["t"][s] - c["t0"]) / c["period"]
        ax.plot(ph, r["cd"][s], color=COL[L], lw=1.3, label=L)
        bx.plot(ph, r["cl"][s], color=COL[L], lw=1.3, label=L)
    for a, keys in ((ax, ("cd_max", "cd_min")), (bx, ("cl_max", "cl_min"))):
        for k in keys:
            a.axhline(ref["value"][k], color=plots.INK2, ls="--", lw=0.9)
        a.set_xlabel(r"$(t - t_0)/T$")
    ax.set_ylabel(r"$c_D$")
    bx.set_ylabel(r"$c_L$")
    bx.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return plots.to_svg(fig, "cycle")


def _conv_fig(runs, have, ms, comp, ref):
    keys = ["cd_mean", "cl_amp", "St"]
    fig, axs = plots.wide(3, 3.3)
    hr = np.array([2.0 ** (len(have) - 1 - i) for i in range(len(have))])     # h / h_finest
    for a, k in zip(axs, keys):
        v = [runs[L]["sm"][k] for L in have]
        r0, u = ref["value"][k], ref["unc"][k]
        a.axhspan(r0 - u, r0 + u, color=plots.GRID, alpha=0.9, lw=0)
        a.axhline(r0, color=plots.INK2, lw=0.9, ls="--")
        a.plot(hr, v, "o-", color=plots.SERIES[0], ms=5, lw=1.2)
        if ms:
            a.errorbar([0.5], [ms[k]["extrap"]], yerr=[ms[k]["gci_abs"]], fmt="D", mfc="none", color=plots.SERIES[1], ms=6, capsize=3)
        a.set_xscale("log", base=2)
        a.set_xticks([0.5] + list(hr))
        a.set_xticklabels(["ext."] + [f"{x:g}" for x in hr])
        a.set_xlabel(r"$h/h_{\rm finest}$")
        a.set_title({"cd_mean": r"$\bar c_D$", "cl_amp": r"$\Delta c_L$", "St": r"$St$"}[k], fontsize=10)
    return plots.to_svg(fig, "conv")


def _timestep_fig(runs, numerics, piso, ref):
    """Lift amplitude and Strouhal number on M2 against the time step: PISO (1 outer corrector) vs PIMPLE (2)."""
    mesh = numerics["timestep_study"]["mesh"]
    names = sorted([n for n in runs if n == mesh or n.startswith(mesh + "_dtx")], key=lambda n: -runs[n]["meta"]["dt"])
    pn = sorted(piso, key=lambda n: -piso[n]["meta"]["dt"])
    fig, axs = plots.wide(2, 3.3)
    for a, k, lab in ((axs[0], "cl_amp", r"$\Delta c_L$"), (axs[1], "St", r"$St$")):
        a.plot([piso[n]["meta"]["dt"] * 1e3 for n in pn], [piso[n]["sm"][k] for n in pn], "s--", color=plots.SERIES[1],
               ms=5, lw=1.2, label="PISO (1 outer corrector)")
        a.plot([runs[n]["meta"]["dt"] * 1e3 for n in names], [runs[n]["sm"][k] for n in names], "o-", color=plots.SERIES[0],
               ms=5, lw=1.4, label="PIMPLE (2 outer correctors)")
        a.axhline(ref["value"][k], color=plots.INK2, lw=0.9, ls=":")
        a.set_xscale("log", base=2)
        ticks = sorted({round(piso[n]["meta"]["dt"] * 1e3, 4) for n in pn} | {round(runs[n]["meta"]["dt"] * 1e3, 4) for n in names})
        a.set_xticks(ticks)
        a.set_xticklabels([f"{x:g}" for x in ticks])
        a.set_xlabel(r"Time step $\Delta t$ (ms), mesh " + mesh)
        a.set_ylabel(lab)
    axs[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.25), ncol=2)
    return plots.to_svg(fig, "timestep")


def _scaling_fig(scaling):
    sc = sorted(scaling, key=lambda s: s["cores"])
    one = next(s for s in sc if s["cores"] == 1)
    c = np.array([s["cores"] for s in sc])
    us = np.array([s["loop_us_per_cell_step"] for s in sc])
    fig, (ax, bx) = plots.wide(2, 3.3)
    ax.plot(c, one["loop_us_per_cell_step"] / us, "o-", color=plots.SERIES[0], ms=5, lw=1.4)
    ax.plot(c, c, "--", color=plots.MUTED, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(c)
    ax.set_xticklabels([str(x) for x in c])
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Speed-up")
    bx.plot(c, us, "o-", color=plots.SERIES[1], ms=5, lw=1.4)
    bx.set_xscale("log", base=2)
    bx.set_xticks(c)
    bx.set_xticklabels([str(x) for x in c])
    bx.set_xlabel("MPI ranks")
    bx.set_ylabel(r"Wall $\mu$s per cell per step")
    return plots.to_svg(fig, "scaling")


if __name__ == "__main__":
    main()
