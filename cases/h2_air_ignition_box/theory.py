"""Theory: what can be known about the parcel without OpenFOAM.

  python -m cases.h2_air_ignition_box.theory        # computes everything, writes results/<case>/theory.json + theory.npz

No formula gives the temperature history. Theory gives the start state, two conservation laws, two bounds on
the final temperature, and the equations themselves, which are solved here a second time by an independent
program (mech.py with SciPy) as a cross-check of both OpenFOAM and the reference.
"""
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp
import yaml
from scipy.optimize import brentq

from cfdlib.report import m, tex_num
from . import measure
from . import mech as M

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RTOL = 1e-10                                   # the independent solution's relative tolerance (Radau)


def load():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    ref = yaml.safe_load((HERE / problem["reference"]["file"]).read_text(encoding="utf-8"))
    return problem, numerics, ref


def given(problem):
    return {k: v["value"] for k, v in problem["given"].items()}


def parcel(g, conv):
    chem, therm = M.scheme_files()
    mech = M.Mechanism(chem, therm, conv)
    return M.Parcel(mech, g["p"], g["T_0"], dict(H2=g["x_H2"], O2=g["x_O2"], N2=g["x_N2"]))


def _level_times(sol, levels, t_end):
    """Exact crossing times of the independent solution: bracket on a fine grid, then a root search."""
    tt = np.linspace(0.0, t_end, 20001)
    T = sol(tt)[-1]
    out = {}
    for L in levels:
        k = int(np.argmax(T >= L))
        out[str(int(L))] = float(brentq(lambda t: sol(t)[-1] - L, tt[k - 1], tt[k], xtol=1e-16, rtol=1e-14)) if T[k] >= L and k > 0 else None
    return out


def _solve(P, g, levels, rtol=RTOL, method="Radau"):
    s = P.solve(g["t_end"], T_ign=g["T_ign"], rtol=rtol, method=method)
    s["levels"] = _level_times(s["sol"], levels, g["t_end"])
    s["T_end"] = float(s["T"][-1])
    return s


def sympy_checks():
    """Symbolic self-checks of the equations (independent of any data)."""
    t = sp.symbols("t", positive=True)
    T = sp.Function("T")(t)
    v = sp.symbols("v", positive=True)
    n = 3
    sig = [sp.Function(f"sigma{i}")(t) for i in range(n)]
    w = sp.symbols(f"omega0:{n}")
    hf = [sp.Function(f"h{i}") for i in range(n)]
    H = sum(sig[i] * hf[i](T) for i in range(n))
    cp = [sp.diff(hf[i](T), T) for i in range(n)]
    dT = -v * sum(hf[i](T) * w[i] for i in range(n)) / sum(sig[i] * cp[i] for i in range(n))
    dH = sp.diff(H, t).subs(sp.Derivative(T, t), dT)
    for i in range(n):
        dH = dH.subs(sp.Derivative(sig[i], t), v * w[i])
    energy_ok = sp.simplify(dH) == 0
    # the data's polynomials: enthalpy is the integral of the heat capacity, entropy of heat capacity over T
    Ts = sp.symbols("T", positive=True)
    a = sp.symbols("a1:8")
    cp_R = sum(a[k] * Ts ** k for k in range(5))
    h_RT = sum(a[k] * Ts ** k / (k + 1) for k in range(5)) + a[5] / Ts
    s_R = a[0] * sp.log(Ts) + sum(a[k] * Ts ** k / k for k in range(1, 5)) + a[6]
    poly_ok = sp.simplify(sp.diff(h_RT * Ts, Ts) - cp_R) == 0 and sp.simplify(sp.diff(s_R, Ts) - cp_R / Ts) == 0
    return bool(energy_ok), bool(poly_ok)


def compute():
    problem, numerics, ref = load()
    g = given(problem)
    levels = numerics["extraction"]["levels"]
    of = M.openfoam_conventions()
    std = M.standard_conventions(of.weights)
    P = parcel(g, std)
    Pof = parcel(g, of)
    mech = P.mech
    t_end = g["t_end"]

    # ---- the independent solution, with its own checks ----
    base = _solve(P, g, levels)
    base_of = _solve(Pof, g, levels)
    tol_runs = [dict(method=mt, rtol=rt, **{k: _solve(P, g, levels, rt, mt)[k] for k in ("t_ign", "T_end", "steps")})
                for mt, rt in (("Radau", 1e-8), ("Radau", 1e-12), ("LSODA", 1e-12))]
    tol_runs.insert(1, dict(method="Radau", rtol=RTOL, t_ign=base["t_ign"], T_end=base["T_end"], steps=base["steps"]))
    spread_t = max(abs(r["t_ign"] / base["t_ign"] - 1) for r in tol_runs)
    spread_T = max(abs(r["T_end"] - base["T_end"]) for r in tol_runs)

    def one_change(**kw):                      # the standard conventions with one constant replaced by OpenFOAM's
        c = M.Conventions("variant", **{**dict(R=std.R, R_cal=std.R_cal, p_std=std.p_std, weights=std.weights), **kw})
        s = _solve(parcel(g, c), g, levels)
        return dict(t_ign=s["t_ign"], T_end=s["T_end"], dt_ign_rel=s["t_ign"] / base["t_ign"] - 1, dT_end=s["T_end"] - base["T_end"])
    conv_effect = dict(R_cal=one_change(R_cal=of.R_cal), p_std=one_change(p_std=of.p_std), R=one_change(R=of.R),
                       all=dict(t_ign=base_of["t_ign"], T_end=base_of["T_end"], dt_ign_rel=base_of["t_ign"] / base["t_ign"] - 1,
                                dT_end=base_of["T_end"] - base["T_end"]))

    # conservation in the independent solution, at the solver's own points
    q = {}
    for name, PP in (("std", P), ("of", Pof)):
        Tcc, scc = PP.complete_combustion()
        q[name] = dict(T_cc=Tcc, q=float(PP.h0 - scc @ PP.mech.h(g["T_0"])), res_cc=float(scc @ PP.mech.h(Tcc) - PP.h0))
    dh = max(abs(P.enthalpy(base["sigma"][k], base["T"][k]) - P.h0) for k in range(len(base["t"]))) / q["std"]["q"]
    at = base["sigma"] @ mech.comp
    present = P.atoms0 > 0
    datoms = float(np.max(np.abs(at[:, present] / P.atoms0[present] - 1)))

    # ---- where the mixture is heading ----
    eq = {}
    for name, PP, s in (("std", P, base), ("of", Pof, base_of)):
        Teq, seq = PP.equilibrium(s["sigma"][-1], s["T"][-1])
        qf, qr = PP.mech.progress(seq / PP.volume(seq, Teq), Teq)
        eq[name] = dict(T=Teq, Y={sn: float(seq[i] * PP.mech.W[i]) for i, sn in enumerate(PP.mech.species)},
                        net_over_forward=float(np.max(np.abs(qf - qr) / np.maximum(qf, 1e-300))))
    long = P.solve(1.0, rtol=RTOL)
    T_long = float(long["T"][-1])

    # ---- shape of the history: growth rate before ignition, fastest rise ----
    sol = base["sol"]
    t1, t2 = (brentq(lambda t: sol(t)[-1] - (g["T_0"] + d), 0.0, base["t_ign"]) for d in (1.0, 100.0))
    growth = math.log(100.0) / (t2 - t1)        # 1/s: T - T0 grows as exp(growth t) between +1 K and +100 K
    tw = np.linspace(0.9 * base["t_ign"], 1.3 * base["t_ign"], 40001)
    rate = np.array([P.rhs(tt, sol(tt))[-1] for tt in tw[::20]])
    k = int(np.argmax(rate))
    fine = np.linspace(tw[::20][max(k - 1, 0)], tw[::20][min(k + 1, len(rate) - 1)], 401)
    rate_f = np.array([P.rhs(tt, sol(tt))[-1] for tt in fine])
    rise = dict(rate=float(rate_f.max()), t=float(fine[int(np.argmax(rate_f))]))

    # ---- the reading rule's own error, measured on the independent solution ----
    bias = {}
    for N in numerics["runs"]["family_steps"]:
        tt = np.arange(N + 1) * (t_end / N)
        bias[str(N)] = measure.crossing_time(tt, sol(tt)[-1], g["T_ign"]) / base["t_ign"] - 1
    # ---- the reference against the independent solution: how well does the reference fix its own numbers? ----
    rt, rT = np.array(ref["history"]["t_s"]), np.array(ref["history"]["T_K"])
    ref_levels = measure.crossings(rt, rT, levels)
    dT_ref = sol(rt)[-1] - rT
    steep = (rT > g["T_0"] + 10) & (rT < 2100)
    ref_cmp = dict(
        t_ign_printed=ref["ignition"]["time_s"], t_ign_history=ref_levels[str(int(g["T_ign"]))], T_end=ref["final"]["T_K"],
        dt_ign_rel=base["t_ign"] / ref["ignition"]["time_s"] - 1, dT_end=base["T_end"] - ref["final"]["T_K"],
        levels=ref_levels, shift={L: (base["levels"][L] - ref_levels[L]) if ref_levels[L] and base["levels"][L] else None for L in ref_levels},
        dT_max_flat=float(np.max(np.abs(dT_ref[~steep]))), dT_max_steep=float(np.max(np.abs(dT_ref[steep]))),
        rho0_printed_g_cc=5.8967e-4,           # read below from the reference's source file when it is there
        fastest_rise=ref["fastest_rise"])
    senk = M.scheme_files()[0].parent / "senk.out"
    if senk.exists():
        import re
        mm = re.search(r"Density \(gm/cc\)\s*=\s*([0-9.Ee+-]+)", senk.read_text())
        ref_cmp["rho0_printed_g_cc"] = float(mm.group(1))

    # ---- the data: is each species' polynomial continuous where its two ranges meet? ----
    jumps = []
    for i, sn in enumerate(mech.species):
        Tm = mech.T_mid[i]
        lo, hi = mech.a_lo[i], mech.a_hi[i]
        cp_lo, cp_hi = (sum(a[k] * Tm ** k for k in range(5)) for a in (lo, hi))
        h_lo, h_hi = (sum(a[k] * Tm ** k / (k + 1) for k in range(5)) + a[5] / Tm for a in (lo, hi))
        jumps.append(dict(species=sn, cp=float(abs(cp_hi / cp_lo - 1)), h_K=float(abs(h_hi - h_lo) * Tm)))   # h jump as kelvin x R, per mole
    stoich_ok = bool(np.all(mech.dnu @ mech.comp == 0))

    # ---- histories for the figures and for the comparison: a uniform grid and a finer one through the ignition ----
    tg = np.unique(np.concatenate([np.linspace(0, t_end, 2001), np.arange(0.8 * base["t_ign"], 1.6 * base["t_ign"], 2e-8)]))
    hist = {}
    for name, s, PP in (("std", base, P), ("of", base_of, Pof)):
        y = s["sol"](tg)
        hist[f"T_{name}"] = y[-1]
        hist[f"Y_{name}"] = (y[:-1].T * PP.mech.W)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "theory.npz", t=tg, **hist)

    energy_ok, poly_ok = sympy_checks()
    out = dict(
        species=mech.species, elements=mech.elements, n_reactions=len(mech.reactions),
        conventions=dict(std=dict(R=std.R, R_cal=std.R_cal, p_std=std.p_std), of=dict(R=of.R, R_cal=of.R_cal, p_std=of.p_std,
                                                                                   reader_R_joule=of.reader_R_joule)),
        weights={e: std.weights[e] for e in mech.elements}, W={sn: float(mech.W[i]) for i, sn in enumerate(mech.species)},
        start=dict(X={sn: float(P.X0[i]) for i, sn in enumerate(mech.species)}, Y={sn: float(P.Y0[i]) for i, sn in enumerate(mech.species)},
                   W0=P.W0, rho0=P.rho0, rho0_of=Pof.rho0, h0=P.h0, h0_of=Pof.h0, atoms={e: float(P.atoms0[i]) for i, e in enumerate(mech.elements)},
                   phi=(g["x_H2"] / g["x_O2"]) / 2.0, cp0=float(P.sigma0 @ mech.cp(g["T_0"])),
                   hs0_of=float(Pof.h0 - Pof.sigma0 @ Pof.mech.h(298.15))),
        bounds=q, equilibrium=eq, T_long=T_long, t_long=1.0,
        independent=dict(rtol=RTOL, t_ign=base["t_ign"], T_end=base["T_end"], levels=base["levels"], steps=base["steps"], runs=tol_runs,
                         spread_t_ign_rel=spread_t, spread_T_end=spread_T, enthalpy_drift=float(dh), atoms_drift=datoms,
                         growth=growth, rise=rise,
                         Y_end={sn: float(base["sigma"][-1][i] * mech.W[i]) for i, sn in enumerate(mech.species)}),
        independent_of=dict(t_ign=base_of["t_ign"], T_end=base_of["T_end"], levels=base_of["levels"]),
        conv_effect=conv_effect, rule_bias=bias, reference=ref_cmp, data_jumps=jumps,
        checks=dict(energy_sympy=energy_ok, poly_sympy=poly_ok, stoich=stoich_ok),
    )
    (OUT / "theory.json").write_text(json.dumps(out, indent=1))
    return out


def numbers(recompute=False):
    """The theory numbers (from results/<case>/theory.json; computed when missing)."""
    p = OUT / "theory.json"
    if recompute or not p.exists() or not (OUT / "theory.npz").exists():
        return compute()
    return json.loads(p.read_text())


def histories():
    z = np.load(OUT / "theory.npz")
    return {k: z[k] for k in z.files}


# ---- the theory section of the report ------------------------------------------------------------------------
def report_theory(c, g):
    """The report's theory block from the numbers `c` and the given data `g`."""
    st, ind, eq, bd, ref = c["start"], c["independent"], c["equilibrium"]["std"], c["bounds"]["std"], c["reference"]
    W = c["W"]
    R, R_cal, p_std = (c["conventions"]["std"][k] for k in ("R", "R_cal", "p_std"))
    ntot = g["x_H2"] + g["x_O2"] + g["x_N2"]
    worst_cp = max(c["data_jumps"], key=lambda j: j["cp"])
    worst_h = max(c["data_jumps"], key=lambda j: j["h_K"])
    intro = (
        "No formula gives the temperature against time: the 27 reactions between 9 species are a set of coupled equations in which "
        "some species change a million times faster than others. They can only be solved step by step. Theory still fixes a good deal "
        "before any run: the start state; two things that cannot change, the number of atoms of each element and the enthalpy "
        "(the energy of a parcel at constant pressure); two limits for the final temperature; and the shape of the history "
        "(a slow start in which the temperature difference grows like an exponential, then a sudden rise). The equations themselves "
        "are written out below and solved here a second time by an independent program (<code>mech.py</code>: its own reading of the two data "
        "files, the rate laws of the CHEMKIN manual, SciPy's stiff integrators with a relative tolerance of " + m(tex_num(ind["rtol"], 1))
        + "). That gives a second answer next to the reference, with known tolerances, and a tool to measure what a difference in the "
        "constants of two programs is worth.")
    steps = [
        dict(label="The parcel", latex=r"\sigma_i = \frac{\text{moles of species } i}{\text{kg of gas}},\qquad v = \frac{R\,T}{p}\sum_i \sigma_i,\qquad c_i = \frac{\sigma_i}{v}",
             note="One kilogram of gas is followed. Its volume \\(v\\) follows from the ideal-gas law at the fixed pressure, so the gas "
                  "expands as it heats and as the number of molecules changes. \\(c_i\\) is the concentration that the reaction rates need."),
        dict(label="Each reaction", latex=r"q_j = k_{f,j}\prod_i c_i^{\nu'_{ij}} - k_{r,j}\prod_i c_i^{\nu''_{ij}},\qquad "
                                           r"k_f = A\,T^{\beta}\,e^{-E/(R T)},\qquad k_r = \frac{k_f}{K_c},\quad K_c = e^{-\Delta G^{\circ}/(R T)}\left(\frac{p^{\circ}}{R T}\right)^{\Delta\nu}",
             note="Rate of reaction \\(j\\): forward minus backward. \\(A\\), \\(\\beta\\), \\(E\\) come from <code>chem.inp</code>. The backward rate is "
                  "not given: it follows from the forward rate and the thermodynamic data (<code>therm.dat</code>), so that the reactions stop "
                  "exactly at chemical equilibrium. \\(p^{\\circ}\\) is the pressure for which the data are tabulated. Reactions that need a third "
                  "molecule to carry energy away multiply by the concentration of such molecules; four reactions of the scheme change "
                  "between a low-pressure and a high-pressure form (fall-off)."),
        dict(label="Species", latex=r"\frac{d\sigma_i}{dt} = v\,\omega_i,\qquad \omega_i = \sum_j (\nu''_{ij} - \nu'_{ij})\,q_j",
             note="Moles of each species per kilogram change by what the reactions make and use."),
        dict(label="Energy at constant pressure, no heat exchange", latex=r"\sum_i \sigma_i h_i(T) = h_0 = \text{const}\quad\Longrightarrow\quad "
                                                                          r"\frac{dT}{dt} = -\,\frac{v\sum_i h_i\,\omega_i}{\sum_i \sigma_i c_{p,i}}",
             note="\\(h_i\\) is the enthalpy of a mole of species \\(i\\), chemical energy included. The total cannot change, so when the reactions "
                  "turn chemical energy into heat the temperature rises. Checked with SymPy: with this \\(dT/dt\\) the time derivative of the sum is zero."),
        dict(label="Atoms", latex=r"\sum_i a_{ie}\,\sigma_i = \text{const for each element } e",
             note="\\(a_{ie}\\) is the number of atoms of element \\(e\\) in a molecule of species \\(i\\). Reactions only rearrange atoms."),
        dict(label="Where it ends: chemical equilibrium", latex=r"\min_{\sigma}\ \sum_i \sigma_i\,\mu_i(T, p, \sigma)\quad\text{with the atoms and } h_0 \text{ fixed}",
             note="Left alone for long enough, the mixture settles where its Gibbs energy is lowest. That state depends only on the "
                  "thermodynamic data, not on the reaction rates, and gives the temperature that the history must approach."),
    ]
    P = lambda tex, meaning, value, unit: dict(tex=tex, meaning=meaning, value=value, unit=unit)
    derived = [
        dict(label="Mean molar mass of the start mixture", symbol=r"W_0", formula=r"W_0 = \frac{\sum_i n_i W_i}{\sum_i n_i}",
             value_tex=tex_num(1e3 * st["W0"], 6), unit_tex=r"\mathrm{g/mol}",
             why="Turns the given mole ratio into mass fractions and gives the density.",
             params=[P(r"n_i", "moles of hydrogen, oxygen and nitrogen (given)", g["x_H2"], "1 : 1 : 3.76"),
                     P(r"W_{\mathrm{H_2}}", "molar mass of hydrogen", 1e3 * W["H2"], "g/mol"), P(r"W_{\mathrm{O_2}}", "molar mass of oxygen", 1e3 * W["O2"], "g/mol"),
                     P(r"W_{\mathrm{N_2}}", "molar mass of nitrogen", 1e3 * W["N2"], "g/mol")],
             check=rf"\frac{{{g['x_H2']:g}\times{1e3 * W['H2']:.5f} + {g['x_O2']:g}\times{1e3 * W['O2']:.4f} + {g['x_N2']:g}\times{1e3 * W['N2']:.4f}}}{{{ntot:g}}} = {1e3 * st['W0']:.5f}"),
        dict(label="Mass fraction of hydrogen at the start", symbol=r"Y_{\mathrm{H_2}}", formula=r"Y_i = \frac{n_i W_i}{\sum_k n_k W_k}",
             value_tex=tex_num(st["Y"]["H2"], 6), unit_tex="",
             why="The solver works with mass fractions. Oxygen: " + m(f"{st['Y']['O2']:.6f}") + ", nitrogen: " + m(f"{st['Y']['N2']:.6f}") + ". "
                 "The pre-flight checks that the solver starts from these.",
             params=[P(r"n_{\mathrm{H_2}} W_{\mathrm{H_2}}", "grams of hydrogen per mole of oxygen", g["x_H2"] * 1e3 * W["H2"], "g"),
                     P(r"\sum n_k W_k", "grams of mixture per mole of oxygen", ntot * 1e3 * st["W0"], "g")],
             check=rf"\frac{{{g['x_H2'] * 1e3 * W['H2']:.5f}}}{{{ntot * 1e3 * st['W0']:.4f}}} = {st['Y']['H2']:.6f}"),
        dict(label="Density at the start", symbol=r"\rho_0", formula=r"\rho_0 = \frac{p\,W_0}{R\,T_0}", value_tex=tex_num(st["rho0"], 6), unit_tex=r"\mathrm{kg/m^3}",
             why="A check of both programs: the reference prints " + m(tex_num(ref["rho0_printed_g_cc"] * 1e3, 5)) + r" \(\mathrm{kg/m^3}\) and "
                 "OpenFOAM must start from this value (pre-flight).",
             params=[P("p", "pressure (given)", g["p"], "Pa"), P("W_0", "mean molar mass", st["W0"], "kg/mol"),
                     P("R", "gas constant", R, "J/(mol K)"), P("T_0", "start temperature (given)", g["T_0"], "K")],
             check=rf"\frac{{{g['p']:.0f}\times{st['W0']:.7f}}}{{{R:.6f}\times{g['T_0']:.0f}}} = {st['rho0']:.6f}"),
        dict(label="Equivalence ratio", symbol=r"\phi", formula=r"\phi = \frac{n_{\mathrm{H_2}}/n_{\mathrm{O_2}}}{2}", value_tex=f"{st['phi']:g}", unit_tex="",
             why="Burning takes two molecules of hydrogen per molecule of oxygen. This mixture has half of that: it is lean, so the hydrogen "
                 "is used up and half of the oxygen is left over.",
             params=[P(r"n_{\mathrm{H_2}}/n_{\mathrm{O_2}}", "moles of hydrogen per mole of oxygen (given)", g["x_H2"] / g["x_O2"], "")],
             check=rf"\frac{{{g['x_H2'] / g['x_O2']:g}}}{{2}} = {st['phi']:g}"),
        dict(label="Heat released if all hydrogen burns to water", symbol=r"q", formula=r"q = h_0 - \sum_i \sigma_i^{\mathrm{burnt}}\,h_i(T_0)",
             value_tex=tex_num(bd["q"] / 1e6, 5), unit_tex=r"\mathrm{MJ/kg}",
             why="The scale of the energy in this problem. Every energy difference in this report is given as a fraction of it.",
             params=[P("h_0", "enthalpy of the start mixture per kg", st["h0"], "J/kg"),
                     P(r"\sigma_i^{\mathrm{burnt}}", "moles per kg after complete burning: water, the left-over oxygen, nitrogen", 0, "")],
             check=rf"{st['h0'] / 1e6:.5f} - ({(st['h0'] - bd['q']) / 1e6:.5f}) = {bd['q'] / 1e6:.5f}\ \mathrm{{MJ/kg}}"),
        dict(label="Upper limit of the temperature: complete burning", symbol=r"T_{\mathrm{cc}}", formula=r"\sum_i \sigma_i^{\mathrm{burnt}}\,h_i(T_{\mathrm{cc}}) = h_0",
             value_tex=f"{bd['T_cc']:.2f}", unit_tex=r"\mathrm{K}",
             why="If every hydrogen molecule ended as water and nothing else formed, all of \\(q\\) would heat the gas. Real hot gas keeps some "
                 "OH, O, H and unburnt hydrogen, which hold energy back, so the true final temperature is lower.",
             params=[P("h_0", "enthalpy of the start mixture per kg", st["h0"], "J/kg")],
             check=rf"\sum_i \sigma_i^{{\mathrm{{burnt}}}} h_i({bd['T_cc']:.2f}) - h_0 = {tex_num(bd['res_cc'], 1)}\ \mathrm{{J/kg}}"),
        dict(label="Where the temperature must end: chemical equilibrium", symbol=r"T_{\mathrm{eq}}", formula=r"\min G \text{ at fixed } h_0,\ p \text{ and atoms}",
             value_tex=f"{eq['T']:.2f}", unit_tex=r"\mathrm{K}",
             why="The history must approach this value from below and may not pass it. At the end of the millisecond the independent "
                 "solution is " + m(f"{eq['T'] - ind['T_end']:.2f}") + " K short of it: the slow reactions that join atoms back into "
                 "molecules are not finished. After one second the rate equations give " + m(f"{c['T_long']:.4f}") + " K.",
             params=[P(r"p^{\circ}", "pressure of the tabulated data", p_std, "Pa")],
             check=rf"\max_j \frac{{|q_{{f,j}} - q_{{r,j}}|}}{{q_{{f,j}}}} = {tex_num(eq['net_over_forward'], 1)}\ \text{{(all balanced)}}"),
        dict(label="Growth rate of the temperature rise before ignition", symbol=r"\lambda", formula=r"T - T_0 \propto e^{\lambda t}",
             value_tex=tex_num(ind["growth"], 3), unit_tex=r"\mathrm{s^{-1}}",
             why="In the slow start the reactive species multiply (each H atom that reacts makes more than one new carrier), so "
                 "everything grows like an exponential. \\(1/\\lambda = " + f"{1e6 / ind['growth']:.1f}" + "\\,\\mathrm{\\mu s}\\) is the time scale that a "
                 "time step must resolve. Read from the independent solution between 1 K and 100 K above the start.",
             params=[P(r"t_{+1}", "time at which the gas is 1 K warmer", ind["levels"]["1200"] - math.log(200.0) / ind["growth"], "s (about)")],
             check=rf"\lambda = \frac{{\ln 100}}{{t_{{+100\,\mathrm{{K}}}} - t_{{+1\,\mathrm{{K}}}}}} = {tex_num(ind['growth'], 3)}\ \mathrm{{s^{{-1}}}}"),
        dict(label="Ignition time of the independent solution", symbol=r"t_{\mathrm{ign}}", formula=r"T(t_{\mathrm{ign}}) = T_{\mathrm{ign}}",
             value_tex=tex_num(ind["t_ign"], 6), unit_tex=r"\mathrm{s}",
             why="A second answer next to the reference (" + m(tex_num(ref["t_ign_printed"], 5)) + " s). The two differ by "
                 + m(f"{100 * ind['dt_ign_rel'] if 'dt_ign_rel' in ind else 100 * ref['dt_ign_rel']:+.3f}\\,\\%") + ". The independent solution "
                 "changes by less than " + m(tex_num(max(ind["spread_t_ign_rel"], 1e-12), 1)) + " (relative) between two integrators and three tolerances.",
             params=[P(r"T_{\mathrm{ign}}", "temperature that defines ignition (given)", g["T_ign"], "K")],
             check=rf"T({tex_num(ind['t_ign'], 6)}\ \mathrm{{s}}) = {g['T_ign']:.0f}\ \mathrm{{K}}"),
        dict(label="Temperature of the independent solution at the end", symbol=r"T(t_{\mathrm{end}})", formula=r"T(t_{\mathrm{end}})",
             value_tex=f"{ind['T_end']:.3f}", unit_tex=r"\mathrm{K}",
             why="The reference prints " + m(f"{ref['T_end']:.1f}") + " K (five digits). Between the two limits above, as it must be: "
                 + m(rf"{ind['T_end']:.1f} < {eq['T']:.1f} < {bd['T_cc']:.1f}") + " K.",
             params=[P(r"t_{\mathrm{end}}", "duration of interest (given)", g["t_end"], "s")],
             check=rf"{ind['T_end']:.2f} < T_{{\mathrm{{eq}}}} = {eq['T']:.2f} < T_{{\mathrm{{cc}}}} = {bd['T_cc']:.2f}"),
    ]
    self_checks = [
        dict(name="SymPy: with the energy equation above, the enthalpy of the parcel does not change in time", ok=c["checks"]["energy_sympy"]),
        dict(name="SymPy: in the data's polynomials the enthalpy is the integral of the heat capacity, and the entropy of heat capacity over temperature", ok=c["checks"]["poly_sympy"]),
        dict(name=f"Each of the {c['n_reactions']} reactions, as read from the file, has the same atoms on both sides", ok=c["checks"]["stoich"]),
        dict(name="The independent solution keeps the atoms of every element to " + m(tex_num(max(ind["atoms_drift"], 1e-17), 1))
                  + " and the enthalpy to " + m(tex_num(max(ind["enthalpy_drift"], 1e-17), 1)) + " of the heat released",
             ok=ind["atoms_drift"] < 1e-9 and ind["enthalpy_drift"] < 1e-9),
        dict(name="Two integrators (Radau, LSODA) and three tolerances give the same ignition time to " + m(tex_num(max(ind["spread_t_ign_rel"], 1e-12), 1))
                  + " (relative) and the same final temperature to " + m(tex_num(max(ind["spread_T_end"], 1e-12), 1)) + " K",
             ok=ind["spread_t_ign_rel"] < 1e-6 and ind["spread_T_end"] < 1e-4),
        dict(name="The rate equations, run for one second, arrive at the equilibrium found by minimising the Gibbs energy (difference "
                  + m(tex_num(max(abs(c["T_long"] - eq["T"]), 1e-12), 1)) + " K): rates and thermodynamic data are consistent",
             ok=abs(c["T_long"] - eq["T"]) < 1e-4),
        dict(name="The final temperature lies between the history's end and the limit of complete burning", ok=ind["T_end"] < eq["T"] < bd["T_cc"]),
        dict(name="The start density agrees with the one the reference prints (" + m(tex_num(ref["rho0_printed_g_cc"] * 1e3, 5)) + r" \(\mathrm{kg/m^3}\), five digits)",
             ok=abs(st["rho0"] / (ref["rho0_printed_g_cc"] * 1e3) - 1) < 1e-4),
        dict(name="The data's two temperature ranges join at 1000 K, the start temperature: the largest jump of a heat capacity there is "
                  + m(tex_num(100 * worst_cp["cp"], 2) + r"\,\%") + f" ({worst_cp['species']}) and of an enthalpy " + m(tex_num(worst_h["h_K"], 2))
                  + f" K times the gas constant per mole ({worst_h['species']}); small enough not to matter here",
             ok=worst_cp["cp"] < 5e-3 and worst_h["h_K"] < 1.0),
    ]
    return dict(intro=intro, steps=steps, derived=derived, self_checks=self_checks)


if __name__ == "__main__":
    c = compute()
    ind, ref = c["independent"], c["reference"]
    print(f"start: W0 {1e3 * c['start']['W0']:.5f} g/mol, rho0 {c['start']['rho0']:.7f} (OpenFOAM constants {c['start']['rho0_of']:.7f}), Y {c['start']['Y']}")
    print(f"bounds: T_cc {c['bounds']['std']['T_cc']:.3f} K, q {c['bounds']['std']['q']:.5e} J/kg, T_eq {c['equilibrium']['std']['T']:.4f} K, T(1 s) {c['T_long']:.4f} K")
    print(f"independent: t_ign {ind['t_ign']:.9e} s, T_end {ind['T_end']:.5f} K, spread {ind['spread_t_ign_rel']:.1e} / {ind['spread_T_end']:.1e} K; "
          f"atoms {ind['atoms_drift']:.1e}, enthalpy {ind['enthalpy_drift']:.1e}; growth {ind['growth']:.4e} 1/s; fastest rise {ind['rise']}")
    for r in ind["runs"]:
        print("   ", r)
    print("conventions:", json.dumps(c["conv_effect"], indent=1))
    print("reference vs independent:", json.dumps({k: v for k, v in ref.items()}, indent=1))
    print("rule bias:", c["rule_bias"])
    print("data jumps:", [(j["species"], f"{j['cp']:.1e}", f"{j['h_K']:.2e}") for j in c["data_jumps"]])
    print("checks:", c["checks"])
