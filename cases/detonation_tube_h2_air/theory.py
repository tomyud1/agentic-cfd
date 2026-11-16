"""Theory for the detonation tube: what can be known without the simulation.

  python -m cases.detonation_tube_h2_air.theory        # compute, self-check and print (cached in results/.../theory.json)

Nothing here uses the flow solver. The gas is described by cfdlib.chem (the scheme's own thermodynamic data, read with
the constants OpenFOAM uses). What is computed:
  - the jump relations across a steady wave, derived from the three conservation laws with SymPy and checked;
  - the Chapman-Jouguet state (cfdlib.chem.cj_detonation), which must reproduce the reference file;
  - the state at the front before burning (a shock of the CJ speed with the composition unchanged);
  - the structure of the steady wave behind that shock with the scheme's own reaction rates (the ZND model): how
    long the gas waits before it burns (induction time and length). This sizes the cells;
  - the expansion wave that follows the detonation in a closed tube (the gas must come to rest at the closed end),
    along the path of constant entropy with the burnt gas in chemical equilibrium;
  - the start state, mass and energy in the tube.
"""
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp
import yaml
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from cfdlib import chem

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
T_STD = 298.15                      # K: OpenFOAM's sensible enthalpy is the enthalpy above the value at this temperature


def load():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8")) if (HERE / "numerics.yaml").exists() else {}
    ref = yaml.safe_load((HERE / problem["reference"]["file"]).read_text(encoding="utf-8"))
    return problem, numerics, ref


def given(problem):
    return {k: float(v["value"]) for k, v in problem["given"].items()}


def moles(g):
    return dict(H2=g["x_H2"], O2=g["x_O2"], N2=g["x_N2"])


_MECH = {}


def mech():
    """The scheme read with OpenFOAM's constants: what the solver's own fields must be checked with."""
    if "m" not in _MECH:
        _MECH["m"] = chem.build(chem.openfoam_conventions())
    return _MECH["m"]


def chemical_enthalpy(mc):
    """Enthalpy of each species at the standard temperature, J/kg: what OpenFOAM leaves out of its sensible enthalpy."""
    return mc.h(T_STD) / mc.W


# ---- the jump relations across a steady wave (SymPy) ----------------------------------------------------------------
def jump_relations():
    """Mass, momentum and energy across a wave of speed D into gas at rest, in the frame of the wave. Returns the
    solved relations and the self-checks (each residual must simplify to zero)."""
    rho1, D, p1, h1, v2 = sp.symbols("rho_1 D p_1 h_1 v_2", positive=True)
    w, p2, h2 = sp.symbols("w p_2 h_2")
    v1 = 1 / rho1
    mass = sp.Eq(rho1 * D, w / v2)
    mom = sp.Eq(p1 + rho1 * D ** 2, p2 + w ** 2 / v2)
    ene = sp.Eq(h1 + D ** 2 / 2, h2 + w ** 2 / 2)
    sol = sp.solve([mass, mom, ene], [w, p2, h2], dict=True)[0]
    rayleigh = p1 + rho1 ** 2 * D ** 2 * (v1 - v2)
    energy = h1 + D ** 2 / 2 * (1 - (v2 / v1) ** 2)
    hugoniot = sp.simplify(sol[h2] - h1 - sp.Rational(1, 2) * (sol[p2] - p1) * (v1 + v2))
    checks = [
        dict(name="The three conservation laws are satisfied by the solved relations", ok=all(sp.simplify(e.lhs - e.rhs).subs(sol).simplify() == 0 for e in (mass, mom, ene))),
        dict(name="Pressure behind the wave: the straight line in pressure and volume (Rayleigh line)", ok=sp.simplify(sol[p2] - rayleigh) == 0),
        dict(name="Enthalpy behind the wave", ok=sp.simplify(sol[h2] - energy) == 0),
        dict(name="The speed drops out: enthalpy rise = half the pressure rise times the sum of the volumes (Hugoniot)", ok=hugoniot == 0),
    ]
    return dict(w=sp.latex(sol[w]), p2=sp.latex(sp.simplify(sol[p2])), h2=sp.latex(sp.simplify(sol[h2])), checks=checks)


# ---- the steady wave behind the shock, with the scheme's reaction rates (ZND) --------------------------------------
def znd(mc, p1, T1, mol, D, t_end=2.0e-5, n_out=1200):
    """A parcel that has just crossed a shock of speed D, followed in time. In the frame of the wave the flow is
    steady, so mass, momentum and energy tie pressure, volume and temperature to the composition:
        p = p1 + m^2 (v1 - v),   h(sigma, T) + (m v)^2 / 2 = h1 + D^2 / 2,   p v = n R T      (m = D / v1)
    Differentiated, these give two linear equations for dT/dt and dv/dt from the reaction rates. The distance
    behind the shock grows with the parcel's speed relative to the wave, m v. Returns arrays over time."""
    fresh = chem.Parcel(mc, p1, T1, mol)
    v1, R = 1.0 / fresh.rho0, mc.conv.R
    mflux = D / v1
    sh = chem.shock_frozen(mc, p1, T1, mol, D)
    y0 = np.concatenate([fresh.sigma0, [sh["T"], 1.0 / sh["rho"], 0.0]])

    def rhs(t, y):
        sigma, T, v = np.maximum(y[:-3], 0.0), y[-3], y[-2]
        ds = v * mc.omega(sigma / v, T)
        n, cp = sigma.sum(), float(sigma @ mc.cp(T))
        # energy:  cp dT + m^2 v dv = -sum h_i dsigma_i ;  state on the Rayleigh line:  n R dT - (p1 + m^2 v1 - 2 m^2 v) dv = -R T dn
        A = np.array([[cp, mflux ** 2 * v], [n * R, -(p1 + mflux ** 2 * v1 - 2 * mflux ** 2 * v)]])
        b = np.array([-float(mc.h(T) @ ds), -R * T * ds.sum()])
        dT, dv = np.linalg.solve(A, b)
        return np.concatenate([ds, [dT, dv, mflux * v]])

    t_eval = np.concatenate([[0.0], np.geomspace(1e-10, t_end, n_out)])
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="Radau", rtol=1e-9, atol=1e-16, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)
    sigma, T, v, x = sol.y[:-3].T, sol.y[-3], sol.y[-2], sol.y[-1]
    p = p1 + mflux ** 2 * (v1 - v)
    return dict(t=sol.t, x=x, T=T, p=p, rho=1.0 / v, u=D - mflux * v, Y=sigma * mc.W, shock=sh)


def znd_lengths(z, T_cj, Y_h2o_cj, iw):
    """Induction time and length: where the temperature rises fastest. Also where the water has reached half and
    95 % of its final value, and how far the temperature still is from the CJ value at fixed distances."""
    rate = np.gradient(z["T"], z["t"])
    k = int(np.argmax(rate))
    w = z["Y"][:, iw]

    def where(frac):
        j = int(np.argmax(w >= frac * Y_h2o_cj))
        return float(np.interp(frac * Y_h2o_cj, w[j - 1:j + 1], z["x"][j - 1:j + 1]))
    return dict(t_ind=float(z["t"][k]), x_ind=float(z["x"][k]), T_ind=float(z["T"][k]), x_half_water=where(0.5), x_95_water=where(0.95),
                T_at_1mm=float(np.interp(1e-3, z["x"], z["T"])), p_at_1mm=float(np.interp(1e-3, z["x"], z["p"])),
                T_end=float(z["T"][-1]), p_end=float(z["p"][-1]), x_end=float(z["x"][-1]))


# ---- the expansion behind the wave in a closed tube (Taylor wave) --------------------------------------------------
def taylor_wave(mc, p1, T1, mol, cj, n=240, p_low=0.2):
    """From the CJ state down a path of constant entropy with the gas in chemical equilibrium: dh = v dp. Along the
    wave u - integral of dp / (rho c) is constant (c: speed of sound of the equilibrium gas, from the path itself),
    and each state travels at x / t = u + c. The gas is at rest where u = 0: the state at the closed end.
    Returns arrays (p, T, rho, u, c, xi = x/t) from the front to the end state, and the end state."""
    prod = chem.Parcel(mc, p1, T1, mol)
    ps = cj["p"] * np.geomspace(1.0, p_low, n)
    sigma, T = np.array(cj["sigma"]), cj["T"]
    h = float(sigma @ mc.h(T))
    v = 1.0 / cj["rho"]
    hs, vs, Ts, Ss = [h], [v], [T], [sigma]
    for j in range(1, n):
        dp = ps[j] - ps[j - 1]
        v_new = v
        for _ in range(4):                                       # trapezoid rule for dh = v dp, the new volume found by repeating
            prod.p, prod.h0 = ps[j], h + 0.5 * (v + v_new) * dp
            T_new, s_new = prod.equilibrium(sigma, T)
            v_new = prod.volume(s_new, T_new)
        h, v, T, sigma = prod.h0, v_new, T_new, s_new
        hs.append(h), vs.append(v), Ts.append(T), Ss.append(sigma)
    rho = 1.0 / np.array(vs)
    c = np.sqrt(np.gradient(ps, rho))                            # equilibrium speed of sound along the path
    integrand = 1.0 / (rho * c)
    du = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(ps))])    # negative: p falls
    u = cj["u"] + du
    k = int(np.argmax(u <= 0.0))
    assert k > 0, "taylor_wave: the gas does not come to rest in the range of pressures; lower p_low"
    f = u[k - 1] / (u[k - 1] - u[k])
    end = {q: float(a[k - 1] + f * (a[k] - a[k - 1])) for q, a in dict(p=ps, T=np.array(Ts), rho=rho, c=c).items()}
    sl = slice(0, k)
    arr = dict(p=np.append(ps[sl], end["p"]), T=np.append(np.array(Ts)[sl], end["T"]), rho=np.append(rho[sl], end["rho"]),
               u=np.append(u[sl], 0.0), c=np.append(c[sl], end["c"]))
    arr["xi"] = arr["u"] + arr["c"]
    return arr, dict(end, c_front=float(c[0]), gamma_e=float(rho[0] * c[0] ** 2 / ps[0]))


def taylor_profile(tw, D, x, t):
    """Pressure and temperature at positions x and time t of the ideal picture: a front of no thickness at D t,
    the expansion wave behind it, gas at rest from the closed end to c_end t. (No ignition layer, no reaction zone.)"""
    xi = np.asarray(x) / t
    order = np.argsort(tw["xi"])
    p = np.interp(xi, tw["xi"][order], tw["p"][order])
    T = np.interp(xi, tw["xi"][order], tw["T"][order])
    return np.where(xi <= D, p, np.nan), np.where(xi <= D, T, np.nan)


# ---- all numbers --------------------------------------------------------------------------------------------------
def numbers(force=False):
    """Everything above for this problem, as plain numbers (cached). Arrays go to theory.npz."""
    cache, arrays = OUT / "theory.json", OUT / "theory.npz"
    if cache.exists() and arrays.exists() and not force:
        return json.loads(cache.read_text())
    problem, numerics, ref = load()
    g = given(problem)
    mc, mol = mech(), moles(g)
    fresh = chem.Parcel(mc, g["p_0"], g["T_0"], mol)
    hc = chemical_enthalpy(mc)
    Y0 = {s: float(y) for s, y in zip(mc.species, fresh.Y0)}

    def total_energy(p, T):                                      # J/m^3 of the start mixture at (p, T): rho (enthalpy) - p
        par = chem.Parcel(mc, p, T, mol)
        return par.rho0 * par.h0 - p, par.rho0
    e0, rho0 = total_energy(g["p_0"], g["T_0"])
    e_ign, rho_ign = total_energy(g["p_ign"], g["T_ign"])
    start = dict(Y=Y0, rho0=rho0, rho_ign=rho_ign, W0=fresh.W0, c0=chem.sound_speed_frozen(mc, g["T_0"], fresh.sigma0),
                 mass_per_area=rho0 * (g["L"] - g["L_ign"]) + rho_ign * g["L_ign"],
                 energy_per_area=e0 * (g["L"] - g["L_ign"]) + e_ign * g["L_ign"],
                 layer_surplus=(e_ign - e0 * rho_ign / rho0) * g["L_ign"],     # J/m^2 above the same mass of fresh gas
                 atoms_per_kg={e: float(a) for e, a in zip(mc.elements, fresh.atoms0)})

    cj = chem.cj_detonation(mc, g["p_0"], g["T_0"], mol)
    r = ref["cj"]
    X_cj = cj["sigma"] / cj["sigma"].sum()
    Y_cj = {s: float(y) for s, y in zip(mc.species, cj["sigma"] * mc.W)}
    q_cj = float(sum(hc[i] * (fresh.Y0[i] - cj["sigma"][i] * mc.W[i]) for i in range(len(mc.species))))     # J/kg released up to the CJ state
    cjn = dict(D=cj["D"], p=cj["p"], T=cj["T"], rho=cj["rho"], u=cj["u"], w=cj["w"], mach=cj["mach"], Y=Y_cj, q=q_cj,
               X={s: float(x) for s, x in zip(mc.species, X_cj)},
               vs_reference=dict(D=cj["D"] / r["speed_m_s"] - 1, p=cj["p"] / r["pressure_Pa"] - 1, T=cj["T"] / r["temperature_K"] - 1))
    vn = chem.shock_frozen(mc, g["p_0"], g["T_0"], mol, r["speed_m_s"])
    rv = r["front_before_burning"]
    vnn = dict(p=vn["p"], T=vn["T"], rho=vn["rho"], u=vn["u"], c=chem.sound_speed_frozen(mc, vn["T"], fresh.sigma0),
               vs_reference=dict(p=vn["p"] / rv["pressure_Pa"] - 1, T=vn["T"] / rv["temperature_K"] - 1))

    z = znd(mc, g["p_0"], g["T_0"], mol, r["speed_m_s"])
    iw = mc.idx["H2O"]
    zl = znd_lengths(z, cj["T"], Y_cj["H2O"], iw)
    tw, tend = taylor_wave(mc, g["p_0"], g["T_0"], mol, cj)

    jr = jump_relations()
    # numbers into the jump relations: the reference's CJ state must satisfy them with this module's enthalpies
    v1, v2 = 1 / rho0, 1 / r["density_kg_m3"]
    D = r["speed_m_s"]
    h2 = float(cj["sigma"] @ mc.h(r["temperature_K"]))
    res = dict(mass=rho0 * D * v2 / r["gas_speed_relative_to_wave_m_s"] - 1,
               momentum=(g["p_0"] + rho0 ** 2 * D ** 2 * (v1 - v2)) / r["pressure_Pa"] - 1,
               energy=(fresh.h0 + D ** 2 / 2 * (1 - (v2 / v1) ** 2) - h2) / (D ** 2 / 2))
    checks = jr["checks"] + [
        dict(name="The CJ state of the reference file satisfies the three relations with this module's enthalpies (largest relative residual "
                  f"{max(abs(x) for x in res.values()):.1e})", ok=max(abs(x) for x in res.values()) < 1e-5),
        dict(name=f"cfdlib.chem.cj_detonation reproduces the reference file (speed {cjn['vs_reference']['D']:+.1e}, pressure "
                  f"{cjn['vs_reference']['p']:+.1e}, temperature {cjn['vs_reference']['T']:+.1e}, relative)",
             ok=max(abs(x) for x in cjn["vs_reference"].values()) < 1e-5),
        dict(name=f"The state at the front before burning reproduces the reference file (pressure {vnn['vs_reference']['p']:+.1e}, "
                  f"temperature {vnn['vs_reference']['T']:+.1e})", ok=max(abs(x) for x in vnn["vs_reference"].values()) < 1e-5),
        dict(name=f"The steady wave ends at the CJ state: after {zl['x_end'] * 1e3:.1f} mm its pressure is {zl['p_end'] / cj['p'] - 1:+.2%} and its "
                  f"temperature {zl['T_end'] / cj['T'] - 1:+.2%} from the CJ values", ok=abs(zl["p_end"] / cj["p"] - 1) < 0.01 and abs(zl["T_end"] / cj["T"] - 1) < 0.01),
        dict(name=f"The speed of sound found along the expansion path equals the speed at which the burnt gas leaves the wave ({tend['c_front']:.1f} against "
                  f"{cj['w']:.1f} m/s)", ok=abs(tend["c_front"] / cj["w"] - 1) < 5e-3),
        dict(name="Along the expansion the entropy path is consistent: pressure, temperature and density fall together and the gas comes to rest "
                  f"at {tend['p'] / cj['p']:.3f} of the CJ pressure", ok=bool(np.all(np.diff(tw["T"]) < 0) and np.all(np.diff(tw["rho"]) < 0))),
    ]
    out = dict(given=g, start=start, cj=cjn, front=vnn, znd=zl, taylor=tend, jump=dict(w=jr["w"], p2=jr["p2"], h2=jr["h2"]),
               checks=[dict(name=c["name"], ok=bool(c["ok"])) for c in checks],
               x_front_cj=D * g["t_end"], hc={s: float(x) for s, x in zip(mc.species, hc)},
               q_per_area_per_m=rho0 * q_cj)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays, znd_t=z["t"], znd_x=z["x"], znd_T=z["T"], znd_p=z["p"], znd_rho=z["rho"], znd_u=z["u"], znd_Y=z["Y"],
                        tw_p=tw["p"], tw_T=tw["T"], tw_rho=tw["rho"], tw_u=tw["u"], tw_c=tw["c"], tw_xi=tw["xi"])
    cache.write_text(json.dumps(out, indent=1, default=float))
    return out


def arrays():
    numbers()
    a = np.load(OUT / "theory.npz")
    return {k: a[k] for k in a.files}


# ---- the report's theory block ------------------------------------------------------------------------------------
def report_theory(th, g, numerics):
    """Theory section of the report from the cached numbers `th` and the given data `g`."""
    from cfdlib.report import m, tex_num
    st, cj, fr, zn, tw, r = th["start"], th["cj"], th["front"], th["znd"], th["taylor"], th["cj"]["vs_reference"]
    R = mech().conv.R
    fam = [int(n) for n, _ in numerics["runs"]["family"]]
    intro = (
        "Theory gives much of the answer here, because a detonation that has settled travels as a steady wave. Three conservation laws "
        "across the wave, the thermodynamic data of the gas and one extra condition fix its speed and the state behind it. This is the "
        "Chapman-Jouguet (CJ) state of the reference file, and it does not depend on how fast the reactions are. The same laws give the "
        "state right behind the leading shock before anything has burnt, and, with the reaction rates of the scheme, how far the gas "
        "travels before it burns (the steady structure, known as the ZND model). In a closed tube the burnt gas must come to rest at the "
        "closed end, which fixes the expansion wave that follows the front and the pressure at the end. All of this is computed here with "
        "<code>cfdlib.chem</code> from the scheme's own two files and without the flow solver. What theory does not give: how the wave "
        "forms out of the ignition layer and how long it takes to settle, and whether the steady wave is stable. Those are what the "
        "simulation is for.")
    steps = [
        dict(label="Conservation across a steady wave", latex=r"\rho_1 D = \rho_2 w,\qquad p_1 + \rho_1 D^2 = p_2 + \rho_2 w^2,\qquad h_1 + \tfrac{1}{2}D^2 = h_2 + \tfrac{1}{2}w^2",
             note="Seen by someone travelling with the wave, fresh gas (1) flows in at the wave speed \\(D\\) and burnt gas (2) leaves at the "
                  "speed \\(w\\). Mass, momentum and energy that flow in must flow out. The enthalpy \\(h\\) includes the chemical energy, so the "
                  "heat of burning is inside \\(h_1 - h_2\\)."),
        dict(label="Solved for the state behind the wave", latex=r"p_2 = p_1 + \rho_1^2 D^2\,(v_1 - v_2),\qquad h_2 - h_1 = \tfrac{1}{2}\,(p_2 - p_1)\,(v_1 + v_2),\qquad v = 1/\rho",
             note="The first relation is a straight line in pressure against volume whose slope is set by the wave speed. The second no longer "
                  "contains the speed: it is the curve of all states that burnt gas can reach. Both were derived from the three laws with SymPy."),
        dict(label="The extra condition (Chapman-Jouguet)", latex=r"w = c_2\qquad\Longleftrightarrow\qquad D_{\mathrm{CJ}} = \min_{v_2} D",
             note="A wave that nothing pushes from behind settles at the lowest speed for which a burnt state exists. At that speed the burnt "
                  "gas leaves the wave at exactly its own speed of sound \\(c_2\\), so nothing that happens behind the wave can catch up with it. "
                  "The burnt gas is taken in chemical equilibrium (the composition of lowest Gibbs energy at its pressure and enthalpy)."),
        dict(label="Right behind the shock, before burning", latex=r"\text{the same three laws with the composition of the fresh gas}",
             note="The leading shock compresses and heats the gas faster than any reaction. This state has the highest pressure of the "
                  "steady wave, well above the CJ pressure; the pressure then falls to the CJ value as the gas burns and speeds up."),
        dict(label="The steady structure with the reaction rates", latex=r"\frac{d\sigma_i}{dt} = v\,\omega_i(\sigma, T, v),\qquad p = p_1 + \rho_1^2 D^2 (v_1 - v),\qquad "
                                                                          r"h(\sigma, T) + \tfrac{1}{2}(\rho_1 D v)^2 = h_1 + \tfrac{1}{2}D^2,\qquad \frac{dx}{dt} = \rho_1 D\,v",
             note="A parcel is followed from the shock on. \\(\\sigma_i\\) are its moles of each species per kilogram and \\(\\omega_i\\) the rates "
                  "of the 27 reactions. Pressure and temperature follow from the two conservation relations at every instant, and \\(x\\) is the "
                  "distance behind the shock. For a while almost nothing happens (the reactive species multiply unseen), then the gas burns quickly."),
        dict(label="The expansion behind the wave in a closed tube", latex=r"u - \int \frac{dp}{\rho\,c} = \text{const},\qquad \frac{x}{t} = u + c,\qquad u = 0 \text{ at the closed end}",
             note="The burnt gas leaves the front moving forward at \\(u_{\\mathrm{CJ}}\\), but the end wall is at rest. An expansion wave "
                  "slows it down. In this wave each state travels at its own fixed speed \\(u + c\\), so the profile stretches in proportion to time, "
                  "and the gas is at rest between the wall and the point \\(x = c_{\\mathrm{end}}\\,t\\). Computed along the path of constant entropy "
                  "with the gas in chemical equilibrium."),
        dict(label="What the closed tube keeps", latex=r"\frac{d}{dt}\int \rho\,dV = 0,\qquad \frac{d}{dt}\int \rho\,Y_i\,\frac{a_{ie}}{W_i}\,dV = 0,\qquad "
                                                         r"\frac{d}{dt}\int \left(\rho\,h + \tfrac{1}{2}\rho\,u^2 - p\right) dV = 0",
             note="Mass, the atoms of each element (\\(a_{ie}\\) atoms of element \\(e\\) in species \\(i\\)) and total energy: nothing enters or "
                  "leaves, no wall moves and no heat is exchanged. The solver is built to keep the first two. It is not built to keep the third, "
                  "so the energy balance is the physical check of every run."),
    ]
    P = lambda tex, meaning, value, unit: dict(tex=tex, meaning=meaning, value=value, unit=unit)
    v1, v2 = 1 / st["rho0"], 1 / cj["rho"]
    mflux = st["rho0"] * cj["D"]                                 # mass flow through the wave, kg/(m^2 s)
    heat_per_mm = st["rho0"] * cj["q"] * 1e-3
    derived = [
        dict(label="Density of the fresh mixture", symbol=r"\rho_1", formula=r"\rho_1 = \frac{p_0\,W_0}{R\,T_0}", value_tex=tex_num(st["rho0"], 6), unit_tex=r"\mathrm{kg/m^3}",
             why="The mass that the wave takes in per metre. Mass fractions of the start mixture: hydrogen " + m(f"{st['Y']['H2']:.6f}") + ", oxygen "
                 + m(f"{st['Y']['O2']:.6f}") + ", nitrogen " + m(f"{st['Y']['N2']:.6f}") + ". The pre-flight checks that the solver starts from these.",
             params=[P("p_0", "start pressure (given)", g["p_0"], "Pa"), P("W_0", "mean molar mass of the mixture", st["W0"], "kg/mol"),
                     P("R", "gas constant (OpenFOAM's value)", R, "J/(mol K)"), P("T_0", "start temperature (given)", g["T_0"], "K")],
             check=rf"\frac{{{g['p_0']:.0f}\times{st['W0']:.7f}}}{{{R:.6f}\times{g['T_0']:.0f}}} = {st['rho0']:.6f}"),
        dict(label="Chapman-Jouguet speed", symbol=r"D_{\mathrm{CJ}}", formula=r"w = c_2 \text{ on the curve of burnt states}", value_tex=f"{cj['D']:.2f}", unit_tex=r"\mathrm{m/s}",
             why="The speed the wave must settle to, and the requirement of the problem (within 2 %). Mach number " + m(f"{cj['mach']:.3f}") + " against the fresh gas. "
                 "This value was computed here and agrees with the reference file to " + m(tex_num(abs(r["D"]), 1)) + " (relative).",
             params=[P(r"\rho_1", "density of the fresh gas", st["rho0"], "kg/m^3"), P("p_1", "start pressure", g["p_0"], "Pa"),
                     P("w", "speed at which the burnt gas leaves the wave", cj["w"], "m/s"), P("c_2", "speed of sound of the burnt gas (equilibrium)", tw["c_front"], "m/s")],
             check=rf"{st['rho0']:.4f}\times{cj['D']:.1f} = {st['rho0'] * cj['D']:.1f} = {cj['rho']:.4f}\times{cj['w']:.1f}"),
        dict(label="Pressure and temperature behind the wave (CJ)", symbol=r"p_{\mathrm{CJ}}", formula=r"p_2 = p_1 + \rho_1^2 D^2\,(v_1 - v_2)", value_tex=f"{cj['p'] / 1e6:.4f}", unit_tex=r"\mathrm{MPa}",
             why="The burnt state the simulation is compared with: " + m(f"{cj['p'] / g['p_0']:.2f}") + " times the start pressure, temperature "
                 + m(f"{cj['T']:.1f}" + r"\,\mathrm{K}") + ", density " + m(f"{cj['rho']:.4f}" + r"\,\mathrm{kg/m^3}") + ", gas speed " + m(f"{cj['u']:.1f}" + r"\,\mathrm{m/s}")
                 + " in the direction of the wave. The burnt gas still holds " + m(f"{100 * cj['X']['H2']:.1f}") + " % of unburnt hydrogen by moles: at "
                 + m(f"{cj['T']:.0f}" + r"\,\mathrm{K}") + " part of the water falls apart again.",
             params=[P("v_1", "volume of a kilogram of fresh gas", v1, "m^3/kg"), P("v_2", "volume of a kilogram of burnt gas", v2, "m^3/kg"), P("D", "wave speed", cj["D"], "m/s")],
             check=rf"p_1 + {mflux:.1f}^2\,({v1:.4f} - {v2:.4f}) = {(g['p_0'] + mflux ** 2 * (v1 - v2)) / 1e6:.4f}\ \mathrm{{MPa}}"),
        dict(label="Pressure right behind the shock, before burning", symbol=r"p_{\mathrm{s}}", formula=r"\text{three laws, composition unchanged}", value_tex=f"{fr['p'] / 1e6:.4f}", unit_tex=r"\mathrm{MPa}",
             why="The highest pressure of the steady wave: " + m(f"{fr['p'] / cj['p']:.2f}") + " times the CJ pressure, at " + m(f"{fr['T']:.0f}" + r"\,\mathrm{K}")
                 + ". A simulation shows this peak only if its cells are much smaller than the stretch in which the gas waits to burn; the peak pressure "
                 "found is therefore a measure of how well the front is resolved.",
             params=[P("D", "wave speed (CJ)", cj["D"], "m/s"), P(r"\rho_{\mathrm{s}}", "density behind the shock", fr["rho"], "kg/m^3")],
             check=rf"p_1 + {mflux:.1f}^2\,({v1:.4f} - {1 / fr['rho']:.4f}) = {(g['p_0'] + mflux ** 2 * (v1 - 1 / fr['rho'])) / 1e6:.4f}\ \mathrm{{MPa}}"),
        dict(label="Wait before burning: induction time and length", symbol=r"x_{\mathrm{ind}}", formula=r"x_{\mathrm{ind}} = \int_0^{t_{\mathrm{ind}}} \rho_1 D\,v\,dt", value_tex=f"{zn['x_ind'] * 1e3:.3f}", unit_tex=r"\mathrm{mm}",
             why="Behind the shock the gas waits " + m(f"{zn['t_ind'] * 1e6:.3f}" + r"\,\mathrm{\mu s}") + " before its temperature rises fastest, and travels this far from the shock "
                 "meanwhile. It is the length the cells are measured against: the grids of this report put " + ", ".join(f"{zn['x_ind'] / (g['L'] / n):.1f}" for n in fam[:-1])
                 + f" and {zn['x_ind'] / (g['L'] / fam[-1]):.1f} cells in it. The water reaches 95 % of its CJ value " + m(f"{zn['x_95_water'] * 1e3:.2f}" + r"\,\mathrm{mm}")
                 + " behind the shock, so the steady burning front sits well inside the required " + m(r"1\,\mathrm{mm}") + ".",
             params=[P(r"t_{\mathrm{ind}}", "time from the shock to the fastest temperature rise", zn["t_ind"], "s"), P(r"\rho_1 D", "mass flow through the wave", st["rho0"] * cj["D"], "kg/(m^2 s)"),
                     P(r"v_{\mathrm{s}}", "volume of a kilogram right behind the shock (the hand check keeps it for the whole wait)", 1 / fr["rho"], "m^3/kg")],
             check=rf"\rho_1 D\,v_{{\mathrm{{s}}}}\,t_{{\mathrm{{ind}}}} = {mflux:.1f}\times{1 / fr['rho']:.4f}\times{zn['t_ind'] * 1e6:.3f}\ \mathrm{{\mu s}} = {mflux / fr['rho'] * zn['t_ind'] * 1e3:.3f}\ \mathrm{{mm}}"),
        dict(label="Pressure at the closed end", symbol=r"p_{\mathrm{end}}", formula=r"u_{\mathrm{CJ}} + \int_{p_{\mathrm{CJ}}}^{p_{\mathrm{end}}} \frac{dp}{\rho\,c} = 0", value_tex=f"{tw['p'] / 1e6:.4f}", unit_tex=r"\mathrm{MPa}",
             why="The pressure of the gas at rest behind the expansion wave: " + m(f"{tw['p'] / cj['p']:.3f}") + " of the CJ pressure, at " + m(f"{tw['T']:.0f}" + r"\,\mathrm{K}")
                 + ". The gas is at rest from the wall to " + m(r"c_{\mathrm{end}}\,t") + ", which is " + m(f"{tw['c'] * g['t_end'] * 1e3:.1f}" + r"\,\mathrm{mm}")
                 + " at the end time, about half of the way to the front. The simulation's end pressure is compared with this value.",
             params=[P(r"u_{\mathrm{CJ}}", "gas speed behind the front", cj["u"], "m/s"), P(r"c_{\mathrm{end}}", "speed of sound of the gas at rest", tw["c"], "m/s"),
                     P(r"\gamma_e", "ratio that ties pressure to density along the path, at the front (the hand check keeps it constant)", tw["gamma_e"], "")],
             check=rf"p_{{\mathrm{{CJ}}}}\left(1 - \tfrac{{\gamma_e - 1}}{{2}}\tfrac{{u_{{\mathrm{{CJ}}}}}}{{c_2}}\right)^{{\frac{{2\gamma_e}}{{\gamma_e - 1}}}} = "
                   rf"{cj['p'] * (1 - (tw['gamma_e'] - 1) / 2 * cj['u'] / tw['c_front']) ** (2 * tw['gamma_e'] / (tw['gamma_e'] - 1)) / 1e6:.4f}\ \mathrm{{MPa}}"),
        dict(label="Where the front is at the end time, if it travelled at the CJ speed from the start", symbol=r"x_{\mathrm{f}}", formula=r"x_{\mathrm{f}} = D_{\mathrm{CJ}}\,t_{\mathrm{end}}",
             value_tex=f"{th['x_front_cj'] * 1e3:.2f}", unit_tex=r"\mathrm{mm}",
             why="The tube is " + m(f"{g['L'] * 1e3:.0f}" + r"\,\mathrm{mm}") + " long, so the wave does not reach the far wall within the time of interest and the gas ahead of it must "
                 "stay untouched. The wave speed is read over the second half of the time, in which a CJ wave covers " + m(f"{th['x_front_cj'] * 5e2:.1f}" + r"\,\mathrm{mm}") + ".",
             params=[P(r"D_{\mathrm{CJ}}", "CJ speed", cj["D"], "m/s"), P(r"t_{\mathrm{end}}", "end of the time of interest (given)", g["t_end"], "s")],
             check=rf"{cj['D']:.2f}\times{g['t_end'] * 1e6:.0f}\times 10^{{-6}} = {th['x_front_cj']:.5f}\ \mathrm{{m}}"),
        dict(label="Heat released per kilogram up to the CJ state", symbol=r"q", formula=r"q = \sum_i h_i^{\circ}\,\left(Y_{i,1} - Y_{i,\mathrm{CJ}}\right)", value_tex=f"{cj['q'] / 1e6:.4f}", unit_tex=r"\mathrm{MJ/kg}",
             why="The scale of the energy balance: every energy error in this report is given as a fraction of the heat released so far. One millimetre of tube releases "
                 + m(f"{heat_per_mm:.0f}" + r"\,\mathrm{J}") + " per square metre of cross-section. The ignition layer holds " + m(f"{st['layer_surplus']:.0f}" + r"\,\mathrm{J/m^2}")
                 + " more than the same mass of fresh gas, as much as the burning of " + m(f"{st['layer_surplus'] / heat_per_mm:.1f}" + r"\,\mathrm{mm}") + " of mixture. The layer is itself unburnt "
                 "mixture, so it burns too.",
             params=[P(r"h_i^{\circ}", "chemical energy of species i per kilogram (its enthalpy at the standard temperature)", 0, "J/kg"), P(r"Y_{\mathrm{H_2O,CJ}}", "mass fraction of water at the CJ state", cj["Y"]["H2O"], "")],
             check=rf"\rho_1\,q\times 1\ \mathrm{{mm}} = {st['rho0']:.4f}\times{cj['q']:.0f}\times 0.001 = {heat_per_mm:.0f}\ \mathrm{{J/m^2}}"),
    ]
    z_p, z_T = zn["p_end"] / cj["p"] - 1, zn["T_end"] / cj["T"] - 1
    names = [c["name"] for c in th["checks"]]
    self_checks = [dict(name=n, ok=c["ok"]) for n, c in zip(names[:4], th["checks"][:4])] + [
        dict(name="The CJ state of the reference file satisfies the three conservation relations with this report's enthalpies", ok=th["checks"][4]["ok"]),
        dict(name="The CJ state computed here reproduces the reference file: speed to " + m(tex_num(abs(r["D"]), 1)) + ", pressure to " + m(tex_num(abs(r["p"]), 1))
                  + ", temperature to " + m(tex_num(abs(r["T"]), 1)) + " (relative)", ok=th["checks"][5]["ok"]),
        dict(name="The state right behind the shock reproduces the reference file: pressure to " + m(tex_num(abs(fr["vs_reference"]["p"]), 1)) + ", temperature to "
                  + m(tex_num(abs(fr["vs_reference"]["T"]), 1)), ok=th["checks"][6]["ok"]),
        dict(name="The steady structure ends at the CJ state: " + m(f"{zn['x_end'] * 1e3:.0f}" + r"\,\mathrm{mm}") + " behind the shock its pressure is " + m(f"{100 * z_p:+.2f}\\,\\%")
                  + " and its temperature " + m(f"{100 * z_T:+.2f}\\,\\%") + " from the CJ values", ok=th["checks"][7]["ok"]),
        dict(name="The speed of sound found along the expansion path equals the speed at which the burnt gas leaves the wave: " + m(f"{tw['c_front']:.1f}") + " against "
                  + m(f"{cj['w']:.1f}" + r"\,\mathrm{m/s}"), ok=th["checks"][8]["ok"]),
        dict(name="Along the expansion, pressure, temperature and density fall together, and the gas comes to rest at " + m(f"{tw['p'] / cj['p']:.3f}") + " of the CJ pressure",
             ok=th["checks"][9]["ok"]),
    ]
    return dict(intro=intro, steps=steps, derived=derived, self_checks=self_checks)


if __name__ == "__main__":
    th = numbers(force=True)
    for k in ("start", "cj", "front", "znd", "taylor"):
        print(k, json.dumps({a: b for a, b in th[k].items() if not isinstance(b, dict)}, default=float))
    for c in th["checks"]:
        print("  ok  " if c["ok"] else "  FAIL", c["name"])
