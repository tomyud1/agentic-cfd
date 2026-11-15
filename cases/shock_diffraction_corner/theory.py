"""What can be known about the shock at the corner without simulating it. Built with SymPy.

Exact (proven here from the conservation laws across a jump):
  the state behind the flat shock, its speed, what the duct brings into the chamber per second,
  and how far the signal from the corner has spread at a given time (so: which part of the shock
  is still flat, and that no wave reaches the top, the floor or the right side before t_end).
Estimates (approximate theories, named as such in the report):
  the speed of the bent shock along the step face (Whitham's geometrical shock dynamics) and the
  pressure the gas would reach if it followed the wall round the corner (Prandtl-Meyer expansion).

evaluate(given) returns the report's theory block and `Flow`, the numbers.
"""
import numpy as np
import sympy as sp
from scipy import integrate, optimize

from cfdlib.report import tex_num

g = sp.Symbol("gamma", positive=True)
M = sp.Symbol("M_s", positive=True)
rho0, p0 = sp.symbols("rho_0 p_0", positive=True)


def shock_relations():
    """A flat shock of Mach number M moving into gas at rest (state 0); state 2 is behind it.
    Returns the relations and the checks that mass, momentum and energy are conserved across it."""
    c0 = sp.sqrt(g * p0 / rho0)
    W = M * c0
    rho2 = rho0 * (g + 1) * M ** 2 / ((g - 1) * M ** 2 + 2)
    p2 = p0 * (2 * g * M ** 2 - (g - 1)) / (g + 1)
    u2 = 2 * c0 * (M ** 2 - 1) / ((g + 1) * M)
    rel = W - u2                                               # gas speed behind the shock, seen from the shock
    E0, E2 = p0 / (g - 1), p2 / (g - 1) + rho2 * u2 ** 2 / 2   # energy per unit volume
    checks = {
        r"shock: mass, \(\rho_0 W = \rho_2 (W - u_2)\)": sp.simplify(rho0 * W - rho2 * rel) == 0,
        r"shock: momentum, \(p_0 + \rho_0 W^2 = p_2 + \rho_2 (W - u_2)^2\)":
            sp.simplify(p0 + rho0 * W ** 2 - p2 - rho2 * rel ** 2) == 0,
        r"shock: energy, \(h_0 + \tfrac12 W^2 = h_2 + \tfrac12 (W - u_2)^2\) with \(h = \gamma p/((\gamma-1)\rho)\)":
            sp.simplify(g / (g - 1) * p0 / rho0 + W ** 2 / 2 - g / (g - 1) * p2 / rho2 - rel ** 2 / 2) == 0,
        r"inflow: the mass the duct brings in is the mass the shock has added, \(\rho_2 u_2 = (\rho_2 - \rho_0) W\)":
            sp.simplify(rho2 * u2 - (rho2 - rho0) * W) == 0,
        r"inflow: the same for energy, \((E_2 + p_2) u_2 = (E_2 - E_0) W\) with \(E = p/(\gamma-1) + \tfrac12\rho u^2\)":
            sp.simplify((E2 + p2) * u2 - (E2 - E0) * W) == 0,
    }
    return dict(c0=c0, W=W, rho2=rho2, p2=p2, u2=u2, E2=E2), checks


def area_rule():
    """Whitham's rule for a shock whose front area A changes: g(M) = -(1/A) dA/dM. It follows from
    applying the relation that holds along a forward sound wave in a duct of slowly changing area,
    dp + rho c du = -rho c^2 u/(u + c) dA/A, to the state right behind the shock."""
    r = shock_relations()[0]
    c = sp.sqrt(g * r["p2"] / r["rho2"])
    gM = sp.simplify((sp.diff(r["p2"], M) + r["rho2"] * c * sp.diff(r["u2"], M)) * (r["u2"] + c) / (r["rho2"] * c ** 2 * r["u2"]))
    return gM


def prandtl_meyer():
    """Turning angle nu of a steady supersonic flow that expands from Mach 1 to Mach number
    sqrt(1 + s^2), and the check that it satisfies d nu = sqrt(M^2 - 1) dq/q with constant total enthalpy."""
    s = sp.Symbol("s", positive=True)                          # s = sqrt(M^2 - 1)
    k = sp.sqrt((g + 1) / (g - 1))
    nu = k * sp.atan(s / k) - sp.atan(s)
    # dq/q = dM / (M (1 + (gamma-1)/2 M^2)), M = sqrt(1+s^2), dM/ds = s/M
    target = s * (s / (1 + s ** 2)) / (1 + (g - 1) / 2 * (1 + s ** 2))
    ok = sp.simplify(sp.diff(nu, s) - target) == 0
    return nu, s, ok


class Flow:
    """The numbers of this problem (SI units)."""

    def __init__(self, given):
        v = {k: x["value"] for k, x in given.items()}
        self.given = v
        gm = self.gamma = v["gamma"]
        self.L, self.H, self.Hs, self.t_end = v["L"], v["H"], v["H_s"], v["t_end"]
        self.p0, self.T0, self.R, self.Ms = v["p_0"], v["T_0"], v["R"], v["M_s"]
        self.rho0 = self.p0 / (self.R * self.T0)
        self.c0 = np.sqrt(gm * self.R * self.T0)
        r, _ = shock_relations()
        sub = {g: gm, M: self.Ms, rho0: self.rho0, p0: self.p0}
        f = lambda e: float(e.subs(sub).evalf(30))
        self.W, self.rho2, self.p2, self.u2 = f(r["W"]), f(r["rho2"]), f(r["p2"]), f(r["u2"])
        self.T2 = self.p2 / (self.R * self.rho2)
        self.c2 = np.sqrt(gm * self.p2 / self.rho2)
        self.M2 = self.u2 / self.c2
        self.E0 = self.p0 / (gm - 1)
        self.E2 = self.p2 / (gm - 1) + 0.5 * self.rho2 * self.u2 ** 2
        # per metre of depth (the third direction)
        self.mass0, self.energy0 = self.rho0 * self.L * self.H, self.E0 * self.L * self.H
        self.mdot = self.rho2 * self.u2 * (self.H - self.Hs)
        self.edot = (self.E2 + self.p2) * self.u2 * (self.H - self.Hs)
        self.s_max = self.u2 + self.c2
        # the signal from the corner: a circle of radius c2 t whose centre moves with the gas
        self.v_flat = np.sqrt(self.c2 ** 2 - (self.W - self.u2) ** 2)   # speed of its end point along the shock
        self.mach_angle = np.arcsin(self.c2 / self.u2)
        self.t_top = (self.H - self.Hs) / self.c2                       # signal reaches the top wall
        self.t_right = self.L / self.W                                  # shock reaches the right side
        # Prandtl-Meyer: if the gas followed the wall round the corner
        nu, s, _ = prandtl_meyer()
        nu_f = sp.lambdify(s, nu.subs(g, gm), "numpy")
        self.nu2 = float(nu_f(np.sqrt(self.M2 ** 2 - 1)))
        self.nu_max = (np.sqrt((gm + 1) / (gm - 1)) - 1) * np.pi / 2
        s90 = optimize.brentq(lambda x: nu_f(x) - self.nu2 - np.pi / 2, 1e-6, 1e6)
        self.M_turned = float(np.sqrt(1 + s90 ** 2))
        ratio = lambda Mach: (1 + (gm - 1) / 2 * Mach ** 2) ** (-gm / (gm - 1))   # p / total pressure
        self.p_turned = self.p2 * ratio(self.M_turned) / ratio(self.M2)
        # Whitham's estimate of the shock that runs down the step face
        gM = sp.lambdify(M, area_rule().subs(g, gm), "numpy")
        self.n_strong = float(1e6 * gM(1e6))                            # g(M) -> n/M for a strong shock
        omega = lambda a, b: integrate.quad(lambda m_: np.sqrt(gM(m_) / m_), a, b, epsabs=1e-12, epsrel=1e-12)[0]
        self.Mw = optimize.brentq(lambda m_: omega(m_, self.Ms) - np.pi / 2, 1.0001, self.Ms)
        self.Mw_strong = self.Ms * np.exp(-np.pi / 2 / np.sqrt(self.n_strong))
        self.d_wall = self.Mw * self.c0 * self.t_end                    # estimated distance down the step face
        self.x_top = self.W * self.t_end

    def flat_from(self, t):
        """Height above which the shock is still flat at time t (no signal from the corner yet)."""
        return self.Hs + self.v_flat * t

    def signal_top(self, t):
        """Highest point the corner's signal has reached at time t."""
        return self.Hs + self.c2 * t

    def totals(self, t):
        """Mass and energy in the chamber at time t, per metre of depth: the start plus the exact inflow."""
        return self.mass0 + self.mdot * t, self.energy0 + self.edot * t


def evaluate(given):
    fl = Flow(given)
    rel, checks = shock_relations()
    nu, s, pm_ok = prandtl_meyer()
    checks[r"expansion round a corner: the turning angle \(\nu(M)\) satisfies \(d\nu = \sqrt{M^2-1}\,dq/q\) at constant total enthalpy"] = pm_ok
    gm = fl.gamma
    n_closed = 1 + 2 / gm + np.sqrt(2 * gm / (gm - 1))
    checks[r"area rule: for a very strong shock \(g(M) \to n/M\) with \(n = 1 + 2/\gamma + \sqrt{2\gamma/(\gamma-1)}\) "
           rf"(\({n_closed:.4f}\) for this \(\gamma\); the derived rule gives \({fl.n_strong:.4f}\))"] = abs(fl.n_strong - n_closed) < 1e-4
    t = fl.t_end
    checks[rf"no wave reaches the top wall before \(t_{{\mathrm{{end}}}}\): the corner's signal needs \({1e3 * fl.t_top:.3f}\,\mathrm{{ms}}\)"] = fl.t_top > t
    checks[rf"the shock does not reach the right side before \(t_{{\mathrm{{end}}}}\): it needs \({1e3 * fl.t_right:.3f}\,\mathrm{{ms}}\)"] = fl.t_right > t
    checks[r"the gas behind the shock moves faster than sound, so the corner cannot signal back into the duct "
           rf"(\(M_2 = {fl.M2:.3f} > 1\))"] = fl.M2 > 1

    c0s, Ws, r2s, p2s, u2s, c2s, T0s, Rs, Hs_, Hh, ts, E2s = sp.symbols("c_0 W rho_2 p_2 u_2 c_2 T_0 R H_s H t_end E_2", positive=True)
    names = {g: r"\gamma", M: "M_s", rho0: r"\rho_0", p0: "p_0", c0s: "c_0", Ws: "W", r2s: r"\rho_2", p2s: "p_2", u2s: "u_2", E2s: "E_2",
             c2s: "c_2", T0s: "T_0", Rs: "R", Hs_: "H_s", Hh: "H", ts: r"t_{\mathrm{end}}"}
    par = {
        g: dict(tex=r"\gamma", meaning="ratio of specific heats", value=gm, unit=""),
        M: dict(tex="M_s", meaning="shock Mach number", value=fl.Ms, unit=""),
        Rs: dict(tex="R", meaning="specific gas constant", value=fl.R, unit=r"\mathrm{J/(kg\,K)}"),
        T0s: dict(tex="T_0", meaning="temperature of the still air", value=fl.T0, unit=r"\mathrm{K}"),
        p0: dict(tex="p_0", meaning="pressure of the still air", value=fl.p0, unit=r"\mathrm{Pa}"),
        rho0: dict(tex=r"\rho_0", meaning="density of the still air", value=fl.rho0, unit=r"\mathrm{kg/m^3}"),
        c0s: dict(tex="c_0", meaning="speed of sound in the still air", value=fl.c0, unit=r"\mathrm{m/s}"),
        Ws: dict(tex="W", meaning="speed of the flat shock", value=fl.W, unit=r"\mathrm{m/s}"),
        r2s: dict(tex=r"\rho_2", meaning="density behind the flat shock", value=fl.rho2, unit=r"\mathrm{kg/m^3}"),
        p2s: dict(tex="p_2", meaning="pressure behind the flat shock", value=fl.p2, unit=r"\mathrm{Pa}"),
        u2s: dict(tex="u_2", meaning="gas speed behind the flat shock", value=fl.u2, unit=r"\mathrm{m/s}"),
        E2s: dict(tex="E_2", meaning="energy per unit volume behind the flat shock", value=fl.E2, unit=r"\mathrm{J/m^3}"),
        c2s: dict(tex="c_2", meaning="speed of sound behind the flat shock", value=fl.c2, unit=r"\mathrm{m/s}"),
        Hs_: dict(tex="H_s", meaning="height of the corner", value=fl.Hs, unit=r"\mathrm{m}"),
        Hh: dict(tex="H", meaning="height of the chamber", value=fl.H, unit=r"\mathrm{m}"),
        ts: dict(tex=r"t_{\mathrm{end}}", meaning="time of interest", value=fl.t_end, unit=r"\mathrm{s}"),
    }

    def entry(key, label, symbol, expr, unit_tex, why, expect, sig=6):
        used = [par[q] for q in par if q in expr.free_symbols]
        num = {q: (rf"\left({tex_num(par[q]['value'], sig)}\right)" if par[q]["unit"] else tex_num(par[q]["value"], sig))
               for q in par if q in expr.free_symbols}
        value = float(expr.subs({q: par[q]["value"] for q in par}).evalf(30))
        if abs(value - expect) > 1e-9 * max(abs(expect), 1e-300):
            raise AssertionError(f"theory entry {key}: formula gives {value}, the Flow object has {expect}")
        return dict(key=key, label=label, symbol=symbol, value=value, value_tex=tex_num(value, sig), unit_tex=unit_tex,
                    formula=rf"{symbol} = {sp.latex(expr, symbol_names=names)}",
                    check=rf"\begin{{aligned}}{symbol} &= {sp.latex(expr, symbol_names=num)}\\ &= {tex_num(value, sig)}\ {unit_tex}"
                          r"\end{aligned}", why=why, params=used)

    derived = [
        entry("c0", "Speed of sound in the still air", "c_0", sp.sqrt(g * Rs * T0s), r"\mathrm{m/s}",
              "Small pressure changes travel at this speed. The shock Mach number is the shock's speed divided by it.", fl.c0),
        entry("rho0", "Density of the still air", r"\rho_0", p0 / (Rs * T0s), r"\mathrm{kg/m^3}",
              "From the ideal-gas law. The solver is given pressure and temperature; this is what it must then hold.", fl.rho0),
        entry("W", "Speed of the flat shock", "W", M * c0s, r"\mathrm{m/s}",
              "Requested output (a). It also gives how far along the top wall the shock is at the time of interest, "
              rf"\(W t_{{\mathrm{{end}}}} = {fl.x_top:.4f}\,\mathrm{{m}}\) (requested output (e)).", fl.W, sig=7),
        entry("p2", "Pressure behind the flat shock", "p_2", p0 * (2 * g * M ** 2 - (g - 1)) / (g + 1), r"\mathrm{Pa}",
              rf"Requested output (a). The shock raises the pressure \({fl.p2 / fl.p0:.2f}\) times.", fl.p2, sig=7),
        entry("rho2", "Density behind the flat shock", r"\rho_2", rho0 * (g + 1) * M ** 2 / ((g - 1) * M ** 2 + 2),
              r"\mathrm{kg/m^3}", r"Requested output (a). However strong the shock, the density rises at most "
              r"\((\gamma+1)/(\gamma-1) = 6\) times; this shock is close to that limit.", fl.rho2, sig=7),
        entry("u2", "Gas speed behind the flat shock", "u_2", 2 * c0s * (M ** 2 - 1) / ((g + 1) * M), r"\mathrm{m/s}",
              "Requested output (a). The shock sets the still air in motion in its own direction.", fl.u2, sig=7),
        entry("c2", "Speed of sound behind the flat shock", "c_2", sp.sqrt(g * p2s / r2s), r"\mathrm{m/s}",
              rf"The gas behind the shock is hot (\({fl.T2:.0f}\,\mathrm{{K}}\)), so sound is fast there. Still, the gas moves "
              rf"faster: its Mach number is \(M_2 = u_2/c_2 = {fl.M2:.3f}\). Nothing that happens at the corner can travel "
              "back into the duct, so the state of the air entering the chamber stays the one behind the shock.", fl.c2),
        entry("mdot", "Mass entering through the duct per second, per metre of depth", r"\dot m", r2s * u2s * (Hh - Hs_),
              r"\mathrm{kg/(s\,m)}", "Requested output (b): the mass in the chamber must be the mass at the start plus this rate "
              "times the time.", fl.mdot),
        entry("edot", "Energy entering through the duct per second, per metre of depth", r"\dot E",
              (E2s + p2s) * u2s * (Hh - Hs_), r"\mathrm{W/m}",
              "Requested output (b). The gas carries its internal and kinetic energy in, "
              r"\(E_2 = p_2/(\gamma-1) + \tfrac12\rho_2 u_2^2\) per unit volume, and the gas behind it does work pushing it in "
              r"(the extra \(p_2\)).", fl.edot, sig=4),
        entry("v_flat", "Speed at which the flat part of the shock is used up", r"v_{\mathrm{flat}}",
              sp.sqrt(c2s ** 2 - (Ws - u2s) ** 2), r"\mathrm{m/s}",
              "The corner's signal is a sound wave: a circle of radius \\(c_2 t\\) whose centre drifts with the gas at "
              "\\(u_2\\). The shock runs ahead of that centre at \\(W - u_2\\), so the circle cuts the shock at a height "
              rf"\(v_{{\mathrm{{flat}}}}\,t\) above the corner. At the time of interest the shock is flat above "
              rf"\(y = H_s + v_{{\mathrm{{flat}}}} t_{{\mathrm{{end}}}} = {fl.flat_from(t):.4f}\,\mathrm{{m}}\). Requested outputs "
              "(a) and (c) are measured there.", fl.v_flat, sig=5),
        entry("s_max", "Fastest signal behind the flat shock", "u_2 + c_2", u2s + c2s, r"\mathrm{m/s}",
              "Sound running with the flow. An explicit solver must not let a signal cross a whole cell in one time step, "
              "so this speed sets the time step (see Decisions).", fl.s_max),
        dict(key="p_turned", label="Estimate: pressure if the gas followed the wall round the corner", symbol=r"p_{90}",
             value=fl.p_turned, value_tex=tex_num(fl.p_turned, 3), unit_tex=r"\mathrm{Pa}",
             formula=r"\begin{aligned}&\nu(M_{90}) = \nu(M_2) + \frac{\pi}{2}\\[2pt] &\nu(M) = \sqrt{\tfrac{\gamma+1}{\gamma-1}}\,"
                     r"\arctan\sqrt{\tfrac{\gamma-1}{\gamma+1}(M^2-1)}\\ &\qquad\quad - \arctan\sqrt{M^2-1}\\[2pt] "
                     r"&\frac{p_{90}}{p_2} = \left(\frac{1+\tfrac{\gamma-1}{2}M_2^2}{1+\tfrac{\gamma-1}{2}M_{90}^2}\right)^{\frac{\gamma}{\gamma-1}}\end{aligned}",
             check=rf"\begin{{aligned}}\nu(M_2) &= {np.degrees(fl.nu2):.2f}^\circ,\quad M_{{90}} = {fl.M_turned:.2f}\\ "
                   rf"p_{{90}} &= {tex_num(fl.p_turned, 3)}\ \mathrm{{Pa}} = {tex_num(fl.p_turned / fl.p0, 2)}\,p_0\end{{aligned}}",
             why="A steady supersonic stream can turn round a sharp corner through an expansion fan (Prandtl-Meyer flow). "
                 rf"Turning by the full \(90^\circ\) would take this stream to Mach \({fl.M_turned:.1f}\) and to a pressure "
                 "far below that of the still air in the chamber. That cannot last: the stream leaves the wall at the corner, "
                 "and a second, weaker shock must bring the over-expanded gas back up in pressure. It also warns that the "
                 "pressure and density next to the corner get very low, which is where a solver can produce a negative "
                 "value and stop. An estimate of what to expect, not a number the simulation is checked against.",
             params=[par[g], dict(tex="M_2", meaning="Mach number of the gas behind the flat shock", value=fl.M2, unit=""), par[p2s]]),
        dict(key="Mw", label="Estimate: Mach number of the bent shock on the step face (Whitham's theory)", symbol="M_w",
             value=fl.Mw, value_tex=tex_num(fl.Mw, 4), unit_tex="",
             formula=r"\begin{aligned}&\int_{M_w}^{M_s}\sqrt{\frac{g(M)}{M}}\,dM = \frac{\pi}{2}\\[2pt] &g(M) = -\frac{1}{A}\frac{dA}{dM} = "
                     r"\frac{\left(\dfrac{dp}{dM} + \rho c\,\dfrac{du}{dM}\right)(u + c)}{\rho c^2 u}\end{aligned}",
             check=rf"\begin{{aligned}}M_w &= {fl.Mw:.4f}\\ M_s e^{{-\pi/(2\sqrt n)}} &= {fl.Mw_strong:.4f}\quad\text{{(strong-shock form)}}\\ "
                   rf"M_w c_0 t_{{\mathrm{{end}}}} &= {fl.d_wall:.4f}\ \mathrm{{m}}\end{{aligned}}",
             why="Whitham's geometrical shock dynamics treats each piece of the shock front as if it moved along a narrow "
                 "tube: where the front spreads out (round the corner) the tube widens and the shock weakens, by the rule "
                 r"\(g(M)\), in which \(p, \rho, u, c\) are the state behind a shock of Mach number \(M\). For a "
                 r"\(90^\circ\) corner it gives this Mach number at the wall. The bent shock would then have run "
                 rf"\({fl.d_wall:.3f}\,\mathrm{{m}}\) down the step face at the time of interest, short of the floor "
                 rf"(\(H_s = {fl.Hs}\,\mathrm{{m}}\)): the flow should still have no length of its own. An approximate "
                 "theory, written here from memory of Whitham (1957) and derived again in code; the simulation is compared "
                 "with it but not judged by it.",
             params=[par[g], par[M], par[c0s], par[ts]]),
    ]
    steps = [
        dict(label="Governing equations: mass, momentum and energy of a gas without friction or heat conduction, in two dimensions",
             latex=r"\frac{\partial}{\partial t}\begin{pmatrix}\rho\\ \rho u\\ \rho v\\ E\end{pmatrix} + \frac{\partial}{\partial x}"
                   r"\begin{pmatrix}\rho u\\ \rho u^2 + p\\ \rho u v\\ u(E+p)\end{pmatrix} + \frac{\partial}{\partial y}"
                   r"\begin{pmatrix}\rho v\\ \rho u v\\ \rho v^2 + p\\ v(E+p)\end{pmatrix} = 0,\qquad E = \frac{p}{\gamma-1} + \frac12\rho(u^2+v^2)",
             note=r"The Euler equations. Each line says: what a patch of the plane holds changes only by what flows through "
                  "its edges. That stays true across a jump, which is why a method built on these balances moves a shock "
                  "at the right speed."),
        dict(label="Across the flat shock: three balances fix the state behind it",
             latex=r"\frac{p_2}{p_0} = \frac{2\gamma M_s^2 - (\gamma-1)}{\gamma+1},\qquad \frac{\rho_2}{\rho_0} = "
                   r"\frac{(\gamma+1)M_s^2}{(\gamma-1)M_s^2+2},\qquad u_2 = \frac{2c_0}{\gamma+1}\left(M_s - \frac{1}{M_s}\right),\qquad W = M_s c_0",
             note="The shock jump conditions (Rankine-Hugoniot) for a shock moving into gas at rest. SymPy checks below "
                  "that mass, momentum and energy flowing into the jump equal what flows out."),
        dict(label="What the duct brings in",
             latex=r"M(t) = \rho_0 L H + \rho_2 u_2 (H - H_s)\,t,\qquad E_{\mathrm{tot}}(t) = \frac{p_0}{\gamma-1} L H + (E_2 + p_2)\,u_2 (H - H_s)\,t",
             note="Per metre of depth. All other sides are walls, so nothing else enters or leaves."),
        dict(label="Where the corner has been felt",
             latex=r"(x - u_2 t)^2 + (y - H_s)^2 \le (c_2 t)^2\quad\text{and the two lines from the corner that touch this circle},"
                   r"\qquad \sin\mu = \frac{c_2}{u_2}",
             note=rf"The corner starts to matter at \(t = 0\). Its signal spreads as a sound wave in gas that moves at \(u_2\): "
                  rf"a circle that grows and drifts. Because the gas is faster than sound, the corner itself lies outside the "
                  rf"circle, and the disturbed region is closed by the two lines from the corner that touch it, at the angle "
                  rf"\(\mu = {np.degrees(fl.mach_angle):.1f}^\circ\) to the flow (the first ray of the expansion fan). "
                  "Outside this region the flow is exactly the flat shock."),
        dict(label="No length of its own: the picture only stretches",
             latex=r"\rho,\ u,\ v,\ p = \text{functions of }\ \frac{x}{t},\ \frac{y - H_s}{t}\qquad\text{(until a wave meets a far wall)}",
             note="The equations have no length or time scale, the corner is a point and the start state is uniform. "
                  "The same holds for the numerical method on equal square cells with a time step in fixed proportion to the "
                  "cell size: its answer after a number of steps depends on the cell count from the corner only. So the "
                  r"picture at \(t_{\mathrm{end}}/2\) on one grid is the picture at \(t_{\mathrm{end}}\) on the grid with cells "
                  "twice as large. Comparing two times is then the same as comparing two grids (requested output (d))."),
    ]
    theory = dict(
        intro="The whole flow has no exact solution: the bent shock, the expansion at the corner and the rolled-up layer "
              "behind it must be computed. Three things are known exactly and are used as checks. First, the flat part of "
              "the shock and the state behind it, from the conservation laws across a jump. Second, the mass and energy the "
              "duct brings in, because the air enters faster than sound and nothing can change it. Third, the region the "
              "corner can have influenced, which says which part of the shock must still be flat. Two approximate "
              "theories add an estimate each: how fast the bent shock runs down the step face, and how low the pressure "
              "gets at the corner.",
        steps=steps, derived=derived, self_checks=[dict(name=k, ok=bool(v)) for k, v in checks.items()])
    return theory, fl


if __name__ == "__main__":
    from pathlib import Path
    import yaml
    prob = yaml.safe_load((Path(__file__).parent / "problem.yaml").read_text(encoding="utf-8"))
    th, fl = evaluate(prob["given"])
    for c in th["self_checks"]:
        print("ok  " if c["ok"] else "FAIL", c["name"])
    for d in th["derived"]:
        print(f"{d['key']:10s} {d['value']:.10g}")
    print(f"T2 {fl.T2:.2f} K, M2 {fl.M2:.4f}, flat above y = {fl.flat_from(fl.t_end):.4f} m, signal top {fl.signal_top(fl.t_end):.4f} m, "
          f"x_top {fl.x_top:.4f} m, t_top {fl.t_top:.3e} s, t_right {fl.t_right:.3e} s")
    print(f"Whitham: Mw {fl.Mw:.4f} (strong form {fl.Mw_strong:.4f}, n {fl.n_strong:.4f}), distance {fl.d_wall:.4f} m; "
          f"Prandtl-Meyer: nu2 {np.degrees(fl.nu2):.2f} deg, M90 {fl.M_turned:.2f}, p90/p0 {fl.p_turned / fl.p0:.3e}")
