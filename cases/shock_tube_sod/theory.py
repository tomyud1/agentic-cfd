"""Exact solution of the shock-tube problem (the Riemann problem of gas dynamics), built with SymPy.

Nothing is typed in from a textbook table. The pieces are written down as formulas and SymPy proves
each one: the shock relations satisfy the three conservation laws across a jump, the expansion fan
satisfies the Euler equations, and the pressure between the waves is solved twice (SymPy and SciPy).
The assembled solution is then checked as a whole: its total mass, momentum and energy at the time
of interest must equal what the tube started with.

evaluate(given) returns the report's theory block and `Exact`, the solution as numbers: point
values, wave positions and exact cell averages on any grid.
"""
import numpy as np
import sympy as sp
from scipy import optimize

from cfdlib.report import tex_num

g = sp.Symbol("gamma", positive=True)


def _rat(x):
    return sp.Rational(str(x))


class _Shown(float):
    """A value for a theory card's symbol table. The report template shows it with 4 significant digits
    through Python's 'g' format, which writes a five-digit number as 1e+04. This float prints such a
    number in full instead; it behaves like any other float in arithmetic."""

    def __format__(self, spec):
        v = float(self)
        return f"{v:.5g}" if spec.endswith("g") and 1e4 <= abs(v) < 1e5 else format(v, spec)


def shock_relations():
    """A shock moving into gas at rest (state 1) that raises the pressure by the ratio P = p2/p1.
    Returns the symbolic relations and the self-checks that they conserve mass, momentum and energy."""
    rho1, p1, P = sp.symbols("rho1 p1 P", positive=True)
    c1sq = g * p1 / rho1
    S2 = c1sq * ((g + 1) * P + (g - 1)) / (2 * g)                # shock speed squared
    rho2 = rho1 * ((g + 1) * P + (g - 1)) / ((g - 1) * P + (g + 1))
    p2 = P * p1
    # Seen from the shock, gas enters at speed S and leaves at S - u2. Mass: rho1 S = rho2 (S - u2).
    rel2 = sp.sqrt(S2) * rho1 / rho2                              # S - u2
    u2 = sp.sqrt(S2) - rel2
    checks = {
        r"shock: momentum is conserved across the jump, \(p_1 + \rho_1 S^2 = p_2 + \rho_2 (S - u_2)^2\)":
            sp.simplify(p1 + rho1 * S2 - p2 - rho2 * rel2 ** 2) == 0,
        r"shock: energy is conserved across the jump, \(h_1 + \tfrac12 S^2 = h_2 + \tfrac12 (S - u_2)^2\) with \(h = \gamma p/((\gamma-1)\rho)\)":
            sp.simplify(g / (g - 1) * p1 / rho1 + S2 / 2 - g / (g - 1) * p2 / rho2 - rel2 ** 2 / 2) == 0,
        r"shock: the gas speed behind it is \(u_2 = (p_2 - p_1)\sqrt{A/(p_2 + B)}\), \(A = 2/((\gamma+1)\rho_1)\), \(B = (\gamma-1)p_1/(\gamma+1)\)":
            sp.simplify(u2 ** 2 - (p2 - p1) ** 2 * (2 / ((g + 1) * rho1)) / (p2 + (g - 1) / (g + 1) * p1)) == 0,
    }
    return dict(rho1=rho1, p1=p1, P=P, S=sp.sqrt(S2), rho2=rho2, u2=u2), checks


def fan_relations():
    """The expansion fan that runs to the left into gas at rest (state L). Everything depends on
    xi = (x - x0)/t only. Returns the relations and the checks that they satisfy the Euler equations."""
    cL, rhoL = sp.symbols("c_L rho_L", positive=True)
    # The fan is written with s = (local speed of sound)/c_L as the variable: s is positive, so SymPy
    # can combine the powers s^(2/(gamma-1)) for any gamma. The ray variable follows from s:
    #   s = 2/(gamma+1) - (gamma-1)/(gamma+1) xi/c_L   <=>   xi = c_L (2 - (gamma+1) s)/(gamma-1)
    s = sp.Symbol("s", positive=True)
    a = 2 / (g - 1)
    pL = rhoL * cL ** 2 / g
    xi = cL * (2 - (g + 1) * s) / (g - 1)
    ds_dxi = -(g - 1) / ((g + 1) * cL)
    rho = rhoL * s ** a
    u = 2 / (g + 1) * (cL + xi)
    p = pL * s ** (a + 2)                                          # 2 gamma/(gamma-1) = a + 2
    E = p / (g - 1) + rho * u ** 2 / 2

    def residual(q, flux):
        """(d q/dt + d flux/dx) t for functions of xi = (x - x0)/t: -xi dq/dxi + dflux/dxi, with
        d/dxi = (ds/dxi) d/ds; divided by the common power of s so that a polynomial is left."""
        return sp.simplify(sp.expand(sp.powsimp(sp.expand((-xi * sp.diff(q, s) + sp.diff(flux, s)) * ds_dxi / s ** (a - 1)))))

    checks = {
        r"fan: mass equation \(\partial_t \rho + \partial_x(\rho u) = 0\)": residual(rho, rho * u) == 0,
        r"fan: momentum equation \(\partial_t(\rho u) + \partial_x(\rho u^2 + p) = 0\)":
            residual(rho * u, rho * u ** 2 + p) == 0,
        r"fan: energy equation \(\partial_t E + \partial_x(u(E + p)) = 0\)": residual(E, u * (E + p)) == 0,
        r"fan: each ray \(x - x_0 = \xi t\) moves at the speed of a left-running sound wave, \(\xi = u - c\)":
            sp.simplify(u - cL * s - xi) == 0,
        r"fan: no entropy is made, \(p/\rho^\gamma = p_L/\rho_L^\gamma\)":
            sp.simplify(sp.expand_log(sp.log(p) - g * sp.log(rho) - sp.log(pL) + g * sp.log(rhoL), force=True)) == 0,
        r"fan: it starts at the undisturbed gas, \(\rho = \rho_L,\ u = 0\) at \(\xi = -c_L\) (\(s = 1\))":
            sp.simplify(rho.subs(s, 1) - rhoL) == 0 and sp.simplify(u.subs(s, 1)) == 0 and sp.simplify(xi.subs(s, 1) + cL) == 0,
        r"fan: written with \(s = c/c_L\), the velocity is \(u = 2c_L(1 - s)/(\gamma - 1)\)":
            sp.simplify(u - a * cL * (1 - s)) == 0,
    }
    # Integrals over the fan, needed for exact cell averages: x = x0 + t xi(s), so
    # dx = -t c_L (gamma+1)/(gamma-1) ds. The functions of s to integrate: rho, rho u and E.
    q = dict(rho=rho, rhou=rho * a * cL * (1 - s), E=p / (g - 1) + rho * (a * cL * (1 - s)) ** 2 / 2)
    return dict(cL=cL, rhoL=rhoL, s=s, q=q), checks


class Exact:
    """The exact solution as numbers (SI units). x0 is where the wall was."""

    def __init__(self, gamma, rhoL, pL, rhoR, pR, L, x0, p_star, fan):
        self.gamma, self.rhoL, self.pL, self.rhoR, self.pR, self.L, self.x0 = gamma, rhoL, pL, rhoR, pR, L, x0
        gm = gamma
        self.cL, self.cR = np.sqrt(gm * pL / rhoL), np.sqrt(gm * pR / rhoR)
        self.p_star = p_star
        self.u_star = 2 * self.cL / (gm - 1) * (1 - (p_star / pL) ** ((gm - 1) / (2 * gm)))
        self.rho_starL = rhoL * (p_star / pL) ** (1 / gm)
        P = p_star / pR
        self.rho_starR = rhoR * ((gm + 1) * P + (gm - 1)) / ((gm - 1) * P + (gm + 1))
        self.S = self.cR * np.sqrt(((gm + 1) * P + (gm - 1)) / (2 * gm))
        self.c_starL = self.cL * (p_star / pL) ** ((gm - 1) / (2 * gm))
        self.c_starR = np.sqrt(gm * p_star / self.rho_starR)
        # wave speeds, left to right: fan head, fan tail, contact (the border between the two gases), shock
        self.speeds = dict(head=-self.cL, tail=self.u_star - self.c_starL, contact=self.u_star, shock=self.S)
        self.s_max = self.u_star + self.c_starR              # fastest signal anywhere in the solution
        sub = {g: _rat(gamma), fan["cL"]: self.cL, fan["rhoL"]: rhoL}
        self._anti = {k: sp.lambdify(fan["s"], sp.integrate(v.subs(sub), fan["s"]), "numpy") for k, v in fan["q"].items()}

    def positions(self, t):
        return {k: self.x0 + v * t for k, v in self.speeds.items()}

    def _s(self, x, t):
        gm = self.gamma
        return 2 / (gm + 1) - (gm - 1) / (gm + 1) * (x - self.x0) / (t * self.cL)

    def primitive(self, x, t):
        """Point values (rho, u, p) at positions x and time t > 0."""
        x = np.asarray(x, float)
        gm, w = self.gamma, self.positions(t)
        s = np.clip(self._s(x, t), self.c_starL / self.cL, 1.0)
        rho = np.where(x < w["tail"], self.rhoL * s ** (2 / (gm - 1)),
                       np.where(x < w["contact"], self.rho_starL, np.where(x < w["shock"], self.rho_starR, self.rhoR)))
        u = np.where(x < w["tail"], 2 * self.cL / (gm - 1) * (1 - s), np.where(x < w["shock"], self.u_star, 0.0))
        p = np.where(x < w["tail"], self.pL * s ** (2 * gm / (gm - 1)), np.where(x < w["shock"], self.p_star, self.pR))
        return rho, u, p

    def cumulative(self, x, t):
        """Integrals from 0 to x of the conserved quantities: rho, rho u and E = p/(gamma-1) + rho u^2/2."""
        x = np.asarray(x, float)
        gm, w = self.gamma, self.positions(t)
        E_of = lambda rho, u, p: p / (gm - 1) + rho * u ** 2 / 2
        const = dict(
            L=dict(rho=self.rhoL, rhou=0.0, E=E_of(self.rhoL, 0.0, self.pL)),
            sL=dict(rho=self.rho_starL, rhou=self.rho_starL * self.u_star, E=E_of(self.rho_starL, self.u_star, self.p_star)),
            sR=dict(rho=self.rho_starR, rhou=self.rho_starR * self.u_star, E=E_of(self.rho_starR, self.u_star, self.p_star)),
            R=dict(rho=self.rhoR, rhou=0.0, E=E_of(self.rhoR, 0.0, self.pR)))
        dxds = -t * self.cL * (gm + 1) / (gm - 1)
        s = self._s(np.clip(x, w["head"], w["tail"]), t)
        out = {}
        for k in ("rho", "rhou", "E"):
            out[k] = (const["L"][k] * np.minimum(x, w["head"]) + dxds * (self._anti[k](s) - self._anti[k](1.0))
                      + const["sL"][k] * (np.clip(x, w["tail"], w["contact"]) - w["tail"])
                      + const["sR"][k] * (np.clip(x, w["contact"], w["shock"]) - w["contact"])
                      + const["R"][k] * (np.maximum(x, w["shock"]) - w["shock"]))
        return out

    def cell_averages(self, edges, t):
        """What an exact finite-volume solution holds in each cell: the cell averages of rho, rho u and E,
        and the velocity and pressure that follow from them (as the solver forms them): (rho, u, p)."""
        edges = np.asarray(edges, float)
        if t <= 0:
            xc = 0.5 * (edges[1:] + edges[:-1])
            return (np.where(xc < self.x0, self.rhoL, self.rhoR), np.zeros_like(xc), np.where(xc < self.x0, self.pL, self.pR))
        G = self.cumulative(edges, t)
        h = np.diff(edges)
        rho, rhou, E = (np.diff(G[k]) / h for k in ("rho", "rhou", "E"))
        return rho, rhou / rho, (self.gamma - 1) * (E - 0.5 * rhou ** 2 / rho)

    def totals(self, t):
        G = self.cumulative(np.array([0.0, self.L]), t)
        return {k: float(v[1] - v[0]) for k, v in G.items()}


def solve_star(gamma, rhoL, pL, rhoR, pR):
    """Pressure between the waves: the fan and the shock must leave the gas between them at one speed.
    Solved twice, with SymPy (30 digits) and with SciPy's bracketing solver."""
    p = sp.Symbol("p", positive=True)
    gm, rL, PL, rR, PR = (_rat(v) for v in (gamma, rhoL, pL, rhoR, pR))
    cL = sp.sqrt(gm * PL / rL)
    u_fan = 2 * cL / (gm - 1) * (1 - (p / PL) ** ((gm - 1) / (2 * gm)))
    u_shock = (p - PR) * sp.sqrt(2 / ((gm + 1) * rR) / (p + (gm - 1) / (gm + 1) * PR))
    f = u_fan - u_shock
    p_sympy = float(sp.nsolve(f, p, float(sp.sqrt(PL * PR)), prec=30))
    p_scipy = float(optimize.brentq(sp.lambdify(p, f, "math"), pR, pL, xtol=1e-9, rtol=1e-15))
    return p_sympy, p_scipy


def evaluate(given):
    """(theory block of the report, Exact). given: the `given` block of problem.yaml."""
    gv = {k: v["value"] for k, v in given.items()}
    gamma, rhoL, pL, rhoR, pR, L, R, t_end = (gv[k] for k in ("gamma", "rho_L", "p_L", "rho_R", "p_R", "L", "R", "t_end"))
    x0 = L / 2
    shock, checks_shock = shock_relations()
    fan, checks_fan = fan_relations()
    p_sympy, p_scipy = solve_star(gamma, rhoL, pL, rhoR, pR)
    ex = Exact(gamma, rhoL, pL, rhoR, pR, L, x0, p_sympy, fan)
    w = ex.positions(t_end)
    tot0 = dict(rho=(rhoL + rhoR) * L / 2, rhou=0.0, E=(pL + pR) * L / (2 * (gamma - 1)))
    tot = ex.totals(t_end)
    # the symbolic shock relations, evaluated at the solved pressure, must give the same numbers as Exact
    sub = {g: gamma, shock["rho1"]: rhoR, shock["p1"]: pR, shock["P"]: p_sympy / pR}
    u_from_shock = float(shock["u2"].subs(sub))
    checks = dict(checks_fan)
    checks.update(checks_shock)
    checks.update({
        r"pressure between the waves: SymPy (30 digits) and SciPy (bracketing) agree to \(10^{-12}\) of \(p_L\)":
            abs(p_sympy - p_scipy) < 1e-12 * pL,
        r"wave pattern: \(p_R < p^* < p_L\), so the left wave is an expansion fan and the right wave a shock":
            pR < ex.p_star < pL,
        r"the gas speed behind the shock equals the gas speed behind the fan (one \(u^*\)), to \(10^{-12}\)":
            abs(u_from_shock - ex.u_star) < 1e-12 * ex.u_star,
        r"the fan ends at the left middle state: at its last ray \(\rho = \rho^*_L\), \(u = u^*\), \(p = p^*\)":
            bool(np.allclose(ex.primitive(np.array([w["tail"] - 1e-13]), t_end), [[ex.rho_starL], [ex.u_star], [ex.p_star]],
                             rtol=1e-9)),
        r"the shock compresses and heats: entropy rises, \(p^*/\rho_R^{*\gamma} > p_R/\rho_R^\gamma\)":
            ex.p_star / ex.rho_starR ** gamma > pR / rhoR ** gamma,
        r"whole solution at \(t_{\mathrm{end}}\): total mass equals the starting mass, to \(10^{-12}\)":
            abs(tot["rho"] - tot0["rho"]) < 1e-12 * tot0["rho"],
        r"whole solution at \(t_{\mathrm{end}}\): total energy equals the starting energy, to \(10^{-12}\)":
            abs(tot["E"] - tot0["E"]) < 1e-12 * tot0["E"],
        r"whole solution at \(t_{\mathrm{end}}\): total momentum equals the push of the end walls, \((p_L - p_R)\,t_{\mathrm{end}}\), to \(10^{-12}\)":
            abs(tot["rhou"] - (pL - pR) * t_end) < 1e-12 * (pL - pR) * t_end,
        r"no wave has reached an end of the tube at \(t_{\mathrm{end}}\): fan head at \(x > 0\), shock at \(x < L\)":
            0 < w["head"] and w["shock"] < L,
    })
    bad = [k for k, v in checks.items() if not bool(v)]
    if bad:
        raise AssertionError(f"Exact solution failed a self-check: {bad}")

    # ---- derived quantities, each with its formula, a hand check and what it is for -----------------
    Sy = {k: sp.Symbol(k, positive=True) for k in
          ("rho_L", "p_L", "rho_R", "p_R", "L", "t", "c_L", "c_R", "ps", "us", "rsL", "rsR", "S", "csL", "csR")}
    names = {Sy["rho_L"]: r"\rho_L", Sy["p_L"]: "p_L", Sy["rho_R"]: r"\rho_R", Sy["p_R"]: "p_R", Sy["L"]: "L",
             Sy["t"]: r"t_{\mathrm{end}}", Sy["c_L"]: "c_L", Sy["c_R"]: "c_R", Sy["ps"]: "p^*", Sy["us"]: "u^*",
             Sy["rsL"]: r"\rho^*_L", Sy["rsR"]: r"\rho^*_R", Sy["S"]: "S", Sy["csL"]: "c^*_L", Sy["csR"]: "c^*_R", g: r"\gamma"}
    par = {
        g: dict(tex=r"\gamma", meaning="ratio of specific heats of the air", value=gamma, unit=""),
        Sy["rho_L"]: dict(tex=r"\rho_L", meaning="density left of the wall", value=rhoL, unit=r"\mathrm{kg/m^3}"),
        Sy["p_L"]: dict(tex="p_L", meaning="pressure left of the wall", value=pL, unit=r"\mathrm{Pa}"),
        Sy["rho_R"]: dict(tex=r"\rho_R", meaning="density right of the wall", value=rhoR, unit=r"\mathrm{kg/m^3}"),
        Sy["p_R"]: dict(tex="p_R", meaning="pressure right of the wall", value=pR, unit=r"\mathrm{Pa}"),
        Sy["L"]: dict(tex="L", meaning="length of the tube", value=L, unit=r"\mathrm{m}"),
        Sy["t"]: dict(tex=r"t_{\mathrm{end}}", meaning="time of interest", value=t_end, unit=r"\mathrm{s}"),
        Sy["c_L"]: dict(tex="c_L", meaning="speed of sound left of the wall", value=ex.cL, unit=r"\mathrm{m/s}"),
        Sy["c_R"]: dict(tex="c_R", meaning="speed of sound right of the wall", value=ex.cR, unit=r"\mathrm{m/s}"),
        Sy["ps"]: dict(tex="p^*", meaning="pressure between the waves", value=ex.p_star, unit=r"\mathrm{Pa}"),
        Sy["us"]: dict(tex="u^*", meaning="gas speed between the waves", value=ex.u_star, unit=r"\mathrm{m/s}"),
        Sy["rsL"]: dict(tex=r"\rho^*_L", meaning="density between the fan and the border of the two gases",
                        value=ex.rho_starL, unit=r"\mathrm{kg/m^3}"),
        Sy["rsR"]: dict(tex=r"\rho^*_R", meaning="density between the border of the two gases and the shock",
                        value=ex.rho_starR, unit=r"\mathrm{kg/m^3}"),
        Sy["S"]: dict(tex="S", meaning="speed of the shock", value=ex.S, unit=r"\mathrm{m/s}"),
        Sy["csL"]: dict(tex="c^*_L", meaning="speed of sound behind the fan", value=ex.c_starL, unit=r"\mathrm{m/s}"),
        Sy["csR"]: dict(tex="c^*_R", meaning="speed of sound behind the shock", value=ex.c_starR, unit=r"\mathrm{m/s}"),
    }

    for p_ in par.values():
        p_["value"] = _Shown(p_["value"])

    def entry(key, label, symbol, expr, unit_tex, why, expect, sig=6, prefix=""):
        used = [par[s] for s in par if s in expr.free_symbols]
        num = {s: (rf"\left({tex_num(par[s]['value'], sig)}\right)" if par[s]["unit"] else tex_num(par[s]["value"], sig))
               for s in par if s in expr.free_symbols}
        value = float(expr.subs({s: par[s]["value"] for s in par}).evalf(30))
        if abs(value - expect) > 1e-9 * max(abs(expect), 1e-300):
            raise AssertionError(f"theory entry {key}: formula gives {value}, the solution object has {expect}")
        return dict(key=key, label=label, symbol=symbol, value=value, value_tex=tex_num(value, sig), unit_tex=unit_tex,
                    formula=rf"{symbol} = {prefix}{sp.latex(expr, symbol_names=names)}",
                    check=rf"\begin{{aligned}}{symbol} &= {sp.latex(expr, symbol_names=num)}\\ &= {tex_num(value, sig)}\ {unit_tex}"
                          r"\end{aligned}", why=why, params=used)

    y = Sy
    Pr = y["ps"] / y["p_R"]
    half = sp.Rational(1, 2)
    derived = [
        entry("c_L", "Speed of sound left of the wall", "c_L", sp.sqrt(g * y["p_L"] / y["rho_L"]), r"\mathrm{m/s}",
              "Small pressure changes travel through a gas at the speed of sound. The front of the expansion fan runs "
              "into the still gas on the left at exactly this speed.", ex.cL),
        entry("c_R", "Speed of sound right of the wall", "c_R", sp.sqrt(g * y["p_R"] / y["rho_R"]), r"\mathrm{m/s}",
              "The shock must move faster than this, or the gas ahead would be warned of it.", ex.cR),
        dict(key="p_star", label="Pressure between the waves", symbol="p^*", value=ex.p_star,
             value_tex=tex_num(ex.p_star, 7), unit_tex=r"\mathrm{Pa}",
             formula=r"\begin{aligned}&\frac{2c_L}{\gamma-1}\left[1-\left(\frac{p^*}{p_L}\right)^{\frac{\gamma-1}{2\gamma}}\right]"
                     r" = (p^*-p_R)\sqrt{\frac{A}{p^*+B}},\\[4pt] &A = \frac{2}{(\gamma+1)\rho_R},\quad B = \frac{\gamma-1}{\gamma+1}\,p_R"
                     r"\end{aligned}",
             check=rf"\begin{{aligned}}\text{{left side}} &= {tex_num(ex.u_star, 9)}\ \mathrm{{m/s}}\\"
                   rf"\text{{right side}} &= {tex_num(u_from_shock, 9)}\ \mathrm{{m/s}}\end{{aligned}}",
             why="The fan lowers the pressure of the left gas and pushes it to the right; the shock raises the pressure of "
                 "the right gas and pushes it to the right too. Between them the pressure and the gas speed must be the "
                 "same on both sides of the border between the two gases. This one equation fixes that pressure; it has "
                 "no closed-form solution and is solved numerically, twice. Requested output (b).",
             params=[par[s] for s in (g, y["c_L"], y["p_L"], y["p_R"], y["rho_R"])]),
        entry("u_star", "Gas speed between the waves", "u^*",
              2 * y["c_L"] / (g - 1) * (1 - (y["ps"] / y["p_L"]) ** ((g - 1) / (2 * g))), r"\mathrm{m/s}",
              "The speed of all the gas between the fan and the shock, and so the speed at which the border between the "
              "two gases moves. Requested outputs (b) and (c).", ex.u_star, sig=7),
        entry("rho_starL", "Density between the fan and the border", r"\rho^*_L",
              y["rho_L"] * (y["ps"] / y["p_L"]) ** (1 / g), r"\mathrm{kg/m^3}",
              "The left gas has expanded through the fan without friction or heat, so its pressure and density stay linked "
              r"by \(p/\rho^\gamma = \text{constant}\). Requested output (b).", ex.rho_starL, sig=7),
        entry("rho_starR", "Density between the border and the shock", r"\rho^*_R",
              y["rho_R"] * ((g + 1) * Pr + (g - 1)) / ((g - 1) * Pr + (g + 1)), r"\mathrm{kg/m^3}",
              "The right gas has been compressed by the shock. A shock heats the gas more than a gentle compression "
              "would, so this density differs from the left gas's at the same pressure: the two gases meet at a jump in "
              "density (and temperature) that moves with the flow. Requested output (b).", ex.rho_starR, sig=7),
        entry("S", "Speed of the shock", "S", y["c_R"] * sp.sqrt(((g + 1) * Pr + (g - 1)) / (2 * g)), r"\mathrm{m/s}",
              "From the conservation of mass, momentum and energy across the jump. Requested output (c).", ex.S, sig=7),
        entry("Ms", "Mach number of the shock", r"M_s", y["S"] / y["c_R"], "",
              "The shock's speed compared with the speed of sound in the gas it runs into: above 1 by definition, and a "
              "measure of its strength.", ex.S / ex.cR),
        entry("tail", "Speed of the last ray of the fan", r"u^* - c^*_L",
              y["us"] - y["c_L"] * (y["ps"] / y["p_L"]) ** ((g - 1) / (2 * g)), r"\mathrm{m/s}",
              "The back of the fan is a sound wave running to the left through gas that itself moves to the right at "
              r"\(u^*\). Here the two nearly cancel, so the back of the fan barely moves. Requested output (c).",
              ex.speeds["tail"]),
        entry("x_head", "Front of the fan at the time of interest", r"x_{\mathrm{head}}", y["L"] / 2 - y["c_L"] * y["t"],
              r"\mathrm{m}", r"Every wave starts at the wall, \(x = L/2\), and moves at a constant speed. Requested output (c).",
              w["head"]),
        entry("x_tail", "Back of the fan at the time of interest", r"x_{\mathrm{tail}}",
              y["L"] / 2 + (y["us"] - y["csL"]) * y["t"], r"\mathrm{m}", "Requested output (c).", w["tail"]),
        entry("x_contact", "Border between the two gases at the time of interest", r"x_{\mathrm{contact}}",
              y["L"] / 2 + y["us"] * y["t"], r"\mathrm{m}",
              "It moves with the gas. In the literature this border is called the contact discontinuity. Requested output (c).",
              w["contact"]),
        entry("x_shock", "Shock at the time of interest", r"x_{\mathrm{shock}}", y["L"] / 2 + y["S"] * y["t"], r"\mathrm{m}",
              r"It is still \(" + tex_num(L - w["shock"], 3) + r"\,\mathrm{m}\) from the right end, so the exact solution "
              "holds at the time of interest. Requested output (c).", w["shock"]),
        entry("mass", "Mass of the air in the tube, per unit of cross-section", "M", (y["rho_L"] + y["rho_R"]) * y["L"] / 2,
              r"\mathrm{kg/m^2}", "The ends are closed, so this cannot change. Requested output (d).", tot0["rho"]),
        entry("energy", "Energy of the air in the tube, per unit of cross-section", r"E_{\mathrm{tot}}",
              (y["p_L"] + y["p_R"]) * y["L"] / (2 * (g - 1)), r"\mathrm{J/m^2}",
              r"Internal energy plus kinetic energy, \(\int (p/(\gamma-1) + \tfrac12\rho u^2)\,dx\). The air starts at rest, "
              "so at the start it is all internal energy. No heat or work crosses the walls, so it cannot change. "
              "Requested output (d).", tot0["E"]),
        entry("s_max", "Fastest signal in the solution", r"u^* + c^*_R", y["us"] + sp.sqrt(g * y["ps"] / y["rsR"]),
              r"\mathrm{m/s}",
              "Sound running to the right through the shocked gas, which itself moves to the right. An explicit solver "
              "must not let this signal cross a whole cell in one time step, so it sets the time step (see Decisions).",
              ex.s_max),
    ]
    steps = [
        dict(label="Governing equations: mass, momentum and energy of a gas without friction or heat conduction, in one dimension",
             latex=r"\frac{\partial}{\partial t}\begin{pmatrix}\rho\\ \rho u\\ E\end{pmatrix} + \frac{\partial}{\partial x}"
                   r"\begin{pmatrix}\rho u\\ \rho u^2 + p\\ u(E+p)\end{pmatrix} = 0,\qquad E = \frac{p}{\gamma-1} + \frac12\rho u^2",
             note=r"These are the Euler equations. \(E\) is the energy per unit volume. Written like this, each line says: "
                  "what a slice of the tube holds changes only by what flows in and out through its two faces. That stays "
                  "true across a jump, where derivatives do not exist."),
        dict(label="No length or time scale: the solution depends on \\((x - x_0)/t\\) only",
             latex=r"\rho,\ u,\ p = \text{functions of }\ \xi = \frac{x - x_0}{t},\qquad x_0 = \frac{L}{2}",
             note="Two uniform states meeting at a point have no length scale, so the picture at a later time is the "
                  "earlier one stretched. Every wave leaves the wall position at a constant speed. Three waves form: an "
                  "expansion fan to the left, a shock to the right and, between them, the border of the two gases."),
        dict(label="Left: the expansion fan (a smooth drop in pressure and density)",
             latex=r"s = \frac{2}{\gamma+1} - \frac{\gamma-1}{\gamma+1}\,\frac{\xi}{c_L},\qquad \rho = \rho_L\,s^{\frac{2}{\gamma-1}},\quad "
                   r"u = \frac{2}{\gamma+1}\,(c_L + \xi),\quad p = p_L\,s^{\frac{2\gamma}{\gamma-1}}\qquad (-c_L \le \xi \le u^* - c^*_L)",
             note=r"\(s\) is the local speed of sound divided by \(c_L\). The velocity rises linearly across the fan. "
                  "SymPy checks below that these three functions satisfy the three equations above."),
        dict(label="Right: the shock (a jump), from conservation across it",
             latex=r"\frac{\rho^*_R}{\rho_R} = \frac{(\gamma+1)\,p^*/p_R + (\gamma-1)}{(\gamma-1)\,p^*/p_R + (\gamma+1)},\qquad "
                   r"S = c_R\sqrt{\frac{(\gamma+1)\,p^*/p_R + (\gamma-1)}{2\gamma}},\qquad u^* = S\left(1 - \frac{\rho_R}{\rho^*_R}\right)",
             note="These are the Rankine-Hugoniot relations for a shock moving into gas at rest. SymPy checks below that "
                  "mass, momentum and energy flowing into the jump equal what flows out."),
        dict(label="Between them: one pressure and one speed; the density jumps at the border of the two gases",
             latex=r"p = p^*,\quad u = u^*\quad (x_{\mathrm{tail}} < x < x_{\mathrm{shock}}),\qquad "
                   r"\rho = \rho^*_L\ \ (x < x_{\mathrm{contact}}),\quad \rho = \rho^*_R\ \ (x > x_{\mathrm{contact}})",
             note=r"\(p^*\) follows from requiring that the fan and the shock give the same \(u^*\) (first card below)."),
    ]
    theory = dict(
        intro="This problem has an exact solution, so the simulation can be checked against it everywhere, not only "
              "at a few numbers. The solution is put together from three pieces, each proven by SymPy below, and one "
              "equation for the pressure between the waves, solved numerically in two independent ways. The assembled "
              "solution is checked once more as a whole: its mass, momentum and energy at the time of interest equal "
              "what the tube started with. One thing theory also says about the simulation: a method that captures "
              "jumps over a few cells cannot reach the accuracy it has in smooth flow, so the error should fall "
              "roughly in proportion to the cell size, not to its square (see Comparison).",
        steps=steps, derived=derived, self_checks=[dict(name=k, ok=bool(v)) for k, v in checks.items()])
    return theory, ex


if __name__ == "__main__":
    from pathlib import Path
    import yaml
    prob = yaml.safe_load((Path(__file__).parent / "problem.yaml").read_text(encoding="utf-8"))
    th, ex = evaluate(prob["given"])
    for c in th["self_checks"]:
        print("ok  " if c["ok"] else "FAIL", c["name"])
    for d in th["derived"]:
        print(f"{d['key']:10s} {d['value']:.10g}")
    print("positions", ex.positions(prob["given"]["t_end"]["value"]))
    print("totals", ex.totals(prob["given"]["t_end"]["value"]))
