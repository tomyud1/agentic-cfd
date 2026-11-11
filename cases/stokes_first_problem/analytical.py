"""Exact solution of Stokes' first problem (impulsively started plate), derived with SymPy.

SymPy reduces the PDE to an ODE with the similarity variable, solves it, and self-checks the result
(PDE, wall condition, far-field and initial condition). The requested quantities (wall shear stress,
1 % penetration depth, momentum) are derived from it symbolically and evaluated with the given data;
each carries a hand-check with the numbers substituted.
"""
import numpy as np
import sympy as sp
from scipy import special

from cfdlib.report import tex_num

y, t = sp.symbols("y t", positive=True)
U, nu, rho = sp.symbols("U nu rho", positive=True)
eta = sp.Symbol("eta", positive=True)
f = sp.Function("f")
FRACTION = sp.Rational(1, 100)           # "u has fallen to 1 % of U"


def derive():
    checks = {}
    # Governing equation for u = u(y, t), v = 0: du/dt = nu d2u/dy2 (pressure gradient zero, convection
    # u du/dx + v du/dy vanishes identically). Similarity: u = U f(eta), eta = y / (2 sqrt(nu t)).
    # Chain rule with F1 = f'(eta), F2 = f''(eta): u_t = U F1 eta_t, u_yy = U F2 eta_y^2 (eta_yy = 0).
    eta_of = y / (2 * sp.sqrt(nu * t))
    F1, F2 = sp.symbols("F1 F2")
    pde = U * F1 * sp.diff(eta_of, t) - nu * U * F2 * sp.diff(eta_of, y) ** 2
    # Multiply by t/U and write y through eta: must leave an ODE in eta only, -(F2 + 2 eta F1)/4.
    ode_expr = sp.simplify((pde * t / U).subs(y, 2 * eta * sp.sqrt(nu * t)))
    checks[r"with \(\eta = y/(2\sqrt{\nu t})\) the PDE becomes the ODE \(f'' + 2\eta f' = 0\) (no \(t\) left)"] = \
        sp.simplify(ode_expr + (F2 + 2 * eta * F1) / 4) == 0 and sp.diff(eta_of, y, 2) == 0

    ode = sp.Eq(f(eta).diff(eta, 2) + 2 * eta * f(eta).diff(eta), 0)
    gen = sp.dsolve(ode, f(eta)).rhs                    # C1 + C2 erf(eta)
    C = sorted(gen.free_symbols - {eta}, key=str)
    sol = sp.solve([sp.Eq(gen.subs(eta, 0), 1), sp.Eq(sp.limit(gen, eta, sp.oo), 0)], C, dict=True)[0]
    f_expr = sp.simplify(gen.subs(sol))
    checks[r"SymPy's solution with \(f(0) = 1,\ f(\infty) = 0\) is \(f = \operatorname{erfc}\eta\)"] = \
        sp.simplify((f_expr - sp.erfc(eta)).rewrite(sp.erf)) == 0

    u_expr = U * sp.erfc(y / (2 * sp.sqrt(nu * t)))
    checks[r"\(u = U\operatorname{erfc}\big(y/2\sqrt{\nu t}\big)\) satisfies \(\partial u/\partial t = \nu\,\partial^2 u/\partial y^2\)"] = \
        sp.simplify(sp.diff(u_expr, t) - nu * sp.diff(u_expr, y, 2)) == 0
    checks[r"\(u(0, t) = U\) (no slip on the moving plate)"] = sp.simplify(u_expr.subs(y, 0) - U) == 0
    checks[r"\(u \to 0\) as \(y \to \infty\) (water at rest far away)"] = sp.limit(u_expr, y, sp.oo) == 0
    checks[r"\(u \to 0\) as \(t \to 0^+\) for every \(y > 0\) (water initially at rest)"] = sp.limit(u_expr, t, 0, "+") == 0

    mu = rho * nu
    tau_w = sp.simplify(mu * sp.diff(u_expr, y).subs(y, 0))         # = -mu U / sqrt(pi nu t)
    tau_mag = sp.simplify(-tau_w)
    P = sp.simplify(rho * sp.integrate(u_expr, (y, 0, sp.oo)))       # momentum per unit plate area
    checks[r"momentum balance \(dP/dt = -\tau_w\): the plate's drag is the only force on the water"] = \
        sp.simplify(sp.diff(P, t) - tau_mag) == 0
    tt = sp.Symbol("tt", positive=True)
    checks[r"\(\int_0^t |\tau_w|\,dt' = P(t)\) (impulse equals momentum)"] = \
        sp.simplify(sp.integrate(tau_mag.subs(t, tt), (tt, 0, t)) - P) == 0
    # Curvature at the wall vanishes (u_t = 0 there), so a two-point wall gradient is second order.
    u_yy0 = sp.simplify(sp.diff(u_expr, y, 2).subs(y, 0))
    ratio3 = sp.simplify(sp.diff(u_expr, y, 3).subs(y, 0) / sp.diff(u_expr, y).subs(y, 0))
    checks[r"\(\partial^2 u/\partial y^2 = 0\) at the plate"] = u_yy0 == 0

    # 1 % penetration depth: erfc(eta1) = 1/100, solved numerically twice (SymPy and SciPy) and compared.
    eta1_sympy = float(sp.nsolve(sp.erfc(eta) - FRACTION, eta, 1.8, prec=30))
    eta1_scipy = float(special.erfcinv(float(FRACTION)))
    checks[r"\(\eta_1\) from SymPy (nsolve) and SciPy (erfcinv) agree to \(10^{-12}\)"] = abs(eta1_sympy - eta1_scipy) < 1e-12
    if not all(checks.values()):
        raise AssertionError(f"Exact solution failed a self-check: {checks}")
    return dict(u=u_expr, tau_w=tau_w, tau_mag=tau_mag, P=P, eta1=eta1_sympy, ratio3=ratio3, checks=checks)


def evaluate(given, H_domain):
    """Steps, derived quantities (with hand-checks), self-checks, and numeric functions of the exact solution."""
    d = derive()
    Uv, nuv, rhov, tend = (given[k]["value"] for k in ("U", "nu", "rho", "t_end"))
    tex = sp.latex
    E1 = sp.Symbol(r"eta_1", positive=True)
    Hs = sp.Symbol("H", positive=True)
    params = {
        U: dict(tex="U", meaning="plate speed", value=Uv, unit=r"\mathrm{m/s}"),
        nu: dict(tex=r"\nu", meaning="kinematic viscosity of water", value=nuv, unit=r"\mathrm{m^2/s}"),
        rho: dict(tex=r"\rho", meaning="density of water", value=rhov, unit=r"\mathrm{kg/m^3}"),
        t: dict(tex="t", meaning="time since the plate started", value=tend, unit=r"\mathrm{s}"),
        E1: dict(tex=r"\eta_1", meaning=r"similarity coordinate where \(\operatorname{erfc}\eta_1 = 0.01\) (solved numerically)",
                 value=d["eta1"], unit=""),
        Hs: dict(tex="H", meaning="depth of the simulated water layer (model choice, see Decisions)", value=H_domain,
                 unit=r"\mathrm{m}"),
    }

    def entry(key, label, symbol, expr, unit_tex, why, tval=tend, prefix=""):
        sub = {U: Uv, nu: nuv, rho: rhov, t: tval, E1: d["eta1"], Hs: H_domain}
        names = {}
        used = []
        for s, p in params.items():
            if s in expr.free_symbols:
                pv = dict(p, value=tval) if s is t else p
                used.append(pv)
                # Numbers only (SI units, listed with each symbol in the table above): keeps the check narrow
                # enough for the half-width card.
                names[s] = rf"\left({tex_num(pv['value'])}\right)" if pv["unit"] else tex_num(pv["value"])
        value = float(expr.subs(sub).evalf(30))
        return dict(key=key, label=label, symbol=symbol, value=value, value_tex=tex_num(value), unit_tex=unit_tex,
                    formula=rf"{symbol} = {prefix}{tex(expr, symbol_names={E1: r'\eta_1', Hs: 'H'})}",
                    check=rf"\begin{{aligned}}{symbol} &= {tex(expr, symbol_names=names)}\\ &= {tex_num(value)}\ {unit_tex}"
                          r"\end{aligned}",
                    why=why, params=used)

    delta = 2 * E1 * sp.sqrt(nu * t)
    derived = [
        entry("tau_w", r"Wall shear stress at \(t = t_{\mathrm{end}}\)", r"|\tau_w|", d["tau_mag"], r"\mathrm{Pa}",
              r"Friction stress of the water on the plate (it acts against the motion, in \(-x\)). It is what keeps "
              r"pulling the plate back; it falls like \(1/\sqrt{t}\) because the velocity gradient at the plate "
              r"weakens as the moving layer thickens. Requested output (b).",
              prefix=r"\mu\left|\frac{\partial u}{\partial y}\right|_{y=0} = "),
        entry("delta_1", r"1 % penetration depth at \(t = 1\ \mathrm{s}\)", r"\delta_{1\%}", delta, r"\mathrm{m}",
              r"How far the motion has reached: the height where \(u = 0.01\,U\). It grows like \(\sqrt{\nu t}\), the "
              r"diffusion length. Requested output (c).", tval=1.0),
        entry("delta_5", r"1 % penetration depth at \(t = 5\ \mathrm{s}\)", r"\delta_{1\%}", delta, r"\mathrm{m}",
              "Same formula, five times later: deeper by a factor " + r"\(\sqrt{5}\)" + ".", tval=5.0),
        entry("delta_20", r"1 % penetration depth at \(t = 20\ \mathrm{s}\)", r"\delta_{1\%}", delta, r"\mathrm{m}",
              r"At the end of the period of interest. It sets how deep the simulated water layer must be.", tval=tend),
        entry("P", r"x-momentum per unit plate area at \(t = t_{\mathrm{end}}\)", "P", d["P"], r"\mathrm{kg/(m\,s)}",
              r"Total momentum the plate has given to the water, \(\rho\int_0^\infty u\,dy\); equal to the time "
              r"integral of the wall shear stress (the impulse of the drag, in \(\mathrm{N\,s/m^2}\)). Requested output (d).",
              prefix=r"\rho\int_0^\infty u\,dy = "),
        entry("far", r"Velocity left at the top of the simulated layer at \(t_{\mathrm{end}}\)", r"u(H)/U",
              sp.erfc(Hs / (2 * sp.sqrt(nu * t))), "",
              r"How much of the motion the exact solution puts at the depth where the simulation stops. It must be "
              r"negligible for the finite layer to stand in for the very deep water of the problem.", tval=tend),
    ]
    # Re based on the 1 % depth at t_end: the convection term is identically zero, so Re does not enter
    # the equations, but it indicates whether the laminar solution could be disturbed (see Assumptions).
    Re_delta = Uv * derived[3]["value"] / nuv
    steps = [
        dict(label="Governing equation: x-momentum for a flow that depends only on y and t",
             latex=r"\frac{\partial u}{\partial t} = \nu\,\frac{\partial^2 u}{\partial y^2},\qquad v = 0,\ \ \frac{\partial p}{\partial x} = 0",
             note=r"With \(u = u(y, t)\) and \(v = 0\), continuity holds automatically and the convection terms "
                  r"\(u\,\partial u/\partial x + v\,\partial u/\partial y\) are zero: the Navier–Stokes equations reduce exactly to a diffusion equation. "
                  "Momentum spreads from the plate into the water by viscosity, like heat from a suddenly heated wall."),
        dict(label="Conditions: plate moves from t = 0, water initially and far away at rest",
             latex=r"u(0, t) = U\ (t > 0),\qquad u(y, 0) = 0,\qquad u(y \to \infty, t) = 0",
             note=r"The plate jumps from rest to \(U\), so at \(t = 0\) the velocity is discontinuous at the plate."),
        dict(label=r"Similarity: no length scale is given, so the profile depends only on \(y/\sqrt{\nu t}\)",
             latex=r"u = U f(\eta),\quad \eta = \frac{y}{2\sqrt{\nu t}}\quad\Longrightarrow\quad f'' + 2\eta f' = 0,\ \ f(0) = 1,\ f(\infty) = 0",
             note=r"Profiles at different times collapse onto one curve when plotted against \(\eta\) (checked by SymPy below)."),
        dict(label="Solution",
             latex=r"u(y, t) = U\,\operatorname{erfc}\!\left(\frac{y}{2\sqrt{\nu t}}\right),\qquad "
                   r"\operatorname{erfc}\eta = \frac{2}{\sqrt{\pi}}\int_\eta^\infty e^{-s^2}\,ds",
             note=r"\(\operatorname{erfc}\) is the complementary error function: 1 at the plate, falling to 0 within a few units of \(\eta\)."),
        dict(label="Requested quantities",
             latex=r"\tau_w = \mu\left.\frac{\partial u}{\partial y}\right|_{0} = -\frac{\mu U}{\sqrt{\pi\nu t}},\quad "
                   r"\delta_{1\%} = 2\eta_1\sqrt{\nu t},\ \operatorname{erfc}\eta_1 = 0.01,\quad "
                   r"P = \rho\int_0^\infty u\,dy = 2\rho U\sqrt{\frac{\nu t}{\pi}} = \int_0^t |\tau_w|\,dt'",
             note=r"The wall shear stress is infinite at \(t = 0\) (the jump) and decays like \(1/\sqrt{t}\); its time integral is finite "
                  r"and equals the momentum in the water. \(\tau_w < 0\): the water pulls the plate back (\(\mu = \rho\nu\))."),
        dict(label="Consequence for the numerics: the wall gradient",
             latex=r"\left.\frac{\partial^2 u}{\partial y^2}\right|_{0} = 0\ \Rightarrow\ "
                   r"\frac{U - u(h/2)}{h/2} = -\left.\frac{\partial u}{\partial y}\right|_0\left(1 - \frac{h^2}{48\,\nu t}\right) + O(h^4)",
             note=r"Because \(u\) stays equal to \(U\) at the plate, its curvature there is zero, so the two-point wall gradient "
                  r"a finite-volume solver uses (plate to first cell centre, distance \(h/2\)) is second-order accurate. "
                  r"Its own error is small once \(\sqrt{\nu t}\) spans many cells; at early times (\(\sqrt{\nu t} \sim h\)) it is not. "
                  r"The cell values themselves also carry an error of the same order, so this term is a scale, not the whole error."),
    ]
    if sp.simplify(d["ratio3"] + 1 / (2 * nu * t)) != 0:
        raise AssertionError("u_yyy/u_y at the wall is not -1/(2 nu t)")
    d["checks"][r"\(u_{yyy}/u_y = -1/(2\nu t)\) at the plate (gives the \(h^2/(48\nu t)\) term)"] = True

    subs = {U: Uv, nu: nuv, rho: rhov}
    u_num = sp.lambdify((y, t), d["u"].subs(subs), modules=[{"erfc": special.erfc}, "numpy"])
    tau_num = sp.lambdify(t, d["tau_mag"].subs(subs), "numpy")
    P_num = sp.lambdify(t, d["P"].subs(subs), "numpy")
    eta1 = d["eta1"]
    return dict(
        steps=steps, derived=derived, self_checks=d["checks"], eta1=eta1, mu=rhov * nuv, Re_delta=Re_delta,
        u_formula=r"u = U\operatorname{erfc}\big(y/2\sqrt{\nu t}\big)",
    ), dict(u=lambda yy, tv: np.asarray(u_num(np.asarray(yy, float), tv), float),
            tau=lambda tv: np.asarray(tau_num(np.asarray(tv, float)), float),
            P=lambda tv: np.asarray(P_num(np.asarray(tv, float)), float),
            delta=lambda tv: 2 * eta1 * np.sqrt(nuv * np.asarray(tv, float)))
