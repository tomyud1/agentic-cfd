"""Analytical solution of plane Poiseuille flow, derived symbolically with SymPy.

The derivation is solved and self-checked by SymPy (not by the agent's reasoning). Each step
is exported as LaTeX for the report, and the formulas are evaluated with the given data.
Every derived quantity also carries a hand-check: the formula with the given numbers substituted.
"""
import sympy as sp

from cfdlib.report import tex_num

y = sp.Symbol("y", real=True)
H, mu, G = sp.symbols("H mu G", positive=True)   # G = -dp/dx > 0 drives flow in +x
u = sp.Function("u")


def derive():
    # x-momentum for steady, fully developed, incompressible, Newtonian flow:
    #   0 = -dp/dx + mu d2u/dy2   ->   mu u'' = -G
    ode = sp.Eq(mu * u(y).diff(y, 2), -G)
    bcs = {u(0): 0, u(H): 0}                       # no slip on both plates
    u_expr = sp.factor(sp.dsolve(ode, u(y), ics=bcs).rhs)

    # Self-checks: the solution must satisfy the ODE and both boundary conditions exactly.
    checks = {
        r"satisfies the momentum equation \(\mu\, u'' = -G\)": sp.simplify(mu * u_expr.diff(y, 2) + G) == 0,
        r"satisfies \(u(0) = 0\)": sp.simplify(u_expr.subs(y, 0)) == 0,
        r"satisfies \(u(H) = 0\)": sp.simplify(u_expr.subs(y, H)) == 0,
    }
    if not all(checks.values()):
        raise AssertionError(f"Analytical solution failed self-check: {checks}")

    # Readable form for display, checked to be identical to SymPy's own result.
    u_display = r"\frac{G}{2\mu}\, y\,(H - y)"
    if sp.simplify(G / (2 * mu) * y * (H - y) - u_expr) != 0:
        raise AssertionError("Display form of u(y) differs from the derived solution")
    checks[r"display form \(" + u_display + r"\) equals the derived solution"] = True

    u_max = sp.simplify(u_expr.subs(y, H / 2))
    q = sp.simplify(sp.integrate(u_expr, (y, 0, H)))
    u_bulk = sp.simplify(q / H)
    tau_w = sp.simplify(mu * u_expr.diff(y).subs(y, 0))
    return dict(ode=ode, u=u_expr, u_display=u_display, u_max=u_max, q=q, u_bulk=u_bulk, tau_w=tau_w, checks=checks)


def evaluate(given):
    """Derivation steps, derived quantities (with hand-checks) and a numeric u(y) function."""
    d = derive()
    rho, nu, Hv = given["rho"]["value"], given["nu"]["value"], given["H"]["value"]
    muv, Gv = rho * nu, -given["dpdx"]["value"]
    subs = {H: Hv, mu: muv, G: Gv}
    num = lambda e: float(e.subs(subs))
    tex = sp.latex

    # What each symbol means in THIS problem, and its value (for the hand-check panels).
    params = {
        G: dict(tex="G", meaning=r"driving pressure gradient, \(G = -\,dp/dx\) (pressure falls along the flow)",
                value=Gv, unit=r"\mathrm{Pa/m}"),
        H: dict(tex="H", meaning="gap between the plates", value=Hv, unit=r"\mathrm{m}"),
        mu: dict(tex=r"\mu", meaning=rf"dynamic viscosity of water, \(\mu = \rho\nu = {tex_num(rho)} \times {tex_num(nu)}\)",
                 value=muv, unit=r"\mathrm{Pa\,s}"),
    }
    plugged = {s: rf"\left({tex_num(p['value'])}\,{p['unit']}\right)" for s, p in params.items()}

    def entry(key, label, symbol, expr, unit_tex, unit, why, prefix=""):
        """One derived quantity: formula, value, and the formula with numbers substituted."""
        value = num(expr)
        used = [params[s] for s in (G, H, mu) if s in expr.free_symbols]
        return dict(
            key=key, label=label, symbol=symbol, value=value, value_tex=tex_num(value), unit=unit, unit_tex=unit_tex,
            formula=rf"{symbol} = {prefix}{tex(expr)}",
            check=rf"{symbol} = {tex(expr, symbol_names=plugged)} = {tex_num(value)}\,{unit_tex}",
            why=why, params=used,
        )

    steps = [
        dict(label="Governing equation: x-momentum for steady, fully developed flow",
             latex=r"0 = -\frac{dp}{dx} + \mu \frac{d^2u}{dy^2}"
                   r"\quad\Longrightarrow\quad " + tex(d["ode"]) + r",\qquad G \equiv -\frac{dp}{dx}",
             note="The pressure force pushing the fluid forward is balanced by viscous friction between layers."),
        dict(label="Boundary conditions: no slip at both plates", latex=r"u(0) = 0, \qquad u(H) = 0",
             note="Fluid touching a stationary wall does not move."),
        dict(label="Solution: integrate twice, apply both conditions", latex=r"u(y) = " + d["u_display"],
             note="A parabola: zero at the walls, fastest in the middle."),
    ]
    derived = [
        entry("u_max", "Maximum velocity", r"u_{\max}", d["u_max"], r"\mathrm{m/s}", "m/s",
              "The fastest fluid, on the centreline (y = H/2). It is what a velocity probe in the middle of the gap would read.",
              prefix=r"u\!\left(\tfrac{H}{2}\right) = "),
        entry("q", "Flow rate per unit width", "q", d["q"], r"\mathrm{m^2/s}", "m²/s",
              "Volume of fluid passing per second, per metre of plate width. This is what a pump must deliver, "
              "and in design it is usually the requirement.",
              prefix=r"\int_0^H u\,dy = "),
        entry("u_bulk", "Mean (bulk) velocity", r"\bar{u}", d["u_bulk"], r"\mathrm{m/s}", "m/s",
              r"Average velocity over the gap, \(\bar u = q/H\). It enters the Reynolds number and pressure-drop "
              r"correlations. For this parabola \(\bar u = \tfrac{2}{3}u_{\max}\).",
              prefix=r"\frac{q}{H} = "),
        entry("tau_w", "Wall shear stress", r"\tau_w", d["tau_w"], r"\mathrm{Pa}", "Pa",
              r"Friction stress the fluid exerts on each plate; times the plate area it gives the drag force. "
              r"It must balance the pressure force: \(2\tau_w = G H\) (two walls, per unit area).",
              prefix=r"\mu \left.\frac{du}{dy}\right|_{y=0} = "),
    ]
    u_func = sp.lambdify(y, d["u"].subs(subs), "numpy")
    return dict(
        steps=steps, derived=derived,
        self_checks=d["checks"], mu=muv, G=Gv,
        u_formula=r"u(y) = " + d["u_display"],
    ), u_func
