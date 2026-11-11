"""What theory can say about the lid-driven cavity without a simulation (no exact solution exists).

SymPy does the algebra and self-checks it:
  * scaling: the dimensionless momentum equation has a single parameter, 1/Re;
  * the stream function satisfies continuity identically, and mass conservation forces
    the net flow through any vertical line to vanish (psi = 0 on every wall);
  * derived numbers (Re, time scales, cell Peclet numbers, benchmark grid spacing), each with
    its formula and a hand-check with the numbers substituted.
"""
import sympy as sp

from cfdlib.report import tex_num

# Dimensional symbols
L, U, nu, h = sp.symbols("L U nu h", positive=True)
# Dimensionless coordinates and fields: X = x/L, Y = y/L, f = u/U, g = v/U, P = p/(rho U^2)
X, Y = sp.symbols("X Y", real=True)
f, g, P = (sp.Function(n)(X, Y) for n in ("f", "g", "P"))
psi = sp.Function("psi")


def derive():
    checks = {}
    # x-momentum, written in dimensional form with x = L X, u = U f, p/rho = U^2 P:
    #   u du/dx + v du/dy = -d(p/rho)/dx + nu (d2u/dx2 + d2u/dy2)
    dx = lambda e: sp.diff(e, X) / L
    dy = lambda e: sp.diff(e, Y) / L
    u, v, pk = U * f, U * g, U ** 2 * P
    lhs = u * dx(u) + v * dy(u)
    rhs = -dx(pk) + nu * (dx(dx(u)) + dy(dy(u)))
    # Divide by the convective scale U^2/L: the only parameter left must be nu/(U L) = 1/Re.
    scaled = sp.expand((lhs - rhs) / (U ** 2 / L))
    Re = sp.Symbol("Re", positive=True)
    target = (f * sp.diff(f, X) + g * sp.diff(f, Y) + sp.diff(P, X)
              - (sp.diff(f, X, 2) + sp.diff(f, Y, 2)) / Re)
    checks[r"dimensionless momentum equation contains only \(1/Re\), with \(Re = UL/\nu\)"] = \
        sp.simplify(scaled - target.subs(Re, U * L / nu)) == 0

    # Stream function: u = d psi/dy, v = -d psi/dx satisfies continuity identically.
    x, y = sp.symbols("x y", real=True)
    ps = psi(x, y)
    checks[r"\(u = \partial\psi/\partial y,\ v = -\partial\psi/\partial x\) satisfies \(\nabla\cdot\mathbf{u} = 0\)"] = \
        sp.simplify(sp.diff(sp.diff(ps, y), x) + sp.diff(-sp.diff(ps, x), y)) == 0
    # Net flow through a vertical line x = const: integral of u dy = psi(x, L) - psi(x, 0) = 0 - 0.
    net = sp.integrate(sp.diff(ps, y), (y, 0, L))
    checks[r"\(\int_0^L u\,dy = \psi(x, L) - \psi(x, 0)\), which is 0 because \(\psi = 0\) on every wall"] = \
        sp.simplify(net - (ps.subs(y, L) - ps.subs(y, 0))) == 0
    if not all(checks.values()):
        raise AssertionError(f"Theory self-check failed: {checks}")
    return checks


def evaluate(given, meshes, h_ghia):
    """Derivation steps, derived quantities with hand-checks, self-checks."""
    checks = derive()
    Lv, Uv, nuv = (given[k]["value"] for k in ("L", "U", "nu"))
    tex = sp.latex
    params = {
        U: dict(tex="U", meaning="lid speed", value=Uv, unit=r"\mathrm{m/s}"),
        L: dict(tex="L", meaning="side of the cavity", value=Lv, unit=r"\mathrm{m}"),
        nu: dict(tex=r"\nu", meaning="kinematic viscosity of the fluid", value=nuv, unit=r"\mathrm{m^2/s}"),
    }
    subs = {U: Uv, L: Lv, nu: nuv}
    plugged = {s: rf"\left({tex_num(p['value'])}\,{p['unit']}\right)" for s, p in params.items()}

    def entry(key, label, symbol, expr, unit_tex, why, extra=None):
        sub, used, names = dict(subs), [params[s] for s in (U, L, nu) if s in expr.free_symbols], dict(plugged)
        if extra:
            for s, p in extra.items():
                sub[s] = p["value"]
                used.append(p)
                names[s] = rf"\left({tex_num(p['value'])}\,{p['unit']}\right)" if p["unit"] else tex_num(p["value"])
        value = float(expr.subs(sub))
        return dict(key=key, label=label, symbol=symbol, value=value, value_tex=tex_num(value), unit_tex=unit_tex,
                    formula=rf"{symbol} = {tex(expr)}",
                    check=rf"{symbol} = {tex(expr, symbol_names=names)} = {tex_num(value)}\ {unit_tex}",
                    why=why, params=used)

    hf = dict(tex="h", meaning=f"cell size of the finest mesh, \\(L/{meshes[-1]}\\)", value=Lv / meshes[-1], unit=r"\mathrm{m}")
    hc = dict(tex="h", meaning=f"cell size of the coarsest mesh, \\(L/{meshes[0]}\\)", value=Lv / meshes[0], unit=r"\mathrm{m}")
    NG = sp.Symbol("N_G", positive=True)
    ng = dict(tex="N_G", meaning="number of grid intervals per side in the benchmark (129 grid points)",
              value=round(1 / h_ghia), unit="")
    derived = [
        entry("Re", "Reynolds number", "Re", U * L / nu, "",
              "The only parameter of the dimensionless problem (Step 2): it compares the fluid's inertia, which "
              "carries momentum around the cavity, with viscous friction, which spreads it out. Two cavities with "
              "the same \\(Re\\) have the same flow pattern. The benchmark column used is the one for this \\(Re\\)."),
        entry("t_conv", "Lid transit time", r"t_c", L / U, r"\mathrm{s}",
              "Time for the lid to move one cavity width. Flow features are carried around the cavity on this time scale."),
        entry("t_visc", "Viscous diffusion time", r"t_\nu", L ** 2 / nu, r"\mathrm{s}",
              "Time for friction to spread momentum across the cavity. Its ratio to the transit time is \\(Re\\): "
              "here momentum is carried around faster than it diffuses, so the vortex is pushed off-centre."),
        entry("Pe_coarse", "Cell Péclet number, coarsest mesh", r"Pe_h", U * h / nu, "",
              "Convection versus diffusion across ONE cell, at the lid speed. Above about 2, central differencing of "
              "the convection term can produce wiggles; this is why a linear-upwind (upwind-biased, still second "
              "order) scheme was chosen (Decisions).", extra={h: hc}),
        entry("Pe_fine", "Cell Péclet number, finest mesh", r"Pe_h", U * h / nu, "",
              "On the finest mesh diffusion dominates within each cell, so any scheme behaves well there.",
              extra={h: hf}),
        entry("h_ghia", "Benchmark grid spacing", r"h_G", 1 / NG, r"\ (\text{in units of } L)",
              "Ghia et al. stored their solution only at grid points this far apart, so their vortex-centre "
              "coordinates and the location of the velocity minimum are known only to within about one spacing. "
              "Location differences smaller than this cannot be judged.", extra={NG: ng}),
    ]
    steps = [
        dict(label="Governing equations: continuity and momentum for a steady, incompressible, Newtonian fluid",
             latex=r"\nabla\cdot\mathbf{u} = 0,\qquad (\mathbf{u}\cdot\nabla)\,\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu\,\nabla^2\mathbf{u}",
             note="Mass is conserved, and each fluid parcel's momentum changes only through pressure and viscous "
                  "forces. The left side (convection) makes the equations nonlinear: no exact solution is known."),
        dict(label="Scaling: measure lengths in L, velocities in U, pressure in ρU²",
             latex=r"(\mathbf{u}^*\cdot\nabla^*)\,\mathbf{u}^* = -\nabla^* p^* + \frac{1}{Re}\,\nabla^{*2}\mathbf{u}^*,\qquad Re = \frac{UL}{\nu}",
             note="One number, Re, decides the shape of the flow (checked by SymPy below). This is why the benchmark "
                  "reports x/L, u/U and ψ/(UL): those are the same for every cavity at Re = 100."),
        dict(label="Boundary conditions (no slip)",
             latex=r"\mathbf{u} = (U, 0)\ \text{on the lid } (y = L),\qquad \mathbf{u} = \mathbf{0}\ \text{on the other three walls}",
             note="At the two top corners the wall velocity jumps from U to 0. There the velocity gradient is "
                  "infinite (a corner singularity), so local errors near those corners shrink slowly with refinement "
                  "in every method, including the benchmark."),
        dict(label="Stream function: the requested measure of the vortex",
             latex=r"u = \frac{\partial \psi}{\partial y},\quad v = -\frac{\partial \psi}{\partial x},\qquad "
                   r"\psi = 0 \text{ on all walls},\qquad \int_0^L u(x, y)\,dy = 0",
             note="Lines of constant ψ are streamlines, and the difference of ψ between two points is the flow rate "
                  "between them. The walls are one streamline (ψ = 0). The lid drags the fluid clockwise, so "
                  "ψ &lt; 0 inside; its minimum marks the vortex centre and |ψ_min| is the flow rate circulating around it. "
                  "Because no fluid crosses the walls, the net flow through any vertical line is zero: the forward flow "
                  "under the lid is exactly returned lower down. This is a check on the simulation."),
        dict(label="What else theory predicts (without numbers)",
             latex=r"Re \to 0:\ \text{flow symmetric about } x = L/2;\qquad Re > 0:\ \text{vortex shifted towards the lid's downstream side } (x > L/2)",
             note="Slow (Stokes) flow is reversible, so its pattern is mirror-symmetric about the vertical centreline. "
                  "Inertia carries momentum in the lid direction and breaks that symmetry. Viscous-flow theory "
                  "(Moffatt 1964, J. Fluid Mech. 18) also predicts small counter-rotating eddies in the bottom corners, "
                  "with ψ &gt; 0; the benchmark's Table V lists two of them (not requested here)."),
    ]
    return dict(steps=steps, derived=derived, self_checks=checks)
