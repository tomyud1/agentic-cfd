"""What theory can say about the DFG 3D-1Z cylinder flow without a simulation (no exact solution exists).

SymPy does the algebra and self-checks it:
  * the inflow profile: zero on all four walls, maximum U_m at the centre of the inlet, mean (4/9) U_m;
  * it is NOT the exact fully developed flow of a square duct (its Laplacian is not constant), so the
    flow re-arranges itself downstream of the inlet: a property of the benchmark, not an error;
  * mirror symmetry about the mid-plane z = H/2 (geometry and inflow): no side force, w = 0 on the mid-plane;
  * derived numbers (Re, force scale, blockage, offset, boundary-layer scale), each with its formula and
    a hand-check with the numbers substituted.
"""
import sympy as sp

from cfdlib.report import tex_num

y, z = sp.symbols("y z", real=True)
H, D, Um, nu, L, rho = sp.symbols("H D U_m nu L rho", positive=True)


def inflow(yy=y, zz=z):
    return 16 * Um * yy * zz * (H - yy) * (H - zz) / H ** 4


def derive():
    checks = {}
    u = inflow()
    walls = [u.subs(y, 0), u.subs(y, H), u.subs(z, 0), u.subs(z, H)]
    checks[r"inflow \(u = 0\) on all four walls (no slip)"] = all(sp.simplify(w) == 0 for w in walls)
    crit = sp.solve([sp.diff(u, y), sp.diff(u, z)], [y, z], dict=True)
    inside = [c for c in crit if all(0 < c[s].subs(H, 1) < 1 for s in (y, z))]
    checks[r"inflow maximum at the centre \(y = z = H/2\) with \(u = U_m\)"] = \
        len(inside) == 1 and inside[0] == {y: H / 2, z: H / 2} and sp.simplify(u.subs({y: H / 2, z: H / 2}) - Um) == 0
    Ubar = sp.integrate(u, (y, 0, H), (z, 0, H)) / H ** 2
    checks[r"mean inflow velocity \(\bar U = \frac{1}{H^2}\iint u\,dy\,dz = \frac{4}{9}U_m\)"] = sp.simplify(Ubar - sp.Rational(4, 9) * Um) == 0
    lap = sp.simplify(sp.diff(u, y, 2) + sp.diff(u, z, 2))
    # Fully developed duct flow needs nu * Laplacian(u) = dp/dx = constant over the section. Here it varies:
    checks[r"the inflow is not the developed square-duct flow: \(\nabla^2 u\) varies over the section"] = \
        sp.simplify(lap.subs({y: H / 2, z: H / 2}) - lap.subs({y: H / 2, z: 0})) != 0
    checks[r"inflow symmetric about the mid-plane: \(u(y, H - z) = u(y, z)\)"] = sp.simplify(u.subs(z, H - z) - u) == 0
    if not all(checks.values()):
        raise AssertionError(f"Theory self-check failed: {checks}")
    return checks, dict(Ubar=Ubar, lap=lap)


def evaluate(given):
    """Derivation steps, derived quantities with hand-checks, self-checks."""
    checks, ex = derive()
    v = {k: given[k]["value"] for k in ("H", "L", "D", "xc", "yc", "Um", "nu", "rho")}
    tex = sp.latex
    Ubar_s = sp.Symbol(r"\bar U", positive=True)
    yc_s = sp.Symbol("y_c", positive=True)
    xc_s = sp.Symbol("x_c", positive=True)
    params = {
        Um: dict(tex="U_m", meaning="maximum inflow velocity", value=v["Um"], unit=r"\mathrm{m/s}"),
        D: dict(tex="D", meaning="cylinder diameter", value=v["D"], unit=r"\mathrm{m}"),
        H: dict(tex="H", meaning="side of the square channel section, also the cylinder length", value=v["H"], unit=r"\mathrm{m}"),
        L: dict(tex="L", meaning="channel length", value=v["L"], unit=r"\mathrm{m}"),
        nu: dict(tex=r"\nu", meaning="kinematic viscosity", value=v["nu"], unit=r"\mathrm{m^2/s}"),
        rho: dict(tex=r"\rho", meaning="density", value=v["rho"], unit=r"\mathrm{kg/m^3}"),
        yc_s: dict(tex="y_c", meaning="height of the cylinder axis", value=v["yc"], unit=r"\mathrm{m}"),
        xc_s: dict(tex="x_c", meaning="distance of the cylinder axis from the inlet", value=v["xc"], unit=r"\mathrm{m}"),
    }
    Ubar_v = float(ex["Ubar"].subs(Um, v["Um"]))
    params[Ubar_s] = dict(tex=r"\bar U", meaning=r"mean inflow velocity, \(\tfrac{4}{9}U_m\)", value=Ubar_v, unit=r"\mathrm{m/s}")

    def entry(key, label, symbol, expr, unit_tex, why):
        sub, names, used = {}, {}, []
        for s_, p in params.items():
            if s_ in expr.free_symbols:
                sub[s_] = p["value"]
                used.append(p)
                names[s_] = tex_num(p["value"])
        value = float(expr.subs(sub))
        return dict(key=key, label=label, symbol=symbol, value=value, value_tex=tex_num(value), unit_tex=unit_tex,
                    formula=rf"{symbol} = {tex(expr)}",
                    check=rf"{symbol} = {tex(expr, symbol_names=names)} = {tex_num(value)}",
                    why=why, params=used)

    derived = [
        entry("Ubar", "Mean inflow velocity", r"\bar U", sp.Rational(4, 9) * Um, r"\mathrm{m/s}",
              r"The velocity scale of the benchmark: the Reynolds number and both force coefficients are defined with it. "
              r"In 3D the mean of the product of two parabolas is \(\tfrac{2}{3} \times \tfrac{2}{3} = \tfrac{4}{9}\) of the "
              r"maximum (SymPy check below)."),
        entry("Re", "Reynolds number", "Re", Ubar_s * D / nu, "",
              r"Inertia versus friction at the scale of the cylinder. At \(Re = 20\) friction is strong: the flow stays "
              r"steady (no vortex shedding; the problem says so, and the 2D benchmark only sheds at \(Re = 100\)), and a "
              r"short, closed region of reversed flow sits behind the cylinder."),
        entry("Fscale", "Force scale (dynamic pressure times frontal area)", r"F_0", rho * Ubar_s ** 2 * D * H / 2,
              r"\mathrm{N}",
              r"The denominator of \(c_D\) and \(c_L\): a coefficient is the force in units of this scale. At low \(Re\) "
              r"friction makes a large part of the drag, and the confinement speeds the flow up around the cylinder, so "
              r"\(c_D\) is several times larger than at high \(Re\)."),
        entry("blockage", "Blockage ratio", r"D/H", D / H, "",
              r"The cylinder fills a quarter of the channel height, and it spans the channel from wall to wall: the walls "
              r"are close on all sides, so the flow is strongly confined."),
        entry("offset", "Vertical offset of the cylinder", r"\frac{H/2 - y_c}{D}", (H / 2 - yc_s) / D, "",
              r"The cylinder sits a twentieth of its diameter below the mid-height. A centred cylinder would have exactly zero "
              r"lift by symmetry; this small offset is the only reason the lift is not zero, so \(c_L\) is hundreds of times "
              r"smaller than \(c_D\) and is a small difference of large pressure and friction forces: hard to compute "
              r"accurately."),
        entry("delta", "Boundary-layer thickness scale", r"\delta", D / sp.sqrt(Ubar_s * D / nu), r"\mathrm{m}",
              r"Laminar boundary layers on the cylinder have a thickness of order \(D/\sqrt{Re}\) (a scaling, not an exact "
              r"value; at this low \(Re\) the layer is thick, a fifth of the diameter). The mesh must resolve it with "
              r"several cells: see the mesh table."),
        entry("upstream", "Distance from the inlet to the cylinder", r"\frac{x_c}{D}", xc_s / D, "",
              r"The inflow profile is not the developed duct flow (SymPy check below), so the flow re-arranges between "
              r"inlet and cylinder; over \(5\,D\) at \(Re = 20\) this adjustment is part of the problem as posed, the same "
              r"for every code that solves it."),
    ]
    steps = [
        dict(label="Governing equations: steady, incompressible, Newtonian, constant density",
             latex=r"\nabla\cdot\mathbf{u} = 0,\qquad (\mathbf{u}\cdot\nabla)\,\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu\,\nabla^2\mathbf{u}",
             note="The convection term on the left makes the equations nonlinear; with a cylinder in a duct no exact solution "
                  "is known. The flow is computed; theory gives scales, symmetry and a balance used as checks."),
        dict(label="Boundary conditions",
             latex=r"x = 0:\ u = \frac{16\,U_m\,y z (H - y)(H - z)}{H^4},\ v = w = 0;\qquad x = L:\ \nu\frac{\partial \mathbf{u}}{\partial n} - \frac{p}{\rho}\,\mathbf{n} = 0;"
                   r"\qquad \text{walls, cylinder: } \mathbf{u} = 0",
             note="The outflow condition is the stress-free ('do-nothing') condition of the benchmark. The finite-volume model "
                  "uses zero normal gradient of velocity and zero pressure at the outlet, the same condition for flow leaving "
                  "parallel to the axis, which holds 20 diameters behind the cylinder."),
        dict(label="Requested quantities (3D normalisation with the cylinder length)",
             latex=r"c_D = \frac{2F_D}{\rho\,\bar U^2 D H},\qquad c_L = \frac{2F_L}{\rho\,\bar U^2 D H},\qquad "
                   r"\mathbf{F} = \oint_{\mathrm{cyl}} \left(-p\,\mathbf{n} + \rho\nu\,(\nabla\mathbf{u} + \nabla\mathbf{u}^T)\cdot\mathbf{n}\right)dA",
             note="The force is the pressure plus the friction integrated over the whole cylinder surface; drag is its "
                  "x component, lift its y component."),
        dict(label="Symmetry about the mid-plane z = H/2",
             latex=r"\mathbf{u}(x, y, H - z) = \big(u, v, -w\big)(x, y, z)\ \Rightarrow\ w = 0 \text{ on } z = H/2,\qquad F_z = 0",
             note="Geometry and inflow are mirror-symmetric about the mid-plane, so (for the steady, stable solution) the flow "
                  "is too: no fluid crosses the mid-plane and the side force vanishes. The simulation does not impose this; it "
                  "is checked."),
        dict(label="Steady x-momentum balance of the whole channel (check of the force output)",
             latex=r"F_D + F_w = \int_{\mathrm{in}}\!\left(p + \rho u^2\right)dA - \int_{\mathrm{out}}\!\left(p + \rho u^2\right)dA",
             note="In steady flow the drag on the cylinder plus the friction on the four channel walls must equal the net "
                  "pressure and momentum flux through inlet and outlet (viscous normal stresses at inlet and outlet are "
                  "neglected: they vanish for flow parallel to the axis). It tests the force output independently of the "
                  "force integration on the cylinder."),
    ]
    return dict(steps=steps, derived=derived, self_checks=checks, Ubar=Ubar_v)


if __name__ == "__main__":
    import yaml
    from pathlib import Path
    g = yaml.safe_load((Path(__file__).parent / "problem.yaml").read_text(encoding="utf-8"))["given"]
    th = evaluate(g)
    for d in th["derived"]:
        print(d["key"], d["value"], d["check"])
    print(th["self_checks"])
