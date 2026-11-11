"""What theory can say about the DFG 2D-2 cylinder flow without a simulation (no exact solution exists).

SymPy does the algebra and self-checks it:
  * the parabolic inflow: zero at both walls, maximum U_m at mid-height, mean (2/3) U_m;
  * it is the exact fully developed (Poiseuille) channel flow: nu u'' = constant pressure gradient;
  * for that flow the pressure force on a channel section balances the friction on its walls
    (the empty-channel limit of the momentum balance used as a check on the simulation);
  * derived numbers (Re, time scales, blockage, offset, boundary-layer scale), each with its formula
    and a hand-check with the numbers substituted.
"""
import sympy as sp

from cfdlib.report import tex_num

y, x = sp.symbols("y x", real=True)
H, D, Um, nu, L, Lc = sp.symbols("H D U_m nu L L_c", positive=True)


def inflow(yy=y):
    return 4 * Um * yy * (H - yy) / H ** 2


def derive():
    checks = {}
    u = inflow()
    checks[r"inflow \(u(0) = u(H) = 0\) (no slip at both walls)"] = sp.simplify(u.subs(y, 0)) == 0 and sp.simplify(u.subs(y, H)) == 0
    ymax = sp.solve(sp.diff(u, y), y)
    checks[r"inflow maximum at \(y = H/2\) with \(u = U_m\)"] = ymax == [H / 2] and sp.simplify(u.subs(y, H / 2) - Um) == 0
    Umean = sp.integrate(u, (y, 0, H)) / H
    checks[r"mean inflow velocity \(U_{\mathrm{mean}} = \frac{1}{H}\int_0^H u\,dy = \frac{2}{3}U_m\)"] = sp.simplify(Umean - 2 * Um / 3) == 0
    # Fully developed channel flow: 0 = -dp/dx + nu u''  ->  dp/dx = nu u'' must be a constant
    dpdx = sp.simplify(nu * sp.diff(u, y, 2))
    checks[r"the parabola solves the channel momentum equation: \(\nu\,u'' = \partial p/\partial x = -8\nu U_m/H^2\) (constant)"] = \
        sp.simplify(dpdx + 8 * nu * Um / H ** 2) == 0 and not dpdx.has(y)
    # Momentum balance of an empty channel section of length Lc: pressure force = friction on the two walls
    tau_w = nu * sp.diff(u, y).subs(y, 0)                    # kinematic wall shear stress (per unit density)
    checks[r"empty channel: pressure force \(H\,\Delta p\) equals the friction \(2\tau_w L_c\) on both walls"] = \
        sp.simplify(-dpdx * Lc * H - 2 * tau_w * Lc) == 0
    if not all(checks.values()):
        raise AssertionError(f"Theory self-check failed: {checks}")
    return checks, dict(Umean=Umean, dpdx=dpdx, tau_w=tau_w)


def evaluate(given, h_surface_fine=None, dt_fine=None):
    """Derivation steps, derived quantities with hand-checks, self-checks."""
    checks, ex = derive()
    v = {k: given[k]["value"] for k in ("H", "L", "D", "xc", "yc", "Um", "nu", "rho")}
    tex = sp.latex
    Umean_s = sp.Symbol(r"U_{\mathrm{mean}}", positive=True)
    yc_s = sp.Symbol("y_c", positive=True)
    params = {
        Um: dict(tex="U_m", meaning="maximum inflow velocity", value=v["Um"], unit=r"\mathrm{m/s}"),
        D: dict(tex="D", meaning="cylinder diameter", value=v["D"], unit=r"\mathrm{m}"),
        H: dict(tex="H", meaning="channel height", value=v["H"], unit=r"\mathrm{m}"),
        L: dict(tex="L", meaning="channel length", value=v["L"], unit=r"\mathrm{m}"),
        nu: dict(tex=r"\nu", meaning="kinematic viscosity", value=v["nu"], unit=r"\mathrm{m^2/s}"),
        yc_s: dict(tex="y_c", meaning="height of the cylinder centre", value=v["yc"], unit=r"\mathrm{m}"),
    }
    Umean_v = float(ex["Umean"].subs(Um, v["Um"]))
    params[Umean_s] = dict(tex=r"U_{\mathrm{mean}}", meaning=r"mean inflow velocity, \(\tfrac{2}{3}U_m\)", value=Umean_v,
                           unit=r"\mathrm{m/s}")

    def entry(key, label, symbol, expr, unit_tex, why, extra=None):
        sub, names, used = {}, {}, []
        allp = dict(params, **(extra or {}))
        for s_, p in allp.items():
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
        entry("Umean", "Mean inflow velocity", r"U_{\mathrm{mean}}", sp.Rational(2, 3) * Um, r"\mathrm{m/s}",
              r"The velocity scale of the benchmark: force coefficients and the Strouhal number are defined with it "
              r"(mean of the parabola, checked by SymPy below)."),
        entry("Re", "Reynolds number", "Re", Umean_s * D / nu, "",
              r"Inertia versus friction at the scale of the cylinder. The DFG benchmark 2D-2 is defined at "
              r"\(Re = 100\); at this value the wake behind a cylinder is unstable and sheds vortices periodically "
              r"(the problem statement says so; for an unconfined cylinder the onset is near \(Re \approx 47\), "
              r"Williamson 1996, Annu. Rev. Fluid Mech. 28, approximate)."),
        entry("blockage", "Blockage ratio", r"D/H", D / H, "",
              r"The cylinder fills a quarter of the channel: the walls are close, so the flow is strongly confined and "
              r"the numbers cannot be compared with an unconfined cylinder."),
        entry("offset", "Vertical offset of the cylinder", r"\frac{H/2 - y_c}{D}", (H / 2 - yc_s) / D, "",
              r"The cylinder sits slightly below the centreline. A centred cylinder would give a mean lift of zero by "
              r"symmetry; the small offset gives a small non-zero mean lift and makes the start of shedding "
              r"reproducible (no random perturbation is needed)."),
        entry("t_conv", "Convective time", r"t_c", D / Umean_s, r"\mathrm{s}",
              r"Time for the flow to pass one diameter. The shedding period is a few of these (a Strouhal number "
              r"\(St = t_c / T\) of order 0.3 means \(T \approx 3\,t_c\))."),
        entry("t_through", "Flow-through time", r"t_L", L / Umean_s, r"\mathrm{s}",
              r"Time for the mean flow to cross the whole channel. A start-up disturbance leaves through the outlet "
              r"after about this time; the developing run covers several of them."),
        entry("delta", "Boundary-layer thickness scale", r"\delta", D / sp.sqrt(Umean_s * D / nu), r"\mathrm{m}",
              r"Laminar boundary layers on the cylinder have a thickness of order \(D/\sqrt{Re}\) (scaling, not an exact "
              r"value). The mesh must resolve it with several cells: see the mesh table."),
        entry("dpdx", "Pressure gradient of the undisturbed channel flow", r"-\frac{\partial p}{\partial x}",
              8 * nu * Um / H ** 2, r"\mathrm{m/s^2}",
              r"Kinematic pressure gradient (\(p/\rho\)) that drives the parabolic flow without the cylinder (SymPy check below). "
              r"It sets the scale of the wall friction in the momentum balance."),
        entry("Fwall", "Wall friction of the undisturbed channel, per unit depth", r"F_{w}", 2 * (4 * nu * Um / H) * L,
              r"\mathrm{m^3/s^2}",
              r"Kinematic friction force of the parabolic flow on both walls over the full length \(L\). In the momentum "
              r"balance it is of the same order as the cylinder drag (\(\tfrac12 c_D U_{\mathrm{mean}}^2 D \approx 0.16\) for "
              r"\(c_D \approx 3.2\)), so the balance cannot ignore it."),
    ]
    steps = [
        dict(label="Governing equations: incompressible, Newtonian, constant density, two-dimensional",
             latex=r"\nabla\cdot\mathbf{u} = 0,\qquad \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\,\mathbf{u} "
                   r"= -\frac{1}{\rho}\nabla p + \nu\,\nabla^2\mathbf{u}",
             note="Convection makes the equations nonlinear and, at this Reynolds number, the steady solution unstable: "
                  "no exact solution is known. The flow is computed; theory gives scales, limits and checks."),
        dict(label="Boundary conditions",
             latex=r"x = 0:\ u = \frac{4U_m\,y\,(H - y)}{H^2},\ v = 0;\qquad x = L:\ \nu\frac{\partial \mathbf{u}}{\partial n} - \frac{p}{\rho}\,\mathbf{n} = 0;"
                   r"\qquad \text{walls, cylinder: } \mathbf{u} = 0",
             note="The outflow condition is the 'do-nothing' (stress-free) condition of the benchmark. The finite-volume "
                  "model uses zero normal gradient of velocity and zero pressure at the outlet, which is the same condition "
                  "when the flow leaves parallel to the axis (the wake is weak 2 m downstream)."),
        dict(label="Requested quantities",
             latex=r"c_D = \frac{2F_D}{\rho\,U_{\mathrm{mean}}^2 D},\quad c_L = \frac{2F_L}{\rho\,U_{\mathrm{mean}}^2 D},\quad "
                   r"St = \frac{f D}{U_{\mathrm{mean}}},\quad \bar c = \tfrac12(c_{\max} + c_{\min}),\quad \Delta c = c_{\max} - c_{\min}",
             note="Forces per unit depth, integrated over the cylinder surface (pressure and viscous parts). Mean and "
                  "amplitude follow the benchmark's definitions over one cycle (amplitude = peak to peak)."),
        dict(label="Symmetry: what shedding must look like",
             latex=r"y_c = H/2:\quad c_L(t + T/2) = -c_L(t),\qquad c_D(t + T/2) = c_D(t)\ \Rightarrow\ f_D = 2 f_L",
             note="For a centred cylinder the flow half a period later is the mirror image of the flow now. Lift changes "
                  "sign under the mirror, drag does not, so drag oscillates at twice the lift frequency and the mean lift is "
                  "zero. The offset here is small, so this should hold closely; it is checked on the simulated signals."),
        dict(label="Time-averaged x-momentum balance of the whole channel (check of the force output)",
             latex=r"\overline{F_{D}} + \overline{F_{w}} = \overline{\int_{\mathrm{in}}\!\left(\tfrac{p}{\rho} + u^2\right)dy} "
                   r"- \overline{\int_{\mathrm{out}}\!\left(\tfrac{p}{\rho} + u^2\right)dy} - \frac{M(t_1) - M(t_0)}{t_1 - t_0}",
             note="Over whole cycles, the drag on the cylinder plus the friction on the channel walls must equal the "
                  "net pressure and momentum flux through inlet and outlet, minus the change of the momentum \\(M\\) stored "
                  "in the channel (all per unit depth and density; viscous normal stresses at inlet and outlet are neglected, "
                  "they vanish for parallel flow). Without the cylinder it reduces to the pressure-friction balance checked "
                  "by SymPy below."),
    ]
    return dict(steps=steps, derived=derived, self_checks=checks, Umean=Umean_v)
