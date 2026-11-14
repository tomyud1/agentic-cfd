"""What theory can say about the turbulent channel without a simulation.

There is no exact solution of a turbulent flow. SymPy derives and self-checks what is exact anyway:
  * the mean momentum balance: the total shear stress is a straight line, and the wall shear stress is
    fixed by the driving force (so the friction velocity is known before any run);
  * the laminar solution of the same equations (not the answer wanted; the turbulent flow must carry
    less flow than it);
  * the mean velocity right at the wall, u+ = y+ - y+^2 / (2 Re_tau) + ...;
  * the balance of the whole box: the bulk velocity changes at the rate (driving force - wall shear),
    which gives the time the bulk velocity needs to settle and what an imbalance of the averaged wall
    shear means.
Derived numbers come with their formula and a hand check with the numbers substituted.
"""
import sympy as sp

from cfdlib.report import tex_num

y, f, nu, delta, rho, t = sp.symbols("y f nu delta rho t", positive=True)
u_tau, Ub, n_exp, eps, Lx = sp.symbols("u_tau U_b n epsilon L_x", positive=True)
N_FRICTION = sp.Rational(7, 4)      # exponent of a Blasius-type friction law, tau_w ~ U_b^(7/4); planning only


def derive():
    checks = {}
    # --- mean momentum along x, per unit mass, statistically steady, uniform in x and z:
    #     0 = f + d/dy ( nu dU/dy - <u'v'> ) = f + dT/dy,   T = total shear stress / rho
    T = sp.Function("T")
    sol = sp.dsolve(sp.Eq(T(y).diff(y), -f), T(y), ics={T(delta): 0}).rhs     # symmetry: no stress on the centre plane
    checks[r"total stress \(\tau/\rho = f(\delta - y)\) satisfies \(\mathrm{d}(\tau/\rho)/\mathrm{d}y = -f\) and vanishes on the centre plane"] = \
        sp.simplify(sol.diff(y) + f) == 0 and sol.subs(y, delta) == 0
    checks[r"wall value \(\tau_w/\rho = f\delta\), so \(u_\tau = \sqrt{f\delta}\) and \(\tau = \tau_w(1 - y/\delta)\)"] = \
        sp.simplify(sol.subs(y, 0) - f * delta) == 0 and sp.simplify(sol - f * delta * (1 - y / delta)) == 0

    # --- laminar solution: nu U'' = -f, U = 0 on both plates
    U = sp.Function("U")
    lam = sp.dsolve(sp.Eq(nu * U(y).diff(y, 2), -f), U(y), ics={U(0): 0, U(2 * delta): 0}).rhs
    lam_centre = sp.simplify(lam.subs(y, delta))
    lam_bulk = sp.simplify(sp.integrate(lam, (y, 0, 2 * delta)) / (2 * delta))
    checks[r"laminar profile \(U = f\,y(2\delta - y)/(2\nu)\) satisfies \(\nu U'' = -f\) and no slip on both plates"] = \
        sp.simplify(nu * lam.diff(y, 2) + f) == 0 and lam.subs(y, 0) == 0 and sp.simplify(lam.subs(y, 2 * delta)) == 0
    re_tau = sp.sqrt(f * delta) * delta / nu
    checks[r"laminar centre speed \(= \tfrac12 Re_\tau u_\tau\) and laminar bulk speed \(= \tfrac13 Re_\tau u_\tau\)"] = \
        sp.simplify(lam_centre - re_tau * sp.sqrt(f * delta) / 2) == 0 and sp.simplify(lam_bulk - re_tau * sp.sqrt(f * delta) / 3) == 0

    # --- right at the wall the turbulent stress vanishes like y^3 (no slip and continuity), so
    #     nu dU/dy = f (delta - y) + O(y^3)  ->  U = (f / nu) (delta y - y^2 / 2) + O(y^4)
    near = sp.integrate(f * (delta - y) / nu, (y, 0, y))
    yp = sp.Symbol("y_plus", positive=True)
    Re = sp.Symbol("Re_tau", positive=True)
    near_plus = sp.simplify((near / sp.sqrt(f * delta)).subs(y, yp * nu / sp.sqrt(f * delta)).subs(nu, sp.sqrt(f * delta) * delta / Re))
    checks[r"near the wall \(u^+ = y^+ - (y^+)^2/(2Re_\tau)\) (up to terms of order \((y^+)^4\))"] = \
        sp.simplify(near_plus - (yp - yp ** 2 / (2 * Re))) == 0

    # --- the whole box: d U_b / dt = f - tau_w / (rho delta)   (tau_w: mean of the two plates)
    #     with tau_w / rho = u_tau^2 (U_b / U_eq)^n near the settled state U_eq, a small gap g = U_b - U_eq obeys
    #     dg/dt = - (n u_tau^2 / (delta U_eq)) g
    Ueq, g = sp.symbols("U_eq g", positive=True)
    rhs = u_tau ** 2 / delta - u_tau ** 2 / delta * ((Ueq + g) / Ueq) ** n_exp
    rate = sp.simplify(-sp.series(rhs, g, 0, 2).removeO() / g)
    checks[r"a small gap of the bulk velocity decays at the rate \(n u_\tau^2/(\delta U_b)\) when \(\tau_w \propto U_b^{\,n}\)"] = \
        sp.simplify(rate - n_exp * u_tau ** 2 / (delta * Ueq)) == 0
    # the imbalance of the wall shear that goes with a relative gap g / U_eq
    imb = sp.simplify(sp.series(((Ueq + g) / Ueq) ** n_exp - 1, g, 0, 2).removeO())
    checks[r"a relative gap \(g/U_b\) of the bulk velocity shows as a wall-shear imbalance of \(n\,g/U_b\)"] = \
        sp.simplify(imb - n_exp * g / Ueq) == 0
    # --- the mean of U_b over a window of length T. With xi(t) the part of the wall shear that the eddies add,
    #     dU_b/dt = -(U_b - U_eq)/T_s + xi. Integrate over the window (DU = change of U_b over it):
    #     DU / T = -(mean(U_b) - U_eq) / T_s + mean(xi)   ->   mean(U_b) - U_eq = T_s (mean(xi) - DU / T)
    Ts, T, DU, xibar, Ubar = sp.symbols("T_s T Delta_U xi_bar U_bar", real=True)
    solved = sp.solve(sp.Eq(DU / T, -(Ubar - Ueq) / Ts + xibar), Ubar)[0]
    checks[r"window mean: \(\overline{U_b} - U_{eq} = T_s\,(\overline{\xi} - \Delta U_b/T)\), so its statistical error is \(T_s\) times that of \(\overline{\xi}\)"] = \
        sp.simplify(solved - Ueq - Ts * (xibar - DU / T)) == 0
    if not all(checks.values()):
        raise AssertionError(f"Theory self-check failed: {checks}")
    return dict(checks=checks, total_stress=sol, laminar=lam, lam_centre=lam_centre, lam_bulk=lam_bulk)


def evaluate(given, ub_reference, balance_tol):
    """Derivation steps, derived quantities with hand checks, self-checks.
    ub_reference: the reference's bulk velocity, used only for the two planning numbers that need one
    (settling time, flow-through time). balance_tol: the stop rule's limit on the wall-shear imbalance."""
    d = derive()
    gv = {k: given[k]["value"] for k in ("delta", "f", "nu", "rho", "Lx")}
    P = {
        f: dict(tex="f", meaning="driving force per unit mass", value=gv["f"], unit=r"\mathrm{m/s^2}"),
        delta: dict(tex=r"\delta", meaning="half the distance between the plates", value=gv["delta"], unit=r"\mathrm{m}"),
        nu: dict(tex=r"\nu", meaning="kinematic viscosity", value=gv["nu"], unit=r"\mathrm{m^2/s}"),
        rho: dict(tex=r"\rho", meaning="density", value=gv["rho"], unit=r"\mathrm{kg/m^3}"),
        Lx: dict(tex="L_x", meaning="length of the periodic box", value=gv["Lx"], unit=r"\mathrm{m}"),
    }
    ut = float(sp.sqrt(gv["f"] * gv["delta"]))
    extra_ut = dict(tex=r"u_\tau", meaning="friction velocity (derived above)", value=ut, unit=r"\mathrm{m/s}")
    extra_ub = dict(tex="U_b", meaning="bulk velocity; here the reference's value, for planning only", value=ub_reference,
                    unit=r"\mathrm{m/s}")
    extra_n = dict(tex="n", meaning="exponent of the friction law \\(\\tau_w \\propto U_b^{\\,n}\\); 7/4 for a Blasius-type "
                                    "law (an approximation, used for planning only)", value=float(N_FRICTION), unit="")
    extra_eps = dict(tex=r"\epsilon", meaning="limit on the wall-shear imbalance in the stop rule", value=balance_tol, unit="")

    def entry(key, label, symbol, expr, unit_tex, why, extra=None, sig=4):
        sub = {s: p["value"] for s, p in P.items()}
        used = [P[s] for s in (f, delta, nu, rho, Lx) if s in expr.free_symbols]
        names = {s: rf"\left({tex_num(p['value'], 10 if s is nu else 4)}\,{p['unit']}\right)" for s, p in P.items()}
        for s, p in (extra or {}).items():
            sub[s] = p["value"]
            used.append(p)
            names[s] = rf"\left({tex_num(p['value'])}\,{p['unit']}\right)" if p["unit"] else tex_num(p["value"])
        value = float(expr.subs(sub))
        return dict(key=key, label=label, symbol=symbol, value=value, value_tex=tex_num(value, sig), unit_tex=unit_tex,
                    formula=rf"{symbol} = {sp.latex(expr)}",
                    check=rf"{symbol} = {sp.latex(expr, symbol_names=names)} = {tex_num(value, sig)}\ {unit_tex}",
                    why=why, params=used)

    derived = [
        entry("tau_w", "Mean wall shear stress", r"\tau_w", rho * f * delta, r"\mathrm{Pa}",
              "Known before any run: in a settled flow the two plates carry exactly the driving force. The simulation "
              "does not prescribe it; its time-averaged wall shear must come out at this value (request g)."),
        entry("u_tau", "Friction velocity", r"u_\tau", sp.sqrt(f * delta), r"\mathrm{m/s}",
              "The velocity scale of the turbulence near the wall, \\(\\sqrt{\\tau_w/\\rho}\\). Every result is reported "
              "in units of it (wall units)."),
        entry("re_tau", "Friction Reynolds number", r"Re_\tau", sp.sqrt(f * delta) * delta / nu, "",
              "Half the plate distance measured in viscous lengths \\(\\nu/u_\\tau\\): the ratio of the largest to the "
              "smallest eddies. At 180 the flow is turbulent but the range of eddy sizes is small enough to resolve "
              "all of them on a grid.", sig=6),
        entry("l_visc", "Viscous length (one wall unit)", r"\ell_\nu", nu / sp.sqrt(f * delta), r"\mathrm{m}",
              "The size of the smallest eddies near the wall scales with it. Cell sizes are judged in this unit: "
              "\\(y^+ = y/\\ell_\\nu\\)."),
        entry("t_unit", "Time unit (eddy turnover time)", r"T", delta / sp.sqrt(f * delta), r"\mathrm{s}",
              "Time for the largest eddies to turn over once. Run lengths and averaging times are counted in it."),
        entry("lam_centre", "Laminar centre speed (not the answer)", r"U_{c,lam}", d["lam_centre"], r"\mathrm{m/s}",
              "The same equations have this laminar solution. Turbulence carries momentum to the walls far better "
              "than viscosity alone, so the turbulent flow is several times slower; a run that drifts toward this "
              "value has lost its turbulence."),
        entry("lam_bulk", "Laminar bulk speed (upper bound)", r"U_{b,lam}", d["lam_bulk"], r"\mathrm{m/s}",
              "An upper bound for the bulk velocity of the turbulent flow at the same driving force."),
        entry("u_wall", "Mean velocity at the centre of a wall cell of height 1 wall unit", r"u^+(0.5)",
              sp.Rational(1, 2) - sp.Rational(1, 8) / (sp.sqrt(f * delta) * delta / nu), "",
              "Exact limit at the wall: \\(u^+ = y^+ - (y^+)^2/(2Re_\\tau)\\). The profile of any correct simulation "
              "must start on this line, whatever the turbulence does further out."),
        entry("t_settle", "Settling time of the bulk velocity", r"T_s", delta * Ub / (n_exp * u_tau ** 2), r"\mathrm{s}",
              "The bulk velocity is driven by the difference between the driving force and the wall shear, and "
              "forgets a disturbance only this slowly: about nine time units for a factor \\(e\\). It sets how much "
              "of each run is start-up and has to be cut from the averages.",
              extra={Ub: extra_ub, n_exp: extra_n, u_tau: extra_ut}),
        entry("gap", "Unsettled bulk velocity that the balance limit allows", r"g/U_b", eps / n_exp, "",
              "If the averaged wall shear differs from the driving force by \\(\\epsilon\\), the bulk velocity can "
              "still be this far (as a fraction) from its settled value. The stop rule's limit on the imbalance "
              "was chosen from this.", extra={eps: extra_eps, n_exp: extra_n}),
        entry("t_flow", "Flow-through time of the box", r"T_x", Lx / Ub, r"\mathrm{s}",
              "Time for the mean flow to cross the periodic box once. Much shorter than the settling time: "
              "the flow passes through the box many times before the averages settle.", extra={Ub: extra_ub}),
    ]
    steps = [
        dict(label="Governing equations: incompressible flow driven by a constant force",
             latex=r"\nabla\cdot\mathbf{u} = 0,\qquad \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\,\mathbf{u} "
                   r"= -\frac{1}{\rho}\nabla p + \nu\,\nabla^2\mathbf{u} + f\,\mathbf{e}_x",
             note="Solved as they stand, in three dimensions and in time, with no model added. The turbulent flow "
                  "is chaotic: no formula gives it, and two runs never agree instant by instant. Only its averages "
                  "are reproducible, so every requested output is an average."),
        dict(label="Average over x, z and time: the mean momentum balance",
             latex=r"0 = f + \frac{\mathrm{d}}{\mathrm{d}y}\left(\nu\,\frac{\mathrm{d}\langle u\rangle}{\mathrm{d}y} - \langle u'v'\rangle\right)"
                   r"\quad\Rightarrow\quad \frac{\tau(y)}{\rho} = \nu\,\frac{\mathrm{d}\langle u\rangle}{\mathrm{d}y} - \langle u'v'\rangle = f\,(\delta - y)",
             note="Exact for a settled flow. The total shear stress (viscous part plus the part carried by the "
                  "eddies) is a straight line from the wall to zero on the centre plane, and its wall value is "
                  "\\(\\tau_w = \\rho f\\delta\\). Theory does not say how the total splits into its two parts; "
                  "that is the turbulence, and the simulation has to find it."),
        dict(label="The laminar solution of the same equations (not the answer wanted)",
             latex=r"U_{lam}(y) = \frac{f}{2\nu}\,y\,(2\delta - y),\qquad U_{c,lam} = \tfrac12 Re_\tau\,u_\tau,\qquad U_{b,lam} = \tfrac13 Re_\tau\,u_\tau",
             note="Without eddies the whole stress is viscous and the profile is a parabola. The start field must "
                  "push the flow away from this solution, which is stable to small disturbances."),
        dict(label="Right at the wall the eddies die out and the profile is known",
             latex=r"u^+ = y^+ - \frac{(y^+)^2}{2\,Re_\tau} + O\!\left((y^+)^4\right),\qquad u^+ = \frac{\langle u\rangle}{u_\tau},\quad y^+ = \frac{y\,u_\tau}{\nu}",
             note="No slip and continuity make the turbulent stress vanish like \\(y^3\\) at the wall, so the "
                  "viscous stress carries everything there. Further out (roughly \\(y^+ > 30\\)) experiments and "
                  "simulations follow the logarithmic law \\(u^+ = \\tfrac{1}{\\kappa}\\ln y^+ + B\\); the constants "
                  "are empirical (\\(\\kappa \\approx 0.41\\), \\(B \\approx 5.2\\), textbook values quoted from "
                  "memory of Pope, Turbulent Flows, 2000, so approximate), and at this low Reynolds number the "
                  "logarithmic region is short. The law is drawn for orientation, not used as a check."),
        dict(label="The whole box: why the bulk velocity settles slowly",
             latex=r"\frac{\mathrm{d}U_b}{\mathrm{d}t} = f - \frac{\tau_{w,low} + \tau_{w,high}}{2\,\rho\,\delta}"
                   r"\quad\Rightarrow\quad T_s = \frac{\delta\,U_b}{n\,u_\tau^2}\ \text{ for } \tau_w \propto U_b^{\,n}",
             note="Exact at every instant (the periodic sides cancel). Averaged over a window it says: mean wall "
                  "shear minus driving force equals minus the change of the bulk velocity over the window, divided "
                  "by its length. So the averaged wall shear against the driving force measures how well the "
                  "average has settled (request g). The exponent \\(n\\) comes from an empirical friction law and is "
                  "used only to plan run lengths."),
        dict(label="An average over a finite time has a statistical error",
             latex=r"\sigma_{\overline{q}} \approx \sigma_q\,\sqrt{\frac{2\,T_{corr}}{T_{avg}}}",
             note="Successive samples of a turbulent signal resemble each other for about its memory "
                  "\\(T_{corr}\\), so an average over \\(T_{avg}\\) holds only about \\(T_{avg}/(2T_{corr})\\) "
                  "independent samples. No grid refinement removes this error; only a longer average does, with "
                  "the square root of the time. Every result below is quoted with it, estimated from the scatter of "
                  "block averages (cfdlib.signals.stationary). That estimate needs blocks several memories long."),
        dict(label="The bulk velocity has a long memory: its error from the balance of the box",
             latex=r"\frac{\mathrm{d}U_b}{\mathrm{d}t} = -\frac{U_b - U_{eq}}{T_s} + \xi(t)"
                   r"\quad\Rightarrow\quad \overline{U_b} - U_{eq} = T_s\left(\overline{\xi} - \frac{\Delta U_b}{T_{avg}}\right),"
                   r"\qquad \sigma_{\overline{U_b}} = T_s\,\sigma_{\overline{\xi}}",
             note="Near its settled value the bulk velocity is pulled back with the settling time \\(T_s\\) and "
                  "pushed by \\(\\xi\\), what the eddies add to the wall shear from moment to moment. So the "
                  "memory of the bulk velocity is \\(T_s\\), about nine time units: a run of 40 time units holds "
                  "about two independent samples of it, and blocks cannot be made long enough. The push "
                  "\\(\\xi = f - \\tau_w/(\\rho\\delta) + (U_b - \\overline{U_b})/T_s\\) is computed from the recorded "
                  "wall shear and bulk velocity; it has a short memory, so its mean has a sound block estimate, and "
                  "\\(T_s\\) times that is the error of the mean bulk velocity. This was added after two runs of the "
                  "coarsest grid disagreed by five times the plain block estimate (Section 5)."),
    ]
    return dict(steps=steps, derived=derived, self_checks=d["checks"],
                values={e["key"]: e["value"] for e in derived})


if __name__ == "__main__":
    import yaml
    from pathlib import Path
    here = Path(__file__).resolve().parent
    prob = yaml.safe_load((here / "problem.yaml").read_text(encoding="utf-8"))
    out = evaluate(prob["given"], 15.7, 3e-3)
    for k, v in out["self_checks"].items():
        print("OK " if v else "FAIL", k)
    for e in out["derived"]:
        print(e["label"], "=", e["value"], "|", e["check"])
