"""What can be known about the turbulent flat plate without the simulation.

There is no exact solution of the whole flow. But the Spalart-Allmaras model has an exact solution in the
layer next to the wall where the total shear stress is constant: there the model's variable is
nu_tilde = kappa u_tau y, all the way to the wall. SymPy checks that this satisfies the model's transport
equation (with and without the f_t2 term), and the velocity profile in wall units follows by one
integration: the model's own law of the wall. The simulation must reproduce it near the wall, so it
tests the implementation of the model directly, without any other code.
"""
import math
import re

import numpy as np
import sympy as sp
from scipy import integrate

from cfdlib.report import tex_num

from .common import REF

# Model constants, as printed on NASA's page of the model (reference/tmr/spalart.html)
CONSTANTS = dict(cb1=0.1355, sigma=sp.Rational(2, 3), cb2=0.622, kappa=0.41, cw2=0.3, cw3=2.0, cv1=7.1, ct3=1.2, ct4=0.5)


def page_constants():
    """The constants as NASA's page gives them (parsed from the saved page), to check CONSTANTS against."""
    text = (REF / "tmr" / "spalart.html").read_text(errors="replace")
    i = text.index("The constants are:")
    block = text[i:i + 3000]
    out = {}
    for name, key in (("c_{b1}", "cb1"), ("c_{b2}", "cb2"), ("\\kappa", "kappa"), ("c_{w2}", "cw2"), ("c_{w3}", "cw3"),
                      ("c_{v1}", "cv1"), ("c_{t3}", "ct3"), ("c_{t4}", "ct4")):
        m = re.search(re.escape(name) + r"\s*=\s*([0-9.]+)", block)
        out[key] = float(m.group(1)) if m else None
    m = re.search(r"\\sigma\s*=\s*([0-9]+)\s*/\s*([0-9]+)", block)
    out["sigma"] = int(m.group(1)) / int(m.group(2)) if m else None
    far = re.search(r"\\nu_\{t,farfield\}\s*=\s*([0-9.]+)", text)
    out["nut_far_low"] = float(far.group(1)) if far else None
    return out


def derive():
    """Symbolic self-checks. Returns {statement: True}."""
    y, ut, nu, kap = sp.symbols("y u_tau nu kappa", positive=True)
    cb1, cb2, sig, cv1 = sp.symbols("c_b1 c_b2 sigma c_v1", positive=True)
    ft2 = sp.Symbol("f_t2", real=True)                       # any value: the check must hold for all of them
    cw1 = cb1 / kap ** 2 + (1 + cb2) / sig
    nt = kap * ut * y                                        # the candidate inner solution
    chi = nt / nu
    fv1 = chi ** 3 / (chi ** 3 + cv1 ** 3)
    fv2 = 1 - chi / (1 + chi * fv1)
    dudy = ut ** 2 / (nu + nt * fv1)                         # constant total stress: (nu + nu_t) du/dy = u_tau^2
    S = dudy + nt / (kap * y) ** 2 * fv2                     # Omega = du/dy in a parallel flow
    checks = {}
    checks[r"with \(\tilde\nu = \kappa u_\tau y\) and a constant total stress, the model's \(\tilde S\) equals \(u_\tau/(\kappa y)\) at every height"] = \
        sp.simplify(S - ut / (kap * y)) == 0
    r = sp.simplify(nt / (S * kap ** 2 * y ** 2))
    checks[r"then \(r = \tilde\nu/(\tilde S\kappa^2 d^2) = 1\), so \(g = 1\) and \(f_w = 1\)"] = sp.simplify(r - 1) == 0
    fw = 1
    production = cb1 * (1 - ft2) * S * nt
    destruction = (cw1 * fw - cb1 / kap ** 2 * ft2) * (nt / y) ** 2
    diffusion = (sp.diff((nu + nt) * sp.diff(nt, y), y) + cb2 * sp.diff(nt, y) ** 2) / sig
    checks[r"the model's transport equation is satisfied exactly (production \(-\) destruction \(+\) diffusion \(= 0\)), for any value of \(f_{t2}\)"] = \
        sp.simplify(production - destruction + diffusion) == 0
    yp = sp.Symbol("y^+", positive=True)
    slope = 1 / (1 + kap * yp * (kap * yp) ** 3 / ((kap * yp) ** 3 + cv1 ** 3))
    checks[r"at the wall the profile is \(u^+ = y^+\): \(du^+/dy^+ \to 1\) as \(y^+ \to 0\)"] = sp.limit(slope, yp, 0) == 1
    checks[r"far from the wall it is logarithmic with slope \(1/\kappa\): \(y^+\,du^+/dy^+ \to 1/\kappa\)"] = \
        sp.simplify(sp.limit(yp * slope, yp, sp.oo) - 1 / kap) == 0
    pc = page_constants()
    num = {k: float(v) for k, v in CONSTANTS.items()}
    checks["the constants used here are the ones printed on NASA's page of the model"] = \
        all(pc[k] is not None and abs(pc[k] - num[k]) < 1e-12 for k in num)
    f3 = 3.0 ** 3 / (3.0 ** 3 + num["cv1"] ** 3)
    checks[r"the inflow value \(\tilde\nu = 3\nu\) gives \(\nu_t = 3 f_{v1}(3)\,\nu\), the value NASA's page prints for it"] = \
        pc["nut_far_low"] is not None and abs(3 * f3 - pc["nut_far_low"]) < 5e-7
    if not all(checks.values()):
        raise AssertionError(f"Theory self-check failed: {checks}")
    return checks


def inner_slope(yplus, kappa=CONSTANTS["kappa"], cv1=CONSTANTS["cv1"]):
    """du+/dy+ of the model's inner solution: 1 / (1 + chi f_v1(chi)) with chi = kappa y+."""
    chi = kappa * np.asarray(yplus, float)
    return 1.0 / (1.0 + chi * chi ** 3 / (chi ** 3 + cv1 ** 3))


def inner_law(yplus):
    """u+ of the model's inner solution at the given y+ (any order), by integrating inner_slope from the wall."""
    yp = np.atleast_1d(np.asarray(yplus, float))
    order = np.argsort(yp)
    out = np.empty_like(yp)
    last_y, last_u = 0.0, 0.0
    for k in order:
        seg, _ = integrate.quad(inner_slope, last_y, yp[k], epsabs=1e-13, epsrel=1e-12, limit=200)
        last_u += seg
        last_y = yp[k]
        out[k] = last_u
    return out


def log_intercept(kappa=CONSTANTS["kappa"]):
    """B of the model's own log law, u+ -> ln(y+)/kappa + B: the limit of u+ - ln(y+)/kappa, evaluated at two
    large y+ (their difference shows the limit is reached)."""
    b = [float(inner_law(yp)[0] - math.log(yp) / kappa) for yp in (1e5, 1e6)]
    return b[1], abs(b[1] - b[0])


def evaluate(G, ref, grids):
    """Steps, derived quantities with hand-checks, self-checks. grids: first cell heights, coarse to fine."""
    checks = derive()
    c = {k: float(v) for k, v in CONSTANTS.items()}
    U, nu, L, x1, x2, xo = G["U_inf"], G["nu"], G["L"], G["x_cf"], G["x_2"], G["x_out"]
    cf_ref = ref["incompressible"]["cf_at_station"]["FUN3D incompressible SA"][0]["cf"]
    B, dB = log_intercept()
    law = ref["profiles"]["files"]["u+y+theory.dat"]
    m = re.search(r"kappa = ([0-9.]+), B = ([0-9.]+)", law)
    kap_w, B_w = float(m.group(1)), float(m.group(2))

    def d(key, label, symbol, formula, value, unit_tex, why, params, check):
        return dict(key=key, label=label, symbol=symbol, formula=formula, value=value, value_tex=tex_num(value),
                    unit_tex=unit_tex, why=why, params=params, check=check)

    pU = dict(tex=r"U_\infty", meaning="free-stream speed", value=U, unit=r"\mathrm{m/s}")
    pnu = dict(tex=r"\nu", meaning="kinematic viscosity", value=nu, unit=r"\mathrm{m^2/s}")
    pL = dict(tex="L", meaning="reference length", value=L, unit=r"\mathrm{m}")
    u_tau = U * math.sqrt(cf_ref / 2)
    cw1 = c["cb1"] / c["kappa"] ** 2 + (1 + c["cb2"]) / c["sigma"]
    fv1_3 = 27.0 / (27.0 + c["cv1"] ** 3)
    derived = [
        d("Re_L", "Reynolds number per reference length", r"Re_L", r"Re_L = \frac{U_\infty L}{\nu}", U * L / nu, "",
          "Compares the inertia of the stream with friction. It is the only parameter of the incompressible problem; "
          "the Reynolds number based on the distance from the leading edge, \\(Re_x = Re_L\\,x/L\\), says how far the "
          "boundary layer has developed.", [pU, pL, pnu],
          rf"Re_L = \frac{{{tex_num(U)} \cdot {tex_num(L)}}}{{{tex_num(nu)}}} = {tex_num(U * L / nu)}"),
        d("Re_x1", "Reynolds number at the first station", r"Re_{x_1}", r"Re_{x_1} = \frac{U_\infty x_1}{\nu}", U * x1 / nu, "",
          "Where NASA tabulates the skin friction.",
          [pU, dict(tex="x_1", meaning="first station", value=x1, unit=r"\mathrm{m}"), pnu],
          rf"Re_{{x_1}} = \frac{{{tex_num(U)} \cdot {tex_num(x1, 6)}}}{{{tex_num(nu)}}} = {tex_num(U * x1 / nu)}"),
        d("Re_out", "Reynolds number at the end of the plate", r"Re_{x_{\mathrm{out}}}",
          r"Re_{x_{\mathrm{out}}} = \frac{U_\infty x_{\mathrm{out}}}{\nu}", U * xo / nu, "",
          "The largest value on the plate; the problem statement quotes it.",
          [pU, dict(tex=r"x_{\mathrm{out}}", meaning="end of the plate", value=xo, unit=r"\mathrm{m}"), pnu],
          rf"Re_{{x_{{\mathrm{{out}}}}}} = \frac{{{tex_num(U)} \cdot {tex_num(xo)}}}{{{tex_num(nu)}}} = {tex_num(U * xo / nu)}"),
        d("cw1", "Model constant that closes the wall layer", r"c_{w1}",
          r"c_{w1} = \frac{c_{b1}}{\kappa^2} + \frac{1 + c_{b2}}{\sigma}", cw1, "",
          "Not a free constant: this value makes production, destruction and diffusion of the model cancel in the "
          "wall layer (third self-check below). A code with another value cannot give the model's law of the wall.",
          [dict(tex="c_{b1}", meaning="production constant", value=c["cb1"], unit=""),
           dict(tex=r"\kappa", meaning="von Kármán constant", value=c["kappa"], unit=""),
           dict(tex="c_{b2}", meaning="diffusion constant", value=c["cb2"], unit=""),
           dict(tex=r"\sigma", meaning="diffusion constant (2/3)", value=c["sigma"], unit="")],
          rf"c_{{w1}} = \frac{{{tex_num(c['cb1'])}}}{{{tex_num(c['kappa'])}^2}} + \frac{{1 + {tex_num(c['cb2'])}}}{{2/3}} = {tex_num(cw1, 6)}"),
        d("nut_in", "Eddy viscosity of the inflow", r"\nu_{t,\infty}/\nu",
          r"\frac{\nu_{t,\infty}}{\nu} = \chi f_{v1}(\chi),\quad f_{v1} = \frac{\chi^3}{\chi^3 + c_{v1}^3},\quad \chi = \frac{\tilde\nu_\infty}{\nu}",
          3 * fv1_3, "",
          "The free stream carries a little modelled turbulence, enough for the boundary layer to become turbulent "
          "at once. The largest eddy viscosity in the boundary layer is about a thousand times this.",
          [dict(tex=r"\chi", meaning="turbulence variable of the inflow over the viscosity (given)", value=G["nut_in"], unit=""),
           dict(tex="c_{v1}", meaning="model constant", value=c["cv1"], unit="")],
          rf"\frac{{\nu_{{t,\infty}}}}{{\nu}} = 3 \cdot \frac{{27}}{{27 + {tex_num(c['cv1'])}^3}} = {tex_num(3 * fv1_3, 6)}"),
        d("B_sa", "Intercept of the model's own log law", r"B_{\mathrm{SA}}",
          r"B_{\mathrm{SA}} = \lim_{y^+\to\infty}\left[\int_0^{y^+}\frac{dy'}{1 + \kappa y' f_{v1}(\kappa y')} - \frac{\ln y^+}{\kappa}\right]",
          B, "",
          f"The model's velocity profile far above the viscous layer is \\(u^+ = \\ln(y^+)/\\kappa + B_{{\\mathrm{{SA}}}}\\). "
          f"The law of the wall NASA plots uses \\(\\kappa = {kap_w}\\), \\(B = {B_w}\\) (White 1974), so the model sits "
          f"\\({B - B_w:+.2f}\\) above it in the log layer, by construction and not through any numerical error. "
          f"The integral is evaluated numerically; its values at \\(y^+ = 10^5\\) and \\(10^6\\) differ by \\({tex_num(dB, 1)}\\).",
          [dict(tex=r"\kappa", meaning="von Kármán constant", value=c["kappa"], unit=""),
           dict(tex="c_{v1}", meaning="model constant in \\(f_{v1}\\)", value=c["cv1"], unit="")],
          rf"B_{{\mathrm{{SA}}}} = {B:.4f}\ \ \text{{(integrated to }} y^+ = 10^6\text{{)}}"),
        d("u_tau", "Friction velocity at the first station (from NASA's skin friction)", r"u_\tau",
          r"u_\tau = U_\infty\sqrt{C_f/2}", u_tau, r"\mathrm{m/s}",
          "The velocity scale of the wall layer. Used here only to size the grids before any run: it turns the "
          "height of the first cell into wall units.",
          [pU, dict(tex="C_f", meaning="skin friction at \\(x_1\\), NASA's incompressible result on the finest grid",
                    value=cf_ref, unit="")],
          rf"u_\tau = {tex_num(U)}\sqrt{{{tex_num(cf_ref, 6)}/2}} = {tex_num(u_tau)}\ \mathrm{{m/s}}"),
        d("yplus_coarse", "Wall-cell centre in wall units, coarsest grid", r"y^+_1",
          r"y^+_1 = \frac{(\Delta y_1/2)\,u_\tau}{\nu}", 0.5 * grids[0] * u_tau / nu, "",
          "The model is integrated to the wall (no wall functions), which needs the first cell centre at \\(y^+\\) of "
          "about 1 or below. Even NASA's coarsest grid meets this at the station.",
          [dict(tex=r"\Delta y_1", meaning="height of the wall cell, coarsest grid", value=grids[0], unit=r"\mathrm{m}"),
           dict(tex=r"u_\tau", meaning="friction velocity (above)", value=u_tau, unit=r"\mathrm{m/s}"), pnu],
          rf"y^+_1 = \frac{{{tex_num(0.5 * grids[0])} \cdot {tex_num(u_tau)}}}{{{tex_num(nu)}}} = {tex_num(0.5 * grids[0] * u_tau / nu, 3)}"),
        d("yplus_fine", "Wall-cell centre in wall units, finest grid", r"y^+_1",
          r"y^+_1 = \frac{(\Delta y_1/2)\,u_\tau}{\nu}", 0.5 * grids[-1] * u_tau / nu, "",
          "Twenty times closer to the wall than needed: on the finest grid the wall layer is resolved far beyond "
          "what the model requires, so the wall shear stress there is not limited by the wall spacing.",
          [dict(tex=r"\Delta y_1", meaning="height of the wall cell, finest grid", value=grids[-1], unit=r"\mathrm{m}"),
           dict(tex=r"u_\tau", meaning="friction velocity (above)", value=u_tau, unit=r"\mathrm{m/s}"), pnu],
          rf"y^+_1 = \frac{{{tex_num(0.5 * grids[-1])} \cdot {tex_num(u_tau)}}}{{{tex_num(nu)}}} = {tex_num(0.5 * grids[-1] * u_tau / nu, 3)}"),
    ]
    steps = [
        dict(label="Governing equations: time-averaged (RANS) momentum with an eddy viscosity",
             latex=r"\nabla\cdot\mathbf{u} = 0,\qquad (\mathbf{u}\cdot\nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + "
                   r"\nabla\cdot\left[(\nu + \nu_t)\left(\nabla\mathbf{u} + \nabla\mathbf{u}^T\right)\right],\qquad \nu_t = \tilde\nu f_{v1}",
             note="The turbulent fluctuations are not computed. Their effect on the mean flow is modelled as extra "
                  "mixing, an eddy viscosity \\(\\nu_t\\) that varies in space and is far larger than \\(\\nu\\) inside "
                  "the boundary layer. The equations are nonlinear and coupled to the model: no exact solution of the whole flow."),
        dict(label="The Spalart-Allmaras model: one transport equation for the variable \\(\\tilde\\nu\\) (NASA's standard form, SA)",
             latex=r"\mathbf{u}\cdot\nabla\tilde\nu = c_{b1}(1 - f_{t2})\tilde S\tilde\nu - \left[c_{w1}f_w - \frac{c_{b1}}{\kappa^2}f_{t2}\right]"
                   r"\left(\frac{\tilde\nu}{d}\right)^2 + \frac{1}{\sigma}\left[\nabla\cdot\left((\nu + \tilde\nu)\nabla\tilde\nu\right) + c_{b2}|\nabla\tilde\nu|^2\right]",
             note="Production by the mean shear, destruction near walls (\\(d\\) is the distance to the wall) and "
                  "diffusion. \\(\\tilde S = \\Omega + \\tilde\\nu f_{v2}/(\\kappa^2 d^2)\\) with the vorticity \\(\\Omega\\); "
                  "the functions \\(f_{v1}, f_{v2}, f_w, f_{t2}\\) and the constants are those of NASA's page "
                  "(checked by code against the saved page)."),
        dict(label="Wall layer: the total shear stress is constant close to the wall",
             latex=r"(\nu + \nu_t)\frac{du}{dy} = \frac{\tau_w}{\rho} = u_\tau^2\quad\Longrightarrow\quad "
                   r"\frac{du^+}{dy^+} = \frac{1}{1 + \chi f_{v1}(\chi)},\qquad \chi = \frac{\tilde\nu}{\nu}",
             note="With no pressure gradient, the momentum carried by the mean flow is negligible near the wall, so the "
                  "stress the wall feels is passed unchanged through the first few per cent of the boundary layer."),
        dict(label="The model's exact solution in that layer, and its law of the wall",
             latex=r"\tilde\nu = \kappa u_\tau y\ \ (\chi = \kappa y^+),\qquad u^+(y^+) = \int_0^{y^+}\frac{dy'}{1 + \kappa y' f_{v1}(\kappa y')}",
             note="The model was built so that \\(\\tilde\\nu\\) is a straight line from the wall up; SymPy confirms it "
                  "satisfies the transport equation exactly, with or without \\(f_{t2}\\). The integral gives "
                  "\\(u^+ = y^+\\) at the wall and a log law above, with the intercept \\(B_{\\mathrm{SA}}\\) listed below. "
                  "This is what the simulated profile must follow for small \\(y/\\delta\\), on any grid fine enough: a check "
                  "of the implementation that needs no other code."),
        dict(label="Drag from the momentum the stream loses (the physical check of every run)",
             latex=r"\frac{F}{\rho} = \int_0^{x_{\mathrm{out}}}\frac{\tau_w}{\rho}\,dx = -\oint_{\text{outer sides}}u\,(\mathbf{u}\cdot\mathbf{n})\,ds"
                   r" + \int_{\text{inflow}}\frac{p}{\rho}\,dy\quad\text{(+ small viscous terms)}",
             note="Integrating the momentum equation over the whole domain: the friction force on the plate equals the "
                  "x-momentum that leaves through the outflow and the top minus what entered, corrected for the pressure "
                  "on the inflow side (the outflow and the top are at the free-stream pressure, 0). The two sides are "
                  "computed independently, the left from the wall shear stress and the right from the fluxes on the "
                  "outer boundaries, so their agreement checks convergence and the force extraction together."),
    ]
    return dict(steps=steps, derived=derived, self_checks=checks, B_sa=B, kappa_white=kap_w, B_white=B_w,
                u_tau_est=u_tau)
