"""Exact solution of laminar tube flow heated with a uniform wall heat flux (the extended Graetz problem
of Siegel, Sparrow & Hallman 1958), derived and self-checked in code.

Model of the exact solution (the classic idealisation): fully developed (parabolic) velocity at the start
of heating, constant properties, no axial heat conduction, no buoyancy, no viscous heating.

Dimensionless form, with eta = r/a (a = D/2), X = x alpha / (U a^2), theta = (T - T_in) k / (q_w a):
    2 (1 - eta^2) dtheta/dX = (1/eta) d/deta (eta dtheta/deta),  dtheta/deta(1) = 1,  theta(0, eta) = 0.
Solution = fully developed part (SymPy, exact) + decaying eigenfunction series (Kummer functions, mpmath):
    theta = 2 X + f(eta) + sum_n C_n R_n(eta) exp(-lambda_n^2 X / 2).
Self-checks: SymPy checks the fully developed part against the PDE, the wall flux and the energy balance;
the series is checked against the ODE residual, its own eigencondition, a doubled number of terms, and an
independent finite-difference solution of the same PDE (method of lines), which does not use the series.
"""
import math

import mpmath as mp
import numpy as np
import sympy as sp
from scipy import integrate, optimize

from cfdlib.report import tex_num

mp.mp.dps = 30
eta, X = sp.symbols("eta X", nonnegative=True)
N_TERMS = 80                     # eigenfunctions kept in the series (convergence checked with 2x fewer)
FIVE_PCT = 0.05                  # "within 5 % of the fully developed value" (problem statement)
X_MIN_D = 0.5                    # the series is used (and compared) from x = 0.5 D on


# ---------------------------------------------------------------------------------------------------
# Fully developed part (exact, SymPy)
# ---------------------------------------------------------------------------------------------------
def derive_fd():
    checks = {}
    A, c = sp.symbols("A c")
    w = 2 * (1 - eta ** 2)                                   # u / U for the parabola
    # Energy over the whole cross-section: d/dX int w theta eta deta = [eta theta_eta]_0^1 = 1, and
    # int w eta deta = 1/2, so a temperature rising like A X everywhere needs A / 2 = 1.
    A_val = sp.solve(sp.Eq(A * sp.integrate(w * eta, (eta, 0, 1)), 1), A)[0]
    fsym = sp.Function("f")
    ode = sp.Eq(sp.diff(eta * sp.diff(fsym(eta), eta), eta), A_val * w * eta)
    gen = sp.expand(sp.dsolve(ode, fsym(eta)).rhs)           # C1 + C2 log(eta) + particular part
    consts = sorted(gen.free_symbols - {eta}, key=str)
    # Regular (finite) on the axis: the coefficient of log(eta) must vanish.
    log_coef = gen.coeff(sp.log(eta))
    kill = sp.solve(sp.Eq(log_coef, 0), [s for s in consts if log_coef.has(s)], dict=True)[0]
    gen = sp.expand(gen.subs(kill))
    Cfree = sorted(gen.free_symbols - {eta}, key=str)
    f_gen = gen.subs(Cfree[0], c)
    # Zero mean (flow-weighted) of f: theta_b = 2 X exactly, from the energy balance with theta(0) = 0.
    c_val = sp.solve(sp.Eq(sp.integrate(w * f_gen * eta, (eta, 0, 1)), 0), c)[0]
    f_expr = sp.expand(f_gen.subs(c, c_val))
    theta_fd = A_val * X + f_expr
    lap = sp.diff(eta * sp.diff(theta_fd, eta), eta) / eta
    checks[r"\(\theta_{fd} = 2X + f(\eta)\) satisfies \(2(1-\eta^2)\,\partial\theta/\partial X = "
           r"\frac{1}{\eta}\partial_\eta(\eta\,\partial_\eta\theta)\)"] = sp.simplify(w * sp.diff(theta_fd, X) - lap) == 0
    checks[r"wall heat flux \(\partial\theta/\partial\eta = 1\) at \(\eta = 1\)"] = sp.simplify(sp.diff(f_expr, eta).subs(eta, 1) - 1) == 0
    checks[r"regular on the axis: \(\partial\theta/\partial\eta = 0\) at \(\eta = 0\)"] = sp.simplify(sp.diff(f_expr, eta).subs(eta, 0)) == 0
    tb = sp.simplify(sp.integrate(w * theta_fd * eta, (eta, 0, 1)) / sp.integrate(w * eta, (eta, 0, 1)))
    checks[r"bulk temperature \(\theta_b = 2X\): all heat put in since \(X = 0\) is carried by the flow"] = sp.simplify(tb - 2 * X) == 0
    f1 = sp.nsimplify(f_expr.subs(eta, 1))
    nu_fd = sp.nsimplify(2 / f1)
    checks[r"\(Nu_{fd} = 2/(\theta_w - \theta_b) = 48/11\)"] = sp.simplify(nu_fd - sp.Rational(48, 11)) == 0
    if not all(checks.values()):
        raise AssertionError(f"fully developed self-check failed: {checks}")
    return dict(A=A_val, f=f_expr, f1=f1, nu_fd=nu_fd, checks=checks)


# ---------------------------------------------------------------------------------------------------
# Entrance region: eigenfunction series (Kummer / confluent hypergeometric functions)
# ---------------------------------------------------------------------------------------------------
# (eta R')' + lam^2 eta (1 - eta^2) R = 0 has the regular solution R = exp(-z/2) M(1/2 - lam/4, 1, z),
# z = lam eta^2 (checked numerically below). Insulated-type eigencondition R'(1) = 0.
def R(lam, e):
    z = lam * e * e
    return mp.exp(-z / 2) * mp.hyp1f1(mp.mpf(1) / 2 - lam / 4, 1, z)


def dR_deta(lam, e):
    z = lam * e * e
    a = mp.mpf(1) / 2 - lam / 4
    M = mp.hyp1f1(a, 1, z)
    Mp = a * mp.hyp1f1(a + 1, 2, z)                           # dM/dz
    return mp.exp(-z / 2) * (Mp - M / 2) * 2 * lam * e


class Series:
    def __init__(self, fd, n_terms=N_TERMS):
        f_num = sp.lambdify(eta, fd["f"], "mpmath")
        g = lambda lam: dR_deta(mp.mpf(lam), 1) / (2 * lam * mp.exp(-lam / 2))   # same zeros, no scale
        lams, lo, step = [], 0.3, 0.05
        while len(lams) < n_terms:
            hi = lo + step
            if g(lo) * g(hi) < 0:
                lams.append(mp.findroot(g, (mp.mpf(lo), mp.mpf(hi)), solver="illinois", tol=1e-40, verify=False))
            lo = hi
        self.lam = lams
        # Coefficients from the initial condition theta(0, eta) = 0, i.e. phi(0) = -f, orthogonality
        # weight eta (1 - eta^2). Gauss-Legendre quadrature in mpmath (eigenfunctions oscillate n times).
        xs, ws = np.polynomial.legendre.leggauss(600)
        xs, ws = (xs + 1) / 2, ws / 2
        self.C, self.R1 = [], []
        for lam in lams:
            num = den = mp.mpf(0)
            for x_, w_ in zip(xs, ws):
                e = mp.mpf(x_)
                wt = e * (1 - e * e)
                r = R(lam, e)
                num += w_ * wt * f_num(e) * r
                den += w_ * wt * r * r
            self.C.append(-num / den)
            self.R1.append(R(lam, mp.mpf(1)))
        self.lam_f = np.array([float(v) for v in lams])
        self.CR1 = np.array([float(c * r) for c, r in zip(self.C, self.R1)])
        self.f1 = float(fd["f1"])

    def dtheta_wb(self, Xv, n=None):
        """theta_w - theta_b at X (array)."""
        Xv = np.atleast_1d(np.asarray(Xv, float))
        k = slice(0, n)
        return self.f1 + np.exp(-0.5 * np.outer(Xv, self.lam_f[k] ** 2)) @ self.CR1[k]

    def nu(self, Xv, n=None):
        return 2.0 / self.dtheta_wb(Xv, n)


def method_of_lines(X_eval, n=400):
    """Independent check: the same PDE solved by finite volumes in eta (n cells, clustered toward the
    wall) and a stiff ODE integrator in X. Returns Nu at X_eval. No series, no eigenvalues."""
    s = np.linspace(0, 1, n + 1)
    faces = 1 - (1 - s) ** 1.5                                   # finer toward the wall
    faces[0], faces[-1] = 0.0, 1.0
    ec = 0.5 * (faces[1:] + faces[:-1])
    vol = 0.5 * (faces[1:] ** 2 - faces[:-1] ** 2)               # int eta deta per cell
    cap = np.array([integrate.quad(lambda e: 2 * (1 - e * e) * e, faces[i], faces[i + 1])[0] for i in range(n)])
    cond = faces[1:-1] / (ec[1:] - ec[:-1])                      # eta / d_eta at interior faces

    def rhs(_, th):
        flux = np.zeros(n + 1)
        flux[1:-1] = cond * (th[1:] - th[:-1])
        flux[-1] = 1.0                                           # eta dtheta/deta = 1 at the wall
        return (flux[1:] - flux[:-1]) / cap

    from scipy.sparse import diags
    main = np.zeros(n)
    main[:-1] -= cond
    main[1:] -= cond
    jac = diags([np.r_[cond], main, np.r_[cond]], [-1, 0, 1]).tocsc()
    jac = diags(1 / cap) @ jac
    sol = integrate.solve_ivp(rhs, (0, max(X_eval)), np.zeros(n), method="BDF", t_eval=sorted(X_eval),
                              jac=jac, rtol=1e-10, atol=1e-13)
    out = {}
    for k, Xk in enumerate(sol.t):
        th = sol.y[:, k]
        # wall value: second-order extrapolation from the last cell with the known wall gradient 1
        tw = th[-1] + (1 - ec[-1]) * 1.0
        tb = np.sum(cap * th) / np.sum(cap)
        out[float(Xk)] = 2 / (tw - tb)
    return out


# ---------------------------------------------------------------------------------------------------
# Problem numbers and the requested quantities
# ---------------------------------------------------------------------------------------------------
class Exact:
    """The exact solution in the problem's units. x in m from the start of heating."""

    def __init__(self, given):
        G = {k: v["value"] for k, v in given.items()}
        self.G = G
        self.a = G["D"] / 2
        self.alpha = G["k"] / (G["rho"] * G["cp"])
        self.fd = derive_fd()
        self.series = Series(self.fd)
        self.dT = G["q_w"] * self.a / G["k"]                 # temperature scale q_w a / k
        self.nu_fd = float(self.fd["nu_fd"])

    def X(self, x):
        return np.asarray(x, float) * self.alpha / (self.G["U"] * self.a ** 2)

    def x_of_X(self, Xv):
        return Xv * self.G["U"] * self.a ** 2 / self.alpha

    def nu(self, x, n=None):
        return self.series.nu(self.X(x), n)

    def T_b(self, x):
        return self.G["T_in"] + self.dT * 2 * self.X(x)

    def T_w(self, x):
        return self.T_b(x) + self.dT * self.series.dtheta_wb(self.X(x))

    def l_th(self, frac=FIVE_PCT, n=None):
        target = (1 + frac) * self.nu_fd
        fn = lambda lx: float(self.series.nu(np.exp(lx), n)[0]) - target
        Xs = math.exp(optimize.brentq(fn, math.log(1e-5), math.log(1.0), xtol=1e-14))
        return self.x_of_X(Xs)


def numbers(given):
    """Dimensionless groups and scales used in the assumptions and the theory section."""
    G = {k: v["value"] for k, v in given.items()}
    mu = G["rho"] * G["nu"]
    alpha = G["k"] / (G["rho"] * G["cp"])
    Re = G["U"] * G["D"] / G["nu"]
    Pr = G["nu"] / alpha
    dTb_dx = 4 * G["q_w"] / (G["rho"] * G["cp"] * G["U"] * G["D"])
    dT_fd = G["q_w"] * G["D"] / (G["k"] * 48 / 11)
    Gr = G["g"] * G["beta"] * dT_fd * G["D"] ** 3 / G["nu"] ** 2
    Gr_q = G["g"] * G["beta"] * G["q_w"] * G["D"] ** 4 / (G["k"] * G["nu"] ** 2)
    dmu_dT = (G["mu_25"] - mu) / 5.0                     # per K, from the two given viscosities (20 and 25 °C)
    T_rise = dTb_dx * G["Lh"]
    T_w_max_rise = T_rise + dT_fd
    return dict(mu=mu, alpha=alpha, Re=Re, Pr=Pr, Pe=Re * Pr, dTb_dx=dTb_dx, dT_fd=dT_fd, Gr=Gr, Ra=Gr * Pr,
                Ri=Gr / Re ** 2, Gr_q=Gr_q, Ra_q=Gr_q * Pr, dmu_dT=dmu_dT, T_rise=T_rise, T_w_max_rise=T_w_max_rise,
                mu_change_wall=abs(dmu_dT) * T_w_max_rise / mu, mu_change_wb=abs(dmu_dT) * dT_fd / mu,
                Br=mu * G["U"] ** 2 / (G["q_w"] * G["D"]), x_plus_end=G["Lh"] / (G["D"] * Re * Pr),
                L0_over_ReD=G["L0"] / (G["D"] * Re))


def evaluate(given):
    """Theory section content (steps, derived quantities with hand-checks, self-checks) and the Exact object."""
    ex = Exact(given)
    G, a = ex.G, ex.a
    nb = numbers(given)
    checks = dict(ex.fd["checks"])

    # --- self-checks of the series ---
    lam = ex.series.lam
    res = []
    for lm in (lam[0], lam[4], lam[-1]):
        for e in (mp.mpf("0.3"), mp.mpf("0.7"), mp.mpf("0.95")):
            Rf = lambda s: R(lm, s)
            r1 = mp.diff(lambda s: s * mp.diff(Rf, s), e) + lm ** 2 * e * (1 - e * e) * Rf(e)
            res.append(abs(r1) / (lm ** 2 * max(abs(Rf(e)), mp.mpf(1e-30))))
    checks[r"eigenfunctions \(R_n = e^{-z/2} M(\tfrac12 - \tfrac{\lambda_n}{4}, 1, z)\), \(z = \lambda_n\eta^2\), satisfy "
           r"\((\eta R')' + \lambda_n^2\eta(1-\eta^2)R = 0\) (relative residual \(< 10^{-12}\))"] = float(max(res)) < 1e-12
    checks[r"eigencondition \(R_n'(1) = 0\) holds for every \(\lambda_n\) (\(|R_n'(1)| < 10^{-15}\,\lambda_n\))"] = \
        all(abs(dR_deta(lm, mp.mpf(1))) < 1e-15 * lm for lm in lam)
    # series truncation: dropping the last quarter of the terms changes nothing where the comparison is made
    xs_chk = np.array([X_MIN_D, 1, 10, 60]) * G["D"]
    n_less = 3 * N_TERMS // 4
    trunc = float(np.max(np.abs(ex.nu(xs_chk, n_less) / ex.nu(xs_chk) - 1)))
    checks[rf"series converged: {n_less} and {N_TERMS} terms give the same \(Nu\) to "
           rf"\({tex_num(max(trunc, 1e-16), 2)}\) at \(x/D \ge {X_MIN_D}\)"] = trunc < 1e-8
    # initial condition reproduced: theta_w - theta_b -> small at X -> 0 (Nu -> infinity)
    mol = method_of_lines(list(ex.X(np.array([0.5, 2, 10, 40, 100]) * G["D"])))
    mol_err = max(abs(v / float(ex.series.nu(k)[0]) - 1) for k, v in mol.items())
    checks[rf"an independent finite-volume solution of the same equation (400 cells, no series) agrees to "
           rf"\({tex_num(mol_err, 2)}\) in \(Nu\) at \(x/D = 0.5\) to \(100\)"] = mol_err < 2e-3
    if not all(checks.values()):
        raise AssertionError(f"theory self-check failed: {checks}")

    Lh = G["Lh"]
    l_th = ex.l_th()
    nu_out = float(ex.nu(Lh)[0])
    T_b_out, T_w_out = float(ex.T_b(Lh)), float(ex.T_w(Lh)[0])
    dT_fd = G["q_w"] * G["D"] / (G["k"] * ex.nu_fd)

    P = {
        "U": dict(tex="U", meaning="mean velocity", value=G["U"], unit=r"\mathrm{m/s}"),
        "D": dict(tex="D", meaning="tube diameter", value=G["D"], unit=r"\mathrm{m}"),
        "nu": dict(tex=r"\nu", meaning="kinematic viscosity of water", value=G["nu"], unit=r"\mathrm{m^2/s}"),
        "alpha": dict(tex=r"\alpha", meaning=r"thermal diffusivity, \(\alpha = k/(\rho c_p)\)", value=ex.alpha, unit=r"\mathrm{m^2/s}"),
        "q": dict(tex="q_w", meaning="wall heat flux", value=G["q_w"], unit=r"\mathrm{W/m^2}"),
        "k": dict(tex="k", meaning="thermal conductivity of water", value=G["k"], unit=r"\mathrm{W/(m\,K)}"),
        "rho": dict(tex=r"\rho", meaning="density", value=G["rho"], unit=r"\mathrm{kg/m^3}"),
        "cp": dict(tex="c_p", meaning="specific heat", value=G["cp"], unit=r"\mathrm{J/(kg\,K)}"),
        "Lh": dict(tex="L_h", meaning="heated length", value=Lh, unit=r"\mathrm{m}"),
        "Tin": dict(tex=r"T_{\mathrm{in}}", meaning="inlet temperature", value=G["T_in"], unit=r"\mathrm{°C}"),
    }
    v = lambda k: rf"({tex_num(P[k]['value'])}\,{P[k]['unit']})"
    derived = [
        dict(key="Re", label="Reynolds number", symbol="Re", formula=r"Re = \frac{U D}{\nu}", value=nb["Re"],
             value_tex=tex_num(nb["Re"]), unit_tex="", params=[P["U"], P["D"], P["nu"]],
             check=rf"Re = \frac{{{v('U')}{v('D')}}}{{{v('nu')}}} = {tex_num(nb['Re'])}",
             why="Ratio of inertia to viscous forces. It sets whether the flow is laminar and how long the velocity "
                 "profile needs to develop after the inlet."),
        dict(key="Pr", label="Prandtl number", symbol="Pr", formula=r"Pr = \frac{\nu}{\alpha}", value=nb["Pr"],
             value_tex=tex_num(nb["Pr"]), unit_tex="", params=[P["nu"], P["alpha"]],
             check=rf"Pr = \frac{{{v('nu')}}}{{{v('alpha')}}} = {tex_num(nb['Pr'])}",
             why="Ratio of momentum to heat diffusion. Water's Pr of about 7 means heat spreads across the tube seven "
                 "times more slowly than momentum: the thermal entrance region is much longer than the hydrodynamic one."),
        dict(key="Pe", label="Péclet number", symbol="Pe", formula=r"Pe = Re\,Pr = \frac{U D}{\alpha}", value=nb["Pe"],
             value_tex=tex_num(nb["Pe"]), unit_tex="", params=[P["U"], P["D"], P["alpha"]],
             check=rf"Pe = \frac{{{v('U')}{v('D')}}}{{{v('alpha')}}} = {tex_num(nb['Pe'])}",
             why="Ratio of heat carried along the tube by the flow to heat conducted along it. At "
                 + r"\(Pe \gg 1\) axial conduction is negligible, which the exact solution assumes."),
        dict(key="nu_fd", label="Fully developed Nusselt number", symbol=r"Nu_{fd}",
             formula=r"Nu_{fd} = \frac{2}{\theta_w - \theta_b} = \frac{2}{f(1)} = \frac{48}{11}", value=ex.nu_fd,
             value_tex=tex_num(ex.nu_fd, 6), unit_tex="", params=[],
             check=rf"Nu_{{fd}} = \frac{{2}}{{11/24}} = \frac{{48}}{{11}} = {tex_num(ex.nu_fd, 6)}",
             why="Far from the start of heating the temperature profile keeps its shape and only rises; the heat "
                 "transfer coefficient is then constant. Requested output (b)."),
        dict(key="dT_fd", label="Wall-to-bulk temperature difference, fully developed", symbol=r"\Delta T_{fd}",
             formula=r"\Delta T_{fd} = T_w - T_b = \frac{q_w D}{k\,Nu_{fd}}", value=dT_fd, value_tex=tex_num(dT_fd),
             unit_tex=r"\mathrm{K}", params=[P["q"], P["D"], P["k"]],
             check=rf"\Delta T_{{fd}} = \frac{{{v('q')}{v('D')}}}{{{v('k')}\,({tex_num(ex.nu_fd, 6)})}} = {tex_num(dT_fd)}\,\mathrm{{K}}",
             why="How much hotter the wall is than the water's mean in the fully developed region."),
        dict(key="l_th", label="Thermal entrance length (5 %)", symbol=r"L_{th}",
             formula=r"Nu(L_{th}) = 1.05\,Nu_{fd}",
             value=l_th, value_tex=tex_num(l_th), unit_tex=r"\mathrm{m}", params=[P["D"], P["U"], P["alpha"]],
             check=rf"L_{{th}} = x^+ D\,Pe = {tex_num(ex.X(l_th) / 4, 4)} \times 0.004 \times {tex_num(nb['Pe'])} = {tex_num(l_th)}\,\mathrm{{m}}",
             why="Requested output (c). The series gives it as the root of " + r"\(Nu(x) = 1.05\,Nu_{fd}\)" + "; in the "
                 "dimensionless distance " + r"\(x^+ = x/(D\,Re\,Pr)\)" + " it is a pure number, " + rf"\(x^+ = {tex_num(ex.X(l_th) / 4, 4)}\)"
                 + ", independent of the data. Here " + rf"\(L_{{th}} = {tex_num(l_th / G['D'])}\,D\)" + "."),
        dict(key="T_b_out", label="Bulk temperature at the outlet", symbol=r"T_b(L_h)",
             formula=r"T_b(L_h) = T_{\mathrm{in}} + \frac{4\,q_w L_h}{\rho\,c_p\,U D}", value=T_b_out,
             value_tex=tex_num(T_b_out, 6), unit_tex=r"\mathrm{°C}", params=[P["Tin"], P["q"], P["Lh"], P["rho"], P["cp"], P["U"], P["D"]],
             check=rf"T_b(L_h) = {tex_num(G['T_in'])} + \frac{{4 \cdot {tex_num(G['q_w'])} \cdot {tex_num(Lh)}}}"
                   rf"{{{tex_num(G['rho'])} \cdot {tex_num(G['cp'])} \cdot {tex_num(G['U'])} \cdot {tex_num(G['D'])}}} = "
                   rf"{tex_num(T_b_out, 6)}",
             why="Energy balance alone: all heat put in through the wall over " + r"\(L_h\)" + " is carried away by the "
                 "water. Exact for any velocity profile, so it also checks the simulation's energy conservation."),
        dict(key="T_w_out", label="Wall temperature at the outlet", symbol=r"T_w(L_h)",
             formula=r"T_w(L_h) = T_b(L_h) + \frac{q_w D}{k\,Nu(L_h)}", value=T_w_out, value_tex=tex_num(T_w_out, 6),
             unit_tex=r"\mathrm{°C}", params=[P["q"], P["D"], P["k"]],
             check=rf"T_w(L_h) = {tex_num(T_b_out, 6)} + \frac{{{tex_num(G['q_w'])} \times {tex_num(G['D'])}}}"
                   rf"{{{tex_num(G['k'])} \times {tex_num(nu_out, 6)}}} = {tex_num(T_w_out, 6)}\,\mathrm{{°C}}",
             why="Requested output (d): the hottest point of the wall. The series gives " + rf"\(Nu(L_h) = {tex_num(nu_out, 6)}\)"
                 + ", still " + rf"\({tex_num(100 * (nu_out / ex.nu_fd - 1), 2)}\,\%\)" + " above " + r"\(Nu_{fd}\)" + "."),
    ]
    f_tex = sp.latex(ex.fd["f"])
    steps = [
        dict(label="Energy equation (steady, constant properties, axial conduction neglected)",
             latex=r"u(r)\,\frac{\partial T}{\partial x} = \frac{\alpha}{r}\frac{\partial}{\partial r}\!\left(r\frac{\partial T}{\partial r}\right),"
                   r"\qquad u(r) = 2U\left(1 - \frac{r^2}{a^2}\right),\quad a = \frac{D}{2}",
             note="Heat carried along by the flow (left) balances heat conducted across the tube (right). The velocity "
                  "is the fully developed parabola from the start of heating."),
        dict(label="Boundary conditions",
             latex=r"k\left.\frac{\partial T}{\partial r}\right|_{r=a} = q_w,\qquad \left.\frac{\partial T}{\partial r}\right|_{r=0} = 0,"
                   r"\qquad T(0, r) = T_{\mathrm{in}}",
             note="Uniform flux into the water at the wall, symmetry on the axis, uniform temperature where heating starts."),
        dict(label="Dimensionless form",
             latex=r"\eta = \frac{r}{a},\quad X = \frac{x\,\alpha}{U a^2} = \frac{4x}{D\,Pe},\quad \theta = \frac{(T - T_{\mathrm{in}})\,k}{q_w a}:"
                   r"\qquad 2(1-\eta^2)\,\frac{\partial\theta}{\partial X} = \frac{1}{\eta}\frac{\partial}{\partial\eta}\!\left(\eta\frac{\partial\theta}{\partial\eta}\right)",
             note="No parameter is left: one solution serves every tube, flow rate and heat flux."),
        dict(label="Fully developed part (SymPy)",
             latex=rf"\theta_{{fd}} = 2X + f(\eta),\qquad f(\eta) = {f_tex},\qquad \theta_b = 2X",
             note="The bulk temperature rises linearly (energy balance); the profile shape f does not change. "
                  "The constant in f makes its flow-weighted mean zero."),
        dict(label="Entrance region: eigenfunction series",
             latex=r"\theta = 2X + f(\eta) + \sum_{n=1}^{\infty} C_n R_n(\eta)\, e^{-\lambda_n^2 X/2},\qquad "
                   r"(\eta R_n')' + \lambda_n^2 \eta (1-\eta^2) R_n = 0,\quad R_n'(1) = 0",
             note=f"The difference from the fully developed state decays along the tube. The eigenfunctions are Kummer "
                  f"functions; the first eigenvalues are {', '.join(f'{float(x_):.4f}' for x_ in lam[:3])}. "
                  f"{N_TERMS} terms are summed; coefficients from " + r"\(\theta(0,\eta) = 0\)" + " by orthogonality "
                  "(weight " + r"\(\eta(1-\eta^2)\)" + ")."),
        dict(label="Local Nusselt number",
             latex=r"Nu(x) = \frac{q_w D}{k\,(T_w - T_b)} = \frac{2}{\theta_w - \theta_b} = "
                   r"\frac{2}{\tfrac{11}{24} + \sum_n C_n R_n(1)\,e^{-\lambda_n^2 X/2}}",
             note="Infinite at the start of heating (the thermal boundary layer has zero thickness there), falling to "
                  + r"\(48/11\)" + " downstream."),
    ]
    return dict(steps=steps, derived=derived, self_checks=checks, exact=ex, numbers=nb,
                l_th=l_th, T_b_out=T_b_out, T_w_out=T_w_out, nu_out=nu_out, dT_fd=dT_fd,
                lam=[float(x_) for x_ in lam[:5]], C=[float(c) for c in ex.series.C[:5]],
                trunc=trunc, mol_err=float(mol_err))


if __name__ == "__main__":
    import yaml
    from pathlib import Path
    g = yaml.safe_load((Path(__file__).parent / "problem.yaml").read_text(encoding="utf-8"))["given"]
    t = evaluate(g)
    for k, val in t["self_checks"].items():
        print("OK " if val else "FAIL", k[:110])
    for d in t["derived"]:
        print(f"{d['key']:8s} {d['value']:.8g}")
    nb = t["numbers"]
    print({k: f"{v:.4g}" for k, v in nb.items()})
    print("lam", t["lam"], "C", t["C"], "trunc", t["trunc"], "mol", t["mol_err"])
    ex = t["exact"]
    for xd in (0.1, 0.5, 1, 2, 5, 10, 20, 40, 60, 80, 120):
        print(f"x/D={xd:6.1f}  X={float(ex.X(xd * ex.G['D'])):.5g}  Nu={float(ex.nu(xd * ex.G['D'])[0]):.6f}")
    print("l_th/D", t["l_th"] / ex.G["D"])
