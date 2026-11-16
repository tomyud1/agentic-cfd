"""Exact solution of the shock-tube (Riemann) problem for two gas mixtures whose heat capacities change with
temperature and whose compositions do not change (no reactions, no mixing: the border between them stays sharp).
Written for rung (b) of gpucfd's reacting ladder. NumPy and SciPy; the thermodynamic data comes from a
cfdlib.chem.Mechanism (or gpucfd.reacting.Gases), nothing else of the project is used.

    ex = MixtureRiemann(mech, left=dict(T=2500.0, p=3e6, Y=YL), right=dict(T=300.0, p=101325.0, Y=YR), x0=0.06)
    f = ex.sample(x, t)                      # point values: rho, u, p, T, Y (species first)
    f = ex.cell_averages(edges, t)           # averages as a finite-volume solver holds them
    ex.star, ex.waves                        # the states between the waves; the speeds of the waves

Method. Each side is one mixture. Across a shock the three jump conditions hold with the mixture's own enthalpy
h(T); along an expansion the entropy is constant, dT/dln p = R T / cp(T), and the velocity changes by
du = -+ dp / (rho c). The pressure between the waves is the one for which both sides give the same velocity
(found by bisection). Inside an expansion fan the state depends on x/t only.
Checks of this file itself are in gpucfd/tests/test_react_mix.py: its shock against cfdlib.chem.shock_frozen,
the entropy along its expansion, and the mass, momentum and energy of the whole solution.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


class Mixture:
    """One mixture of fixed composition: thermodynamic functions of temperature, for arrays of T."""

    def __init__(self, mech, Y):
        self.mech, self.Y = mech, np.asarray(Y, float)
        self.sigma = self.Y / mech.W                               # mol/kg
        self.R = float(self.sigma.sum()) * mech.conv.R            # gas constant, J/(kg K)

    def _a(self, T):
        m = self.mech
        return np.where((np.asarray(T, float)[..., None] >= m.T_mid)[..., None], m.a_hi, m.a_lo)   # (..., species, 7)

    def cp(self, T):
        a, T = self._a(T), np.asarray(T, float)[..., None]
        return (a[..., 0] + T * (a[..., 1] + T * (a[..., 2] + T * (a[..., 3] + T * a[..., 4])))) @ self.sigma * self.mech.conv.R

    def h(self, T):
        a, T = self._a(T), np.asarray(T, float)[..., None]
        return (T * (a[..., 0] + T * (a[..., 1] / 2 + T * (a[..., 2] / 3 + T * (a[..., 3] / 4 + T * a[..., 4] / 5)))) + a[..., 5]) @ self.sigma * self.mech.conv.R

    def s0(self, T):
        """Entropy at the standard pressure, J/(kg K), up to a constant of the composition."""
        a, T = self._a(T), np.asarray(T, float)[..., None]
        return (a[..., 0] * np.log(T) + T * (a[..., 1] + T * (a[..., 2] / 2 + T * (a[..., 3] / 3 + T * a[..., 4] / 4))) + a[..., 6]) @ self.sigma * self.mech.conv.R

    def e(self, T):
        return self.h(T) - self.R * np.asarray(T, float)

    def c(self, T):
        cp = self.cp(T)
        return np.sqrt(cp / (cp - self.R) * self.R * np.asarray(T, float))


class _Wave:
    """The wave on one side (sign -1: the left side, +1: the right side) as a function of the pressure behind it."""

    def __init__(self, mix, T, p, u, sign, p_low):
        self.mix, self.T0, self.p0, self.u0, self.sign = mix, float(T), float(p), float(u), sign
        self.rho0, self.c0 = p / (mix.R * T), float(mix.c(T))
        # the expansion: T and u against ln p, from the state down to p_low
        def rhs(lp, y):
            Tk = y[0]
            return [mix.R * Tk / float(mix.cp(Tk)), sign * mix.R * Tk / float(mix.c(Tk))]
        self.fan = solve_ivp(rhs, (np.log(p), np.log(p_low)), [T, u], rtol=1e-12, atol=1e-12, dense_output=True, method="DOP853").sol

    def shock(self, p):
        """State behind a shock to pressure p: (T, rho, u, speed of the shock)."""
        m, v0, h0 = self.mix, 1.0 / self.rho0, float(self.mix.h(self.T0))
        def f(x):                                                  # x = v / v0; energy across the shock
            v = x * v0
            return float(m.h(p * v / m.R)) - h0 - 0.5 * (p - self.p0) * (v0 + v)
        # the search stays below 6000 K, where the data ends (above it the polynomials mean nothing)
        x = brentq(f, 0.02, min(1.0 - 1e-12, 6000.0 * m.R / (p * v0)), xtol=1e-15, rtol=1e-15)
        v = x * v0
        flux = np.sqrt((p - self.p0) / (v0 - v))                   # mass per area and time through the shock
        return p * v / m.R, 1.0 / v, self.u0 + self.sign * (p - self.p0) / flux, self.u0 + self.sign * flux * v0

    def behind(self, p):
        """(T, rho, u) behind the wave at pressure p."""
        if p > self.p0 * (1 + 1e-10):
            return self.shock(p)[:3]
        if p >= self.p0:                                           # no wave at all
            return self.T0, self.rho0, self.u0
        T, u = self.fan(np.log(p))
        return float(T), p / (self.mix.R * float(T)), float(u)


class MixtureRiemann:
    def __init__(self, mech, left, right, x0=0.0):
        self.mech, self.x0 = mech, float(x0)
        self.mix = [Mixture(mech, left["Y"]), Mixture(mech, right["Y"])]
        p_low = 1e-3 * min(left["p"], right["p"])
        self.side = [_Wave(self.mix[0], left["T"], left["p"], left.get("u", 0.0), -1, p_low),
                     _Wave(self.mix[1], right["T"], right["p"], right.get("u", 0.0), +1, p_low)]
        L, R = self.side
        apart = lambda lp: R.behind(np.exp(lp))[2] - L.behind(np.exp(lp))[2]                                # noqa: E731
        hi = np.log(min(L.p0, R.p0))                               # upwards from the lower pressure, so that no shock is
        while apart(hi) < 0:                                       # asked for that is stronger than the data allows (6000 K)
            hi += 0.2
        lp = brentq(apart, np.log(p_low * 1.001), hi, xtol=1e-14, rtol=1e-14)
        self.p_star = float(np.exp(lp))
        (TL, rL, uL), (TR, rR, uR) = L.behind(self.p_star), R.behind(self.p_star)
        self.u_star = 0.5 * (uL + uR)
        self.star = dict(p=self.p_star, u=self.u_star, T_left=TL, rho_left=rL, T_right=TR, rho_right=rR)
        self.waves = {"contact": self.u_star}
        for w, name, Ts in ((L, "left", TL), (R, "right", TR)):
            if self.p_star > w.p0:
                self.waves[name + "_head"] = self.waves[name + "_tail"] = w.shock(self.p_star)[3]
            else:
                self.waves[name + "_head"] = w.u0 + w.sign * w.c0
                self.waves[name + "_tail"] = self.u_star + w.sign * float(w.mix.c(Ts))
        self.fastest = max(abs(v) for v in self.waves.values())
        self._tables = {}

    def _fan_table(self, k):
        """Inside the fan of side k: x/t = u + sign c against ln p, tabulated for interpolation."""
        if k not in self._tables:
            w = self.side[k]
            lp = np.linspace(np.log(w.p0), np.log(self.p_star), 20001)
            T, u = w.fan(lp)
            xi = u + w.sign * w.mix.c(T)
            order = np.argsort(xi)
            self._tables[k] = (xi[order], lp[order], T[order], u[order])
        return self._tables[k]

    def sample(self, x, t):
        """Point values at positions x and time t > 0: dict(rho, u, p, T, Y (species first), side (0 left gas, 1 right gas))."""
        xi = (np.asarray(x, float) - self.x0) / t
        L, R = self.side
        rho, u, p, T = (np.empty_like(xi) for _ in range(4))
        side = (xi > self.u_star).astype(int)
        for k, w in enumerate((L, R)):
            head, tail = self.waves[("left", "right")[k] + "_head"], self.waves[("left", "right")[k] + "_tail"]
            mine = side == k
            out = mine & ((xi <= head) if k == 0 else (xi >= head))            # undisturbed
            star = mine & ((xi >= tail) if k == 0 else (xi <= tail))           # between the wave and the border
            fan = mine & ~out & ~star
            T[out], p[out], u[out] = w.T0, w.p0, w.u0
            T[star], p[star], u[star] = self.star["T_left" if k == 0 else "T_right"], self.p_star, self.u_star
            if fan.any():
                tx, tlp, tT, tu = self._fan_table(k)
                T[fan], u[fan], p[fan] = np.interp(xi[fan], tx, tT), np.interp(xi[fan], tx, tu), np.exp(np.interp(xi[fan], tx, tlp))
            rho[mine] = p[mine] / (w.mix.R * T[mine])
        Y = np.where(side[None, ...] == 0, self.mix[0].Y.reshape((-1,) + (1,) * xi.ndim), self.mix[1].Y.reshape((-1,) + (1,) * xi.ndim))
        return dict(rho=rho, u=u, p=p, T=T, Y=Y, side=side)

    def energy(self, f):
        """Total energy per volume (chemical energy included) of sampled values."""
        e = np.where(f["side"] == 0, self.mix[0].e(f["T"]), self.mix[1].e(f["T"]))
        return f["rho"] * (e + 0.5 * f["u"] ** 2)

    def cell_averages(self, edges, t, sub=64):
        """Averages over the cells between `edges` as a finite-volume solver holds them: the conserved numbers
        (species, momentum, energy) are averaged over `sub` points per cell, and velocity, pressure and
        temperature are formed from the averages (the temperature by solving the mixture's energy for it in the
        cells where the gas is not uniform). Returns dict(rho, u, p, T, Y, E)."""
        edges = np.asarray(edges, float)
        n, m = len(edges) - 1, self.mech
        x = edges[:-1, None] + (np.arange(sub) + 0.5)[None, :] / sub * np.diff(edges)[:, None]
        f = self.sample(x, t)
        ri = (f["rho"][None] * f["Y"]).mean(-1)
        rho, mom, E = ri.sum(0), (f["rho"] * f["u"]).mean(-1), self.energy(f).mean(-1)
        u = mom / rho
        T = f["T"].mean(-1)
        mixed = np.nonzero((np.ptp(f["T"], axis=-1) > 1e-9 * T) | (np.ptp(f["side"], axis=-1) > 0))[0]
        for i in mixed:
            mix = Mixture(m, ri[:, i] / rho[i])
            T[i] = brentq(lambda Tk: float(mix.e(Tk)) - (E[i] / rho[i] - 0.5 * u[i] ** 2), 100.0, 6000.0, xtol=1e-11, rtol=1e-14)
        p = rho * (m.conv.R * ((ri / m.W[:, None]).sum(0) / rho)) * T
        return dict(rho=rho, u=u, p=p, T=T, Y=ri / rho, E=E)
