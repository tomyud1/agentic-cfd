"""Exact solution of the Riemann problem of gas dynamics (two uniform states of an ideal gas, constant gamma,
released at x0 at time zero), written for gpucfd's compressible ladder. Plain NumPy, no other project code.

    ex = Riemann(1.4, left=(rho, u, p), right=(rho, u, p), x0=0.5)
    rho, u, p = ex.primitive(x, t)            point values
    rho, u, p = ex.cell_averages(edges, t)    exact averages over cells, formed as a finite-volume solver holds them
    ex.waves                                  speeds of the waves, left to right

The pressure between the waves is found by Newton's iteration on the two wave curves (Toro, Riemann Solvers and
Numerical Methods for Fluid Dynamics, chapter 4). Cell averages are exact: the conserved quantities (density,
momentum, energy) are integrated in closed form over each uniform stretch and over each expansion fan, and
velocity and pressure are then formed from the averaged conserved quantities.
Checks of this file itself are in gpucfd/tests/test_euler_sod.py (conservation of the whole solution, against
sampled averages, and against OpenFOAM's stored results).
"""
import numpy as np


class Riemann:
    def __init__(self, gamma, left, right, x0=0.0):
        self.g, self.x0 = float(gamma), float(x0)
        self.L, self.R = tuple(map(float, left)), tuple(map(float, right))
        g = self.g
        (rl, ul, pl), (rr, ur, pr) = self.L, self.R
        self.cl, self.cr = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)
        if 2 * (self.cl + self.cr) / (g - 1) <= ur - ul:
            raise ValueError("the two states pull apart so fast that a vacuum forms: not covered")

        def curve(p, rk, pk, ck):
            """Velocity change across the wave on one side as a function of the pressure behind it, and its slope."""
            if p > pk:                                             # shock
                A, B = 2 / ((g + 1) * rk), (g - 1) / (g + 1) * pk
                q = np.sqrt(A / (p + B))
                return (p - pk) * q, q * (1 - 0.5 * (p - pk) / (p + B))
            return 2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1), (p / pk) ** (-(g + 1) / (2 * g)) / (rk * ck)

        p = max(1e-12 * (pl + pr), 0.5 * (pl + pr) - 0.125 * (ur - ul) * (rl + rr) * (self.cl + self.cr))
        for _ in range(100):
            fl, dl = curve(p, rl, pl, self.cl)
            fr, dr = curve(p, rr, pr, self.cr)
            step = (fl + fr + ur - ul) / (dl + dr)
            p_new = max(p - step, 1e-14 * p)
            done = abs(p_new - p) <= 1e-15 * (p_new + p)
            p = p_new
            if done:
                break
        self.p_star = p
        self.u_star = 0.5 * (ul + ur) + 0.5 * (curve(p, rr, pr, self.cr)[0] - curve(p, rl, pl, self.cl)[0])
        # each side: the density behind its wave, and the wave's speeds (a shock has one, a fan a head and a tail)
        self.side = {}
        for name, (rk, uk, pk), ck, sgn in (("left", self.L, self.cl, -1.0), ("right", self.R, self.cr, 1.0)):
            P = p / pk
            if p > pk:
                rs = rk * ((g + 1) * P + (g - 1)) / ((g - 1) * P + (g + 1))
                S = uk + sgn * ck * np.sqrt((g + 1) / (2 * g) * P + (g - 1) / (2 * g))
                self.side[name] = dict(kind="shock", rho=rs, head=S, tail=S)
            else:
                rs = rk * P ** (1 / g)
                cs = ck * P ** ((g - 1) / (2 * g))
                self.side[name] = dict(kind="fan", rho=rs, head=uk + sgn * ck, tail=self.u_star + sgn * cs)
        self.rho_star_left, self.rho_star_right = self.side["left"]["rho"], self.side["right"]["rho"]
        self.waves = dict(left_head=self.side["left"]["head"], left_tail=self.side["left"]["tail"], contact=self.u_star,
                          right_tail=self.side["right"]["tail"], right_head=self.side["right"]["head"])
        self.fastest = max(abs(ul) + self.cl, abs(ur) + self.cr, abs(self.u_star) + np.sqrt(g * p / self.rho_star_left),
                           abs(self.u_star) + np.sqrt(g * p / self.rho_star_right))

    def positions(self, t):
        return {k: self.x0 + v * t for k, v in self.waves.items()}

    # ---- the fans: everything as a function of s = (local speed of sound) / (speed of sound of the undisturbed gas)
    def _fan_s(self, xi, name):
        g = self.g
        (rk, uk, pk), ck, sgn = (self.L, self.cl, -1.0) if name == "left" else (self.R, self.cr, 1.0)
        return (2 + sgn * (g - 1) * (xi - uk) / ck) / (g + 1)

    def _fan_state(self, s, name):
        g = self.g
        (rk, uk, pk), ck, sgn = (self.L, self.cl, -1.0) if name == "left" else (self.R, self.cr, 1.0)
        return rk * s ** (2 / (g - 1)), uk - sgn * 2 * ck / (g - 1) * (1 - s), pk * s ** (2 * g / (g - 1))

    def _fan_integral(self, xi, name):
        """Antiderivatives with respect to xi = (x - x0) / t of rho, rho u and E inside a fan."""
        g = self.g
        (rk, uk, pk), ck, sgn = (self.L, self.cl, -1.0) if name == "left" else (self.R, self.cr, 1.0)
        s = self._fan_s(xi, name)
        a, k = 2 / (g - 1), 2 * ck / (g - 1)
        A = uk - sgn * k                                           # u = A + sgn k s
        P = lambda n: s ** (n + 1) / (n + 1)                       # noqa: E731  (integral of s^n)
        jac = sgn * (g + 1) / (g - 1) * ck                         # d xi = jac ds
        rho = rk * P(a)
        mom = rk * (A * P(a) + sgn * k * P(a + 1))
        E = pk / (g - 1) * P(a + 2) + 0.5 * rk * (A * A * P(a) + 2 * A * sgn * k * P(a + 1) + k * k * P(a + 2))
        return jac * rho, jac * mom, jac * E

    def primitive(self, x, t):
        x = np.asarray(x, float)
        xi = (x - self.x0) / t
        w, g = self.waves, self.g
        out = []
        fanL = self._fan_state(np.clip(self._fan_s(xi, "left"), 1e-300, None), "left")
        fanR = self._fan_state(np.clip(self._fan_s(xi, "right"), 1e-300, None), "right")
        star = ((self.rho_star_left, self.u_star, self.p_star), (self.rho_star_right, self.u_star, self.p_star))
        for i in range(3):
            out.append(np.where(xi < w["left_head"], self.L[i], np.where(xi < w["left_tail"], fanL[i], np.where(
                xi < w["contact"], star[0][i], np.where(xi < w["right_tail"], star[1][i], np.where(
                    xi < w["right_head"], fanR[i], self.R[i]))))))
        return tuple(out)

    def cumulative(self, x, t):
        """Integrals from x0 to x of rho, rho u and E = p / (gamma - 1) + rho u^2 / 2 at time t (exact)."""
        xi = (np.asarray(x, float) - self.x0) / t
        w, g = self.waves, self.g
        cons = lambda r, u, p: np.array([r, r * u, p / (g - 1) + 0.5 * r * u * u])      # noqa: E731
        pieces = [(-np.inf, w["left_head"], cons(*self.L), None), (w["left_head"], w["left_tail"], None, "left"),
                  (w["left_tail"], w["contact"], cons(self.rho_star_left, self.u_star, self.p_star), None),
                  (w["contact"], w["right_tail"], cons(self.rho_star_right, self.u_star, self.p_star), None),
                  (w["right_tail"], w["right_head"], None, "right"), (w["right_head"], np.inf, cons(*self.R), None)]
        total = np.zeros((3,) + xi.shape)
        for a, b, const, fan in pieces:                            # integral from 0 to xi = sum over the pieces
            lo, hi = np.clip(0.0, a, b), np.clip(xi, a, b)
            if b <= a:
                continue
            if fan is None:
                total += const.reshape(3, *([1] * xi.ndim)) * (hi - lo)
            else:
                Fh, Fl = self._fan_integral(hi, fan), self._fan_integral(np.asarray(lo), fan)
                total += np.array([Fh[i] - Fl[i] for i in range(3)])
        return total * t

    def cell_averages(self, edges, t):
        """(rho, u, p) of every cell between consecutive edges: averages of the conserved quantities, then
        u = (rho u) / rho and p = (gamma - 1) (E - (rho u)^2 / (2 rho)), as a finite-volume solver forms them."""
        edges = np.asarray(edges, float)
        c = self.cumulative(edges, t)
        rho, mom, E = (c[:, 1:] - c[:, :-1]) / np.diff(edges)
        return rho, mom / rho, (self.g - 1) * (E - 0.5 * mom * mom / rho)
