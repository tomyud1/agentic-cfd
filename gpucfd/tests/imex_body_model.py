"""Why IMEX with immersed bodies was unstable: a numpy model of the time step (DESIGN.md, "IMEX with
immersed bodies"). Not a rung of the ladder (it runs on the CPU and tests no solver code); it is the
evidence for the cause, kept so that it can be rerun.

    python -m gpucfd.tests.imex_body_model            (about 3 minutes; needs scipy, no GPU)

The model: 2D, uniform grid, the linear part of the step (no convection), a box whose x- side is an
inflow, x+ an outflow (or a wall: closed box) and y sides walls, with the bodies' ghost rules taken from
gpucfd.bodies.build, the same lists the solver gets. Every operator is a sparse matrix, so the step is a
matrix too and its growth can be computed instead of observed. It reproduces the solver's growth per
step to three digits in every variant tried on the 384 x 72 channel (DESIGN.md has the table).

It prints and writes to tests/results/imex_body_model.json:
1. mu, the largest eigenvalues of the ghost feedback M = A^-1 D_g E G (pressure -> its gradient on the
   fluid faces -> the ghost rule -> the divergence through the faces in bodies -> a pressure solve);
2. for which mu the pressure recurrence p_new = -mu [(1 + r) p - r p_old] is stable, for IMEX's substeps
   and for the explicit SSP-RK3, and the growth per step it predicts for the mu found;
3. the growth per step of the model's IMEX step in six variants, with the projection that corrects fluid
   faces only and with the consistent one.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from gpucfd import bodies as B


class Model:
    """State: u (M), v (M), ub (ny: the normal velocity on the outflow face, not stored in u), p (M)."""

    def __init__(self, n, length, bodies, nu, open_=True):
        nx, ny = n
        self.nx, self.ny, self.nu, self.open = nx, ny, nu, open_
        self.n, self.length, self.ncomp = (nx, ny, 1), (length[0], length[1], 1.0), 2  # what bodies.build reads
        self.kinds, self.shape, self.stretched = [1, 2 if open_ else 1, 1, 1, 0, 0], (1, ny, nx), False
        hx, hy = length[0] / nx, length[1] / ny
        self.hx, self.hy = hx, hy
        M = self.M = nx * ny
        idx = lambda j, i: i + nx * j
        self.rules = rules = B.build(self, bodies)
        fl = [r["mask"].copy() for r in rules]
        nw = 2 * M + ny
        Lw, G, D = sp.lil_matrix((2 * M, nw)), sp.lil_matrix((nw, M)), sp.lil_matrix((M, nw))
        ax, ay = 1 / hx ** 2, 1 / hy ** 2
        for j in range(ny):
            for i in range(nx):
                c = idx(j, i)
                D[c, c] -= 1 / hx
                if i + 1 < nx:
                    D[c, idx(j, i + 1)] += 1 / hx
                elif open_:
                    D[c, 2 * M + j] += 1 / hx
                D[c, M + c] -= 1 / hy
                if j + 1 < ny:
                    D[c, M + idx(j + 1, i)] += 1 / hy
                if i >= 1:  # u: face 0 is the inflow face (fixed); beyond a wall in y the ghost value is -u
                    G[c, c] += 1 / hx
                    G[c, idx(j, i - 1)] -= 1 / hx
                    d = -2.0 * ax
                    Lw[c, idx(j, i - 1)] += ax
                    if i + 1 < nx:
                        Lw[c, idx(j, i + 1)] += ax
                    elif open_:
                        Lw[c, 2 * M + j] += ax
                    for jj in (j - 1, j + 1):
                        if 0 <= jj < ny:
                            Lw[c, idx(jj, i)] += ay
                            d -= ay
                        else:
                            d -= 2 * ay
                    Lw[c, c] += d
                if j >= 1:  # v: ghost -v beyond a velocity side, v itself beyond the outflow (zero gradient)
                    G[M + c, c] += 1 / hy
                    G[M + c, idx(j - 1, i)] -= 1 / hy
                    d = -2.0 * ay
                    Lw[M + c, M + idx(j - 1, i)] += ay
                    if j + 1 < ny:
                        Lw[M + c, M + idx(j + 1, i)] += ay
                    for ii in (i - 1, i + 1):
                        if 0 <= ii < nx:
                            Lw[M + c, M + idx(j, ii)] += ax
                            d -= ax
                        elif ii < 0 or not open_:
                            d -= 2 * ax
                    Lw[M + c, M + c] += d
            if open_:
                G[2 * M + j, idx(j, nx - 1)] = -2 / hx  # outflow face: the pressure is zero on the side
        Lw = Lw.tocsr()
        self.Lh, self.Lb = Lw[:, :2 * M].tocsr(), Lw[:, 2 * M:].tocsr()  # implicit part; from the outflow face values
        self.G, self.D = G.tocsr(), D.tocsr()
        inter = np.ones(2 * M)
        inter[[idx(j, 0) for j in range(ny)]] = 0
        inter[[M + idx(0, i) for i in range(nx)]] = 0
        self.inter, self.fluid = inter, np.concatenate(fl)
        self.flq = self.fluid * inter                        # faces the projection corrects
        self.flw = np.concatenate([self.flq, np.ones(ny)])
        rows, cols, vals = [], [], []
        for c, r in enumerate(rules):                        # S: the ghost rule (without its constants)
            f = np.flatnonzero(r["mask"] > 0)
            rows += list(c * M + f)
            cols += list(c * M + f)
            vals += [1.0] * f.size
            for k in range(4):
                rows += list(c * M + r["idx"])
                cols += list(c * M + r["nb"][:, k])
                vals += list(r["w"][:, k])
        self.S = sp.csr_matrix((vals, (rows, cols)), shape=(2 * M, 2 * M))
        mu_, mv_ = fl[0].reshape(ny, nx), fl[1].reshape(ny, nx)
        act = np.zeros((ny, nx))
        for j in range(ny):
            for i in range(nx):
                act[j, i] = float((i > 0 and mu_[j, i] > 0) or (i < nx - 1 and mu_[j, i + 1] > 0) or (i == nx - 1 and open_)
                                  or (j > 0 and mv_[j, i] > 0) or (j < ny - 1 and mv_[j + 1, i] > 0))
        self.act = act.ravel()
        self.ai = ai = np.flatnonzero(self.act)
        self.A = (self.D @ sp.diags(self.flw) @ self.G).tocsr()
        Aa = self.A[ai][:, ai]
        self.closed = not open_
        if self.closed:  # singular: zero mean; the rim faces take the net flow out (k_ib_cons in the solver)
            one = sp.csr_matrix(np.ones((ai.size, 1)))
            self.Alu = spl.splu(sp.bmat([[Aa, one], [one.T, None]]).tocsc())
            ri, rs = [], []
            for c in range(2):
                for f in rules[c]["idx"]:
                    j, i = divmod(int(f), nx)
                    I, J = [i, j], [i, j]
                    if I[c] == 0:
                        continue
                    J[c] -= 1
                    sg = self.act[J[0] + nx * J[1]] - self.act[f]
                    if sg != 0:
                        ri.append(c * M + f)
                        rs.append(sg * hx)
            self.rim, self.rsa = np.array(ri), np.array(rs) / len(ri)
        else:
            self.Alu = spl.splu(Aa.tocsc())
        # the consistent pressure operator A + D_g E G (open box only)
        body = np.concatenate([(1.0 - self.fluid) * inter, np.zeros(ny)])
        Sw = sp.bmat([[self.S, None], [None, sp.identity(ny)]]).tocsr()
        self.DgEG = (self.D @ sp.diags(body) @ Sw @ sp.diags(self.flw) @ self.G).tocsr()
        self._clu, self._h = None, {}

    def face_coords(self, c):
        hh = (self.hx, self.hy)
        ax = [np.arange(self.n[d]) * hh[d] + (0.0 if d == c else 0.5 * hh[d]) for d in range(2)]
        Y, X = np.meshgrid(ax[1], ax[0], indexing="ij")
        return X[None], Y[None], np.full((1, self.ny, self.nx), 0.5)

    def poisson(self, dv):
        x = self.Alu.solve(np.vstack([dv[self.ai], np.zeros((1, dv.shape[1]))]))[:-1] if self.closed else self.Alu.solve(dv[self.ai])
        phi = np.zeros_like(dv)
        phi[self.ai] = x
        return phi

    def project(self, q, ub, ghost=True, consistent=False):
        """As the solver: ghost faces from q, outflow face, (closed: rim correction), solve, correct the fluid
        faces. consistent: the pressure operator includes the ghost faces, which are then set from the result."""
        M = self.M
        if ghost:
            q = self.S @ q
        if self.open:
            ub = q[self.nx - 1:M:self.nx].copy()
        w = np.vstack([q, ub])
        if self.closed:
            w[self.rim] -= self.rsa[:, None] * (self.act @ (self.D @ w))[None, :]
        dv = self.act[:, None] * (self.D @ w)
        if consistent:
            if self._clu is None:
                self._clu = spl.splu((self.A + self.DgEG)[self.ai][:, self.ai].tocsc())
            phi = np.zeros_like(dv)
            phi[self.ai] = self._clu.solve(dv[self.ai])
        else:
            phi = self.poisson(dv)
        w = w - self.flw[:, None] * (self.G @ phi)
        q = w[:2 * M]
        return (self.S @ q if consistent else q), w[2 * M:], phi

    def helm(self, c, rows):
        """(I - c Lap) with the faces in bodies "free" (unknowns like fluid faces, as first coded), "fixed"
        (they keep their value during the solve) or "rule" (their ghost rule is their equation)."""
        key = (round(c, 14), rows)
        if key not in self._h:
            H = (sp.identity(2 * self.M) - c * self.Lh).tocsr()
            if rows != "free":
                body = sp.diags(1.0 - self.fluid)
                H = sp.diags(self.fluid) @ H + body - (body @ self.S if rows == "rule" else 0 * body)
            self._h[key] = spl.splu(H.tocsc())
        return self._h[key]

    AB = (4 / 15, 1 / 15, 1 / 6)  # Spalart, Moser & Rogers: substeps 2 ab dt = 8/15, 2/15, 1/3 of dt

    def step_imex(self, q, ub, p, dt, rows="rule", gmask=True, reghost=False, consistent=False):
        """gmask False: the pressure gradient also on the faces in bodies (cause 1). reghost: ghost faces set
        again after the projection."""
        M = self.M
        body = 1.0 - self.fluid
        for ab in self.AB:
            cdt = 2 * ab * dt
            gp = (self.G @ p)[:2 * M] * (self.flq if gmask else self.inter)[:, None]
            rhs = q + dt * ab * self.nu * (self.Lh @ q + 2 * (self.Lb @ ub)) - cdt * gp
            if rows == "fixed":
                rhs = self.fluid[:, None] * rhs + body[:, None] * q
            elif rows == "rule":
                rhs = self.fluid[:, None] * rhs
            q = self.helm(ab * dt * self.nu, rows).solve(rhs)
            q, ub, phi = self.project(q, ub, consistent=consistent)
            if reghost:
                q = self.S @ q
            p = p + phi / cdt
        return q, ub, p

    def growth(self, dt, steps=160, **kw):
        """Growth per step of the dominant mode: geometric mean over the last quarter of `steps` steps."""
        rng = np.random.default_rng(0)
        q = rng.standard_normal((2 * self.M, 1)) * self.inter[:, None]
        ub, p, g = np.zeros((self.ny, 1)), np.zeros((self.M, 1)), []
        for k in range(steps):
            q, ub, p = self.step_imex(q, ub, p, dt, **kw)
            nrm = float(np.sqrt(np.sum((q * self.fluid[:, None]) ** 2)))
            if not np.isfinite(nrm) or nrm == 0:
                return float("inf")
            g.append(nrm)
            q, ub, p = q / nrm, ub / nrm, p / nrm
        return float(np.exp(np.mean(np.log(g[-steps // 4:]))))

    def feedback(self, k=6):
        """Largest eigenvalues of M = A^-1 D_g E G."""
        M2 = 2 * self.M
        body = (1.0 - self.fluid) * self.inter

        def apply(x):
            p = np.zeros((self.M, 1))
            p[self.ai, 0] = x
            g = body[:, None] * (self.S @ (self.flq[:, None] * (self.G @ p)[:M2]))
            w = np.vstack([g, np.zeros((self.ny, 1))])
            if self.closed:
                w[self.rim] -= self.rsa[:, None] * (self.act @ (self.D @ w))[None, :]
            return self.poisson(self.act[:, None] * (self.D @ w))[self.ai, 0]
        w = spl.eigs(spl.LinearOperator((self.ai.size,) * 2, matvec=apply, dtype=float), k=k, which="LM",
                     return_eigenvectors=False, tol=1e-8, maxiter=5000)
        return w[np.argsort(-np.abs(w))]


def radius_imex(mu, c=(8 / 15, 2 / 15, 1 / 3)):
    """Growth per step of p_{k+1} = -mu [(1 + r_k) p_k - r_k p_{k-1}], r_k = c_{k-1} / c_k, over the stages of a step."""
    T = np.eye(2, dtype=complex)
    for k in range(3):
        r = c[k - 1] / c[k]
        T = np.array([[-mu * (1 + r), mu * r], [1, 0]], dtype=complex) @ T
    return float(np.max(np.abs(np.linalg.eigvals(T))))


def radius_rk3(mu):
    """The same for SSP-RK3 with the incremental pressure (u* = a u0 + b (u + dt F - dt grad p), b = 1, 1/4, 2/3):
    p1 = -mu [p0 + 2/3 d0], p2 = -mu [2 p1 - p0 + 2 d0], p3 = -mu [5/4 p2 - 1/4 p1 + d0 / 3], d0 = p0 - p_(-1)."""
    cols = []
    for p0, d0 in ((1, 0), (0, 1)):
        p1 = -mu * (p0 + 2 / 3 * d0)
        p2 = -mu * (2 * p1 - p0 + 2 * d0)
        p3 = -mu * (1.25 * p2 - 0.25 * p1 + d0 / 3)
        cols.append([p3, p3 - p2])
    return float(np.max(np.abs(np.linalg.eigvals(np.array(cols, dtype=complex).T))))


def limit(f):
    """Largest positive mu for which the recurrence does not grow."""
    xs = np.linspace(0.001, 1.5, 6000)
    return float(next(x for x in xs if f(x) > 1.0))


VARIANTS = (("as first coded: faces in bodies free in the solve, pressure gradient on every face", dict(rows="free", gmask=False)),
            ("pressure gradient on fluid faces only", dict(rows="free")),
            ("faces in bodies fixed during the solve", dict(rows="fixed")),
            ("ghost rule as rows of the solve", dict(rows="rule")),
            ("ghost rule as rows, ghost faces set again after the projection", dict(rows="rule", reghost=True)),
            ("ghost rule as rows, consistent projection (the design)", dict(rows="rule", consistent=True)))

if __name__ == "__main__":
    out = dict(limits=dict(imex=limit(radius_imex), equal_substeps=limit(lambda x: radius_imex(x, (1, 1, 1))), rk3=limit(radius_rk3)))
    L = out["limits"]
    print(f"the pressure recurrence is stable for mu below {L['imex']:.3f} (IMEX substeps 8/15, 2/15, 1/3), "
          f"{L['equal_substeps']:.3f} (equal substeps), {L['rk3']:.3f} (SSP-RK3)", flush=True)
    cyl = lambda xc: [B.Cylinder((xc, 0.2), 0.05)]
    ctr = (0.5, 0.5, 0.5)
    cases = (("channel 64 x 32, cylinder at 8 cells per diameter", (64, 32), (0.8, 0.4), cyl(0.2), True),
             ("DFG channel 384 x 72 (17.6 cells per diameter)", (384, 72), (2.2, 0.41), cyl(0.2), True),
             ("DFG channel 384 x 72, cylinder moved 0.37 cell", (384, 72), (2.2, 0.41), cyl(0.2 + 0.37 * 2.2 / 384), True),
             ("closed Couette cell 32 x 32 (rung 10)", (32, 32), (1.0, 1.0),
              [B.Cylinder(ctr, 0.2, velocity=B.rotation(ctr, 5.0)), B.Inside(B.Cylinder(ctr, 0.45))], False),
             ("closed Couette cell 64 x 64", (64, 64), (1.0, 1.0),
              [B.Cylinder(ctr, 0.2, velocity=B.rotation(ctr, 5.0)), B.Inside(B.Cylinder(ctr, 0.45))], False))
    out["feedback"], models = [], {}
    for label, n, length, bod, open_ in cases:
        m = models[label] = Model(n, length, bod, 1e-3, open_)
        w = m.feedback()
        mu = complex(w[0])
        r = dict(case=label, mu=[[float(x.real), float(x.imag)] for x in w], growth_imex=radius_imex(mu), growth_rk3=radius_rk3(mu))
        out["feedback"].append(r)
        print(f"{label}: mu = " + ", ".join(f"{x.real:+.3f}{'' if abs(x.imag) < 1e-9 else f'{x.imag:+.3f}i'}" for x in w[:4])
              + f"; predicted growth per step without viscosity: IMEX {r['growth_imex']:.3f}, RK3 {r['growth_rk3']:.3f}", flush=True)
    out["variants"] = []
    for label in [c[0] for c in cases[1:3]]:
        m = models[label]
        for d in (0.02, 0.31, 1.0, 4.0):
            dt = d * m.hx ** 2 / m.nu
            for name, kw in VARIANTS:
                g = m.growth(dt, **kw)
                out["variants"].append(dict(case=label, d=d, variant=name, growth=g))
                print(f"{label}, nu dt / h^2 = {d}: {name}: growth per step {g:.3f}", flush=True)
    path = Path(__file__).parent / "results" / "imex_body_model.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"written {path}")
    sys.exit(0)
