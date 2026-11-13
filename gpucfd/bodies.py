"""Immersed bodies for gcfd.NavierStokes (sharp interface, ghost-cell method).

A body is a shape given by its signed distance phi (negative inside, |grad phi| = 1), optionally with a
velocity field (a rotating cylinder, a moving wall). The geometry is fixed in time.

    from gpucfd import gcfd, bodies
    cyl = bodies.Cylinder(center=(0.2, 0.2, 0), radius=0.05)          # axis z: a circle in 2D
    s = gcfd.NavierStokes(n, length, nu, boundaries=..., bodies=[cyl])

How the solver uses it (src/gcfd.cu, k_ib_set): every face of a velocity component that is not in the
fluid (phi <= 0 at the face) gets its value from a rule built here.
  (On a stretched grid h below is the local cell size at the face: see build.)
- Ghost face (within 1.5 h of the surface): G, its closest surface point B = G - phi n and the image
  point I = B + s_I n in the fluid. The velocity along the normal is taken as linear through the body's
  velocity at B and the fluid velocity at I (bi- or trilinear interpolation from FLUID faces only:
  s_I >= sqrt(dim) h guarantees that for an exact signed distance), so
      u_G = U_B + (u_I - U_B) phi_G / s_I.
  Second order: the error of a linear profile over a distance O(h) is O(h^2).
- Deeper faces: the body's velocity (they only enter the pressure equation's decoupled cells).
The pressure equation is solved with zero gradient through every face in a body (a staircase wall for
the pressure, the usual choice of ghost-cell methods); the velocity on those faces is never changed by
the projection, so at a steady state the rules above hold exactly. (Scheme "imex": the projection
includes the ghost faces, so the rules hold after every step: DESIGN.md, "IMEX with immersed bodies".)

Shapes: Cylinder, Sphere, Box, HalfSpace, Inside (fluid inside a shape: a pipe, the outer cylinder of a
Couette cell) and SDF (any exact signed distance function). Bodies must not cross a periodic side (the
signed distance is evaluated at the raw coordinates) unless the SDF itself is periodic.
"""
import numpy as np


class Body:
    """Base class: sdf(X, Y, Z) -> signed distance (negative inside). velocity: None (at rest) or a
    function (X, Y, Z) -> (u, v, w) evaluated on arrays (the body's velocity at those points)."""
    velocity = None

    def sdf(self, X, Y, Z):
        raise NotImplementedError

    def vel(self, X, Y, Z, comp):
        if self.velocity is None:
            return np.zeros(np.shape(X))
        r = self.velocity(X, Y, Z)
        v = r[comp] if comp < len(r) else 0.0
        return np.broadcast_to(np.asarray(v, np.float64), np.shape(X)).astype(np.float64)


class Cylinder(Body):
    """Infinite circular cylinder along axis (0 x, 1 y, 2 z) through center: a circle in 2D (axis 2)."""

    def __init__(self, center, radius, axis=2, velocity=None):
        self.center = np.asarray(tuple(center) + (0.0,) * (3 - len(center)), np.float64)
        self.radius, self.axis, self.velocity = float(radius), int(axis), velocity

    def sdf(self, X, Y, Z):
        P = (X, Y, Z)
        r2 = sum((P[d] - self.center[d]) ** 2 for d in range(3) if d != self.axis)
        return np.sqrt(r2) - self.radius


class Sphere(Body):
    def __init__(self, center, radius, velocity=None):
        self.center = np.asarray(center, np.float64)
        self.radius, self.velocity = float(radius), velocity

    def sdf(self, X, Y, Z):
        return np.sqrt((X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2 + (Z - self.center[2]) ** 2) - self.radius


class Box(Body):
    """Axis-aligned box lo..hi (exact signed distance, also at edges and corners)."""

    def __init__(self, lo, hi, velocity=None):
        self.lo, self.hi, self.velocity = np.asarray(lo, np.float64), np.asarray(hi, np.float64), velocity

    def sdf(self, X, Y, Z):
        c, e = 0.5 * (self.lo + self.hi), 0.5 * (self.hi - self.lo)
        q = [np.abs(P - c[d]) - e[d] for d, P in enumerate((X, Y, Z))]
        out = np.sqrt(sum(np.maximum(a, 0.0) ** 2 for a in q))
        return out + np.minimum(np.maximum(np.maximum(q[0], q[1]), q[2]), 0.0)


class HalfSpace(Body):
    """Solid on the side of the plane through point that the normal points away from: phi = n.(x - point)."""

    def __init__(self, point, normal, velocity=None):
        n = np.asarray(normal, np.float64)
        self.point, self.n, self.velocity = np.asarray(point, np.float64), n / np.linalg.norm(n), velocity

    def sdf(self, X, Y, Z):
        return self.n[0] * (X - self.point[0]) + self.n[1] * (Y - self.point[1]) + self.n[2] * (Z - self.point[2])


class Inside(Body):
    """The fluid is inside the given shape (the solid is outside it): a pipe, a Couette cell's outer wall."""

    def __init__(self, shape, velocity=None):
        self.shape, self.velocity = shape, velocity

    def sdf(self, X, Y, Z):
        return -self.shape.sdf(X, Y, Z)


class SDF(Body):
    """Any exact signed distance function f(X, Y, Z) (negative inside)."""

    def __init__(self, f, velocity=None):
        self.f, self.velocity = f, velocity

    def sdf(self, X, Y, Z):
        return self.f(X, Y, Z)


def rotation(center, omega, axis=2):
    """Velocity field of a rigid rotation about the line through center along axis, rate omega."""
    c = np.asarray(tuple(center) + (0.0,) * (3 - len(center)), np.float64)
    a, b = [d for d in range(3) if d != axis]

    def f(X, Y, Z):
        P = (X, Y, Z)
        out = [0.0 * X, 0.0 * X, 0.0 * X]
        out[a] = -omega * (P[b] - c[b])
        out[b] = omega * (P[a] - c[a])
        return tuple(out)
    return f


# ------------------------------------------------------------------------------------------- geometry
def phi(bodies, X, Y, Z):
    """Signed distance to the union of the bodies, and which body is nearest."""
    a = np.stack([np.asarray(b.sdf(X, Y, Z), np.float64) * np.ones(np.shape(X)) for b in bodies])
    k = np.argmin(a, axis=0)
    return np.take_along_axis(a, k[None], 0)[0], k


def normals(bodies, which, P, active, eps):
    """Unit outward normals (grad phi by central differences of the nearest body's distance) at points P."""
    n = np.zeros((3,) + P[0].shape)
    for k, b in enumerate(bodies):
        sel = which == k
        if not np.any(sel):
            continue
        Q = [p[sel] for p in P]
        for d in active:
            Qp, Qm = list(Q), list(Q)
            Qp[d] = Q[d] + eps
            Qm[d] = Q[d] - eps
            n[d][sel] = (b.sdf(*Qp) - b.sdf(*Qm)) / (2 * eps)
    norm = np.sqrt(sum(n[d] ** 2 for d in active))
    if np.any(norm < 0.5):
        raise ValueError("bodies: a signed distance has |grad| far from 1 near the surface (not an exact SDF?)")
    return n / norm


def interp_nodes(ns, comp, I, active):
    """Multilinear interpolation of component comp at points I: node indices (flat, wrapped in periodic
    directions; -1 where a node is outside a bounded direction) and weights, shape (m, 2^dim)."""
    m = I[0].size
    per = [ns.kinds[2 * d] == 0 for d in range(3)]
    lo, fr = [], []
    for d in range(3):
        if d not in active:
            lo.append(np.zeros(m, np.int64))
            fr.append(None)
            continue
        h = ns.length[d] / ns.n[d]
        off = 0.0 if d == comp else 0.5 * h
        t = (I[d].ravel() - off) / h
        i0 = np.floor(t).astype(np.int64)
        lo.append(i0)
        fr.append(t - i0)
    nodes, weights = [], []
    combos = [(a, b, c) for c in ((0, 1) if 2 in active else (0,)) for b in ((0, 1) if 1 in active else (0,))
              for a in ((0, 1) if 0 in active else (0,))]
    for cmb in combos:
        idx = []
        w = np.ones(m)
        bad = np.zeros(m, bool)
        for d in range(3):
            i = lo[d] + cmb[d]
            if d in active:
                w = w * (fr[d] if cmb[d] else 1.0 - fr[d])
            if per[d]:
                i = np.mod(i, ns.n[d])
            else:
                bad |= (i < 0) | (i >= ns.n[d])
                i = np.clip(i, 0, ns.n[d] - 1)
            idx.append(i)
        flat = idx[0] + ns.n[0] * (idx[1] + ns.n[1] * idx[2])
        flat[bad] = -1
        nodes.append(flat)
        weights.append(w)
    return np.stack(nodes, 1), np.stack(weights, 1)


def interp_nodes_s(ns, comp, I, active):
    """interp_nodes on a stretched grid: the weights are linear between the component's true node
    positions (periodic directions are uniform there and keep the uniform rule). A point beyond the
    first or last node of a bounded direction gets a node index -1 with a non-zero weight, as in
    interp_nodes, so the caller rejects the stencil."""
    m = I[0].size
    per = [ns.kinds[2 * d] == 0 for d in range(3)]
    lo, fr = [], []
    for d in range(3):
        if d not in active:
            lo.append(np.zeros(m, np.int64))
            fr.append(None)
            continue
        p = I[d].ravel()
        if per[d]:
            h = ns.length[d] / ns.n[d]
            t = (p - (0.0 if d == comp else 0.5 * h)) / h
            i0 = np.floor(t).astype(np.int64)
            lo.append(i0)
            fr.append(t - i0)
            continue
        x = ns._axis(d, d == comp)
        xe = np.concatenate(([2 * x[0] - x[1]], x, [2 * x[-1] - x[-2]]))  # one virtual node beyond each end
        i0 = np.clip(np.searchsorted(x, p, side="right") - 1, -1, x.size - 1).astype(np.int64)
        lo.append(i0)
        fr.append((p - xe[i0 + 1]) / (xe[i0 + 2] - xe[i0 + 1]))
    nodes, weights = [], []
    combos = [(a, b, c) for c in ((0, 1) if 2 in active else (0,)) for b in ((0, 1) if 1 in active else (0,))
              for a in ((0, 1) if 0 in active else (0,))]
    for cmb in combos:
        idx = []
        w = np.ones(m)
        bad = np.zeros(m, bool)
        for d in range(3):
            i = lo[d] + cmb[d]
            if d in active:
                w = w * (fr[d] if cmb[d] else 1.0 - fr[d])
            if per[d]:
                i = np.mod(i, ns.n[d])
            else:
                bad |= (i < 0) | (i >= ns.n[d])
                i = np.clip(i, 0, ns.n[d] - 1)
            idx.append(i)
        flat = idx[0] + ns.n[0] * (idx[1] + ns.n[1] * idx[2])
        flat[bad] = -1
        nodes.append(flat)
        weights.append(w)
    return np.stack(nodes, 1), np.stack(weights, 1)


def local_size(ns, comp, active):
    """Stretched grids: the local cell size at every face of component comp (shape (nz, ny, nx)): the
    largest distance to a neighbouring node of that component, over the active directions. Beyond a
    bounded side the neighbour is the mirror image of the first cell. On a uniform grid this is the
    largest of the cell sizes, the h of the uniform rules."""
    out = np.zeros(ns.shape)
    for d in active:
        w = ns.widths(d)
        if d == comp:  # nodes are the low faces: neighbours one cell away on either side
            a = np.maximum(w, np.concatenate(([w[0]], w[:-1])))
        else:          # nodes are cell centres: neighbours half of each adjoining cell away
            dm = np.concatenate(([w[0]], 0.5 * (w[:-1] + w[1:]), [w[-1]]))
            a = np.maximum(dm[:-1], dm[1:])
        shp = [1, 1, 1]
        shp[2 - d] = -1
        out = np.maximum(out, a.reshape(shp))
    return out


def build(ns, bodies, band=1.5, s_image=None):
    """Per velocity component: fluid mask and the rules of the faces in bodies (see the module text).
    Returns a list of dicts (mask, idx, a0, nb, w, n_ghost).
    On a stretched grid the one cell size h of the rules (ghost band, image distance and its steps) is
    the local cell size at each ghost face (local_size), and the image point is interpolated between the
    true node positions (interp_nodes_s). With uniform face positions these are the uniform rules."""
    active = [d for d in range(3) if ns.n[d] > 1]
    h = [ns.length[d] / ns.n[d] for d in range(3)]
    hmax = max(h[d] for d in active)  # stretched: the mean cell size, only a scale for the two tolerances below
    stretched = bool(getattr(ns, "stretched", False))
    interp = interp_nodes_s if stretched else interp_nodes
    fI = (np.sqrt(len(active)) + 0.05) if s_image is None else float(s_image)
    nsb = 2 ** len(active)
    out = []
    for c in range(ns.ncomp):
        X, Y, Z = ns.face_coords(c)
        ph, which = phi(bodies, X, Y, Z)
        fluid = (ph > 1e-12 * hmax).ravel()
        solid = np.flatnonzero(~fluid)
        m = solid.size
        # the cell size of the rules at every face in a body
        hl = local_size(ns, c, active).ravel()[solid] if stretched else np.full(m, hmax)
        a0 = np.zeros(m)
        nb = np.zeros((m, nsb), np.int64)
        w = np.zeros((m, nsb))
        first_fluid = int(np.flatnonzero(fluid)[0]) if np.any(fluid) else 0
        nb[:] = first_fluid  # weight 0 (deep faces): any fluid face
        phs, ws = ph.ravel()[solid], which.ravel()[solid]
        P = [A.ravel()[solid] for A in (X, Y, Z)]
        g = phs > -band * hl
        # deep faces: the body's velocity there
        for k, b in enumerate(bodies):
            sel = (~g) & (ws == k)
            if np.any(sel):
                a0[sel] = b.vel(*[p[sel] for p in P], c)
        gi = np.flatnonzero(g)
        if gi.size:
            Pg = [p[gi] for p in P]
            nrm = normals(bodies, ws[gi], Pg, active, 1e-6 * hmax)
            B = [Pg[d] - phs[gi] * nrm[d] for d in range(3)]
            UB = np.zeros(gi.size)
            for k, b in enumerate(bodies):
                sel = ws[gi] == k
                if np.any(sel):
                    UB[sel] = b.vel(*[x[sel] for x in B], c)
            hg = hl[gi]
            sI = fI * hg
            todo = np.ones(gi.size, bool)
            NB = np.zeros((gi.size, nsb), np.int64)
            WT = np.zeros((gi.size, nsb))
            while np.any(todo):
                if np.any(sI[todo] > 4.0 * hg[todo]):
                    bad = np.flatnonzero(todo & (sI > 4.0 * hg))[0]
                    raise ValueError(f"bodies: no fluid stencil for the image point of the face at "
                                     f"{tuple(float(Pg[d][bad]) for d in range(3))} (component {c}): body too close "
                                     "to a non-periodic side or another body, or features thinner than a few cells")
                t = np.flatnonzero(todo)
                I = [B[d][t] + sI[t] * nrm[d][t] for d in range(3)]
                nodes, wts = interp(ns, c, I, active)
                okn = (nodes >= 0) & fluid[np.maximum(nodes, 0)]
                ok = np.all(okn | (wts == 0.0), axis=1)
                NB[t[ok]] = np.where(nodes[ok] >= 0, nodes[ok], first_fluid)
                WT[t[ok]] = wts[ok]
                todo[t[ok]] = False
                sI[t[~ok]] += 0.25 * hg[t[~ok]]
            r = phs[gi] / sI  # <= 0: the ghost face lies on the far side of the surface
            a0[gi] = (1.0 - r) * UB
            w[gi] = r[:, None] * WT
            nb[gi] = NB
        out.append(dict(mask=fluid.astype(np.float64), idx=solid.astype(np.int64), a0=a0, nb=nb, w=w,
                        n_ghost=int(gi.size)))
    return out
