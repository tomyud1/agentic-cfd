"""Python wrapper of the gpucfd CUDA library (build it first: bash gpucfd/build.sh).

Arrays are numpy float64 of shape (nz, ny, nx) (x fastest), interior cells only.

    from gpucfd import gcfd
    s = gcfd.Poisson((64, 64, 1), (1.0, 1.0, 1.0), ["neumann"] * 4 + ["periodic"] * 2)
    x, hist, ms = s.solve(f, tol=1e-8, method="pcg")
"""
import ctypes
import os
from pathlib import Path

import numpy as np

BC = {"periodic": 0, "neumann": 1, "dirichlet": 2}
# "mixed": double-precision outer iteration (solution, residual) with a single-precision multigrid inside
PRECISION = {"double": 8, "float": 4, "mixed": 6}
# GCFD_LIB: another build of the library (used to check that a change leaves results bit-identical)
LIB = Path(os.environ.get("GCFD_LIB", Path(__file__).parent / "build" / "libgcfd.so"))
# Explicit diffusion limit of scheme "rk3": dt <= VISCOUS_DT h_min^2 / nu (the one place this factor is written)
VISCOUS_DT = 0.15
_lib = None

_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_lp = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")


def lib():
    global _lib
    if _lib is None:
        if not LIB.exists():
            raise FileNotFoundError(f"{LIB} missing: run  bash gpucfd/build.sh")
        L = ctypes.CDLL(str(LIB))
        L.gcfd_poisson_create.restype = ctypes.c_void_p
        L.gcfd_poisson_create.argtypes = [_ip, _dp, _ip, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.gcfd_poisson_destroy.argtypes = [ctypes.c_void_p]
        L.gcfd_poisson_levels.restype = ctypes.c_int
        L.gcfd_poisson_levels.argtypes = [ctypes.c_void_p, _ip]
        L.gcfd_poisson_apply.argtypes = [ctypes.c_void_p, _dp, _dp]
        L.gcfd_poisson_solve.restype = ctypes.c_int
        L.gcfd_poisson_solve.argtypes = [ctypes.c_void_p, _dp, _dp, ctypes.c_double, ctypes.c_int, ctypes.c_int,
                                         _dp, _dp]
        L.gcfd_poisson_bench_vcycle.restype = ctypes.c_double
        L.gcfd_poisson_bench_vcycle.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.gcfd_bandwidth.restype = ctypes.c_double
        L.gcfd_bandwidth.argtypes = [ctypes.c_double, ctypes.c_int]
        L.gcfd_device.restype = ctypes.c_int
        L.gcfd_device.argtypes = [ctypes.c_char_p, ctypes.c_int]
        L.gcfd_ns_create.restype = ctypes.c_void_p
        L.gcfd_ns_create.argtypes = [_ip, _dp, ctypes.c_double, ctypes.c_int, _ip, _dp, _dp]
        if hasattr(L, "gcfd_ns_create2"):  # inflow/outflow (absent in builds before 2026-11-12)
            L.gcfd_ns_create2.restype = ctypes.c_void_p
            L.gcfd_ns_create2.argtypes = [_ip, _dp, ctypes.c_double, ctypes.c_int, _ip, _dp]
            L.gcfd_ns_set_bnd.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, _dp]
            L.gcfd_ns_get_bnd.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, _dp]
        L.gcfd_ns_destroy.argtypes = [ctypes.c_void_p]
        L.gcfd_ns_set.argtypes = [ctypes.c_void_p, ctypes.c_int, _dp]
        L.gcfd_ns_get.argtypes = [ctypes.c_void_p, ctypes.c_int, _dp]
        L.gcfd_ns_run.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_double, _dp]
        L.gcfd_ns_scheme.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        L.gcfd_set_graphs.argtypes = [ctypes.c_int]
        if hasattr(L, "gcfd_set_tail"):  # absent in builds before 2026-11-12 (compared by same_as_build)
            L.gcfd_set_tail.argtypes = [ctypes.c_longlong]
        if hasattr(L, "gcfd_ns_body"):  # immersed bodies, pressure and right-hand side (from 2026-11-12)
            L.gcfd_ns_body.argtypes = [ctypes.c_void_p, ctypes.c_int, _dp, ctypes.c_int, _lp, _dp, _lp, _dp]
            L.gcfd_ns_body_done.argtypes = [ctypes.c_void_p]
            L.gcfd_ns_get_p.argtypes = [ctypes.c_void_p, _dp]
            L.gcfd_ns_rhs.argtypes = [ctypes.c_void_p, ctypes.c_int, _dp]
        L.gcfd_ns_project.restype = ctypes.c_double
        L.gcfd_ns_project.argtypes = [ctypes.c_void_p]
        if hasattr(L, "gcfd_ns_create3"):  # stretched grids (from 2026-11-13)
            vp = ctypes.c_void_p
            L.gcfd_ns_create3.restype = ctypes.c_void_p
            L.gcfd_ns_create3.argtypes = [_ip, _dp, ctypes.c_double, ctypes.c_int, _ip, _dp, vp, vp, vp]
            L.gcfd_poisson_create_s.restype = ctypes.c_void_p
            L.gcfd_poisson_create_s.argtypes = [_ip, _dp, _ip, ctypes.c_int, ctypes.c_int, ctypes.c_int, vp, vp, vp]
            L.gcfd_set_lines.argtypes = [ctypes.c_int]
            L.gcfd_set_ns_sweeps.argtypes = [ctypes.c_int]
        if hasattr(L, "gcfd_ns_body_ok"):  # bodies on stretched grids (from 2026-11-13, stage 2)
            L.gcfd_ns_body_ok.restype = ctypes.c_int
            L.gcfd_ns_body_ok.argtypes = [ctypes.c_void_p]
        if hasattr(L, "gcfd_ns_set_p"):  # IMEX with bodies, and setting the pressure (from 2026-11-13, stage 3)
            L.gcfd_ns_set_p.argtypes = [ctypes.c_void_p, _dp]
        _lib = L
    return _lib


def device():
    buf = ctypes.create_string_buffer(256)
    cc = lib().gcfd_device(buf, 256)
    return dict(name=buf.value.decode(), compute_capability=cc / 10)


def use_graphs(on=True):
    """CUDA graphs (default on): each pressure solve or time step is recorded once and replayed with one
    launch, loops decided on the GPU. Off: the CPU launches every kernel. Results are bit-identical."""
    lib().gcfd_set_graphs(1 if on else 0)


def set_tail(cells=4096):
    """Multigrid levels with at most this many cells run inside one thread block (one launch for the
    whole coarse end of a V-cycle); 0 switches it off. Results are bit-identical. Applies to graphs
    recorded afterwards (create solvers after calling it)."""
    lib().gcfd_set_tail(int(cells))


def set_lines(directions="xyz"):
    """Stretched grids, for experiments: the directions whose bounded sides may use zebra line relaxation
    in the pressure multigrid ("xyz", the default: every bounded direction; "": the point smoother).
    Applies to solvers created afterwards."""
    lib().gcfd_set_lines(sum(1 << "xyz".index(c) for c in directions))


def set_ns_sweeps(sweeps=2):
    """Stretched grids: smoothing sweeps before and after the coarse-grid correction in the pressure
    solver of NavierStokes objects created afterwards."""
    lib().gcfd_set_ns_sweeps(int(sweeps))


def _faces(n, length, grid):
    """Per direction: None (uniform) or the n + 1 face positions as a float64 array; and the C arguments."""
    faces = [None, None, None]
    for d, g in enumerate(grid or ()):
        if g is None:
            continue
        a = np.ascontiguousarray(g, np.float64)
        if a.shape != (n[d] + 1,):
            raise ValueError(f"grid[{d}]: {n[d] + 1} face positions expected (n + 1), got shape {a.shape}")
        if n[d] == 1:
            raise ValueError(f"grid[{d}]: a direction with one cell cannot be stretched")
        faces[d] = a
    return faces, [None if a is None else a.ctypes.data_as(ctypes.c_void_p) for a in faces]


def bandwidth(mbytes=512, reps=20):
    """Measured device copy bandwidth (GB/s, read + write)."""
    return lib().gcfd_bandwidth(float(mbytes), int(reps))


class Poisson:
    """Laplacian(p) = f on a box of n = (nx, ny, nz) cells and size length = (lx, ly, lz).
    bc: 6 names (x-, x+, y-, y+, z-, z+) from BC. nz = 1 makes it 2D. precision: "double" or "float".
    nu: smoothing sweeps before and after the coarse-grid correction.
    grid: stretched grid, per direction None (uniform) or the n + 1 face positions from 0 to the length
    (gpucfd.grids makes them); bounded directions only. The equation is then Laplacian(p) = f with the
    cell sizes from the lists, the residual norms are volume-weighted, and a singular problem needs
    sum(f * cell volume) = 0 and returns p with zero volume-weighted mean."""

    def __init__(self, n, length, bc, precision="double", nu=(2, 2), grid=None):
        self.n = tuple(int(v) for v in n)
        self.shape = (self.n[2], self.n[1], self.n[0])
        self.bc = list(bc)
        self.precision = precision
        self.faces, fargs = _faces(self.n, length, grid)
        self.stretched = any(a is not None for a in self.faces)
        if self.stretched:
            h = lib().gcfd_poisson_create_s(np.array(self.n, np.int32), np.array(length, np.float64),
                                            np.array([BC[b] for b in bc], np.int32), PRECISION[precision],
                                            int(nu[0]), int(nu[1]), *fargs)
        else:
            h = lib().gcfd_poisson_create(np.array(self.n, np.int32), np.array(length, np.float64),
                                          np.array([BC[b] for b in bc], np.int32), PRECISION[precision],
                                          int(nu[0]), int(nu[1]))
        if not h:
            raise ValueError("gcfd: invalid grid or boundary conditions (see stderr)")
        self.h = h

    def close(self):
        if getattr(self, "h", None):
            lib().gcfd_poisson_destroy(self.h)
            self.h = None

    __del__ = close

    def levels(self):
        dims = np.zeros(3 * 64, np.int32)
        k = lib().gcfd_poisson_levels(self.h, dims)
        return [tuple(dims[3 * q:3 * q + 3]) for q in range(k)]

    def apply(self, p):
        p = np.ascontiguousarray(p, np.float64).reshape(self.shape)
        out = np.empty_like(p)
        lib().gcfd_poisson_apply(self.h, p.ravel(), out.ravel())
        return out

    def solve(self, f, x0=None, tol=1e-8, maxit=100, method="pcg"):
        """Returns (x, hist, ms): solution, relative residual after each iteration, GPU time (ms, data on
        the GPU, excluding host transfers)."""
        f = np.ascontiguousarray(f, np.float64).reshape(self.shape)
        x = np.zeros(self.shape) if x0 is None else np.array(x0, np.float64).reshape(self.shape)
        hist = np.zeros(maxit + 1)
        ms = np.zeros(1)
        it = lib().gcfd_poisson_solve(self.h, f.ravel(), x.ravel(), float(tol), int(maxit),
                                      {"mg": 0, "pcg": 1}[method], hist, ms)
        return x, hist[:it + 1], float(ms[0])

    def bench_vcycle(self, reps=20):
        """GPU milliseconds per V-cycle (call after a solve, which leaves a right-hand side on the GPU)."""
        return lib().gcfd_poisson_bench_vcycle(self.h, int(reps))


class NavierStokes:
    """Incompressible Navier-Stokes on a box (staggered grid, all fields in double precision).
    Velocity component c is stored on the low c-faces of the cells: u[k, j, i] at x = i h, y = (j + 1/2) h,
    z = (k + 1/2) h, and so on. nz = 1 makes it 2D. pressure: "mixed" or "double" (projection solver).
    walls: per direction, True for walls at both ends (no slip), False for periodic. On a wall direction
    the normal component's face 0 is the low wall and stays zero.
    wall_velocity: {"x-"|"x+"|"y-"|"y+"|"z-"|"z+": (ux, uy, uz)} for moving walls (tangential part used).
    force: constant body force per unit mass (gx, gy, gz).
    boundaries: per side, overriding walls: {"x-": spec, ...} with spec one of
        "wall" or ("wall", (ux, uy, uz))      no slip (moving wall: tangential part used)
        ("inflow", (u, v, w))                 prescribed velocity, uniform
        ("inflow", f)                         prescribed velocity profile: f(x, y, z) -> (u, v, w), called
                                              with arrays of the points on the side
        "outflow"                             pressure zero, zero normal gradient of the velocity
      Both sides of a direction must be set (or neither: periodic). Without an outflow the prescribed
      velocities must carry zero net flow (incompressible), else ValueError.
    scheme: "rk3" (explicit, third order, time step limited by 0.15 h^2 / nu as well as the flow speed)
    or "imex" (diffusion implicit by Crank-Nicolson, second order, time step limited by the flow speed
    only). htol: relative tolerance of the implicit diffusion solves.
    bodies: immersed solid bodies (gpucfd.bodies: Cylinder, Sphere, Box, HalfSpace, Inside, SDF), no
    slip on their surfaces (with their velocity if given). See bodies.py for the method; forces() gives
    the force on them, surface_pressure() the pressure on their surface. image_distance: distance of the
    ghost faces' image points from the surface, in cells (default sqrt(dim) + 0.05, the least that keeps
    their interpolation stencils in the fluid).
    grid: stretched grid, per direction None (uniform) or the n + 1 face positions from 0 to the length
    (gpucfd.grids.tanh clusters cells toward walls, gpucfd.grids.clustered around a body); bounded
    directions only. With bodies the ghost rules use the local cell size (bodies.build). Use dt_cfl()
    for the time step: max_dt() takes the smallest cell with the largest velocity."""

    SIDES = ("x-", "x+", "y-", "y+", "z-", "z+")

    def __init__(self, n, length, nu, pressure="mixed", walls=(False, False, False), wall_velocity=None,
                 force=(0.0, 0.0, 0.0), scheme="rk3", htol=1e-10, boundaries=None, bodies=None, image_distance=None,
                 grid=None):
        self.n = tuple(int(v) for v in n)
        self.shape = (self.n[2], self.n[1], self.n[0])
        self.length = tuple(float(v) for v in length)
        self.ncomp = 3 if self.n[2] > 1 else 2
        self.walls = tuple(bool(w) for w in walls)
        self.faces, fargs = _faces(self.n, self.length, grid)
        self.stretched = any(a is not None for a in self.faces)
        if self.stretched:
            kinds, planes = self._sides(wall_velocity or {}, boundaries or {})
            self.kinds = kinds
            h = lib().gcfd_ns_create3(np.array(self.n, np.int32), np.array(length, np.float64), float(nu),
                                      PRECISION[pressure], np.array(kinds, np.int32), np.array(force, np.float64),
                                      *fargs)
            if not h:
                raise ValueError("gcfd: invalid set-up (see stderr)")
            self.h = h
            for (side, comp), a in planes.items():
                lib().gcfd_ns_set_bnd(h, side, comp, np.ascontiguousarray(a, np.float64).ravel())
        elif boundaries is None:  # walls only: the original interface (also understood by older builds)
            self.kinds = [1 if self.walls[s // 2] else 0 for s in range(6)]
            uw = np.zeros((6, 3))
            for side, vel in (wall_velocity or {}).items():
                uw[self.SIDES.index(side)] = vel
            h = lib().gcfd_ns_create(np.array(self.n, np.int32), np.array(length, np.float64), float(nu),
                                     PRECISION[pressure], np.array(self.walls, np.int32), uw.ravel(),
                                     np.array(force, np.float64))
            if not h:
                raise ValueError("gcfd: invalid grid (periodic directions need even cell counts; at least 2 "
                                 "cells in x and y; no z walls in 2D)")
            self.h = h
        else:
            kinds, planes = self._sides(wall_velocity or {}, boundaries)
            self.kinds = kinds
            h = lib().gcfd_ns_create2(np.array(self.n, np.int32), np.array(length, np.float64), float(nu),
                                      PRECISION[pressure], np.array(kinds, np.int32), np.array(force, np.float64))
            if not h:
                raise ValueError("gcfd: invalid set-up (see stderr)")
            self.h = h
            for (side, comp), a in planes.items():
                lib().gcfd_ns_set_bnd(h, side, comp, np.ascontiguousarray(a, np.float64).ravel())
        self.nu = float(nu)
        self.scheme = scheme
        self.g = tuple(float(v) for v in force)
        lib().gcfd_ns_scheme(h, {"rk3": 0, "imex": 1}[scheme], float(htol))
        self.bodies = list(bodies or [])
        self.fluid = None  # per component: True on fluid faces (bodies only)
        if self.bodies:
            if scheme != "rk3" and not hasattr(lib(), "gcfd_ns_set_p"):
                # Builds before 2026-11-13 (stage 3): IMEX with bodies grows unstably there (DESIGN.md,
                # "IMEX with immersed bodies": a pressure gradient inside the body, and the ghost faces'
                # answer to the pressure correction fed back into the pressure).
                raise ValueError('this build of the library needs scheme="rk3" with bodies (rebuild: bash gpucfd/build.sh)')
            from gpucfd import bodies as _b
            rules = _b.build(self, self.bodies, s_image=image_distance)
            for c, r in enumerate(rules):
                lib().gcfd_ns_body(h, c, r["mask"], int(r["idx"].size), r["idx"], r["a0"],
                                   np.ascontiguousarray(r["nb"].ravel()), np.ascontiguousarray(r["w"].ravel()))
            lib().gcfd_ns_body_done(h)
            if self.stretched and not (hasattr(lib(), "gcfd_ns_body_ok") and lib().gcfd_ns_body_ok(h)):
                raise ValueError("gcfd: bodies refused on this stretched grid (see stderr)")
            self.fluid = [r["mask"].reshape(self.shape) > 0 for r in rules]
            self.n_ghost = [r["n_ghost"] for r in rules]

    def _sides(self, wall_velocity, boundaries):
        spec = {}
        for s, side in enumerate(self.SIDES):
            if self.walls[s // 2]:
                spec[side] = ("wall", wall_velocity.get(side, (0.0, 0.0, 0.0)))
        for side, b in boundaries.items():
            if side not in self.SIDES:
                raise ValueError(f"unknown side {side!r} (one of {self.SIDES})")
            spec[side] = (b, None) if isinstance(b, str) else tuple(b)
        kinds, planes = [0] * 6, {}
        for s, side in enumerate(self.SIDES):
            if side not in spec:
                continue
            kind, val = spec[side]
            if kind == "outflow":
                kinds[s] = 2
                continue
            if kind not in ("wall", "inflow"):
                raise ValueError(f"{side}: unknown boundary {kind!r} (wall, inflow or outflow)")
            kinds[s] = 1
            d = s // 2
            for comp in range(3):
                X, Y, Z = self.side_coords(side, comp)
                if val is None:
                    v = 0.0
                elif callable(val):
                    r = val(X, Y, Z)
                    v = r[comp] if comp < len(r) else 0.0
                else:
                    v = val[comp] if comp < len(val) else 0.0
                if kind == "wall" and comp == d:
                    v = 0.0  # a wall carries no flow through it
                planes[(s, comp)] = np.broadcast_to(np.asarray(v, np.float64), X.shape).copy()
        for d in range(3):
            if (kinds[2 * d] == 0) != (kinds[2 * d + 1] == 0):
                raise ValueError(f"direction {'xyz'[d]}: set both sides (or neither, for periodic)")
        if 2 not in kinds:
            net = sum(self._plane_flux(s, planes[(s, s // 2)]) for s in range(6) if kinds[s] == 1)
            scale = sum(abs(self._plane_flux(s, planes[(s, s // 2)])) for s in range(6) if kinds[s] == 1)
            if abs(net) > 1e-12 * max(scale, 1e-300) and abs(net) > 1e-300:
                raise ValueError(f"no outflow side, but the prescribed velocities carry a net outflow of {net:.3g}: "
                                 "an incompressible flow needs zero (add an outflow side or balance the inflow)")
        return kinds, planes

    def side_coords(self, side, comp):
        """Coordinates (X, Y, Z) where component comp is prescribed on a side: the side's plane, at that
        component's face positions along the side (shape (n_b, n_a) over the two other directions a < b)."""
        s = self.SIDES.index(side)
        d = s // 2
        c = []
        for e in range(3):
            if e == d:
                c.append(np.array([0.0 if s % 2 == 0 else self.length[e]]))
            else:
                c.append(self._axis(e, e == comp))
        Z, Y, X = np.meshgrid(c[2], c[1], c[0], indexing="ij")
        a, b = [e for e in range(3) if e != d]
        shp = (self.n[b], self.n[a])
        return X.reshape(shp), Y.reshape(shp), Z.reshape(shp)

    def _axis(self, d, face):
        """Positions along direction d: the low faces of the cells (face=True) or the cell centres."""
        if self.faces[d] is None:
            h = self.length[d] / self.n[d]
            return np.arange(self.n[d]) * h + (0.0 if face else 0.5 * h)
        f = self.faces[d]
        return f[:-1].copy() if face else 0.5 * (f[:-1] + f[1:])

    def widths(self, d):
        """Cell widths along direction d (n values)."""
        if self.faces[d] is None:
            return np.full(self.n[d], self.length[d] / self.n[d])
        return np.diff(self.faces[d])

    def _hmin(self):
        return min(float(self.widths(d).min()) for d in range(3) if self.n[d] > 1)

    def _plane_flux(self, s, un):
        """Volume flow OUT of the domain through side s for normal velocity un on its plane."""
        d = s // 2
        if self.stretched:
            a, b = [e for e in range(3) if e != d]
            area = np.outer(self.widths(b), self.widths(a))
            return float(np.sum(np.asarray(un).reshape(area.shape) * area)) * (1.0 if s % 2 else -1.0)
        area = np.prod([self.length[e] / self.n[e] for e in range(3) if e != d])
        return float(np.sum(un) * area) * (1.0 if s % 2 else -1.0)

    def get_boundary(self, side, comp):
        """Boundary data of a side: the prescribed velocity component, or on an outflow side the current
        normal velocity on the boundary face (comp = the side's direction)."""
        s = self.SIDES.index(side)
        a, b = [e for e in range(3) if e != s // 2]
        out = np.zeros((self.n[b], self.n[a]))
        lib().gcfd_ns_get_bnd(self.h, s, comp, out.ravel())
        return out

    def flux(self, side):
        """Volume flow out of the domain through a side (negative: inflow). 2D: per unit depth times lz."""
        s = self.SIDES.index(side)
        d = s // 2
        if self.kinds[s] == 0:
            raise ValueError(f"{side} is periodic")
        if s % 2 == 0:  # the low boundary face is stored: face 0 of the normal component
            un = np.take(self.get()[d], 0, axis=2 - d)
        else:
            un = self.get_boundary(side, d)
        return self._plane_flux(s, un)

    def close(self):
        if getattr(self, "h", None):
            lib().gcfd_ns_destroy(self.h)
            self.h = None

    __del__ = close

    def face_coords(self, comp):
        """Coordinates (X, Y, Z) of the faces where component comp is stored (each of shape (nz, ny, nx))."""
        c = [self._axis(d, d == comp) for d in range(3)]
        Z, Y, X = np.meshgrid(c[2], c[1], c[0], indexing="ij")
        return X, Y, Z

    def set(self, comps):
        for s, a in enumerate(comps):
            lib().gcfd_ns_set(self.h, s, np.ascontiguousarray(a, np.float64).reshape(self.shape).ravel())

    def get(self):
        out = []
        for s in range(self.ncomp):
            a = np.empty(self.shape)
            lib().gcfd_ns_get(self.h, s, a.ravel())
            out.append(a)
        return out

    def project(self):
        """Make the current velocity discretely divergence-free; returns max |div u| afterwards."""
        return lib().gcfd_ns_project(self.h)

    def set_pressure(self, p):
        """Set the pressure that the incremental projection carries (IMEX, or any scheme with bodies), to
        continue from a stored state: set(velocity) and set_pressure(pressure()) of that state."""
        if not (self.scheme == "imex" or self.bodies):
            raise ValueError("the explicit scheme without bodies carries no pressure between steps")
        lib().gcfd_ns_set_p(self.h, np.ascontiguousarray(p, np.float64).reshape(self.shape).ravel())

    def run(self, dt, nsteps, ptol=1e-10):
        """Advance; returns dict(ms (GPU time), max_div (after), poisson_its (mean per projection),
        helm_its (IMEX: mean CG iterations per diffusion solve))."""
        st = np.zeros(4)
        lib().gcfd_ns_run(self.h, float(dt), int(nsteps), float(ptol), st)
        if not np.isfinite(st[1]):
            hmin = self._hmin()
            limit = "dt < 0.8 h / max|u|" + (" and dt < 0.2 h^2 / nu" if self.scheme == "rk3" else "")
            raise FloatingPointError(f"gcfd: solution blew up (dt {dt:g}; stability needs roughly {limit} "
                                     f"with h = {hmin:g})")
        return dict(ms=float(st[0]), max_div=float(st[1]), poisson_its=float(st[2]), helm_its=float(st[3]))

    def max_dt(self, umax, cfl=0.8):
        """Largest stable time step for velocity scale umax: convective limit, plus the explicit
        diffusion limit for scheme "rk3"."""
        lim = self.dt_limits(umax, cfl)
        return lim["convective"] if lim["viscous"] is None else min(lim["convective"], lim["viscous"])

    def dt_limits(self, umax, cfl=0.8):
        """The two time-step limits for velocity scale umax, on the smallest cell h_min:
        dict(convective = cfl h_min / umax, viscous = VISCOUS_DT h_min^2 / nu, or None for scheme "imex",
        which has no such limit, h_min). max_dt is the smaller of the two."""
        h = self._hmin() if self.stretched else min(self.length[d] / self.n[d] for d in range(3) if self.n[d] > 1)
        return dict(convective=cfl * h / umax, viscous=None if self.scheme == "imex" else VISCOUS_DT * h * h / self.nu, h_min=h)

    def dt_cfl(self, cfl=0.8):
        """Convective time-step limit from the current velocity: cfl / max over cells of
        (|u| / dx + |v| / dy + |w| / dz), each velocity the larger of the cell's two faces. On a stretched
        grid this is far less strict than max_dt (small cells at a no-slip wall carry small velocities).
        Infinite for a fluid at rest. Scheme "rk3" also needs dt < 0.15 h_min^2 / nu (included)."""
        r = self.courant_rate()
        dt = np.inf if r == 0.0 else cfl / r
        return dt if self.scheme == "imex" else min(dt, VISCOUS_DT * self._hmin() ** 2 / self.nu)

    def courant_rate(self, q=None):
        """Largest over the cells of |u| / dx + |v| / dy + |w| / dz (each velocity the larger of the cell's
        two faces), from the current velocity or from q = get(). Times a time step it is the Courant number
        of that step, with each cell's own size: on a grid stretched toward a wall this is the number that
        matters, where the largest velocity over the smallest cell overstates it many times. An observe
        function can return it as 'courant_rate' for cfdlib.gpu.run_transient's Courant check."""
        q = self.get() if q is None else q
        rate = np.zeros(self.shape)
        for d in range(self.ncomp):
            a = np.abs(q[d])
            ax = 2 - d
            if self.kinds[2 * d] == 0:
                hi = np.roll(a, -1, axis=ax)
            else:  # the high boundary face is not stored: the last cell uses its low face
                hi = np.concatenate((np.take(a, range(1, self.n[d]), axis=ax), np.take(a, [-1], axis=ax)), axis=ax)
            shp = [1, 1, 1]
            shp[ax] = -1
            rate += np.maximum(a, hi) / self.widths(d).reshape(shp)
        return float(rate.max())

    # ----------------------------------------------------------------- pressure, forces (post-processing)
    def pressure(self):
        """Pressure per unit density at the cell centres (shape (nz, ny, nx)), defined up to a constant
        unless there is an outflow. Inside bodies the values are meaningless (cells decoupled there)."""
        p = np.empty(self.shape)
        lib().gcfd_ns_get_p(self.h, p.ravel())
        return p

    def rhs(self, comp):
        """Explicit momentum right-hand side of component comp at its faces: - convection + nu Laplacian +
        body force (no pressure term), from the current velocity."""
        out = np.empty(self.shape)
        lib().gcfd_ns_rhs(self.h, int(comp), out.ravel())
        return out

    def cell_coords(self):
        if self.stretched:
            c = [self._axis(d, False) for d in range(3)]
        else:
            c = [(np.arange(self.n[d]) + 0.5) * self.length[d] / self.n[d] for d in range(3)]
        Z, Y, X = np.meshgrid(c[2], c[1], c[0], indexing="ij")
        return X, Y, Z

    def forces(self, box):
        """Force per unit density of the fluid on whatever lies inside box = ((x0, x1), (y0, y1), (z0, z1))
        (2D: per unit depth; any z range). The box must lie in the fluid and away from non-periodic sides.

        Exact discrete momentum balance: summed over the fluid faces in the box, the solver's own momentum
        equation telescopes to the fluxes through the box surface plus those into the bodies, so
            F_on_body = - [ sum_box (rhs - grad p - g) vol + sum_surface (flux out) area ]
        with the solver's discrete fluxes (convection, viscous stress, pressure). It includes the unsteady
        term (rhs - grad p is the momentum change rate) and gives the same force for any box around the
        same bodies, to round-off (rung 10 checks it)."""
        if self.stretched:
            return self._forces_s(box)
        q = self.get()
        p = self.pressure()
        act = [d for d in range(3) if self.n[d] > 1]
        h = [self.length[d] / self.n[d] for d in range(3)]
        vol = float(np.prod([h[d] for d in act]))
        per = [self.kinds[2 * d] == 0 for d in range(3)]
        F = np.zeros(3)

        def take(a, idx):  # a[k, j, i] at index triples (lists over directions), wrapped if periodic
            I = []
            for d in range(3):
                i = np.asarray(idx[d])
                if per[d]:
                    i = np.mod(i, self.n[d])
                elif np.any(i < 0) or np.any(i >= self.n[d]):
                    raise ValueError("forces: the box reaches a non-periodic side")
                I.append(i)
            return a[I[2], I[1], I[0]]

        for c in range(self.ncomp):
            R = self.rhs(c) - self.g[c]
            rng = []
            for d in range(3):
                if self.n[d] == 1:
                    rng.append(np.array([0]))
                    continue
                off = 0.0 if d == c else 0.5 * h[d]
                lo = int(np.ceil((box[d][0] - off) / h[d] - 1e-9))
                hi = int(np.floor((box[d][1] - off) / h[d] + 1e-9))
                if hi < lo:
                    raise ValueError("forces: empty box")
                rng.append(np.arange(lo, hi + 1))
            K, J, I = np.meshgrid(rng[2], rng[1], rng[0], indexing="ij")
            idx = [I, J, K]
            e = [np.eye(3, dtype=int)[d] for d in range(3)]

            def sh(ix, v):
                return [ix[d] + v[d] for d in range(3)]
            fl = take(self.fluid[c], idx) if self.fluid is not None else np.ones(I.shape, bool)
            gp = (take(p, idx) - take(p, sh(idx, -e[c]))) / h[c]
            S = np.sum(np.where(fl, take(R, idx) - gp, 0.0)) * vol

            def flux(d, ix):  # flux of c-momentum in direction d at the + side of faces ix
                qf, qp = take(q[c], ix), take(q[c], sh(ix, e[d]))
                if d == c:
                    U = 0.5 * (qf + qp)
                    return U * U - self.nu * (qp - qf) / h[d] + take(p, ix)
                a = 0.5 * (take(q[d], sh(ix, e[d])) + take(q[d], sh(ix, e[d] - e[c])))
                return a * 0.5 * (qf + qp) - self.nu * (qp - qf) / h[d]
            out = 0.0
            for d in act:
                top = [x.take([-1], axis=2 - d) for x in idx]
                bot = [x.take([0], axis=2 - d) for x in idx]
                bot = sh(bot, -e[d])
                if self.fluid is not None:
                    for ix in (top, sh(top, e[d]), bot):
                        if not np.all(take(self.fluid[c], ix)):
                            raise ValueError("forces: the box surface cuts a body")
                out += np.sum(flux(d, top) - flux(d, bot)) * vol / h[d]
            F[c] = -(S + out)
        return F

    def pressure_at(self, points):
        """Pressure interpolated (bi- or trilinear, from cell centres) at points (array (m, 3))."""
        P = np.atleast_2d(np.asarray(points, np.float64))
        if self.stretched:
            return self._pressure_at_s(P)
        p = self.pressure()
        act = [d for d in range(3) if self.n[d] > 1]
        lo, fr = [], []
        for d in range(3):
            if d not in act:
                lo.append(np.zeros(len(P), int))
                fr.append(np.zeros(len(P)))
                continue
            t = P[:, d] / (self.length[d] / self.n[d]) - 0.5
            i0 = np.floor(t).astype(int)
            lo.append(i0)
            fr.append(t - i0)
        out = np.zeros(len(P))
        for c2 in (0, 1) if 2 in act else (0,):
            for c1 in (0, 1):
                for c0 in (0, 1):
                    cm = (c0, c1, c2)
                    w = np.ones(len(P))
                    ix = []
                    for d in range(3):
                        w = w * ((fr[d] if cm[d] else 1 - fr[d]) if d in act else 1.0)
                        i = lo[d] + (cm[d] if d in act else 0)
                        i = np.mod(i, self.n[d]) if self.kinds[2 * d] == 0 else np.clip(i, 0, self.n[d] - 1)
                        ix.append(i)
                    out += w * p[ix[2], ix[1], ix[0]]
        return out

    def surface_pressure(self, points, s=(1.5, 2.5, 3.5)):
        """Pressure on a body's surface at points (on or near the surface): sampled along the outward
        normal at distances s (times the largest cell size) in the fluid, extrapolated to the surface by
        the quadratic through the samples (the interpolation stencils then hold fluid cells only)."""
        from gpucfd import bodies as _b
        P = np.atleast_2d(np.asarray(points, np.float64))
        act = [d for d in range(3) if self.n[d] > 1]
        hmax = max(self.length[d] / self.n[d] for d in act)
        X = [P[:, d] for d in range(3)]
        ph, which = _b.phi(self.bodies, *X)
        nrm = _b.normals(self.bodies, which, X, act, 1e-6 * hmax)
        B = [X[d] - ph * nrm[d] for d in range(3)]
        if self.stretched:  # the local cell size: the largest width of the cell that holds the surface point
            hl = np.zeros(len(P))
            for d in act:
                f = self.faces[d] if self.faces[d] is not None else np.linspace(0.0, self.length[d], self.n[d] + 1)
                i = np.clip(np.searchsorted(f, B[d], side="right") - 1, 0, self.n[d] - 1)
                hl = np.maximum(hl, np.diff(f)[i])
            sn = np.asarray(s, np.float64)
            vals = [self.pressure_at(np.stack([B[d] + si * hl * nrm[d] for d in range(3)], 1)) for si in sn]
            out = np.zeros(len(P))
            for a, si in enumerate(sn):  # Lagrange weights at distance 0 (they depend on the ratios only)
                w = np.prod([(0.0 - sj) / (si - sj) for b, sj in enumerate(sn) if b != a])
                out += w * vals[a]
            return out
        sv = np.asarray(s, np.float64) * hmax
        vals = [self.pressure_at(np.stack([B[d] + si * nrm[d] for d in range(3)], 1)) for si in sv]
        out = np.zeros(len(P))
        for a, si in enumerate(sv):  # Lagrange weights at distance 0
            w = np.prod([(0.0 - sj) / (si - sj) for b, sj in enumerate(sv) if b != a])
            out += w * vals[a]
        return out

    # ------------------------------------------------------------------------------ stretched grids
    def _forces_s(self, box):
        """forces() on a stretched grid: the same balance with the true box of every velocity unknown
        (width dm, centre to centre, along its own direction; the cell width dc across) and the fluxes of
        the stretched stencil (src/gcfd.cu, rhs_s)."""
        q = self.get()
        p = self.pressure()
        act = [d for d in range(3) if self.n[d] > 1]
        per = [self.kinds[2 * d] == 0 for d in range(3)]
        dc = [self.widths(d) for d in range(3)]
        # per face 0..n: distance between the two cell centres, and the share of the flow through the
        # half cell on the face's high side (a mirror cell beyond a bounded side; periodic: uniform)
        dm = [np.concatenate(([w[0]], 0.5 * (w[:-1] + w[1:]), [w[-1]])) for w in dc]
        fw = [np.concatenate(([0.5], w[1:] / (w[:-1] + w[1:]), [0.5])) for w in dc]
        F = np.zeros(3)

        def wrap(d, i):
            return np.mod(i, self.n[d]) if per[d] else i

        def take(a, idx):
            I = []
            for d in range(3):
                i = np.asarray(idx[d])
                if per[d]:
                    i = np.mod(i, self.n[d])
                elif np.any(i < 0) or np.any(i >= self.n[d]):
                    raise ValueError("forces: the box reaches a non-periodic side")
                I.append(i)
            return a[I[2], I[1], I[0]]

        for c in range(self.ncomp):
            R = self.rhs(c) - self.g[c]
            rng = []
            for d in range(3):
                if self.n[d] == 1:
                    rng.append(np.array([0]))
                    continue
                x = self._axis(d, d == c)
                tol = 1e-9 * self.length[d]
                sel = np.flatnonzero((x >= box[d][0] - tol) & (x <= box[d][1] + tol))
                if sel.size == 0:
                    raise ValueError("forces: empty box")
                rng.append(sel)
            K, J, I = np.meshgrid(rng[2], rng[1], rng[0], indexing="ij")
            idx = [I, J, K]
            e = [np.eye(3, dtype=int)[d] for d in range(3)]

            def sh(ix, v):
                return [ix[d] + v[d] for d in range(3)]

            def width(d, ix):  # width of this component's box along d
                return (dm[d] if d == c else dc[d])[wrap(d, ix[d])]
            vol = np.ones(I.shape)
            for d in act:
                vol = vol * width(d, idx)
            fl = take(self.fluid[c], idx) if self.fluid is not None else np.ones(I.shape, bool)
            gp = (take(p, idx) - take(p, sh(idx, -e[c]))) / dm[c][wrap(c, idx[c])]
            S = np.sum(np.where(fl, take(R, idx) - gp, 0.0) * vol)

            def flux(d, ix):  # flux of c-momentum in direction d at the + side of faces ix
                qf, qp = take(q[c], ix), take(q[c], sh(ix, e[d]))
                if d == c:  # the + side is the centre of cell ix: midway between its two faces
                    U = 0.5 * (qf + qp)
                    return U * U - self.nu * (qp - qf) / dc[d][wrap(d, ix[d])] + take(p, ix)
                f = fw[c][wrap(c, ix[c])]
                a = f * take(q[d], sh(ix, e[d])) + (1.0 - f) * take(q[d], sh(ix, e[d] - e[c]))
                return a * 0.5 * (qf + qp) - self.nu * (qp - qf) / dm[d][wrap(d, ix[d]) + 1]
            out = 0.0
            for d in act:
                top = [x.take([-1], axis=2 - d) for x in idx]
                bot = [x.take([0], axis=2 - d) for x in idx]
                bot = sh(bot, -e[d])
                if self.fluid is not None:
                    for ix in (top, sh(top, e[d]), bot):
                        if not np.all(take(self.fluid[c], ix)):
                            raise ValueError("forces: the box surface cuts a body")
                area = np.ones(top[0].shape)
                for d2 in act:
                    if d2 != d:
                        area = area * width(d2, top)
                out += np.sum((flux(d, top) - flux(d, bot)) * area)
            F[c] = -(S + out)
        return F

    def _pressure_at_s(self, P):
        """pressure_at() on a stretched grid: linear between the true cell centres (the nearest centre's
        value beyond the first or last one, as on a uniform grid)."""
        p = self.pressure()
        act = [d for d in range(3) if self.n[d] > 1]
        lo, hi, fr = [], [], []
        for d in range(3):
            if d not in act:
                lo.append(np.zeros(len(P), int))
                hi.append(np.zeros(len(P), int))
                fr.append(np.zeros(len(P)))
                continue
            n = self.n[d]
            if self.kinds[2 * d] == 0:  # periodic: uniform
                t = P[:, d] / (self.length[d] / n) - 0.5
                i0 = np.floor(t).astype(int)
                lo.append(np.mod(i0, n))
                hi.append(np.mod(i0 + 1, n))
                fr.append(t - i0)
                continue
            xc = self._axis(d, False)
            i0 = np.searchsorted(xc, P[:, d], side="right") - 1
            a, b = np.clip(i0, 0, n - 1), np.clip(i0 + 1, 0, n - 1)
            dx = xc[b] - xc[a]
            lo.append(a)
            hi.append(b)
            fr.append(np.where(dx > 0, (P[:, d] - xc[a]) / np.where(dx > 0, dx, 1.0), 0.0))
        out = np.zeros(len(P))
        for c2 in (0, 1) if 2 in act else (0,):
            for c1 in (0, 1):
                for c0 in (0, 1):
                    cm = (c0, c1, c2)
                    w = np.ones(len(P))
                    ix = []
                    for d in range(3):
                        w = w * ((fr[d] if cm[d] else 1 - fr[d]) if d in act else 1.0)
                        ix.append(hi[d] if (cm[d] and d in act) else lo[d])
                    out += w * p[ix[2], ix[1], ix[0]]
        return out
