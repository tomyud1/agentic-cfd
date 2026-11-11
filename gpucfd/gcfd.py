"""Python wrapper of the gpucfd CUDA library (build it first: bash gpucfd/build.sh).

Arrays are numpy float64 of shape (nz, ny, nx) (x fastest), interior cells only.

    from gpucfd import gcfd
    s = gcfd.Poisson((64, 64, 1), (1.0, 1.0, 1.0), ["neumann"] * 4 + ["periodic"] * 2)
    x, hist, ms = s.solve(f, tol=1e-8, method="pcg")
"""
import ctypes
from pathlib import Path

import numpy as np

BC = {"periodic": 0, "neumann": 1, "dirichlet": 2}
# "mixed": double-precision outer iteration (solution, residual) with a single-precision multigrid inside
PRECISION = {"double": 8, "float": 4, "mixed": 6}
LIB = Path(__file__).parent / "build" / "libgcfd.so"
_lib = None

_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


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
        L.gcfd_ns_create.argtypes = [_ip, _dp, ctypes.c_double, ctypes.c_int]
        L.gcfd_ns_destroy.argtypes = [ctypes.c_void_p]
        L.gcfd_ns_set.argtypes = [ctypes.c_void_p, ctypes.c_int, _dp]
        L.gcfd_ns_get.argtypes = [ctypes.c_void_p, ctypes.c_int, _dp]
        L.gcfd_ns_run.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_double, _dp]
        L.gcfd_ns_project.restype = ctypes.c_double
        L.gcfd_ns_project.argtypes = [ctypes.c_void_p]
        _lib = L
    return _lib


def device():
    buf = ctypes.create_string_buffer(256)
    cc = lib().gcfd_device(buf, 256)
    return dict(name=buf.value.decode(), compute_capability=cc / 10)


def bandwidth(mbytes=512, reps=20):
    """Measured device copy bandwidth (GB/s, read + write)."""
    return lib().gcfd_bandwidth(float(mbytes), int(reps))


class Poisson:
    """Laplacian(p) = f on a box of n = (nx, ny, nz) cells and size length = (lx, ly, lz).
    bc: 6 names (x-, x+, y-, y+, z-, z+) from BC. nz = 1 makes it 2D. precision: "double" or "float".
    nu: smoothing sweeps before and after the coarse-grid correction."""

    def __init__(self, n, length, bc, precision="double", nu=(2, 2)):
        self.n = tuple(int(v) for v in n)
        self.shape = (self.n[2], self.n[1], self.n[0])
        self.bc = list(bc)
        self.precision = precision
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
    """Incompressible Navier-Stokes on a periodic box (staggered grid, all fields in double precision).
    Velocity component c is stored on the low c-faces of the cells: u[k, j, i] at x = i h, y = (j + 1/2) h,
    z = (k + 1/2) h, and so on. nz = 1 makes it 2D. pressure: "mixed" or "double" (projection solver)."""

    def __init__(self, n, length, nu, pressure="mixed"):
        self.n = tuple(int(v) for v in n)
        self.shape = (self.n[2], self.n[1], self.n[0])
        self.length = tuple(float(v) for v in length)
        self.ncomp = 3 if self.n[2] > 1 else 2
        h = lib().gcfd_ns_create(np.array(self.n, np.int32), np.array(length, np.float64), float(nu),
                                 PRECISION[pressure])
        if not h:
            raise ValueError("gcfd: invalid grid (even cell counts, at least 2 in x and y)")
        self.h = h

    def close(self):
        if getattr(self, "h", None):
            lib().gcfd_ns_destroy(self.h)
            self.h = None

    __del__ = close

    def face_coords(self, comp):
        """Coordinates (Z, Y, X) of the faces where component comp is stored."""
        c = []
        for d in range(3):
            h = self.length[d] / self.n[d]
            c.append(np.arange(self.n[d]) * h + (0.0 if d == comp else 0.5 * h))
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

    def run(self, dt, nsteps, ptol=1e-10):
        """Advance; returns dict(ms (GPU time), max_div (after), poisson_its (mean per projection))."""
        st = np.zeros(3)
        lib().gcfd_ns_run(self.h, float(dt), int(nsteps), float(ptol), st)
        if not np.isfinite(st[1]):
            hmin = min(self.length[d] / self.n[d] for d in range(3) if self.n[d] > 1)
            raise FloatingPointError(f"gcfd: solution blew up (dt {dt:g}; stability needs roughly dt < 0.8 h / max|u| "
                                     f"and dt < 0.2 h^2 / nu with h = {hmin:g})")
        return dict(ms=float(st[0]), max_div=float(st[1]), poisson_its=float(st[2]))
