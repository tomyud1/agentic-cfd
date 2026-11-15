"""Python wrapper of gpucfd's compressible solver (build it first: bash gpucfd/build.sh compressible).

The Euler equations for an ideal gas with a constant ratio of specific heats, with shocks. Design and
verification: gpucfd/COMPRESSIBLE.md. Arrays are numpy float64 of shape (nz, ny, nx), x fastest.

    from gpucfd.euler import Euler
    s = Euler((400, 1, 1), (1.0, 1.0, 1.0), gamma=1.4, sides={"x-": "wall", "x+": "wall"})
    s.set(rho=rho, u=0.0, p=p)
    st = s.run(5e-4, cfl=0.8)          # to that time; st["steps"], st["ms"], ...
    f = s.get()                        # dict of rho, u, v, w, p, E

Sides are named "x-", "x+", "y-", "y+", "z-", "z+". Each is "periodic", "wall" (frictionless), "outflow"
(zero gradient), or ("inflow", state) / ("supersonic", state) with state = dict(rho=, u=, v=, w=, p=).
"inflow" puts the state beyond the side and lets the flux formula decide what crosses; "supersonic" sends
exactly the state's own flux through the side (right when the gas enters faster than sound).
One side can be split: s.segment("x-", {"y": (j0, j1)}, ("supersonic", state)) changes the cells j0..j1-1.
"""
import ctypes
import os
from pathlib import Path

import numpy as np

LIB = Path(os.environ.get("GCFDC_LIB", Path(__file__).parent / "build" / "libgcfdc.so"))
# "mixed": the state is stored in double, the fluxes are computed in float (conserves like double, see COMPRESSIBLE.md)
PRECISION = {"float": 4, "double": 8, "mixed": 6}
BC = {"periodic": 0, "wall": 1, "inflow": 2, "outflow": 3, "supersonic": 4}
RECON = {"first": 0, "minmod": 1, "vanleer": 2, "mc": 3, "central": 4, "weno5": 5}
FLUX = {"rusanov": 0, "hll": 1, "hllc": 2, "hybrid": 3}
SIDES = {"x-": 0, "x+": 1, "y-": 2, "y+": 3, "z-": 4, "z+": 5}
AXES = "xyz"
_lib = None
_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


def lib():
    global _lib
    if _lib is None:
        if not LIB.exists():
            raise FileNotFoundError(f"{LIB} missing: run  bash gpucfd/build.sh compressible")
        L = ctypes.CDLL(str(LIB))
        vp, d, i, l = ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_long
        L.gcfdc_create.restype = vp
        L.gcfdc_create.argtypes = [_ip, _dp, d, i]
        L.gcfdc_destroy.argtypes = [vp]
        L.gcfdc_set.argtypes = [vp, _dp]
        L.gcfdc_get.argtypes = [vp, _dp]
        L.gcfdc_scheme.argtypes = [vp, i, i, i, d]
        L.gcfdc_add_state.restype = i
        L.gcfdc_add_state.argtypes = [vp, _dp]
        L.gcfdc_side.argtypes = [vp, i, i, i]
        L.gcfdc_segment.restype = i
        L.gcfdc_segment.argtypes = [vp, i, _ip, _ip, i, i]
        L.gcfdc_reduce.argtypes = [vp, _dp]
        L.gcfdc_bytes.restype = d
        L.gcfdc_bytes.argtypes = [vp]
        L.gcfdc_time.restype = d
        L.gcfdc_time.argtypes = [vp]
        L.gcfdc_weno_eps.argtypes = [vp, d]
        L.gcfdc_run_fixed.restype = d
        L.gcfdc_run_fixed.argtypes = [vp, d, l, ctypes.c_void_p]
        L.gcfdc_advance.argtypes = [vp, d, d, i, l, _dp]
        _lib = L
    return _lib


class Blowup(RuntimeError):
    """The run produced a value that is not a number, or a cell with no gas or no pressure left."""


class Euler:
    def __init__(self, n, lengths, gamma=1.4, sides=None, precision="mixed", recon="vanleer", flux="hybrid",
                 stages=3, sensor=0.5):
        self.n = tuple(int(v) for v in n)
        self.h = tuple(float(L) / m for L, m in zip(lengths, self.n))
        self.gamma, self.precision = float(gamma), precision
        self.shape = (self.n[2], self.n[1], self.n[0])
        self.cells = int(np.prod(self.n))
        self.volume = float(np.prod(self.h))
        self._s = lib().gcfdc_create(np.array(self.n, np.int32), np.array(self.h, np.float64), self.gamma, PRECISION[precision])
        self.scheme(recon, flux, stages, sensor)
        for name, spec in (sides or {}).items():
            self.side(name, spec)

    # ---- set-up ----
    def scheme(self, recon=None, flux=None, stages=None, sensor=None):
        """recon: first, minmod, vanleer, mc, central (no limiter) or weno5. flux: rusanov, hll, hllc, or hybrid
        (HLLC, with HLL on faces that lie along a strong shock). stages: 1, 2 or 3 (Euler, SSP-RK2, SSP-RK3).
        sensor: relative pressure jump between neighbours that counts as a strong shock (hybrid only)."""
        self.recon, self.flux = recon or self.recon, flux or self.flux
        self.stages = stages or self.stages
        self.sensor = self.sensor if sensor is None else sensor
        lib().gcfdc_scheme(self._s, RECON[self.recon], FLUX[self.flux], self.stages, self.sensor)

    def weno_eps(self, eps):
        """WENO5 only: differences below sqrt(eps) of the values themselves are treated as smooth. Default 1e-20
        in double and 1e-10 in float, where smaller values let round-off noise steer the scheme (rung 2)."""
        self._weno_eps = float(eps)
        lib().gcfdc_weno_eps(self._s, float(eps))

    def _kind(self, spec):
        if isinstance(spec, str):
            if spec in ("inflow", "supersonic"):
                raise ValueError(f"side type {spec} needs a state: ('{spec}', dict(rho=, u=, v=, w=, p=))")
            return BC[spec], 0
        kind, st = spec
        arr = np.array([st["rho"], st.get("u", 0.0), st.get("v", 0.0), st.get("w", 0.0), st["p"]], np.float64)
        k = lib().gcfdc_add_state(self._s, arr)
        if k < 0:
            raise ValueError("too many prescribed states (8)")
        return BC[kind], k

    def side(self, name, spec):
        d = SIDES[name] // 2
        kind, st = self._kind(spec)
        lib().gcfdc_side(self._s, SIDES[name], kind, st)

    def segment(self, name, ranges, spec):
        """Part of a side: ranges = {axis: (first cell, one past the last)} for the directions along the side."""
        d = SIDES[name] // 2
        if spec == "periodic":
            raise ValueError("a part of a side cannot be periodic")
        lo, hi = np.zeros(2, np.int32), np.zeros(2, np.int32)
        for k, t in enumerate(((d + 1) % 3, (d + 2) % 3)):
            lo[k], hi[k] = ranges.get(AXES[t], (0, self.n[t]))
        kind, st = self._kind(spec)
        if lib().gcfdc_segment(self._s, SIDES[name], lo, hi, kind, st) < 0:
            raise ValueError("too many segments on one side (4)")

    # ---- fields ----
    def set_conserved(self, U):
        """U: array (5, nz, ny, nx) of rho, rho u, rho v, rho w, E."""
        U = np.ascontiguousarray(np.broadcast_to(np.asarray(U, np.float64), (5,) + self.shape))
        lib().gcfdc_set(self._s, U.reshape(-1))

    def get_conserved(self):
        U = np.empty(5 * self.cells, np.float64)
        lib().gcfdc_get(self._s, U)
        return U.reshape((5,) + self.shape)

    def set(self, rho, u=0.0, v=0.0, w=0.0, p=None):
        rho, u, v, w, p = (np.broadcast_to(np.asarray(a, np.float64), self.shape) for a in (rho, u, v, w, p))
        self.set_conserved(np.stack([rho, rho * u, rho * v, rho * w, p / (self.gamma - 1) + 0.5 * rho * (u * u + v * v + w * w)]))

    def get(self):
        U = self.get_conserved()
        rho = U[0]
        u, v, w = U[1] / rho, U[2] / rho, U[3] / rho
        return dict(rho=rho, u=u, v=v, w=w, E=U[4], p=(self.gamma - 1) * (U[4] - 0.5 * rho * (u * u + v * v + w * w)))

    def centres(self, axis):
        k = AXES.index(axis)
        return (np.arange(self.n[k]) + 0.5) * self.h[k]

    # ---- diagnostics ----
    def _reduce(self):
        r = np.empty(8, np.float64)
        lib().gcfdc_reduce(self._s, r)
        return r

    def totals(self):
        """Mass, momentum (3) and energy in the domain (sums over the cells times the cell volume; in 1D and 2D
        the unused directions have the lengths given at creation)."""
        r = self._reduce()
        return dict(mass=r[0] * self.volume, momentum=tuple(r[1:4] * self.volume), energy=r[4] * self.volume)

    def limits(self):
        r = self._reduce()
        return dict(speed_over_h=r[5], rho_min=r[6], p_min=r[7])

    @property
    def time(self):
        return lib().gcfdc_time(self._s)

    @property
    def bytes(self):
        return lib().gcfdc_bytes(self._s)

    # ---- time stepping ----
    def run(self, tspan, cfl=0.8, every=1, max_steps=10 ** 9):
        """Advance by tspan. The time step is cfl / max over the cells of the sum over directions of
        (|u| + c) / h, found again every `every` steps (a copy of a few numbers to the CPU each time)."""
        st = np.empty(8, np.float64)
        lib().gcfdc_advance(self._s, float(tspan), float(cfl), int(every), int(max_steps), st)
        out = dict(steps=int(st[0]), advanced=st[1], dt=st[2], ms=st[3], speed_over_h=st[4], rho_min=st[5], p_min=st[6],
                   status=int(st[7]))
        if out["status"] == 1:
            raise Blowup(f"after {out['steps']} steps: max speed/h {st[4]}, lowest density {st[5]}, lowest pressure {st[6]} "
                         f"(scheme {self.recon}/{self.flux}, {self.stages} stages, cfl {cfl})")
        return out

    def run_fixed(self, dt, steps, history=False):
        """`steps` steps of size dt, nothing copied to the CPU. history=True: also the totals after every step,
        as arrays mass, momentum (steps, 3), energy, and the lowest density and pressure."""
        hist = np.empty((steps, 8), np.float64) if history else None
        ms = lib().gcfdc_run_fixed(self._s, float(dt), int(steps), hist.ctypes.data if history else None)
        out = dict(steps=int(steps), dt=float(dt), ms=ms)
        if history:
            out.update(mass=hist[:, 0] * self.volume, momentum=hist[:, 1:4] * self.volume, energy=hist[:, 4] * self.volume,
                       rho_min=hist[:, 6], p_min=hist[:, 7])
        return out

    def close(self):
        if self._s:
            lib().gcfdc_destroy(self._s)
            self._s = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
