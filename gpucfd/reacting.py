"""Python wrapper of gpucfd's reacting solver (build it first: bash gpucfd/build.sh reacting).

Gas with shock waves that burns: the Euler equations for a mixture of ideal gases whose heat capacities change
with temperature, with the mass of every species carried along, and the reactions of every cell advanced on the
GPU inside each time step. Design and verification: gpucfd/REACTING.md. Arrays are numpy float64 of shape
(nz, ny, nx), x fastest; species arrays have the species first: (species, nz, ny, nx).

    from cfdlib import chem
    from gpucfd.reacting import Reacting
    mech = chem.build(chem.openfoam_conventions())                 # OpenFOAM's hydrogen scheme
    s = Reacting(mech, (1200, 1, 1), (0.12, 1.0, 1.0), sides={"x-": "wall", "x+": "wall"})
    x = s.centres("x")
    s.set(T=np.where(x < 1.5e-3, 2500.0, 300.0), p=np.where(x < 1.5e-3, 3.0e6, 101325.0), moles=dict(H2=2, O2=1, N2=3.76))
    st = s.run(5e-5, cfl=0.4)          # flow and chemistry to that time; st["steps"], st["ms"], st["p_peak"]
    f = s.get()                        # rho, u, v, w, p, T, c, E, Y (species first)
    s.guard()                          # how often the guard was used

The gases: a cfdlib.chem.Mechanism (species, thermodynamic data, reactions). Without reactions: inert(mech) keeps
a mechanism's species and drops its reactions; constant_gas(...) makes species with constant heat capacities.
Sides and parts of sides as in gpucfd.euler; a prescribed state is dict(T=, p=, Y= or moles=, u=, v=, w=).
"""
import ctypes
import os
from pathlib import Path

import numpy as np
import scipy.constants as const

from cfdlib import chem
from gpucfd.chemistry import METHOD, pack

LIB = Path(os.environ.get("GCFDR_LIB", Path(__file__).parent / "build" / "libgcfdr.so"))
# "mixed": the state is stored in double; fluxes, temperatures and reaction rates are computed in float
PRECISION = {"float": 4, "double": 8, "mixed": 6}
BC = {"periodic": 0, "wall": 1, "inflow": 2, "outflow": 3, "supersonic": 4}
RECON = {"first": 0, "minmod": 1, "vanleer": 2, "mc": 3, "central": 4, "weno5": 5}
FLUX = {"rusanov": 0, "hll": 1, "hllc": 2, "hybrid": 3}
SPLIT = {"flow-chemistry": 0, "chemistry-flow": 1, "strang": 2}
SIDES = {"x-": 0, "x+": 1, "y-": 2, "y+": 3, "z-": 4, "z+": 5}
AXES = "xyz"
NEXT = 7
# The chemistry setting a Reacting starts with: chosen by the tuning budget of
# dev/compare/detonation_tube_h2_air/plan.yaml (gpucfd/tests/test_react_chem.py, record test_react_chem.json).
# Picked on 2026-11-16: the explicit method at tolerance 1e-4 with the rule for quiet cells (23.5 million cells x
# steps per second at the state behind a detonation's shock against 18.5 for the chemistry kernel's own default).
CHEMISTRY = dict(method="bs23", rtol=1e-4, atol=1e-14, quiet=1.0, split="flow-chemistry", max_substeps=100000)
_lib = None
_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


def lib():
    global _lib
    if _lib is None:
        if not LIB.exists():
            raise FileNotFoundError(f"{LIB} missing: run  bash gpucfd/build.sh reacting")
        L = ctypes.CDLL(str(LIB))
        vp, d, i, l = ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_long
        L.gcfdr_create.restype = vp
        L.gcfdr_create.argtypes = [_ip, _dp, i, _ip, _dp]
        L.gcfdr_destroy.argtypes = [vp]
        L.gcfdr_set.argtypes = [vp, _dp, _dp, _dp]
        L.gcfdr_get.argtypes = [vp, _dp, _dp]
        L.gcfdr_scheme.argtypes = [vp, i, i, i, d]
        L.gcfdr_chemistry.argtypes = [vp, i, i, d, d, i, d, i]
        L.gcfdr_add_state.restype = i
        L.gcfdr_add_state.argtypes = [vp, _dp, d]
        L.gcfdr_side.argtypes = [vp, i, i, i]
        L.gcfdr_segment.restype = i
        L.gcfdr_segment.argtypes = [vp, i, _ip, _ip, i, i]
        L.gcfdr_reduce.argtypes = [vp, _dp]
        L.gcfdr_nred.restype = i
        L.gcfdr_nred.argtypes = [vp]
        L.gcfdr_bytes.restype = d
        L.gcfdr_bytes.argtypes = [vp]
        L.gcfdr_time.restype = d
        L.gcfdr_time.argtypes = [vp]
        L.gcfdr_weno_eps.argtypes = [vp, d]
        L.gcfdr_floor.argtypes = [vp, d]
        L.gcfdr_timed.argtypes = [vp, i]
        L.gcfdr_counts.argtypes = [vp, _ip, _dp, _dp, i]
        L.gcfdr_chem_stats.argtypes = [vp, _ip, _ip, _ip]
        L.gcfdr_run_fixed.restype = d
        L.gcfdr_run_fixed.argtypes = [vp, d, l, ctypes.c_void_p]
        L.gcfdr_advance.argtypes = [vp, d, d, i, l, _dp]
        _lib = L
    return _lib


class Gases(chem.Mechanism):
    """Species with their thermodynamic data and no reactions, in the form of a cfdlib.chem.Mechanism (so the
    same code reads it: heat capacities, enthalpies, the kernel's packing)."""

    def __init__(self, species, W, a_lo, a_hi, T_mid, conv, comp=None, elements=None):
        ns = len(species)
        self.conv, self.species, self.reactions = conv, list(species), []
        self.idx = {s: i for i, s in enumerate(self.species)}
        self.W, self.a_lo, self.a_hi, self.T_mid = (np.array(a, float) for a in (W, a_lo, a_hi, T_mid))
        self.elements = list(elements) if elements is not None else []
        self.comp = np.array(comp, float) if comp is not None else np.zeros((ns, 0))
        z = np.zeros(0)
        self.nu_f = self.nu_r = self.dnu = self.eff = np.zeros((0, ns))
        self.lnA = self.beta = self.Ta = self.lnA0 = self.beta0 = self.Ta0 = z
        self.third = self.fall = self.is_troe = np.zeros(0, bool)
        self.troe = np.zeros((0, 4))


def inert(mech):
    """The species of a mechanism with its reactions dropped."""
    return Gases(mech.species, mech.W, mech.a_lo, mech.a_hi, mech.T_mid, mech.conv, mech.comp, mech.elements)


def constant_gas(names=("gas",), W=0.02896, gamma=1.4):
    """Species with constant heat capacities (ratio gamma) and molar mass W (kg/mol); one value for all or one
    per species. The ideal gas of gpucfd.euler in the form the reacting solver reads."""
    ns = len(names)
    W, gamma = np.broadcast_to(np.asarray(W, float), (ns,)), np.broadcast_to(np.asarray(gamma, float), (ns,))
    a = np.zeros((ns, 7))
    a[:, 0] = gamma / (gamma - 1)                                 # cp / R
    conv = chem.Conventions("standard", R=const.R, R_cal=const.R / const.calorie, p_std=const.atm)
    return Gases(names, W, a, a, np.full(ns, 1000.0), conv)


class Blowup(RuntimeError):
    """The run produced a value that is not a number, or a cell with no gas or no pressure left."""


class Reacting:
    def __init__(self, mech, n, lengths, sides=None, precision="mixed", recon="vanleer", flux="hybrid", stages=3, sensor=0.5,
                 chemistry=True):
        """chemistry: True (the setting CHEMISTRY), False (no reactions), or a dict that changes entries of
        CHEMISTRY. A mechanism without reactions never reacts."""
        self.mech, self.n = mech, tuple(int(v) for v in n)
        self.h = tuple(float(L) / m for L, m in zip(lengths, self.n))
        self.precision, self.ns = precision, len(mech.species)
        self.shape = (self.n[2], self.n[1], self.n[0])
        self.cells = int(np.prod(self.n))
        self.volume = float(np.prod(self.h))
        im, dm = pack(mech)
        self._s = lib().gcfdr_create(np.array(self.n, np.int32), np.array(self.h, np.float64), PRECISION[precision], im, dm)
        if not self._s:
            raise ValueError("the reaction scheme is larger than the compiled limits (src/gcfdk_core.h)")
        self.nred = lib().gcfdr_nred(self._s)
        self.recon, self.flux, self.stages, self.sensor = recon, flux, stages, sensor
        self.scheme()
        self.chem = dict(CHEMISTRY, on=bool(chemistry) and len(mech.reactions) > 0)
        self.chemistry(**(chemistry if isinstance(chemistry, dict) else {}))
        for name, spec in (sides or {}).items():
            self.side(name, spec)

    # ---- set-up ----
    def scheme(self, recon=None, flux=None, stages=None, sensor=None):
        """As gpucfd.euler.Euler.scheme: reconstruction, flux formula, stages (1, 2 or 3), shock sensor."""
        self.recon, self.flux = recon or self.recon, flux or self.flux
        self.stages = stages or self.stages
        self.sensor = self.sensor if sensor is None else sensor
        lib().gcfdr_scheme(self._s, RECON[self.recon], FLUX[self.flux], self.stages, self.sensor)

    def chemistry(self, on=None, method=None, rtol=None, atol=None, quiet=None, split=None, max_substeps=None):
        """on: reactions switched on or off. method, rtol, atol, quiet: as gpucfd.chemistry.Chemistry.control
        ("bs23", "dp45", "ros34", "ros34a"). split: the order of the two parts of a time step, "flow-chemistry"
        (the default), "chemistry-flow" or "strang" (half the chemistry, the flow, the other half)."""
        for k, v in dict(on=on, method=method, rtol=rtol, atol=atol, quiet=quiet, split=split, max_substeps=max_substeps).items():
            if v is not None:
                self.chem[k] = v
        c = self.chem
        lib().gcfdr_chemistry(self._s, int(c["on"]), METHOD[c["method"]], float(c["rtol"]), float(c["atol"]), int(c["max_substeps"]),
                              float(c["quiet"]), SPLIT[c["split"]])

    def weno_eps(self, eps):
        lib().gcfdr_weno_eps(self._s, float(eps))

    def floor(self, T):
        """The guard's temperature floor (default 60 K): a cell that ends a stage colder gets the floor's energy."""
        lib().gcfdr_floor(self._s, float(T))

    def _mixture(self, Y=None, moles=None):
        """Mass fractions (species first) from Y or from relative moles {species: number}."""
        if moles is not None:
            x = np.array([moles.get(s, 0.0) for s in self.mech.species], np.float64)
            return x * self.mech.W / (x @ self.mech.W)
        return np.asarray(Y, np.float64)

    def _kind(self, spec):
        if isinstance(spec, str):
            if spec in ("inflow", "supersonic"):
                raise ValueError(f"side type {spec} needs a state: ('{spec}', dict(T=, p=, Y= or moles=, u=, v=, w=))")
            return BC[spec], 0
        kind, st = spec
        m = self.mech
        Y = self._mixture(st.get("Y"), st.get("moles"))
        sigma = Y / m.W
        rho = st["p"] / (m.conv.R * sigma.sum() * st["T"])
        arr = np.concatenate([rho * Y, [st.get("u", 0.0), st.get("v", 0.0), st.get("w", 0.0), st["p"]]]).astype(np.float64)
        k = lib().gcfdr_add_state(self._s, arr, chem.sound_speed_frozen(m, float(st["T"]), sigma))
        if k < 0:
            raise ValueError("too many prescribed states (8)")
        return BC[kind], k

    def side(self, name, spec):
        kind, st = self._kind(spec)
        lib().gcfdr_side(self._s, SIDES[name], kind, st)

    def segment(self, name, ranges, spec):
        """Part of a side: ranges = {axis: (first cell, one past the last)} for the directions along the side."""
        d = SIDES[name] // 2
        if spec == "periodic":
            raise ValueError("a part of a side cannot be periodic")
        lo, hi = np.zeros(2, np.int32), np.zeros(2, np.int32)
        for k, t in enumerate(((d + 1) % 3, (d + 2) % 3)):
            lo[k], hi[k] = ranges.get(AXES[t], (0, self.n[t]))
        kind, st = self._kind(spec)
        if lib().gcfdr_segment(self._s, SIDES[name], lo, hi, kind, st) < 0:
            raise ValueError("too many segments on one side (4)")

    # ---- fields ----
    def set(self, T=None, p=None, rho=None, Y=None, moles=None, u=0.0, v=0.0, w=0.0):
        """The state of every cell from two of temperature, pressure and density, the mixture (mass fractions Y,
        species first, or relative moles), and the velocity. Scalars are spread over the cells. Clears the time
        and every counter."""
        m = self.mech
        Y = self._mixture(Y, moles)
        Y = np.broadcast_to(Y.reshape(Y.shape + (1,) * (4 - Y.ndim)) if Y.ndim == 1 else Y, (self.ns,) + self.shape)
        rs = np.tensordot(1.0 / m.W, Y, axes=1) * m.conv.R        # gas constant of the mixture, J/(kg K)
        if rho is None:
            rho = np.asarray(p, np.float64) / (rs * np.asarray(T, np.float64))
        elif T is None:
            T = np.asarray(p, np.float64) / (rs * np.asarray(rho, np.float64))
        rho, T = (np.broadcast_to(np.asarray(a, np.float64), self.shape) for a in (rho, T))
        vel = np.stack([np.broadcast_to(np.asarray(a, np.float64), self.shape) for a in (u, v, w)])
        lib().gcfdr_set(self._s, np.ascontiguousarray(rho * Y).reshape(-1), np.ascontiguousarray(vel).reshape(-1), np.ascontiguousarray(T).reshape(-1))

    def get(self):
        """dict(rho, u, v, w, p, T, c (speed of sound), E (total energy per volume, chemical energy included),
        Y and rho_i (species first))."""
        U, pr = np.empty((self.ns + 4) * self.cells), np.empty(3 * self.cells)
        lib().gcfdr_get(self._s, U, pr)
        U, pr = U.reshape((self.ns + 4,) + self.shape), pr.reshape((3,) + self.shape)
        ri = U[:self.ns]
        rho = ri.sum(0)
        return dict(rho=rho, u=U[self.ns] / rho, v=U[self.ns + 1] / rho, w=U[self.ns + 2] / rho, E=U[self.ns + 3], T=pr[0], p=pr[1], c=pr[2],
                    rho_i=ri, Y=ri / rho)

    def centres(self, axis):
        k = AXES.index(axis)
        return (np.arange(self.n[k]) + 0.5) * self.h[k]

    # ---- diagnostics ----
    def _reduce(self):
        r = np.empty(self.nred, np.float64)
        lib().gcfdr_reduce(self._s, r)
        return r

    def _totals(self, r):
        """From rows of reductions (..., nred): the totals in the domain."""
        ns, V = self.ns, self.volume
        out = dict(species=r[..., :ns] * V, mass=r[..., :ns].sum(-1) * V, momentum=r[..., ns:ns + 3] * V, energy=r[..., ns + 3] * V)
        if self.mech.comp.shape[1]:
            out["atoms"] = (out["species"] / self.mech.W) @ self.mech.comp      # mol of atoms of each element (mech.elements)
        return out

    def _limits(self, r):
        o = self.ns + 4
        return dict(speed_over_h=r[..., o], rho_min=r[..., o + 1], p_min=r[..., o + 2], T_min=r[..., o + 3], T_max=r[..., o + 4],
                    p_max=r[..., o + 5], speed_max=r[..., o + 6])

    def totals(self):
        """Mass of each species, total mass, momentum (3), total energy, and atoms of each element in the domain
        (sums over the cells times the cell volume)."""
        return self._totals(self._reduce())

    def limits(self):
        """Largest signal speed over cell size, lowest density and pressure, lowest and highest temperature,
        highest pressure, highest gas speed."""
        return self._limits(self._reduce())

    def guard(self, clear=False):
        """Uses of the guard since set() (or the last clear): first_order (cells x stages x directions whose
        reconstruction fell back to the cell's own average, because a face had no pressure or no gas),
        species_flat (cells x stages x directions in which a species came out negative at a face and was put
        back to the cell's average, that species only), species (cells x stages with a negative species set to
        zero) with mass_moved (kg, summed), cold (cells x stages raised to the temperature floor) with
        energy_added (J, summed)."""
        g, a, ms = np.zeros(4, np.int32), np.zeros(2), np.zeros(2)
        lib().gcfdr_counts(self._s, g, a, ms, int(clear))
        return dict(first_order=int(g[0]), species_flat=int(g[3]), species=int(g[1]), cold=int(g[2]), mass_moved=float(a[0] * self.volume),
                    energy_added=float(a[1] * self.volume))

    def timed(self, on=True):
        """Measure the time of flow and of chemistry apart (the GPU is made to finish after each part)."""
        lib().gcfdr_timed(self._s, int(on))

    def times(self):
        """Seconds spent in flow and in chemistry since set(), when timed() is on."""
        g, a, ms = np.zeros(4, np.int32), np.zeros(2), np.zeros(2)
        lib().gcfdr_counts(self._s, g, a, ms, 0)
        return dict(flow_s=ms[0] / 1e3, chemistry_s=ms[1] / 1e3)

    def chem_stats(self):
        """Per cell (nz, ny, nx): accepted and rejected chemistry sub-steps since set(), and the status (0 ok,
        1 sub-step limit reached, 2 broke down)."""
        a, b, c = (np.empty(self.cells, np.int32) for _ in range(3))
        lib().gcfdr_chem_stats(self._s, a, b, c)
        return dict(substeps=a.reshape(self.shape), rejected=b.reshape(self.shape), status=c.reshape(self.shape))

    @property
    def time(self):
        return lib().gcfdr_time(self._s)

    @property
    def bytes(self):
        return lib().gcfdr_bytes(self._s)

    # ---- time stepping ----
    def run(self, tspan, cfl=0.4, every=1, max_steps=10 ** 9):
        """Advance by tspan. The time step is cfl / max over the cells of the sum over directions of
        (|u| + c) / h, found again every `every` steps. Returns steps, ms, the last dt, and p_peak: the highest
        pressure seen at any of those looks."""
        st = np.empty(9, np.float64)
        lib().gcfdr_advance(self._s, float(tspan), float(cfl), int(every), int(max_steps), st)
        out = dict(steps=int(st[0]), advanced=st[1], dt=st[2], ms=st[3], speed_over_h=st[4], rho_min=st[5], p_min=st[6],
                   status=int(st[7]), p_peak=st[8])
        if out["status"] == 1:
            raise Blowup(f"after {out['steps']} steps: max speed/h {st[4]}, lowest density {st[5]}, lowest pressure {st[6]} "
                         f"(scheme {self.recon}/{self.flux}, {self.stages} stages, cfl {cfl})")
        return out

    def run_fixed(self, dt, steps, history=False):
        """`steps` steps of size dt, nothing copied to the CPU. history=True: also, after every step, the totals
        (species, mass, momentum, energy, atoms) and the limits (as limits())."""
        hist = np.empty((steps, self.nred), np.float64) if history else None
        ms = lib().gcfdr_run_fixed(self._s, float(dt), int(steps), hist.ctypes.data if history else None)
        out = dict(steps=int(steps), dt=float(dt), ms=ms)
        if history:
            out.update(self._totals(hist))
            out.update(self._limits(hist))
        return out

    def close(self):
        if self._s:
            lib().gcfdr_destroy(self._s)
            self._s = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
