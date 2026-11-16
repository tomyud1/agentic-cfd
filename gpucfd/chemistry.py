"""Python wrapper of gpucfd's chemistry kernel (build it first: bash gpucfd/build.sh chemistry).

The reactions of N independent cells, advanced on the GPU over a time step, at constant pressure or at constant
volume. Design and verification: gpucfd/CHEMISTRY.md. The reaction scheme is a cfdlib.chem.Mechanism (read from
CHEMKIN files); nothing about a scheme is written here or in the kernel.

    from cfdlib import chem
    from gpucfd.chemistry import Chemistry
    mech = chem.build()                                   # OpenFOAM's hydrogen scheme, CHEMKIN's constants
    c = Chemistry(mech, n=4096, constant="volume")        # or "pressure"
    c.set(T=np.linspace(1000, 1400, 4096), p=202650.0, moles=dict(H2=1, O2=1, N2=3.76))
    c.control(level=1200.0)                               # also record when each cell first reaches 1200 K
    c.advance(1e-8, steps=1000)                           # 1000 steps of 1e-8 s; each cell takes its own sub-steps
    s = c.get()                                           # T, p, rho, Y (n, species), sigma (mol/kg)
    st = c.stats()                                        # sub-steps, rejected sub-steps, status, crossing times

Each cell keeps its enthalpy (constant pressure) or internal energy (constant volume) exactly: the temperature
is found from it. Species are mass fractions Y on the outside and moles per kg (sigma = Y / molar mass) inside.
"""
import ctypes
import os
from pathlib import Path

import numpy as np

LIB = Path(os.environ.get("GCFDK_LIB", Path(__file__).parent / "build" / "libgcfdk.so"))
# "mixed": species and energy stored in double, rates computed in float, reactions applied in double
PRECISION = {"float": 4, "double": 8, "mixed": 6}
# "ros34": the implicit method with its matrix found by differences; "ros34a": with the matrix written out exactly
METHOD = {"bs23": 0, "dp45": 1, "ros34": 2, "ros34a": 3}
MODE = {"pressure": 0, "volume": 1}
# The setting a Chemistry starts with. Integrator and tolerances: chosen by the tuning budget (48 runs,
# tests/test_chem_batch.py tune, record tests/results/test_chem_tune.json) as the cheapest at steps of 1e-8 s
# that keeps every cell within a tenth of case 20's required accuracy. Precision: chosen by the rule of
# tests/test_chem_precision.py. At steps of 1e-6 s and more "ros34" is about as cheap (CHEMISTRY.md).
DEFAULT = dict(precision="mixed", method="bs23", rtol=1e-6, atol=1e-14)
_lib = None
_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


def lib():
    global _lib
    if _lib is None:
        if not LIB.exists():
            raise FileNotFoundError(f"{LIB} missing: run  bash gpucfd/build.sh chemistry")
        L = ctypes.CDLL(str(LIB))
        vp, d, i, l = ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_long
        L.gcfdk_create.restype = vp
        L.gcfdk_create.argtypes = [l, i, i, _ip, _dp]
        L.gcfdk_destroy.argtypes = [vp]
        L.gcfdk_set.argtypes = [vp, _dp, _dp, _dp]
        L.gcfdk_get.argtypes = [vp, _dp, _dp, _dp, _dp]
        L.gcfdk_control.argtypes = [vp, i, d, d, i, i, d, d]
        L.gcfdk_advance.restype = d
        L.gcfdk_advance.argtypes = [vp, d, l, ctypes.c_void_p]
        L.gcfdk_stats.argtypes = [vp, _ip, _ip, _ip, _dp]
        L.gcfdk_rates.argtypes = [vp, _dp]
        L.gcfdk_jacobian.argtypes = [vp, _dp]
        L.gcfdk_bytes.restype = d
        L.gcfdk_bytes.argtypes = [vp]
        L.gcfdk_time.restype = d
        L.gcfdk_time.argtypes = [vp]
        _lib = L
    return _lib


def pack(mech):
    """A cfdlib.chem.Mechanism as the two arrays the kernel reads (layout: gcfdk_create in src/gcfdk.cu)."""
    ns, nr = len(mech.species), len(mech.reactions)
    act = [i for i in range(ns) if np.any(mech.dnu[:, i] != 0)]
    im, rows, per_reaction = [], [], []
    for j in range(nr):
        sides = []
        for nu in (mech.nu_f[j], mech.nu_r[j]):
            if np.any(nu != np.round(nu)):
                raise ValueError(f"reaction {j}: only whole numbers of molecules are handled")
            side = [i for i in range(ns) for _ in range(int(nu[i]))]
            if len(side) > 3:
                raise ValueError(f"reaction {j}: more than three molecules on one side")
            sides.append(side + [-1] * (3 - len(side)))
        kind, row = 0, -1
        troe = np.zeros(4)
        if mech.fall[j]:
            kind |= 2
            if mech.is_troe[j]:
                kind |= 4 | (8 if np.isfinite(mech.troe[j, 3]) else 0)
                troe = np.nan_to_num(mech.troe[j], nan=0.0)
        elif mech.third[j]:
            kind |= 1
        if kind & 3:
            row = len(rows)
            rows.append(mech.eff[j])
        im += sides[0] + sides[1] + [kind, row]
        per_reaction += [mech.lnA[j], mech.beta[j], mech.Ta[j], mech.lnA0[j], mech.beta0[j], mech.Ta0[j], *troe, mech.dnu[j].sum()]
    dm = [mech.conv.R, mech.conv.p_std]
    for i in range(ns):
        dm += [mech.T_mid[i], *mech.a_lo[i], *mech.a_hi[i]]
    dm += per_reaction
    for row in rows:
        dm += list(row)
    dm += list(mech.W)
    return np.array([ns, nr, len(act), len(rows)] + act + im, np.int32), np.array(dm, np.float64)


class Chemistry:
    def __init__(self, mech, n, constant="pressure", precision=None, method=None, rtol=None, atol=None):
        precision, method = precision or DEFAULT["precision"], method or DEFAULT["method"]
        rtol, atol = rtol or DEFAULT["rtol"], atol or DEFAULT["atol"]
        self.mech, self.n, self.constant, self.precision = mech, int(n), constant, precision
        self.ns, self.nr = len(mech.species), len(mech.reactions)
        im, dm = pack(mech)
        self._s = lib().gcfdk_create(self.n, PRECISION[precision], MODE[constant], im, dm)
        if not self._s:
            raise ValueError("the reaction scheme is larger than the kernel's compiled limits (src/gcfdk.cu)")
        self.method, self.rtol, self.atol, self.fixed, self.max_substeps, self.level, self.quiet = method, rtol, atol, 0, 100000, 0.0, 0.0
        self.control()

    # ---- set-up ----
    def control(self, method=None, rtol=None, atol=None, fixed=None, max_substeps=None, level=None, quiet=None):
        """method: "bs23" (explicit, orders 3 and 2), "dp45" (explicit, 5 and 4), "ros34" (implicit, 4 and 3, its
        matrix by differences) or "ros34a" (the same with its matrix written out exactly: fewer rate evaluations,
        and atoms kept to round-off in mixed precision too).
        rtol, atol: a sub-step is accepted when its estimated error is below atol (mol/kg) + rtol x the species.
        fixed > 0: that many equal sub-steps per step and no error control. level > 0: record, per cell, when the
        temperature first reaches it (a straight line over the step in which it does). quiet > 0: a cell whose
        species would change over the whole step, at their present rates, by less than quiet times the allowed
        error takes one plain step and no more rate evaluations (cold gas; 0 switches the rule off)."""
        for k, v in dict(method=method, rtol=rtol, atol=atol, fixed=fixed, max_substeps=max_substeps, level=level, quiet=quiet).items():
            if v is not None:
                setattr(self, k, v)
        lib().gcfdk_control(self._s, METHOD[self.method], float(self.rtol), float(self.atol), int(self.fixed), int(self.max_substeps), float(self.level), float(self.quiet))

    def set(self, T, p=None, rho=None, Y=None, moles=None, sigma=None):
        """The state of every cell: temperature, pressure or density, and the mixture as mass fractions Y
        (n, species), as moles per kg sigma (n, species), or as relative moles {species: number}. Scalars and
        single rows are spread over the cells. Clears the time, the counters and the crossing times."""
        m = self.mech
        if moles is not None:
            x = np.array([moles.get(s, 0.0) for s in m.species], np.float64)
            sigma = x / (x @ m.W)
        elif Y is not None:
            sigma = np.asarray(Y, np.float64) / m.W
        sig = np.ascontiguousarray(np.broadcast_to(np.asarray(sigma, np.float64), (self.n, self.ns)).T)        # (species, n)
        T = np.ascontiguousarray(np.broadcast_to(np.asarray(T, np.float64), (self.n,)))
        if (p is None) == (rho is None):
            raise ValueError("give the pressure or the density, one of the two")
        if self.constant == "pressure":
            pv = np.broadcast_to(np.asarray(p if p is not None else rho * sig.sum(0) * m.conv.R * T, np.float64), (self.n,))
        else:
            pv = np.broadcast_to(np.asarray(1.0 / rho if rho is not None else sig.sum(0) * m.conv.R * T / p, np.float64), (self.n,))
        self.pv = np.ascontiguousarray(pv)
        lib().gcfdk_set(self._s, T, self.pv, sig.reshape(-1))

    # ---- state ----
    def get(self):
        """dict(T, p, rho, Y (n, species), sigma (n, species, mol/kg), energy (J/kg, as stored: enthalpy or
        internal energy), h (the sub-step each cell would take next, s))."""
        T, e, h = (np.empty(self.n) for _ in range(3))
        sig = np.empty(self.ns * self.n)
        lib().gcfdk_get(self._s, T, e, sig, h)
        sig = sig.reshape(self.ns, self.n).T
        return dict(T=T, sigma=sig, Y=sig * self.mech.W, energy=e, h=h, **self._state(T, sig))

    def _state(self, T, sig):
        rt = sig.sum(-1) * self.mech.conv.R * T
        return dict(p=self.pv, rho=self.pv / rt) if self.constant == "pressure" else dict(p=rt / self.pv, rho=1.0 / self.pv)

    def stats(self):
        """Per cell: accepted and rejected sub-steps since set(), status (0 ok, 1 sub-step limit, 2 broke down),
        and the time of the first crossing of the level (nan: not reached)."""
        a, b, c = (np.empty(self.n, np.int32) for _ in range(3))
        tc = np.empty(self.n)
        lib().gcfdk_stats(self._s, a, b, c, tc)
        return dict(substeps=a, rejected=b, status=c, t_cross=np.where(tc < 0, np.nan, tc))

    def rates(self):
        """What the kernel computes for the present state, for the tests: dict(qf, qr (n, reactions; mol/(m3 s)),
        dsigma_dt (n, species; mol/(kg s)), h, cp (n, species; per mole), T (found from the stored energy), v)."""
        nq = 2 * self.nr + 3 * self.ns + 2
        out = np.empty(nq * self.n)
        lib().gcfdk_rates(self._s, out)
        o = out.reshape(self.n, nq)
        nr, ns = self.nr, self.ns
        return dict(qf=o[:, :nr], qr=o[:, nr:2 * nr], dsigma_dt=o[:, 2 * nr:2 * nr + ns], h=o[:, 2 * nr + ns:2 * nr + 2 * ns],
                    cp=o[:, 2 * nr + 2 * ns:2 * nr + 3 * ns], T=o[:, -2], v=o[:, -1])

    def jacobian(self):
        """For the tests: the matrix the method "ros34a" uses, (n, reacting species, reacting species): how the
        rate of change of each reacting species (mol/(kg s)) changes with each reacting species (mol/kg) while the
        cell's energy and its volume (or pressure) are held. Second value: the indices of the reacting species."""
        act = [i for i in range(self.ns) if np.any(self.mech.dnu[:, i] != 0)]
        out = np.empty(len(act) ** 2 * self.n)
        lib().gcfdk_jacobian(self._s, out)
        return out.reshape(self.n, len(act), len(act)), act

    @property
    def time(self):
        return lib().gcfdk_time(self._s)

    @property
    def bytes(self):
        return lib().gcfdk_bytes(self._s)

    # ---- time stepping ----
    def advance(self, dt, steps=1, history=False):
        """`steps` steps of size dt. Returns dict(ms, steps, dt); with history=True also T (steps, n) and sigma
        (steps, n, species) after every step (held on the GPU during the run: steps x cells x (species + 1)
        numbers, so for small numbers of cells)."""
        hist = np.empty((steps, 1 + self.ns, self.n)) if history else None
        ms = lib().gcfdk_advance(self._s, float(dt), int(steps), hist.ctypes.data if history else None)
        out = dict(ms=ms, steps=int(steps), dt=float(dt))
        if history:
            out.update(T=hist[:, 0, :], sigma=np.swapaxes(hist[:, 1:, :], 1, 2))
        return out

    def close(self):
        if self._s:
            lib().gcfdk_destroy(self._s)
            self._s = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
