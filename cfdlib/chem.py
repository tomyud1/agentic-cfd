"""Chemistry without flow: a reader for CHEMKIN reaction schemes, the reaction rates, a well-mixed parcel at
constant pressure, and the chemical equilibrium. An independent second answer next to a reacting solver.

Written by the solver agent of case 20 (ignition of hydrogen and air) as its independent solution and moved here
unchanged in substance (cases/h2_air_ignition_box/mech.py is the record of the original). There it agreed with
OpenFOAM's chemFoam to 3e-7 in the ignition time once both used the same constants, and with the CHEMKIN II
reference to 0.03 %.

Nothing here uses OpenFOAM's chemistry: the two CHEMKIN files are parsed by this module, the rate laws are
those of the CHEMKIN-II manual (Kee, Rupley and Miller, Sandia report SAND89-8009, 1989; cited from memory),
and the equations are integrated with SciPy. What it gives:
  - theory before a run: the start state, bounds on the final temperature, the equilibrium the mixture is
    heading for (`Parcel.complete_combustion`, `Parcel.equilibrium`);
  - a second solution with stated tolerances (`Parcel.solve`; at constant volume `VolumeParcel.solve`);
  - what a difference of conventions between two programs is worth: `Conventions` (the constant that converts
    activation energies, the standard pressure of the thermodynamic data, the atomic weights);
  - the health of a reacting history: `parcel_health` (atoms, sum of mass fractions, temperature bounds).

    from cfdlib import chem
    mech = chem.build(chem.openfoam_conventions())                 # OpenFOAM's hydrogen scheme, read as OpenFOAM reads it
    parcel = chem.Parcel(mech, 202650.0, 1000.0, dict(H2=1, O2=1, N2=3.76))
    sol = parcel.solve(1e-3, T_ign=1200.0)                         # t, T, sigma (mol/kg), t_ign

Units inside: mol, m, s, K, J. A parcel is described per kilogram of gas: sigma_i = moles of species i per kg.
"""
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.constants as const
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

CM3 = 1.0e-6                         # m^3 in a cm^3: CHEMKIN's rate constants are in cm, mol, s


@dataclass
class Conventions:
    """The constants a program uses when it reads the scheme. The scheme's files do not fix them."""
    name: str
    R: float                         # universal gas constant, J/(mol K)
    R_cal: float                     # the same in cal/(mol K): converts the activation energies of chem.inp
    p_std: float                     # standard pressure of the thermodynamic data, Pa (in the equilibrium constants)
    weights: dict = field(default_factory=dict)   # atomic weights, g/mol (only mass fractions and density need them)


def standard_conventions(weights):
    """Today's constants (CODATA, through SciPy), the thermochemical calorie, and 1 atm as the standard
    pressure of the thermodynamic data: the convention of CHEMKIN, whose data files these are."""
    return Conventions("standard", R=const.R, R_cal=const.R / const.calorie, p_std=const.atm, weights=dict(weights))


def openfoam_conventions():
    """The constants OpenFOAM uses, read from its own files (not typed here): the gas constant
    (cfdlib.foam.gas_constant), the calorie conversion in its CHEMKIN reader, the standard pressure in its
    etc/controlDict and its table of atomic weights."""
    from cfdlib import foam
    src = Path(foam.foam_env("FOAM_SRC"))
    lexer = (src / "thermophysicalModels/reactionThermo/chemistryReaders/chemkinReader/chemkinLexer.L").read_text()
    r_cal = float(re.search(r"RRcal\s*=\s*([0-9.eE+-]+)\s*;", lexer).group(1))
    r_joule = float(re.search(r"RRjoule\s*=\s*([0-9.eE+-]+)\s*;", lexer).group(1))
    etc = (Path(foam.foam_env("WM_PROJECT_DIR")) / "etc" / "controlDict").read_text()
    si = etc[etc.index("SICoeffs"):]
    p_std = float(re.search(r"\bPstd\s+Pstd\s+\[[^\]]*\]\s*([0-9.eE+-]+)\s*;", si).group(1))
    table = (src / "thermophysicalModels/specie/atomicWeights/atomicWeights.C").read_text()
    weights = {m.group(1).upper(): float(m.group(2)) for m in re.finditer(r'\{"(\w+)",\s*([0-9.]+)\}', table)}
    c = Conventions("OpenFOAM", R=foam.gas_constant() / 1000.0, R_cal=r_cal, p_std=p_std, weights=weights)
    c.reader_R_joule = r_joule       # the reader's other gas constant, J/(mol K): shows that its two constants disagree
    return c


def scheme_files(folder=None):
    """(chem.inp, therm.dat) of a scheme folder; default: the hydrogen scheme that ships with OpenFOAM
    ($FOAM_TUTORIALS/combustion/chemFoam/h2/chemkin: 10 species, 27 reaction lines)."""
    if folder is None:
        from cfdlib import foam
        folder = Path(foam.foam_env("FOAM_TUTORIALS")) / "combustion/chemFoam/h2/chemkin"
    return Path(folder) / "chem.inp", Path(folder) / "therm.dat"


# ---- reading the files --------------------------------------------------------------------------------------
def read_thermo(path):
    """therm.dat (NASA polynomials with 7 coefficients for two temperature ranges, fixed columns):
    {species: dict(elements {symbol: count}, T_low, T_high, T_mid, hi[7], lo[7])}."""
    lines = Path(path).read_text().splitlines()
    out, i = {}, 0
    while i < len(lines):
        ln = lines[i]
        if len(ln) >= 80 and ln[79] == "1" and i + 3 < len(lines) and lines[i + 1][79:80] == "2":
            name = ln[:18].split()[0]
            el = {}
            for k in range(4):
                sym, num = ln[24 + 5 * k:26 + 5 * k].strip(), ln[26 + 5 * k:29 + 5 * k].strip()
                if sym and num and float(num) != 0:
                    el[sym.upper()] = int(float(num))
            t_low, t_high, t_mid = float(ln[45:55]), float(ln[55:65]), float(ln[65:73])
            nums = []
            for row in lines[i + 1:i + 4]:
                nums += [float(row[15 * k:15 * k + 15]) for k in range(5) if row[15 * k:15 * k + 15].strip()]
            assert len(nums) == 14, f"{name}: {len(nums)} coefficients"
            out[name] = dict(elements=el, T_low=t_low, T_high=t_high, T_mid=t_mid, hi=nums[:7], lo=nums[7:])
            i += 4
        else:
            i += 1
    return out


def _side(text, species):
    """One side of a reaction equation: ({species: count}, third body 'M' present, fall-off collider or None)."""
    collider = None
    m = re.search(r"\(\+(\w+)\)", text)
    if m:
        collider = m.group(1)
        text = text.replace(m.group(0), "")
    nu, third = {}, False
    for term in text.split("+"):
        if term == "M":
            third = True
            continue
        n = 1
        if term not in species:
            mm = re.fullmatch(r"(\d+)(\w+)", term)
            if not mm or mm.group(2) not in species:
                raise ValueError(f"unknown species '{term}'")
            n, term = int(mm.group(1)), mm.group(2)
        nu[term] = nu.get(term, 0) + n
    return nu, third, collider


def read_scheme(path):
    """chem.inp: (elements, species, reactions). Each reaction: dict(eq, lhs, rhs, A, beta, E, third, collider,
    low (A, beta, E) or None, troe [..] or None, eff {species: efficiency}, dup). Units as in the file
    (cm, mol, s, cal/mol)."""
    text = "\n".join(l.split("!")[0].rstrip() for l in Path(path).read_text().splitlines())
    def block(key):
        m = re.search(rf"^\s*{key}\w*\s*(.*?)^\s*END", text, re.S | re.M | re.I)
        return m.group(1)
    elements, species = block("ELEM").split(), block("SPEC").split()
    reactions = []
    num = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"
    for ln in block("REAC").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if "=" in ln:
            m = re.fullmatch(rf"(.*?)\s+({num})\s+({num})\s+({num})", ln)
            eq = m.group(1).replace(" ", "")
            if "=>" in eq or "<=>" in eq:
                raise ValueError(f"reaction arrow not handled: {eq}")
            left, right = eq.split("=")
            lhs, t1, c1 = _side(left, species)
            rhs, t2, c2 = _side(right, species)
            assert t1 == t2 and c1 == c2, eq
            reactions.append(dict(eq=eq, lhs=lhs, rhs=rhs, A=float(m.group(2)), beta=float(m.group(3)), E=float(m.group(4)),
                                  third=t1, collider=c1, low=None, troe=None, eff={}, dup=False))
            continue
        r = reactions[-1]
        up = ln.upper()
        if up.startswith("DUP"):
            r["dup"] = True
        elif up.startswith("LOW"):
            r["low"] = tuple(float(x) for x in re.findall(num, ln[3:]))
        elif up.startswith("TROE"):
            r["troe"] = [float(x) for x in re.findall(num, ln[4:])]
        else:
            pairs = re.findall(rf"(\w+)\s*/\s*({num})\s*/", ln)
            if not pairs:
                raise ValueError(f"line not understood: {ln}")
            for s, v in pairs:
                assert s in species, ln
                r["eff"][s] = float(v)
    return elements, species, reactions


class Mechanism:
    """The scheme as arrays, ready to give reaction rates, for one set of conventions."""

    def __init__(self, chem_path, therm_path, conv):
        self.conv = conv
        self.elements_listed, self.species, self.reactions = read_scheme(chem_path)
        th = read_thermo(therm_path)
        sp = self.species
        N, R = len(sp), len(self.reactions)
        self.thermo = {s: th[s] for s in sp}
        self.a_hi = np.array([th[s]["hi"] for s in sp])
        self.a_lo = np.array([th[s]["lo"] for s in sp])
        self.T_mid = np.array([th[s]["T_mid"] for s in sp])
        self.elements = sorted({e for s in sp for e in th[s]["elements"]})
        self.comp = np.array([[th[s]["elements"].get(e, 0) for e in self.elements] for s in sp], dtype=float)
        self.W = np.array([sum(n * conv.weights[e] for e, n in th[s]["elements"].items()) for s in sp]) * 1e-3   # kg/mol
        idx = {s: i for i, s in enumerate(sp)}
        self.idx = idx
        self.nu_f, self.nu_r = np.zeros((R, N)), np.zeros((R, N))
        self.lnA, self.beta, self.Ta = np.zeros(R), np.zeros(R), np.zeros(R)
        self.third, self.fall = np.zeros(R, bool), np.zeros(R, bool)
        self.eff = np.zeros((R, N))
        self.lnA0, self.beta0, self.Ta0 = np.zeros(R), np.zeros(R), np.zeros(R)
        self.troe = np.full((R, 4), np.nan)
        for j, r in enumerate(self.reactions):
            for s, n in r["lhs"].items():
                self.nu_f[j, idx[s]] = n
            for s, n in r["rhs"].items():
                self.nu_r[j, idx[s]] = n
            order = sum(r["lhs"].values())
            self.beta[j], self.Ta[j] = r["beta"], r["E"] / conv.R_cal
            self.third[j] = r["third"]
            if r["collider"]:
                self.fall[j] = True
                assert r["low"] is not None, r["eq"]
                self.lnA[j] = math.log(r["A"] * CM3 ** (order - 1))
                a0, b0, e0 = r["low"]
                self.lnA0[j], self.beta0[j], self.Ta0[j] = math.log(a0 * CM3 ** order), b0, e0 / conv.R_cal
                if r["troe"]:
                    self.troe[j, :len(r["troe"])] = r["troe"]
            else:
                self.lnA[j] = math.log(r["A"] * CM3 ** (order - 1 + (1 if r["third"] else 0)))
            if r["collider"] and r["collider"] != "M":
                self.eff[j, idx[r["collider"]]] = 1.0            # one named collision partner
            elif r["collider"] or r["third"]:
                self.eff[j] = 1.0
                for s, v in r["eff"].items():
                    self.eff[j, idx[s]] = v
        self.dnu = self.nu_r - self.nu_f
        self.is_troe = np.isfinite(self.troe[:, 0])

    # ---- thermodynamics of each species, per mole ----
    def _a(self, T):
        return np.where((T >= self.T_mid)[:, None], self.a_hi, self.a_lo)

    def cp(self, T):
        a = self._a(T)
        return self.conv.R * (a[:, 0] + T * (a[:, 1] + T * (a[:, 2] + T * (a[:, 3] + T * a[:, 4]))))

    def h(self, T):
        a = self._a(T)
        return self.conv.R * T * (a[:, 0] + T * (a[:, 1] / 2 + T * (a[:, 2] / 3 + T * (a[:, 3] / 4 + T * a[:, 4] / 5))) + a[:, 5] / T)

    def s_over_R(self, T):
        a = self._a(T)
        return a[:, 0] * math.log(T) + T * (a[:, 1] + T * (a[:, 2] / 2 + T * (a[:, 3] / 3 + T * a[:, 4] / 4))) + a[:, 6]

    def g_over_RT(self, T):
        """Gibbs energy at the standard pressure, divided by R T."""
        return self.h(T) / (self.conv.R * T) - self.s_over_R(T)

    # ---- reaction rates ----
    def rate_constants(self, c, T):
        """(kf, kr, M) per reaction, SI units; M is the collision-partner concentration (1 where none)."""
        lnT = math.log(T)
        kf = np.exp(self.lnA + self.beta * lnT - self.Ta / T)
        M = self.eff @ c
        if self.fall.any():
            f = self.fall
            k0 = np.exp(self.lnA0[f] + self.beta0[f] * lnT - self.Ta0[f] / T)
            Pr = np.maximum(k0 * M[f] / kf[f], 1e-300)
            F = np.ones(Pr.size)
            tr = self.is_troe[f]
            if tr.any():
                a, T3, T1, T2 = (self.troe[f][tr][:, k] for k in range(4))
                Fc = (1 - a) * np.exp(-T / T3) + a * np.exp(-T / T1) + np.where(np.isfinite(T2), np.exp(-np.nan_to_num(T2, nan=0.0) / T), 0.0)
                lF, lP = np.log10(Fc), np.log10(Pr[tr])
                cc, nn = -0.4 - 0.67 * lF, 0.75 - 1.27 * lF
                F[tr] = 10.0 ** (lF / (1 + ((lP + cc) / (nn - 0.14 * (lP + cc))) ** 2))
            kf = kf.copy()
            kf[f] = kf[f] * Pr / (1 + Pr) * F
        Kc = np.exp(-self.dnu @ self.g_over_RT(T)) * (self.conv.p_std / (self.conv.R * T)) ** self.dnu.sum(axis=1)
        return kf, kf / Kc, np.where(self.third, M, 1.0)

    def progress(self, c, T):
        """Forward and reverse rate of each reaction, mol/(m^3 s)."""
        kf, kr, M = self.rate_constants(c, T)
        return kf * M * np.prod(c[None, :] ** self.nu_f, axis=1), kr * M * np.prod(c[None, :] ** self.nu_r, axis=1)

    def omega(self, c, T):
        """Net production rate of each species, mol/(m^3 s)."""
        qf, qr = self.progress(c, T)
        return self.dnu.T @ (qf - qr)


class Parcel:
    """A closed, well-mixed parcel at constant pressure without heat exchange: the given problem."""

    def __init__(self, mech, p, T0, moles):
        """moles: {species: relative number of moles} of the start mixture."""
        self.mech, self.p, self.T0 = mech, p, T0
        x = np.array([moles.get(s, 0.0) for s in mech.species], dtype=float)
        self.X0 = x / x.sum()
        self.W0 = float(self.X0 @ mech.W)                        # mean molar mass, kg/mol
        self.sigma0 = self.X0 / self.W0                          # mol/kg
        self.Y0 = self.sigma0 * mech.W
        self.rho0 = p * self.W0 / (mech.conv.R * T0)
        self.h0 = float(self.sigma0 @ mech.h(T0))                # J/kg, constant in time
        self.atoms0 = mech.comp.T @ self.sigma0                  # mol of atoms per kg, per element

    def volume(self, sigma, T):
        return sigma.sum() * self.mech.conv.R * T / self.p      # m^3/kg

    def rhs(self, t, y):
        sigma, T = y[:-1], y[-1]
        v = self.volume(sigma, T)
        w = self.mech.omega(sigma / v, T)
        return np.append(v * w, -v * (self.mech.h(T) @ w) / (sigma @ self.mech.cp(T)))

    def enthalpy(self, sigma, T):
        return float(sigma @ self.mech.h(T))

    def solve(self, t_end, T_ign=None, rtol=1e-10, atol=1e-20, method="Radau", t_eval=None):
        """Integrate to t_end. Returns dict(t, T, sigma (rows = times), t_ign, steps, sol). With t_eval the
        values at those times come from the integrator's own interpolation. At the default tolerances the
        ignition time of case 20 is good to about 3e-7 of its value, not to 1e-10: gpucfd's kernel and
        OpenFOAM taken to a step of zero agree with each other to 1e-7 and differ from this by 3e-7."""
        events = None
        if T_ign is not None:
            def ev(t, y):
                return y[-1] - T_ign
            ev.direction = 1
            events = ev
        y0 = np.append(self.sigma0, self.T0)
        sol = solve_ivp(self.rhs, (0.0, t_end), y0, method=method, rtol=rtol, atol=atol, events=events,
                        dense_output=True, t_eval=t_eval)
        if not sol.success:
            raise RuntimeError(sol.message)
        t_ign = float(sol.t_events[0][0]) if events is not None and len(sol.t_events[0]) else None
        return dict(t=sol.t, T=sol.y[-1], sigma=sol.y[:-1].T, t_ign=t_ign, steps=int(sol.nfev), sol=sol.sol)

    # ---- where the mixture is heading: bounds on the final temperature ----
    def complete_combustion(self):
        """All hydrogen burnt to water, nothing else formed: (temperature, moles per kg). An upper bound."""
        m = self.mech
        s = self.sigma0.copy()
        i = m.idx
        burnt = min(s[i["H2"]], 2 * s[i["O2"]])
        s[i["H2"]] -= burnt
        s[i["O2"]] -= burnt / 2
        s[i["H2O"]] += burnt
        T = brentq(lambda T: s @ m.h(T) - self.h0, self.T0, 5000.0, xtol=1e-10)
        return float(T), s

    def equilibrium(self, guess_sigma, guess_T):
        """Chemical equilibrium at the parcel's enthalpy and pressure, by minimising the Gibbs energy with the
        elements conserved (element potentials). Unknowns: one potential per element, the total moles, the
        temperature. Independent of the reaction rates. Returns (T, sigma)."""
        m = self.mech
        act = (m.comp.sum(axis=1) > 0) & np.array([all(self.atoms0[k] > 0 for k in np.nonzero(m.comp[i])[0]) for i in range(len(m.species))])
        E = m.comp[act][:, self.atoms0 > 0]
        b = self.atoms0[self.atoms0 > 0]
        lp = math.log(self.p / m.conv.p_std)

        def sig(x):
            lam, ln_n, T = x[:-2], x[-2], x[-1]
            return np.exp(E @ lam - m.g_over_RT(T)[act] - lp + ln_n)

        def res(x):
            s = sig(x)
            return np.concatenate([(E.T @ s - b) / b, [s.sum() / math.exp(x[-2]) - 1.0,
                                                       (s @ m.h(x[-1])[act] - self.h0) / abs(self.h0)]])
        g = guess_sigma[act]
        mu = m.g_over_RT(guess_T)[act] + np.log(np.maximum(g, 1e-30) / g.sum()) + lp
        lam0 = np.linalg.lstsq(E, mu, rcond=None)[0]
        x = root(res, np.concatenate([lam0, [math.log(g.sum()), guess_T]]), method="hybr", tol=1e-13)
        if not x.success or np.max(np.abs(res(x.x))) > 1e-9:
            raise RuntimeError(f"equilibrium not found: {x.message}")
        out = np.zeros(len(m.species))
        out[act] = sig(x.x)
        return float(x.x[-1]), out


class VolumeParcel(Parcel):
    """A closed, well-mixed parcel at constant volume without heat exchange: what one cell of a flow solver is
    during the chemistry part of a time step (density and internal energy do not change). The start state is
    given as for Parcel (pressure, temperature, moles); the pressure then follows the gas. Added for the
    constant-volume mode of gpucfd's chemistry kernel (gpucfd/CHEMISTRY.md, rung 2)."""

    def __init__(self, mech, p, T0, moles):
        super().__init__(mech, p, T0, moles)
        self.p0, self.v0 = p, 1.0 / self.rho0
        self.u0 = self.h0 - float(self.sigma0.sum()) * mech.conv.R * T0      # internal energy, J/kg, constant in time

    def volume(self, sigma, T):
        return self.v0

    def pressure(self, sigma, T):
        """Pressure of the mixture sigma (mol/kg; one state or rows of states) at temperature T."""
        return np.asarray(sigma).sum(axis=-1) * self.mech.conv.R * T / self.v0

    def energy(self, sigma, T):
        return float(sigma @ self.mech.h(T)) - float(sigma.sum()) * self.mech.conv.R * T

    def rhs(self, t, y):
        sigma, T = y[:-1], y[-1]
        R = self.mech.conv.R
        w = self.mech.omega(sigma / self.v0, T)
        return np.append(self.v0 * w, -self.v0 * ((self.mech.h(T) - R * T) @ w) / (sigma @ (self.mech.cp(T) - R)))

    def complete_combustion(self):
        """All hydrogen burnt to water at constant volume: (temperature, moles per kg). An upper bound."""
        m = self.mech
        s = super().complete_combustion()[1]
        T = brentq(lambda T: s @ (m.h(T) - m.conv.R * T) - self.u0, self.T0, 6000.0, xtol=1e-10)
        return float(T), s

    def equilibrium(self, guess_sigma, guess_T):
        raise NotImplementedError("the equilibrium solver is for a given enthalpy and pressure (Parcel), not for a given energy and volume")


def build(conv=None, folder=None):
    """The mechanism for a set of conventions (default: the standard ones, with OpenFOAM's atomic weights,
    which are those of CHEMKIN's own table) from a scheme folder (default: OpenFOAM's hydrogen scheme)."""
    conv = conv or standard_conventions(openfoam_conventions().weights)
    chem, therm = scheme_files(folder)
    return Mechanism(chem, therm, conv)


# ---- reading a history (temperature and mass fractions at many times) -----------------------------------------
def crossing_time(t, T, level):
    """The time at which T first reaches `level`: the first point with T >= level, and a straight line
    between it and the point before. None if the history never gets there."""
    t, T = np.asarray(t, float), np.asarray(T, float)
    k = int(np.argmax(T >= level))
    if T[k] < level:
        return None
    if k == 0:
        return float(t[0])
    return float(t[k - 1] + (level - T[k - 1]) * (t[k] - t[k - 1]) / (T[k] - T[k - 1]))


def atoms(Y, mech):
    """Moles of atoms of each element per kg of gas at every point: Y (points x species) -> (points x elements,
    in the order of mech.elements). Use a mechanism built with the SOLVER's atomic weights: with other weights
    the atoms seem to change (case 20: 3e-5 with hydrogen at 1.00794 where OpenFOAM has 1.00797)."""
    return (np.asarray(Y, float) / mech.W) @ mech.comp


def enthalpy(T, Y, mech):
    """Enthalpy per kg (chemical energy included) at every point, from temperature and mass fractions."""
    T = np.asarray(T, float)
    sigma = np.asarray(Y, float) / mech.W
    return np.array([sigma[k] @ mech.h(T[k]) for k in range(len(T))])


def parcel_health(T, Y, mech, T_low, T_high, tol_sum=1e-9, tol_atoms=1e-8):
    """What a sound reacting history must do, whatever the solver: the temperature stays between T_low and
    T_high (for a parcel without heat exchange: the start and the temperature of complete burning), no mass
    fraction is negative, they add up to one, and the atoms of each element present are kept.
    Returns dict(ok, checks[{name, ok, value}], atoms_change, sum_off, lowest_Y, T_min, T_max)."""
    T, Y = np.asarray(T, float), np.asarray(Y, float)
    n = atoms(Y, mech)
    present = n[0] > 0
    change = float(np.max(np.abs(n[:, present] / n[0, present] - 1)))
    sum_off, low = float(np.max(np.abs(Y.sum(axis=1) - 1))), float(Y.min())
    checks = [dict(name="temperature inside its bounds", ok=bool(T.min() >= T_low * (1 - 1e-6) and T.max() <= T_high * (1 + 1e-6)),
                   value=f"{T.min():.4f} to {T.max():.4f} K, bounds {T_low:.4f} to {T_high:.4f}"),
              dict(name="no negative mass fraction", ok=bool(low >= -1e-12), value=f"lowest {low:.2e}"),
              dict(name="mass fractions add up to one", ok=bool(sum_off <= tol_sum), value=f"off by at most {sum_off:.1e}"),
              dict(name="atoms of each element kept", ok=bool(change <= tol_atoms), value=f"largest relative change {change:.1e}")]
    return dict(ok=all(c["ok"] for c in checks), checks=checks, atoms_change=change, sum_off=sum_off, lowest_Y=low,
                T_min=float(T.min()), T_max=float(T.max()))


# ---- detonation theory: what the gas alone fixes, with no simulation ------------------------------------------
def sound_speed_frozen(mech, T, sigma):
    """Speed of sound with the composition held fixed: sqrt(gamma R T / W), sigma in mol/kg."""
    cp = float(sigma @ mech.cp(T))                               # J/(kg K)
    r = float(sigma.sum()) * mech.conv.R                         # gas constant of the mixture, J/(kg K)
    return math.sqrt(cp / (cp - r) * r * T)


def shock_frozen(mech, p1, T1, moles, D):
    """The state right behind a shock of speed D that runs into gas at rest, before anything reacts (the
    composition does not change; heat capacities change with temperature). In a detonation this is the
    'von Neumann' state at the front. Returns dict(p, T, rho, u (gas speed, laboratory frame), ratio v2/v1)."""
    fresh = Parcel(mech, p1, T1, moles)
    v1, s = 1.0 / fresh.rho0, fresh.sigma0
    r = float(s.sum()) * mech.conv.R

    def state(x):                                                # x = v2 / v1
        p2 = p1 + D * D * (1 - x) / v1
        T2 = brentq(lambda T: float(s @ mech.h(T)) - fresh.h0 - 0.5 * D * D * (1 - x * x), T1, 6000.0, xtol=1e-10)
        return p2, T2
    x = brentq(lambda x: r * state(x)[1] / state(x)[0] - x * v1, 0.05, 0.95, xtol=1e-13)
    p2, T2 = state(x)
    return dict(p=p2, T=T2, rho=1.0 / (x * v1), u=D * (1 - x), ratio=x)


def cj_detonation(mech, p1, T1, moles, x_range=(0.45, 0.70)):
    """The Chapman-Jouguet detonation of a mixture at rest at (p1, T1): the slowest wave whose burnt gas, in
    chemical equilibrium, satisfies the conservation of mass, momentum and energy across the wave. A detonation
    that runs freely travels at this speed. It follows from the thermodynamic data alone, not from the reaction
    rates.

    Method (the definition, no shortcut): for a ratio x = v2 / v1 of the specific volumes, mass and momentum give
    the pressure p2 = p1 + D^2 (1 - x) / v1 and energy gives the enthalpy h2 = h1 + D^2 (1 - x^2) / 2; the
    equilibrium at (h2, p2) has a volume of its own, and D(x) is the speed for which that volume is x v1. The
    minimum of D(x) is the CJ speed. Returns dict(D, p, T, rho, u (gas speed behind the wave, laboratory frame),
    w (gas speed relative to the wave: the equilibrium speed of sound there), ratio, sigma (mol/kg), mach
    (D over the frozen speed of sound of the fresh gas), fresh dict(rho, h, c))."""
    from scipy.optimize import minimize_scalar
    fresh = Parcel(mech, p1, T1, moles)
    v1, h1 = 1.0 / fresh.rho0, fresh.h0
    c1 = sound_speed_frozen(mech, T1, fresh.sigma0)
    prod = Parcel(mech, p1, T1, moles)                           # carries pressure and enthalpy into the equilibrium solver
    T_guess, s_guess = fresh.complete_combustion()
    s_guess = s_guess + 1e-3 * s_guess.sum()                     # a little of every species: a start with exact zeros does not converge
    guess = list(prod.equilibrium(s_guess, 0.9 * T_guess))       # equilibrium at the fresh gas's enthalpy and pressure

    last = dict(p=p1, h=h1)

    def burnt(x, D):
        """Equilibrium at the pressure and enthalpy that mass, momentum and energy give for (x, D). The solver
        starts from the last state it solved and walks to the new one, with shorter strides where it fails."""
        p2, h2 = p1 + D * D * (1 - x) / v1, h1 + 0.5 * D * D * (1 - x * x)
        pa, ha, done, stride = last["p"], last["h"], 0.0, 1.0
        while done < 1.0:
            g = min(1.0, done + stride)
            prod.p, prod.h0 = pa * (p2 / pa) ** g, ha + g * (h2 - ha)
            try:
                with np.errstate(over="ignore", invalid="ignore"):   # a stride that is too long overflows; it is then halved
                    T, s = prod.equilibrium(guess[1], guess[0])
            except RuntimeError:
                stride /= 2
                if stride < 1e-4:
                    raise
                continue
            guess[0], guess[1], done = T, s, g
        last["p"], last["h"] = p2, h2
        return T, s, prod.volume(s, T)

    def speed(x):
        return brentq(lambda D: burnt(x, D)[2] - x * v1, 3.5 * c1, 6.5 * c1, xtol=1e-9, rtol=1e-13)
    best = minimize_scalar(speed, bounds=x_range, method="bounded", options=dict(xatol=1e-7))
    x, D = float(best.x), float(best.fun)
    if min(x - x_range[0], x_range[1] - x) < 1e-4:
        raise RuntimeError(f"CJ point at the edge of x_range {x_range}: widen it")
    T, s, v = burnt(x, D)
    return dict(D=D, p=float(prod.p), T=float(T), rho=1.0 / v, u=D * (1 - x), w=D * x, ratio=x, sigma=s, mach=D / c1,
                fresh=dict(rho=fresh.rho0, h=h1, c=c1))
