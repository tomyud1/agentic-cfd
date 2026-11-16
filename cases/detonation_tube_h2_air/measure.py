"""Reading rules: how each output is read from what the solver wrote. Fixed before the production runs.

What the solver writes (system/controlDict of the template):
  postProcessing/line/<time>/*.xy      the state in every cell along the tube, at `samples` equally spaced times
  postProcessing/peaks/0/*.dat         after every step: the highest pressure, temperature and heat release, and where
  postProcessing/volume, mass/0/*.dat  after every step: integrals over the tube (mass, pressure, and the mass-weighted
                                       sums of sensible enthalpy, kinetic energy and every species)
"""
import re
from pathlib import Path

import numpy as np

from cfdlib import chem

SPECIES_LINE = ("H2", "H", "O2", "O", "OH", "HO2", "H2O2", "H2O", "N2")


# ---- reading the files --------------------------------------------------------------------------------------------
def read_table(path):
    """A volFieldValue file: (time, {field: array}), one row per step."""
    path = Path(path)
    head = [l for l in path.read_text().splitlines()[:10] if l.startswith("# Time")][0]
    cols = re.findall(r"\((\w+)\)", head)
    a = np.loadtxt(path, comments="#", ndmin=2)
    assert a.shape[1] == len(cols) + 1, f"{path}: {a.shape[1]} columns for {len(cols)} names"
    return a[:, 0], {c: a[:, i + 1] for i, c in enumerate(cols)}, head


def read_peaks(path):
    """The fieldMinMax file: {field: dict(t, min, x_min, max, x_max)}; x is the position along the tube. A parallel
    run adds a processor column after each position."""
    out, head = {}, ""
    vec = re.compile(r"\(([^)]*)\)")
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            head = line if "Time" in line else head
            continue
        parts = [s.strip() for s in line.split("\t") if s.strip()]
        locs = [float(vec.match(s).group(1).split()[0]) for s in parts if s.startswith("(")]
        nums = [float(s) for s in parts[2:] if not s.startswith("(")]
        d = out.setdefault(parts[1], dict(t=[], min=[], x_min=[], max=[], x_max=[]))
        d["t"].append(float(parts[0])), d["min"].append(nums[0]), d["x_min"].append(locs[0])
        d["max"].append(nums[-2] if len(nums) == 4 else nums[1]), d["x_max"].append(locs[1])
    return {f: {k: np.array(v) for k, v in d.items()} for f, d in out.items()}, head


def read_line(case):
    """The sampled states along the tube: dict(t (nt,), x (nx,), <field> (nt, nx)); U is the x component."""
    dirs = sorted((d for d in (Path(case) / "postProcessing" / "line").iterdir() if d.is_dir()), key=lambda d: float(d.name))
    out, x = {}, None
    for d in dirs:
        f = next(d.glob("*.xy"))
        names = f.stem.split("_")[1:]
        a = np.loadtxt(f, ndmin=2)
        assert a.shape[1] == 1 + sum(3 if n == "U" else 1 for n in names), f"{f}: unexpected columns"
        if x is None:
            x = a[:, 0]
        assert len(a) == len(x), f"{f}: {len(a)} points, expected {len(x)}"
        k = 1
        for n in names:
            out.setdefault(n, []).append(a[:, k])
            k += 3 if n == "U" else 1
    res = {n: np.array(v) for n, v in out.items()}
    res.update(t=np.array([float(d.name) for d in dirs]), x=x, header=str(next(dirs[-1].glob("*.xy")).name))
    return res


# ---- fronts -------------------------------------------------------------------------------------------------------
def foremost_crossing(x, f, level):
    """The position furthest down the tube where f falls through `level`, on a straight line between two cell
    centres. NaN if f never reaches the level or still holds it in the last cell."""
    k = np.nonzero(f >= level)[0]
    if not len(k) or k[-1] == len(x) - 1:
        return float("nan")
    k = k[-1]
    return float(x[k] + (level - f[k]) * (x[k + 1] - x[k]) / (f[k + 1] - f[k]))


def fronts(line, p_mid, w_mid):
    """Pressure front and burning front at every sampled time, and the distance between them."""
    xp = np.array([foremost_crossing(line["x"], p, p_mid) for p in line["p"]])
    xb = np.array([foremost_crossing(line["x"], w, w_mid) for w in line["H2O"]])
    return xp, xb, xp - xb


def speed(t, xp, t_a, t_b):
    """The problem's definition: distance run by the pressure front between two times, over the time. Also the
    slope of a straight line through every sample between them (a cross-check), and the largest and smallest
    speed over one sampling interval (how steadily the front moves)."""
    ka, kb = int(np.argmin(np.abs(t - t_a))), int(np.argmin(np.abs(t - t_b)))
    assert abs(t[ka] - t_a) < 1e-12 and abs(t[kb] - t_b) < 1e-12, "no sample at the two times of the speed"
    D = (xp[kb] - xp[ka]) / (t[kb] - t[ka])
    sl = slice(ka, kb + 1)
    fit = float(np.polyfit(t[sl], xp[sl], 1)[0])
    local = np.diff(xp[sl]) / np.diff(t[sl])
    return dict(D=float(D), D_fit=fit, D_min=float(local.min()), D_max=float(local.max()), x_a=float(xp[ka]), x_b=float(xp[kb]))


# ---- the burnt gas behind the wave --------------------------------------------------------------------------------
def sound_speed(mech, line, k):
    """Speed of sound (composition held fixed) in every cell of sample k, from temperature and composition."""
    Y = np.array([line[s][k] if s in line else np.zeros_like(line["x"]) for s in mech.species]).T
    sigma = Y / mech.W
    return np.array([chem.sound_speed_frozen(mech, T, s) for T, s in zip(line["T"][k], sigma)])


def burnt_state(mech, line, k, x_front, D, distances):
    """The state where the burnt gas moves away from the front at its own speed of sound: u + c = D, the first such
    point behind the front. Nothing behind that point can reach the front, so it is the end of the wave proper;
    in the ideal picture it is the CJ state. Also the state at fixed distances behind the front."""
    x, u = line["x"], line["U"][k]
    c = sound_speed(mech, line, k)
    f = u + c - D
    i_front = int(np.searchsorted(x, x_front))
    idx = np.arange(i_front)
    # Inside the front itself the gas is still cold and slow (u + c < D). Behind it comes the stretch in which the hot
    # gas is slower than its sound relative to the front (u + c > D): the search starts at the front end of that stretch.
    above = idx[f[:i_front] > 0]
    out = dict(found=False, subsonic_cells=int(len(above)))
    below = idx[:above[-1]][f[:above[-1]] <= 0] if len(above) else []
    if len(below):
        j = below[-1]                                            # closest point behind that stretch with u + c <= D
        out["subsonic_cells"] = int(np.sum(f[j + 1:i_front] > 0))
        if j + 1 < len(x) and f[j + 1] > 0:
            a = f[j] / (f[j] - f[j + 1])
            val = lambda q: float(q[j] + a * (q[j + 1] - q[j]))
            out.update(found=True, x=val(x), d=x_front - val(x), p=val(line["p"][k]), T=val(line["T"][k]), rho=val(line["rho"][k]),
                       u=val(u), c=val(c), H2O=val(line["H2O"][k]))
    out["at"] = [dict(d=d, p=float(np.interp(x_front - d, x, line["p"][k])), T=float(np.interp(x_front - d, x, line["T"][k])),
                      u=float(np.interp(x_front - d, x, u)), H2O=float(np.interp(x_front - d, x, line["H2O"][k]))) for d in distances]
    return out


# ---- what the tube holds ------------------------------------------------------------------------------------------
def totals(mech, hc, t, vol, mass, he):
    """From the integrals after every step: mass, atoms of each element, heat released so far and total energy
    (sensible enthalpy + chemical enthalpy + kinetic energy - pressure, integrated over the tube). hc: chemical
    enthalpy of each species per kg (theory.chemical_enthalpy), with the SOLVER's constants."""
    sp = [s for s in mech.species if s in mass]
    chem_e = sum(hc[mech.idx[s]] * mass[s] for s in sp)
    E = mass[he] + chem_e + mass["K"] - (vol["p"] if he == "h" else 0.0)
    atoms = {e: sum(mech.comp[mech.idx[s], k] * mass[s] / mech.W[mech.idx[s]] for s in sp) for k, e in enumerate(mech.elements)}
    return dict(t=t, mass=vol["rho"], E=E, chem=chem_e, kinetic=mass["K"], atoms={e: a for e, a in atoms.items() if a[0] > 0},
                sumY=sum(mass[s] for s in sp) / vol["rho"])


def thin(n, keep):
    """Indices of about `keep` of n points, first and last included."""
    return np.unique(np.round(np.linspace(0, n - 1, min(n, keep))).astype(int))
