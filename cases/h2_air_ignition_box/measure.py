"""The rules that read the outputs from a temperature and species history. Each is fixed in numerics.yaml
before the production runs and applied alike to OpenFOAM's history, to the reference's printed history and to
the independent solution (which measures a rule's own error)."""
import numpy as np


def crossing_time(t, T, level):
    """The time at which T first reaches `level`: the first point with T >= level, and a straight line
    between it and the point before. None if the history never gets there. t[0] is the start."""
    t, T = np.asarray(t, float), np.asarray(T, float)
    k = int(np.argmax(T >= level))
    if T[k] < level:
        return None
    if k == 0:
        return float(t[0])
    return float(t[k - 1] + (level - T[k - 1]) * (t[k] - t[k - 1]) / (T[k] - T[k - 1]))


def crossings(t, T, levels):
    return {str(int(L)): crossing_time(t, T, L) for L in levels}


def fastest_rise(t, T):
    """Largest rate of temperature rise between two consecutive points, and the middle of that interval."""
    t, T = np.asarray(t, float), np.asarray(T, float)
    r = np.diff(T) / np.diff(t)
    k = int(np.argmax(r))
    return float(r[k]), float(0.5 * (t[k] + t[k + 1]))


def atoms(Y, mech):
    """Moles of atoms of each element per kg of gas, at every point: Y (points x species) -> (points x elements)."""
    return (np.asarray(Y, float) / mech.W) @ mech.comp


def enthalpy(T, Y, mech):
    """Enthalpy per kg (chemical energy included) at every point, from temperature and mass fractions, with the
    thermodynamic data read by mech.py (not OpenFOAM's own number)."""
    T = np.asarray(T, float)
    sigma = np.asarray(Y, float) / mech.W
    return np.array([sigma[k] @ mech.h(T[k]) for k in range(len(T))])


def sensible(T, Y, mech, T_std=298.15):
    """Sensible enthalpy per kg as OpenFOAM defines it: the enthalpy minus each species' enthalpy at T_std."""
    sigma = np.asarray(Y, float) / mech.W
    return enthalpy(T, Y, mech) - sigma @ mech.h(T_std)


def thin(n, keep):
    """Indices of about `keep` of n points, the first and the last included."""
    if n <= keep:
        return np.arange(n)
    idx = np.unique(np.round(np.linspace(0, n - 1, keep)).astype(int))
    return idx
