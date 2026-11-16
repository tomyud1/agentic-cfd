"""A file of the developing agent: the review of case 21 (a detonation in a tube). It recomputes the solver
agent's main numbers from the stored profiles of its runs with rules of its own, and from the problem statement
and the reference file, without the case's own measuring code.

    PYTHONPATH=. python dev/agent_checks/case21_review.py      (inside WSL; a few seconds; no solver run)

What is recomputed (finest grid, 3200 cells, unless said):
  - the wave speed over the second half of the time, from the pressure profiles at 25 and 50 microseconds, with
    the front taken where the pressure falls through a level of my own (0.5 MPa, and 1.0 MPa as a second try);
  - the distance from the pressure front to the burning front (where water falls through half its CJ share);
  - the mass in the tube, the atoms of each element and the total energy at the end against the start state
    computed from the problem statement with cfdlib.chem (OpenFOAM's constants);
  - the highest pressure at the front against the pressure behind a shock of the measured speed;
  - the wave speed of the other grids and of the run with one pass per time step (the agent's finding that
    OpenFOAM's reacting-example setting runs 17 % slow).
Agreement asked for: speeds within 0.05 % of the agent's (the two front rules differ by a constant shift at
most); the two requirements of the problem met by my numbers too; mass and atoms within 1e-6; my energy
change at the end within 0.05 % of the heat released of the agent's.
"""
import json
from pathlib import Path

import numpy as np
import yaml

from cfdlib import chem

ROOT = Path(__file__).resolve().parents[2]
CASE, RES = ROOT / "cases" / "detonation_tube_h2_air", ROOT / "results" / "detonation_tube_h2_air"
SPECIES = ["H2", "H", "O2", "O", "OH", "HO2", "H2O2", "H2O", "N2"]


def crossing(x, f, level):
    """Furthest point down the tube where f falls through the level (straight line between cell centres)."""
    k = np.nonzero(f >= level)[0][-1]
    return float(x[k] + (level - f[k]) * (x[k + 1] - x[k]) / (f[k + 1] - f[k]))


def speed(z, level, t_a=25e-6, t_b=50e-6):
    t = z["prof_t"]
    a, b = int(np.argmin(abs(t - t_a))), int(np.argmin(abs(t - t_b)))
    assert abs(t[a] - t_a) < 1e-9 and abs(t[b] - t_b) < 1e-9
    return (crossing(z["x"], z["prof_p"][b], level) - crossing(z["x"], z["prof_p"][a], level)) / (t[b] - t[a])


def main():
    g = {k: float(v["value"]) for k, v in yaml.safe_load((CASE / "problem.yaml").read_text(encoding="utf-8"))["given"].items()}
    ref = yaml.safe_load((CASE / "reference" / "cj.yaml").read_text(encoding="utf-8"))["cj"]
    D_cj = ref["speed_m_s"]
    mech = chem.build(chem.openfoam_conventions())
    moles = dict(H2=g["x_H2"], O2=g["x_O2"], N2=g["x_N2"])
    z = np.load(RES / "runs" / "N3200.npz")
    rec = json.loads((RES / "runs" / "N3200.json").read_text(encoding="utf-8"))
    x, dx = z["x"], float(z["x"][1] - z["x"][0])
    ok = True

    def line(tag, good, text):
        nonlocal ok
        ok &= bool(good)
        print(f"{'PASS' if good else 'FAIL'}  {tag}: {text}")

    # ---- wave speed ----
    D1, D2 = speed(z, 0.5e6), speed(z, 1.0e6)
    line("wave speed, 3200 cells", abs(D1 / rec["speed"]["D"] - 1) < 5e-4 and abs(D1 / D_cj - 1) <= 0.02,
         f"mine {D1:.1f} m/s (level 0.5 MPa; {D2:.1f} with 1.0 MPa), the agent's {rec['speed']['D']:.1f}; from the CJ speed {D_cj:.2f}: {100 * (D1 / D_cj - 1):+.2f} % (required: 2 %)")
    # ---- fronts together ----
    Y_half = 0.5 * ref["mole_fractions"]["H2O"] * mech.W[mech.idx["H2O"]] / sum(ref["mole_fractions"][s] * mech.W[mech.idx[s]] for s in ref["mole_fractions"])
    gaps = [crossing(x, p, 0.5e6) - crossing(x, w, Y_half) for t, p, w in zip(z["prof_t"], z["prof_p"], z["prof_H2O"]) if t >= 10e-6 - 1e-9]
    line("burning front behind the pressure front, from 10 us", max(np.abs(gaps)) <= 1e-3,
         f"mine at most {1e3 * max(np.abs(gaps)):.3f} mm over {len(gaps)} profiles (my levels); the agent's {1e3 * rec['gap_max']:.3f} mm (required: 1 mm)")
    # ---- start state from the problem alone; mass, atoms, energy at the end ----
    fresh = chem.Parcel(mech, g["p_0"], g["T_0"], moles)
    hot = chem.Parcel(mech, g["p_ign"], g["T_ign"], moles)
    m0 = hot.rho0 * g["L_ign"] + fresh.rho0 * (g["L"] - g["L_ign"])
    E0 = hot.rho0 * g["L_ign"] * (hot.h0 - g["p_ign"] / hot.rho0) + fresh.rho0 * (g["L"] - g["L_ign"]) * (fresh.h0 - g["p_0"] / fresh.rho0)
    rho, T, p, U = z["prof_rho"][-1], z["prof_T"][-1], z["prof_p"][-1], z["prof_U"][-1]
    Y = np.zeros((len(x), len(mech.species)))
    for s in SPECIES:
        Y[:, mech.idx[s]] = z["end_" + s]
    m_end = float(rho.sum() * dx)
    masses = [float(r.sum() * dx) / m0 - 1 for r in z["prof_rho"]]
    line("mass in the tube", max(np.abs(masses)) < 1e-6, f"largest change over the 10 profiles {max(np.abs(masses)):.1e} against the start mass computed from the problem ({m0:.6f} kg/m2)")
    at_end = (rho[:, None] * chem.atoms(Y, mech)).sum(0) * dx
    at0 = m0 * fresh.atoms0                                       # the same mixture in the layer and outside it
    have = at0 > 0
    d_at = float(np.max(np.abs(at_end[have] / at0[have] - 1)))
    line("atoms of each element at the end", d_at < 1e-6, f"largest change {d_at:.1e} ({', '.join(e for e, h in zip(mech.elements, have) if h)}); sum of the 9 stored species - 1: {np.max(np.abs(Y.sum(1) - 1)):.1e}")
    E_end = float((rho * (chem.enthalpy(T, Y, mech) + 0.5 * U**2) - p).sum() * dx)
    q = rec["energy"]["q_end"]
    mine, theirs = (E_end - E0) / q, rec["energy"]["dE_end"] / q
    line("total energy at the end", abs(mine - theirs) < 5e-4, f"mine {100 * mine:+.3f} % of the heat released, the agent's {100 * theirs:+.3f} % (its heat released: {q:.0f} J/m2; start energy "
         f"mine {E0:.2f}, the agent's {rec['energy']['E0']:.2f} J/m2)")
    # ---- peak at the front ----
    late = z["ts"] >= 25e-6
    vn = chem.shock_frozen(mech, g["p_0"], g["T_0"], moles, D1)
    ratio = float(z["pmax"][late].mean() / vn["p"])
    line("highest pressure at the front", abs(ratio - rec["peak"]["ratio"]) < 5e-3, f"mean over the second half {z['pmax'][late].mean() / 1e6:.3f} MPa = {ratio:.3f} of the {vn['p'] / 1e6:.3f} MPa behind a "
         f"shock of my speed (the agent's {rec['peak']['ratio']:.3f}); the agent expected at least 0.80")
    # ---- the other runs ----
    for name in ("N400", "N800", "N1600", "N800_p1", "N800_p3", "N800_seulex"):
        zz = np.load(RES / "runs" / f"{name}.npz")
        rr = json.loads((RES / "runs" / f"{name}.json").read_text(encoding="utf-8"))
        D = speed(zz, 0.5e6)
        line(f"wave speed, {name} ({rr['passes']} passes, {rr['ode']})", abs(D / rr["speed"]["D"] - 1) < 5e-4, f"mine {D:.1f} m/s ({100 * (D / D_cj - 1):+.2f} % from CJ), the agent's {rr['speed']['D']:.1f}; "
             f"solver time {rr['wall_time']:.0f} s, energy at the end {100 * rr['energy']['end']:+.2f} % of the heat released")
    print(f"{'PASS' if ok else 'FAIL'}  review of case 21")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
