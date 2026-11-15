"""A file of the developing agent. An agent that solves the case does not need it and should not open it.

Review of case 20 (ignition of hydrogen and air; dev/LOOP.md, step 4): the headline numbers recomputed from the
raw histories in results/h2_air_ignition_box/runs with code written for this review, not the case's.

    PYTHONPATH=. python dev/agent_checks/case20_review.py        (inside WSL; reads one table of OpenFOAM's source; a second)

What it recomputes: the ignition time (first reach of 1200 K, straight line between stored points) and the
final temperature of the run with the smallest step, against the reference file; the order with which the
ignition time changes with the time step and its value at a step of zero; the number of H, O and N atoms
from the stored mass fractions, once with OpenFOAM's own atomic masses and once with today's standard ones typed
here. The second count is not constant, and that is the lesson: the masses must be the solver's (hydrogen is
1.00797 in OpenFOAM's table and 1.00794 in the standard one; the difference shows as 3e-5 in the atom count).
"""
import json
import re
from pathlib import Path

import numpy as np
import yaml

from cfdlib import foam

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "h2_air_ignition_box" / "runs"
REF = yaml.safe_load((ROOT / "cases/h2_air_ignition_box/reference/senkin_h2.yaml").read_text(encoding="utf-8"))
PROBLEM = yaml.safe_load((ROOT / "cases/h2_air_ignition_box/problem.yaml").read_text(encoding="utf-8"))
MASS = dict(H=1.00794, O=15.9994, N=14.0067, AR=39.948)                       # g/mol, standard values typed here
ATOMS = dict(H2=dict(H=2), H=dict(H=1), O2=dict(O=2), O=dict(O=1), OH=dict(O=1, H=1), HO2=dict(H=1, O=2),
             H2O2=dict(H=2, O=2), H2O=dict(H=2, O=1), AR=dict(AR=1), N2=dict(N=2))


def first_reach(t, T, level):
    i = int(np.argmax(T >= level))
    return t[i - 1] + (level - T[i - 1]) * (t[i] - t[i - 1]) / (T[i] - T[i - 1])


def main():
    level = PROBLEM["given"]["T_ign"]["value"]
    ref_t, ref_T = REF["ignition"]["time_s"], REF["final"]["T_K"]
    fam = []
    for steps in (2500, 5000, 10000, 20000, 40000, 80000):
        z = np.load(RUNS / f"S{steps}.npz")
        j = json.loads((RUNS / f"S{steps}.json").read_text(encoding="utf-8"))
        fam.append(dict(steps=steps, dt=PROBLEM["given"]["t_end"]["value"] / steps, t_mine=first_reach(z["t"], z["T"], level),
                        t_agent=j["t_ign"], T_end=float(z["T"][-1]), T_agent=j["T_end"], z=z, j=j))
    f = fam[-1]
    print(f"smallest step ({f['steps']} steps): ignition {1e6 * f['t_mine']:.4f} us from the stored history, the agent's from every step "
          f"{1e6 * f['t_agent']:.4f} us; reference {1e6 * ref_t:.3f} us -> {100 * (f['t_mine'] / ref_t - 1):+.4f} % (agent {100 * (f['t_agent'] / ref_t - 1):+.4f} %)")
    print(f"final temperature {f['T_end']:.4f} K (agent {f['T_agent']:.4f}); reference {ref_T} K -> {100 * (f['T_end'] / ref_T - 1):+.5f} %")
    a, b, c = (x["t_agent"] for x in fam[-3:])                      # the three smallest steps, ratio 2
    order = np.log2((a - b) / (b - c))
    zero = c + (c - b)                                              # first order: the next difference halves again, and so on
    print(f"ignition time against the step: {', '.join(f'{1e6 * x['t_agent']:.4f}' for x in fam)} us; order from the three smallest "
          f"{order:.3f}; at a step of zero {1e6 * zero:.4f} us -> {100 * (zero / ref_t - 1):+.4f} % from the reference")
    # atoms of each element per kilogram, from the stored mass fractions
    head = [h for h in f["j"]["headers"] if "volAverage(T)" in h][0].split()
    names = [h[len("volAverage("):-1] for h in head if h.startswith("volAverage(")][4:]      # after T, p, rho, h
    Y = f["z"]["Y"]
    assert Y.shape[1] == len(names) and set(names) == set(ATOMS), names
    table = (Path(foam.foam_env("FOAM_SRC")) / "thermophysicalModels/specie/atomicWeights/atomicWeights.C").read_text()
    solver = {m.group(1).upper(): float(m.group(2)) for m in re.finditer(r'\{"(\w+)",\s*([0-9.]+)\}', table)}

    def atoms_change(mass):
        W = {s: sum(mass[e] * n for e, n in ATOMS[s].items()) for s in names}
        counts = [sum(Y[:, k] / W[s] * ATOMS[s].get(e, 0) for k, s in enumerate(names)) for e in ("H", "O", "N")]
        return max(float(np.max(np.abs(n / n[0] - 1))) for n in counts)
    worst, typed = atoms_change(solver), atoms_change(MASS)
    print(f"atoms of H, O, N over {len(Y)} stored points: largest relative change {worst:.1e} with OpenFOAM's atomic masses "
          f"(H {solver['H']}; required: 1e-5), {typed:.1e} with the standard masses typed here (H {MASS['H']}); "
          f"mass fractions add up to one within {float(np.max(np.abs(Y.sum(axis=1) - 1))):.1e}")
    ok = (abs(f["t_mine"] / ref_t - 1) < 0.01 and abs(f["T_end"] / ref_T - 1) < 0.001 and worst < 1e-5
          and abs(f["t_mine"] / f["t_agent"] - 1) < 1e-5 and 0.8 < order < 1.2)
    print("REVIEW", "agrees with the agent's numbers and the three requirements are met" if ok else "DISAGREES: look")


if __name__ == "__main__":
    main()
