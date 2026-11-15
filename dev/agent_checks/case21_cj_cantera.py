"""A file of the developing agent. An agent that solves the case does not need it and should not open it.

Cross-check of cfdlib.chem.cj_detonation with an independent program: Cantera (thermodynamics and equilibrium
solver of its own), given the same two CHEMKIN files. Cantera is not part of the project's environment; this
script runs in any Python that has it and writes its numbers to case21_cj_cantera.json next to it, which
tests/test_tools.py compares with cfdlib.chem.

    <python with cantera> dev/agent_checks/case21_cj_cantera.py <folder with chem.inp and therm.dat>
    (the folder: $FOAM_TUTORIALS/combustion/chemFoam/h2/chemkin; done on 2026-11-15 with Cantera 3.2.0)

Two checks that do not share code with cfdlib.chem:
  - the CJ speed as the minimum of the wave speed over the volume ratio, with Cantera's equilibrium;
  - at that state, the gas speed relative to the wave against the equilibrium speed of sound, found by a small
    pressure change at constant entropy (the CJ condition itself, not used in the search).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cantera as ct
from scipy.optimize import brentq, minimize_scalar

P1, T1, X1 = 101325.0, 300.0, "H2:2, O2:1, N2:3.76"


def main(folder):
    folder = Path(folder)
    out = Path(tempfile.mkdtemp()) / "h2.yaml"
    subprocess.run([sys.executable, "-m", "cantera.ck2yaml", f"--input={folder / 'chem.inp'}", f"--thermo={folder / 'therm.dat'}",
                    f"--output={out}", "--permissive"], check=True, capture_output=True)
    gas = ct.Solution(str(out))
    gas.TPX = T1, P1, X1
    v1, h1 = 1.0 / gas.density, gas.enthalpy_mass

    def burnt(x, D):
        p2, h2 = P1 + D * D * (1 - x) / v1, h1 + 0.5 * D * D * (1 - x * x)
        gas.TPX = 2500.0, p2, X1
        gas.HP = h2, p2
        gas.equilibrate("HP", rtol=1e-12, max_steps=10000)
        return 1.0 / gas.density

    def speed(x):
        return brentq(lambda D: burnt(x, D) - x * v1, 1400.0, 2700.0, xtol=1e-9, rtol=1e-13)
    best = minimize_scalar(speed, bounds=(0.45, 0.70), method="bounded", options=dict(xatol=1e-7))
    x, D = float(best.x), float(best.fun)
    burnt(x, D)
    p2, T2, rho2, s2 = gas.P, gas.T, gas.density, gas.entropy_mass
    X2 = {s: float(v) for s, v in zip(gas.species_names, gas.X) if v > 1e-7}
    states = []
    for f in (1 - 1e-4, 1 + 1e-4):                               # the equilibrium speed of sound at the CJ state
        gas.SP = s2, p2 * f
        gas.equilibrate("SP", rtol=1e-12, max_steps=10000)
        states.append((gas.P, gas.density))
    a_eq = ((states[1][0] - states[0][0]) / (states[1][1] - states[0][1])) ** 0.5
    rec = dict(program=f"Cantera {ct.__version__}", scheme=str(folder), p1=P1, T1=T1, mixture=X1, D=D, p=p2, T=T2, rho=rho2,
               ratio=x, w=D * x, sound_speed_equilibrium=a_eq, mole_fractions=X2)
    (Path(__file__).with_suffix(".json")).write_text(json.dumps(rec, indent=1))
    print(f"Cantera: D {D:.3f} m/s, p {p2 / 101325:.4f} atm, T {T2:.2f} K, density ratio {1 / x:.4f}; gas speed relative to the wave "
          f"{D * x:.2f} m/s against the equilibrium speed of sound {a_eq:.2f} m/s ({100 * (D * x / a_eq - 1):+.3f} %)")


if __name__ == "__main__":
    main(sys.argv[1])
