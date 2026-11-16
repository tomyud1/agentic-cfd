"""Writes the reference of case 21 (a detonation in a straight tube): the Chapman-Jouguet state of hydrogen and
air from the thermodynamic data of OpenFOAM's hydrogen scheme, computed by cfdlib.chem, with the cross-check by
an independent program (Cantera; dev/agent_checks/case21_cj_cantera.py wrote case21_cj_cantera.json).

    PYTHONPATH=. python setup/cj_h2_air_reference.py        (inside WSL; reads OpenFOAM's files; a second)
"""
import json
from pathlib import Path

import yaml

from cfdlib import chem

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cases" / "detonation_tube_h2_air" / "reference" / "cj.yaml"
P1, T1, MIX = 101325.0, 300.0, dict(H2=2.0, O2=1.0, N2=3.76)


def state(mech):
    cj = chem.cj_detonation(mech, P1, T1, MIX)
    vn = chem.shock_frozen(mech, P1, T1, MIX, cj["D"])
    X = cj["sigma"] / cj["sigma"].sum()
    return dict(speed_m_s=round(cj["D"], 3), pressure_Pa=round(cj["p"], 1), temperature_K=round(cj["T"], 2), density_kg_m3=round(cj["rho"], 5),
                gas_speed_m_s=round(cj["u"], 3), gas_speed_relative_to_wave_m_s=round(cj["w"], 3),
                sound_speed_m_s=dict(equilibrium=round(cj["w"], 3), fixed_composition=round(cj["c_frozen"], 3),
                                     note="a simulated wave ends where the gas leaves the front at its own speed of sound; the two "
                                          "values put that point at slightly different places (case 21)"),
                mole_fractions={s: round(float(x), 6) for s, x in zip(mech.species, X) if x > 1e-6},
                fresh=dict(density_kg_m3=round(cj["fresh"]["rho"], 6), sound_speed_m_s=round(cj["fresh"]["c"], 3)),
                front_before_burning=dict(pressure_Pa=round(vn["p"], 1), temperature_K=round(vn["T"], 2), density_kg_m3=round(vn["rho"], 5),
                                          gas_speed_m_s=round(vn["u"], 3)))


def main():
    std, of = state(chem.build()), state(chem.build(chem.openfoam_conventions()))
    ct = json.loads((ROOT / "dev/agent_checks/case21_cj_cantera.json").read_text())
    out = dict(
        source="Chapman-Jouguet theory: the slowest wave whose burnt gas, in chemical equilibrium, satisfies the conservation of mass, "
               "momentum and energy across the wave. Computed by cfdlib.chem.cj_detonation from the thermodynamic data of OpenFOAM's "
               "hydrogen scheme ($FOAM_TUTORIALS/combustion/chemFoam/h2/chemkin/therm.dat). It does not depend on the reaction rates.",
        conditions=dict(pressure_Pa=P1, temperature_K=T1, mole_ratios=MIX),
        cj=dict(note="with the constants OpenFOAM uses when it reads the data (standard pressure 1 bar): the state an OpenFOAM run should reach", **of),
        cj_standard_constants=dict(note="with the constants the data were written for (standard pressure 1 atm)", **std),
        front_note="front_before_burning: the state right behind a shock of the CJ speed with the composition unchanged. A fully resolved "
                   "detonation shows this pressure at its front; a coarse grid shows less.",
        uncertainty=dict(
            independent_program=dict(program=ct["program"], speed_m_s=round(ct["D"], 3), pressure_Pa=round(ct["p"], 1), temperature_K=round(ct["T"], 2),
                                     relative_difference_in_speed=float(f"{abs(std['speed_m_s'] / ct['D'] - 1):.1e}"),
                                     note="same data files, standard constants, its own thermodynamics and equilibrium solver; at that state the gas "
                                          "leaves the wave at the equilibrium speed of sound to 0.001 %"),
            constants=dict(speed_difference_m_s=round(of["speed_m_s"] - std["speed_m_s"], 3), note="OpenFOAM's constants against the standard ones"),
            not_covered="The scheme has no nitrogen oxides and its data tables are old; values in the literature for this mixture may differ by "
                        "about half a per cent. No published value was transcribed here."))
    out = json.loads(json.dumps(out, default=float))              # plain Python numbers for the YAML writer
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("# Written by setup/cj_h2_air_reference.py; do not edit by hand.\n" + yaml.safe_dump(out, sort_keys=False, width=120), encoding="utf-8")
    print(f"reference: {OUT} | CJ speed {of['speed_m_s']} m/s (standard constants {std['speed_m_s']}, {ct['program']} {ct['D']:.3f})")


if __name__ == "__main__":
    main()
