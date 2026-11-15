"""Reference data of case 20 (ignition of hydrogen and air): the CHEMKIN II (SENKIN) calculation that ships with
OpenFOAM's own chemistry validation tutorial, transcribed by code into a YAML file.

    PYTHONPATH=. python setup/parse_senkin_h2.py        (inside WSL, needs OpenFOAM; writes
                                                         cases/h2_air_ignition_box/reference/senkin_h2.yaml)

Source: $FOAM_TUTORIALS/combustion/chemFoam/h2/chemkin/senk.out (the output of SENKIN for the input senk.inp
next to it), with the reaction mechanism chem.inp and the thermodynamic data therm.dat of the same folder.
Nothing is typed by hand: the conditions are read from senk.inp, the temperature history and the ignition time
from senk.out. What the source does not state (its integration tolerances) is recorded as not stated.
"""
import re
from pathlib import Path

import numpy as np
import yaml

from cfdlib import foam

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(foam.foam_env("FOAM_TUTORIALS")) / "combustion" / "chemFoam" / "h2" / "chemkin"
OUT = ROOT / "cases" / "h2_air_ignition_box" / "reference" / "senkin_h2.yaml"

inp = (SRC / "senk.inp").read_text()
out = (SRC / "senk.out").read_text()
key = lambda k: re.search(rf"^{k}\s+(\S+)(?:\s+(\S+))?", inp, re.M)            # noqa: E731
conp = bool(re.search(r"^CONP", inp, re.M))
p_atm, T0, t_end = float(key("PRES").group(1)), float(key("TEMP").group(1)), float(key("TIME").group(1))
reac = {m.group(1): float(m.group(2)) for m in re.finditer(r"^REAC\s+(\S+)\s+(\S+)", inp, re.M)}
rows = np.array([[float(a), float(b)] for a, b in re.findall(r"^ Time \(sec\) =\s*(\S+)\s+T \(K\) =\s*(\S+)", out, re.M)])
t, T = rows[:, 0], rows[:, 1]
ign = float(re.search(r"Ignition Time \(sec\) =\s*(\S+)", out).group(1))
crit = float(re.search(r"Temp criteria \(K\) =\s*(\S+)", out).group(1))
assert conp and t[0] == 0.0 and abs(T[0] - T0) < 1e-9 and abs(t[-1] - t_end) < 1e-12 and np.all(np.diff(t) > 0)
# the ignition time the source prints, against the first crossing of its criterion in its own printed history
i = int(np.argmax(T >= crit))
cross = float(t[i - 1] + (crit - T[i - 1]) * (t[i] - t[i - 1]) / (T[i] - T[i - 1]))
rate = np.gradient(T, t)
species = re.search(r"SPECIES\s+(.*?)\s+END", (SRC / "chem.inp").read_text(), re.S).group(1).split()
n_reactions = len(re.findall(r"^\s*[A-Z0-9()+]+=[A-Z0-9()+]+\s+[0-9.Ee+-]+\s+[0-9.Ee+-]+\s+[0-9.Ee+-]+", (SRC / "chem.inp").read_text(), re.M))

ref = dict(
    source="CHEMKIN II (SENKIN) output shipped with OpenFOAM " + foam.foam_env("WM_PROJECT_VERSION")
           + ": tutorials/combustion/chemFoam/h2/chemkin/senk.out, for the input senk.inp in the same folder",
    mechanism=dict(files="tutorials/combustion/chemFoam/h2/chemkin/chem.inp and therm.dat (in $FOAM_TUTORIALS)", species=species,
                   reaction_lines=n_reactions, note="hydrogen and oxygen with nitrogen and argon as inert gases; rate constants "
                   "attributed in the file to Marinov (1995) and Hippler (1995)"),
    conditions=dict(constant="pressure" if conp else "volume", pressure_atm=p_atm, pressure_Pa=p_atm * 101325.0, T0_K=T0,
                    t_end_s=t_end, mole_ratios=reac),
    ignition=dict(criterion_K=crit, time_s=ign, meaning="the time at which the temperature first reaches the criterion",
                  first_crossing_of_printed_history_s=cross),
    fastest_rise=dict(time_s=float(t[int(np.argmax(rate))]), rate_K_per_s=float(rate.max())),
    final=dict(time_s=float(t[-1]), T_K=float(T[-1])),
    uncertainty=dict(
        printed_digits=5,
        temperature_resolution_K=0.1,
        time_between_printed_points_s=dict(smallest=float(np.diff(t).min()), at_ignition=float(t[i] - t[i - 1]), largest=float(np.diff(t).max())),
        tolerances="not stated in senk.inp (SENKIN's defaults); no second calculation with tighter tolerances is given",
        note="The ignition time printed by the source and the first crossing of its own printed history differ by "
             f"{abs(cross - ign):.2e} s; that is how well the reference fixes its own ignition time"),
    history=dict(t_s=[float(v) for v in t], T_K=[float(v) for v in T]),
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("# Written by setup/parse_senkin_h2.py from OpenFOAM's tutorial files; do not edit by hand.\n"
               + yaml.safe_dump(ref, sort_keys=False, default_flow_style=None, width=110), encoding="utf-8")
print(f"{OUT.relative_to(ROOT)}: {len(t)} points, constant {ref['conditions']['constant']}, {p_atm} atm, {T0} K, mixture {reac}")
print(f"   ignition ({crit:.0f} K) printed {ign:.4e} s, first crossing of the history {cross:.4e} s; fastest rise at {ref['fastest_rise']['time_s']:.4e} s "
      f"({rate.max():.3e} K/s); final {T[-1]:.1f} K; {len(species)} species, {n_reactions} reaction lines")
