"""A file of the developing agent. An agent that solves the case does not need it and should not open it.

Before case 20 (ignition of hydrogen and air; dev/LOOP.md, step 2): can the tools handle OpenFOAM's chemistry
solver (chemFoam), and how close is its untouched tutorial to the reference?

    PYTHONPATH=. python dev/agent_checks/case20_toolgap.py      (inside WSL, needs OpenFOAM; seconds)

It copies OpenFOAM's hydrogen ignition tutorial to a temporary folder, runs it as it is and calls the tools on
it. The tutorial is NOT a solution of the case. Lines say PASS or GAP (a tool problem) or INFO (how the
untouched tutorial compares with the reference, which sizes the accuracy the problem can ask for).
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import yaml

from cfdlib import contract, foam, foamdict, monitor

ROOT = Path(__file__).resolve().parents[2]
REF = yaml.safe_load((ROOT / "cases/h2_air_ignition_box/reference/senkin_h2.yaml").read_text(encoding="utf-8"))
GAPS = []


def line(kind, name, detail=""):
    print(f"{kind:4s}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if kind == "GAP":
        GAPS.append(name)


def attempt(name, fn):
    try:
        ok, detail = fn()
        line("PASS" if ok else "GAP", name, detail)
    except Exception as e:                                         # noqa: BLE001
        line("GAP", name, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")


tmp = Path(tempfile.mkdtemp(prefix="case20_toolgap_", dir=Path.home()))
case = tmp / "h2"
shutil.copytree(Path(foam.foam_env("FOAM_TUTORIALS")) / "combustion" / "chemFoam" / "h2", case)
wall = foam.run("chemFoam", case, "chemFoam")
print(f"chemFoam ran the tutorial in {wall:.1f} s; files written: {sorted(p.name for p in case.iterdir())}")


def log():
    L = monitor.parse_log(case / "log.chemFoam")
    a = monitor.assess(case / "log.chemFoam", end_time=REF["conditions"]["t_end_s"])
    return len(L["steps"]) > 10 and a["status"] in ("ok", "warn"), f"{len(L['steps'])} steps, status {a['status']}: " + "; ".join(
        f"{c['name']} {c['status']}" for c in a["checks"])


def dictionaries():
    bad = []
    s = foamdict.case_settings(case, unparsed=bad)
    u = contract.uncovered_settings([], case)
    return not bad, f"{len(s)} settings, unparsed {bad}; {len(u)} to document; the reaction mechanism is in CHEMKIN files, outside this check"


def preflight():
    a = monitor.preflight(case, "chemFoam", 1e-4, 1e-3, timeout=120)
    return a["status"] in ("ok", "warn"), f"status {a['status']}, projected {a['est_full_runtime']}"


def fields():
    last = foam.time_dirs(case)[-1]
    names = sorted(p.name for p in last.iterdir() if p.is_file())
    try:
        v = foam.read_internal_field(last / "T")
        return True, f"t = {last.name}: {names}; T read as {v}"
    except ValueError as e:
        return False, f"t = {last.name}: {names}; a one-cell field is written as 'uniform' and read_internal_field refuses it: {e}"


for name, fn in [("monitor reads and judges chemFoam's log", log), ("dictionaries are parsed", dictionaries),
                 ("monitor.preflight on chemFoam", preflight), ("fields of the one-cell case are read", fields)]:
    attempt(name, fn)

# ---- the untouched tutorial against the reference ---------------------------------------------------------
hist = next(case.glob("chemFoam.out"), None)
if hist is None:
    line("GAP", "chemFoam's own history file (time, temperature, pressure)", "not written")
else:
    d = np.array([[float(v) for v in ln.split()] for ln in hist.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")])
    t, T = d[:, 0], d[:, 1]
    crit = REF["ignition"]["criterion_K"]
    i = int(np.argmax(T >= crit))
    ign = float(t[i - 1] + (crit - T[i - 1]) * (t[i] - t[i - 1]) / (T[i] - T[i - 1]))
    rt, rT = np.array(REF["history"]["t_s"]), np.array(REF["history"]["T_K"])
    line("INFO", "history file", f"{len(t)} rows, columns {d.shape[1]}, head: {hist.read_text().splitlines()[0][:80]}")
    line("INFO", "ignition time (first reach of 1200 K)", f"tutorial {ign:.4e} s, reference {REF['ignition']['time_s']:.4e} s, difference {100 * (ign / REF['ignition']['time_s'] - 1):+.2f} %")
    line("INFO", "temperature at the end", f"tutorial {T[-1]:.1f} K, reference {REF['final']['T_K']:.1f} K, difference {100 * (T[-1] / REF['final']['T_K'] - 1):+.3f} %")
    shifted = np.interp(t, rt + (ign - REF["ignition"]["time_s"]), rT)
    line("INFO", "largest temperature difference along the history", f"{np.abs(T - np.interp(t, rt, rT)).max():.1f} K as computed; "
         f"{np.abs(T - shifted).max():.1f} K after shifting the reference by the ignition-time difference")
    line("INFO", "time steps of the tutorial", f"smallest {np.diff(t).min():.2e} s, largest {np.diff(t).max():.2e} s; tolerances as shipped: "
         f"{foamdict.parse(case / 'constant' / 'chemistryProperties').get('odeCoeffs')}")
print(f"\n{len(GAPS)} gaps" + (": " + "; ".join(GAPS) if GAPS else ""))
shutil.rmtree(tmp, ignore_errors=True)
