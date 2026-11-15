"""Tool-gap check before case 18 (shock tube; dev/LOOP.md, step 2): can the tools run OpenFOAM's shock solver
(rhoCentralFoam), read its log and dictionaries, and extract what the problem asks for?

    PYTHONPATH=. python dev/agent_checks/case18_toolgap.py      (inside WSL, needs OpenFOAM; about a minute)

It copies OpenFOAM's own shock-tube tutorial to a temporary folder, runs it and calls the tools on it. The
tutorial is NOT a solution of the case: the solver agent of case 18 writes its own set-up, and nothing from
here goes into the case folder. Each line says PASS or GAP; a GAP is something to fix or to answer before a
fresh agent is launched.
"""
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np

from cfdlib import contract, foam, foamdict, monitor, perf

GAPS = []


def line(ok, name, detail=""):
    print(f"{'PASS' if ok else 'GAP '}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        GAPS.append(name)


def attempt(name, fn):
    """Run one check; an exception is a gap, with its last line."""
    try:
        ok, detail = fn()
        line(ok, name, detail)
    except Exception as e:                                         # noqa: BLE001  (every failure is a finding here)
        line(False, name, f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else traceback.format_exc().splitlines()[-1]}")


tmp = Path(tempfile.mkdtemp(prefix="case18_toolgap_", dir=Path.home()))
case = tmp / "shockTube"
shutil.copytree(Path(foam.foam_env("FOAM_TUTORIALS")) / "compressible" / "rhoCentralFoam" / "shockTube", case)
shutil.copytree(case / "0.orig", case / "0")                      # what the tutorial's own Allrun does first
print("tutorial files:", sorted(str(p.relative_to(case)) for p in case.rglob("*") if p.is_file()))
ctrl = foamdict.parse(case / "system" / "controlDict")
END = float(ctrl["endTime"])


def run_all():
    foam.run("blockMesh", case, "blockMesh")
    if (case / "system" / "setFieldsDict").exists():
        foam.run("setFields", case, "setFields")
    wall = foam.run("rhoCentralFoam", case, "rhoCentralFoam")
    return True, f"wall time {wall:.1f} s to t = {END:g}"


def log_parsed():
    L = monitor.parse_log(case / "log.rhoCentralFoam")
    arr = monitor.log_arrays(L["steps"])
    n = len(L["steps"])
    co = np.asarray(arr["co"], dtype=float)
    have_co = np.isfinite(co).sum()
    ok = n > 10 and L["ended"] and not L["fatal"] and have_co >= n - 1     # the line is printed before each step's 'Time ='
    return ok, f"{n} steps, ended {L['ended']}, Courant number read on {have_co} of {n} steps, residual fields {sorted(L['steps'][-1]['res'])}"


def assessed():
    a = monitor.assess(case / "log.rhoCentralFoam", end_time=END)
    return a["status"] in ("ok", "warn"), f"status {a['status']}: " + "; ".join(f"{c['name']} {c['status']} ({c.get('value')})" for c in a["checks"])


def preflight():
    a = monitor.preflight(case, "rhoCentralFoam", END / 10.0, END, timeout=120)
    checks = "; ".join(f"{c['name']} {c['status']} ({c.get('value')})" for c in a["checks"])
    return (a["status"] in ("ok", "warn") and not a["timed_out"],
            f"status {a['status']}, {a['steps']} steps, projected full run {a['est_full_runtime']} ({a['est_kind']}): {checks}")


def dictionaries():
    bad = []
    s = foamdict.case_settings(case, unparsed=bad)
    thermo = [k for k in s if "thermophysicalProperties" in k]
    return not bad and len(thermo) > 3, f"{len(s)} settings, {len(thermo)} of them thermophysical, unparsed {bad}"


def ignored():
    w = foam.ignored_settings(case)
    return True, f"{len(w)} warnings" + (f": {w[0][:120]}" if w else "")


def uncovered():
    u = contract.uncovered_settings([], case)
    return len(u) > 5, f"with no decisions, {len(u)} settings are reported as undocumented, for example {sorted(u)[:3]}"


def fields():
    last = foam.time_dirs(case)[-1]
    names = sorted(p.name for p in last.iterdir() if p.is_file())
    got = {n: foam.read_internal_field(last / n).shape for n in ("p", "T", "U") if (last / n).exists()}
    rho = (last / "rho").exists()
    return len(got) == 3, f"t = {last.name}: files {names}; read {got}; density written by default: {rho}"


def cell_centres():
    foam.run("postProcess -func writeCellCentres -latestTime", case, "writeCellCentres")
    last = foam.time_dirs(case)[-1]
    c = foam.read_internal_field(last / "C")
    v = foam.read_internal_field(last / "V") if (last / "V").exists() else None
    return c.ndim == 2 and c.shape[1] == 3, f"cell centres {c.shape}, x from {c[:, 0].min():.3g} to {c[:, 0].max():.3g}; cell volumes: {None if v is None else v.shape}"


def density_and_totals():
    """Mass and energy in the tube need the density: is there a tool path to it?"""
    foam.run("postProcess -func 'writeObjects(rho)' -latestTime", case, "writeRho")
    last = foam.time_dirs(case)[-1]
    ok = (last / "rho").exists()
    return ok, "postProcess writeObjects(rho) " + ("wrote the density" if ok else "wrote nothing: the density must come from p and T, or from a function object during the run")


def live_view():
    out = monitor.peek(case)
    return True, f"peek returned {type(out).__name__}"


def performance():
    p = perf.summarize(case, "rhoCentralFoam", cells=1000, cores=1)
    return True, f"keys {sorted(p)[:8]}"


for name, fn in [("run blockMesh, setFields and rhoCentralFoam with foam.run", run_all),
                 ("monitor.parse_log reads the solver's log (steps, Courant number)", log_parsed),
                 ("monitor.assess judges the run", assessed),
                 ("monitor.preflight runs a short copy and projects the full run", preflight),
                 ("foamdict.case_settings parses every dictionary (thermophysical ones too)", dictionaries),
                 ("foam.ignored_settings works on this solver", ignored),
                 ("contract.uncovered_settings lists the settings to document", uncovered),
                 ("foam.read_internal_field reads p, T, U", fields),
                 ("cell centres for profiles along the tube", cell_centres),
                 ("density for mass and energy totals", density_and_totals),
                 ("monitor.peek shows the run", live_view),
                 ("perf.summarize times the run", performance)]:
    attempt(name, fn)

print(f"\n{len(GAPS)} gaps" + (": " + "; ".join(GAPS) if GAPS else ""))
shutil.rmtree(tmp, ignore_errors=True)
