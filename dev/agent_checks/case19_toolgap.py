"""Before case 19 (a shock going round a corner; dev/LOOP.md, step 2): is the problem well posed as written,
and can the tools handle a two-dimensional run of OpenFOAM's shock solver?

    PYTHONPATH=. python dev/agent_checks/case19_toolgap.py [cells per metre]     (inside WSL, needs OpenFOAM)

It builds a throwaway coarse set-up of cases/shock_diffraction_corner/problem.yaml in a temporary folder, runs
it and calls the tools on it. The set-up is NOT a solution of the case and is not tuned: the solver agent
writes its own, and nothing from here goes into the case folder. Lines say PASS or GAP (a tool problem to fix
before a fresh agent is launched) or INFO (what the coarse run says about the problem as posed: does part of
the shock stay undisturbed, does the bent shock stay clear of the floor until the time of interest).
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from cfdlib import foam, foamdict, monitor

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ROOT = Path(__file__).resolve().parents[2]
G = {k: float(v["value"]) for k, v in yaml.safe_load((ROOT / "cases/shock_diffraction_corner/problem.yaml").read_text(encoding="utf-8"))["given"].items()}
g, R, p0, T0, Ms, L, H, Hs, T_END = G["gamma"], G["R"], G["p_0"], G["T_0"], G["M_s"], G["L"], G["H"], G["H_s"], G["t_end"]
rho0, c0 = p0 / (R * T0), np.sqrt(g * R * T0)
W = Ms * c0                                                        # speed of the incoming shock
p1 = p0 * (1 + 2 * g / (g + 1) * (Ms ** 2 - 1))                    # state behind it (the shock's jump conditions)
rho1 = rho0 * (g + 1) * Ms ** 2 / ((g - 1) * Ms ** 2 + 2)
u1 = 2 * c0 / (g + 1) * (Ms - 1 / Ms)
T1 = p1 / (rho1 * R)
HEAD = "FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}\n"
GAPS = []


def line(kind, name, detail=""):
    print(f"{kind:4s}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if kind == "GAP":
        GAPS.append(name)


def field(case, name, cls, dim, internal, inlet):
    wall = "type slip;" if name == "U" else "type zeroGradient;"
    (case / "0" / name).write_text(HEAD.format(cls=cls, obj=name) + f"dimensions {dim};\ninternalField uniform {internal};\nboundaryField\n{{\n"
                                   f"    inlet {{ type fixedValue; value uniform {inlet}; }}\n    walls {{ {wall} }}\n    frontAndBack {{ type empty; }}\n}}\n")


tmp = Path(tempfile.mkdtemp(prefix="case19_toolgap_", dir=Path.home()))
case = tmp / "corner"
tut = Path(foam.foam_env("FOAM_TUTORIALS")) / "compressible" / "rhoCentralFoam" / "shockTube"
for d in ("0", "constant", "system"):
    (case / d).mkdir(parents=True)
for f in ("constant/thermophysicalProperties", "constant/turbulenceProperties", "system/fvSchemes", "system/fvSolution"):
    shutil.copy(tut / f, case / f)
foam.run(f"foamDictionary -entry mixture/specie/molWeight -set {foam.mol_weight(R)!r} constant/thermophysicalProperties", case, "molWeight")
nx, n_low, n_up = round(N * L), round(N * Hs), round(N * (H - Hs))
(case / "system/blockMeshDict").write_text(HEAD.format(cls="dictionary", obj="blockMeshDict") + f"""scale 1;
vertices ( (0 0 0) ({L} 0 0) (0 {Hs} 0) ({L} {Hs} 0) (0 {H} 0) ({L} {H} 0)
           (0 0 0.1) ({L} 0 0.1) (0 {Hs} 0.1) ({L} {Hs} 0.1) (0 {H} 0.1) ({L} {H} 0.1) );
blocks ( hex (0 1 3 2 6 7 9 8) ({nx} {n_low} 1) simpleGrading (1 1 1)
         hex (2 3 5 4 8 9 11 10) ({nx} {n_up} 1) simpleGrading (1 1 1) );
boundary
(
    inlet {{ type patch; faces ( (2 8 10 4) ); }}
    walls {{ type wall; faces ( (0 6 8 2) (0 1 7 6) (1 3 9 7) (3 5 11 9) (4 10 11 5) ); }}
    frontAndBack {{ type empty; faces ( (0 2 3 1) (2 4 5 3) (6 7 9 8) (8 9 11 10) ); }}
);
""")
field(case, "p", "volScalarField", "[1 -1 -2 0 0 0 0]", p0, p1)
field(case, "T", "volScalarField", "[0 0 0 1 0 0 0]", T0, T1)
field(case, "U", "volVectorField", "[0 1 -1 0 0 0 0]", "(0 0 0)", f"({u1} 0 0)")
(case / "system/controlDict").write_text(HEAD.format(cls="dictionary", obj="controlDict") + f"""application rhoCentralFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {T_END!r}; deltaT 1e-8; adjustTimeStep yes; maxCo 0.1; maxDeltaT 1;
writeControl adjustableRunTime; writeInterval {T_END / 2!r}; purgeWrite 0; writeFormat ascii; writePrecision 12;
writeCompression off; timeFormat general; timePrecision 9; runTimeModifiable true;
""")

try:
    foam.run("blockMesh", case, "blockMesh")
    wall = foam.run("rhoCentralFoam", case, "rhoCentralFoam")
    line("PASS", "blockMesh and rhoCentralFoam run in two dimensions with foam.run", f"{nx} x {n_low + n_up} cells, {wall:.1f} s")
except Exception as e:                                            # noqa: BLE001
    line("GAP", "run", str(e).splitlines()[0][:200])
    sys.exit(1)
Lg = monitor.parse_log(case / "log.rhoCentralFoam")
a = monitor.assess(case / "log.rhoCentralFoam", end_time=T_END)
line("PASS" if a["status"] in ("ok", "warn") else "GAP", "monitor.assess judges the run",
     f"{len(Lg['steps'])} steps, status {a['status']}: " + "; ".join(f"{c['name']} {c['status']} ({c.get('value')})" for c in a["checks"]))
bad = []
s = foamdict.case_settings(case, unparsed=bad)
line("PASS" if not bad else "GAP", "foamdict.case_settings parses every dictionary", f"{len(s)} settings, unparsed {bad}")
foam.run("postProcess -func writeCellCentres -latestTime", case, "writeCellCentres")
times = foam.time_dirs(case)
last = times[-1]
C = foam.read_internal_field(last / "C")
x, y = C[:, 0], C[:, 1]
p, rho = foam.read_internal_field(last / "p"), foam.read_internal_field(last / "rho")
U = foam.read_internal_field(last / "U")
line("PASS" if len(times) == 3 and p.shape == (nx * (n_low + n_up),) else "GAP", "fields at half time and at the time of interest are read",
     f"times {[t.name for t in times]}, p {p.shape}, U {U.shape}")
pf = monitor.preflight(case, "rhoCentralFoam", T_END / 20, T_END, timeout=300)
line("PASS" if pf["status"] in ("ok", "warn") else "GAP", "monitor.preflight on the 2D case", f"status {pf['status']}, projected {pf['est_full_runtime']}")

# ---- what the coarse run says about the problem as posed -------------------------------------------------
h = 1.0 / N
print(f"\nexact, from the jump conditions: shock speed {W:.1f} m/s, behind it p {p1:.0f} Pa, rho {rho1:.4f} kg/m3, u {u1:.1f} m/s, "
      f"T {T1:.1f} K, flow Mach number {u1 / np.sqrt(g * R * T1):.3f}; at t_end the undisturbed shock is at x = {W * T_END:.4f} m")


def front(row_y):
    """x of the shock on the row of cells nearest to row_y: the rightmost cell where p is above the mean of p0 and p1."""
    m = np.abs(y - row_y) < 0.5 * h
    hit = x[m][p[m] > 0.5 * (p0 + p1)]
    return float(hit.max() + 0.5 * h) if hit.size else float("nan")


tops = [front(H - (k + 0.5) * h) for k in range(round(0.25 * N))]            # the quarter of the rows nearest the top wall
line("INFO", "shock near the top wall against the exact position",
     f"x = {np.mean(tops):.4f} m, exact {W * T_END:.4f} m, difference {(np.mean(tops) - W * T_END) / h:+.2f} cells; spread over those rows {(max(tops) - min(tops)) / h:.2f} cells")
rows = np.array([[yy, front(yy)] for yy in np.arange(Hs + 0.5 * h, H, h)])
straight = rows[np.abs(rows[:, 1] - np.mean(tops)) < 1.5 * h, 0]
line("INFO", "how much of the shock above the corner is still straight at t_end",
     f"from y = {straight.min():.3f} m to the top wall ({H - straight.min():.3f} m of {H - Hs:.3f} m); behind it, mean p {p[(y > straight.min() + 0.05) & (x > 0.6 * W * T_END) & (x < 0.9 * W * T_END)].mean() / p1:.4f} of exact")
col = np.abs(x - 0.5 * h) < 0.5 * h                                           # the cells along the step face (x = 0, y < Hs)
below = col & (y < Hs)
reached = y[below][p[below] > 1.2 * p0]
line("INFO", "the bent shock on the step face (the wall below the corner)",
     f"has run down to y = {reached.min():.3f} m, {Hs - reached.min():.3f} m below the corner; the floor is {Hs:.2f} m below" if reached.size else "no pressure rise on the face")
floor = y < h
line("INFO", "pressure on the floor at t_end", f"largest p / p0 = {p[floor].max() / p0:.4f} (1 means no wave has arrived)")
right = x > L - h
line("INFO", "pressure on the right wall at t_end", f"largest p / p0 = {p[right].max() / p0:.4f}")
mass = float(np.sum(rho) * h * h)
line("INFO", "mass in the box against start plus inflow",
     f"relative difference {(mass - (rho0 * (L * H) + rho1 * u1 * (H - Hs) * T_END)) / (rho1 * u1 * (H - Hs) * T_END):+.2e} of the inflow")
print(f"\n{len(GAPS)} gaps" + (": " + "; ".join(GAPS) if GAPS else ""))
shutil.rmtree(tmp, ignore_errors=True)
