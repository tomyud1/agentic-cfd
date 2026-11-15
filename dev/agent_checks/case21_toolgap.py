"""A file of the developing agent. An agent that solves the case does not need it and should not open it.

Before case 21 (a detonation in a straight tube; dev/LOOP.md, step 2): does OpenFOAM's own burning solver carry
a detonation at all, how is the wave started, and what does a run cost? This is a TRIAL, not a solution: one
coarse grid, settings taken from OpenFOAM's examples, nothing tuned.

    PYTHONPATH=. python dev/agent_checks/case21_toolgap.py [scheme] [cells] [solver] [length m] [end time s] [hot layer m] [hot pressure Pa]
    scheme: limitedLinear (the example's) or upwind;  cells: default 600;  solver: reactingFoam or rhoReactingFoam

Decided before the first run (2026-11-15):
  - Set-up: a tube 60 mm long, closed at both ends, 0.1 mm cells (the cell size of the engine papers),
    hydrogen and air (2 : 1 : 3.76) at 1 atm and 300 K. The first 3 mm start at 2500 K and 5 MPa (the same
    mixture), which is above the pressure at the front of a CJ detonation, so the wave starts overdriven.
  - End of the run: 20 microseconds (a CJ wave covers 40 mm), or 15 minutes of wall time, whichever is first.
  - What counts as "carries a detonation": the pressure front moves, over the second half of the run, within
    5 % of the CJ speed of cfdlib.chem.cj_detonation, and the gas behind it is hotter than 2500 K.
  - What counts as a gap: the solver stops with an error, or the front falls behind the burning (speed far
    below CJ, cold gas between shock and flame).
First run (60 mm, 3 mm at 5 MPa): the solver ran (212 s, 88 microseconds per cell and step) and carried a
coupled wave at a steady 2133 m/s, 7.8 % above CJ, with 2.3 MPa behind it where CJ has 1.58: still pushed by
the strong start. Second run, decided before it: twice the tube (120 mm), half the hot layer (1.5 mm) at 3 MPa,
to 50 microseconds, wall-time limit 30 minutes: does the speed come down towards CJ?
Second run: yes, from the other side. 1957 m/s over the second half, 1.1 % below CJ and still rising (1951, then 1960),
1.56 MPa and 2920 K behind the wave (CJ 1.58 and 2970), flame 0.02 mm behind the pressure front; 689 s on one core,
57 microseconds per cell and step. The highest pressure is 1.97 MPa where a resolved front has 2.81: the front is
not resolved by 0.1 mm cells. These settings became the problem statement of case 21.
Lines say PASS, GAP or INFO.
"""
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from cfdlib import chem, foam

SCHEME = sys.argv[1] if len(sys.argv) > 1 else "limitedLinear"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
SOLVER = sys.argv[3] if len(sys.argv) > 3 else "reactingFoam"
arg = lambda k, default: float(sys.argv[k]) if len(sys.argv) > k else default
L, X_HOT, T_HOT, P_HOT = arg(4, 0.06), arg(6, 0.003), 2500.0, arg(7, 5.0e6)
P1, T1, MIX = 101325.0, 300.0, dict(H2=2.0, O2=1.0, N2=3.76)
T_END, WALL_LIMIT, WRITES = arg(5, 2.0e-5), 1800, 20
HEAD = "FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}\n"


def line(kind, name, detail=""):
    print(f"{kind:4s}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def field(case, name, dims, values, wall="zeroGradient", cls="volScalarField"):
    inner = (f"uniform {values}" if np.isscalar(values) or isinstance(values, str)
             else f"nonuniform List<scalar> {len(values)}\n(\n" + "\n".join(f"{v:.10g}" for v in values) + "\n)")
    wall_bc = "type zeroGradient;" if wall == "zeroGradient" else wall
    (case / "0" / name).write_text(HEAD.format(cls=cls, obj=name) + f"dimensions {dims};\ninternalField {inner};\nboundaryField\n{{\n"
                                   f"    \"(left|right)\" {{ {wall_bc} }}\n    sides {{ type empty; }}\n}}\n")


def main():
    mech = chem.build(chem.openfoam_conventions())
    cj = chem.cj_detonation(mech, P1, T1, MIX)
    vn = chem.shock_frozen(mech, P1, T1, MIX, cj["D"])
    fresh = chem.Parcel(mech, P1, T1, MIX)
    line("INFO", "theory (cfdlib.chem, OpenFOAM's constants)", f"CJ speed {cj['D']:.1f} m/s, behind the wave {cj['p'] / 1e6:.3f} MPa and {cj['T']:.0f} K; "
         f"at the front {vn['p'] / 1e6:.3f} MPa and {vn['T']:.0f} K")
    case = Path(tempfile.mkdtemp(prefix="case21_toolgap_", dir=Path.home()))
    tut = Path(foam.foam_env("FOAM_TUTORIALS")) / "combustion"
    for d in ("0", "constant", "system"):
        (case / d).mkdir()
    shutil.copytree(tut / "chemFoam/h2/chemkin", case / "chemkin")
    dx = L / N
    x = (np.arange(N) + 0.5) * dx
    hot = x < X_HOT
    Y = dict(zip(mech.species, fresh.Y0))
    field(case, "p", "[1 -1 -2 0 0 0 0]", np.where(hot, P_HOT, P1))
    field(case, "T", "[0 0 0 1 0 0 0]", np.where(hot, T_HOT, T1))
    field(case, "U", "[0 1 -1 0 0 0 0]", "(0 0 0)", wall="type fixedValue; value uniform (0 0 0);", cls="volVectorField")
    field(case, "alphat", "[1 -1 -1 0 0 0 0]", 0, wall="type calculated; value uniform 0;")
    for s in ("H2", "O2", "N2"):
        field(case, s, "[0 0 0 0 0 0 0]", float(Y[s]))
    field(case, "Ydefault", "[0 0 0 0 0 0 0]", 0)
    h = dx
    (case / "system/blockMeshDict").write_text(HEAD.format(cls="dictionary", obj="blockMeshDict") + f"""scale 1;
vertices ((0 0 0) ({L} 0 0) ({L} {h} 0) (0 {h} 0) (0 0 {h}) ({L} 0 {h}) ({L} {h} {h}) (0 {h} {h}));
blocks (hex (0 1 2 3 4 5 6 7) ({N} 1 1) simpleGrading (1 1 1));
edges ();
boundary
(
    left  {{ type wall; faces ((0 4 7 3)); }}
    right {{ type wall; faces ((1 2 6 5)); }}
    sides {{ type empty; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
""")
    (case / "system/controlDict").write_text(HEAD.format(cls="dictionary", obj="controlDict") + f"""application {SOLVER};
startFrom startTime; startTime 0; stopAt endTime; endTime {T_END}; deltaT 1e-9;
writeControl adjustable; writeInterval {T_END / WRITES}; purgeWrite 0; writeFormat ascii; writePrecision 10;
timeFormat general; timePrecision 10; runTimeModifiable true; adjustTimeStep yes; maxCo 0.1; maxDeltaT 5e-9;
""")
    div = "Gauss upwind" if SCHEME == "upwind" else "Gauss limitedLinear 1"
    divU = "Gauss upwind" if SCHEME == "upwind" else "Gauss limitedLinearV 1"
    (case / "system/fvSchemes").write_text(HEAD.format(cls="dictionary", obj="fvSchemes") + f"""ddtSchemes {{ default Euler; }}
gradSchemes {{ default Gauss linear; }}
divSchemes
{{
    default none;
    div(phi,U) {divU};
    div(phi,Yi_h) {div};
    div(phi,K) {div};
    div(phid,p) {div};
    div(phiv,p) {div};
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear orthogonal; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default orthogonal; }}
""")
    (case / "system/fvSolution").write_text(HEAD.format(cls="dictionary", obj="fvSolution") + """solvers
{
    "rho.*" { solver diagonal; }
    "(p|U|h|Yi)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-10; relTol 0; }
    "(p|U|h|Yi)Final" { $p; }
}
PIMPLE { momentumPredictor yes; nOuterCorrectors 2; nCorrectors 2; nNonOrthogonalCorrectors 0; transonic yes; }
""")
    thermo_type = "hePsiThermo" if SOLVER == "reactingFoam" else "heRhoThermo"
    (case / "constant/thermophysicalProperties").write_text(HEAD.format(cls="dictionary", obj="thermophysicalProperties") + f"""thermoType
{{
    type {thermo_type}; mixture reactingMixture; transport sutherland; thermo janaf;
    energy sensibleEnthalpy; equationOfState perfectGas; specie specie;
}}
inertSpecie N2;
CHEMKINFile "<case>/chemkin/chem.inp";
CHEMKINThermoFile "<case>/chemkin/therm.dat";
CHEMKINTransportFile "<case>/chemkin/transportProperties";
""")
    (case / "constant/chemistryProperties").write_text(HEAD.format(cls="dictionary", obj="chemistryProperties") + """chemistryType { solver ode; }
chemistry on;
initialChemicalTimeStep 1e-10;
odeCoeffs { solver seulex; absTol 1e-10; relTol 1e-3; }
""")
    (case / "constant/combustionProperties").write_text(HEAD.format(cls="dictionary", obj="combustionProperties") +
                                                        "combustionModel laminar;\nactive true;\nlaminarCoeffs {}\n")
    (case / "constant/turbulenceProperties").write_text(HEAD.format(cls="dictionary", obj="turbulenceProperties") + "simulationType laminar;\n")
    if SOLVER == "rhoReactingFoam":
        (case / "constant/g").write_text(HEAD.format(cls="uniformDimensionedVectorField", obj="g") + "dimensions [0 1 -2 0 0 0 0];\nvalue (0 0 0);\n")
    try:
        foam.run("blockMesh", case, "blockMesh")
        t0 = time.perf_counter()
        foam.run(SOLVER, case, SOLVER, timeout=WALL_LIMIT)
        wall, ended = time.perf_counter() - t0, "reached the end time"
    except subprocess.TimeoutExpired:
        wall, ended = float(WALL_LIMIT), f"stopped at the wall-time limit of {WALL_LIMIT} s"
    except Exception as e:                                         # noqa: BLE001
        log = case / f"log.{SOLVER}"
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:]) if log.exists() else str(e)[-1500:]
        line("GAP", f"{SOLVER} stopped with an error ({SCHEME}, {N} cells)", f"case kept in {case}")
        print(tail)
        return
    times = [t for t in foam.time_dirs(case) if float(t.name) > 0]
    if len(times) < 6:
        line("GAP", "too few written times to measure a speed", f"{len(times)}; {ended}; case kept in {case}")
        return
    half = 0.5 * (P1 + cj["p"])
    rows = []
    for t in times:
        p, T = foam.read_internal_field(t / "p"), foam.read_internal_field(t / "T")
        i = int(np.max(np.nonzero(p > half)[0])) if (p > half).any() else -1
        j = int(np.max(np.nonzero(T > 1500.0)[0])) if (T > 1500.0).any() else -1       # the burning front
        back = slice(max(i - 60, 0), max(i - 20, 1))                                   # 2 to 6 mm behind the pressure front
        rows.append(dict(t=float(t.name), x_p=x[i] if i >= 0 else np.nan, x_T=x[j] if j >= 0 else np.nan, p_max=float(p.max()),
                         p_back=float(p[back].mean()), T_back=float(T[back].mean())))
    steps = sum(1 for ln in (case / f"log.{SOLVER}").read_text(errors="replace").splitlines() if ln.startswith("Time = "))
    line("PASS", f"{SOLVER} runs a shock with burning in one dimension ({SCHEME}, {N} cells)", f"{ended}; {steps} steps to t = {rows[-1]['t']:.3e} s "
         f"in {wall:.0f} s: {1e6 * wall / (steps * N):.0f} microseconds per cell and step on one core")
    for r in rows[::2]:
        line("INFO", f"t = {1e6 * r['t']:5.2f} us", f"pressure front at {1e3 * r['x_p']:6.2f} mm, burning front at {1e3 * r['x_T']:6.2f} mm, highest pressure "
             f"{r['p_max'] / 1e6:.2f} MPa, behind the front {r['p_back'] / 1e6:.2f} MPa and {r['T_back']:.0f} K")
    late = [r for r in rows if r["t"] >= 0.5 * rows[-1]["t"] and np.isfinite(r["x_p"]) and r["x_p"] < L - 3 * dx]
    if len(late) < 3:
        line("GAP", "the pressure front is not there in the second half of the run")
        return
    D = float(np.polyfit([r["t"] for r in late], [r["x_p"] for r in late], 1)[0])
    k = len(late) // 2                                             # is it still slowing down? the speed in the two halves of that stretch
    D_a, D_b = (float(np.polyfit([r["t"] for r in part], [r["x_p"] for r in part], 1)[0]) for part in (late[:k + 1], late[k:]))
    line("INFO", "trend of the speed", f"{D_a:.0f} m/s then {D_b:.0f} m/s; pressure 2 to 6 mm behind the front {late[0]['p_back'] / 1e6:.2f} then "
         f"{late[-1]['p_back'] / 1e6:.2f} MPa (CJ {cj['p'] / 1e6:.2f}); start: {1e3 * X_HOT:.1f} mm at {P_HOT / 1e6:.1f} MPa and {T_HOT:.0f} K")
    gap = float(np.mean([r["x_p"] - r["x_T"] for r in late]))
    hotb = float(np.mean([r["T_back"] for r in late]))
    ok = abs(D / cj["D"] - 1) < 0.05 and hotb > 2500.0
    line("PASS" if ok else "GAP", "the wave is a detonation near the CJ speed", f"speed over the second half {D:.0f} m/s, {100 * (D / cj['D'] - 1):+.1f} % from CJ "
         f"({cj['D']:.0f}); burning front {1e3 * gap:.2f} mm behind the pressure front; {hotb:.0f} K behind (CJ {cj['T']:.0f}); "
         f"pressure behind {np.mean([r['p_back'] for r in late]) / 1e6:.2f} MPa (CJ {cj['p'] / 1e6:.2f})")
    shutil.rmtree(case, ignore_errors=True)


if __name__ == "__main__":
    main()
