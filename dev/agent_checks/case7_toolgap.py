"""Tool-gap check before case 7 (turbulent flat plate, Spalart-Allmaras; dev/LOOP.md, step 2): can the
tools read a turbulence case's dictionaries and logs, and extract what the problem asks for?

    PYTHONPATH=. python dev/agent_checks/case7_toolgap.py [iterations]      (inside WSL, needs OpenFOAM)

It builds a throwaway setup on NASA's coarsest grid in a temporary folder, runs simpleFoam for a few
hundred iterations and calls the tools on it. The setup is NOT a solution of the case and is not tuned:
the solver agent of case 7 writes its own, and nothing from here goes into the case folder. Each line
says PASS or GAP; a GAP is something to fix before a fresh agent is launched.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from cfdlib import contract, foam, foamdict, monitor, structgrid

ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 300
GRID = Path("cases/turbulent_flat_plate_sa/reference/grids/flatplate_clust2_4levelsdown_35x25.p2dfmt.gz")
U, NU = 50.0, 1.0e-5
HEAD = "FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}\n"
GAPS = []


def line(ok, name, detail=""):
    print(f"{'PASS' if ok else 'GAP '}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        GAPS.append(name)


def field(case, name, cls, dim, internal, bc):
    body = "".join(f"    {p} {{ {v} }}\n" for p, v in bc.items())
    (case / "0" / name).write_text(HEAD.format(cls=cls, obj=name) + f"dimensions {dim};\ninternalField uniform {internal};\n"
                                   f"boundaryField\n{{\n{body}    symmetry {{ type symmetryPlane; }}\n"
                                   f"    frontAndBack {{ type empty; }}\n}}\n")


def setup(case):
    for d in ("0", "constant", "system"):
        (case / d).mkdir(parents=True)
    x, y = structgrid.read_plot3d_2d(GRID)
    plate = structgrid.index_range(x[0], lo=0.0)
    info = structgrid.write_polymesh(case, x, y, {"inlet": ("i-", None), "outlet": ("i+", None), "top": ("j+", None),
                                                  "symmetry": ("j-", (0, plate[0])), "plate": ("j-", plate)},
                                     types={"plate": "wall", "symmetry": "symmetryPlane"})
    u, nt = f"({U} 0 0)", 3 * NU
    field(case, "U", "volVectorField", "[0 1 -1 0 0 0 0]", u, dict(
        inlet=f"type fixedValue; value uniform {u};", outlet="type zeroGradient;",
        top=f"type inletOutlet; inletValue uniform {u}; value uniform {u};", plate="type noSlip;"))
    field(case, "p", "volScalarField", "[0 2 -2 0 0 0 0]", "0", dict(
        inlet="type zeroGradient;", outlet="type fixedValue; value uniform 0;", top="type fixedValue; value uniform 0;",
        plate="type zeroGradient;"))
    field(case, "nuTilda", "volScalarField", "[0 2 -1 0 0 0 0]", nt, dict(
        inlet=f"type fixedValue; value uniform {nt};", outlet="type zeroGradient;",
        top=f"type inletOutlet; inletValue uniform {nt}; value uniform {nt};", plate="type fixedValue; value uniform 0;"))
    field(case, "nut", "volScalarField", "[0 2 -1 0 0 0 0]", "0", dict(
        inlet="type calculated; value uniform 0;", outlet="type calculated; value uniform 0;",
        top="type calculated; value uniform 0;", plate="type fixedValue; value uniform 0;"))
    (case / "constant/transportProperties").write_text(HEAD.format(cls="dictionary", obj="transportProperties")
                                                       + f"transportModel Newtonian;\nnu {NU};\n")
    (case / "constant/turbulenceProperties").write_text(HEAD.format(cls="dictionary", obj="turbulenceProperties")
                                                        + "simulationType RAS;\nRAS { RASModel SpalartAllmaras; turbulence on; printCoeffs on; }\n")
    (case / "system/controlDict").write_text(HEAD.format(cls="dictionary", obj="controlDict") + f"""application simpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {ITER}; deltaT 1; writeControl timeStep; writeInterval {ITER};
purgeWrite 0; writeFormat ascii; writePrecision {max(12, info['digits_needed'])}; writeCompression off; timeFormat general;
timePrecision 6; runTimeModifiable true;
functions
{{
    wallShearStress {{ type wallShearStress; libs (fieldFunctionObjects); patches (plate); writeControl writeTime; }}
    yPlus {{ type yPlus; libs (fieldFunctionObjects); writeControl writeTime; }}
    forces {{ type forceCoeffs; libs (forces); patches (plate); rho rhoInf; rhoInf 1.2; CofR (0 0 0); liftDir (0 1 0);
              dragDir (1 0 0); pitchAxis (0 0 1); magUInf {U}; lRef 2; Aref 2; writeControl timeStep; writeInterval 1; }}
}}
""")
    (case / "system/fvSchemes").write_text(HEAD.format(cls="dictionary", obj="fvSchemes") + """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); div(phi,nuTilda) bounded Gauss upwind;
             div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
""")
    (case / "system/fvSolution").write_text(HEAD.format(cls="dictionary", obj="fvSolution") + """solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-10; relTol 0.05; }
    "(U|nuTilda)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0.05; }
}
SIMPLE { nNonOrthogonalCorrectors 0; consistent yes; residualControl { p 1e-9; U 1e-9; nuTilda 1e-9; } }
relaxationFactors { equations { U 0.9; nuTilda 0.9; } }
""")
    return x, y, plate, info


def main():
    case = Path(tempfile.mkdtemp(prefix="case7gap_", dir=Path.home()))
    try:
        x, y, plate, info = setup(case)
        # 1. dictionaries
        unparsed = []
        settings = foamdict.case_settings(case, unparsed)
        want = ["constant/turbulenceProperties/RAS/RASModel", "system/fvSchemes/wallDist/method",
                "0/nuTilda/boundaryField/inlet/value", "0/nut/boundaryField/plate/type",
                "system/fvSchemes/divSchemes/div(phi,nuTilda)"]
        missing = [k for k in want if k not in settings]
        line(not unparsed and not missing, "dictionary reader: every file of a turbulence case is read, model settings included",
             f"unparsed {unparsed}, missing {missing}, {len(settings)} settings")
        unc = contract.uncovered_settings([dict(settings=["system/*", "0/U/*", "0/p/*", "constant/transportProperties/*"])], case)
        keys = sorted(k.split(": ")[-1] for k in unc)         # each message ends with the setting's path
        line(all(any(k.startswith(p) for k in keys) for p in ("constant/turbulenceProperties/", "0/nuTilda/", "0/nut/")),
             "contract: turbulence settings without a decision are reported as uncovered",
             f"{len(keys)} uncovered, e.g. {[k for k in keys if 'turbulenceProperties' in k][:2]}")
        ign = foam.ignored_settings(case)
        line(isinstance(ign, list), "foam check runs on a turbulence case", f"reports: {ign or 'nothing ignored'}")

        # 2. a short run
        try:
            wall = foam.run("simpleFoam", case, timeout=900)
            crashed = None
        except RuntimeError as e:
            wall, crashed = None, str(e)[-400:]
        log = case / "log.simpleFoam"
        steps = monitor.parse_log(log)
        hist = foam.residual_history(log, fields=("Ux", "Uy", "p", "nuTilda"))
        nres = len(hist.get("nuTilda", [])) if isinstance(hist, dict) else 0
        line(crashed is None, f"simpleFoam runs {ITER} iterations on NASA's coarsest grid with this untuned setup",
             crashed or f"{wall:.1f} s")
        line(nres > 0, "residual history includes the turbulence variable", f"{nres} values of nuTilda"
             + (f", last {hist['nuTilda'][-1]:.2e}; Ux {hist['Ux'][-1]:.2e}, p {hist['p'][-1]:.2e}" if nres else ""))
        try:
            a = monitor.assess(log, residual_targets=dict(Ux=1e-6, p=1e-6, nuTilda=1e-6))
            line(True, "monitor.assess accepts a residual target for the turbulence variable",
                 "; ".join(f"{c['name']}: {c['status']}" for c in a["checks"])[:300] if isinstance(a, dict) and "checks" in a else str(type(a)))
        except Exception as e:                                # noqa: BLE001
            line(False, "monitor.assess accepts a residual target for the turbulence variable", f"{type(e).__name__}: {e}")

        # 3. extraction
        times = foam.time_dirs(case)
        last = times[-1]
        try:
            tw = foam.read_boundary_field(last / "wallShearStress", "plate")
            xf = 0.5 * (x[0, plate[0]:-1] + x[0, plate[0] + 1:])
            cf = np.abs(tw[:, 0]) / (0.5 * U * U)
            k = int(np.argmin(np.abs(xf - 0.97)))
            line(tw.shape == (len(xf), 3), "wall shear stress on the plate can be read face by face (Cf along the plate)",
                 f"{len(xf)} faces; Cf near x = {xf[k]:.2f} after {ITER} iterations: {cf[k]:.4e} (not converged, not a result)")
        except Exception as e:                                # noqa: BLE001
            line(False, "wall shear stress on the plate can be read face by face (Cf along the plate)", f"{type(e).__name__}: {e}")
        try:
            yp = foam.read_boundary_field(last / "yPlus", "plate")
            line(True, "y+ on the plate can be read", f"min {yp.min():.3f}, mean {yp.mean():.3f}, max {yp.max():.3f}")
        except Exception as e:                                # noqa: BLE001
            line(False, "y+ on the plate can be read", f"{type(e).__name__}: {e}")
        try:
            ux = foam.read_internal_field(last / "U")[:, 0].reshape(info["nj"], info["ni"])
            nut = foam.read_internal_field(last / "nut").reshape(info["nj"], info["ni"])
            xc = 0.5 * (x[0, 1:] + x[0, :-1])
            i = int(np.argmin(np.abs(xc - 0.97)))
            line(ux.shape == (info["nj"], info["ni"]) and ux[0, i] < ux[-1, i],
                 "cell values reshape to (nj, ni): a velocity profile is one column, no searching",
                 f"column at x = {xc[i]:.3f}: u from {ux[0, i]:.3f} at the wall cell to {ux[-1, i]:.2f} at the top; "
                 f"largest nut/nu in it {nut[:, i].max() / NU:.1f}")
        except Exception as e:                                # noqa: BLE001
            line(False, "cell values reshape to (nj, ni): a velocity profile is one column, no searching", f"{type(e).__name__}: {e}")
        coeff = sorted((case / "postProcessing/forces").rglob("coefficient*.dat"))
        try:
            rows = foam.read_dat(coeff[-1])
            head = [h for h in coeff[-1].read_text().splitlines() if h.startswith("#")][-1].split()[1:]
            cd = rows[-1][head.index("Cd")]
            line(len(rows) >= ITER - 1, "drag coefficient history is written and read (forceCoeffs)",
                 f"{len(rows)} rows; Cd after {ITER} iterations {cd:.4e} (not converged, not a result)")
        except Exception as e:                                # noqa: BLE001
            line(False, "drag coefficient history is written and read (forceCoeffs)", f"{type(e).__name__}: {e}")
        print(f"\n{len(GAPS)} gap(s)" + (": " + "; ".join(GAPS) if GAPS else ""))
    finally:
        foam.stop(case)
        shutil.rmtree(case, ignore_errors=True)
    return 1 if GAPS else 0


if __name__ == "__main__":
    sys.exit(main())
