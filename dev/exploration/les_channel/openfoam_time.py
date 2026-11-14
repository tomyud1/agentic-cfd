"""OpenFOAM's cost per time step on the channel grids of predict.py, for the speed comparison.

    PYTHONPATH=. python dev/exploration/les_channel/openfoam_time.py [nx ny nz] [steps] [cores, e.g. 8,4]
    (inside WSL, needs OpenFOAM and the developed field stored by `predict.py run`)

Same flow, same cells (blockMesh's points moved onto gpucfd's wall spacing), same time step, started from
gpucfd's developed turbulent field put on the cell centres. Solver settings are those of OpenFOAM's own
channel tutorial (pimpleFoam/LES/periodicPlaneChannel): second-order schemes, PISO with two pressure
corrections, multigrid for the pressure. No turbulence model, as in gpucfd's run. Timed on 1, 4 and 8
cores; seconds per step from the solver's own clock, first 10 steps left out.
This is a cost measurement, not a solution: the run is far too short for statistics.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import predict as P  # noqa: E402

from cfdlib import foam, monitor  # noqa: E402

HEAD = "FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}\n"
CYCLIC = "{a} {{ type cyclic; neighbourPatch {b}; faces ({f}); }}\n"


def cell_field(src, n, y):
    """gpucfd's stored field on the cell centres of the grid n (velocity components, shape (nz, ny, nx))."""
    xc = [(np.arange(n[0]) + 0.5) * P.LX / n[0], 0.5 * (y[:-1] + y[1:]), (np.arange(n[2]) + 0.5) * P.LZ / n[2]]
    out = []
    for c in range(3):
        a = src[f"q{c}"]
        for d, axis in ((0, 2), (1, 1), (2, 0)):
            if d == 1:
                a = P.interp_axis(a, src[f"x{c}{d}"], xc[d], axis, ends=(0.0, P.LY))
            else:
                a = P.interp_axis(a, src[f"x{c}{d}"], xc[d], axis, period=(P.LX, P.LZ)[d // 2])
        out.append(a)
    return out


def build(case, n, y, dt, steps, src):
    shutil.rmtree(case, ignore_errors=True)
    for sub in ("system", "constant", "0"):
        (case / sub).mkdir(parents=True)
    nx, ny, nz = n
    (case / "system/blockMeshDict").write_text(HEAD.format(cls="dictionary", obj="blockMeshDict") + f"""
scale 1;
vertices ((0 0 0) ({P.LX!r} 0 0) ({P.LX!r} 2 0) (0 2 0) (0 0 {P.LZ!r}) ({P.LX!r} 0 {P.LZ!r}) ({P.LX!r} 2 {P.LZ!r}) (0 2 {P.LZ!r}));
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1));
edges ();
boundary
(
    bottomWall {{ type wall; faces ((0 1 5 4)); }}
    topWall {{ type wall; faces ((3 7 6 2)); }}
    {CYCLIC.format(a="inlet", b="outlet", f="(0 4 7 3)")}    {CYCLIC.format(a="outlet", b="inlet", f="(1 2 6 5)")}
    {CYCLIC.format(a="front", b="back", f="(0 3 2 1)")}    {CYCLIC.format(a="back", b="front", f="(4 5 6 7)")}
);
""")
    (case / "system/controlDict").write_text(HEAD.format(cls="dictionary", obj="controlDict") + f"""
application pimpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime {steps * dt!r}; deltaT {dt!r};
writeControl timeStep; writeInterval 1000000; writeFormat ascii; writePrecision 16; timeFormat general; timePrecision 10;
runTimeModifiable false;
""")
    (case / "system/fvSchemes").write_text(HEAD.format(cls="dictionary", obj="fvSchemes") + """
ddtSchemes { default backward; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) Gauss linear; div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
    (case / "system/fvSolution").write_text(HEAD.format(cls="dictionary", obj="fvSolution") + """
solvers
{
    p { solver GAMG; tolerance 0; relTol 0.1; smoother GaussSeidel; }
    pFinal { solver GAMG; smoother DICGaussSeidel; tolerance 1e-06; relTol 0; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0.1; }
    UFinal { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0; }
}
PIMPLE { nOuterCorrectors 1; nCorrectors 2; nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }
""")
    (case / "constant/transportProperties").write_text(
        HEAD.format(cls="dictionary", obj="transportProperties") + f"transportModel Newtonian;\nnu {P.NU!r};\n")
    (case / "constant/turbulenceProperties").write_text(
        HEAD.format(cls="dictionary", obj="turbulenceProperties") + "simulationType laminar;\n")
    (case / "constant/fvOptions").write_text(HEAD.format(cls="dictionary", obj="fvOptions") + """
drive { type vectorSemiImplicitSource; selectionMode all; volumeMode specific; sources { U ((1 0 0) 0); } }
""")
    foam.run("blockMesh", case)
    # move the uniform points onto gpucfd's wall-normal faces (blockMesh's grading is geometric, not tanh)
    pts = case / "constant/polyMesh/points"
    lines = pts.read_text().split("\n")
    k = next(i for i, ln in enumerate(lines) if ln.strip().isdigit() and lines[i + 1].strip() == "(")
    npts = int(lines[k])
    xyz = np.array([ln.strip()[1:-1].split() for ln in lines[k + 2:k + 2 + npts]], dtype=float)
    j = np.rint(xyz[:, 1] / (P.LY / ny)).astype(int)
    assert np.abs(xyz[:, 1] - j * P.LY / ny).max() < 1e-9 and j.min() == 0 and j.max() == ny
    xyz[:, 1] = y[j]
    lines[k + 2:k + 2 + npts] = [f"({a!r} {b!r} {c!r})" for a, b, c in xyz.tolist()]
    pts.write_text("\n".join(lines))
    cyc = "".join(f"    {p} {{ type cyclic; }}\n" for p in ("inlet", "outlet", "front", "back"))
    u, v, w = cell_field(src, n, y)
    vals = np.column_stack([u.ravel(), v.ravel(), w.ravel()])          # cell number = i + nx (j + ny k)
    body = "\n".join(f"({a:.12g} {b:.12g} {c:.12g})" for a, b, c in vals.tolist())
    (case / "0/U").write_text(HEAD.format(cls="volVectorField", obj="U") + "dimensions [0 1 -1 0 0 0 0];\n"
                              f"internalField nonuniform List<vector> {len(vals)}\n(\n{body}\n);\nboundaryField\n{{\n"
                              "    bottomWall { type noSlip; }\n    topWall { type noSlip; }\n" + cyc + "}\n")
    (case / "0/p").write_text(HEAD.format(cls="volScalarField", obj="p") + "dimensions [0 2 -2 0 0 0 0];\n"
                              "internalField uniform 0;\nboundaryField\n{\n    bottomWall { type zeroGradient; }\n"
                              "    topWall { type zeroGradient; }\n" + cyc + "}\n")
    return float((u.mean(axis=(0, 2)) * np.diff(y)).sum() / P.LY)


def timed(case, cores, steps):
    if cores > 1:
        split = {4: "(2 1 2)", 8: "(2 2 2)", 2: "(2 1 1)"}[cores]
        (case / "system/decomposeParDict").write_text(HEAD.format(cls="dictionary", obj="decomposeParDict") +
                                                      f"numberOfSubdomains {cores};\nmethod simple;\ncoeffs {{ n {split}; }}\n")
        foam.run("decomposePar -force", case, "decomposePar")
        foam.run(f"mpirun -np {cores} pimpleFoam -parallel", case, "pimpleFoam", timeout=3600)
    else:
        foam.run("pimpleFoam", case, timeout=3600)
    arr = monitor.log_arrays(monitor.parse_log(case / "log.pimpleFoam")["steps"])
    ex = arr["exec"]
    assert len(ex) >= steps, (len(ex), steps)
    return dict(cores=cores, s_per_step=float((ex[-1] - ex[9]) / (len(ex) - 10)), courant_max=float(np.nanmax(arr["co"])),
                steps=len(ex))


if __name__ == "__main__":
    n = tuple(int(v) for v in sys.argv[1:4]) if len(sys.argv) >= 4 else (64, 64, 64)
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    d = P.load()
    key = f"{n[0]}x{n[1]}x{n[2]}"
    dt = d[f"time_{key}_imex_1e-08"]["dt"] if f"time_{key}_imex_1e-08" in d else d[f"run_{key}_imex"]["dt_mean_window"]
    src = np.load(P.FIELD.with_name("les_channel_field_64x64x64.npz"))
    y = P.wall_grid(n[1], 1.0)
    case = Path(foam.foam_env("FOAM_RUN")) / "agentic" / "les_channel" / f"of_{key}"
    bulk = build(case, n, y, dt, steps, src)
    print(f"{key}: case {case}, dt {dt:.3e} (gpucfd's), bulk velocity of the start field {bulk:.3f}", flush=True)
    rows = []
    for cores in ([int(c) for c in sys.argv[5].split(",")] if len(sys.argv) > 5 else (8, 4, 1)):
        r = timed(case, cores, steps)
        rows.append(r)
        print(f"  {cores} core(s): {r['s_per_step'] * 1e3:.0f} ms per step, largest Courant number {r['courant_max']:.2f}", flush=True)
    best = min(rows, key=lambda r: r["s_per_step"])
    d[f"openfoam_{key}"] = dict(grid=P.describe(n, y), dt=dt, rows=rows, best=best,
                                settings="pimpleFoam, laminar (no model), backward, linear, PISO 2 corrections, GAMG "
                                         "(tutorial periodicPlaneChannel), v" + foam.foam_env("WM_PROJECT_VERSION"))
    P.save(d)
