"""Like-for-like comparison of gpucfd (GPU) and OpenFOAM (CPU) on the same flow.

The flow: decaying Arnold-Beltrami-Childress (ABC) flow in a periodic box [0, 2 pi]^3,
    u = (A sin z + C cos y, B sin x + A cos z, C sin y + B cos x) exp(-nu t),
an exact solution of the incompressible Navier-Stokes equations (its convection term is exactly a
pressure gradient). Same box, same uniform grids, same viscosity, same end time on both sides.

Rules (fixed before running, see gpucfd/DESIGN.md):
1. Same answer: both solvers are judged by their error against the exact solution at the end time.
2. Each solver uses its best settings: the cheapest time step / tolerances whose error is within 10 %
   of its time-converged error on that grid ("tune" step), OpenFOAM on its best rank count (8 or 16).
3. Time = the solver's own time-stepping loop (OpenFOAM: profiling of time.run(); GPU: CUDA events),
   both excluding start-up, mesh generation and file output. Total wall times are recorded as well.
4. The comparison is cost at equal accuracy: error against time on each grid.

    python -m gpucfd.compare.abc tune        # choose each solver's settings (small grids)
    python -m gpucfd.compare.abc run         # the comparison on 32^3, 64^3, 128^3
(inside WSL from the project root; results in gpucfd/compare/results/abc.json)
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from cfdlib import foam, perf
from gpucfd import gcfd

A, B, C = 1.0, 0.7, 0.4
NU = 0.01
T_END = 1.0
L = 2 * np.pi
UMAX = 2.0                      # bound on |u| for time-step sizing (max |u| of this ABC flow is about 1.9)
ROOT = Path(foam.foam_env("FOAM_RUN")) / "gpucfd_compare"
OUT = Path(__file__).parent / "results"


def abc(X, Y, Z, t):
    f = np.exp(-NU * t)
    return (A * np.sin(Z) + C * np.cos(Y)) * f, (B * np.sin(X) + A * np.cos(Z)) * f, (C * np.sin(Y) + B * np.cos(X)) * f


# ------------------------------------------------------------------------------------------ GPU
def gpu(N, cfl, ptol, pressure="mixed"):
    h = L / N
    dt_target = min(cfl * h / UMAX, 0.15 * h * h / NU)      # convective and (explicit) diffusive limits
    steps = int(np.ceil(T_END / dt_target))
    t0 = time.time()
    s = gcfd.NavierStokes((N, N, N), (L, L, L), NU, pressure)
    q0 = []
    for comp in range(3):
        X, Y, Z = s.face_coords(comp)
        q0.append(abc(X, Y, Z, 0.0)[comp])
    s.set(q0)
    st = s.run(T_END / steps, steps, ptol)
    q = s.get()
    err = 0.0
    for comp in range(3):
        X, Y, Z = s.face_coords(comp)
        err = max(err, float(np.max(np.abs(q[comp] - abc(X, Y, Z, T_END)[comp]))))
    s.close()
    return dict(solver="gpucfd", N=N, cells=N ** 3, cfl=cfl, ptol=ptol, steps=steps, err=err, loop_s=st["ms"] / 1e3,
                wall_s=time.time() - t0, max_div=st["max_div"], p_iters=st["poisson_its"])


# ------------------------------------------------------------------------------------------ OpenFOAM
HEADER = "FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}\n"


def write_case(case, N, dt, steps, ncorr, nouter, ptol, ranks, utol=1e-10):
    if case.exists():
        shutil.rmtree(case)
    (case / "0").mkdir(parents=True)
    (case / "constant").mkdir()
    (case / "system").mkdir()
    pairs = [("xmin", "xmax", "(0 4 7 3)", "(1 2 6 5)"), ("ymin", "ymax", "(0 1 5 4)", "(3 7 6 2)"),
             ("zmin", "zmax", "(0 3 2 1)", "(4 5 6 7)")]
    bnd = ""
    for a, b, fa, fb in pairs:
        bnd += f"  {a} {{ type cyclic; neighbourPatch {b}; faces ({fa}); }}\n"
        bnd += f"  {b} {{ type cyclic; neighbourPatch {a}; faces ({fb}); }}\n"
    (case / "system/blockMeshDict").write_text(
        HEADER.format(cls="dictionary", obj="blockMeshDict") + f"scale 1;\nvertices ((0 0 0) ({L!r} 0 0) ({L!r} {L!r} 0) "
        f"(0 {L!r} 0) (0 0 {L!r}) ({L!r} 0 {L!r}) ({L!r} {L!r} {L!r}) (0 {L!r} {L!r}));\n"
        f"blocks (hex (0 1 2 3 4 5 6 7) ({N} {N} {N}) simpleGrading (1 1 1));\nboundary\n(\n{bnd});\n")
    (case / "system/controlDict").write_text(
        HEADER.format(cls="dictionary", obj="controlDict") +
        f"application pimpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime {T_END};\n"
        f"deltaT {T_END / steps!r};\nadjustTimeStep no;\nwriteControl timeStep;\nwriteInterval {steps};\n"
        "writeFormat ascii;\nwritePrecision 17;\ntimeFormat general;\ntimePrecision 12;\nrunTimeModifiable false;\n"
        "profiling { active true; cpuInfo false; memInfo false; sysInfo false; }\n")
    (case / "system/fvSchemes").write_text(
        HEADER.format(cls="dictionary", obj="fvSchemes") +
        "ddtSchemes { default backward; }\ngradSchemes { default Gauss linear; }\n"
        "divSchemes { default none; div(phi,U) Gauss linear; div((nuEff*dev2(T(grad(U))))) Gauss linear; }\n"
        "laplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\n"
        "snGradSchemes { default corrected; }\n")
    gamg = f"solver GAMG; smoother DICGaussSeidel; tolerance {ptol}; relTol 0;"
    (case / "system/fvSolution").write_text(
        HEADER.format(cls="dictionary", obj="fvSolution") +
        f"solvers\n{{\n  p {{ {gamg} }}\n  pFinal {{ {gamg} }}\n"
        f"  \"(U|UFinal)\" {{ solver smoothSolver; smoother symGaussSeidel; tolerance {utol}; relTol 0; }}\n}}\n"
        f"PIMPLE {{ momentumPredictor yes; nOuterCorrectors {nouter}; nCorrectors {ncorr}; "
        "nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }\n")
    (case / "constant/transportProperties").write_text(
        HEADER.format(cls="dictionary", obj="transportProperties") + f"transportModel Newtonian;\nnu {NU!r};\n")
    (case / "constant/turbulenceProperties").write_text(
        HEADER.format(cls="dictionary", obj="turbulenceProperties") + "simulationType laminar;\n")
    xc = (np.arange(N) + 0.5) * L / N                     # blockMesh cell order: x fastest, then y, then z
    Z, Y, X = np.meshgrid(xc, xc, xc, indexing="ij")
    u, v, w = (a.ravel() for a in abc(X, Y, Z, 0.0))
    body = "\n".join(f"({a:.12g} {b:.12g} {c:.12g})" for a, b, c in zip(u, v, w))
    cyc = "boundaryField { \".*\" { type cyclic; } }\n"
    (case / "0/U").write_text(HEADER.format(cls="volVectorField", obj="U") + "dimensions [0 1 -1 0 0 0 0];\n"
                              f"internalField nonuniform List<vector>\n{N ** 3}\n(\n{body}\n);\n" + cyc)
    (case / "0/p").write_text(HEADER.format(cls="volScalarField", obj="p") + "dimensions [0 2 -2 0 0 0 0];\n"
                              "internalField uniform 0;\n" + cyc)
    if ranks > 1:
        (case / "system/decomposeParDict").write_text(
            HEADER.format(cls="dictionary", obj="decomposeParDict") + f"numberOfSubdomains {ranks};\nmethod scotch;\n")


def openfoam(N, co, ncorr=2, nouter=1, ptol=1e-8, ranks=8, utol=1e-10):
    h = L / N
    steps = int(np.ceil(T_END / (co * h / UMAX)))
    case = ROOT / f"abc_N{N}_co{co}_c{ncorr}_o{nouter}_p{ptol:g}_u{utol:g}_r{ranks}"
    t0 = time.time()
    write_case(case, N, T_END / steps, steps, ncorr, nouter, ptol, ranks, utol)
    foam.run("blockMesh", case)
    if ranks > 1:
        foam.run("decomposePar -force", case)
        hw = " --use-hwthread-cpus" if ranks > 8 else ""
        foam.run(f"mpirun{hw} -np {ranks} pimpleFoam -parallel", case, log_name="pimpleFoam", timeout=7200)
        foam.run("reconstructPar -latestTime", case)
    else:
        foam.run("pimpleFoam", case, timeout=7200)
    wall = time.time() - t0
    prof = perf.profiling(case, ranks)
    loop = next(v["total"] for k, v in prof.items() if k.startswith("time.run()"))
    psolve = prof.get("fvMatrix::solve.p", {}).get("total")
    usolve = prof.get("fvMatrix::solve.U", {}).get("total")
    U = np.asarray(foam.read_internal_field(foam.time_dirs(case)[-1] / "U"))
    xc = (np.arange(N) + 0.5) * L / N
    Z, Y, X = np.meshgrid(xc, xc, xc, indexing="ij")
    ex = np.stack([a.ravel() for a in abc(X, Y, Z, T_END)], axis=1)
    err = float(np.max(np.abs(U - ex)))
    log = (case / "log.pimpleFoam").read_text(errors="replace")
    pit = [int(x) for x in __import__("re").findall(r"Solving for p, Initial residual = [^,]+, Final residual = [^,]+, No Iterations (\d+)", log)]
    shutil.rmtree(case, ignore_errors=True)                 # cases are large; the numbers are kept below
    return dict(solver="OpenFOAM", N=N, cells=N ** 3, co=co, ncorr=ncorr, nouter=nouter, ptol=ptol, ranks=ranks,
                utol=utol, steps=steps, err=err, loop_s=loop, p_solve_s=psolve, U_solve_s=usolve, wall_s=wall,
                p_iters=float(np.mean(pit)) if pit else None)


def save(name, rows):
    OUT.mkdir(exist_ok=True)
    path = OUT / "abc.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = rows
    path.write_text(json.dumps(data, indent=1))


def show(r):
    extra = (f"Co {r['co']} corr {r['ncorr']} outer {r['nouter']} ranks {r['ranks']}" if r["solver"] == "OpenFOAM"
             else f"CFL {r['cfl']} ptol {r['ptol']:g}")
    print(f"{r['solver']:8s} N={r['N']:3d} {extra:32s} steps {r['steps']:4d}  err {r['err']:.3e}  "
          f"loop {r['loop_s']:8.2f} s  wall {r['wall_s']:7.1f} s  p its {r['p_iters']}", flush=True)
    return r


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "tune"
    if what == "tune":
        rows = []
        # time-step / tolerance sensitivity on 64^3: the error must be space-dominated
        for co in (0.8, 0.4, 0.2, 0.1):
            rows.append(show(openfoam(64, co)))
        rows.append(show(openfoam(64, 0.4, ncorr=3)))
        rows.append(show(openfoam(64, 0.4, nouter=2)))
        rows.append(show(openfoam(64, 0.4, ptol=1e-6)))
        for cfl in (0.8, 0.4, 0.2):
            rows.append(show(gpu(64, cfl, 1e-10)))
        for ptol in (1e-6, 1e-4):
            rows.append(show(gpu(64, 0.8, ptol)))
        save("tune", rows)
    elif what == "tune2":
        # OpenFOAM gets the same chance as the GPU: looser pressure tolerance, more correctors, rank counts
        rows = []
        for co, nc, pt, r in ((0.4, 3, 1e-6, 8), (0.4, 3, 1e-4, 8), (0.2, 2, 1e-6, 8), (0.2, 2, 1e-4, 8),
                              (0.4, 3, 1e-6, 4), (0.4, 3, 1e-6, 16)):
            rows.append(show(openfoam(64, co, ncorr=nc, ptol=pt, ranks=r)))
        save("tune2", rows)
    elif what == "tune3":
        rows = [show(openfoam(64, 0.4, ncorr=3, ptol=1e-4, ranks=8, utol=u)) for u in (1e-6, 1e-5, 1e-4)]
        for r in rows:
            print(r["N"], r["utol"], "U solve", r["U_solve_s"], "p solve", r["p_solve_s"], "loop", r["loop_s"])
        save("tune3", rows)
    elif what == "ranks":
        rows = [show(openfoam(128, float(sys.argv[2]), ranks=r)) for r in (8, 16)]
        save("ranks", rows)
    elif what == "run":
        # best settings from the tuning runs (python -m gpucfd.compare.abc tune / tune2), fixed before this run:
        # OpenFOAM Co 0.4, 3 pressure correctors, p tolerance 1e-4, 8 ranks; GPU CFL 0.8, ptol 1e-4.
        rows = []
        for N in (32, 64, 128):
            rows.append(show(openfoam(N, 0.4, ncorr=3, ptol=1e-4, ranks=8)))
            rows.append(show(gpu(N, 0.8, 1e-4)))
            rows.append(show(gpu(N, 0.8, 1e-10)))           # strict pressure tolerance, for reference
        rows.append(show(gpu(256, 0.8, 1e-4)))              # beyond what fits OpenFOAM comfortably in memory
        save("run", rows)
