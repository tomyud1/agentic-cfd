"""Smoke test of the OGL GPU plugin: the icoFoam cavity on a fine 2D mesh, pressure solved on the CPU
(PCG/DIC and GAMG) and on the GPU (Ginkgo CG with block Jacobi, executor cuda). Checks that the GPU run
loads, converges and gives the same fields, and prints time per step and pressure iterations.

    python setup/gpu/test_ogl.py [N]        (inside WSL from the project folder; N cells per side, default 400)
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from cfdlib import foam

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
STEPS = 20
root = Path(foam.foam_env("FOAM_RUN")) / "ogl_test"
tut = Path(foam.foam_env("FOAM_TUTORIALS")) / "incompressible/icoFoam/cavity/cavity"
dt = 0.5 * (0.1 / N) / 1.0                                   # Courant number 0.5 at the lid (U = 1, side 0.1 m)

# OGL needs -parallel (at least 2 ranks: OpenFOAM refuses -parallel on 1) and, because Ubuntu's Open MPI
# is not CUDA-aware, forceHostBuffer true (MPI data via host memory; without it: segfault in MPI_Alltoallv).
GKO = ("solver GKOCG; preconditioner BJ; executor cuda; tolerance 1e-06; relTol 0; verbose 1; "
       "forceHostBuffer true; ranksPerGPU {n};")
RUNS = {                                    # name: (pressure solver, MPI ranks or 0 for serial)
    "cpu_PCG": ("solver PCG; preconditioner DIC; tolerance 1e-06; relTol 0;", 0),
    "cpu_GAMG": ("solver GAMG; smoother GaussSeidel; tolerance 1e-06; relTol 0;", 0),
    "cpu_GAMG_np8": ("solver GAMG; smoother GaussSeidel; tolerance 1e-06; relTol 0;", 8),
    "gpu_CG_BJ_np2": (GKO.format(n=2), 2),
    "gpu_CG_none_np2": (GKO.format(n=2).replace("preconditioner BJ", "preconditioner none"), 2),
    "gpu_CG_BJ_neg_np2": (GKO.format(n=2) + " scaling -1;", 2),
    "gpu_CG_MG_neg_np2": (GKO.format(n=2).replace("preconditioner BJ", "preconditioner Multigrid") + " scaling -1;", 2),
    "gpu_CG_BJ_np8": (GKO.format(n=8), 8),
}
if len(sys.argv) > 2:                       # optional: only the named runs (plus cpu_PCG as the reference)
    RUNS = {k: v for k, v in RUNS.items() if k in sys.argv[2:] or k == "cpu_PCG"}


def setup(name, psolver, ranks):
    case = root / name
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(tut, case)
    bm = (case / "system/blockMeshDict").read_text()
    (case / "system/blockMeshDict").write_text(bm.replace("(20 20 1)", f"({N} {N} 1)"))
    cd = (case / "system/controlDict").read_text()
    cd = re.sub(r"deltaT\s+[^;]+;", f"deltaT {dt:g};", cd)
    cd = re.sub(r"endTime\s+[^;]+;", f"endTime {STEPS * dt:g};", cd)
    cd = re.sub(r"writeInterval\s+[^;]+;", f"writeInterval {STEPS};", cd)
    cd = re.sub(r"writeControl\s+[^;]+;", "writeControl timeStep;", cd)
    cd = re.sub(r"writePrecision\s+[^;]+;", "writePrecision 12;", cd)
    if name.startswith("gpu"):
        cd += '\nlibs ("libOGL.so");\n'
    (case / "system/controlDict").write_text(cd)
    fs = (case / "system/fvSolution").read_text()
    fs = re.sub(r"(?s)\bp\s*\{.*?\}", "p { " + psolver + " }", fs, count=1)
    fs = re.sub(r"(?s)\bpFinal\s*\{.*?\}", "pFinal { $p; relTol 0; }", fs, count=1)
    (case / "system/fvSolution").write_text(fs)
    foam.run("blockMesh", case)
    if ranks:
        (case / "system/decomposeParDict").write_text(
            "FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\n"
            f"numberOfSubdomains {ranks};\nmethod {'scotch' if ranks > 1 else 'simple'};\n"
            "coeffs { n (1 1 1); }\n")
        foam.run("decomposePar -force", case)
    return case


res = {}
for name, (ps, ranks) in RUNS.items():
    case = setup(name, ps, ranks)
    try:
        if ranks:
            foam.run(f"mpirun --use-hwthread-cpus -np {ranks} icoFoam -parallel", case, log_name="icoFoam", timeout=1800)
            foam.run("reconstructPar -latestTime", case)
        else:
            foam.run("icoFoam", case, timeout=1800)
        ok = True
    except Exception as e:                                   # noqa: BLE001 - a failure is a finding
        ok = False
        print(name, "FAILED:", e)
    log = (case / "log.icoFoam").read_text(errors="replace")
    it = [int(x) for x in re.findall(r"Solving for p, Initial residual = [^,]+, Final residual = [^,]+, No Iterations (\d+)", log)]
    ex = [float(x) for x in re.findall(r"^ExecutionTime = ([\d.eE+-]+) s", log, re.M)]
    ogl = re.findall(r"\[OGL LOG\][^\n]*", log)
    p = foam.read_internal_field(foam.time_dirs(case)[-1] / "p") if ok else None
    res[name] = dict(ok=ok, p=p, iters=np.mean(it) if it else float("nan"),
                     s_per_step=(ex[-1] - ex[0]) / (len(ex) - 1) if len(ex) > 1 else float("nan"), ogl=ogl[-4:])
    print(f"{name:14s} ok={ok} mean p iterations {res[name]['iters']:.1f}  wall s/step (after step 1) {res[name]['s_per_step']:.4f}")
    for line in res[name]["ogl"]:
        print("   ", line[:160])

ref = res["cpu_PCG"]["p"]
for name, r in res.items():
    if r["p"] is not None and ref is not None:
        d = np.max(np.abs(np.asarray(r["p"]) - np.asarray(ref))) / np.max(np.abs(ref))
        print(f"{name:14s} max |p - p_PCG| / max|p| = {d:.2e}")
print(f"cells {N * N}, {STEPS} steps, dt {dt:g}")
