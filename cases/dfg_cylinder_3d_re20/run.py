"""Build and run the DFG 3D-1Z cylinder case (Re = 20, steady) on the mesh levels in numerics.yaml; save raw outputs.

Usage (repo root, inside WSL):
  python -m cases.dfg_cylinder_3d_re20.run mesh L1 [L2 ...]            # build + checkMesh only (inspect)
  python -m cases.dfg_cylinder_3d_re20.run preflight                     # short runs on L1 (1 core) and L4 (16 ranks)
  python -m cases.dfg_cylinder_3d_re20.run production L1 [L2 ...]        # converged runs, each mapped from the previous level
  python -m cases.dfg_cylinder_3d_re20.run gpu_full NAME LEVEL CONFIG RANKS SOURCE   # converged run with any p solver config
  python -m cases.dfg_cylinder_3d_re20.run timing MESH CONFIG RANKS ITERS [TAG]      # fixed-iteration timing run
  python -m cases.dfg_cylinder_3d_re20.run tune MESH RANKS ITERS CONFIG [CONFIG ...]  # GPU solver tuning (iterations, times)
  python -m cases.dfg_cylinder_3d_re20.run flow LEVEL                    # mid-plane velocity of a finished run -> npz
Outputs: results/dfg_cylinder_3d_re20/{preflight.json, runs/<name>.json + .npz, perf/<name>.json}
Pressure solver configurations (CPU GAMG and GPU OGL variants) are the CONFIGS table below.
"""
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cfdlib import foam, monitor, perf

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS_DIR = OUT / "runs"
PERF_DIR = OUT / "perf"
TIMING_DIR = OUT / "timing"


def load():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    return problem, numerics


def run_root():
    return Path(foam.foam_env("FOAM_RUN")) / "agentic" / HERE.name


# ---------------------------------------------------------------------------------------------
# Mesh: O-grid around the cylinder inside a square box, 8 rectangular blocks around it, extruded in z
# ---------------------------------------------------------------------------------------------
def geometric_first(length, n, ratio):
    """First cell of n cells over `length` with last/first = ratio (blockMesh simpleGrading)."""
    if abs(ratio - 1) < 1e-12 or n == 1:
        return length / n
    r = ratio ** (1.0 / (n - 1))
    return length * (r - 1) / (r ** n - 1)


def scale_of(mesh, numerics):
    """Cell-count multiplier of a mesh name relative to L1 (L1..L4: 1, 2, 4, 8; extras: nc / nc_L1)."""
    mm = numerics["mesh"]
    if mesh in mm["levels"]:
        return 2 ** mm["levels"].index(mesh)
    return mm["extra"][mesh] / mm["nc_L1"]


def mesh_spec(mesh, problem, numerics):
    g, mm = problem["given"], numerics["mesh"]
    H, L, D, xc, yc = (g[k]["value"] for k in ("H", "L", "D", "xc", "yc"))
    R, b = D / 2, mm["box_half"]
    f = scale_of(mesh, numerics)

    def cnt(key, even=False):
        n = mm[key + "_L1"] * f
        n = int(round(n / 2) * 2) if even else int(round(n))
        return max(n, 2 if even else 1)

    nc, nr, nl, nd, nb, nt = (cnt(k) for k in ("nc", "nr", "nl", "nd", "nb", "nt"))
    nz = cnt("nz", even=True)
    X, Y = [0.0, xc - b, xc + b, L], [0.0, yc - b, yc + b, H]
    pts = [(X[i], Y[j]) for j in range(4) for i in range(4)]            # P(i, j) = 4 j + i
    s = R / math.sqrt(2)
    pts += [(xc - s, yc - s), (xc + s, yc - s), (xc + s, yc + s), (xc - s, yc + s)]   # SW SE NE NW = 16..19
    nv = len(pts)
    vertices = [(x, y, 0.0) for x, y in pts] + [(x, y, H) for x, y in pts]
    P = lambda i, j: 4 * j + i
    SW, SE, NE, NW = 16, 17, 18, 19
    Eu, Ed, Er, Ez = mm["upstream_ratio"], mm["downstream_ratio"], mm["radial_ratio"], mm["z_ratio"]

    def blk(name, quad, n, gr=(1, 1)):
        return dict(name=name, v=list(quad) + [q + nv for q in quad], n=n, g=gr)

    blocks = [
        blk("below-upstream", (P(0, 0), P(1, 0), P(1, 1), P(0, 1)), (nl, nb), (Eu, 1)),
        blk("below-box", (P(1, 0), P(2, 0), P(2, 1), P(1, 1)), (nc, nb)),
        blk("below-downstream", (P(2, 0), P(3, 0), P(3, 1), P(2, 1)), (nd, nb), (Ed, 1)),
        blk("upstream", (P(0, 1), P(1, 1), P(1, 2), P(0, 2)), (nl, nc), (Eu, 1)),
        blk("downstream", (P(2, 1), P(3, 1), P(3, 2), P(2, 2)), (nd, nc), (Ed, 1)),
        blk("above-upstream", (P(0, 2), P(1, 2), P(1, 3), P(0, 3)), (nl, nt), (Eu, 1)),
        blk("above-box", (P(1, 2), P(2, 2), P(2, 3), P(1, 3)), (nc, nt)),
        blk("above-downstream", (P(2, 2), P(3, 2), P(3, 3), P(2, 3)), (nd, nt), (Ed, 1)),
        # O-grid: local x along the box side, local y radially inwards (cells shrink towards the cylinder)
        blk("O-south", (P(1, 1), P(2, 1), SE, SW), (nc, nr), (1, 1 / Er)),
        blk("O-east", (P(2, 1), P(2, 2), NE, SE), (nc, nr), (1, 1 / Er)),
        blk("O-north", (P(2, 2), P(1, 2), NW, NE), (nc, nr), (1, 1 / Er)),
        blk("O-west", (P(1, 2), P(1, 1), SW, NW), (nc, nr), (1, 1 / Er)),
    ]
    arcs = []
    for a, b_, (mx, my) in ((SW, SE, (xc, yc - R)), (SE, NE, (xc + R, yc)), (NE, NW, (xc, yc + R)), (NW, SW, (xc - R, yc))):
        arcs += [(a, b_, (mx, my, 0.0)), (a + nv, b_ + nv, (mx, my, H))]

    def side(a, b_):
        return [a, b_, b_ + nv, a + nv]

    patches = {
        "inlet": dict(type="patch", faces=[side(P(0, j + 1), P(0, j)) for j in range(3)]),
        "outlet": dict(type="patch", faces=[side(P(3, j), P(3, j + 1)) for j in range(3)]),
        # all four channel walls: y = 0, y = H, and the side walls z = 0, z = H of every block
        "walls": dict(type="wall", faces=[side(P(i, 0), P(i + 1, 0)) for i in range(3)]
                                         + [side(P(i + 1, 3), P(i, 3)) for i in range(3)]
                                         + [[bk["v"][0], bk["v"][3], bk["v"][2], bk["v"][1]] for bk in blocks]
                                         + [bk["v"][4:] for bk in blocks]),
        "cylinder": dict(type="wall", faces=[side(SE, SW), side(NE, SE), side(NW, NE), side(SW, NW)]),
    }
    cells = sum(bk["n"][0] * bk["n"][1] for bk in blocks) * nz
    h_rad = geometric_first(yc + b - yc - R, nr, 1 / Er)
    r_ = (1 / Er) ** (1 / (nr - 1)) if nr > 1 else 1.0
    h_wall = h_rad * r_ ** (nr - 1)                       # radial size of the cells on the cylinder
    hz_wall = geometric_first(H / 2, nz // 2, Ez)         # z size of the cells at the side walls
    zg = f"((0.5 0.5 {Ez:g}) (0.5 0.5 {1 / Ez:.12g}))"
    return dict(mesh=mesh, nc=nc, nr=nr, nl=nl, nd=nd, nb=nb, nt=nt, nz=nz, cells=cells, cells2d=cells // nz,
                vertices=vertices, blocks=blocks, arcs=arcs, patches=patches, z_grading=zg,
                h=dict(surface=(math.pi * D / 4) / nc, wall=h_wall, box=2 * b / nc, z_wall=hz_wall,
                       z_mid=hz_wall * Ez, outlet=geometric_first(L - xc - b, nd, Ed) * Ed))


# ---------------------------------------------------------------------------------------------
# Pressure linear-solver configurations: CPU (GAMG) and GPU (OGL / Ginkgo)
# ---------------------------------------------------------------------------------------------
GPU_COMMON = dict(executor="cuda", scaling=-1, forceHostBuffer="true", verbose=1)
CONFIGS = {
    # CPU baseline: GAMG with Gauss-Seidel (the benchmark paper's OpenFOAM pressure solver)
    "GAMG": dict(solver="GAMG", smoother="GaussSeidel"),
    "GAMG_DIC": dict(solver="GAMG", smoother="DICGaussSeidel"),
    "PCG_DIC": dict(solver="PCG", preconditioner="DIC"),
    # GPU: Ginkgo CG; preconditioner settings go into a sub-dictionary (see OGL include/OGL/Preconditioner/*.hpp)
    "gpu_MG_default": dict(solver="GKOCG", **GPU_COMMON, matrixFormat="Coo",
                           preconditioner=dict(preconditioner="Multigrid")),
}


def _gpu_mg(name, fmt="Coo", caching=None, adapt="false", **mg):
    """OGL CG + multigrid variant. adaptMinIter (OGL default true) forces every solve to do at least 0.6 times the
    iterations of the previous one and checks the residual only every few iterations: switched off unless stated."""
    pre = dict(preconditioner="Multigrid", **mg)
    if caching is not None:
        pre["caching"] = caching
    CONFIGS[name] = dict(solver="GKOCG", **GPU_COMMON, matrixFormat=fmt, adaptMinIter=adapt, preconditioner=pre)


# Tuning candidates (time-boxed): coarse level size and solver, smoother strength, hierarchy reuse, caching
_gpu_mg("gpu_MG_noadapt")
_gpu_mg("gpu_MG_c1k", minCoarseRows=1000)
_gpu_mg("gpu_MG_c1k_cg_adapt", adapt="true", minCoarseRows=1000, coarseSolver="CG", maxIterCoarse=20, relTolCoarse=0.01)
_gpu_mg("gpu_MG_c1k_cg", minCoarseRows=1000, coarseSolver="CG", maxIterCoarse=20, relTolCoarse=0.01)
_gpu_mg("gpu_MG_c4k_cg", minCoarseRows=4000, coarseSolver="CG", maxIterCoarse=30, relTolCoarse=0.01)
_gpu_mg("gpu_MG_c1k_cg_s2", minCoarseRows=1000, coarseSolver="CG", maxIterCoarse=20, relTolCoarse=0.01, maxIterSmoother=2)
_gpu_mg("gpu_MG_c1k_cg_s3", minCoarseRows=1000, coarseSolver="CG", maxIterCoarse=20, relTolCoarse=0.01, maxIterSmoother=3)
_gpu_mg("gpu_MG_c1k_cg_ssor", minCoarseRows=1000, coarseSolver="CG", maxIterCoarse=20, relTolCoarse=0.01, smoother="SSOR")
_gpu_mg("gpu_MG_c1k_cg_w", minCoarseRows=1000, coarseSolver="CG", maxIterCoarse=20, relTolCoarse=0.01, cycle="w")
_gpu_mg("gpu_MG_c1k_cg_csr", fmt="Csr", minCoarseRows=1000, coarseSolver="CG", maxIterCoarse=20, relTolCoarse=0.01)
_gpu_mg("gpu_MG_c1k_cg_reuse", minCoarseRows=1000, coarseSolver="CG", maxIterCoarse=20, relTolCoarse=0.01,
        reuseHierarchy="true")
_gpu_mg("gpu_MG_c1k_cg_cache5", caching=5, minCoarseRows=1000, coarseSolver="CG", maxIterCoarse=20, relTolCoarse=0.01)
_gpu_mg("gpu_MG_c1k_cg_cache20", caching=20, minCoarseRows=1000, coarseSolver="CG", maxIterCoarse=20, relTolCoarse=0.01)
# Result of the tuning (G36, 4 ranks): CG coarse solver on a small coarsest level, 2 smoothing sweeps, preconditioner
# reused for 20 solves. Used for all GPU timing and converged runs.
_gpu_mg("gpu_best", caching=20, minCoarseRows=1000, coarseSolver="CG", maxIterCoarse=20, relTolCoarse=0.01, maxIterSmoother=2)
CONFIGS["gpu_BJ"] = dict(solver="GKOCG", **GPU_COMMON, matrixFormat="Coo", preconditioner=dict(preconditioner="BJ"))
CONFIGS["gpu_ISAI"] = dict(solver="GKOCG", **GPU_COMMON, matrixFormat="Coo", preconditioner=dict(preconditioner="ISAI"))
CONFIGS["gpu_IC"] = dict(solver="GKOCG", **GPU_COMMON, matrixFormat="Coo", preconditioner=dict(preconditioner="IC"))
# Extra variants defined at run time from the command line: name=key:value,key:value (see _parse_config)


def _fmt_entry(k, v, ind):
    pad = " " * ind
    if isinstance(v, dict):
        body = "\n".join(_fmt_entry(kk, vv, ind + 4) for kk, vv in v.items())
        return f"{pad}{k}\n{pad}{{\n{body}\n{pad}}}"
    return f"{pad}{k:<16}{v};"


def is_gpu(cfg):
    return CONFIGS[cfg]["solver"].startswith("GKO")


def p_block(cfg, ranks, tol, relTol, max_iter=None):
    c = dict(CONFIGS[cfg])
    if is_gpu(cfg):
        c["ranksPerGPU"] = ranks                         # all ranks gather their matrix rows on the one GPU
        if max_iter:
            c["maxIter"] = max_iter
    c.update(tolerance=tol, relTol=relTol)
    return "\n".join(_fmt_entry(k, v, 8) for k, v in c.items())


def _parse_config(spec):
    """'name=base+key:value,key:value' defines a variant of an existing config (values into the preconditioner dict
    for keys that are not top-level OGL keys)."""
    name, rest = spec.split("=", 1)
    base, _, kvs = rest.partition("+")
    c = json.loads(json.dumps(CONFIGS[base]))
    top = {"matrixFormat", "maxIter", "minIter", "evalFrequency", "adaptMinIter", "resNormEval", "normEvalLimit",
           "solver", "ranksPerGPU"}
    for kv in filter(None, kvs.split(",")):
        k, v = kv.split(":", 1)
        if k in top or "preconditioner" not in c:
            c[k] = v
        else:
            c["preconditioner"][k] = v
    CONFIGS[name] = c
    return name


# ---------------------------------------------------------------------------------------------
# Case building and running
# ---------------------------------------------------------------------------------------------
def params_for(mesh, problem, numerics, cfg="GAMG", ranks=1, max_iterations=None, write_interval=None,
               write_format="ascii", res=True, max_iter=None):
    g = problem["given"]
    ms = mesh_spec(mesh, problem, numerics)
    ls, sim, sch = numerics["linear_solvers"], numerics["simple"], numerics["schemes"]
    n_it = max_iterations or sim["max_iterations"]
    return dict(
        ms, H=g["H"]["value"], Um=g["Um"]["value"], nu=g["nu"]["value"], rho=g["rho"]["value"],
        xc=g["xc"]["value"], yc=g["yc"]["value"], U_init=0, max_iterations=n_it,
        write_interval=write_interval or n_it, write_format=write_format, gpu=is_gpu(cfg), cores=ranks,
        balance_interval=1, div_U=sch["div_U"], cfg=cfg,
        p_block=p_block(cfg, ranks, ls["p"]["tol"], ls["p"]["relTol"], max_iter),
        U_solver=ls["U"]["solver"], U_smoother=ls["U"]["smoother"], U_tol=ls["U"]["tol"], U_relTol=ls["U"]["relTol"],
        n_nonorth=sim["n_nonorth"], consistent=sim["consistent"], relax_U=sim["relax_U"],
        res_U=sim["residual_U"] if res else 0, res_p=sim["residual_p"] if res else 0)


def build(name, mesh, problem, numerics, check=True, cache=False, **kw):
    """Render a case, run blockMesh (+ checkMesh). cache: reuse the polyMesh of an earlier build of the same mesh
    (timing runs; blockMesh of a 5 million cell mesh takes minutes and gives the same mesh every time)."""
    prm = params_for(mesh, problem, numerics, **kw)
    case = foam.render_case(HERE / "template", run_root() / name, prm)
    store = run_root() / "_meshcache" / mesh / "polyMesh"
    if cache and store.exists():
        shutil.copytree(store, case / "constant" / "polyMesh")
    else:
        foam.run("blockMesh", case)
        if cache:
            shutil.copytree(case / "constant" / "polyMesh", store)
    if check:
        foam.run("checkMesh", case)
    return case, prm


def map_from(case, source):
    """Initial field = latest time of `source` (a converged run of this case on another mesh)."""
    foam.run(f"mapFields {source} -consistent -sourceTime latestTime", case, "mapFields")


def solve(case, cores, timeout=2700, reconstruct=True, log_name="simpleFoam"):
    if cores > 1:
        foam.run("decomposePar -force", case, "decomposePar")
        hw = " --use-hwthread-cpus" if cores > 8 else ""      # more ranks than physical cores: use the 16 hardware threads
        wall = foam.run(f"mpirun{hw} -np {cores} simpleFoam -parallel", case, log_name, timeout=timeout)
        if reconstruct:
            foam.run("reconstructPar -latestTime", case, "reconstructPar")
    else:
        wall = foam.run("simpleFoam", case, log_name, timeout=timeout)
    return wall


def _dat(case, fo, fname):
    hits = sorted((case / "postProcessing" / fo).rglob(fname))
    rows = []
    for h in hits:
        rows += foam.read_dat(h)
    return rows


def collect(case):
    """Per-iteration series written by the function objects."""
    out = {}
    fc = np.array(_dat(case, "forcesCyl", "force.dat"), dtype=float)
    fw = np.array(_dat(case, "forcesWalls", "force.dat"), dtype=float)
    # force.dat (v2606): iteration, total xyz, pressure xyz, viscous xyz as nine scalar columns
    out["it"] = fc[:, 0]
    out["Fcyl"], out["Fcyl_p"], out["Fcyl_v"] = fc[:, 1:4], fc[:, 4:7], fc[:, 7:10]
    out["Fwall"] = fw[:, 1:4]
    for key, fo in (("mom_in", "momFlux_inlet"), ("mom_out", "momFlux_outlet"), ("p_in", "pForce_inlet"),
                    ("p_out", "pForce_outlet"), ("q_in", "flow_inlet"), ("q_out", "flow_outlet")):
        rows = _dat(case, fo, "surfaceFieldValue.dat")
        out[key] = np.array([r[1] for r in rows])
        out[key + "_it"] = np.array([r[0] for r in rows])
    return out


def _checkmesh_numbers(text):
    out = {}
    for key, pat in (("max_nonorth", r"Mesh non-orthogonality Max: ([\d.eE+-]+)"),
                     ("max_skew", r"Max skewness = ([\d.eE+-]+)"),
                     ("max_aspect", r"Max aspect ratio = ([\d.eE+-]+)")):
        mt = re.search(pat, text)
        out[key] = float(mt.group(1)) if mt else None
    return out


def loop_times(case, log_name="simpleFoam", skip=5):
    """Per-iteration wall clock of the solver loop: (ClockTime list, mean seconds per iteration after `skip`)."""
    log = (case / f"log.{log_name}").read_text(errors="replace")
    ck = [float(x) for x in re.findall(r"ClockTime = ([\d.eE+-]+) s", log)]
    ex = [float(x) for x in re.findall(r"ExecutionTime = ([\d.eE+-]+) s", log)]
    if len(ck) <= skip + 2:
        return ck, None, None
    # last iteration excluded (it writes the fields)
    return ck, (ck[-2] - ck[skip]) / (len(ck) - 2 - skip), (ex[-2] - ex[skip]) / (len(ex) - 2 - skip)


def ogl_times(case, log_name="simpleFoam"):
    """Mean OGL per-solve timings [ms] by label (verbose 1 lines '[OGL LOG][Proc: 0]p: <label>: <ms>')."""
    log = (case / f"log.{log_name}").read_text(errors="replace")
    d = {}
    for obj, lab, v in re.findall(r"\[OGL LOG\]\[Proc: 0\](p\w*): ([\w ]+): ([\d.eE+-]+) \[ms\]", log):
        d.setdefault(f"{obj}:{lab.strip()}", []).append(float(v))
    return {k: dict(mean=float(np.mean(v[3:] if len(v) > 6 else v)), n=len(v), total=float(np.sum(v))) for k, v in d.items()}


def profiling(case, cores):
    """ESI OpenFOAM profiling (controlDict profiling): total seconds per section from the last written time."""
    roots = [case / "processor0"] if cores > 1 else [case]
    out = {}
    for r in roots:
        tdirs = [d for d in foam.time_dirs(r) if (d / "uniform" / "profiling").exists()]
        if not tdirs:
            continue
        text = (tdirs[-1] / "uniform" / "profiling").read_text(errors="replace")
        for blk_ in re.findall(r"\{([^{}]*)\}", text):
            dsc = re.search(r'description\s+"([^"]+)"', blk_)
            tot = re.search(r"totalTime\s+([\d.eE+-]+);", blk_)
            cnt = re.search(r"calls\s+(\d+);", blk_)
            if dsc and tot:
                out[dsc.group(1)] = dict(total=float(tot.group(1)), calls=int(cnt.group(1)) if cnt else None)
    return out


class GpuMem:
    """Sample GPU memory use with nvidia-smi while a run is going (MiB)."""

    def __init__(self, path):
        self.path = Path(path)
        self.base = self.sample()
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "250"],
                                  stdout=open(self.path, "w"), stderr=subprocess.DEVNULL)

    @staticmethod
    def sample():
        try:
            return float(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True).stdout.split()[0])
        except Exception:                          # noqa: BLE001 - no GPU tool: no memory record
            return None

    def stop(self):
        self.p.terminate()
        self.p.wait()
        v = [float(x) for x in self.path.read_text().split() if x.strip()]
        return dict(base_mib=self.base, peak_mib=max(v) if v else None,
                    peak_above_base_mib=(max(v) - self.base) if v and self.base is not None else None)


def record(name, case, prm, cores, wall, extra=None, kind="runs"):
    """Save iteration series + metadata of a finished run."""
    d = OUT / kind
    d.mkdir(parents=True, exist_ok=True)
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    ts = collect(case)
    np.savez_compressed(d / f"{name}.npz", **ts)
    log = case / "log.simpleFoam"
    health = monitor.assess(log)
    pf = perf.summarize(case, "simpleFoam", cells=prm["cells"], cores=cores)
    ck, s_it, cpu_it = loop_times(case)
    pf.update(loop_s_per_iter=s_it, loop_cpu_s_per_iter=cpu_it, profiling=profiling(case, cores),
              ogl=ogl_times(case) if prm["gpu"] else {})
    (PERF_DIR / f"{name}.json").write_text(json.dumps(pf, indent=1))
    cm = (case / "log.checkMesh").read_text() if (case / "log.checkMesh").exists() else ""
    text = log.read_text(errors="replace")
    conv = re.search(r"SIMPLE solution converged in (\d+) iterations", text)
    meta = dict(name=name, mesh=prm["mesh"], cells=prm["cells"], h=prm["h"], cores=cores, cfg=prm["cfg"],
                p_block=prm["p_block"], case_dir=str(case), wall_time=wall, iterations=int(ts["it"][-1]),
                converged=bool(conv), mesh_ok=("Mesh OK." in cm) if cm else None, checkmesh=_checkmesh_numbers(cm),
                health=health, perf=pf, F_last=ts["Fcyl"][-1].tolist(),
                openfoam=foam.foam_env("WM_PROJECT_VERSION"), python=platform.python_version(), run_root=str(run_root()))
    meta.update(extra or {})
    (d / f"{name}.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"{name}: {prm['cells']:,} cells, {cores} ranks, {prm['cfg']}, {meta['iterations']} it, {wall:.1f} s, "
          f"loop {s_it} s/it, p iters {pf['lin_iters'].get('p')}, health {health['status']}, F {meta['F_last']}", flush=True)
    return meta


# ---------------------------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------------------------
def do_mesh(meshes):
    problem, numerics = load()
    for M in meshes:
        case, prm = build(f"mesh_{M}", M, problem, numerics, max_iterations=1)
        cm = (case / "log.checkMesh").read_text()
        print(M, prm["cells"], {k: prm[k] for k in ("nc", "nr", "nl", "nd", "nb", "nt", "nz")},
              {k: round(v, 5) for k, v in prm["h"].items()}, _checkmesh_numbers(cm), "Mesh OK." in cm, flush=True)


def do_preflight():
    """Short runs from the uniform start on the coarsest (1 core) and finest (16 ranks) mesh: health, outputs, timing."""
    problem, numerics = load()
    lv, cores = numerics["mesh"]["levels"], numerics["parallel"]["cores"]
    n = numerics["preflight"]["iterations"]
    targets = {"Ux": numerics["simple"]["residual_U"], "p": numerics["simple"]["residual_p"]}
    out = []
    for M in (lv[0], lv[-1]):
        case, prm = build(f"pf_{M}", M, problem, numerics, ranks=cores[M], max_iterations=n)
        t0 = time.time()
        try:
            wall = solve(case, cores[M], timeout=1200)
            crashed = False
        except RuntimeError as e:
            wall, crashed = time.time() - t0, True
            print(e)
        a = monitor.assess(case / "log.simpleFoam", residual_targets=targets)
        ts = collect(case)
        pf = perf.summarize(case, "simpleFoam", cells=prm["cells"], cores=cores[M])
        ck, s_it, _ = loop_times(case)
        a.update(name=M, cells=prm["cells"], cores=cores[M], wall=wall, crashed=crashed, case=str(case), iterations=n,
                 perf=pf, s_per_iter=s_it, n_force_rows=len(ts["it"]), F_last=ts["Fcyl"][-1].tolist(),
                 q_in=float(ts["q_in"][-1]), q_out=float(ts["q_out"][-1]), profiling=profiling(case, cores[M]),
                 checkmesh=_checkmesh_numbers((case / "log.checkMesh").read_text()),
                 mesh_ok="Mesh OK." in (case / "log.checkMesh").read_text())
        out.append(a)
        print(f"preflight {M}: {a['status']}, {prm['cells']:,} cells, {cores[M]} ranks, {wall:.1f} s for {n} it, {s_it} s/it, "
              f"projected {a['est_full_runtime']} ({a['est_kind']}); rows {a['n_force_rows']}; q_in {a['q_in']} q_out {a['q_out']}; "
              f"mesh {a['checkmesh']}", flush=True)
        for c in a["checks"]:
            print(f"   {c['status']:4s} {c['name']}: {c['value']}")
        for fo in sorted((case / "postProcessing").iterdir()):
            f = next(fo.rglob("*.dat"))
            lines = f.read_text().splitlines()
            print("   ", fo.name, "|", lines[-1][:150] if lines else "EMPTY")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(out, indent=1, default=str))


def do_production(meshes, cfg="GAMG", ranks=None, prefix=""):
    problem, numerics = load()
    lv = numerics["mesh"]["levels"]
    for M in meshes:
        k = lv.index(M)
        n = ranks or numerics["parallel"]["cores"][M]
        name = prefix + M
        case, prm = build(name, M, problem, numerics, cfg=cfg, ranks=n)
        src = None
        if k > 0:
            # always the converged PRODUCTION run of the coarser level, so that CPU and GPU comparison runs start from
            # exactly the same field
            src = run_root() / lv[k - 1]
            map_from(case, src)
        gm = GpuMem(case / "gpumem.txt") if prm["gpu"] else None
        wall = solve(case, n)
        mem = gm.stop() if gm else None
        record(name, case, prm, n, wall, extra=dict(source=str(src) if src else None, gpu_mem=mem))


def do_timing(mesh, cfg, ranks, iters, tag=""):
    """Fixed number of SIMPLE iterations from the uniform start (same for every config): time per iteration and per
    pressure solve, iterations per solve, GPU memory. Fields written once at the end (binary)."""
    problem, numerics = load()
    name = f"t_{mesh}_{cfg}_np{ranks}{tag}"
    case, prm = build(name, mesh, problem, numerics, check=False, cache=True, cfg=cfg, ranks=ranks, max_iterations=iters,
                      write_format="binary", res=False)
    gm = GpuMem(case / "gpumem.txt") if prm["gpu"] else None
    try:
        wall = solve(case, ranks, timeout=2700, reconstruct=False)
        ok = True
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        wall, ok = None, False
        print(f"{name} FAILED: {str(e)[-800:]}", flush=True)
    mem = gm.stop() if gm else None
    if not ok:
        TIMING_DIR.mkdir(parents=True, exist_ok=True)
        (TIMING_DIR / f"{name}.json").write_text(json.dumps(dict(name=name, mesh=mesh, cfg=cfg, cores=ranks, ok=False,
                                                                  cells=prm["cells"], gpu_mem=mem)))
        return None
    meta = record(name, case, prm, ranks, wall, extra=dict(gpu_mem=mem, ok=True, tag=tag, kind="timing"), kind="timing")
    for d in case.glob("processor*"):                         # keep the disk free: timing runs are not re-analysed
        shutil.rmtree(d)
    return meta


def do_variant(name, level, overrides, ranks=8):
    """Sensitivity run: a converged CPU run of one level with numerics.yaml entries overridden (dotted.key=value),
    mapped from the converged next coarser level, recorded like a production run under `name`."""
    problem, numerics = load()
    source = None
    for ov in overrides:
        k, v = ov.split("=", 1)
        if k == "from":                                        # start field from another run (default: coarser level)
            source = v
            continue
        if k == "ranks":
            ranks = int(v)
            continue
        d = numerics
        *path, last = k.split(".")
        for p_ in path:
            d = d[p_]
        d[last] = yaml.safe_load(v)
    lv = numerics["mesh"]["levels"]
    case, prm = build(name, level, problem, numerics, ranks=ranks)
    source = source or lv[lv.index(level) - 1]
    map_from(case, run_root() / source)
    wall = solve(case, ranks)
    record(name, case, prm, ranks, wall, extra=dict(overrides=overrides, variant_of=level, source=source))


def do_tune(mesh, ranks, iters, cfgs):
    for c in cfgs:
        do_timing(mesh, c, ranks, iters, tag="_tune")


def do_flow(M):
    """Mid-plane z = H/2 velocity of a finished run (average of the two cell layers next to it) -> npz."""
    problem, numerics = load()
    case = run_root() / M
    t = foam.time_dirs(case)[-1]
    foam.run(f"postProcess -func writeCellCentres -time {t.name}", case, "writeCellCentres")
    C = foam.read_internal_field(t / "C")
    U = foam.read_internal_field(t / "U")
    H = problem["given"]["H"]["value"]
    zs = np.unique(np.round(C[:, 2], 10))
    below, above = zs[zs < H / 2].max(), zs[zs > H / 2].min()
    lo, hi = np.isclose(C[:, 2], below, atol=1e-9), np.isclose(C[:, 2], above, atol=1e-9)
    # same extrusion: cells of the two layers have the same (x, y), in the same order within a layer
    xl, yl = C[lo, 0], C[lo, 1]
    rd = lambda a: np.round(a, 8)                           # round-off differences between layers must not reorder
    oh = np.lexsort((rd(C[hi, 1]), rd(C[hi, 0])))
    ol = np.lexsort((rd(yl), rd(xl)))
    assert np.allclose(C[hi][oh][:, :2], C[lo][ol][:, :2])
    w = (above - H / 2) / (above - below)                   # linear interpolation to z = H/2
    Umid = w * U[lo][ol] + (1 - w) * U[hi][oh]
    np.savez_compressed(RUNS_DIR / f"{M}_mid.npz", x=xl[ol], y=yl[ol], u=Umid[:, 0], v=Umid[:, 1], w=Umid[:, 2],
                        z_lo=below, z_hi=above)
    print(f"mid-plane of {M}: {len(xl)} points, layers at z = {below:.5f}, {above:.5f}; min u {Umid[:, 0].min():.4f}")


def do_inlet_visc(levels):
    """Viscous x-force through the inlet, nu * sum (u_b - u_P) / d * A over the inlet faces (one-sided face gradient,
    as the solver's Laplacian). It vanishes for the continuous problem (continuity makes the integral of du/dx over the
    inlet zero), but not for the discrete one: the momentum balance neglects it, and this measures how much it is."""
    problem, _ = load()
    g = problem["given"]
    nu, Um, H = g["nu"]["value"], g["Um"]["value"], g["H"]["value"]
    out = {}
    for L in levels:
        case = run_root() / L
        t = foam.time_dirs(case)[-1]
        for fn, f in (("writeCellCentres", "C"), ("writeCellVolumes", "V")):
            if not (t / f).exists():
                foam.run(f"postProcess -func {fn} -time {t.name}", case, fn)
        C = foam.read_internal_field(t / "C")
        U = foam.read_internal_field(t / "U")
        V = foam.read_internal_field(t / "V")
        x0 = C[:, 0].min()                                      # first cell layer at the inlet (same x for every cell)
        s = np.isclose(C[:, 0], x0, atol=1e-9)
        y, z = C[s, 1], C[s, 2]
        ub = 16 * Um * y * z * (H - y) * (H - z) / H ** 4
        A = V[s] / (2 * x0)
        out[L] = dict(F_visc_in=float(nu * np.sum((ub - U[s, 0]) / x0 * A)), area=float(A.sum()), faces=int(s.sum()))
        print(L, out[L], flush=True)
    p = RUNS_DIR / "inlet_visc.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    p.write_text(json.dumps(old | out, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    args = sys.argv[2:]
    for a in list(args):                                       # runtime config variants: name=base+key:value,...
        if "=" in a and "+" in a.split("=", 1)[1] or ("=" in a and a.split("=", 1)[1] in CONFIGS):
            args[args.index(a)] = _parse_config(a)
    if mode == "mesh":
        do_mesh(args or load()[1]["mesh"]["levels"])
    elif mode == "preflight":
        do_preflight()
    elif mode == "production":
        do_production(args)
    elif mode == "gpu_full":                                   # gpu_full PREFIX CONFIG RANKS MESH [MESH ...]
        do_production(args[3:], cfg=args[1], ranks=int(args[2]), prefix=args[0])
    elif mode == "timing":
        do_timing(args[0], args[1], int(args[2]), int(args[3]), args[4] if len(args) > 4 else "")
    elif mode == "tune":
        do_tune(args[0], int(args[1]), int(args[2]), args[3:])
    elif mode == "flow":
        do_flow(args[0])
    elif mode == "inletvisc":
        do_inlet_visc(args)
    elif mode == "variant":                                    # variant NAME LEVEL key=value [key=value ...]
        do_variant(args[0], args[1], args[2:])
    else:
        raise SystemExit(f"unknown mode {mode}")
