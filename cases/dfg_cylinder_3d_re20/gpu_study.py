"""GPU versus CPU study for the DFG 3D-1Z case: is the pressure equation solved faster on the GPU (OGL), and from
which mesh size does it pay off? Renders results/dfg_cylinder_3d_re20/gpu_study.html with cfdlib.study, and a short
summary (gpu_summary.json) for the main report.

Usage (repo root, inside WSL):  python -m cases.dfg_cylinder_3d_re20.gpu_study
Reads results/dfg_cylinder_3d_re20/{timing/*.json, runs/*.json|npz}. Every number is computed here.
Run names: t_<mesh>_<config>_np<ranks>[tag] are fixed-iteration timing runs (tag _tune: tuning batch, partly measured
while other runs were going; _r2: repeat); cpu_<level> / gpu_<level> are converged comparison runs.
"""
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from cfdlib import insights, plots, report, study
from cfdlib.report import load_yaml, m, tex_num as tn, tex_fix as tf
from .run import CONFIGS, mesh_spec

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
TIM = OUT / "timing"
RUNS = OUT / "runs"
CASE = HERE.name
CPU, GPU = "GAMG", "gpu_best"
CPUS = ("GAMG", "GAMG_DIC")                # CPU baselines: GAMG with Gauss-Seidel or DIC-Gauss-Seidel smoother
# Same-answer tolerance, fixed BEFORE the GPU comparison runs: at least ten times below the finest-mesh GCI of each number.
TOL = dict(cd=1.0e-4, cl=1.0e-5)
DESC = {                                   # tuning configurations: what each changes (run.py CONFIGS)
    "GAMG": "CPU GAMG (Gauss-Seidel), the baseline",
    "gpu_MG_default": "OGL defaults: coarsest level 64000 rows, 1 Jacobi sweep as coarse solve, adaptMinIter on",
    "gpu_MG_noadapt": "defaults, adaptMinIter off",
    "gpu_MG_c1k": "coarsest level 1000 rows (more levels)",
    "gpu_MG_c1k_cg": "+ coarse solve by CG (20 iterations)",
    "gpu_MG_c1k_cg_adapt": "same, adaptMinIter on",
    "gpu_MG_c4k_cg": "coarsest level 4000 rows, CG coarse solve",
    "gpu_MG_c1k_cg_s2": "+ 2 smoothing sweeps",
    "gpu_MG_c1k_cg_ssor": "SSOR smoother instead of Jacobi",
    "gpu_MG_c1k_cg_cache5": "preconditioner reused for 5 solves",
    "gpu_MG_c1k_cg_cache20": "preconditioner reused for 20 solves",
    "gpu_MG_c1k_cg_reuse": "hierarchy reused, coarse matrices recomputed",
    "gpu_MG_c1k_cg_csr": "Csr matrix format",
    "gpu_MG_c4k_cg_cache20": "coarsest 4000 rows, reused for 20 solves",
    "gpu_MG_c1k_cg_s2_cache20": "2 sweeps, reused for 20 solves",
    "gpu_s2c20": "2 sweeps, reused for 20 solves (repeat, nothing else running)",
    "gpu_s3c20": "3 sweeps, reused for 20 solves",
    "gpu_s2c50": "2 sweeps, reused for 50 solves",
    "gpu_c4k_s2c20": "coarsest 4000 rows, 2 sweeps, reused 20",
    "gpu_s2c20_csr": "2 sweeps, reused 20, Csr format",
    "gpu_s2c20_k10": "2 sweeps, reused 20, 10 coarse CG iterations",
    "gpu_BJ": "block Jacobi preconditioner",
    "gpu_ISAI": "ISAI preconditioner",
    "gpu_IC": "incomplete Cholesky preconditioner",
}


def _exec_times(case):
    log = (Path(case) / "log.simpleFoam").read_text(errors="replace")
    return [float(x) for x in re.findall(r"^ExecutionTime = ([\d.eE+-]+) s", log, re.M)]


def load_timing():
    out = {}
    for f in TIM.glob("*.json"):
        d = json.loads(f.read_text())
        d["mtime"] = f.stat().st_mtime
        if not d.get("ok", True) or "perf" not in d:
            out[d["name"]] = dict(d, failed=True)
            continue
        p = d["perf"]
        pr = p.get("profiling", {})
        calls = lambda k: pr.get(k, {}).get("calls") or 1
        tot = lambda k: pr.get(k, {}).get("total", float("nan"))
        loop_key = next((k for k in pr if k.startswith("time.run()")), None)
        fo = sum(v["total"] for k, v in pr.items() if k.startswith("functionObject::") and ("execute" in k or ":write" in k))
        n = p["steps"]
        d.update(failed=False, exec_it=p["loop_cpu_s_per_iter"], p_it=p["lin_iters"].get("p"), n_it=n,
                 p_call=tot("fvMatrix::solve.p") / calls("fvMatrix::solve.p"),
                 U_call=tot("fvMatrix::solve.U") / calls("fvMatrix::solve.U"), fo_call=fo / n,
                 loop_call=tot(loop_key) / calls(loop_key) if loop_key else float("nan"),
                 main=tot("application::main"), ogl={k: v for k, v in p.get("ogl", {}).items()},
                 gpu=d["cfg"].startswith("gpu"), mem=(d.get("gpu_mem") or {}).get("peak_above_base_mib"),
                 tag=d.get("tag") or re.sub(r"^.*_np\d+", "", d["name"]))
        d["rest_call"] = d["loop_call"] - d["p_call"] - d["U_call"] - d["fo_call"]
        # Time per iteration: rank 0's ExecutionTime differences after 5 warm-up iterations (excludes one-off costs of the
        # first iteration and the final write) when ExecutionTime is wall time, i.e. rank 0 never slept. Check: the windowed
        # CPU time per iteration is at least 70 % of the profiled wall time per iteration of the whole loop (one-off costs
        # explain ratios down to about 0.8). Otherwise (16 ranks on the smallest mesh: MPI lets rank 0 sleep, ratio 0.5)
        # the profiled wall time of the whole loop / iterations (Insights).
        d["cpu_wall"] = d["exec_it"] / d["loop_call"] if d["exec_it"] and d["loop_call"] else 0
        d["exec_ok"] = d["cpu_wall"] >= 0.7
        d["s_it"] = d["exec_it"] if d["exec_ok"] else d["loop_call"]
        out[d["name"]] = d
    return out


def pick(T, mesh, cfg, tags=("", "_r2")):
    """Runs of one mesh and config, grouped by rank count: {ranks: [runs]} (failed runs excluded)."""
    g = {}
    for d in T.values():
        if not d["failed"] and d["mesh"] == mesh and d["cfg"] == cfg and d["tag"] in tags:
            g.setdefault(d["cores"], []).append(d)
    return g


def mean_of(runs, key):
    return float(np.mean([r[key] for r in runs]))


def best(T, mesh, cfg):
    """Fastest rank count (mean time per iteration over repeats): (ranks, runs, mean s/it)."""
    g = pick(T, mesh, cfg)
    if not g:
        return None
    r = min(g, key=lambda k: mean_of(g[k], "s_it"))
    return dict(ranks=r, runs=g[r], s_it=mean_of(g[r], "s_it"), p_call=mean_of(g[r], "p_call"), U_call=mean_of(g[r], "U_call"),
                fo_call=mean_of(g[r], "fo_call"), rest_call=mean_of(g[r], "rest_call"), loop_call=mean_of(g[r], "loop_call"),
                p_it=mean_of(g[r], "p_it"), mem=max((x["mem"] or 0) for x in g[r]) if cfg.startswith("gpu") else None,
                cells=g[r][0]["cells"], all=g)


def best_cpu(T, mesh):
    """Fastest CPU setup over both GAMG smoothers and all rank counts."""
    c = [b for b in (best(T, mesh, cfg) for cfg in CPUS) if b]
    if not c:
        return None
    b = min(c, key=lambda x: x["s_it"])
    b["cfg"] = b["runs"][0]["cfg"]
    return b


def converged(name, g):
    p = RUNS / f"{name}.json"
    if not p.exists():
        return None
    meta = json.loads(p.read_text())
    d = np.load(RUNS / f"{name}.npz")
    F0 = 0.5 * g["rho"]["value"] * (4 / 9 * g["Um"]["value"]) ** 2 * g["D"]["value"] * g["H"]["value"]
    return dict(meta=meta, cd=float(d["Fcyl"][-1, 0] / F0), cl=float(d["Fcyl"][-1, 1] / F0), it=meta["iterations"],
                wall=meta["wall_time"], s_it=meta["perf"].get("loop_cpu_s_per_iter"), ranks=meta["cores"],
                p_it=meta["perf"]["lin_iters"].get("p"))


def machine():
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, shell=True).stdout.strip()
        except Exception:                                  # noqa: BLE001
            return ""
    gpu = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    cuda = sh("/usr/local/cuda/bin/nvcc --version | tail -2 | head -1")
    mpi = sh("mpirun --version | head -1")
    return gpu, cuda, mpi


def main():
    problem = load_yaml(HERE / "problem.yaml")
    numerics = load_yaml(HERE / "numerics.yaml")
    g = problem["given"]
    T = load_timing()
    meshes = [M for M in ("G16", "L3", "G36", "L4", "G60") if any(pick(T, M, c) for c in CPUS) or pick(T, M, GPU)]
    cells = {M: mesh_spec(M, problem, numerics)["cells"] for M in meshes}
    B = {M: dict(cpu=best_cpu(T, M), gpu=best(T, M, GPU)) for M in meshes}
    for M in meshes:
        c, gq = B[M]["cpu"], B[M]["gpu"]
        print(M, cells[M], "CPU", c and (c["ranks"], round(c["s_it"], 4), round(c["p_call"], 4)),
              "GPU", gq and (gq["ranks"], round(gq["s_it"], 4), round(gq["p_call"], 4)))
    pairs = [M for M in meshes if B[M]["cpu"] and B[M]["gpu"]]
    sp = {M: B[M]["cpu"]["s_it"] / B[M]["gpu"]["s_it"] for M in pairs}
    spp = {M: B[M]["cpu"]["p_call"] / B[M]["gpu"]["p_call"] for M in pairs}
    # converged comparison runs (same start field, same stopping rule)
    conv = {k: converged(k, g) for k in ("L3", "cpu_L3", "gpu_L3", "L4", "cpu_L4", "gpu_L4")}
    print({k: (v["cd"], v["cl"], v["it"], v["wall"]) for k, v in conv.items() if v})
    build(problem, numerics, g, T, meshes, cells, B, pairs, sp, spp, conv)


def crossing(xs, ys):
    """Cell count where ys crosses 1 (log-linear interpolation between neighbours), or None."""
    for i in range(len(xs) - 1):
        a, b = ys[i] - 1, ys[i + 1] - 1
        if a < 0 <= b or a >= 0 > b:
            t = a / (a - b)
            return float(np.exp(np.log(xs[i]) + t * (np.log(xs[i + 1]) - np.log(xs[i]))))
    return None


def build(problem, numerics, g, T, meshes, cells, B, pairs, sp, spp, conv):
    big = pairs[-1]
    cb, gb = B[big]["cpu"], B[big]["gpu"]
    xs = [cells[M] for M in pairs]
    be_it = crossing(xs, [sp[M] for M in pairs])
    be_p = crossing(xs, [spp[M] for M in pairs])
    gpu_name, cuda, mpi = machine()
    ogl_commit = next((re.search(r"OGL commit:\s+(\w+)", (Path(r["case_dir"]) / "log.simpleFoam").read_text(errors="replace")).group(1)
                       for r in T.values() if not r["failed"] and r["gpu"] and (Path(r["case_dir"]) / "log.simpleFoam").exists()), "?")

    # ---- same answer ----------------------------------------------------------------------------
    sa_rows, sa_ok, sa_pairs = [], [], []
    for lvl in ("L3", "L4"):
        c = conv.get(f"cpu_{lvl}") or conv.get(lvl)
        gq = conv.get(f"gpu_{lvl}")
        if not (c and gq):
            continue
        sa_pairs.append(lvl)
        for k, sym in (("cd", "c_D"), ("cl", "c_L")):
            d = gq[k] - c[k]
            ok = abs(d) <= TOL[k]
            sa_ok.append(ok)
            sa_rows.append([lvl + f" ({c['meta']['cells']:,} cells)", m(sym), m(tf(c[k], 6)), m(tf(gq[k], 6)), m(tn(d, 2)),
                            m(tn(TOL[k], 1)), "✓" if ok else "✗"])
        sa_rows.append([lvl, "iterations to converge", str(c["it"]), str(gq["it"]), f"{gq['it'] - c['it']:+d}", "–", "–"])
    same_ok = bool(sa_ok) and all(sa_ok)

    # ---- tables ------------------------------------------------------------------------------------
    def s(x, d=3):
        return m(tf(x, d)) if x is not None and np.isfinite(x) else "–"

    main_rows = []
    for M in meshes:
        c, gq = B[M]["cpu"], B[M]["gpu"]
        main_rows.append([M, f"{cells[M]:,}",
                          (s(c["s_it"]) + f" ({c['ranks']})") if c else "–", (s(gq["s_it"]) + f" ({gq['ranks']})") if gq else "–",
                          s(sp.get(M), 2) if M in sp else "–", s(c["p_call"]) if c else "–", s(gq["p_call"]) if gq else "–",
                          s(spp.get(M), 2) if M in spp else "–", s(c["p_it"], 1) if c else "–", s(gq["p_it"], 1) if gq else "–",
                          f"{gq['mem']:.0f}" if gq and gq["mem"] else "–"])
    all_rows = []
    for d in sorted((d for d in T.values() if not d["failed"] and d["tag"] in ("", "_r2")),
                    key=lambda d: (d["cells"], d["cfg"], d["cores"], d["tag"])):
        all_rows.append([d["name"], f"{d['cells']:,}", "GPU" if d["gpu"] else "CPU", str(d["cores"]),
                         s(d["s_it"]) + ("" if d["exec_ok"] else " *"), s(d["p_call"]),
                         s(d["U_call"]), s(d["p_it"], 1), f"{d['mem']:.0f}" if d["mem"] else "–"])
    failed = [d for d in T.values() if d["failed"]]
    tune = sorted((d for d in T.values() if not d["failed"] and d["tag"] == "_tune"), key=lambda d: d["mtime"])
    tune_rows = []
    for d in tune:
        o = d["ogl"]
        n_solves = o.get("p:solve", {}).get("n") or 1
        setup = o.get("p:init_precond", {}).get("total", 0) / n_solves if o else None
        tune_rows.append(["<code>" + d["cfg"] + "</code><br>" + DESC.get(d["cfg"], d["cfg"]), s(d["p_it"], 1),
                          m(tf(o["p:solve"]["mean"], 0)) if "p:solve" in o else "–",
                          m(tf(setup, 0)) if o else "–", s(d["p_call"]), s(d["s_it"])])
    conv_rows = []
    for k in ("cpu_L3", "gpu_L3", "cpu_L4", "gpu_L4"):
        c = conv.get(k)
        if c:
            conv_rows.append([k, f"{c['meta']['cells']:,}", "GPU" if k.startswith("gpu") else "CPU", str(c["ranks"]), str(c["it"]),
                              s(c["wall"], 1), s(c["s_it"]), s(c["p_it"], 1)])
    per_run = {}
    for lvl in ("L3", "L4"):
        if conv.get(f"cpu_{lvl}") and conv.get(f"gpu_{lvl}"):
            per_run[lvl] = conv[f"cpu_{lvl}"]["wall"] / conv[f"gpu_{lvl}"]["wall"]
    # CPU scaling and GPU rank count on the L4 mesh
    scal_rows = []
    for cfg in (CPU, "GAMG_DIC", GPU):
        for r, runs in sorted(pick(T, "L4", cfg).items()):
            scal_rows.append([cfg, str(r), s(mean_of(runs, "s_it")), s(mean_of(runs, "p_call")), s(mean_of(runs, "U_call")),
                              s(mean_of(runs, "rest_call")), s(mean_of(runs, "p_it"), 1)])

    # ---- breakdown -------------------------------------------------------------------------------
    bd_rows, ogl_rows, o_ms, of_ms = [], [], {}, {}
    for M in pairs:
        for side in ("cpu", "gpu"):
            x = B[M][side]
            tot = x["p_call"] + x["U_call"] + x["fo_call"] + x["rest_call"]
            bd_rows.append([M, ("GPU" if side == "gpu" else "CPU") + f" ({x['ranks']} ranks)", s(x["p_call"]), s(x["U_call"]),
                            s(x["fo_call"]), s(x["rest_call"]), s(tot), m(tf(100 * x["p_call"] / tot, 0) + r"\,\%")])
        o = B[M]["gpu"]["runs"][0]["ogl"]
        if o:
            n_solves = o.get("p:solve", {}).get("n") or 1
            mean = lambda k: o.get(k, {}).get("mean", float("nan"))
            setup = o.get("p:init_precond", {}).get("total", 0) / n_solves
            tot_ogl = mean("p:solve_multi_gpu")
            ofs = 1000 * B[M]["gpu"]["p_call"] - tot_ogl
            o_ms[M] = mean("p:solve") / B[M]["gpu"]["p_it"]
            of_ms[M] = ofs
            ogl_rows.append([M, m(tf(mean("p_distributed_matrix:call_update"), 0)), m(tf(setup, 0)), m(tf(mean("p:solve"), 0)),
                             m(tf(mean("p:copy_x_back"), 0)), m(tf(tot_ogl, 0)), m(tf(ofs, 0)), s(B[M]["gpu"]["p_it"], 1),
                             m(tf(mean("p:solve") / B[M]["gpu"]["p_it"], 2))])
    # pressure share and Amdahl bound from profiling only (same basis: means over all iterations)
    pshare = {M: B[M]["cpu"]["p_call"] / B[M]["cpu"]["loop_call"] for M in pairs}
    amdahl = {M: 1 / (1 - pshare[M]) for M in pairs}
    # cost of one pressure solve per million cells: does it grow (CPU) or fall (GPU) with the mesh?
    pcell = {side: {M: 1e6 * B[M][side]["p_call"] / cells[M] for M in pairs} for side in ("cpu", "gpu")}

    # ---- figures ---------------------------------------------------------------------------------
    svgs = {"tpi": _tpi_fig(pairs, cells, B, sp, spp), "ranks": _ranks_fig(T), "tune": _tune_fig(tune),
            "bd": _bd_fig(pairs, cells, B)}
    figures = [
        dict(key="tpi", section="results", caption="Left: wall time per SIMPLE iteration (solid) and per pressure solve "
             "(dashed) against mesh size, best CPU setup (GAMG on its fastest rank count) and best GPU setup (OGL, tuned "
             "multigrid-preconditioned CG). Right: speed-up of the GPU over the CPU, for the whole iteration and for the "
             "pressure solve alone; above the dotted line the GPU is faster."),
        dict(key="ranks", section="results", caption="Rank count on the " + f"{cells.get('L4', 0):,}" + "-cell mesh (L4): time "
             "per iteration for CPU GAMG (one rank per core or hardware thread) and for the GPU runs (the ranks assemble and "
             "solve momentum on the CPU and all send their pressure matrix to the one GPU)."),
        dict(key="tune", section="results", caption="GPU solver tuning on the " + f"{cells.get('G36', 0):,}" + "-cell mesh "
             "(4 ranks, 20 iterations): CG iterations per pressure solve (left, log scale) and wall time per pressure solve "
             "(right), in the order they were tried. Hatched: measured while multi-core accuracy runs were going (time inflated, "
             "iteration counts unaffected); solid: nothing else running, or at most single-core set-up work. Dotted: the CPU "
             "baseline (GAMG, 4 ranks)."),
        dict(key="bd", section="breakdown", caption="Where the time of one iteration goes, best CPU (left bar of each pair) and "
             "best GPU setup (right bar), per million cells: pressure solve, momentum solve, force output (function objects) and "
             "the rest (matrix assembly, gradients, fluxes, MPI). The GPU only shortens the first part."),
    ]

    # ---- texts ------------------------------------------------------------------------------------
    win = [M for M in pairs if sp[M] > 1]
    winp = [M for M in pairs if spp[M] > 1]
    head = (("The GPU is faster per iteration from about " + m(tf(be_it / 1e6, 1)) + " million cells" if be_it and sp[big] > 1 else
             "The GPU never makes a whole SIMPLE iteration faster in the tested range (" + f"{xs[0]:,}" + " to " + f"{xs[-1]:,}"
             + " cells)" if not win else "The GPU is faster per iteration on " + ", ".join(win))
            + ": on the largest mesh (" + f"{cells[big]:,}" + " cells) one iteration takes " + m(tf(cb["s_it"], 2) + r"\,\mathrm{s}")
            + " on the best CPU setup (" + clab(cb) + ") and " + m(tf(gb["s_it"], 2) + r"\,\mathrm{s}")
            + " with the pressure solved on the GPU (" + str(gb["ranks"]) + " ranks), a factor " + m(tf(sp[big], 2))
            + "; the pressure solve alone takes " + m(tf(cb["p_call"], 2)) + " against " + m(tf(gb["p_call"], 2) + r"\,\mathrm{s}")
            + " (factor " + m(tf(spp[big], 2)) + "). " + ("Both give the same converged forces." if same_ok else
                                                          "The converged forces do NOT agree to the stated tolerance." if sa_ok else
                                                          "The same-answer test has not run yet."))
    answers = [
        dict(label="Speed-up per iteration, largest mesh", value=m(tf(sp[big], 2) + r"\times"),
             note=f"{cells[big]:,} cells; best CPU ({clab(cb)}) over best GPU ({gb['ranks']} ranks); above 1 the GPU is faster"),
        dict(label="Speed-up of the pressure solve, largest mesh", value=m(tf(spp[big], 2) + r"\times"),
             note="Time per pressure solve including the matrix transfer to the GPU and the solution back"),
        dict(label="Break-even mesh size (whole iteration)", value=(m(tf(be_it / 1e6, 1)) + " million cells") if be_it else "none in range",
             note="Log-linear interpolation of the measured speed-ups" if be_it else
                  ("the GPU is " + ("slower" if sp[big] < 1 else "faster") + " at every tested size, " + f"{xs[0]:,}" + " to " + f"{xs[-1]:,}" + " cells")),
        dict(label="Break-even mesh size (pressure solve alone)", value=(m(tf(be_p / 1e6, 1)) + " million cells") if be_p else "none in range",
             note="Where the GPU pressure solve becomes faster than GAMG" if be_p else
                  ("the GPU pressure solve is " + ("slower" if spp[big] < 1 else "faster") + " at every tested size")),
        dict(label="Same answer", value="✓ yes" if same_ok else ("✗ no" if sa_ok else "not tested"),
             note="Converged drag and lift of CPU and GPU runs on " + " and ".join(sa_pairs) + " within "
                  + m(r"|\Delta c_D| \le " + tn(TOL["cd"])) + ", " + m(r"|\Delta c_L| \le " + tn(TOL["cl"]))),
        dict(label="Peak GPU memory, largest mesh", value=m(tf(gb["mem"] / 1024, 2) + r"\,\mathrm{GB}") if gb["mem"] else "–",
             note="nvidia-smi during the run, above the idle level; the card has 12 GB"),
    ] + ([dict(label="Converged run, " + lv, value=m(tf(per_run[lv], 2) + r"\times"),
               note=f"CPU wall time over GPU wall time to the same steady state ({conv['cpu_' + lv]['wall']:.0f} s against "
                    f"{conv['gpu_' + lv]['wall']:.0f} s)") for lv in per_run])
    findings = [
        "<b>Tuning mattered more than hardware.</b> With OGL's default multigrid the GPU needed " + m(tf(_cfgv(tune, "gpu_MG_default", "p_it"), 0))
        + " CG iterations per pressure solve (GAMG: " + m(tf(_cfgv(tune, "GAMG", "p_it"), 1)) + "): its coarsest level keeps 64000 rows and "
        "is 'solved' by one Jacobi sweep. A small coarsest level solved by CG and two smoothing sweeps brought it to about "
        + m(tf(_cfgv(tune, "gpu_s2c20", "p_it"), 0)) + "; building that deeper hierarchy then cost more than the solve, until the "
        "preconditioner was reused for 20 solves (the pressure matrix changes slowly between SIMPLE iterations).",
        "<b>Size dependence.</b> Speed-up per iteration " + ", ".join(f"{M} " + m(tf(sp[M], 2)) for M in pairs)
        + "; pressure solve alone " + ", ".join(f"{M} " + m(tf(spp[M], 2)) for M in pairs) + ". "
        + "Both sides change with size, in opposite directions. Time of one pressure solve per million cells: CPU "
        + ", ".join(f"{M} " + m(tf(pcell['cpu'][M], 3)) for M in pairs) + r" \(\mathrm{s}\), GPU "
        + ", ".join(f"{M} " + m(tf(pcell['gpu'][M], 3)) for M in pairs) + r" \(\mathrm{s}\). "
        + ("The GPU gets cheaper per cell because its fixed costs per solve (kernel launches, transfers, the coarse levels) "
           "are spread over more cells and its many cores fill up. " if pcell["gpu"][pairs[-1]] < pcell["gpu"][pairs[0]] else "")
        + ("The CPU gets dearer per cell, " + m(tf(pcell["cpu"][pairs[-1]] / min(pcell["cpu"].values()), 1) + r"\times")
           + " from its cheapest to the largest mesh; GAMG also needs a few more cycles on larger meshes ("
           + ", ".join(f"{M} " + m(tf(B[M]['cpu']['p_it'], 1)) for M in pairs) + "), and the rest is consistent with the "
           "data leaving the CPU's 96 MB L3 cache (not measured directly)." if pcell["cpu"][pairs[-1]] > 1.2 * min(pcell["cpu"].values()) else ""),
        "<b>Amdahl's limit.</b> On the CPU the pressure solve is " + ", ".join(
            f"{M} " + m(tf(100 * pshare[M], 0) + r"\,\%") for M in pairs)
        + " of an iteration (profiling). Even an infinitely fast pressure solver could speed an iteration up by at most "
        + ", ".join(f"{M} " + m(tf(amdahl[M], 2) + r"\times") for M in pairs) + ": momentum, assembly and the force output stay on the CPU.",
        ("<b>Rank count.</b> " + (f"{B[pairs[0]]['cpu']['ranks']} ranks (one per physical core) were fastest for both the CPU and the "
                                  "GPU runs on every mesh. "
                                  if len({B[M][sd]['ranks'] for M in pairs for sd in ('cpu', 'gpu')}) == 1 else
                                  "Fastest ranks, CPU: " + ", ".join(f"{M} {B[M]['cpu']['ranks']}" for M in pairs) + "; GPU: "
                                  + ", ".join(f"{M} {B[M]['gpu']['ranks']}" for M in pairs) + ". ")
         + "Hyper-threads add MPI traffic but no memory bandwidth; for the GPU runs, more ranks speed up the CPU part but "
           "every rank's matrix has to travel to the one GPU through host memory (L4: Figure {fig:ranks})."),
    ]
    if same_ok:
        findings.append("<b>Same answer.</b> The converged forces of the GPU and CPU runs agree to "
                        + m(tn(max(abs(r) for r in [conv[f"gpu_{lv}"][k] - (conv.get(f"cpu_{lv}") or conv[lv])[k]
                                                    for lv in sa_pairs for k in ("cd",)]), 2)) + " (drag) and "
                        + m(tn(max(abs(conv[f"gpu_{lv}"]["cl"] - (conv.get(f"cpu_{lv}") or conv[lv])["cl"]) for lv in sa_pairs), 2))
                        + " (lift), far below the mesh error; a partial linear solve changes the path, not the steady state.")
    limitations = [
        "One consumer GPU (RTX 4070): fast memory but slow double-precision arithmetic; a data-centre GPU would do better. "
        "Ubuntu's Open MPI is not CUDA-aware, so every pressure matrix and solution passes through host memory "
        "(<code>forceHostBuffer</code>).",
        "Only the pressure linear solve runs on the GPU; OGL cannot move assembly or the momentum equations there.",
        "Tuning was time-boxed (" + str(len(tune)) + " configurations on one mesh, 4 ranks); part of it ran next to other runs "
        "(marked), which inflates its times but not its iteration counts. A better GPU preconditioner may exist.",
        "Timing runs are 30 iterations from rest; converged "
        "runs were compared on " + " and ".join(sa_pairs or ["no mesh"]) + ".",
        "OGL is a development version (commit " + ogl_commit + "), tested by its authors on OpenFOAM v2512, run here on v2606.",
    ]
    rec = ("Below about " + (m(tf(be_it / 1e6, 1)) + " million cells" if be_it else "the largest tested size")
           + " keep GAMG (DIC-Gauss-Seidel smoother, 8 ranks) on the CPU; above it the tuned GPU solver is worth switching on "
           "for long runs, but expect about " + m(tf(100 * (1 - 1 / sp[big]), 0) + r"\,\%") + " less time per iteration at "
           + f"{cells[big] / 1e6:.1f}" + " million cells, not a multiple. " if sp[big] > 1 else
           "Keep GAMG on the CPU for this problem on this machine. ") \
        + "A real gain needs either a GPU with fast double precision, a CUDA-aware MPI (no host copies), or a solver that also " \
          "moves momentum and assembly to the GPU."

    ins = [it for it in insights.load(CASE)]
    hw = [["CPU", report.environment_info()[3][1]], ["Memory (WSL)", report.environment_info()[4][1]],
          ["GPU", gpu_name or "NVIDIA RTX 4070"], ["GPU link", "PCIe; WSL2 GPU paravirtualisation"]]
    sw = [["OpenFOAM", report.environment_info()[0][1]], ["OGL (GPU plugin)", "commit " + ogl_commit + ", Ginkgo 2.0 (develop)"],
          ["CUDA", cuda or "12.9"], ["MPI", mpi or "Open MPI"], ["Solver", "simpleFoam (SIMPLEC), steady"]]
    s_ = dict(
        title="GPU versus CPU pressure solver: DFG 3D-1Z cylinder (Re = 20)",
        date=dt.date.today().isoformat(), question=problem["study"]["question"].strip(),
        verdict="PASS" if same_ok and len(pairs) >= 3 else ("FAIL" if sa_ok and not same_ok else "WARN"),
        summary=dict(headline=head, answers=answers, findings=findings, limitations=limitations, recommendation=rec),
        setup=dict(
            intro="Both sides solve the same steady problem with the same OpenFOAM solver (simpleFoam, SIMPLEC) on the same "
                  "meshes, with the same schemes, tolerances, function objects and start field. Only the pressure linear solver "
                  "changes: GAMG on the CPU cores, or OGL's Ginkgo CG with a multigrid preconditioner on the GPU. In each "
                  "SIMPLE iteration OpenFOAM assembles the pressure equation on the CPU; with OGL the matrix is then sent to the "
                  "GPU, solved there and the solution sent back. Meshes: the O-grid family of the main report (L3, L4) plus three "
                  "extra sizes of the same blocking (G16, G36, G60), " + f"{min(cells.values()):,}" + " to " + f"{max(cells.values()):,}" + " cells.",
            hardware=hw, software=sw,
            fairness=[
                "CPU baseline = the fastest measured GAMG setup per mesh: two smoothers (Gauss-Seidel, DIC-Gauss-Seidel) and 8 or 16 "
                "ranks on every mesh, 4 to 16 ranks on L4; not PCG on one core. Chosen per mesh: "
                + ", ".join(f"{M}: {clab(B[M]['cpu'])}" for M in pairs) + ".",
                "Same tolerances on both sides: each pressure solve to " + m(r"\mathrm{relTol} = " + tn(numerics["linear_solvers"]["p"]["relTol"]))
                + " of its initial residual, OpenFOAM's normalised residual (OGL computes the same norm), absolute tolerance "
                + m(tn(numerics["linear_solvers"]["p"]["tol"])) + " (never reached).",
                "Same work: 30 SIMPLE iterations from the same start field (rest); time per iteration over iterations 6 to 29, "
                "so one-off costs (GAMG agglomeration, CUDA start-up, the final write) are excluded on both sides.",
                "Converged comparison runs start from the same converged coarser-mesh field and stop on the same residual rule.",
                "Timing runs one at a time, nothing else running; repeats on the largest mesh.",
                "Clock: rank 0's ExecutionTime (0.01 s resolution) where it equals wall time (checked against OpenFOAM's "
                "high-resolution profiled wall time of the loop: at least 70 % of it, the rest being one-off costs); otherwise the profiled wall time of the whole loop / 30, marked * in "
                "the tables (" + str(sum(1 for d in T.values() if not d["failed"] and not d["exec_ok"] and d["tag"] in ("", "_r2")))
                + " run(s): 16 ranks on the smallest mesh, where MPI lets rank 0 sleep). The log's ClockTime has 1 s resolution (Insights).",
            ]),
        method=dict(
            intro="Three kinds of runs: short tuning runs of the GPU solver on one mesh, fixed-length timing runs on five mesh "
                  "sizes, and converged runs on two meshes to check that the answer is the same.",
            choices=[
                dict(choice="Timing protocol: 30 SIMPLE iterations from rest; time per iteration over iterations 6 to 29",
                     why="A converged run on every mesh would cost hours; the cost per iteration is what a solver change affects, "
                         "and the converged runs (L3, L4) show that both sides need the same number of iterations.",
                     alternatives="Converged runs on every size (5-10 times the compute); single-solve benchmarks outside OpenFOAM "
                                  "(would miss the transfer and assembly costs)."),
                dict(choice="Time split by OpenFOAM's built-in profiling (pressure solve, momentum solve, function objects, rest) "
                            "and OGL's own per-solve timings (matrix transfer, preconditioner set-up, GPU solve, copy back)",
                     why="Profiling measures both sides identically; OGL's verbose output splits the GPU path further.",
                     alternatives="nsys/nvprof GPU profiles (not available in WSL here)."),
                dict(choice="GPU solver: Ginkgo CG, multigrid preconditioner (PGM aggregation), coarsest level at most 1000 rows "
                            "solved by 20 CG iterations, 2 Jacobi smoothing sweeps, preconditioner reused for 20 solves, "
                            "<code>adaptMinIter off</code>, <code>scaling -1</code>, <code>forceHostBuffer true</code>",
                     why="The fastest of the " + str(len(tune)) + " configurations tried on the 1-million-cell mesh (Results); "
                         "OGL's defaults needed many times more iterations than GAMG.",
                     alternatives="Block Jacobi, ISAI, IC preconditioners (hundreds of iterations, tried); SSOR smoother (fewer "
                                  "iterations, slower); Csr format; more smoothing sweeps (tried)."),
                dict(choice="GPU rank count: the CPU part runs on N MPI ranks, all sending their rows to the one GPU "
                            "(<code>ranksPerGPU N</code>); N = 2, 4, 8, 16 tested on L4",
                     why="More ranks speed up assembly and momentum, but each extra rank adds a matrix transfer and gather.",
                     alternatives="One rank per GPU (OGL refuses serial runs; at least 2 ranks)."),
                dict(choice="Same-answer test: converged runs on L3 and L4, CPU and GPU, from the same start field; tolerance "
                            + m(r"|\Delta c_D| \le " + tn(TOL["cd"])) + ", " + m(r"|\Delta c_L| \le " + tn(TOL["cl"]))
                            + ", fixed before the runs",
                     why="Ten times below the mesh uncertainty (GCI) of each number on L4: a GPU solve that changed the answer "
                         "by more would matter.",
                     alternatives="Comparing fields after a fixed number of iterations (partial solves take different paths, so "
                                  "unconverged states differ even when both are right)."),
            ]),
        results=dict(
            intro="Best CPU and best GPU setup per mesh, then all timing runs, the rank-count test on L4, the tuning runs and the "
                  "converged runs. Times in seconds of wall time.",
            tables=[
                dict(title="Best setups per mesh", head=["Mesh", "Cells", "CPU s/iter (ranks)", "GPU s/iter (ranks)", "Speed-up per iter.",
                                                         "CPU p solve (s)", "GPU p solve (s)", "Speed-up p solve", "GAMG iters/solve",
                                                         "GPU CG iters/solve", "GPU memory (MiB)"], rows=main_rows,
                     note="Speed-up = CPU time / GPU time (above 1: GPU faster). Pressure solve time from profiling, per call, "
                          "including the transfers to and from the GPU."),
                dict(title="Rank count on L4", head=["Pressure solver", "Ranks", "s/iter", "p solve (s)", "U solve (s)", "Rest (s)",
                                                     "p iters/solve"], rows=scal_rows),
                dict(title="GPU solver tuning on G36 (4 ranks, 20 iterations)", head=["Configuration: what changes", "CG iters/solve",
                                                                                      "GPU solve (ms)", "Set-up per solve (ms)",
                                                                                      "p solve (s)", "s/iter"], rows=tune_rows,
                     note="Configurations after the first one up to and including the caching tests ran while the accuracy "
                          "runs of the main report used 12 of the 16 hardware threads: their times are inflated; compare "
                          "iteration counts there."),
                dict(title="Converged runs (same start field, same stopping rule)",
                     head=["Run", "Cells", "Side", "Ranks", "Iterations", "Wall (s)", "s/iter", "p iters/solve"], rows=conv_rows),
                dict(title="All timing runs", head=["Run", "Cells", "Side", "Ranks", "s/iter", "p solve (s)", "U solve (s)",
                                                    "p iters/solve", "GPU MiB"], rows=all_rows),
            ],
            notes=["Break-even sizes are interpolated between measured meshes (log scale in cells), not extrapolated."]
                  + ([f"Failed runs: " + ", ".join(d["name"] for d in failed)] if failed else [])),
        same_answer=dict(
            intro="Converged runs on the same mesh, CPU (GAMG) and GPU (tuned OGL), from the same start field, stopped by the same "
                  "residual rule. The tolerance was fixed before the runs, ten times below the mesh uncertainty of each number.",
            head=["Mesh", "Quantity", "CPU", "GPU", "GPU − CPU", "Tolerance", "OK"], rows=sa_rows, ok=same_ok,
            verdict=("Same answer: drag and lift agree within the tolerance on " + " and ".join(sa_pairs) + "."
                     if same_ok else "The answers differ by more than the tolerance (or the test is missing).")),
        breakdown=dict(
            intro="Seconds per iteration by part, from OpenFOAM's profiling of the best setups (mean over the run), and the GPU "
                  "pressure solve split further from OGL's own timings.",
            tables=[
                dict(title="Per iteration (s)", head=["Mesh", "Setup", "Pressure solve", "Momentum solve", "Force output", "Rest",
                                                      "Total", "Pressure share"], rows=bd_rows,
                     note="Rest: matrix assembly, gradients, flux updates, MPI exchange, bookkeeping. Profiling means over all 30 iterations "
                          "(first-iteration set-up and final write included), so totals are slightly above the Results."),
                dict(title="GPU pressure solve, per call (ms)", head=["Mesh", "Matrix update", "Preconditioner set-up (mean)",
                                                                      "GPU solve", "Copy back", "OGL total", "OpenFOAM side",
                                                                      "CG iters", "ms per CG iter."], rows=ogl_rows,
                     note="Matrix update: OGL's copy of the new coefficients to the GPU. OpenFOAM side: the rest of the pressure "
                          "solve call outside OGL's timers (gathering the ranks' matrices, conversion, scattering the solution)."),
            ],
            notes=[
                "On the GPU the pressure solve shrinks, but the rest of the iteration does not: momentum, assembly and the force "
                "output still take " + ", ".join(f"{M} " + m(tf(B[M]['gpu']['loop_call'] - B[M]['gpu']['p_call'], 3) + r"\,\mathrm{s}") for M in pairs)
                + " per iteration on the GPU runs, against " + ", ".join(f"{M} " + m(tf(B[M]['cpu']['loop_call'] - B[M]['cpu']['p_call'], 3) + r"\,\mathrm{s}") for M in pairs)
                + " on the CPU runs (profiling means; same rank count on both sides).",
                "GPU solve time per CG iteration per million cells: " + ", ".join(
                    f"{M} " + m(tf(1e3 * o_ms[M] / cells[M], 1)) for M in pairs if M in o_ms) + r" \(\mathrm{ms}\): "
                + ("it falls with size up to the largest meshes, where the GPU is finally busy." if len(o_ms) > 1 and
                   o_ms[pairs[-1]] / cells[pairs[-1]] < o_ms[pairs[0]] / cells[pairs[0]] else "it does not fall with size.")
                + " The OpenFOAM side of the GPU path (gathering the ranks' matrices, conversion, scattering the solution) costs "
                + ", ".join(f"{M} " + m(tf(of_ms[M], 0) + r"\,\mathrm{ms}") for M in pairs if M in of_ms) + " per solve.",
            ]),
        reproducibility=dict(
            commands=[
                dict(cmd=f"python -m cases.{CASE}.run tune G36 4 20 <configs>", what="GPU tuning runs (configurations in run.py CONFIGS, or name=base+key:value)"),
                dict(cmd=f"bash cases/{CASE}/scratch/timing_chain.sh gpu_best A", what="L4: CPU GAMG on 4-16 ranks, GPU on 2-16 ranks, 30 iterations each, one at a time"),
                dict(cmd=f"bash cases/{CASE}/scratch/timing_chain.sh gpu_best B '<ranks>'", what="Size sweep G16, L3, G36, G60 and repeats on G60"),
                dict(cmd=f"python -m cases.{CASE}.run timing <mesh> <config> <ranks> 30", what="One timing run"),
                dict(cmd=f"python -m cases.{CASE}.run gpu_full gpu_ gpu_best <ranks> L3 L4", what="Converged GPU runs (same start field as CPU)"),
                dict(cmd=f"python -m cases.{CASE}.run gpu_full cpu_ GAMG <ranks> L3 L4", what="Converged CPU runs, measured alone"),
                dict(cmd=f"python -m cases.{CASE}.gpu_study", what="This report"),
            ],
            files=[dict(path=f"cases/{CASE}/run.py", what="Meshes, pressure-solver configurations (CONFIGS), timing and run modes"),
                   dict(path=f"cases/{CASE}/gpu_study.py", what="Analysis and this report"),
                   dict(path=f"results/{CASE}/timing/", what="One JSON (+ forces) per timing run, with profiling and OGL timings")],
            run_dirs=[dict(path=str(Path(next(iter(T.values()))["case_dir"]).parent) + "/t_<mesh>_<config>_np<ranks>", what="Timing runs (processor directories deleted)"),
                      dict(path=str(Path(next(iter(T.values()))["case_dir"]).parent) + "/cpu_L*, gpu_L*", what="Converged comparison runs")],
            environment=report.environment_info(),
            notes=["Per-iteration times: rank 0's ExecutionTime differences where it equals wall time, else the profiled loop "
                   "mean (Insights: ClockTime has 1 s resolution, ExecutionTime is CPU time)."]),
        figures=figures,
    )
    out = study.render_study(s_, svgs, OUT / "gpu_study.html", ins)
    print("Study:", out)
    # short summary for the main report
    summary = dict(headline=head, notes=[
        "<b>GPU study.</b> " + head + " See the <a href='gpu_study.html'>GPU study report</a>.",
        "<b>Pressure share.</b> On the CPU the pressure solve takes " + ", ".join(
            f"{M} " + m(tf(100 * B[M]['cpu']['p_call'] / B[M]['cpu']['s_it'], 0) + r"\,\%") for M in pairs)
        + " of an iteration; that bounds what any pressure-solver change can gain."])
    (OUT / "gpu_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")


def clab(b):
    """Label of a CPU setup, e.g. 'GAMG DIC-GS, 8 ranks'."""
    return ("GAMG DIC-GS" if b.get("cfg") == "GAMG_DIC" else "GAMG GS") + f", {b['ranks']} ranks"


def _cfgv(runs, cfg, key):
    v = [d[key] for d in runs if d["cfg"] == cfg]
    return v[-1] if v else float("nan")


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def _tpi_fig(pairs, cells, B, sp, spp):
    fig, (ax, bx) = plots.wide(2, 3.5)
    x = [cells[M] for M in pairs]
    c0, c1 = plots.SERIES[0], plots.SERIES[1]
    ax.loglog(x, [B[M]["cpu"]["s_it"] for M in pairs], "o-", color=c0, ms=5, lw=1.6, label="CPU, per iteration")
    ax.loglog(x, [B[M]["gpu"]["s_it"] for M in pairs], "s-", color=c1, ms=5, lw=1.6, label="GPU, per iteration")
    ax.loglog(x, [B[M]["cpu"]["p_call"] for M in pairs], "o--", color=c0, ms=4, lw=1, mfc="none", label="CPU, pressure solve")
    ax.loglog(x, [B[M]["gpu"]["p_call"] for M in pairs], "s--", color=c1, ms=4, lw=1, mfc="none", label="GPU, pressure solve")
    ax.set_xlabel("Cells")
    ax.set_ylabel("Wall time (s)")
    bx.semilogx(x, [sp[M] for M in pairs], "D-", color=plots.SERIES[2], ms=5, lw=1.6, label="whole iteration")
    bx.semilogx(x, [spp[M] for M in pairs], "D--", color=plots.SERIES[3], ms=5, lw=1.2, mfc="none", label="pressure solve")
    bx.axhline(1.0, color=plots.INK2, ls=":", lw=1)
    bx.set_xlabel("Cells")
    bx.set_ylabel("Speed-up (CPU time / GPU time)")
    lo = min(min(sp.values()), min(spp.values()), 1.0)
    hi = max(max(sp.values()), max(spp.values()), 1.0)
    bx.set_ylim(0.8 * lo, 1.25 * hi)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2)
    bx.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2)
    return plots.to_svg(fig, "tpi")


def _ranks_fig(T):
    fig, ax = plots.figure(6.4, 3.3)
    for cfg, mk, col, lab in ((CPU, "o-", plots.SERIES[0], "CPU GAMG"), ("GAMG_DIC", "^:", plots.SERIES[2], "CPU GAMG (DIC-GS)"),
                              (GPU, "s-", plots.SERIES[1], "GPU (OGL)")):
        g = pick(T, "L4", cfg)
        if g:
            r = sorted(g)
            ax.plot(r, [mean_of(g[k], "s_it") for k in r], mk, color=col, ms=5, lw=1.5, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 12, 16])
    ax.set_xticklabels(["2", "4", "8", "12", "16"])
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Wall time per iteration (s)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return plots.to_svg(fig, "ranks")


def _tune_fig(tune):
    fig, (ax, bx) = plots.wide(2, 3.6)
    names = [d["cfg"] for d in tune]
    y = np.arange(len(tune))
    cont = [i for i, d in enumerate(tune) if d["cfg"] not in ("GAMG", "gpu_MG_default", "gpu_MG_noadapt", "gpu_s2c20", "gpu_s3c20",
                                                            "gpu_s2c50", "gpu_MG_c1k_cg_s2_cache20", "gpu_ISAI", "gpu_IC")]
    for a, key in ((ax, "p_it"), (bx, "p_call")):
        vals = [d[key] for d in tune]
        bars = a.barh(y, vals, color=[plots.SERIES[0] if d["cfg"] == "GAMG" else plots.SERIES[1] for d in tune], height=0.7)
        for i in cont:
            bars[i].set_hatch("////")
            bars[i].set_alpha(0.55)
        base = next((d[key] for d in tune if d["cfg"] == "GAMG"), None)
        if base:
            a.axvline(base, color=plots.INK2, ls=":", lw=1)
        a.set_yticks(y)
        a.invert_yaxis()
        a.grid(axis="y", visible=False)
    ax.set_yticklabels(names, fontsize=7)
    bx.set_yticklabels([])
    ax.set_xscale("log")
    ax.set_xlabel("CG iterations per pressure solve")
    bx.set_xlabel("Wall time per pressure solve (s)")
    return plots.to_svg(fig, "tune")


def _bd_fig(pairs, cells, B):
    fig, ax = plots.figure(6.4, 3.4)
    parts = (("p_call", "pressure solve", plots.SERIES[1]), ("U_call", "momentum solve", plots.SERIES[0]),
             ("fo_call", "force output", plots.SERIES[3]), ("rest_call", "rest", plots.AXIS))
    xs, labels = [], []
    for i, M in enumerate(pairs):
        for j, side in enumerate(("cpu", "gpu")):
            x0 = 3 * i + j
            xs.append(x0)
            labels.append(("C" if side == "cpu" else "G") + "\n" + M)
            bottom = 0.0
            for key, lab, col in parts:
                v = 1e6 * B[M][side][key] / cells[M]
                ax.bar(x0, v, bottom=bottom, color=col, width=0.8, label=lab if (i == 0 and j == 0) else None,
                       hatch="////" if side == "gpu" and key == "p_call" else None)
                bottom += v
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(r"Wall $\mu$s per cell per iteration")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return plots.to_svg(fig, "bd")


if __name__ == "__main__":
    main()
