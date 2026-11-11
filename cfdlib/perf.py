"""Performance data of OpenFOAM runs, for optimisation (CPU cores, MPI, later GPU).

foam.run() already writes <case>/perf.<name>.json for every command: wall time, CPU time of the
child processes (user + system) and the peak memory of the largest child. summarize() adds what
the solver log says and normalises it, so runs of different sizes can be compared:

    from cfdlib import perf
    p = perf.summarize(case_dir, "pimpleFoam", cells=52000, cores=4)

Key numbers:
  us_per_cell_step   wall microseconds per cell per time step (or iteration). Lower is better; compare
                     across meshes and core counts to see scaling.
  cpu_efficiency     CPU time / (wall time x cores). Near 1: all cores busy. Low: waiting (I/O, MPI).
  lin_iters          mean linear-solver iterations per solve, per field (pressure usually dominates;
                     this is what a GPU linear solver would accelerate).
  write_share        fraction of wall time not spent in the solver loop (start-up, writing, function objects).
  lin_iters_by_solve per field, mean iterations of the 1st, 2nd, ... solve within a time step and of the
                     last one (PISO/PIMPLE solve p several times per step; v2606 names them all "p").
"""
import json
import re
from pathlib import Path

import numpy as np


def machine():
    """Cores, memory and MPI of this machine (for the performance appendix)."""
    import os
    import shutil
    mem = None
    try:
        kb = int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal")))
        mem = kb / 1024 ** 2
    except (OSError, StopIteration):
        pass
    gpu = "none found"
    if shutil.which("nvidia-smi"):
        import subprocess
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20).stdout.strip() or gpu
        except (OSError, subprocess.SubprocessError):
            pass
    return dict(logical_cores=os.cpu_count(), physical_cores=os.cpu_count() // 2 if os.cpu_count() else None,
                memory_gb=mem, mpirun=bool(shutil.which("mpirun")) or Path("/usr/bin/mpirun").exists(), gpu=gpu)


def profiling(case_dir, cores=1):
    """OpenFOAM's built-in profiling (controlDict: profiling { active true; }), from the last written time
    of the case (processor0 for parallel runs): {section description: dict(total=wall s, calls=n)}.

    High-resolution wall time per section, e.g. "fvMatrix::solve.p" (the pressure solve) and
    "fvMatrix::solve.U". Use it for timing: the log's ClockTime has 1 s resolution and ExecutionTime is
    rank 0's CPU time. GAMG's "lduMatrix::solver.coarsestLevelCorr" entry is a nesting artefact; use
    fvMatrix::solve.<field> for solve times.
    """
    from cfdlib import foam
    root = Path(case_dir) / "processor0" if cores > 1 else Path(case_dir)
    tdirs = [d for d in foam.time_dirs(root) if (d / "uniform" / "profiling").exists()]
    out = {}
    if not tdirs:
        return out
    text = (tdirs[-1] / "uniform" / "profiling").read_text(errors="replace")
    for blk in re.findall(r"\{([^{}]*)\}", text):
        dsc = re.search(r'description\s+"([^"]+)"', blk)
        tot = re.search(r"totalTime\s+([\d.eE+-]+);", blk)
        cnt = re.search(r"calls\s+(\d+);", blk)
        if dsc and tot:
            out[dsc.group(1)] = dict(total=float(tot.group(1)), calls=int(cnt.group(1)) if cnt else None)
    return out


def summarize(case_dir, log_name, cells, cores=1):
    """Normalised performance numbers for one run (see module docstring)."""
    case_dir = Path(case_dir)
    log = (case_dir / f"log.{log_name}").read_text(errors="replace")
    rec = {}
    pj = case_dir / f"perf.{log_name}.json"
    if pj.exists():
        rec = json.loads(pj.read_text())
    steps = len(re.findall(r"^Time = ", log, re.M))
    ex = [float(x) for x in re.findall(r"^ExecutionTime = ([\d.eE+-]+) s", log, re.M)]
    ck = [float(x) for x in re.findall(r"ClockTime = ([\d.eE+-]+) s", log)]
    iters, by_pos = {}, {}
    solve = re.compile(r"Solving for (\w+), Initial residual = [^,]+, Final residual = [^,]+, No Iterations (\d+)")
    # Per time step, number the solves of each field by position (1st, 2nd, ... last). v2606 logs every
    # pressure solve as "p" (no "pFinal"), so the final solve of a step is found by position, not by name.
    for block in re.split(r"^Time = ", log, flags=re.M)[1:]:
        seen = {}
        for f, n in solve.findall(block):
            iters.setdefault(f, []).append(int(n))
            seen.setdefault(f, []).append(int(n))
        for f, v in seen.items():
            for k, n in enumerate(v):
                by_pos.setdefault(f, {}).setdefault(k, []).append(n)
            by_pos.setdefault(f, {}).setdefault("last", []).append(v[-1])
    wall = rec.get("wall_s") or (ck[-1] if ck else None)
    loop = (ck[-1] - ck[0]) if len(ck) >= 2 else None
    out = dict(
        case=str(case_dir), solver=log_name, cells=int(cells), cores=int(cores), steps=steps,
        wall_s=wall, cpu_s=rec.get("cpu_s"), solver_cpu_s=ex[-1] if ex else None, peak_mem_mb=rec.get("peak_mem_mb"),
        us_per_cell_step=(1e6 * wall / (cells * steps)) if wall and steps else None,
        cpu_efficiency=(rec["cpu_s"] / (wall * cores)) if rec.get("cpu_s") and wall else None,
        write_share=(1 - loop / wall) if loop is not None and wall else None,
        lin_iters={f: float(np.mean(v)) for f, v in iters.items() if sum(v) > 0},
        lin_solves_per_step={f: len(v) / steps for f, v in iters.items() if steps and sum(v) > 0},
        # mean iterations of the 1st, 2nd, ... solve of a field within a step, and of the last one
        lin_iters_by_solve={f: {("last" if k == "last" else f"solve {k + 1}"): float(np.mean(v)) for k, v in d.items()}
                            for f, d in by_pos.items() if f in iters and sum(iters[f]) > 0},
    )
    return out
