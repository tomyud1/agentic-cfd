"""Solver-log monitoring: health checks, divergence warnings and run-time estimates.

Use it on short pre-flight runs (catch blow-ups and bad settings before a long run) and on
production runs (document that each run was healthy).
"""
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import foam
from .report import tex_num


def _ratio(x):
    """Trend ratio as report math, e.g. ×7.5e-8 -> LaTeX inside \\( \\)."""
    return r"\(\times\," + tex_num(x, 3) + r"\)"


def parse_log(path):
    """Per time step/iteration: time, max Courant, first initial residual per field, local continuity
    error, cumulative execution time. Plus flags for fatal errors."""
    text = Path(path).read_text(errors="replace")
    steps, cur = [], None
    for line in text.splitlines():
        if line.startswith("Time = "):
            if cur:
                steps.append(cur)
            try:
                t = float(line.split("=", 1)[1].strip().rstrip("s"))
            except ValueError:
                t = float("nan")
            cur = dict(time=t, res={}, co=None, cont=None, exec=None)
        elif cur is None:
            continue
        elif "Courant Number mean" in line:
            cur["co"] = float(line.split()[-1])
        elif "Solving for" in line:
            m = re.search(r"Solving for (\w+), Initial residual = ([^,]+)", line)
            if m and m.group(1) not in cur["res"]:
                cur["res"][m.group(1)] = float(m.group(2))
        elif "time step continuity errors" in line:
            m = re.search(r"sum local = ([^,]+)", line)
            if m and cur["cont"] is None:
                cur["cont"] = float(m.group(1))
        elif line.startswith("ExecutionTime"):
            cur["exec"] = float(line.split("=")[1].split()[0])
    if cur:
        steps.append(cur)
    return dict(
        steps=steps,
        fatal="FOAM FATAL" in text,
        # Match the crash itself: the startup banner always says
        # "trapFpe: Floating point exception trapping enabled", which is not an error.
        fpe="sigFpe::sigHandler" in text or bool(re.search(r"Floating point exception\s*\(core dumped\)", text)),
        nan=bool(re.search(r"\bnan\b", text, re.I)),
        ended=bool(re.search(r"^End\s*$", text, re.M)),
    )


# Below these absolute levels a quantity is round-off noise and its trend means nothing
# (e.g. continuity errors of 1e-28 -> 1e-19 in a flow with v = 0 exactly; case 3).
ROUNDOFF = {"residual": 1e-10, "continuity": 1e-12}


def _trend(values, floor=0.0):
    """Ratio of the median of the last quarter to the median of the first quarter (skipping step 1).

    Returns (ratio, late_level); ratio is None if there are too few values or the late level is
    below `floor` (round-off: no meaningful trend)."""
    v = np.array([x for x in values[1:] if x is not None and np.isfinite(x) and x > 0])
    if len(v) < 8:
        return None, None
    q = max(2, len(v) // 4)
    late = float(np.median(v[-q:]))
    if late < floor:
        return None, late
    return late / float(np.median(v[:q])), late


def assess(log_path, end_time=None, courant_limit=1.0, residual_targets=None):
    """Health checks for a solver log. Status: ok / warn / fail.

    end_time: the full run's end time, for a run-time projection (est_full_runtime, seconds).
    residual_targets: steady runs that stop on residualControl, e.g. {"Ux": 1e-6, "p": 1e-5}. The
      projection is then made from the residual decay rate instead of end_time, and is only a LOWER
      BOUND: SIMPLE converges fast at first and slowly later, and more slowly on finer meshes. For the
      finest mesh, extrapolate the iteration counts of the coarser production runs instead.
    """
    log = parse_log(log_path)
    s = log["steps"]
    checks = []

    def add(name, status, value, meaning):
        checks.append(dict(name=name, status=status, value=value, meaning=meaning))

    add("Solver finished without errors", "fail" if (log["fatal"] or log["fpe"]) else "ok",
        "fatal error" if log["fatal"] else "floating-point exception" if log["fpe"] else "no errors",
        "A crash or floating-point exception usually means the solution blew up or a setting is invalid.")
    add("No NaN values", "fail" if log["nan"] else "ok", "NaN found" if log["nan"] else "none",
        "NaN (not a number) means the calculation produced invalid values.")
    co = [x["co"] for x in s if x["co"] is not None]
    if co:
        co_max = max(co)
        add("Max Courant number", "warn" if co_max > courant_limit else "ok", f"{co_max:.3g}",
            f"Fluid should not cross more than about one cell per time step (limit used: {courant_limit}).")
        tr, _ = _trend(co)
        if tr is not None:
            add("Courant number trend", "fail" if tr > 5 else "ok", f"{_ratio(tr)} (late vs early)",
                "A steadily growing Courant number is the typical signature of a blow-up.")
    fields = sorted({f for x in s for f in x["res"]})
    for f in fields:
        tr, late = _trend([x["res"].get(f) for x in s], ROUNDOFF["residual"])
        if tr is None and late is not None:
            add(f"Residual trend: {f}", "ok", r"round-off level (\(< 10^{-10}\))",
                "The residual is at round-off level, so its trend carries no information.")
        elif tr is not None:
            add(f"Residual trend: {f}", "warn" if tr > 10 else "ok", f"{_ratio(tr)} (late vs early)",
                "Residuals should fall or level off. Growth suggests divergence (or a near-zero field "
                "whose normalised residual is only noise).")
    cont = [x["cont"] for x in s if x["cont"] is not None]
    tr, late = _trend(cont, ROUNDOFF["continuity"])
    if tr is None and late is not None:
        add("Continuity-error trend", "ok", r"round-off level (\(< 10^{-12}\))",
            "The continuity error is at round-off level, so its trend carries no information.")
    elif tr is not None:
        add("Continuity-error trend", "fail" if tr > 100 else "ok", f"{_ratio(tr)} (late vs early)",
            "Mass-conservation error per step; strong growth means the solution is degrading.")

    est = None
    ex = [(x["time"], x["exec"]) for x in s if x["exec"] is not None]
    if len(ex) >= 2 and end_time:
        half = ex[len(ex) // 2:] if len(ex) >= 4 else ex
        dt_sim = (half[-1][0] - half[0][0])
        dt_wall = (half[-1][1] - half[0][1])
        if dt_sim > 0:
            est = dt_wall / dt_sim * (end_time - ex[-1][0]) + ex[-1][1]
    est_kind = "fixed run length" if est is not None else None
    if residual_targets and len(ex) >= 2:
        need = []
        for f, target in residual_targets.items():
            r = np.array([x["res"].get(f, np.nan) for x in s], dtype=float)
            idx = np.nonzero(np.isfinite(r) & (r > 0))[0]
            idx = idx[len(idx) // 2:]                              # decay rate over the second half
            if len(idx) >= 4:
                slope = np.polyfit(idx, np.log10(r[idx]), 1)[0]    # decades per iteration
                need.append(len(s) + (np.log10(target) - np.log10(r[idx[-1]])) / slope if slope < 0 else np.inf)
        if need:
            per_step = (ex[-1][1] - ex[0][1]) / max(1, len(ex) - 1)
            est_iter = max(need)
            finite = bool(np.isfinite(est_iter))
            est = per_step * est_iter if finite else None
            est_kind = "lower bound (steady: early residual decay rate)" if finite else "residuals not falling"
            checks.append(dict(name="Projected iterations to residual targets", status="ok" if finite else "warn",
                               value=r"\(\ge " + f"{est_iter:.0f}" + r"\)" if finite else "not falling",
                               meaning="Extrapolated from the decay rate of the short run; real runs usually need more."))
    status = "fail" if any(c["status"] == "fail" for c in checks) else \
             "warn" if any(c["status"] == "warn" for c in checks) else "ok"
    return dict(status=status, checks=checks, steps=len(s), last_time=s[-1]["time"] if s else None,
                wall_time=ex[-1][1] if ex else None, est_full_runtime=est, est_kind=est_kind)


def preflight(case_dir, solver, end_time_short, end_time_full, timeout=600, courant_limit=1.0,
              residual_targets=None):
    """Run a short copy of a prepared case (mesh already built) and assess it.

    Copies case_dir to <case_dir>_preflight (the prepared case itself is not touched), sets endTime
    to end_time_short, runs the solver with a timeout, and returns assess() plus a projected run
    time for end_time_full (see assess() for residual_targets on steady runs).
    """
    case_dir = Path(case_dir)
    pf = case_dir.parent / (case_dir.name + "_preflight")
    if pf.exists():
        shutil.rmtree(pf)
    shutil.copytree(case_dir, pf, ignore=shutil.ignore_patterns("log.*", "postProcessing", "processor*"))
    foam.run(f"foamDictionary -entry endTime -set {end_time_short} system/controlDict", pf, "setEndTime")
    timed_out = False
    try:
        foam.run(solver, pf, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    except RuntimeError:
        pass                                   # a crash is a finding; assess() reports it
    a = assess(pf / f"log.{solver.split()[0]}", end_time=end_time_full, courant_limit=courant_limit,
               residual_targets=residual_targets)
    a.update(case=str(pf), end_time_short=end_time_short, timed_out=timed_out)
    return a
