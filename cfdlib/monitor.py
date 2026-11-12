"""Solver-log monitoring: health checks, divergence warnings, run-time estimates and the live stop.

Use it on short pre-flight runs (catch blow-ups and bad settings before a long run), on production
runs (document that each run was healthy), and while a run goes (LiveStop: end it once the physical
check written down before the run holds).
"""
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import foam
from .report import tex_num
from .verification import iteration_error


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


# ---------------------------------------------------------------------------------------------
# Live stop: end a run as soon as a physical check holds, while it runs
# ---------------------------------------------------------------------------------------------
# A fixed end time is a guess made before the run: too short and the answer is not converged
# (gpucfd G1 on case 4: drag 16 % low after 3 s), too long and the run wastes time. The live stop
# replaces the guess with the rule the agent writes down before the run (the end time becomes a
# maximum): every few seconds the run's own output is read, the rule is applied, and once it holds
# the solver writes the current time step and ends normally.

STOP_FILE = "STOP_NOW"


def add_stop_switch(case_dir, action="writeNow"):
    """Add OpenFOAM's `abort` function object to system/controlDict (once).

    When the file <case>/STOP_NOW appears, the solver writes the current time step and ends normally,
    so fields, logs and postProcessing output are complete, as if endTime had been reached. Only the
    master rank looks for the file and tells the others, so parallel runs stop the same way. The
    function object deletes a leftover file when the run starts."""
    cd = Path(case_dir) / "system" / "controlDict"
    text = cd.read_text()
    if "liveStop" in text:
        return
    block = ("    liveStop    // cfdlib.monitor.LiveStop: write and end when <case>/" + STOP_FILE + " appears\n"
             "    {\n        type            abort;\n        libs            (utilityFunctionObjects);\n"
             f'        file            "<case>/{STOP_FILE}";\n        action          {action};\n    }}\n\n')
    m = re.search(r"^functions\s*\{[^\n]*\n", text, re.M)
    text = text[:m.end()] + block + text[m.end():] if m else text + "\nfunctions\n{\n" + block + "}\n"
    cd.write_text(text)


def read_dat_live(case_dir, fo, fname):
    """Rows of a postProcessing .dat file of a run that may still be writing it: complete lines only,
    all start-time folders (restarts) in time order."""
    def start(p):
        try:
            return float(p.parent.name)
        except ValueError:
            return math.inf
    rows = []
    for h in sorted((Path(case_dir) / "postProcessing" / fo).rglob(fname), key=start):
        text = h.read_text(errors="replace")
        rows += foam.parse_dat(text[:text.rfind("\n") + 1])
    return rows


def settled(values, tol, n=4):
    """Has a quantity stopped changing? `values`: one number per cycle (periodic flow, e.g. the drag
    maximum of each shedding cycle) or per equal chunk of time or iterations (steady flow).

    settled : the largest change between consecutive values over the last n values is below tol (the
              periodicity rule of the DFG cases), and no drift is left: if the last n changes all point
              the same way, the change still to come (geometric extrapolation at their average decay
              rate) must be below tol. A run of equal small changes in one direction is not settled.
    floor   : the changes have stopped shrinking and flip sign (the noise test of
              cfdlib.verification.iteration_error: round-off, solver tolerances, time sampling of the
              peaks), so running longer will not bring them below tol. The run can stop; the spread
              is then the quantity's iteration uncertainty, and the stop is a warning.
    running : neither yet.
    Returns dict(state, spread, remaining): remaining is the extrapolated change still to come of a
    one-way trend (0 when the changes do not all point the same way)."""
    v = np.asarray(values, float)
    if len(v) < n:
        return dict(state="running", spread=None, remaining=None)
    spread = float(np.max(np.abs(np.diff(v[-n:]))))
    it = iteration_error(v)
    # Drift: the last n changes all point the same way. Extrapolate them geometrically at their average
    # decay rate over the window (ratio of the last to the first change, per step; at least 0.5, i.e. one
    # more half-size step): a constant or growing drift has no bound (inf). iteration_error's own
    # estimate needs steady ratios, which per-cycle peaks rarely give (case 4: ratios 0.4 to 1.2 while
    # the changes shrink 10x), so it only serves the floor test here.
    d = np.diff(v[-n - 1:])
    remaining = 0.0
    if len(d) == n and len(np.unique(np.sign(d))) == 1:
        rho = max(0.5, float(abs(d[-1] / d[0])) ** (1.0 / (n - 1)))
        remaining = math.inf if rho >= 1.0 else float(abs(d[-1])) * rho / (1.0 - rho)
    drift = remaining > tol
    state = "settled" if spread < tol and not drift else "floor" if it["noise"] else "running"
    return dict(state=state, spread=spread, remaining=remaining)


def judge(quantities, tolerances, n=4):
    """Apply settled() to several quantities (dict name -> per-cycle values, name -> tol). Stop when
    every quantity is settled or at its floor; status ok if all settled, warn if any is at a floor."""
    res = {k: settled(quantities[k], tolerances[k], n) for k in tolerances}
    states = {k: r["state"] for k, r in res.items()}
    stop = all(s != "running" for s in states.values())
    floors = [k for k, s in states.items() if s == "floor"]
    if not stop:
        reason = "running: " + ", ".join(k for k, s in states.items() if s == "running")
    elif floors:
        reason = "floor reached (changes stopped shrinking) for " + ", ".join(
            f"{k} (spread {res[k]['spread']:.2g} > tol {tolerances[k]:.2g})" for k in floors)
    else:
        reason = "all settled: " + ", ".join(f"{k} {res[k]['spread']:.2g} < {tolerances[k]:.2g}" for k in res)
    return dict(stop=stop, status="warn" if floors else "ok", reason=reason, quantities=res)


class LiveStop:
    """Ends an OpenFOAM run once a physical check holds. Use: foam.run(cmd, case, watch=LiveStop(case, decide)).

    decide(case_dir) -> None (nothing to judge yet) or a dict with at least stop (bool), status
    ('ok' / 'warn') and reason, plus t (simulated time); judge() builds one. It reads the run's own
    output (read_dat_live) and applies the stopping rule written down before the run. Every decision is
    saved to <case>/live_stop.json, so the record shows when and why the run ended."""

    def __init__(self, case_dir, decide, log=print):
        self.case, self.decide, self.log = Path(case_dir), decide, log
        self.history, self.decision = [], None
        add_stop_switch(self.case)

    def __call__(self):
        if self.decision is not None:
            return
        d = self.decide(self.case)
        if d is None:
            return
        self.history.append(d)
        self.log(f"live check t = {d.get('t', float('nan')):.4g}: {d['reason']}")
        if d["stop"]:
            (self.case / STOP_FILE).write_text("action=writeNow\n")
            self.decision = d
            self.log(f"live stop requested ({d['status']}) at t = {d.get('t', float('nan')):.4g}")
        (self.case / "live_stop.json").write_text(json.dumps(self.summary(), indent=1, default=float))

    def summary(self):
        return dict(stopped=self.decision is not None, decision=self.decision, checks=len(self.history),
                    last=self.history[-1] if self.history else None)
