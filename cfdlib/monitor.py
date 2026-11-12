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
        states = {k: q["state"] for k, q in d.get("quantities", {}).items()}
        if len(self.history) == 1 or states != getattr(self, "_states", None):   # log changes only
            self.log(f"live check t = {d.get('t', float('nan')):.4g}: {d['reason']}")
        self._states = states
        if d["stop"]:
            (self.case / STOP_FILE).write_text("action=writeNow\n")
            self.decision = d
            self.log(f"live stop requested ({d['status']}) at t = {d.get('t', float('nan')):.4g}")
        (self.case / "live_stop.json").write_text(json.dumps(self.summary(), indent=1, default=float))

    def summary(self):
        return dict(stopped=self.decision is not None, decision=self.decision, checks=len(self.history),
                    last=self.history[-1] if self.history else None)


# ---------------------------------------------------------------------------------------------
# Peek: look into a run at any moment, from its first step to now
# ---------------------------------------------------------------------------------------------
# The agent calls this when IT chooses (after a minute or after hours); nothing runs between calls,
# so a long run costs no tokens while the agent is not looking. Everything comes from files the run
# writes anyway (solver log, postProcessing, live_stop.json, the PID file of foam.run), so it works
# on any OpenFOAM case, while the run goes or after it ended or crashed.

def _entry(text, key):
    m = re.search(rf"^\s*{key}\s+([^;]+);", text, re.M)
    return m.group(1).strip() if m else None


def _alive(case):
    """True / False if foam.run's PID file says whether the solver runs, None if there is no PID file."""
    import os
    pids = list(Path(case).glob("*.pid"))
    if not pids:
        return None
    for p in pids:
        try:
            os.killpg(int(p.read_text().strip()), 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            continue
    return False


def _series(case, t_from=None):
    """Every postProcessing .dat file as (label, time, {column name: values}); vectors split into x/y/z,
    columns that never change dropped."""
    out = []
    root = Path(case) / "postProcessing"
    if not root.exists():
        return out
    files = {}
    for f in root.glob("*/*/*"):                       # <function object>/<start time>/<file>
        if f.is_file() and (f.suffix == ".dat" or not f.suffix):   # .dat files and probes (U, p, ...)
            files.setdefault((f.parent.parent.name, f.name), []).append(f)
    for (fo, fname), _ in sorted(files.items()):
        rows = [r for r in read_dat_live(case, fo, fname) if r and isinstance(r[0], float)]
        if len(rows) < 2:
            continue
        first = next(iter(sorted(files[(fo, fname)])))
        hdr = [ln for ln in first.read_text(errors="replace").splitlines()[:40] if ln.startswith("#")]
        names = hdr[-1].lstrip("#").split()[1:] if hdr else []
        width = min(len(r) for r in rows)
        t = np.array([r[0] for r in rows], float)
        keep = t >= t_from if t_from is not None else np.ones(len(t), bool)
        cols = {}
        for j in range(1, width):
            base = names[j - 1] if j - 1 < len(names) else f"col{j}"
            vals = [r[j] for r in rows]
            if isinstance(vals[0], tuple):
                for c, comp in enumerate("xyz"[:len(vals[0])]):
                    cols[f"{base}_{comp}"] = np.array([v[c] for v in vals], float)
            elif isinstance(vals[0], float):
                cols[base] = np.array(vals, float)
        # drop columns that do not change beyond round-off (a fixed inflow rate) or are round-off themselves
        # (the z force of a 2D case: 1e-24 next to 1e-3)
        scale = max((float(np.max(np.abs(v))) for v in cols.values()), default=0.0)
        cols = {k: v[keep] for k, v in cols.items()
                if np.ptp(v) > 1e-8 * np.max(np.abs(v)) and np.max(np.abs(v)) > 1e-12 * scale}
        if cols:
            out.append((f"{fo}/{fname}", t[keep], cols))
    return out


def _digest(label, t, cols, windows, ncols=3):
    """Numbers, not a picture: the shown history cut into `windows` equal time windows, each column as
    min / mean / max per window. Trends, growth or decay of an oscillation, jumps and the start-up
    transient all show in the rows; an agent reads these more reliably than a figure."""
    keys = list(cols)[:ncols]
    edges = np.linspace(t[0], t[-1], windows + 1)
    head = f"{label}: {windows} windows of {edges[1] - edges[0]:.4g}, min / mean / max per window" + (
        f" (first {ncols} of {len(cols)} columns)" if len(cols) > ncols else "")
    rows = [head, "  " + f"{'t_end':>10}" + "".join(f"  {k[:29]:>29}" for k in keys)]
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t <= b)
        if not m.any():
            continue
        rows.append(f"  {b:>10.5g}" + "".join(
            f"  {np.min(cols[k][m]):>9.4g} {np.mean(cols[k][m]):>9.4g} {np.max(cols[k][m]):>9.4g}" for k in keys))
    return rows


def peek(case_dir, plot=False, t_from=None, files=None, courant_limit=1.0, windows=8, csv_dir=None):
    """Snapshot of an OpenFOAM run from its first step to now, as numbers: state, progress and time
    left, health so far, the live stopping rule's last verdict, then the history of the residuals,
    the Courant number and every monitored series (postProcessing) as min / mean / max over `windows`
    equal time windows (_digest; windows=0: one line per series). Returns dict(text, png, csv, ...).
    files: substrings selecting postProcessing series (default all); t_from: show the history from
    this time on; csv_dir: also write the selected series at full resolution, one CSV per file;
    plot: also write <case>/peek.png, a figure for a human (see watch())."""
    case = Path(case_dir)
    logs = [p for p in case.glob("log.*") if p.name.endswith("Foam")]
    if not logs:
        return dict(text=f"{case}: no solver log (log.*Foam) yet", png=None)
    log = max(logs, key=lambda p: p.stat().st_mtime)
    cd = (case / "system" / "controlDict").read_text()
    t_start, t_end = float(_entry(cd, "startTime") or 0), float(_entry(cd, "endTime") or "nan")
    parsed = parse_log(log)
    s = parsed["steps"]
    a = assess(log, end_time=t_end, courant_limit=courant_limit)
    alive = _alive(case)
    state = ("running" if alive else "crashed" if (parsed["fatal"] or parsed["fpe"]) else
             "finished" if parsed["ended"] else "not running (killed or stopped?)" if alive is False else
             "unknown (no PID file; finished runs have none)")
    lines = [f"{case.name}: {log.name}, {state}"]
    if s:
        t_now = s[-1]["time"]
        frac = (t_now - t_start) / (t_end - t_start) if t_end > t_start else float("nan")
        ex = [(x["time"], x["exec"]) for x in s if x["exec"] is not None]
        wall = ex[-1][1] if ex else float("nan")
        msg = (f"progress: t = {t_now:.6g} of {t_start:g}..{t_end:g} ({100 * frac:.1f} %), {len(s)} steps, "
               f"{wall:.0f} s of solver time")
        if state == "running" and a["est_full_runtime"] is not None:
            msg += f", about {a['est_full_runtime'] - wall:.0f} s left to endTime at the current rate"
        lines.append(msg)
    bad = [c for c in a["checks"] if c["status"] != "ok"]
    lines.append(f"health: {a['status']}" + ("" if not bad else ": " + "; ".join(
        f"{c['name']} {c['status']} ({re.sub(r'[\\\\()]|times|,', '', str(c['value'])).strip()})" for c in bad)))
    ls_path = case / "live_stop.json"
    if ls_path.exists():
        ls = json.loads(ls_path.read_text())
        last = ls.get("decision") or ls.get("last") or {}
        lines.append(f"live stop: {'stop requested' if ls.get('stopped') else 'watching'} after {ls.get('checks')} checks; "
                     f"last verdict at t = {last.get('t', float('nan')):.5g}: {last.get('reason', '-')}")
    ser = [x for x in _series(case, t_from) if not files or any(f in x[0] for f in files)]
    if windows and len(s) >= windows:
        t = np.array([x["time"] for x in s])
        keep = t >= t_from if t_from is not None else np.ones(len(t), bool)
        lc = {f"res {f}": np.array([x["res"].get(f, np.nan) for x in s])[keep]
              for f in sorted({f for x in s for f in x["res"]})}
        lc["Courant max"] = np.array([x["co"] if x["co"] is not None else np.nan for x in s])[keep]
        lc = {k: v for k, v in lc.items() if len(v) and np.isfinite(v).all()}
        if lc and keep.sum() >= windows:
            lines += _digest("solver log", t[keep], lc, windows, ncols=4)
    for label, t, cols in ser:
        if len(t) < 4:
            continue
        if windows and len(t) >= windows:
            lines += _digest(label, t, cols, windows)
        else:
            late = t >= t[-1] - 0.25 * (t[-1] - t[0])
            parts = [f"{k} {v[-1]:.5g} [{v[late].min():.5g}, {v[late].max():.5g}]" for k, v in list(cols.items())[:3]]
            lines.append(f"{label}: " + "; ".join(parts) + (f"; +{len(cols) - 3} columns" if len(cols) > 3 else "")
                         + "  (last value [min, max over the last quarter])")
    csvs = []
    if csv_dir:
        out = Path(csv_dir)
        out.mkdir(parents=True, exist_ok=True)
        for label, t, cols in ser:
            f = out / (label.replace("/", "_").replace(".dat", "") + ".csv")
            np.savetxt(f, np.column_stack([t] + list(cols.values())), delimiter=",", fmt="%.10g",
                       header="t," + ",".join(cols), comments="")
            csvs.append(str(f))
        lines.append(f"raw series (CSV, full resolution): {', '.join(csvs)}")
    png = None
    if plot and (s or ser):
        png = _peek_plot(case, log.name, s, ser, t_from)
        lines.append(f"figure: {png}")
    return dict(text="\n".join(lines), png=str(png) if png else None, csv=csvs, state=state, health=a["status"])


def watch(case_dir, every=60.0, **kw):
    """For a human engineer: refresh <case>/peek.png and <case>/peek.html (the figure plus the text,
    reloading itself every `every` seconds in a browser) until the run is no longer running. Prints
    the page's path once. The agent does not need this: it calls peek() when it chooses."""
    import html
    import os
    import time as _time
    case = Path(case_dir).resolve()
    page = case / "peek.html"
    distro = os.environ.get("WSL_DISTRO_NAME")
    where = ("\\\\wsl.localhost\\" + distro + str(page).replace("/", "\\")) if distro else str(page)
    print(f"open in a browser (reloads every {every:g} s): {where}", flush=True)
    while True:
        r = peek(case, plot=True, **kw)
        name, state = html.escape(case.name), html.escape(str(r.get("state", "?")))
        page.write_text(
            f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="{every:g}">'
            f"<title>{name}: {state}</title>"
            '<body style="font-family:sans-serif;background:#fcfcfb;color:#0b0b0b;margin:16px">'
            f"<h3>{name}: {state}, updated {_time.strftime('%H:%M:%S')}</h3>"
            f'<img src="peek.png?{int(_time.time())}" style="max-width:100%">'
            f'<pre style="font-size:12px;overflow-x:auto">{html.escape(r["text"])}</pre></body>')
        if r.get("state") != "running":
            print(f"run {r.get('state')}; page left at its last state", flush=True)
            return r
        _time.sleep(every)


def _peek_plot(case, title, steps, ser, t_from):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = []
    if steps:
        t = np.array([x["time"] for x in steps])
        keep = t >= t_from if t_from is not None else np.ones(len(t), bool)
        res = {f: np.array([x["res"].get(f, np.nan) for x in steps])[keep] for f in sorted({f for x in steps for f in x["res"]})}
        panels.append(("initial residuals", t[keep], res, True))
        co = np.array([x["co"] if x["co"] is not None else np.nan for x in steps])[keep]
        if np.isfinite(co).any():
            panels.append(("max Courant number", t[keep], {"Co": co}, False))
    for label, t, cols in ser:
        panels.append((label, t, dict(list(cols.items())[:3]), False))
    nc = 3
    nr = -(-len(panels) // nc)
    fig, axs = plt.subplots(nr, nc, figsize=(12, 2.3 * nr), squeeze=False)
    for ax, (label, t, cols, logy) in zip(axs.flat, panels):
        step = max(1, len(t) // 4000)                    # at most ~4000 points per line
        for k, v in cols.items():
            ax.plot(t[::step], v[::step], lw=0.8, label=k)
        if logy:
            ax.set_yscale("log")
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="best")
    for ax in list(axs.flat)[len(panels):]:
        ax.axis("off")
    span = "whole history" if t_from is None else f"history from t = {t_from:g}"
    fig.suptitle(f"{case.name} ({title}): {span}", fontsize=10)
    fig.tight_layout()
    png = case / "peek.png"
    fig.savefig(png, dpi=70)
    plt.close(fig)
    return png


if __name__ == "__main__":
    # python -m cfdlib.monitor peek  <case dir> [--files forcesCyl,probes] [--from T] [--windows N] [--csv DIR] [--plot]
    # python -m cfdlib.monitor watch <case dir> [--every S] [--files ...]        (a live page for a human)
    import argparse
    ap = argparse.ArgumentParser(description="Look into an OpenFOAM run: progress, health, live-stop verdict, history")
    ap.add_argument("cmd", choices=["peek", "watch"])
    ap.add_argument("case")
    ap.add_argument("--from", dest="t_from", type=float)
    ap.add_argument("--files", help="comma-separated substrings of postProcessing series to show")
    ap.add_argument("--windows", type=int, default=8, help="time windows of the history table (0: one line per series)")
    ap.add_argument("--csv", dest="csv_dir", help="also write the selected series in full as CSV files to this directory")
    ap.add_argument("--courant-limit", type=float, default=1.0)
    ap.add_argument("--plot", action="store_true", help="also write <case>/peek.png (a figure for a human)")
    ap.add_argument("--every", type=float, default=60.0, help="watch: seconds between page updates")
    a = ap.parse_args()
    kw = dict(t_from=a.t_from, files=a.files.split(",") if a.files else None, courant_limit=a.courant_limit,
              windows=a.windows)
    if a.cmd == "watch":
        watch(a.case, every=a.every, **kw)
    else:
        print(peek(a.case, plot=a.plot, csv_dir=a.csv_dir, **kw)["text"])
