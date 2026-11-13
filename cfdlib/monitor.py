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
    return parse_log_text(Path(path).read_text(errors="replace"))


def parse_log_text(text):
    """parse_log on the text of a log, or of a piece of it that starts at a 'Time =' line (peek's sync)."""
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
    v = np.asarray(values, dtype=float)[1:]               # None -> nan
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 8:
        return None, None
    q = max(2, len(v) // 4)
    late = float(np.median(v[-q:]))
    if late < floor:
        return None, late
    return late / float(np.median(v[:q])), late


def log_arrays(steps):
    """parse_log steps as arrays (nan where a step has no value): time, co, cont, exec, res{field}."""
    def col(k):
        return np.array([x[k] for x in steps], dtype=float)
    fields = sorted({f for x in steps for f in x["res"]})
    return dict(time=col("time"), co=col("co"), cont=col("cont"), exec=col("exec"),
                res={f: np.array([x["res"].get(f) for x in steps], dtype=float) for f in fields})


def assess(log_path, end_time=None, courant_limit=1.0, residual_targets=None):
    """Health checks for a solver log. Status: ok / warn / fail.

    end_time: the full run's end time, for a run-time projection (est_full_runtime, seconds).
    residual_targets: steady runs that stop on residualControl, e.g. {"Ux": 1e-6, "p": 1e-5}. The
      projection is then made from the residual decay rate instead of end_time, and is only a LOWER
      BOUND: SIMPLE converges fast at first and slowly later, and more slowly on finer meshes. For the
      finest mesh, extrapolate the iteration counts of the coarser production runs instead.
    """
    log = parse_log(log_path)
    return assess_arrays(log_arrays(log["steps"]), log, end_time, courant_limit, residual_targets)


def assess_arrays(arr, flags, end_time=None, courant_limit=1.0, residual_targets=None):
    """assess() on a log history held as arrays (log_arrays, or peek's solver_log.csv) plus the error
    flags fatal / fpe / nan of parse_log, so a long run need not be re-parsed from its text."""
    n = len(arr["time"])
    checks = []

    def add(name, status, value, meaning):
        checks.append(dict(name=name, status=status, value=value, meaning=meaning))

    add("Solver finished without errors", "fail" if (flags["fatal"] or flags["fpe"]) else "ok",
        "fatal error" if flags["fatal"] else "floating-point exception" if flags["fpe"] else "no errors",
        "A crash or floating-point exception usually means the solution blew up or a setting is invalid.")
    add("No NaN values", "fail" if flags["nan"] else "ok", "NaN found" if flags["nan"] else "none",
        "NaN (not a number) means the calculation produced invalid values.")
    co = arr["co"][np.isfinite(arr["co"])]
    if co.size:
        co_max = float(co.max())
        add("Max Courant number", "warn" if co_max > courant_limit else "ok", f"{co_max:.3g}",
            f"Fluid should not cross more than about one cell per time step (limit used: {courant_limit}).")
        tr, _ = _trend(co)
        if tr is not None:
            add("Courant number trend", "fail" if tr > 5 else "ok", f"{_ratio(tr)} (late vs early)",
                "A steadily growing Courant number is the typical signature of a blow-up.")
    for f in sorted(arr["res"]):
        tr, late = _trend(arr["res"][f], ROUNDOFF["residual"])
        if tr is None and late is not None:
            add(f"Residual trend: {f}", "ok", r"round-off level (\(< 10^{-10}\))",
                "The residual is at round-off level, so its trend carries no information.")
        elif tr is not None:
            add(f"Residual trend: {f}", "warn" if tr > 10 else "ok", f"{_ratio(tr)} (late vs early)",
                "Residuals should fall or level off. Growth suggests divergence (or a near-zero field "
                "whose normalised residual is only noise).")
    tr, late = _trend(arr["cont"][np.isfinite(arr["cont"])], ROUNDOFF["continuity"])
    if tr is None and late is not None:
        add("Continuity-error trend", "ok", r"round-off level (\(< 10^{-12}\))",
            "The continuity error is at round-off level, so its trend carries no information.")
    elif tr is not None:
        add("Continuity-error trend", "fail" if tr > 100 else "ok", f"{_ratio(tr)} (late vs early)",
            "Mass-conservation error per step; strong growth means the solution is degrading.")

    est = None
    m = np.isfinite(arr["exec"])
    ext, exe = arr["time"][m], arr["exec"][m]
    if len(ext) >= 2 and end_time:
        h = len(ext) // 2 if len(ext) >= 4 else 0
        dt_sim, dt_wall = ext[-1] - ext[h], exe[-1] - exe[h]
        if dt_sim > 0:
            est = float(dt_wall / dt_sim * (end_time - ext[-1]) + exe[-1])
    est_kind = "fixed run length" if est is not None else None
    if residual_targets and len(ext) >= 2:
        need = []
        for f, target in residual_targets.items():
            r = arr["res"].get(f, np.full(n, np.nan))
            idx = np.nonzero(np.isfinite(r) & (r > 0))[0]
            idx = idx[len(idx) // 2:]                              # decay rate over the second half
            if len(idx) >= 4:
                slope = np.polyfit(idx, np.log10(r[idx]), 1)[0]    # decades per iteration
                need.append(n + (np.log10(target) - np.log10(r[idx[-1]])) / slope if slope < 0 else np.inf)
        if need:
            per_step = (exe[-1] - exe[0]) / max(1, len(exe) - 1)
            est_iter = max(need)
            finite = bool(np.isfinite(est_iter))
            est = per_step * est_iter if finite else None
            est_kind = "lower bound (steady: early residual decay rate)" if finite else "residuals not falling"
            checks.append(dict(name="Projected iterations to residual targets", status="ok" if finite else "warn",
                               value=r"\(\ge " + f"{est_iter:.0f}" + r"\)" if finite else "not falling",
                               meaning="Extrapolated from the decay rate of the short run; real runs usually need more."))
    status = "fail" if any(c["status"] == "fail" for c in checks) else \
             "warn" if any(c["status"] == "warn" for c in checks) else "ok"
    return dict(status=status, checks=checks, steps=n, last_time=float(arr["time"][-1]) if n else None,
                wall_time=float(exe[-1]) if len(exe) else None, est_full_runtime=est, est_kind=est_kind)


def preflight(case_dir, solver, end_time_short, end_time_full, timeout=600, courant_limit=1.0,
              residual_targets=None):
    """Run a short copy of a prepared case (mesh already built) and assess it.

    Copies case_dir to <case_dir>_preflight (the prepared case itself is not touched), sets endTime
    to end_time_short, runs the solver with a timeout, and returns assess() plus a projected run
    time for end_time_full (see assess() for residual_targets on steady runs).

    solver: the command as you would run it, e.g. "simpleFoam" or "mpirun -np 8 simpleFoam -parallel".
    A parallel command needs the case decomposed already (decomposePar): its processor folders are
    then copied too. A serial command leaves them out.
    """
    case_dir = Path(case_dir)
    pf = case_dir.parent / (case_dir.name + "_preflight")
    if pf.exists():
        shutil.rmtree(pf)
    parallel = "-parallel" in solver.split()
    if parallel and not any(case_dir.glob("processor*")):
        raise ValueError(f"preflight: '{solver}' is a parallel command, but {case_dir} has no processor "
                         "folders. Run decomposePar on the prepared case first")
    skip = ("log.*", "postProcessing") if parallel else ("log.*", "postProcessing", "processor*")
    shutil.copytree(case_dir, pf, ignore=shutil.ignore_patterns(*skip))
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
    for w in foam.ignored_settings(pf):
        a["checks"].append(dict(name="Settings the solver ignores", status="warn", value="ignored", meaning=w))
        if a["status"] == "ok":
            a["status"] = "warn"
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


class DatTail:
    """Incremental reader of a postProcessing .dat file that a running solver keeps writing. Each call
    returns every row so far as one float array (time first, vectors flattened), but reads and parses
    only the bytes added since the previous call, so a live check every few seconds costs the same at
    step 1000 as at step 1000000. Start-time folders (restarts) are read in time order; only complete
    lines are taken; rows of another width (a header change) are skipped."""

    def __init__(self, case_dir, fo, fname):
        self.root, self.fname = Path(case_dir) / "postProcessing" / fo, fname
        self.offsets, self.width = {}, None
        self.buf, self.n = np.empty((0, 0)), 0

    def __call__(self):
        def start(p):
            try:
                return float(p.parent.name)
            except ValueError:
                return math.inf
        new = []
        for src in sorted(self.root.glob(f"*/{self.fname}"), key=start):
            off = self.offsets.get(src, 0)
            with open(src, "rb") as fh:
                fh.seek(off)
                data = fh.read()
            data = data[:data.rfind(b"\n") + 1]
            self.offsets[src] = off + len(data)
            for r in foam.parse_dat(data.decode(errors="replace")):
                if not r or not isinstance(r[0], float):
                    continue
                flat = [x for v in r for x in (v if isinstance(v, tuple) else (v,)) if isinstance(x, float)]
                if self.width is None:
                    self.width = len(flat)
                if len(flat) == self.width:
                    new.append(flat)
        if new:
            if self.n + len(new) > len(self.buf):                # grow by doubling: appends cost O(new)
                buf = np.empty((max(2 * len(self.buf), self.n + len(new), 1024), self.width))
                if self.n:
                    buf[:self.n] = self.buf[:self.n]
                self.buf = buf
            self.buf[self.n:self.n + len(new)] = new
            self.n += len(new)
        return self.buf[:self.n]


# ---------------------------------------------------------------------------------------------
# Steady runs: stop on the iteration error, not on residuals
# ---------------------------------------------------------------------------------------------
# A steady SIMPLE run approaches its answer geometrically, so the error still left in a monitored
# value is about |last change| * rho / (1 - rho) (verification.iteration_error). Stopping when that
# estimate is below a tolerance chosen from the physics (a small fraction of the mesh error, say)
# replaces residual targets, which say nothing about the error of the answer (primer pitfall 10).
# Replayed on case 2's four recorded meshes (16 combinations of tolerance and sampling) the true
# remaining error was always below the estimate, at 0.6 to 0.97 of the tolerance.

def _fallback_error(s, m=5, safety=10.0):
    """Remaining change of a sampled series when iteration_error has no estimate (inf). Two conservative
    cases, else inf (case 6, insights 9 and 15: values written with 12 digits stop changing smoothly
    near convergence, so the change ratios scatter or a change is exactly zero, and the fitted rate is
    never accepted although the value has long converged):
    - block bound: the total change (sum of |changes|) in each of the last three blocks of m samples
      shrinks twice in a row; at the slower of the two block ratios r, what is left is at most
      S * r / (1 - r) (S the last block's total). Sums, not maxima, so a drift of one or two units of the
      last printed digit per sample still shows its decay. The result is multiplied by `safety` (10)
      before the caller compares it with the tolerance, so a fast mode dying out cannot fake convergence.
    - output precision: the last 2m changes are at most a few units of the smallest change the file
      resolves (q, the quantum of the printed digits), they do not all point one way (no drift), and the
      series has moved by more than 1000 q since the start (a probe the solution has not reached yet
      does not count). Error = 10 q.
    Returns (error, how)."""
    d = np.diff(s)
    a = np.abs(d)
    if len(d) < 3 * m:
        return math.inf, None
    s1, s2, s3 = a[-3 * m:-2 * m].sum(), a[-2 * m:-m].sum(), a[-m:].sum()
    if 0.0 < s3 < s2 < s1:
        r = max(s3 / s2, s2 / s1)
        return float(s3 * r / (1.0 - r)) * safety, "block bound"
    nz = a[-6 * m:][a[-6 * m:] > 0]
    q = float(nz.min()) if len(nz) else 0.0
    tail = d[-2 * m:]
    one_way = np.all(tail > 0) or np.all(tail < 0)
    if q > 0 and a[-2 * m:].max() <= 10 * q and not one_way \
            and float(np.max(s) - np.min(s)) > 1000 * q:
        return 10 * q, "output precision"
    return math.inf, None


def _cycle(x):
    """Period (2 to 6 rows) of a regular cycle in the rows x, or None: the values repeat after p rows
    ten times more closely than they follow each other. Random scatter (a noise floor) does not: its
    values p rows apart differ as much as neighbours do."""
    x = np.asarray(x, float)
    step = np.max(np.abs(np.diff(x))) if len(x) > 1 else 0.0
    for p in range(2, 7):
        if step > 0 and len(x) > 2 * p and np.max(np.abs(x[p:] - x[:-p])) < 0.1 * step:
            return p
    return None


def steady_check(values, tol, every=10, names=None):
    """Iteration error of each monitored series of a steady run, now.

    values: (n,) or (n, k) array, one row per iteration (or per equal step). tol: scalar or one per
    column, in the units of values. every: sampling interval in rows; the changes between samples must
    shrink visibly (if the error never settles, sample more sparsely). Returns dict(ok, status, reason,
    quantities): per column state 'settled' (error < tol), 'floor' (changes stopped shrinking above
    tol: running longer will not help, so it counts as done with a warning), 'running', or 'oscillating'
    (consecutive rows repeat in a cycle of 2 to 6 rows with steps above tol, whatever the samples say:
    the iteration flips between states, so the run is NOT converged and the check does not hold; status
    'warn'). 'settled' also needs the largest change between consecutive rows, over the last `every`
    rows, to be below tol. These guards see a cycle only if `values` has every iteration: a function
    object that writes every n-th iteration hides a cycle of period n (write every iteration).

    The error is iteration_error's geometric estimate. While the decay is visibly slowing down (the
    fitted rate three samples ago was clearly faster: a fast mode dying out and a slower one taking
    over), that estimate is too small, so it is multiplied by 10 (a trap test with two modes stopped
    18 times too early without this). Where there is no estimate (rounded output near convergence), a
    conservative fallback is tried (_fallback_error: a block bound with the same factor of 10, or the
    output precision of a value that has stopped drifting); `how` says which was used."""
    v = np.asarray(values, float)
    v = v[:, None] if v.ndim == 1 else v
    k = v.shape[1]
    tols = np.broadcast_to(np.asarray(tol, float), (k,))
    names = list(names) if names is not None else [f"c{j}" for j in range(k)]
    s = v[::-1][::every][::-1]                   # samples aligned on the newest row
    q = {}
    for j in range(k):
        e = iteration_error(s[:, j])
        err, how = e["error"], "noise floor" if e["noise"] else "geometric"
        if math.isfinite(err) and not e["noise"] and e["rho"] is not None and len(s) > 8:
            before = iteration_error(s[:-3, j])
            if before["rho"] is not None and (1.0 - e["rho"]) < (1.0 - before["rho"]) / 1.2:
                err, how = err * 10.0, "geometric, slowing (x10)"
        if not math.isfinite(err):
            err, how = _fallback_error(s[:, j])
        state = "settled" if err < tols[j] else "floor" if e["noise"] else "running"
        if state in ("settled", "floor"):
            # The samples say done; do the rows between them agree? A solution that flips between two
            # states on every iteration is constant when sampled at an even interval (case 7, insight 3:
            # drag 0.00275 / 0.00318 in turn, stopped as "settled, change zero") and looks like a noise
            # floor at an odd one.
            x = v[-(max(every, 12) + 1):, j]
            step = float(np.max(np.abs(np.diff(x)))) if len(x) > 1 else 0.0
            period = _cycle(x) if step >= tols[j] else None
            if period:
                state, err, how = "oscillating", step, f"cycle of period {period} between rows"
            elif state == "settled" and step >= tols[j]:
                state, err, how = "running", step, "consecutive rows still differ by more than tol"
        if np.ptp(v[:, j]) == 0.0:
            # Never changed: truly constant (upstream of a heater) or not reached by the solution yet (far
            # downstream of a moving front). The data cannot tell, so it does not decide the stop.
            state, how = "constant", "never changed"
        q[names[j]] = dict(state=state, error=err, rho=e["rho"], tol=float(tols[j]), how=how)
    cycling = [n for n, x in q.items() if x["state"] == "oscillating"]
    running = [n for n, x in q.items() if x["state"] == "running"] + cycling
    floors = [n for n, x in q.items() if x["state"] == "floor"]
    const = [n for n, x in q.items() if x["state"] == "constant"]
    if len(const) == k:
        return dict(ok=False, status="ok", quantities=q,
                    reason="running: no monitored value has changed yet")
    judged = [n for n in q if n not in const]
    worst = max(judged, key=lambda n: q[n]["error"] / q[n]["tol"] if q[n]["tol"] > 0 else math.inf)
    if cycling:
        c = max(cycling, key=lambda n: q[n]["error"] / q[n]["tol"] if q[n]["tol"] > 0 else math.inf)
        return dict(ok=False, status="warn", quantities=q,
                    reason=f"oscillating: {len(cycling)} of {k} repeat in a cycle between rows (worst {c}: "
                           f"{q[c]['error']:.2g} between consecutive rows, tol {q[c]['tol']:.2g}; {q[c]['how']}). "
                           "If this lasts, the iteration cycles instead of converging and running longer will "
                           "not help: lower a relaxation factor")
    if running:
        reason = f"running: {len(running)} of {k} not settled (worst {worst}: error {q[worst]['error']:.2g})"
    elif floors:
        reason = "floor reached (changes stopped shrinking) for " + ", ".join(
            f"{n} ({q[n]['error']:.2g} > tol {q[n]['tol']:.2g})" for n in floors)
    else:
        reason = f"all settled: largest error {q[worst]['error']:.2g} ({worst}, tol {q[worst]['tol']:.2g})"
    if const:
        reason += f"; constant so far, not judged: {', '.join(const)}"
    return dict(ok=not running, status="warn" if floors else "ok", reason=reason, quantities=q)


def steady_decider(fo, fname, tol, columns=None, every=10, hold=2, names=None):
    """decide(case) for LiveStop on a steady run: read postProcessing/<fo>/*/<fname> incrementally
    (any function object: probes, forces, surfaceFieldValue of a wall heat flux ...), apply
    steady_check, and stop once it has held for `hold` consecutive checks.

    columns: indices into the flattened value columns (after the iteration column; a vector is three
    columns); None = all. tol, every, names: as in steady_check, for the chosen columns.

    The checks come at intervals of wall time (foam.run's watch_every), so a small grid passes
    thousands of iterations between two of them and runs far past the point where the rule first held
    (case 7: 8433 iterations run on 816 cells, rule held from about 350). The decision that stops the
    run therefore also carries t_first_held, the iteration at which the rule first held `hold` times in
    a row on the recorded history (steady_replay): quote that, not the iterations run, when you say how
    fast a grid converged or extrapolate the cost of a finer one."""
    tails, streak = {}, {}

    def decide(case):
        key = str(case)
        a = tails.setdefault(key, DatTail(case, fo, fname))()
        if len(a) < 6 * every:
            return None
        v = a[:, 1:] if columns is None else a[:, [1 + c for c in columns]]
        r = steady_check(v, tol, every, names)
        streak[key] = streak.get(key, 0) + 1 if r["ok"] else 0
        r.update(stop=streak[key] >= hold, t=float(a[-1, 0]))
        if r["stop"]:
            i = steady_replay(v, tol, every, hold)["index"]
            r["t_first_held"] = float(a[i, 0]) if i is not None else None
        return r
    return decide


def steady_replay(values, tol, every=10, hold=2):
    """Where would the steady stop have ended a recorded run? Applies steady_check after every
    `every` rows, as if checked live that often. Returns dict(index, check) for the first row where it
    held `hold` times in a row, or index None. Use it on a coarse mesh's history to test a tolerance
    and sampling interval before relying on them live."""
    v = np.asarray(values, float)
    held = 0
    for i in range(every, len(v), every):
        r = steady_check(v[:i + 1], tol, every)
        held = held + 1 if r["ok"] else 0
        if held >= hold:
            return dict(index=i, check=r)
    return dict(index=None, check=None)


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
    if len(d) == n and (np.all(d > 0) or np.all(d < 0)):
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


def _sources(case):
    """postProcessing files (<function object>/<start time>/<file>: .dat files and probes such as U),
    in time order of their start folder (restarts write a new folder)."""
    root = Path(case) / "postProcessing"
    if not root.exists():
        return []

    def key(f):
        try:
            return (f.parent.parent.name, f.name, float(f.parent.name))
        except ValueError:
            return (f.parent.parent.name, f.name, math.inf)
    return sorted((f for f in root.glob("*/*/*") if f.is_file() and (f.suffix == ".dat" or not f.suffix)), key=key)


def _expand(names, row):
    """CSV column names of a postProcessing row: vectors '(a b c)' become name_x, name_y, name_z."""
    cols = []
    for j, v in enumerate(row[1:]):
        base = names[j] if j < len(names) else f"col{j + 1}"
        if isinstance(v, tuple):
            cols += [f"{base}_{c}" for c in "xyz"[:len(v)]]
        elif isinstance(v, float):
            cols.append(base)
    return cols


def _flat(row):
    out = []
    for v in row:
        if isinstance(v, tuple):
            out += v
        elif isinstance(v, float):
            out.append(v)
    return out


def sync(case_dir, out_dir=None):
    """Bring the raw-history CSV files in <case>/peek/ up to date by APPENDING only what the run wrote
    since the last call: a byte offset per source (solver log, each postProcessing file) is kept in
    peek/.state.json, so the cost of a call does not grow with the length of the run. Only complete
    time steps are taken from the log while the solver runs (up to the last 'Time =' line). A log or
    source that got shorter means a new run in the same case: the CSV files start over.
    Returns (out_dir, state)."""
    case = Path(case_dir)
    out = Path(out_dir) if out_dir else case / "peek"
    out.mkdir(parents=True, exist_ok=True)
    sp = out / ".state.json"
    st = json.loads(sp.read_text()) if sp.exists() else {}
    logs = [p for p in case.glob("log.*") if p.name.endswith("Foam")]
    log = max(logs, key=lambda p: p.stat().st_mtime) if logs else None

    def reset():
        for f in [*out.glob("*.csv"), *out.glob("*.f64")]:
            f.unlink()
        return dict(log=log.name if log else None, offset=0, fields=None,
                    flags=dict(fatal=False, fpe=False, nan=False, ended=False), dat={})

    if not st or (log and st.get("log") != log.name) or (log and log.stat().st_size < st["offset"]) or any(
            not (case / rel).exists() or (case / rel).stat().st_size < d["offset"] for rel, d in st.get("dat", {}).items()):
        st = reset()

    def append(name, header, rows):
        # CSV to grep, plus the same rows as raw float64 (<name>.f64) for fast loading (load_csv)
        rows = np.array(list(rows), dtype=float).reshape(-1, len(header))
        f = out / f"{name}.csv"
        new = not f.exists()
        with open(f, "a") as fh:
            if new:
                fh.write(",".join(header) + "\n")
            fh.writelines(",".join(f"{v:.10g}" for v in r) + "\n" for r in rows)
        with open(f.with_suffix(".f64"), "ab") as fb:
            rows.tofile(fb)

    if log:
        alive = _alive(case)
        with open(log, "rb") as fh:
            fh.seek(st["offset"])
            data = fh.read()
        if alive is False or b"\nEnd" in data or b"FOAM FATAL" in data or b"sigFpe" in data:
            chunk = data                                   # the run is over: take everything
        else:
            i = data.rfind(b"\nTime = ")                   # complete steps only
            chunk = data[:i + 1] if i >= 0 else b""
        if chunk:
            p = parse_log_text(chunk.decode(errors="replace"))
            for k in st["flags"]:
                st["flags"][k] = st["flags"][k] or p[k]
            if p["steps"]:
                if st["fields"] is None:
                    st["fields"] = sorted({f for x in p["steps"] for f in x["res"]})
                nan = float("nan")
                append("solver_log", ["t", "exec_s", "courant_max", "continuity_local"] + [f"res_{f}" for f in st["fields"]],
                       ([x["time"]] + [nan if x[k] is None else x[k] for k in ("exec", "co", "cont")]
                        + [x["res"].get(f, nan) for f in st["fields"]] for x in p["steps"]))
            st["offset"] += len(chunk)
    for src in _sources(case):
        rel = str(src.relative_to(case))
        d = st["dat"].setdefault(rel, dict(offset=0, names=None))
        with open(src, "rb") as fh:
            fh.seek(d["offset"])
            data = fh.read()
        data = data[:data.rfind(b"\n") + 1]               # complete lines only
        if not data:
            continue
        text = data.decode(errors="replace")
        hdr = [ln for ln in text.splitlines() if ln.startswith("#")]
        if hdr and d["names"] is None:
            d["names"] = hdr[-1].lstrip("#").split()[1:]
        rows = [r for r in foam.parse_dat(text) if r and isinstance(r[0], float)]
        d["offset"] += len(data)
        if not rows:
            continue
        name = f"{src.parent.parent.name}_{src.name}".replace(".dat", "")
        f = out / f"{name}.csv"
        cols = ["t"] + _expand(d["names"] or [], rows[0])
        if f.exists():                                   # a restart folder: same columns as the first
            cols = f.open().readline().strip().split(",")
        append(name, cols, (fl for fl in map(_flat, rows) if len(fl) == len(cols)))
    sp.write_text(json.dumps(st))
    return out, st


def load_csv(path, columns=None):
    """(column names, dict name -> array) of a peek CSV. Reads the float64 twin <name>.f64 that sync()
    writes next to it (fast for millions of rows); without it, parses the text (about 3 s per
    million rows of 10 columns). columns: names to keep (default all)."""
    path = Path(path)
    with open(path) as fh:
        names = fh.readline().strip().split(",")
    b = path.with_suffix(".f64")
    if b.exists():
        v = np.fromfile(b, dtype=float)
    else:
        v = np.array(path.read_text().partition("\n")[2].replace(",", " ").split(), dtype=float)
    v = v[:v.size - v.size % len(names)].reshape(-1, len(names))
    keep = names if columns is None else ["t"] + [c for c in names[1:] if c in columns]
    return names, {c: v[:, names.index(c)] for c in keep}


def _shown(cols, t, t_from):
    """Columns for display: from t_from on, without columns that do not change beyond round-off
    (a fixed inflow rate) or are round-off themselves (the z force of a 2D case: 1e-24 next to 1e-3)."""
    keep = t >= t_from if t_from is not None else np.ones(len(t), bool)
    if not keep.any():
        return t[keep], {}
    scale = max((float(np.max(np.abs(v))) for v in cols.values()), default=0.0)
    return t[keep], {k: v[keep] for k, v in cols.items()
                     if np.ptp(v) > 1e-8 * np.max(np.abs(v)) and np.max(np.abs(v)) > 1e-12 * scale}


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
    the Courant number and every monitored series as min / mean / max over `windows` equal time
    windows (_digest; windows=0: one line per series). Returns dict(text, png, csv, ...).

    The raw history is kept in <case>/peek/ (csv_dir): solver_log.csv (every step's residuals, Courant
    number, continuity error, solver time) and one CSV per postProcessing file. sync() appends only
    what is new since the last call, so peeking at a run of millions of steps stays cheap; the files
    are always complete, whatever files / t_from show. files: substrings selecting the series shown
    (default all); t_from: show the history from this time on; plot: also write <case>/peek.png, the
    history as a figure, to look at or to show a human (see watch())."""
    case = Path(case_dir)
    logs = [p for p in case.glob("log.*") if p.name.endswith("Foam")]
    if not logs:
        return dict(text=f"{case}: no solver log (log.*Foam) yet", png=None, csv=[], state="not started")
    log = max(logs, key=lambda p: p.stat().st_mtime)
    out, st = sync(case, csv_dir)
    cd = (case / "system" / "controlDict").read_text()
    t_start, t_end = float(_entry(cd, "startTime") or 0), float(_entry(cd, "endTime") or "nan")
    fl = st["flags"]
    lp = out / "solver_log.csv"
    arr = None
    if lp.exists():
        _, c = load_csv(lp)
        arr = dict(time=c["t"], co=c["courant_max"], cont=c["continuity_local"], exec=c["exec_s"],
                   res={k[4:]: v for k, v in c.items() if k.startswith("res_")})
    empty = dict(time=np.array([]), co=np.array([]), cont=np.array([]), exec=np.array([]), res={})
    a = assess_arrays(arr or empty, fl, end_time=t_end, courant_limit=courant_limit)
    alive = _alive(case)
    state = ("running" if alive else "crashed" if (fl["fatal"] or fl["fpe"]) else
             "finished" if fl["ended"] else "not running (killed or stopped?)" if alive is False else
             "unknown (no PID file; finished runs have none)")
    lines = [f"{case.name}: {log.name}, {state}"]
    n = a["steps"]
    if n:
        t_now = a["last_time"]
        frac = (t_now - t_start) / (t_end - t_start) if t_end > t_start else float("nan")
        msg = (f"progress: t = {t_now:.6g} of {t_start:g}..{t_end:g} ({100 * frac:.1f} %), {n} steps, "
               f"{a['wall_time'] or float('nan'):.0f} s of solver time")
        if state == "running" and a["est_full_runtime"] is not None:
            msg += f", about {a['est_full_runtime'] - a['wall_time']:.0f} s left to endTime at the current rate"
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
    # history tables: the solver log, then each selected series (loaded from the CSV files)
    log_shown = None
    if arr is not None and n:
        lc = {f"res {f}": v for f, v in arr["res"].items()} | {"Courant max": arr["co"]}
        lc = {k: v for k, v in lc.items() if np.isfinite(v).all()}
        t, lc = _shown(lc, arr["time"], t_from)
        log_shown = (t, lc)
        if windows and lc and len(t) >= windows:
            lines += _digest("solver log", t, lc, windows, ncols=4)
    ser = []
    for f in sorted(out.glob("*.csv")):
        if f.name == "solver_log.csv" or (files and not any(x.replace("/", "_").replace(".dat", "") in f.stem
                                                              for x in files)):
            continue
        _, c = load_csv(f)
        t, cols = _shown({k: v for k, v in c.items() if k != "t"}, c["t"], t_from)
        if cols and len(t) >= 4:
            ser.append((f.stem, t, cols))
    for label, t, cols in ser:
        if windows and len(t) >= windows:
            lines += _digest(label, t, cols, windows)
        else:
            late = t >= t[-1] - 0.25 * (t[-1] - t[0])
            parts = [f"{k} {v[-1]:.5g} [{v[late].min():.5g}, {v[late].max():.5g}]" for k, v in list(cols.items())[:3]]
            lines.append(f"{label}: " + "; ".join(parts) + (f"; +{len(cols) - 3} columns" if len(cols) > 3 else "")
                         + "  (last value [min, max over the last quarter])")
    csvs = []
    for f in sorted(out.glob("*.csv")):
        with open(f) as fh:
            head = fh.readline().strip()
        csvs.append(f"{f.name} ({f.stat().st_size / 1e6:.3g} MB: {head})")
    lines.append(f"raw history, every step, in {out}/ (CSV, appended at each peek; grep it or load it "
                 f"with cfdlib.monitor.load_csv):")
    lines += [f"  {c}" for c in csvs]
    png = None
    if plot and (log_shown or ser):
        png = _peek_plot(case, log.name, log_shown, ser, t_from)
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


def _peek_plot(case, title, log_shown, ser, t_from):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = []
    if log_shown:
        t, lc = log_shown
        res = {k[4:]: v for k, v in lc.items() if k.startswith("res ")}
        if res:
            panels.append(("initial residuals", t, res, True))
        if "Courant max" in lc:
            panels.append(("max Courant number", t, {"Co": lc["Courant max"]}, False))
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
    #   (keeps the raw history as CSV in <case>/peek/, appending only what is new)
    # python -m cfdlib.monitor watch <case dir> [--every S] [--files ...]        (a live page for a human)
    import argparse
    ap = argparse.ArgumentParser(description="Look into an OpenFOAM run: progress, health, live-stop verdict, history")
    ap.add_argument("cmd", choices=["peek", "watch"])
    ap.add_argument("case")
    ap.add_argument("--from", dest="t_from", type=float)
    ap.add_argument("--files", help="comma-separated substrings of postProcessing series to show")
    ap.add_argument("--windows", type=int, default=8, help="time windows of the history table (0: one line per series)")
    ap.add_argument("--csv", dest="csv_dir", help="directory for the raw-history CSV files (default <case>/peek/)")
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
