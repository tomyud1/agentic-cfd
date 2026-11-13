"""gpucfd as a tool of the agent: when it fits a problem, and a run loop that records a transient run in
the same kind of files as the OpenFOAM runs (runs/<name>.json + .npz), so a case's analysis can read
both and compare them.

When to choose gpucfd instead of OpenFOAM (measured numbers and limits: gpucfd/DESIGN.md):
- It fits: incompressible, laminar, constant density and viscosity, single phase, no heat transfer;
  a rectangular box (periodic, wall, inflow or outflow sides) with solid bodies inside it given as
  shapes (cylinder, sphere, box, or any signed-distance function); a grid fine enough for central
  differences (cell Reynolds number U h / nu about 2 or below), uniform or stretched per direction
  (gpucfd.grids: cells clustered toward walls, or a band of fine cells around a body).
- It does not fit: turbulence models (RANS/LES), heat transfer, multiphase, compressible flow, STL
  geometry, body-fitted or unstructured grids. A stretched grid is still rows and columns: the fine
  band around a body crosses the whole domain in each direction (case 4: 3.85 times fewer cells
  than uniform, not the 30 times of a body-fitted grid).
- Where it pays: many time steps on grids of 10^4 to 10^7 cells (one RTX 4070 runs a 2D step in a
  few milliseconds). Bodies force the explicit RK3 scheme, whose time step is limited by
  0.15 h^2 / nu with the smallest cell as well as by the flow speed; at high resolution that limit
  dominates (case 4 at 140 cells per diameter: 3.3 times below the convective limit).
- With bodies and the explicit scheme, check the pressure tolerance on the finest grid, not only on
  the coarsest: the pressure noise grows as the time step shrinks (case 4: 1e-8 on the coarsest
  level, 1e-10 at 140 cells per diameter). tolerance_check() does this for any level.
- An immersed body's result depends a little on where the grid cuts it; position_spread() reports
  that spread from runs with the body shifted by a fraction of a cell.
- Steady problems: march to steady state with gpucfd.steady (grid sequencing and an error-based stop).

check_fit() turns this into a checklist for a given problem; its reasons go into decisions.yaml.
run_transient() runs and records a transient case (live check, stop rule, a partial record that survives
a killed run); write_run() saves it in the layout of the OpenFOAM runs.
"""
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

BLOCKERS = {
    "turbulence_model": "needs a turbulence model (gpucfd resolves laminar flow only)",
    "heat_transfer": "needs heat transfer or a transported scalar",
    "multiphase": "needs several phases or a free surface",
    "compressible": "needs compressible flow",
    "variable_properties": "needs variable density or viscosity",
    "stl_geometry": "the geometry comes as STL/CAD (gpucfd takes shapes or signed-distance functions)",
    "non_box_domain": "the domain is not a rectangular box (bodies inside a box are fine)",
    "body_fitted_grid": "needs a body-fitted or unstructured grid (gpucfd grids are rows and columns of cells, "
                        "stretched per direction, with the bodies immersed)",
}


def check_fit(cell_reynolds=None, needs_stretched_grid=False, **needs):
    """Whether gpucfd can solve a problem. needs: flags named as in BLOCKERS (True = the problem needs it).
    cell_reynolds: U h / nu on the finest grid you can afford (above about 2 central differences
    oscillate). needs_stretched_grid: thin layers that a uniform grid cannot resolve at an affordable size;
    since 2026-11-13 a note, not a blocker (stretched grids with bodies: gpucfd/DESIGN.md, rungs 12 and 13).
    Returns dict(ok, blockers[], notes[])."""
    unknown = set(needs) - set(BLOCKERS)
    if unknown:
        raise ValueError(f"unknown flags {sorted(unknown)}; known: {sorted(BLOCKERS)}")
    blockers = [BLOCKERS[k] for k, v in needs.items() if v]
    notes = []
    if needs_stretched_grid:
        notes.append("thin layers: use a grid stretched per direction (gpucfd.grids.tanh toward walls, "
                     "gpucfd.grids.clustered around a body). The fine bands cross the whole domain, and with bodies the "
                     "explicit time step follows the smallest cell squared: estimate cells and steps before choosing gpucfd. "
                     "Below a few hundred thousand cells a step costs several milliseconds almost whatever the size, so "
                     "there fewer cells buy accuracy per cell, not time (case 4: a third of the cells, nearly the same time)")
    if cell_reynolds is not None:
        if cell_reynolds > 10:
            blockers.append(f"cell Reynolds number {cell_reynolds:.3g}: central differences would oscillate")
        elif cell_reynolds > 2:
            notes.append(f"cell Reynolds number {cell_reynolds:.3g} > 2: expect wiggles; check them on the finer grids")
    return dict(ok=not blockers, blockers=blockers, notes=notes)


def available():
    """dict(name, compute_capability) of the GPU if the gpucfd library loads, else None (the reason is printed)."""
    try:
        from gpucfd import gcfd
        return gcfd.device()
    except Exception as e:  # noqa: BLE001 - any failure means: not available on this machine
        print("gpucfd not available:", e)
        return None


def _save_partial(path, ts, series):
    """Write what a run has recorded so far to path (.npz), replacing the previous copy in one move."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, t=np.array(ts), **{k: np.array(v) for k, v in series.items()})
    tmp.replace(path)


def run_transient(s, dt, t0, t_end, sample_dt, observe, u_ref, log=print, tag="", log_every=1.0, ptol=1e-10,
                  cfl_limit=1.0, watch=None, stop_on_watch=False, watch_every=None, partial=None):
    """Advance gpucfd simulation s from time t0 to t_end and sample it about every sample_dt.

    dt is the largest time step to use; it is reduced slightly so that t_end - t0 is a whole number of
    steps. Samples are taken every m steps, m = the largest whole number with m dt <= sample_dt (at least 1),
    so sampling never shortens the time step (the last interval may be shorter).
    observe(s) -> dict of name -> number or array, called at every sample (and at t0). If it returns
    'umax' (largest velocity magnitude), the Courant number is checked against cfl_limit.
    u_ref: velocity scale for the divergence check (max |div u| h / u_ref).
    watch(t, series) -> optional text, called every watch_every time units (default: log_every): the run's
    live check (a periodicity or steadiness test); its text is logged and its first non-empty answer is
    recorded. For a periodic flow set watch_every to about one period: a run can only stop at a check, so
    with a coarse interval the run lengths, and the wall times compared afterwards, jump in large steps.
    stop_on_watch: end the run at that first answer (t_end is then the longest the run may go).
    partial: path of an .npz file that receives everything recorded so far at every log line (same layout
    as the file write_run saves). A run that is killed from outside then keeps its record up to the last
    log line, so a stop rule can be applied to a run without losing it. write_run removes
    <runs_dir>/<name>.partial.npz once the full record is saved.
    Returns dict(t, series, perf, health, dt, watch_first).

    A blow-up (non-finite values) or Ctrl-C ends the run early instead of raising (health status 'fail' or
    'warn'), so what was recorded up to then is returned and can be saved."""
    n_total = max(1, math.ceil((t_end - t0) / dt - 1e-9))
    dt = (t_end - t0) / n_total
    nsub = max(1, int(math.floor(sample_dt / dt + 1e-9)))
    # the cell size of the Courant and divergence checks: on a stretched grid the smallest cell (strict)
    h = s._hmin() if getattr(s, "stretched", False) else min(s.length[d] / s.n[d] for d in range(3) if s.n[d] > 1)
    cells = int(np.prod(s.n))
    ts, series = [], {}
    gpu_ms = obs_s = its = 0.0
    steps = 0
    max_div = 0.0
    cfl = 0.0
    error = None
    interrupted = None
    first = None
    next_log = t0 + log_every
    watch_every = log_every if watch_every is None else watch_every
    next_watch = t0 + watch_every
    wall0 = time.time()

    def sample(t):
        nonlocal obs_s, cfl
        a = time.time()
        o = observe(s)
        obs_s += time.time() - a
        ts.append(t)
        for k, v in o.items():
            series.setdefault(k, []).append(np.asarray(v, float))
        if "umax" in o and len(ts) > 1:  # not the start field (a projected start from rest can be far off)
            cfl = max(cfl, float(o["umax"]) * dt / h)

    sample(t0)
    while steps < n_total:
        m = min(nsub, n_total - steps)
        try:
            st = s.run(dt, m, ptol)
        except FloatingPointError as e:
            error = str(e)
            log(f"{tag}t {t0 + (steps + m) * dt:.4f}: {e}")
            break
        except KeyboardInterrupt:
            interrupted = f"interrupted at t = {t0 + steps * dt:.4f} (of {t_end:g})"
            log(f"{tag}{interrupted}")
            break
        gpu_ms += st["ms"]
        its += st["poisson_its"] * m
        steps += m
        max_div = max(max_div, st["max_div"])
        t = t0 + steps * dt
        sample(t)
        last_step = steps == n_total
        due_log = t >= next_log - 1e-9 or last_step
        msg = None
        if watch and (t >= next_watch - 1e-9 or last_step):
            while next_watch <= t + 1e-9:
                next_watch += watch_every
            msg = watch(t, {k: np.array(v) for k, v in series.items()} | {"t": np.array(ts)})
            if msg and first is None:
                first = dict(t=t, text=msg)
        stop = bool(msg) and stop_on_watch
        if due_log or stop:
            while due_log and next_log <= t + 1e-9:
                next_log += log_every
            last = {k: v[-1] for k, v in series.items() if np.ndim(v[-1]) == 0}
            vals = " ".join(f"{k} {v:.6g}" for k, v in last.items())
            log(f"{tag}t {t:.3f} {vals} | CFL {cfl:.2f} div {max_div * h / u_ref:.1e} | GPU {gpu_ms / 1000:.1f} s "
                f"wall {time.time() - wall0:.1f} s" + (f" | {msg}" if msg else ""))
            if partial and not last_step and not stop:
                _save_partial(partial, ts, series)
        if stop:
            break
    wall = time.time() - wall0
    nan = any(not np.all(np.isfinite(np.asarray(v))) for v in series.values())
    checks = []

    def add(name, ok, value, meaning, warn=False):
        checks.append(dict(name=name, status="ok" if ok else ("warn" if warn else "fail"), value=value, meaning=meaning))
    add("Solver finished without errors", error is None, error or "no errors",
        "A blow-up means the time step is too large for the flow speed or the viscous limit.")
    if interrupted:
        add("Run not interrupted", False, interrupted,
            "The run was stopped from the keyboard; the record ends there.", warn=True)
    add("No NaN values", not nan, "none" if not nan else "found", "NaN (not a number) means invalid values.")
    add("Max Courant number", cfl <= cfl_limit, f"{cfl:.2f}",
        f"Fluid should not cross more than about one cell per time step (limit used: {cfl_limit:g}).", warn=True)
    rel_div = max_div * h / u_ref
    add("Divergence after projection", rel_div < 1e-6, f"{rel_div:.1e}",
        "max |div u| h / U after each step: the velocity must stay divergence-free (mass conservation).", warn=True)
    status = "fail" if any(c["status"] == "fail" for c in checks) else \
        ("warn" if any(c["status"] == "warn" for c in checks) else "ok")
    perf = dict(gpu_s=gpu_ms / 1000, wall_s=wall, observe_s=obs_s, steps=steps, cells=cells, sample_steps=nsub,
                poisson_its=its / max(steps, 1),
                ms_per_step=gpu_ms / max(steps, 1), us_per_cell_step=1000 * gpu_ms / max(steps * cells, 1))
    return dict(t=np.array(ts), series={k: np.array(v) for k, v in series.items()}, perf=perf,
                health=dict(status=status, checks=checks, steps=steps, last_time=ts[-1]), dt=dt, watch_first=first)


def write_run(runs_dir, name, result, meta):
    """Save a run_transient result as runs_dir/<name>.npz (t + every series) and <name>.json (meta, perf,
    health, environment). Returns the json record."""
    from gpucfd import gcfd
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(runs_dir / f"{name}.npz", t=result["t"], **result["series"])
    rec = dict(name=name, tool="gpucfd", **meta, dt=result["dt"], start_time=float(result["t"][0]),
               end_time=float(result["t"][-1]), health=result["health"], perf=result["perf"],
               watch_first=result["watch_first"], gpu=gcfd.device()["name"], python=platform.python_version(),
               wall_time=result["perf"]["wall_s"])
    (runs_dir / f"{name}.json").write_text(json.dumps(rec, indent=1, default=float))
    (runs_dir / f"{name}.partial.npz").unlink(missing_ok=True)  # the full record replaces run_transient's partial one
    return rec


def tolerance_check(make, dt, t0, length, sample_dt, observe, u_ref, tols=(1e-10, 1e-9, 1e-8), keys=None,
                    rule=1e-4, log=print, **run_args):
    """Which pressure tolerance may a level use? Runs the same stretch of flow once per tolerance and
    compares the sampled outputs with those of the strictest one (the same-answer rule).

    Do this on every level of a study, the finest included: with bodies the scheme is explicit, the time
    step shrinks faster than the cell size, and the error a fixed tolerance leaves in the pressure then
    weighs more in the forces (case 4: 1e-8 held on the coarsest level, the finest needed 1e-10, and with
    1e-8 its forces jittered so much that the run never became periodic).

    make() -> a new simulation holding the start field at time t0 (for example the level's saved last
    field); it is closed here after its run. length: flow time to compare over (about one period of an
    unsteady flow). observe, sample_dt, u_ref and run_args as in run_transient.
    keys: the series to compare (default: every scalar series except 'umax'). The differences are
    absolute, max over the samples of |value - strictest value|: compare coefficients, or scale the
    outputs in observe. rule: the largest difference accepted.
    Returns dict(rule, keys, runs=[dict(tol, diff{key}, worst, passes, ms_per_step, poisson_its, divergence)],
    chosen), chosen = the loosest tolerance that passes together with every stricter one."""
    tols = sorted(float(v) for v in tols)
    base, runs = None, []
    for tol in tols:
        s = make()
        try:
            res = run_transient(s, dt, t0, t0 + length, sample_dt, observe, u_ref, log=lambda m: None, ptol=tol, **run_args)
        finally:
            s.close()
        if keys is None:
            keys = [k for k, v in res["series"].items() if k != "umax" and np.ndim(v) == 1]
        if base is None:
            base = res
        diff = {k: float(np.max(np.abs(res["series"][k] - base["series"][k]))) if len(res["t"]) == len(base["t"]) else math.inf
                for k in keys}
        worst = max(diff.values())
        r = dict(tol=tol, diff=diff, worst=worst, passes=bool(worst <= rule and res["health"]["status"] != "fail"),
                 ms_per_step=res["perf"]["ms_per_step"], poisson_its=res["perf"]["poisson_its"],
                 divergence=next(c["value"] for c in res["health"]["checks"] if c["name"].startswith("Divergence")))
        runs.append(r)
        log(f"pressure tolerance {tol:g}: largest difference {worst:.2e} ({'passes' if r['passes'] else 'fails'} the rule "
            f"{rule:g}), {r['ms_per_step']:.2f} ms per step, {r['poisson_its']:.1f} pressure iterations per step")
    chosen = tols[0]
    for r in runs:
        if not r["passes"]:
            break
        chosen = r["tol"]
    return dict(rule=rule, keys=list(keys), runs=runs, chosen=chosen)


def position_spread(by_position):
    """How much do the outputs move with where the grid cuts an immersed body?

    A body in gpucfd is immersed in a grid of rows and columns, and the result depends a little on where
    its surface falls between the grid lines. A grid-convergence study does not see this error: every
    level of it cuts the body at the same relative place. To measure it, rerun a level with the body
    shifted by a fraction of a cell (0 and 1/2 cell at least; 0, 1/4, 1/2, 3/4 for a proper spread) and
    quote the outputs as middle +- half_range next to the grid-convergence index. Measured on a cylinder:
    drag moved by 0.5 % of its value at 35 cells per diameter and by 0.1 % at 70 and 140; the pressure
    at one point of the surface by 2 % at 35 (gpucfd/tests/shift_study.py, case 4's shifted runs).

    by_position: {label: {quantity: value}} from those runs.
    Returns {quantity: dict(values{label}, middle, half_range)}; two positions give a range, not a statistic."""
    labels = list(by_position)
    out = {}
    for q in by_position[labels[0]]:
        vals = {lb: float(by_position[lb][q]) for lb in labels if q in by_position[lb]}
        lo, hi = min(vals.values()), max(vals.values())
        out[q] = dict(values=vals, middle=0.5 * (lo + hi), half_range=0.5 * (hi - lo))
    return out
