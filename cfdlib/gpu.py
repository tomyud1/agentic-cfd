"""gpucfd as a tool of the agent: when it fits a problem, and a run loop that records a transient run in
the same kind of files as the OpenFOAM runs (runs/<name>.json + .npz), so a case's analysis can read
both and compare them.

When to choose gpucfd instead of OpenFOAM (measured numbers and limits: gpucfd/DESIGN.md):
- It fits: incompressible, constant density and viscosity, single phase, no heat transfer; laminar,
  or turbulent with the eddies resolved in time on the grid (no model: see below);
  a rectangular box (periodic, wall, inflow or outflow sides) with solid bodies inside it given as
  shapes (cylinder, sphere, box, or any signed-distance function); for a laminar flow a grid fine
  enough for central differences (cell Reynolds number U h / nu about 2 or below), uniform or
  stretched per direction (gpucfd.grids: cells clustered toward walls, or a band of fine cells around
  a body).
- Turbulence without a model (measured 2026-11-14 on one flow, the plane channel at friction Reynolds
  number 180; gpucfd/DESIGN.md has the numbers): the scheme conserves kinetic energy, so a turbulent
  flow runs stably with cells far larger than the laminar rule allows, and the grid is judged in wall
  units. gpucfd.turbulence has the start field and the plane averages, cfdlib.signals.stationary the
  time average with its statistical error. Per step it was 2 to 9 times faster than OpenFOAM on 8
  cores of the same PC (larger grids, larger gain), with OpenFOAM at its tutorial's settings.
- It does not fit: turbulence models (RANS/LES), heat transfer, multiphase, compressible flow, STL
  geometry, body-fitted or unstructured grids. A stretched grid is still rows and columns: the fine
  band around a body crosses the whole domain in each direction (case 4: 3.85 times fewer cells
  than uniform, not the 30 times of a body-fitted grid).
- Where it pays: many time steps on grids of 10^4 to 10^7 cells (one RTX 4070 runs a 2D step in a
  few milliseconds). The explicit RK3 scheme's time step is limited by 0.15 h^2 / nu with the
  smallest cell as well as by the flow speed; at high resolution that limit dominates (case 4 at
  140 cells per diameter: 3.3 times below the convective limit). scheme="imex" has only the
  flow-speed limit. With a body its step costs about twice an explicit one, so it pays when the
  viscous limit is more than about twice below the convective one (case 4: 1.55 times faster at
  140 cells per diameter, slower at 35); both schemes give the same answer, so the levels of a
  study may differ in scheme. pick_scheme() does this comparison for a level.
- With bodies, check the pressure tolerance on the finest grid, not only on the coarsest: the
  pressure noise grows as the time step shrinks (case 4, explicit: 1e-8 on the coarsest level,
  1e-10 at 140 cells per diameter; IMEX: 1e-9 and 1e-10). tolerance_check() does this for any level.
- An immersed body's result depends a little on where the grid cuts it; position_spread() reports
  that spread from runs with the body shifted by a fraction of a cell.
- Steady problems: march to steady state with gpucfd.steady (grid sequencing and an error-based stop).
  Around a body, a simulation made with scheme="imex" settles to the same state in fewer steps (DFG
  2D-1 at 17.6 cells per diameter: same drag to 3e-8, 18 GPU-s against 50; one case, CFL 0.8, larger
  CFL numbers untried with bodies).

check_fit() turns this into a checklist for a given problem; its reasons go into decisions.yaml.
time_step() gives the step of a run with both limits and the one that set it. run_transient() runs and
records a transient case (live check, stop rule, a partial record that survives a killed run); write_run()
saves it in the layout of the OpenFOAM runs (its perf block names each cost by its clock: perf_record).
cfdlib.runrecord.read() returns a record of either tool in one shape. tolerance_check() and
read_tolerance_check() make and read a pressure-tolerance check.
"""
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

BLOCKERS = {
    "turbulence_model": "needs a turbulence model, RANS or LES (gpucfd has none; it can compute a turbulent flow "
                        "directly on a grid fine enough to resolve it: check_fit(wall_units=...))",
    "heat_transfer": "needs heat transfer or a transported scalar",
    "multiphase": "needs several phases or a free surface",
    "compressible": "needs compressible flow",
    "variable_properties": "needs variable density or viscosity",
    "stl_geometry": "the geometry comes as STL/CAD (gpucfd takes shapes or signed-distance functions)",
    "non_box_domain": "the domain is not a rectangular box (bodies inside a box are fine)",
    "body_fitted_grid": "needs a body-fitted or unstructured grid (gpucfd grids are rows and columns of cells, "
                        "stretched per direction, with the bodies immersed)",
}


def check_fit(cell_reynolds=None, needs_stretched_grid=False, wall_units=None, **needs):
    """Whether gpucfd can solve a problem. needs: flags named as in BLOCKERS (True = the problem needs it).
    cell_reynolds: U h / nu on the finest grid you can afford (above about 2 central differences
    oscillate). For a laminar flow; not the measure for turbulence resolved in time (use wall_units).
    needs_stretched_grid: thin layers that a uniform grid cannot resolve at an affordable size;
    since 2026-11-13 a note, not a blocker (stretched grids with bodies: gpucfd/DESIGN.md, rungs 12 and 13).
    wall_units: for a turbulent flow along walls computed without a model, dict(dx, dy_wall, dz): the cell
    sizes of the grid you can afford in wall units (length x u_tau / nu; dx along the flow, dz across it,
    dy_wall the height of the wall cell). Gives notes from the one flow measured, not blockers.
    Returns dict(ok, blockers[], notes[])."""
    unknown = set(needs) - set(BLOCKERS)
    if unknown:
        raise ValueError(f"unknown flags {sorted(unknown)}; known: {sorted(BLOCKERS)}")
    blockers = [BLOCKERS[k] for k, v in needs.items() if v]
    notes = []
    if wall_units is not None:
        dx, dyw, dz = (float(wall_units[k]) for k in ("dx", "dy_wall", "dz"))
        notes.append("turbulence resolved without a model (gpucfd.turbulence; cost and limits in gpucfd/DESIGN.md): "
                     "measured on one flow only, the plane channel at friction Reynolds number 180 with a wall cell of "
                     "1 wall unit. With dx+ 35 and dz+ 12 the mean flow rate came out 7.5 % low; with dx+ 12 and dz+ 12 "
                     "about 1 % low (that run had not settled). Outputs are time averages with a statistical error "
                     "(cfdlib.signals.stationary): plan for tens of time units delta / u_tau per grid")
        if dx > 20 or dz > 15:
            notes.append(f"dx+ {dx:.3g}, dz+ {dz:.3g}: coarser than the grid that came within about 1 %; expect an error of "
                         "several per cent in the mean flow and say so, or refine")
        if dyw > 2:
            notes.append(f"wall cell of {dyw:.3g} wall units: untested above 1; the wall shear is formed from that cell")
        cell_reynolds = None      # the cell Reynolds number of the mean flow is large by design in such a run
    if needs_stretched_grid:
        notes.append("thin layers: use a grid stretched per direction (gpucfd.grids.tanh toward walls, "
                     "gpucfd.grids.clustered around a body). The fine bands cross the whole domain, and the "
                     "explicit time step follows the smallest cell squared (scheme imex does not, at about twice the cost per "
                     "step with a body): estimate cells and steps before choosing gpucfd. "
                     "Below a few hundred thousand cells a step costs several milliseconds almost whatever the size, so "
                     "there fewer cells buy accuracy per cell, not time (case 4: a third of the cells, nearly the same time)")
    if cell_reynolds is not None:
        if cell_reynolds > 10:
            blockers.append(f"cell Reynolds number {cell_reynolds:.3g}: central differences would oscillate")
        elif cell_reynolds > 2:
            notes.append(f"cell Reynolds number {cell_reynolds:.3g} > 2: expect wiggles; check them on the finer grids")
    return dict(ok=not blockers, blockers=blockers, notes=notes)


def pick_scheme(h_min, nu, u, cfl=0.8, step_cost=2.0):
    """Which time scheme for one grid level: explicit "rk3" or "imex" (implicit diffusion)?

    The explicit scheme's step is the smaller of the flow-speed limit cfl h / u and the viscous limit
    0.15 h^2 / nu; IMEX has only the first, but a step costs more. h_min: smallest cell size, u: the
    velocity scale used for the time step. step_cost: time of an IMEX step over an explicit one at the
    tolerance each needs. The default 2.0 was measured with an immersed body (case 4: 1.7 to 2.1,
    because the body makes IMEX's systems non-symmetric); it is not measured without a body, so there
    time about 200 steps of each scheme and pass the ratio. Both schemes reach the same answer (gpucfd
    rung 14; case 4's forces to 0.01 %), so the levels of one study may use different schemes.

    Returns dict(scheme, dt_rk3, dt_imex, fewer_steps, expected_gain, why). expected_gain is
    fewer_steps / step_cost: an estimate, to be replaced by a timing before it is quoted (case 4 at
    140 cells per diameter: estimated 1.66, measured 1.55; at 70: 0.83 and 0.96; at 35: 0.5 and 0.6).
    The pressure tolerance has to be checked per scheme (tolerance_check): on case 4's coarsest level
    IMEX needed 1e-9 where the explicit scheme passed with 1e-8."""
    from gpucfd.gcfd import VISCOUS_DT          # the solver's own factor (0.15); importing it needs no GPU
    dt_imex = cfl * h_min / u
    dt_rk3 = min(dt_imex, VISCOUS_DT * h_min * h_min / nu)
    fewer = dt_imex / dt_rk3
    gain = fewer / step_cost
    scheme = "imex" if gain > 1.0 else "rk3"
    why = (f"the viscous limit is {fewer:.2f} times below the flow-speed limit; an IMEX step costs about {step_cost:g} "
           f"explicit ones, so IMEX would take about {1 / gain:.2f} times the explicit scheme's time")
    return dict(scheme=scheme, dt_rk3=dt_rk3, dt_imex=dt_imex, fewer_steps=fewer, expected_gain=gain, why=why)


def available():
    """dict(name, compute_capability) of the GPU if the gpucfd library loads, else None (the reason is printed)."""
    try:
        from gpucfd import gcfd
        return gcfd.device()
    except Exception as e:  # noqa: BLE001 - any failure means: not available on this machine
        print("gpucfd not available:", e)
        return None


def time_step(s, u_scale=None, cfl=0.8, factor=1.0):
    """The time step for a run of simulation s, and what set it. Use its `dt` for run_transient and pass
    the whole dict to write_run as meta["time_step"], so the record says which limit held the run back.

    u_scale: a velocity scale fixed in advance (a run from rest has nothing else): dt = factor x the
    smaller of the convective limit cfl h_min / u_scale and, for scheme "rk3", the viscous limit
    VISCOUS_DT h_min^2 / nu. Take the scale at least as large as the largest velocity the flow reaches,
    or the Courant check of the run warns. (Case 4's warning on its coarsest level, 1.15 to 1.18, is not
    that: the developed flow reaches 2.47 against a scale of 2.25, a Courant number of 0.88. It comes from
    the first steps after the start from rest, where the sampled velocity bound passes 3.)
    u_scale=None: from the field s holds now (a developed field, for example the coarser level's last
    field put on this grid), with NavierStokes.dt_cfl: the local velocities and cell sizes, which on a
    stretched grid allows a larger step than the smallest cell with the largest velocity. Not for a start
    from rest, and not tried on a case yet: watch the Courant check of the first such run.
    factor: a time-step study's multiplier (0.5 = half the step).
    Returns dict(dt, convective, viscous (None for scheme "imex"), set_by ("convective", "viscous" or
    "current field"), scheme, cfl, u_scale, h_min, factor)."""
    if u_scale is None:
        lim = s.dt_limits(1.0, cfl)
        return dict(dt=factor * s.dt_cfl(cfl), convective=None, viscous=lim["viscous"], set_by="current field",
                    scheme=s.scheme, cfl=cfl, u_scale=None, h_min=lim["h_min"], factor=factor)
    lim = s.dt_limits(u_scale, cfl)
    by = "viscous" if lim["viscous"] is not None and lim["viscous"] < lim["convective"] else "convective"
    return dict(dt=factor * s.max_dt(u_scale, cfl), convective=lim["convective"], viscous=lim["viscous"], set_by=by,
                scheme=s.scheme, cfl=cfl, u_scale=u_scale, h_min=lim["h_min"], factor=factor)


def _save_partial(path, ts, series):
    """Write what a run has recorded so far to path (.npz), replacing the previous copy in one move."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, t=np.array(ts), **{k: np.array(v) for k, v in series.items()})
    tmp.replace(path)


def run_transient(s, dt, t0, t_end, sample_dt, observe, u_ref, log=print, tag="", log_every=1.0, ptol=1e-10,
                  cfl_limit=1.0, watch=None, stop_on_watch=False, watch_every=None, partial=None, adapt_cfl=None):
    """Advance gpucfd simulation s from time t0 to t_end and sample it about every sample_dt.

    dt is the largest time step to use; it is reduced slightly so that t_end - t0 is a whole number of
    steps. Samples are taken every m steps, m = the largest whole number with m dt <= sample_dt (at least 1),
    so sampling never shortens the time step (the last interval may be shorter).
    observe(s) -> dict of name -> number or array, called at every sample (and at t0). If it returns
    'umax' (largest velocity magnitude), the Courant number is checked against cfl_limit, as umax dt over
    the smallest cell. On a grid stretched toward a wall that overstates it many times (the smallest
    cells carry the smallest velocities): return 'courant_rate' = s.courant_rate() as well, and the check
    uses each cell's own size and velocity.
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
    adapt_cfl: a Courant number to hold, for a flow whose time-step limit moves while it runs (in a
    turbulent channel it moved by half within a few time units, so a fixed step either breaks the limit
    or wastes steps). observe must return 'courant_rate'; after every sample the step becomes adapt_cfl /
    courant_rate, never above dt. Samples are then every m steps with m fixed from dt, so unevenly spaced
    in time: average them by time (cfdlib.signals.stationary does). The steps of a run with adapt_cfl
    depend on the samples, so it does not repeat a run without it step for step.
    Returns dict(t, series, perf, health, dt, watch_first), with adapt_cfl also dt_min and dt_max (dt is
    then the mean step).

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
    umax_seen, t_umax = 0.0, t0
    error = None
    interrupted = None
    first = None
    next_log = t0 + log_every
    watch_every = log_every if watch_every is None else watch_every
    next_watch = t0 + watch_every
    wall0 = time.time()

    def sample(t, dt):
        nonlocal obs_s, cfl, umax_seen, t_umax
        a = time.time()
        o = observe(s)
        obs_s += time.time() - a
        ts.append(t)
        for k, v in o.items():
            series.setdefault(k, []).append(np.asarray(v, float))
        if len(ts) > 1:  # not the start field (a projected start from rest can be far off)
            if "courant_rate" in o:
                cfl = max(cfl, float(o["courant_rate"]) * dt)
            elif "umax" in o:
                cfl = max(cfl, float(o["umax"]) * dt / h)
            if "umax" in o and float(o["umax"]) > umax_seen:
                umax_seen, t_umax = float(o["umax"]), t

    sample(t0, dt)
    adaptive = adapt_cfl is not None
    if adaptive and "courant_rate" not in series:
        raise ValueError("adapt_cfl needs 'courant_rate' from observe (s.courant_rate())")
    t, dt_now, dt_lo, dt_hi = t0, dt, math.inf, 0.0
    t_tol = 1e-9 * max(1.0, abs(t_end - t0))
    while (t < t_end - t_tol) if adaptive else (steps < n_total):
        if adaptive:
            rate = float(series["courant_rate"][-1])
            dt_now = min(dt, adapt_cfl / rate) if rate > 0 else dt
            m = nsub
            if t + m * dt_now >= t_end - t_tol:  # the last stretch: whole steps that end at t_end
                m = min(nsub, max(1, math.ceil((t_end - t) / dt_now - 1e-9)))
                dt_now = (t_end - t) / m
        else:
            m = min(nsub, n_total - steps)
        try:
            st = s.run(dt_now, m, ptol)
        except FloatingPointError as e:
            error = str(e)
            log(f"{tag}t {t + m * dt_now:.4f}: {e}")
            break
        except KeyboardInterrupt:
            interrupted = f"interrupted at t = {t:.4f} (of {t_end:g})"
            log(f"{tag}{interrupted}")
            break
        gpu_ms += st["ms"]
        its += st["poisson_its"] * m
        steps += m
        max_div = max(max_div, st["max_div"])
        t = t + m * dt_now if adaptive else t0 + steps * dt
        dt_lo, dt_hi = min(dt_lo, dt_now), max(dt_hi, dt_now)
        sample(t, dt_now)
        last_step = (t >= t_end - t_tol) if adaptive else steps == n_total
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
        f"Fluid should not cross more than about one cell per time step (limit used: {cfl_limit:g})."
        + ("" if cfl <= cfl_limit else f" The largest velocity sampled was {umax_seen:.3g}, at t = {t_umax:.4g}. Just after a "
           "start from rest that is the start transient: a smaller step for the first stretch removes it. Later in the run "
           "it is the flow itself: take the time step from a velocity scale at least that large (cfdlib.gpu.time_step)."),
        warn=True)
    rel_div = max_div * h / u_ref
    add("Divergence after projection", rel_div < 1e-6, f"{rel_div:.1e}",
        "max |div u| h / U after each step: the velocity must stay divergence-free (mass conservation).", warn=True)
    status = "fail" if any(c["status"] == "fail" for c in checks) else \
        ("warn" if any(c["status"] == "warn" for c in checks) else "ok")
    return dict(t=np.array(ts), series={k: np.array(v) for k, v in series.items()},
                perf=perf_record(gpu_ms / 1000, wall, obs_s, steps, cells, its / max(steps, 1), nsub),
                health=dict(status=status, checks=checks, steps=steps, last_time=ts[-1], umax=umax_seen or None,
                            t_umax=t_umax if umax_seen else None),
                watch_first=first, **(dict(dt=(ts[-1] - t0) / max(steps, 1), dt_min=dt_lo, dt_max=dt_hi) if adaptive else dict(dt=dt)))


def perf_record(gpu_s, wall_s, observe_s, steps, cells, poisson_its, sample_steps):
    """The `perf` block of a gpucfd run record. Two clocks, each named in its keys:
      gpu_s   time inside the solver's steps;  ms_per_step_gpu, us_per_cell_step_gpu
      wall_s  elapsed time of the whole run, sampling the outputs included (observe_s of it);
              ms_per_step_wall, us_per_cell_step_wall: the ones to set against another tool's wall time
    ms_per_step and us_per_cell_step are the GPU-time values under their older names, kept because
    records and analyses written before 2026-11-13 use them. Read a record with cfdlib.runrecord.read."""
    n, nc = max(steps, 1), max(steps * cells, 1)
    return dict(gpu_s=gpu_s, wall_s=wall_s, observe_s=observe_s, steps=steps, cells=cells, sample_steps=sample_steps,
                poisson_its=poisson_its, ms_per_step=1e3 * gpu_s / n, us_per_cell_step=1e6 * gpu_s / nc,
                ms_per_step_gpu=1e3 * gpu_s / n, us_per_cell_step_gpu=1e6 * gpu_s / nc,
                ms_per_step_wall=1e3 * wall_s / n, us_per_cell_step_wall=1e6 * wall_s / nc)


def write_run(runs_dir, name, result, meta):
    """Save a run_transient result as runs_dir/<name>.npz (t + every series) and <name>.json (meta, perf,
    health, environment). Returns the json record.

    meta: what the case knows about the run. Give at least level, cells, h (smallest cell size), source
    (the run whose last field it started from, or None), pressure_tol, scheme, and time_step (the dict
    from time_step(): both limits and which one set the step). cfdlib.runrecord.read returns this record
    and an OpenFOAM one in the same shape."""
    from gpucfd import gcfd
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(runs_dir / f"{name}.npz", t=result["t"], **result["series"])
    rec = dict(name=name, tool="gpucfd", **meta, dt=result["dt"], start_time=float(result["t"][0]),
               end_time=float(result["t"][-1]), health=result["health"], perf=result["perf"],
               watch_first=result["watch_first"], gpu=gcfd.device()["name"], python=platform.python_version(),
               wall_time=result["perf"]["wall_s"])
    rec.update({k: result[k] for k in ("dt_min", "dt_max") if k in result})   # a run with adapt_cfl: dt is its mean step
    (runs_dir / f"{name}.json").write_text(json.dumps(rec, indent=1, default=float))
    (runs_dir / f"{name}.partial.npz").unlink(missing_ok=True)  # the full record replaces run_transient's partial one
    return rec


def tolerance_check(make, dt, t0, length, sample_dt, observe, u_ref, tols=(1e-10, 1e-9, 1e-8), keys=None,
                    rule=1e-4, log=print, **run_args):
    """Which pressure tolerance may a level use? Runs the same stretch of flow once per tolerance and
    compares the sampled outputs with those of the strictest one (the same-answer rule).

    Do this on every level of a study, the finest included: the pressure update divides by the time
    step, which shrinks with the cell size (with the explicit scheme faster than it), and the error a
    fixed tolerance leaves in the pressure then weighs more in the forces (case 4: 1e-8 held on the coarsest level, the finest needed 1e-10, and with
    1e-8 its forces jittered so much that the run never became periodic).

    make() -> a new simulation holding the start field at time t0 (for example the level's saved last
    field); it is closed here after its run. length: flow time to compare over (about one period of an
    unsteady flow). observe, sample_dt, u_ref and run_args as in run_transient.
    keys: the series to compare (default: every scalar series except 'umax'). The differences are
    absolute, max over the samples of |value - strictest value|: compare coefficients, or scale the
    outputs in observe. rule: the largest difference accepted.
    Returns dict(rule, keys, runs=[dict(tol, diff{key}, worst, passes, ms_per_step, poisson_its, wall_s, steps,
    divergence)], chosen), chosen = the loosest tolerance that passes together with every stricter one.
    Save it as JSON; read_tolerance_check() reads it back (and the older layout of case 4's first checks)."""
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
                 wall_s=res["perf"]["wall_s"], steps=res["perf"]["steps"],
                 divergence=next(c["value"] for c in res["health"]["checks"] if c["name"].startswith("Divergence")))
        runs.append(r)
        log(f"pressure tolerance {tol:g}: largest difference {worst:.2e} ({'passes' if r['passes'] else 'fails'} the rule "
            f"{rule:g}), {r['ms_per_step']:.2f} ms per step, {r['poisson_its']:.1f} pressure iterations per step")
    return dict(rule=rule, keys=list(keys), runs=runs, chosen=_loosest_passing(runs))


def _loosest_passing(runs):
    """The loosest tolerance that passes together with every stricter one (runs sorted strictest first)."""
    chosen = runs[0]["tol"]
    for r in runs:
        if not r["passes"]:
            break
        chosen = r["tol"]
    return chosen


def read_tolerance_check(path, rule=None):
    """A stored pressure-tolerance check in the layout tolerance_check returns, whichever way it was
    made. Case 4's first checks (2026-11-12 and 13, gpu_tuning/ptol_<level>.json) are a list of runs with
    d_cd, d_cl and a perf block; they hold no rule, so pass the case's rule (its numerics.yaml) and
    `passes` and `chosen` are worked out here by tolerance_check's own definition. A newer file keeps the
    rule it was made with; runs written before 2026-11-13 have no wall_s (None here)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, dict):
        for r in d["runs"]:
            r.setdefault("wall_s", None)
            r.setdefault("steps", None)
        return d
    if rule is None:
        raise ValueError(f"{path}: an older tolerance check stores no rule; pass rule=")
    runs = []
    for r in sorted(d, key=lambda r: r["tol"]):
        diff = dict(cd=r["d_cd"], cl=r["d_cl"])
        worst = max(abs(v) for v in diff.values())
        p = r["perf"]
        runs.append(dict(tol=r["tol"], diff=diff, worst=worst, passes=bool(worst <= rule and r.get("health") != "fail"),
                         ms_per_step=p["ms_per_step"], poisson_its=p["poisson_its"], wall_s=p["wall_s"], steps=p["steps"],
                         divergence=r.get("div")))
    return dict(rule=rule, keys=["cd", "cl"], runs=runs, chosen=_loosest_passing(runs))


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
