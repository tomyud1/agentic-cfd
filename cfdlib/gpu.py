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
  units. gpucfd.turbulence has the start field, the plane averages, the error of the slowly settling
  bulk velocity and a stop rule; cfdlib.signals.stationary the time average with its statistical error.
  Per step it was 2 to 9 times faster than OpenFOAM on 8 cores of the same PC (larger grids, larger
  gain), with OpenFOAM at its tutorial's settings. Case 17 is the worked example: its finest
  affordable grid (12 wall units along and across the flow) gave the mean flow rate to 0.3 % but did
  not resolve the profiles.
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
time_step() gives the step of a run with both limits and the one that set it. preflight() is the short
run before a long one (health, cost per step, projected run time); step_check() measures what the time
step does to the answer; save_state()/load_state() are the restart files; peek() looks into a running
case. run_transient() runs and
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
                     "measured on one flow only, the plane channel at friction Reynolds number 180 (case 17: cells square "
                     "in the wall plane, wall cell 2 to 1 wall units, all directions refined together). With dx+ = dz+ = "
                     "24, 16 and 12 the mean flow rate came out +1.5 %, -1.6 % and -0.3 % from the reference, each known to "
                     "about 0.2 %: no steady trend, and the good last value is partly cancellation. On that finest grid the "
                     "mean velocity profile is still 0.31 friction velocities low at y+ 22 and the fluctuation profiles are "
                     "off by up to 0.14, so judge the profiles, not only the bulk value. An earlier run with dx+ 35 and "
                     "dz+ 12 was 7.5 % low. Outputs are time averages with a statistical error (cfdlib.signals.stationary; "
                     "the bulk velocity settles slowly, about 9 time units delta / u_tau, so use "
                     "gpucfd.turbulence.bulk_average for it): about 40 time units after the start-up gave it to 0.2 %")
        if dx > 20 or dz > 15:
            notes.append(f"dx+ {dx:.3g}, dz+ {dz:.3g}: coarser than the finest grid measured (12 wall units, which did not "
                         "resolve the profiles); expect an error of several per cent in the mean flow and say so, or refine")
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


def save_state(s, path, t=0.0):
    """Store what a run needs to continue from this moment: the velocity, the pressure the scheme carries
    (IMEX, or any scheme with bodies) and the time, in one .npz file, replaced in one move. gpucfd has
    no restart files of its own. Keep such files outside git (they are large)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q = s.get()
    carried = s.scheme == "imex" or bool(getattr(s, "bodies", None))
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, t=float(t), n=np.array(s.n), **{f"q{k}": a for k, a in enumerate(q)},
             **(dict(p=s.pressure()) if carried else {}))
    tmp.replace(path)
    return path


def load_state(s, path):
    """Put a state stored by save_state into simulation s (same grid) and return its time. For a start on
    a finer grid, load it into a simulation of the stored grid and use gpucfd.steady.prolong."""
    d = np.load(path)
    if tuple(d["n"]) != tuple(s.n):
        raise ValueError(f"{path} holds a grid of {tuple(d['n'])} cells, the simulation has {tuple(s.n)}")
    s.set([d[f"q{k}"] for k in range(s.ncomp)])
    if "p" in d.files:
        s.set_pressure(d["p"])
    return float(d["t"])


def run_transient(s, dt, t0, t_end, sample_dt, observe, u_ref, log=print, tag="", log_every=1.0, ptol=1e-10,
                  cfl_limit=1.0, watch=None, stop_on_watch=False, watch_every=None, partial=None, adapt_cfl=None,
                  sample_steps=None, cfl_steps=None, checkpoint=None, checkpoint_every=None):
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
    With adapt_cfl the natural settings are steps, not times:
      sample_steps: sample every this many steps (sample_dt is then ignored, and dt may be None: no cap).
      cfl_steps: look at the Courant rate and re-choose the step every this many steps, between the
        samples too (s.courant_rate(): it reads the field back, about the cost of half a step on a
        million cells, far less than a sample of plane averages).
    The Courant number OVERSHOOTS the target by however much the rate grows between two looks, and the
    rate is an extreme of the field (one cell somewhere). Measured in a turbulent channel (case 17):
    up to 35 % within 10 steps over 40 time units, and a short pre-flight sees less than a long run
    (20 % in 4 time units). A look every 5 steps with a target of 0.75 stayed below 0.9.
    checkpoint: path of a file that receives the state (save_state) every checkpoint_every time units and
    at the end of the run: a killed run can continue (load_state), and the next grid level can start from
    this one's last field.
    Returns dict(t, series, perf, health, dt, watch_first), with adapt_cfl also dt_min and dt_max (dt is
    then the mean step), with checkpoint also state (the file).

    A blow-up (non-finite values) or Ctrl-C ends the run early instead of raising (health status 'fail' or
    'warn'), so what was recorded up to then is returned and can be saved."""
    if adapt_cfl is not None and dt is None:
        if sample_steps is None:
            raise ValueError("run_transient: without a largest step dt, give sample_steps")
        dt, n_total = math.inf, 0
    else:
        n_total = max(1, math.ceil((t_end - t0) / dt - 1e-9))
        dt = (t_end - t0) / n_total
    nsub = int(sample_steps) if sample_steps else max(1, int(math.floor(sample_dt / dt + 1e-9)))
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
    next_ckpt = t0 + checkpoint_every if (checkpoint and checkpoint_every) else math.inf

    def advance(dt_c, c):
        nonlocal gpu_ms, its, steps, max_div, dt_lo, dt_hi
        st = s.run(dt_c, c, ptol)
        gpu_ms += st["ms"]
        its += st["poisson_its"] * c
        steps += c
        max_div = max(max_div, st["max_div"])
        dt_lo, dt_hi = min(dt_lo, dt_c), max(dt_hi, dt_c)

    while (t < t_end - t_tol) if adaptive else (steps < n_total):
        try:
            if adaptive:       # nsub steps to the next sample, the step re-chosen every `look` of them
                look = min(int(cfl_steps), nsub) if cfl_steps else nsub
                rate, done = float(series["courant_rate"][-1]), 0
                while done < nsub:
                    if done:   # a look between two samples
                        rate = float(s.courant_rate())
                        cfl = max(cfl, rate * dt_now)        # what the steps just taken reached
                    if not (rate > 0 or math.isfinite(dt)):
                        raise ValueError("run_transient: a fluid at rest and no largest step dt")
                    dt_now = min(dt, adapt_cfl / rate) if rate > 0 else dt
                    c = min(look, nsub - done)
                    ends = t + c * dt_now >= t_end - t_tol
                    if ends:   # the last stretch: whole steps that end at t_end
                        c = min(c, max(1, math.ceil((t_end - t) / dt_now - 1e-9)))
                        dt_now = (t_end - t) / c
                    advance(dt_now, c)
                    t, done = (t_end if ends else t + c * dt_now), done + c
                    if ends:
                        break
            else:
                advance(dt, min(nsub, n_total - steps))
                t = t0 + steps * dt
        except FloatingPointError as e:
            error = str(e)
            log(f"{tag}after t {t:.4f}: {e}")
            break
        except KeyboardInterrupt:
            interrupted = f"interrupted at t = {t:.4f} (of {t_end:g})"
            log(f"{tag}{interrupted}")
            break
        sample(t, dt_now)
        if t >= next_ckpt - 1e-9:
            save_state(s, checkpoint, t)
            while next_ckpt <= t + 1e-9:
                next_ckpt += checkpoint_every
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
    if checkpoint and error is None and interrupted is None and not nan:
        save_state(s, checkpoint, ts[-1])                    # the field at the record's end
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
                watch_first=first, **(dict(state=str(checkpoint)) if checkpoint else {}),
                **(dict(dt=(ts[-1] - t0) / max(steps, 1), dt_min=dt_lo, dt_max=dt_hi) if adaptive else dict(dt=dt)))


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


def thin(result, keep):
    """A run_transient result with every keep-th sample of each series (and the last one), for a run
    that was sampled often for its step control or stop rule: the stored profiles need not be closer in
    time than the averages use. The scalar series are also kept in full, as <name>_all with their times
    t_all, so the Courant and wall-shear history of the whole run stays checkable."""
    if keep <= 1:
        return result
    n = len(result["t"])
    idx = np.unique(np.append(np.arange(0, n, keep), n - 1))
    series = {k: v[idx] for k, v in result["series"].items()}
    series["t_all"] = result["t"]
    series.update({f"{k}_all": v for k, v in result["series"].items() if np.ndim(v) == 1})
    return dict(result, t=result["t"][idx], series=series)


def solver_settings(s):
    """The arguments a gpucfd simulation was made with, for a run record: every setting of the solver
    that is not a number of the run loop. A report's decisions can be checked against this list."""
    return dict(cells=[int(v) for v in s.n], box=[float(v) for v in s.length], viscosity=float(s.nu), scheme=s.scheme,
                force=[float(v) for v in getattr(s, "g", ())], side_kinds=[int(v) for v in getattr(s, "kinds", ())],
                stretched=bool(getattr(s, "stretched", False)), bodies=len(getattr(s, "bodies", None) or ()))


def write_run(runs_dir, name, result, meta, store_every=1, solver=None):
    """Save a run_transient result as runs_dir/<name>.npz (t + every series) and <name>.json (meta, perf,
    health, environment). Returns the json record.

    meta: what the case knows about the run. Give at least level, cells, h (smallest cell size), source
    (the run whose last field it started from, or None), pressure_tol, scheme, and time_step (the dict
    from time_step(): both limits and which one set the step). cfdlib.runrecord.read returns this record
    and an OpenFOAM one in the same shape.
    store_every: keep every so many samples of the array series (thin()); the record says so.
    solver: the simulation; its settings (solver_settings) go into the record as `solver`."""
    from gpucfd import gcfd
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    result = thin(result, store_every)
    np.savez_compressed(runs_dir / f"{name}.npz", t=result["t"], **result["series"])
    rec = dict(name=name, tool="gpucfd", **meta, dt=result["dt"], start_time=float(result["t"][0]),
               end_time=float(result["t"][-1]), health=result["health"], perf=result["perf"],
               watch_first=result["watch_first"], gpu=gcfd.device()["name"], python=platform.python_version(),
               wall_time=result["perf"]["wall_s"])
    rec.update({k: result[k] for k in ("dt_min", "dt_max", "state") if k in result})   # adapt_cfl: dt is the mean step
    if store_every > 1:
        rec["store_every"] = int(store_every)
    if solver is not None:
        rec["solver"] = solver_settings(solver)
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
    What to compare when there is no body: the pressure cannot change an integral of the flow (a bulk
    velocity; the pressure gradient has no mean), and without a body no force is taken from the
    pressure. Case 17 (a channel) compared the bulk velocity and the wall shear and every tolerance from
    1e-10 to 1e-6 passed eight orders below its rule: that says the choice is safe over a short stretch,
    not which tolerance is needed. There the tolerance is a cost setting (1e-6 against 1e-8: 9 to 20 %
    per step), and the number to watch is the divergence left after the projection, in each run's health
    record. Over a long run two turbulent flows part ways whatever the tolerance.
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


def _gpu_memory_mib():
    """Memory in use on the GPU in MiB (nvidia-smi), or None when it cannot be read."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - a missing tool must not end a run
        return None


def preflight(s, dt, t0, length, sample_dt, observe, u_ref, time_units=None, **run_args):
    """A short run before a long one (the gpucfd counterpart of cfdlib.monitor.preflight): is the run
    healthy, what does a step cost, and what will the long run cost?

    Runs simulation s (holding its start field) for `length` flow time with run_transient (same
    arguments; adapt_cfl, sample_steps, cfl_steps go through run_args) and returns dict(health,
    ms_per_step_wall, ms_per_step_gpu, poisson_its, observe_share (of the wall time, for the sampling),
    steps, dt_mean, steps_per_unit, wall_per_unit_s, gpu_memory_mib, result (the run_transient result)),
    and with time_units also projected_wall_s = wall_per_unit_s x time_units.
    The steps per time unit come from the second half of the short run (past the start field's own
    noise). For a run whose outputs are time averages the cost is seconds per time unit times the time
    units the stop rule will need, and the second factor is only known as a range before the run: project
    both ends. A short run also sees smaller extremes than a long one (see run_transient on the Courant
    overshoot), so leave a margin on what it shows."""
    res = run_transient(s, dt, t0, t0 + length, sample_dt, observe, u_ref, **run_args)
    t, p = res["t"], res["perf"]
    late = t >= t0 + 0.5 * length
    if "dt_min" in res and "courant_rate" in res["series"] and late.sum() > 1:
        steps_per_unit = float(np.mean(res["series"]["courant_rate"][late])) / run_args["adapt_cfl"]
    else:
        steps_per_unit = 1.0 / res["dt"]
    wall_per_unit = steps_per_unit * p["ms_per_step_wall"] / 1000.0
    out = dict(health=res["health"], ms_per_step_wall=p["ms_per_step_wall"], ms_per_step_gpu=p["ms_per_step_gpu"],
               poisson_its=p["poisson_its"], observe_share=p["observe_s"] / max(p["wall_s"], 1e-12), steps=p["steps"],
               dt_mean=res["dt"], steps_per_unit=steps_per_unit, wall_per_unit_s=wall_per_unit,
               gpu_memory_mib=_gpu_memory_mib(), result=res)
    if time_units is not None:
        out["projected_wall_s"] = wall_per_unit * time_units
    return out


def step_check(make, dt, t0, length, sample_dt, observe, u_ref, keys=None, log=print, **run_args):
    """What does the time step do to the answer? Runs the same stretch of flow from the same field with
    the step dt and with dt / 2 (fixed steps) and compares the sampled outputs time by time.

    For a chaotic flow this is the only cheap time-step check: two time AVERAGES each carry their
    statistical error, so comparing them needs runs as long as the averages themselves (case 17: half
    the step changed the mean bulk velocity by -0.11 % +- 0.26 %, which says nothing below a quarter of
    a per cent). Two runs from the same field stay together for a while before they part ways; over that
    stretch their difference is the time-step error. Keep `length` short (a fraction of a time unit of
    the flow) and read `growth`: a difference that grows many times over the stretch means the runs
    have started to part, and only the early samples measure the step.
    make, observe, sample_dt, u_ref, run_args as in tolerance_check (sample_dt a whole multiple of dt).
    keys: the series to compare (default: every scalar series except umax and courant_rate).
    Returns dict(dt, keys, diff{key: largest |value(dt) - value(dt/2)|}, first{key: the same at the
    first sample}, growth{key: diff / first}, t (the sample times), by_time{key: array})."""
    runs = []
    for step in (dt, 0.5 * dt):
        s = make()
        try:
            runs.append(run_transient(s, step, t0, t0 + length, sample_dt, observe, u_ref, log=lambda m: None, **run_args))
        finally:
            s.close()
    a, b = runs
    if len(a["t"]) != len(b["t"]) or not np.allclose(a["t"], b["t"]):
        raise ValueError("step_check: the two runs sampled at different times; make sample_dt a whole multiple of dt")
    if keys is None:
        keys = [k for k, v in a["series"].items() if k not in ("umax", "courant_rate") and np.ndim(v) == 1]
    by = {k: np.abs(a["series"][k] - b["series"][k])[1:] for k in keys}
    diff = {k: float(v.max()) for k, v in by.items()}
    first = {k: float(v[0]) for k, v in by.items()}
    growth = {k: (diff[k] / first[k] if first[k] > 0 else math.inf) for k in keys}
    for k in keys:
        log(f"time step {dt:.3g} against half of it, {k}: largest difference {diff[k]:.2e} (first sample {first[k]:.2e})")
    return dict(dt=dt, keys=list(keys), diff=diff, first=first, growth=growth, t=a["t"][1:], by_time=by,
                health=[r["health"]["status"] for r in runs])


def peek(path, windows=6):
    """A look into a gpucfd run while it runs (or after): reads the partial record that run_transient
    writes at every log line (partial=...), or a finished runs/<name>.npz, and returns text: the time
    reached and, per scalar series, its mean over the last `windows` equal stretches of the record, so a
    level that is still moving shows as a trend.   python -m cfdlib.gpu peek <file.npz>"""
    d = np.load(path)
    t = d["t_all"] if "t_all" in d.files else d["t"]
    lines = [f"{path}: {len(t)} samples, t = {t[0]:.4g} to {t[-1]:.4g}"]
    edges = np.linspace(t[0], t[-1], windows + 1)
    lines.append("  mean over " + "  ".join(f"{a:.4g}-{b:.4g}" for a, b in zip(edges[:-1], edges[1:])))
    for k in d.files:
        v = d[k + "_all"] if k + "_all" in d.files else d[k]
        if k.endswith("_all") or k in ("t", "t_all") or np.ndim(v) != 1 or len(v) != len(t):
            continue
        cells = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (t >= a) & (t <= b)
            cells.append(f"{v[m].mean():.6g}" if m.any() else "-")
        lines.append(f"  {k:14s} " + "  ".join(cells) + f"   last {v[-1]:.6g}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "peek":
        print(peek(sys.argv[2]))
    else:
        sys.exit("usage: python -m cfdlib.gpu peek <runs/name.partial.npz or runs/name.npz>")
