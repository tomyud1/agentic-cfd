"""gpucfd as a tool of the agent: when it fits a problem, and a run loop that records a transient run in
the same kind of files as the OpenFOAM runs (runs/<name>.json + .npz), so a case's analysis can read
both and compare them.

When to choose gpucfd instead of OpenFOAM (measured numbers and limits: gpucfd/DESIGN.md):
- It fits: incompressible, laminar, constant density and viscosity, single phase, no heat transfer;
  a rectangular box (periodic, wall, inflow or outflow sides) with solid bodies inside it given as
  shapes (cylinder, sphere, box, or any signed-distance function); a uniform grid fine enough for
  central differences (cell Reynolds number U h / nu about 2 or below).
- It does not fit: turbulence models (RANS/LES), heat transfer, multiphase, compressible flow, STL
  geometry, body-fitted or stretched grids (thin boundary layers need a uniform grid fine everywhere).
- Where it pays: many time steps on grids of 10^4 to 10^7 cells (one RTX 4070 runs a 2D step in a
  few milliseconds). Bodies force the explicit RK3 scheme, whose time step is limited by
  0.15 h^2 / nu as well as by the flow speed; at high resolution and low Reynolds number that limit
  dominates (DFG 2D-1 at 70 cells per diameter).
- Steady problems: march to steady state with gpucfd.steady (grid sequencing and an error-based stop).

check_fit() turns this into a checklist for a given problem; its reasons go into decisions.yaml.
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
}


def check_fit(cell_reynolds=None, needs_stretched_grid=False, **needs):
    """Whether gpucfd can solve a problem. needs: flags named as in BLOCKERS (True = the problem needs it).
    cell_reynolds: U h / nu on the finest grid you can afford (above about 2 central differences
    oscillate). needs_stretched_grid: thin layers that a uniform grid cannot resolve at an affordable size.
    Returns dict(ok, blockers[], notes[])."""
    unknown = set(needs) - set(BLOCKERS)
    if unknown:
        raise ValueError(f"unknown flags {sorted(unknown)}; known: {sorted(BLOCKERS)}")
    blockers = [BLOCKERS[k] for k, v in needs.items() if v]
    notes = []
    if needs_stretched_grid:
        blockers.append("needs a stretched or body-fitted grid (gpucfd grids are uniform)")
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


def run_transient(s, dt, t0, t_end, sample_dt, observe, u_ref, log=print, tag="", log_every=1.0, ptol=1e-10,
                  cfl_limit=1.0, watch=None, stop_on_watch=False):
    """Advance gpucfd simulation s from time t0 to t_end and sample it about every sample_dt.

    dt is the largest time step to use; it is reduced slightly so that t_end - t0 is a whole number of
    steps. Samples are taken every m steps, m = the largest whole number with m dt <= sample_dt (at least 1),
    so sampling never shortens the time step (the last interval may be shorter).
    observe(s) -> dict of name -> number or array, called at every sample (and at t0). If it returns
    'umax' (largest velocity magnitude), the Courant number is checked against cfl_limit.
    u_ref: velocity scale for the divergence check (max |div u| h / u_ref).
    watch(t, series) -> optional text, called every log_every time units: the run's live check (a
    periodicity or steadiness test); its text is logged and its first non-empty answer is recorded.
    stop_on_watch: end the run at that first answer (t_end is then the longest the run may go).
    Returns dict(t, series, perf, health, dt, watch_first).

    A blow-up (non-finite values) ends the run early with health status 'fail' instead of raising, so
    what was recorded up to then is kept."""
    n_total = max(1, math.ceil((t_end - t0) / dt - 1e-9))
    dt = (t_end - t0) / n_total
    nsub = max(1, int(math.floor(sample_dt / dt + 1e-9)))
    h = min(s.length[d] / s.n[d] for d in range(3) if s.n[d] > 1)
    cells = int(np.prod(s.n))
    ts, series = [], {}
    gpu_ms = obs_s = its = 0.0
    steps = 0
    max_div = 0.0
    cfl = 0.0
    error = None
    first = None
    next_log = t0 + log_every
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
        gpu_ms += st["ms"]
        its += st["poisson_its"] * m
        steps += m
        max_div = max(max_div, st["max_div"])
        t = t0 + steps * dt
        sample(t)
        if t >= next_log - 1e-9 or steps == n_total:
            next_log += log_every
            msg = watch(t, {k: np.array(v) for k, v in series.items()} | {"t": np.array(ts)}) if watch else None
            if msg and first is None:
                first = dict(t=t, text=msg)
            last = {k: v[-1] for k, v in series.items() if np.ndim(v[-1]) == 0}
            vals = " ".join(f"{k} {v:.6g}" for k, v in last.items())
            log(f"{tag}t {t:.3f} {vals} | CFL {cfl:.2f} div {max_div * h / u_ref:.1e} | GPU {gpu_ms / 1000:.1f} s "
                f"wall {time.time() - wall0:.1f} s" + (f" | {msg}" if msg else ""))
            if msg and stop_on_watch:
                break
    wall = time.time() - wall0
    nan = any(not np.all(np.isfinite(np.asarray(v))) for v in series.values())
    checks = []

    def add(name, ok, value, meaning, warn=False):
        checks.append(dict(name=name, status="ok" if ok else ("warn" if warn else "fail"), value=value, meaning=meaning))
    add("Solver finished without errors", error is None, error or "no errors",
        "A blow-up means the time step is too large for the flow speed or the viscous limit.")
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
    return rec
