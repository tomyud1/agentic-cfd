"""One shape for the record of a run, whichever tool made it.

Every run of a case leaves results/<case>/runs/<name>.json. OpenFOAM records (a case's run.py with
cfdlib.perf.summarize) and gpucfd records (cfdlib.gpu.write_run) name things differently, and one key
means two things: `perf.us_per_cell_step` is on wall time for OpenFOAM and on GPU solver time, without
the sampling of forces, for gpucfd. Dividing one by the other once gave "50 times faster"; with wall
time on both sides it was 25 to 45. So a comparison of tools reads its records here:

    from cfdlib import runrecord
    r = runrecord.read("results/<case>/runs/M4.json")      # or a record already loaded as a dict
    rows = runrecord.table("results/<case>/runs", ["M1", "M2", "G1", "C3i"])

read() returns the same keys for both tools. Every cost carries its clock in its name; a key that a
tool or an older record does not have is None, never a guess:

  name, tool ("openfoam" or "gpucfd"), level, source, scheme, cores, health ("ok", "warn", "fail")
  cells, steps, dt, t0, t1, flow_time          steps: time steps, or iterations of a steady run
  wall_s                                       elapsed time of the whole run: the fair basis between tools
  ms_per_step_wall, us_per_cell_step_wall      wall_s per step, and per cell and step
  gpu_s, observe_s                             gpucfd only: time inside the solver, and time spent sampling
  ms_per_step_gpu, us_per_cell_step_gpu        gpucfd only: solver time per step; do not set these against
                                               another tool's wall time
  h, h_min                                     cell size as recorded (a number, or a dict of named sizes)
                                               and the smallest of them
  pressure_its, pressure_tol                   mean pressure iterations per solve; the solver tolerance
  time_step                                    gpucfd records from 2026-11-13 on: dict(dt, convective,
                                               viscous, set_by, scheme, cfl, u_scale, h_min); else None
  walls, wall_spread                           every timing of this run (the run itself, then any repeat
                                               added with add_timing) and (max - min) / mean of them;
                                               wall_spread is None for a single timing
  record                                       the record as stored

Timings are single runs unless repeated: the same case run twice on one PC differed by 3 % (case 4,
2026-11-13). Where a time carries a headline, repeat the run, add the repeat with add_timing() and
quote wall_spread next to the time.
"""
import json
import math
from pathlib import Path


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def read(path_or_record):
    """A run record of either tool in the common shape (keys: module docstring)."""
    rec = path_or_record
    if not isinstance(rec, dict):
        rec = json.loads(Path(path_or_record).read_text(encoding="utf-8"))
    perf = rec.get("perf") or {}
    gpucfd = rec.get("tool") == "gpucfd"
    cells = rec.get("cells") or perf.get("cells")
    steps = perf.get("steps") or rec.get("steps") or rec.get("iterations")
    wall = _num(perf.get("wall_s")) if gpucfd else (_num(rec.get("wall_time")) or _num(perf.get("wall_s")))
    t0, t1 = _num(rec.get("start_time")), _num(rec.get("end_time"))
    h = rec.get("h")
    h_min = min((v for v in h.values() if _num(v)), default=None) if isinstance(h, dict) else _num(h)
    per_step = lambda t: 1e3 * t / steps if t is not None and steps else None
    per_cell = lambda t: 1e6 * t / (steps * cells) if t is not None and steps and cells else None
    gpu_s = _num(perf.get("gpu_s")) if gpucfd else None
    walls = [wall] + [_num(t.get("wall_s")) for t in rec.get("timings") or []]
    walls = [w for w in walls if w is not None]
    health = rec.get("health")
    return dict(
        name=rec.get("name"), tool="gpucfd" if gpucfd else "openfoam", level=rec.get("level"), source=rec.get("source"),
        scheme=rec.get("scheme"), cores=rec.get("cores") or perf.get("cores"),
        health=health.get("status") if isinstance(health, dict) else health,
        cells=cells, steps=steps, dt=_num(rec.get("dt")), t0=t0, t1=t1,
        flow_time=t1 - t0 if t0 is not None and t1 is not None else None,
        wall_s=wall, ms_per_step_wall=per_step(wall), us_per_cell_step_wall=per_cell(wall),
        gpu_s=gpu_s, observe_s=_num(perf.get("observe_s")) if gpucfd else None,
        ms_per_step_gpu=per_step(gpu_s), us_per_cell_step_gpu=per_cell(gpu_s),
        h=h, h_min=h_min,
        pressure_its=_num(perf.get("poisson_its")) if gpucfd else _num((perf.get("lin_iters") or {}).get("p")),
        pressure_tol=_num(rec.get("pressure_tol")), time_step=rec.get("time_step"),
        walls=walls, wall_spread=(max(walls) - min(walls)) / (sum(walls) / len(walls)) if len(walls) > 1 else None,
        record=rec)


def table(runs_dir, names):
    """read() for several runs of one folder, in the order given: a list of rows."""
    return [read(Path(runs_dir) / f"{n}.json") for n in names]


def add_timing(path, wall_s, note="", steps=None, gpu_s=None, when=None):
    """Add a repeated timing of a run to its record (key `timings`), without touching the run's own
    numbers: read() then reports every timing in `walls` and their spread in `wall_spread`.
    The repeat must be the same run (same grid, step, tolerance, start field and stop), timed alone;
    say in `note` how it was made. when: an ISO date (default: today)."""
    import datetime
    path = Path(path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec.setdefault("timings", []).append(dict(wall_s=float(wall_s), steps=steps, gpu_s=gpu_s, note=note,
                                              when=when or datetime.date.today().isoformat()))
    path.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    return read(rec)
