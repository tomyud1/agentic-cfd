"""The one-page progress report of case 17: OpenFOAM next to gpucfd on the turbulent channel.

    PYTHONPATH=. python dev/compare/turbulent_channel_re180/brief.py [out.html]      (repository root; needs no GPU)

Reads the recorded runs: gpucfd's (results/turbulent_channel_re180/runs, dev/optimize/turbulent_channel_re180/runs)
and OpenFOAM's (runs/ next to this file, written by openfoam.py). While OpenFOAM's record is missing, its
accuracy cells say so and the page is a draft. Default output: results/turbulent_channel_re180/brief.html.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "dev" / "optimize" / "turbulent_channel_re180"))
import stage as S  # noqa: E402
from cases.turbulent_channel_re180.sketch import sketch  # noqa: E402
from cfdlib import brief, study  # noqa: E402
from cfdlib.report import m, m_fix, m_pct, m_times  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "turbulent_channel_re180" / "brief.html"
RUNS = HERE / "runs"
QS = S.QS


def load(name):
    p = RUNS / name
    return json.loads(p.read_text()) if p.exists() else None


def grid(n):
    return m(rf"{n[0]}\times{n[1]}\times{n[2]}")


g, ref, limits, numerics = S.setup()
gp, gp_new = S.assess(None), S.assess("x15z15")                 # gpucfd: the grid shared with OpenFOAM, and the finer one
of = load("openfoam_G3.json")                                   # OpenFOAM on the shared grid (None while it runs)
tune = load("tune.json")
of_new_time = load("time_new.json")                             # OpenFOAM timed on the finer grid (None until timed)
draft = of is None

NAMES = dict(bulk="Mean speed", centre="Speed at the centre", cf="Friction coefficient", urms_peak="Peak swirl strength",
             urms_peak_y="Place of that peak", U="Profile: mean velocity", u_rms="Profile: swirls along the flow",
             v_rms="Profile: swirls toward the wall", w_rms="Profile: swirls across the flow", uv="Profile: shear stress")


def cell(c, key):
    if c is None:
        return brief.wait()
    v = c["checks"][key]
    if key in ("bulk", "centre", "cf", "urms_peak"):
        text = m_pct(v["value"] / 100.0, 2, signed=True)
    elif key == "urms_peak_y":
        text = m(f"{v['value']:+.1f}")
    else:
        text = m(f"{v['value']:+.3f}")
    return text if v["ok"] else brief.miss(text)


def limit(key):
    v = gp["checks"][key]
    if key in ("bulk", "centre", "cf", "urms_peak"):
        return m_pct(v["limit"] / 100.0, 0)
    return m_fix(v["limit"], 1) + " wall units" if key == "urms_peak_y" else m_fix(v["limit"], 2)


def met(c):
    return brief.wait() if c is None else f"{sum(v['ok'] for v in c['checks'].values())} of {len(c['checks'])}"


acc_rows = [[NAMES[k], limit(k), cell(of, k), cell(gp, k), cell(gp_new, k)] for k in NAMES]
acc_rows.append(["<b>Limits met</b>", "", met(of), met(gp), met(gp_new)])

# speed: per step on the shared grid, per time unit of flow, and the whole run as it was made
of_ms = 1e3 * tune["rows"][tune["best"]]["s_per_step"] if tune else None
if of:
    of_ms = of["ms_per_step"]
of_steps = 1.0 / float(of["dt"]) if of else (1.0 / float(tune["dt"]) if tune else None)
ms = lambda x: brief.wait() if x is None else m_fix(x, 0) + " ms"                    # noqa: E731
sec = lambda x: brief.wait() if x is None else m_fix(x, 0) + " s"                     # noqa: E731
hrs = lambda x: brief.wait() if x is None else m_fix(x, 1) + " h"                     # noqa: E731
ratio = lambda a, b: brief.wait() if a is None or b is None else m_times(a / b)       # noqa: E731
of_unit = of["s_per_unit"] if of else (of_ms * of_steps / 1e3 if of_ms else None)
of_new_ms = 1e3 * of_new_time["s_per_step"] if of_new_time else None
gp_end = float(json.loads((ROOT / "results" / "turbulent_channel_re180" / "runs" / "G3.json").read_text(encoding="utf-8"))["end_time"])
of_new_run = of_new_ms * gp_new["steps_per_unit"] * (gp_new["span"] + 7.0) / 3.6e6 if of_new_ms else None
speed_rows = [
    [f"One time step, {grid(gp['n'])}", ms(of_ms), ms(gp["ms_per_step"]), ratio(of_ms, gp["ms_per_step"])],
    [f"One time unit of flow, {grid(gp['n'])}", sec(of_unit), sec(gp["s_per_unit"]), ratio(of_unit, gp["s_per_unit"])],
    [f"Whole run, {grid(gp['n'])}", (hrs(of["wall_s"] / 3600) + f" for {of['t_end']:g} time units") if of else brief.wait(),
     hrs(gp["wall_h"]) + f" for {gp_end:.0f} time units", ratio(of["wall_s"] / 3600 / of["t_end"] if of else None, gp["wall_h"] / gp_end)],
    [f"One time step, {grid(gp_new['n'])}", ms(of_new_ms), ms(gp_new["ms_per_step"]), ratio(of_new_ms, gp_new["ms_per_step"])],
    [f"Whole run, {grid(gp_new['n'])}", (hrs(of_new_run) + " (projected)") if of_new_run else brief.wait("not run"), hrs(gp_new["wall_h"]),
     ratio(of_new_run, gp_new["wall_h"])],
]

n_all = len(gp["checks"])
gp_met, new_met = sum(v["ok"] for v in gp["checks"].values()), sum(v["ok"] for v in gp_new["checks"].values())
fail, warn = [], []
if draft:
    warn.append("OpenFOAM's run on this problem has not finished: its accuracy is not known yet")
else:
    if not of["usable"]:
        warn.append("OpenFOAM's averages are not settled by the rule written before its run")
    warn.append("OpenFOAM was not run on the finer grid; its time there is projected from a short timing")
warn.append("gpucfd's finer grid is one grid, with no finer one to confirm it")
verdict, reasons = study.verdict(fail, warn)

if draft:
    headline = (f"DRAFT: OpenFOAM is still running. On the same grid gpucfd is {ratio(of_ms, gp['ms_per_step'])} faster per time step. "
                f"gpucfd meets {gp_met} of {n_all} limits on that grid and all {new_met} on a finer one.")
    plain = (f"Draft: OpenFOAM still running. gpucfd meets {gp_met} of {n_all} limits on the shared grid and {new_met} of {n_all} on a finer "
             f"grid; per time step it is {of_ms / gp['ms_per_step']:.1f} times faster than OpenFOAM on 8 cores")
    meaning = ["OpenFOAM's accuracy on this problem is being computed now.",
               "gpucfd can compute this turbulent flow correctly on a desktop GPU, in under two hours on the finer grid."]
else:
    of_met = sum(v["ok"] for v in of["checks"].values())
    headline = (f"On the same grid OpenFOAM meets {of_met} of {n_all} limits and gpucfd {gp_met}; gpucfd is "
                f"{ratio(of['ms_per_step'], gp['ms_per_step'])} faster per time step. On a finer grid gpucfd meets all {new_met}.")
    plain = (f"Same grid ({gp['n'][0]} x {gp['n'][1]} x {gp['n'][2]}): OpenFOAM meets {of_met} of {n_all} limits, gpucfd {gp_met}; gpucfd is "
             f"{of['ms_per_step'] / gp['ms_per_step']:.1f} times faster per step. On a finer grid gpucfd meets {new_met} of {n_all}")
    ob, gb = of["checks"]["bulk"]["value"], gp["checks"]["bulk"]["value"]
    closer = "gpucfd is closer to the reference" if abs(gb) < abs(ob) else "OpenFOAM is closer to the reference"
    meaning = [f"On the same cells and with the same time step {closer}: the mean speed differs by {m_pct(gb / 100.0, 2, signed=True)} "
               f"with gpucfd and by {m_pct(ob / 100.0, 2, signed=True)} with OpenFOAM."]
    weak = all(of["checks"][q]["value"] < gp["checks"][q]["value"] < 0 for q in ("v_rms", "w_rms"))
    if ob > 0 and weak and of["checks"]["uv"]["value"] < 0:      # said only when the records show it
        meaning.append("With both tools the swirls toward the wall and across the flow come out too weak on this grid, more so with OpenFOAM. "
                       "With OpenFOAM the shear stress they carry is too low as well, so the flow is braked too little and runs too fast. "
                       "The likely cause is that OpenFOAM's numerical scheme damps small swirls more than gpucfd's. That cause was not tested.")
    if of_new_run:
        meaning.append(f"gpucfd reaches an answer inside every limit on this PC in {hrs(gp_new['wall_h'])}. OpenFOAM's run on the coarser grid "
                       f"took {hrs(of['wall_s'] / 3600)}, and the finer grid would take it about {hrs(of_new_run)}.")
    else:
        meaning.append(f"gpucfd reaches an answer inside every limit on this PC in {hrs(gp_new['wall_h'])}. OpenFOAM's run on the coarser "
                       f"grid took {hrs(of['wall_s'] / 3600)}.")
    series = np.load(RUNS / "openfoam_G3.npz")                    # is OpenFOAM's mean speed settled? first against last ten time units
    t, ub = series["t"], series["bulk"]
    first = float(ub[(t >= of["t_start"]) & (t < of["t_start"] + 10.0)].mean())
    last = float(ub[(t > of["t_end"] - 10.0) & (t <= of["t_end"])].mean())
    of_drift = 100.0 * (last / first - 1.0)

b = dict(
    title="Turbulent channel flow without a turbulence model: OpenFOAM and gpucfd",
    date="2026-11-14",
    goal="Simulate a turbulent flow directly, with every swirl computed and no turbulence model, for the first time in this project, "
         "and see how accurate and how fast each of our two tools is on it.",
    story="OpenFOAM computed the project's first turbulent flow earlier, with a turbulence model (case 7, a flat plate). For this case "
          "gpucfd, the project's GPU solver, was extended to compute turbulence directly and solved the problem first. "
          "OpenFOAM then ran the same problem on the same grid with the same time step.",
    problem=dict(requirements=[
        "Fluid between two flat plates at rest, pushed along the plates by a constant force. The box repeats along the flow and across it.",
        f"Fast enough to be turbulent: friction Reynolds number {m_fix(g['u_tau'] * g['delta'] / g['nu'], 0)}.",
        "No turbulence model: the grid and the time step must resolve the swirls. The answers are averages over time.",
        "Asked for: the mean speed, the speed at the centre, the friction coefficient, and the profiles of the mean velocity, "
        "of the swirl strength in the three directions and of the shear stress.",
        "Reference: the published simulation of Vreman and Kuerten (2014), two independent codes.",
        "Pass limits, set before the runs: 1 % for the two speeds, 2 % for friction and the peak, and a largest difference along each profile.",
    ]),
    results=dict(headline=headline, meaning=meaning, limits=[
        "OpenFOAM uses the settings of its own channel tutorial, tuned only for speed (three timed tries of its pressure solver), "
        "not for accuracy. Another time scheme may do better: that would be a new test. "
        "It runs on 8 CPU cores, gpucfd on one desktop GPU: two different pieces of hardware in one PC.",
        *([] if draft else [
            f"OpenFOAM's mean speed was still changing during its averaging time: by {m_pct(of_drift / 100.0, 1, signed=True)} from the "
            f"first ten time units to the last ten. Its wall friction was within {m_pct(abs(of['imbalance']), 1)} of the driving force, "
            "inside the rule written before the run, so the run was not extended. A longer run would move its mean speed further in "
            "that direction."]),
        "The finer grid was run with gpucfd only. It is one grid, with no finer one to confirm it.",
        "Averaging times differ between the runs (30 to 38 time units); the statistical error of the mean speed is about 0.2 %.",
    ]),
    accuracy=dict(
        title="Accuracy: difference from the reference",
        head=["Requirement", "Limit", f"OpenFOAM<br>{grid(gp['n'])}", f"gpucfd<br>{grid(gp['n'])}", f"gpucfd<br>{grid(gp_new['n'])}"],
        rows=acc_rows,
        note="Per cent for the first four rows; wall units for the place of the peak; friction velocities for the profiles (largest "
             "difference along the profile). Red is outside its limit."),
    speed=dict(
        title="Speed",
        head=["", "OpenFOAM<br>8 CPU cores", "gpucfd<br>RTX 4070", "gpucfd faster by"],
        rows=speed_rows,
        note="A time unit is the time a large swirl needs to turn over once; a run needs 40 to 60 of them. "
             "Whole runs include the start-up. The two whole runs on the first grid differ in length, so their \"faster by\" compares "
             "the time per time unit."),
    links=[dict(href="report.html", text="the full report of this case (the solver agent's, verdict FAIL on its own grids)"),
           dict(href="optimize_study.html", text="the finer grid (17b)")],
    verdict=json.loads((ROOT / "results" / "turbulent_channel_re180" / "results.json").read_text(encoding="utf-8"))["verdict"],
    summary=dict(plain=plain, reasons=reasons),      # the comparison's own caveats (study.verdict), kept in the JSON
)

if __name__ == "__main__":
    from gpucfd import turbulence
    ut, nu, delta = g["u_tau"], g["nu"], g["delta"]
    uc0 = float(turbulence.reichardt(delta * ut / nu))
    svg = sketch(lambda yy: turbulence.reichardt(np.minimum(yy, 2.0 - yy) * delta * ut / nu) / uc0)
    page = brief.render_brief(b, svg, OUT)
    print(f"brief: {page}  verdict {verdict}{'  (DRAFT: OpenFOAM not finished)' if draft else ''}")
