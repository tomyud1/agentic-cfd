"""The page of case 17's optimize stage: the new grid's profiles next to the first grid's.

    PYTHONPATH=. python dev/optimize/turbulent_channel_re180/study.py      (repository root; needs no GPU)

Reads the recorded runs (the case's finest run in results/turbulent_channel_re180/runs, the stage's runs in
runs/ next to this file) and writes results/turbulent_channel_re180/optimize_study.html and .json.
Every number is formed here or in stage.py (assess, gaps) from those records.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage as S  # noqa: E402
from cfdlib import plots, report, signals, study  # noqa: E402
from cfdlib.report import m, m_fix, m_pct, m_times  # noqa: E402

OUT = S.ROOT / "results" / "turbulent_channel_re180" / "optimize_study.html"
ORDER = (None, "z15", "y15", "x15", "x15z15")
LABEL = {None: "First grid (case 17)", "z15": "Finer across the flow", "y15": "Finer toward the wall", "x15": "Finer along the flow",
         "x15z15": "New grid"}
PROFILE = dict(U="Mean velocity", u_rms="Swirl strength along the flow", v_rms="Swirl strength toward the wall",
               w_rms="Swirl strength across the flow", uv="Shear stress carried by the swirls")
SHORT = dict(U="Mean velocity", u_rms="Along the flow", v_rms="Toward the wall", w_rms="Across the flow", uv="Shear stress")
C_FIRST, C_NEW, C_REF = "#c0392b", "#1f6fb2", "#222222"
QS = S.QS


def grid(n):
    return m(rf"{n[0]}\times{n[1]}\times{n[2]}")


def sgn(x, d=3):
    return m(f"{x:+.{d}f}")


def size(x, d=3):
    return m_fix(abs(x), d)


# ------------------------------------------------------------------------------------------ numbers
g, ref, limits, numerics = S.setup()
A = {k: S.assess(k) for k in ORDER}
base, new = A[None], A["x15z15"]
noise = json.loads((HERE / "noise.json").read_text())
final = json.loads((HERE / "final.json").read_text())
plan = S.PLAN
recs_new, t_new, d_new, y_new, yf_new = S.load_variant("x15z15")
health = {k: [r["health"]["status"] for r in S.load_variant(k)[0]] for k in ORDER if k}
met = {k: sum(A[k]["checks"][q]["ok"] for q in QS) for k in ORDER}
n_checks = len(new["checks"])
met_all = {k: sum(c["ok"] for c in A[k]["checks"].values()) for k in ORDER}
share = {k: {q: abs(A[k]["checks"][q]["value"]) / limits[q] for q in QS} for k in ORDER}
worst_new = max(QS, key=lambda q: share["x15z15"][q])
worst_base = max(QS, key=lambda q: share[None][q])

# the last refinement step in each direction that leads to the new grid, and what a second-order estimate leaves
r = 1.5
steps = []
for other, what in (("z15", "along the flow"), ("x15", "across the flow")):
    ch = new["bulk"] / A[other]["bulk"] - 1.0
    steps.append(dict(what=what, frm=A[other]["n"], change=ch, left=abs(ch) / (r ** 2 - 1.0)))

# evidence of the final run
plates = signals.stationary(t_new, d_new["tau_low"] - d_new["tau_high"])
plates_sigma = abs(plates["mean"]) / plates["stderr"]
w = signals._weights(t_new)
line = g["f"] * g["delta"] - g["f"] * yf_new[:d_new["uv"].shape[1]]
stress = g["nu"] * np.average(d_new["dUdy"], axis=0, weights=w) - np.average(d_new["uv"], axis=0, weights=w) - line
stress_max = float(np.max(np.abs(stress)))
stop_text = (recs_new[-1].get("watch_first") or {}).get("text", "")
ended_on_rule = stop_text.startswith("stop rule holds")
acc = numerics["acceptance"]
pred = final["predicted"]
stage_h = sum(A[k]["wall_h"] for k in ORDER if k)

fail, warn = [], []
if not new["all_ok"]:
    fail.append("the new grid misses a limit of the case: " + ", ".join(k for k, c in new["checks"].items() if not c["ok"]))
if any(h != "ok" for hs in health.values() for h in hs):
    fail.append("a run of the stage failed a health check")
if not ended_on_rule:
    warn.append("the final run ended at its time limit, not on its stop rule")
if plates_sigma > acc["halves_sigma"] or stress_max > acc["total_stress_abs"]:
    warn.append("the final run's averages are not settled (the two plates differ, or the total shear stress leaves its straight line)")
warn.append("the new grid is one grid, with no finer grid to confirm it: the last refinement along the flow still moved the mean speed "
            f"by {100 * abs(steps[0]['change']):.1f} per cent and the last across the flow by {100 * abs(steps[1]['change']):.1f} per cent, "
            "in opposite directions")
warn.append("the time-step check of case 17 was not repeated on the new grid")
verdict, reasons = study.verdict(fail, warn)


# ------------------------------------------------------------------------------------------ figures
def fig_profiles():
    f, (a, b) = plots.wide(2, h=3.9)
    for k, c, lab, kw in ((None, C_FIRST, "First grid", dict(marker="o", mfc="none", ms=4.5, ls="none")),
                          ("x15z15", C_NEW, "New grid", dict(marker="o", ms=3.2, ls="none"))):
        G = A[k]["gaps"]
        yu = np.array(G["U"]["y"])
        a.semilogx(yu, np.array(G["U"]["diff"]) + ref["at"]("U", yu), color=c, label=lab, **kw)
        for q in ("u_rms", "v_rms", "w_rms"):
            yq = np.array(G[q]["y"])
            b.plot(yq, np.array(G[q]["diff"]) + ref["at"](q, yq), color=c, label=lab if q == "u_rms" else None, **kw)
    yy = np.geomspace(1.0, 178.0, 300)
    a.semilogx(yy, ref["at"]("U", yy), color=C_REF, lw=1.2, label="Reference", zorder=1)
    yl = np.linspace(1.0, 178.0, 400)
    for q in ("u_rms", "v_rms", "w_rms"):
        b.plot(yl, ref["at"](q, yl), color=C_REF, lw=1.2, label="Reference" if q == "u_rms" else None, zorder=1)
    a.set_xlabel("Distance from the wall (wall units)")
    a.set_ylabel("Mean velocity (friction velocities)")
    a.set_title("Mean velocity")
    a.legend(loc="upper left")
    b.set_xlabel("Distance from the wall (wall units)")
    b.set_ylabel("Swirl strength (friction velocities)")
    b.set_title("Swirl strength in the three directions")
    b.set_xlim(0, 180)
    b.set_ylim(0, 3.4)
    b.legend(loc="upper right")
    return plots.to_svg(f, "profiles")


def fig_diff():
    f, axes = plt.subplots(2, 3, figsize=(plots.COLUMN_IN, 5.8), layout="constrained")
    flat = axes.ravel()
    for ax, q in zip(flat, QS):
        ax.axhspan(-limits[q], limits[q], color="#2e9e5b", alpha=0.13, lw=0, label="Inside the limit")
        ax.axhline(0.0, color=C_REF, lw=0.8)
        for k, c, lab in ((None, C_FIRST, "First grid"), ("x15z15", C_NEW, "New grid")):
            G = A[k]["gaps"][q]
            ax.semilogx(G["y"], G["diff"], color=c, lw=1.6, label=lab)
        top = 1.25 * max(limits[q], max(abs(v) for k in (None, "x15z15") for v in A[k]["gaps"][q]["diff"]))
        ax.set_ylim(-top, top)
        ax.set_title(PROFILE[q], fontsize=10)
        ax.set_xlabel("Distance from the wall (wall units)")
    flat[0].set_ylabel("Ours minus reference")
    flat[3].set_ylabel("Ours minus reference")
    h, lab = flat[0].get_legend_handles_labels()
    flat[5].axis("off")
    flat[5].legend(h, lab, loc="center", frameon=False)
    return plots.to_svg(f, "diff")


def fig_directions():
    f, ax = plots.figure(w=plots.COLUMN_IN * 0.82, h=3.9)
    cols = {None: C_FIRST, "z15": "#e0a030", "y15": "#8e6bbf", "x15": "#3aa6a0", "x15z15": C_NEW}
    wbar = 0.16
    for i, k in enumerate(ORDER):
        ax.bar(np.arange(len(QS)) + (i - 2) * wbar, [share[k][q] for q in QS], wbar, color=cols[k], label=LABEL[k])
    ax.axhline(1.0, color=C_REF, lw=1.2, ls="--")
    ax.annotate("limit", (len(QS) - 0.55, 1.03), ha="right", va="bottom")
    ax.set_xticks(np.arange(len(QS)))
    ax.set_xticklabels([SHORT[q] for q in QS])
    ax.set_ylabel("Gap as a share of its limit")
    ax.set_ylim(0, 1.15 * max(max(v.values()) for v in share.values()))
    ax.legend(loc="upper right", ncols=2)
    return plots.to_svg(f, "directions")


svgs = dict(profiles=fig_profiles(), diff=fig_diff(), directions=fig_directions())

# ------------------------------------------------------------------------------------------ the page
pct = lambda c, d=2: m_pct(c["value"] / 100.0, d, signed=True)                      # noqa: E731
yes = lambda ok: "yes" if ok else "<b>no</b>"                                       # noqa: E731
row_names = dict(bulk="Mean speed through the channel", centre="Speed at the centre", cf="Friction coefficient",
                 urms_peak="Peak of the swirl strength along the flow", urms_peak_y="Where that peak sits")
criteria_rows = []
for k, name in row_names.items():
    cb, cn = base["checks"][k], new["checks"][k]
    if k == "urms_peak_y":
        criteria_rows.append([name, m_fix(cb["limit"], 1) + " wall units", sgn(cb["value"], 1), sgn(cn["value"], 1), yes(cn["ok"])])
    else:
        criteria_rows.append([name, m_pct(cb["limit"] / 100.0, 0), pct(cb), pct(cn), yes(cn["ok"])])
for q in QS:
    cb, cn = base["checks"][q], new["checks"][q]
    criteria_rows.append([PROFILE[q], m_fix(cb["limit"], 2), sgn(cb["value"]) if cb["ok"] else "<b>" + sgn(cb["value"]) + "</b>", sgn(cn["value"]),
                          yes(cn["ok"])])

profile_rows = [[PROFILE[q], m_fix(limits[q], 2), size(base["checks"][q]["value"]), size(new["checks"][q]["value"]),
                 m_pct(share["x15z15"][q], 0)] for q in QS]
direction_rows = [[LABEL[k], grid(A[k]["n"])] + [size(A[k]["checks"][q]["value"], 2) if A[k]["checks"][q]["ok"]
                                                 else "<b>" + size(A[k]["checks"][q]["value"], 2) + "</b>" for q in QS]
                  + [f"{met[k]} of {len(QS)}"] for k in ORDER]
cost_rows = [[LABEL[k], grid(A[k]["n"]), m_fix(A[k]["cells"] / 1e6, 2), m_fix(A[k]["steps_per_unit"], 0), m_fix(A[k]["ms_per_step"], 0),
              m_fix(A[k]["s_per_unit"], 0), m_fix(A[k]["span"], 0), m_fix(A[k]["wall_h"], 2)] for k in ORDER]
step_rows = [[f"{s['what'].capitalize()}: from {grid(s['frm'])} to {grid(new['n'])}", m_pct(s["change"], 2, signed=True), m_pct(s["left"], 2)]
             for s in steps]
pred_rows = [[PROFILE[q], sgn(pred["pred"][q]), sgn(new["checks"][q]["value"]), sgn(new["checks"][q]["value"] - pred["pred"][q]),
              m_fix(noise["scatter"][q], 3)] for q in QS]
evidence_rows = [
    ["How the run ended", "on its stop rule" if ended_on_rule else "at its time limit", "on the stop rule"],
    ["Time averaged", m_fix(new["span"], 0) + " time units", "at least " + m_fix(plan["final"]["stop"]["min_time"], 0)],
    ["Statistical error of the mean speed", m_pct(new["bulk_stderr"] / new["bulk"], 2), "at most " + m_pct(plan["final"]["stop"]["tol"], 2)],
    ["Wall friction against the driving force", m_pct(new["imbalance"], 2, signed=True), "close to zero when settled"],
    ["The two plates against each other", m_fix(plates_sigma, 1) + " standard errors apart", "at most " + m_fix(acc["halves_sigma"], 0)],
    ["Total shear stress against its exact straight line", m_fix(stress_max, 4), "at most " + m_fix(acc["total_stress_abs"], 2)],
]

headline = (f"A grid with cells {m_times(r)} finer along the flow and across it ({grid(new['n'])}) meets all {n_checks} limits of case 17. "
            f"The first grid ({grid(base['n'])}) met {met_all[None]} of them: its mean speed was right, its profiles were not. "
            f"The run on the new grid took {m_fix(new['wall_h'], 1)} hours on the desktop GPU. "
            "It is one grid, with no finer one to confirm it.")
plain = (f"A grid 1.5 times finer along and across the flow ({new['n'][0]} x {new['n'][1]} x {new['n'][2]}) meets all {n_checks} limits of case 17 "
         f"(the first grid met {met_all[None]}): mean speed {100 * (new['bulk'] / ref['scalars']['bulk']['value'] - 1):+.2f} % with a statistical "
         f"error of {100 * new['bulk_stderr'] / new['bulk']:.2f} %, in {new['wall_h']:.1f} GPU hours. One grid, no finer one to confirm it")

s = dict(
    title="Turbulent channel: which grid gets the profiles right?",
    date="2026-11-14",
    kicker="follow-up study",
    question="Case 17 got the mean speed of a turbulent channel flow right but missed its profiles. "
             "What is the cheapest grid on this PC that gets the profiles right too?",
    verdict=verdict,
    verdict_label={"PASS": "Every limit met", "WARN": "Every limit met, on one grid only", "FAIL": "Not a valid result"}[verdict],
    summary=dict(
        headline=headline,
        plain=plain,
        reasons=reasons,
        answers=[
            dict(label="Limits of case 17 met by the new grid", value=f"{met_all['x15z15']} of {n_checks}",
                 note=f"the first grid: {met_all[None]} of {n_checks}"),
            dict(label="Mean speed against the reference", value=pct(new["checks"]["bulk"]),
                 note=f"statistical error {m_pct(new['bulk_stderr'] / new['bulk'], 2)}, limit {m_pct(acc['bulk_pct'] / 100.0, 0)}"),
            dict(label="Worst profile, as a share of its limit", value=m_pct(share["x15z15"][worst_new], 0),
                 note=f"the first grid: {m_pct(share[None][worst_base], 0)} (above 100 % is a miss)"),
            dict(label="GPU time of the run on the new grid", value=m_fix(new["wall_h"], 1) + " hours",
                 note=f"a time unit costs {m_times(new['s_per_unit'] / base['s_per_unit'])} that of the first grid"),
        ],
        findings=[
            "The cells across the flow were the main problem. Making only them finer brought "
            f"{met['z15']} of the {len(QS)} profiles inside their limits (Figure {{fig:directions}}).",
            f"The cells along the flow come second. Making only them finer brought {met['x15']} profiles inside, and it moved the mean speed "
            f"from {pct(base['checks']['bulk'], 1)} to {pct(A['x15']['checks']['bulk'], 1)}.",
            "Toward the wall the first grid was already fine enough: more cells there changed nothing.",
            "The first grid's good mean speed was partly luck. Its two grid errors pushed in opposite directions and cancelled.",
        ],
        limitations=[
            "One grid, with no finer grid to confirm it. The last refinement in each direction still moved the mean speed by "
            f"{m_pct(abs(steps[0]['change']), 1)} and {m_pct(abs(steps[1]['change']), 1)}, in opposite directions, so the small "
            "difference from the reference again includes some cancellation.",
            "The verdict of case 17 itself stays FAIL. That verdict belongs to the agent's solve; this study was made afterwards by the "
            "developer agent and stands next to it.",
            "The three one-direction runs averaged only 10 time units each. They are good for ranking the directions, not for precise numbers.",
            "One flow at one Reynolds number. The time-step check of case 17 was not repeated on the new grid.",
        ],
        recommendation="For this flow on gpucfd, use cells of about 8 wall units along the flow and across it, and a first cell of one wall "
                       "unit at the wall. Cells of 12 wall units are too coarse for the profiles.",
    ),
    setup=dict(
        intro="The flow is water-like fluid pushed between two flat plates, fast enough to be turbulent: full of swirls that keep changing. "
              "The solver (gpucfd, the project's GPU program) computes every swirl directly. Two research groups published very exact "
              "results for this flow (Vreman and Kuerten 2014); that is the reference. All runs here are the same problem with the same "
              "settings. Only the grid differs.",
        terms=[
            dict(term="Grid and cells", meaning="the box of fluid is cut into small cells, and the solver keeps one value per cell. "
                 "It cannot see a swirl that is smaller than a cell. A grid is written as cells along the flow, toward the wall and across the flow."),
            dict(term="Mean speed", meaning="how fast the fluid moves through the channel on average. One number."),
            dict(term="Profile", meaning="a curve from the wall to the middle of the channel: how the velocity, or the strength of the swirls, "
                 "changes with the distance from the wall. Harder to get right than the mean speed."),
            dict(term="Swirl strength", meaning="how strongly the velocity jumps around its average at one height (the root mean square of the "
                 "fluctuation). There is one for each direction."),
            dict(term="Friction velocity", meaning="a speed built from the friction on the wall. All speeds on this page are multiples of it; "
                 "the mean speed is about 15.7 of them."),
            dict(term="Wall unit", meaning="the natural small length of the flow at the wall. Half the channel height is 180 wall units. "
                 "The streaks of fast and slow fluid at the wall lie about 100 wall units apart."),
            dict(term="Gap", meaning="the largest difference between our profile and the reference, anywhere along the profile."),
            dict(term="Limit", meaning="how large a gap may be to pass. The solver agent of case 17 wrote the limits down before its runs; "
                 "they were not changed here."),
            dict(term="Time unit", meaning="half the channel height divided by the friction velocity: roughly the time a large swirl needs "
                 "to turn over once."),
            dict(term="Statistical error", meaning="a turbulent flow never repeats, so an average over a finite time is a little uncertain. "
                 "This number says by how much."),
        ],
        hardware=[[k, v] for k, v in report.environment_info(gpu=recs_new[0].get("gpu")) if k in ("GPU", "CPU", "Memory", "OS")],
        software=[["Solver", "gpucfd, second order in space, implicit diffusion (" + str(plan["numerics"]["scheme"]) + ")"],
                  ["Time step", f"chosen by the solver as the flow changes (Courant target {m_fix(plan['numerics']['cfl'], 2)})"],
                  ["Turbulence model", "none: every swirl is computed"],
                  ["Reference", "Vreman and Kuerten 2014, mean of their two codes"]],
        fairness=[
            "Same problem, same solver settings and same pass limits in every run. Only the grid differs.",
            "The limits and the reference are those of case 17, fixed before that case's runs.",
            "The target, the budget and the rule for picking the new grid were written down before the first run of this study.",
            f"Not like for like: the first grid's result is an average over {m_fix(base['span'], 0)} time units, the new grid's over "
            f"{m_fix(new['span'], 0)}, the one-direction runs' over {m_fix(A['z15']['span'], 0)}.",
        ],
    ),
    method=dict(
        intro="First find out which direction of the grid is too coarse, with three short runs. Then make one full run on the cheapest grid "
              "that should pass, and judge it by the limits of case 17.",
        choices=[
            dict(choice="Refine one direction at a time",
                 why=f"Three short runs, each with the cells {m_times(r)} finer in one direction only, show which direction matters. "
                     "Refining everything at once would not say where the cells are needed.",
                 alternatives="Refine all three directions together: projected at about 4 GPU hours for one run, with nothing learned "
                              "about where the cost can be saved."),
            dict(choice=f"Short runs of {m_fix(A['z15']['span'], 0)} time units for the three tests",
                 why="The gaps of the first grid are much larger than the scatter of a short average, measured beforehand on the first "
                     f"grid's own record (for the mean velocity {m_fix(noise['scatter']['U'], 3)} against a gap of "
                     f"{size(base['checks']['U']['value'], 2)}).",
                 alternatives="Long runs for each test: three times the cost for the same ranking."),
            dict(choice="Each run starts from the nearest finished flow field",
                 why="The field of the first grid is interpolated onto the new grid, so the flow is already turbulent. After 4 time units "
                     "its speed is scaled once to where the wall friction says it is heading, and it settles for 3 more. This start-up "
                     "takes 7 time units; the runs of case 17 needed about 20.",
                 alternatives="Start from rest with a disturbance, or wait for the mean speed to drift into place by itself."),
            dict(choice="Pick the new grid as the first grid plus the measured changes",
                 why="The cheapest grid whose predicted gaps all stay within 0.8 of their limits. For the picked grid the prediction is just "
                     "the first grid's result plus the change measured for each of its two refinements.",
                 alternatives="A formula that assumes the error falls with the square of the cell size. It failed its own check here: "
                              "these grids are too coarse for that rule (section 5)."),
            dict(choice="The final run stops by a rule fixed beforehand",
                 why=f"It ends when the statistical error of the mean speed is below {m_pct(plan['final']['stop']['tol'], 2)} after at least "
                     f"{m_fix(plan['final']['stop']['min_time'], 0)} time units, or after {m_fix(plan['final']['max_gpu_hours'], 0)} GPU hours.",
                 alternatives="A fixed run length chosen by eye."),
        ],
    ),
    results=dict(
        title="What the grids give",
        intro="Figure {fig:profiles} shows what a profile is: the reference curves with our points on top. On this scale both grids look "
              "close, so the tables below use the difference from the reference instead.",
        tables=[
            dict(title="The first grid and the new grid", question="Do the profiles now agree with the reference?", figure="diff",
                 head=["Profile", "Limit", "Gap, first grid", "Gap, new grid", "Share of its limit used, new grid"], rows=profile_rows,
                 result=f"All {len(QS)} gaps of the new grid are inside their limits. The first grid missed all {len(QS) - met[None]}.",
                 meaning="The new grid resolves the swirls at the wall well enough for every profile. The closest call is the "
                         + PROFILE[worst_new].lower() + ", at " + m_pct(share["x15z15"][worst_new], 0) + " of its limit.",
                 note="Gaps and limits are in friction velocities (for the shear stress, in their square)."),
            dict(title="One direction at a time", question="Which direction of the grid was too coarse?", figure="directions",
                 head=["Grid", "Cells along, toward the wall, across"] + [SHORT[q] for q in QS] + ["Limits met"], rows=direction_rows,
                 result=f"Finer across the flow: {met['z15']} of {len(QS)} profiles inside. Finer along the flow: {met['x15']}. "
                        f"Finer toward the wall: {met['y15']}, with gaps as large as on the first grid.",
                 meaning="The cells across the flow matter most, because they must catch the thin streaks at the wall. The cells along the "
                         "flow come second. Toward the wall nothing is gained, so the new grid keeps the first grid's cells there.",
                 note="Gaps in friction velocities; bold is outside its limit."),
        ],
        notes=[
            "With only the cells across the flow made finer, the swirl strength along the flow changes from too strong right at the wall "
            "to slightly too weak further out. It needs both refinements to come inside its limit.",
            "The mean speed barely shows any of this. It is the profiles that tell a resolved flow from an unresolved one.",
        ],
    ),
    same_answer=dict(
        title="Does the new grid meet every limit of the case?",
        intro="These are the limits the solver agent of case 17 set before its runs: the four numbers the problem asked for, the place of "
              "one peak, and the five profiles.",
        question="Is every output of the new grid within its limit?",
        head=["Output", "Limit", "First grid", "New grid", "New grid within its limit?"], rows=criteria_rows,
        ok=bool(new["all_ok"]),
        verdict=(f"The new grid meets all {n_checks} limits." if new["all_ok"] else "The new grid misses at least one limit.")
                + f" The first grid met {met_all[None]}.",
        result=f"New grid: {met_all['x15z15']} of {n_checks} within their limits. First grid: {met_all[None]} of {n_checks}.",
        meaning="On this grid gpucfd reproduces the published flow, in the mean speed and in the profiles.",
        note="Numbers are differences from the reference: in per cent for the first four rows, in wall units for the place of the peak, "
             "in friction velocities for the profiles (largest difference along the profile). Bold is outside its limit.",
    ),
    sections=[dict(
        key="sure", title="How sure is this result?", after="same_answer",
        intro="Meeting the limits on one grid is not the same as knowing the answer has stopped changing. Three checks say how far "
              "this result can be trusted.",
        tables=[
            dict(title="The run itself", question="Is the final run a settled, healthy average?",
                 head=["Check", "Final run", "Wanted"], rows=evidence_rows,
                 result="The run ended on its stop rule and its averages are settled.",
                 meaning="The numbers of the new grid are a sound average of that grid's flow."),
            dict(title="The last refinement steps", question="Has the mean speed stopped changing with the grid?",
                 head=["Last step that leads to the new grid", "Change of the mean speed", "Estimate of what is still left"], rows=step_rows,
                 result="No. Each last step still moved the mean speed by more than half a per cent, in opposite directions.",
                 meaning="About half a per cent of grid error is probably left from each direction, and the two partly cancel. "
                         "The mean speed of the new grid is trustworthy to about half a per cent, not to the "
                         + m_pct(abs(new["checks"]["bulk"]["value"]) / 100.0, 2) + " it happens to show. A finer grid would settle it.",
                 note="The estimate assumes the error falls with the square of the cell size, which these grids do not fully obey. "
                      "The run with finer cells along the flow only had not fully settled, so the second row is less certain."),
            dict(title="Prediction against measurement", question="Did the picked grid behave as predicted?",
                 head=["Profile", "Predicted gap", "Measured gap", "Difference", "Scatter of a short average"], rows=pred_rows,
                 result="Four of the five profiles came out close to the prediction. The mean velocity came out better than predicted.",
                 meaning="Adding up the changes measured one direction at a time is a usable guide for picking a grid, and no more than that."),
        ],
        notes=[
            "The original plan was a formula that splits each gap into shares of the three directions. Its own check failed: carried to "
            "infinitely fine cells it does not go to zero. So it was used only to rank the directions.",
        ],
    )],
    breakdown=dict(
        title="What it costs",
        intro="All runs on one desktop GPU, one at a time.",
        tables=[dict(title="Cost of each run",
                     head=["Grid", "Cells", "Million cells", "Steps per time unit", "Milliseconds per step", "Seconds per time unit",
                           "Time units averaged", "GPU hours with start-up"], rows=cost_rows,
                     result=f"The new grid has {m_times(new['cells'] / base['cells'], 2)} the cells of the first grid and costs "
                            f"{m_times(new['s_per_unit'] / base['s_per_unit'])} as much per time unit.",
                     meaning=f"A resolved run of this flow fits in under two hours on this PC. The whole study took "
                             f"{m_fix(stage_h, 1)} GPU hours of the {m_fix(plan['budget_gpu_hours'], 0)} set aside.")],
        notes=[
            "Making the cells finer across the flow lowers the number of steps per time unit: the better resolved flow has milder extremes, "
            "so the solver may take longer steps.",
            f"The cheap option: finer across the flow only costs {m_times(A['z15']['s_per_unit'] / base['s_per_unit'])} the first grid and "
            f"brings {met['z15']} of the {len(QS)} profiles inside.",
        ],
    ),
    reproducibility=dict(
        commands=[
            dict(cmd="bash dev/optimize/turbulent_channel_re180/chain.sh z15 y15 x15", what="the three one-direction runs (GPU)"),
            dict(cmd="bash dev/optimize/turbulent_channel_re180/final.sh x15z15 x15", what="the run on the new grid (GPU)"),
            dict(cmd="bash dev/optimize/turbulent_channel_re180/stage.sh table", what="the gaps of every run and the choice of the grid"),
            dict(cmd="PYTHONPATH=. python dev/optimize/turbulent_channel_re180/study.py", what="this page, from the recorded runs (no GPU)"),
        ],
        files=[
            dict(path="dev/optimize/turbulent_channel_re180/plan.yaml", what="target, rules and budget, written before the runs"),
            dict(path="dev/optimize/turbulent_channel_re180/stage.py", what="the runs and how the gaps are formed"),
            dict(path="dev/optimize/turbulent_channel_re180/runs/", what="the record of every run of this study"),
            dict(path="results/turbulent_channel_re180/runs/G3.json", what="the first grid's run, from case 17"),
            dict(path="cases/turbulent_channel_re180/", what="the case: problem, limits, reference (read, not changed)"),
        ],
        run_dirs=[dict(path="~/gpucfd_runs/turbulent_channel_re180/", what="flow fields and restart files (outside git)")],
        environment=report.environment_info(gpu=recs_new[0].get("gpu")),
    ),
    figures=[
        dict(key="profiles", section="results",
             caption="What a profile is. Left: the mean velocity from the wall to the middle of the channel. Right: the strength of the "
                     "swirls in the three directions (from the top: along the flow, across the flow, toward the wall). Black lines: the reference. Open red circles: the first grid. Blue dots: the new grid."),
        dict(key="diff", section="results",
             caption="Ours minus the reference along each profile. Red: the first grid. Blue: the new grid. A curve passes when it stays "
                     "inside the green band everywhere."),
        dict(key="directions", section="results",
             caption="The gap of each profile as a share of its limit, for the five grids. A bar above the dashed line misses its limit."),
    ],
)

insights = [
    dict(stage="gpu", logged="2026-11-14 13:25", problem="A gap given as one number could not be split into shares of the three directions",
         diagnosis="The gap is the largest difference along a profile. Its place moves from run to run and its sign with it, so the "
                   "changes of that one number do not add up.",
         fix="The comparison is made at each height of the profile, and the largest difference is taken at the end.",
         lesson="Compare profiles point by point when refining one direction at a time.",
         guideline="dev/LOOP.md, step 8."),
    dict(stage="gpu", logged="2026-11-14 13:30", problem="The planned formula for predicting a grid failed its own check",
         diagnosis="It assumes the error falls with the square of the cell size. Carried to infinitely fine cells it left a mean-velocity "
                   "gap several times the scatter, so these grids are too coarse for that rule.",
         fix="The formula was used only to rank the directions. The grid was picked as the first grid plus the two measured changes.",
         lesson="Check a prediction model on its own data before trusting what it predicts.",
         guideline="gpucfd/DESIGN.md, channel section."),
    dict(stage="gpu", logged="2026-11-14 13:35", problem="One run was still speeding up while it was averaged",
         diagnosis="The run with finer cells along the flow was scaled to its expected speed after 4 time units, while its field was still "
                   "adapting to the new grid. The scaling came too early for that run.",
         fix="The final run started from that run's field, the nearest one, and its wall friction then matched the driving force.",
         lesson="When the wall friction is still off after settling, scale once more.",
         guideline="gpucfd.turbulence.rescale_to_balance (its text)."),
]

if __name__ == "__main__":
    page = study.render_study(s, svgs, OUT, insights=insights)
    print(f"study: {page}  verdict {verdict}")
    for x in reasons:
        print("  -", x)
