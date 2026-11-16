"""The plan page of the rotating detonation engine (RDE) challenge: the problem and the steps to it.

    PYTHONPATH=. python dev/rde/plan_page.py [out.html]        (repository root; needs no solver and no GPU)

A plan, not a result: it holds no computed answer. The set-up numbers are those of the 2D test in
Cheng, Sheng and Wang, Energies 17 (2024) 769, section 4.7 (read 2026-11-15); the final problem statement is
fixed with the supervisor. Default output: results/rde/plan.html.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from cfdlib import plots  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "rde" / "plan.html"

# The set-up of the first paper's 2D test (section 4.7), in its own units.
SETUP = dict(length_cm=12.0, height_cm=4.0, supply_atm=8.0, supply_K=540.0, cell_mm=0.1)
CELLS = round(10 * SETUP["length_cm"] / SETUP["cell_mm"]) * round(10 * SETUP["height_cm"] / SETUP["cell_mm"])

RED, BLUE, ORANGE = "#d03b3b", plots.SERIES[0], plots.SERIES[1]
FRESH = "#dbe9fb"


def arrow(ax, a, b, color, lw=1.2, size=10, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=size, color=color, lw=lw))


def fresh_layer(ax, x0, y0, sx, sy, xw=8.0, h=1.4, L=12.0):
    """The wedge of fresh gas and the detonation front in a rectangle drawn at (x0, y0) with scales sx, sy."""
    P = lambda x, y: (x0 + sx * x, y0 + sy * y)                                    # noqa: E731
    back = h * (1 - (L - xw) / L)                                                  # layer height at the right side
    for pts in ([(xw, 0), (xw, h), (L, back), (L, 0)], [(0, 0), (0, back), (xw, 0)]):
        ax.add_patch(Polygon([P(*p) for p in pts], closed=True, facecolor=FRESH, edgecolor=BLUE, lw=0.8))
    ax.plot(*zip(P(xw, 0), P(xw, h)), color=RED, lw=3, solid_capstyle="butt")
    return P


def engine_sketch():
    """The ring-shaped chamber, and how it is cut and unrolled into a rectangle."""
    fig, ax = plots.sketch(8.6, 3.5)
    ink, ink2 = plots.INK, plots.INK2
    cx, rx, ry, yb, yt = 2.0, 1.3, 0.38, 0.75, 3.0
    th = np.linspace(0, np.pi, 80)
    ax.add_patch(Ellipse((cx, yt), 2 * rx, 2 * ry, facecolor="#e1e0d9", edgecolor=ink2, lw=1.1))      # top: the gap, seen from above
    ax.add_patch(Ellipse((cx, yt), 2 * 0.95, 2 * 0.27, facecolor="#f7f7f4", edgecolor=ink2, lw=1.0))
    ax.plot(cx + rx * np.cos(th), yb - ry * np.sin(th), color=ink2, lw=1.1)                            # bottom, front half
    ax.plot(cx + rx * np.cos(th), yb + ry * np.sin(th), "--", color=ink2, lw=0.7)                      # bottom, back half
    for x in (cx - rx, cx + rx):
        ax.plot([x, x], [yb, yt], color=ink2, lw=1.1)
    front = lambda x, y: y - ry * np.sqrt(1 - ((x - cx) / rx) ** 2)                # noqa: E731  (the front surface at height y)
    ax.plot([1.45, 1.45], [front(1.45, yb), front(1.45, yb + 0.75)], color=RED, lw=3, solid_capstyle="butt")
    arrow(ax, (1.55, 0.85), (2.35, 0.8), RED, lw=1.4)
    ax.text(0.85, 1.75, "detonation wave\nruns round\nthe ring", fontsize=8.5, color=RED, ha="left", va="center")
    ax.plot([2.95, 2.95], [front(2.95, yb), front(2.95, yt)], "--", color=ink, lw=1.1)
    for x in (1.3, 2.0, 2.7):
        arrow(ax, (x, -0.2), (x, 0.3), BLUE)
        arrow(ax, (x, 3.45), (x, 3.95), ORANGE)
    ax.text(cx, -0.28, "fuel and air enter", fontsize=9, color=BLUE, ha="center", va="top")
    ax.text(cx, 4.02, "burnt gas leaves: thrust", fontsize=9, color=ORANGE, ha="center", va="bottom")
    ax.text(cx - rx - 0.12, 1.9, "thin ring-shaped\ngap between\ntwo cylinders", fontsize=8.5, color=ink2, ha="right", va="center")

    arrow(ax, (3.75, 1.85), (4.95, 1.85), ink, lw=1.6, size=14)
    ax.text(4.35, 2.0, "cut along the\ndashed line\nand unroll", fontsize=9, color=ink, ha="center", va="bottom")

    x0, y0, w, hgt = 5.35, 1.2, 3.9, 1.3
    ax.add_patch(Rectangle((x0, y0), w, hgt, facecolor="none", edgecolor=ink2, lw=1.1))
    fresh_layer(ax, x0, y0, w / 12.0, hgt / 4.0)
    for x in (x0, x0 + w):
        ax.plot([x, x], [y0, y0 + hgt], "--", color=ink, lw=1.1)
    ax.text(x0 + w / 2, y0 + hgt + 0.12, "the same chamber, laid flat", fontsize=9.5, color=ink, ha="center", va="bottom")
    ax.text(x0 + w / 2, y0 - 0.12, "the two cut edges are the same line:\nwhat leaves on the right comes back on the left",
            fontsize=8.5, color=ink2, ha="center", va="top")
    ax.set_xlim(-1.35, 9.5)
    ax.set_ylim(-0.75, 4.45)
    return plots.to_svg(fig, name="rde_engine")


def problem_sketch():
    """The flat (unwrapped) engine: a rectangle that repeats sideways, fed from below, open above."""
    fig, ax = plots.sketch(8.6, 4.25)
    ink, ink2 = plots.INK, plots.INK2
    L, H, h, xw = SETUP["length_cm"], SETUP["height_cm"], 1.4, 8.0
    fresh_layer(ax, 0, 0, 1, 1, xw=xw, h=h, L=L)
    ax.plot([0, L], [0, 0], color=ink, lw=1.6)
    ax.plot([0, L], [H, H], ":", color=ink2, lw=1.2)
    for x in (0, L):
        ax.plot([x, x], [0, H], "--", color=ink2, lw=1.1)
    ax.text(-0.18, H / 2, "repeats", fontsize=8.5, color=ink2, ha="right", va="center", rotation=90)
    ax.text(L + 0.18, H / 2, "repeats", fontsize=8.5, color=ink2, ha="left", va="center", rotation=90)

    arrow(ax, (xw + 0.2, 0.72), (xw + 1.45, 0.72), RED, lw=1.6, size=12)
    ax.text(xw + 0.25, 1.95, "detonation wave:\na shock with a flame\nright behind it", fontsize=8.5, color=RED, ha="left", va="center")
    ax.text(0.3, 0.2, "fresh mixture, filling up again", fontsize=8.5, color=BLUE, ha="left", va="center")
    ax.plot([xw, 5.6], [h, H], "--", color=ORANGE, lw=1.4)
    ax.text(5.55, 3.5, "shock wave trailing the front", fontsize=8.5, color=ORANGE, ha="right", va="center")
    ax.text(3.0, 2.1, "burnt gas, hot and fast", fontsize=9, color=ink2, ha="center", va="center")

    for x in np.arange(1.0, L, 2.0):
        arrow(ax, (x, H + 0.08), (x, H + 0.62), ink2)
    ax.text(L / 2, H + 0.72, "open end: the burnt gas leaves, no wave may bounce back", fontsize=9, color=ink2, ha="center", va="bottom")
    for x in np.arange(0.5, L, 1.0):
        arrow(ax, (x, -0.6), (x, -0.06), BLUE, size=8)
    ax.text(L / 2, -0.7, "hydrogen and air, already mixed, enter through small nozzles "
            f"(supply: {SETUP['supply_atm']:g} atm, {SETUP['supply_K']:g} K)", fontsize=9, color=BLUE, ha="center", va="top")

    arrow(ax, (0, -1.6), (L, -1.6), ink, lw=1, size=9, style="<|-|>")
    ax.text(L / 2, -1.68, f"{L:g} cm, once round the ring", fontsize=10, color=ink, ha="center", va="top")
    arrow(ax, (-1.05, 0), (-1.05, H), ink, lw=1, size=9, style="<|-|>")
    ax.text(-1.2, H / 2, f"{H:g} cm, along the engine", fontsize=10, color=ink, ha="right", va="center", rotation=90)
    ax.set_xlim(-1.9, L + 0.9)
    ax.set_ylim(-2.25, H + 1.15)
    return plots.to_svg(fig, name="rde_problem", allow=('"fresh mixture',))      # that label sits inside its own blue layer on purpose


PAPER1 = dict(href="https://www.mdpi.com/1996-1073/17/4/769", text="Cheng, Sheng and Wang, Energies 17 (2024) 769")
PAPER2 = dict(href="https://arxiv.org/abs/2508.16105", text="Kwak, Kim, Ryu and Shin, arXiv:2508.16105")

page = dict(
    title="The next challenge: a rotating detonation engine",
    date="2026-11-16",
    goal="Simulate a rotating detonation engine with an AI agent: first a flat, two-dimensional version of it, "
         "with every result on the way checked against a known answer.",
    story="Prof. Frankel proposed this engine as a challenge: his lab studies it. Every flow the project had solved before it had "
          "constant density and no burning. This engine needs shock waves and burning together, which is new for the workflow, "
          "for our use of OpenFOAM and for gpucfd, our GPU solver. So we do not jump: we climb to it in seven steps.",
    engine=[
        "A jet or rocket engine burns its fuel in a steady flame. A detonation engine burns it in a <b>detonation</b>: "
        "a shock wave with a flame locked right behind it, moving faster than sound.",
        "In a <b>rotating</b> detonation engine the chamber is a thin gap between two cylinders. Fuel and air enter at one end. "
        "One detonation wave runs round and round the ring and burns the fresh gas as fast as it arrives.",
        "Why people want it: burning in a detonation is faster and promises a higher efficiency than burning in a flame.",
        "The gap is thin, so the flow across it matters little. Cut the ring, unroll it, and the chamber becomes a flat "
        "rectangle. That is the two-dimensional problem.",
    ],
    requirements=[
        f"A rectangle {SETUP['length_cm']:g} cm long (once round the ring) and {SETUP['height_cm']:g} cm high (along the engine). "
        "Its left and right sides are the same line.",
        "Bottom: hydrogen and air, mixed in the exact ratio for complete burning, enter through small nozzles "
        f"from a supply at {SETUP['supply_atm']:g} atm and {SETUP['supply_K']:g} K.",
        "Top: an open end. The burnt gas leaves, and no wave may bounce back in.",
        "Start: a small hot spot at high pressure lights the wave. The run goes on until the wave repeats itself, turn after turn.",
        "Asked for: the speed of the wave, the mass of gas passing through per second (in and out must balance), "
        "and the pressure and temperature along the ring.",
        "Checked against: two published simulations of this kind of flat engine, and then a case from the lab "
        "whose answer the agent does not see.",
    ],
    problem_note=f"The numbers are those of the 2D test in the first paper (its section 4.7), which used cells of "
                 f"{SETUP['cell_mm']:g} mm: {CELLS:,} cells. The final problem statement is fixed with Prof. Frankel.".replace(",", " "),
    hard=dict(head=["The engine needs", "Where we are today"], rows=[
        ["Gas that is squeezed and expands, with shock waves", "<b>Done</b> (steps 1 and 2). OpenFOAM's shock solver and a new shock solver in gpucfd both match the exact solutions."],
        ["Burning: a chain of chemical reactions between hydrogen and air", "<b>Done</b> (step 3). OpenFOAM's chemistry solver and new chemistry on the GPU in gpucfd both match the reference."],
        ["Shock and burning in one solver", "<b>Done with OpenFOAM</b> (step 4). <b>In progress for gpucfd:</b> its shock solver and its chemistry are being joined."],
        ["Very small cells at the wave: the burning zone is far thinner than the engine", "Measured in one dimension (step 4): the wave's speed is right even with cells larger than the burning zone; the pressure peak at the front is not. What two dimensions need is measured on step 5. The second paper needed cells of half a micrometre to resolve the zone fully."],
        ["Checks and reports that understand such flows", "Four cases solved. What each one broke was turned into a check, a tool or a note for the next agent."],
    ]),
    steps=[
        dict(name="Shock tube", what="A tube with high pressure on one side of a thin wall and low pressure on the other. The wall is removed.",
             new="shock waves, changing density", answer="an exact formula", where="this PC, seconds", status="done",
             result='Solved on 2026-11-15 by a fresh agent in 37 minutes. The answer is within about 0.01 % of the exact one. '
                    'One expectation was not met: the error falls more slowly with the grid than the agent had predicted. '
                    '<a href="../shock_tube_sod/brief.html">One-page report</a>'),
        dict(name="A shock going round a corner", what="A shock wave runs along a wall that suddenly ends and bends round the corner.",
             new="shocks in two dimensions", answer="the exact jump across the flat part of the shock", where="this PC, minutes", status="done",
             result='Solved on 2026-11-15 by a fresh agent in about 50 minutes. All three requirements are met: the flat part of the shock '
                    'moves at the exact speed to within 0.01 %, mass and energy are kept, and the shock stays flat. '
                    'In two dimensions the solver needs a smaller time step, or the run breaks at the corner. '
                    '<a href="../shock_diffraction_corner/brief.html">One-page report</a>'),
        dict(name="Ignition in a closed box", what="Hot hydrogen and air at rest. How long until the mixture ignites?",
             new="chemistry alone, with no flow", answer="an independent chemistry code", where="this PC, seconds", status="done",
             result='Solved on 2026-11-15 by a fresh agent in about 35 minutes. The ignition time is within 0.03 % of the reference and the '
                    'final temperature within 0.002 %. The chemistry of one cell costs several hundred times more than its flow, '
                    'so chemistry on the GPU was built for gpucfd the same day; it meets the same requirements. '
                    '<a href="../h2_air_ignition_box/brief.html">One-page report</a>'),
        dict(name="Detonation in a straight tube", what="A detonation wave runs down a tube of hydrogen and air.",
             new="shock and burning together", answer="the wave's speed follows from the gas alone, with no simulation",
             where="this PC, minutes", status="done", mark="The three solvers are compared here",
             result='Solved with OpenFOAM on 2026-11-16 by a fresh agent in about an hour and a half. The wave runs 0.5 % below the '
                    'theoretical speed (required: 2 %) and the flame stays within 0.06 mm of the shock. OpenFOAM needed five passes per '
                    'time step to keep the energy; with one it lost 15 % of the heat. Almost all the time is chemistry. '
                    'The GPU solver is being built for this step now. '
                    '<a href="../detonation_tube_h2_air/brief.html">One-page report</a>'),
        dict(name="Detonation cells", what="In two dimensions a detonation front is not flat: it wrinkles into a regular pattern of cells.",
             new="the fine structure of the wave", answer="published cell patterns", where="this PC or the lab: to be measured", status="planned"),
        dict(name="The flat engine", what="The problem of this page.",
             new="injection, the open end, a wave that runs for many turns", answer="two published simulations",
             where="probably the lab's machines", status="planned"),
        dict(name="A blind engine case from the lab", what="A variant the lab has computed and not published. The agent does not see the answer.",
             new="no answer to lean on", answer="the lab compares, after we report", where="the lab's machines", status="planned"),
    ],
    loop="Each step is one case of the project's loop: a fresh AI agent solves it with the workflow and writes a checked report, "
         "then the tools and the workflow are improved with what it learned. Each step ends in a one-page report with "
         "accuracy and speed side by side.",
    ways=[
        dict(letter="A", name="OpenFOAM's own burning solver", have="Done on step 4: the wave speed is within 0.5 % of the theory.",
             idea="It was made for flames, not for shocks. The second paper shows that a solver of this kind can carry a detonation, "
                  "on another branch of OpenFOAM than ours.",
             risk="It may smear the shock or get its speed wrong. Found: with the setting of its own examples it lost 15 % of the heat "
                  "and the wave ran 17 % too slow; five passes per time step fix that, at five times the cost."),
        dict(letter="B", name="A published detonation solver", have="Not started. Lowest priority since Prof. Frankel's answers.",
             idea="Research groups have built solvers for exactly this job on top of OpenFOAM. The first paper describes one.",
             risk="That paper lists no public link to its code, so another one may be needed, and it may not build on our version."),
        dict(letter="C", name="Our own solver in gpucfd", have="Its shock solver (steps 1 and 2) and its chemistry on the GPU (step 3) are built and tested. "
                                                                 "They are being joined now, for step 4.",
             idea="A GPU solver for detonations, open to everyone. The flat engine is a rectangle that repeats sideways: "
                  "the kind of grid gpucfd is built for.",
             risk="The most work. New code must pass steps 1 to 3 before it is trusted on step 4."),
    ],
    ways_note="All three are run on step 4, the straight tube, on the same grid, because there the right answer is known without any "
              "simulation. Each gets a small tuning budget fixed in advance. The report shows accuracy and speed of the three side by side, "
              "whatever the outcome.",
    rules=[
        "Every step has a known answer, and its pass limits are written down before the runs.",
        "Code computes every number in a report. Nothing is typed in by hand.",
        "What was expected before the runs is shown next to what was found, and never changed afterwards. "
        "A step is called failed only when the experiment failed.",
        "Tools are compared on the same problem and the same grid, each with a tuning budget fixed in advance.",
    ],
    answered=[                                                    # Prof. Frankel's email of 2026-11-15 (CHANGELOG.md)
        dict(q="Is this flat engine the right target?",
             a="Yes. It is a generic engine; the lab has no specific design yet."),
        dict(q="Full hydrogen chemistry, or a one-reaction model first?",
             a="Full chemistry from the start. A one-reaction model predicts a temperature much too high. The one-reaction option is dropped."),
        dict(q="Which published detonation solver does the lab trust?",
             a="None was named. Chemistry on the GPU is the key to speed in the lab's own code, so it was built first (step 3). "
               "The published solver now has the lowest priority."),
        dict(q="Does OpenFOAM run the chemistry on the GPU? (his question to us)",
             a="As installed, it does not. gpucfd now does."),
        dict(q="Is there a lab result for step 7?",
             a="The lab has run the flat engine in two dimensions with its own code and sent a video of it."),
    ],
    questions=[
        "The set-up behind the lab's video, so that step 6 matches it. In the video the fuel enters through separate small openings, not through an even layer.",
        "How accurate must an engine answer be? Each step states its required accuracy before the runs.",
        "Which lab result stays hidden from the agent for step 7?",
        "Which GPUs can gpucfd be tested on, and how are they reached?",
    ],
    words=[
        dict(name="Shock wave", meaning="A very thin front where pressure, density and temperature jump. It moves faster than sound."),
        dict(name="Detonation", meaning="A shock wave with a flame right behind it. The shock heats the gas so much that it burns at once, and the burning keeps the shock going."),
        dict(name="Mixed in the exact ratio", meaning="Just enough air to burn all the hydrogen, with nothing left over."),
        dict(name="Repeats", meaning="What leaves through one side comes back in through the opposite side. It stands for the closed ring."),
        dict(name="Cell", meaning="One small box of the grid. The solver stores one value of each quantity per cell; smaller cells show finer detail and cost more."),
        dict(name="Solver", meaning="The program that computes the flow, step by step in time."),
        dict(name="Blind case", meaning="A problem whose answer exists but is hidden from the agent until it has reported."),
        dict(name="Tuning budget", meaning="A fixed number of short timed tries to find a tool's best settings, so that the search has an end."),
    ],
    sources=[PAPER1, PAPER2],
)

if __name__ == "__main__":
    env = Environment(loader=FileSystemLoader([str(HERE), str(ROOT / "cfdlib" / "templates")]), undefined=StrictUndefined, autoescape=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(env.get_template("plan.html.j2").render(r=page, engine_svg=engine_sketch(), problem_svg=problem_sketch()),
                   encoding="utf-8")
    print(f"plan page: {OUT}")
