"""Problem sketch: side view of the channel (two plates, driving force, mean profile, periodic box), drawn by code."""
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from cfdlib import plots


def sketch(profile):
    """profile: function y/delta in [0, 2] -> mean velocity / centre velocity (a shape for the arrows; the
    start profile of the runs, not a result)."""
    fig, ax = plots.sketch(8.6, 4.4)
    ink, ink2, blue, orange = plots.INK, plots.INK2, plots.SERIES[0], plots.SERIES[1]
    L, H, t = 9.0, 3.6, 0.22                   # drawing units: box length, plate distance, plate thickness
    wall = dict(facecolor="#e1e0d9", edgecolor=ink2, hatch="////", lw=1)
    ax.add_patch(Rectangle((-0.6, -t), L + 1.2, t, **wall))
    ax.add_patch(Rectangle((-0.6, H), L + 1.2, t, **wall))
    ax.text(L / 2, H + t + 0.12, r"plate at rest, no slip ($y = 2\delta$)", ha="center", va="bottom", fontsize=9, color=ink2)
    ax.text(L / 2, -t - 0.12, r"plate at rest, no slip ($y = 0$)", ha="center", va="top", fontsize=9, color=ink2)

    # the periodic box
    for x in (0.0, L):
        ax.plot([x, x], [0, H], "--", color=ink2, lw=1.1)
    ax.text(L + 0.12, H * 0.5, "periodic:\nthe flow\nrepeats", fontsize=8.5, color=ink2, ha="left", va="center")

    # mean velocity profile (arrows) near the left side
    x0, scale = 0.9, 2.3
    ys = np.linspace(0.0, 2.0, 60)
    ax.plot(x0 + scale * profile(ys), ys * H / 2, color=blue, lw=1.6)
    ax.plot([x0, x0], [0, H], color=blue, lw=0.8)
    for yy in np.linspace(0.18, 1.82, 8):
        ax.add_patch(FancyArrowPatch((x0, yy * H / 2), (x0 + scale * float(profile(yy)), yy * H / 2), arrowstyle="-|>",
                                     mutation_scale=8, color=blue, lw=1))
    ax.text(x0 + scale + 0.25, 1.05, r"mean velocity $\langle u\rangle(y)$", fontsize=9.5, color=blue, ha="left", va="center")

    # eddies (sense only)
    rng = np.random.default_rng(3)
    for cx, cy, r in ((5.0, 1.9, 0.33), (6.1, 2.2, 0.42), (7.3, 1.75, 0.36), (8.2, 2.3, 0.28), (6.6, 1.45, 0.2)):
        th = np.linspace(0.2 * np.pi, 1.9 * np.pi, 60) + rng.uniform(0, 2 * np.pi)
        ax.plot(cx + r * np.cos(th), cy + r * 0.75 * np.sin(th), color=ink2, lw=0.9)
    ax.text(6.1, 3.15, "eddies of all sizes, computed, not modelled", fontsize=8.5, color=ink2, ha="center", va="center")

    # driving force
    ax.add_patch(FancyArrowPatch((4.3, 0.42), (5.5, 0.42), arrowstyle="-|>", mutation_scale=14, color=orange, lw=2.2))
    ax.text(5.7, 0.45, r"driving force $f$" + "\n(per unit mass)", fontsize=9, color=orange, ha="left", va="center")

    # dimensions
    ax.add_patch(FancyArrowPatch((0, -0.95), (L, -0.95), arrowstyle="<|-|>", mutation_scale=9, color=ink, lw=1))
    ax.text(L / 2, -1.02, r"$L_x = 4\pi\delta$", ha="center", va="top", fontsize=11, color=ink)
    ax.add_patch(FancyArrowPatch((-1.0, 0), (-1.0, H), arrowstyle="<|-|>", mutation_scale=9, color=ink, lw=1))
    ax.text(-1.1, H / 2, r"$2\delta$", ha="right", va="center", fontsize=11, color=ink)

    # axes
    ox, oy = -2.6, -1.3
    for dx, dy, lab in ((0.6, 0, "$x$ (flow)"), (0, 0.6, "$y$")):
        ax.add_patch(FancyArrowPatch((ox, oy), (ox + dx, oy + dy), arrowstyle="-|>", mutation_scale=9, color=ink, lw=1))
        ax.text(ox + dx + (0.08 if dx else 0), oy + dy + (0.06 if dy else 0), lab, fontsize=10, color=ink,
                ha="left" if dx else "center", va="center" if dx else "bottom")
    ax.text(ox, oy - 0.15, r"$z$ out of the page: periodic, width $L_z = \frac{4}{3}\pi\delta$", fontsize=8.5, color=ink2,
            ha="left", va="top")

    ax.set_xlim(-2.9, L + 1.6)
    ax.set_ylim(-1.9, H + 0.75)
    return plots.to_svg(fig, name="sketch")
