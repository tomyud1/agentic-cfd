"""Problem sketch: square cavity, moving lid, the two centrelines and the primary vortex (drawn by code)."""
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from cfdlib import plots


def sketch(vortex_centre):
    """vortex_centre: (x/L, y/L) of the benchmark primary vortex, read from the reference file."""
    fig, ax = plots.figure(6.4, 5.2)
    ax.set_axis_off()
    ax.grid(False)
    S, t = 4.0, 0.22                       # drawing units: cavity side, wall thickness
    ink, ink2, blue, orange = plots.INK, plots.INK2, plots.SERIES[0], plots.SERIES[1]
    wall = dict(facecolor="#e1e0d9", edgecolor=ink2, hatch="////", lw=1)

    # Fixed walls (left, right, bottom) and the lid
    ax.add_patch(Rectangle((-t, -t), t, S + t, **wall))
    ax.add_patch(Rectangle((S, -t), t, S + t, **wall))
    ax.add_patch(Rectangle((0, -t), S, t, **wall))
    ax.add_patch(Rectangle((-t, S), S + 2 * t, 0.12, facecolor=orange, edgecolor=ink2, lw=1))
    for xa in (0.6, 1.9, 3.2):
        ax.add_patch(FancyArrowPatch((xa, S + 0.32), (xa + 0.75, S + 0.32), arrowstyle="-|>", mutation_scale=12,
                                     color=orange, lw=2))
    ax.text(S / 2, S + 0.55, r"moving lid: $u = U,\ v = 0$", ha="center", va="bottom", fontsize=11, color=ink)
    ax.text(-t - 0.12, S / 2, "fixed wall\n" + r"$\mathbf{u} = 0$", ha="right", va="center", fontsize=9, color=ink2)
    ax.text(S + t + 0.12, S / 2, "fixed wall\n" + r"$\mathbf{u} = 0$", ha="left", va="center", fontsize=9, color=ink2)
    ax.text(S / 2, -t - 0.12, r"fixed wall: $\mathbf{u} = 0$", ha="center", va="top", fontsize=9, color=ink2)

    # Centrelines where the benchmark tables are given
    ax.plot([S / 2, S / 2], [0, S], "--", color=blue, lw=1.2)
    ax.plot([0, S], [S / 2, S / 2], "--", color=plots.SERIES[2], lw=1.2)
    ax.text(S / 2 + 0.06, 0.12, r"$x = L/2$: $u$ (a), (c)", color=blue, fontsize=9, ha="left", va="bottom")
    ax.text(0.08, S / 2 + 0.06, r"$y = L/2$: $v$ (b)", color=plots.SERIES[2], fontsize=9, ha="left", va="bottom")

    # Primary vortex: clockwise circulation around the benchmark centre
    xc, yc = vortex_centre[0] * S, vortex_centre[1] * S
    th = np.linspace(0.15 * np.pi, 1.85 * np.pi, 120)
    rx, ry = 0.95, 0.75
    xs, ys = xc + rx * np.cos(-th), yc + ry * np.sin(-th)
    ax.plot(xs, ys, color=ink, lw=1.3)
    ax.add_patch(FancyArrowPatch((xs[-2], ys[-2]), (xs[-1], ys[-1]), arrowstyle="-|>", mutation_scale=12, color=ink, lw=1.3))
    ax.plot([xc], [yc], "o", color=ink, ms=4)
    ax.annotate("primary vortex (d)\ncentre and " + r"$\psi_{\min}$", xy=(xc, yc), xytext=(3.0, 1.25),
                fontsize=9, color=ink, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8, shrinkA=2, shrinkB=3))
    # Small corner eddies (sense only)
    for cx, cy in ((0.18, 0.18), (S - 0.22, 0.22)):
        ax.add_patch(plots.plt.Circle((cx, cy), 0.12, fill=False, color=ink2, lw=0.8, ls=":"))
    ax.text(S - 0.45, 0.45, "corner eddies", fontsize=8, color=ink2, ha="right", va="bottom")

    # Dimension L
    ax.add_patch(FancyArrowPatch((0, -0.75), (S, -0.75), arrowstyle="<|-|>", mutation_scale=9, color=ink, lw=1))
    ax.text(S / 2, -0.82, r"$L$", ha="center", va="top", fontsize=12, color=ink)
    ax.add_patch(FancyArrowPatch((S + 1.7, 0), (S + 1.7, S), arrowstyle="<|-|>", mutation_scale=9, color=ink, lw=1))
    ax.text(S + 1.78, S / 2, r"$L$", ha="left", va="center", fontsize=12, color=ink)

    # Axes indicator at the origin (bottom-left corner)
    ox, oy = -1.1, -0.95
    for dx, dy, lab in ((0.6, 0, "$x$"), (0, 0.6, "$y$")):
        ax.add_patch(FancyArrowPatch((ox, oy), (ox + dx, oy + dy), arrowstyle="-|>", mutation_scale=9, color=ink, lw=1))
        ax.text(ox + dx + (0.06 if dx else 0), oy + dy + (0.06 if dy else 0), lab, fontsize=10, color=ink,
                ha="left" if dx else "center", va="center" if dx else "bottom")
    ax.text(ox, oy - 0.12, "origin at the bottom-left corner;\ncavity very long in $z$ (2D flow)", fontsize=8,
            color=ink2, ha="left", va="top")

    ax.set_xlim(-1.4, S + 2.1)
    ax.set_ylim(-1.6, S + 1.0)
    ax.set_aspect("equal")
    return plots.to_svg(fig)
