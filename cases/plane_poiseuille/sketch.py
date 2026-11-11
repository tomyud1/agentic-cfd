"""Problem sketch: two plates, gap H, pressure-driven parabolic profile (drawn by code)."""
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from cfdlib import plots


def sketch():
    fig, ax = plots.figure(7.0, 3.2)
    ax.set_axis_off()
    ax.grid(False)
    L, Hd, t = 10.0, 3.0, 0.35            # drawing units: plate length, gap, plate thickness
    ink, blue = plots.INK, plots.SERIES[0]

    # Plates (hatched solid walls)
    for yb in (-t, Hd):
        ax.add_patch(Rectangle((0, yb), L, t, facecolor="#e1e0d9", edgecolor=plots.INK2, hatch="////", lw=1))
    ax.text(L / 2, Hd + t + 0.15, r"stationary plate:  $u = 0$ (no slip)", ha="center", va="bottom", color=ink, fontsize=10)
    ax.text(L / 2, -t - 0.15, r"stationary plate:  $u = 0$ (no slip)", ha="center", va="top", color=ink, fontsize=10)

    # Parabolic velocity profile with arrows
    x0, umax = 3.0, 3.2
    yy = np.linspace(0, Hd, 200)
    prof = umax * 4 * (yy / Hd) * (1 - yy / Hd)
    ax.plot([x0, x0], [0, Hd], color=plots.INK2, lw=1)
    ax.plot(x0 + prof, yy, color=blue, lw=2)
    for yi in np.linspace(0.3, Hd - 0.3, 7):
        ui = umax * 4 * (yi / Hd) * (1 - yi / Hd)
        ax.add_patch(FancyArrowPatch((x0, yi), (x0 + ui, yi), arrowstyle="-|>", mutation_scale=9,
                                     color=blue, lw=1.2, shrinkA=0, shrinkB=0))
    ax.text(x0 + umax + 0.12, Hd / 2, r"$u_{\max}$", color=blue, fontsize=12, va="center", ha="left")
    yl = Hd * 0.75                                      # label the curve just outside it
    ax.text(x0 + umax * 4 * (yl / Hd) * (1 - yl / Hd) + 0.2, yl, r"$u(y)$", color=blue, fontsize=12,
            va="bottom", ha="left")

    # Gap dimension H
    xd = 0.8
    ax.add_patch(FancyArrowPatch((xd, 0), (xd, Hd), arrowstyle="<|-|>", mutation_scale=9, color=ink, lw=1))
    ax.text(xd - 0.15, Hd / 2, r"$H$", ha="right", va="center", fontsize=12, color=ink)

    # Pressure gradient
    ax.add_patch(FancyArrowPatch((7.4, Hd / 2), (9.6, Hd / 2), arrowstyle="-|>", mutation_scale=14,
                                 color=plots.SERIES[1], lw=2.5))
    ax.text(8.5, Hd / 2 + 0.25, r"$dp/dx < 0$", ha="center", va="bottom", fontsize=12, color=ink)
    ax.text(8.5, Hd / 2 - 0.25, "pressure falls\nalong the flow", ha="center", va="top", fontsize=9, color=plots.INK2)

    # Axes indicator, below the lower plate and its label
    ox, oy = 0.2, -2.3
    for dx, dy, lab in ((0.9, 0, "$x$"), (0, 0.9, "$y$")):
        ax.add_patch(FancyArrowPatch((ox, oy), (ox + dx, oy + dy), arrowstyle="-|>", mutation_scale=9, color=ink, lw=1))
        ax.text(ox + dx + (0.1 if dx else 0), oy + dy + (0.1 if dy else 0), lab, fontsize=10, color=ink,
                ha="left" if dx else "center", va="center" if dx else "bottom")
    ax.text(ox + 1.5, oy + 0.1, r"origin $y = 0$ on the lower plate; the plates extend far in $x$ and $z$ (2D flow)",
            fontsize=9, color=plots.INK2, va="center")

    ax.set_xlim(-0.3, L + 0.2)
    ax.set_ylim(-2.6, Hd + 1.0)
    ax.set_aspect("equal")
    return plots.to_svg(fig)
