"""Problem sketch: the domain, its five sides, the plate, the boundary layer and the two stations (not to scale)."""
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from cfdlib import plots


def sketch(G):
    fig, ax = plots.figure(8.6, 4.6)
    ax.set_axis_off()
    ax.grid(False)
    ink, ink2, blue, orange, green = plots.INK, plots.INK2, plots.SERIES[0], plots.SERIES[1], plots.SERIES[2]
    # drawing units: x from -1.6 (inflow) to 9.6 (outflow), plate from 0; height 3.2 (the real domain is far taller
    # than the boundary layer: the sketch exaggerates the layer)
    xin, xout, H, t = -1.6, 9.6, 3.2, 0.22
    sx = xout / G["x_out"]                                     # drawing units per metre along the plate
    ax.plot([xin, xout, xout], [H, H, 0], color=ink2, lw=1)
    ax.plot([xin, xin], [0, H], color=ink2, lw=1)
    ax.plot([xin, 0], [0, 0], color=green, lw=1.6, ls="--")
    ax.add_patch(Rectangle((0, -t), xout, t, facecolor="#e1e0d9", edgecolor=ink2, hatch="////", lw=1))
    # inflow arrows
    for ya in (0.45, 1.25, 2.05, 2.85):
        ax.add_patch(FancyArrowPatch((xin - 1.25, ya), (xin - 0.15, ya), arrowstyle="-|>", mutation_scale=11, color=blue, lw=1.6))
    ax.text(xin - 0.7, H + 0.12, "inflow\n" + r"$u = U_\infty$, $\tilde\nu = 3\nu$", ha="center", va="bottom", fontsize=9, color=ink)
    # boundary layer edge (exaggerated), grows like x^0.8
    xb = np.linspace(0, xout, 80)
    yb = 1.25 * (xb / xout) ** 0.8
    ax.plot(xb, yb, color=orange, lw=1.6)
    ax.text(6.9, 1.28, "edge of the turbulent\nboundary layer", ha="center", va="bottom", fontsize=9, color=orange)
    # velocity profile at the first station
    for x, lab in ((G["x_cf"] * sx, r"$x_1$"), (G["x_2"] * sx, r"$x_2$")):
        d = 1.25 * (x / xout) ** 0.8
        ax.plot([x, x], [0, d], color=ink2, lw=0.8, ls=":")
        ax.text(x, -t - 0.1, lab, ha="center", va="top", fontsize=10, color=ink)
    x1 = G["x_cf"] * sx
    d1 = 1.25 * (x1 / xout) ** 0.8
    yy = np.linspace(0.0, d1, 30)
    uu = 0.85 * (yy / d1) ** (1 / 7)
    ax.plot(x1 + uu, yy, color=blue, lw=1.4)
    for ya in (0.25 * d1, 0.6 * d1, 0.95 * d1):
        ax.add_patch(FancyArrowPatch((x1, ya), (x1 + 0.85 * (ya / d1) ** (1 / 7), ya), arrowstyle="-|>", mutation_scale=7,
                                     color=blue, lw=1))
    # labels of the sides
    ax.text(0.5 * (xin + xout), H + 0.12, r"upper side: free stream (far field), pressure $p_\infty$", ha="center", va="bottom",
            fontsize=9, color=ink)
    ax.text(xout + 0.15, 0.5 * H, "outflow\n" + r"$p = p_\infty$", ha="left", va="center", fontsize=9, color=ink)
    ax.text(0.5 * xin, -0.12, "symmetry", ha="center", va="top", fontsize=9, color=green)
    ax.text(1.9, -t - 0.1, r"plate: smooth wall, no slip, $\tilde\nu = 0$", ha="center", va="top", fontsize=9, color=ink)
    # dimensions
    yd = -1.05
    ax.add_patch(FancyArrowPatch((0, yd), (xout, yd), arrowstyle="<|-|>", mutation_scale=9, color=ink, lw=1))
    ax.text(0.5 * xout, yd - 0.07, r"plate, $0 \leq x \leq x_{\mathrm{out}} = 2L$", ha="center", va="top", fontsize=9, color=ink)
    ax.add_patch(FancyArrowPatch((xin, yd), (0, yd), arrowstyle="<|-|>", mutation_scale=9, color=ink, lw=1))
    ax.text(0.5 * xin, yd - 0.07, r"$L/3$", ha="center", va="top", fontsize=9, color=ink)
    ax.add_patch(FancyArrowPatch((xout + 1.55, 0), (xout + 1.55, H), arrowstyle="<|-|>", mutation_scale=9, color=ink, lw=1))
    ax.text(xout + 1.63, 0.5 * H, r"$H = L$", ha="left", va="center", fontsize=9, color=ink)
    # axes
    ox, oy = 0.0, 0.0
    ax.add_patch(FancyArrowPatch((ox, oy), (ox + 0.7, oy), arrowstyle="-|>", mutation_scale=9, color=ink, lw=1.2))
    ax.add_patch(FancyArrowPatch((ox, oy), (ox, oy + 0.7), arrowstyle="-|>", mutation_scale=9, color=ink, lw=1.2))
    ax.text(ox + 0.5, oy + 0.32, "$x$", fontsize=10, color=ink, ha="center", va="bottom")
    ax.text(ox - 0.12, oy + 0.75, "$y$", fontsize=10, color=ink, ha="right", va="bottom")
    ax.text(0.5 * (xin + xout), yd - 0.62, "Not to scale: the boundary layer is at most 3 % of the domain height.", ha="center", va="top",
            fontsize=8, color=ink2)
    ax.set_xlim(xin - 1.5, xout + 2.6)
    ax.set_ylim(yd - 1.0, H + 0.95)
    ax.set_aspect("equal")
    return plots.to_svg(fig, "sketch")
