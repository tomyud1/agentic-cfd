"""Problem sketch: tube with an unheated entrance and a uniformly heated section (drawn by code, not to scale)."""
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from cfdlib import plots


def sketch():
    fig, ax = plots.sketch(9.0, 4.4)
    ax.set_axis_off()
    ax.grid(False)
    ink, blue, red = plots.INK, plots.SERIES[0], plots.SERIES[1]
    x_in, x0, x_out, R, t = 0.0, 3.0, 12.0, 1.0, 0.18      # drawing units

    # Tube walls: insulated (grey) and heated (orange) parts
    for yb in (R, -R - t):
        ax.add_patch(Rectangle((x_in, yb), x0 - x_in, t, facecolor="#e1e0d9", edgecolor=plots.INK2, lw=1))
        ax.add_patch(Rectangle((x0, yb), x_out - x0, t, facecolor="#f6c9b3", edgecolor=red, lw=1))
    ax.plot([x_in, x_out], [0, 0], ls=(0, (6, 3)), color=plots.MUTED, lw=0.8)

    # Wall heat flux arrows into the water
    for xq in np.linspace(x0 + 0.5, x_out - 0.5, 8):
        for s in (1, -1):
            ax.add_patch(FancyArrowPatch((xq, s * (R + t + 0.55)), (xq, s * (R + t + 0.05)), arrowstyle="-|>",
                                         mutation_scale=9, color=red, lw=1.2))
    ax.text((x0 + x_out) / 2, R + t + 0.65, r"uniform heat flux $q_w$ into the water", ha="center", va="bottom",
            color=red, fontsize=10)
    ax.text((x_in + x0) / 2, R + t + 0.12, "insulated", ha="center", va="bottom", color=plots.INK2, fontsize=9)

    # Inlet: uniform velocity profile
    for yi in np.linspace(-0.75, 0.75, 5):
        ax.add_patch(FancyArrowPatch((x_in - 1.1, yi), (x_in - 0.15, yi), arrowstyle="-|>", mutation_scale=8,
                                     color=blue, lw=1.1))
    ax.text(x_in - 0.65, -1.05, r"$U,\ T_{\mathrm{in}}$", ha="center", va="top", color=blue, fontsize=11)

    # Developed parabolic profile inside the heated section
    xp, um = 7.5, 1.6
    yy = np.linspace(-R, R, 100)
    ax.plot([xp, xp], [-R, R], color=plots.INK2, lw=0.8)
    ax.plot(xp + um * (1 - (yy / R) ** 2), yy, color=blue, lw=1.8)
    ax.text(xp + um + 0.1, 0.25, r"$u(r)$", color=blue, fontsize=11, va="bottom", ha="left")

    # Outlet
    ax.add_patch(FancyArrowPatch((x_out + 0.15, 0.0), (x_out + 1.0, 0.0), arrowstyle="-|>", mutation_scale=10,
                                 color=blue, lw=1.4))
    ax.text(x_out + 0.6, -0.15, "outlet", ha="center", va="top", color=plots.INK2, fontsize=9)

    # Dimensions below the tube
    yd = -R - t - 0.55
    for (a, b, lab) in ((x_in, x0, r"$L_0 = 20\,D$"), (x0, x_out, r"$L_h = 120\,D$")):
        ax.add_patch(FancyArrowPatch((a, yd), (b, yd), arrowstyle="<|-|>", mutation_scale=8, color=ink, lw=0.9))
        ax.text((a + b) / 2, yd - 0.12, lab, ha="center", va="top", fontsize=10, color=ink)
    ax.add_patch(FancyArrowPatch((1.6, -R), (1.6, R), arrowstyle="<|-|>", mutation_scale=8, color=ink, lw=0.9))
    ax.text(1.75, 0.35, r"$D$", ha="left", va="center", fontsize=11, color=ink)

    # x origin at the start of heating, r from the axis
    ax.plot([x0, x0], [-R - t - 0.2, R + t + 0.05], color=plots.INK2, lw=0.8, ls=":")
    ax.text(x0, -R - t - 1.25, r"$x = 0$", ha="center", va="top", fontsize=10, color=ink)
    ox, oy = -1.3, -2.85
    for dx, dy, lab in ((0.8, 0, "$x$"), (0, 0.5, "$r$")):
        ax.add_patch(FancyArrowPatch((ox, oy), (ox + dx, oy + dy), arrowstyle="-|>", mutation_scale=8, color=ink, lw=0.9))
        ax.text(ox + dx + (0.15 if dx else 0), oy + dy + (0.15 if dy else 0), lab, fontsize=10, color=ink,
                ha="left" if dx else "center", va="center" if dx else "bottom")

    # Gravity
    ax.add_patch(FancyArrowPatch((-1.0, 2.4), (-1.0, 1.5), arrowstyle="-|>", mutation_scale=10, color=ink, lw=1.2))
    ax.text(-0.85, 2.0, r"$g$", ha="left", va="center", fontsize=11, color=ink)

    ax.set_xlim(-1.6, 13.4)
    ax.set_ylim(-3.1, 2.6)
    ax.set_aspect("equal")
    return plots.to_svg(fig)
