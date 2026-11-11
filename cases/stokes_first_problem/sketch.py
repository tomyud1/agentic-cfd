"""Problem sketch: plate set in motion under deep still water; the moving layer thickens with time."""
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.special import erfc

from cfdlib import plots


def sketch():
    fig, ax = plots.figure(8.6, 4.4)
    ax.set_axis_off()
    ax.grid(False)
    ink, ink2 = plots.INK, plots.INK2
    L, Hd, t = 14.0, 5.6, 0.4                 # drawing units: plate length, water shown, plate thickness

    # Water (light tint) and the plate (hatched solid) below it
    ax.add_patch(Rectangle((0, 0), L, Hd, facecolor="#eef4fb", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0, -t), L, t, facecolor="#e1e0d9", edgecolor=ink2, hatch="////", lw=1, zorder=1))
    ax.add_patch(FancyArrowPatch((L - 3.6, -t - 0.55), (L - 0.6, -t - 0.55), arrowstyle="-|>", mutation_scale=14,
                                 color=plots.SERIES[1], lw=2.5))
    ax.text(L - 3.8, -t - 0.55, r"plate moves at $U$ for $t > 0$", ha="right", va="center", fontsize=10, color=ink)
    ax.text(0.2, -t - 0.55, r"plate $y = 0$: water sticks to it, $u = U$", ha="left", va="center", fontsize=9, color=ink2)

    # Velocity profiles u(y) at three times, drawn from a common vertical line (not to scale)
    x0, ul = 2.2, 4.2                          # profile origin and drawn length of U
    yy = np.linspace(0, Hd - 0.3, 300)
    ax.plot([x0, x0], [0, Hd - 0.3], color=ink2, lw=1)
    scale = [0.3, 0.6, 1.2]                  # sqrt(nu t) in drawing units for t1 < t2 < t3
    labels = [r"$t_1$", r"$t_2$", r"$t_3$"]
    for k, (s, lab) in enumerate(zip(scale, labels)):
        prof = ul * erfc(yy / (2 * s))
        ax.plot(x0 + prof, yy, color=plots.SERIES[0], lw=1.8 - 0.3 * k, alpha=1 - 0.25 * k)
        yl = 2 * s * 0.48                       # where u is about U / 2
        ax.annotate(lab, (x0 + ul * erfc(yl / (2 * s)), yl), xytext=(7.0, 1.0 + 0.8 * k), textcoords="data",
                    va="center",
                    fontsize=11, color=plots.SERIES[0], arrowprops=dict(arrowstyle="-", color=plots.SERIES[0], lw=0.8))
    ax.text(x0 + ul / 2, Hd - 0.05, r"profiles $u(y)$ at $t_1 < t_2 < t_3$", ha="center", va="top", fontsize=10,
            color=plots.SERIES[0])

    # Penetration depth of the latest profile
    xd = 9.0
    d3 = 2 * 1.8214 * scale[-1]          # erfc(1.8214) = 0.01
    ax.add_patch(FancyArrowPatch((xd, 0), (xd, d3), arrowstyle="<|-|>", mutation_scale=9, color=ink, lw=1))
    ax.text(xd + 0.2, d3 / 2, r"$\delta_{1\%}(t)$: height where $u = 0.01\,U$" + "\n" + r"grows like $\sqrt{\nu t}$",
            ha="left", va="center", fontsize=10, color=ink)
    ax.text(xd + 0.2, Hd - 0.5, "deep water, at rest at " + r"$t = 0$" + "\nand far above the plate",
            ha="left", va="top", fontsize=9, color=ink2)

    # Axes indicator
    ox, oy = 0.3, 3.4
    for dx, dy, lab in ((0.9, 0, "$x$"), (0, 0.9, "$y$")):
        ax.add_patch(FancyArrowPatch((ox, oy), (ox + dx, oy + dy), arrowstyle="-|>", mutation_scale=9, color=ink, lw=1))
        ax.text(ox + dx + (0.1 if dx else 0), oy + dy + (0.1 if dy else 0), lab, fontsize=10, color=ink,
                ha="left" if dx else "center", va="center" if dx else "bottom")

    ax.set_xlim(-0.2, L + 0.2)
    ax.set_ylim(-t - 1.0, Hd + 0.2)
    ax.set_aspect("equal")
    return plots.to_svg(fig, name="sketch")
