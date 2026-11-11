"""Problem sketch: the channel, the cylinder, the inflow profile, the outflow and the measured outputs (drawn by code)."""
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from cfdlib import plots


def sketch(given, probe):
    H, L, D, xc, yc = (given[k]["value"] for k in ("H", "L", "D", "xc", "yc"))
    fig, ax = plots.wide(1, 3.9)
    ax.set_axis_off()
    ax.grid(False)
    ink, ink2, blue, orange = plots.INK, plots.INK2, plots.SERIES[0], plots.SERIES[1]
    t = 0.025
    wall = dict(facecolor="#e1e0d9", edgecolor=ink2, hatch="////", lw=1)
    ax.add_patch(Rectangle((0, -t), L, t, **wall))
    ax.add_patch(Rectangle((0, H), L, t, **wall))
    ax.text(1.35, H + t + 0.012, r"no-slip walls: $\mathbf{u} = 0$", ha="center", va="bottom", fontsize=9, color=ink2)
    # Cylinder
    ax.add_patch(Circle((xc, yc), D / 2, facecolor="#c3c2b7", edgecolor=ink, lw=1.2, hatch="////"))
    # Inflow parabola
    yy = np.linspace(0, H, 60)
    Um = given["Um"]["value"]
    s = 0.12 / Um                                     # drawing scale: m of x per m/s
    u = 4 * Um * yy * (H - yy) / H ** 2
    ax.plot(-0.2 + s * u, yy, color=blue, lw=1.4)
    ax.plot([-0.2, -0.2], [0, H], color=blue, lw=0.8)
    for yi in (0.08, 0.205, 0.33):
        ui = 4 * Um * yi * (H - yi) / H ** 2
        ax.add_patch(FancyArrowPatch((-0.2, yi), (-0.2 + s * ui, yi), arrowstyle="-|>", mutation_scale=8, color=blue, lw=1))
    ax.text(-0.24, -0.16, r"inflow $u = 4U_m y(H-y)/H^2$", color=blue, fontsize=9, ha="left", va="top")
    # Outflow
    ax.text(L + 0.03, H / 2, "free outflow\n(no stress)", ha="left", va="center", fontsize=9, color=ink2)
    # Wake sketch (alternating vortices) and the probe
    for i, xv in enumerate(np.arange(0.8, 2.0, 0.27)):
        yv = yc + (0.05 if i % 2 else -0.05)
        ax.add_patch(Circle((xv, yv), 0.035, fill=False, color=ink2, lw=0.8, ls=":"))
    ax.plot([probe[0]], [probe[1]], "s", color=orange, ms=5)
    ax.annotate("wake probe", xy=probe, xytext=(0.62, 0.33), fontsize=8, color=orange, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=orange, lw=0.8, shrinkA=1, shrinkB=4))
    # Forces on the cylinder
    ax.add_patch(FancyArrowPatch((xc + 0.07, yc - 0.1), (xc + 0.25, yc - 0.1), arrowstyle="-|>", mutation_scale=10,
                                 color=orange, lw=1.5))
    ax.text(xc + 0.16, yc - 0.125, r"$F_D$ (a)", color=orange, fontsize=9, ha="center", va="top")
    ax.add_patch(FancyArrowPatch((xc - 0.1, yc + 0.06), (xc - 0.1, yc + 0.15), arrowstyle="-|>", mutation_scale=10,
                                 color=orange, lw=1.5))
    ax.text(xc - 0.13, yc + 0.11, r"$F_L$ (b)", color=orange, fontsize=9, ha="right", va="center")
    ax.text(1.3, 0.06, "vortex street: lift frequency " + r"$f$" + " (c), picture (d)", fontsize=9, color=ink2,
            ha="center", va="center")
    # Dimensions
    ax.add_patch(FancyArrowPatch((0, -0.11), (L, -0.11), arrowstyle="<|-|>", mutation_scale=8, color=ink, lw=0.9))
    ax.text(L / 2, -0.12, r"$L$", ha="center", va="top", fontsize=11, color=ink)
    ax.add_patch(FancyArrowPatch((L + 0.32, 0), (L + 0.32, H), arrowstyle="<|-|>", mutation_scale=8, color=ink, lw=0.9))
    ax.text(L + 0.34, 0.06, r"$H$", ha="left", va="center", fontsize=11, color=ink)
    ax.plot([xc, xc], [0, yc - D / 2], ls="--", color=plots.MUTED, lw=0.8)
    ax.plot([0, xc - D / 2], [yc, yc], ls="--", color=plots.MUTED, lw=0.8)
    ax.text(0.03, yc + 0.008, r"$y_c$", fontsize=9, color=ink2, ha="left", va="bottom")
    ax.text(xc + 0.012, 0.01, r"$x_c$", fontsize=9, color=ink2, ha="left", va="bottom")
    ax.text(xc + 0.065, yc + 0.045, r"$D$", fontsize=10, color=ink, ha="left", va="bottom")
    ax.set_xlim(-0.25, L + 0.45)
    ax.set_ylim(-0.2, H + 0.08)
    ax.set_aspect("equal")
    return plots.to_svg(fig, "sketch")
