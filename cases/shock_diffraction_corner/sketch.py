"""Problem sketch: the chamber, the duct's opening, the corner, and the shock a little after t = 0."""
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from cfdlib import plots


def sketch(fl):
    """fl: theory.Flow (the exact numbers of the problem). The bent shock is drawn as a schematic curve."""
    fig, ax = plots.sketch(7.8, 5.6)
    ink, ink2, wall = plots.INK, plots.INK2, "#d9d8d2"
    L, H, Hs = fl.L, fl.H, fl.Hs
    d = 0.6                                                        # length of duct drawn to the left
    t = 0.45 * fl.t_end                                            # an early time
    xs, yf, yw = fl.W * t, fl.flat_from(t), Hs - fl.Mw * fl.c0 * t
    ax.add_patch(Rectangle((0, 0), L, H, facecolor="#eef4fb", edgecolor="none"))
    ax.add_patch(Rectangle((-d, Hs), d + xs, H - Hs, facecolor="#c9dcf2", edgecolor="none"))
    ax.add_patch(Rectangle((-d, 0), d, Hs, facecolor=wall, edgecolor="none", hatch="////"))
    ax.plot([-d, 0, 0, L, L, -d], [Hs, Hs, 0, 0, H, H], color=ink, lw=2)
    ax.plot([0, 0], [Hs, H], color=plots.SERIES[1], lw=2.5, ls=(0, (4, 2)))
    ax.plot([xs, xs], [yf, H], color=ink, lw=2.2)
    th = np.linspace(0, np.pi / 2, 60)                             # the bent part: a quarter of an ellipse, schematic
    ax.plot(xs * np.cos(th), yf - (yf - yw) * np.sin(th), color=ink, lw=2.2)
    ax.plot([0], [Hs], "o", color=ink, ms=5)
    ax.add_patch(FancyArrowPatch((xs, 0.93), (xs + 0.13, 0.93), arrowstyle="-|>", mutation_scale=12, color=ink, lw=1.4))
    ax.add_patch(FancyArrowPatch((-d + 0.08, 0.86), (-0.08, 0.86), arrowstyle="-|>", mutation_scale=14, color=plots.SERIES[0], lw=1.8))
    ax.text(-d / 2, 0.67, "duct: air behind\nthe shock\n" r"$p_2,\ \rho_2,\ u_2$", ha="center", va="center", fontsize=9.5, color=ink)
    ax.text(0.025, 0.545, r"corner $(0, H_s)$", ha="left", va="center", fontsize=9, color=ink)
    ax.text(0.72, 0.47, r"still air  $p_0,\ T_0$", ha="center", va="center", fontsize=10.5, color=ink)
    ax.text(xs + 0.03, 0.82, "flat shock,\n" r"speed $W = M_s c_0$", ha="left", va="center", fontsize=9.5, color=ink)
    k = 40
    ax.annotate("bent shock", xy=(xs * np.cos(th[k]), yf - (yf - yw) * np.sin(th[k])), xytext=(0.36, 0.4), fontsize=9.5, color=ink,
                ha="left", va="center", arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
    ax.annotate("step face (wall)", xy=(0, 0.12), xytext=(0.1, 0.12), fontsize=9.5, color=ink2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
    ax.annotate(r"opening: the shock is here at $t = 0$", xy=(0, 0.985), xytext=(-0.12, 1.1), fontsize=9.5, color=plots.SERIES[1],
                ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=plots.SERIES[1], lw=0.8))
    ax.text(L / 2 + 0.2, H + 0.03, "top wall", ha="center", va="bottom", fontsize=9.5, color=ink2)
    ax.text(L + 0.03, H / 2, "right side (wall)", ha="left", va="center", fontsize=9.5, color=ink2, rotation=90)
    ax.text(L / 2, -0.035, "floor (wall)", ha="center", va="top", fontsize=9.5, color=ink2)
    ax.text(-d / 2, -0.035, "solid step under the duct", ha="center", va="top", fontsize=9.5, color=ink2)
    # dimensions
    ya = -0.17
    ax.add_patch(FancyArrowPatch((0, ya), (L, ya), arrowstyle="<|-|>", mutation_scale=9, color=ink2, lw=0.9))
    ax.text(L / 2, ya - 0.02, r"$L$", ha="center", va="top", fontsize=10.5, color=ink2)
    xa = L + 0.17
    ax.add_patch(FancyArrowPatch((xa, 0), (xa, H), arrowstyle="<|-|>", mutation_scale=9, color=ink2, lw=0.9))
    ax.text(xa + 0.02, H / 2, r"$H$", ha="left", va="center", fontsize=10.5, color=ink2)
    xa = -d - 0.07
    ax.add_patch(FancyArrowPatch((xa, 0), (xa, Hs), arrowstyle="<|-|>", mutation_scale=9, color=ink2, lw=0.9))
    ax.text(xa - 0.02, Hs / 2, r"$H_s$", ha="right", va="center", fontsize=10.5, color=ink2)
    # axes, drawn in the lower right of the chamber
    x0, y0 = 0.74, 0.04
    ax.add_patch(FancyArrowPatch((x0, y0), (x0 + 0.16, y0), arrowstyle="-|>", mutation_scale=9, color=ink, lw=1.2))
    ax.add_patch(FancyArrowPatch((x0, y0), (x0, y0 + 0.16), arrowstyle="-|>", mutation_scale=9, color=ink, lw=1.2))
    ax.text(x0 + 0.175, y0, r"$x$", ha="left", va="center", fontsize=10.5, color=ink)
    ax.text(x0, y0 + 0.175, r"$y$", ha="center", va="bottom", fontsize=10.5, color=ink)
    ax.set_xlim(-d - 0.2, L + 0.3)
    ax.set_ylim(-0.3, H + 0.22)
    return plots.to_svg(fig, name="sketch")
