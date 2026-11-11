"""Problem sketch: side view (x-y) of the channel with the cylinder and the inflow profile, and the square
cross-section (z-y) seen from the inlet, with the cylinder spanning wall to wall (drawn by code, to scale)."""
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from cfdlib import plots


def sketch(given):
    H, L, D, xc, yc = (given[k]["value"] for k in ("H", "L", "D", "xc", "yc"))
    Um = given["Um"]["value"]
    fig, (ax, bx) = plots.wide(2, 3.6, gridspec_kw=dict(width_ratios=[3.3, 1]))
    ink, ink2, blue, orange = plots.INK, plots.INK2, plots.SERIES[0], plots.SERIES[1]
    wall = dict(facecolor="#e1e0d9", edgecolor=ink2, hatch="////", lw=1)
    t = 0.025
    for a in (ax, bx):
        a.set_axis_off()
        a.grid(False)
    # ---- side view (x-y), at the mid-plane z = H/2 ---------------------------------------------
    ax.add_patch(Rectangle((0, -t), L, t, **wall))
    ax.add_patch(Rectangle((0, H), L, t, **wall))
    ax.text(2.0, H + t + 0.012, r"no-slip walls (all four)", ha="center", va="bottom", fontsize=9, color=ink2)
    ax.add_patch(Circle((xc, yc), D / 2, facecolor="#c3c2b7", edgecolor=ink, lw=1.2, hatch="////"))
    yy = np.linspace(0, H, 60)
    s = 0.2 / Um                                     # drawing scale: m of x per m/s
    u = 4 * Um * yy * (H - yy) / H ** 2              # profile in the mid-plane z = H/2
    ax.plot(-0.28 + s * u, yy, color=blue, lw=1.4)
    ax.plot([-0.28, -0.28], [0, H], color=blue, lw=0.8)
    for yi in (0.07, 0.205, 0.34):
        ui = 4 * Um * yi * (H - yi) / H ** 2
        ax.add_patch(FancyArrowPatch((-0.28, yi), (-0.28 + s * ui, yi), arrowstyle="-|>", mutation_scale=8, color=blue, lw=1))
    ax.text(-0.3, H + 0.075, r"inflow, maximum $U_m$", color=blue, fontsize=9, ha="left", va="bottom")
    ax.text(L + 0.04, H / 2, "free\noutflow", ha="left", va="center", fontsize=9, color=ink2)
    # recirculation sketch behind the cylinder (steady, closed)
    for sgn in (1, -1):
        ax.add_patch(Circle((xc + 0.085, yc + sgn * 0.022), 0.02, fill=False, color=ink2, lw=0.8, ls=":"))
    ax.annotate("reversed flow (c)", xy=(xc + 0.11, yc - 0.03), xytext=(0.9, 0.06), fontsize=8, color=ink2, ha="left",
                va="center", arrowprops=dict(arrowstyle="-", color=ink2, lw=0.7, shrinkA=1, shrinkB=2))
    ax.add_patch(FancyArrowPatch((xc + 0.08, yc + 0.1), (xc + 0.3, yc + 0.1), arrowstyle="-|>", mutation_scale=10,
                                 color=orange, lw=1.5))
    ax.text(xc + 0.33, yc + 0.1, r"$F_D$ (a)", color=orange, fontsize=9, ha="left", va="center")
    ax.add_patch(FancyArrowPatch((xc - 0.12, yc - 0.02), (xc - 0.12, yc + 0.11), arrowstyle="-|>", mutation_scale=10,
                                 color=orange, lw=1.5))
    ax.text(xc - 0.15, yc + 0.07, r"$F_L$ (b)", color=orange, fontsize=9, ha="right", va="center")
    ax.add_patch(FancyArrowPatch((0, -0.11), (L, -0.11), arrowstyle="<|-|>", mutation_scale=8, color=ink, lw=0.9))
    ax.text(L / 2, -0.125, r"$L$", ha="center", va="top", fontsize=11, color=ink)
    ax.plot([xc, xc], [0, yc - D / 2], ls="--", color=plots.MUTED, lw=0.8)
    ax.text(xc + 0.012, 0.01, r"$x_c$", fontsize=9, color=ink2, ha="left", va="bottom")
    ax.plot([xc + D / 2, xc + 0.7], [yc, yc], ls="--", color=plots.MUTED, lw=0.8)
    ax.text(xc + 0.72, yc, r"$y_c$", fontsize=9, color=ink2, ha="left", va="center")
    ax.text(xc - 0.045, yc - 0.06, r"$D$", fontsize=10, color=ink, ha="right", va="top")
    ax.text(0.8, H + 0.075, r"side view, mid-plane $z = H/2$", fontsize=9, color=ink, ha="left", va="bottom")
    ax.set_xlim(-0.32, L + 0.3)
    ax.set_ylim(-0.2, H + 0.12)
    ax.set_aspect("equal")
    # ---- cross-section (z-y) at the cylinder ---------------------------------------------------
    bx.add_patch(Rectangle((-t, -t), H + 2 * t, t, **wall))
    bx.add_patch(Rectangle((-t, H), H + 2 * t, t, **wall))
    bx.add_patch(Rectangle((-t, 0), t, H, **wall))
    bx.add_patch(Rectangle((H, 0), t, H, **wall))
    bx.add_patch(Rectangle((0, yc - D / 2), H, D, facecolor="#c3c2b7", edgecolor=ink, lw=1.0, hatch="////"))
    bx.plot([H / 2, H / 2], [0, yc - D / 2], ls="--", color=plots.MUTED, lw=0.8)
    bx.plot([H / 2, H / 2], [yc + D / 2, H], ls="--", color=plots.MUTED, lw=0.8)
    bx.add_patch(FancyArrowPatch((0, -0.07), (H, -0.07), arrowstyle="<|-|>", mutation_scale=8, color=ink, lw=0.9))
    bx.text(H / 2, -0.085, r"$H$ (along $z$)", ha="center", va="top", fontsize=10, color=ink)
    bx.text(H / 2, H + 0.075, "cross-section", fontsize=9, color=ink, ha="center", va="bottom")
    bx.text(H / 2 + 0.01, H - 0.03, r"$z = H/2$", fontsize=8, color=ink2, ha="left", va="top")
    bx.set_xlim(-0.06, H + 0.06)
    bx.set_ylim(-0.2, H + 0.12)
    bx.set_aspect("equal")
    return plots.to_svg(fig, "sketch")
