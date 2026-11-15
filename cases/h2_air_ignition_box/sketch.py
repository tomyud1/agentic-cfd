"""Problem sketch: a parcel of gas held at constant pressure, with no heat exchange."""
from matplotlib.patches import FancyArrowPatch, Rectangle

from cfdlib import plots


def sketch():
    """A cylinder with a free piston stands for 'the pressure is held constant, the parcel is free to expand'.
    It is a picture of the constraints, not of a real device: the problem has no geometry."""
    fig, ax = plots.sketch(7.6, 4.6)
    ink, ink2, wall = plots.INK, plots.INK2, "#d9d8d2"
    w, h, tw = 3.0, 2.0, 0.22
    for x0, burnt in ((0.0, False), (4.6, True)):
        hh = h if not burnt else 1.55 * h                       # the hot gas takes more room (schematic)
        ax.fill_between([x0, x0 + w], 0, hh, color="#eef4fb" if not burnt else "#fde7dc", lw=0)   # the gas (not a shape that text must avoid)
        ax.add_patch(Rectangle((x0 - tw, -tw), tw, 3.6 + tw, facecolor=wall, edgecolor="none", hatch="////"))
        ax.add_patch(Rectangle((x0 + w, -tw), tw, 3.6 + tw, facecolor=wall, edgecolor="none", hatch="////"))
        ax.add_patch(Rectangle((x0 - tw, -tw), w + 2 * tw, tw, facecolor=wall, edgecolor="none", hatch="////"))
        ax.plot([x0, x0, x0 + w, x0 + w], [3.6, 0, 0, 3.6], color=ink, lw=1.6)
        ax.add_patch(Rectangle((x0, hh), w, 0.16, facecolor=ink2, edgecolor=ink, lw=1.0))
        ax.add_patch(FancyArrowPatch((x0 + w / 2, hh + 0.95), (x0 + w / 2, hh + 0.2), arrowstyle="-|>", mutation_scale=13, color=plots.SERIES[0], lw=1.8))
        ax.text(x0 + w / 2 + 0.15, hh + 0.62, r"$p$ = const", ha="left", va="center", fontsize=10, color=plots.SERIES[0])
    ax.text(w / 2, 1.25, r"$\mathrm{H_2 + O_2 + 3.76\,N_2}$", ha="center", va="center", fontsize=11, color=ink)
    ax.text(w / 2, 0.75, r"at rest, $T_0$", ha="center", va="center", fontsize=10.5, color=ink)
    ax.text(4.6 + w / 2, 1.9, "burnt gas", ha="center", va="center", fontsize=11, color=ink)
    ax.text(4.6 + w / 2, 1.4, r"$\mathrm{H_2O,\ O_2,\ N_2}$ + a little", ha="center", va="center", fontsize=10, color=ink)
    ax.text(4.6 + w / 2, 1.0, r"$\mathrm{OH,\ O,\ H,\ H_2}$", ha="center", va="center", fontsize=10, color=ink)
    ax.text(w / 2, -0.55, r"start, $t = 0$", ha="center", va="center", fontsize=10.5, color=ink2)
    ax.text(4.6 + w / 2, -0.55, r"after ignition", ha="center", va="center", fontsize=10.5, color=ink2)
    ax.add_patch(FancyArrowPatch((w + 0.45, 1.0), (4.6 - 0.45, 1.0), arrowstyle="-|>", mutation_scale=15, color=ink, lw=1.6))
    ax.text((w + 4.6) / 2, 1.3, "time", ha="center", va="center", fontsize=10, color=ink)
    ax.annotate("walls let no heat through", xy=(4.6 + w + tw / 2, 0.45), xytext=(4.6 + w + 0.6, 0.45), fontsize=9.5, color=ink2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
    ax.annotate("free piston:\nthe gas expands\nas it heats", xy=(4.6 + w - 0.1, 1.55 * h + 0.08), xytext=(4.6 + w + 0.6, 2.75), fontsize=9.5, color=ink2, ha="left",
                va="center", arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
    ax.set_xlim(-0.5, 10.6)
    ax.set_ylim(-0.9, 4.3)
    return plots.to_svg(fig, name="sketch")
