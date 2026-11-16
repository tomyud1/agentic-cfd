"""Problem sketch: the closed tube, the ignition layer, and the wave on its way (schematic, not a result)."""
from matplotlib.patches import FancyArrowPatch, Rectangle

from cfdlib import plots


def sketch():
    fig, ax = plots.sketch(8.6, 4.4)
    ink, ink2, wall = plots.INK, plots.INK2, "#d9d8d2"
    L, H, tw = 10.0, 1.0, 0.18
    y0 = 2.6
    xf = 5.2                                                     # where the wave is drawn
    ax.fill_between([0, xf], y0, y0 + H, color="#fde7dc", lw=0)                      # burnt gas
    ax.fill_between([xf, L], y0, y0 + H, color="#eef4fb", lw=0)                      # fresh gas
    ax.fill_between([0, 0.45], y0, y0 + H, color="#f6b89c", lw=0)                    # where the ignition layer was
    for x0 in (-tw, L):
        ax.add_patch(Rectangle((x0, y0 - tw), tw, H + 2 * tw, facecolor=wall, edgecolor="none", hatch="////"))
    ax.plot([0, L, L, 0, 0], [y0, y0, y0 + H, y0 + H, y0], color=ink, lw=1.5)
    ax.plot([xf, xf], [y0, y0 + H], color=plots.SERIES[3], lw=2.6)
    ax.add_patch(FancyArrowPatch((xf + 0.1, y0 + H / 2), (xf + 1.3, y0 + H / 2), arrowstyle="-|>", mutation_scale=14, color=plots.SERIES[3], lw=1.8))
    ax.text(xf + 1.45, y0 + H / 2, "wave", ha="left", va="center", fontsize=10.5, color=plots.SERIES[3])
    ax.text(2.8, y0 + H / 2, "burnt gas", ha="center", va="center", fontsize=10.5, color=ink)
    ax.text(8.9, y0 + H / 2 + 0.18, r"$\mathrm{H_2 + O_2 + 3.76\,N_2}$", ha="center", va="center", fontsize=9.5, color=ink)
    ax.text(8.9, y0 + H / 2 - 0.2, r"at rest, $p_0$, $T_0$", ha="center", va="center", fontsize=9.5, color=ink)
    ax.annotate("ignition layer at the start:\n" + r"$T_{\mathrm{ign}}$, $p_{\mathrm{ign}}$, thickness $L_{\mathrm{ign}}$", xy=(0.22, y0 + H), xytext=(0.9, y0 + H + 0.75),
                fontsize=9.5, color=ink2, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
    ax.text(-0.45, y0 + H / 2, "closed end", rotation=90, ha="center", va="center", fontsize=9, color=ink2)
    ax.text(L + 0.45, y0 + H / 2, "closed end", rotation=90, ha="center", va="center", fontsize=9, color=ink2)
    ax.annotate("", xy=(0, y0 - 0.45), xytext=(L, y0 - 0.45), arrowprops=dict(arrowstyle="<->", color=ink2, lw=0.9))
    ax.text(L / 2, y0 - 0.68, r"$L$", ha="center", va="center", fontsize=10.5, color=ink2)
    # the pressure along the tube, schematic
    yb, sc = 0.0, 1.35
    xs = [0.0, 0.52 * xf, xf - 0.28, xf - 0.07, xf, xf, L]
    ps = [0.36, 0.36, 0.86, 1.0, 1.0, 0.06, 0.06]
    ax.plot(xs, [yb + sc * p for p in ps], color=plots.SERIES[0], lw=1.8)
    ax.plot([0, L], [yb, yb], color=ink2, lw=0.8)
    ax.text(-0.45, yb + 0.7, "pressure", rotation=90, ha="center", va="center", fontsize=9, color=plots.SERIES[0])
    ax.annotate("jump at the front", xy=(xf, yb + sc * 0.6), xytext=(xf + 0.7, yb + sc * 0.75), fontsize=9, color=ink2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
    ax.annotate("the gas burns and\nthe pressure falls", xy=(xf - 0.2, yb + sc * 0.93), xytext=(xf - 2.3, yb + sc * 1.12), fontsize=9, color=ink2, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
    ax.text(0.26 * xf, yb + sc * 0.36 + 0.2, "gas at rest again", ha="center", va="center", fontsize=9, color=ink2)
    ax.text(0.84 * xf, yb + sc * 0.2, "expansion", ha="center", va="center", fontsize=9, color=ink2)
    ax.text(8.6, yb + 0.28, r"$p_0$", ha="center", va="center", fontsize=9.5, color=ink2)
    ax.set_xlim(-0.9, L + 0.9)
    ax.set_ylim(-0.3, y0 + H + 1.25)
    return plots.to_svg(fig, name="sketch")
