"""Problem sketch: the tube before the wall is removed, and the three waves some time after."""
from matplotlib.patches import FancyArrowPatch, Rectangle

from cfdlib import plots


def sketch(w, L):
    """w: exact wave positions at the time of interest (head, tail, contact, shock); L: tube length."""
    fig, ax = plots.figure(9.0, 4.6)
    ax.set_axis_off()
    ax.grid(False)
    ink, ink2 = plots.INK, plots.INK2
    W, H = 14.0, 1.1                              # drawing units: tube length and height
    X = {k: v / L * W for k, v in w.items()}
    dark, mid1, mid2, light = "#9dbfe6", "#c9dcf2", "#dbe7f6", "#eef4fb"

    # ---- the tube at t = 0 ----
    y0 = 5.2
    ax.add_patch(Rectangle((0, y0), W / 2, H, facecolor=dark, edgecolor=ink2, lw=1))
    ax.add_patch(Rectangle((W / 2, y0), W / 2, H, facecolor=light, edgecolor=ink2, lw=1))
    ax.plot([W / 2, W / 2], [y0 - 0.25, y0 + H + 0.25], color=plots.SERIES[1], lw=3)
    ax.text(W / 4, y0 + H + 0.25, r"left: high pressure $p_L$, density $\rho_L$, at rest", ha="center", va="bottom",
            fontsize=10, color=ink)
    ax.text(3 * W / 4, y0 + H + 0.25, r"right: low pressure $p_R$, density $\rho_R$, at rest", ha="center", va="bottom",
            fontsize=10, color=ink)
    ax.text(W / 2, y0 - 0.4, r"thin wall at $x = L/2$, removed at $t = 0$", ha="center", va="top", fontsize=10,
            color=plots.SERIES[1])
    ax.text(-0.3, y0 + H / 2, r"$t = 0$", ha="right", va="center", fontsize=11, color=ink)

    # ---- the tube at the time of interest ----
    y1 = 1.9
    ax.add_patch(Rectangle((0, y1), X["head"], H, facecolor=dark, edgecolor="none"))
    n = 8                                         # the fan: pressure falls smoothly from left to right
    for i in range(n):
        f = (i + 0.5) / n
        col = tuple((1 - f) * int(dark[j:j + 2], 16) / 255 + f * int(mid1[j:j + 2], 16) / 255 for j in (1, 3, 5))
        ax.add_patch(Rectangle((X["head"] + i * (X["tail"] - X["head"]) / n, y1), (X["tail"] - X["head"]) / n, H,
                               facecolor=col, edgecolor="none"))
    ax.add_patch(Rectangle((X["tail"], y1), X["contact"] - X["tail"], H, facecolor=mid1, edgecolor="none"))
    ax.add_patch(Rectangle((X["contact"], y1), X["shock"] - X["contact"], H, facecolor=mid2, edgecolor="none"))
    ax.add_patch(Rectangle((X["shock"], y1), W - X["shock"], H, facecolor=light, edgecolor="none"))
    ax.add_patch(Rectangle((0, y1), W, H, facecolor="none", edgecolor=ink2, lw=1))
    ax.plot([X["contact"]] * 2, [y1, y1 + H + 0.65], color=ink, lw=1.2, ls="--")
    ax.plot([X["shock"]] * 2, [y1, y1 + H + 0.3], color=ink, lw=2)
    ax.plot([X["head"]] * 2, [y1, y1 + H], color=ink2, lw=0.8, ls=":")
    ax.plot([X["tail"]] * 2, [y1, y1 + H], color=ink2, lw=0.8, ls=":")
    ya = y1 + H / 2                                # arrows inside the tube: which way each wave moves
    for xa, dx in ((X["head"], -0.9), (X["contact"], 0.9), (X["shock"], 0.9)):
        ax.add_patch(FancyArrowPatch((xa, ya), (xa + dx, ya), arrowstyle="-|>", mutation_scale=12, color=ink, lw=1.4))
    ax.text(0.5 * (X["head"] + X["tail"]), y1 + H + 0.25, "expansion fan", ha="center", va="bottom", fontsize=10, color=ink)
    ax.text(X["contact"], y1 + H + 0.75, "border of the two gases", ha="center", va="bottom", fontsize=10, color=ink)
    ax.text(X["shock"] + 0.15, y1 + H + 0.12, "shock", ha="left", va="bottom", fontsize=10, color=ink)
    ax.text(0.5 * X["head"], y1 - 0.2, r"still $p_L$, $\rho_L$, at rest", ha="center", va="top", fontsize=9.5, color=ink2)
    ax.text(0.5 * (X["tail"] + X["shock"]), y1 - 0.2, r"one pressure $p^*$, one speed $u^*$", ha="center", va="top",
            fontsize=9.5, color=ink2)
    ax.text(0.5 * (X["shock"] + W), y1 - 0.2, r"still $p_R$, $\rho_R$, at rest", ha="center", va="top", fontsize=9.5,
            color=ink2)
    ax.text(-0.3, y1 + H / 2, r"$t = t_{\mathrm{end}}$", ha="right", va="center", fontsize=11, color=ink)

    # ---- the x axis ----
    ya = 0.75
    ax.add_patch(FancyArrowPatch((0, ya), (W + 0.6, ya), arrowstyle="-|>", mutation_scale=10, color=ink2, lw=1))
    for xt, lab in ((0, r"$x = 0$"), (W / 2, r"$L/2$"), (W, r"$L$")):
        ax.plot([xt, xt], [ya - 0.1, ya + 0.1], color=ink2, lw=1)
        ax.text(xt, ya - 0.2, lab, ha="center", va="top", fontsize=10, color=ink2)
    ax.set_xlim(-1.9, W + 0.9)
    ax.set_ylim(0.0, y0 + H + 0.9)
    return plots.to_svg(fig, name="sketch")
