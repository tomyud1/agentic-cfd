"""Figures of the report, each drawn from the saved runs. Every function returns an inline SVG."""
import numpy as np
from matplotlib.colors import LogNorm

from cfdlib import plots
from . import measure

S = plots.SERIES
GREYS = ["#b9b8b2", "#898781", "#52514e", "#0b0b0b"]


def signal_edge(fl, t, n=200):
    """The exact edge of the region the corner has influenced behind the flat shock: the line from the
    corner that touches the signal's circle, then the circle up to the shock."""
    mu = fl.mach_angle
    xt, yt = t * (fl.u2 - fl.c2 ** 2 / fl.u2), fl.Hs + fl.c2 * t * np.cos(mu)
    a0 = np.pi / 2 + mu                                           # angle of the touching point seen from the centre
    a1 = np.arcsin(fl.v_flat / fl.c2)                             # angle of the point where the circle meets the shock
    a = np.linspace(a0, a1, n)
    return np.r_[0.0, xt, fl.u2 * t + fl.c2 * t * np.cos(a)], np.r_[fl.Hs, yt, fl.Hs + fl.c2 * t * np.sin(a)]


def flow_map(d, fl, N, labels):
    """Density at t_end and a shadow picture (size of the density gradient), with the waves named.
    labels: list of (text, xy point, xy text, panel index)."""
    fig, axs = plots.wide(2, h=4.9)
    rho = d["map_rho"].astype(float)
    h = fl.L / N
    ext = (0, fl.L, 0, fl.H)
    im = axs[0].imshow(rho, origin="lower", extent=ext, cmap="viridis", interpolation="nearest", rasterized=True)
    cb = fig.colorbar(im, ax=axs[0], shrink=0.8, pad=0.02)
    cb.set_label(r"density $\rho$ (kg/m$^3$)")
    gy, gx = np.gradient(rho, h)
    s = np.hypot(gx, gy) * h / fl.rho0                            # change of density per cell, in units of rho0
    im2 = axs[1].imshow(np.maximum(s, 1e-4), origin="lower", extent=ext, cmap="gray_r", norm=LogNorm(1e-3, 1.0),
                        interpolation="nearest", rasterized=True)
    cb2 = fig.colorbar(im2, ax=axs[1], shrink=0.8, pad=0.02)
    cb2.set_label(r"change of density per cell, $|\nabla\rho|\,h/\rho_0$")
    xe, ye = signal_edge(fl, fl.t_end)
    axs[1].plot(xe, ye, color=S[1], lw=1.2, ls="--", label="exact edge of the corner's signal")
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8.5)
    for ax, title in zip(axs, ("Density", "Shadow picture")):
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel(r"$x$ (m)")
        ax.set_title(title)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axs[0].set_ylabel(r"$y$ (m)")
    for text, xy, xyt, k in labels:
        axs[k].annotate(text, xy=xy, xytext=xyt, fontsize=8.5, color="black", ha="center", va="center",
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85),
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.8, shrinkA=0, shrinkB=0))
    return plots.to_svg(fig, name="map", allow=("overlaps",))


def corner_zoom(data, fl, names, box=(0.0, 0.45, 0.05, 0.55)):
    """The region below the corner on three grids: the rolled-up layer changes with the grid."""
    fig, axs = plots.wide(len(names), h=3.9)
    x0, x1, y0, y1 = box
    vmax = max(float(data[n]["map_rho"].max()) for n in names)
    for ax, n in zip(axs, names):
        rho = data[n]["map_rho"]
        N = rho.shape[0]
        i0, i1, j0, j1 = int(x0 * N / fl.L), int(x1 * N / fl.L), int(y0 * N / fl.H), int(y1 * N / fl.H)
        im = ax.imshow(rho[j0:j1, i0:i1], origin="lower", extent=(x0, x1, y0, y1), cmap="viridis", vmin=0, vmax=0.75 * vmax,
                       interpolation="nearest", rasterized=True)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(f"{N} cells per side")
        ax.set_xlabel(r"$x$ (m)")
    axs[0].set_ylabel(r"$y$ (m)")
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.015)
    cb.set_label(r"density (kg/m$^3$)")
    return plots.to_svg(fig, name="corner")


def front_similarity(d, fl, N):
    """The whole shock front at the ten written times, each scaled by its time."""
    fig, ax = plots.figure(6.4, 4.4)
    y = measure.centres(N, fl.H)
    n = len(d["times"])
    for k in range(n):
        t = d["times"][k]
        f = d["front_low"][k]
        top = (fl.H - fl.Hs) / fl.t_end                           # the panel ends at the top wall's height at t_end
        ok = np.isfinite(f) & ((y - fl.Hs) / t <= top)
        ax.plot(f[ok] / t, (y[ok] - fl.Hs) / t, color=(0.35 + 0.5 * (1 - k / n),) * 3 if k < n - 1 else S[0],
                lw=1.0 if k < n - 1 else 1.8, label=None if 0 < k < n - 1 else (r"$t = t_{\mathrm{end}}/10$" if k == 0 else r"$t = t_{\mathrm{end}}$"))
    ax.axvline(fl.W, color=S[1], lw=1.0, ls="--", label=r"exact flat shock, $x/t = W$")
    ax.axhline(fl.v_flat, color=S[2], lw=1.0, ls=":", label="exact end of the flat part")
    ax.set_xlabel(r"$x/t$ (m/s)")
    ax.set_ylabel(r"$(y - H_s)/t$ (m/s)")
    ax.set_xlim(0, 2100)
    ax.set_ylim(-1000, (fl.H - fl.Hs) / fl.t_end)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    return plots.to_svg(fig, name="front")


def flat_part(data, runs, fl, names):
    """Left: the top row of cells at t_end against the exact state behind the shock. Right: where the shock
    is in every row above the corner, as a distance from the exact flat shock in cells."""
    fig, axs = plots.wide(2, h=3.9)
    ax = axs[0]
    fine = names[-1]
    N = runs[fine]["N"]
    x = measure.centres(N, fl.L)
    for key, exact, col, lab in (("map_p", fl.p2, S[0], r"pressure $p/p_2$"), ("map_rho", fl.rho2, S[1], r"density $\rho/\rho_2$"),
                                 ("map_u", fl.u2, S[2], r"velocity $u/u_2$")):
        ax.plot(x, data[fine][key][-1].astype(float) / exact, color=col, lw=1.4, label=lab)
    for xm, lab in (((fl.u2 - fl.c2) * fl.t_end, r"$(u_2 - c_2)\,t$"), (fl.u2 * fl.t_end, r"$u_2 t$")):
        ax.axvline(xm, color=plots.MUTED, lw=0.8, ls=":")
        ax.text(xm, 1.046, lab, ha="center", va="bottom", fontsize=9, color=plots.INK2)
    ax.set_xlim(0, fl.x_top + 0.03)
    ax.set_ylim(0.955, 1.045)
    ax.set_xlabel(r"$x$ (m), top row of cells")
    ax.set_ylabel("value / exact value behind the shock")
    ax.set_title(f"Behind the flat shock, {N} cells per side", pad=16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5, ncol=3, columnspacing=1.0, handlelength=1.5)
    ax = axs[1]
    for n, col in zip(names, GREYS[-len(names):]):
        Nn = runs[n]["N"]
        y = measure.centres(Nn, fl.H)
        f = data[n]["front_low"][-1]
        dev = (f - f[-1]) / (fl.L / Nn)
        ok = np.isfinite(f) & (y > fl.Hs + 0.2) & (dev > -6)       # only what the panel shows (the layout check sees the whole line)
        ax.plot(dev[ok], y[ok], color=col, lw=1.4, label=f"{Nn}")
    ax.axhline(fl.flat_from(fl.t_end), color=S[2], lw=1.0, ls=":")
    ax.text(-5.85, fl.flat_from(fl.t_end) + 0.008, "exact end of the flat part", ha="left", va="bottom", fontsize=8.5, color=plots.INK2)
    ax.set_xlim(-6, 1)
    ax.set_ylim(fl.Hs + 0.2, fl.H)
    ax.set_xlabel("shock position minus its position in the top row (cells)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_title("Is the flat part straight?")
    ax.legend(title="cells per side", loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5, title_fontsize=8.5, ncol=4,
              columnspacing=1.0, handlelength=1.5)
    return plots.to_svg(fig, name="flat")


def two_times(d, fl, N, n_snap):
    """The density along the step face and along the line y = H_s at t_end/2 and t_end, against distance / time."""
    fig, axs = plots.wide(2, h=3.7)
    i_half, i_end = n_snap // 2 - 1, n_snap - 1
    h = fl.L / N
    dist = (np.arange(N // 2) + 0.5) * h                          # down from the corner
    x = measure.centres(N, fl.L)
    for ax, get, s, xlabel, title in (
            (axs[0], lambda k: d["wall_rho"][k][::-1], dist, r"distance below the corner / time (m/s)", "Along the step face"),
            (axs[1], lambda k: d["line_rho"][k], x, r"$x$ / time (m/s)", r"Along the line $y = H_s$")):
        ax.plot(s / d["times"][i_end], get(i_end), color=S[0], lw=1.8, label=r"$t = t_{\mathrm{end}}$")
        ax.plot(s / d["times"][i_half], get(i_half), color=S[1], lw=1.2, ls="--", label=r"$t = t_{\mathrm{end}}/2$")
        ax.set_xlim(0, s[-1] / d["times"][i_end])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axs[0].set_ylabel(r"density (kg/m$^3$)")
    axs[0].legend(loc="upper left", fontsize=9)
    return plots.to_svg(fig, name="similar")


def convergence(runs, pairs, fam, fl):
    """Left: how much the answer changes from one grid level to the next. Right: the bent shock on the step face."""
    fig, axs = plots.wide(2, h=3.7)
    ax = axs[0]
    Nf = np.array([runs[p["fine"]]["N"] for p in pairs])
    ax.loglog(Nf, [100 * p["field_change"] for p in pairs], "o-", color=S[0], label="density field, whole chamber")
    ax.loglog(Nf, [100 * runs[p["fine"]]["sim_wall"] for p in pairs], "s--", color=S[1], lw=1.4, label="density along the step face, two times")
    ax.loglog(Nf, [100 * runs[p["fine"]]["sim_line"] for p in pairs], "^--", color=S[2], lw=1.4, label=r"density along $y = H_s$, two times")
    ref = 100 * pairs[0]["field_change"] * 1.6
    ax.loglog(Nf, ref * Nf[0] / Nf, color=plots.MUTED, lw=1.0, ls=":", label="slope of first order")
    ax.set_xticks(Nf)
    ax.set_xticklabels([str(n) for n in Nf])
    ax.minorticks_off()
    ax.set_xlabel("cells per side of the finer grid")
    ax.set_ylabel("mean change from the coarser grid (%)")
    ax.set_title("Change from one grid to the next")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=8.5, ncol=1)
    ax = axs[1]
    N = np.array([runs[n]["N"] for n in fam])
    ax.plot(1000 * fl.L / N, [runs[n]["d_wall"] for n in fam], "o-", color=S[0])
    ax.set_xscale("log")
    ax.set_xticks(1000 * fl.L / N)
    ax.set_xticklabels([f"{1000 * fl.L / n:g}\n({n})" for n in N])
    ax.minorticks_off()
    ax.set_xlabel("cell size (mm), with the cells per side in brackets")
    ax.set_ylabel("distance run down the step face (m)")
    ax.set_title("Bent shock on the step face")
    return plots.to_svg(fig, name="convergence")


def conservation(data, runs, fam, fl):
    """Mass and energy in the chamber minus (start + exact inflow), as a fraction of the inflow so far, every step."""
    fig, axs = plots.wide(2, h=3.5)
    for ax, key, rate, start, title in ((axs[0], "c_mass", fl.mdot, fl.mass0, "Mass"), (axs[1], "c_energy", fl.edot, fl.energy0, "Energy")):
        for n, col in zip(fam, GREYS[-len(fam):]):
            t = data[n]["c_t"]
            err = np.abs(data[n][key] - start - rate * t) / (rate * t)
            ax.semilogy(1e3 * t, np.maximum(err, 1e-17), color=col, lw=1.0, label=f"{runs[n]['N']}")
        ax.axhline(1e-4, color=S[1], lw=1.2, ls="--")
        ax.text(0.44, 1.6e-4, "required: 0.01 %", ha="right", va="bottom", fontsize=9, color=S[1])
        ax.set_ylim(1e-17, 1e-2)
        ax.set_xlabel("time (ms)")
        ax.set_title(title)
    axs[0].set_ylabel("|difference| / inflow so far")
    axs[1].legend(title="cells per side", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, title_fontsize=8.5)
    return plots.to_svg(fig, name="conservation")
