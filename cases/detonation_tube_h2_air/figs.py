"""Figures of the report (matplotlib through cfdlib.plots). Each takes the dict of analyze.compute()."""
import numpy as np

from cfdlib import plots

C = plots.SERIES
MM, US, MPA = 1e3, 1e6, 1e-6
DASH = (0, (4, 2))


def _grid_label(c, n):
    return f"{c['runs'][n]['N']} cells"


def _legend(fig, ax, ncol=4):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=ncol, fontsize=8.5)


def xt(c):
    """Where the fronts are against time (finest grid), and the speed of the pressure front on every grid."""
    g, th, fam = c["g"], c["th"], c["fam"]
    d = c["data"][fam[-1]]
    D = th["cj"]["D"]
    fig, (a, b) = plots.wide(2, h=3.9)
    a.axvspan(US * g["t_end"] / 2, US * g["t_end"], color="#f1f0ea", lw=0)
    a.plot(US * d["t"], MM * d["xp"], color=C[0], label="pressure front")
    a.plot(US * d["t"][::4], MM * d["xb"][::4], "o", ms=3.0, mfc="none", mew=0.8, color=C[1], label="burning front")
    tt = np.array([0.0, g["t_end"]])
    a.plot(US * tt, MM * (d["xp"][-1] - D * (g["t_end"] - tt)), color=plots.INK, lw=0.9, ls=DASH, label="slope of the CJ speed")
    a.set_xlabel(r"time ($\mathrm{\mu s}$)")
    a.set_ylabel("distance from the closed end (mm)")
    a.set_title(f"The two fronts, {_grid_label(c, fam[-1])}")
    a.set_xlim(0, US * g["t_end"])
    a.set_ylim(0, MM * g["L"])
    a.legend(loc="upper left", fontsize=8.5, frameon=False)
    b.axhspan(-2, 2, color="#f1f0ea", lw=0)
    b.axhline(0, color=plots.INK, lw=0.9, ls=DASH)
    for k, n in enumerate(fam):
        e = c["data"][n]
        w = 4                                                    # speed over 2 microseconds (4 samples)
        v = (e["xp"][w:] - e["xp"][:-w]) / (e["t"][w:] - e["t"][:-w])
        b.plot(US * 0.5 * (e["t"][w:] + e["t"][:-w]), 100 * (v / D - 1), color=C[k], label=_grid_label(c, n), lw=1.3)
    b.set_xlabel(r"time ($\mathrm{\mu s}$)")
    b.set_ylabel("front speed minus CJ speed (%)")
    b.set_title("Speed of the pressure front")
    b.set_xlim(0, US * g["t_end"])
    b.set_ylim(-8, 4)
    _legend(fig, b)
    return plots.to_svg(fig, name="xt")


def profiles(c):
    """Pressure and temperature along the tube at five times on the finest grid, with the ideal profile at the end."""
    g, th, tha, fam = c["g"], c["th"], c["tha"], c["fam"]
    d = c["data"][fam[-1]]
    fig, (a, b) = plots.wide(2, h=3.9)
    ks = [k for k, t in enumerate(d["prof_t"]) if round(US * t) in (10, 20, 30, 40, 50)]
    greys = ["#b9b8b2", "#96958f", "#74736e", "#4d4c48", C[0]]
    for col, k in zip(greys, ks):
        lab = rf"${US * d['prof_t'][k]:.0f}\ \mathrm{{\mu s}}$"
        a.plot(MM * d["x"], MPA * d["prof_p"][k], color=col, lw=1.2, label=lab)
        b.plot(MM * d["x"], d["prof_T"][k], color=col, lw=1.2, label=lab)
    xi = np.linspace(0, th["cj"]["D"], 400)
    order = np.argsort(tha["tw_xi"])
    xs = xi * g["t_end"] * d["xp"][-1] / (th["cj"]["D"] * g["t_end"])            # the ideal wave drawn with its front at the simulated front
    a.plot(MM * xs, MPA * np.interp(xi, tha["tw_xi"][order], tha["tw_p"][order]), color=C[1], lw=1.1, ls=DASH, label="ideal wave (Theory)")
    b.plot(MM * xs, np.interp(xi, tha["tw_xi"][order], tha["tw_T"][order]), color=C[1], lw=1.1, ls=DASH, label="ideal wave (Theory)")
    a.set_ylabel("pressure (MPa)")
    b.set_ylabel("temperature (K)")
    for ax in (a, b):
        ax.set_xlabel("distance from the closed end (mm)")
        ax.set_xlim(0, MM * g["L"])
    a.set_ylim(0, None)
    b.set_ylim(0, 3600)
    a.set_title("Pressure")
    b.set_title("Temperature")
    _legend(fig, a, ncol=6)
    return plots.to_svg(fig, name="profiles")


def structure(c):
    """The state behind the front at the end time on every grid, against the steady structure of Theory."""
    th, tha, fam = c["th"], c["tha"], c["fam"]
    cj = th["cj"]
    fig, (a, b) = plots.wide(2, h=3.9)
    for k, n in enumerate(fam):
        d, r = c["data"][n], c["runs"][n]
        s = MM * (d["xp"][-1] - d["x"])
        m = (s > -0.6) & (s < 6.5)
        a.plot(s[m], d["prof_p"][-1][m] / cj["p"], color=C[k], lw=1.3, label=_grid_label(c, n))
        b.plot(s[m], d["prof_T"][-1][m] / cj["T"], color=C[k], lw=1.3, label=_grid_label(c, n))
        if r["burnt"]["found"]:
            a.plot([MM * r["burnt"]["d"]], [r["burnt"]["p"] / cj["p"]], "o", ms=5, mfc="white", mew=1.2, color=C[k])
            b.plot([MM * r["burnt"]["d"]], [r["burnt"]["T"] / cj["T"]], "o", ms=5, mfc="white", mew=1.2, color=C[k])
    zx = MM * tha["znd_x"]
    mz = zx < 6.5
    a.plot(np.concatenate([[0.0, 0.0], zx[mz]]), np.concatenate([[c["g"]["p_0"] / cj["p"], tha["znd_p"][0] / cj["p"]], tha["znd_p"][mz] / cj["p"]]), color=plots.INK, lw=1.0, ls=DASH,
           label="steady wave (Theory)")
    b.plot(np.concatenate([[0.0, 0.0], zx[mz]]), np.concatenate([[c["g"]["T_0"] / cj["T"], tha["znd_T"][0] / cj["T"]], tha["znd_T"][mz] / cj["T"]]), color=plots.INK, lw=1.0, ls=DASH,
           label="steady wave (Theory)")
    for ax in (a, b):
        ax.axhline(1.0, color=plots.MUTED, lw=0.8, ls=":")
        ax.set_xlabel("distance behind the pressure front (mm)")
        ax.set_xlim(-0.6, 6.5)
    a.set_ylabel("pressure / CJ pressure")
    b.set_ylabel("temperature / CJ temperature")
    a.set_title("Pressure behind the front")
    b.set_title("Temperature behind the front")
    a.set_ylim(0, 2.0)
    b.set_ylim(0, 1.3)
    _legend(fig, a, ncol=5)
    return plots.to_svg(fig, name="structure")


def attached(c):
    """Distance between the two fronts, and the highest pressure, against time on every grid."""
    g, th, fam, ext = c["g"], c["th"], c["fam"], c["numerics"]["extraction"]
    fig, (a, b) = plots.wide(2, h=3.9)
    for k, n in enumerate(fam):
        d = c["data"][n]
        a.semilogy(US * d["t"][1:], MM * np.maximum(d["gap"][1:], 1e-4), color=C[k], lw=1.3, label=_grid_label(c, n))
        w = max(1, len(d["ts"]) // 400)
        pm = np.convolve(d["pmax"], np.ones(w) / w, mode="valid")
        b.plot(US * d["ts"][w - 1:], pm / th["front"]["p"], color=C[k], lw=1.1, label=_grid_label(c, n))
    a.axhline(1.0, color=plots.INK, lw=0.9, ls=DASH)
    a.axvline(US * ext["attached_from"], color=plots.MUTED, lw=0.8, ls=":")
    a.set_xlabel(r"time ($\mathrm{\mu s}$)")
    a.set_ylabel("pressure front minus burning front (mm)")
    a.set_title("Distance between the fronts")
    a.set_xlim(0, US * g["t_end"])
    a.set_ylim(0.01, 3)
    b.axhline(1.0, color=plots.INK, lw=0.9, ls=DASH)
    b.axhline(th["cj"]["p"] / th["front"]["p"], color=plots.MUTED, lw=0.8, ls=":")
    b.set_xlabel(r"time ($\mathrm{\mu s}$)")
    b.set_ylabel("highest pressure / shock pressure of a CJ wave")
    b.set_title("Highest pressure in the tube")
    b.set_xlim(0, US * g["t_end"])
    b.set_ylim(0.4, 1.6)
    _legend(fig, a)
    return plots.to_svg(fig, name="attached")


def conservation(c):
    """Total energy against time: the family, and the passes per time step."""
    g, fam = c["g"], c["fam"]
    fig, (a, b) = plots.wide(2, h=3.9)

    def curve(ax, n, **kw):
        d = c["data"][n]
        m = d["ts"] > 2e-6
        ax.plot(US * d["ts"][m], 100 * d["dE"][m] / d["q"][m], **kw)
    for k, n in enumerate(fam):
        curve(a, n, color=C[k], lw=1.3, label=_grid_label(c, n))
    a.axhline(0, color=plots.MUTED, lw=0.8, ls=":")
    a.set_xlabel(r"time ($\mathrm{\mu s}$)")
    a.set_ylabel("change of total energy (% of the heat released)")
    a.set_title("Five passes per step, four grids")
    a.set_xlim(0, US * g["t_end"])
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5, frameon=False, ncol=4)
    styles = [dict(color=plots.MUTED, ls=DASH), dict(color=C[1]), dict(color=C[0]), dict(color=plots.INK, ls=(0, (1, 1.5)))]
    for sty, n in zip(styles, c["passes"]):
        curve(b, n, lw=1.3, label=f"{c['runs'][n]['passes']} pass" + ("es" if c["runs"][n]["passes"] > 1 else ""), **sty)
    b.axhline(0, color=plots.MUTED, lw=0.8, ls=":")
    b.set_xlabel(r"time ($\mathrm{\mu s}$)")
    b.set_ylabel("change of total energy (% of the heat released)")
    b.set_title(f"Passes per time step, {c['runs'][c['passes'][0]]['N']} cells")
    b.set_xlim(0, US * g["t_end"])
    b.set_ylim(-2.4, 0.4)                                        # one pass per step leaves the frame (the caption gives its value)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5, frameon=False, ncol=4)
    return plots.to_svg(fig, name="conservation")


def convergence(c):
    """The wave speed against the cell size, and what each setting does to it on one grid."""
    fam, runs = c["fam"], c["runs"]
    fig, (a, b) = plots.wide(2, h=3.9)
    dx = np.array([1e6 * runs[n]["dx"] for n in fam])
    err = np.array([100 * runs[n]["speed_err"] for n in fam])
    a.axhspan(-2, 2, color="#f1f0ea", lw=0)
    a.axhline(0, color=plots.INK, lw=0.9, ls=DASH)
    a.semilogx(dx, err, "o-", color=C[0], ms=5)
    a.set_xlabel(r"cell size ($\mathrm{\mu m}$)")
    a.set_ylabel("wave speed minus CJ speed (%)")
    a.set_title("Cell size")
    a.set_xticks(dx)
    a.set_xticklabels([f"{x:g}" for x in dx])
    a.minorticks_off()
    lim = max(2.6, 1.25 * float(np.max(np.abs(err))))
    a.set_ylim(-lim, lim)
    labels, vals = zip(*c["study_points"])
    xs = np.arange(len(vals))
    b.axhspan(-2, 2, color="#f1f0ea", lw=0)
    b.axhline(0, color=plots.INK, lw=0.9, ls=DASH)
    b.plot(xs, 100 * np.array(vals), "o", color=C[0], ms=5)
    b.set_xticks(xs)
    b.set_xticklabels([{"looser tolerance": "tolerance"}.get(s, s) for s in labels], rotation=55, ha="right", rotation_mode="anchor", fontsize=8.5)
    b.set_ylabel("wave speed minus CJ speed (%)")
    b.set_title(f"One setting changed at a time, {runs[c['study_grid']]['N']} cells")
    limb = max(2.6, 1.15 * float(np.max(np.abs(100 * np.array(vals)))))
    b.set_ylim(-limb, limb if limb > 2.7 else 2.6)
    return plots.to_svg(fig, name="convergence")
