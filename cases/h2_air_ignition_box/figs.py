"""Figures of the report (matplotlib through cfdlib.plots)."""
import numpy as np
from matplotlib.ticker import NullFormatter

from cfdlib import plots

C = plots.SERIES
MS = 1e3          # seconds -> milliseconds
SPECIES_TEX = {"H2": r"$\mathrm{H_2}$", "O2": r"$\mathrm{O_2}$", "H2O": r"$\mathrm{H_2O}$", "OH": r"$\mathrm{OH}$", "H": r"$\mathrm{H}$",
               "O": r"$\mathrm{O}$", "HO2": r"$\mathrm{HO_2}$", "H2O2": r"$\mathrm{H_2O_2}$", "N2": r"$\mathrm{N_2}$"}


def history(c):
    """Temperature against time: OpenFOAM (finest step), the reference's printed points, the independent solution."""
    d, ref, hist, g, th = c["data"][c["fam"][-1]], c["ref"], c["hist"], c["g"], c["th"]
    rt, rT = np.array(ref["history"]["t_s"]), np.array(ref["history"]["T_K"])
    fig, (a, b) = plots.wide(2, h=3.9)
    a.plot(MS * d["t"], d["T"], color=C[0], label="OpenFOAM (chemFoam), smallest step")
    a.plot(MS * rt[::12], rT[::12], "o", ms=3.2, mfc="none", color=plots.INK, mew=0.8, label="reference (CHEMKIN II), printed points")
    a.plot(MS * hist["t"], hist["T_std"], color=C[1], lw=1.0, ls=(0, (4, 2)), label="independent solution (Theory)")
    a.axhline(th["equilibrium"]["std"]["T"], color=plots.MUTED, lw=0.8, ls=":")
    a.set_xlabel("time (ms)")
    a.set_ylabel("temperature (K)")
    a.set_title("The whole millisecond")
    a.set_xlim(0, MS * g["t_end"])
    a.set_ylim(900, 2400)
    t0 = c["th"]["independent"]["t_ign"]
    lo, hi = t0 - 4.0e-7, t0 + 4.0e-7
    m = (d["t"] > lo) & (d["t"] < hi)
    b.plot(1e6 * (d["t"][m] - t0), d["T"][m], color=C[0])
    mr = (rt > lo) & (rt < hi)
    b.plot(1e6 * (rt[mr] - t0), rT[mr], "o", ms=3.6, mfc="none", color=plots.INK, mew=0.8)
    mh = (hist["t"] > lo) & (hist["t"] < hi)
    b.plot(1e6 * (hist["t"][mh] - t0), hist["T_std"][mh], color=C[1], lw=1.0, ls=(0, (4, 2)))
    b.axhline(g["T_ign"], color=plots.MUTED, lw=0.8, ls=":")
    b.set_xlabel(r"time minus the independent solution's ignition time ($\mathrm{\mu s}$)")
    b.set_ylabel("temperature (K)")
    b.set_title("Around the ignition temperature")
    b.set_xlim(-0.4, 0.4)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=8.5)
    return plots.to_svg(fig, name="history")


def species(c):
    """Moles of the six requested species per kilogram of gas against time (finest step)."""
    d, th, g = c["data"][c["fam"][-1]], c["th"], c["g"]
    sp = th["species"]
    W = np.array([th["W"][s] for s in sp])
    sig = d["Y"] / W
    t0 = th["independent_of"]["t_ign"]
    fig, (a, b) = plots.wide(2, h=3.9)
    for k, s in enumerate(("H2", "O2", "H2O")):
        a.plot(MS * d["t"], sig[:, sp.index(s)], color=C[k], label=SPECIES_TEX[s])
    a.axvline(MS * t0, color=plots.MUTED, lw=0.8, ls=":")
    a.set_xlabel("time (ms)")
    a.set_ylabel("moles per kg of gas")
    a.set_title("Fuel, oxygen and water")
    a.set_xlim(0, MS * g["t_end"])
    styles = {"OH": dict(color=C[3]), "H": dict(color=plots.INK, lw=1.3), "O": dict(color=plots.MUTED, ls=(0, (5, 2)))}   # not the left panel's colours
    for s in ("OH", "H", "O"):
        y = sig[:, sp.index(s)]
        b.semilogy(MS * d["t"][1:], y[1:], label=SPECIES_TEX[s], **styles[s])
    b.axvline(MS * t0, color=plots.MUTED, lw=0.8, ls=":")
    b.set_xlabel("time (ms)")
    b.set_ylabel("moles per kg of gas")
    b.set_title("Short-lived species (logarithmic scale)")
    b.set_xlim(0, MS * g["t_end"])
    b.set_ylim(1e-10, 2)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=6, fontsize=9.5)
    return plots.to_svg(fig, name="species")


def conservation(c):
    """Atoms and enthalpy of the parcel at every stored step of the family's runs."""
    acc, th = c["numerics"]["acceptance"], c["th"]
    fig, (a, b) = plots.wide(2, h=3.9)
    d = c["data"][c["fam"][-1]]
    els = [e for e in th["elements"] if th["start"]["atoms"][e] > 0]
    for k, e in enumerate(els):
        y = np.abs(d["atoms_dev"][:, k])
        a.semilogy(MS * d["t"][1:], np.maximum(y[1:], 1e-17), color=C[k], lw=1.0, label={"H": "hydrogen atoms", "O": "oxygen atoms", "N": "nitrogen atoms"}[e])
    a.axhline(acc["req_conserved"], color=plots.INK, lw=1.0, ls=(0, (4, 2)), label="required limit")
    a.set_ylim(1e-17, 1e-3)
    a.set_xlabel("time (ms)")
    a.set_ylabel("|change| / start value")
    a.set_title("Atoms of each element (smallest step)")
    for k, n in enumerate((c["fam"][0], c["fam"][-1])):
        dd = c["data"][n]
        b.plot(MS * dd["t"], 1e8 * dd["energy"], color=C[k], lw=1.2, label=f"{c['runs'][n]['steps']} steps")
    b.set_xlabel("time (ms)")
    b.set_ylabel(r"enthalpy change / heat released  ($\times 10^{-8}$)")
    b.set_title("Enthalpy of the parcel")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=6, fontsize=8.5)
    return plots.to_svg(fig, name="conservation")


def convergence(c):
    """Error against the time step, for the ignition time and the final temperature."""
    st, dts = c["study"], np.array(c["dts"])
    ind_of = c["th"]["independent_of"]
    fig, (a, b) = plots.wide(2, h=3.6)
    e1 = np.array(st["t_ign"]["err"]) / ind_of["t_ign"]
    a.loglog(1e6 * dts, 100 * e1, "o-", color=C[0], label="OpenFOAM against the independent solution")
    a.loglog(1e6 * dts, 100 * e1[-1] * dts / dts[-1], color=plots.MUTED, lw=1.0, ls=(0, (4, 2)), label="first order: error in proportion to the step")
    a.set_xlabel(r"time step ($\mathrm{\mu s}$)")
    a.set_ylabel("ignition time: difference (%)")
    a.set_title("Ignition time")
    e2 = np.array(st["T_end"]["err"])
    b.loglog(1e6 * dts, 1e3 * e2, "o-", color=C[0])
    b.loglog(1e6 * dts, 1e3 * e2[-1] * dts / dts[-1], color=plots.MUTED, lw=1.0, ls=(0, (4, 2)))
    b.set_xlabel(r"time step ($\mathrm{\mu s}$)")
    b.set_ylabel("final temperature: difference (mK)")
    b.set_title("Temperature at the end")
    for ax in (a, b):                                   # one tick per level; the default log ticks crowd each other
        ax.set_xticks(1e6 * dts)
        ax.set_xticklabels([f"{1e6 * x:g}" for x in dts], fontsize=8.5)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=8.5)
    return plots.to_svg(fig, name="convergence")


def cost(c):
    """Where the time goes within a run, and accuracy against solver time for every run."""
    runs, data = c["runs"], c["data"]
    fig, (a, b) = plots.wide(2, h=3.9)
    main = f"S{c['numerics']['runs']['timing']['steps']}"
    loose = next((n for n in runs if n.startswith(main + "_rel0.1_abs1e-12")), None)
    rkf = main + "_RKF45" if main + "_RKF45" in runs else None
    for k, (n, lab) in enumerate(((main, "tight tolerances"), (loose, "the tutorial's tolerances"), (rkf, "tight tolerances, method RKF45"))):
        if n and "cost_us" in data[n]:
            per = runs[n]["steps"] / len(data[n]["cost_us"])                 # steps in each stretch of the stored cost profile
            used = np.cumsum(data[n]["cost_us"] * per) * 1e-6             # the solver's clock has 0.01 s steps: summed, not per step
            a.plot(MS * data[n]["cost_t"], used, color=C[k], lw=1.4, label=f"time used so far: {lab}")
    a.set_xlabel("time in the history (ms)")
    a.set_ylabel("solver time used so far (s)")
    a.set_title(f"Along a run of {runs[main]['steps']} steps")
    a.set_xlim(0, MS * c["g"]["t_end"])
    groups = [("time step", "o", C[0], "fixed step, tight tolerances"), ("tolerance", "s", C[1], "fixed step, other tolerances"),
              ("adjustable", "^", C[2], "step set by the chemistry solver"), ("method", "D", C[3], "other chemistry solvers")]
    seen = set()
    for study, mk, col, lab in groups:
        xs, ys = [], []
        for n, r in runs.items():
            if study in r["studies"] and n not in seen and r.get("d_of_t") is not None and r["table"]:
                seen.add(n)
                xs.append(r["wall_time"])
                ys.append(max(100 * abs(r["d_of_t"]), 1e-6))
        b.loglog(xs, ys, mk, color=col, ms=6, mfc="none", mew=1.4, label=lab)
    b.axhline(100 * c["numerics"]["acceptance"]["req_ignition"], color=plots.INK, lw=1.0, ls=(0, (4, 2)), label="required: 1 % of the reference")
    b.set_xlabel("solver time of the whole run (s)")
    b.set_ylabel("ignition time: difference (%)")
    b.set_title("Accuracy against cost, every run")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=3, fontsize=8.5)
    return plots.to_svg(fig, name="cost")
