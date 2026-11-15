"""The numbers of the analysis: every run against the reference and against the independent solution, the
time-step study, the comparison of the two histories, and the cost. Used by analyze.py and figs.py."""
import json

import numpy as np

from cfdlib import verification
from . import run, theory


def compute():
    problem, numerics, ref = theory.load()
    g = theory.given(problem)
    th = theory.numbers()
    hist = theory.histories()
    specs = run.specs(numerics)
    runs, data, crashed = {}, {}, {}
    for n in specs:
        p = run.RUNS / f"{n}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        if r.get("crashed"):
            crashed[n] = r
            continue
        runs[n] = r
        z = np.load(run.RUNS / f"{n}.npz")
        data[n] = {k: z[k] for k in z.files}
    t_ref, T_ref = ref["ignition"]["time_s"], ref["final"]["T_K"]
    ind, ind_of = th["independent"], th["independent_of"]
    for n, r in runs.items():
        r["d_ref_t"] = r["t_ign"] / t_ref - 1 if r["t_ign"] else None
        r["d_ref_T"] = r["T_end"] / T_ref - 1
        r["d_of_t"] = r["t_ign"] / ind_of["t_ign"] - 1 if r["t_ign"] else None
        r["d_of_T"] = r["T_end"] - ind_of["T_end"]
        r["atoms_max"] = max(r["atoms_drift"].values()) if "atoms_drift" in r else None
        r["dt"] = g["t_end"] / r["steps"] if not r["adjust"] else None
        if r.get("phase_s"):
            r["phase_us"] = [1e6 * s / k if k else None for s, k in zip(r["phase_s"], r["phase_steps"])]

    # ---- the time-step family ----
    fam = [f"S{n}" for n in numerics["runs"]["family_steps"] if f"S{n}" in runs]
    planned = [f"S{n}" for n in numerics["runs"]["family_steps"]]
    F = [runs[n] for n in fam]
    dts = [r["dt"] for r in F]
    study = {}
    for key, val, exact in (("t_ign", [r["t_ign"] for r in F], ind_of["t_ign"]), ("T_end", [r["T_end"] for r in F], ind_of["T_end"])):
        err = [abs(v - exact) for v in val]
        s3 = verification.three_grid_study(val, r=2.0, p_assumed=1.0) if len(val) >= 3 else None
        study[key] = dict(values=val, err=err, orders=verification.order_from_errors(dts, err) if len(val) > 1 else [],
                          p=s3["p"] if s3 else None, extrap=s3["extrap"] if s3 else None, monotonic=s3["monotonic"] if s3 else None,
                          gci_abs=s3["gci_abs"] if s3 else None,
                          orders3=[verification.three_grid_study(val[i:i + 3], r=2.0, p_assumed=1.0)["p"] for i in range(len(val) - 2)])
    study["t_ign"]["extrap_vs_of"] = study["t_ign"]["extrap"] / ind_of["t_ign"] - 1
    study["T_end"]["extrap_vs_of"] = study["T_end"]["extrap"] - ind_of["T_end"]
    study["t_ign"]["extrap_vs_ref"] = study["t_ign"]["extrap"] / t_ref - 1
    study["T_end"]["extrap_vs_ref"] = study["T_end"]["extrap"] / T_ref - 1

    # ---- the finest history against the reference's printed history ----
    fine = F[-1]
    d = data[fam[-1]]
    levels = [str(int(L)) for L in numerics["extraction"]["levels"]]
    ref_lv = th["reference"]["levels"]
    shift = {L: fine["levels"][L] - ref_lv[L] for L in levels}
    shift_ind = {L: fine["levels"][L] - ind_of["levels"][L] for L in levels}
    rt, rT = np.array(ref["history"]["t_s"]), np.array(ref["history"]["T_K"])
    flat = (rT < g["T_0"] + 10) | (rT > 2100)
    dT_pts = np.interp(rt, d["t"], d["T"]) - rT
    steep_levels = [L for L in levels if int(L) <= 2000]
    history = dict(shift=shift, shift_ind=shift_ind, shift_max=max(abs(shift[L]) for L in steep_levels), steep_levels=steep_levels,
                   dT_flat_max=float(np.max(np.abs(dT_pts[flat]))), dT_flat_mean=float(np.mean(dT_pts[flat & (rT > 2100)])),
                   dT_steep_max=float(np.max(np.abs(dT_pts[~flat]))), n_points=len(rt), n_flat=int(flat.sum()),
                   rise_rate=fine["rise_rate"], rise_t=fine["rise_t"])

    # ---- species: the finest run against the independent solution with OpenFOAM's constants ----
    sp = th["species"]
    W = np.array([th["W"][s] for s in sp])
    sig_run = d["Y"] / W                                            # mol/kg
    sig_ind = np.column_stack([np.interp(d["t"], hist["t"], hist["Y_of"][:, i]) for i in range(len(sp))]) / W
    species = {}
    for i, s in enumerate(sp):
        peak = float(sig_run[:, i].max())
        if peak > 0:
            species[s] = dict(start=float(sig_run[0, i]), end=float(sig_run[-1, i]), peak=peak, t_peak=float(d["t"][int(np.argmax(sig_run[:, i]))]),
                              end_ind=float(sig_ind[-1, i]), dmax=float(np.max(np.abs(sig_run[:, i] - sig_ind[:, i])) / peak),
                              end_eq=th["equilibrium"]["of"]["Y"][s] / W[i])
    asked = ["H2", "O2", "H2O", "OH", "H", "O"]
    species_dmax = max(species[s]["dmax"] for s in asked)

    # ---- cost ----
    main10 = runs.get(f"S{numerics['runs']['timing']['steps']}")
    notable = runs.get(f"S{numerics['runs']['timing']['steps']}_notable")
    total_solver = sum(r["wall_time"] for r in runs.values())
    return dict(problem=problem, numerics=numerics, ref=ref, g=g, th=th, hist=hist, specs=specs, runs=runs, data=data, crashed=crashed,
                fam=fam, planned=planned, dts=dts, study=study, history=history, species=species, species_dmax=species_dmax, asked=asked,
                main10=main10, notable=notable, total_solver=total_solver, t_ref=t_ref, T_ref=T_ref)
