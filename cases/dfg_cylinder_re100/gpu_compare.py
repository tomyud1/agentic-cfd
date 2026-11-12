"""gpucfd against OpenFOAM on case 4 (DFG 2D-2): the same numbers, judged by the same rules, and the time each takes.

Usage (repo root, inside WSL):  python -m cases.dfg_cylinder_re100.gpu_compare
Reads results/dfg_cylinder_re100/runs/G*.json|npz (gpu_run.py), analysis.json (analyze.py, the OpenFOAM result)
and the case YAML files. Writes results/dfg_cylinder_re100/gpu_compare.json. Every number is computed here.

Rules (numerics.yaml, gpucfd block, fixed before the runs):
- accuracy: the case's own rule applied to G3: band = U_sim + U_ref, U_sim = GCI of G1..G3 + time-step error;
- same answer: |G3 - M4| <= U_G3 + U_M4 for every primary quantity.
"""
import json
from pathlib import Path

import numpy as np

from cfdlib.report import load_yaml
from . import signals as S
from .analyze import QKEYS, reference, study

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"


def analyse(name, given, numerics):
    meta = json.loads((RUNS / f"{name}.json").read_text())
    d = np.load(RUNS / f"{name}.npz")
    D, Umean = given["D"]["value"], 2 * given["Um"]["value"] / 3
    t, cd, cl = d["t"], d["cd"], d["cl"]
    cyc = S.cycles(t, cd, cl)
    n = numerics["time"]["cycles_measured"]
    sm = S.summary(cyc, n)
    sm["St"] = sm["f"] * D / Umean
    win = (t >= cyc[-n]["t0"]) & (t <= cyc[-1]["t1"])
    f_fft = S.dominant_frequency(t[win], cl[win])
    fD = S.dominant_frequency(t[win], cd[win])
    f_probe = S.dominant_frequency(t[win], d["probe_v"][win])
    mass = float(np.max(np.abs(d["q_in"] + d["q_out"])) / np.max(np.abs(d["q_in"])))
    acc = numerics["acceptance"]
    sp = sm["spread"]
    periodic = bool(max(sp["cl_max"], sp["cl_min"]) <= acc["periodic_cl_peak"] and max(sp["cd_max"], sp["cd_min"]) <= acc["periodic_cd_peak"]
                    and sp["period"] / sm["period"] <= acc["periodic_period_rel"])
    return dict(name=name, meta=meta, sm=sm, St_fft=f_fft * D / Umean, St_probe=f_probe * D / Umean, ratio_fD=fD / f_fft,
                mass=mass, periodic=periodic, t=t, cd=cd, cl=cl)


def main():
    problem = load_yaml(HERE / "problem.yaml")
    numerics = load_yaml(HERE / "numerics.yaml")
    g, G, acc = problem["given"], numerics["gpucfd"], numerics["acceptance"]
    ref = reference(load_yaml(HERE / problem["reference"]["file"]))
    of = json.loads((OUT / "analysis.json").read_text())
    lv = G["levels"]
    tname = f"{G['dt_check']['level']}_dtx{G['dt_check']['factor']:g}"
    runs = {nm: analyse(nm, g, numerics) for nm in lv + [tname]}
    fine = runs[lv[-1]]
    ms = {k: study([runs[L]["sm"][k] for L in lv]) for k in QKEYS}
    # Time-step error: G2 against G2 with dt / 2. With two time steps no order can be observed, so the change
    # itself is taken as the time error of G3 (whose dt is 3.3 times smaller still): conservative.
    base = runs[G["dt_check"]["level"]]
    u_time = {k: abs(base["sm"][k] - runs[tname]["sm"][k]) for k in QKEYS}
    comp, same = {}, {}
    for k in QKEYS:
        sim = fine["sm"][k]
        u_sim = ms[k]["gci_abs"] + u_time[k]
        band = u_sim + ref["unc"][k]
        diff = sim - ref["value"][k]
        status = "ok" if abs(diff) <= band else ("warn" if abs(diff) <= 2 * band else "fail")
        comp[k] = dict(sim=sim, ref=ref["value"][k], u_ref=ref["unc"][k], u_sim=u_sim, u_time=u_time[k], gci=ms[k]["gci_abs"],
                       band=band, diff=diff, rel=diff / abs(ref["value"][k]), ratio=abs(diff) / band, status=status,
                       p=ms[k]["p"], monotonic=ms[k]["monotonic"], extrap=ms[k]["extrap"])
        o = of["comp"][k]
        d_of = sim - o["sim"]
        tol = u_sim + o["u_sim"]
        same[k] = dict(gpu=sim, openfoam=o["sim"], diff=d_of, tol=tol, ok=abs(d_of) <= tol, u_gpu=u_sim, u_openfoam=o["u_sim"],
                       openfoam_status=o["status"])
    fail, warn = [], []
    for nm, r in runs.items():
        if r["meta"]["health"]["status"] == "fail":
            fail.append(f"run {nm} failed a health check")
        if not r["periodic"]:
            (fail if nm == lv[-1] else warn).append(f"run {nm} not periodic to the set tolerance")
    for k in acc["primary"]:
        if comp[k]["status"] == "fail":
            fail.append(f"{k}: more than twice the combined uncertainty from the reference")
        elif comp[k]["status"] == "warn":
            warn.append(f"{k}: more than the combined uncertainty from the reference (less than twice)")
        if not ms[k]["monotonic"]:
            warn.append(f"{k}: grid convergence G1-G3 not monotonic")
        elif not acc["order_range"][0] <= ms[k]["p"] <= acc["order_range"][1]:
            warn.append(f"{k}: observed order {ms[k]['p']:.2f} outside {acc['order_range']}")
    for k in acc["secondary"]:
        if comp[k]["status"] == "fail":
            warn.append(f"{k} (secondary): more than twice the combined uncertainty from the reference")
    verdict = "FAIL" if fail else "WARN" if warn else "PASS"
    same_ok = all(same[k]["ok"] for k in acc["primary"])
    # Cost: each tool's whole protocol (coarsest from rest, then each finer level 3 s from the coarser field)
    cost_gpu = [dict(level=L, cells=runs[L]["meta"]["cells"], D_h=runs[L]["meta"]["D_h"], steps=runs[L]["meta"]["perf"]["steps"],
                     gpu_s=runs[L]["meta"]["perf"]["gpu_s"], wall_s=runs[L]["meta"]["perf"]["wall_s"],
                     observe_s=runs[L]["meta"]["perf"]["observe_s"], ms_per_step=runs[L]["meta"]["perf"]["ms_per_step"],
                     err_cd_mean=runs[L]["sm"]["cd_mean"] - ref["value"]["cd_mean"],
                     err_St=runs[L]["sm"]["St"] - ref["value"]["St"]) for L in lv]
    cost_of = []
    for L in numerics["mesh"]["levels"]:
        mt = json.loads((RUNS / f"{L}.json").read_text())
        sm = of["runs"][L]["sm"]
        cost_of.append(dict(level=L, cells=mt["cells"], cores=mt["cores"], steps=mt["perf"]["steps"], wall_s=mt["wall_time"],
                            err_cd_mean=sm["cd_mean"] - ref["value"]["cd_mean"],
                            err_St=sm["f"] * g["D"]["value"] / (2 * g["Um"]["value"] / 3) - ref["value"]["St"]))
    out = dict(verdict=verdict, fail=fail, warn=warn, comp=comp, same=same, same_ok=same_ok, ms=ms, u_time=u_time,
               runs={nm: dict(sm=r["sm"], St_fft=r["St_fft"], St_probe=r["St_probe"], ratio_fD=r["ratio_fD"], mass=r["mass"],
                              periodic=r["periodic"], meta=r["meta"]) for nm, r in runs.items()},
               cost_gpu=cost_gpu, cost_openfoam=cost_of)
    (OUT / "gpu_compare.json").write_text(json.dumps(out, indent=1, default=float))
    print("verdict", verdict, "| fail", fail, "| warn", warn, "| same answer as OpenFOAM:", same_ok)
    for k in QKEYS:
        c, s = comp[k], same[k]
        print(f"{k:8s} G3 {c['sim']:.5f} ref {c['ref']:.5f} diff {c['diff']:+.5f} ({100 * c['rel']:+.2f}%) band {c['band']:.5f} "
              f"(gci {c['gci']:.5f} time {c['u_time']:.5f} ref {c['u_ref']:.5f}) {c['status']} | p {c['p']:.2f} | "
              f"M4 {s['openfoam']:.5f} G3-M4 {s['diff']:+.5f} tol {s['tol']:.5f} {'same' if s['ok'] else 'DIFFERENT'}")
    for nm, r in runs.items():
        print(f"{nm}: cd_mean {r['sm']['cd_mean']:.5f} cl_amp {r['sm']['cl_amp']:.5f} St {r['sm']['St']:.5f} (fft {r['St_fft']:.5f}, "
              f"probe {r['St_probe']:.5f}) fD/fL {r['ratio_fD']:.4f} mass {r['mass']:.1e} periodic {r['periodic']} "
              f"spread cl {max(r['sm']['spread']['cl_max'], r['sm']['spread']['cl_min']):.1e}")
    tg = sum(c["gpu_s"] for c in cost_gpu)
    tw = sum(c["wall_s"] for c in cost_gpu)
    to = sum(c["wall_s"] for c in cost_of)
    for c in cost_gpu:
        print(f"gpucfd {c['level']}: {c['cells']} cells, D/h {c['D_h']:.1f}, {c['steps']} steps, GPU {c['gpu_s']:.1f} s, "
              f"wall {c['wall_s']:.1f} s ({c['ms_per_step']:.2f} ms/step), cd_mean err {c['err_cd_mean']:+.4f}, St err {c['err_St']:+.5f}")
    for c in cost_of:
        print(f"OpenFOAM {c['level']}: {c['cells']} cells, {c['cores']} ranks, {c['steps']} steps, wall {c['wall_s']:.1f} s, "
              f"cd_mean err {c['err_cd_mean']:+.4f}, St err {c['err_St']:+.5f}")
    print(f"total: gpucfd GPU {tg:.0f} s, wall {tw:.0f} s; OpenFOAM wall {to:.0f} s")


if __name__ == "__main__":
    main()
