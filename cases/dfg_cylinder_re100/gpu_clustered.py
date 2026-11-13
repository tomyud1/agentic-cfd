"""Case 4 on gpucfd grids clustered around the cylinder, next to the uniform-grid run and OpenFOAM.

Usage (repo root, inside WSL):  python -m cases.dfg_cylinder_re100.gpu_clustered [C1 C2 ...]
Reads results/dfg_cylinder_re100/runs/C*.json|npz (gpu_run.py clustered), gpu_compare.json (the uniform G
levels and OpenFOAM, gpu_compare.py) and the case YAML files. Writes results/dfg_cylinder_re100/gpu_clustered.json.
Every number is computed here.

What it answers (numerics.yaml, gpucfd.clustered, fixed before the runs):
- at the drag error of the uniform G3 run, does the clustered protocol take less wall time?
- how close does the finest clustered level come to FeatFlow, and in what wall time against OpenFOAM's study?
"""
import json
import sys
from pathlib import Path

from cfdlib.report import load_yaml
from .analyze import QKEYS, reference, study
from .gpu_compare import analyse
from .gpu_run import run_spec, runs_where

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"


def main(names=None):
    problem = load_yaml(HERE / "problem.yaml")
    numerics = load_yaml(HERE / "numerics.yaml")
    g = problem["given"]
    C = numerics["gpucfd"]["clustered"]
    tgt = C["target"]             # the target fixed before the runs: drag within tgt["rel"], study time below tgt["wall"]
    explicit = numerics["gpucfd"]["scheme"]   # this analysis covers the explicit runs; the IMEX ones are gpu_imex.py's
    names = names or [L for L in C["levels"] if (RUNS / f"{L}.json").exists()]
    ref = reference(load_yaml(HERE / problem["reference"]["file"]))
    uni = json.loads((OUT / "gpu_compare.json").read_text())
    def row(nm):
        r = analyse(nm, g, numerics)
        m, p = r["meta"], r["meta"]["perf"]
        return dict(level=nm, n=m["n"], cells=m["cells"], D_h=m["D_h"], dt=m["dt"], t0=m["start_time"], t1=m["end_time"],
                    steps=p["steps"], gpu_s=p["gpu_s"], wall_s=p["wall_s"], ms_per_step=p["ms_per_step"],
                    poisson_its=p["poisson_its"], health=m["health"]["status"], source=m.get("source"),
                    pressure_tol=m.get("pressure_tol"),
                    periodic=r["periodic"], periodic_from=(m["watch_first"] or {}).get("t"), mass=r["mass"],
                    St_fft=r["St_fft"], St_probe=r["St_probe"],
                    sm={k: r["sm"][k] for k in QKEYS},
                    rel={k: (r["sm"][k] - ref["value"][k]) / abs(ref["value"][k]) for k in QKEYS})
    rows, wall, gpu_s = [], 0.0, 0.0
    for nm in names:
        r = row(nm)
        wall += r["wall_s"]
        gpu_s += r["gpu_s"]
        rows.append(dict(r, wall_protocol=wall, gpu_protocol=gpu_s))
    # Check runs: the explicit runs of the table (numerics.yaml, gpucfd.runs) that have a variant, by variant.
    by = {r["level"]: r for r in rows}
    variants = {}
    for v in C.get("variants", {}):
        w = 0.0
        for nm in runs_where(numerics, family="clustered", variant=v, scheme=explicit):
            if (RUNS / f"{nm}.json").exists():
                r = row(nm)
                w += r["wall_s"]
                variants[nm] = dict(r, wall_variant_levels=w)
    # the run with the cylinder moved (variant s) on each level, where it was made
    moved = {run_spec(nm, numerics)["level"]: nm for nm in variants if run_spec(nm, numerics)["variant"] == "s"}
    # Two positions of the cylinder in the grid (variant s): both values, their middle and half their distance,
    # per level and quantity, relative to FeatFlow. Two cuts give a range, not a distribution.
    cuts = {}
    for L in C["levels"]:
        if L in by and L in moved:
            a, b = by[L]["rel"], variants[moved[L]]["rel"]
            cuts[L] = dict(D_h=by[L]["D_h"], first=a, shifted=b, middle={k: 0.5 * (a[k] + b[k]) for k in QKEYS},
                           half_range={k: 0.5 * abs(a[k] - b[k]) for k in QKEYS})
    ug = uni["cost_gpu"]
    uni_rows, w = [], 0.0
    for c in ug:
        w += c["wall_s"]
        sm = uni["runs"][c["level"]]["sm"]
        uni_rows.append(dict(level=c["level"], cells=c["cells"], D_h=c["D_h"], steps=c["steps"], gpu_s=c["gpu_s"], wall_s=c["wall_s"],
                             ms_per_step=c["ms_per_step"], wall_protocol=w,
                             rel={k: (sm[k] - ref["value"][k]) / abs(ref["value"][k]) for k in QKEYS}))
    of_rows, w = [], 0.0
    for c in uni["cost_openfoam"]:
        w += c["wall_s"]
        of_rows.append(dict(level=c["level"], cells=c["cells"], cores=c["cores"], wall_s=c["wall_s"], wall_protocol=w,
                            rel_cd_mean=c["err_cd_mean"] / abs(ref["value"]["cd_mean"])))
    g3 = uni_rows[-1]
    # The cheap question: the clustered level with the uniform G3's cells per diameter at the body, its drag
    # error and the wall time of the protocol up to it, against G1 + G2 + G3. Stated as measured: the errors
    # are close, not equal, so `no_larger_error` says whether the clustered one is at most the uniform one.
    match = min(rows, key=lambda r: abs(r["D_h"] - g3["D_h"]))
    cheap = dict(level=match["level"], D_h=match["D_h"], rel_cd_mean=match["rel"]["cd_mean"], rel_cd_max=match["rel"]["cd_max"],
                 rel_cl_max=match["rel"]["cl_max"], rel_St=match["rel"]["St"], wall=match["wall_protocol"], cells=match["cells"],
                 uniform_rel_cd_mean=g3["rel"]["cd_mean"], uniform_rel_cd_max=g3["rel"]["cd_max"], uniform_rel_cl_max=g3["rel"]["cl_max"],
                 uniform_rel_St=g3["rel"]["St"], uniform_wall=g3["wall_protocol"], uniform_cells=g3["cells"],
                 faster=bool(match["wall_protocol"] < g3["wall_protocol"]), time_ratio=match["wall_protocol"] / g3["wall_protocol"],
                 no_larger_error=bool(abs(match["rel"]["cd_mean"]) <= abs(g3["rel"]["cd_mean"])))
    # What the clustered levels say about their own error, with no reference: observed order and the grid
    # convergence index of the finest level (the case's U_sim without the time-step part), per quantity.
    conv = {}
    if len(rows) >= 3:
        for k in QKEYS:
            st = study([r["sm"][k] for r in rows[-3:]])
            diff = rows[-1]["sm"][k] - ref["value"][k]
            band = st["gci_abs"] + ref["unc"][k]
            conv[k] = dict(p=st["p"], monotonic=st["monotonic"], gci_reliable=st["gci_reliable"], gci_basis=st["gci_basis"],
                           extrap=st["extrap"], gci_abs=st["gci_abs"],
                           gci_rel=st["gci_abs"] / abs(ref["value"][k]), u_ref_rel=ref["unc"][k] / abs(ref["value"][k]),
                           diff=diff, band=band, status="ok" if abs(diff) <= band else ("warn" if abs(diff) <= 2 * band else "fail"))
    conv_shifted = {}
    sh = [variants.get(moved.get(L)) for L in C["levels"]]
    if all(sh) and len(sh) >= 3:  # the same for the three levels with the shifted cylinder
        for k in QKEYS:
            st = study([r["sm"][k] for r in sh[-3:]])
            conv_shifted[k] = dict(p=st["p"], monotonic=st["monotonic"], gci_reliable=st["gci_reliable"],
                                   gci_basis=st["gci_basis"], gci_abs=st["gci_abs"],
                                   gci_rel=st["gci_abs"] / abs(ref["value"][k]))
    fine = rows[-1]
    target = dict(level=fine["level"], rel_cd_mean=fine["rel"]["cd_mean"], rel_cd_max=fine["rel"]["cd_max"], wall=fine["wall_protocol"],
                  want_rel=tgt["rel"], want_wall=tgt["wall"],
                  accuracy_met=bool(abs(fine["rel"]["cd_mean"]) <= tgt["rel"] and abs(fine["rel"]["cd_max"]) <= tgt["rel"]),
                  time_met=bool(fine["wall_protocol"] < tgt["wall"]),
                  stage2_met=bool(abs(fine["rel"]["cd_mean"]) <= tgt["stage2_rel"] and abs(fine["rel"]["cd_max"]) <= tgt["stage2_rel"]))
    if fine["level"] in cuts:  # the target on both cuts
        c3s = variants[moved[fine["level"]]]
        target.update(shifted_rel_cd_mean=c3s["rel"]["cd_mean"], shifted_rel_cd_max=c3s["rel"]["cd_max"],
                      shifted_wall=c3s["wall_variant_levels"],
                      shifted_accuracy_met=bool(abs(c3s["rel"]["cd_mean"]) <= tgt["rel"] and abs(c3s["rel"]["cd_max"]) <= tgt["rel"]),
                      shifted_time_met=bool(c3s["wall_variant_levels"] < tgt["wall"]),
                      shifted_periodic=c3s["periodic"])
    out = dict(clustered=rows, uniform=uni_rows, openfoam=of_rows, cheap_question=cheap, target=target, convergence=conv,
               variants=variants, cuts=cuts, convergence_shifted=conv_shifted,
               reference={k: ref["value"][k] for k in QKEYS})
    (OUT / "gpu_clustered.json").write_text(json.dumps(out, indent=1, default=float))

    def line(r):
        e = r["rel"]
        return (f"cd_max {100 * e['cd_max']:+.2f} % cd_mean {100 * e['cd_mean']:+.2f} % cl_max {100 * e['cl_max']:+.2f} % "
                f"cl_amp {100 * e['cl_amp']:+.2f} % St {100 * e['St']:+.2f} %")
    for r in rows:
        print(f"{r['level']}: {r['cells']} cells, D/h {r['D_h']:.0f}, dt {r['dt']:.3g}, t {r['t0']:g}-{r['t1']:.2f}, {r['steps']} steps, "
              f"{r['ms_per_step']:.2f} ms/step ({r['poisson_its']:.1f} its), GPU {r['gpu_s']:.0f} s, wall {r['wall_s']:.0f} s "
              f"(protocol so far {r['wall_protocol']:.0f} s) | {line(r)} | periodic {r['periodic']} health {r['health']}")
    for r in variants.values():
        print(f"{r['level']}: {r['cells']} cells, D/h {r['D_h']:.0f}, t {r['t0']:g}-{r['t1']:.2f}, {r['ms_per_step']:.2f} ms/step, "
              f"wall {r['wall_s']:.0f} s (variant levels so far {r['wall_variant_levels']:.0f} s), from {r['source']} | {line(r)} | "
              f"periodic {r['periodic']} health {r['health']}")
    for L, c in cuts.items():
        print(f"cut spread {L}: " + " ".join(f"{k} {100 * c['middle'][k]:+.2f} +- {100 * c['half_range'][k]:.2f} %"
                                           for k in ("cd_max", "cd_mean", "cl_max", "cl_amp", "St")))
    for r in uni_rows:
        print(f"uniform {r['level']}: {r['cells']} cells, D/h {r['D_h']:.1f}, {r['ms_per_step']:.2f} ms/step, wall {r['wall_s']:.0f} s "
              f"(protocol so far {r['wall_protocol']:.0f} s) | {line(r)}")
    for r in of_rows:
        print(f"OpenFOAM {r['level']}: {r['cells']} cells, wall {r['wall_s']:.0f} s (so far {r['wall_protocol']:.0f} s), cd_mean "
              f"{100 * r['rel_cd_mean']:+.2f} %")
    for k, c in conv.items():
        print(f"convergence {k}: order {c['p']:.2f} monotonic {c['monotonic']} GCI {100 * c['gci_rel']:.2f} % reference unc "
              f"{100 * c['u_ref_rel']:.2f} % -> {c['status']}")
    for k, c in conv_shifted.items():
        print(f"convergence, shifted set, {k}: order {c['p']:.2f} monotonic {c['monotonic']} GCI {100 * c['gci_rel']:.2f} %")
    print("cheap question:", cheap)
    print("target:", target)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
