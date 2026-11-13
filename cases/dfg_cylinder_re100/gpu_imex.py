"""Case 4 on gpucfd with the IMEX scheme (implicit diffusion) on the clustered grids, against the recorded
explicit runs on the same grids.

Usage (repo root, inside WSL):  python -m cases.dfg_cylinder_re100.gpu_imex
Reads results/dfg_cylinder_re100/runs/C*.json|npz (gpu_run.py clustered: C1..C3 explicit, C1i..C3i IMEX and
the variants that were run) and the case YAML files. Writes results/dfg_cylinder_re100/gpu_imex.json.
Every number is computed here; the explicit records and gpu_clustered.json are only read.

What it answers (numerics.yaml, gpucfd.clustered.imex, fixed before the runs):
- same answer: is each IMEX level within the rule of its explicit twin (drag and Strouhal number 0.05 %,
  lift 0.3 %)?
- what does the three-level study cost when the finest level uses IMEX, against the explicit study and
  OpenFOAM's?
"""
import json
import sys
from pathlib import Path

from cfdlib import gpu
from cfdlib.report import load_yaml
from .analyze import QKEYS, reference
from .gpu_compare import analyse
from .gpu_run import run_spec, runs_where

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"


def main():
    problem = load_yaml(HERE / "problem.yaml")
    numerics = load_yaml(HERE / "numerics.yaml")
    g = problem["given"]
    C = numerics["gpucfd"]["clustered"]
    RULE = C["imex"]["same_answer"]            # the same-answer rule, relative to the explicit value (fixed before the runs)
    OPENFOAM_STUDY_WALL = C["target"]["wall"]  # OpenFOAM's whole study with the live stop: the target's time
    ref = reference(load_yaml(HERE / problem["reference"]["file"]))
    # The benchmark's finest tabulated level is not its converged value: its three finest levels head for a mean
    # drag about 0.2 % higher (analyze.reference, Richardson). Mean drag is quoted against both.
    ext = ref["seq"]["cd_mean"]["extrap"]

    def row(nm):
        r = analyse(nm, g, numerics)
        m, p = r["meta"], r["meta"]["perf"]
        return dict(level=nm, scheme=m.get("scheme"), cells=m["cells"], D_h=m["D_h"], dt=m["dt"], t0=m["start_time"], t1=m["end_time"],
                    steps=p["steps"], gpu_s=p["gpu_s"], wall_s=p["wall_s"], ms_per_step=p["ms_per_step"],
                    poisson_its=p["poisson_its"], health=m["health"]["status"], source=m.get("source"),
                    pressure_tol=m.get("pressure_tol"), periodic=r["periodic"], periodic_from=(m["watch_first"] or {}).get("t"),
                    sm={k: r["sm"][k] for k in QKEYS},
                    rel={k: (r["sm"][k] - ref["value"][k]) / abs(ref["value"][k]) for k in QKEYS},
                    rel_cd_mean_converged=(r["sm"]["cd_mean"] - ext) / abs(ext))
    # every run on a clustered grid that the table lists (numerics.yaml, gpucfd.runs) and that was made
    rows = {nm: row(nm) for nm in sorted(runs_where(numerics, family="clustered")) if (RUNS / f"{nm}.json").exists()}
    spec = {nm: run_spec(nm, numerics) for nm in rows}
    pairs = []
    for nm, r in rows.items():  # every IMEX run with its explicit twin: the run on the same level and variant
        twin = next((x for x in rows if spec[x]["scheme"] != "imex" and (spec[x]["level"], spec[x]["variant"]) ==
                     (spec[nm]["level"], spec[nm]["variant"])), None)
        if spec[nm]["scheme"] != "imex" or twin is None:
            continue
        e = rows[twin]
        d = {k: (r["sm"][k] - e["sm"][k]) / abs(e["sm"][k]) for k in RULE}
        pairs.append(dict(imex=nm, explicit=twin, diff=d, same={k: bool(abs(d[k]) <= RULE[k]) for k in RULE},
                          same_answer=bool(all(abs(d[k]) <= RULE[k] for k in RULE)),
                          steps_ratio=e["steps"] / r["steps"] * (r["t1"] - r["t0"]) / (e["t1"] - e["t0"]),
                          dt_ratio=r["dt"] / e["dt"], ms_ratio=r["ms_per_step"] / e["ms_per_step"],
                          wall_ratio=e["wall_s"] / r["wall_s"], gpu_per_flow_second=[e["gpu_s"] / (e["t1"] - e["t0"]),
                                                                                     r["gpu_s"] / (r["t1"] - r["t0"])]))
    out = dict(rule=RULE, runs=rows, pairs=pairs, reference={k: ref["value"][k] for k in QKEYS},
               reference_converged=dict(cd_mean=ext, order=ref["seq"]["cd_mean"]["p"]),
               openfoam_study_wall=OPENFOAM_STUDY_WALL)
    # The study: C1 and C2 explicit, the finest level with IMEX (each level the scheme that is faster on it).
    for label, names in (("explicit", ["C1", "C2", "C3"]), ("imex_finest", ["C1", "C2", "C3i"]), ("imex_all", ["C1i", "C2i", "C3i"]),
                         ("explicit_shifted", ["C1s", "C2s", "C3s"]), ("imex_finest_shifted", ["C1s", "C2s", "C3si"]),
                         ("imex_finest_long_band", ["C1", "C2w", "C3wi"])):
        if all(n in rows for n in names):
            out.setdefault("studies", {})[label] = dict(levels=names, wall=sum(rows[n]["wall_s"] for n in names),
                                                        gpu=sum(rows[n]["gpu_s"] for n in names), finest_rel=rows[names[-1]]["rel"],
                                                        finest_rel_cd_mean_converged=rows[names[-1]]["rel_cd_mean_converged"])
    if "C3i" in rows and "C3si" in rows:  # the finest IMEX level at two positions of the cylinder in the grid
        sp = gpu.position_spread({"first": rows["C3i"]["rel"], "shifted": rows["C3si"]["rel"]})
        out["cuts_C3i"] = sp
    (OUT / "gpu_imex.json").write_text(json.dumps(out, indent=1, default=float))

    def line(r):
        e = r["rel"]
        c = f" (against the converged benchmark {100 * r['rel_cd_mean_converged']:+.3f} %)" if "rel_cd_mean_converged" in r else ""
        return (f"cd_max {100 * e['cd_max']:+.3f} % cd_mean {100 * e['cd_mean']:+.3f} %{c} cl_max {100 * e['cl_max']:+.2f} % "
                f"cl_amp {100 * e['cl_amp']:+.2f} % St {100 * e['St']:+.3f} %")
    for nm, r in rows.items():
        print(f"{nm} ({r['scheme']}): dt {r['dt']:.3g}, tol {r['pressure_tol']:g}, t {r['t0']:g}-{r['t1']:.2f}, {r['steps']} steps, "
              f"{r['ms_per_step']:.2f} ms/step ({r['poisson_its']:.1f} its), GPU {r['gpu_s']:.0f} s, wall {r['wall_s']:.0f} s, from "
              f"{r['source']} | {line(r)} | periodic {r['periodic']} health {r['health']}")
    for p in pairs:
        print(f"{p['imex']} against {p['explicit']}: " + " ".join(f"{k} {100 * v:+.3f} %{'' if p['same'][k] else ' (!)'}" for k, v in p["diff"].items())
              + f" -> same answer: {p['same_answer']}; time step x{p['dt_ratio']:.2f}, per step x{p['ms_ratio']:.2f}, wall time / {p['wall_ratio']:.2f}")
    for k, s in out.get("studies", {}).items():
        print(f"study {k} ({' + '.join(s['levels'])}): wall {s['wall']:.0f} s, GPU {s['gpu']:.0f} s; finest {line(dict(rel=s['finest_rel'], rel_cd_mean_converged=s['finest_rel_cd_mean_converged']))}")
    if "cuts_C3i" in out:
        print("C3i over two positions: " + " ".join(f"{k} {100 * v['middle']:+.3f} +- {100 * v['half_range']:.3f} %" for k, v in out["cuts_C3i"].items()))


if __name__ == "__main__":
    main()
