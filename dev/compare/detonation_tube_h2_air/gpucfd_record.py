"""The record of gpucfd's runs of case 21 for the one-page report: gpucfd.json, formed from gpucfd_study.json
(written by gpucfd_run.py) next to the numbers of OpenFOAM's solution of the case (results/detonation_tube_h2_air)
and of OpenFOAM's trial run. It runs no solver; it was written after the runs of rung d were finished.

    bash dev/compare/run.sh detonation_tube_h2_air gpucfd_run.py record

Form (cfdlib/brief.py): answers {letter: {diff}} on 3200 cells, the finest grid of OpenFOAM's solution, formed as
the case's report forms OpenFOAM's cells; runs [{N, cells, steps_done, wall_s}] on OpenFOAM's four grids, and rows
of their own (key: label) for the grids on which no detonation forms and for the tube on a full GPU; plain.
"""
import datetime as dt
import json
from pathlib import Path

from cfdlib import chem
from cfdlib.report import lines, m, m_pct, tex_num
from gpucfd.tests import test_react_tube as T

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDY, OUT = HERE / "gpucfd_study.json", HERE / "gpucfd.json"
RES = ROOT / "results" / "detonation_tube_h2_air"
TRIAL = dict(cells=1200, speed_dev=-0.011, us_per_cell_step=57.0, wall_s=689.0, cores=1, p_peak=1.97e6, p_back=1.56e6, T_back=2920.0, dt=5e-9,
             source="dev/agent_checks/case21_toolgap.py, as the task statement of stage 2 gave it")
sc = lambda x, sig=1: m(tex_num(x, sig)) if x else m("0")                                                 # noqa: E731


def record():
    st = json.loads(STUDY.read_text(encoding="utf-8"))
    of = json.loads((RES / "results.json").read_text(encoding="utf-8"))["data"]["runs"]
    g, mech, moles, cj = T.problem()

    def row(r):
        """One run in short; the pressure behind a non-burning shock of the run's own speed stands next to its peak."""
        out = {k: r[k] for k in ("N", "h", "cells", "steps", "gpu_s", "wall_s", "detonation", "mass", "atoms", "energy", "status", "substeps_per_cell_step", "ns_per_cell_step",
                                 "p_peak", "guard", "given")}
        out["tag"] = r.get("tag", "")
        for k in ("energy_of_heat", "times", "rows_identical"):
            if k in r:
                out[k] = r[k]
        if "heat_released" in r:                                  # the run's totals are those of a slice one cell wide: per square metre of the tube's cross-section
            out["heat_released_J_per_m2"] = r["heat_released"] / (r["h"] * r["cells"] / r["N"])
        if r["detonation"]:
            vn = chem.shock_frozen(mech, g["p_0"], g["T_0"], moles, r["speed"])
            out.update(speed=r["speed"], speed_dev=r["speed_dev"], speed_first=r["speed_first"], speed_second=r["speed_second"], gap_max=r["gap_max"], gap_mean=r["gap_mean"],
                       sonic=r["sonic"], back=r["back"], energy_rebuilt=r["energy_rebuilt"], p_shock_of_this_speed=float(vn["p"]), peak_ratio=r["p_peak"]["mean"] / float(vn["p"]))
        else:
            out["shock"] = r["shock"]
        return out

    like = [row(r) for r in st["like"]]
    d1, border, d4, d2, d3 = ([row(r) for r in st[k]] for k in ("d1", "d1_border", "d4", "d2", "d3"))
    fine = like[-1]                                               # 3200 cells: the finest grid of OpenFOAM's solution
    o = of["N3200"]
    lost = sorted([r for r in d1 + border if not r["detonation"]], key=lambda r: r["N"])
    held = sorted([r for r in d1 + border + like if r["detonation"]], key=lambda r: r["N"])
    stacked = next(r for r in d3 if "times" not in r and r["tag"].endswith("chosen chemistry setting"))
    timed = {("bs23" if "bs23" in r["tag"] else "chosen"): r for r in d3 if "times" in r}

    answers = dict(
        a=dict(value=fine["speed"], diff=m_pct(fine["speed_dev"], 2, signed=True)),
        b=dict(value=fine["gap_max"], diff=lines("limit " + m(r"1\,\mathrm{mm}"), "found " + m(f"{1e3 * fine['gap_max']:.2f}" + r"\,\mathrm{mm}"))),
        c=dict(value=fine["sonic"], diff=lines("p " + m_pct(fine["sonic"]["p"] / cj["p"] - 1, 1, signed=True), "T " + m_pct(fine["sonic"]["T"] / cj["T"] - 1, 1, signed=True))),
        d=dict(value=dict(p_peak=fine["p_peak"]["mean"], p_shock=fine["p_shock_of_this_speed"]), diff=m_pct(fine["peak_ratio"] - 1, 0, signed=True)),
        e=dict(value=dict(mass=fine["mass"], atoms=fine["atoms"], energy_of_heat=fine["energy_of_heat"]),
               diff=lines("mass " + sc(max(fine["mass"], fine["atoms"])), "energy " + m(tex_num(100 * fine["energy_of_heat"], 1) + r"\,\%"))))

    runs = [dict(N=r["N"], cells=r["cells"], steps_done=r["steps"], wall_s=r["wall_s"], gpu_s=r["gpu_s"], speed_dev=r["speed_dev"], gap_max=r["gap_max"],
                 openfoam=dict(wall_s=of[f"N{r['N']}"]["wall_time"], cores=of[f"N{r['N']}"]["cores"], speed_dev=of[f"N{r['N']}"]["speed_err"], gap_max=of[f"N{r['N']}"]["gap_max"]))
            for r in like]
    for r in lost:
        if r["N"] in (4800, 19200):
            runs.append(dict(label=f"{r['N']} cells: no detonation forms ({r['steps']} steps)", N=r["N"], cells=r["cells"], steps_done=r["steps"], wall_s=r["wall_s"], gpu_s=r["gpu_s"]))
    runs.append(dict(label=f"the 1200-cell tube in 1024 rows, a full GPU ({stacked['steps']} steps)", cells=stacked["cells"], steps_done=stacked["steps"], wall_s=stacked["wall_s"],
                     gpu_s=stacked["gpu_s"], ns_per_cell_step=stacked["ns_per_cell_step"]))

    ratio = o["wall_time"] / fine["wall_s"]
    plain = (f"gpucfd's joined solver, built after this case was solved, runs the same tube on the same grids. On 3200 cells its wave travels at "
             + m(f"{fine['speed']:.1f}" + r"\,\mathrm{m/s}") + f" ({100 * fine['speed_dev']:+.2f} % from the theory), with the burning front "
             + m(f"{1e3 * fine['gap_max']:.2f}" + r"\,\mathrm{mm}") + f" behind the pressure front, in {fine['wall_s']:.0f} s on the GPU against {o['wall_time']:.0f} s on "
             f"{o['cores']} cores, and it keeps mass and energy to rounding. On finer grids ({lost[0]['N']} cells and more) gpucfd finds no detonation at all: the hot layer of "
             "the problem is too weak to ignite the gas behind its own shock, and on coarser cells it is the smearing of the scheme that holds shock and flame together. "
             "OpenFOAM was not run on such grids")
    not_like = ["One GPU (RTX 4070) against 8 CPU cores (OpenFOAM's solution) or 1 core (its trial).",
                "In one dimension the GPU is almost idle: a step costs about 0.7 to 1.2 ms whatever the number of cells. The stacked tube shows the cost on a full GPU.",
                "gpucfd takes three flux evaluations per step at Courant number 0.4 with its own step size; OpenFOAM's solution takes a fixed step (Courant number about 0.4 of "
                "the wave) with five passes per step; its trial took 5e-9 s.",
                "Chemistry: gpucfd's explicit method at relative tolerance 1e-4 in mixed precision; OpenFOAM's RKF45 at 1e-6 in double (its trial: seulex at 1e-3).",
                "Flow scheme: gpucfd is a density-based solver with a Riemann solver and conserves energy to rounding; reactingFoam is pressure-based and loses 0.2 % of the "
                "heat released.",
                "Tuning: gpucfd's flow setting was fixed before the runs and its chemistry setting chosen with 12 runs; OpenFOAM's solution had its own studies (passes, "
                "time step, chemistry method)."]
    rec = dict(tool="gpucfd", case=HERE.name, date=dt.date.today().isoformat(), device=st["device"], startup_s=st["startup_s"], plain=plain,
               busy_cpu=bool(st["machine_before"]["solver_processes"] or st["like_machine_before"]["solver_processes"]), machine=dict(before=st["machine_before"], after=st["machine_after"]),
               answers=answers, runs=runs, flow=st["flow"], chemistry=st["chemistry"], reference=cj,
               requirements_met=dict(
                   on_openfoams_finest_grid_3200_cells=dict(speed=abs(fine["speed_dev"]) <= 0.02, attached=fine["gap_max"] <= 1e-3),
                   on_every_grid_with_a_detonation=dict(grids=[r["N"] for r in held], speed=all(abs(r["speed_dev"]) <= 0.02 for r in held), attached=all(r["gap_max"] <= 1e-3 for r in held)),
                   on_gpucfds_finest_grid=dict(N=max(r["N"] for r in d1), speed=False, attached=False, why="no detonation forms on 4000 cells and more"),
                   grids_without_a_detonation=[r["N"] for r in lost]),
               like_for_like=like, grids=d1, border=border, stronger_patch=dict(given=st["d4_given"], runs=d4, note="NOT the case's problem: plan.yaml, rung d4"),
               other_settings=d2, full_gpu=d3,
               cost=dict(one_row=dict(ms_per_step={r["N"]: 1e3 * r["gpu_s"] / r["steps"] for r in like + d1}),
                         full_gpu=dict(cells=stacked["cells"], steps=stacked["steps"], ns_per_cell_step=stacked["ns_per_cell_step"],
                                       flow_ns=1e9 * timed["chosen"]["times"]["flow_s"] / (stacked["cells"] * stacked["steps"]),
                                       chemistry_ns=1e9 * timed["chosen"]["times"]["chemistry_s"] / (stacked["cells"] * stacked["steps"]),
                                       chemistry_share=timed["chosen"]["times"]["chemistry_s"] / timed["chosen"]["gpu_s"],
                                       chemistry_ns_bs23_default=1e9 * timed["bs23"]["times"]["chemistry_s"] / (stacked["cells"] * stacked["steps"]))),
               openfoam=dict(solution={n: {k: v[k] for k in ("N", "cores", "wall_time", "us_per_cell_step", "speed_err", "gap_max", "steps", "dt")} for n, v in of.items()
                                       if n in ("N400", "N800", "N1600", "N3200")},
                             finest=dict(speed=json.loads((RES / "runs" / "N3200.json").read_text(encoding="utf-8"))["speed"]["D"], wall_s=o["wall_time"], cores=o["cores"]),
                             trial=TRIAL, ratio_wall_time_3200_cells=ratio, not_like_for_like=not_like))
    OUT.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    print(f"3200 cells: gpucfd {fine['speed']:.1f} m/s ({100 * fine['speed_dev']:+.2f} %), gap {1e3 * fine['gap_max']:.3f} mm, highest pressure {fine['p_peak']['mean'] / 1e6:.3f} MPa "
          f"= {fine['peak_ratio']:.2f} of the shock pressure {fine['p_shock_of_this_speed'] / 1e6:.3f} MPa, energy {fine['energy_of_heat']:+.1e} of the heat released "
          f"({fine['heat_released_J_per_m2']:.4g} J per square metre of the tube's cross-section), {fine['wall_s']:.1f} s; OpenFOAM {100 * o['speed_err']:+.2f} %, gap {1e3 * o['gap_max']:.3f} mm, "
          f"{o['wall_time']:.0f} s on {o['cores']} cores: {ratio:.0f} times the wall time", flush=True)
    for r in like + d1 + border:
        print(f"  {r['N']:6d} cells: " + (f"{100 * r['speed_dev']:+.2f} %, gap {1e3 * r['gap_max']:.3f} mm, peak {r['p_peak']['mean'] / 1e6:.2f} MPa = {r['peak_ratio']:.2f} of the shock's, "
                                         f"{1e3 * r['gpu_s'] / r['steps']:.2f} ms per step" if r["detonation"] else f"no detonation; shock at {r['shock']['speed']:.0f} m/s"), flush=True)
    for r in d4:
        print(f"  stronger patch, {r['N']:6d} cells: {r['speed']:.1f} m/s ({100 * r['speed_dev']:+.2f} %), gap {1e3 * r['gap_max']:.3f} mm, peak {r['p_peak']['mean'] / 1e6:.2f} MPa "
              f"({r['p_peak']['low'] / 1e6:.2f} to {r['p_peak']['high'] / 1e6:.2f}) = {r['peak_ratio']:.2f} of the shock's {r['p_shock_of_this_speed'] / 1e6:.2f} MPa", flush=True)
    c = rec["cost"]["full_gpu"]
    print(f"full GPU: {c['ns_per_cell_step']:.1f} ns per cell and step; flow {c['flow_ns']:.1f}, chemistry {c['chemistry_ns']:.1f} ({100 * c['chemistry_share']:.0f} %), "
          f"chemistry with the kernel's default {c['chemistry_ns_bs23_default']:.1f}", flush=True)
    print(f"wrote {OUT.name}", flush=True)


if __name__ == "__main__":
    record()
