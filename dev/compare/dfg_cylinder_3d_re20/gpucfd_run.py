"""Case 5 with gpucfd: the 3D cylinder in a square channel at Re 20 (DFG 3D-1Z), drag and lift as the case
defines them. The plan (grids, time step, stop rule, pre-flight, budget) is plan.yaml, written before any run.

    PYTHONPATH=. python dev/compare/dfg_cylinder_3d_re20/gpucfd_run.py preflight    # the short run of the plan
    PYTHONPATH=. python dev/compare/dfg_cylinder_3d_re20/gpucfd_run.py run          # the grids, coarse to fine
    PYTHONPATH=. python dev/compare/dfg_cylinder_3d_re20/gpucfd_run.py record       # gpucfd.json from the runs

Writes preflight.json, runs/<level>.json (after each grid, so a later failure keeps the earlier grids) and
gpucfd.json: one value per requirement, the run times, the checks.
"""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cfdlib import foam, verification as V
from cfdlib.gpu import _gpu_memory_mib
from cfdlib.report import m, tex_num as tn
from gpucfd import bodies as B
from gpucfd import gcfd, grids, steady

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CASE = ROOT / "cases" / HERE.name
RUNS = HERE / "runs"
PLAN = yaml.safe_load((HERE / "plan.yaml").read_text(encoding="utf-8"))
G = {k: v["value"] for k, v in yaml.safe_load((CASE / "problem.yaml").read_text(encoding="utf-8"))["given"].items()}
UBAR = 4.0 * G["Um"] / 9.0                                         # mean of the inflow profile over the inlet plane
BOX = ((G["xc"] - G["D"], G["xc"] + G["D"]), (G["yc"] - G["D"], G["yc"] + G["D"]), (0.0, G["H"]))   # wall to wall


def inflow(X, Y, Z):
    H = G["H"]
    return (16 * G["Um"] * Y * Z * (H - Y) * (H - Z) / H ** 4, 0 * X, 0 * X)


def make(level):
    n, gr, H = PLAN["grids"]["levels"][level], PLAN["grids"], G["H"]
    h = H / n[1]
    band = grids.body_band(G["xc"], G["D"], gr["band_x"]["ahead"], gr["band_x"]["behind"], length=G["L"])
    gx = grids.clustered(n[0], G["L"], band, h, tuple(gr["grow_x"]), cap_max=gr["far_field_cap"])
    bnd = {"x-": ("inflow", inflow), "x+": "outflow", "y-": "wall", "y+": "wall", "z-": "wall", "z+": "wall"}
    s = gcfd.NavierStokes(n, (G["L"], H, H), G["nu"], boundaries=bnd, bodies=[B.Cylinder((G["xc"], G["yc"]), G["D"] / 2)],
                          grid=(gx, None, None), scheme=PLAN["scheme"])
    return s, h, grids.describe(gx)


def from_rest(s):
    q = []
    for c in range(3):
        X, Y, Z = s.face_coords(c)
        q.append(np.where(s.fluid[c], inflow(X, Y, Z)[c], 0.0))
    s.set(q)
    return s.project()


def quantities(s):
    F = s.forces(BOX)                                              # per unit density; the z component is not computed
    k = 2.0 / (UBAR ** 2 * G["D"] * G["H"])
    return dict(cd=k * F[0], cl=k * F[1])


def step_of(h):
    ts = PLAN["time_step"]
    return ts["cfl"] * h / (ts["speed_factor"] * G["Um"])


def preflight():
    pf, mem0 = PLAN["preflight"], _gpu_memory_mib()
    t0 = time.perf_counter()
    s, h, desc = make("G1")
    build = time.perf_counter() - t0
    div0 = from_rest(s)
    mem1 = _gpu_memory_mib()
    step = step_of(h)
    st = s.run(step, 200, PLAN["pressure_tolerance"])
    q = quantities(s)
    rate = s.courant_rate()
    cells = {k: int(np.prod(v)) for k, v in PLAN["grids"]["levels"].items()}
    per_cell_step = st["ms"] / 200 / cells["G1"] * 1e6            # ns of GPU time per cell and step
    per_mib = (mem1 - mem0) / cells["G1"]
    # the cost of the plan's runs at this price: G1 for its minimum time, the finer grids for 10 s of flow time
    # (the 2D version of this flow needed 6 to 12 on its sequenced grids)
    proj = {k: (37.5 if k == "G1" else 10.0) / step_of(G["H"] / PLAN["grids"]["levels"][k][1]) * per_cell_step * cells[k] * 1e-9
            for k in cells}
    go = pf["go_on_if"]
    out = dict(grid=desc, n=PLAN["grids"]["levels"]["G1"], cells=cells, build_s=build, start_divergence=div0, dt=step,
               ms_per_step=st["ms"] / 200, ns_per_cell_step=per_cell_step, poisson_its=st["poisson_its"], helm_its=st["helm_its"],
               max_div=st["max_div"], courant=rate * step, after_200_steps=q, memory_mib=dict(before=mem0, with_G1=mem1),
               memory_G3_gb=per_mib * cells["G3"] / 1024, projected_gpu_s=proj, projected_total_gpu_s=sum(proj.values()),
               busy_cpu=bool(foam.running()))
    out["go_on"] = dict(pressure_iterations=bool(st["poisson_its"] <= go["pressure_iterations_at_most"]),
                        memory=bool(out["memory_G3_gb"] <= go["memory_G3_at_most_gb"]),
                        time=bool(out["projected_total_gpu_s"] <= go["projected_total_gpu_s_at_most"]))
    s.close()
    (HERE / "preflight.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=1, default=float))
    print("grid along x:", desc)


def run(levels):
    stop, RUNS_ = PLAN["stop"], RUNS
    RUNS_.mkdir(exist_ok=True)
    done = [json.loads((RUNS_ / f"{k}.json").read_text()) for k in PLAN["grids"]["levels"] if (RUNS_ / f"{k}.json").exists()]
    spent, prev, res = sum(r["gpu_s"] for r in done), None, []
    for lev in levels:
        left = stop["total_gpu_s"] - spent
        cap = min(stop["caps_gpu_s"][lev], left)
        if cap < 60:
            print(f"[{lev}] not started: {left:.0f} s of the budget left")
            break
        mem0 = _gpu_memory_mib()
        s, h, desc = make(lev)
        div0 = from_rest(s) if prev is None else steady.prolong(prev, s)
        if prev is not None:
            prev.close()
        step, hs = step_of(h), [r["h"] for r in res] + [h]
        if not res:
            tol = lambda k, v: 1e-5 * abs(v)                                           # noqa: E731
        else:
            fs = {k: [r["values"][k] for r in res] for k in ("cd", "cl")}

            def tol(k, v, fs=fs, hs=hs):                           # 1 % of the grid's own error estimate
                f, p = fs[k] + [v], 2.0
                if len(f) >= 3:
                    p_obs = V.order_unequal(f[-3:], hs[-3:])
                    p = p_obs if np.isfinite(p_obs) and 0.5 < p_obs < 6 else 2.0
                return max(0.01 * abs(v - f[-2]) / ((hs[-2] / hs[-1]) ** p - 1), 1e-7 * abs(v))
        floor = None if not res else {k: v for k, v in res[-1]["rho"].items() if v < 0.999}
        out = steady.march(s, quantities, step, tol, chunk=stop["check_every"], gpu_max=cap, ptol=PLAN["pressure_tolerance"],
                           t_min=37.5 if not res else 0.0, rho_min=floor, tag=f"[{lev}] ")
        rate = s.courant_rate()
        out.update(level=lev, n=list(s.n), cells=int(np.prod(s.n)), h=h, D_h=G["D"] / h, dt=step, grid=desc, prolong_div=div0,
                   courant=rate * step, memory_mib=dict(before=mem0, with_grid=_gpu_memory_mib()),
                   q_in=s.flux("x-"), q_out=s.flux("x+"), busy_cpu=bool(foam.running()), gpu_cap_s=cap,
                   date=dt.datetime.now().isoformat(timespec="minutes"))
        (RUNS_ / f"{lev}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
        print(f"[{lev}] done: t {out['t']:.1f} steps {out['steps']} gpu {out['gpu_s']:.0f} s converged {out['converged']} "
              f"cd {out['values']['cd']:.5f} cl {out['values']['cl']:.6f} Courant {out['courant']:.2f} "
              f"flow in/out {out['q_in']:.6g}/{out['q_out']:.6g}", flush=True)
        spent += out["gpu_s"]
        res.append(out)
        prev = s
    if prev is not None:
        prev.close()


def record():
    """gpucfd.json: the values on the finest grid run, judged by the case's rule, next to OpenFOAM's."""
    runs = [json.loads((RUNS / f"{k}.json").read_text()) for k in PLAN["grids"]["levels"] if (RUNS / f"{k}.json").exists()]
    an = json.loads((ROOT / "results" / HERE.name / "analysis.json").read_text(encoding="utf-8"))
    ref, of = an["ref"], an["comp"]
    fine, answers, judged = runs[-1], {}, {}
    for k, letter, digits in (("cd", "a", 4), ("cl", "b", 5)):     # decimals as in the case's report
        f, hs = [r["values"][k] for r in runs], [r["h"] for r in runs]
        st = V.three_grid_study(f, h=hs) if len(f) >= 3 else None
        u_sim = st["gci_abs"] if st else float("nan")
        band = u_sim + ref["unc"][k]
        diff = f[-1] - ref["val"][k]
        ratio = abs(diff) / band if st else float("nan")
        # The case's rule needs a band, and a band needs three grids that approach a limit. Where they do not
        # (gci_reliable False), there is no band and the value counts as a miss: never "inside" a made-up band.
        valid = bool(st and st["gci_reliable"])
        judged[k] = dict(values=f, value=f[-1], reference=ref["val"][k], u_ref=ref["unc"][k], diff=diff, rel=diff / abs(ref["val"][k]),
                         study=st, band=band if valid else None, band_rel=band / abs(ref["val"][k]) if valid else None,
                         band_ratio=ratio if valid else None,
                         status=("not converged" if not valid else "ok" if ratio <= 1 else "warn" if ratio <= 2 else "fail"),
                         iteration_error=fine["errors"][k], noise_limited=fine["noise_limited"][k],
                         openfoam=dict(value=of[k]["sim"], rel=of[k]["rel"], band_ratio=abs(of[k]["diff"]) / of[k]["band"]))
        mark = {"ok": "✓", "warn": "⚠", "fail": "✗"}.get(judged[k]["status"], "")              # as the case's report marks OpenFOAM's
        pct = 100 * diff / abs(ref["val"][k])
        answers[letter] = dict(value=diff / abs(ref["val"][k]), diff=m(f"{diff:+.{digits}f}") + " (" + m(tn(pct, 2 if abs(pct) < 100 else 3) + r"\,\%")
                               + ")<br>" + (f"{ratio:.2f} band {mark}, a wide band: " + m(r"\pm " + tn(100 * band / abs(ref["val"][k]), 2) + r"\,\%")
                                            if valid else "no band ✗: its grids do not converge"))
    within = abs(judged["cd"]["rel"]) <= 0.01
    # What a run as accurate as OpenFOAM's would need in this grid family: the order from the last two grids'
    # distance to the reference, the cells per diameter that brings the drag to OpenFOAM's distance, and its memory.
    e, d_h = [abs(r["values"]["cd"] - ref["val"]["cd"]) for r in runs], [r["D_h"] for r in runs]
    p_ref = float(np.log(e[-2] / e[-1]) / np.log(d_h[-1] / d_h[-2]))
    need = d_h[-1] * (e[-1] / abs(of["cd"]["diff"])) ** (1.0 / p_ref)
    idle = runs[0]["memory_mib"]["before"]
    gb_per_cell = (fine["memory_mib"]["with_grid"] - idle) / 1024.0 / fine["cells"]
    cells_need = fine["cells"] * (need / d_h[-1]) ** 3
    to_match = dict(order_from_reference=p_ref, cells_per_diameter=need, cells=cells_need, memory_gb=cells_need * gb_per_cell,
                    memory_gb_finest_run=fine["cells"] * gb_per_cell, openfoam_cells=an_cells(),
                    cells_ratio_finest=fine["cells"] / an_cells())
    label = lambda r: f"gpucfd {r['level']}, {r['D_h']:.0f} cells per diameter"                 # noqa: E731
    plain = (f"gpucfd on its finest grid ({fine['cells']:,} cells, {fine['D_h']:.0f} across the cylinder's diameter): the drag is "
             + m(tn(100 * judged["cd"]["rel"], 2) + r"\,\%") + " from the reference; the lift does not converge on its grids "
             f"({', '.join(f'{v:.4f}' for v in judged['cl']['values'])}).")
    rec = dict(tool="gpucfd", case=HERE.name, date=dt.date.today().isoformat(), busy_cpu=any(r["busy_cpu"] for r in runs), plain=plain,
               answers=answers, judged=judged, drag_within_1_percent=bool(within), to_match_openfoam_drag=to_match,
               runs=[dict(label=label(r), level=r["level"], cells=r["cells"], D_h=r["D_h"], wall_s=r["wall_s"], gpu_s=r["gpu_s"],
                          steps=r["steps"], flow_time=r["t"], converged=r["converged"], courant=r["courant"],
                          cd=r["values"]["cd"], cl=r["values"]["cl"]) for r in runs],
               total_gpu_s=sum(r["gpu_s"] for r in runs), preflight=json.loads((HERE / "preflight.json").read_text()))
    (HERE / "gpucfd.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    for k, j in judged.items():
        print(f"{k}: " + ", ".join(f"{v:.6g}" for v in j["values"]) + f" | reference {j['reference']:.6g} | finest {100 * j['rel']:+.3f} % | "
              f"band {j['band_rel']} ratio {j['band_ratio']} ({j['status']}) | OpenFOAM {100 * j['openfoam']['rel']:+.3f} %, band ratio {j['openfoam']['band_ratio']:.2f}"
              + (f" | order {j['study']['p']:.2f}, extrapolated {j['study']['extrap']:.6g}, GCI reliable {j['study']['gci_reliable']}" if j["study"] else ""))
    print(f"drag within 1 %: {within}; total GPU time {rec['total_gpu_s']:.0f} s")
    print("to match OpenFOAM's drag in this grid family:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in to_match.items()})


def an_cells():
    """Cells of OpenFOAM's finest grid, from the case's cost table."""
    t = json.loads((ROOT / "results" / HERE.name / "results.json").read_text(encoding="utf-8"))["performance"]
    return int(t["rows"][-1][1].replace(",", ""))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "preflight"
    if mode == "preflight":
        preflight()
    elif mode == "run":
        run(sys.argv[2:] or list(PLAN["grids"]["levels"]))
    else:
        record()
