"""Case 2 with gpucfd: the lid-driven cavity at Re 100 on OpenFOAM's four grids, with the four outputs the
case asks for, formed as in cases/lid_driven_cavity_re100/analyze.py against the same benchmark tables.
The plan (grids, time step, stop rule, what counts as agreement) is plan.yaml, written before the runs.

    PYTHONPATH=. python dev/compare/lid_driven_cavity_re100/gpucfd_run.py            # all runs of the plan

Writes gpucfd.json next to this file: one value per requirement, the run times per grid, the checks.
"""
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from cases.lid_driven_cavity_re100.analyze import _vortex, study
from cfdlib import foam
from cfdlib.report import m, tex_num as tn
from gpucfd import gcfd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CASE = ROOT / "cases" / HERE.name
OUT = HERE / "gpucfd.json"
SCAL = ["u_min", "y_umin", "psi_min", "x_c", "y_c"]


def solve(N, L, U, nu, cfl, ptol, plan):
    """March from rest to the steady state. Returns the velocity (u on x-faces, v on y-faces) and the run record."""
    t0 = time.perf_counter()
    s = gcfd.NavierStokes((N, N, 1), (L, L, L), nu, walls=(True, True, False), wall_velocity={"y+": (U, 0, 0)},
                          scheme=plan["scheme"])
    unit = L / U                                                   # one time unit
    per_unit = int(math.ceil(unit / (cfl * (L / N) / U)))          # steps per time unit at this cfl
    old = [np.zeros(s.shape), np.zeros(s.shape)]
    t, gpu, change, pits = 0, 0.0, math.inf, 0.0
    while t < plan["stop"]["cap_time_units"] and change > plan["stop"]["settle"]:
        st = s.run(unit / per_unit, per_unit, ptol)
        gpu += st["ms"] / 1e3
        pits += st["poisson_its"]
        t += 1
        new = s.get()
        change = max(float(np.max(np.abs(a - b))) for a, b in zip(new, old)) / U
        old = new
    s.close()
    return old[0][0], old[1][0], dict(N=N, cells=N * N, cfl=cfl, ptol=ptol, time_units=t, steps=t * per_unit,
                                      last_change=change, converged=bool(change <= plan["stop"]["settle"]),
                                      max_div=st["max_div"], poisson_its=pits / t, gpu_s=gpu,
                                      wall_s=time.perf_counter() - t0)


def extract(N, u, v, L, U, y_g, x_g):
    """The case's outputs from gpucfd's staggered fields, dimensionless (x / L, u / U, psi / (U L)).
    u[j, i] sits on the x-face i (x = i h) at the cell-centre height, v[j, i] on the y-face j."""
    u, v = u / U, v / U
    c = (np.arange(N) + 0.5) / N
    u_vc, v_hc = u[:, N // 2], v[N // 2, :]                        # the centre lines are rows of stored values
    y_line, u_line = np.r_[0.0, c, 1.0], np.r_[0.0, u_vc, 1.0]     # wall values from the boundary conditions
    x_line, v_line = np.r_[0.0, c, 1.0], np.r_[0.0, v_hc, 0.0]
    su, sv = CubicSpline(y_line, u_line), CubicSpline(x_line, v_line)
    k = int(np.argmin(u_vc))
    r = minimize_scalar(su, bounds=(c[max(k - 1, 0)], c[min(k + 1, N - 1)]), method="bounded", options=dict(xatol=1e-10))
    psi = np.zeros((N + 1, N + 1))                                 # at the grid corners, from the face fluxes
    psi[1:, :N] = np.cumsum(u, axis=0) / N
    node = np.arange(N + 1) / N
    vort = _vortex(psi, node, node)
    return dict(N=N, u_at_ghia=su(y_g).tolist(), v_at_ghia=sv(x_g).tolist(), u_min=float(r.fun), y_umin=float(r.x),
                psi_min=vort["psi"], x_c=vort["x"], y_c=vort["y"], net_flow=float(psi[N, N // 2]),
                lid_psi=float(np.max(np.abs(psi[-1]))))


def bands(rows, key):
    """Per-point grid-error band of a profile on the finest grid, as analyze.py forms it."""
    chg = [float(np.sqrt(np.mean((np.array(a[key]) - np.array(b[key])) ** 2))) for a, b in zip(rows[:-1], rows[1:])]
    p = math.log(chg[-2] / chg[-1]) / math.log(2)
    f2, f3 = np.array(rows[-2][key]), np.array(rows[-1][key])
    return 1.25 * np.abs(f3 - f2) / (2 ** p - 1), p


def main():
    plan = yaml.safe_load((HERE / "plan.yaml").read_text(encoding="utf-8"))
    problem = yaml.safe_load((CASE / "problem.yaml").read_text(encoding="utf-8"))
    acc = yaml.safe_load((CASE / "numerics.yaml").read_text(encoding="utf-8"))["acceptance"]
    ref = yaml.safe_load((CASE / problem["reference"]["file"]).read_text(encoding="utf-8"))
    of = json.loads((ROOT / "results" / HERE.name / "results.json").read_text(encoding="utf-8"))["data"]
    L, U, nu = (problem["given"][k]["value"] for k in ("L", "U", "nu"))
    tu, tv = np.array(ref["u_vertical_centreline"]["rows"], float), np.array(ref["v_horizontal_centreline"]["rows"], float)
    iu, iv = (tu[:, 0] > 0) & (tu[:, 0] < 1), (tv[:, 0] > 0) & (tv[:, 0] < 1)
    y_g, u_g, x_g, v_g = tu[iu, 0], tu[iu, 1], tv[iv, 0], tv[iv, 1]
    G, h_G = of["ghia"], of["h_ghia"]

    busy = [p[3][:60] for p in foam.running()]
    t0 = time.perf_counter()
    name = gcfd.device()                                           # loads the library and starts the GPU
    startup = time.perf_counter() - t0
    print(f"GPU: {name}; start-up {startup:.2f} s; other solver processes on the CPU: {len(busy)}", flush=True)

    rows, runs = [], []
    for N in plan["grids"]["N"]:
        u, v, run = solve(N, L, U, nu, plan["time_step"]["cfl"], plan["pressure_tolerance"], plan)
        x = extract(N, u, v, L, U, y_g, x_g)
        du, dv = np.array(x["u_at_ghia"]) - u_g, np.array(x["v_at_ghia"]) - v_g
        x.update(du_max=float(np.abs(du).max()), du_rms=float(np.sqrt(np.mean(du ** 2))),
                 dv_max=float(np.abs(dv).max()), dv_rms=float(np.sqrt(np.mean(dv ** 2))),
                 d_umin=x["u_min"] - G["u_min"], d_yumin=x["y_umin"] - G["y_umin"],
                 d_psi_rel=(x["psi_min"] - G["psi_min"]) / abs(G["psi_min"]), d_xc=x["x_c"] - G["x_c"], d_yc=x["y_c"] - G["y_c"])
        rows.append(x)
        runs.append(run)
        print(f"N {N:4d}: {run['time_units']} time units, {run['steps']} steps, change {run['last_change']:.1e}, "
              f"gpu {run['gpu_s']:.1f} s, wall {run['wall_s']:.1f} s | du {x['du_max']:.5f} dv {x['dv_max']:.5f} "
              f"u_min {x['u_min']:.6f} psi_min {x['psi_min']:.6f} centre ({x['x_c']:.4f}, {x['y_c']:.4f})", flush=True)
    # ---- the check run: do the two speed settings change the answer? ---------------------------------
    ck = plan["check_run"]
    u, v, crun = solve(ck["N"], L, U, nu, ck["cfl"], ck["pressure_tolerance"], plan)
    cx = extract(ck["N"], u, v, L, U, y_g, x_g)
    base = rows[plan["grids"]["N"].index(ck["N"])]
    check = dict(N=ck["N"], cfl=ck["cfl"], ptol=ck["pressure_tolerance"], run=crun,
                 profiles=float(max(np.abs(np.array(cx[k]) - np.array(base[k])).max() for k in ("u_at_ghia", "v_at_ghia"))),
                 u_min=abs(cx["u_min"] - base["u_min"]), psi_min_rel=abs(cx["psi_min"] - base["psi_min"]) / abs(base["psi_min"]))
    check["ok"] = bool(check["profiles"] < 1e-5 and check["u_min"] < 1e-5 and check["psi_min_rel"] < 1e-5)
    print(f"check run (cfl {ck['cfl']}, ptol {ck['pressure_tolerance']:g}): profiles {check['profiles']:.1e}, "
          f"u_min {check['u_min']:.1e}, psi_min {check['psi_min_rel']:.1e} -> {'ok' if check['ok'] else 'NOT ok'}", flush=True)

    # ---- grid study of gpucfd's own outputs, and the case's acceptance rules ---------------------------
    fine = rows[-1]
    ms = {k: study([r[k] for r in rows]) for k in SCAL}
    band_u, p_u = bands(rows, "u_at_ghia")
    band_v, p_v = bands(rows, "v_at_ghia")
    loc_tol = acc["location_tol_ghia_cells"] * h_G
    fail, warn = [], []
    if not all(r["converged"] for r in runs):
        fail.append("a run reached the time cap before it settled")
    for lab, key in (("(a) u profile", "du_max"), ("(b) v profile", "dv_max")):
        if fine[key] > acc["profile_max_diff_warn"]:
            fail.append(f"{lab}: {fine[key]:.4f} U above {acc['profile_max_diff_warn']}")
        elif fine[key] > acc["profile_max_diff_pass"]:
            warn.append(f"{lab}: {fine[key]:.4f} U above {acc['profile_max_diff_pass']}")
    if abs(fine["d_umin"]) > acc["u_min_diff_pass"]:
        fail.append("(c): u_min differs from Ghia by more than the tolerance")
    if abs(fine["d_yumin"]) > loc_tol:
        fail.append("(c): place of u_min further than one benchmark spacing from Ghia")
    if abs(fine["d_psi_rel"]) > acc["psi_min_rel_pass"]:
        fail.append("(d): psi_min differs from Ghia by more than the tolerance")
    if abs(fine["d_xc"]) > loc_tol or abs(fine["d_yc"]) > loc_tol:
        fail.append("(d): vortex centre further than one benchmark spacing from Ghia")
    for k in SCAL:
        if ms[k]["gci"] > acc["gci_max"]:
            warn.append(f"GCI of {k} {100 * ms[k]['gci']:.3f} % above {100 * acc['gci_max']:g} %")
        if not ms[k]["monotonic"]:
            warn.append(f"{k}: not monotonic on the three finest grids")
        elif not acc["order_range"][0] <= ms[k]["p"] <= acc["order_range"][1]:
            warn.append(f"{k}: observed order {ms[k]['p']:.2f} outside {acc['order_range']}")
    if max(abs(r["net_flow"]) for r in rows) > acc["mass_residual_max"]:
        fail.append("net flow through the vertical centre line is not zero")
    verdict = "FAIL" if fail else "WARN" if warn else "PASS"

    # ---- the two tools on equal grids ------------------------------------------------------------------
    ofr = {r["N"]: r for r in of["rows"]}
    per_grid = []
    for r in rows:
        o = ofr[r["N"]]
        per_grid.append(dict(N=r["N"],
                             profiles=float(max(np.abs(np.array(r[k]) - np.array(o[k])).max() for k in ("u_at_ghia", "v_at_ghia"))),
                             u_min=abs(r["u_min"] - o["u_min"]), psi_min_rel=abs(r["psi_min"] - o["psi_min"]) / abs(o["psi_min"]),
                             centre=math.hypot(r["x_c"] - o["x_c"], r["y_c"] - o["y_c"])))
    o, om, op = ofr[fine["N"]], of["mesh_study"], of["profiles"]
    agree = {}
    for key, band, lab in (("u_at_ghia", band_u, "a"), ("v_at_ghia", band_v, "b")):
        d = np.abs(np.array(fine[key]) - np.array(o[key]))
        allow = band + np.array(op[key]["band"])
        worst = int(np.argmax(d / allow))                         # the point that uses most of its allowance
        agree[lab] = dict(difference=float(d[worst]), allowance=float(allow[worst]), largest_difference=float(d.max()),
                          ok=bool(np.all(d <= allow)))
    for key, lab in (("u_min", "c"), ("psi_min", "d")):
        d = abs(fine[key] - o[key])
        allow = ms[key]["gci_abs"] + om[key]["gci_abs"]
        agree[lab] = dict(difference=d, allowance=allow, ok=bool(d <= allow))
    rule = dict(a=(fine["du_max"], o["du_max"], acc["profile_max_diff_pass"]), b=(fine["dv_max"], o["dv_max"], acc["profile_max_diff_pass"]),
                c=(abs(fine["d_umin"]), abs(o["d_umin"]), acc["u_min_diff_pass"]),
                d=(abs(fine["d_psi_rel"]), abs(o["d_psi_rel"]), acc["psi_min_rel_pass"]))
    for lab, (g, f, lim) in rule.items():
        agree[lab].update(gpucfd=g, openfoam=f, limit=lim, same_side_of_the_rule=bool((g <= lim) == (f <= lim)))

    # ---- one cell per requirement, written as the case's report writes OpenFOAM's -----------------------
    pct = lambda x: m(rf"{tn(100 * x, 2)}\,\%")
    answers = dict(
        a=dict(value=fine["du_max"], diff=m(rf"\le {tn(fine['du_max'], 2)}\,U")),
        b=dict(value=fine["dv_max"], diff=m(rf"\le {tn(fine['dv_max'], 2)}\,U")),
        c=dict(value=fine["d_umin"], diff=m(rf"{tn(fine['d_umin'], 2)}\,U") + "<div class='sub'>position "
               + m(rf"{tn(fine['d_yumin'], 2)}\,L") + f" = {tn(fine['d_yumin'] / h_G, 2)} benchmark cells</div>"),
        d=dict(value=fine["d_psi_rel"], diff=pct(fine["d_psi_rel"]) + "<div class='sub'>centre "
               + m(rf"({tn(fine['d_xc'], 2)}, {tn(fine['d_yc'], 2)})\,L") + "</div>"))
    plain = (f"gpucfd on the same four grids: on the finest its centre-line velocities differ from OpenFOAM's by at most "
             + m(rf"{tn(per_grid[-1]['profiles'], 2)}\,U") + f", and it reaches the steady flow there in {runs[-1]['wall_s']:.0f} s.")
    rec = dict(tool="gpucfd", case=HERE.name, date=dt.date.today().isoformat(), device=name, startup_s=startup, plain=plain,
               busy_cpu=bool(busy), busy_processes=busy, answers=answers, runs=runs, grids=rows,
               mesh_study=ms, profile_order=dict(u=p_u, v=p_v), profile_band_max=dict(u=float(band_u.max()), v=float(band_v.max())),
               by_case_rules=dict(verdict=verdict, fail=fail, warn=warn), check_run=check,
               against_openfoam=dict(per_grid=per_grid, finest=agree, agree=bool(all(a["ok"] for a in agree.values()))))
    OUT.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"by the case's rules: {verdict}; fail {fail}; warn {warn}")
    print("orders: " + ", ".join(f"{k} p={ms[k]['p']:.2f} GCI={100 * ms[k]['gci']:.4f}%" for k in SCAL) + f"; profiles u {p_u:.2f} v {p_v:.2f}")
    for g in per_grid:
        print(f"gpucfd - OpenFOAM, N {g['N']:4d}: profiles {g['profiles']:.1e} U, u_min {g['u_min']:.1e} U, "
              f"psi_min {100 * g['psi_min_rel']:.4f} %, centre {g['centre']:.1e} L")
    for lab, a in agree.items():
        print(f"  ({lab}) difference {a['difference']:.2e}, allowed {a['allowance']:.2e}, "
              f"{'agree' if a['ok'] else 'DO NOT agree'}; vs benchmark: gpucfd {a['gpucfd']:.4g}, OpenFOAM {a['openfoam']:.4g}, limit {a['limit']}")
    print(f"record: {OUT}")


if __name__ == "__main__":
    main()
