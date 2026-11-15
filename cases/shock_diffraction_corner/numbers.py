"""Every number of the analysis, computed from the saved runs (results/shock_diffraction_corner/runs).

  python -m cases.shock_diffraction_corner.numbers      prints them (used to size the acceptance rules
                                                        from the pre-flight runs, before the production runs)
analyze.py builds the report from compute().
"""
import json
from pathlib import Path

import numpy as np
import yaml

from cfdlib import verification
from . import measure, theory
from .run import OUT, RUNS, specs, steps_of

HERE = Path(__file__).resolve().parent


def load_run(name):
    f = RUNS / f"{name}.json"
    if not f.exists():
        return None, None
    meta = json.loads(f.read_text())
    return meta, (None if meta.get("crashed") else np.load(RUNS / f"{name}.npz"))


def one_run(meta, d, fl, numerics):
    """The outputs of one finished run, read by the fixed rules."""
    ext, N, h = numerics["extraction"], meta["N"], meta["h"]
    flat, wall = meta["flat"], meta["wall"]
    t = np.array([f["t"] for f in flat])
    late = t >= ext["speed_from"] * fl.t_end - 1e-12
    xm = np.array([f.get("x_mean", np.nan) for f in flat])
    speed = float(np.polyfit(t[late], xm[late], 1)[0])
    f_end = flat[-1]
    dw = np.array([w["d"] for w in wall])
    out = dict(
        N=N, h=h, cores=meta["cores"], steps=meta["steps"], wall_time=meta["wall_time"],
        speed=speed, speed_err=speed / fl.W - 1, speed_pt=f_end["x_mean"] / fl.t_end, speed_pt_err=f_end["x_mean"] / (fl.W * fl.t_end) - 1,
        x_top=f_end["x_top"], x_top_err_cells=(f_end["x_top"] - fl.x_top) / h, flat_rows=f_end["rows"],
        dev_cells_end=f_end["dev_mean_cells"], dev_line_end=f_end["dev_line_cells"],
        dev_cells_max=float(max(f["dev_mean_cells"] for f in flat if f.get("ok"))), dev_times=int(sum(1 for f in flat if f.get("ok"))),
        state={k: f_end[k] for k in ("p", "rho", "u")}, v_max=f_end["v_max"], p_row_spread=f_end.get("p_row_spread"),
        region=(f_end["region_rows"], f_end["region_cols"]),
        state_all=float(max(abs(f[k]["err"]) for f in flat if "p" in f for k in ("p", "rho", "u"))),
        ripple_p_all=float(max(f["p"]["ripple"] for f in flat if "p" in f)),
        d_wall=float(dw[-1]), wall_speed=float(np.polyfit(t[late], dw[late], 1)[0]), wall_peak=wall[-1]["peak"],
        Mw_dist=float(dw[-1] / (fl.t_end * fl.c0)),
        balance=meta["health"]["balance"], lows=meta["lows"], far=meta["far"], health=meta["health"]["status"],
        co_max=float(next(c["value"].split()[0] for c in meta["health"]["checks"] if c["name"] == "Max Courant number")),
        us_per_cell_step=meta["perf"]["us_per_cell_step"], cpu_efficiency=meta["perf"].get("cpu_efficiency"),
        peak_mem_mb=meta["perf"].get("peak_mem_mb"))
    out["Mw_peak"] = float(np.sqrt(((fl.gamma + 1) * out["wall_peak"] / fl.p0 + (fl.gamma - 1)) / (2 * fl.gamma)))
    # balance at the written times only (the request), next to the per-step one
    n_snap = numerics["snapshots"]
    k = (np.arange(1, n_snap + 1) * (len(d["c_t"]) // n_snap)) - 1
    m_ex, e_ex = fl.totals(d["c_t"][k])
    out["written"] = dict(t=d["c_t"][k].tolist(), mass=d["c_mass"][k].tolist(), energy=d["c_energy"][k].tolist(),
                          mass_err=((d["c_mass"][k] - m_ex) / (fl.mdot * d["c_t"][k])).tolist(),
                          energy_err=((d["c_energy"][k] - e_ex) / (fl.edot * d["c_t"][k])).tolist(),
                          mass_exact=np.asarray(m_ex).tolist(), energy_exact=np.asarray(e_ex).tolist())
    # the same picture at two times: along the step face (cells counted down from the corner) and along y = H_s
    i_half, i_end = n_snap // 2 - 1, n_snap - 1
    wr = d["wall_rho"][:, ::-1]
    a, b = wr[i_half][:N // 4], measure.restrict1(wr[i_end])
    out["sim_wall"] = measure.mean_rel_diff(a, b)
    a2, b2 = d["line_rho"][i_half][:N // 2], measure.restrict1(d["line_rho"][i_end])
    out["sim_line"] = measure.mean_rel_diff(a2, b2)
    out["sim_wall_max"] = float(np.max(np.abs(a - b)) / np.mean(b))
    out["sim_line_max"] = float(np.max(np.abs(a2 - b2)) / np.mean(b2))
    out["extra"] = extras(d, fl, N, ext)
    return out


def extras(d, fl, N, ext):
    """Numbers read from the fields at t_end that describe the pattern (none is a requirement)."""
    from scipy.ndimage import map_coordinates
    h = fl.L / N
    rho, p, u, v = (d[k].astype(float) for k in ("map_rho", "map_p", "map_u", "map_v"))
    c = np.sqrt(fl.gamma * p / rho)
    mach = np.hypot(u, v) / c
    half = N // 2
    out = dict(inlet_mach_min=float((u[half:, 0] / c[half:, 0]).min()), mach_max=float(mach.max()), p_min=float(p.min()), rho_min=float(rho.min()))
    j, i = np.unravel_index(np.argmin(p), p.shape)
    out["p_min_at"] = ((i + 0.5) * h, (j + 0.5) * h)
    # Ripples inside the expansion fan: along an arc round the corner the density of a smooth fan only falls.
    # The arc runs from 45 degrees above the duct's direction down to where the density is lowest; the measure
    # is the up-and-down beyond that single fall, as a fraction of the fall.
    R = 0.15 * fl.Hs / 0.5 * (fl.t_end / 0.00045)                  # 0.15 m at the time of interest
    ang = np.radians(45) - np.arange(0, int(np.radians(135) * R / (0.5 * h))) * (0.5 * h / R)
    xs, ys = R * np.cos(ang), fl.Hs + R * np.sin(ang)
    prof = map_coordinates(rho, [ys / h - 0.5, xs / h - 0.5], order=1, mode="nearest")
    k = int(np.argmin(prof))
    fall = prof[0] - prof[k]
    out.update(fan_radius=R, fan_wiggle=float((np.sum(np.abs(np.diff(prof[:k + 1]))) - fall) / fall),
               fan_angle_min=float(np.degrees(ang[k])), fan_rho_min=float(prof[k]), fan_arc=[np.degrees(ang).tolist(), prof.tolist()])
    # The border between duct air and chamber air, where it crosses the row y = 0.6 H: the steepest change of
    # density between the middle of the chamber and the bent shock, and the jumps of density and pressure across it.
    jr = int(0.6 * N * fl.H / fl.H)
    front = measure.front_x(p[jr], ext["low_level"] * fl.p0, h)[0]
    i0, i1 = int(0.35 * N), int(front / h) - ext["margin_cells"]
    g = np.abs(np.diff(rho[jr, i0:i1]))
    ic = i0 + int(np.argmax(g))
    w = 4
    out["contact"] = dict(x=(ic + 1) * h, y=(jr + 0.5) * h, rho_ratio=float(rho[jr, ic + 1 + w] / rho[jr, ic - w]),
                          p_ratio=float(p[jr, ic + 1 + w] / p[jr, ic - w]), front=float(front), x_duct_air=fl.u2 * fl.t_end)
    return out


def pair(fine, coarse, d_f, d_c, fl):
    """Two grids of ratio 2: (1) the fine grid at t_end/2 against the coarse grid at t_end, cell for cell (the
    method has no length scale, so they should agree to round-off); (2) the change of the density field at
    t_end from the coarse grid to the fine one (fine averaged over 2 x 2 cells)."""
    out = dict(fine=fine, coarse=coarse)
    if "rho_end" in d_c.files:
        a, b = d_f["rho_half_w"], d_c["rho_end"]
        out["discrete"] = float(np.max(np.abs(a - b)) / fl.rho0)
        out["discrete_mean"] = float(np.mean(np.abs(a - b)) / fl.rho0)
        c = d_f["rho_end_c"]
        out["field_change"] = measure.mean_rel_diff(c, b)
        out["field_change_max"] = float(np.max(np.abs(c - b)) / fl.rho0)
    return out


def compute():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    th, fl = theory.evaluate(problem["given"])
    sp = specs(numerics)
    runs, data, crashed = {}, {}, {}
    for name in sp:
        meta, d = load_run(name)
        if meta is None:
            continue
        if meta.get("crashed"):
            crashed[name] = meta
            continue
        runs[name], data[name] = one_run(meta, d, fl, numerics), d
        runs[name].update(name=name, spc=sp[name]["spc"], flux=sp[name]["flux"], limiter=sp[name]["limiter"],
                          co_x=fl.s_max * meta["dt"] / meta["h"])
    fam = [f"N{N}" for N in numerics["runs"]["family"] if f"N{N}" in runs]
    pairs = [pair(f, c, data[f], data[c], fl) for c, f in zip(fam[:-1], fam[1:])]
    # how the answers change with the grid: observed order of the level-to-level changes
    conv = {}
    if len(fam) >= 3:
        h = [runs[n]["h"] for n in fam]
        for key in ("d_wall", "x_top", "speed"):
            vals = [runs[n][key] for n in fam]
            conv[key] = verification.level_sequence(vals) if len(vals) >= 3 else None
        ch = [p["field_change"] for p in pairs]
        conv["field_orders"] = [float(np.log2(ch[i] / ch[i + 1])) for i in range(len(ch) - 1)]
        for key in ("sim_wall", "sim_line"):
            v = [runs[n][key] for n in fam]
            conv[key + "_orders"] = [float(np.log2(v[i] / v[i + 1])) for i in range(len(v) - 1)]
    # other time steps and schemes against the main run on the same grid, whole density field at t_end
    variants = {}
    for name, r in runs.items():
        base = f"N{r['N']}"
        if name != base and base in data and r["N"] <= 400:
            variants[name] = dict(field_vs_main=measure.mean_rel_diff(data[name]["rho_end"], data[base]["rho_end"]),
                                  d_wall_vs_main=r["d_wall"] / runs[base]["d_wall"] - 1)
    bias = {N: measure.rule_bias(fl, numerics["extraction"], N) for N in numerics["runs"]["family"]}
    return dict(problem=problem, numerics=numerics, theory=th, fl=fl, specs=sp, runs=runs, data=data, crashed=crashed,
                family=fam, pairs=pairs, conv=conv, variants=variants, bias=bias)


if __name__ == "__main__":
    c = compute()
    fl = c["fl"]
    for name, r in c["runs"].items():
        s = r["state"]
        print(f"{name}: Co_x {r['co_x']:.2f} (solver's max {r['co_max']:.3f}), {r['steps']} steps, {r['wall_time']:.1f} s on {r['cores']} core(s), "
              f"{r['us_per_cell_step']:.3f} us/cell/step\n"
              f"   flat shock: speed {r['speed']:.2f} ({r['speed_err']:+.2e}; x/t {r['speed_pt_err']:+.2e}), top wall {r['x_top']:.5f} m "
              f"({r['x_top_err_cells']:+.3f} cells), {r['flat_rows']} rows, deviation {r['dev_cells_end']:.2e} cells at t_end, "
              f"{r['dev_cells_max']:.2e} over {r['dev_times']} written times\n"
              f"   state behind ({r['region'][0]} x {r['region'][1]} cells): p {s['p']['err']:+.2e} [{s['p']['lo']:+.2e}, {s['p']['hi']:+.2e}], "
              f"rho {s['rho']['err']:+.2e} [{s['rho']['lo']:+.2e}, {s['rho']['hi']:+.2e}], u {s['u']['err']:+.2e} [{s['u']['lo']:+.2e}, "
              f"{s['u']['hi']:+.2e}]; worst mean over the written times {r['state_all']:.2e}; ripple p {s['p']['ripple']:.2e} "
              f"(largest over times {r['ripple_p_all']:.2e}); |v|/u2 {r['v_max']:.1e}; p along the shock {r['p_row_spread']:.1e}\n"
              f"   wall shock: d {r['d_wall']:.5f} m, speed from fit {r['wall_speed']:.1f} m/s, Mach from distance {r['Mw_dist']:.3f}, "
              f"from the level behind {r['Mw_peak']:.3f} (Whitham {fl.Mw:.3f})\n"
              f"   balance per step {r['balance']}; lows {r['lows']}; far walls {r['far']['floor']:.1e}, {r['far']['right']:.1e}\n"
              f"   two times: step face {r['sim_wall']:.3e} (max {r['sim_wall_max']:.2e}), line y = Hs {r['sim_line']:.3e} (max {r['sim_line_max']:.2e})")
        e = r["extra"]
        print(f"   pattern: inlet Mach min {e['inlet_mach_min']:.4f}, Mach max {e['mach_max']:.2f}, p min {e['p_min']:.0f} Pa at "
              f"({e['p_min_at'][0]:.3f}, {e['p_min_at'][1]:.3f}); fan ripple {e['fan_wiggle']:.3e} (arc to {e['fan_angle_min']:.0f} deg, "
              f"rho min {e['fan_rho_min']:.3f}); contact {e['contact']}")
    for name, m in c["crashed"].items():
        print(f"{name}: stopped by the solver at step {m['steps_done']} of {m['steps']}, t = {m['t_stop']:.3e} s, solver's Courant {m['co_max']:.3f}")
    for p in c["pairs"]:
        print("pair", p)
    print("convergence", c["conv"])
    print("variants", c["variants"])
    print("rule bias (cells)", c["bias"])
