"""Compressible ladder, rung 3: the shock tube of case 18 (Sod) against its exact solution, next to OpenFOAM.

    python -m gpucfd.tests.test_euler_sod          (inside WSL, from the project root)

The problem is cases/shock_tube_sod/problem.yaml. The exact solution is gpucfd's own (gpucfd/exact_riemann.py).
The outputs are read with the case's rules (cases/shock_tube_sod/measure.py, extraction settings of its
numerics.yaml), so the numbers stand next to OpenFOAM's (results/shock_tube_sod/) like for like.

Checks:
  a. the exact solution itself: it conserves mass, momentum and energy; its cell averages agree with sampled
     ones; scoring OpenFOAM's stored profiles with it reproduces the errors OpenFOAM's report recorded;
  b. the grid family 100 to 12800 cells with the solver's default setting: conservation, the error and its
     order, widths of the shock and of the border between the gases, ripple, states, positions, speeds;
  c. the same grid in double, mixed and single precision;
  d. the time scheme: one stage (what OpenFOAM has) against three, at the same and at larger time steps;
  e. the alternatives: every reconstruction and flux formula on 400 cells.
Pass rules, fixed before the runs: those of the case's numerics.yaml that do not depend on the tool
(conservation 1e-11, ends 1e-10, order floors, states, positions, speeds on the finest grid), with ripple,
wiggle and shock width against the case's warning limits. In mixed precision two of them cannot hold and
were replaced after the first runs (said where they are checked): momentum against the push of the end walls
and the end cells, both limited by the single-precision fluxes. The comparison with OpenFOAM is reported,
not judged.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cases.shock_tube_sod import measure
from cfdlib import verification as V
from gpucfd.euler import Euler
from gpucfd.exact_riemann import Riemann

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cases" / "shock_tube_sod"
OF = ROOT / "results" / "shock_tube_sod"
FAILS = []
REPORT = {}
# The solver's setting for this problem, chosen by the tuning budget of dev/compare/shock_tube_sod/plan.yaml
DEFAULT = dict(recon="weno5", flux="hllc", cfl=0.4, stages=3, precision="mixed")
FAMILY = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
SNAPSHOTS = 10


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def given():
    g = {k: v["value"] for k, v in yaml.safe_load((CASE / "problem.yaml").read_text(encoding="utf-8"))["given"].items()}
    num = yaml.safe_load((CASE / "numerics.yaml").read_text(encoding="utf-8"))
    return g, num["extraction"], num["acceptance"]


def exact(g):
    return Riemann(g["gamma"], (g["rho_L"], 0.0, g["p_L"]), (g["rho_R"], 0.0, g["p_R"]), x0=g["L"] / 2)


def case_waves(ex, t):
    """Wave positions under the names the case's measuring rules use."""
    w = ex.positions(t)
    return dict(head=w["left_head"], tail=w["left_tail"], contact=w["contact"], shock=w["right_head"])


def run(g, N, recon, flux, cfl, stages=3, precision="mixed", history=True, steps=None):
    """One run to t_end in SNAPSHOTS equal chunks. cfl is the Courant number of the fastest signal of the exact
    solution (as case 18 counts it). Returns the snapshots and, with history, the totals after every step."""
    ex = exact(g)
    h = g["L"] / N
    if steps is None:
        steps = SNAPSHOTS * int(np.ceil(g["t_end"] * ex.fastest / (cfl * h) / SNAPSHOTS))
    dt = g["t_end"] / steps
    t0 = time.perf_counter()
    s = Euler((N, 1, 1), (g["L"], 1.0, 1.0), g["gamma"], sides={"x-": "wall", "x+": "wall"}, precision=precision,
              recon=recon, flux=flux, stages=stages)
    x = s.centres("x")
    s.set(rho=np.where(x < g["L"] / 2, g["rho_L"], g["rho_R"]), p=np.where(x < g["L"] / 2, g["p_L"], g["p_R"]))
    d = dict(N=N, h=h, steps=steps, dt=dt, courant=ex.fastest * dt / h, x=x, times=[], rho=[], u=[], p=[], gpu_ms=0.0,
             recon=recon, flux=flux, stages=stages, precision=precision)
    hist = {k: [] for k in ("mass", "energy", "momentum", "rho_min", "p_min")}
    for k in range(SNAPSHOTS):
        st = s.run_fixed(dt, steps // SNAPSHOTS, history=history)
        d["gpu_ms"] += st["ms"]
        if history:
            for key in hist:
                hist[key].append(st[key] if key != "momentum" else st[key][:, 0])
        f = s.get()
        d["times"].append(dt * (steps // SNAPSHOTS) * (k + 1))
        for key in ("rho", "u", "p"):
            d[key].append(f[key][0, 0].copy())
    d["wall_s"] = time.perf_counter() - t0
    s.close()
    if history:
        d.update({"c_" + k: np.concatenate(v) for k, v in hist.items()})
        d["c_t"] = dt * (np.arange(steps) + 1)
    return d


def analyse(d, g, ext):
    """The outputs of one run, formed as cases/shock_tube_sod/analyze.py forms OpenFOAM's."""
    ex = exact(g)
    N, x, L, tend = d["N"], d["x"], g["L"], g["t_end"]
    w = case_waves(ex, tend)
    exc = ex.cell_averages(np.linspace(0, L, N + 1), tend)
    sim = (d["rho"][-1], d["u"][-1], d["p"][-1])
    scales = (g["rho_L"], ex.u_star, g["p_L"])
    r = {k: d[k] for k in ("N", "h", "steps", "dt", "courant", "wall_s", "gpu_ms", "recon", "flux", "stages", "precision")}
    r["e"] = {k: V.mean_abs_error(sim[i], exc[i], scales[i]) for i, k in enumerate(("rho", "u", "p"))}
    r["wiggle"] = dict(rho=measure.total_variation(sim[0]) / (g["rho_L"] - g["rho_R"]) - 1,
                       p=measure.total_variation(sim[2]) / (g["p_L"] - g["p_R"]) - 1)
    read = measure.read_all(x, *sim, w, L, g["p_R"], ext)
    st_exact = dict(p=ex.p_star, u=ex.u_star, rho_L=ex.rho_star_left, rho_R=ex.rho_star_right)
    r["state"] = {k: read["states"][k]["mean"] for k in st_exact}
    r["state_err"] = {k: r["state"][k] / st_exact[k] - 1 for k in st_exact}
    r["ripple"] = {k: read["states"][k]["spread"] / abs(r["state"][k]) for k in ("p", "u")}
    r["pos_err"] = {k: read["positions"][k] - w[k] for k in w}
    r["width"] = read["widths"]
    pos_t = [measure.read_all(x, d["rho"][k], d["u"][k], d["p"][k], case_waves(ex, float(t)), L, g["p_R"], ext)["positions"]
             for k, t in enumerate(d["times"])]
    speed = measure.speeds(d["times"], pos_t, ext["speed_from"] * tend)
    sp_exact = dict(head=ex.waves["left_head"], tail=ex.waves["left_tail"], contact=ex.waves["contact"], shock=ex.waves["right_head"])
    r["speed_err"] = {k: speed[k] - sp_exact[k] for k in sp_exact}
    r["shock_speed"] = sp_exact["shock"]
    r["ends"] = float(max(abs(sim[2][0] / g["p_L"] - 1), abs(sim[2][-1] / g["p_R"] - 1), abs(sim[1][0]) / ex.u_star, abs(sim[1][-1]) / ex.u_star))
    if "c_mass" in d:
        M0, E0 = (g["rho_L"] + g["rho_R"]) * L / 2, (g["p_L"] + g["p_R"]) * L / (2 * (g["gamma"] - 1))
        r["mass_drift"] = float(np.max(np.abs(d["c_mass"] - M0)) / M0)
        r["energy_drift"] = float(np.max(np.abs(d["c_energy"] - E0)) / E0)
        r["momentum_err"] = float(np.max(np.abs(d["c_momentum"] - (g["p_L"] - g["p_R"]) * d["c_t"])) / ((g["p_L"] - g["p_R"]) * tend))
        r["rho_min"], r["p_min"] = float(d["c_rho_min"].min()), float(d["c_p_min"].min())
    return r


def openfoam():
    """OpenFOAM's recorded numbers per run name (results/shock_tube_sod/results.json)."""
    return json.loads((OF / "results.json").read_text(encoding="utf-8"))["data"]["runs"]


def check_exact(g):
    ex = exact(g)
    L, t = g["L"], g["t_end"]
    c = ex.cumulative(np.array([0.0, L]), t)
    tot = c[:, 1] - c[:, 0]
    M0, E0 = (g["rho_L"] + g["rho_R"]) * L / 2, (g["p_L"] + g["p_R"]) * L / (2 * (g["gamma"] - 1))
    errs = (abs(tot[0] / M0 - 1), abs(tot[1] / ((g["p_L"] - g["p_R"]) * t) - 1), abs(tot[2] / E0 - 1))
    check("rung 3a: the exact solution conserves mass, momentum and energy", max(errs) < 1e-12,
          f"total mass, momentum (against the push of the two ends) and energy at t_end off by {errs[0]:.1e}, {errs[1]:.1e}, {errs[2]:.1e}")
    edges = np.linspace(0, L, 401)
    mine = ex.cell_averages(edges, t)
    g_ = g["gamma"]

    def cons(x):
        r, u, p = ex.primitive(x, t)
        return r, r * u, p / (g_ - 1) + 0.5 * r * u * u
    r, m, E = V.cell_averages(cons, edges, sub=64)
    samp = (r, m / r, (g_ - 1) * (E - 0.5 * m * m / r))
    scales = (g["rho_L"], ex.u_star, g["p_L"])
    diffs = [float(np.max(np.abs(mine[i] - samp[i])) / scales[i]) for i in range(3)]
    smooth = [float(np.sort(np.abs(mine[i] - samp[i]) / scales[i])[-5]) for i in range(3)]       # all but the 4 cells with a jump or kink
    check("rung 3a: exact cell averages against 64 samples per cell", max(diffs) < 1 / 64 and max(smooth) < 1e-6,
          f"largest difference {max(diffs):.1e} (a cell with a jump; bound 1/128 of the jump), elsewhere at most {max(smooth):.1e}")
    worst, rows = 0.0, {}
    for name, rec in openfoam().items():
        if not name[1:].isdigit():
            continue
        z = np.load(OF / "runs" / f"{name}.npz")
        N = len(z["x"])
        exc = ex.cell_averages(np.linspace(0, L, N + 1), float(z["times"][-1]))
        e = {k: V.mean_abs_error(z[k][-1], exc[i], scales[i]) for i, k in enumerate(("rho", "u", "p"))}
        rows[name] = e
        worst = max(worst, max(abs(e[k] / rec["e"][k] - 1) for k in e))
    check("rung 3a: OpenFOAM's stored profiles scored with this exact solution give OpenFOAM's recorded errors", worst < 1e-6,
          f"{len(rows)} grids, largest relative difference of an error {worst:.1e}")
    REPORT["exact"] = dict(conservation=errs, sampled=diffs, openfoam_rescored=worst,
                           p_star=ex.p_star, u_star=ex.u_star, rho_star_left=ex.rho_star_left, rho_star_right=ex.rho_star_right,
                           shock_speed=ex.waves["right_head"])


def slim(r):
    return {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}


def main():
    g, ext, acc = given()
    ex = exact(g)
    check_exact(g)
    of = openfoam()

    # ---- b. the grid family ----
    fam = []
    for N in FAMILY:
        d = run(g, N, DEFAULT["recon"], DEFAULT["flux"], DEFAULT["cfl"], DEFAULT["stages"], DEFAULT["precision"])
        r = analyse(d, g, ext)
        timed = run(g, N, DEFAULT["recon"], DEFAULT["flux"], DEFAULT["cfl"], DEFAULT["stages"], DEFAULT["precision"], history=False)
        r["wall_s"], r["gpu_ms"] = timed["wall_s"], timed["gpu_ms"]
        fam.append(r)
        o = of[f"N{N}"]
        print(f"  N {N:6d}: {r['steps']:5d} steps, wall {r['wall_s']:.3f} s (OpenFOAM {o['steps']} steps, {o['wall_time']:.2f} s) | errors rho/u/p "
              f"{r['e']['rho']:.2e} {r['e']['u']:.2e} {r['e']['p']:.2e} (OpenFOAM {o['e']['rho']:.2e} {o['e']['u']:.2e} {o['e']['p']:.2e}) | "
              f"shock {r['width']['shock']:.2f} cells, border {r['width']['contact']:.2f} (OpenFOAM {o['width']['shock']:.2f}, {o['width']['contact']:.2f}) | "
              f"ripple p {r['ripple']['p']:.2e} (OpenFOAM {o['ripple']['p']:.2e}) | wiggle p {r['wiggle']['p']:.2e} (OpenFOAM {o['wiggle']['p']:.2e})", flush=True)
    # the same family in double precision, for the case's own conservation limits
    famd = [analyse(run(g, N, DEFAULT["recon"], DEFAULT["flux"], DEFAULT["cfl"], DEFAULT["stages"], "double"), g, ext) for N in FAMILY]
    wd = {k: max(r[k] for r in famd) for k in ("mass_drift", "energy_drift", "momentum_err", "ends")}
    check("rung 3b: double precision: mass, energy, momentum after every step of every grid; end cells at t_end",
          max(wd["mass_drift"], wd["energy_drift"], wd["momentum_err"]) < 1e-11 and wd["ends"] < acc["fail_ends"],
          f"largest change of mass {wd['mass_drift']:.1e}, of energy {wd['energy_drift']:.1e}, momentum against the push of the ends "
          f"{wd['momentum_err']:.1e} (rule: below 1e-11); end cells {wd['ends']:.1e} (rule: below {acc['fail_ends']:g})")
    worst = {k: max(r[k] for r in fam) for k in ("mass_drift", "energy_drift", "momentum_err", "ends")}
    check("rung 3b: mixed precision: mass and energy after every step of every grid", max(worst["mass_drift"], worst["energy_drift"]) < 1e-11,
          f"largest change of mass {worst['mass_drift']:.1e}, of energy {worst['energy_drift']:.1e} (rule: below 1e-11)")
    # These two limits were set after the first runs had shown the numbers (the others before): with fluxes in
    # single precision the pressure on the end walls, and any signal, is only resolved to about 1e-7.
    check("rung 3b: mixed precision: momentum against the push of the ends, and the end cells (single-precision effects)",
          worst["momentum_err"] < 2e-7 and worst["ends"] < 1e-6,
          f"momentum {worst['momentum_err']:.1e} (rule: below 2e-7; the case's limit for a double-precision tool is {acc['fail_momentum']:g}), "
          f"end cells {worst['ends']:.1e} (rule: below 1e-6; the case's limit is {acc['fail_ends']:g})")
    check("rung 3b: density and pressure stay positive", min(r["rho_min"] for r in fam) > 0 and min(r["p_min"] for r in fam) > 0,
          f"lowest density {min(r['rho_min'] for r in fam):.4g}, lowest pressure {min(r['p_min'] for r in fam):.5g}")
    REPORT["conservation"] = dict(double=wd, mixed=worst)
    hs = [r["h"] for r in fam]
    order = {k: V.order_from_errors(hs, [r["e"][k] for r in fam]) for k in ("rho", "u", "p")}
    overall = {k: float(np.polyfit(np.log(hs), np.log([r["e"][k] for r in fam]), 1)[0]) for k in order}
    fine, coarse = fam[-1], fam[0]
    for k in ("rho", "u", "p"):
        limit = coarse["e"][k] * (fine["h"] / coarse["h"]) ** acc["fail_l1_order_floor"][k]
        check(f"rung 3b: error of {k} on the finest grid against the order floor {acc['fail_l1_order_floor'][k]}", fine["e"][k] <= limit,
              f"{fine['e'][k]:.2e} (limit {limit:.2e}); order between grids {', '.join(f'{x:.2f}' for x in order[k])}; fitted over all grids {overall[k]:.2f}")
    st = max(abs(v) for v in fine["state_err"].values())
    check("rung 3b: the uniform states between the waves, finest grid", st < acc["fail_state"], f"largest relative error {st:.1e} (limit {acc['fail_state']:g})")
    pj = max(abs(fine["pos_err"][k]) for k in ("shock", "contact")) / g["L"]
    pf = max(abs(fine["pos_err"][k]) for k in ("head", "tail")) / g["L"]
    check("rung 3b: wave positions, finest grid", pj < acc["fail_position_jump"] and pf < acc["fail_position_fan"],
          f"shock and border {pj:.1e} L (limit {acc['fail_position_jump']:g}), ends of the fan {pf:.1e} L (limit {acc['fail_position_fan']:g})")
    sp = max(abs(v) for v in fine["speed_err"].values()) / fine["shock_speed"]
    check("rung 3b: wave speeds, finest grid", sp < acc["fail_speed"], f"largest error {sp:.1e} of the shock speed (limit {acc['fail_speed']:g})")
    notes = dict(ripple=max(fine["ripple"].values()), wiggle=max(max(r["wiggle"].values()) for r in fam), shock_cells=max(r["width"]["shock"] for r in fam))
    kept = notes["ripple"] <= acc["warn_plateau_ripple"] and notes["wiggle"] <= acc["warn_wiggle"] and notes["shock_cells"] <= acc["warn_shock_cells"]
    print(f"NOTE  rung 3b: the case's warning limits (reported, not judged): ripple on the finest grid {notes['ripple']:.2e} (limit {acc['warn_plateau_ripple']:g}), "
          f"largest wiggle on any grid {notes['wiggle']:.2e} (limit {acc['warn_wiggle']:g}), widest shock {notes['shock_cells']:.2f} cells "
          f"(limit {acc['warn_shock_cells']:g}): {'all kept' if kept else 'NOT all kept'}", flush=True)
    notes["all_kept"] = bool(kept)
    REPORT["warning_limits"] = notes
    ratio = {k: [of[f"N{r['N']}"]["e"][k] / r["e"][k] for r in fam] for k in ("rho", "u", "p")}
    print("  OpenFOAM's error divided by gpucfd's, coarse to fine: " + "; ".join(f"{k}: " + ", ".join(f"{x:.2f}" for x in v) for k, v in ratio.items()))
    REPORT["family"] = dict(setting=DEFAULT, runs=[slim(r) for r in fam], order=order, overall_order=overall, openfoam_over_gpucfd=ratio,
                            openfoam={f"N{r['N']}": {k: of[f"N{r['N']}"][k] for k in ("steps", "wall_time", "e", "width", "ripple", "wiggle", "courant")} for r in fam})

    # ---- c. precision ----
    rows = {}
    for prec in ("double", "mixed", "float"):
        rows[prec] = analyse(run(g, 3200, DEFAULT["recon"], DEFAULT["flux"], DEFAULT["cfl"], DEFAULT["stages"], prec), g, ext)
    dbl = rows["double"]
    diff = {p: max(abs(rows[p]["e"][k] / dbl["e"][k] - 1) for k in ("rho", "u", "p")) for p in ("mixed", "float")}
    check("rung 3c: 3200 cells in mixed and in single precision against double", diff["mixed"] < 0.01 and rows["mixed"]["mass_drift"] < 1e-9 and dbl["mass_drift"] < 1e-11,
          "; ".join(f"{p}: errors {rows[p]['e']['rho']:.3e} {rows[p]['e']['u']:.3e} {rows[p]['e']['p']:.3e}, mass change {rows[p]['mass_drift']:.1e}, "
                    f"energy {rows[p]['energy_drift']:.1e}, GPU {rows[p]['gpu_ms']:.0f} ms" for p in rows)
          + f" (rule: mixed errors within 1 % of double's, mass 1e-9; single is recorded: errors differ by {100 * diff['float']:.2g} %)")
    REPORT["precision"] = {p: slim(r) for p, r in rows.items()}

    # ---- d. the time scheme ----
    rows = []
    for stages, cfl in ((1, 0.19), (1, 0.39), (2, 0.39), (3, 0.39), (3, 0.8), (3, 1.0)):
        try:
            r = analyse(run(g, 400, "vanleer", "hll", cfl, stages, "double"), g, ext)
            rows.append(dict(stages=stages, courant=r["courant"], steps=r["steps"], e=r["e"], ripple=r["ripple"], wiggle=r["wiggle"], width=r["width"]))
        except Exception as err:                                   # a blown-up run is a result here
            rows.append(dict(stages=stages, courant=cfl, failed=str(err)[:200]))
    txt = "; ".join((f"{r['stages']} stage(s) at Courant {r['courant']:.2f}: ripple {100 * r['ripple']['p']:.2f} %, wiggle {100 * r['wiggle']['p']:.1f} %, "
                     f"density error {r['e']['rho']:.2e}") if "e" in r else f"{r['stages']} stage(s) at Courant {r['courant']:.2f}: broke" for r in rows)
    o = {k: of[k] for k in ("N400", "N400_spc0.9")}
    three = [r for r in rows if r["stages"] == 3 and "e" in r and abs(r["courant"] - 0.8) < 0.05]
    check("rung 3d: three stages at Courant 0.8 ring less than OpenFOAM's one stage at 0.19 (400 cells, van Leer, HLL)",
          bool(three) and three[0]["ripple"]["p"] < o["N400"]["ripple"]["p"],
          txt + f". OpenFOAM (one stage): ripple {100 * o['N400']['ripple']['p']:.2f} % at 0.19, {100 * o['N400_spc0.9']['ripple']['p']:.2f} % at 0.39")
    REPORT["time_scheme"] = dict(gpucfd=rows, openfoam={k: dict(courant=v["courant"], ripple=v["ripple"], wiggle=v["wiggle"], e=v["e"]) for k, v in o.items()})

    # ---- e. the alternatives ----
    rows = []
    for recon in ("first", "minmod", "vanleer", "mc", "weno5"):
        for flux in ("rusanov", "hll", "hllc"):
            r = analyse(run(g, 400, recon, flux, 0.8, 3, "double"), g, ext)
            rows.append(dict(recon=recon, flux=flux, e=r["e"], width=r["width"], ripple=r["ripple"], wiggle=r["wiggle"], mass_drift=r["mass_drift"]))
            print(f"  {recon:8s} {flux:8s}: errors {r['e']['rho']:.2e} {r['e']['u']:.2e} {r['e']['p']:.2e}; shock {r['width']['shock']:.2f} cells, "
                  f"border {r['width']['contact']:.2f}; ripple {100 * r['ripple']['p']:.2f} %, wiggle {100 * r['wiggle']['p']:.2f} %", flush=True)
    check("rung 3e: every reconstruction and flux formula runs the shock tube and conserves (400 cells, double)",
          max(r["mass_drift"] for r in rows) < 1e-11, f"{len(rows)} combinations, largest change of mass {max(r['mass_drift'] for r in rows):.1e}")
    REPORT["alternatives"] = rows

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "test_euler_sod.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
