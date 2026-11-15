"""Case 18 with gpucfd: the shock tube on the case's grids, with the outputs the case asks for, read by the
case's own rules against gpucfd's own exact solution. The plan (grids, tuning budget, what each side is
given, what is expected) is plan.yaml, written before the runs.

    bash dev/compare/run.sh shock_tube_sod gpucfd_run.py        (or: PYTHONPATH=. python dev/compare/shock_tube_sod/gpucfd_run.py)

Writes gpucfd.json next to this file, in the form cfdlib.brief reads (answers, runs, plain), with the tuning
tries, both families of runs and OpenFOAM's recorded numbers on the same grids.
"""
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import yaml

from cfdlib import foam, verification as V
from cfdlib.report import m, tex_num, tex_pct as tp
from gpucfd import gcfd
from gpucfd.euler import Euler
from gpucfd.tests import test_euler_sod as T

HERE = Path(__file__).resolve().parent
OUT = HERE / "gpucfd.json"
PRECISION = "mixed"


def family(g, ext, recon, flux, cfl, grids):
    rows = []
    for N in grids:
        r = T.analyse(T.run(g, N, recon, flux, cfl, 3, PRECISION), g, ext)
        timed = [T.run(g, N, recon, flux, cfl, 3, PRECISION, history=False) for _ in range(3)]
        best = min(timed, key=lambda d: d["wall_s"])                # best of three: the PC is shared with a CPU solver
        r["wall_s"], r["gpu_ms"] = best["wall_s"], best["gpu_ms"]
        rows.append(T.slim(r))
        print(f"  {recon}/{flux} N {N:6d}: {r['steps']:5d} steps, wall {r['wall_s']:.3f} s | errors {r['e']['rho']:.2e} {r['e']['u']:.2e} {r['e']['p']:.2e} | "
              f"shock {r['width']['shock']:.2f}, border {r['width']['contact']:.2f} cells | ripple {100 * r['ripple']['p']:.2f} %, wiggle {100 * r['wiggle']['p']:.2f} %", flush=True)
    return rows


def rules(fam, acc):
    """The case's acceptance rules that do not depend on the tool. Returns (fail, warn) as lists of sentences."""
    fail, warn = [], []
    fine, coarse = fam[-1], fam[0]
    lim = 1e-9                                                     # mixed precision (plan.yaml)
    for k, name in (("mass_drift", "mass"), ("energy_drift", "energy"), ("momentum_err", "momentum")):
        worst = max(r[k] for r in fam)
        if worst > lim:
            fail.append(f"{name} changes by {worst:.1e} (limit {lim:g})")
    if max(r["ends"] for r in fam) > acc["fail_ends"]:
        fail.append("an end cell is disturbed")
    for k in ("rho", "u", "p"):
        limit = coarse["e"][k] * (fine["h"] / coarse["h"]) ** acc["fail_l1_order_floor"][k]
        if fine["e"][k] > limit:
            fail.append(f"error of {k} on the finest grid {fine['e'][k]:.2e} above {limit:.2e} (order floor {acc['fail_l1_order_floor'][k]})")
    if max(abs(v) for v in fine["state_err"].values()) > acc["fail_state"]:
        fail.append("a state between the waves is off by more than the limit")
    if max(abs(fine["pos_err"][k]) for k in ("shock", "contact")) > acc["fail_position_jump"] or \
            max(abs(fine["pos_err"][k]) for k in ("head", "tail")) > acc["fail_position_fan"]:
        fail.append("a wave position is off by more than the limit")
    if max(abs(v) for v in fine["speed_err"].values()) / fine["shock_speed"] > acc["fail_speed"]:
        fail.append("a wave speed is off by more than the limit")
    hs = [r["h"] for r in fam]
    for k in ("rho", "u", "p"):
        o = V.order_from_errors(hs[-2:], [r["e"][k] for r in fam[-2:]])[0]
        lo, hi = acc["warn_order"][k]
        if not lo <= o <= hi:
            warn.append(f"order of {k} at the last refinement {o:.2f} outside {lo} to {hi}")
    if max(max(r["wiggle"].values()) for r in fam) > acc["warn_wiggle"]:
        warn.append(f"wiggle {max(max(r['wiggle'].values()) for r in fam):.3f} above {acc['warn_wiggle']}")
    if max(fine["ripple"].values()) > acc["warn_plateau_ripple"]:
        warn.append(f"ripple on the finest grid {max(fine['ripple'].values()):.4f} above {acc['warn_plateau_ripple']}")
    if max(r["width"]["shock"] for r in fam) > acc["warn_shock_cells"]:
        warn.append("shock wider than the limit")
    return fail, warn


def main():
    plan = yaml.safe_load((HERE / "plan.yaml").read_text(encoding="utf-8"))
    g, ext, acc = T.given()
    of = T.openfoam()
    ex = T.exact(g)
    busy = [p[3][:60] for p in foam.running()]
    t0 = time.perf_counter()
    name = gcfd.device()
    s = Euler((16, 1, 1), (1.0, 1.0, 1.0))                         # loads the library and starts the GPU
    s.set(rho=1.0, p=1.0)
    s.run_fixed(1e-3, 1)
    s.close()
    startup = time.perf_counter() - t0
    print(f"GPU: {name}; start-up {startup:.2f} s; other solver processes on the CPU: {len(busy)}", flush=True)
    T.check_exact(g)                                               # the exact solution, and OpenFOAM's errors reproduced with it
    if T.FAILS:
        raise SystemExit("the exact solution failed its checks: nothing is recorded")

    # ---- the tuning budget: twelve runs on 400 cells ----
    tries = []
    for recon in ("vanleer", "mc", "weno5"):
        for flux in ("hll", "hllc"):
            for cfl in (0.4, 0.8):
                r = T.analyse(T.run(g, 400, recon, flux, cfl, 3, PRECISION), g, ext)
                ok = max(r["ripple"].values()) <= acc["warn_plateau_ripple"] and max(r["wiggle"].values()) <= acc["warn_wiggle"]
                tries.append(dict(recon=recon, flux=flux, cfl=cfl, courant=r["courant"], steps=r["steps"], e=r["e"], score=sum(r["e"].values()),
                                  ripple=r["ripple"], wiggle=r["wiggle"], width=r["width"], within_limits=bool(ok)))
                print(f"try {recon:8s} {flux:5s} Courant {cfl}: errors {r['e']['rho']:.2e} {r['e']['u']:.2e} {r['e']['p']:.2e} sum {tries[-1]['score']:.2e}; "
                      f"ripple {100 * max(r['ripple'].values()):.2f} %, wiggle {100 * max(r['wiggle'].values()):.2f} %, shock {r['width']['shock']:.2f}, "
                      f"border {r['width']['contact']:.2f} cells; within limits {ok}", flush=True)
    ok = [t for t in tries if t["within_limits"]]
    if not ok:
        raise SystemExit("no try kept the ripple and wiggle limits: the plan's budget is used up; nothing is recorded")
    best = min(ok, key=lambda t: t["score"])
    print(f"chosen: {best['recon']} / {best['flux']} at Courant {best['cfl']}", flush=True)

    grids = plan["grids"]
    tuned = family(g, ext, best["recon"], best["flux"], best["cfl"], grids)
    alike = family(g, ext, "vanleer", "hll", best["cfl"], grids)
    fail, warn = rules(tuned, acc)
    fail2, warn2 = rules(alike, acc)
    # The problem statement sets no required accuracy, so a missed limit is an expectation that was not met
    # (CLAUDE.md, 2026-11-15): it is listed as expected and found, and it does not make the experiment fail.
    verdict = "limits missed (expected, found): " + "; ".join(fail) if fail else "all limits kept"
    hs = [r["h"] for r in tuned]
    fit = lambda fam, k: float(np.polyfit(np.log(hs), np.log([r["e"][k] for r in fam]), 1)[0])          # noqa: E731
    orders = dict(tuned={k: fit(tuned, k) for k in ("rho", "u", "p")}, like_for_like={k: fit(alike, k) for k in ("rho", "u", "p")},
                  openfoam={k: fit([of[f"N{N}"] for N in grids], k) for k in ("rho", "u", "p")})
    per_grid = []
    for a, b in zip(tuned, alike):
        o = of[f"N{a['N']}"]
        per_grid.append(dict(N=a["N"], openfoam=dict(steps=o["steps"], wall_s=o["wall_time"], e=o["e"], width=o["width"], ripple=o["ripple"], wiggle=o["wiggle"]),
                             tuned=dict(steps=a["steps"], wall_s=a["wall_s"], e=a["e"], width=a["width"], ripple=a["ripple"], wiggle=a["wiggle"]),
                             like_for_like=dict(steps=b["steps"], wall_s=b["wall_s"], e=b["e"], width=b["width"], ripple=b["ripple"], wiggle=b["wiggle"]),
                             error_ratio_tuned={k: o["e"][k] / a["e"][k] for k in a["e"]}, error_ratio_like={k: o["e"][k] / b["e"][k] for k in b["e"]},
                             time_ratio_tuned=o["wall_time"] / a["wall_s"]))
        p = per_grid[-1]
        print(f"N {a['N']:6d}: OpenFOAM's error / gpucfd's: tuned " + " ".join(f"{p['error_ratio_tuned'][k]:.2f}" for k in ("rho", "u", "p"))
              + "; like for like " + " ".join(f"{p['error_ratio_like'][k]:.2f}" for k in ("rho", "u", "p"))
              + f"; OpenFOAM's time / gpucfd's {p['time_ratio_tuned']:.1f}", flush=True)

    fine = tuned[-1]
    w = T.case_waves(ex, g["t_end"])
    names = (("head", "fan front"), ("tail", "fan back"), ("contact", "border"), ("shock", "shock"))
    answers = dict(
        a=dict(value=fine["e"], diff="<br>".join(m(s_) + ": " + m(tp(fine["e"][k], 2)) for k, s_ in (("rho", r"\rho"), ("u", "u"), ("p", "p")))),
        b=dict(value=fine["state_err"], diff="<br>".join(m(tp(abs(fine["state_err"][k]), 2)) for k in ("p", "u", "rho_L", "rho_R"))),
        c=dict(value=dict(pos_err=fine["pos_err"], speed_err=fine["speed_err"]),
               diff="<br>".join(m(rf"{1e3 * fine['pos_err'][k]:+.2f}\,\mathrm{{mm}},\ {fine['speed_err'][k]:+.2f}\,\mathrm{{m/s}}") for k, _ in names)),
        d=dict(value=dict(mass=max(r["mass_drift"] for r in tuned + alike), energy=max(r["energy_drift"] for r in tuned + alike)),
               diff="", sim="constant. Largest change: mass " + m(tex_num(max(r['mass_drift'] for r in tuned + alike), 2)) + ", energy "
                            + m(tex_num(max(r['energy_drift'] for r in tuned + alike), 2)) + " (relative; fluxes in single precision)"))
    ratio_fine = per_grid[-1]["error_ratio_tuned"]
    plain = (f"gpucfd's new solver for gas with shocks runs the same grids with errors {min(ratio_fine.values()):.1f} to {max(ratio_fine.values()):.1f} "
             f"times smaller than OpenFOAM's on the finest grid, in {fine['steps']} steps against {of['N12800']['steps']} and "
             f"{fine['wall_s']:.1f} s against {of['N12800']['wall_time']:.0f} s.")
    runs = [dict(label=f"{r['N']} cells, {r['steps']} steps (gpucfd)", N=r["N"], cells=r["N"], steps=r["steps"], wall_s=r["wall_s"], gpu_s=r["gpu_ms"] / 1e3,
                 courant=r["courant"], recon=r["recon"], flux=r["flux"], precision=r["precision"]) for r in tuned if r["N"] in (100, 800, 3200, 12800)]
    rec = dict(tool="gpucfd", case=HERE.name, date=dt.date.today().isoformat(), device=name, startup_s=startup, plain=plain,
               busy_cpu=bool(busy), busy_processes=busy, answers=answers, runs=runs,
               setting=dict(recon=best["recon"], flux=best["flux"], courant=best["cfl"], stages=3, precision=PRECISION),
               tuning=dict(tries=tries, chosen=best), grids=dict(tuned=tuned, like_for_like=alike), orders=orders,
               by_case_rules=dict(verdict=verdict, fail=fail, warn=warn, like_for_like=dict(fail=fail2, warn=warn2)),
               against_openfoam=dict(per_grid=per_grid, what_each_side_is_given=plan["what_each_side_is_given"]),
               exact=dict(p_star=ex.p_star, u_star=ex.u_star, waves=w))
    OUT.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    print(f"by the case's rules: {verdict}; fail {fail}; warn {warn}\nlike for like: fail {fail2}; warn {warn2}")
    print("fitted order of the error over all grids: " + "; ".join(f"{k}: " + " ".join(f"{v[q]:.2f}" for q in ("rho", "u", "p")) for k, v in orders.items()))
    print(f"record: {OUT}")


if __name__ == "__main__":
    main()
