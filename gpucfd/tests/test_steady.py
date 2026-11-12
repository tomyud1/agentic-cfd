"""Verification ladder, rung 11: steady-state tooling (gpucfd/steady.py).

    python -m gpucfd.tests.test_steady          (inside WSL, from the project root)

1. The iteration-error estimator (cfdlib.verification.iteration_error) on sequences with a known limit:
   exact for one geometric mode, conservative for an oscillating one, and never too small while a
   fast mode is still dying out once the slow rate is known (from a coarser grid: sequence() passes it).
   Without it a slow mode hidden under a fast one is invisible; the first grid gets a minimum time.
2. prolong(): a converged circular Couette flow (rung 10) interpolated from 32^2 onto 64^2 is still
   second-order close to the exact solution and divergence-free after the projection.
3. sequence() against cold starts: Couette on 32^2, 64^2, 128^2, each grid either started from the
   previous grid's solution or from rest. Same stopping rule (iteration error below 1 % of the grid's
   discretisation error), so the steady answers must agree to within the stopping tolerances, and the
   sequenced study must cost less GPU time.
"""
import json
import sys
from pathlib import Path

import numpy as np

from cfdlib.verification import iteration_error
from gpucfd import bodies as B
from gpucfd import gcfd, steady
from gpucfd.tests.test_bodies import C0, R1, R2, W1, NU_C, couette_exact, couette_p, face_errors

FAILS = []
REPORT = {}


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------------------ 1. estimator
def test_estimator():
    t = np.arange(60.0)
    cases = {"one mode": 5 - 2 * 0.9 ** t, "oscillating": 5 + 2 * (-0.8) ** t,
             "two modes": 5 + 3 * 0.3 ** t + 0.01 * 0.95 ** t}
    worst = {}
    for name, v in cases.items():
        ratios = []
        rho_min = 0.95 if name == "two modes" else None  # the slow rate, as a coarser grid would measure it
        for m in range(4, 60):
            e = iteration_error(v[:m + 1], rho_min=rho_min)["error"]
            if np.isfinite(e):
                ratios.append(e / abs(v[m] - 5))
        worst[name] = (min(ratios), max(ratios))
    check("rung 11: iteration error estimate exact for one geometric mode",
          abs(worst["one mode"][0] - 1) < 1e-9 and abs(worst["one mode"][1] - 1) < 1e-9,
          f"estimate / true error in [{worst['one mode'][0]:.6f}, {worst['one mode'][1]:.6f}]")
    check("rung 11: iteration error estimate never too small (oscillating; fast mode dying out, slow rate known)",
          worst["oscillating"][0] >= 1 and worst["two modes"][0] >= 0.99,
          f"estimate / true error: oscillating >= {worst['oscillating'][0]:.2f}, two modes >= {worst['two modes'][0]:.3f}")
    rng = np.random.default_rng(0)
    caught, covered, sizes = 0, 0, []
    for _ in range(1000):  # a converged value with solver noise (standard deviation 1e-8) on top
        v = 0.0263 + 1e-8 * rng.standard_normal(40)
        e = iteration_error(v)
        if e["noise"]:
            caught += 1
            covered += e["error"] >= abs(v[-1] - 0.0263)
            sizes.append(e["error"])
    ok = caught >= 950 and covered >= 0.95 * caught and 3e-9 < min(sizes) and max(sizes) < 1e-7
    check("rung 11: a noise floor is recognised and sized (instead of marching forever)", ok,
          f"{caught} of 1000 noisy histories recognised; estimate >= true error in {100 * covered / caught:.1f} %; "
          f"estimates {min(sizes):.1e} to {max(sizes):.1e} for noise 1e-8")
    worst["noise_caught"] = caught
    REPORT["estimator"] = worst


# ------------------------------------------------------------------------------ Couette helpers
def couette_ns(N):
    ctr = [C0[0], C0[1], 0.5]
    bod = [B.Cylinder(ctr, R1, velocity=B.rotation(ctr, W1)), B.Inside(B.Cylinder(ctr, R2))]
    return gcfd.NavierStokes((N, N, 1), (1.0, 1.0, 1.0), NU_C, walls=(True, True, False), bodies=bod)


T_MIN = 1.0  # first grid's minimum marching time: 4 e-folds of the slowest viscous mode ((R2 - R1)^2 / nu / pi^2 = 0.25)


def rest(s):
    s.set([np.zeros(s.shape) for _ in range(s.ncomp)])
    s.project()


# probes at r = 0.28 and 0.37 (gap R1 = 0.2 to R2 = 0.45): at least 2.6 cells from both walls on 32^2, outside the
# band near the surface where the immersed-boundary pressure is less accurate (rung 10)
PTS = np.array([[0.78, 0.5, 0.5], [0.87, 0.5, 0.5], [0.697989898732, 0.697989898732, 0.5], [0.761629509039, 0.761629509039, 0.5]])


def quantities(s):
    p = s.pressure_at(PTS)
    return dict(dp1=p[1] - p[0], dp2=p[3] - p[2])


def exact_q():
    r2 = ((PTS[:, 0] - 0.5) ** 2 + (PTS[:, 1] - 0.5) ** 2)
    pe = couette_p(r2)
    return dict(dp1=pe[1] - pe[0], dp2=pe[3] - pe[2])


# ------------------------------------------------------------------------------ 2. prolong
def test_prolong():
    ex = couette_exact()
    a = couette_ns(32)
    rest(a)
    m = steady.march(a, quantities, a.max_dt(1.0), lambda k, v: 1e-6 * abs(v), chunk=0.02, t_min=T_MIN,
                     t_max=20 * T_MIN, log=None)
    print(f"      32^2 converged {m['converged']} at t {m['t']:.2f} ({m['gpu_s']:.1f} GPU s)", flush=True)
    e32 = face_errors(a, ex)[1]
    b = couette_ns(64)
    div = steady.prolong(a, b)
    e_pro = face_errors(b, ex)[1]
    a.close()
    b.close()
    check("rung 11: prolong() of a converged 32^2 Couette flow onto 64^2 stays second-order close",
          e_pro < 1.5 * e32 and div / 64 < 1e-8,
          f"rms velocity error {e_pro:.2e} after prolongation vs {e32:.2e} on 32^2; max |div u| h / U {div / 64:.1e} "
          f"(the projection's own tolerance level)")
    REPORT["prolong"] = dict(rms_32=e32, rms_prolonged=e_pro, div=div)


# ------------------------------------------------------------------------------ 3. sequence vs cold
def test_sequence():
    kw = dict(chunk=0.02, t_min=T_MIN, t_max=20 * T_MIN, dt_of=lambda s: s.max_dt(1.0), log=lambda m: print("      " + m, flush=True))
    runs = {mode: steady.sequence(couette_ns, rest, quantities, [32, 64, 128], r=2.0, cold=(mode == "cold"), **kw)
            for mode in ("sequenced", "cold")}
    ok, worst, lines = True, 0.0, []
    for a, b in zip(runs["sequenced"]["levels"], runs["cold"]["levels"]):
        for k in a["values"]:
            d = abs(a["values"][k] - b["values"][k])
            allow = 3 * (a["errors"][k] + b["errors"][k])
            ok &= d <= allow
            worst = max(worst, d / allow if allow > 0 else np.inf)
        lines.append(f"N {a['level']}: t {a['t']:.2f} vs {b['t']:.2f}, GPU {a['gpu_s']:.2f} s vs {b['gpu_s']:.2f} s")
    conv = all(x["converged"] for r in runs.values() for x in r["levels"])
    check("rung 11: sequenced and cold-started grids reach the same steady answers (within the stop tolerances)",
          ok and conv, f"worst difference / allowance {worst:.2f}; all converged {conv}")
    g = {m: sum(x["gpu_s"] for x in r["levels"][1:]) for m, r in runs.items()}
    check("rung 11: grid sequencing costs less GPU time on the refined grids",
          g["sequenced"] < g["cold"], f"{g['sequenced']:.2f} s vs {g['cold']:.2f} s from rest; " + "; ".join(lines))
    ex = exact_q()
    for k, st in runs["sequenced"]["study"].items():
        fine = runs["sequenced"]["levels"][-1]["values"][k]
        print(f"      {k}: finest {fine:.6g}, Richardson {st['richardson']}, order {st['order']}, exact {ex[k]:.6g}",
              flush=True)
    REPORT["sequence"] = {m: dict(levels=[{k: v for k, v in x.items() if k != "history"} for x in r["levels"]],
                                  study=r["study"]) for m, r in runs.items()}
    REPORT["sequence"]["exact"] = ex


if __name__ == "__main__":
    test_estimator()
    test_prolong()
    test_sequence()
    out = Path(__file__).parent / "results" / "test_steady.json"
    out.write_text(json.dumps(dict(report=REPORT, fails=FAILS), indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))
