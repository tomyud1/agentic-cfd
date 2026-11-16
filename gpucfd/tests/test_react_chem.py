"""Reacting ladder, cheaper chemistry at detonation states: the implicit method with an exact matrix ("ros34a")
and the rule for quiet cells.

    python -m gpucfd.tests.test_react_chem           (inside WSL, from the project root; about a minute)
    python -m gpucfd.tests.test_react_chem tune      (the tuning budget: 12 runs, about ten minutes)

Pass rules, expectations and the budget were written before the runs:
dev/compare/detonation_tube_h2_air/plan.yaml, "chemistry_cheaper".
  j1. The exact matrix against central differences of cfdlib.chem's rates (independent code), at constant
      volume and at constant pressure, double precision: within 1e-5 of each row's largest entry.
  j2. Order of ros34a with fixed sub-steps: 4 within 0.4.
  j3. Atoms over a burning parcel in mixed precision: ros34a within 1e-12 (the old ros34 is shown next to it).
  tune. State (i): 250 000 cells at the state behind a shock of the CJ speed, 400 steps of 5e-9 s; state (ii):
      fresh gas at 300 K and 1 atm. The cheapest mixed-precision setting that keeps the accuracy rule becomes
      gpucfd.reacting.CHEMISTRY.
"""
import json
import sys
from pathlib import Path

import numpy as np

from cfdlib import chem
from gpucfd.chemistry import DEFAULT, Chemistry

OUT = Path(__file__).parent / "results"
FAILS = []
REPORT = {}
P1, T1, MIX = 101325.0, 300.0, dict(H2=2.0, O2=1.0, N2=3.76)
LEVEL, N, DT, STEPS = 2000.0, 250_000, 5e-9, 400


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def temperature(mech, sigma, energy, T, volume):
    """The temperature at which the mixture sigma (mol/kg) has that internal energy (volume held) or enthalpy."""
    R = mech.conv.R
    for _ in range(50):
        f = sigma @ mech.h(T) - (sigma.sum() * R * T if volume else 0.0)
        d = sigma @ mech.cp(T) - (sigma.sum() * R if volume else 0.0)
        dT = (energy - f) / d
        T += dT
        if abs(dT) < 1e-14 * T:
            break
    return T


def rate(mech, sigma, energy, pv, T, volume):
    """d(sigma)/dt (mol/(kg s)) by cfdlib.chem, with the energy and the volume (or the pressure) held."""
    T = temperature(mech, sigma, energy, T, volume)
    v = pv if volume else sigma.sum() * mech.conv.R * T / pv
    return v * mech.omega(sigma / v, T)


def matrix_by_differences(mech, sigma, T, p, volume, act, rel=1e-4):
    R = mech.conv.R
    v = sigma.sum() * R * T / p
    energy = sigma @ mech.h(T) - (sigma.sum() * R * T if volume else 0.0)
    pv = v if volume else p
    J = np.zeros((len(act), len(act)))
    for b, k in enumerate(act):
        d = rel * sigma[k]
        sp, sm = sigma.copy(), sigma.copy()
        sp[k] += d
        sm[k] -= d
        J[:, b] = (rate(mech, sp, energy, pv, T, volume) - rate(mech, sm, energy, pv, T, volume))[act] / (2 * d)
    return J


def detonation_states(mech):
    """Three states with every species present: a parcel that started behind the shock of a CJ detonation, in
    its temperature rise (2000 K) and burnt (after 2 microseconds); and the CJ state itself."""
    cj = chem.cj_detonation(mech, P1, T1, MIX)
    vn = chem.shock_frozen(mech, P1, T1, MIX, cj["D"])
    parcel = chem.VolumeParcel(mech, vn["p"], vn["T"], MIX)
    sol = parcel.solve(DT * STEPS, T_ign=LEVEL)
    y = sol["sol"](sol["t_ign"])
    rise = dict(T=float(y[-1]), sigma=y[:-1], p=float(parcel.pressure(y[:-1], y[-1])))
    burnt = dict(T=float(sol["T"][-1]), sigma=sol["sigma"][-1], p=float(parcel.pressure(sol["sigma"][-1], sol["T"][-1])))
    return cj, vn, sol, [("rise behind the shock", rise), ("burnt at constant volume", burnt), ("CJ state", dict(T=cj["T"], sigma=cj["sigma"], p=cj["p"]))]


def main():
    mech = chem.build()
    ns = len(mech.species)
    cj, vn, sol, special = detonation_states(mech)
    REPORT["front"] = dict(p=vn["p"], T=vn["T"])

    # ---- j1: the exact matrix against differences of cfdlib.chem's rates ----
    rng = np.random.default_rng(3)
    n = 300
    T = rng.uniform(900.0, 3500.0, n)
    p = 101325.0 * 10 ** rng.uniform(-0.5, np.log10(60.0), n)
    x = 10 ** rng.uniform(-3, 0, (n, ns))
    sigma = x / (x @ mech.W)[:, None]
    T = np.append(T, [s["T"] for _, s in special])
    p = np.append(p, [s["p"] for _, s in special])
    sigma = np.vstack([sigma] + [s["sigma"] for _, s in special])
    REPORT["j1"] = {}
    for constant in ("volume", "pressure"):
        c = Chemistry(mech, len(T), constant=constant, precision="double")
        c.set(T=T, p=p, sigma=sigma)
        J, act = c.jacobian()
        c.close()
        worst, where = 0.0, 0
        named = {}
        for k in range(len(T)):
            ref = matrix_by_differences(mech, sigma[k], T[k], p[k], constant == "volume", act)
            d = float(np.max(np.abs(J[k] - ref) / np.max(np.abs(ref), axis=1, keepdims=True)))
            if d > worst:
                worst, where = d, k
            if k >= n:
                named[special[k - n][0]] = d
        REPORT["j1"][constant] = dict(states=len(T), worst=worst, at=dict(T=T[where], p=p[where]), detonation=named)
        check(f"j1 exact matrix against differences of cfdlib.chem's rates, constant {constant}", worst <= 1e-5,
              f"{len(T)} states: largest difference {worst:.1e} of a row's largest entry (at {T[where]:.0f} K, {p[where] / 101325:.1f} atm; limit 1e-5); "
              + ", ".join(f"{k} {v:.1e}" for k, v in named.items()))

    # ---- j2: order of ros34a with fixed sub-steps, from the rise behind the shock, at constant volume ----
    rise, span = special[0][1], 2e-8

    def fixed(method, k):
        c = Chemistry(mech, 1, constant="volume", precision="double", method=method)
        c.control(fixed=k)
        c.set(T=rise["T"], p=rise["p"], sigma=rise["sigma"])
        c.advance(span, 1)
        s = c.get()
        c.close()
        return np.append(s["sigma"][0], s["T"][0])
    exact = fixed("dp45", 20000)
    scale = np.append(np.maximum(np.abs(exact[:-1]), 1e-3 * np.abs(exact[:-1]).max()), exact[-1])
    REPORT["j2"] = {}
    for method in ("ros34a", "ros34"):
        # from 1e-9 s down to 1.6e-11 s: at this state the order reaches 4 only for short sub-steps (the first run
        # of this test stopped at 160 and found 3.42)
        counts = (20, 40, 80, 160, 320, 640, 1280)
        errs = [float(np.max(np.abs(fixed(method, k) - exact) / scale)) for k in counts]
        obs = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
        REPORT["j2"][method] = dict(substeps=list(counts), errors=errs, orders=obs)
        check(f"j2 order of {method} with fixed sub-steps is 4 (within 0.4, last refinement)", abs(obs[-1] - 4) <= 0.4,
              "errors " + ", ".join(f"{e:.2e}" for e in errs) + "; orders " + ", ".join(f"{o:.2f}" for o in obs))

    # ---- j3: atoms over a burning parcel in mixed precision ----
    REPORT["j3"] = {}
    T_ref = float(sol["T"][-1])
    for method in ("ros34a", "ros34", "bs23"):
        c = Chemistry(mech, 64, constant="volume", precision="mixed", method=method, rtol=1e-4)
        c.control(level=LEVEL)
        c.set(T=vn["T"], p=vn["p"], moles=MIX)
        a0 = chem.atoms(c.get()["Y"], mech)
        c.advance(DT, STEPS)
        s, st = c.get(), c.stats()
        c.close()
        a1 = chem.atoms(s["Y"], mech)
        present = a0[0] > 0
        atoms = float(np.max(np.abs(a1[:, present] / a0[:, present] - 1)))
        d_t, d_T = float(st["t_cross"][0] / sol["t_ign"] - 1), float(s["T"][0] / T_ref - 1)
        REPORT["j3"][method] = dict(atoms=atoms, d_t=d_t, d_T=d_T, substeps=float(st["substeps"].mean() / STEPS), status=int(st["status"].max()))
        line = (f"atoms changed by {atoms:.1e}; time to 2000 K {100 * d_t:+.4f} %, temperature after 2 microseconds {100 * d_T:+.5f} %, "
                f"{st['substeps'].mean() / STEPS:.2f} sub-steps per step")
        if method == "ros34a":
            check("j3 ros34a in mixed precision keeps atoms to round-off and the parcel's answer", atoms <= 1e-12 and abs(d_t) <= 1e-3 and abs(d_T) <= 1e-4
                  and st["status"].max() == 0, line + " (limits 1e-12, 0.1 %, 0.01 %)")
        else:
            print(f"      {method}, the same parcel: {line}", flush=True)
    finish("test_react_chem.json")


def tune():
    """The budget of 12 runs. Writes test_react_chem_tune.json; the choice is printed and recorded."""
    mech = chem.build()
    cj, vn, sol, _ = detonation_states(mech)
    T_ref = float(sol["T"][-1])
    c = Chemistry(mech, 16, constant="volume")                   # start-up of the GPU library, not timed
    c.set(T=vn["T"], p=vn["p"], moles=MIX)
    c.advance(DT, 1)
    c.close()
    rows = []
    for precision, method, rtol in ([("mixed", me, tol) for me in ("bs23", "ros34", "ros34a") for tol in (1e-4, 1e-6)]
                                    + [("double", "ros34a", 1e-4), ("double", "ros34a", 1e-6)]):
        c = Chemistry(mech, N, constant="volume", precision=precision, method=method, rtol=rtol, atol=DEFAULT["atol"])
        c.control(level=LEVEL)
        c.set(T=vn["T"], p=vn["p"], moles=MIX)
        a0 = chem.atoms(c.get()["Y"][:64], mech)
        out = c.advance(DT, STEPS)
        s, st = c.get(), c.stats()
        c.close()
        a1 = chem.atoms(s["Y"][:64], mech)
        present = a0[0] > 0
        row = dict(state="front", precision=precision, method=method, rtol=rtol, gpu_s=out["ms"] / 1e3, cell_steps_per_s=N * STEPS / (out["ms"] / 1e3),
                   substeps_per_step=float(st["substeps"].mean() / STEPS), rejected_per_step=float(st["rejected"].mean() / STEPS), status=int(st["status"].max()),
                   d_t=float(st["t_cross"][0] / sol["t_ign"] - 1), d_T=float(s["T"][0] / T_ref - 1), atoms=float(np.max(np.abs(a1[:, present] / a0[:, present] - 1))))
        row["accurate"] = bool(abs(row["d_t"]) <= 1e-3 and abs(row["d_T"]) <= 1e-4 and row["status"] == 0 and row["atoms"] <= 1e-10)
        rows.append(row)
        print(f"{'ok  ' if row['accurate'] else 'OUT '} {precision:6s} {method:6s} tolerance {rtol:.0e}: {row['gpu_s']:6.1f} s, {row['cell_steps_per_s'] / 1e6:6.1f} million cells x steps "
              f"per second, {row['substeps_per_step']:5.2f} sub-steps per step ({row['rejected_per_step']:.2f} rejected), time to 2000 K {100 * row['d_t']:+.4f} %, end "
              f"temperature {100 * row['d_T']:+.5f} %, atoms {row['atoms']:.1e}, status {row['status']}", flush=True)
    good = [r for r in rows if r["accurate"] and r["precision"] == "mixed"]
    best = min(good, key=lambda r: r["gpu_s"])
    dflt = next(r for r in rows if r["precision"] == "mixed" and r["method"] == DEFAULT["method"] and r["rtol"] == DEFAULT["rtol"])
    print(f"cheapest mixed-precision setting that keeps the accuracy rule: {best['method']} tolerance {best['rtol']:.0e}, "
          f"{dflt['gpu_s'] / best['gpu_s']:.2f} times cheaper than the chemistry kernel's default", flush=True)
    # state (ii): fresh gas, the quiet rule off and on
    fresh = []
    for method, rtol in (("bs23", DEFAULT["rtol"]), (best["method"], best["rtol"])):
        ends = {}
        for quiet in (0.0, 1.0):
            c = Chemistry(mech, N, constant="volume", precision="mixed", method=method, rtol=rtol, atol=DEFAULT["atol"])
            c.control(quiet=quiet)
            c.set(T=T1, p=P1, moles=MIX)
            out = c.advance(DT, STEPS)
            s, st = c.get(), c.stats()
            c.close()
            ends[quiet] = s["sigma"][0]
            fresh.append(dict(state="fresh", method=method, rtol=rtol, quiet=quiet, gpu_s=out["ms"] / 1e3, cell_steps_per_s=N * STEPS / (out["ms"] / 1e3),
                              substeps_per_step=float(st["substeps"].mean() / STEPS), status=int(st["status"].max())))
        fresh[-1]["changed"] = fresh[-2]["changed"] = float(np.max(np.abs(ends[1.0] - ends[0.0])))
        print(f"fresh gas, {method} tolerance {rtol:.0e}: {fresh[-2]['cell_steps_per_s'] / 1e6:.0f} million cells x steps per second, with the quiet rule "
              f"{fresh[-1]['cell_steps_per_s'] / 1e6:.0f} ({fresh[-2]['gpu_s'] / fresh[-1]['gpu_s']:.1f} times cheaper); species differ by at most {fresh[-1]['changed']:.1e} mol/kg", flush=True)
    keep = fresh[-1]["changed"] <= 1e-12 and fresh[-1]["gpu_s"] < fresh[-2]["gpu_s"]
    choice = dict(method=best["method"], rtol=best["rtol"], atol=DEFAULT["atol"], quiet=1.0 if keep else 0.0)
    print(f"chosen for gpucfd.reacting.CHEMISTRY: {choice}")
    REPORT.update(front=dict(p=vn["p"], T=vn["T"]), cells=N, dt=DT, steps=STEPS, reference=dict(t_2000=sol["t_ign"], T_end=T_ref), rows=rows, fresh=fresh,
                  best=best, default=dflt, gain=dflt["gpu_s"] / best["gpu_s"], quiet_kept=bool(keep), choice=choice)
    finish("test_react_chem_tune.json")


def finish(name):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    tune() if len(sys.argv) > 1 and sys.argv[1] == "tune" else main()
