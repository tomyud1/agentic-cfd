"""Chemistry ladder, rung 2: one parcel. Case 20 (ignition of hydrogen and air) at constant pressure against
cfdlib.chem and the CHEMKIN II reference; the same parcel at constant volume against cfdlib.chem; the order of
each integrator.

    python -m gpucfd.tests.test_chem_parcel          (inside WSL, from the project root; about a minute)

Pass rules and expectations were written before the runs: dev/compare/h2_air_ignition_box/plan.yaml.
Required by the problem: ignition time within 1 % and final temperature within 0.1 % of the reference, atoms of
each element constant within 0.001 %. Own, tighter rules: ignition time within 1e-6 and final temperature within
1e-3 K of cfdlib.chem with the same constants; atoms within 1e-12 at every step; the energy rebuilt from the
output within 1e-10 of the heat released; orders within 0.4 of 3, 5 and 4.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cfdlib import chem
from gpucfd.chemistry import Chemistry

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cases" / "h2_air_ignition_box"
FAILS = []
REPORT = {}
TIGHT = dict(method="dp45", rtol=1e-10, atol=1e-20)


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def given():
    """Case 20 as given, and its reference: (dict of values, moles, reference ignition time, reference final T)."""
    prob = yaml.safe_load((CASE / "problem.yaml").read_text(encoding="utf-8"))
    g = {k: v["value"] for k, v in prob["given"].items()}
    ref = yaml.safe_load((CASE / "reference" / "senkin_h2.yaml").read_text(encoding="utf-8"))
    return g, dict(H2=g["x_H2"], O2=g["x_O2"], N2=g["x_N2"]), ref["ignition"]["time_s"], ref["final"]["T_K"]


def run_parcel(mech, g, moles, constant="pressure", precision="double", dt=1e-8, t_end=None, T0=None, history=True, **control):
    """One cell from the case's start state to t_end in steps of dt. Returns the history (t, T, Y, sigma after
    every step, the start included), the ignition time by the case's reading rule and by the kernel's own
    record, the end state, the sub-steps and the times."""
    steps = int(round((t_end or g["t_end"]) / dt))
    t0 = time.perf_counter()
    c = Chemistry(mech, 1, constant=constant, precision=precision)
    c.control(level=g["T_ign"], **control)
    c.set(T=T0 or g["T_0"], p=g["p"], moles=moles)
    start = c.get()
    out = c.advance(dt, steps, history=history)
    end, st = c.get(), c.stats()
    wall = time.perf_counter() - t0
    c.close()
    r = dict(constant=constant, precision=precision, dt=dt, steps=steps, wall_s=wall, gpu_ms=out["ms"], control=dict(control),
             T_end=float(end["T"][0]), p_end=float(end["p"][0]), rho_end=float(end["rho"][0]), sigma_end=end["sigma"][0], energy=float(end["energy"][0]),
             substeps=int(st["substeps"][0]), rejected=int(st["rejected"][0]), status=int(st["status"][0]), t_ign_kernel=float(st["t_cross"][0]))
    if history:
        r["t"] = dt * np.arange(steps + 1)
        r["T"] = np.concatenate([start["T"], out["T"][:, 0]])
        r["sigma"] = np.concatenate([start["sigma"], out["sigma"][:, 0, :]])
        r["Y"] = r["sigma"] * mech.W
        r["t_ign"] = chem.crossing_time(r["t"], r["T"], g["T_ign"])
    return r


def joint_jump(mech, sigma):
    """The data's own jump: therm.dat gives each species two polynomials joined at 1000 K, and they do not give
    exactly the same enthalpy there. Returns the jump of the mixture's enthalpy at the joint, J/kg."""
    Tm = float(mech.T_mid[0])
    assert np.all(mech.T_mid == Tm)
    w = np.array([1, Tm / 2, Tm ** 2 / 3, Tm ** 3 / 4, Tm ** 4 / 5, 1 / Tm, 0])
    return abs(float(sigma @ (mech.conv.R * Tm * ((mech.a_hi - mech.a_lo) @ w))))


def conserved(r, mech, parcel, q):
    """Atoms (largest relative change over every step) and the energy rebuilt by cfdlib.chem from the output
    temperature and species (every 20th step, the first 400 and the last), relative to the heat released.
    The energy is given twice: over the steps whose temperature is not exactly on the joint of the data's two
    polynomials (1000 K, which is also the start temperature), and over all steps. For a narrow band of energies
    the data have no temperature (joint_jump); the kernel then reports the joint itself, and the rebuilt energy
    is off by at most that jump. The parcel sits there for the first microseconds: the first reactions take up
    heat. No program can do better with these data.
    Returns dict(atoms, energy, energy_on_joint, steps_on_joint, jump, Y_min, sumY)."""
    n = chem.atoms(r["Y"], mech)
    present = n[0] > 0
    atoms = float(np.max(np.abs(n[:, present] / n[0, present] - 1)))
    idx = np.unique(np.concatenate([np.arange(0, len(r["t"]), 20), np.arange(min(400, len(r["t"]))), [len(r["t"]) - 1]]))
    if r["constant"] == "pressure":
        e = np.array([r["sigma"][k] @ mech.h(r["T"][k]) for k in idx])
    else:
        e = np.array([parcel.energy(r["sigma"][k], r["T"][k]) for k in idx])
    de = np.abs(e - r["energy"]) / q                                       # against the energy the kernel stores
    on = np.abs(r["T"] - mech.T_mid[0]) <= 1e-9
    on[0] = False                                                          # the start is a given state, not a found one
    away = ~on[idx]
    return dict(atoms=atoms, energy=float(de[away].max()), energy_on_joint=float(de.max()), steps_on_joint=int(on.sum()),
                jump=joint_jump(mech, r["sigma"][0]) / q, Y_min=float(r["Y"].min()), sumY=float(np.max(np.abs(r["Y"].sum(1) - 1))))


def energy_line(c, q, what):
    return (f"atoms {c['atoms']:.1e}; {what} rebuilt by cfdlib.chem: {c['energy']:.1e} of the heat released ({q:.4e} J/kg) away from the data's joint at "
            f"1000 K, {c['energy_on_joint']:.1e} on it ({c['steps_on_joint']} steps; the data's own jump there is {c['jump']:.1e}); lowest mass fraction "
            f"{c['Y_min']:.1e}, sum off by {c['sumY']:.1e}")


def slim(r):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if k not in ("t", "T", "sigma", "Y")}


def main():
    g, moles, t_ref, T_ref = given()
    mechs = {"CHEMKIN": chem.build(), "OpenFOAM": chem.build(chem.openfoam_conventions())}
    REPORT.update(given=g, reference=dict(t_ign=t_ref, T_end=T_ref))

    # ---- constant pressure: case 20 ----
    state_at_ignition = None
    for name, mech in mechs.items():
        parcel = chem.Parcel(mech, g["p"], g["T_0"], moles)
        T_cc, s_cc = parcel.complete_combustion()
        q = parcel.h0 - float(s_cc @ mech.h(g["T_0"]))                    # heat released by complete burning, J/kg
        ind = parcel.solve(g["t_end"], T_ign=g["T_ign"], rtol=1e-10, atol=1e-20)
        r = run_parcel(mech, g, moles, **TIGHT)
        cons = conserved(r, mech, parcel, q)
        atoms = cons["atoms"]
        health = chem.parcel_health(r["T"], r["Y"], mech, g["T_0"], T_cc, tol_sum=1e-12, tol_atoms=1e-12)
        d_t, d_T = r["t_ign"] / ind["t_ign"] - 1, r["T_end"] - float(ind["T"][-1])
        rec = dict(slim(r), independent=dict(t_ign=ind["t_ign"], T_end=float(ind["T"][-1])), d_ind_t=d_t, d_ind_T=d_T, d_ref_t=r["t_ign"] / t_ref - 1,
                   d_ref_T=r["T_end"] / T_ref - 1, T_cc=T_cc, q=q, health=health["checks"], **cons)
        REPORT[f"pressure_{name}"] = rec
        print(f"constant pressure, {name}'s constants: ignition {1e6 * r['t_ign']:.4f} us (cfdlib.chem {1e6 * ind['t_ign']:.4f}), final {r['T_end']:.4f} K "
              f"(cfdlib.chem {float(ind['T'][-1]):.4f}); {r['substeps']} sub-steps, {r['rejected']} rejected, {r['wall_s']:.1f} s", flush=True)
        check(f"rung 2: status and health of the run ({name}'s constants)", r["status"] == 0 and health["ok"],
              "; ".join(f"{c['name']}: {c['value']}" for c in health["checks"]))
        check(f"rung 2: the kernel's own record of the ignition time equals the reading of the history ({name})",
              abs(r["t_ign_kernel"] / r["t_ign"] - 1) < 1e-12, f"{r['t_ign_kernel']:.12e} and {r['t_ign']:.12e}")
        check(f"rung 2: constant pressure against cfdlib.chem, {name}'s constants (ignition time within 1e-6, final temperature within 1e-3 K)",
              abs(d_t) <= 1e-6 and abs(d_T) <= 1e-3, f"ignition time {d_t:+.1e}, final temperature {d_T:+.1e} K (expected within 1e-7 and 1e-5 K)")
        check(f"rung 2: atoms within 1e-12 at every step, energy within 1e-10 of the heat released, and on the data's joint within its jump ({name})",
              atoms <= 1e-12 and cons["energy"] <= 1e-10 and cons["energy_on_joint"] <= 1.01 * cons["jump"] + 1e-10, energy_line(cons, q, "enthalpy"))
        if name == "CHEMKIN":
            check("rung 2, REQUIRED: ignition time within 1 % of the reference", abs(rec["d_ref_t"]) <= 0.01,
                  f"{1e6 * r['t_ign']:.3f} us against {1e6 * t_ref:.2f} us: {100 * rec['d_ref_t']:+.3f} % (expected -0.022 %)")
            check("rung 2, REQUIRED: temperature at 1 ms within 0.1 % of the reference", abs(rec["d_ref_T"]) <= 0.001,
                  f"{r['T_end']:.3f} K against {T_ref:.1f} K: {100 * rec['d_ref_T']:+.4f} % (expected -0.002 %)")
            check("rung 2, REQUIRED: atoms of each element constant within 0.001 %", atoms <= 1e-5, f"largest change {atoms:.1e}")
            k = int(np.argmax(r["T"] >= g["T_ign"]))
            state_at_ignition = dict(T=float(r["T"][k]), sigma=r["sigma"][k].copy())
            # the answer must not depend on the step: 1e-6 s against the independent solution read at the same times
            big = run_parcel(mech, g, moles, dt=1e-6, **TIGHT)
            t_ind = chem.crossing_time(big["t"], ind["sol"](big["t"])[-1], g["T_ign"])
            fine_at = r["T"][::100]
            dT = float(np.max(np.abs(big["T"] - fine_at)))
            REPORT["pressure_step_1e-6"] = dict(slim(big), t_ign_independent_same_reading=t_ind, d_t=big["t_ign"] / t_ind - 1, dT_max_against_small_step=dT)
            check("rung 2: the answer does not depend on the step (1e-6 s against 1e-8 s; ignition time by the same reading within 1e-6)",
                  abs(big["t_ign"] / t_ind - 1) <= 1e-6 and abs(big["T_end"] - r["T_end"]) <= 1e-3,
                  f"ignition time read at steps of 1e-6 s: kernel {1e6 * big['t_ign']:.5f} us, cfdlib.chem read the same way {1e6 * t_ind:.5f} us "
                  f"({big['t_ign'] / t_ind - 1:+.1e}); final temperature differs by {big['T_end'] - r['T_end']:+.1e} K; largest difference of the "
                  f"temperature at the common times {dT:.1e} K")
        else:
            of = json.loads((ROOT / "results" / "h2_air_ignition_box" / "results.json").read_text(encoding="utf-8"))["data"]["study"]
            rec["openfoam_step_to_zero"] = dict(t_ign=of["t_ign"]["extrap"], T_end=of["T_end"]["extrap"])
            print(f"      recorded: OpenFOAM extrapolated to a step of zero {1e6 * of['t_ign']['extrap']:.4f} us, {of['T_end']['extrap']:.4f} K; "
                  f"gpucfd differs by {r['t_ign'] / of['t_ign']['extrap'] - 1:+.1e} and {r['T_end'] - of['T_end']['extrap']:+.1e} K", flush=True)

    # ---- constant volume ----
    mech = mechs["CHEMKIN"]
    parcel = chem.VolumeParcel(mech, g["p"], g["T_0"], moles)
    T_cc, s_cc = parcel.complete_combustion()
    q = parcel.u0 - parcel.energy(s_cc, g["T_0"])
    ind = parcel.solve(g["t_end"], T_ign=g["T_ign"], rtol=1e-10, atol=1e-20)
    p_ind = float(parcel.pressure(ind["sigma"][-1], ind["T"][-1]))
    u_drift = max(abs(parcel.energy(ind["sigma"][k], ind["T"][k]) - parcel.u0) for k in range(0, len(ind["t"]), 50)) / q
    r = run_parcel(mech, g, moles, constant="volume", **TIGHT)
    cons = conserved(r, mech, parcel, q)
    health = chem.parcel_health(r["T"], r["Y"], mech, g["T_0"], T_cc, tol_sum=1e-12, tol_atoms=1e-12)
    d_t, d_T, d_p = r["t_ign"] / ind["t_ign"] - 1, r["T_end"] - float(ind["T"][-1]), r["p_end"] / p_ind - 1
    REPORT["volume"] = dict(slim(r), independent=dict(t_ign=ind["t_ign"], T_end=float(ind["T"][-1]), p_end=p_ind, energy_drift=u_drift), d_ind_t=d_t,
                            d_ind_T=d_T, d_ind_p=d_p, T_cc=T_cc, q=q, health=health["checks"], **cons)
    print(f"constant volume: ignition {1e6 * r['t_ign']:.4f} us (cfdlib.chem {1e6 * ind['t_ign']:.4f}), final {r['T_end']:.4f} K ({float(ind['T'][-1]):.4f}), "
          f"final pressure {r['p_end']:.2f} Pa ({p_ind:.2f}), {r['p_end'] / g['p']:.4f} times the start; cfdlib.chem's own energy drift {u_drift:.1e}", flush=True)
    check("rung 2: constant volume, status and health", r["status"] == 0 and health["ok"], "; ".join(f"{c['name']}: {c['value']}" for c in health["checks"]))
    check("rung 2: constant volume against cfdlib.chem (ignition time within 1e-6, final temperature within 1e-3 K, final pressure within 1e-8)",
          abs(d_t) <= 1e-6 and abs(d_T) <= 1e-3 and abs(d_p) <= 1e-8, f"ignition time {d_t:+.1e}, final temperature {d_T:+.1e} K, final pressure {d_p:+.1e}")
    check("rung 2: constant volume, atoms within 1e-12, internal energy within 1e-10 of the heat released, and on the data's joint within its jump",
          cons["atoms"] <= 1e-12 and cons["energy"] <= 1e-10 and cons["energy_on_joint"] <= 1.01 * cons["jump"] + 1e-10, energy_line(cons, q, "internal energy"))

    # ---- order of each integrator: fixed sub-steps over 0.2 microseconds of the rise, from the state at ignition ----
    # Sub-steps of 1e-8 s and less: the fastest reactions there act in about 1e-8 s, and above that the explicit
    # methods are unstable (5e-8 s breaks down) and no method shows its order yet. Judged on the last refinement;
    # the fifth-order method reaches round-off (3e-14) there and comes down to 5 from above.
    span, orders = 2e-7, {}

    def fixed(method, n):
        c = Chemistry(mech, 1, constant="pressure", precision="double", method=method)
        c.control(fixed=n)
        c.set(T=state_at_ignition["T"], p=g["p"], sigma=state_at_ignition["sigma"])
        c.advance(span, 1)
        s = c.get()
        c.close()
        return np.append(s["sigma"][0], s["T"][0])
    exact = fixed("dp45", 20000)
    scale = np.append(np.maximum(np.abs(exact[:-1]), 1e-3 * np.abs(exact[:-1]).max()), exact[-1])
    for method, want, counts in (("bs23", 3, (20, 40, 80, 160)), ("dp45", 5, (10, 20, 40, 80)), ("ros34", 4, (20, 40, 80, 160))):
        errs = [float(np.max(np.abs(fixed(method, n) - exact) / scale)) for n in counts]
        obs = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
        orders[method] = dict(substeps=list(counts), errors=errs, orders=obs)
        check(f"rung 2: order of {method} with fixed sub-steps is {want} (within 0.4, last refinement)", abs(obs[-1] - want) <= 0.4,
              "errors " + ", ".join(f"{e:.2e}" for e in errs) + "; orders " + ", ".join(f"{o:.2f}" for o in obs))
    REPORT["orders"] = dict(span=span, start=dict(T=state_at_ignition["T"]), methods=orders)

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "test_chem_parcel.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
