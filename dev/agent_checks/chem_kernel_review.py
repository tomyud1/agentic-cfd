"""A file of the developing agent: an independent check of gpucfd's chemistry kernel (gpucfd/CHEMISTRY.md) at
the state its own ladder did not cover: the gas right behind the shock of a detonation (step 4 of the
detonation-engine ladder, case 21). The ladder tested 2 atm and 1000 to 1400 K; here the start is the state
behind a shock of the Chapman-Jouguet speed in hydrogen and air (about 28 atm and 1540 K), burning at constant
volume, as one cell of a flow solver does during the chemistry part of a time step.

    PYTHONPATH=. python dev/agent_checks/chem_kernel_review.py      (inside WSL; GPU; under a minute)

Reference: cfdlib.chem.VolumeParcel with SciPy's Radau at relative tolerance 1e-10 (good to about 3e-7 in an
ignition time, measured in CHEMISTRY.md).

Written before the run.
Pass rules (kernel in double with tolerance 1e-8, and with its default setting; time steps of 2.5e-9 s):
  - the time at which the temperature first reaches 2000 K is within 0.1 % of the reference;
  - temperature and pressure at the end (2 microseconds) are within 0.01 % of the reference;
  - every cell reports status 0, the 16 cells give identical numbers, atoms change by less than 1e-12.
Expected, not required: the 2000 K time is between 0.2 and 2 microseconds; the end state is near 3400 K at
about twice the start pressure; at steps of 1e-8 s the kernel needs more than one sub-step per step in the
rise (the chemistry is faster at 28 atm than at 2 atm); a million cells that all ignite together advance at
50 to 200 million cells x steps per second at steps of 5e-9 s (the step case 21's trial took on 0.1 mm cells).

Found (2026-11-15): every rule met (2000 K at 0.41869 us in the reference; the kernel -0.005 %, end state to
1e-7, atoms 1e-15). The speed expectation is missed: 18 to 19 million while igniting (10 sub-steps per step)
and 16 million once burnt (11 sub-steps; timed after the first run raised the question). The first run of this
file failed on its own atom count: it divided by zero for an element the mixture does not hold; rules unchanged.
"""
import json
import time
from pathlib import Path

import numpy as np

from cfdlib import chem
from gpucfd.chemistry import DEFAULT, Chemistry

OUT = Path(__file__).with_suffix(".json")
P1, T1, MIX = 101325.0, 300.0, dict(H2=2.0, O2=1.0, N2=3.76)
LEVEL, T_END = 2000.0, 2.0e-6
SETTINGS = {"double, tolerance 1e-8": dict(precision="double", rtol=1e-8, atol=1e-14), "default (mixed, 1e-6)": dict(DEFAULT)}


def main():
    mech = chem.build()
    cj = chem.cj_detonation(mech, P1, T1, MIX)
    vn = chem.shock_frozen(mech, P1, T1, MIX, cj["D"])
    ref = chem.VolumeParcel(mech, vn["p"], vn["T"], MIX)
    r = ref.solve(T_END, T_ign=LEVEL)
    T_ref, s_ref = float(r["T"][-1]), r["sigma"][-1]
    p_ref = float(ref.pressure(s_ref, T_ref))
    atoms0 = mech.comp.T @ ref.sigma0
    print(f"start: behind a shock of {cj['D']:.1f} m/s: {vn['p'] / 101325:.2f} atm, {vn['T']:.1f} K")
    print(f"reference (cfdlib.chem.VolumeParcel): 2000 K at {1e6 * r['t_ign']:.5f} us; at {1e6 * T_END:.0f} us {T_ref:.3f} K, {p_ref / vn['p']:.4f} times the start pressure")
    rows, ok = [], True
    for name, kw in SETTINGS.items():
        for dt in (1e-8, 2.5e-9):
            steps = int(round(T_END / dt))
            c = Chemistry(mech, 16, constant="volume", **kw)
            c.control(level=LEVEL)
            c.set(T=vn["T"], p=vn["p"], moles=MIX)
            c.advance(dt, steps)
            s, st = c.get(), c.stats()
            c.close()
            have = atoms0 > 0                                    # the scheme knows elements this mixture does not hold
            now = s["sigma"] @ mech.comp
            atoms = float(max(np.max(np.abs(now[:, have] / atoms0[have] - 1)), np.max(np.abs(now[:, ~have]), initial=0.0)))
            row = dict(setting=name, dt=dt, steps=steps, t_cross=float(st["t_cross"][0]), d_t=float(st["t_cross"][0] / r["t_ign"] - 1), T_end=float(s["T"][0]),
                       d_T=float(s["T"][0] / T_ref - 1), d_p=float(s["p"][0] / p_ref - 1), substeps_per_step=float(st["substeps"][0] / steps), rejected=int(st["rejected"][0]),
                       status=int(st["status"].max()), identical=bool(np.all(s["T"] == s["T"][0]) and np.all(st["t_cross"] == st["t_cross"][0])), atoms=atoms)
            row["judged"] = dt == 2.5e-9
            row["pass"] = bool(abs(row["d_t"]) <= 1e-3 and abs(row["d_T"]) <= 1e-4 and abs(row["d_p"]) <= 1e-4 and row["status"] == 0 and row["identical"] and atoms < 1e-12)
            ok &= row["pass"] or not row["judged"]
            rows.append(row)
            print(f"{'PASS' if row['pass'] else ('FAIL' if row['judged'] else 'INFO')}  {name}, steps of {dt:.1e} s: 2000 K at {1e6 * row['t_cross']:.5f} us ({100 * row['d_t']:+.4f} %), "
                  f"end {row['T_end']:.3f} K ({100 * row['d_T']:+.5f} %), pressure {100 * row['d_p']:+.5f} %, {row['substeps_per_step']:.2f} sub-steps per step, "
                  f"{row['rejected']} rejected, status {row['status']}, identical {row['identical']}, atoms {atoms:.1e}", flush=True)
    # speed where it is hardest: a million cells that all ignite together, the whole ignition, the trial's time step
    n, dt, steps = 1_000_000, 5e-9, 200
    c = Chemistry(mech, n, constant="volume")
    c.set(T=vn["T"], p=vn["p"], moles=MIX)
    c.advance(dt, 1)                                             # first call: start-up of the kernel
    c.set(T=vn["T"], p=vn["p"], moles=MIX)
    t0 = time.perf_counter()
    out = c.advance(dt, steps)
    wall = time.perf_counter() - t0
    st, s = c.stats(), c.get()
    speed = dict(cells=n, dt=dt, steps=steps, gpu_ms=out["ms"], wall_s=wall, cell_steps_per_s=n * steps / (out["ms"] / 1e3), substeps_per_step=float(st["substeps"].mean() / steps),
                 status=int(st["status"].max()), T_end=float(s["T"][0]))
    print(f"INFO  a million cells igniting together, {steps} steps of {dt:.1e} s (to {1e6 * dt * steps:.0f} us, {speed['T_end']:.0f} K): {speed['cell_steps_per_s'] / 1e6:.0f} million cells x steps "
          f"per second, {speed['substeps_per_step']:.2f} sub-steps per step, status {speed['status']}")
    # added after the first run (a question it raised): what do the same cells cost once they are burnt?
    out2 = c.advance(dt, steps)
    st2, s2 = c.stats(), c.get()
    c.close()
    speed["burnt"] = dict(steps=steps, gpu_ms=out2["ms"], cell_steps_per_s=n * steps / (out2["ms"] / 1e3), substeps_per_step=float((st2["substeps"].mean() - st["substeps"].mean()) / steps),
                          status=int(st2["status"].max()), T_end=float(s2["T"][0]), p_end_atm=float(s2["p"][0] / 101325))
    b = speed["burnt"]
    print(f"INFO  the same cells, burnt ({steps} more steps, {b['T_end']:.0f} K, {b['p_end_atm']:.1f} atm): {b['cell_steps_per_s'] / 1e6:.0f} million cells x steps per second, "
          f"{b['substeps_per_step']:.2f} sub-steps per step, status {b['status']}")
    OUT.write_text(json.dumps(dict(start=dict(p=vn["p"], T=vn["T"], D=cj["D"]), reference=dict(t_2000=r["t_ign"], T_end=T_ref, p_end=p_ref, p_ratio=p_ref / vn["p"]),
                                   rows=rows, speed=speed, passed=bool(ok)), indent=1, default=float), encoding="utf-8")
    print(f"{'PASS' if ok else 'FAIL'}  the kernel at the state behind a detonation's shock | record: {OUT.name}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
