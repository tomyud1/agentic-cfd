"""Reacting ladder, rung (c): uniform gas at rest that ignites. Flow and chemistry are both switched on; every
cell must give the answer of one parcel at constant volume, and nothing may start to move.

    python -m gpucfd.tests.test_react_ignite          (inside WSL, from the project root; about two minutes)

The mixture of case 20 (H2 : O2 : N2 = 1 : 1 : 3.76) at 1000 K and 2 atm in a closed box, 64 cells in one
dimension and 32 x 32 in two. Reference: cfdlib.chem.VolumeParcel with the same constants.
Pass rules (dev/compare/detonation_tube_h2_air/plan.yaml, rung c): ignition time (first crossing of 1200 K,
straight line over the step) within 1e-4; temperature at 1 ms within 0.1 K and pressure within 1e-5; the
highest gas speed exactly zero and highest and lowest temperature the same number after every step; mass, atoms
and energy within 1e-12; every cell's chemistry status 0; the guard not used.
"""
import json
import sys
from pathlib import Path

import numpy as np

from cfdlib import chem
from gpucfd import reacting
from gpucfd.reacting import Reacting

OUT = Path(__file__).parent / "results"
FAILS = []
REPORT = {}
P0, T0, MIX, LEVEL = 202650.0, 1000.0, dict(H2=1.0, O2=1.0, N2=3.76), 1200.0


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def box(mech, n, dt, steps, **chemistry):
    lengths = tuple(1e-3 * k for k in n)
    sides = {a + s: "wall" for a, k in zip("xyz", n) if k > 1 for s in "-+"}
    s = Reacting(mech, n, lengths, sides=sides, chemistry=chemistry or True)
    s.set(T=T0, p=P0, moles=MIX)
    start = s.totals()
    st = s.run_fixed(dt, steps, history=True)
    f, cs, guard = s.get(), s.chem_stats(), s.guard()
    setting = dict(s.chem)
    s.close()
    t = dt * np.arange(steps + 1)
    T = np.concatenate([[T0], st["T_max"]])
    drift = lambda key: float(np.max(np.abs(st[key] / start[key] - 1)))                                     # noqa: E731
    used = start["atoms"] > 0
    return dict(n=list(n), dt=dt, steps=steps, setting=setting, gpu_s=st["ms"] / 1e3, t_ign=chem.crossing_time(t, T, LEVEL), T_end=float(f["T"].max()),
                p_end=float(f["p"].max()), still=float(st["speed_max"].max()), spread_T=float(np.max(st["T_max"] - st["T_min"])),
                spread_end=float(max(np.ptp(f["T"]), np.ptp(f["p"]) / P0, np.ptp(f["Y"], axis=(1, 2, 3)).max())),
                mass=drift("mass"), energy=drift("energy"), atoms=float(np.max(np.abs(st["atoms"][:, used] / start["atoms"][used] - 1))),
                status=int(cs["status"].max()), substeps=float(cs["substeps"].mean() / steps), guard=guard)


def main():
    mech = chem.build()
    parcel = chem.VolumeParcel(mech, P0, T0, MIX)
    ref = parcel.solve(1e-3, T_ign=LEVEL)
    t_ref, T_ref, p_ref = ref["t_ign"], float(ref["T"][-1]), float(parcel.pressure(ref["sigma"][-1], ref["T"][-1]))
    print(f"reference (cfdlib.chem.VolumeParcel): ignition {1e6 * t_ref:.4f} microseconds, {T_ref:.4f} K and {p_ref:.2f} Pa at 1 ms", flush=True)
    REPORT.update(reference=dict(t_ign=t_ref, T_end=T_ref, p_end=p_ref), runs=[])
    settings = [("the chemistry kernel's default", dict(method="bs23", rtol=1e-6, quiet=0.0))]
    if reacting.CHEMISTRY["method"] != "bs23" or reacting.CHEMISTRY["quiet"]:
        settings.append(("the joined solver's setting", {}))
    for tag, setting in settings:
        for n in ((64, 1, 1), (32, 32, 1)):
            a = box(mech, n, 1e-8, 30000, **setting)              # ignition
            b = box(mech, n, 1e-7, 10000, **setting)              # the end state
            REPORT["runs"].append(dict(tag=tag, ignition=a, end=b))
            name = f"c {n[0]} x {n[1]} cells, {tag} ({a['setting']['method']}, tolerance {a['setting']['rtol']:.0e}, quiet {a['setting']['quiet']:g})"
            d_t, d_T, d_p = a["t_ign"] / t_ref - 1, b["T_end"] - T_ref, b["p_end"] / p_ref - 1
            check(f"{name}: every cell gives the parcel's answer", abs(d_t) <= 1e-4 and abs(d_T) <= 0.1 and abs(d_p) <= 1e-5,
                  f"ignition {1e6 * a['t_ign']:.4f} microseconds ({d_t:+.1e}; limit 1e-4), at 1 ms {b['T_end']:.4f} K ({d_T:+.1e} K; limit 0.1) and "
                  f"{b['p_end']:.2f} Pa ({d_p:+.1e}; limit 1e-5); {a['substeps']:.2f} and {b['substeps']:.2f} sub-steps per step")
            check(f"{name}: nothing moves", max(a["still"], b["still"]) == 0 and max(a["spread_T"], b["spread_T"]) == 0 and max(a["spread_end"], b["spread_end"]) == 0,
                  f"highest gas speed {max(a['still'], b['still']):.1e} m/s, highest minus lowest temperature {max(a['spread_T'], b['spread_T']):.1e} K at every step")
            worst = {k: max(a[k], b[k]) for k in ("mass", "atoms", "energy")}
            unused = all(v == 0 for r in (a, b) for v in (r["guard"]["first_order"], r["guard"]["species_flat"], r["guard"]["species"], r["guard"]["cold"]))
            check(f"{name}: mass, atoms and energy kept, chemistry healthy, guard not used", max(worst.values()) <= 1e-12 and max(a["status"], b["status"]) == 0 and unused,
                  f"mass {worst['mass']:.1e}, atoms {worst['atoms']:.1e}, energy {worst['energy']:.1e}; status {max(a['status'], b['status'])}; guard {b['guard']}")
    OUT.mkdir(exist_ok=True)
    (OUT / "test_react_ignite.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
