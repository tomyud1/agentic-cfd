"""Chemistry ladder, rung 3, speed: cells times steps per second, and memory.

    python -m gpucfd.tests.test_chem_speed          (inside WSL, from the project root; nothing else on the GPU)

Batches of cells are advanced with the wrapper's default integrator and tolerances at a flow-sized step (1e-8 s)
and at a large one (1e-6 s), in single, mixed and double precision, and timed with the GPU finished at both ends.
The cells of a batch are
  - all in one state taken from case 20's parcel: before ignition (at 100 microseconds), in the rise (when the
    temperature passes 1200, 1500 and 1800 K) or burnt (at 600 microseconds); or
  - a mixed population: the 4096 cells of rung 3 (start temperatures 1000 to 1400 K) after 50 microseconds, when
    some have burnt, some are igniting and some have not started, repeated to fill the batch. "tiled" keeps the
    cells interleaved (neighbours do different amounts of work); "sorted" puts equal cells next to each other.
Every timing: set the state, one step to settle the sub-step size, then the timed steps; best of three. The
number of timed steps is sized from one trial step for about a second, at most 50 steps (3 at the large step), so
that a batch stays in its state. A batch size is skipped when one step takes more than 10 s.

Speed is recorded, not judged (plan: dev/compare/h2_air_ignition_box/plan.yaml). Judged: no cell breaks, and the
memory per cell is what the kernel says it allocates. OpenFOAM's recorded cost per step and phase of case 20 is
read from results/h2_air_ignition_box/runs/ and stored next to the numbers.
"""
import json
import sys
from pathlib import Path

import numpy as np

from cfdlib import chem, foam
from gpucfd import chemistry
from gpucfd.chemistry import Chemistry
from gpucfd.tests.test_chem_batch import N as NPOP, T_HIGH, T_LOW
from gpucfd.tests.test_chem_parcel import ROOT, TIGHT, given, run_parcel

FAILS = []
REPORT = {}
SIZES = (1000, 10000, 100000, 1000000, 4000000, 16000000)
STEPS = {1e-8: 50, 1e-6: 3}


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def states(mech, g, moles):
    """The single states (name -> temperature, mol/kg) and the mixed population (4096 temperatures, 4096 mixtures)."""
    r = run_parcel(mech, g, moles, dt=1e-6, **TIGHT)
    at = lambda k: (float(r["T"][k]), r["sigma"][k].copy())                                              # noqa: E731
    out = {"before ignition": at(100), "burnt": at(600)}
    for level in (1200, 1500, 1800):
        out[f"rise, {level} K"] = at(int(np.argmax(r["T"] >= level)))
    c = Chemistry(mech, NPOP, precision="double")
    T0 = np.linspace(T_LOW, T_HIGH, NPOP)
    c.set(T=T0, p=g["p"], moles=moles)
    c.advance(1e-7, 500)
    s = c.get()
    c.close()
    share = dict(not_started=float(np.mean(s["T"] < T0 + 10)), burnt=float(np.mean(s["T"] > 2000)))
    share["igniting"] = 1 - share["not_started"] - share["burnt"]
    return out, (s["T"], s["sigma"]), share


def timed(mech, g, state, n, precision, dt, order="tiled", **control):
    """One timing. state: (T, sigma) of one cell or of a population that is repeated to n cells."""
    T, sigma = state
    if np.ndim(T):
        idx = np.arange(n) % len(T)
        if order == "sorted":
            idx = np.sort(idx)
        T, sigma = T[idx], sigma[idx]
    c = Chemistry(mech, n, precision=precision, **control)

    def fresh():
        c.set(T=T, p=g["p"], sigma=sigma)
        c.advance(dt, 1)
    fresh()
    trial = c.advance(dt, 1)["ms"]
    if trial > 1e4:
        c.close()
        return None
    steps = int(min(STEPS[dt], max(1, 1000 / trial)))
    best = 1e300
    for _ in range(3):
        fresh()
        best = min(best, c.advance(dt, steps)["ms"])
    st, bytes_ = c.stats(), c.bytes
    setting = dict(method=c.method, rtol=c.rtol, atol=c.atol)
    c.close()
    rate = n * steps / (best / 1e3)
    return dict(cells=n, precision=precision, dt=dt, steps=steps, order=order, ms_per_step=best / steps, cell_steps_per_s=rate, ns_per_cell_step=1e9 / rate,
                substeps_per_step=float(st["substeps"].mean() / (steps + 1)), rejected_per_step=float(st["rejected"].mean() / (steps + 1)),
                broke=int((st["status"] > 0).sum()), bytes_per_cell=bytes_ / n, **setting)


def show(tag, r):
    print(f"  {tag:34s} {r['cells']:9d} cells {r['precision']:6s} dt {r['dt']:g}: {r['cell_steps_per_s'] / 1e6:9.3f} million cells x steps / s, "
          f"{r['ms_per_step']:9.3f} ms per step, {r['substeps_per_step']:7.2f} sub-steps per cell and step ({r['method']})", flush=True)


def openfoam():
    """Case 20's recorded cost: microseconds of one core per step, by phase (slow start, rise, burnt gas)."""
    out = {}
    for name in ("S80000", "S10000", "S2500", "S10000_RKF45", "S10000_rel0.1_abs1e-12"):
        r = json.loads((ROOT / "results" / "h2_air_ignition_box" / "runs" / f"{name}.json").read_text(encoding="utf-8"))
        us = [1e6 * s / n for s, n in zip(r["phase_s"], r["phase_steps"])]
        out[name] = dict(steps=r["steps"], dt=1e-3 / r["steps"], solver=r["ode"], rel=r["rel"], abs=r["abs"], us_per_step=r["us_per_step"], phase_us=us,
                         cell_steps_per_s=[1e6 / u for u in us], wall_s=r["wall_time"])
    return out


def large_step(mech, g, pop):
    """Does the tuning's winner at steps of 1e-6 s hold on a full GPU? The tuning timed 4096 cells, where a step
    takes as long as its slowest cell. Here: that setting, and the default integrator with the same tolerances, on
    a million cells of the mixed population, in double (as tuned) and in the default precision."""
    won = json.loads((Path(__file__).parent / "results" / "test_chem_tune.json").read_text())["chosen"]["1e-06"]
    rows = []
    for precision in dict.fromkeys(("double", chemistry.DEFAULT["precision"])):
        for method in dict.fromkeys((won["method"], chemistry.DEFAULT["method"])):
            r = timed(mech, g, pop, 1000000, precision, 1e-6, method=method, rtol=won["rtol"], atol=won["atol"])
            r["state"] = "mixed population"
            rows.append(r)
            show(f"tuned for 1e-6 s: {method}, rtol {won['rtol']:g}", r)
    return rows


def main():
    g, moles, _, _ = given()
    mech = chem.build()
    busy = [p[3][:60] for p in foam.running()]
    single, pop, share = states(mech, g, moles)
    out = Path(__file__).parent / "results"
    if sys.argv[1:] == ["large"]:                     # only the section added after the first full run; merged into the record
        rec = json.loads((out / "test_chem_speed.json").read_text())
        rec["large_step"] = large_step(mech, g, pop)
        (out / "test_chem_speed.json").write_text(json.dumps(rec, indent=1, default=float))
        return
    REPORT.update(busy_cpu=bool(busy), busy_processes=busy, default=dict(chemistry.DEFAULT), population_share=share,
                  states={k: dict(T=v[0]) for k, v in single.items()}, openfoam=openfoam(), scan=[], phases=[], methods=[], order=[])
    print(f"mixed population after 50 us: {100 * share['not_started']:.0f} % not started, {100 * share['igniting']:.0f} % igniting, {100 * share['burnt']:.0f} % burnt", flush=True)
    rows = []

    # ---- against the number of cells ----
    for tag, state in (("before ignition", single["before ignition"]), ("mixed population", pop)):
        for precision in ("float", "mixed", "double"):
            for dt in (1e-8, 1e-6):
                for n in SIZES:
                    r = timed(mech, g, state, n, precision, dt)
                    if r is None:
                        print(f"  {tag}, {n} cells, {precision}, dt {dt:g}: one step takes more than 10 s; larger batches skipped", flush=True)
                        break
                    r["state"] = tag
                    REPORT["scan"].append(r); rows.append(r)
                    show(tag, r)
    # ---- by state, a million cells ----
    for tag, state in single.items():
        for precision in ("float", "mixed", "double"):
            for dt in (1e-8, 1e-6):
                r = timed(mech, g, state, 1000000, precision, dt)
                if r is None:
                    continue
                r["state"] = tag
                REPORT["phases"].append(r); rows.append(r)
                show(tag, r)
    # ---- the three integrators with the default tolerances, and sorted cells, on the mixed population ----
    for precision in dict.fromkeys(("double", chemistry.DEFAULT["precision"])):
        for dt in (1e-8, 1e-6):
            for method in ("bs23", "dp45", "ros34"):
                r = timed(mech, g, pop, 1000000, precision, dt, method=method)
                if r is None:
                    continue
                r["state"] = "mixed population"
                REPORT["methods"].append(r); rows.append(r)
                show(f"mixed population, {method}", r)
    for dt in (1e-8, 1e-6):
        for order in ("tiled", "sorted"):
            r = timed(mech, g, pop, 1000000, chemistry.DEFAULT["precision"], dt, order=order)
            r["state"] = "mixed population"
            REPORT["order"].append(r); rows.append(r)
            show(f"mixed population, {order}", r)
    REPORT["large_step"] = large_step(mech, g, pop)
    rows += REPORT["large_step"]

    check("rung 3, speed: no cell broke in any timing", all(r["broke"] == 0 for r in rows), f"{len(rows)} timings")
    ns = len(mech.species)
    want = {"float": (ns + 3) * 4 + 21, "mixed": (ns + 3) * 8 + 21, "double": (ns + 3) * 8 + 21}
    off = max(abs(r["bytes_per_cell"] / want[r["precision"]] - 1) for r in rows)
    check("rung 3, speed: memory per cell is the species, three more numbers and 21 bytes of book-keeping", off < 1e-12,
          f"{want['float']} bytes in single, {want['mixed']} in mixed and double precision ({ns} species)")
    REPORT["bytes_per_cell"] = want
    out.mkdir(exist_ok=True)
    (out / "test_chem_speed.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
