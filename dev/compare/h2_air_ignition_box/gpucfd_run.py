"""Case 20 with gpucfd's chemistry kernel: the ignition of the hydrogen mixture, with the outputs the case asks
for, read by the case's own rules, next to OpenFOAM's recorded numbers. The plan (pass rules, expectations, the
tuning budget, what each side is given) is plan.yaml, written before the runs.

    bash dev/compare/run.sh h2_air_ignition_box gpucfd_run.py     (or: PYTHONPATH=. python dev/compare/h2_air_ignition_box/gpucfd_run.py)

Writes gpucfd.json next to this file, in the form cfdlib.brief reads (answers, runs, plain). The answers are
those of one parcel in 80000 steps (OpenFOAM's finest run) in double precision with relative tolerance 1e-8
(OpenFOAM's), with CHEMKIN's constants, which are the reference's own. Next to them: the same with OpenFOAM's
constants, the same with the kernel's default setting, and the speed numbers of gpucfd/tests/test_chem_speed.py.
Run the ladder first (bash gpucfd/run_ladder.sh chemistry): this script reads its speed record.
"""
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import yaml

from cases.h2_air_ignition_box import measure
from cfdlib import chem, foam
from cfdlib.report import lines, m, m_pct, tex_num
from gpucfd import chemistry, gcfd
from gpucfd.chemistry import Chemistry
from gpucfd.tests import test_chem_parcel as P

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "gpucfd.json"
RES = ROOT / "results" / "h2_air_ignition_box"
LIKE = dict(precision="double", rtol=1e-8, atol=1e-14)             # like for like with OpenFOAM's tolerance
US = r"\,\mathrm{\mu s}"
sc = lambda x, sig=1: m(tex_num(x, sig)) if x else m("0")                                                 # noqa: E731


def outputs(r, mech, g, ref, th, t_ref, T_ref, numerics):
    """What the case asks for, from one history, by the case's rules (cases/h2_air_ignition_box/numbers.py)."""
    parcel = chem.Parcel(mech, g["p"], g["T_0"], dict(H2=g["x_H2"], O2=g["x_O2"], N2=g["x_N2"]))
    q = parcel.h0 - float(parcel.complete_combustion()[1] @ mech.h(g["T_0"]))
    cons = P.conserved(r, mech, parcel, q)
    levels = [str(int(L)) for L in numerics["extraction"]["levels"]]
    mine = measure.crossings(r["t"], r["T"], numerics["extraction"]["levels"])
    shift = {L: mine[L] - th["reference"]["levels"][L] for L in levels}
    steep = [L for L in levels if int(L) <= 2000]
    worst = max(steep, key=lambda L: abs(shift[L]))
    rt, rT = np.array(ref["history"]["t_s"]), np.array(ref["history"]["T_K"])
    flat = (rT < g["T_0"] + 10) | (rT > 2100)
    dT = np.interp(rt, r["t"], r["T"]) - rT
    return dict(t_ign=r["t_ign"], T_end=r["T_end"], d_ref_t=r["t_ign"] / t_ref - 1, d_ref_T=r["T_end"] / T_ref - 1, levels=mine, shift=shift, shift_worst=shift[worst],
                dT_flat_max=float(np.max(np.abs(dT[flat]))), atoms=cons["atoms"], energy=cons["energy_on_joint"], energy_away_from_joint=cons["energy"],
                data_jump=cons["jump"], Y_min=cons["Y_min"], sumY=cons["sumY"], substeps=r["substeps"], rejected=r["rejected"], status=r["status"],
                steps=r["steps"], wall_s=r["wall_s"], gpu_s=r["gpu_ms"] / 1e3, precision=r["precision"], setting=r["setting"])


def main():
    plan = yaml.safe_load((HERE / "plan.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((ROOT / "cases" / "h2_air_ignition_box" / "numerics.yaml").read_text(encoding="utf-8"))
    ref = yaml.safe_load((ROOT / "cases" / "h2_air_ignition_box" / "reference" / "senkin_h2.yaml").read_text(encoding="utf-8"))
    th = json.loads((RES / "theory.json").read_text(encoding="utf-8"))
    of = json.loads((RES / "results.json").read_text(encoding="utf-8"))["data"]
    speed = json.loads((ROOT / "gpucfd" / "tests" / "results" / "test_chem_speed.json").read_text(encoding="utf-8"))
    g, moles, t_ref, T_ref = P.given()
    mechs = {"CHEMKIN": chem.build(), "OpenFOAM": chem.build(chem.openfoam_conventions())}
    busy = [p[3][:60] for p in foam.running()]
    t0 = time.perf_counter()
    name = gcfd.device()["name"]
    c = Chemistry(mechs["CHEMKIN"], 16)
    c.set(T=g["T_0"], p=g["p"], moles=moles)
    c.advance(1e-8, 1)
    c.close()
    startup = time.perf_counter() - t0
    print(f"GPU: {name}; start-up {startup:.2f} s; other solver processes on the CPU: {len(busy)}", flush=True)

    def parcel(mech, steps, **kw):
        r = P.run_parcel(mech, g, moles, dt=g["t_end"] / steps, **kw)
        r["setting"] = dict(chemistry.DEFAULT, **{k: v for k, v in kw.items() if k in chemistry.DEFAULT})
        return r

    # ---- the answers: 80000 steps, like for like; then OpenFOAM's constants; then the default setting ----
    fine = of["runs"]["S80000"]
    main_ = outputs(parcel(mechs["CHEMKIN"], 80000, **LIKE), mechs["CHEMKIN"], g, ref, th, t_ref, T_ref, numerics)
    same_constants = outputs(parcel(mechs["OpenFOAM"], 80000, **LIKE), mechs["OpenFOAM"], g, ref, th, t_ref, T_ref, numerics)
    default = outputs(parcel(mechs["CHEMKIN"], 80000), mechs["CHEMKIN"], g, ref, th, t_ref, T_ref, numerics)
    for tag, x in (("like for like, CHEMKIN's constants", main_), ("like for like, OpenFOAM's constants", same_constants), ("default setting", default)):
        print(f"{tag}: ignition {1e6 * x['t_ign']:.4f} us ({100 * x['d_ref_t']:+.4f} % from the reference), final {x['T_end']:.4f} K ({100 * x['d_ref_T']:+.5f} %), atoms "
              f"{x['atoms']:.1e}, energy {x['energy']:.1e}, {x['substeps']} sub-steps, {x['wall_s']:.1f} s", flush=True)
    ex = of["study"]
    print(f"OpenFOAM, 80000 steps: {1e6 * fine['t_ign']:.4f} us, {fine['T_end']:.4f} K; extrapolated to a step of zero {1e6 * ex['t_ign']['extrap']:.4f} us, "
          f"{ex['T_end']['extrap']:.4f} K; gpucfd with OpenFOAM's constants differs from that by {same_constants['t_ign'] / ex['t_ign']['extrap'] - 1:+.1e} and "
          f"{same_constants['T_end'] - ex['T_end']['extrap']:+.1e} K", flush=True)
    requirements = dict(ignition=abs(main_["d_ref_t"]) <= 0.01, final=abs(main_["d_ref_T"]) <= 0.001, conserved=main_["atoms"] <= 1e-5)

    # ---- the same runs as OpenFOAM's speed rows: one parcel, a row of output at every step ----
    runs = []
    for steps in (2500, 10000, 40000, 80000):
        r = parcel(mechs["CHEMKIN"], steps, **LIKE)
        o = of["runs"][f"S{steps}"]
        runs.append(dict(steps=steps, cells=1, wall_s=r["wall_s"], gpu_s=r["gpu_ms"] / 1e3, substeps=r["substeps"], t_ign=r["t_ign"], T_end=r["T_end"],
                         openfoam=dict(wall_s=o["wall_time"], t_ign=o["t_ign"], T_end=o["T_end"]), **LIKE))
        print(f"one parcel, {steps} steps: {r['wall_s']:.1f} s (OpenFOAM {o['wall_time']:.1f} s on one core); ignition {1e6 * r['t_ign']:.4f} us (OpenFOAM "
              f"{1e6 * o['t_ign']:.4f}); {r['substeps']} sub-steps", flush=True)
    # 4096 parcels at once, the default setting, the same 80000 steps (no history)
    t0 = time.perf_counter()
    c = Chemistry(mechs["CHEMKIN"], 4096)
    c.control(level=g["T_ign"])
    c.set(T=g["T_0"], p=g["p"], moles=moles)
    out = c.advance(g["t_end"] / 80000, 80000)
    s, st = c.get(), c.stats()
    c.close()
    many = dict(label="4096 parcels at once, 80000 time steps (gpucfd's default setting)", cells=4096, time_steps=80000, wall_s=time.perf_counter() - t0,     # not "steps": that key would put it in the one-parcel row
 gpu_s=out["ms"] / 1e3,
                t_ign=float(st["t_cross"][0]), T_end=float(s["T"][0]), identical=bool(np.all(s["T"] == s["T"][0])), **chemistry.DEFAULT)
    runs.append(many)
    print(f"4096 parcels, 80000 steps, default setting: {many['wall_s']:.1f} s; ignition {1e6 * many['t_ign']:.4f} us, final {many['T_end']:.4f} K", flush=True)

    # ---- speed: this kernel's batches next to OpenFOAM's cost per step and phase ----
    o80, pick = speed["openfoam"]["S80000"], lambda rows, **kw: next(r for r in rows if all(r[k] == v for k, v in kw.items()))      # noqa: E731
    dflt = chemistry.DEFAULT["precision"]
    phases = {}
    for tag, key in (("slow start", "before ignition"), ("rise", "rise, 1500 K"), ("burnt gas", "burnt")):
        r = pick(speed["phases"], state=key, precision=dflt, dt=1e-8)
        k = ("slow start", "rise", "burnt gas").index(tag)
        phases[tag] = dict(gpucfd_cell_steps_per_s=r["cell_steps_per_s"], openfoam_cell_steps_per_s=o80["cell_steps_per_s"][k], openfoam_us_per_step=o80["phase_us"][k],
                           ratio=r["cell_steps_per_s"] / o80["cell_steps_per_s"][k])
        print(f"{tag}: gpucfd {r['cell_steps_per_s'] / 1e6:.1f} million cells x steps per second ({dflt}, a million cells, 1e-8 s); OpenFOAM {o80['phase_us'][k]:.0f} us per "
              f"step on one core ({o80['cell_steps_per_s'][k]:.0f} per second, 1.25e-8 s, tolerance 1e-8): ratio {phases[tag]['ratio']:.0f}", flush=True)

    answers = dict(
        a=dict(value=dict(shift=main_["shift"], dT_flat_max=main_["dT_flat_max"]),
               diff=lines("rise: " + m(f"{1e6 * main_['shift_worst']:+.2f}" + US), "flat parts: " + m(f"{main_['dT_flat_max']:.2f}" + r"\,\mathrm{K}"))),
        b=dict(value=main_["t_ign"], diff=m_pct(main_["d_ref_t"], 3, signed=True)),
        c=dict(value=main_["T_end"], diff=lines(m_pct(main_["d_ref_T"], 4, signed=True), m(f"{main_['T_end'] - T_ref:+.3f}" + r"\,\mathrm{K}"))),
        e=dict(value=dict(atoms=main_["atoms"], energy=main_["energy"]), diff=lines("atoms " + sc(main_["atoms"]), "energy " + sc(main_["energy"]))))
    plain = (f"gpucfd's new chemistry kernel solves the same parcel: ignition at {1e6 * main_['t_ign']:.2f} microseconds ({100 * main_['d_ref_t']:+.3f} % from the reference) "
             f"and " + m(f"{main_['T_end']:.2f}" + r"\,\mathrm{K}") + " at the end. "
             f"For one parcel the GPU is slower than OpenFOAM on one core ({runs[3]['wall_s']:.0f} s against "
             f"{runs[3]['openfoam']['wall_s']:.0f} s for 80000 steps); its gain is in numbers: 4096 parcels take {many['wall_s']:.0f} s, and a million cells before ignition "
             f"advance at about {phases['slow start']['gpucfd_cell_steps_per_s'] / 1e6:.0f} million cells x steps per second (OpenFOAM's chemistry on a mesh of many "
             "cells was not timed, so there is no fair ratio yet)")
    rec = dict(tool="gpucfd", case=HERE.name, date=dt.date.today().isoformat(), device=name, startup_s=startup, plain=plain, busy_cpu=bool(busy), busy_processes=busy,
               answers=answers, runs=runs, requirements_met=requirements, like_for_like=main_, like_for_like_openfoam_constants=same_constants, default_setting=default,
               default=dict(chemistry.DEFAULT), reference=dict(t_ign=t_ref, T_end=T_ref),
               openfoam=dict(finest=dict(steps=80000, t_ign=fine["t_ign"], T_end=fine["T_end"], d_ref_t=fine["d_ref_t"], d_ref_T=fine["d_ref_T"], atoms=fine["atoms_max"],
                                         energy=fine["energy_dev"], wall_s=fine["wall_time"]),
                             step_to_zero=dict(t_ign=ex["t_ign"]["extrap"], T_end=ex["T_end"]["extrap"]), cost=speed["openfoam"]),
               against_openfoam=dict(same_constants_vs_step_to_zero=dict(t_ign=same_constants["t_ign"] / ex["t_ign"]["extrap"] - 1, T_end_K=same_constants["T_end"] - ex["T_end"]["extrap"]),
                                     phases_at_flow_step=phases, what_each_side_is_given=plan["what_each_side_is_given"]),
               speed_record="gpucfd/tests/results/test_chem_speed.json")
    OUT.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    print(f"the problem's requirements: {requirements}\nrecord: {OUT}")


if __name__ == "__main__":
    main()
