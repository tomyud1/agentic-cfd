"""Case 19 with gpucfd: the record that cfdlib.brief reads (gpucfd.json next to this file).

    PYTHONPATH=. python dev/compare/shock_diffraction_corner/gpucfd_record.py        (no solver runs; needs no GPU)

It is formed from the recorded run of rung 4 of gpucfd's compressible ladder
(gpucfd/tests/results/test_euler_corner.json, written by `bash gpucfd/tests/run_corner.sh 1600`). That run read
gpucfd's fields with the case's own rules (cases/shock_diffraction_corner/measure.py) and stored OpenFOAM's
numbers, read the same way, next to them. The plan with its pass rules was written before the runs: plan.yaml.

The cells are formed as the case's report forms OpenFOAM's (cases/shock_diffraction_corner/analyze.py, answers
a to d), on 800 x 800 cells, which is OpenFOAM's finest grid, with the setting the plan's rule chose before
the runs. gpucfd's second setting and its 1600 x 1600 grid are in gpucfd/COMPRESSIBLE.md.
"""
import datetime as dt
import json
from pathlib import Path

from cfdlib.report import lines, m, m_pct, tex_num

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "gpucfd" / "tests" / "results" / "test_euler_corner.json"
N_SHOWN = 800                                    # the finest grid both tools ran


def sc(x, sig=1):
    return m(tex_num(x, sig))


def sgn(x, dec=3):
    return m_pct(x, dec, signed=True)


def main():
    rec = json.loads(SRC.read_text(encoding="utf-8"))
    fam = {r["N"]: r for r in rec["family"]}
    of = {r["N"]: r["openfoam"] for r in rec["openfoam"]}
    both = sorted(set(fam) & set(of))                              # the grids both tools ran
    assert N_SHOWN == max(both), (N_SHOWN, both)
    g, o = fam[N_SHOWN], of[N_SHOWN]
    st = g["state"]
    worst_dev = max(fam[N]["dev_cells_max"] for N in both)
    mass = max(abs(fam[N]["mass_vs_inflow"]) for N in both)
    energy = max(abs(fam[N]["energy_vs_inflow"]) for N in both)
    answers = dict(
        a=dict(value=dict(speed=g["speed_err"], p=st["p"]["err"], rho=st["rho"]["err"], u=st["u"]["err"]),
               diff=lines(sgn(g["speed_err"]), sgn(st["p"]["err"]), sgn(st["rho"]["err"]), sgn(st["u"]["err"]))),
        b=dict(value=dict(mass=mass, energy=energy), diff=lines("mass " + sc(mass), "energy " + sc(energy))),
        c=dict(value=dict(dev_cells=worst_dev, ripple=st["p"]["ripple"]),
               diff=lines(sc(worst_dev) + " cells", "ripple " + m_pct(st["p"]["ripple"], 1))),
        d=dict(value=dict(step_face=g["sim_wall"], line=g["sim_line"]),
               diff=lines("step face " + m_pct(g["sim_wall"], 1), "line " + m_pct(g["sim_line"], 1))))
    runs = [dict(N=N, cells=fam[N]["cells"], steps=fam[N]["steps"], wall_s=fam[N]["wall_s"]) for N in both]
    runs += [dict(label=f"{N} by {N} cells, {fam[N]['steps']} steps (gpucfd only)", N=N, cells=fam[N]["cells"],
                  steps=fam[N]["steps"], wall_s=fam[N]["wall_s"]) for N in sorted(set(fam) - set(of))]
    ripple_g, ripple_o = g["ripple_p_all"], o["ripple_p_all"]      # the largest at any written time, the same measure for both
    plain = (f"gpucfd's new shock solver meets the same three requirements on the same grids, and on {N_SHOWN} by {N_SHOWN} "
             f"cells it takes {g['wall_s']:.0f} s on the GPU against OpenFOAM's {o['wall_time']:.0f} s on {o['cores']} cores. "
             f"Its pressure ripple behind the shock is {'larger' if ripple_g > ripple_o else 'smaller'}: at most "
             f"\\({100 * ripple_g:.1f}\\,\\%\\) at any written time against OpenFOAM's \\({100 * ripple_o:.1f}\\,\\%\\)")
    out = dict(tool="gpucfd", case="shock_diffraction_corner", date=dt.date.today().isoformat(), source=str(SRC.relative_to(ROOT)).replace("\\", "/"),
               setting=rec["chosen"], grid_shown=N_SHOWN, plain=plain, busy_cpu=bool(rec.get("busy_cpu")),
               answers=answers, runs=runs,
               openfoam_same_rules=dict(speed=o["speed_err"], state=o["state"], dev_cells=max(of[N]["dev_cells_max"] for N in both),
                                        step_face=o["sim_wall"], line=o["sim_line"], ripple_any_time=ripple_o))
    (HERE / "gpucfd.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"record: {HERE / 'gpucfd.json'} | speed {100 * g['speed_err']:+.3f} % (OpenFOAM {100 * o['speed_err']:+.3f} %), "
          f"off a straight line {worst_dev:.1e} cells, {g['wall_s']:.1f} s against {o['wall_time']:.0f} s")


if __name__ == "__main__":
    main()
