"""Rung d5 of plan.yaml: does the wave speed settle at the Chapman-Jouguet speed on fine grids? A trial with
gpucfd only; NOT case 21's problem. The tube is four times longer (0.48 m, 200 microseconds) and the ignition
patch is at 10 MPa, chosen without a flow solver (why_no_detonation.py, part "cpu") before any run of this file.
The plan, the readings and what was expected were written in plan.yaml (rung d5) before the runs.

    bash dev/compare/run.sh detonation_tube_h2_air gpucfd_long_tube.py            (about twelve minutes; writes gpucfd_long_tube.json)
    bash dev/compare/run.sh detonation_tube_h2_air gpucfd_long_tube.py summary    (expected and found, from the saved record; runs nothing)
"""
import json
from pathlib import Path

import numpy as np

from gpucfd.tests import test_react_tube as T

HERE = Path(__file__).resolve().parent
OUT = HERE / "gpucfd_long_tube.json"
LONG = dict(L=0.48, t_end=2e-4)
WINDOWS = ((5e-5, 1e-4), (1e-4, 1.5e-4), (1.5e-4, 2e-4))           # s


def windows(r):
    """Wave speed (m/s) over each window: straight-line fit of the pressure front's place."""
    t, x = np.asarray(r["samples"]["t"]), np.asarray(r["samples"]["x_p"])
    return [float(np.polyfit(t[(t >= a - 1e-12) & (t <= b + 1e-12)], x[(t >= a - 1e-12) & (t <= b + 1e-12)], 1)[0]) for a, b in WINDOWS]


def run(g, mech, moles, cj, N, p_ign):
    r = T.tube(g, mech, moles, cj, N, given=dict(LONG, p_ign=p_ign))
    s = T.slim(r)
    head = f"patch {p_ign / 1e6:.0f} MPa, {N} cells ({1e6 * r['h']:.2f} micrometres)"
    if r["broke"]:
        print(f"{head}: BROKE {r['broke']}", flush=True)
        return s
    healthy = r["status"] == 0 and max(r["mass"], r["atoms"], r["energy"]) <= 1e-9
    kept = f"chemistry status {r['status']}; mass {r['mass']:.1e}, atoms {r['atoms']:.1e}, energy {r['energy']:.1e}: {'healthy' if healthy else 'NOT HEALTHY'}"
    s["healthy"] = bool(healthy)
    if not r["detonation"]:
        print(f"{head}: NO DETONATION ({T.line(r, cj)})", flush=True)
        return s
    s["windows"] = windows(r)
    s["windows_dev"] = [v / cj["D"] - 1 for v in s["windows"]]
    print(f"{head}, {r['steps']} steps, {r['gpu_s']:.0f} s on the GPU: speed over 50-100, 100-150, 150-200 microseconds "
          + ", ".join(f"{v:.1f}" for v in s["windows"]) + " m/s (" + ", ".join(f"{100 * d:+.2f}" for d in s["windows_dev"]) + " % from CJ); burning front at most "
          f"{1e3 * r['gap_max']:.3f} mm behind; highest pressure {r['p_peak']['low'] / 1e6:.2f} to {r['p_peak']['high'] / 1e6:.2f} MPa (mean {r['p_peak']['mean'] / 1e6:.2f}); "
          f"sonic point: {r['sonic']['p'] / 1e6:.3f} MPa, {r['sonic']['T']:.0f} K (CJ {cj['p'] / 1e6:.3f}, {cj['T']:.0f}); " + kept, flush=True)
    return s


def main():
    g, mech, moles, cj = T.problem()
    print(f"reference: CJ speed {cj['D']:.2f} m/s; flow setting {T.FLOW}; tube {LONG['L']} m, {1e6 * LONG['t_end']:.0f} microseconds", flush=True)
    rec = dict(reference=cj, given=dict(g, **LONG), flow=T.FLOW, windows_s=WINDOWS, grid=[], patch=[], finer=None)

    def save():
        OUT.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")

    for N in (4800, 9600, 19200, 38400):
        rec["grid"].append(run(g, mech, moles, cj, N, 10e6))
        save()
    for p in (8e6, 12e6):
        rec["patch"].append(run(g, mech, moles, cj, 19200, p))
        save()
    a, b = rec["grid"][-2], rec["grid"][-1]
    if a.get("windows") and b.get("windows"):
        moved = abs(b["windows"][-1] - a["windows"][-1]) / cj["D"]
        expect_s = 4 * b["wall_s"]
        go = moved > 1e-3 and expect_s < 900
        print(f"between the two finest grids the last window moved by {100 * moved:.3f} % of CJ; 76800 cells expected to take {expect_s:.0f} s: {'run' if go else 'not run'}", flush=True)
        rec["finer_rule"] = dict(moved=moved, expected_s=expect_s, run=bool(go))
        if go:
            rec["finer"] = run(g, mech, moles, cj, 76800, 10e6)
    save()
    summary(rec)


def alive(r):
    """The reading rule of plan.yaml: a pressure front at every sample from 10 microseconds on, and the burning
    front within 1 mm of it."""
    return bool(r.get("windows")) and r["gap_max"] <= 1e-3


def summary(rec):
    """What was expected (plan.yaml, d5) next to what was found, from the saved record; runs nothing."""
    cj, g = rec["reference"], rec["given"]
    runs = rec["grid"] + rec["patch"] + ([rec["finer"]] if rec["finer"] else [])
    name = lambda r: f"{r['given']['p_ign'] / 1e6:.0f} MPa on {1e6 * r['h']:.2f} micrometres"
    # the distance between shock and burning in a steady wave, from numbers on record: the gas leaves the shock at
    # D / (density ratio) and burns after the ignition delay of why_no_detonation.py (constant volume, 400 K hotter)
    why = json.loads((HERE / "why_no_detonation.json").read_text(encoding="utf-8"))
    delay = why["cpu"]["shocked"][-1]["ignites_after_s"]
    ratio = cj["p_front"] / g["p_0"] * g["T_0"] / cj["T_front"]
    length = cj["D"] / ratio * delay
    print(f"distance between shock and burning in a steady wave, estimated without a flow solver: {1e3 * length:.3f} mm "
          f"(the gas leaves the shock at {cj['D'] / ratio:.0f} m/s and ignites after {1e6 * delay:.2f} microseconds)", flush=True)
    out = dict(induction_length_estimate=length, runs=[])
    for r in runs:
        t, gap, pk = (np.asarray(v, float) for v in (r["samples"]["t"], np.subtract(r["samples"]["x_p"], r["samples"]["x_b"]), r["samples"]["p_peak"]))
        row = dict(p_ign=r["given"]["p_ign"], N=r["N"], h=r["h"], alive=alive(r), cells_in_estimate=length / r["h"])
        if alive(r):
            late = t >= 1e-4 - 1e-12
            row.update(last_window_dev=r["windows_dev"][-1], gap_late=float(gap[late].mean()), gap_in_cells=float(gap[late].mean() / r["h"]))
            print(f"  {name(r)} ({length / r['h']:.1f} cells in the estimated distance): a detonation to the end; last window {100 * row['last_window_dev']:+.2f} % from CJ; "
                  f"burning {1e3 * row['gap_late']:.3f} mm behind the shock ({row['gap_in_cells']:.1f} cells)", flush=True)
        else:
            k = int(np.argmax(gap > 1e-3))
            before = (t >= 1e-5 - 1e-12) & (t < t[k] - 5e-6)
            row.update(lost_at_s=float(t[k]), lost_at_x=float(r["samples"]["x_p"][k]), p_peak_before=[float(pk[before].min()), float(pk[before].max())],
                       gap_before=[float(gap[before].min()), float(gap[before].max())])
            print(f"  {name(r)} ({length / r['h']:.1f} cells in the estimated distance): THE DETONATION IS LOST at {1e6 * t[k]:.0f} microseconds, {1e3 * row['lost_at_x']:.0f} mm along; "
                  f"before that the highest pressure swings between {pk[before].min() / 1e6:.2f} and {pk[before].max() / 1e6:.2f} MPa and the burning is "
                  f"{1e3 * gap[before].min():.3f} to {1e3 * gap[before].max():.3f} mm behind the shock", flush=True)
        out["runs"].append(row)
    ok = [r for r in runs if alive(r)]
    print(f"expected, found  a detonation in every run: {len(ok)} of {len(runs)}", flush=True)
    grid = [r for r in rec["grid"] + ([rec["finer"]] if rec["finer"] else [])]
    last = [r["windows_dev"][-1] for r in grid if alive(r)]
    print("expected, found  last window within 1 % of CJ on every grid: " + ", ".join(f"{100 * r['windows_dev'][-1]:+.2f} %" if alive(r) else "lost" for r in grid), flush=True)
    if last:
        print(f"expected, found  the grids within 0.2 % of each other over the last window: spread {100 * (max(last) - min(last)):.2f} % of CJ over the {len(last)} grids that keep the detonation", flush=True)
    three = [r for r in (rec["patch"][:1] + [rec["grid"][2]] + rec["patch"][1:])]
    if all(alive(r) for r in three):
        d = [r["windows_dev"][-1] for r in three]
        print("expected, found  the patches (8, 10, 12 MPa, 25 micrometres) within 0.5 % of each other over the last window: "
              + ", ".join(f"{100 * v:+.2f} %" for v in d) + f"; spread {100 * (max(d) - min(d)):.2f} %", flush=True)
    print("expected, found  each run's last two windows within 0.5 % of each other: "
          + ", ".join(f"{100 * abs(r['windows'][2] / r['windows'][1] - 1):.2f} %" if alive(r) else "lost" for r in runs), flush=True)
    print(f"healthy runs: {sum(1 for r in runs if r.get('healthy'))} of {len(runs)}", flush=True)
    out.update(alive=len(ok), of=len(runs), grid_spread=max(last) - min(last) if last else None)
    rec["summary"] = out
    OUT.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    print(f"wrote {OUT.name}", flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["summary"]:
        summary(json.loads(OUT.read_text(encoding="utf-8")))
    else:
        main()
