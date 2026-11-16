"""Why case 21 gives no detonation on fine grids: the checks behind that statement in gpucfd/REACTING.md and in
plan.yaml (rung d, found), kept as code with a record of their numbers.

    bash dev/compare/run.sh detonation_tube_h2_air why_no_detonation.py        (about two minutes; writes why_no_detonation.json)

cpu   Independent of the GPU solver (cfdlib.chem and gpucfd/exact_mixture.py): the ignition patch burnt at
      constant volume; the shock it then drives into the fresh gas (exact shock-tube solution with frozen
      compositions), for the problem's patch pressure and for stronger ones (rung d4 took the first whose shock is
      faster than the CJ speed); how long shocked fresh gas takes to ignite at constant volume.
gpu   The wave followed in time: on 3200, 4000 and 4800 cells with the fixed setting, and on 4800 cells with a
      tighter chemistry setting, the implicit method, first-order reconstruction, WENO5, and double precision.
"""
import json
from pathlib import Path

import numpy as np

from cfdlib import chem
from gpucfd.exact_mixture import MixtureRiemann
from gpucfd.reacting import Reacting
from gpucfd.tests import test_react_tube as T

HERE = Path(__file__).resolve().parent
OUT = HERE / "why_no_detonation.json"
WAIT = 2e-4                                                      # how long a shocked parcel is watched, s
PATCHES = (3e6, 6e6, 8e6, 10e6, 12e6, 15e6)


def ignition(mech, p, Tk, moles):
    """Seconds until gas at (p, Tk) held at constant volume is 400 K hotter; None if not within WAIT."""
    s = chem.VolumeParcel(mech, p, Tk, moles).solve(WAIT, T_ign=Tk + 400.0)
    t = s["t_ign"]
    return float(t) if t is not None and np.isfinite(t) else None


def cpu(g, mech, moles, cj):
    x = np.array([moles.get(s, 0.0) for s in mech.species])
    Yf = x * mech.W / (x @ mech.W)
    out = dict(patches=[], shocked=[])
    for p_ign in PATCHES:
        patch = chem.VolumeParcel(mech, p_ign, g["T_ign"], moles)
        sol = patch.solve(2e-6, T_ign=3000.0)
        Tb, sb = float(sol["T"][-1]), sol["sigma"][-1]
        pb, Yb = float(patch.pressure(sb, Tb)), sb * mech.W
        ex = MixtureRiemann(mech, dict(T=Tb, p=pb, Y=Yb / Yb.sum()), dict(T=g["T_0"], p=g["p_0"], Y=Yf))
        r = dict(p_ign=p_ign, burnt_T=Tb, burnt_p=pb, burns_in_s=float(sol["t_ign"]), shock_p=ex.p_star, shock_T=ex.star["T_right"], shock_speed=ex.waves["right_head"],
                 ignites_after_s=ignition(mech, ex.p_star, ex.star["T_right"], moles))
        out["patches"].append(r)
        print(f"patch at {p_ign / 1e6:.0f} MPa: burns to {Tb:.0f} K and {pb / 1e6:.2f} MPa; its shock into the fresh gas: {ex.p_star / 1e6:.3f} MPa, {ex.star['T_right']:.0f} K, "
              f"{ex.waves['right_head']:.0f} m/s ({100 * (ex.waves['right_head'] / cj['D'] - 1):+.1f} % from CJ); that gas "
              + (f"ignites after {1e6 * r['ignites_after_s']:.2f} microseconds" if r["ignites_after_s"] else f"does not ignite within {1e6 * WAIT:.0f} microseconds"), flush=True)
    for p, Tk in ((1.65e6, 1050.0), (1.85e6, 1120.0), (2.0e6, 1180.0), (cj["p_front"], cj["T_front"])):
        t = ignition(mech, p, Tk, moles)
        out["shocked"].append(dict(p=p, T=Tk, ignites_after_s=t))
        print(f"fresh gas at {p / 1e6:.2f} MPa and {Tk:.0f} K: " + (f"ignites after {1e6 * t:.2f} microseconds" if t else f"does not ignite within {1e6 * WAIT:.0f} microseconds"), flush=True)
    return out


def follow(g, mech, moles, cj, N, tag, t_end=3e-5, cfl=0.4, **kw):
    """The shock (pressure above 1.5 times the start), the burning front and the state just behind the shock,
    every microsecond."""
    s = Reacting(mech, (N, 1, 1), (g["L"], 1.0, 1.0), sides={"x-": "wall", "x+": "wall"}, **kw)
    x = s.centres("x")
    hot = (x < g["L_ign"])[None, None, :]
    s.set(T=np.where(hot, g["T_ign"], g["T_0"]) * np.ones(s.shape), p=np.where(hot, g["p_ign"], g["p_0"]) * np.ones(s.shape), moles=moles)
    iw, h, rows = mech.idx["H2O"], g["L"] / N, []
    for k in range(1, int(round(t_end / 1e-6)) + 1):
        s.run(1e-6, cfl=cfl)
        f = s.get()
        p, Tk, yw = f["p"][0, 0], f["T"][0, 0], f["Y"][iw, 0, 0]
        xs, xb = T.crossing(x, p, 1.5 * g["p_0"]), T.crossing(x, yw, 0.5 * cj["Y_H2O"])
        js = int(np.searchsorted(x, xs)) - 1
        behind = slice(max(js - int(0.3e-3 / h), 0), max(js - 2, 1))          # 0.3 mm behind the shock, without the two cells of its rise
        rows.append(dict(t=1e-6 * k, shock=xs, burning=xb, p_max=float(p.max()), T_max=float(Tk.max()), p_behind=float(p[behind].mean()), T_behind=float(Tk[behind].mean())))
    status, guard = int(s.chem_stats()["status"].max()), s.guard()
    s.close()
    a, b = rows[min(4, len(rows) - 1)], rows[-1]
    coupled = (b["shock"] - b["burning"]) <= 1e-3
    print(f"{N} cells, {tag}: at {1e6 * a['t']:.0f} microseconds the gas behind the shock has {a['p_behind'] / 1e6:.2f} MPa and {a['T_behind']:.0f} K, burning front "
          f"{1e3 * (a['shock'] - a['burning']):.2f} mm behind; at {1e6 * b['t']:.0f} microseconds {b['p_behind'] / 1e6:.2f} MPa, {b['T_behind']:.0f} K, {1e3 * (b['shock'] - b['burning']):.2f} mm "
          f"behind: {'a detonation' if coupled else 'NO detonation'}", flush=True)
    return dict(N=N, tag=tag, detonation=bool(coupled), status=status, guard=guard, history=rows)


def gpu(g, mech, moles, cj):
    runs = [follow(g, mech, moles, cj, N, "fixed setting") for N in (3200, 4000, 4800)]
    for tag, kw in (("chemistry bs23 at 1e-6, quiet rule off", dict(chemistry=dict(method="bs23", rtol=1e-6, quiet=0.0))),
                    ("chemistry ros34a at 1e-6, quiet rule off", dict(chemistry=dict(method="ros34a", rtol=1e-6, quiet=0.0))),
                    ("first-order reconstruction", dict(recon="first")),
                    ("WENO5 with HLLC", dict(recon="weno5", flux="hllc")),
                    ("double precision (15 microseconds)", dict(precision="double", t_end=1.5e-5))):
        runs.append(follow(g, mech, moles, cj, 4800, tag, **kw))
    return runs


if __name__ == "__main__":
    g, mech, moles, cj = T.problem()
    rec = dict(reference=cj, cpu=cpu(g, mech, moles, cj), gpu=gpu(g, mech, moles, cj))
    OUT.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    print(f"wrote {OUT.name}")
