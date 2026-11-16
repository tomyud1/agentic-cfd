"""Reacting ladder, rung (b): two different gases with heat capacities that change with temperature, no reactions.

    python -m gpucfd.tests.test_react_mix          (inside WSL, from the project root; about a minute)

Pass rules and expectations were written before the runs: dev/compare/detonation_tube_h2_air/plan.yaml, rung b.
  b0. The exact solution (gpucfd/exact_mixture.py) by itself: its shock against cfdlib.chem.shock_frozen, the
      entropy along its expansion, the mass, momentum and energy of the whole solution.
  b1. Shock tube 0.12 m: left water vapour and nitrogen (2 : 3.76) at 2500 K and 3 MPa, right the fresh gas of
      case 21 (H2 : O2 : N2 = 2 : 1 : 3.76, 300 K, 1 atm), to 30 microseconds, 200 to 3200 cells.
  b2. A border between the same two gases at 1 atm, moving at 500 m/s once round a periodic tube: pressure and
      velocity should stay uniform. Measures the known weakness of a conservative scheme at such a border.
"""
import json
import sys
from pathlib import Path

import numpy as np

from cfdlib import chem
from gpucfd.exact_mixture import Mixture, MixtureRiemann
from gpucfd.reacting import Reacting, inert

OUT = Path(__file__).parent / "results"
FAILS = []
REPORT = {}
L, X0, T_END = 0.12, 0.06, 3.0e-5
LEFT = dict(T=2500.0, p=3.0e6, moles=dict(H2O=2.0, N2=3.76))
RIGHT = dict(T=300.0, p=101325.0, moles=dict(H2=2.0, O2=1.0, N2=3.76))
FAMILY = (200, 400, 800, 1600, 3200)
SNAPSHOTS = 10


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def fractions(mech, moles):
    x = np.array([moles.get(s, 0.0) for s in mech.species])
    return x * mech.W / (x @ mech.W)


def crossing(x, f, level):
    """The rightmost place where f falls through `level` (f above it on the left), straight line between cells."""
    k = int(np.max(np.nonzero(f > level)[0]))
    if k + 1 >= len(x):
        return float(x[k])
    return float(x[k] + (f[k] - level) / (f[k] - f[k + 1]) * (x[k + 1] - x[k]))


def exact_solution(mech):
    YL, YR = fractions(mech, LEFT["moles"]), fractions(mech, RIGHT["moles"])
    return MixtureRiemann(mech, dict(T=LEFT["T"], p=LEFT["p"], Y=YL), dict(T=RIGHT["T"], p=RIGHT["p"], Y=YR), x0=X0), YL, YR


def check_exact(mech, ex):
    w, st = ex.waves, ex.star
    ref = chem.shock_frozen(mech, RIGHT["p"], RIGHT["T"], RIGHT["moles"], w["right_head"])
    d = max(abs(st["p"] / ref["p"] - 1), abs(st["T_right"] / ref["T"] - 1), abs(st["rho_right"] / ref["rho"] - 1), abs(st["u"] / ref["u"] - 1))
    check("b0 exact solution: its shock against cfdlib.chem.shock_frozen", d <= 1e-8,
          f"shock at {w['right_head']:.2f} m/s; pressure, temperature, density, gas speed behind it agree within {d:.1e} (limit 1e-8)")
    left = ex.side[0]
    lp = np.linspace(np.log(left.p0), np.log(st["p"]), 50)
    T = left.fan(lp)[0]
    s = left.mix.s0(T) - left.mix.R * lp
    ds = float(np.max(np.abs(s - s[0])) / left.mix.R)
    check("b0 exact solution: entropy along the expansion", ds <= 1e-9, f"changes by {ds:.1e} of the gas constant (limit 1e-9)")
    # the whole solution: each stretch between two waves integrated by itself
    t = T_END
    cuts = [0.0] + [X0 + w[k] * t for k in ("left_head", "left_tail", "contact", "right_head")] + [L]
    tot = np.zeros(3)
    for a, b in zip(cuts[:-1], cuts[1:]):
        x = a + (np.arange(8000) + 0.5) / 8000 * (b - a)            # midpoints: never on a wave
        f = ex.sample(x, t)
        tot += (b - a) * np.array([f["rho"], f["rho"] * f["u"], ex.energy(f)]).mean(1)
    mixL, mixR = ex.mix
    rL, rR = LEFT["p"] / (mixL.R * LEFT["T"]), RIGHT["p"] / (mixR.R * RIGHT["T"])
    start = np.array([rL * X0 + rR * (L - X0), (LEFT["p"] - RIGHT["p"]) * t, rL * float(mixL.e(LEFT["T"])) * X0 + rR * float(mixR.e(RIGHT["T"])) * (L - X0)])
    err = np.abs(tot / start - 1)
    REPORT["b0"] = dict(shock=dict(speed=w["right_head"], apart=d), entropy=ds, totals=err.tolist(), star=st, waves=w)
    check("b0 exact solution: mass, momentum (against the push of the walls) and energy of the whole solution", float(err.max()) <= 1e-6,
          f"{err[0]:.1e}, {err[1]:.1e}, {err[2]:.1e} (limit 1e-6)")


def tube(mech, ex, YL, YR, N, recon, flux, cfl=0.4, precision="mixed"):
    h = L / N
    signal = max(ex.side[0].c0, ex.side[1].c0, abs(ex.waves["right_head"]), abs(ex.u_star) + float(ex.mix[0].c(ex.star["T_left"])),
                 abs(ex.u_star) + float(ex.mix[1].c(ex.star["T_right"])))
    steps = SNAPSHOTS * int(np.ceil(T_END * signal / (cfl * h) / SNAPSHOTS))
    dt = T_END / steps
    s = Reacting(mech, (N, 1, 1), (L, 1.0, 1.0), sides={"x-": "wall", "x+": "wall"}, precision=precision, recon=recon, flux=flux, chemistry=False)
    x = s.centres("x")
    left = (x < X0)[None, None, :]
    s.set(T=np.where(left, LEFT["T"], RIGHT["T"]), p=np.where(left, LEFT["p"], RIGHT["p"]), Y=np.where(left[None], YL.reshape(-1, 1, 1, 1), YR.reshape(-1, 1, 1, 1)))
    hist, front, ms = [], [], 0.0
    for k in range(SNAPSHOTS):
        st = s.run_fixed(dt, steps // SNAPSHOTS, history=True)
        ms += st["ms"]
        hist.append(st)
        front.append(crossing(x, s.get()["p"][0, 0], 0.5 * (RIGHT["p"] + ex.p_star)))
    f, guard = s.get(), s.guard()
    s.close()
    t = T_END * (np.arange(SNAPSHOTS) + 1) / SNAPSHOTS
    cat = lambda key: np.concatenate([h_[key] for h_ in hist])                                              # noqa: E731
    mass, energy, species = cat("mass"), cat("energy"), cat("species")
    used = species[0] > 0
    sim = dict(rho=f["rho"][0, 0], u=f["u"][0, 0], p=f["p"][0, 0], T=f["T"][0, 0])
    Y = f["Y"][:, 0, 0]
    exc = ex.cell_averages(np.linspace(0, L, N + 1), T_END)
    scales = dict(rho=LEFT["p"] / (ex.mix[0].R * LEFT["T"]), u=ex.u_star, p=LEFT["p"], T=LEFT["T"])
    w = {k: X0 + v * T_END for k, v in ex.waves.items()}
    mid = (x > w["contact"] + 0.25 * (w["right_head"] - w["contact"])) & (x < w["contact"] + 0.75 * (w["right_head"] - w["contact"]))
    star = dict(p=ex.p_star, u=ex.u_star, rho=ex.star["rho_right"], T=ex.star["T_right"])
    between = (x > w["left_tail"] + 2 * h) & (x < w["right_head"] - 4 * h)
    lo, hi = np.minimum(YL, YR)[:, None], np.maximum(YL, YR)[:, None]
    return dict(N=N, recon=recon, flux=flux, precision=precision, steps=steps, gpu_ms=ms,
                e={k: float(np.mean(np.abs(sim[k] - exc[k])) / scales[k]) for k in sim},
                shock_speed=float(np.polyfit(t[SNAPSHOTS // 2 - 1:], front[SNAPSHOTS // 2 - 1:], 1)[0]), shock_speed_exact=ex.waves["right_head"],
                plateau={k: float(sim[k][mid].mean() / star[k] - 1) for k in star},
                wiggle_p=float(np.max(np.abs(sim["p"][between] / ex.p_star - 1))), wiggle_u=float(np.max(np.abs(sim["u"][between] - ex.u_star)) / float(ex.mix[1].c(ex.star["T_right"]))),
                Y_min=float(Y.min()), Y_max=float(Y.max()), sum_off=float(np.max(np.abs(Y.sum(0) - 1))), Y_outside=float(max((Y - hi).max(), (lo - Y).max())),
                mass=float(np.max(np.abs(mass / mass[0] - 1))), energy=float(np.max(np.abs(energy / energy[0] - 1))),
                species=float(np.max(np.abs(species[:, used] / species[0, used] - 1))), guard=guard, x=x, sim=sim, exact=exc)


def moving_border(mech, YL, YR, N, recon="vanleer", flux="hybrid", cfl=0.4, speed=500.0):
    """Gas of the left kind at 2500 K in the middle half, gas of the right kind at 300 K outside, one pressure, one
    velocity, once round a periodic tube."""
    h, p0 = L / N, 101325.0
    cmax = max(float(Mixture(mech, YL).c(2500.0)), float(Mixture(mech, YR).c(300.0)))
    t_round = L / speed
    steps = int(np.ceil(t_round * (speed + cmax) / (cfl * h)))
    s = Reacting(mech, (N, 1, 1), (L, 1.0, 1.0), sides={"x-": "periodic", "x+": "periodic"}, recon=recon, flux=flux, chemistry=False)
    x = s.centres("x")
    hot = ((x > 0.25 * L) & (x < 0.75 * L))[None, None, :]
    s.set(T=np.where(hot, 2500.0, 300.0), p=p0, Y=np.where(hot[None], YL.reshape(-1, 1, 1, 1), YR.reshape(-1, 1, 1, 1)), u=speed)
    st = s.run_fixed(t_round / steps, steps, history=True)
    f, guard = s.get(), s.guard()
    s.close()
    Y = f["Y"][:, 0, 0]
    used = st["species"][0] > 0
    lo, hi = np.minimum(YL, YR)[:, None], np.maximum(YL, YR)[:, None]
    return dict(N=N, steps=steps, p_during=float(max(st["p_max"].max() / p0 - 1, 1 - st["p_min"].min() / p0)), p_end=float(np.max(np.abs(f["p"] / p0 - 1))),
                u_end=float(np.max(np.abs(f["u"] - speed)) / float(Mixture(mech, YR).c(300.0))), Y_min=float(Y.min()), Y_max=float(Y.max()),
                sum_off=float(np.max(np.abs(Y.sum(0) - 1))), Y_outside=float(max((Y - hi).max(), (lo - Y).max())),
                mass=float(np.max(np.abs(st["mass"] / st["mass"][0] - 1))), energy=float(np.max(np.abs(st["energy"] / st["energy"][0] - 1))),
                species=float(np.max(np.abs(st["species"][:, used] / st["species"][0, used] - 1))), guard=guard)


def bounded(r):
    return r["Y_min"] >= 0 and r["Y_max"] <= 1 + 1e-12 and r["sum_off"] <= 1e-12 and r["mass"] <= 1e-12 and r["energy"] <= 1e-12 and r["species"] <= 1e-12


def bounds_line(r):
    return (f"lowest mass fraction {r['Y_min']:.1e}, highest 1{r['Y_max'] - 1:+.1e}, sum off by {r['sum_off']:.1e}; a mass fraction leaves the range of its two "
            f"end values by {r['Y_outside']:.1e}; mass {r['mass']:.1e}, each species {r['species']:.1e}, energy {r['energy']:.1e}; guard {r['guard']}")


def slim(r):
    return {k: v for k, v in r.items() if k not in ("x", "sim", "exact")}


def main():
    mech = inert(chem.build())
    ex, YL, YR = exact_solution(mech)
    print(f"exact: between the waves {ex.p_star / 1e6:.4f} MPa and {ex.u_star:.2f} m/s; left of the border {ex.star['T_left']:.1f} K, right {ex.star['T_right']:.1f} K; "
          f"shock {ex.waves['right_head']:.1f} m/s, border {ex.waves['contact']:.1f} m/s, fan {ex.waves['left_head']:.1f} to {ex.waves['left_tail']:.1f} m/s", flush=True)
    check_exact(mech, ex)

    # ---- b1: the shock tube ----
    fam = [tube(mech, ex, YL, YR, N, "vanleer", "hybrid") for N in FAMILY]
    for r in fam:
        print(f"      {r['N']:5d} cells, {r['steps']} steps: mean error " + " / ".join(f"{100 * r['e'][k]:.4f}" for k in ("rho", "u", "p", "T"))
              + f" % (density / velocity / pressure / temperature); shock speed {100 * (r['shock_speed'] / r['shock_speed_exact'] - 1):+.3f} %; wiggle between fan and shock: "
              f"pressure {100 * r['wiggle_p']:.2f} %, velocity {100 * r['wiggle_u']:.2f} % of the speed of sound", flush=True)
    fine = fam[-1]
    d = abs(fine["shock_speed"] / fine["shock_speed_exact"] - 1)
    check("b1 3200 cells: speed of the shock", d <= 2e-3, f"{fine['shock_speed']:.2f} m/s against {fine['shock_speed_exact']:.2f}: {100 * d:.3f} % (limit 0.2 %)")
    worst = max(abs(v) for v in fine["plateau"].values())
    check("b1 3200 cells: state between border and shock", worst <= 5e-3, ", ".join(f"{k} {100 * v:+.3f} %" for k, v in fine["plateau"].items()) + " (limit 0.5 %)")
    check("b1 3200 cells: mean errors along the tube", max(fine["e"].values()) <= 2e-3, ", ".join(f"{k} {100 * v:.4f} %" for k, v in fine["e"].items()) + " (limit 0.2 %)")
    orders = {k: float(-np.polyfit(np.log([r["N"] for r in fam]), np.log([r["e"][k] for r in fam]), 1)[0]) for k in ("rho", "u", "p", "T")}
    # At least 0.6: a second-order scheme smears a border between two gases over a width that shrinks as the cell size
    # to the power 2/3, so a field that jumps there (temperature) cannot do better than 0.67. The first limit, 0.7 for
    # every field, was a wrong guess (found: temperature 0.66; REACTING.md). Rule set in the review, 2026-11-16.
    check("b1 the errors fall with the cell size", min(orders.values()) >= 0.6,
          "fitted orders " + ", ".join(f"{k} {v:.2f}" for k, v in orders.items()) + " (at least 0.6; 0.67 is the best possible at a border between gases)")
    check("b1 mass fractions bounded; species, mass and energy kept (every grid)", all(bounded(r) for r in fam), "worst grid: " + bounds_line(max(fam, key=lambda r: r["species"])))
    wig = [r["wiggle_p"] for r in fam]
    print(f"expected, found: pressure wiggle at most 2 % on every grid: found {100 * min(wig):.2f} to {100 * max(wig):.2f} %; a mass fraction outside its end values "
          f"by at most 1e-3: found {max(r['Y_outside'] for r in fam):.1e}; guard not used with van Leer: {fine['guard']}", flush=True)
    info = []
    for N in FAMILY:
        try:
            r = tube(mech, ex, YL, YR, N, "weno5", "hllc")
            info.append(slim(r))
            print(f"      for information, WENO5 with HLLC, {N} cells: mean error " + " / ".join(f"{100 * r['e'][k]:.4f}" for k in ("rho", "u", "p", "T"))
                  + f" %; pressure wiggle {100 * r['wiggle_p']:.2f} %; " + bounds_line(r), flush=True)
        except Exception as err:                                   # noqa: BLE001
            info.append(dict(N=N, broke=str(err)[:200]))
            print(f"      for information, WENO5 with HLLC, {N} cells: broke: {str(err)[:200]}", flush=True)
    REPORT["b1"] = dict(family=[slim(r) for r in fam], orders=orders, weno5_hllc=info)

    # ---- b2: a moving border at one pressure ----
    REPORT["b2"] = []
    for N in (400, 1600):
        r = moving_border(mech, YL, YR, N)
        REPORT["b2"].append(r)
        check(f"b2 moving border, {N} cells: mass fractions bounded; species, mass and energy kept", bounded(r), bounds_line(r))
        print(f"expected, found ({N} cells, {r['steps']} steps): pressure disturbed by at most 1 %: found {100 * r['p_during']:.2f} % during the run, {100 * r['p_end']:.2f} % at the "
              f"end; velocity by at most 1 % of the speed of sound: found {100 * r['u_end']:.2f} %", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "test_react_mix.json").write_text(json.dumps(REPORT, indent=1, default=float))
    np.savez(OUT / "test_react_mix_profiles.npz", x=fine["x"], **{"sim_" + k: v for k, v in fine["sim"].items()}, **{"exact_" + k: fine["exact"][k] for k in fine["sim"]})
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
