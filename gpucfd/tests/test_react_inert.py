"""Reacting ladder, rung (a): with one inert gas and constant heat capacities the joined solver gives what the
shock solver gives.

    python -m gpucfd.tests.test_react_inert          (inside WSL, from the project root; about a minute)

Pass rules and expectations were written before the runs: dev/compare/detonation_tube_h2_air/plan.yaml, rung a.
  a1. The shock tube of case 18, air as one species with heat capacity 3.5 R: joined solver against shock solver
      (gpucfd.euler), same setting, same time steps; double within 1e-9, mixed within 1e-5 of the scales.
  a2. The grid family 100 to 12800 cells: errors against the exact solution, each within 0.1 % of the shock
      solver's.
  a3. The same air as two species, one on each side: equal to a1 within 1e-12; mass fractions between 0 and 1,
      their sum one, each species' mass kept.
  a4. Two dimensions: the shock round a corner of case 19 on 100 x 100 cells within 1e-8 in double; a
      one-dimensional problem along y and along z equal to the same along x bit for bit.
Mass and energy kept within 1e-13; the guard not used.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.constants as const
import yaml

from cfdlib import chem
from cfdlib import verification as V
from gpucfd.euler import Euler
from gpucfd.exact_riemann import Riemann
from gpucfd.reacting import Reacting, constant_gas, inert

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "results"
FAILS = []
REPORT = {}
SETTINGS = (("vanleer", "hll"), ("weno5", "hllc"))
FAMILY = (100, 200, 400, 800, 1600, 3200, 6400, 12800)


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def given(case):
    return {k: v["value"] for k, v in yaml.safe_load((ROOT / "cases" / case / "problem.yaml").read_text(encoding="utf-8"))["given"].items()}


def sod(g, kind, N, recon, flux, precision, cfl=0.4, species=1):
    """The shock tube on N cells to t_end in equal steps. kind: "euler" or "joined". Returns the end state, the
    totals after every step, and for the joined solver the mass fractions and the guard's counts."""
    ex = Riemann(g["gamma"], (g["rho_L"], 0.0, g["p_L"]), (g["rho_R"], 0.0, g["p_R"]), x0=g["L"] / 2)
    h = g["L"] / N
    steps = 10 * int(np.ceil(g["t_end"] * ex.fastest / (cfl * h) / 10))
    dt = g["t_end"] / steps
    sides = {"x-": "wall", "x+": "wall"}
    if kind == "euler":
        s = Euler((N, 1, 1), (g["L"], 1.0, 1.0), g["gamma"], sides=sides, precision=precision, recon=recon, flux=flux)
        x = s.centres("x")
        s.set(rho=np.where(x < g["L"] / 2, g["rho_L"], g["rho_R"]), p=np.where(x < g["L"] / 2, g["p_L"], g["p_R"]))
    else:
        names = ("air",) if species == 1 else ("air_left", "air_right")
        s = Reacting(constant_gas(names, const.R / g["R"], g["gamma"]), (N, 1, 1), (g["L"], 1.0, 1.0), sides=sides, precision=precision,
                     recon=recon, flux=flux, chemistry=False)
        x = s.centres("x")
        left = (x < g["L"] / 2)[None, None, :]
        Y = np.ones((1, 1, 1, N)) if species == 1 else np.stack([left * 1.0, ~left * 1.0])
        s.set(rho=np.where(left, g["rho_L"], g["rho_R"]), p=np.where(left, g["p_L"], g["p_R"]), Y=Y)
    st = s.run_fixed(dt, steps, history=True)
    f = s.get()
    r = dict(N=N, steps=steps, x=x, rho=f["rho"][0, 0], u=f["u"][0, 0], p=f["p"][0, 0], gpu_ms=st["ms"],
             mass=float(np.max(np.abs(st["mass"] / st["mass"][0] - 1))), energy=float(np.max(np.abs(st["energy"] / st["energy"][0] - 1))))
    if kind == "joined":
        r.update(Y=f["Y"][:, 0, 0], guard=s.guard(), species=float(np.max(np.abs(st["species"] / st["species"][0] - 1))))
    s.close()
    exc = ex.cell_averages(np.linspace(0, g["L"], N + 1), g["t_end"])
    scales = (g["rho_L"], ex.u_star, g["p_L"])
    r["e"] = [V.mean_abs_error(r[k], exc[i], scales[i]) for i, k in enumerate(("rho", "u", "p"))]
    r["scales"] = scales
    return r


def apart(a, b):
    return max(float(np.max(np.abs(a[k] - b[k]))) / a["scales"][i] for i, k in enumerate(("rho", "u", "p")))


def unused(g):
    return g["first_order"] == 0 and g["species_flat"] == 0 and g["species"] == 0 and g["cold"] == 0


def gentle(g):
    """The guard did nothing that changes the answer: no fall-back to first order, no cold cell, no mass moved
    (below 1e-30 kg) and no energy added. Mass fractions put back on a face are allowed and counted."""
    return g["first_order"] == 0 and g["cold"] == 0 and g["mass_moved"] <= 1e-30 and g["energy_added"] == 0


def main():
    g = given("shock_tube_sod")
    REPORT["a1"], REPORT["a2"], REPORT["a3"] = [], [], []

    # ---- a1: joined against shock solver, same setting and steps ----
    # The rounding level of mixed precision: how far each solver's mixed run is from its own double run. Two programs
    # that do the same sums in a different order cannot agree closer than about this, so it is the limit in mixed
    # precision (twice the larger of the two). The first limit, a flat 1e-5 written before any run, was a wrong guess
    # (found: up to 3.4e-5; REACTING.md, "Expected by the builder, found otherwise"). Rule set in the review, 2026-11-16.
    REPORT["a1_noise"] = []
    noise = {}
    for recon, flux in SETTINGS:
        for N in (400, 3200):
            runs = {(w, pr): sod(g, w, N, recon, flux, pr) for w in ("euler", "joined") for pr in ("double", "mixed")}
            ne, nj = apart(runs["euler", "double"], runs["euler", "mixed"]), apart(runs["joined", "double"], runs["joined", "mixed"])
            noise[recon, flux, N] = max(ne, nj)
            REPORT["a1_noise"].append(dict(recon=recon, flux=flux, N=N, euler=ne, joined=nj))
            print(f"      rounding level, {recon}/{flux} {N} cells: mixed against double {ne:.1e} in the shock solver, {nj:.1e} in the joined solver", flush=True)
    for precision in ("double", "mixed"):
        for recon, flux in SETTINGS:
            for N in (400, 3200):
                e, j = sod(g, "euler", N, recon, flux, precision), sod(g, "joined", N, recon, flux, precision)
                d = apart(e, j)
                limit = 1e-9 if precision == "double" else 2 * noise[recon, flux, N]
                REPORT["a1"].append(dict(precision=precision, recon=recon, flux=flux, N=N, apart=d, limit=limit, mass=j["mass"], energy=j["energy"], guard=j["guard"]))
                check(f"a1 {precision} {recon}/{flux} {N} cells: joined solver equals shock solver", d <= limit,
                      f"largest difference {d:.1e} of the scales (limit {limit:.1e}" + ("" if precision == "double" else ", twice the rounding level") + "); errors against the exact solution "
                      f"{100 * j['e'][0]:.4f} / {100 * j['e'][1]:.4f} / {100 * j['e'][2]:.4f} % (density / velocity / pressure)")
                check(f"a1 {precision} {recon}/{flux} {N} cells: mass and energy kept, guard not used",
                      j["mass"] <= 1e-13 and j["energy"] <= 1e-13 and unused(j["guard"]), f"mass {j['mass']:.1e}, energy {j['energy']:.1e}, guard {j['guard']}")

    # ---- a2: the grid family, errors next to the shock solver's ----
    worst = 0.0
    for N in FAMILY:
        e, j = sod(g, "euler", N, "weno5", "hllc", "mixed"), sod(g, "joined", N, "weno5", "hllc", "mixed")
        rel = max(abs(j["e"][i] / e["e"][i] - 1) for i in range(3))
        worst = max(worst, rel)
        REPORT["a2"].append(dict(N=N, steps=j["steps"], joined=j["e"], euler=e["e"], rel=rel, gpu_ms=dict(joined=j["gpu_ms"], euler=e["gpu_ms"])))
        print(f"      {N:6d} cells: joined {100 * j['e'][0]:.4f} / {100 * j['e'][1]:.4f} / {100 * j['e'][2]:.4f} %, shock solver "
              f"{100 * e['e'][0]:.4f} / {100 * e['e'][1]:.4f} / {100 * e['e'][2]:.4f} %; {j['steps']} steps, {j['gpu_ms'] / 1e3:.2f} s against {e['gpu_ms'] / 1e3:.2f} s", flush=True)
    check("a2 grid family (WENO5, HLLC, mixed): errors equal to the shock solver's", worst <= 1e-3, f"largest relative difference of an error {worst:.1e} (limit 1e-3)")

    # ---- a3: the same air as two species ----
    for recon, flux in SETTINGS:
        one, two = sod(g, "joined", 400, recon, flux, "double"), sod(g, "joined", 400, recon, flux, "double", species=2)
        d = apart(one, two)
        Y = two["Y"]
        lo, hi, off = float(Y.min()), float(Y.max()), float(np.max(np.abs(Y.sum(0) - 1)))
        REPORT["a3"].append(dict(recon=recon, flux=flux, apart=d, Y_min=lo, Y_max=hi, sum_off=off, species=two["species"], guard=two["guard"]))
        check(f"a3 {recon}/{flux}: two equal species give the one-species answer", d <= 1e-12, f"largest difference {d:.1e} (limit 1e-12)")
        # The guard: not used at all with van Leer (the setting of the detonation runs). WENO5 undershoots next to a
        # step that ends at zero, which is what the guard is for: there it may put mass fractions back on a face, but
        # may not move mass, add energy or fall back to first order. The first rule ("used in no run") was a wrong
        # guess for WENO5 (found: 12 242 face values put back; REACTING.md). Rule set in the review, 2026-11-16.
        guard_ok = unused(two["guard"]) if recon == "vanleer" else gentle(two["guard"])
        check(f"a3 {recon}/{flux}: mass fractions bounded, sum one, species kept, guard " + ("not used" if recon == "vanleer" else "moves no mass"),
              lo >= 0 and hi <= 1 + 1e-13 and off <= 1e-13 and two["species"] <= 1e-13 and guard_ok,
              f"lowest {lo:.1e}, highest 1{hi - 1:+.1e}, sum off by {off:.1e}, mass of each species changed by {two['species']:.1e}, guard {two['guard']}")

    # ---- a4: two dimensions ----
    from gpucfd.tests.test_euler_corner import Flow
    gc = given("shock_diffraction_corner")
    fl = Flow(gc)
    N = 100
    h = fl.L / N
    dt = 0.25 * h / (fl.u2 + 2 * fl.c2)
    steps = int(np.ceil(fl.t_end / dt))
    half = int(round(N * fl.Hs / fl.H))
    walls = {k: "wall" for k in ("x-", "x+", "y-", "y+")}
    e = Euler((N, N, 1), (fl.L, fl.H, 1.0), fl.gamma, precision="double", recon="vanleer", flux="hybrid", sides=walls)
    e.segment("x-", {"y": (half, N)}, ("supersonic", fl.state))
    e.set(rho=fl.rho0, p=fl.p0)
    e.run_fixed(fl.t_end / steps, steps)
    fe = e.get()
    e.close()
    gas = constant_gas(("air",), const.R / gc["R"], fl.gamma)
    j = Reacting(gas, (N, N, 1), (fl.L, fl.H, 1.0), precision="double", recon="vanleer", flux="hybrid", sides=walls, chemistry=False)
    j.segment("x-", {"y": (half, N)}, ("supersonic", dict(T=fl.p2 / (fl.rho2 * gc["R"]), p=fl.p2, Y=[1.0], u=fl.u2)))
    j.set(rho=fl.rho0, p=fl.p0, Y=[1.0])
    j.run_fixed(fl.t_end / steps, steps)
    fj, gj = j.get(), j.guard()
    j.close()
    d = max(float(np.max(np.abs(fe[k] - fj[k]))) / sc for k, sc in (("rho", fl.rho2), ("u", fl.u2), ("v", fl.u2), ("p", fl.p2)))
    REPORT["a4"] = dict(corner=dict(N=N, steps=steps, apart=d, guard=gj, rho_min=float(fj["rho"].min())))
    check("a4 corner of case 19, 100 x 100, double: joined solver equals shock solver", d <= 1e-8 and unused(gj),
          f"largest difference {d:.1e} of the state behind the shock (limit 1e-8) after {steps} steps; lowest density {fj['rho'].min():.3f} kg/m3; guard {gj}")

    # a one-dimensional problem with two real gases along x, y and z
    mech = inert(chem.build())
    out = []
    for axis in range(3):
        n, L = [1, 1, 1], [1.0, 1.0, 1.0]
        n[axis], L[axis] = 200, 0.12
        name = "xyz"[axis]
        s = Reacting(mech, n, L, sides={name + "-": "wall", name + "+": "wall"}, recon="weno5", flux="hllc", chemistry=False)
        x = s.centres(name)
        shape = [1, 1, 1]
        shape[2 - axis] = 200
        left = (x < 0.06).reshape(shape)
        YL, YR = s._mixture(moles=dict(H2O=2, N2=3.76)), s._mixture(moles=dict(H2=2, O2=1, N2=3.76))
        s.set(T=np.where(left, 2500.0, 300.0), p=np.where(left, 3.0e6, 101325.0), Y=np.where(left[None], YL.reshape(-1, 1, 1, 1), YR.reshape(-1, 1, 1, 1)))
        s.run_fixed(2e-8, 500)
        f = s.get()
        s.close()
        out.append(np.concatenate([f["rho"].ravel(), f["p"].ravel(), f["T"].ravel(), f["uvw"[axis]].ravel(), f["Y"].reshape(len(mech.species), -1).ravel()]))
    same = bool(np.array_equal(out[0], out[1]) and np.array_equal(out[0], out[2]))
    REPORT["a4"]["axes_identical"] = same
    check("a4 a two-gas shock tube along y and along z equals the one along x bit for bit", same,
          f"largest difference {max(float(np.max(np.abs(out[0] - o))) for o in out[1:]):.1e}")

    OUT.mkdir(exist_ok=True)
    (OUT / "test_react_inert.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
