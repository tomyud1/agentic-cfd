"""Chemistry ladder, rung 1: the kernel's rates and thermodynamics against cfdlib.chem.

    python -m gpucfd.tests.test_chem_rates          (inside WSL, from the project root)

2000 random states (300 to 3500 K, 0.1 to 50 atm, random mixtures of all ten species of OpenFOAM's hydrogen
scheme, a third of the entries absent) are given to the kernel, at constant pressure and at constant volume.
Compared with cfdlib.chem, which reads the same two CHEMKIN files and shares no code with the kernel: forward and
reverse rate of each reaction, the rate of change of each species, enthalpy and heat capacity of each species,
the energy the kernel stores per cell, and the temperature it finds back from that energy (from a guess 2 % off).

Pass (plan: dev/compare/h2_air_ignition_box/plan.yaml), double precision: rates within 1e-12, enthalpies and heat
capacities within 1e-13, species within 1e-12 of their gross rate, temperature within 1e-9 K. Single and mixed
precision are recorded, not judged (they feed rung 4).
"""
import json
import sys
from pathlib import Path

import numpy as np

from cfdlib import chem
from gpucfd.chemistry import Chemistry

FAILS = []
REPORT = {}
N = 2000


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def states(mech, seed=1):
    rng = np.random.default_rng(seed)
    ns = len(mech.species)
    T = rng.uniform(300.0, 3500.0, N)
    p = 101325.0 * 10 ** rng.uniform(-1, np.log10(50.0), N)
    x = 10 ** rng.uniform(-8, 0, (N, ns)) * (rng.random((N, ns)) > 1 / 3)
    x[:, -1] += 1e-3                                              # never an empty cell
    x /= x.sum(1, keepdims=True)
    sigma = x / (x @ mech.W)[:, None]
    return T, p, sigma


def reference(mech, T, sigma, vol):
    """cfdlib.chem at each state (temperature, mol/kg, m3/kg): qf, qr, d(sigma)/dt, its gross rate, h, cp, enthalpy
    and internal energy per kg."""
    out = {k: [] for k in ("qf", "qr", "ds", "gross", "h", "cp", "H", "U", "v")}
    for k in range(len(T)):
        v = vol[k]
        qf, qr = mech.progress(sigma[k] / v, T[k])
        h = mech.h(T[k])
        out["qf"].append(qf); out["qr"].append(qr)
        out["ds"].append(v * (mech.dnu.T @ (qf - qr))); out["gross"].append(v * (np.abs(mech.dnu).T @ (qf + qr)))
        out["h"].append(h); out["cp"].append(mech.cp(T[k]))
        out["H"].append(sigma[k] @ h); out["U"].append(sigma[k] @ (h - mech.conv.R * T[k])); out["v"].append(v)
    return {k: np.array(v) for k, v in out.items()}


def rel(a, b, floor=0.0):
    """Largest |a - b| / |b| over the entries where b is not zero; where b is zero a must be zero. With a floor
    (single precision): relative to at least that share of the row's largest entry, and of 1e-30."""
    nz = b != 0
    zero_ok = bool(np.all(a[~nz] == 0))
    scale = np.abs(b) if not floor else np.maximum(np.abs(b), np.maximum(floor * np.abs(b).max(axis=1, keepdims=True), 1e-30))
    return float(np.max(np.abs(a[nz] - b[nz]) / scale[nz])), zero_ok


def compare(mech, T, p, sigma, ref, constant, precision):
    """Largest differences over all states. Rates: relative to each rate. Species: relative to the gross rate of
    that species (what the reactions make plus what they use). Enthalpy of a species: relative to |h| + R T (h
    itself passes through zero). Stored energy: relative to the sum of the species' |h| + R T.
    Single and mixed precision cannot hold numbers below 1e-38: there a rate or a gross rate counts as at least
    1e-12 of the largest one in its cell.
    In double precision the rates are compared at the temperature the kernel found back from the stored energy
    (itself compared with the given one): a reverse rate changes by up to half its value per kelvin, so the
    1e-11 K of the temperature search would otherwise show as 1e-12 in the rates (plan.yaml, note after rung 1).
    The difference at the given temperature is recorded next to it."""
    c = Chemistry(mech, N, constant=constant, precision=precision)
    c.set(T=T, p=p, sigma=sigma)
    k, s = c.rates(), c.get()
    c.close()
    floor = 0.0 if precision == "double" else 1e-12
    at_given = dict(qf=rel(k["qf"], ref["qf"], floor)[0], qr=rel(k["qr"], ref["qr"], floor)[0])
    if precision == "double":
        vol = sigma.sum(1) * mech.conv.R * k["T"] / p if constant == "pressure" else ref["v"]
        ref = dict(ref, **{q: v for q, v in reference(mech, k["T"], sigma, vol).items() if q in ("qf", "qr", "ds", "gross")})
    qf, zf = rel(k["qf"], ref["qf"], floor)
    qr, zr = rel(k["qr"], ref["qr"], floor)
    gross = np.maximum(ref["gross"], 1e-300) if not floor else np.maximum(ref["gross"], np.maximum(floor * ref["gross"].max(axis=1, keepdims=True), 1e-30))
    rt = mech.conv.R * T[:, None]
    e_scale = (sigma * (np.abs(ref["h"]) + rt)).sum(1)
    return dict(constant=constant, precision=precision, qf=qf, qr=qr, zeros=zf and zr, rates_at_given_T=at_given,
                species=float(np.max(np.abs(k["dsigma_dt"] - ref["ds"]) / gross)),
                h=float(np.max(np.abs(k["h"] - ref["h"]) / (np.abs(ref["h"]) + rt))), cp=rel(k["cp"], ref["cp"])[0],
                energy=float(np.max(np.abs(s["energy"] - ref["H" if constant == "pressure" else "U"]) / e_scale)),
                T_found=float(np.max(np.abs(k["T"] - T))), volume=float(np.max(np.abs(k["v"] / ref["v"] - 1))))


def main():
    mech = chem.build()
    T, p, sigma = states(mech)
    ref = reference(mech, T, sigma, sigma.sum(1) * mech.conv.R * T / p)
    REPORT.update(states=N, T_range=[float(T.min()), float(T.max())], p_range=[float(p.min()), float(p.max())],
                  species=mech.species, reactions=len(mech.reactions), rows=[])
    for constant in ("pressure", "volume"):
        for precision in ("double", "mixed", "float"):
            r = compare(mech, T, p, sigma, ref, constant, precision)
            REPORT["rows"].append(r)
            line = (f"forward rates {r['qf']:.1e}, reverse {r['qr']:.1e} (at the given temperature {r['rates_at_given_T']['qf']:.1e}, {r['rates_at_given_T']['qr']:.1e}), species {r['species']:.1e} of the gross rate, enthalpy {r['h']:.1e}, "
                    f"heat capacity {r['cp']:.1e}, stored energy {r['energy']:.1e}, temperature found back {r['T_found']:.1e} K")
            if precision == "double":
                ok = (r["qf"] <= 1e-12 and r["qr"] <= 1e-12 and r["zeros"] and r["species"] <= 1e-12 and r["h"] <= 1e-13 and r["cp"] <= 1e-13
                      and r["energy"] <= 1e-13 and r["T_found"] <= 1e-9 and r["volume"] <= 1e-13)
                check(f"rung 1: kernel equals cfdlib.chem to round-off (constant {constant}, double, {N} states, {len(mech.reactions)} reactions)", ok, line)
            else:
                print(f"      recorded, constant {constant}, {precision}: {line}", flush=True)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "test_chem_rates.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
