"""Compressible ladder, rung 2: smooth flows with exact solutions, the designed order of accuracy.

    python -m gpucfd.tests.test_euler_smooth          (inside WSL, from the project root)

1. A density wave carried by a uniform stream (1D along x, and 2D along the diagonal): rho = 1 + 0.2 sin(2 pi (x - u t)),
   velocity and pressure constant. Exact for the Euler equations; here they act as a linear transport
   equation, so each reconstruction shows its own order: 2 for the piecewise-linear ones, 5 for WENO5
   (with the time step cut like h^(5/3), so that the third-order time scheme does not hide it).
2. An isentropic vortex carried across a periodic box (2D, nonlinear, exact): order 2 for every
   reconstruction. The scheme reconstructs density, velocity and pressure from cell averages and uses one
   flux per face, which is second order for a nonlinear flow in any case; WENO5 lowers the error, not the order.
3. The time scheme: three stages, order 3 (WENO5 at a fixed ratio of time step to cell size).
4. Single and mixed precision (state in double, fluxes in float) against double on the density wave.

Pass rules, fixed before the runs: finest-pair order of the mean absolute density error within 1.9 to 2.1
(no limiter), at least 1.7 (van Leer, MC: limiters clip the crests), at least 1.3 (minmod), 4.7 to 5.3
(WENO5, linear wave), at least 1.8 (vortex), at least 2.8 (time). Mixed: error within 3e-6 of double's and mass kept to
1e-9. Float: its loss of mass is recorded (first measured after the runs: about 1e-4 in 24 000 steps).
"""
import json
import sys
from pathlib import Path

import numpy as np

from cfdlib import verification as V
from gpucfd.euler import Euler

FAILS = []
REPORT = {}
GAMMA = 1.4
AMP = 0.2


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def wave_average(n, dims, t, vel):
    """Exact cell averages of rho = 1 + AMP sin(2 pi (x [+ y] - c t)) on the unit box."""
    h = 1.0 / n
    xc = (np.arange(n) + 0.5) * h
    f = np.sin(np.pi * h) / (np.pi * h)                       # the average of a sine over a cell, per direction
    if dims == 1:
        return 1 + AMP * f * np.sin(2 * np.pi * (xc - vel[0] * t))[None, None, :]
    X, Y = np.meshgrid(xc, xc)
    return 1 + AMP * f * f * np.sin(2 * np.pi * (X + Y - (vel[0] + vel[1]) * t))[None]


def wave(n, dims, recon, precision="double", cfl=0.4, fifth=False, flux="hllc", T=1.0, stages=3, mass=False):
    vel = (1.0, 0.5 if dims == 2 else 0.0)
    shape = (n, n if dims == 2 else 1, 1)
    s = Euler(shape, (1.0, 1.0, 1.0), GAMMA, precision=precision, recon=recon, flux=flux, stages=stages)
    s.set(rho=wave_average(n, dims, 0.0, vel), u=vel[0], v=vel[1], p=1.0)
    h = 1.0 / n
    speed = sum(abs(v) + np.sqrt(GAMMA * 1.0 / (1 - AMP)) for v in vel[:dims])
    dt = cfl * h / speed * ((h * 32) ** (2 / 3) if fifth else 1.0)
    steps = int(np.ceil(T / dt))
    m0 = s.totals()["mass"]
    st = s.run_fixed(T / steps, steps)
    err = V.mean_abs_error(s.get()["rho"], wave_average(n, dims, T, vel), 1.0)
    m1 = s.totals()["mass"]
    s.close()
    return (err, steps, st["ms"], m1 / m0 - 1) if mass else (err, steps, st["ms"])


def orders(name, key, Ns, errs, lo, hi):
    o = V.order_from_errors([1.0 / n for n in Ns], errs)
    check(name, lo <= o[-1] <= hi, f"N {list(Ns)}: errors {', '.join(f'{e:.2e}' for e in errs)}; orders {', '.join(f'{x:.2f}' for x in o)} "
          f"(rule: finest pair {lo:g} to {hi:g})")
    REPORT[key] = dict(N=list(Ns), err=errs, order=o, rule=[lo, hi])


def vortex_fields(n, L, t, beta=5.0, u0=1.0, v0=1.0):
    """Isentropic vortex (Shu 1998) centred in a periodic box of side L, carried at (u0, v0). Point values."""
    xc = (np.arange(n) + 0.5) * L / n
    X, Y = np.meshgrid(xc, xc)
    dx = (X - L / 2 - u0 * t + L / 2) % L - L / 2
    dy = (Y - L / 2 - v0 * t + L / 2) % L - L / 2
    r2 = dx * dx + dy * dy
    e = np.exp(0.5 * (1 - r2))
    u = u0 - beta / (2 * np.pi) * e * dy
    v = v0 + beta / (2 * np.pi) * e * dx
    Temp = 1 - (GAMMA - 1) * beta ** 2 / (8 * GAMMA * np.pi ** 2) * e * e
    rho = Temp ** (1 / (GAMMA - 1))
    return rho[None], u[None], v[None], (rho * Temp)[None]


def vortex(n, recon, L=20.0, T=2.0, cfl=0.4):
    s = Euler((n, n, 1), (L, L, 1.0), GAMMA, precision="double", recon=recon, flux="hllc")
    rho, u, v, p = vortex_fields(n, L, 0.0)
    s.set(rho=rho, u=u, v=v, p=p)
    st = s.run(T, cfl=cfl, every=10)
    err = V.mean_abs_error(s.get()["rho"], vortex_fields(n, L, T)[0], 1.0)
    s.close()
    return err, st["steps"]


def main():
    Ns = (32, 64, 128, 256, 512)
    for dims in (1, 2):
        for recon, lo, hi in (("central", 1.9, 2.1), ("vanleer", 1.7, 2.3), ("mc", 1.7, 2.3), ("minmod", 1.3, 2.3)):
            errs = [wave(n, dims, recon)[0] for n in Ns]
            orders(f"rung 2: density wave {dims}D, {recon}", f"wave{dims}d_{recon}", Ns, errs, lo, hi)
        N5 = (32, 64, 128, 256) if dims == 1 else (32, 64, 128)
        errs = [wave(n, dims, "weno5", fifth=True)[0] for n in N5]
        orders(f"rung 2: density wave {dims}D, weno5 (time step ~ h^(5/3))", f"wave{dims}d_weno5", N5, errs, 4.7, 5.3)
    # every flux formula gives the same order (they differ only in how much they smear)
    for flux in ("rusanov", "hll", "hybrid"):
        errs = [wave(n, 2, "central", flux=flux)[0] for n in Ns[:4]]
        orders(f"rung 2: density wave 2D, no limiter, flux {flux}", f"wave2d_central_{flux}", Ns[:4], errs, 1.9, 2.1)
    # the time scheme: WENO5 at a fixed ratio of time step to cell size, so the error left is the time error
    errs = [wave(n, 1, "weno5", cfl=0.8)[0] for n in (64, 128, 256, 512)]
    orders("rung 2: time scheme, three stages (WENO5 at CFL 0.8)", "time_rk3", (64, 128, 256, 512), errs, 2.8, 3.2)
    errs = [wave(n, 1, "weno5", cfl=0.2, stages=2)[0] for n in (64, 128, 256, 512)]
    orders("rung 2: time scheme, two stages (WENO5 at CFL 0.2)", "time_rk2", (64, 128, 256, 512), errs, 1.8, 2.2)
    # nonlinear and two-dimensional
    Nv = (64, 128, 256, 512)
    for recon in ("central", "vanleer", "weno5"):
        errs = [vortex(n, recon)[0] for n in Nv]
        orders(f"rung 2: isentropic vortex, {recon}", f"vortex_{recon}", Nv, errs, 1.8, 3.5 if recon == "weno5" else 2.3)
    # single and mixed against double precision: error and total mass after one period
    rows = []
    for n in (1024, 4096):
        row = dict(N=n)
        for prec in ("double", "mixed", "float"):
            err, steps, ms, mass = wave(n, 1, "vanleer", prec, cfl=0.4, mass=True)
            row.update({prec: err, prec + "_mass": mass, prec + "_ms": ms, "steps": steps})
        rows.append(row)
    txt = "; ".join(f"N {r['N']} ({r['steps']} steps): error double {r['double']:.2e}, mixed {r['mixed']:.2e}, float {r['float']:.2e}; "
                    f"mass change double {r['double_mass']:+.0e}, mixed {r['mixed_mass']:+.0e}, float {r['float_mass']:+.0e}" for r in rows)
    check("rung 2: mixed precision against double, density wave 1D (van Leer)",
          all(abs(r["mixed"] - r["double"]) < 3e-6 and abs(r["mixed_mass"]) < 1e-9 for r in rows),
          txt + " (rule: mixed within 3e-6 of double, mass kept to 1e-9)")
    check("rung 2: single precision loses mass (recorded as a limit, rule: below 1e-3 here)",
          all(abs(r["float_mass"]) < 1e-3 for r in rows), "float mass change " + ", ".join(f"{r['float_mass']:+.1e}" for r in rows))
    REPORT["precision"] = rows

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "test_euler_smooth.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()
