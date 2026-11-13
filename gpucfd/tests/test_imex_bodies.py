"""Verification ladder, rung 14: the IMEX scheme (implicit diffusion) with immersed bodies.

    python -m gpucfd.tests.test_imex_bodies [part ...]     (inside WSL, from the project root)

Until 2026-11-13 a body needed the explicit scheme: IMEX grew without bound. DESIGN.md, "IMEX with
immersed bodies", has the two causes and the design (a projection that carries its correction onto the
ghost faces, the ghost rule as rows of the diffusion solve). The checks, criteria fixed there in advance:

1. growth    No growth with convection negligible (the linear part of the step), open channel with the
             cylinder, time steps from 0.01 to 4 times h^2 / nu: uniform grid at two positions of the
             cylinder (the second has the stronger feedback, mu = 0.34) and a clustered grid.
2. flow      No growth with the flow: Re 100 from rest at the convective time step, 2 s.
             After it: the ghost faces equal their rule applied to the stored velocity, and the velocity
             is divergence-free with them (what the projection promises).
3. couette   Circular Couette flow (exact), N = 32, 64, 128 at the convective time step: order 2.
4. steady    The steady state is the explicit scheme's and does not depend on the time step: Couette on
             uniform and stretched grids; DFG 2D-1 (Re 20) drag and lift on a uniform and a clustered grid.
5. time      Second order in time with a body: the Re 100 channel from a developed state, four time steps.
6. 3d        The Couette flow extruded along z equals the 2D solution.
7. repro     Two identical runs bit-identical, CUDA graphs on = off.
"""
import json
import sys
from pathlib import Path

import numpy as np

from gpucfd import bodies as B
from gpucfd import gcfd
from gpucfd.tests import test_bodies as tb
from gpucfd.tests import test_stretched_bodies as tsb

FAILS = []
REPORT = {}
RESULTS = Path(__file__).parent / "results"
DFG = tb.DFG


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def channel(n, scheme, grid=None, xc=DFG["xc"], scale=1.0):
    """The DFG channel with the cylinder; scale multiplies the inflow (1e-6: convection negligible)."""
    Um = DFG["Um"] * scale

    def inflow(X, Y, Z):
        return (4 * Um * Y * (DFG["H"] - Y) / DFG["H"] ** 2, 0 * X, 0 * X)
    s = gcfd.NavierStokes(n, (DFG["L"], DFG["H"], 1.0), DFG["nu"], scheme=scheme, grid=grid,
                          boundaries={"x-": ("inflow", inflow), "x+": "outflow", "y-": "wall", "y+": "wall"},
                          bodies=[B.Cylinder((xc, DFG["yc"]), DFG["D"] / 2)])
    q = []
    for c in range(2):
        X, Y, Z = s.face_coords(c)
        q.append(np.where(s.fluid[c], inflow(X, Y, Z)[c], 0.0))
    s.set(q)
    s.project()
    return s


def hmin(s):
    return s._hmin() if s.stretched else min(s.length[d] / s.n[d] for d in range(2))


# --------------------------------------------------------------------------------------- 1. no growth
def growth(n, grid, xc, d, steps=200):
    scale = 1e-6
    s = channel(n, "imex", grid, xc, scale)
    h = hmin(s)
    dt = d * h * h / DFG["nu"]
    prev, ch = s.get(), []
    for k in range(steps):
        st = s.run(dt, 1, 1e-12)
        cur = s.get()
        ch.append(max(float(np.max(np.abs(a - b))) for a, b in zip(cur, prev)) / scale)
        prev = cur
    s.close()
    half = steps // 2
    # the change per step over the second half: largest value there against the largest of the 10 steps before it
    g = max(ch[half:]) / max(ch[half - 10:half])
    return dict(d=d, growth=float(g), first=ch[0], last=ch[-1], poisson_its=st["poisson_its"], helm_its=st["helm_its"])


def test_growth():
    n1, g1 = tsb.dfg_grid(0)
    cases = (("uniform 384 x 72", (384, 72, 1), None, DFG["xc"]),
             ("uniform 384 x 72, cylinder moved 0.37 cell", (384, 72, 1), None, DFG["xc"] + 0.37 * DFG["L"] / 384),
             ("clustered 128 x 56", n1, g1, DFG["xc"]))
    rows, worst = [], 0.0
    for label, n, grid, xc in cases:
        for d in (0.01, 0.15, 0.31, 1.0, 4.0):
            r = growth(n, grid, xc, d)
            r["grid"] = label
            rows.append(r)
            worst = max(worst, r["growth"])
            print(f"      {label}, nu dt / h^2 = {d}: change per step {r['first']:.1e} -> {r['last']:.1e}, largest of the last 100 / "
                  f"before them {r['growth']:.3f}; iterations pressure {r['poisson_its']:.1f}, diffusion {r['helm_its']:.1f}", flush=True)
    check("rung 14: IMEX with a body, convection negligible: the change per step does not grow (200 steps, 5 time steps, 3 grids)",
          worst <= 1.0, f"largest ratio (last 100 steps against the 10 before) {worst:.3f}; the scheme before this change grew "
          "1.17 (first position) and 1.59 (second) per step at the smallest time step")
    REPORT["growth"] = rows


# ----------------------------------------------------------------------------- 2. the flow, and the promise
def rule_error(s):
    """Largest difference between the stored value of a face in a body and its rule applied to the stored fluid faces."""
    rules = B.build(s, s.bodies)
    q = s.get()
    worst = 0.0
    for c, r in enumerate(rules):
        a = q[c].ravel()
        want = r["a0"] + np.sum(r["w"] * a[r["nb"]], axis=1)
        if r["idx"].size:
            worst = max(worst, float(np.max(np.abs(a[r["idx"]] - want))))
    return worst


def test_flow():
    n1, g1 = tsb.dfg_grid(1)
    rows = []
    for label, n, grid in (("uniform 384 x 72", (384, 72, 1), None), ("clustered 256 x 112", n1, g1)):
        s = channel(n, "imex", grid)
        h = hmin(s)
        dt = 0.8 * h / (1.5 * DFG["Um"])
        steps = int(round(2.0 / dt))
        umax, ms, ptol = 0.0, 0.0, 1e-9
        for k in range(10):
            st = s.run(dt, steps // 10, ptol)
            ms += st["ms"]
            umax = max(umax, max(float(np.max(np.abs(a[f]))) for a, f in zip(s.get(), s.fluid)))
        re = rule_error(s)
        r = dict(grid=label, dt=dt, d=dt * DFG["nu"] / h ** 2, steps=10 * (steps // 10), umax=umax, rule=re,
                 div=st["max_div"] * h / (1.5 * DFG["Um"]), poisson_its=st["poisson_its"], helm_its=st["helm_its"],
                 ms_step=ms / (10 * (steps // 10)))
        s.close()
        se = channel(n, "rk3", grid)  # for contrast: the explicit scheme's ghost faces belong to u*, not to u
        se.run(se.max_dt(1.5 * DFG["Um"]), 20, ptol)
        r["rule_rk3"] = rule_error(se)
        se.close()
        rows.append(r)
        print(f"      {label}: dt {dt:.2e} (nu dt / h^2 = {r['d']:.2f}), {r['steps']} steps, max |u| {umax:.3f}; ghost rule kept to "
              f"{re:.1e} (explicit scheme after 20 steps: {r['rule_rk3']:.1e}); max |div u| h / U {r['div']:.1e}; iterations pressure "
              f"{r['poisson_its']:.1f}, diffusion {r['helm_its']:.1f}; {r['ms_step']:.1f} ms per step", flush=True)
    check("rung 14: Re 100 channel from rest at CFL 0.8, 2 s of flow: velocities bounded (below 3 times the inflow maximum)",
          all(r["umax"] < 3 * DFG["Um"] for r in rows), ", ".join(f"{r['grid']}: max |u| {r['umax']:.3f}" for r in rows) +
          f" (inflow maximum {DFG['Um']})")
    check("rung 14: after a step the ghost faces follow their rule (1e-12) and the velocity is divergence-free with them (tolerance 1e-9: below 1e-6)",
          all(r["rule"] < 1e-12 and r["div"] < 1e-6 for r in rows),
          "; ".join(f"{r['grid']}: rule {r['rule']:.1e}, max |div u| h / U {r['div']:.1e}" for r in rows))
    REPORT["flow"] = rows


# ------------------------------------------------------------------------------------------ 3, 4. Couette
def couette(N, scheme, dt_factor=1.0, grid=None, dim3=False, image=None, tol=1e-10):
    if dim3:
        n, length = (N, N, 4), (1.0, 1.0, 4.0 / N)
    else:
        n, length = (N, N, 1), (1.0, 1.0, 1.0)
    ctr = [0.5, 0.5, 0.5]
    bod = [B.Cylinder(ctr, tb.R1, velocity=B.rotation(ctr, tb.W1)), B.Inside(B.Cylinder(ctr, tb.R2))]
    s = gcfd.NavierStokes(n, length, tb.NU_C, walls=(True, True, False), bodies=bod, scheme=scheme, grid=grid,
                          image_distance=image, htol=1e-13)
    ex = tb.couette_exact(2)
    q = []
    for c in range(s.ncomp):
        X, Y, Z = s.face_coords(c)
        q.append(np.where(s.fluid[c], ex(X, Y, Z)[c], 0.0))
    s.set(q)
    s.project()
    dt = s.max_dt(1.0) * dt_factor
    chunk = max(20, int(round(0.5 / dt)))  # about half a time unit between looks
    st, t = tb.steady(s, dt, chunk=chunk, tol=tol, ptol=1e-13)
    out = dict(u=s.get(), fluid=s.fluid, st=st, t=t, dt=dt, d=dt * tb.NU_C / hmin(s) ** 2)
    out["umax"], out["urms"] = tb.face_errors(s, ex)
    s.close()
    return out


def fluid_diff(a, b):
    return max(float(np.max(np.abs(x - y)[f])) for x, y, f in zip(a["u"], b["u"], a["fluid"]))


def test_couette():
    Ns = (32, 64, 128)
    rs = [couette(N, "imex") for N in Ns]
    urms, umax = [r["urms"] for r in rs], [r["umax"] for r in rs]
    o = tb.order(urms)
    ref = json.loads((RESULTS / "test_bodies.json").read_text())["report"]["couette"]["u_rms"]
    check("rung 14: circular Couette flow with IMEX at the convective time step, velocity order 2 (rung 10's rule)",
          tb.ib_ordered(urms),
          f"rms error {', '.join(f'{e:.2e}' for e in urms)} on N = {Ns}, order {', '.join(f'{v:.2f}' for v in o)} (mean "
          f"{tb.mean_order(urms):.2f}); explicit scheme (rung 10): {', '.join(f'{e:.2e}' for e in ref)}; nu dt / h^2 = "
          f"{', '.join(f'{r['d']:.2f}' for r in rs)}; iterations pressure {', '.join(f'{r['st']['poisson_its']:.1f}' for r in rs)}")
    REPORT["couette"] = dict(N=list(Ns), u_rms=urms, u_max=umax, order_rms=o, u_rms_explicit=ref, d=[r["d"] for r in rs],
                             max_div=[r["st"]["max_div"] / N for r, N in zip(rs, Ns)])


def dfg_steady(n, grid, scheme, tol=1e-8):
    """DFG 2D-1 (Re 20) to a steady state: drag and lift coefficients, stopped when both change by less than tol per time unit."""
    Um, D = DFG["Um"], DFG["D"]
    Ubar = 2 * Um / 3
    s = channel(n, scheme, grid)
    dt = s.max_dt(1.5 * Um)
    chunk, t, prev, ms = int(np.ceil(1.0 / dt)), 0.0, None, 0.0
    while t < 200:
        st = s.run(dt, chunk, 1e-11)
        t, ms = t + dt * chunk, ms + st["ms"]
        F = s.forces(tsb.BOX)
        cur = (2 * F[0] / (Ubar ** 2 * D), 2 * F[1] / (Ubar ** 2 * D))
        if prev is not None and abs(cur[0] - prev[0]) < tol * dt * chunk and abs(cur[1] - prev[1]) < tol * dt * chunk:
            break
        prev = cur
    s.close()
    return dict(cD=cur[0], cL=cur[1], t=t, gpu_s=ms / 1000, dt=dt, poisson_its=st["poisson_its"])


def test_steady():
    TOL = 1e-7  # of the wall speed (1): the marches stop at about 1e-9; a slip growing with the time step would be 1e-4 or more
    rows, worst = [], 0.0
    for label, N, grid in (("uniform N = 32", 32, None), ("uniform N = 64", 64, None),
                           ("cell sizes varying by 3, N = 32", 32, (tsb.wavy(32, 1.0), tsb.wavy(32, 1.0), None))):
        e = couette(N, "rk3", grid=grid)
        a = couette(N, "imex", grid=grid)
        b = couette(N, "imex", 0.25, grid=grid)
        r = dict(grid=label, imex_vs_rk3=fluid_diff(a, e), imex_dt_vs_quarter=fluid_diff(a, b), d=a["d"], t=[e["t"], a["t"], b["t"]],
                 steps=[int(round(x["t"] / x["dt"])) for x in (e, a, b)])
        rows.append(r)
        worst = max(worst, r["imex_vs_rk3"], r["imex_dt_vs_quarter"])
        print(f"      Couette, {label}: IMEX - RK3 {r['imex_vs_rk3']:.1e}; IMEX at dt (nu dt / h^2 = {a['d']:.2f}) - at dt / 4 "
              f"{r['imex_dt_vs_quarter']:.1e}; steps to steady: RK3 {r['steps'][0]}, IMEX {r['steps'][1]}", flush=True)
    check("rung 14: Couette steady state, IMEX = explicit scheme and independent of the time step (1e-7 of the wall speed)",
          worst < TOL, f"largest difference over the fluid faces {worst:.1e} (3 grids; IMEX at dt, at dt / 4, and RK3)")
    REPORT["steady_couette"] = rows
    n1, g1 = tsb.dfg_grid(0)
    rows = []
    for label, n, grid in (("uniform 384 x 72", (384, 72, 1), None), ("clustered 128 x 56", n1, g1)):
        e, a = dfg_steady(n, grid, "rk3"), dfg_steady(n, grid, "imex")
        r = dict(grid=label, rk3=e, imex=a, d_cD=abs(a["cD"] - e["cD"]) / e["cD"], d_cL=abs(a["cL"] - e["cL"]) / e["cD"])
        rows.append(r)
        print(f"      DFG 2D-1, {label}: cD RK3 {e['cD']:.8f}, IMEX {a['cD']:.8f}; cL {e['cL']:.8f}, {a['cL']:.8f}; GPU {e['gpu_s']:.0f} s "
              f"and {a['gpu_s']:.0f} s (time steps {e['dt']:.2e}, {a['dt']:.2e})", flush=True)
    check("rung 14: DFG 2D-1 (Re 20) steady drag and lift, IMEX = explicit scheme (1e-6 of the drag)",
          all(r["d_cD"] < 1e-6 and r["d_cL"] < 1e-6 for r in rows),
          "; ".join(f"{r['grid']}: drag {r['d_cD']:.1e}, lift {r['d_cL']:.1e} of the drag" for r in rows))
    REPORT["steady_dfg"] = rows


# ------------------------------------------------------------------------------------- 5. order in time
def test_time():
    n = (384, 72, 1)
    Um = DFG["Um"]
    h = DFG["L"] / n[0]
    dt0 = 0.8 * h / (1.5 * Um)
    T, nbase = 0.2, 20
    dt0 = T / nbase  # 0.01: CFL 0.79
    s = channel(n, "imex")
    s.run(dt0 / 4, 400, 1e-11)  # 1 s of flow from rest: the wake is growing (a smooth unsteady state)
    u0, p0 = s.get(), s.pressure()
    fl = s.fluid
    sols = []
    for k in range(4):
        s.set(u0)
        s.set_pressure(p0)
        s.run(dt0 / 2 ** k, nbase * 2 ** k, 1e-12)
        sols.append(s.get())
    s.close()
    e = channel(n, "rk3")
    e.set(u0)
    e.set_pressure(p0)
    dte = dt0 / 16
    assert dte <= e.max_dt(1.5 * Um)
    e.run(dte, nbase * 16, 1e-12)
    ue = e.get()
    e.close()

    def l2(a, b):
        return float(np.sqrt(sum(np.sum((x - y)[f] ** 2) for x, y, f in zip(a, b, fl)) / sum(int(f.sum()) for f in fl)))
    change = l2(sols[-1], u0)
    dif = [l2(sols[k], sols[k + 1]) for k in range(3)]
    orders = tb.order(dif)
    vs_e = [l2(sols[k], ue) for k in range(4)]
    check("rung 14: IMEX with a body is second order in time (Re 100 channel, 0.2 s, time steps dt to dt / 8; order from successive differences >= 1.8)",
          all(v >= 1.8 for v in orders),
          f"differences {', '.join(f'{v:.2e}' for v in dif)} (rms over the fluid faces; the flow itself changes by {change:.1e}): order "
          f"{', '.join(f'{v:.2f}' for v in orders)}; dt = {dt0:g} (CFL {dt0 * 1.5 * Um / h:.2f}, nu dt / h^2 = {dt0 * DFG['nu'] / h ** 2:.2f})")
    check("rung 14: the IMEX solutions converge to the explicit scheme's (difference at dt / 8 below the last IMEX difference)",
          vs_e[-1] <= dif[-1] and vs_e[0] > vs_e[1] > vs_e[2] > vs_e[3],
          f"IMEX at dt, dt / 2, dt / 4, dt / 8 against RK3 at dt / 16: {', '.join(f'{v:.2e}' for v in vs_e)} "
          f"(order {', '.join(f'{v:.2f}' for v in tb.order(vs_e))})")
    REPORT["time"] = dict(dt=dt0, differences=dif, orders=orders, vs_explicit=vs_e, flow_change=change)


# ------------------------------------------------------------------------------------------- 6, 7
def test_3d():
    a = couette(32, "imex", image=np.sqrt(3) + 0.05, tol=1e-9)
    r = couette(32, "imex", dim3=True, tol=1e-9)
    u = r["u"]
    d = max(max(float(np.max(np.abs(u[c][k] - a["u"][c][0]))) for k in range(4)) for c in range(2))
    d = max(d, float(np.max(np.abs(u[2]))))
    check("rung 14: Couette in 3D with IMEX (extruded along z) equals the 2D solution", d < 1e-9, f"max difference {d:.1e} (N = 32)")
    REPORT["couette_3d"] = d


def test_repro():
    def run(graphs):
        gcfd.use_graphs(graphs)
        n, grid = tsb.dfg_grid(0)
        s = channel(n, "imex", grid)
        s.run(0.8 * hmin(s) / (1.5 * DFG["Um"]), 30, 1e-8)
        u = s.get() + [s.pressure()]
        s.close()
        gcfd.use_graphs(True)
        return u
    a, b, c = run(True), run(True), run(False)
    same = all(np.array_equal(x, y) for x, y in zip(a, b))
    same_g = all(np.array_equal(x, y) for x, y in zip(a, c))
    check("rung 14: IMEX with a body on a clustered grid, two identical runs bit-identical, CUDA graphs on = off", same and same_g,
          f"second run identical: {same}; graphs off identical: {same_g}")
    REPORT["reproducible"] = dict(rerun=bool(same), graphs_off=bool(same_g))


PARTS = dict(growth=test_growth, flow=test_flow, couette=test_couette, steady=test_steady, time=test_time,
             **{"3d": test_3d}, repro=test_repro)

if __name__ == "__main__":
    args = sys.argv[1:]
    for name in (args or PARTS):
        PARTS[name]()
    if not args:
        (RESULTS / "test_imex_bodies.json").write_text(json.dumps(dict(report=REPORT, fails=FAILS), indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))
