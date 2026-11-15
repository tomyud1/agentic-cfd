"""Compressible ladder, rung 4: a Mach 5.09 shock going round a sharp corner (case 19), and before it the
question which flux formula keeps a flat strong shock flat.

    python -m gpucfd.tests.test_euler_corner [finest N, default 1600]      (inside WSL, from the project root)
    python -m gpucfd.tests.test_euler_corner only 3200 vanleer hybrid       (one finer grid, its own result file)

The problem is cases/shock_diffraction_corner/problem.yaml with its required_accuracy. The plan with the pass
rules and the expectations was written before the runs: dev/compare/shock_diffraction_corner/plan.yaml.
The exact state behind the shock is computed here from the jump conditions (and checked against the three
conservation laws). The outputs are read with the case's own rules (cases/shock_diffraction_corner/measure.py,
extraction settings of its numerics.yaml), so they stand next to OpenFOAM's numbers of the same case
(results/shock_diffraction_corner/results.json) like for like.

  a. A flat Mach 5.09 shock in a plain duct, the still air disturbed by noise of 1e-4: every flux formula with
     van Leer and with WENO5. Which stay straight?
  b. The corner on 200 x 200 with each of them: which survive the expansion round the corner?
  c. The corner on N x N, N = 100 ... finest, with the chosen setting: the problem's requirements.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cases.shock_diffraction_corner import measure
from cfdlib import foam
from gpucfd.euler import Blowup, Euler

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cases" / "shock_diffraction_corner"
OF = ROOT / "results" / "shock_diffraction_corner" / "results.json"
OUT = Path(__file__).parent / "results"
FAILS = []
REPORT = {}
CFL = 0.4
PRECISION = "mixed"
SNAPSHOTS = 10
RECORD = 1600          # the run to this grid is the recorded one; a run to another finest grid writes files of its own
TAG = ""
# best first, by how sharp each kept the border between the two gases in the shock tube (rung 3e)
PREFERENCE = [("weno5", "hllc"), ("weno5", "hybrid"), ("weno5", "hll"), ("weno5", "rusanov"),
              ("vanleer", "hllc"), ("vanleer", "hybrid"), ("vanleer", "hll"), ("vanleer", "rusanov")]


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


class Flow:
    """The exact part of the problem: the state behind a shock of Mach number M_s running into still air."""

    def __init__(self, g):
        self.L, self.H, self.Hs, self.t_end, self.gamma = g["L"], g["H"], g["H_s"], g["t_end"], g["gamma"]
        gm, M = g["gamma"], g["M_s"]
        self.p0, self.rho0 = g["p_0"], g["p_0"] / (g["R"] * g["T_0"])
        self.c0 = np.sqrt(gm * g["R"] * g["T_0"])
        self.W = M * self.c0                                                   # shock speed
        self.p2 = self.p0 * (1 + 2 * gm / (gm + 1) * (M * M - 1))
        self.rho2 = self.rho0 * (gm + 1) * M * M / ((gm - 1) * M * M + 2)
        self.u2 = self.W * (1 - self.rho0 / self.rho2)
        self.c2 = np.sqrt(gm * self.p2 / self.rho2)
        self.E0 = self.p0 / (gm - 1)
        self.E2 = self.p2 / (gm - 1) + 0.5 * self.rho2 * self.u2 ** 2
        self.mdot = self.rho2 * self.u2 * (self.H - self.Hs)                   # per metre of depth
        self.edot = (self.E2 + self.p2) * self.u2 * (self.H - self.Hs)
        self.mass0, self.energy0 = self.rho0 * self.L * self.H, self.E0 * self.L * self.H
        self.v_flat = np.sqrt(self.c2 ** 2 - (self.W - self.u2) ** 2)          # the corner's signal along the shock
        self.x_top = self.W * self.t_end
        self.state = dict(rho=self.rho2, u=self.u2, p=self.p2)

    def flat_from(self, t):
        return self.Hs + self.v_flat * t

    def signal_top(self, t):
        return self.Hs + self.c2 * t

    def jump_residuals(self):
        """The three conservation laws across the shock, seen from the shock (each relative; zero if exact)."""
        a, b = self.W, self.W - self.u2
        h0, h2 = self.gamma / (self.gamma - 1) * self.p0 / self.rho0, self.gamma / (self.gamma - 1) * self.p2 / self.rho2
        return (self.rho0 * a / (self.rho2 * b) - 1, (self.p0 + self.rho0 * a * a) / (self.p2 + self.rho2 * b * b) - 1,
                (h0 + 0.5 * a * a) / (h2 + 0.5 * b * b) - 1)


def given():
    g = {k: v["value"] for k, v in yaml.safe_load((CASE / "problem.yaml").read_text(encoding="utf-8"))["given"].items()}
    ext = yaml.safe_load((CASE / "numerics.yaml").read_text(encoding="utf-8"))["extraction"]
    return g, ext


def duct(fl, recon, flux, nx=400, ny=100, noise=1e-4, seed=1):
    """A flat shock of the problem's strength in a plain duct with noisy still air. Returns how straight it stays."""
    s = Euler((nx, ny, 1), (fl.L, fl.L * ny / nx, 1.0), fl.gamma, precision=PRECISION, recon=recon, flux=flux,
              sides={"x-": ("supersonic", fl.state), "x+": "wall", "y-": "wall", "y+": "wall"})
    rng = np.random.default_rng(seed)
    s.set(rho=fl.rho0 * (1 + noise * (2 * rng.random(s.shape) - 1)), p=fl.p0)
    try:
        st = s.run(fl.t_end, cfl=CFL)
    except Blowup as err:
        s.close()
        return dict(recon=recon, flux=flux, broke=str(err)[:160])
    f = s.get()
    s.close()
    h = fl.L / nx
    p = f["p"][0]
    xs = measure.front_x(p, 0.5 * (fl.p0 + fl.p2), h)
    x = (np.arange(nx) + 0.5) * h
    behind = (x < xs.mean() - 6 * h) & (x > xs.mean() - 40 * h)
    return dict(recon=recon, flux=flux, steps=st["steps"], dev_cells=float(np.max(np.abs(xs - xs.mean())) / h),
                p_along=float(np.max(p[:, behind].max(axis=0) - p[:, behind].min(axis=0)) / fl.p2),
                v_max=float(np.max(np.abs(f["v"][0][:, behind])) / fl.u2), x_err_cells=float((xs.mean() - fl.W * fl.t_end) / h))


def corner(fl, ext, N, recon, flux, keep_fields=False):
    """The corner problem on N x N cells. Returns the outputs read by the case's rules."""
    t0 = time.perf_counter()
    half = int(round(N * fl.Hs / fl.H))
    s = Euler((N, N, 1), (fl.L, fl.H, 1.0), fl.gamma, precision=PRECISION, recon=recon, flux=flux,
              sides={k: "wall" for k in ("x-", "x+", "y-", "y+")})
    s.segment("x-", {"y": (half, N)}, ("supersonic", fl.state))
    s.set(rho=fl.rho0, p=fl.p0)
    h = fl.L / N
    out = dict(N=N, h=h, recon=recon, flux=flux, cells=N * N, bytes=s.bytes, steps=0, gpu_s=0.0)
    flat, wall, written, wall_rho, line_rho = [], [], [], [], []
    lows = dict(rho=np.inf, p=np.inf)
    for k in range(1, SNAPSHOTS + 1):
        t = fl.t_end * k / SNAPSHOTS
        st = s.run(fl.t_end / SNAPSHOTS, cfl=CFL)
        out["steps"] += st["steps"]
        out["gpu_s"] += st["ms"] / 1e3
        lows = dict(rho=min(lows["rho"], st["rho_min"]), p=min(lows["p"], st["p_min"]))
        f = s.get()
        rho, u, v, p = (f[q][0] for q in ("rho", "u", "v", "p"))
        fp = measure.flat_part(p, rho, u, v, t, fl, ext, h)
        fp["t"] = t
        flat.append(fp)
        yw, peak = measure.wall_shock(p[:half, 0], fl.p0, h, ext["wall_start"])
        wall.append(dict(t=t, d=fl.Hs - yw, peak=peak))
        tot = s.totals()
        written.append(dict(t=t, mass_err=(tot["mass"] - fl.mass0 - fl.mdot * t) / (fl.mdot * t),
                            energy_err=(tot["energy"] - fl.energy0 - fl.edot * t) / (fl.edot * t)))
        wall_rho.append(rho[:half, 0].copy())
        line_rho.append(0.5 * (rho[half - 1] + rho[half]))
    out["wall_s"] = time.perf_counter() - t0
    s.close()
    # as cases/shock_diffraction_corner/numbers.py forms OpenFOAM's outputs
    tt = np.array([q["t"] for q in flat])
    late = tt >= ext["speed_from"] * fl.t_end - 1e-12
    xm = np.array([q.get("x_mean", np.nan) for q in flat])
    end = flat[-1]
    speed = float(np.polyfit(tt[late], xm[late], 1)[0])
    out.update(speed=speed, speed_err=speed / fl.W - 1, x_top_err_cells=(end["x_top"] - fl.x_top) / h, flat_rows=end["rows"],
               dev_cells_end=end["dev_mean_cells"], dev_line_end=end["dev_line_cells"],
               dev_cells_max=float(max(q["dev_mean_cells"] for q in flat if q.get("ok"))),
               state={q: end[q] for q in ("p", "rho", "u")}, v_max=end["v_max"], p_row_spread=end.get("p_row_spread"),
               state_all=float(max(abs(q[c]["err"]) for q in flat if "p" in q for c in ("p", "rho", "u"))),
               ripple_p_all=float(max(q["p"]["ripple"] for q in flat if "p" in q)),
               d_wall=float(wall[-1]["d"]), wall_peak=float(wall[-1]["peak"]),
               mass_vs_inflow=float(max(abs(q["mass_err"]) for q in written)), energy_vs_inflow=float(max(abs(q["energy_err"]) for q in written)),
               lows=lows, far=dict(floor=float(np.max(np.abs(p[0] / fl.p0 - 1))), right=float(np.max(np.abs(p[:, -1] / fl.p0 - 1)))),
               us_per_cell_step=1e6 * out["gpu_s"] / (N * N * max(out["steps"], 1)))
    i_half, i_end = SNAPSHOTS // 2 - 1, SNAPSHOTS - 1
    a, b = wall_rho[i_half][::-1][:N // 4], measure.restrict1(wall_rho[i_end][::-1])
    a2, b2 = line_rho[i_half][:N // 2], measure.restrict1(line_rho[i_end])
    out.update(sim_wall=measure.mean_rel_diff(a, b), sim_line=measure.mean_rel_diff(a2, b2))
    if keep_fields:
        out["_rho"], out["_p"] = rho, p
    return out


def picture(r, fl, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    rho = r["_rho"]
    k = max(1, rho.shape[0] // 800)
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
    im = ax.imshow(rho[::k, ::k], origin="lower", extent=(0, fl.L, 0, fl.H), cmap="viridis")
    ax.plot([0, 0], [0, fl.Hs], color="white", lw=3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"gpucfd: density at t_end, {r['N']} x {r['N']} cells ({r['recon']}, {r['flux']})", fontsize=9)
    fig.colorbar(im, ax=ax, label="density (kg/m³)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path.name)


def slim(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def main():
    global TAG
    finest = int(sys.argv[1]) if len(sys.argv) > 1 else RECORD
    TAG = "" if finest == RECORD else f"_to{finest}"
    g, ext = given()
    fl = Flow(g)
    res = fl.jump_residuals()
    check("rung 4: the state behind the shock satisfies the three conservation laws across it", max(abs(v) for v in res) < 1e-13,
          f"mass, momentum, energy residuals {res[0]:.1e}, {res[1]:.1e}, {res[2]:.1e}; shock speed {fl.W:.2f} m/s, behind it p {fl.p2:.0f} Pa, "
          f"rho {fl.rho2:.4f} kg/m3, u {fl.u2:.2f} m/s (Mach {fl.u2 / fl.c2:.3f})")
    REPORT["exact"] = dict(W=fl.W, p2=fl.p2, rho2=fl.rho2, u2=fl.u2, M2=fl.u2 / fl.c2, residuals=res)
    busy = [q[3][:60] for q in foam.running()]
    REPORT.update(busy_cpu=bool(busy), busy_processes=busy)
    print(f"  OpenFOAM processes running on the CPU while the times are taken: {len(busy)}", flush=True)

    # ---- a. which flux formula keeps a flat Mach 5.09 shock flat ----
    ducts = [duct(fl, recon, flux) for recon, flux in PREFERENCE]
    for d in ducts:
        print("  duct " + (f"{d['recon']:8s} {d['flux']:8s}: shock off a straight line by {d['dev_cells']:.4f} cells, pressure along the shock "
                           f"{100 * d['p_along']:.3f} % of p2, cross velocity {100 * d['v_max']:.3f} % of u2, position {d['x_err_cells']:+.2f} cells"
                           if "broke" not in d else f"{d['recon']:8s} {d['flux']:8s}: BROKE ({d['broke']})"), flush=True)
    REPORT["duct"] = ducts
    straight = [(d["recon"], d["flux"]) for d in ducts if "broke" not in d and d["dev_cells"] <= 0.1]
    check("rung 4a: at least one flux formula keeps the flat Mach 5.09 shock straight to 0.1 cell (noisy still air, 400 x 100)", bool(straight),
          f"straight: {', '.join('/'.join(q) for q in straight) or 'none'}; not straight or broken: "
          f"{', '.join('/'.join((d['recon'], d['flux'])) for d in ducts if (d['recon'], d['flux']) not in straight) or 'none'}")

    # ---- b. which survive the corner ----
    survive = []
    for recon, flux in PREFERENCE:
        try:
            r = corner(fl, ext, 200, recon, flux)
            survive.append(dict(recon=recon, flux=flux, dev_cells_max=r["dev_cells_max"], ripple=r["ripple_p_all"], rho_min=r["lows"]["rho"],
                                p_min=r["lows"]["p"], d_wall=r["d_wall"], state_all=r["state_all"], sim_wall=r["sim_wall"]))
            print(f"  corner 200 x 200 {recon:8s} {flux:8s}: ran; flat part off by {r['dev_cells_max']:.1e} cells, ripple {100 * r['ripple_p_all']:.2f} %, "
                  f"lowest density {r['lows']['rho']:.4f}, lowest pressure {r['lows']['p']:.0f}", flush=True)
        except Blowup as err:
            survive.append(dict(recon=recon, flux=flux, broke=str(err)[:160]))
            print(f"  corner 200 x 200 {recon:8s} {flux:8s}: BROKE ({str(err)[:120]})", flush=True)
    REPORT["corner_200"] = survive
    ok = [(q["recon"], q["flux"]) for q in survive if "broke" not in q and (q["recon"], q["flux"]) in straight]
    check("rung 4b: a setting that is straight in the duct also runs the corner on 200 x 200", bool(ok),
          f"run and straight: {', '.join('/'.join(q) for q in ok) or 'none'}; broke: "
          f"{', '.join('/'.join((q['recon'], q['flux'])) for q in survive if 'broke' in q) or 'none'}")
    if not ok:
        return finish()
    recon, flux = ok[0]
    print(f"  chosen for the corner: {recon} / {flux}", flush=True)
    REPORT["chosen"] = dict(recon=recon, flux=flux, cfl=CFL, precision=PRECISION)

    # ---- c. the grid family, with the chosen setting and with a second one for comparison ----
    fam = family(fl, ext, recon, flux, 100, finest, "rung 4c")
    if fam:
        REPORT["family"] = [slim(r) for r in fam]
        REPORT["picture"] = picture(fam[-1], fl, OUT / f"euler_corner_density{TAG}.png")
        REPORT["openfoam"] = against_openfoam(fam)
    # van Leer with the hybrid flux: OpenFOAM's limiter, a sharper flux. Not part of the choice above; run because
    # WENO5 only survives the corner with the most smearing flux, and this is the natural alternative.
    second = ("vanleer", "hybrid")
    if second in ok and second != (recon, flux):
        fam2 = family(fl, ext, *second, 100, finest, "rung 4c, second setting")
        if fam2:
            REPORT["family_second"] = [slim(r) for r in fam2]
            REPORT["picture_second"] = picture(fam2[-1], fl, OUT / f"euler_corner_density_vanleer_hybrid{TAG}.png")
            REPORT["openfoam_second"] = against_openfoam(fam2)
    finish()


def family(fl, ext, recon, flux, first, finest, tag):
    """The corner on first, 2 first, ... finest cells per side, with the problem's requirements checked."""
    fam, N = [], first
    while N <= finest:
        try:
            r = corner(fl, ext, N, recon, flux, keep_fields=(2 * N > finest))
        except Blowup as err:
            check(f"{tag}: the run on {N} x {N} does not break", False, str(err)[:200])
            break
        fam.append(r)
        print(f"  {recon}/{flux} N {N:5d}: {r['steps']} steps, GPU {r['gpu_s']:.1f} s, wall {r['wall_s']:.1f} s, {r['bytes'] / 2 ** 20:.0f} MB | speed {100 * r['speed_err']:+.3f} %, "
              f"state p {100 * r['state']['p']['err']:+.3f} % rho {100 * r['state']['rho']['err']:+.3f} % u {100 * r['state']['u']['err']:+.3f} % | "
              f"mass {r['mass_vs_inflow']:.1e} energy {r['energy_vs_inflow']:.1e} of the inflow | flat part off by {r['dev_cells_max']:.1e} cells, "
              f"ripple {100 * r['ripple_p_all']:.2f} % | step face {r['d_wall']:.4f} m, top {r['x_top_err_cells']:+.2f} cells | two times: face "
              f"{100 * r['sim_wall']:.2f} %, line {100 * r['sim_line']:.2f} % | lowest rho {r['lows']['rho']:.4f}", flush=True)
        N *= 2
    if not fam:
        return fam
    fine = fam[-1]
    st = max(abs(fine["state"][q]["err"]) for q in ("p", "rho", "u"))
    check(f"{tag} ({recon}/{flux}), required: flat part on the finest grid ({fine['N']} x {fine['N']}): speed within 0.5 %, state behind within 1 %",
          abs(fine["speed_err"]) <= 0.005 and st <= 0.01, f"speed {100 * fine['speed_err']:+.4f} %, largest state error {100 * st:.4f} %")
    mw, ew = max(r["mass_vs_inflow"] for r in fam), max(r["energy_vs_inflow"] for r in fam)
    check(f"{tag} ({recon}/{flux}), required: mass and energy within 0.01 % of the inflow at every written time, every grid", max(mw, ew) <= 1e-4,
          f"largest difference: mass {mw:.1e}, energy {ew:.1e} of the inflow")
    dv = max(r["dev_cells_max"] for r in fam)
    check(f"{tag} ({recon}/{flux}), required: flat part straight to within one cell on every grid, every written time", dv <= 1.0,
          f"largest deviation {dv:.1e} cells")
    check(f"{tag} ({recon}/{flux}): no wave has reached the floor or the right side (the flow has no length of its own yet)",
          max(max(r["far"].values()) for r in fam) < 1e-6, f"largest pressure change there {max(max(r['far'].values()) for r in fam):.1e}")
    return fam


def against_openfoam(fam):
    """OpenFOAM's recorded numbers of case 19 on the same grids (read by the same rules), printed next to these."""
    if not OF.exists():
        print("  results/shock_diffraction_corner/results.json is not there: the comparison with OpenFOAM is open", flush=True)
        return None
    of = json.loads(OF.read_text(encoding="utf-8"))["data"]["runs"]
    rows = []
    for r in fam:
        o = of.get(f"N{r['N']}")
        if not o:
            continue
        rows.append(dict(N=r["N"], openfoam={k: o[k] for k in ("steps", "wall_time", "cores", "speed_err", "dev_cells_max", "ripple_p_all", "d_wall",
                                                                 "x_top_err_cells", "sim_wall", "sim_line", "state_all")}
                         | dict(state={q: o["state"][q]["err"] for q in ("p", "rho", "u")}, mass_vs_inflow=o["balance"]["mass_vs_inflow"],
                                energy_vs_inflow=o["balance"]["energy_vs_inflow"], rho_min=o["lows"]["rho"])))
        o = rows[-1]["openfoam"]
        print(f"  N {r['N']:5d} OpenFOAM | gpucfd: speed {100 * o['speed_err']:+.3f} | {100 * r['speed_err']:+.3f} %; state p {100 * o['state']['p']:+.3f} | "
              f"{100 * r['state']['p']['err']:+.3f} %, rho {100 * o['state']['rho']:+.3f} | {100 * r['state']['rho']['err']:+.3f} %; ripple "
              f"{100 * o['ripple_p_all']:.2f} | {100 * r['ripple_p_all']:.2f} %; step face {o['d_wall']:.4f} | {r['d_wall']:.4f} m; two times face "
              f"{100 * o['sim_wall']:.2f} | {100 * r['sim_wall']:.2f} %, line {100 * o['sim_line']:.2f} | {100 * r['sim_line']:.2f} %; lowest rho "
              f"{o['rho_min']:.4f} | {r['lows']['rho']:.4f}; steps {o['steps']} | {r['steps']}; time {o['wall_time']:.1f} s on {o['cores']} core(s) | "
              f"{r['wall_s']:.1f} s", flush=True)
    return rows


def one(N, recon, flux):
    """A single finer grid, outside the ladder run:  python -m gpucfd.tests.test_euler_corner only 3200 vanleer hybrid"""
    g, ext = given()
    fl = Flow(g)
    fam = family(fl, ext, recon, flux, N, N, f"rung 4c, {N} x {N} alone")
    if fam:
        REPORT["run"] = slim(fam[0])
        REPORT["picture"] = picture(fam[0], fl, OUT / f"euler_corner_density_N{N}_{recon}_{flux}.png")
    OUT.mkdir(exist_ok=True)
    (OUT / f"test_euler_corner_N{N}_{recon}_{flux}.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


def finish():
    OUT.mkdir(exist_ok=True)
    (OUT / f"test_euler_corner{TAG}.json").write_text(json.dumps(REPORT, indent=1, default=float))
    print(f"\n{len(FAILS)} failed" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(len(FAILS))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "only":
        one(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    main()
