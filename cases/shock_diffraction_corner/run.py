"""Build and run the shock at the corner for every run in numerics.yaml; save what the analysis needs.

Usage (repository root, inside WSL):
  python -m cases.shock_diffraction_corner.run preflight              # short tests on the coarsest and finest grid (the
                                                                      # finest on 1, 8 and 16 cores), then in full: the
                                                                      # two coarsest grids and the time-step study
  python -m cases.shock_diffraction_corner.run production [NAME ...]  # every other run (default: all not yet made), one at a time
  python -m cases.shock_diffraction_corner.run recollect [NAME ...]   # read finished runs again (after a change of a reading rule)
  python -m cases.shock_diffraction_corner.run list                   # the runs and their time steps
Outputs: results/shock_diffraction_corner/preflight.json, runs/<name>.json + runs/<name>.npz
Run names: N<cells per side>, plus _spc<steps per cell> or _<flux>_<limiter> when a run differs from the main choice.
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cfdlib import contract, foam, foamdict, monitor, perf
from cfdlib.report import tex_num
from . import measure, theory

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"


def load():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    return problem, numerics


def run_root():
    return Path(foam.foam_env("FOAM_RUN")) / "agentic" / HERE.name


def specs(numerics):
    """Every run as {name: dict(name, N, spc, flux, limiter, studies)}: grid family, time-step study, scheme variants."""
    main, out = numerics["main"], {}

    def add(N, spc, flux, limiter, study):
        name = f"N{N}"
        if spc != main["steps_per_cell"]:
            name += f"_spc{spc:g}"
        if (flux, limiter) != (main["flux"], main["limiter"]):
            name += f"_{flux}_{limiter}"
        s = out.setdefault(name, dict(name=name, N=N, spc=spc, flux=flux, limiter=limiter, studies=[]))
        s["studies"].append(study)

    for N in numerics["runs"]["family"]:
        add(N, main["steps_per_cell"], main["flux"], main["limiter"], "grid")
    ts = numerics["runs"]["timestep"]
    for spc in ts["steps_per_cell"]:
        add(ts["cells"], spc, main["flux"], main["limiter"], "time step")
    for N, spc in numerics["runs"].get("stress", []):
        add(N, spc, main["flux"], main["limiter"], "stress")
    sc = numerics["runs"]["schemes"]
    add(sc["cells"], main["steps_per_cell"], main["flux"], main["limiter"], "scheme")
    for flux, limiter in sc["variants"]:
        add(sc["cells"], main["steps_per_cell"], flux, limiter, "scheme")
    return out


def steps_of(spec, numerics):
    n = spec["spc"] * spec["N"]
    assert abs(n - round(n)) < 1e-9 and round(n) % numerics["snapshots"] == 0, \
        f"{spec['name']}: {n} steps is not a whole multiple of the {numerics['snapshots']} snapshots"
    return int(round(n))


def ranks_of(spec, numerics, ranks=None):
    if ranks is not None:
        return ranks
    par = numerics["parallel"]
    return par["ranks"] if spec["N"] >= par["from_cells"] else 1


def solver_cmd(solver, ranks):
    if ranks == 1:
        return solver
    return f"mpirun {'--use-hwthread-cpus ' if ranks > 8 else ''}-np {ranks} {solver} -parallel"


def params_for(spec, fl, numerics, ranks, end_steps=None):
    N, n = spec["N"], steps_of(spec, numerics)
    n_low = round(N * fl.Hs / fl.H)
    assert abs(n_low - N * fl.Hs / fl.H) < 1e-9, "the corner must lie on a cell face"
    dt = fl.t_end / n
    limiter = spec["limiter"]
    return dict(N=N, n_low=n_low, n_up=N - n_low, L=repr(fl.L), H=repr(fl.H), Hs=repr(fl.Hs), depth=repr(numerics["depth"]),
                steps=n, dt=repr(dt), end_time=repr(dt * end_steps) if end_steps else repr(fl.t_end),
                write_every=n // numerics["snapshots"], flux=spec["flux"], limiter=limiter,
                limiter_vector=limiter if limiter == "upwind" else limiter + "V",
                p0=repr(fl.p0), T0=repr(fl.T0), p2=repr(fl.p2), T2=repr(fl.T2), u2=repr(fl.u2),
                mol_weight=repr(foam.mol_weight(fl.R)), Cp=repr(fl.gamma * fl.R / (fl.gamma - 1)), ranks=ranks)


def make(spec, fl, numerics, prefix="", ranks=None, **kw):
    """Render the case and make the grid (and cut it into parts for a run on several cores)."""
    ranks = ranks_of(spec, numerics, ranks)
    case = foam.render_case(HERE / "template", run_root() / f"{prefix}{spec['name']}", params_for(spec, fl, numerics, ranks, **kw))
    foam.run("blockMesh", case)
    foam.run("checkMesh", case)
    if ranks > 1:
        foam.run("decomposePar", case)
    return case, ranks


def _dat(case, fo, fname):
    hits = sorted((Path(case) / "postProcessing" / fo).rglob(fname))
    rows = foam.read_dat(hits[0])
    head = [l for l in hits[0].read_text().splitlines() if l.startswith("#")][-1]
    return np.array(rows, dtype=float), head


def totals(case, fl, numerics):
    """Mass and energy in the chamber and the mass flow through the opening after every step, with the exact
    values they must have: the start plus the exact inflow rate times the time."""
    c, head_c = _dat(case, "conserved", "volFieldValue.dat")
    q, head_q = _dat(case, "inflow", "surfaceFieldValue.dat")
    # The files print the time with 10 digits; use step number times the time step the case ran with.
    dt = float(foamdict.parse(Path(case) / "system/controlDict")["deltaT"])
    t = dt * (np.arange(len(c)) + 1)
    assert np.allclose(t, c[:, 0], rtol=1e-8, atol=0), "totals file: rows are not one per time step"
    depth = numerics["depth"]
    return dict(t=t, mass=c[:, 1] / depth, energy=c[:, 2] / depth, inflow=-q[:, 1] / depth, head=[head_c, head_q])


def balance(tot, fl):
    """Largest difference between the chamber's mass and energy and (start + exact inflow), as a fraction of
    the inflow up to that time and as a fraction of the total."""
    m_ex, e_ex = fl.totals(tot["t"])
    dm, de = tot["mass"] - m_ex, tot["energy"] - e_ex
    return dict(mass_vs_inflow=float(np.max(np.abs(dm) / (fl.mdot * tot["t"]))), energy_vs_inflow=float(np.max(np.abs(de) / (fl.edot * tot["t"]))),
                mass_vs_total=float(np.max(np.abs(dm) / m_ex)), energy_vs_total=float(np.max(np.abs(de) / e_ex)),
                inflow_rate=float(np.max(np.abs(tot["inflow"] / fl.mdot - 1))))


def collect(case, spec, fl, numerics):
    """Read the written fields of a finished run and apply the reading rules. Returns (arrays, numbers)."""
    N, ext = spec["N"], numerics["extraction"]
    h, half = fl.L / N, spec["N"] // 2
    tds = [t for t in foam.time_dirs(case) if float(t.name) > 0]
    n_snap = numerics["snapshots"]
    full = len(tds) == n_snap
    steps, dt = steps_of(spec, numerics), fl.t_end / steps_of(spec, numerics)
    every = steps // n_snap
    d = dict(times=[], front_low=[], wall_p=[], wall_rho=[], line_rho=[], line_p=[])
    flat, wall = [], []
    lows = dict(rho=np.inf, p=np.inf, T=np.inf)
    for k, td in enumerate(tds):
        t = (k + 1) * every * dt                                    # exact; the folder name has 10 digits
        assert abs(t - float(td.name)) < 1e-8 * t, f"{td}: not snapshot {k + 1}"
        p = foam.read_internal_field(td / "p").reshape(N, N)
        rho = foam.read_internal_field(td / "rho").reshape(N, N)
        U = foam.read_internal_field(td / "U")
        u, v = U[:, 0].reshape(N, N), U[:, 1].reshape(N, N)
        lows = dict(rho=min(lows["rho"], float(rho.min())), p=min(lows["p"], float(p.min())),
                    T=min(lows["T"], float((p / (rho * fl.R)).min())))
        d["times"].append(t)
        d["front_low"].append(measure.front_x(p, ext["low_level"] * fl.p0, h))
        d["wall_p"].append(p[:half, 0].copy())
        d["wall_rho"].append(rho[:half, 0].copy())
        d["line_rho"].append(0.5 * (rho[half - 1] + rho[half]))
        d["line_p"].append(0.5 * (p[half - 1] + p[half]))
        flat.append(dict(measure.flat_part(p, rho, u, v, t, fl, ext, h), t=t))
        yw, peak = measure.wall_shock(p[:half, 0], fl.p0, h, ext["wall_start"])
        wall.append(dict(t=t, y=yw, d=fl.Hs - yw, peak=peak))
        if full and k == n_snap // 2 - 1:                           # t_end / 2: the part that the coarser grid covers at t_end
            d["rho_half_w"] = rho[N // 4:3 * N // 4, :half].copy()
    d = {k: np.array(v) for k, v in d.items()}
    if full:
        d.update(map_rho=rho.astype(np.float32), map_p=p.astype(np.float32), map_u=u.astype(np.float32), map_v=v.astype(np.float32),
                 rho_end_c=measure.restrict(rho))
        if N <= 400:
            d["rho_end"] = rho
    # the far walls at the last written time: has any wave arrived there?
    far = dict(floor=float(np.max(np.abs(p[0] / fl.p0 - 1))), right=float(np.max(np.abs(p[:, -1] / fl.p0 - 1))),
               first_cell_rho=float(rho[0, -1]))
    return d, dict(flat=flat, wall=wall, lows=lows, far=far, snapshots=len(tds))


def health(log, nums, tot, fl, end_time=None):
    """monitor.assess on the log, plus the checks that belong to this flow."""
    a = monitor.assess(log, end_time=end_time, courant_limit=0.5)
    lows = nums["lows"]
    a["checks"].append(dict(
        name="Density, pressure and temperature stay positive", status="ok" if min(lows.values()) > 0 else "fail",
        value=rf"lowest density \({tex_num(lows['rho'], 4)}\,\mathrm{{kg/m^3}}\), pressure \({lows['p'] / 1e3:.1f}\,\mathrm{{kPa}}\), "
              rf"temperature \({lows['T']:.1f}\,\mathrm{{K}}\) at the written times",
        meaning="The gas expands strongly round the corner. A scheme that overshoots there produces a negative "
                "density or pressure, which is not a gas."))
    a["balance"] = balance(tot, fl)
    a["status"] = "fail" if any(c["status"] == "fail" for c in a["checks"]) else \
                  "warn" if any(c["status"] == "warn" for c in a["checks"]) else "ok"
    return a


def save(name, d, tot, meta):
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{name}.npz", c_t=tot["t"], c_mass=tot["mass"], c_energy=tot["energy"], c_inflow=tot["inflow"], **d)
    (RUNS / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))


def summary_line(name, meta):
    f, w, h = meta["flat"][-1], meta["wall"][-1], meta["health"]
    b = h["balance"]
    s = (f"{name}: {meta['steps']} steps on {meta['cores']} core(s), solver {meta['wall_time']:.1f} s, health {h['status']}; "
         f"balance vs inflow: mass {b['mass_vs_inflow']:.1e}, energy {b['energy_vs_inflow']:.1e}; lows {meta['lows']}; "
         f"far walls {meta['far']['floor']:.1e} / {meta['far']['right']:.1e}; wall shock d = {w['d']:.4f} m (level behind {w['peak']:.4g} Pa)")
    if f.get("ok"):
        s += (f"; flat part: {f['rows']} rows, x = {f['x_mean']:.5f} m, deviation {f['dev_mean_cells']:.3f} cells "
              f"(from a line {f['dev_line_cells']:.3f}, tilt {f['tilt_cells']:.3f})")
        if "p" in f:
            s += (f"; behind it ({f['region_rows']} x {f['region_cols']} cells): p {f['p']['err']:+.2e} ripple {f['p']['ripple']:.2e}, "
                  f"rho {f['rho']['err']:+.2e} ripple {f['rho']['ripple']:.2e}, u {f['u']['err']:+.2e} ripple {f['u']['ripple']:.2e}, "
                  f"|v|/u2 {f['v_max']:.1e}, p along the shock {f.get('p_row_spread', float('nan')):.1e}")
    return s


def do_one(spec, fl, numerics):
    name, solver = spec["name"], numerics["solver"]
    t0 = time.time()
    case, ranks = make(spec, fl, numerics)
    try:
        wall = foam.run(solver_cmd(solver, ranks), case, log_name=solver, timeout=3600)
    except RuntimeError:
        return crashed(case, spec, fl, numerics, ranks, time.time() - t0)
    if ranks > 1:
        foam.run("reconstructPar -fields '(p rho U)'", case, "reconstructPar")
    return finish(case, spec, fl, numerics, ranks, wall, time.time() - t0)


def crashed(case, spec, fl, numerics, ranks, total):
    """A run that the solver ended with an error is a result too: record when it stopped and how."""
    solver = numerics["solver"]
    log = monitor.parse_log(case / f"log.{solver}")
    co = [s["co"] for s in log["steps"] if s["co"] is not None]
    steps = steps_of(spec, numerics)
    tot = totals(case, fl, numerics)
    meta = dict(spec, crashed=True, steps=steps, dt=fl.t_end / steps, h=fl.L / spec["N"], cores=ranks, case_dir=str(case),
                steps_done=len(log["steps"]), t_stop=float(tot["t"][-1]), co_max=float(max(co)), co_last=float(co[-1]),
                fpe=bool(log["fpe"]), total_time=total, balance=balance(tot, fl), snapshots=len([t for t in foam.time_dirs(case) if float(t.name) > 0]))
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{spec['name']}.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"{spec['name']}: STOPPED BY THE SOLVER at step {meta['steps_done']} of {steps} (t = {meta['t_stop']:.4e} s), floating-point "
          f"exception: {meta['fpe']}; the solver's Courant number reached {meta['co_max']:.3f}", flush=True)
    return meta


def finish(case, spec, fl, numerics, ranks, wall, total):
    name, solver = spec["name"], numerics["solver"]
    d, nums = collect(case, spec, fl, numerics)
    tot = totals(case, fl, numerics)
    steps = steps_of(spec, numerics)
    meta = dict(spec, steps=steps, dt=fl.t_end / steps, h=fl.L / spec["N"], cores=ranks, case_dir=str(case), wall_time=wall,
                total_time=total, steps_logged=len(tot["t"]), mesh_ok="Mesh OK." in (case / "log.checkMesh").read_text(),
                health=health(case / f"log.{solver}", nums, tot, fl), perf=perf.summarize(case, solver, cells=spec["N"] ** 2, cores=ranks),
                openfoam=foam.foam_env("WM_PROJECT_VERSION"), python=platform.python_version(), run_root=str(run_root()), **nums)
    save(name, d, tot, meta)
    print(summary_line(name, meta), flush=True)
    return meta


def do_recollect(names):
    problem, numerics = load()
    fl = theory.Flow(problem["given"])
    sp_all = specs(numerics)
    for n in names or [n for n in sp_all if (RUNS / f"{n}.json").exists()]:
        old = json.loads((RUNS / f"{n}.json").read_text())
        finish(Path(old["case_dir"]), sp_all[n], fl, numerics, old["cores"], old["wall_time"], old["total_time"])


def do_preflight(part, args):
    """part: short (coarsest grid, every check), timing [ranks ...] (finest grid, short), full [names ...]
    (whole runs that are cheap and size the rules), or all. Each part updates its own entry of preflight.json."""
    problem, numerics = load()
    fl = theory.Flow(problem["given"])
    solver, sp_all, fam, pfn = numerics["solver"], specs(numerics), numerics["runs"]["family"], numerics["preflight"]
    pj = OUT / "preflight.json"
    out = json.loads(pj.read_text()) if pj.exists() else {}
    out.update(mol_weight=foam.mol_weight(fl.R), gas_constant=foam.gas_constant())
    out.setdefault("short", [])
    out.setdefault("timing", [])
    if part in ("short", "all"):
        out["short"] = []
        _preflight_short(out, fl, numerics, sp_all, fam, pfn, solver)
        pj.write_text(json.dumps(out, indent=1, default=str))
    if part in ("timing", "all"):
        tested = [int(a) for a in args] if part == "timing" and args else pfn["ranks_tested"]
        out["timing"] = [r for r in out["timing"] if r["ranks"] not in tested]
        _preflight_timing(out, fl, numerics, sp_all, fam, pfn, solver, tested, pj)
    if part in ("full", "all"):
        ts_names = sorted((n for n, s in sp_all.items() if "time step" in s["studies"]), key=lambda n: sp_all[n]["spc"])
        names = args if part == "full" and args else list(dict.fromkeys([f"N{fam[0]}", f"N{fam[1]}"] + ts_names))
        for n in names:
            do_one(sp_all[n], fl, numerics)
        for N in fam:
            lo, hi = measure.rule_bias(fl, numerics["extraction"], N)
            print(f"shock-position rule on the exact cell averages, N = {N}: bias {lo:+.3f} to {hi:+.3f} cells")
    if any(r["status"] == "fail" for r in out["short"] + out["timing"]):
        raise SystemExit("Pre-flight FAILED: fix the setup before production runs.")


def _preflight_short(out, fl, numerics, sp_all, fam, pfn, solver):
    # ---- coarsest grid, short: every output file, the start state, the cell order, the balances ----
    spec = sp_all[f"N{fam[0]}"]
    N, h = spec["N"], fl.L / spec["N"]
    case, _ = make(spec, fl, numerics, prefix="pf_")
    pf = monitor.preflight(case, solver, end_time_short=pfn["fraction"] * fl.t_end, end_time_full=fl.t_end, timeout=600,
                           courant_limit=0.5)
    pfc = Path(pf["case"])
    foam.run("postProcess -func writeCellCentres -latestTime", pfc, "writeCellCentres")
    td = [t for t in foam.time_dirs(pfc) if float(t.name) > 0][-1]
    C = foam.read_internal_field(td / "C")
    xc, yc = np.meshgrid(measure.centres(N, fl.L), measure.centres(N, fl.H))
    order = float(max(np.max(np.abs(C[:, 0].reshape(N, N) - xc)), np.max(np.abs(C[:, 1].reshape(N, N) - yc))))
    d, nums = collect(pfc, spec, fl, numerics)
    tot = totals(pfc, fl, numerics)
    hl = health(pfc / f"log.{solver}", nums, tot, fl, end_time=fl.t_end)
    b = hl["balance"]
    n_short = round(pfn["fraction"] * steps_of(spec, numerics))
    f = nums["flat"][-1]
    dt = fl.t_end / steps_of(spec, numerics)
    co_x = fl.s_max * dt / h                                         # the usual Courant number along x behind the shock
    co_solver_expected = 0.5 * (2 * fl.u2 + fl.c2) * dt / h          # what rhoCentralFoam should print for that state
    co_printed = float(next(c["value"].split()[0] for c in hl["checks"] if c["name"] == "Max Courant number"))
    checks = hl["checks"] + [
        dict(name="Cells are in rows and columns as the analysis assumes", status="ok" if order < 1e-12 else "fail",
             value=rf"largest difference \({tex_num(order, 2)}\,\mathrm{{m}}\)",
             meaning="The analysis places cell number i + N j at ((i + 1/2) h, (j + 1/2) h). OpenFOAM's own cell centres must agree."),
        dict(name="Start state as given", status="ok" if abs(nums["far"]["first_cell_rho"] / fl.rho0 - 1) < 1e-12 else "fail",
             value=rf"density in an undisturbed cell: \({nums['far']['first_cell_rho']:.13g}\,\mathrm{{kg/m^3}}\)",
             meaning="The solver forms the density from pressure, temperature and its own gas constant. A cell that no wave has "
                     "reached must hold the given density."),
        dict(name="Totals recorded after every step", status="ok" if len(tot["t"]) == n_short and tot["mass"][0] > 0 else "fail",
             value=f"{len(tot['t'])} rows for {n_short} steps", meaning="Mass and energy of the chamber and the mass flow through "
             "the opening are needed after every step for the conservation check."),
        dict(name="The opening delivers the exact mass flow", status="ok" if b["inflow_rate"] < 1e-12 else "fail",
             value=rf"largest relative difference from \(\rho_2 u_2 (H - H_s)\): \({tex_num(b['inflow_rate'], 2)}\)",
             meaning="The air enters faster than sound, so the face flux at the opening must be the flux of the state behind the shock, from the first step on."),
        dict(name="Mass and energy equal the start plus the exact inflow, after every step",
             status="ok" if max(b["mass_vs_inflow"], b["energy_vs_inflow"]) < 1e-9 else "fail",
             value=rf"mass \({tex_num(b['mass_vs_inflow'], 2)}\), energy \({tex_num(b['energy_vs_inflow'], 2)}\) of the inflow so far",
             meaning="The physical check of every step: the walls let nothing through, so the chamber's mass and energy may change only by what the duct brings in."),
        dict(name="The solver's own Courant number, against the usual one", status="ok" if co_printed < 0.5 else "warn",
             value=rf"largest printed \({co_printed:.3f}\); in the uniform state behind the shock it is \({co_solver_expected:.3f}\); the usual "
                   rf"Courant number along \(x\) there is \({co_x:.3f}\)",
             meaning="rhoCentralFoam prints its own measure: half the sum over a cell's faces of the larger one-sided flux speed. "
                     "In two dimensions it adds both directions, so it can be above the usual number along one direction; the "
                     "largest value is found where the gas turns round the corner."),
    ]
    status = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "ok"
    pf.update(checks=checks, status=status, N=N, name=spec["name"], n_steps=n_short, headers=tot["head"],
              first_rows=[[float(tot[k][i]) for k in ("t", "mass", "energy", "inflow")] for i in range(3)], balance=b, flat=f)
    out["short"].append(pf)
    print(f"preflight {spec['name']}: {status}; {pf['wall_time']} s for {n_short} steps; projected {pf['est_full_runtime']} s")
    for c in checks:
        print(f"   {c['status']:4s} {c['name']}: {c['value']}")
    print("   output files:", tot["head"], "| first rows:", pf["first_rows"])
    dec = yaml.safe_load((HERE / "decisions.yaml").read_text(encoding="utf-8")) if (HERE / "decisions.yaml").exists() else []
    out["decisions"] = contract.check_decisions(dec) + contract.uncovered_settings(dec, pfc)
    print("   decisions:", out["decisions"] or "cover every setting")


def _preflight_timing(out, fl, numerics, sp_all, fam, pfn, solver, tested, pj):
    # ---- finest grid, short, on several numbers of cores: health and the cost of the full run ----
    spec = sp_all[f"N{fam[-1]}"]
    n_full = steps_of(spec, numerics)
    for ranks in tested:
        case, _ = make(spec, fl, numerics, prefix=f"pf_r{ranks}_", ranks=ranks)
        cmd = solver_cmd(solver, ranks)
        pf = monitor.preflight(case, cmd, end_time_short=pfn["fraction_finest"] * fl.t_end, end_time_full=fl.t_end,
                               timeout=900, courant_limit=0.5)
        rec = json.loads((Path(pf["case"]) / f"perf.{cmd.split()[0]}.json").read_text())
        n_short = round(pfn["fraction_finest"] * n_full)
        # seconds per step from the solver's own clock over the second half of the short run (start-up left out)
        log = monitor.parse_log(Path(pf["case"]) / f"log.{cmd.split()[0]}")
        ex = [s["exec"] for s in log["steps"] if s["exec"] is not None]
        per_step = (ex[-1] - ex[len(ex) // 2]) / (len(ex) - 1 - len(ex) // 2)
        row = dict(ranks=ranks, status=pf["status"], steps=pf["steps"], wall_short=rec["wall_s"], n_short=n_short,
                   s_per_step=per_step, projected=per_step * n_full, us_per_cell_step=1e6 * per_step / spec["N"] ** 2,
                   checks=pf["checks"], N=spec["N"])
        out["timing"].append(row)
        out["timing"].sort(key=lambda r: r["ranks"])
        print(f"preflight {spec['name']} on {ranks} core(s): {pf['status']}; {rec['wall_s']:.1f} s for {pf['steps']} steps; "
              f"{per_step:.3f} s per step after start-up; full run projected {row['projected']:.0f} s; "
              f"{row['us_per_cell_step']:.3f} microseconds per cell and step", flush=True)
        pj.write_text(json.dumps(out, indent=1, default=str))


def do_production(names):
    problem, numerics = load()
    fl = theory.Flow(problem["given"])
    sp_all = specs(numerics)
    todo = names or [n for n in sp_all if not (RUNS / f"{n}.json").exists()]
    for n in todo:                                  # one at a time: wall times are then comparable
        do_one(sp_all[n], fl, numerics)
    (OUT / "production.done").write_text(json.dumps(dict(runs=todo, finished=time.strftime("%Y-%m-%d %H:%M:%S"))))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    if mode == "preflight":
        do_preflight(sys.argv[2] if len(sys.argv) > 2 else "all", sys.argv[3:])
    elif mode == "production":
        do_production(sys.argv[2:])
    elif mode == "recollect":
        do_recollect(sys.argv[2:])
    elif mode == "list":
        _, num = load()
        for n, s in specs(num).items():
            print(f"{n:28s} {s['N']:5d} cells per side, {steps_of(s, num):6d} steps, {s['flux']}/{s['limiter']}, studies: {', '.join(s['studies'])}")
    else:
        raise SystemExit(f"unknown mode {mode}")
