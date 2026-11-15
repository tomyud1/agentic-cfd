"""Build and run the shock tube for every run in numerics.yaml; save the raw outputs.

Usage (repository root, inside WSL):
  python -m cases.shock_tube_sod.run preflight              # gas constant, short tests on the coarsest and finest
                                                            # grid, the time-step study and the coarsest full run
  python -m cases.shock_tube_sod.run production [NAME ...]  # every other run (default: all not yet made), one at a time
  python -m cases.shock_tube_sod.run list                   # the runs and their time steps
Outputs: results/shock_tube_sod/gas_constant.json, preflight.json, runs/<name>.json + runs/<name>.npz
Run names: N<cells>, plus _spc<steps per cell> or _<flux>_<limiter> when a run differs from the main choice.
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
MOL_WEIGHT_GUESS = 28.97      # only the starting value of the measurement in gas_constant()


def load():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    return problem, numerics


def run_root():
    return Path(foam.foam_env("FOAM_RUN")) / "agentic" / HERE.name


def specs(numerics):
    """Every run as {name: dict(name, N, spc, flux, limiter, studies)}, in the order: grid family,
    time-step study, scheme variants. A run that belongs to several studies appears once."""
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


def params_for(spec, problem, numerics, mol_weight, steps=None, write_every=None, end_steps=None):
    g = {k: v["value"] for k, v in problem["given"].items()}
    n = steps or steps_of(spec, numerics)
    dt = g["t_end"] / n
    limiter = spec["limiter"]
    return dict(N=spec["N"], L=g["L"], x0=g["L"] / 2, side=numerics["side"], half=numerics["side"] / 2,
                steps=n, dt=repr(dt), end_time=repr(dt * end_steps) if end_steps else repr(g["t_end"]),
                write_every=write_every or n // numerics["snapshots"],
                flux=spec["flux"], limiter=limiter, limiter_vector=limiter if limiter == "upwind" else limiter + "V",
                p_L=repr(g["p_L"]), p_R=repr(g["p_R"]), T_L=repr(g["p_L"] / (g["rho_L"] * g["R"])),
                T_R=repr(g["p_R"] / (g["rho_R"] * g["R"])), mol_weight=repr(mol_weight),
                Cp=repr(g["gamma"] * g["R"] / (g["gamma"] - 1)))


def make(spec, problem, numerics, mol_weight, prefix="", **kw):
    """Render the case, make the grid and set the start state. Pre-flight uses its own directories."""
    case = foam.render_case(HERE / "template", run_root() / f"{prefix}{spec['name']}",
                            params_for(spec, problem, numerics, mol_weight, **kw))
    foam.run("blockMesh", case)
    foam.run("checkMesh", case)
    foam.run("setFields", case)
    return case


def gas_constant(problem, numerics, redo=False):
    """OpenFOAM's universal gas constant, measured: a 10-cell case is advanced by one step with a guessed
    molar mass W, and the first cell (still undisturbed) gives RR = p W / (rho T). The molar mass that
    makes OpenFOAM's gas constant RR / W equal to the given R follows. The value is built into the
    program from two rounded constants, so it is measured here, not typed from memory."""
    f = OUT / "gas_constant.json"
    if f.exists() and not redo:
        return json.loads(f.read_text())
    g = {k: v["value"] for k, v in problem["given"].items()}
    spec = dict(name="gas_constant", N=10, spc=2.0, flux=numerics["main"]["flux"], limiter=numerics["main"]["limiter"])
    case = make(spec, problem, numerics, MOL_WEIGHT_GUESS, steps=20, write_every=1, end_steps=1)
    foam.run(numerics["solver"], case, timeout=120)
    td = [t for t in foam.time_dirs(case) if float(t.name) > 0][-1]
    rho, p, T = (foam.read_internal_field(td / k)[0] for k in ("rho", "p", "T"))
    RR = p * MOL_WEIGHT_GUESS / (rho * T)
    out = dict(RR=float(RR), mol_weight=float(RR / g["R"]), R=g["R"], measured_from=dict(rho=float(rho), p=float(p), T=float(T)),
               openfoam=foam.foam_env("WM_PROJECT_VERSION"))
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1))
    return out


def conserved_history(case):
    """Time and the volume integrals of rho, rho u (x component) and rho E after every step."""
    hits = sorted((Path(case) / "postProcessing" / "conserved").rglob("volFieldValue.dat"))
    rows = foam.read_dat(hits[0])
    head = [l for l in hits[0].read_text().splitlines() if l.startswith("#")][-1]
    return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2][0] for r in rows]),
            np.array([r[3] for r in rows]), head)


def collect(case, spec, problem, numerics):
    """Read the written fields and the per-step totals of a finished run into arrays."""
    g = {k: v["value"] for k, v in problem["given"].items()}
    N = spec["N"]
    tds = [t for t in foam.time_dirs(case) if float(t.name) > 0]
    d = dict(x=(np.arange(N) + 0.5) * g["L"] / N, times=np.array([float(t.name) for t in tds]),
             rho=np.array([foam.read_internal_field(t / "rho") for t in tds]),
             u=np.array([foam.read_internal_field(t / "U")[:, 0] for t in tds]),
             p=np.array([foam.read_internal_field(t / "p") for t in tds]),
             T=np.array([foam.read_internal_field(t / "T") for t in tds]))
    t_printed, d["c_mass"], d["c_mom"], d["c_energy"], head = conserved_history(case)
    # The file prints the time with 10 digits, too few for a check at round-off level: use step number
    # times the time step the case really ran with (read from its controlDict, because the pre-flight
    # copy is rewritten by foamDictionary with 6 digits and so runs with a slightly different step).
    dt = float(foamdict.parse(Path(case) / "system/controlDict")["deltaT"])
    d["c_t"] = dt * (np.arange(len(t_printed)) + 1)
    assert np.allclose(d["c_t"], t_printed, rtol=1e-8, atol=0), "totals file: rows are not one per time step"
    return d, head


def health(log, d, spec, problem, numerics, th_ex, end_time=None):
    """monitor.assess on the log, plus the checks that belong to this flow: the gas state stays
    physical, the ends are still undisturbed, and mass and energy are conserved step by step."""
    g = {k: v["value"] for k, v in problem["given"].items()}
    _, ex = th_ex
    a = monitor.assess(log, end_time=end_time, courant_limit=1.0)
    rho_min, p_min = float(d["rho"].min()), float(d["p"].min())
    a["checks"].append(dict(
        name="Density and pressure stay positive", status="ok" if min(rho_min, p_min) > 0 else "fail",
        value=rf"lowest density \({rho_min:.4g}\), lowest pressure \({p_min:.5g}\)",
        meaning="A negative density or pressure is not a gas; a scheme that overshoots at a jump can produce one."))
    ends = max(abs(d["p"][-1, 0] / g["p_L"] - 1), abs(d["p"][-1, -1] / g["p_R"] - 1),
               abs(d["u"][-1, 0]) / ex.u_star, abs(d["u"][-1, -1]) / ex.u_star)
    M0, E0 = (g["rho_L"] + g["rho_R"]) * g["L"] / 2, (g["p_L"] + g["p_R"]) * g["L"] / (2 * (g["gamma"] - 1))
    a.update(rho_min=rho_min, p_min=p_min, ends=float(ends),
             # start: the first recorded total against the exact one (how well the grid holds the start state);
             # drift: the largest change from that first value during the run (what the time stepping loses)
             mass_start=float(abs(d["c_mass"][0] / M0 - 1)), energy_start=float(abs(d["c_energy"][0] / E0 - 1)),
             mass_drift=float(np.max(np.abs(d["c_mass"] - d["c_mass"][0])) / M0),
             energy_drift=float(np.max(np.abs(d["c_energy"] - d["c_energy"][0])) / E0),
             momentum_err=float(np.max(np.abs(d["c_mom"] - (g["p_L"] - g["p_R"]) * d["c_t"])) / ((g["p_L"] - g["p_R"]) * g["t_end"])))
    a["status"] = "fail" if any(c["status"] == "fail" for c in a["checks"]) else \
                  "warn" if any(c["status"] == "warn" for c in a["checks"]) else "ok"
    return a


def errors(d, k, ex, g):
    """L1 errors of density, velocity and pressure of snapshot k against the exact cell averages."""
    N = len(d["x"])
    r, u, p = ex.cell_averages(np.linspace(0, g["L"], N + 1), float(d["times"][k]))
    # The exact density and pressure only fall from left to right. Any extra up-and-down in the
    # simulation (total variation above the exact one) is a wiggle the scheme made.
    return dict(rho=measure.l1(d["rho"][k], r, g["rho_L"]), u=measure.l1(d["u"][k], u, ex.u_star),
                p=measure.l1(d["p"][k], p, g["p_L"]),
                wiggle_rho=measure.total_variation(d["rho"][k]) / (g["rho_L"] - g["rho_R"]) - 1,
                wiggle_p=measure.total_variation(d["p"][k]) / (g["p_L"] - g["p_R"]) - 1)


def do_one(spec, problem, numerics, th_ex, mw):
    name = spec["name"]
    g = {k: v["value"] for k, v in problem["given"].items()}
    t0 = time.time()
    case = make(spec, problem, numerics, mw["mol_weight"])
    wall = foam.run(numerics["solver"], case, timeout=1800)
    d, head = collect(case, spec, problem, numerics)
    log = case / f"log.{numerics['solver']}"
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{name}.npz", **d)
    steps = steps_of(spec, numerics)
    meta = dict(spec, steps=steps, dt=g["t_end"] / steps, h=g["L"] / spec["N"], case_dir=str(case), wall_time=wall,
                total_time=time.time() - t0, steps_logged=len(d["c_t"]),
                mesh_ok="Mesh OK." in (case / "log.checkMesh").read_text(),
                health=health(log, d, spec, problem, numerics, th_ex), errors=errors(d, -1, th_ex[1], g),
                perf=perf.summarize(case, numerics["solver"], cells=spec["N"], cores=1),
                openfoam=foam.foam_env("WM_PROJECT_VERSION"), python=platform.python_version(), run_root=str(run_root()))
    (RUNS / f"{name}.json").write_text(json.dumps(meta, indent=1))
    e, h = meta["errors"], meta["health"]
    print(f"{name}: {steps} steps, solver {wall:.1f} s, health {h['status']}; L1 errors rho {e['rho']:.3e}, u {e['u']:.3e}, "
          f"p {e['p']:.3e}; wiggle rho {e['wiggle_rho']:.1e}, p {e['wiggle_p']:.1e}; mass drift {h['mass_drift']:.1e} "
          f"(start {h['mass_start']:.1e}), energy drift {h['energy_drift']:.1e} (start {h['energy_start']:.1e}), "
          f"momentum {h['momentum_err']:.1e}, ends {h['ends']:.1e}", flush=True)
    return meta


def do_preflight():
    problem, numerics = load()
    g = {k: v["value"] for k, v in problem["given"].items()}
    th_ex = theory.evaluate(problem["given"])
    ex = th_ex[1]
    sp_all = specs(numerics)
    mw = gas_constant(problem, numerics, redo=True)
    print(f"gas constant: OpenFOAM's RR = {mw['RR']:.12g} J/(kmol K) -> molWeight {mw['mol_weight']:.12g} for R = {g['R']}")
    fam = numerics["runs"]["family"]
    frac = numerics["preflight"]["fraction"]
    out = dict(gas_constant=mw, short=[], decisions=None)
    for N in (fam[0], fam[-1]):
        spec = sp_all[f"N{N}"]
        case = make(spec, problem, numerics, mw["mol_weight"], prefix="pf_")
        pf = monitor.preflight(case, numerics["solver"], end_time_short=repr(frac * g["t_end"]), end_time_full=g["t_end"],
                               timeout=600)
        pfc = Path(pf["case"])
        foam.run("postProcess -func writeCellCentres -latestTime", pfc, "writeCellCentres")
        d, head = collect(pfc, spec, problem, numerics)
        h = health(pfc / f"log.{numerics['solver']}", d, spec, problem, numerics, th_ex, end_time=g["t_end"])
        td = [t for t in foam.time_dirs(pfc) if float(t.name) > 0][-1]
        Cx = foam.read_internal_field(td / "C")[:, 0]
        n_short = round(frac * steps_of(spec, numerics))
        checks = h["checks"] + [
            dict(name="Cells are in order along the tube", status="ok" if np.allclose(Cx, d["x"], rtol=0, atol=1e-12) else "fail",
                 value=rf"largest difference \({tex_num(float(np.max(np.abs(Cx - d['x']))), 2)}\,\mathrm{{m}}\)",
                 meaning="The analysis places cell number i at (i + 1/2) h. OpenFOAM's own cell centres must agree."),
            dict(name="Start state as given", status="ok" if (abs(d["rho"][0, 0] / g["rho_L"] - 1) < 1e-12
                                                               and abs(d["rho"][0, -1] / g["rho_R"] - 1) < 1e-12) else "fail",
                 value=rf"density in the first and last cell: \({d['rho'][0, 0]:.13g}\) and \({d['rho'][0, -1]:.13g}\)",
                 meaning="The solver forms the density from pressure, temperature and its own gas constant. The end cells "
                         "are still undisturbed, so they must hold the given densities."),
            dict(name="Totals recorded after every step", status="ok" if len(d["c_t"]) == n_short and d["c_mass"][0] > 0 else "fail",
                 value=f"{len(d['c_t'])} rows for {n_short} steps",
                 meaning="Mass, momentum and energy of the whole tube are needed after every step for the conservation check."),
            dict(name="Mass and energy unchanged, momentum equals the push of the end walls",
                 status="ok" if max(h["mass_drift"], h["energy_drift"], h["momentum_err"]) < 1e-9 else "fail",
                 value=rf"mass \({tex_num(h['mass_drift'], 2)}\), energy \({tex_num(h['energy_drift'], 2)}\), "
                       rf"momentum \({tex_num(h['momentum_err'], 2)}\) (relative)",
                 meaning="The physical check of every step. Closed ends let no mass or energy through, and the only force "
                         "on the air is the pressure on the two end walls."),
        ]
        for w in foam.ignored_settings(pfc):
            checks.append(dict(name="Settings the solver ignores", status="warn", value="ignored", meaning=w))
        status = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "ok"
        pf.update(checks=checks, status=status, N=N, name=spec["name"], n_steps=n_short, header=head,
                  first_rows=[[float(d[k][i]) for k in ("c_t", "c_mass", "c_mom", "c_energy")] for i in range(min(3, len(d["c_t"])))],
                  mass_drift=h["mass_drift"], energy_drift=h["energy_drift"], momentum_err=h["momentum_err"], ends=h["ends"],
                  mass_start=h["mass_start"], energy_start=h["energy_start"])
        out["short"].append(pf)
        print(f"preflight {spec['name']}: {status}; {pf['wall_time']} s for {n_short} steps; projected {pf['est_full_runtime']} s")
        for c in checks:
            print(f"   {c['status']:4s} {c['name']}: {c['value']}")
        print("   totals file:", head, "| first rows:", pf["first_rows"])
        if N == fam[-1]:
            dec = yaml.safe_load((HERE / "decisions.yaml").read_text(encoding="utf-8")) if (HERE / "decisions.yaml").exists() else []
            out["decisions"] = contract.check_decisions(dec) + contract.uncovered_settings(dec, pfc)
            print("   decisions:", out["decisions"] or "cover every setting")

    # Time-step study on one grid, and the coarsest grid in full: cheap, and they size the acceptance rules.
    ts = numerics["runs"]["timestep"]
    ts_names = sorted((n for n, s in sp_all.items() if "time step" in s["studies"]), key=lambda n: sp_all[n]["spc"])
    names = ts_names + [f"N{fam[0]}"]
    metas = {n: do_one(sp_all[n], problem, numerics, th_ex, mw) for n in dict.fromkeys(names)}
    study = []
    prev = None
    for n in ts_names:                                # largest time step first
        d = np.load(RUNS / f"{n}.npz")
        row = dict(name=n, spc=sp_all[n]["spc"], courant=ex.s_max * metas[n]["dt"] / metas[n]["h"], errors=metas[n]["errors"])
        if prev is not None:
            row["change"] = {k: measure.l1(d[k][-1], prev[k][-1], s) for k, s in (("rho", g["rho_L"]), ("u", ex.u_star), ("p", g["p_L"]))}
        study.append(row)
        prev = d
    out["timestep"] = study
    for r in study:
        e = r["errors"]
        print(f"time step {r['name']}: Courant {r['courant']:.3f}, L1 rho {e['rho']:.4e}, u {e['u']:.4e}, p {e['p']:.4e}; "
              f"wiggle rho {e['wiggle_rho']:.2e}, p {e['wiggle_p']:.2e}; change from the larger step: {r.get('change')}")
    # The coarsest grid and the time-step grid, read with the extraction rules: these numbers size the acceptance rules.
    ext, w = numerics["extraction"], ex.positions(g["t_end"])
    out["sizing"] = {}
    for n in (f"N{fam[0]}", f"N{ts['cells']}"):
        d = np.load(RUNS / f"{n}.npz")
        N = len(d["x"])
        sim = measure.read_all(d["x"], d["rho"][-1], d["u"][-1], d["p"][-1], w, g["L"], g["p_R"], ext)
        bias = measure.read_all(d["x"], *ex.cell_averages(np.linspace(0, g["L"], N + 1), g["t_end"]), w, g["L"], g["p_R"], ext)
        out["sizing"][n] = dict(errors=metas[n]["errors"], sim=sim, exact_cells=bias)
        print(f"{n}: errors {metas[n]['errors']}")
        for key, exact in (("p", ex.p_star), ("u", ex.u_star), ("rho_L", ex.rho_starL), ("rho_R", ex.rho_starR)):
            s, b = sim["states"][key], bias["states"][key]
            print(f"   state {key}: sim {s['mean']:.8g} (spread {s['spread']:.2e}, {s['cells']} cells), rel. error "
                  f"{s['mean'] / exact - 1:+.2e}; the rule on the exact cells: {b['mean'] / exact - 1:+.2e}")
        for key in ("head", "tail", "contact", "shock"):
            print(f"   position {key}: sim {sim['positions'][key]:.6f}, exact {w[key]:.6f}, error "
                  f"{(sim['positions'][key] - w[key]) * N / g['L']:+.3f} cells; the rule on the exact cells: "
                  f"{(bias['positions'][key] - w[key]) * N / g['L']:+.3f} cells")
        print(f"   widths in cells: sim {sim['widths']}, exact cells {bias['widths']}")
    (OUT / "preflight.json").write_text(json.dumps(out, indent=1, default=str))
    if any(p["status"] == "fail" for p in out["short"]):
        raise SystemExit("Pre-flight FAILED: fix the setup before production runs.")


def do_production(names):
    problem, numerics = load()
    th_ex = theory.evaluate(problem["given"])
    mw = gas_constant(problem, numerics)
    sp_all = specs(numerics)
    todo = names or [n for n in sp_all if not (RUNS / f"{n}.json").exists()]
    for n in todo:                                  # one at a time: wall times are then comparable
        do_one(sp_all[n], problem, numerics, th_ex, mw)
    (OUT / "production.done").write_text(json.dumps(dict(runs=todo, finished=time.strftime("%Y-%m-%d %H:%M:%S"))))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    if mode == "preflight":
        do_preflight()
    elif mode == "production":
        do_production(sys.argv[2:])
    elif mode == "list":
        _, num = load()
        for n, s in specs(num).items():
            print(f"{n:28s} {s['N']:5d} cells, {steps_of(s, num):6d} steps, {s['flux']}/{s['limiter']}, studies: {', '.join(s['studies'])}")
    else:
        raise SystemExit(f"unknown mode {mode}")
