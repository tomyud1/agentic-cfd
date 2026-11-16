"""Build and run the tube for every run in numerics.yaml; save what the analysis needs.

Usage (repository root, inside WSL; or through py.sh):
  python -m cases.detonation_tube_h2_air.run preflight              # short tests on the coarsest and finest grid, then the pre-flight's full runs
  python -m cases.detonation_tube_h2_air.run production [NAME ...]  # every other run (default: all not yet made), one at a time
  python -m cases.detonation_tube_h2_air.run recollect [NAME ...]   # read finished runs again (after a change of a reading rule)
  python -m cases.detonation_tube_h2_air.run list                   # the runs
Outputs: results/detonation_tube_h2_air/preflight.json, runs/<name>.json + runs/<name>.npz
Run names: N<cells> = the main settings on that grid; a suffix names what differs: _s<steps> time step, _p<passes>,
_serial (one core, with profiling, the first part of the time only), _nochem (chemistry switched off: the cost of the flow alone),
_seulex / _rel<tolerance> / _allcells (chemistry solver, its tolerance, no cut-off temperature).
"""
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cfdlib import chem, contract, foam, monitor, perf
from cfdlib.report import tex_num
from . import measure, theory

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"
SOLVER = "reactingFoam"
SCHEME_DIR = "$FOAM_TUTORIALS/combustion/chemFoam/h2/chemkin"      # read where it is shipped; OpenFOAM expands the variable
PROFILE_TIMES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)   # of t_end: states along the tube kept for the figures
KEEP = ("p", "T", "rho", "U", "Qdot", "H2", "H2O", "OH")
MASS_TOL = 1.0e-6               # health: the mass in the closed tube and the atoms of each element may change by this fraction at most


def run_root():
    return Path(foam.foam_env("FOAM_RUN")) / "agentic" / HERE.name


def specs(numerics):
    """Every run as {name: dict(...)}; stage 'preflight' or 'production'."""
    main, R, P, out = numerics["main"], numerics["runs"], numerics["preflight"], {}
    steps_of = {int(n): int(s) for n, s in R["family"]}

    def add(stage, study, N, name=None, **kw):
        s = dict(N=int(N), steps=steps_of[int(N)], passes=main["passes"], ode=main["ode"], rel=main["rel_tol"], t_react=main["t_react"],
                 cores=int(R["cores"][int(N)]), chemistry=True, profiling=False, fraction=1.0)
        s.update(kw)
        name = name or f"N{N}"
        if name in out:
            out[name]["studies"].append(study)
            return
        out[name] = dict(s, name=name, stage=stage, studies=[study])

    n, fr = P["serial"]["cells"], P["serial"]["fraction"]           # one core, the first part of the time only
    add("preflight", "cores", n, f"N{n}_serial", cores=1, profiling=True, fraction=fr)
    add("preflight", "cores", n, f"N{n}_serial_nochem", cores=1, profiling=True, chemistry=False, fraction=fr)
    add("preflight", "cores", n)
    add("preflight", "time step", P["timestep"]["cells"])
    for s in P["timestep"]["steps"]:
        add("preflight", "time step", P["timestep"]["cells"], f"N{P['timestep']['cells']}_s{s}", steps=s)
    add("preflight", "passes", P["passes"]["cells"])
    for k in P["passes"]["values"]:
        add("preflight", "passes", P["passes"]["cells"], f"N{P['passes']['cells']}_p{k}", passes=k)
    for N, _ in R["family"]:
        add("production", "grid", N)
    N = R["chemistry"]["cells"]
    add("production", "chemistry", N)
    for v in R["chemistry"]["variants"]:
        if "ode" in v:
            add("production", "chemistry", N, f"N{N}_{v['ode']}", ode=v["ode"])
        elif "rel_tol" in v:
            add("production", "chemistry", N, f"N{N}_rel{v['rel_tol']:g}", rel=v["rel_tol"])
        else:
            add("production", "chemistry", N, f"N{N}_allcells", t_react=v["t_react"])
    return out


def params_for(spec, g, numerics, th, t_stop=None):
    main = numerics["main"]
    h, t_end = main["cross_section"], g["t_end"]
    Y = th["start"]["Y"]
    return dict(p_0=repr(g["p_0"]), T_0=repr(g["T_0"]), Y_H2=repr(Y["H2"]), Y_O2=repr(Y["O2"]), Y_N2=repr(Y["N2"]),
                scheme_dir=SCHEME_DIR, chemistry="on" if spec["chemistry"] else "off", init_chem_dt=repr(main["init_chem_dt"]),
                ode_solver=spec["ode"], abs_tol=repr(main["abs_tol"]), rel_tol=repr(spec["rel"]), T_react=repr(float(spec["t_react"])),
                L=repr(g["L"]), h=repr(h), h_half=repr(h / 2), N=spec["N"], L_ign=repr(g["L_ign"]), p_ign=repr(g["p_ign"]), T_ign=repr(g["T_ign"]),
                ddt=main["ddt"], limiter=main["limiter"], lin_tol=repr(main["lin_tol"]), n_outer=spec["passes"], n_corr=main["correctors"],
                t_end=repr(t_stop or t_end), dt=repr(t_end / spec["steps"]), write_interval=repr(t_end / 2),
                profiling="true" if spec["profiling"] else "false", sample_interval=repr(t_end / main["samples"]), cores=spec["cores"],
                first_dt=repr(main["first_dt"]), max_co=repr(main["max_co"]))


def write_decomposition(case, N, cores, stripes):
    """Which rank owns each cell: the tube is cut into cores x stripes equal stretches, dealt to the ranks in turn.
    The chemistry costs time only in cells the wave has reached, so ranks that owned one end of the tube each would
    be busy one after the other. With stripes every rank holds its share of burnt gas at any time."""
    blocks = cores * stripes
    owner = (np.arange(N) * blocks // N) % cores
    head = ('FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       labelList;\n    location    "constant";\n'
            '    object      cellDecomposition;\n}\n\n')
    (Path(case) / "constant" / "cellDecomposition").write_text(head + f"{N}\n(\n" + "\n".join(str(int(o)) for o in owner) + "\n)\n")


def command(spec):
    if spec["cores"] == 1:
        return SOLVER
    hw = "--use-hwthread-cpus " if spec["cores"] > 8 else ""
    return f"mpirun {hw}-np {spec['cores']} {SOLVER} -parallel"


def make(spec, ctx, prefix="", t_stop=None):
    g, numerics, th, mech = ctx
    layer_cells = spec["N"] * g["L_ign"] / g["L"]
    if abs(layer_cells - round(layer_cells)) > 1e-9:               # insight 7: 600 cells made the layer 1.6 mm thick
        raise ValueError(f"{spec['N']} cells: the ignition layer would be {layer_cells} cells; it must be a whole number")
    case = foam.render_case(HERE / "template", run_root() / f"{prefix}{spec['name']}", params_for(spec, g, numerics, th, t_stop))
    foam.run("blockMesh", case)
    foam.run("checkMesh", case)
    foam.run("setFields", case)
    if spec["cores"] > 1:
        write_decomposition(case, spec["N"], spec["cores"], numerics["runs"]["stripes_per_rank"])
        foam.run("decomposePar", case)
    return case


def mesh_ok(case):
    return "Mesh OK." in (Path(case) / "log.checkMesh").read_text(errors="replace")


# ---- reading a finished run ---------------------------------------------------------------------------------------
def collect(case, spec, ctx, full=True, log_name=SOLVER):
    """Apply the reading rules of measure.py. Returns (arrays for the figures, numbers)."""
    g, numerics, th, mech = ctx
    ext, main = numerics["extraction"], numerics["main"]
    t_end, A = g["t_end"], numerics["main"]["cross_section"] ** 2
    cj, st = th["cj"], th["start"]
    hc = theory.chemical_enthalpy(mech)
    case = Path(case)
    line = measure.read_line(case)
    t = line["t"]
    p_mid = g["p_0"] + ext["pressure_front"] * (cj["p"] - g["p_0"])
    w_mid = ext["burning_front"] * cj["Y"]["H2O"]
    xp, xb, gap = measure.fronts(line, p_mid, w_mid)
    pk, head_pk = measure.read_peaks(next((case / "postProcessing/peaks").rglob("fieldMinMax.dat")))
    tv, vol, head_v = measure.read_table(next((case / "postProcessing/volume").rglob("volFieldValue.dat")))
    tm, mass, head_m = measure.read_table(next((case / "postProcessing/mass").rglob("volFieldValue.dat")))
    assert len(tv) == len(tm) == len(pk["p"]["t"]), "per-step files differ in length"
    tot = measure.totals(mech, hc, tv, vol, mass, "h")
    m0, E0 = st["mass_per_area"] * A, st["energy_per_area"] * A
    chem0 = sum(hc[mech.idx[s]] * st["Y"][s] for s in st["Y"]) * m0
    q = chem0 - tot["chem"]                                       # heat released since the start, J
    dE = tot["E"] - E0
    dx, dt = g["L"] / spec["N"], t_end / spec["steps"]
    num = dict(steps_done=len(tv), t_last=float(tv[-1]), dx=dx, dt=dt, courant_wave=cj["D"] * dt / dx, cells_per_induction=th["znd"]["x_ind"] / dx,
               headers=[head_pk, head_v, head_m, line["header"]], samples=len(t),
               first=dict(mass=float(tot["mass"][0] / m0 - 1), energy=float(dE[0] / (st["layer_surplus"] * A)),
                          p_max=float(pk["p"]["max"][0]), T_max=float(pk["T"]["max"][0]), dt=float(tv[0])),
               qdot_abs_max=float(max(abs(pk["Qdot"]["min"].min()), pk["Qdot"]["max"].max())), dt_max=float(np.diff(tv).max()) if len(tv) > 1 else float(tv[0]),
               mass_dev=float(np.max(np.abs(tot["mass"] / tot["mass"][0] - 1))),          # against the first record; the start is checked apart
               atoms_dev={e: float(np.max(np.abs(a / a[0] - 1))) for e, a in tot["atoms"].items()},
               atoms_start={e: float(a[0] / (st["atoms_per_kg"][e] * m0) - 1) for e, a in tot["atoms"].items()},
               sumY_dev=float(np.max(np.abs(tot["sumY"] - 1))),
               T_min=float(pk["T"]["min"].min()), T_max=float(pk["T"]["max"].max()), p_min=float(pk["p"]["min"].min()), p_max=float(pk["p"]["max"].max()),
               energy_startup=float(dE[int(np.argmin(np.abs(tv - 1e-6)))] / (st["layer_surplus"] * A)) if tv[-1] >= 1e-6 else None)
    hot = np.array([(T > spec["t_react"]).sum() for T in line["T"]])          # cells the chemistry works on, at each sample
    num["hot_cell_steps"] = float(np.trapezoid(hot, t) / dt) if len(t) > 1 else 0.0
    keep = measure.thin(len(tv), ext["store_points"])
    arr = dict(t=t, xp=xp, xb=xb, gap=gap, ts=tv[keep], pmax=pk["p"]["max"][keep], x_pmax=pk["p"]["x_max"][keep], x_qmax=pk["Qdot"]["x_max"][keep],
               dE=dE[keep], q=q[keep], mass_rel=(tot["mass"] / m0 - 1)[keep], x=line["x"])
    if full:
        t_a, t_b = (f * t_end for f in ext["speed_window"])
        sp = measure.speed(t, xp, t_a, t_b)
        D = sp["D"]
        late = tv >= t_a - 1e-12
        att = t >= ext["attached_from"] - 1e-12
        att_s = tv >= ext["attached_from"] - 1e-12
        gap_step = pk["p"]["x_max"] - pk["Qdot"]["x_max"]            # per step: highest pressure against highest heat release
        ka = int(np.argmin(np.abs(tv - t_a)))
        vn = chem.shock_frozen(mech, g["p_0"], g["T_0"], theory.moles(g), D)
        b = measure.burnt_state(mech, line, -1, xp[-1], D, ext["burnt_distances"])
        c_end = th["taylor"]["c"]
        pl = (line["x"] >= ext["end_plateau"][0] * c_end * t_end) & (line["x"] <= ext["end_plateau"][1] * c_end * t_end)
        num.update(
            speed=sp, speed_err=D / cj["D"] - 1, speed_fit_err=sp["D_fit"] / cj["D"] - 1, x_front_end=float(xp[-1]),
            gap_max=float(np.nanmax(np.abs(gap[att]))), gap_mean=float(np.nanmean(gap[att])), gap_end=float(gap[-1]),
            gap_step_max=float(np.max(np.abs(gap_step[att_s]))), gap_step_mean=float(np.mean(gap_step[att_s])),
            gap_early_max=float(np.nanmax(np.abs(gap[~att]))) if (~att).any() else None,
            peak=dict(mean=float(pk["p"]["max"][late].mean()), low=float(pk["p"]["max"][late].min()), high=float(pk["p"]["max"][late].max()),
                      vn=vn["p"], ratio=float(pk["p"]["max"][late].mean() / vn["p"]), T_vn=vn["T"]),
            burnt=b,
            end_p=float(line["p"][-1][pl].mean()), end_p_spread=float(np.ptp(line["p"][-1][pl]) / line["p"][-1][pl].mean()),
            end_T=float(line["T"][-1][pl].mean()), right_dev=float(abs(line["p"][-1][-1] / g["p_0"] - 1)), right_T=float(line["T"][-1][-1]),
            energy=dict(end=float(dE[-1] / q[-1]), rate_late=float((dE[-1] - dE[ka]) / (q[-1] - q[ka])), worst=float(np.max(np.abs(dE)) / q[-1]),
                        q_end=float(q[-1] / A), dE_end=float(dE[-1] / A), E0=float(E0 / A),
                        kinetic_end=float(tot["kinetic"][-1] / A)),
            q_vs_cj=float((q[-1] - q[ka]) / A / (st["rho0"] * (sp["x_b"] - sp["x_a"])) / cj["q"] - 1),
        )
        ks = [int(np.argmin(np.abs(t - f * t_end))) for f in PROFILE_TIMES]
        arr.update(prof_t=t[ks], **{f"prof_{n}": line[n][ks] for n in KEEP})
        arr.update(c_end=measure.sound_speed(mech, line, -1), **{f"end_{s}": line[s][-1] for s in measure.SPECIES_LINE})
    return arr, num


def health(case, num, spec, g, full=True, log_name=SOLVER):
    """monitor.assess on the log, plus what a sound run of this problem must do."""
    # The log is judged from the first step of the planned size on. The run starts from rest with a step that grows
    # from 1e-12 s, so over the first steps the Courant number rises a million times: monitor.assess reads that as a blow-up.
    log = monitor.parse_log(Path(case) / f"log.{log_name}")
    tt = np.array([s["time"] for s in log["steps"]])
    full_size = np.nonzero(np.diff(tt, prepend=0.0) >= 0.99 * num["dt"])[0]
    k = int(full_size[0]) if len(full_size) else 0
    a = monitor.assess_arrays(monitor.log_arrays(log["steps"][k:]), log, end_time=g["t_end"])
    num["ramp_steps"] = k
    add = a["checks"].append
    if full:
        add(dict(name="The run reaches the end of the time of interest", status="ok" if abs(num["t_last"] / g["t_end"] - 1) < 1e-9 else "fail",
                 value=f"{num['steps_done']} steps (planned {spec['steps']}, plus the growing first steps)", meaning="The history must cover the whole time."))
    add(dict(name="Temperature and pressure never fall below the start values", status="ok" if num["T_min"] >= g["T_0"] * (1 - 1e-3) and num["p_min"] >= g["p_0"] * (1 - 1e-3) else "fail",
             value=rf"lowest temperature \({num['T_min']:.3f}\,\mathrm{{K}}\), lowest pressure \({num['p_min']:.1f}\,\mathrm{{Pa}}\)",
             meaning="Nothing in a closed tube of burning gas cools the gas below its start state. A value below it is an undershoot of the scheme at the front."))
    add(dict(name="The mass in the tube does not change", status="ok" if num["mass_dev"] < MASS_TOL else "fail",
             value=rf"largest relative change \({tex_num(num['mass_dev'], 2)}\)" if num["mass_dev"] else "none",
             meaning="The tube is closed."))
    a["status"] = "fail" if any(c["status"] == "fail" for c in a["checks"]) else "warn" if any(c["status"] == "warn" for c in a["checks"]) else "ok"
    return a


def summary_line(name, meta):
    s = f"{name}: {meta['steps_done']} steps, wall {meta['wall_time']:.1f} s on {meta['cores']} cores, health {meta['health']['status']}"
    if "speed" in meta:
        e, b = meta["energy"], meta["burnt"]
        s += (f"; speed {meta['speed']['D']:.1f} ({meta['speed_err']:+.2%}), gap max {meta['gap_max'] * 1e6:.0f} um (per step {meta['gap_step_max'] * 1e6:.0f}), "
              f"peak {meta['peak']['mean'] / 1e5:.2f} bar = {meta['peak']['ratio']:.3f} of vN, energy end {e['end']:+.3%} rate {e['rate_late']:+.3%} "
              f"start-up {meta['energy_startup']:+.3%}, mass {meta['mass_dev']:.1e}, atoms {max(meta['atoms_dev'].values()):.1e}, "
              f"sonic point {'p %.3f T %.3f of CJ at %.2f mm' % (b['p'] / CJ['p'], b['T'] / CJ['T'], b['d'] * 1e3) if b['found'] else 'not found'}, "
              f"end p {meta['end_p'] / 1e5:.3f} bar, {meta.get('us_per_cell_step', float('nan')):.2f} us/cell/step")
    else:
        s += (f"; stopped at {meta['t_last'] * 1e6:.2f} us, front {meta['front_last'] * 1e3:.4f} mm, burning front {meta['burning_last'] * 1e3:.4f} mm, "
              f"energy {meta['energy_last'] if meta['energy_last'] is None else format(meta['energy_last'], '+.3%')}, hot cell steps {meta['hot_cell_steps']:.3e}")
    return s


CJ = {}


def context():
    problem, numerics, _ = theory.load()
    g = theory.given(problem)
    th = theory.numbers()
    CJ.update(th["cj"])
    return g, numerics, th, theory.mech()


def do_one(spec, ctx, prefix=""):
    t0 = time.time()
    case = make(spec, ctx, prefix, t_stop=spec["fraction"] * ctx[0]["t_end"] if spec["fraction"] < 1 else None)
    try:
        wall = foam.run(command(spec), case, log_name=SOLVER, timeout=3600)
    except RuntimeError as e:
        RUNS.mkdir(parents=True, exist_ok=True)
        log = (case / f"log.{SOLVER}").read_text(errors="replace")
        meta = dict(spec, crashed=True, case_dir=str(case), steps_done=len(re.findall(r"^Time = ", log, re.M)), error=str(e)[-800:])
        (RUNS / f"{spec['name']}.json").write_text(json.dumps(meta, indent=1, default=float))
        print(f"{spec['name']}: STOPPED BY THE SOLVER after {meta['steps_done']} steps\n{meta['error']}", flush=True)
        return meta
    try:
        return finish(case, spec, ctx, wall, time.time() - t0)
    except Exception:                               # a run that cannot be read must not stop the runs after it; it shows as missing
        import traceback
        print(f"{spec['name']}: RUN FINISHED BUT COULD NOT BE READ (wall {wall:.1f} s)\n{traceback.format_exc()}", flush=True)
        return None


def finish(case, spec, ctx, wall, total):
    g = ctx[0]
    whole = spec["fraction"] >= 1
    arr, num = collect(case, spec, ctx, full=whole)
    if not whole:                                   # a run of the first part of the time: what it is compared on
        num.update(front_last=float(arr["xp"][-1]), burning_last=float(arr["xb"][-1]),
                   energy_last=float(arr["dE"][-1] / arr["q"][-1]) if spec["chemistry"] else None)
    p = perf.summarize(case, SOLVER, spec["N"], spec["cores"])
    prof = perf.profiling(case, spec["cores"]) if spec["profiling"] else {}
    meta = dict(spec, case_dir=str(case), wall_time=wall, total_time=total, mesh_ok=mesh_ok(case),
                health=health(case, num, spec, g, full=whole), openfoam=foam.foam_env("WM_PROJECT_VERSION"), python=platform.python_version(),
                run_root=str(run_root()), us_per_cell_step=p["us_per_cell_step"], cpu_efficiency=p["cpu_efficiency"], cpu_s=p["cpu_s"],
                peak_mem_mb=p["peak_mem_mb"], lin_iters=p["lin_iters"], solver_cpu_s=p["solver_cpu_s"],
                profiling={k: v for k, v in prof.items()}, load=json.loads((Path(case) / f"perf.{SOLVER}.json").read_text()).get("load_1min"), **num)
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{spec['name']}.npz", **arr)
    (RUNS / f"{spec['name']}.json").write_text(json.dumps(meta, indent=1, default=float))
    print(summary_line(spec["name"], meta), flush=True)
    return meta


# ---- pre-flight ---------------------------------------------------------------------------------------------------
def do_preflight(only_full=False):
    ctx = context()
    g, numerics, th, mech = ctx
    sp_all, pfn = specs(numerics), numerics["preflight"]
    out = dict(short=[], gas_constant=foam.gas_constant())
    if not only_full:
        for N in pfn["short"]["cells"]:
            spec = sp_all[f"N{N}"]
            case = make(spec, ctx, prefix="pf_")
            cmd = command(spec)
            pf = monitor.preflight(case, cmd, end_time_short=pfn["short"]["fraction"] * g["t_end"], end_time_full=g["t_end"], timeout=900)
            pfc, log_name = Path(pf["case"]), cmd.split()[0]
            arr, num = collect(pfc, spec, ctx, full=False, log_name=log_name)
            hl = health(pfc, num, spec, g, full=False, log_name=log_name)
            n_short = round(pfn["short"]["fraction"] * spec["steps"])
            n_samp = round(pfn["short"]["fraction"] * numerics["main"]["samples"])
            f = num["first"]
            checks = hl["checks"] + [
                dict(name="The grid passes checkMesh", status="ok" if mesh_ok(case) else "fail", value="Mesh OK" if mesh_ok(case) else "not OK", meaning="Equal cells in one row."),
                dict(name="One row per time step in the three per-step files, and the sampled states along the tube at the planned times",
                     status="ok" if num["steps_done"] >= n_short and num["samples"] == n_samp and num["dt_max"] <= num["dt"] * 1.02 else "fail",
                     value=f"{num['steps_done']} rows ({n_short} steps of the planned size plus the growing first steps); {num['samples']} sampled states, {n_samp} planned",
                     meaning="Fronts, totals and peaks are read from these files. No step may be longer than the planned one."),
                dict(name="The solver starts from the start state of Theory", status="ok" if abs(f["mass"]) < 1e-7 and abs(f["energy"]) < 1e-3 and abs(f["p_max"] / g["p_ign"] - 1) < 1e-6 else "fail",
                     value=rf"after the first step: mass differs by \({f['mass'] * 1e12:.2f} \times 10^{{-12}}\) (relative), total energy by \({100 * f['energy']:.3f}\,\%\) of what the "
                           rf"ignition layer holds above the gas around it, highest pressure \({f['p_max'] / 1e6:.4f}\,\mathrm{{MPa}}\)",
                     meaning="Checks the mole ratio, the molar masses, the gas constant, the layer and the bookkeeping of the energy, all against values computed without the solver."),
                dict(name="The chemistry is running", status="ok" if num["qdot_abs_max"] > 1e9 else "fail",
                     value=rf"largest heat release rate in the short run \({num['qdot_abs_max'] / 1e12:.2f} \times 10^{{12}}\,\mathrm{{W/m^3}}\)",
                     meaning="A heat release that stays at zero would mean the reactions are switched off or the output is not written."),
                dict(name="Atoms kept during the short run", status="ok" if max(num["atoms_dev"].values()) < MASS_TOL else "fail",
                     value=rf"largest relative change \({max(num['atoms_dev'].values()) * 1e12:.2f} \times 10^{{-12}}\)", meaning="The reactions only rearrange atoms."),
            ]
            status = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "ok"
            pf.update(checks=checks, status=status, name=spec["name"], N=N, cores=spec["cores"], n_steps=n_short, headers=num["headers"], first=f,
                      front=float(arr["xp"][-1]), wall_short=json.loads((pfc / f"perf.{log_name}.json").read_text())["wall_s"])
            out["short"].append(pf)
            print(f"preflight {spec['name']}: {status}; {pf['wall_short']:.1f} s for {num['steps_done']} steps on {spec['cores']} cores; front at {pf['front'] * 1e3:.3f} mm", flush=True)
            for c in checks:
                print(f"   {c['status']:4s} {c['name']}: {c['value']}")
            print("   output files:", *num["headers"], sep="\n      ")
        dec = yaml.safe_load((HERE / "decisions.yaml").read_text(encoding="utf-8")) if (HERE / "decisions.yaml").exists() else []
        out["decisions"] = contract.check_decisions(dec) + contract.uncovered_settings(dec, out["short"][-1]["case"])
        out["ignored"] = foam.ignored_settings(out["short"][-1]["case"])
        print("   decisions:", out["decisions"] or "cover every setting", "| ignored settings:", out["ignored"] or "none")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "preflight.json").write_text(json.dumps(out, indent=1, default=str))
        if any(r["status"] == "fail" for r in out["short"]):
            raise SystemExit("Pre-flight FAILED: fix the setup before the full runs.")
    for name, spec in sp_all.items():
        if spec["stage"] == "preflight" and not (RUNS / f"{name}.json").exists():
            do_one(spec, ctx)
    (OUT / "preflight.done").write_text(json.dumps(dict(finished=time.strftime("%Y-%m-%d %H:%M:%S"))))


def do_production(names):
    ctx = context()
    sp_all = specs(ctx[1])
    todo = names or [n for n, s in sp_all.items() if not (RUNS / f"{n}.json").exists()]
    for n in todo:                                  # one at a time: the timings are then comparable
        do_one(sp_all[n], ctx)
    (OUT / "production.done").write_text(json.dumps(dict(runs=todo, finished=time.strftime("%Y-%m-%d %H:%M:%S"))))


def do_recollect(names):
    ctx = context()
    sp_all = specs(ctx[1])
    for n in names or [n for n in sp_all if (RUNS / f"{n}.json").exists()]:
        old = json.loads((RUNS / f"{n}.json").read_text())
        if not old.get("crashed"):
            finish(Path(old["case_dir"]), sp_all[n], ctx, old["wall_time"], old["total_time"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    if mode == "preflight":
        do_preflight(only_full="full" in sys.argv[2:])
    elif mode == "production":
        do_production(sys.argv[2:])
    elif mode == "recollect":
        do_recollect(sys.argv[2:])
    elif mode == "list":
        _, num, _ = theory.load()
        for n, s in specs(num).items():
            print(f"{n:22s} {s['stage']:10s} {s['N']:5d} cells {s['steps']:6d} steps, {s['passes']} passes, {s['ode']}, rel {s['rel']:g}, "
                  f"Treact {s['t_react']:g}, {s['cores']} cores, studies: {', '.join(s['studies'])}")
    else:
        raise SystemExit(f"unknown mode {mode}")
