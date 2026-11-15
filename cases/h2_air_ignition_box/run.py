"""Build and run the parcel for every run in numerics.yaml; save what the analysis needs.

Usage (repository root, inside WSL; or through py.sh):
  python -m cases.h2_air_ignition_box.run preflight               # scheme check, short tests on the coarsest and finest level,
                                                                  # then the three coarsest levels in full
  python -m cases.h2_air_ignition_box.run production [NAME ...]   # every other run (default: all not yet made), one at a time
  python -m cases.h2_air_ignition_box.run recollect [NAME ...]    # read finished runs again (after a change of a reading rule)
  python -m cases.h2_air_ignition_box.run list                    # the runs
Outputs: results/h2_air_ignition_box/preflight.json, runs/<name>.json + runs/<name>.npz
Run names: S<steps> = fixed time step t_end / steps with the main settings; a suffix names what differs
(_rel<r>_abs<a> tolerances, _<method>, _notable); adj_rel<r>_abs<a> = time step set by the chemistry solver.
"""
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cfdlib import contract, foam, monitor
from cfdlib.report import tex_num
from . import measure, theory
from . import mech as M

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"
SOLVER = "chemFoam"
SCHEME_DIR = "$FOAM_TUTORIALS/combustion/chemFoam/h2/chemkin"      # read where it is shipped; OpenFOAM expands the variable
TABLE = ["T", "p", "rho", "h"]                                     # then every species of the scheme


def run_root():
    return Path(foam.foam_env("FOAM_RUN")) / "agentic" / HERE.name


def specs(numerics):
    """Every run as {name: dict(name, steps or None, adjust, chem, ode, rel, abs, table, studies)}."""
    main, R, out = numerics["main"], numerics["runs"], {}

    def add(study, steps=None, chem=None, ode=None, rel=None, abs_=None, table=True):
        chem, ode = chem or main["chem"], ode or main["ode"]
        rel, abs_ = rel if rel is not None else main["rel_tol"], abs_ if abs_ is not None else main["abs_tol"]
        name = f"S{steps}" if steps else "adj"
        if chem != main["chem"]:
            name += f"_{chem}"
        elif ode != main["ode"]:
            name += f"_{ode}"
        if (rel, abs_) != (main["rel_tol"], main["abs_tol"]) or not steps:
            name += f"_rel{rel:g}_abs{abs_:g}"
        if not table:
            name += "_notable"
        s = out.setdefault(name, dict(name=name, steps=steps, adjust=not steps, chem=chem, ode=ode, rel=rel, abs=abs_, table=table, studies=[]))
        s["studies"].append(study)

    for n in R["family_steps"]:
        add("time step", steps=n)
    add("tolerance", steps=R["tolerance"]["steps"])
    for rel, abs_ in R["tolerance"]["pairs"]:
        add("tolerance", steps=R["tolerance"]["steps"], rel=rel, abs_=abs_)
    for rel, abs_ in R["adjustable"]["pairs"]:
        add("adjustable", rel=rel, abs_=abs_)
    add("method", steps=R["methods"]["steps"])
    for chem, ode in R["methods"]["variants"]:
        add("method", steps=R["methods"]["steps"], chem=chem, ode=None if ode == "none" else ode)
    add("timing", steps=R["timing"]["steps"])
    add("timing", steps=R["timing"]["steps"], table=False)
    return out


def params_for(spec, g, numerics, species):
    t_end, adj = g["t_end"], numerics["runs"]["adjustable"]
    return dict(chem_solver=spec["chem"], ode_solver=spec["ode"], rel_tol=repr(spec["rel"]), abs_tol=repr(spec["abs"]),
                init_chem_dt=repr(numerics["main"]["init_chem_dt"]), scheme_dir=SCHEME_DIR,
                x_H2=repr(g["x_H2"]), x_O2=repr(g["x_O2"]), x_N2=repr(g["x_N2"]), p=repr(g["p"]), T_0=repr(g["T_0"]),
                t_end=repr(t_end), dt=repr(adj["first_dt"] if spec["adjust"] else t_end / spec["steps"]),
                max_dt=repr(adj["max_dt"]), adjust="on" if spec["adjust"] else "off",
                write_interval=repr(t_end / 2 if spec["adjust"] else t_end),
                table=spec["table"], table_fields=" ".join(TABLE + list(species)))


def make(spec, g, numerics, species, prefix=""):
    return foam.render_case(HERE / "template", run_root() / f"{prefix}{spec['name']}", params_for(spec, g, numerics, species))


# ---- reading a finished run -----------------------------------------------------------------------------------
def read_out(case):
    """chemFoam's own history file: time, temperature, pressure after every step."""
    a = np.loadtxt(Path(case) / "chemFoam.out", comments="#", ndmin=2)
    head = (Path(case) / "chemFoam.out").read_text().splitlines()[0]
    return a[:, 0], a[:, 1], a[:, 2], head


def read_table(case):
    """The table written at every step by the function object: {column: array}, and its header line."""
    f = sorted((Path(case) / "postProcessing" / "state").rglob("volFieldValue.dat"))[0]
    head = [l for l in f.read_text().splitlines()[:8] if l.startswith("# Time")][0]
    cols = re.findall(r"volAverage\((\w+)\)", head)
    a = np.loadtxt(f, comments="#", ndmin=2)
    assert a.shape[1] == len(cols) + 1, f"{f}: {a.shape[1]} columns for {len(cols)} names"
    return {c: a[:, i + 1] for i, c in enumerate(cols)}, head


def exec_times(case):
    """The solver's own clock (processor time, printed to 0.01 s) after every step."""
    return np.array([float(x) for x in re.findall(r"^ExecutionTime = ([\d.eE+-]+) s", (Path(case) / f"log.{SOLVER}").read_text(errors="replace"), re.M)])


def collect(case, spec, g, numerics, th, mech):
    """Apply the reading rules to a finished (or short) run. Returns (arrays for the figures, numbers)."""
    ext = numerics["extraction"]
    t_out, T_out, p_out, head_out = read_out(case)
    n = len(t_out)
    if spec["adjust"]:
        t = t_out
    else:
        dt = g["t_end"] / spec["steps"]
        t = dt * (np.arange(n) + 1)                               # exact; the file adds the step n times
        assert np.allclose(t, t_out, rtol=1e-9, atol=0), "history file: rows are not one per time step"
    st = th["start"]
    tt, TT = np.append(0.0, t), np.append(g["T_0"], T_out)
    num = dict(steps=n, t_last=float(t_out[-1]), T_end=float(T_out[-1]), t_ign=measure.crossing_time(tt, TT, g["T_ign"]),
               levels=measure.crossings(tt, TT, ext["levels"]), p_dev=float(np.max(np.abs(p_out / g["p"] - 1))),
               T_max=float(TT.max()), T_min=float(TT.min()), dt_min=float(np.diff(tt).min()), dt_max=float(np.diff(tt).max()),
               headers=[head_out])
    num["rise_rate"], num["rise_t"] = measure.fastest_rise(tt, TT)
    k1, k2 = (int(np.argmax(TT >= L)) if TT.max() >= L else len(TT) - 1 for L in ext["phases"])
    num["phase_steps"] = [k1, k2 - k1, len(TT) - 1 - k2]
    ex = exec_times(case)
    if len(ex) == n and n > 1:
        num["loop_s"] = float(ex[-1] - ex[0])
        num["us_per_step"] = 1e6 * num["loop_s"] / (n - 1)
        e = np.append(ex[0], ex)                                   # e[k] = clock after step k (k = 0: before the loop, about)
        num["phase_s"] = [float(e[k1] - e[0]), float(e[k2] - e[k1]), float(e[-1] - e[k2])]
        edges = np.round(np.linspace(0, n, min(200, n) + 1)).astype(int)        # the cost of a step along the run, in 200 stretches
        cost = dict(cost_t=0.5 * (tt[edges[:-1]] + tt[edges[1:]]), cost_us=1e6 * np.diff(e[edges]) / np.diff(edges))
    else:
        cost = {}
    if spec["table"]:
        tab, head = read_table(case)
        assert len(tab["T"]) == n and np.allclose(tab["T"], T_out, rtol=1e-12), "table and history file disagree"
        sp = mech.species
        Y = np.vstack([[st["Y"][s] for s in sp], np.column_stack([tab[s] for s in sp])])
        rho = np.append(st["rho0_of"], tab["rho"])
        hs = np.append(st["hs0_of"], tab["h"])
        q = th["bounds"]["of"]["q"]
        at = measure.atoms(Y, mech)
        present = at[0] > 0
        drift = np.max(np.abs(at[:, present] / at[0, present] - 1), axis=0)
        H = measure.enthalpy(TT, Y, mech)
        Wmix = 1.0 / (Y / mech.W).sum(axis=1)
        num.update(
            headers=[head_out, head], first_row={c: float(tab[c][0]) for c in tab},
            atoms_drift={e: float(d) for e, d in zip(np.array(mech.elements)[present], drift)},
            atoms_drift_at={e: float(tt[int(np.argmax(np.abs(at[:, i] / at[0, i] - 1)))]) for i, e in enumerate(mech.elements) if present[i]},
            sumY_dev=float(np.max(np.abs(Y.sum(axis=1) - 1))), Y_min=float(Y.min()),
            energy_dev=float(np.max(np.abs(H - th["start"]["h0_of"])) / q), energy_end=float((H[-1] - th["start"]["h0_of"]) / q),
            hs_dev=float(np.max(np.abs(hs - measure.sensible(TT, Y, mech))) / q),
            state_dev=float(np.max(np.abs(rho * mech.conv.R * TT / (g["p"] * Wmix) - 1))),
            Y_end={s: float(Y[-1, i]) for i, s in enumerate(sp)},
            Y_peak={s: [float(Y[:, i].max()), float(tt[int(np.argmax(Y[:, i]))])] for i, s in enumerate(sp)})
        keep = measure.thin(len(tt), ext["store_points"])
        arr = dict(t=tt[keep], T=TT[keep], Y=Y[keep], rho=rho[keep], atoms_dev=(at[:, present] / at[0, present] - 1)[keep],
                   energy=((H - th["start"]["h0_of"]) / q)[keep])
    else:
        keep = measure.thin(len(tt), ext["store_points"])
        arr = dict(t=tt[keep], T=TT[keep])
    arr.update(cost)
    return arr, num


def health(case, num, g, th, full=True):
    """monitor.assess on the log, plus the checks that belong to a reacting parcel."""
    a = monitor.assess(Path(case) / f"log.{SOLVER}", end_time=g["t_end"])
    Tcc = th["bounds"]["of"]["T_cc"]
    add = a["checks"].append
    if full:
        add(dict(name="The run reaches the end of the time of interest", status="ok" if abs(num["t_last"] / g["t_end"] - 1) < 1e-9 else "fail",
                 value=rf"last time \({tex_num(num['t_last'], 6)}\,\mathrm{{s}}\)", meaning="The history must cover the whole millisecond."))
    add(dict(name="The temperature stays between the start and the limit of complete burning",
             status="ok" if num["T_min"] > g["T_0"] - 0.01 and num["T_max"] < Tcc else "fail",
             value=rf"lowest \({num['T_min']:.4f}\,\mathrm{{K}}\), highest \({num['T_max']:.2f}\,\mathrm{{K}}\)",
             meaning="Nothing cools the parcel, and it cannot get hotter than complete burning would make it (Theory)."))
    add(dict(name="The pressure stays as given", status="ok" if num["p_dev"] < 1e-12 else "fail",
             value=rf"largest relative change \({tex_num(num['p_dev'], 1) if num['p_dev'] else '0'}\)", meaning="The problem holds the pressure constant."))
    if "Y_min" in num:
        add(dict(name="No negative mass fraction, and the mass fractions add up to one",
                 status="ok" if num["Y_min"] >= -1e-12 and num["sumY_dev"] < 1e-9 else "fail",
                 value=rf"lowest mass fraction \({tex_num(num['Y_min'], 2) if num['Y_min'] else '0'}\), sum off by at most \({tex_num(num['sumY_dev'], 1) if num['sumY_dev'] else '0'}\)",
                 meaning="A chemistry solver that oversteps makes a species negative; mass fractions that do not add up mean lost or made mass."))
    a["status"] = "fail" if any(c["status"] == "fail" for c in a["checks"]) else "warn" if any(c["status"] == "warn" for c in a["checks"]) else "ok"
    return a


def summary_line(name, meta):
    s = (f"{name}: {meta['steps']} steps, wall {meta['wall_time']:.2f} s, {meta.get('us_per_step', float('nan')):.1f} us/step, health {meta['health']['status']}; "
         f"t_ign {meta['t_ign']:.9e} s, T_end {meta['T_end']:.5f} K")
    if "atoms_drift" in meta:
        s += (f"; atoms {max(meta['atoms_drift'].values()):.1e}, energy {meta['energy_dev']:.1e}, hs {meta['hs_dev']:.1e}, state {meta['state_dev']:.1e}, "
              f"sumY {meta['sumY_dev']:.1e}, Ymin {meta['Y_min']:.1e}")
    return s


def context():
    problem, numerics, _ = theory.load()
    g = theory.given(problem)
    th = theory.numbers()
    mech = M.build(M.openfoam_conventions())     # the constants OpenFOAM uses: its own data are checked with its own constants
    return g, numerics, th, mech


def do_one(spec, ctx):
    g, numerics, th, mech = ctx
    t0 = time.time()
    case = make(spec, g, numerics, mech.species)
    try:
        wall = foam.run(SOLVER, case, timeout=1800)
    except RuntimeError as e:
        RUNS.mkdir(parents=True, exist_ok=True)
        log = (case / f"log.{SOLVER}").read_text(errors="replace")
        meta = dict(spec, crashed=True, case_dir=str(case), steps_done=len(re.findall(r"^Time = ", log, re.M)), error=str(e)[-600:])
        (RUNS / f"{spec['name']}.json").write_text(json.dumps(meta, indent=1, default=float))
        print(f"{spec['name']}: STOPPED BY THE SOLVER after {meta['steps_done']} steps", flush=True)
        return meta
    return finish(case, spec, ctx, wall, time.time() - t0)


def finish(case, spec, ctx, wall, total):
    g, numerics, th, mech = ctx
    arr, num = collect(case, spec, g, numerics, th, mech)
    perf = json.loads((Path(case) / f"perf.{SOLVER}.json").read_text())
    meta = dict(spec, case_dir=str(case), wall_time=wall, total_time=total, cpu_s=perf.get("cpu_s"), peak_mem_mb=perf.get("peak_mem_mb"),
                health=health(case, num, g, th), openfoam=foam.foam_env("WM_PROJECT_VERSION"), python=platform.python_version(),
                run_root=str(run_root()), **num)
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{spec['name']}.npz", **arr)
    (RUNS / f"{spec['name']}.json").write_text(json.dumps(meta, indent=1, default=float))
    print(summary_line(spec["name"], meta), flush=True)
    return meta


# ---- pre-flight -------------------------------------------------------------------------------------------------
def scheme_check(mech):
    """Does OpenFOAM read the scheme as mech.py does? OpenFOAM's converter (chemkinToFoam) writes its reading
    to two files; every rate constant, exponent, activation temperature, collision efficiency and molar mass
    in them is compared with the independent reading. Returns dict(ok, reactions, worst, notes)."""
    d = run_root() / "scheme_check"
    d.mkdir(parents=True, exist_ok=True)
    src = Path(foam.foam_env("FOAM_TUTORIALS")) / "combustion/chemFoam/h2/chemkin"
    foam.run(f"chemkinToFoam {src}/chem.inp {src}/therm.dat {src}/transportProperties reactionsFoam thermoFoam", d, "chemkinToFoam")
    text = (d / "reactionsFoam").read_text()
    blocks = re.split(r"un-named-reaction-\d+\s*\{", text)[1:]
    num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    worst, notes, kinds = 0.0, [], []
    ok = len(blocks) == len(mech.reactions)
    for j, b in enumerate(blocks[:len(mech.reactions)]):
        kind = re.search(r"type\s+(\w+);", b).group(1)
        kinds.append(kind)
        triples = [(float(a), float(be), float(ta)) for a, be, ta in re.findall(rf"\bA\s+({num});\s*beta\s+({num});\s*Ta\s+({num});", b)]
        r = mech.reactions[j]
        order = sum(r["lhs"].values())
        mine = []
        if mech.fall[j]:                                           # OpenFOAM lists the low-pressure form first, then the high-pressure one
            mine.append((np.exp(mech.lnA0[j]) * 1e3 ** order, mech.beta0[j], mech.Ta0[j]))
            mine.append((np.exp(mech.lnA[j]) * 1e3 ** (order - 1), mech.beta[j], mech.Ta[j]))
        else:
            mine.append((np.exp(mech.lnA[j]) * 1e3 ** (order - 1 + (1 if mech.third[j] else 0)), mech.beta[j], mech.Ta[j]))
        if len(triples) != len(mine):
            ok = False
            notes.append(f"reaction {j + 1} ({r['eq']}): {len(triples)} rate laws in OpenFOAM's reading, {len(mine)} here")
            continue
        for (a1, b1, t1), (a2, b2, t2) in zip(triples, mine):       # per kmol in OpenFOAM, per mol here
            dev = max(abs(a1 / a2 - 1), abs(b1 - b2), abs(t1 - t2) / max(abs(t2), 1.0))
            worst = max(worst, dev)
        effs = dict((s, float(v)) for s, v in re.findall(rf"\((\w+)\s+({num})\)", b))
        if mech.fall[j] or mech.third[j]:
            if any(abs(effs.get(s, np.nan) - mech.eff[j, i]) > 1e-12 for i, s in enumerate(mech.species)):
                ok = False
                notes.append(f"reaction {j + 1} ({r['eq']}): collision efficiencies differ")
        elif effs:
            ok = False
            notes.append(f"reaction {j + 1} ({r['eq']}): OpenFOAM has collision partners, the independent reading none")
        if ("Troe" in kind) != bool(mech.is_troe[j]) or ("FallOff" in kind) != bool(mech.fall[j]) or ("thirdBody" in kind) != bool(mech.third[j]):
            ok = False
            notes.append(f"reaction {j + 1} ({r['eq']}): kind {kind} differs")
        if mech.is_troe[j]:
            tr = [float(re.search(rf"\b{k}\s+({num});", b).group(1)) for k in ("alpha", "Tsss", "Ts", "Tss")]
            if not np.allclose(tr, mech.troe[j], rtol=1e-12):
                ok = False
                notes.append(f"reaction {j + 1}: fall-off shape parameters differ")
    th_text = (d / "thermoFoam").read_text()
    w_dev = 0.0
    for i, s in enumerate(mech.species):
        mw = float(re.search(rf"^{s}\s*\{{\s*specie\s*\{{\s*molWeight\s+({num});", th_text, re.M).group(1))
        w_dev = max(w_dev, abs(mw / (1e3 * mech.W[i]) - 1))
    ok = ok and worst < 1e-8 and w_dev < 1e-9
    return dict(ok=bool(ok), reactions=len(blocks), worst=float(worst), w_dev=float(w_dev), notes=notes,
                kinds={k: kinds.count(k) for k in sorted(set(kinds))})


def do_preflight():
    ctx = context()
    g, numerics, th, mech = ctx
    sp_all, pfn = specs(numerics), numerics["preflight"]
    fam = numerics["runs"]["family_steps"]
    out = dict(short=[], gas_constant=foam.gas_constant())
    sc = scheme_check(mech)
    out["scheme"] = sc
    print(f"scheme check: {'ok' if sc['ok'] else 'FAIL'}; {sc['reactions']} reactions, kinds {sc['kinds']}; largest difference of a rate "
          f"parameter {sc['worst']:.1e}, of a molar mass {sc['w_dev']:.1e}; {sc['notes'] or 'no notes'}", flush=True)
    st = th["start"]
    for n in (fam[0], fam[-1]):
        spec = sp_all[f"S{n}"]
        case = make(spec, g, numerics, mech.species, prefix="pf_")
        pf = monitor.preflight(case, SOLVER, end_time_short=pfn["fraction"] * g["t_end"], end_time_full=g["t_end"], timeout=600)
        pfc = Path(pf["case"])
        arr, num = collect(pfc, spec, g, numerics, th, mech)
        hl = health(pfc, num, g, th, full=False)
        fr = num["first_row"]
        n_short = round(pfn["fraction"] * n)
        start_dev = max(abs(fr["rho"] / st["rho0_of"] - 1), abs(fr["h"] / st["hs0_of"] - 1), *(abs(fr[s] / st["Y"][s] - 1) for s in ("H2", "O2", "N2")))
        checks = hl["checks"] + [
            dict(name="One row per time step in both output files", status="ok" if num["steps"] == n_short else "fail",
                 value=f"{num['steps']} rows for {n_short} steps", meaning="The history and the table of species are needed after every step."),
            dict(name="The solver starts from the start state of Theory", status="ok" if start_dev < 1e-6 else "fail",
                 value=rf"after the first step, density, sensible enthalpy and the three mass fractions differ from the start values by at most \({tex_num(start_dev, 1)}\) (relative)",
                 meaning="Checks the mole ratio, the molar masses and the gas constant that the solver uses. The density is compared with "
                         "the value for OpenFOAM's own gas constant."),
            dict(name="Radicals appear from the first step", status="ok" if fr["H"] > 0 and fr["HO2"] > 0 and fr["OH"] > 0 else "fail",
                 value=rf"mass fractions after one step: H \({tex_num(fr['H'], 2)}\), HO2 \({tex_num(fr['HO2'], 2)}\), OH \({tex_num(fr['OH'], 2)}\)",
                 meaning="The start mixture holds no reactive species; the first reactions must make some, or nothing would ever ignite. "
                         "An output column that stays at zero would be a silent failure."),
            dict(name="Atoms and energy kept during the short run", status="ok" if max(num["atoms_drift"].values()) < 1e-9 and num["energy_dev"] < 1e-5 else "fail",
                 value=rf"atoms to \({tex_num(max(max(num['atoms_drift'].values()), 1e-17), 1)}\), enthalpy to \({tex_num(max(num['energy_dev'], 1e-17), 1)}\) of the heat released",
                 meaning="The physical check of every step (see Simulation)."),
        ]
        status = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "ok"
        pf.update(checks=checks, status=status, name=spec["name"], n_steps=n_short, headers=num["headers"],
                  first_row=fr, us_per_step=num.get("us_per_step"), projected=pf.get("est_full_runtime"),
                  wall_short=json.loads((pfc / f"perf.{SOLVER}.json").read_text())["wall_s"])
        out["short"].append(pf)
        print(f"preflight {spec['name']}: {status}; {pf['wall_short']:.2f} s for {num['steps']} steps; projected {pf['est_full_runtime']} s; "
              f"{num.get('us_per_step')} us/step", flush=True)
        for c in checks:
            print(f"   {c['status']:4s} {c['name']}: {c['value']}")
        print("   output files:", num["headers"], "| first row:", {k: fr[k] for k in ("T", "rho", "h", "H2", "H", "HO2")})
    dec = yaml.safe_load((HERE / "decisions.yaml").read_text(encoding="utf-8")) if (HERE / "decisions.yaml").exists() else []
    out["decisions"] = contract.check_decisions(dec) + contract.uncovered_settings(dec, out["short"][-1]["case"])
    out["ignored"] = foam.ignored_settings(out["short"][-1]["case"])
    print("   decisions:", out["decisions"] or "cover every setting")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(out, indent=1, default=str))
    if any(r["status"] == "fail" for r in out["short"]) or not sc["ok"]:
        raise SystemExit("Pre-flight FAILED: fix the setup before production runs.")
    for n in pfn["full_steps"]:
        do_one(sp_all[f"S{n}"], ctx)


def do_production(names):
    ctx = context()
    sp_all = specs(ctx[1])
    todo = names or [n for n in sp_all if not (RUNS / f"{n}.json").exists()]
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
        do_preflight()
    elif mode == "production":
        do_production(sys.argv[2:])
    elif mode == "recollect":
        do_recollect(sys.argv[2:])
    elif mode == "list":
        _, num, _ = theory.load()
        for n, s in specs(num).items():
            print(f"{n:34s} {'adjustable' if s['adjust'] else str(s['steps']) + ' steps':>12s}  {s['chem']}/{s['ode']}, rel {s['rel']:g}, abs {s['abs']:g}, "
                  f"table {s['table']}, studies: {', '.join(s['studies'])}")
    else:
        raise SystemExit(f"unknown mode {mode}")
