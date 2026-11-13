"""Build and run the heated-tube case (laminar water, uniform wall heat flux) on the meshes of numerics.yaml.

Usage (repo root, inside WSL):
  python -m cases.heated_pipe_re200.run preflight            # short runs on L1 and L4 from rest + full L1 run
  python -m cases.heated_pipe_re200.run production [L1 ...]  # live-stopped runs, coarse to fine, grid sequencing
  python -m cases.heated_pipe_re200.run buoyancy             # 3D half-tube, same mesh with g = 0 and g = 9.81
Outputs: results/heated_pipe_re200/preflight.json, runs/<name>.json + runs/<name>.npz, runs/DONE_<mode>
"""
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import optimize

from cfdlib import foam, monitor, perf

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"


def load():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    return problem, numerics


def given(problem):
    return {k: v["value"] for k, v in problem["given"].items()}


def run_root():
    return Path(foam.foam_env("FOAM_RUN")) / "agentic" / HERE.name


# ------------------------------------------------------------------------------------------------
# Meshes
# ------------------------------------------------------------------------------------------------
def first_cell(n, L, E):
    """First cell length of n geometric cells over L with last/first ratio E."""
    if abs(E - 1) < 1e-12:
        return L / n
    r = E ** (1.0 / (n - 1))
    return L * (r - 1) / (r ** n - 1)


def expansion_for_last(n, L, last):
    """Ratio E (last/first) such that the last of n geometric cells over L has length `last`."""
    return math.exp(optimize.brentq(lambda lE: first_cell(n, L, math.exp(lE)) * math.exp(lE) - last, -15, 15, xtol=1e-14))


def axial_spec(G, num, i):
    m = num["mesh"]
    n0, n1 = m["Nx_unheated"][i], m["Nx_heated"][i]
    Eh = m["heated_expansion"]
    dx0 = first_cell(n1, G["Lh"], Eh)
    E0 = expansion_for_last(n0, G["L0"], dx0)
    return dict(n0=n0, n1=n1, Eh=Eh, E0=E0, dx0=dx0, dx_inlet=first_cell(n0, G["L0"], E0), dx_out=dx0 * Eh)


def mesh_wedge(G, num, i):
    """Axisymmetric wedge: two blocks along x (unheated, heated), Nr cells from axis to wall."""
    a = G["D"] / 2
    th = math.radians(num["geometry"]["wedge_half_angle_deg"])
    c, s = a * math.cos(th), a * math.sin(th)
    xs = [-G["L0"], 0.0, G["Lh"]]
    v = [(x, 0.0, 0.0) for x in xs] + [(x, c, -s) for x in xs] + [(x, c, s) for x in xs]
    ax = axial_spec(G, num, i)
    Nr, Er = num["mesh"]["Nr"][i], num["mesh"]["radial_expansion"]
    blocks = [f"hex (0 1 4 3 0 1 7 6) ({ax['n0']} {Nr} 1) simpleGrading ({ax['E0']:.12g} {Er} 1)",
              f"hex (1 2 5 4 1 2 8 7) ({ax['n1']} {Nr} 1) simpleGrading ({ax['Eh']:.12g} {Er} 1)"]
    patches = {
        "inlet": dict(type="patch", faces=[(0, 0, 6, 3)]),
        "outlet": dict(type="patch", faces=[(2, 5, 8, 2)]),
        "unheatedWall": dict(type="wall", faces=[(3, 6, 7, 4)]),
        "heatedWall": dict(type="wall", faces=[(4, 7, 8, 5)]),
        "front": dict(type="wedge", faces=[(0, 3, 4, 1), (1, 4, 5, 2)]),
        "back": dict(type="wedge", faces=[(0, 1, 7, 6), (1, 2, 8, 7)]),
        "axis": dict(type="empty", faces=[(0, 1, 1, 0), (1, 2, 2, 1)]),
    }
    fmt = lambda p: tuple(f"{x:.12g}" for x in p)
    return dict(wedge=True, vertices=[fmt(p) for p in v], blocks=blocks, edges=[], patches=patches,
                cells=Nr * (ax["n0"] + ax["n1"]), Nr=Nr, axial=ax)


def mesh_half3d(G, num):
    """Half tube z >= 0 (symmetry plane z = 0 contains gravity, -y): O-grid cross-section with a
    rectangular core [-s, s] x [0, s] and three outer blocks (bottom, side, top) to the wall arc."""
    b = num["buoyancy"]
    a = G["D"] / 2
    sc = b["core_half_width"] * a
    Nc, Nrr = b["Nc"], b["Nr_outer"]
    Er = b["radial_expansion_outer"]
    ax = axial_spec(G, num, b["axial_level"])
    xs = [-G["L0"], 0.0, G["Lh"]]
    q = a / math.sqrt(2)
    # cross-section points (y, z): core 0..3, wall 4..7
    cs = [(-sc, 0), (sc, 0), (sc, sc), (-sc, sc), (-a, 0), (a, 0), (q, q), (-q, q)]
    v = [(x, y, z) for x in xs for (y, z) in cs]                  # index = 8 * ix + k
    V = lambda ix, k: 8 * ix + k
    # 2D blocks in (y, z): (p0 p1 p2 p3) counter-clockwise seen from +x (y right, z up => x1 x2 x3 right-handed
    # with x3 = +x). Local directions: x1 = first edge, x2 = second edge, x3 = axial.
    quads = [
        ((0, 1, 2, 3), (Nc, Nc // 2), (1, 1)),                    # core
        ((4, 0, 3, 7), (Nrr, Nc // 2), (1.0 / Er, 1)),            # bottom (y < 0): wall -> core along x1
        ((3, 2, 6, 7), (Nc, Nrr), (1, Er)),                       # side (z > 0): core -> wall along x2
        ((1, 5, 6, 2), (Nrr, Nc // 2), (Er, 1)),                  # top (y > 0): core -> wall along x1
    ]
    blocks, edges = [], []
    for ib, (n_ax, E_ax) in enumerate(((ax["n0"], ax["E0"]), (ax["n1"], ax["Eh"]))):
        for (p, (n1, n2), (e1, e2)) in quads:
            vs = [V(ib, k) for k in p] + [V(ib + 1, k) for k in p]
            blocks.append(f"hex ({' '.join(map(str, vs))}) ({n1} {n2} {n_ax}) simpleGrading ({e1:.12g} {e2:.12g} {E_ax:.12g})")
    for ix, x in enumerate(xs):
        # arc midpoints: wall from angle 180 deg (point 4) through 135 (7), 45 (6) to 0 (5), measured from +y
        for (k0, k1, phi) in ((4, 7, 157.5), (7, 6, 90.0), (6, 5, 22.5)):
            ph = math.radians(phi)
            edges.append(f"arc {V(ix, k0)} {V(ix, k1)} ({x:.12g} {a * math.cos(ph):.12g} {a * math.sin(ph):.12g})")
    faces_x = lambda ix, outward: [tuple(V(ix, k) for k in (p if outward else p[::-1])) for (p, _, _) in quads]
    wall = lambda ib: [(V(ib, 4), V(ib, 7), V(ib + 1, 7), V(ib + 1, 4)), (V(ib, 7), V(ib, 6), V(ib + 1, 6), V(ib + 1, 7)),
                       (V(ib, 6), V(ib, 5), V(ib + 1, 5), V(ib + 1, 6))]
    sym = lambda ib: [(V(ib, 4), V(ib + 1, 4), V(ib + 1, 0), V(ib, 0)), (V(ib, 0), V(ib + 1, 0), V(ib + 1, 1), V(ib, 1)),
                      (V(ib, 1), V(ib + 1, 1), V(ib + 1, 5), V(ib, 5))]
    patches = {
        "inlet": dict(type="patch", faces=faces_x(0, False)),
        "outlet": dict(type="patch", faces=faces_x(2, True)),
        "unheatedWall": dict(type="wall", faces=wall(0)),
        "heatedWall": dict(type="wall", faces=wall(1)),
        "symmetryPlane": dict(type="symmetryPlane", faces=sym(0) + sym(1)),
    }
    fmt = lambda p: tuple(f"{x:.12g}" for x in p)
    per_section = Nc * (Nc // 2) + 2 * Nrr * (Nc // 2) + Nc * Nrr
    return dict(wedge=False, vertices=[fmt(p) for p in v], blocks=blocks, edges=edges, patches=patches,
                cells=per_section * (ax["n0"] + ax["n1"]), axial=ax, per_section=per_section)


# ------------------------------------------------------------------------------------------------
# Cases
# ------------------------------------------------------------------------------------------------
def params(problem, num, mesh, g_y=0.0, max_iterations=None, relax_U=None, p_solver=None):
    G = given(problem)
    alpha = G["k"] / (G["rho"] * G["cp"])
    c = num["convergence"]
    probes = [(px * G["D"], pr * G["D"] / 2, 0.0) for px, pr in num["probes"]]
    if not mesh["wedge"]:                                       # 3D: same points, inside the half tube (z > 0)
        probes = [(x, 0.0, r if r > 0.01 * G["D"] else 0.01 * G["D"]) for x, r, _ in probes]
    return dict(mesh, U=G["U"], T_in=G["T_in"], wall_gradient=f"{G['q_w'] / G['k']:.12g}", nu=G["nu"],
                beta=G["beta"], Pr=f"{G['nu'] / alpha:.12g}", g_y=g_y,
                div_U=num["schemes"]["div_U"], div_T=num["schemes"]["div_T"],
                p_relTol=num["linear_solvers"]["p_relTol"], p_tol=num["linear_solvers"]["p_tol"], p_solver=p_solver or num["linear_solvers"]["p_solver"],
                consistent=num["relaxation"]["consistent"], relax_p=num["relaxation"]["p_rgh"],
                relax_U=relax_U or num["relaxation"]["U"],
                relax_T=num["relaxation"]["T"], max_iterations=max_iterations or c["max_iterations"],
                write_every=c["write_every"], balance_every=c["balance_every"],
                probes=[tuple(f"{x:.12g}" for x in p) for p in probes])


def build(name, problem, num, mesh, cores=1, **kw):
    case = foam.render_case(HERE / "template", run_root() / name, params(problem, num, mesh, **kw))
    foam.run("blockMesh", case)
    foam.run("checkMesh", case)
    if cores > 1:
        (case / "system" / "decomposeParDict").write_text(
            "FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\n"
            f"numberOfSubdomains {cores};\nmethod scotch;\n")
    return case


def at_output_precision(s, digits=12, n=20):
    """True when the last n changes of a sampled series are all within a few units of its last printed digit
    (the probes are written with 12 significant digits): the value is converged to output precision. Only for
    a value that HAS moved (range of its history above 1000 units): a probe the solution has not reached yet
    also shows zero changes (replay on L1: a false stop at iteration 40 without this condition)."""
    d = np.abs(np.diff(s[-(n + 1):]))
    if len(d) < n:
        return False
    quantum = 10.0 ** (math.floor(math.log10(max(abs(s[-1]), 1e-300))) - (digits - 1))
    moved = float(np.max(s) - np.min(s)) > 1000 * quantum
    return moved and float(d.max()) <= 10 * quantum


def live_decider(num):
    """Live stop rule: monitor.steady_check on the T probes and the axial velocity probes, every `every`
    iterations, held `hold` times. Work-around (insight): steady_check reports 'running, error inf' forever
    for a value converged to its 12 printed digits, because its consecutive changes are then exactly zero or
    one unit of the last digit (a zero change in the window makes the ratio undefined). Such a column counts as
    settled here: its remaining change is below 1e-11 relative, far below the tolerances."""
    c = num["convergence"]
    nprobe = len(num["probes"])
    every, hold = c["every"], c["hold"]
    tails, streak = {}, {}

    def check(v, tol, names):
        r = monitor.steady_check(v, tol, every, names)
        s = v[::-1][::every][::-1]
        for j, n in enumerate(names):
            q = r["quantities"][n]
            if q["state"] == "running" and at_output_precision(s[:, j]):
                q.update(state="settled", error=0.0, note="converged to output precision")
        running = [n for n, q in r["quantities"].items() if q["state"] == "running"]
        floors = [n for n, q in r["quantities"].items() if q["state"] == "floor"]
        r.update(ok=not running, status="warn" if floors else "ok",
                 reason=(f"running: {len(running)} of {len(names)} (e.g. {running[0]})" if running else
                         "floor: " + ", ".join(floors) if floors else "all settled"))
        return r

    def decide(case):
        key = str(case)
        tT = tails.setdefault(key + "T", monitor.DatTail(case, "probes", "T"))()
        tU = tails.setdefault(key + "U", monitor.DatTail(case, "probes", "U"))()
        m = min(len(tT), len(tU))
        if m < 6 * every:
            return None
        a = check(tT[:m, 1:], c["tol_T"], [f"T{i}" for i in range(nprobe)])
        b = check(tU[:m, [1 + 3 * i for i in range(nprobe)]], c["tol_U"], [f"Ux{i}" for i in range(nprobe)])
        ok = a["ok"] and b["ok"]
        streak[key] = streak.get(key, 0) + 1 if ok else 0
        return dict(stop=streak[key] >= hold, status="warn" if "warn" in (a["status"], b["status"]) else "ok",
                    reason=f"T: {a['reason']} | U: {b['reason']}", t=float(tT[m - 1, 0]),
                    quantities={**a["quantities"], **b["quantities"]})
    return decide


# ------------------------------------------------------------------------------------------------
# Reading results
# ------------------------------------------------------------------------------------------------
def read_patch(path, patch):
    """Values of a boundary patch in an OpenFOAM field file (ASCII): (n,) or (n, 3) array."""
    text = Path(path).read_text()
    bf = text[text.index("boundaryField"):]
    m = re.search(rf"\n\s*{patch}\s*\n\s*\{{", bf)
    body = bf[m.end():]
    body = body[:body.index("\n    }")]
    mv = re.search(r"\bvalue\s+(uniform|nonuniform)\s+", body)
    rest = body[mv.end():]
    if mv.group(1) == "uniform":
        val = rest[:rest.index(";")].strip()
        return np.array(val.strip("()").split(), float)
    mlist = re.match(r"List<(\w+)>\s*(\d+)\s*\(", rest)
    n = int(mlist.group(2))
    inner = rest[mlist.end():]
    inner = inner[:inner.rindex(")")]
    arr = np.array(inner.replace("(", " ").replace(")", " ").split(), float)
    return arr.reshape(n, -1) if mlist.group(1) == "vector" else arr[:n]


def wall_temperature(case, patch, T, C, Cf, gradient):
    """Face values of T on a wall patch, as OpenFOAM's fixedGradient / zeroGradient evaluate them:
    T_f = T_owner + gradient * n_f . (C_f - C_owner). These BCs do not write their face values, so they
    are rebuilt here from the owner cells (constant/polyMesh/owner and boundary). n_f is radial (wall faces
    are flat quads with their corners on the circle; in the wedge the radial direction is +y)."""
    pm = Path(case) / "constant" / "polyMesh"
    btext = (pm / "boundary").read_text()
    mb = re.search(rf"\n\s*{patch}\s*\n\s*\{{([^}}]*)\}}", btext)
    n = int(re.search(r"nFaces\s+(\d+);", mb.group(1)).group(1))
    start = int(re.search(r"startFace\s+(\d+);", mb.group(1)).group(1))
    otext = (pm / "owner").read_text()
    body = otext[otext.index("(", otext.index("// *")) + 1:]
    owner = np.array(body[:body.index(")")].split(), int)[start:start + n]
    nf = np.zeros_like(Cf)
    nf[:, 1:] = Cf[:, 1:]
    nf /= np.linalg.norm(nf, axis=1)[:, None]
    d = np.einsum("ij,ij->i", nf, Cf - C[owner])
    assert np.all(d > 0), "wall distance must be positive"
    return T[owner] + gradient * d


def dat_columns(case, fo, fname="surfaceFieldValue.dat"):
    rows = monitor.read_dat_live(case, fo, fname)
    return np.array([[x for v in r for x in (v if isinstance(v, tuple) else (v,))] for r in rows], float)


def collect(case, name, G, num, mesh, wall, cores, live):
    """Post-process one finished run into runs/<name>.json and .npz."""
    # mapFields copies EVERY field of the coarser case's latest time, including the C*/V written there by
    # postProcess; they are not inputs, so remove them from 0/ (the report contract reads 0/).
    for f in ("C", "Cx", "Cy", "Cz", "V"):
        (Path(case) / "0" / f).unlink(missing_ok=True)
    for f in ("writeCellCentres", "writeCellVolumes"):
        foam.run(f"postProcess -func {f} -latestTime", case, f)
    t = foam.time_dirs(case)[-1]
    C = foam.read_internal_field(t / "C")
    Vc = foam.read_internal_field(t / "V")
    U = foam.read_internal_field(t / "U")
    T = foam.read_internal_field(t / "T")
    Cw = read_patch(t / "C", "heatedWall")
    Cuw = read_patch(t / "C", "unheatedWall")
    Tw = wall_temperature(case, "heatedWall", T, C, Cw, G["q_w"] / G["k"])
    Tuw = wall_temperature(case, "unheatedWall", T, C, Cuw, 0.0)
    log = case / f"log.{num['solver']}"
    out = dict(name=name, case_dir=str(case), cells=int(len(T)), wall_time=wall, cores=cores,
               iterations=int(float(t.name)), live_stop=live.summary() if live else None,
               health=monitor.assess(log), mesh_ok="Mesh OK." in (case / "log.checkMesh").read_text(),
               perf=perf.summarize(case, num["solver"], len(T), cores), axial=mesh["axial"])
    arrays = {}
    if mesh["wedge"]:
        Nr = mesh["Nr"]
        n0, n1 = mesh["axial"]["n0"], mesh["axial"]["n1"]
        def grid(a):
            b0 = a[:Nr * n0].reshape(Nr, n0)
            b1 = a[Nr * n0:].reshape(Nr, n1)
            return np.concatenate([b0, b1], axis=1)                  # [j (r), i (x)]
        x, r = grid(C[:, 0]), grid(np.hypot(C[:, 1], C[:, 2]))
        assert np.all(np.diff(x, axis=1) > 0) and np.all(np.diff(r, axis=0) > 0), "cell ordering"
        ux, uy, Tg, Vg = grid(U[:, 0]), grid(U[:, 1]), grid(T), grid(Vc)
        ow = np.argsort(Cw[:, 0])
        ouw = np.argsort(Cuw[:, 0])
        xw = np.concatenate([Cuw[ouw, 0], Cw[ow, 0]])
        Twall = np.concatenate([Tuw[ouw], Tw[ow]])
        assert np.allclose(xw, x[-1], rtol=0, atol=1e-9), "wall faces and columns"
        Tb = np.sum(ux * Tg * Vg, axis=0) / np.sum(ux * Vg, axis=0)
        flow = np.sum(ux * Vg, axis=0)                               # proportional to the flux at each station
        arrays = dict(x=x[0], r=r[:, 0], ux=ux, uy=uy, T=Tg, V=Vg, Tw=Twall, Tb=Tb, flow=flow)
    else:
        arrays = dict(C=C, U=U, T=T, V=Vc, Cw=Cw, Tw=Tw, Cuw=Cuw, Tuw=Tuw)
    # energy and mass balance from the function objects (last row)
    fl = {p: dat_columns(case, f"{p}Flux")[-1, 1] for p in ("inlet", "outlet")}
    en = {p: dat_columns(case, f"{p}Enthalpy")[-1, 1] for p in ("inlet", "outlet")}
    out.update(phi_in=float(fl["inlet"]), phi_out=float(fl["outlet"]), phiT_in=float(en["inlet"]),
               phiT_out=float(en["outlet"]))
    # heated wall area from the mesh (sum of face areas): read via the patch's magSf is not written; use
    # the wedge/3D geometry instead in analyze. Probe histories for the convergence figure:
    pr = monitor.read_dat_live(case, "probes", "T")
    arrays["probe_T"] = np.array([[x for x in r] for r in pr], float)
    pu = monitor.read_dat_live(case, "probes", "U")
    arrays["probe_Ux"] = np.array([[r[0]] + [v[0] for v in r[1:]] for r in pu], float)
    arrays["res"] = np.array([0])
    res = foam.residual_history(log, ("Ux", "p_rgh", "T"))
    out["final_residuals"] = {k: (v[-1] if v else None) for k, v in res.items()}
    for k, v in res.items():
        arrays[f"res_{k}"] = np.array(v, float)
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{name}.npz", **arrays)
    (RUNS / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# ------------------------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------------------------
def do_preflight():
    problem, num = load()
    G = given(problem)
    n_short = num["preflight"]["iterations"]
    res = []
    last = len(num["mesh"]["levels"]) - 1
    gamg = "solver GAMG; smoother DICGaussSeidel;"
    for i, tag, extra in ((0, "", {}), (last, "", {}), (last, "gamg", dict(p_solver=gamg))):
        lv = num["mesh"]["levels"][i]
        mesh = mesh_wedge(G, num, i)
        case = build(f"pf{lv}{tag}", problem, num, mesh, **extra)
        pf = monitor.preflight(case, num["solver"], end_time_short=n_short, end_time_full=num["convergence"]["max_iterations"],
                               timeout=900)
        pf.update(level=lv, variant=tag or "chosen", p_solver=extra.get("p_solver", num["linear_solvers"]["p_solver"]),
                  cells=mesh["cells"], axial=mesh["axial"],
                  seconds_per_iteration=pf["wall_time"] / n_short if pf.get("wall_time") else None)
        pcase = Path(pf["case"])
        files = {}
        for d in sorted((pcase / "postProcessing").glob("*/*/*")):
            lines = d.read_text().splitlines()
            files[str(d.relative_to(pcase))] = lines[:3] + lines[-2:]
        pf["postProcessing_head"] = files
        print(f"preflight {lv} ({mesh['cells']} cells): {pf['status']}, {pf['seconds_per_iteration']:.4g} s/it")
        for ch in pf["checks"]:
            print(f"   {ch['status']:4s} {ch['name']}: {ch['value']}")
        for k, v in files.items():
            print("  ", k, "|", " || ".join(l[:150] for l in v))
        res.append(pf)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(res, indent=1, default=str))
    # Coarsest mesh run to the live-stop rule: its error against the exact solution sizes the acceptance
    # tolerances (with the expected order 2) before any production run.
    mesh = mesh_wedge(G, num, 0)
    case = build("pfL1full", problem, num, mesh)
    live = monitor.LiveStop(case, live_decider(num))
    wall = foam.run(num["solver"], case, watch=live, watch_every=2.0)
    rec = collect(case, "pfL1full", G, num, mesh, wall, 1, live)
    print(f"pfL1full: {rec['iterations']} iterations, {wall:.1f} s, health {rec['health']['status']}")


def do_production(levels, prefix="", sequence=True):
    problem, num = load()
    G = given(problem)
    decide = live_decider(num)
    i_first = num["mesh"]["levels"].index(levels[0])
    prev = run_root() / f"{prefix}{num['mesh']['levels'][i_first - 1]}" if (sequence and i_first > 0) else None
    RUNS.mkdir(parents=True, exist_ok=True)
    marker = RUNS / f"DONE_production{prefix}"
    marker.unlink(missing_ok=True)
    for lv in levels:
        i = num["mesh"]["levels"].index(lv)
        mesh = mesh_wedge(G, num, i)
        name = f"{prefix}{lv}"
        case = build(name, problem, num, mesh)
        if sequence and prev is not None:
            foam.map_fields(case, prev)
        live = monitor.LiveStop(case, decide)
        t0 = time.time()
        wall = foam.run(num["solver"], case, watch=live, watch_every=5.0)
        rec = collect(case, name, G, num, mesh, wall, 1, live)
        print(f"{name}: {rec['cells']} cells, {rec['iterations']} iterations, {wall:.1f} s, health {rec['health']['status']}, "
              f"stop: {(rec['live_stop'] or {}).get('decision', {}) and rec['live_stop']['decision']['reason']}", flush=True)
        prev = case
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def do_buoyancy(which=("g0", "g1")):
    problem, num = load()
    G = given(problem)
    b = num["buoyancy"]
    mesh = mesh_half3d(G, num)
    decide = live_decider(num)
    marker = RUNS / "DONE_buoyancy"
    marker.unlink(missing_ok=True)
    prev = None
    for tag in which:
        g_y = 0.0 if tag == "g0" else -G["g"]
        name = f"B3D_{tag}"
        case = build(name, problem, num, mesh, cores=b["cores"], g_y=g_y)
        if prev is not None:
            foam.map_fields(case, prev)                      # start the buoyant run from the g = 0 answer
        foam.run("decomposePar -force", case, "decomposePar")
        live = monitor.LiveStop(case, decide)
        wall = foam.run(f"mpirun -np {b['cores']} {num['solver']} -parallel", case, log_name=num["solver"], watch=live,
                        watch_every=5.0)
        foam.run("reconstructPar -latestTime", case, "reconstructPar")
        rec = collect(case, name, G, num, mesh, wall, b["cores"], live)
        print(f"{name}: {rec['cells']} cells, {rec['iterations']} iterations, {wall:.1f} s, health {rec['health']['status']}",
              flush=True)
        prev = case
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    if mode == "preflight":
        do_preflight()
    elif mode == "production":
        _, n = load()
        do_production(sys.argv[2:] or n["mesh"]["levels"])
    elif mode == "buoyancy":
        do_buoyancy(tuple(sys.argv[2:]) or ("g0", "g1"))
    elif mode == "collect":                                 # re-read a finished wedge run: collect <name> <level>
        p, n = load()
        Gv = given(p)
        nm, lv = sys.argv[2], sys.argv[3]
        cs = run_root() / nm
        ms = mesh_wedge(Gv, n, n["mesh"]["levels"].index(lv))
        perf_j = json.loads((cs / f"perf.{n['solver']}.json").read_text())

        class _Live:                                        # the live-stop record left in the case
            def summary(self):
                return json.loads((cs / "live_stop.json").read_text())
        r = collect(cs, nm, Gv, n, ms, perf_j["wall_s"], 1, _Live())
        print(nm, r["iterations"], "iterations")
    elif mode == "mesh":                                    # print mesh data only
        p, n = load()
        Gv = given(p)
        for i, lv in enumerate(n["mesh"]["levels"]):
            ms = mesh_wedge(Gv, n, i)
            print(lv, ms["cells"], {k: f"{v:.4g}" for k, v in ms["axial"].items()})
    else:
        raise SystemExit(f"unknown mode {mode}")
