"""Build and run the lid-driven cavity (Re = 100) on the meshes in numerics.yaml; save raw outputs.

Usage (repo root, inside WSL):
  python -m cases.lid_driven_cavity_re100.run preflight          # short runs, coarsest + finest mesh
  python -m cases.lid_driven_cavity_re100.run production [N ...]  # full runs (default: every mesh)
Outputs: results/lid_driven_cavity_re100/preflight.json, runs/N<N>.json + runs/N<N>.npz
"""
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import yaml

from cfdlib import foam, monitor

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"
SOLVER_FIELDS = ("Ux", "Uy", "p")


def load():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    return problem, numerics


def run_root():
    return Path(foam.foam_env("FOAM_RUN")) / "agentic" / HERE.name


def params_for(N, problem, numerics, max_iterations=None, div_U=None):
    g = problem["given"]
    L, U, nu = (g[k]["value"] for k in ("L", "U", "nu"))
    c = numerics["convergence"]
    return dict(
        L=L, U=U, nu=nu, N=N, dz=numerics["mesh"]["dz_over_L"] * L,
        max_iterations=max_iterations or c["max_iterations"], write_every=c["write_every"],
        res_U=c["residual_U"], res_p=c["residual_p"], relax_U=numerics["relaxation"]["U"],
        div_U=div_U or numerics["schemes"]["div_U"], p_relTol=numerics["linear_solvers"]["p_relTol"],
        probes=[(px * L, py * L) for px, py in numerics["probes"]],
    )


def build(N, problem, numerics, max_iterations=None, prefix="N", div_U=None):
    """Render and mesh one case. Pre-flight uses its own directories (prefix 'pf'), so re-running
    it never deletes a production case."""
    case = foam.render_case(HERE / "template", run_root() / f"{prefix}{N}", params_for(N, problem, numerics, max_iterations, div_U))
    foam.run("blockMesh", case)
    foam.run("checkMesh", case)
    return case


def residual_decay_rate(log_path):
    """Mean decades of residual reduction per iteration over the second half of a log (per field)."""
    hist = foam.residual_history(log_path, SOLVER_FIELDS)
    rates = {}
    for f, h in hist.items():
        h = np.array([x for x in h if x > 0])
        if len(h) < 20:
            continue
        k = len(h) // 2
        n = np.arange(len(h))[k:]
        slope = np.polyfit(n, np.log10(h[k:]), 1)[0]
        rates[f] = dict(slope=float(slope), last=float(h[-1]))
    return rates


def do_preflight():
    problem, numerics = load()
    Ns, n_short = numerics["mesh"]["N"], numerics["preflight"]["iterations"]
    c = numerics["convergence"]
    out = []
    for N in (Ns[0], Ns[-1]):
        case = build(N, problem, numerics, prefix="pf")
        pf = monitor.preflight(case, numerics["solver"], end_time_short=n_short,
                               end_time_full=c["max_iterations"], timeout=900)
        # Steady runs stop on residualControl, so the useful projection is: how many iterations until
        # the residual targets are met, at the decay rate seen in the second half of the short run?
        rates = residual_decay_rate(Path(pf["case"]) / f"log.{numerics['solver']}")
        target = {"Ux": c["residual_U"], "Uy": c["residual_U"], "p": c["residual_p"]}
        need = []
        for f, r in rates.items():
            if r["slope"] < 0:
                need.append(n_short + math.log10(target[f] / r["last"]) / r["slope"])
            else:
                need.append(float("inf"))
        it_est = max(need) if need else float("inf")
        per_it = pf["wall_time"] / n_short if pf.get("wall_time") else None
        pf.update(N=N, cells=N * N, decay=rates, iterations_to_converge=it_est,
                  seconds_per_iteration=per_it,
                  est_converged_runtime=(per_it * it_est) if (per_it and math.isfinite(it_est)) else None)
        out.append(pf)
        print(f"preflight N={N}: {pf['status']}; {per_it:.4g} s/iteration; residual decay "
              + ", ".join(f"{f} {r['slope']:.3g} dec/it (last {r['last']:.2e})" for f, r in rates.items())
              + f"; est. {it_est:.0f} iterations, {pf['est_converged_runtime'] or float('nan'):.0f} s")
        for ch in pf["checks"]:
            print(f"   {ch['status']:4s} {ch['name']}: {ch['value']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(out, indent=1, default=str))
    if any(p["status"] == "fail" for p in out):
        raise SystemExit("Pre-flight FAILED: fix the setup before production runs.")


def read_points(path):
    """constant/polyMesh/points (ASCII) as an (n, 3) array."""
    text = Path(path).read_text()
    i = text.index("(", text.index("\n", text.index("}")))       # list start after the FoamFile header
    body = text[i + 1:text.rindex(")")]
    return np.array(body.replace("(", " ").replace(")", " ").split(), dtype=float).reshape(-1, 3)


def do_production(Ns, prefix="N", div_U=None):
    problem, numerics = load()
    L = problem["given"]["L"]["value"]
    U = problem["given"]["U"]["value"]
    c = numerics["convergence"]
    RUNS.mkdir(parents=True, exist_ok=True)
    for N in Ns:
        case = build(N, problem, numerics, prefix=prefix, div_U=div_U)
        wall = foam.run(numerics["solver"], case)
        log = case / f"log.{numerics['solver']}"
        foam.run("postProcess -func writeCellCentres -latestTime", case, "writeCellCentres")
        foam.run("postProcess -func streamFunction -latestTime", case, "streamFunction")
        t = foam.time_dirs(case)[-1]
        C = foam.read_internal_field(t / "C")
        Uc = foam.read_internal_field(t / "U")
        p = foam.read_internal_field(t / "p")
        # Arrange cells on the N x N grid by index (never by exact coordinates).
        h = L / N
        i = np.floor(C[:, 0] / h).astype(int)
        j = np.floor(C[:, 1] / h).astype(int)
        assert i.min() == 0 and i.max() == N - 1 and j.min() == 0 and j.max() == N - 1
        grid = lambda a: _scatter(a, i, j, N)
        ux, uy, pp, xc, yc = grid(Uc[:, 0]), grid(Uc[:, 1]), grid(p), grid(C[:, 0]), grid(C[:, 1])
        # OpenFOAM's own stream function (point field), for a cross-check of the analysis' psi.
        pts = read_points(case / "constant/polyMesh/points")
        sf = foam.read_internal_field(t / "streamFunction")
        bottom = pts[:, 2] < 0.5 * numerics["mesh"]["dz_over_L"] * L          # one z-layer of points
        pi_ = np.rint(pts[bottom, 0] / h).astype(int)
        pj_ = np.rint(pts[bottom, 1] / h).astype(int)
        sf_grid = np.full((N + 1, N + 1), np.nan)
        sf_grid[pj_, pi_] = sf[bottom]

        # Physical steadiness: how much did each probed velocity change over the last 10 % of iterations?
        probe_rows = foam.read_dat(next((case / "postProcessing/probes").rglob("U")))
        it_hist = np.array([r[0] for r in probe_rows])
        vals = np.array([[comp for vec in r[1:] for comp in vec[:2]] for r in probe_rows])   # (n, 2*probes)
        k = max(1, len(it_hist) // 10)
        probe_change = float(np.max(np.abs(vals[-1] - vals[-1 - k])) / U)
        # Steadiness between the last two written fields (if more than one write happened)
        tds = [d for d in foam.time_dirs(case) if d.name != "0"]
        write_change = None
        if len(tds) >= 2 and (tds[-2] / "U").exists():
            U_prev = foam.read_internal_field(tds[-2] / "U")
            write_change = dict(from_iteration=float(tds[-2].name), to_iteration=float(tds[-1].name),
                                max_change=float(np.max(np.abs(Uc - U_prev)) / U))
        iters = int(float(t.name))
        health = monitor.assess(log)
        checkmesh = (case / "log.checkMesh").read_text()
        res = foam.residual_history(log, SOLVER_FIELDS)
        np.savez_compressed(RUNS / f"{prefix}{N}.npz", ux=ux, uy=uy, p=pp, xc=xc, yc=yc, sf=sf_grid,
                            probe_it=it_hist, probe_vals=vals)
        meta = dict(N=N, variant=prefix, div_U=div_U or numerics["schemes"]["div_U"], cells=N * N, h=h, case_dir=str(case), iterations=iters, wall_time=wall,
                    stopped_by_residualControl=iters < c["max_iterations"],
                    final_residuals={f: (v[-1] if v else None) for f, v in res.items()},
                    residual_history=res, probe_change=probe_change, probe_window=int(it_hist[-1] - it_hist[-1 - k]),
                    write_change=write_change, mesh_ok="Mesh OK." in checkmesh, health=health,
                    openfoam=foam.foam_env("WM_PROJECT_VERSION"), python=platform.python_version(),
                    cpu=_cpu_name(), run_root=str(run_root()))
        (RUNS / f"{prefix}{N}.json").write_text(json.dumps(meta, indent=1))
        print(f"N={N}: {iters} iterations, {wall:.1f} s, health {health['status']}, "
              f"probe change {probe_change:.2e} over {meta['probe_window']} its, final res "
              + ", ".join(f"{f} {v:.1e}" for f, v in meta["final_residuals"].items() if v is not None), flush=True)


def _scatter(a, i, j, N):
    g = np.full((N, N), np.nan)
    g[j, i] = a
    assert not np.isnan(g).any()
    return g                                    # g[j, i]: row j (y), column i (x)


def _cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    if mode == "preflight":
        do_preflight()
    elif mode == "production":
        _, num = load()
        do_production([int(a) for a in sys.argv[2:]] or num["mesh"]["N"])
    elif mode == "sensitivity":
        # Same mesh, central-difference convection: does the scheme change the mesh-converged answer?
        _, num = load()
        do_production([int(a) for a in sys.argv[2:]] or [128], prefix="lin", div_U=num["sensitivity"]["div_U"])
    else:
        raise SystemExit(f"unknown mode {mode}")
