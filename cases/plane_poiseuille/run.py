"""Build and run the plane Poiseuille case on every mesh in numerics.yaml; save raw outputs.

Usage (from the repo root, inside WSL):  python -m cases.plane_poiseuille.run
Output: results/plane_poiseuille/raw.json
"""
import json
import platform
from pathlib import Path

import numpy as np
import yaml

from cfdlib import foam, monitor

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name


def main():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text())
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text())
    g = problem["given"]
    H, rho, nu, dpdx = (g[k]["value"] for k in ("H", "rho", "nu", "dpdx"))
    m, conv = numerics["mesh"], numerics["convergence"]
    Lx = m["Lx_over_H"] * H

    run_root = Path(foam.foam_env("FOAM_RUN")) / "agentic" / HERE.name

    def build(Ny):
        params = dict(H=H, Lx=Lx, z0=-Lx / 2, z1=Lx / 2, Nx=m["Nx"], Ny=Ny, nu=nu,
                      accel=-dpdx / rho, max_iterations=conv["iterations"])
        case = foam.render_case(HERE / "template", run_root / f"Ny{Ny}", params)
        foam.run("blockMesh", case)
        foam.run("checkMesh", case)
        return case

    # ---- pre-flight: short runs on the coarsest and finest meshes before production -------------
    preflight = []
    for Ny in (m["Ny"][0], m["Ny"][-1]):
        case = build(Ny)
        pf = monitor.preflight(case, numerics["solver"], end_time_short=2, end_time_full=conv["iterations"], timeout=120)
        pf["Ny"] = Ny
        preflight.append(pf)
        print(f"preflight Ny={Ny}: {pf['status']}, est. full run {pf['est_full_runtime']}")
        if pf["status"] == "fail":
            raise SystemExit("Pre-flight failed; fix the setup before production runs:\n" + json.dumps(pf["checks"], indent=1))

    runs = []
    for Ny in m["Ny"]:
        case = build(Ny)
        wall = {}
        wall["solver"] = foam.run(numerics["solver"], case)
        foam.run("postProcess -func writeCellCentres -latestTime", case, "writeCellCentres")

        t = foam.time_dirs(case)[-1]
        Cy = foam.read_internal_field(t / "C")[:, 1]
        Ux = foam.read_internal_field(t / "U")[:, 0]
        order = np.argsort(Cy)

        wss = foam.read_dat(next((case / "postProcessing/wallShearStress").rglob("wallShearStress.dat")))
        last_t = wss[-1][0]
        # Kinematic wall shear stress vector (tau/rho); x-component magnitude, times rho -> Pa
        tau = {row[1]: rho * abs(row[2][0]) for row in wss if row[0] == last_t}   # row: time patch min max
        flow = foam.read_dat(next((case / "postProcessing/flowRate").rglob("surfaceFieldValue.dat")))
        q_hist = np.array([row[1] for row in flow]) / Lx                           # m^3/s per m of depth
        q = float(q_hist[-1])
        # Iterative convergence, judged physically: has the flow rate stopped changing?
        k = max(1, len(q_hist) // 10)
        q_change = abs(q_hist[-1] - q_hist[-1 - k]) / abs(q_hist[-1])
        force_balance = abs(sum(tau.values()) - (-dpdx) * H) / ((-dpdx) * H)
        checkmesh = (case / "log.checkMesh").read_text()

        runs.append(dict(
            Ny=Ny, cells=Ny * m["Nx"], h=H / Ny, case_dir=str(case),
            iterations=int(float(t.name)), q_change=q_change, force_balance=force_balance,
            converged=bool(q_change < conv["q_change_tol"] and force_balance < conv["force_balance_tol"]),
            final_residuals={k: v[-1] for k, v in foam.residual_history(case / f"log.{numerics['solver']}").items()},
            residual_history=foam.residual_history(case / f"log.{numerics['solver']}"),
            mesh_ok="Mesh OK." in checkmesh,
            y=Cy[order].tolist(), u=Ux[order].tolist(),
            tau_w=tau, q=q, wall_time=wall,
            health=monitor.assess(case / f"log.{numerics['solver']}"),
        ))
        print(f"Ny={Ny:4d}  iters={runs[-1]['iterations']:5d}  converged={runs[-1]['converged']}  "
              f"q={q:.9e}  dq={q_change:.1e}  forceBal={force_balance:.1e}  solver {wall['solver']:.2f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    meta = dict(openfoam=foam.foam_env("WM_PROJECT_VERSION"), solver=numerics["solver"],
                python=platform.python_version(), host=platform.node(),
                cpu=_cpu_name(), run_root=str(run_root))
    (OUT / "raw.json").write_text(json.dumps(dict(meta=meta, preflight=preflight, runs=runs), indent=1))
    print(f"Saved {OUT / 'raw.json'}")


def _cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


if __name__ == "__main__":
    main()
