"""Verification ladder, rung 7b: lid-driven cavity at Re = 100 on the GPU solver, against the benchmark of
Ghia, Ghia & Shin (1982) and against our own OpenFOAM results of case 2 (same Re, same grids).

    python -m gpucfd.compare.cavity [imex] 32 64 128 [256]      (inside WSL, from the project root)

Unit square, lid (y = 1) moving at U = 1 in +x, nu = 0.01. Time-marched from rest until the velocity
changes by less than 1e-7 (max norm) over one time unit. Results in gpucfd/compare/results/cavity.json.
Pass rule (fixed before running, the same quantity case 2 reported): the centreline velocities agree
with Ghia within 0.01 U on the finest grid, and the GPU and OpenFOAM solutions on the same grid
converge toward each other with refinement.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from gpucfd import gcfd

ROOT = Path(__file__).resolve().parents[2]
NU, U = 0.01, 1.0
OUT = Path(__file__).parent / "results"


def ghia():
    r = yaml.safe_load((ROOT / "cases/lid_driven_cavity_re100/reference/ghia1982_re100.yaml").read_text())
    u = np.array(r["u_vertical_centreline"]["rows"], float)       # columns y, u
    v = np.array(r["v_horizontal_centreline"]["rows"], float)     # columns x, v
    return u[np.argsort(u[:, 0])], v[np.argsort(v[:, 0])]


def gpu_cavity(N, ptol=1e-4, tmax=80.0, settle=1e-7, scheme="rk3", cfl=0.8):
    s = gcfd.NavierStokes((N, N, 1), (1.0, 1.0, 1.0), NU, walls=(True, True, False), wall_velocity={"y+": (U, 0, 0)},
                          scheme=scheme)
    h = 1.0 / N
    per_unit = int(np.ceil(1.0 / s.max_dt(U, cfl)))
    t, loop, change, hist = 0.0, 0.0, np.inf, []
    old = [np.zeros(s.shape), np.zeros(s.shape)]
    t0 = time.time()
    while t < tmax and change > settle:
        st = s.run(1.0 / per_unit, per_unit, ptol)
        loop += st["ms"] / 1e3
        t += 1.0
        new = s.get()
        change = max(float(np.max(np.abs(a - b))) for a, b in zip(new, old))
        hist.append(change)
        old = new
    u, v = old[0][0], old[1][0]                       # 2D arrays [j, i]
    # u on the vertical centreline x = 0.5: x-faces i = N/2; positions y = (j + 1/2) h, plus the walls
    yu = np.concatenate([[0.0], (np.arange(N) + 0.5) * h, [1.0]])
    ul = np.concatenate([[0.0], u[:, N // 2], [U]])
    # v on the horizontal centreline y = 0.5: y-faces j = N/2; positions x = (i + 1/2) h
    xv = np.concatenate([[0.0], (np.arange(N) + 0.5) * h, [1.0]])
    vl = np.concatenate([[0.0], v[N // 2, :], [0.0]])
    s.close()
    return dict(N=N, scheme=scheme, t_steady=t, steps=int(t * per_unit), loop_s=loop, wall_s=time.time() - t0, last_change=change,
                max_div=st["max_div"], line_u=(yu, ul), line_v=(xv, vl))


def openfoam_lines(N):
    """Case 2's OpenFOAM solution on N x N (cell centres, cavity side 0.1 m, lid 1 m/s)."""
    f = ROOT / f"results/lid_driven_cavity_re100/runs/N{N}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    ux, uy = d["ux"], d["uy"]                         # [j, i], cell centres
    c = (np.arange(N) + 0.5) / N
    ul = np.concatenate([[0.0], 0.5 * (ux[:, N // 2 - 1] + ux[:, N // 2]), [U]])     # average of the two middle columns
    vl = np.concatenate([[0.0], 0.5 * (uy[N // 2 - 1, :] + uy[N // 2, :]), [0.0]])
    pos = np.concatenate([[0.0], c, [1.0]])
    return (pos, ul), (pos, vl), json.loads(f.with_suffix(".json").read_text()).get("wall_time")


def at_ghia(line, pts):
    return np.interp(pts, line[0], line[1])


if __name__ == "__main__":
    scheme = "imex" if "imex" in sys.argv[1:] else "rk3"
    Ns = [int(a) for a in sys.argv[1:] if a.isdigit()] or [32, 64, 128]
    gu, gv = ghia()
    inner_u, inner_v = gu[1:-1], gv[1:-1]            # Ghia's points without the wall values
    rows, lines = [], {}
    for N in Ns:
        r = gpu_cavity(N, scheme=scheme)
        du = np.abs(at_ghia(r["line_u"], inner_u[:, 0]) - inner_u[:, 1])
        dv = np.abs(at_ghia(r["line_v"], inner_v[:, 0]) - inner_v[:, 1])
        row = dict(N=N, scheme=scheme, t_steady=r["t_steady"], steps=r["steps"], loop_s=r["loop_s"], wall_s=r["wall_s"],
                   last_change=r["last_change"], max_div=r["max_div"],
                   gpu_vs_ghia_u=float(du.max()), gpu_vs_ghia_v=float(dv.max()))
        of = openfoam_lines(N)
        if of:
            (ou, ov, of_wall) = of
            row["openfoam_vs_ghia_u"] = float(np.abs(at_ghia(ou, inner_u[:, 0]) - inner_u[:, 1]).max())
            row["openfoam_vs_ghia_v"] = float(np.abs(at_ghia(ov, inner_v[:, 0]) - inner_v[:, 1]).max())
            fine = np.linspace(0, 1, 401)
            row["gpu_vs_openfoam"] = float(max(np.abs(at_ghia(r["line_u"], fine) - at_ghia(ou, fine)).max(),
                                               np.abs(at_ghia(r["line_v"], fine) - at_ghia(ov, fine)).max()))
            row["openfoam_case2_wall_s"] = of_wall
        lines[N] = dict(u=[a.tolist() for a in r["line_u"]], v=[a.tolist() for a in r["line_v"]])
        rows.append(row)
        print(f"N={N:4d} steady at t={r['t_steady']:.0f} ({r['steps']} steps, change {r['last_change']:.1e}), "
              f"loop {r['loop_s']:.1f} s | max diff from Ghia: GPU u {row['gpu_vs_ghia_u']:.4f} v {row['gpu_vs_ghia_v']:.4f}"
              + (f" | OpenFOAM u {row['openfoam_vs_ghia_u']:.4f} v {row['openfoam_vs_ghia_v']:.4f}"
                 f" | GPU vs OpenFOAM {row['gpu_vs_openfoam']:.4f}" if of else ""), flush=True)
    # self-convergence of the GPU solution: difference between successive grids on the centrelines
    fine = np.linspace(0, 1, 401)
    for a, b in zip(Ns[:-1], Ns[1:]):
        d = max(np.abs(at_ghia(lines[a]["u"], fine) - at_ghia(lines[b]["u"], fine)).max(),
                np.abs(at_ghia(lines[a]["v"], fine) - at_ghia(lines[b]["v"], fine)).max())
        print(f"GPU centrelines, N={a} vs N={b}: max difference {d:.5f} U")
        next(r for r in rows if r["N"] == b)["change_from_coarser"] = float(d)
    OUT.mkdir(exist_ok=True)
    path = OUT / "cavity.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    for r in rows:
        old[str(r["N"]) + ("" if scheme == "rk3" else "_imex")] = r
    if scheme == "rk3":
        old["lines"] = {**old.get("lines", {}), **{str(k): v for k, v in lines.items()}}
    path.write_text(json.dumps(old, indent=1))
