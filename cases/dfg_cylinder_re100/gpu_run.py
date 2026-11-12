"""The DFG 2D-2 cylinder case (Re = 100) on gpucfd: the same problem and protocol as run.py's OpenFOAM levels.

Usage (repo root, inside WSL):
  python -m cases.dfg_cylinder_re100.gpu_run preflight     # short runs on G1 (from rest) and G3 (from G1): cost, health
  python -m cases.dfg_cylinder_re100.gpu_run production [NAME ...]  # G1 0..15 s from rest; G2, G2_dtx0.5, G3 (run_table)
  python -m cases.dfg_cylinder_re100.gpu_run ptol          # pressure-tolerance study on G1 from its developed field
Outputs: results/dfg_cylinder_re100/runs/<G..>.json + .npz (cfdlib.gpu.write_run), gpu_preflight.json;
final velocity fields in ~/gpucfd_runs/dfg_cylinder_re100/ (outside git, like OpenFOAM's run directories).

npz keys: t, Fcyl (force per unit density and unit depth on the cylinder, x y z), probe_v (v at the wake
probe), q_in / q_out (volume flow per unit depth through inlet / outlet, out of the domain positive), umax.
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from cfdlib import gpu
from . import signals as S

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS_DIR = OUT / "runs"
FIELDS = Path.home() / "gpucfd_runs" / HERE.name


def load():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    return problem, numerics


def make(level, problem, numerics):
    from gpucfd import gcfd, bodies as B
    g = {k: v["value"] for k, v in problem["given"].items()}
    G = numerics["gpucfd"]
    k = G["levels"].index(level)
    n = (G["n_G1"][0] * 2 ** k, G["n_G1"][1] * 2 ** k, 1)
    H, Um = g["H"], g["Um"]

    def inflow(X, Y, Z):
        return (4 * Um * Y * (H - Y) / H ** 2, 0 * X, 0 * X)
    bnd = {"x-": ("inflow", inflow), "x+": "outflow", "y-": "wall", "y+": "wall"}
    s = gcfd.NavierStokes(n, (g["L"], H, 1.0), g["nu"], scheme=G["scheme"], boundaries=bnd,
                          bodies=[B.Cylinder((g["xc"], g["yc"]), g["D"] / 2)])
    return s, inflow


def from_rest(s, inflow):
    """Start field: the inflow profile everywhere in the fluid (zero in the body), projected."""
    q = []
    for c in range(2):
        X, Y, Z = s.face_coords(c)
        q.append(np.where(s.fluid[c], inflow(X, Y, Z)[c], 0.0))
    s.set(q)
    return s.project()


def observer(problem, numerics):
    g = {k: v["value"] for k, v in problem["given"].items()}
    G = numerics["gpucfd"]
    box = (tuple(G["force_box"][0]), tuple(G["force_box"][1]), (0.0, 1.0))
    px, py = numerics["probes"][0]

    def probe_v(s, q):
        hx, hy = s.length[0] / s.n[0], s.length[1] / s.n[1]   # v lives at x = (i + 1/2) hx, y = j hy
        fx, fy = px / hx - 0.5, py / hy
        i, j = int(np.floor(fx)), int(np.floor(fy))
        a, b = fx - i, fy - j
        v = q[1][0]
        return (1 - a) * (1 - b) * v[j, i] + a * (1 - b) * v[j, i + 1] + (1 - a) * b * v[j + 1, i] + a * b * v[j + 1, i + 1]

    def observe(s):
        q = s.get()
        F = s.forces(box)
        return dict(Fcyl=F, cd=2 * F[0] / ((2 * g["Um"] / 3) ** 2 * g["D"]), cl=2 * F[1] / ((2 * g["Um"] / 3) ** 2 * g["D"]),
                    probe_v=probe_v(s, q), q_in=s.flux("x-"), q_out=s.flux("x+"),
                    umax=float(np.sqrt(np.max(q[0][0] ** 2) + np.max(q[1][0] ** 2))))
    return observe


def periodic_watch(numerics):
    """Live check: the case's periodicity rule over the last cycles_measured lift cycles."""
    acc, n = numerics["acceptance"], numerics["time"]["cycles_measured"]

    def watch(t, d):
        tt = d["t"]
        if tt[-1] - tt[0] < 1.5:
            return None
        cyc = S.cycles(tt, d["cd"], d["cl"], window=min(3.0, tt[-1] - tt[0]))
        if len(cyc) < n + 1:
            return None
        sp = S.summary(cyc, n)["spread"]
        ok = (max(sp["cl_max"], sp["cl_min"]) < acc["periodic_cl_peak"] and max(sp["cd_max"], sp["cd_min"]) < acc["periodic_cd_peak"]
              and sp["period"] / cyc[-1]["period"] < acc["periodic_period_rel"])
        return f"periodic (cl spread {max(sp['cl_max'], sp['cl_min']):.1e})" if ok else None
    return watch


def save_field(s, name):
    FIELDS.mkdir(parents=True, exist_ok=True)
    q = s.get()
    np.savez_compressed(FIELDS / f"{name}.npz", u=q[0], v=q[1])


def load_field(name, level, problem, numerics):
    """A simulation on `level` holding the saved last field of run `name`."""
    s, _ = make(level, problem, numerics)
    d = np.load(FIELDS / f"{name}.npz")
    s.set([d["u"], d["v"]])
    s.project()
    s._name = name
    return s


def one(name, level, problem, numerics, t0, t1, src=None, factor=1.0, ptol=None, runs_dir=RUNS_DIR, log=print):
    """Run one level from rest (src None) or from the last field of simulation src; returns (s, record)."""
    from gpucfd import steady
    G = numerics["gpucfd"]
    ptol = G["pressure_tol"] if ptol is None else ptol
    s, inflow = make(level, problem, numerics)
    div0 = from_rest(s, inflow) if src is None else steady.prolong(src, s)
    nu = problem["given"]["nu"]["value"]
    dt = s.max_dt(G["u_dt"], G["cfl"]) * factor
    h = min(s.length[d] / s.n[d] for d in range(2))
    log(f"{name}: grid {s.n[0]} x {s.n[1]} ({int(np.prod(s.n))} cells), D/h {0.1 / h:.1f}, dt {dt:.3g} "
        f"(convective {G['cfl'] * h / G['u_dt']:.3g}, viscous {0.15 * h * h / nu:.3g}), t {t0:g} -> {t1:g}, "
        f"start div {div0:.1e}", flush=True)
    res = gpu.run_transient(s, dt, t0, t1, G["sample_dt"], observer(problem, numerics), u_ref=problem["given"]["Um"]["value"],
                            log=lambda m: print(m, flush=True), tag=f"{name} ", watch=periodic_watch(numerics),
                            ptol=ptol, cfl_limit=1.0)
    rec = gpu.write_run(runs_dir, name, res, dict(level=level, n=list(s.n[:2]), cells=int(np.prod(s.n)), h=h, D_h=0.1 / h,
                                                  source=getattr(src, "_name", None), dt_factor=factor, pressure_tol=ptol,
                                                  scheme=G["scheme"], sample_dt=G["sample_dt"], force_per_unit_depth=True))
    save_field(s, name)
    s._name = name
    p = rec["perf"]
    log(f"{name}: done, health {rec['health']['status']}, GPU {p['gpu_s']:.1f} s ({p['ms_per_step']:.2f} ms/step, "
        f"{p['steps']} steps, {p['poisson_its']:.1f} pressure iterations per step), observe {p['observe_s']:.1f} s, "
        f"wall {p['wall_s']:.1f} s; periodic from {rec['watch_first']['t'] if rec['watch_first'] else 'never'}", flush=True)
    return s, rec


def run_table(numerics):
    """name -> (level, source run, t0, t1, dt factor): the same protocol as run.py's OpenFOAM levels."""
    t, G = numerics["time"], numerics["gpucfd"]
    table, prev, t0 = {}, None, 0.0
    for L in G["levels"]:
        t1 = t0 + (t["develop_end"] if prev is None else t["run_length"])
        table[L] = (L, prev, t0, t1, 1.0)
        if L == G["dt_check"]["level"]:
            f = G["dt_check"]["factor"]
            table[f"{L}_dtx{f:g}"] = (L, prev, t0, t1, f)
        prev, t0 = L, t1
    return table


def production(names):
    """Named runs in the order given; each starts from its source run's saved last field."""
    problem, numerics = load()
    table = run_table(numerics)
    for nm in names:
        level, src, t0, t1, f = table[nm]
        source = load_field(src, table[src][0], problem, numerics) if src else None
        s, _ = one(nm, level, problem, numerics, t0, t1, src=source, factor=f)
        s.close()
        if source is not None:
            source.close()


def extend(name, max_length=12.0):
    """Continue run `name` from its saved last field, with the same time step, until the live periodicity check
    passes (at most max_length seconds more), and merge the continuation into its records: the same result as
    one longer run. Needed when the coarser start field is far from the finer level's answer (G1's drag is 16 %
    low), so the fixed 3 s of the OpenFOAM protocol do not reach the periodic state."""
    problem, numerics = load()
    G = numerics["gpucfd"]
    level = run_table(numerics)[name][0]
    rec0 = json.loads((RUNS_DIR / f"{name}.json").read_text())
    d0 = dict(np.load(RUNS_DIR / f"{name}.npz"))
    s = load_field(name, level, problem, numerics)
    dt0, t0 = rec0["dt"], rec0["end_time"]
    n = int(round(max_length / dt0))
    print(f"{name}: continuing from t {t0:g} with dt {dt0:.4g}, until periodic (at most {max_length:g} s)", flush=True)
    res = gpu.run_transient(s, dt0, t0, t0 + n * dt0, G["sample_dt"], observer(problem, numerics),
                            u_ref=problem["given"]["Um"]["value"], log=lambda m: print(m, flush=True), tag=f"{name}+ ",
                            watch=periodic_watch(numerics), stop_on_watch=True, ptol=rec0["pressure_tol"], cfl_limit=1.0)
    p0, p1 = rec0["perf"], res["perf"]
    steps = p0["steps"] + p1["steps"]
    perf = dict(p0, gpu_s=p0["gpu_s"] + p1["gpu_s"], wall_s=p0["wall_s"] + p1["wall_s"], observe_s=p0["observe_s"] + p1["observe_s"],
                steps=steps, poisson_its=(p0["poisson_its"] * p0["steps"] + p1["poisson_its"] * p1["steps"]) / steps)
    perf["ms_per_step"] = 1000 * perf["gpu_s"] / steps
    perf["us_per_cell_step"] = 1e6 * perf["gpu_s"] / (steps * perf["cells"])
    checks = []
    for c0, c1 in zip(rec0["health"]["checks"], res["health"]["checks"]):
        worse = c1 if ["ok", "warn", "fail"].index(c1["status"]) > ["ok", "warn", "fail"].index(c0["status"]) else c0
        checks.append(dict(worse, value=f"{c0['value']} / {c1['value']}"))
    status = max((c["status"] for c in checks), key=["ok", "warn", "fail"].index)
    merged = dict(t=np.concatenate([d0["t"], res["t"][1:]]),
                  series={k: np.concatenate([d0[k], res["series"][k][1:]]) for k in res["series"]},
                  perf=perf, health=dict(status=status, checks=checks, steps=steps, last_time=float(res["t"][-1])),
                  dt=dt0, watch_first=res["watch_first"])
    meta = {k: rec0[k] for k in ("level", "n", "cells", "h", "D_h", "source", "dt_factor", "pressure_tol", "scheme", "sample_dt",
                                 "force_per_unit_depth")}
    meta["segments"] = rec0.get("segments", [[rec0["start_time"], t0]]) + [[t0, float(res["t"][-1])]]
    rec = gpu.write_run(RUNS_DIR, name, merged, meta)
    save_field(s, name)
    s.close()
    print(f"{name}: now t {rec['start_time']:g} -> {rec['end_time']:g}, GPU {perf['gpu_s']:.1f} s, wall {perf['wall_s']:.1f} s, "
          f"health {status}, periodic at {res['watch_first']['t'] if res['watch_first'] else 'never'}", flush=True)


def tune_ptol(tols=(1e-10, 1e-8, 1e-6, 1e-4), length=1.0):
    """Pressure tolerance study on G1 from its developed field (t = develop_end): forces over `length`
    seconds for each tolerance, against the strictest. Rule (numerics.yaml): the loosest tolerance with
    max |delta cd| and |delta cl| below 1e-4. Writes gpu_tuning/ptol.json."""
    problem, numerics = load()
    G = numerics["gpucfd"]
    L = G["levels"][0]
    t0 = numerics["time"]["develop_end"]
    tdir = OUT / "gpu_tuning"
    out = []
    for tol in tols:
        src = load_field(L, L, problem, numerics)
        s, rec = one(f"ptol_{tol:g}", L, problem, numerics, t0, t0 + length, src=src, ptol=tol, runs_dir=tdir)
        s.close()
        src.close()
    ptol_summary(tols)


def ptol_summary(tols=(1e-10, 1e-8, 1e-6, 1e-4)):
    """Compare the tolerance runs in gpu_tuning/ with the strictest; writes gpu_tuning/ptol.json."""
    tdir = OUT / "gpu_tuning"
    recs = [json.loads((tdir / f"ptol_{tol:g}.json").read_text()) for tol in tols]
    data = [np.load(tdir / f"ptol_{tol:g}.npz") for tol in tols]
    out = []
    for tol, rec, d in zip(tols, recs, data):
        r = dict(tol=tol, perf=rec["perf"], health=rec["health"]["status"],
                 d_cd=float(np.max(np.abs(d["cd"] - data[0]["cd"]))), d_cl=float(np.max(np.abs(d["cl"] - data[0]["cl"]))),
                 div=next(c["value"] for c in rec["health"]["checks"] if c["name"].startswith("Divergence")))
        out.append(r)
        print(f"ptol {tol:g}: max |d cd| {r['d_cd']:.2e}, max |d cl| {r['d_cl']:.2e}, {r['perf']['ms_per_step']:.2f} ms/step, "
              f"{r['perf']['poisson_its']:.1f} pressure iterations per step, divergence {r['div']}", flush=True)
    (tdir / "ptol.json").write_text(json.dumps(out, indent=1, default=float))


def preflight():
    """G1 from rest for 0.5 s and G3 from G1 for 0.05 s: stability, outputs, cost per step and per sample."""
    problem, numerics = load()
    G = numerics["gpucfd"]
    out = []
    s1, r1 = one("pf_G1", G["levels"][0], problem, numerics, 0.0, 0.5)
    s3, r3 = one("pf_G3", G["levels"][-1], problem, numerics, 0.5, 0.55, src=s1)
    for r in (r1, r3):
        p = r["perf"]
        n_samp = len(np.load(RUNS_DIR / f"{r['name']}.npz")["t"])
        r["observe_ms_per_sample"] = 1000 * p["observe_s"] / n_samp
        out.append(r)
        print(f"{r['name']}: {r['health']['status']}, {p['ms_per_step']:.2f} ms/step, {r['observe_ms_per_sample']:.1f} ms per sample, "
              + "; ".join(f"{c['name']}: {c['value']}" for c in r["health"]["checks"]))
    for nm in ("pf_G1", "pf_G3"):  # pre-flight records are not runs of the study
        for ext in ("json", "npz"):
            (RUNS_DIR / f"{nm}.{ext}").unlink()
    (OUT / "gpu_preflight.json").write_text(json.dumps(out, indent=1, default=float))
    s1.close()
    s3.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    if mode == "production":
        production(sys.argv[2:] or list(run_table(load()[1])))
    elif mode == "ptol":
        tune_ptol()
    elif mode == "extend":
        for nm in sys.argv[2:]:
            extend(nm)
    elif mode == "ptol_summary":
        ptol_summary()
    elif mode == "preflight":
        preflight()
    else:
        raise SystemExit(f"unknown mode {mode}")
