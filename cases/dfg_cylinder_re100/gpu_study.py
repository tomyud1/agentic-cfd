"""gpucfd next to OpenFOAM on case 4 (DFG 2D-2, the cylinder shedding vortices at Re 100): accuracy against the
FeatFlow benchmark, cost, what limits gpucfd here, and the outcome against the target fixed in advance.
Renders results/dfg_cylinder_re100/gpu_study.html with cfdlib.study.

Usage (repo root, inside WSL):  python -m cases.dfg_cylinder_re100.gpu_study
No run is made and the GPU is not used: everything is read from recorded files in results/dfg_cylinder_re100/:
  analysis.json            OpenFOAM's levels M1..M4 (analyze.py)
  live_stop.json           OpenFOAM's wall time with the live stop (run.py live; replay of the recorded histories)
  gpu_compare.json         gpucfd with equal cells, G1..G3, judged by the case's rules (gpu_compare.py)
  gpu_clustered.json       gpucfd with cells clustered around the cylinder, C1..C3, and the check runs (gpu_clustered.py)
  gpu_imex.json            the same clustered grids with the implicit time scheme (IMEX), C1i..C3i, C3wi (gpu_imex.py)
  runs/<name>.json        one record per run (time step, steps, wall time, pressure tolerance, health)
  gpu_tuning/              pressure-tolerance checks per level and the first runs they replaced
Every number in the report is computed here from those files and the case's YAML files.

Names: M1..M4 OpenFOAM's grids fitted to the cylinder; G1..G3 gpucfd, equal cells; C1..C3 gpucfd, cells clustered
around the cylinder; C1s..C3s the same with the cylinder moved half a C3 cell; C2w the middle clustered grid with a
longer band of fine cells behind the cylinder; the same names with an i (C3i, C3wi) are runs with the implicit scheme.
"""
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

from cfdlib import insights, plots, report, study
from cfdlib.report import load_yaml, m, tex_num as tn, tex_fix as tf
from gpucfd import grids                      # face positions only (NumPy); no solver, no GPU
from .analyze import QKEYS, SYM, reference

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / HERE.name
RUNS = OUT / "runs"
TUNE = OUT / "gpu_tuning"
CASE = HERE.name
# The pressure-tolerance rule of numerics.yaml (gpucfd.pressure_tol): the loosest tolerance whose drag and lift
# stay within this distance of the strictest run. A rule, fixed before the tolerance checks ran.
PTOL_RULE = 1.0e-4
# What was expected of the implicit scheme's three-grid study before it was built (seconds, low and high; written in
# numerics.yaml and gpucfd/DESIGN.md before the runs). A statement made in advance, not a result: it is only compared.
IMEX_EXPECTED_STUDY = (550.0, 800.0)
NAME = dict(cd_max="Drag maximum", cd_min="Drag minimum", cd_mean="Mean drag", cd_amp="Drag swing",
            cl_max="Lift maximum", cl_min="Lift minimum", cl_mean="Mean lift", cl_amp="Lift swing (maximum minus minimum)",
            St="Shedding frequency (Strouhal number)")
FAM = dict(of="OpenFOAM, grid fitted to the cylinder", uni="gpucfd, equal cells", clu="gpucfd, clustered cells",
           shift="gpucfd, clustered, cylinder moved half a cell", wide="gpucfd, clustered, longer fine band")
TFAM = dict(of="OpenFOAM, fitted grid", uni="gpucfd, equal cells", clu="gpucfd, clustered", shift="gpucfd, clustered, moved",
            wide="gpucfd, clustered, long band")       # the same, short, for table cells
SHORT = dict(NAME, cl_amp="Lift swing", St="Shedding frequency")


def nw(text):
    """A table cell that must stay on one line."""
    return f"<span style='white-space:nowrap'>{text}</span>"


TFAM = {k: nw(v) for k, v in TFAM.items()}
COL = dict(of=plots.SERIES[0], uni=plots.SERIES[1], clu=plots.SERIES[2], shift=plots.SERIES[2], wide=plots.SERIES[3])


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ---- number formatting (all math, so the typesetting check passes) ----------------------------------
def P(x, d=2):
    """Signed per cent for a table cell: +0.24"""
    return m(f"{100 * x:+.{d}f}") if x is not None and np.isfinite(x) else "–"


def U(x, d=2):
    """Unsigned per cent for a table cell: 0.75"""
    return m(f"{100 * x:.{d}f}") if x is not None and np.isfinite(x) else "–"


def pt(x, d=2):
    """Signed per cent in a sentence: +0.24 %"""
    return m(f"{100 * x:+.{d}f}" + r"\,\%")


def pa(x, d=2):
    """Size of a per cent value in a sentence: 0.24 %"""
    return m(f"{abs(100 * x):.{d}f}" + r"\,\%")


def sec(x):
    return f"{x:,.0f} s"


def num(x, d=2):
    return m(tf(x, d)) if x is not None and np.isfinite(x) else "–"


def p10(x):
    """A tolerance that is a power of ten, compactly: 10^{-8}."""
    e = int(round(np.log10(x)))
    return m(f"10^{{{e}}}") if np.isclose(x, 10.0 ** e) else m(tn(x, 2))


def times(x, d=1):
    return m(tf(x, d) + r"\times")


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def load():
    problem = load_yaml(HERE / "problem.yaml")
    numerics = load_yaml(HERE / "numerics.yaml")
    g = {k: v["value"] for k, v in problem["given"].items()}
    ref = reference(load_yaml(HERE / problem["reference"]["file"]))
    d = dict(problem=problem, numerics=numerics, g=g, ref=ref, of=jload(OUT / "analysis.json"), live=jload(OUT / "live_stop.json"),
             cmp=jload(OUT / "gpu_compare.json"), cl=jload(OUT / "gpu_clustered.json"), imex=jload(OUT / "gpu_imex.json"),
             first=jload(TUNE / "gpu_compare_first_G3_ptol1e-8.json"), c3first=jload(TUNE / "C3_first_ptol1e-8.json"))
    G = numerics["gpucfd"]
    C = G["clustered"]
    d["ptol"] = {L: jload(TUNE / ("ptol.json" if L == G["levels"][0] else f"ptol_{L}.json")) for L in G["levels"] + C["levels"]}
    return d


def rel_of(sm, ref):
    return {k: (sm[k] - ref["value"][k]) / abs(ref["value"][k]) for k in QKEYS}


def openfoam_rows(d):
    """OpenFOAM's levels: recorded runs (fixed lengths) and the wall time the live stop would have given."""
    g, rows, w, wl = d["g"], [], 0.0, 0.0
    umean = 2 * g["Um"] / 3
    for L in d["numerics"]["mesh"]["levels"]:
        mt = jload(RUNS / f"{L}.json")
        sm = dict(d["of"]["runs"][L]["sm"])
        sm.setdefault("St", sm["f"] * g["D"] / umean)
        lv = d["live"]["runs"][L]
        w += mt["wall_time"]
        wl += lv["replay_wall_est"]
        steps = mt["perf"]["steps"]
        rows.append(dict(level=L, fam="of", cells=mt["cells"], cores=mt["cores"], dt=mt["dt"], t0=mt["start_time"], t1=mt["end_time"],
                         steps=steps, wall_s=mt["wall_time"], wall_protocol=w, ms_wall=1e3 * mt["wall_time"] / steps,
                         us=1e6 * mt["wall_time"] / (steps * mt["cells"]), p_its=mt["perf"]["lin_iters"]["p"],
                         D_h=g["D"] / mt["h"]["wall"], health=mt["health"]["status"], sm=sm, rel=rel_of(sm, d["ref"]),
                         live_est=lv["replay_wall_est"], live_protocol=wl, live_measured=(lv.get("live") or {}).get("wall"),
                         live_fraction=lv["replay_fraction"]))
    return rows


def gpu_row(name, fam, sm, d):
    """One gpucfd run: its record (runs/<name>.json) and its coefficients over the last cycles."""
    G = d["numerics"]["gpucfd"]
    mt = jload(RUNS / f"{name}.json")
    p = mt["perf"]
    dt_conv = G["cfl"] * mt["h"] / G["u_dt"]         # the largest step the flow speed allows on the smallest cell
    return dict(level=name, fam=fam, cells=mt["cells"], n=mt["n"], h=mt["h"], D_h=mt["D_h"], dt=mt["dt"], dt_conv=dt_conv,
                below=dt_conv / mt["dt"], t0=mt["start_time"], t1=mt["end_time"], steps=p["steps"], gpu_s=p["gpu_s"],
                wall_s=p["wall_s"], observe_s=p["observe_s"], ms_gpu=p["ms_per_step"], ms_wall=1e3 * p["wall_s"] / p["steps"],
                us=1e6 * p["wall_s"] / (p["steps"] * p["cells"]), us_gpu=p["us_per_cell_step"], p_its=p["poisson_its"],
                ptol=mt["pressure_tol"], health=mt["health"]["status"], source=mt.get("source"), sm=sm, rel=rel_of(sm, d["ref"]))


def families(d):
    G = d["numerics"]["gpucfd"]
    C = G["clustered"]
    fam = dict(of=openfoam_rows(d))
    fam["uni"] = [gpu_row(L, "uni", d["cmp"]["runs"][L]["sm"], d) for L in G["levels"]]
    by = {r["level"]: r for r in d["cl"]["clustered"]}
    fam["clu"] = [gpu_row(L, "clu", by[L]["sm"], d) for L in C["levels"]]
    fam["shift"] = [gpu_row(L + "s", "shift", d["cl"]["variants"][L + "s"]["sm"], d) for L in C["levels"]]
    for key in ("uni", "clu", "shift"):
        w = 0.0
        for r in fam[key]:
            w += r["wall_s"]
            r["wall_protocol"] = w
    wide = gpu_row("C2w", "wide", d["cl"]["variants"]["C2w"]["sm"], d)
    wide["wall_protocol"] = fam["clu"][0]["wall_s"] + wide["wall_s"]      # C2w started from C1's field
    fam["wide"] = [wide]
    for key in ("clu", "shift", "wide"):
        for r in fam[key]:
            r["periodic"] = (by.get(r["level"]) or d["cl"]["variants"][r["level"]])["periodic"]
    for r in fam["uni"]:
        r["periodic"] = d["cmp"]["runs"][r["level"]]["periodic"]
    return fam


def clustered_grid(d, k, variant=None):
    """Face positions of clustered level k (0 = C1), as gpu_run.py builds them, and what they say about the cells."""
    g, C = d["g"], d["numerics"]["gpucfd"]["clustered"]
    var = C["variants"][variant] if variant else {}
    n0, band = var.get("n_C1", C["n_C1"]), var.get("band", C["band"])
    h = g["D"] / (C["D_h_C1"] * 2 ** k)
    x = grids.clustered(n0[0] * 2 ** k, g["L"], band[0], h, C["grow"][0])
    y = grids.clustered(n0[1] * 2 ** k, g["H"], band[1], h, C["grow"][1])
    fx, fy = int(np.sum(np.diff(x) <= h * (1 + 1e-6))), int(np.sum(np.diff(y) <= h * (1 + 1e-6)))
    cells = (len(x) - 1) * (len(y) - 1)
    return dict(x=x, y=y, h=h, band=band, cells=cells, far_x=grids.describe(x)["ratio"], far_y=grids.describe(y)["ratio"],
                in_band=fx * fy, share_band=fx * fy / cells, equal_cells=g["L"] * g["H"] / h ** 2)


def tolerance_rows(d, fam):
    """Per level: the recorded tolerance checks, the tolerance the rule gives and the one the run used."""
    used = {r["level"]: r["ptol"] for r in fam["uni"] + fam["clu"]}
    out = {}
    for L, runs in d["ptol"].items():
        by = {r["tol"]: r for r in runs}
        strict = min(by)
        ok = [t for t, r in by.items() if max(abs(r["d_cd"]), abs(r["d_cl"])) <= PTOL_RULE]
        out[L] = dict(by=by, strict=strict, rule=max(ok), used=used[L], consistent=bool(np.isclose(max(ok), used[L])),
                      wall=sum(r["perf"]["wall_s"] for r in runs))
    return out


def same_answer(d, fam):
    """The rule of numerics.yaml (gpucfd): |gpucfd finest - OpenFOAM finest| <= the two uncertainties added, for every
    primary quantity. For G3 it is recorded (gpu_compare.json); for C3 the same rule is applied here, with C3's
    grid-convergence index as its uncertainty (no time-step check was made on the clustered grids)."""
    prim = d["numerics"]["acceptance"]["primary"]
    c3, conv, comp = fam["clu"][-1], d["cl"]["convergence"], d["of"]["comp"]
    out = {}
    for k in prim:
        s = d["cmp"]["same"][k]
        dc = c3["sm"][k] - comp[k]["sim"]
        tol = conv[k]["gci_abs"] + comp[k]["u_sim"]
        out[k] = dict(openfoam=comp[k]["sim"], g3=s["gpu"], g3_diff=s["diff"], g3_tol=s["tol"], g3_ok=bool(s["ok"]),
                      c3=c3["sm"][k], c3_diff=dc, c3_tol=tol, c3_ok=bool(abs(dc) <= tol))
    return out


def data_commit(name="gpu_clustered.json"):
    """The commit that last changed a recorded file ('not committed yet' for a file git does not know)."""
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%h %cs", "--", f"results/{CASE}/{name}"],
                           capture_output=True, text=True, cwd=HERE.parents[1])
        return r.stdout.strip() or "not committed yet"
    except Exception:                                      # noqa: BLE001
        return "not read"


def sig1(x, unit=False):
    """A small per cent value to two significant figures, as math and without e-notation: 0.00047 (with unit: 0.00047 %)."""
    s = np.format_float_positional(100 * abs(x), precision=2, unique=False, fractional=False, trim="-")
    return m(s + (r"\,\%" if unit else ""))


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
def main():
    d = load()
    fam = families(d)
    build(d, fam)


def build(d, fam):
    g, numerics, ref, cl, cmp = d["g"], d["numerics"], d["ref"], d["cl"], d["cmp"]
    G = numerics["gpucfd"]
    C = G["clustered"]
    acc = numerics["acceptance"]
    prim = acc["primary"]
    of, uni, clu, shift, wide = fam["of"], fam["uni"], fam["clu"], fam["shift"], fam["wide"][0]
    m2, m4, g3, c1, c2, c3, c2s, c3s = of[1], of[-1], uni[-1], clu[0], clu[1], clu[-1], shift[1], shift[-1]
    tg = cl["target"]
    conv, convs = cl["convergence"], cl["convergence_shifted"]
    cuts = cl["cuts"]
    tol = tolerance_rows(d, fam)
    same = same_answer(d, fam)
    grid = [clustered_grid(d, k) for k in range(len(C["levels"]))]
    gridw = clustered_grid(d, 1, "w")
    uref = {k: ref["unc"][k] / abs(ref["value"][k]) for k in QKEYS}
    of_live = of[-1]["live_protocol"]
    of_rec = of[-1]["wall_protocol"]
    # the benchmark's own finest levels, extrapolated (analyze.reference): where the drag is heading
    ext = ref["seq"]["cd_mean"]["extrap"]
    rel_ext = lambda r: (r["sm"]["cd_mean"] - ext) / abs(ext)
    ext_gap = (ext - ref["value"]["cd_mean"]) / abs(ref["value"]["cd_mean"])
    # first runs that were replaced
    f3 = next(c for c in d["first"]["cost_gpu"] if c["level"] == g3["level"])
    first_total = sum(c["wall_s"] for c in d["first"]["cost_gpu"])
    first_sm = d["first"]["runs"][g3["level"]]["sm"]
    first_rel = rel_of(first_sm, ref)
    first_meta = d["first"]["runs"][g3["level"]]["meta"]
    c3f = d["c3first"]
    tune_wall = sum(t["wall"] for t in tol.values())
    # the implicit scheme (gpu_imex.json): each run beside its explicit twin, and the studies it makes possible
    im = d["imex"]
    ir, ist = im["runs"], im["studies"]
    ipair = {p["imex"]: p for p in im["pairs"]}
    c1i, c2i, c3i, c3wi = ir["C1i"], ir["C2i"], ir["C3i"], ir["C3wi"]
    imex_wall, expl_wall, wide_wall = ist["imex_finest"]["wall"], ist["explicit"]["wall"], ist["imex_finest_long_band"]["wall"]
    imex_same = all(p["same_answer"] for p in im["pairs"])
    diff_cd = max(abs(p["diff"][k]) for p in im["pairs"] for k in ("cd_mean", "cd_max", "St"))
    diff_cl = max(abs(p["diff"][k]) for p in im["pairs"] for k in ("cl_max", "cl_amp"))
    rule_cd, rule_cl = im["rule"]["cd_mean"], im["rule"]["cl_max"]
    pair3 = ipair["C3i"]
    share_imex = c3i["wall_s"] / imex_wall
    imex_rel = ist["imex_finest"]["finest_rel"]
    imex_conv = ist["imex_finest"]["finest_rel_cd_mean_converged"]
    wide_rel = ist["imex_finest_long_band"]["finest_rel"]
    moved_lift = abs(c3["rel"]["cl_max"] - c3s["rel"]["cl_max"])       # what moving the cylinder changes in the lift maximum
    wide_gain = abs(imex_rel["cl_max"]) - abs(wide_rel["cl_max"])
    imex_tol_same = bool(np.isclose(c3i["pressure_tol"], ir["C3"]["pressure_tol"]))
    imex_missed = not (IMEX_EXPECTED_STUDY[0] <= imex_wall <= IMEX_EXPECTED_STUDY[1])

    # ---- verdict (rules in code) ---------------------------------------------------------------
    fail, warn = [], []
    for r in uni + clu + shift + [wide]:
        if r["health"] == "fail":
            fail.append(f"run {r['level']} failed a health check")
    for r in (g3, c3, c3s):
        if not r["periodic"]:
            fail.append(f"the finest run {r['level']} did not reach a periodic state")
    for r in (c1i, c2i, c3i, c3wi):
        if r["health"] == "fail":
            fail.append(f"implicit-scheme run {r['level']} failed a health check")
        if not r["periodic"]:
            fail.append(f"implicit-scheme run {r['level']} did not reach a periodic state")
    if not imex_same:
        warn.append("the implicit scheme's answer differs from the explicit one by more than the rule fixed before the runs allowed")
    target_met = bool(tg["accuracy_met"] and tg["time_met"] and tg["shifted_accuracy_met"] and tg["shifted_time_met"])
    if not target_met:
        warn.append("the target was not met on both positions of the cylinder")
    if max(abs(rel_ext(c3)), abs(rel_ext(c3s))) > tg["want_rel"]:
        warn.append("measured against the value the reference is heading for, gpucfd's drag is outside the target's tolerance")
    bad_c3 = [k for k in prim if not same[k]["c3_ok"]]
    bad_g3 = [k for k in prim if not same[k]["g3_ok"]]
    same_ok = not bad_c3
    if bad_c3:
        warn.append("the finest clustered grid and OpenFOAM disagree by more than their error estimates allow in: " + ", ".join(NAME[k].lower() for k in bad_c3))
    if bad_g3:
        warn.append("the finest equal-cell grid and OpenFOAM disagree by more than their error estimates allow in: " + ", ".join(NAME[k].lower() for k in bad_g3))
    if cmp["verdict"] != "PASS":
        warn.append(f"by the case's own rules the equal-cell study is rated {cmp['verdict']}")
    off_c3 = [k for k in prim if conv[k]["status"] != "ok"]
    if off_c3:
        warn.append("the finest clustered grid is outside its own error band in: " + ", ".join(NAME[k].lower() for k in off_c3))
    verdict = "FAIL" if fail else "WARN" if warn else "PASS"
    print("verdict", verdict, "| fail", fail, "| warn", warn)

    # ---- tables (short, plain headings; detail stays in the cells) ---------------------------------
    acc_rows = [[TFAM[r["fam"]], r["level"], f"{r['cells']:,}", num(r["D_h"], 0), P(r["rel"]["cd_mean"]), P(r["rel"]["cl_max"]),
                 P(r["rel"]["St"]), f"{r['wall_protocol']:,.0f}"] for r in of + uni + clu + shift + [wide]]

    fin_rows = []
    for k in prim:
        oc = d["of"]["comp"][k]
        fin_rows.append([SHORT[k] + " " + m(SYM[k]), num(ref["value"][k], 4), U(uref[k]),
                         P(oc["rel"]), U(oc["u_sim"] / abs(ref["value"][k])),
                         P(c3["rel"][k]) + " / " + P(c3s["rel"][k]),
                         U(conv[k]["gci_rel"]) + " / " + U(convs[k]["gci_rel"])])
    ord_rows = []
    for k in prim:
        o = lambda s: num(s["p"], 2) if s.get("monotonic") and np.isfinite(s["p"]) else "no clear order"
        ord_rows.append([nw(SHORT[k]), o(d["of"]["ms"][k]), o(cmp["ms"][k]), o(conv[k]), o(convs[k])])
    cut_rows = []
    for r, rs in zip(clu, shift):
        c = cuts[r["level"]]
        cut_rows.append([r["level"] + " and " + rs["level"], num(r["D_h"], 0), P(c["first"]["cd_mean"]), P(c["shifted"]["cd_mean"]),
                         P(c["first"]["cl_max"]), P(c["shifted"]["cl_max"]), P(c["first"]["St"]), P(c["shifted"]["St"])])
    band_rows = []
    for r in (c2, c2s, wide, g3):
        e = r["rel"]
        band_rows.append([TFAM[r["fam"]], r["level"], f"{r['cells']:,}", P(e["cd_mean"]), P(e["cl_max"]), P(e["St"]),
                          f"{r['wall_protocol']:,.0f}"])

    cost_rows = [[TFAM[r["fam"]], r["level"], f"{r['cells']:,}", f"{r['steps']:,}", f"{r['wall_s']:,.0f}", num(r["ms_wall"], 1),
                  num(r["us"], 3)] for r in of + uni + clu + shift + [wide]]

    def study_row(label, rows, wall, note=""):
        e = rows[-1]["rel"]
        return [label, " + ".join(r["level"] for r in rows), f"{sum(r['steps'] for r in rows):,}", f"{wall:,.0f}" + note,
                P(e["cd_mean"]), P(e["cl_max"]), P(e["St"])]
    study_rows = [
        study_row("OpenFOAM, all four grids, as recorded", of, of_rec),
        study_row("OpenFOAM, all four grids, each run stopped when periodic", of, of_live, note=" *"),
        study_row("gpucfd, equal cells", uni, g3["wall_protocol"]),
        study_row("gpucfd, clustered", clu, c3["wall_protocol"]),
        study_row("gpucfd, clustered, cylinder moved", shift, c3s["wall_protocol"]),
        study_row("gpucfd, clustered, implicit scheme on the finest grid", [ir[k] for k in ist["imex_finest"]["levels"]], imex_wall),
        study_row("OpenFOAM, first two grids only", of[:2], m2["wall_protocol"]),
        study_row("gpucfd, clustered, first two grids only", clu[:2], c2["wall_protocol"]),
        study_row("gpucfd, clustered, longer band, two grids", [c1, wide], wide["wall_protocol"]),
    ]

    share_rows = [[TFAM[r["fam"]], r["level"], U(r["wall_s"] / rows[-1]["wall_protocol"], 0), f"{r['steps']:,}", m(tn(r["dt"], 3)),
                   m(tn(r["dt_conv"], 3)), num(r["below"], 1)] for rows in (uni, clu) for r in rows]
    tols = sorted({t for v in tol.values() for t in v["by"] if t <= 1e-8}, reverse=True)
    t_loose, t_mid, t_strict = tols[0], tols[1], tols[-1]

    def chg(t, x):
        r = t["by"].get(x)
        return "not run" if r is None else m(tn(max(abs(r["d_cd"]), abs(r["d_cl"])), 2))
    tol_rows = []
    for L in G["levels"] + C["levels"]:
        t = tol[L]
        tol_rows.append([L, f"{next(r['cells'] for r in uni + clu if r['level'] == L):,}", chg(t, t_loose), chg(t, t_mid),
                         p10(t["rule"]), p10(t["used"]) + (" ✓" if t["consistent"] else " ✗"),
                         num(t["by"][t_loose]["perf"]["ms_per_step"], 1), num(t["by"][t_strict]["perf"]["ms_per_step"], 1)])
    cell_rows = [[r["level"], f"{r['cells']:,}", num(r["D_h"], 0), U(gr["share_band"], 0), f"{gr['equal_cells']:,.0f}",
                  times(gr["equal_cells"] / r["cells"], 2)] for r, gr in list(zip(clu, grid)) + [(wide, gridw)]]
    fair_rows = [
        ["First run", p10(first_meta["pressure_tol"]), num(f3["ms_per_step"], 1), f"{f3['wall_s']:,.0f}", f"{first_total:,.0f}",
         P(first_rel["cd_mean"], 3), P(first_rel["cl_max"], 3), P(first_rel["St"], 3)],
        ["Rerun with the setting the rule gives", p10(g3["ptol"]), num(g3["ms_gpu"], 1), f"{g3['wall_s']:,.0f}",
         f"{g3['wall_protocol']:,.0f}", P(g3["rel"]["cd_mean"], 3), P(g3["rel"]["cl_max"], 3), P(g3["rel"]["St"], 3)],
    ]
    sa_rows = [[nw(SHORT[k] + " " + m(SYM[k])), num(same[k]["openfoam"], 4), num(same[k]["c3"], 4), m(f"{same[k]['c3_diff']:+.4f}"),
                num(same[k]["c3_tol"], 4), "✓" if same[k]["c3_ok"] else "✗"] for k in prim]

    imex_grid_rows = []
    for ex, ik in (("C1", "C1i"), ("C2", "C2i"), ("C3", "C3i")):
        e, i, p = ir[ex], ir[ik], ipair[ik]
        imex_grid_rows.append([ex, f"{e['cells']:,}", f"{e['steps']:,} / {i['steps']:,}", num(e["ms_per_step"], 1) + " / " + num(i["ms_per_step"], 1),
                               f"{e['wall_s']:,.0f} / {i['wall_s']:,.0f}", p10(e["pressure_tol"]) + " / " + p10(i["pressure_tol"]),
                               sig1(max(abs(p["diff"][k]) for k in ("cd_mean", "cd_max", "St"))),
                               sig1(max(abs(p["diff"][k]) for k in ("cl_max", "cl_amp")))])
    imex_study_rows = []
    for label, key in (("gpucfd, explicit on all three grids", "explicit"),
                       ("Implicit on the finest grid only", "imex_finest"),
                       ("Implicit on all three grids", "imex_all"),
                       ("Implicit on the finest, with the longer band of small cells", "imex_finest_long_band")):
        s = ist[key]
        imex_study_rows.append([label, " + ".join(s["levels"]), f"{sum(ir[L]['steps'] for L in s['levels']):,}", f"{s['wall']:,.0f}",
                                P(s["finest_rel"]["cd_mean"]), P(s["finest_rel_cd_mean_converged"]), P(s["finest_rel"]["cl_max"]),
                                P(s["finest_rel"]["St"])])

    # ---- numbers used in sentences -------------------------------------------------------------
    drag_lo = min(abs(c3["rel"]["cd_mean"]), abs(c3s["rel"]["cd_mean"]))
    dmax_hi = max(abs(c3["rel"]["cd_max"]), abs(c3s["rel"]["cd_max"]))
    lift_lo, lift_hi = sorted([abs(c3["rel"]["cl_max"]), abs(c3s["rel"]["cl_max"])])
    st_lo, st_hi = sorted([abs(c3["rel"]["St"]), abs(c3s["rel"]["St"])])
    gci_lo, gci_hi = sorted([conv["cd_mean"]["gci_rel"], convs["cd_mean"]["gci_rel"]])
    margin = 1 - c3["wall_protocol"] / of_live
    share_c3 = c3["wall_s"] / c3["wall_protocol"]
    tc3 = tol[c3["level"]]["by"]
    tol_cost = tc3[min(tc3)]["perf"]["ms_per_step"] / tc3[max(t for t in tc3 if t <= 1e-8)]["perf"]["ms_per_step"]
    cellstep = lambda r: r["cells"] * r["steps"]
    of_gpu_same_grid = c2["cells"] / g3["cells"]
    trend_c3 = c2["rel"]["cd_mean"] * c2["rel"]["cd_mean"] / c1["rel"]["cd_mean"]
    behind = lambda band: m(f"{(band[0][1] - g['xc']) / g['D']:g}")
    behind_c, behind_w = behind(C["band"]), behind(C["variants"]["w"]["band"])
    threads = ", ".join(str(r["cores"]) for r in of)
    of_drag_lo, of_drag_hi = min(abs(r["rel"]["cd_mean"]) for r in of), max(abs(r["rel"]["cd_mean"]) for r in of)
    twin = min(of, key=lambda r: abs(r["D_h"] - g3["D_h"]))      # OpenFOAM's grid with the same cell thickness at the cylinder
    live_meas = [r for r in of if r["live_measured"]]
    live_txt = ", ".join(f"{r['level']} {r['live_measured']:,.0f} s (estimate {r['live_est']:,.0f} s)" for r in live_meas)

    # ---- figures -------------------------------------------------------------------------------
    svgs = {"grids": _grids_fig(d, g, grid[0], uni[0]), "cost": _cost_fig(fam, uref, tg), "res": _res_fig(fam, conv),
            "time": _time_fig(fam, tg, [ir[k] for k in ist["imex_finest"]["levels"]]), "step": _step_fig(fam, tol)}
    figures = [
        dict(key="grids", section="setup", caption="The two kinds of grid gpucfd used, each at its coarsest (only every fourth "
             "line is drawn). Top: equal cells everywhere. Bottom: small cells in a band around the cylinder (dashed box), "
             "growing larger away from it. The grid is made of straight rows and columns, so a small column has to run the whole "
             "height of the channel: only " + pa(grid[0]["share_band"], 0) + " of the cells lie inside the box. The cylinder is "
             "cut out of the grid, not part of it. OpenFOAM's grid is not drawn; it bends around the cylinder."),
        dict(key="cost", section="results", caption="<b>How far from the reference, and how long it took.</b> Each point is one grid. "
             "Lower is more accurate; further left is quicker. A line joins the grids of one study, coarsest first. Hollow "
             "markers: the clustered study with the cylinder moved half a cell. Diamond: the middle clustered grid with a longer "
             "band of small cells. Grey area: smaller than the reference's own uncertainty, where it cannot tell results apart. "
             "Dashed lines in the left panel: the target (" + pa(tg["want_rel"]) + " and " + sec(tg["want_wall"]) + ")."),
        dict(key="res", section="results", caption="<b>What decides gpucfd's accuracy.</b> The same differences against the number "
             "of cells across the cylinder's diameter. Left: for the drag, equal and clustered cells lie close together at the "
             "same cell size. So the drag error follows the cell size at the cylinder, not the total number of cells. Dotted "
             "line: the slope of the order seen on the clustered grids (" + m(tf(conv["cd_mean"]["p"], 2)) + "). Right: the lift "
             "is worse on clustered grids until the band of small cells is made longer (diamond)."),
        dict(key="time", section="breakdown", caption="<b>Where each study's time goes.</b> One bar per study, split by grid, "
             "coarsest first. 'With stop rule' is OpenFOAM's time if every run stopped when the flow repeats (an estimate); "
             "'moved' is the cylinder moved half a cell; 'implicit finest' runs the finest grid with the implicit time scheme "
             "(section 5, last tables). The colours only count the grids of one study; the grids differ in size, so the cell counts are "
             "written under each label. gpucfd's third grid has about " + times(g3["cells"] / m4["cells"], 1) + " the cells of OpenFOAM's "
             "fourth, and takes a fraction of its time. That is the low cost per cell and step; it is not the same as "
             "accuracy (section 3). Dashed line: the target time, " + sec(tg["want_wall"]) + ". The finest grid takes most of the time: "
             + pa(m4["wall_s"] / of_rec, 0) + " for OpenFOAM, " + pa(g3["wall_s"] / g3["wall_protocol"], 0)
             + " for gpucfd with equal cells, " + pa(share_c3, 0) + " for gpucfd with clustered cells."),
        dict(key="step", section="breakdown", caption="<b>Time for one time step against the number of cells.</b> gpucfd (one line "
             "per pressure setting, circles equal cells, squares clustered) takes several milliseconds per step however few "
             "the cells. A stricter pressure setting costs more on every grid. OpenFOAM (" + threads + " processor threads on its "
             "four grids) grows with the cells. Dotted line: cost in proportion to the cells."),
    ]

    # ---- texts ---------------------------------------------------------------------------------
    ext_lo, ext_hi = sorted([abs(rel_ext(c3)), abs(rel_ext(c3s))])
    of_test = "would just miss" if abs(m4["rel"]["cd_mean"]) > tg["want_rel"] else "just passes"
    head = (("gpucfd <b>met the target</b> we set before the runs. " if target_met else
             "gpucfd <b>did not meet the target</b> we set before the runs. ")
            + "Its drag was " + pa(c3["rel"]["cd_mean"]) + " and " + pa(c3s["rel"]["cd_mean"]) + " below the published reference "
            "(two positions of the cylinder), and its three-grid study took about " + sec(imex_wall) + " (with the implicit time "
            "scheme on the finest grid; " + sec(expl_wall) + " without it), against about " + sec(of_live) + " for OpenFOAM's "
            "four-grid study. "
            "<b>But the target is easier than it looks.</b> The published reference is itself about " + pa(ext_gap, 1)
            + " too low. Against the value it is heading for, gpucfd's drag is " + pa(ext_lo) + " to " + pa(ext_hi)
            + " low, and OpenFOAM's is within " + pa(rel_ext(m4)) + ". "
            "<b>And OpenFOAM is still cheaper</b> for this accuracy: it reaches the same drag accuracy in " + sec(m2["wall_protocol"]) + ".")
    answers = [
        dict(label="Drag, gpucfd's finest grid, against the published reference",
             value=pt(c3["rel"]["cd_mean"]) + " and " + pt(c3s["rel"]["cd_mean"]),
             note="Two positions of the cylinder. The target was " + pa(tg["want_rel"]) + "."),
        dict(label="The same drag, against the value the reference is heading for",
             value=pt(rel_ext(c3)) + " and " + pt(rel_ext(c3s)),
             note="OpenFOAM: " + pt(rel_ext(m4)) + ". Against the published number itself OpenFOAM is " + pt(m4["rel"]["cd_mean"], 3)
                  + ", so it " + of_test + " the " + pa(tg["want_rel"]) + " test."),
        dict(label="Time for the whole study", value=sec(imex_wall),
             note="Implicit time scheme on the finest grid. Explicit: " + sec(expl_wall) + " (" + sec(c3s["wall_protocol"])
                  + " with the cylinder moved). OpenFOAM: about " + sec(of_live) + " (partly estimated)."),
        dict(label="Quickest way to get the drag within " + pa(tg["want_rel"]),
             value=sec(m2["wall_protocol"]) + " against " + sec(imex_wall),
             note="OpenFOAM's first two grids against gpucfd's three (implicit on the finest): " + times(imex_wall / m2["wall_protocol"], 0)
                  + " quicker."),
        dict(label="Lift, maximum", value=pa(c3["rel"]["cl_max"]) + " to " + pa(c3s["rel"]["cl_max"]) + " low",
             note="OpenFOAM: " + pt(m4["rel"]["cl_max"]) + "."),
        dict(label="gpucfd's own error estimate for the drag", value=pa(gci_lo) + " to " + pa(gci_hi),
             note="Made from its own grids, without the reference. OpenFOAM's: "
                  + pa(d["of"]["comp"]["cd_mean"]["u_sim"] / ref["value"]["cd_mean"]) + "."),
        dict(label="Same answer as OpenFOAM?", value=str(sum(same[k]["c3_ok"] for k in prim)) + " of " + str(len(prim)) + " outputs",
             note=("The " + ", ".join(NAME[k].lower() for k in bad_c3) + " differs" if bad_c3 else "All agree")
                  + " by more than the two error estimates allow." if bad_c3 else "All agree within the two error estimates."),
    ]
    findings = [
        "<b>Did gpucfd meet the target?</b> Yes, as it was defined: drag within " + pa(tg["want_rel"]) + " of the reference, in "
        "less time than OpenFOAM's whole study. It got " + pa(c3["rel"]["cd_mean"]) + " and " + pa(c3s["rel"]["cd_mean"])
        + ", in " + sec(c3["wall_protocol"]) + " and " + sec(c3s["wall_protocol"]) + " (explicit scheme; " + sec(imex_wall)
        + " with the implicit scheme on the finest grid), against about " + sec(of_live) + ".",
        "<b>Is that a fair test of accuracy?</b> Not quite. The reference was itself computed on finer and finer grids, and its "
        "drag was still rising. Its three finest values point to " + m(tf(ext, 4)) + ", which is " + pa(ext_gap) + " above the "
        "number we compare with (" + m(tf(ref["value"]["cd_mean"], 4)) + "). Against that, gpucfd is " + pa(ext_lo) + " to "
        + pa(ext_hi) + " low and OpenFOAM is within " + pa(rel_ext(m4)) + ". So this result does not show that gpucfd is as "
        "accurate as OpenFOAM.",
        "<b>Was the good drag luck?</b> Partly. gpucfd cuts the cylinder out of a grid of rows and columns, and where the cuts "
        "fall changes the answer a little. With the cylinder moved half a cell the drag went from " + pa(c3["rel"]["cd_mean"])
        + " to " + pa(c3s["rel"]["cd_mean"]) + " low. So what can be quoted is a range. Two positions are not a proper spread.",
        "<b>gpucfd costs far less per cell and step. Why is it not much quicker?</b> Per cell and step it is "
        + times(m4["us"] / c3["us"], 0) + " to " + times(m4["us"] / g3["us"], 0) + " cheaper than OpenFOAM. But its finest "
        "run has " + times(c3["cells"] / m4["cells"], 1) + " the cells and " + times(c3["steps"] / m4["steps"], 1) + " the steps, so "
        "it is only " + times(m4["wall_s"] / c3["wall_s"], 1) + " quicker (" + sec(c3["wall_s"]) + " against "
        + sec(m4["wall_s"]) + "). And OpenFOAM's second grid, with " + f"{m2['cells']:,}" + " cells, already has the drag.",
        "<b>Does an implicit time scheme make gpucfd faster?</b> Yes, but not more accurate. It gives the same answer (drag and "
        "frequency within " + sig1(diff_cd, True) + ", lift within " + sig1(diff_cl, True) + ", of the explicit runs). The finest run takes "
        + sec(c3i["wall_s"]) + " instead of " + sec(ir["C3"]["wall_s"]) + ", and the study " + sec(imex_wall) + " instead of "
        + sec(expl_wall) + ". " + ("We had expected " + f"{IMEX_EXPECTED_STUDY[0]:.0f} to {IMEX_EXPECTED_STUDY[1]:.0f} s: that was missed. "
                                   if imex_missed else "") + "The drag is as far from the reference as before ("
        + pt(imex_rel["cd_mean"]) + ").",
        "<b>What holds gpucfd back?</b> Three things (section 5). Its explicit time steps are very small: on the finest grid "
        + times(c3["below"], 1) + " smaller than the flow alone needs. The implicit scheme takes " + times(pair3["steps_ratio"], 1)
        + " fewer steps, but each costs " + times(pair3["ms_ratio"], 1) + " more, so the run gains " + times(ir["C3"]["wall_s"] / c3i["wall_s"], 2)
        + ", not " + times(pair3["steps_ratio"], 1) + ". A finer grid needs the pressure worked out more exactly, "
        "which makes each explicit step " + times(tol_cost, 1) + " dearer; the implicit scheme "
        + ("did not relax that setting on the finest grid. " if imex_tol_same else "needed a setting of " + p10(c3i["pressure_tol"]) + ". ")
        + "And small cells cannot go in one place only: just " + pa(grid[-1]["share_band"], 0) + " of the clustered cells are in "
        "the band around the cylinder. The finest grid still takes " + pa(share_imex, 0) + " of the study's time.",
        "<b>Why were lift and frequency worse on the clustered grid at first?</b> The small cells stopped too soon behind the "
        "cylinder, where the vortices form. With the small cells reaching " + behind_w + " diameters behind it instead of "
        + behind_c + ", the lift maximum went from " + pt(c2["rel"]["cl_max"]) + " to " + pt(wide["rel"]["cl_max"]) + " and the "
        "frequency from " + pt(c2["rel"]["St"]) + " to " + pt(wide["rel"]["St"]) + ", as good as equal cells. But then the "
        "clustered study took " + sec(wide["wall_protocol"]) + " against " + sec(g3["wall_protocol"]) + ": fewer cells bought "
        "almost no time.",
    ] + (["<b>Why the verdict is " + "'usable, with caveats'" + ":</b> " + "; ".join(warn) + "."] if warn else []) \
      + (["<b>Why the verdict is a failure:</b> " + "; ".join(fail) + "."] if fail else [])
    limitations = [
        "<b>Only the drag is close.</b> The lift maximum is " + pa(lift_lo) + " to " + pa(lift_hi) + " low and the frequency "
        + pa(st_lo) + " to " + pa(st_hi) + " high. OpenFOAM: " + pt(m4["rel"]["cl_max"]) + " and " + pt(m4["rel"]["St"]) + ".",
        "<b>gpucfd cannot prove its own accuracy yet.</b> Its error estimate for the drag is " + pa(gci_lo) + " to " + pa(gci_hi)
        + ", which is " + m(tf(gci_lo / tg["want_rel"], 1)) + " to " + m(tf(gci_hi / tg["want_rel"], 1)) + " times the target. "
        "The last step also fell faster than the one before it (it would be " + pt(trend_c3) + " on the earlier trend).",
        "<b>The comparison is not like for like.</b> Different kinds of grid, four grids against three, a processor against a "
        "graphics card, OpenFOAM's implicit time scheme against gpucfd's explicit (or, on the finest grid, partly implicit) one, one "
        "timed run each on one PC (section 1).",
        "<b>The implicit-scheme runs are one cylinder position, one timed run each.</b> The study time " + sec(imex_wall) + " has no "
        "spread, and the longer-band run (" + pt(wide_rel["cl_max"]) + " lift maximum) is a hint, not a result.",
        "<b>OpenFOAM's study time is partly an estimate.</b> Three of its four runs were really made with the stop rule; "
        "the finest, which is " + pa(of[-1]["live_est"] / of_live, 0) + " of the time, was not.",
        "<b>Tuning time is not counted on either side.</b> For gpucfd that is " + sec(tune_wall) + " of pressure checks and a "
        "first run of the finest grid that never settled (" + sec(c3f["wall_time"]) + ").",
        "<b>One flow only:</b> two dimensions, Reynolds number 100, one cylinder, one PC.",
    ]
    rec = ("<b>Still open.</b> (1) The longer band of small cells was run once on the finest grid (implicit scheme, one cylinder "
           "position, " + pa(c3wi["cells"] / c3i["cells"] - 1, 0) + " more cells): the lift maximum went from " + pt(imex_rel["cl_max"])
           + " to " + pt(wide_rel["cl_max"]) + ". That is " + ("less" if wide_gain < moved_lift else "more") + " than moving the cylinder "
           "changes it (" + pa(moved_lift, 1) + "), so it is a hint. (2) The next gain in speed: one pressure solve per time step in "
           "place of three (<code>gpucfd/DESIGN.md</code>, 'IMEX with immersed bodies, stage 3'). (3) More positions of the cylinder, "
           "to turn the range into a spread. "
           "<b>For choosing a tool today:</b> for a round body like this, OpenFOAM with a grid that follows the body is still "
           "the cheaper way to a given accuracy.")

    hw_env = dict((k, v) for k, v in report.environment_info())
    gpu_name = jload(RUNS / f"{c3['level']}.json")["gpu"]
    m4meta = jload(RUNS / f"{m4['level']}.json")
    terms = [
        dict(term="Reference", meaning="Numbers for this exact problem published by another research group (FeatFlow, TU Dortmund), "
             "computed on very fine grids. We use them as the answer to compare with. They are not exact either."),
        dict(term="Drag, lift", meaning="The force on the cylinder along the flow (drag) and across it (lift). Lift swings up and down "
             "as vortices leave the cylinder; we compare its largest value."),
        dict(term="Shedding frequency", meaning="How often vortices leave the cylinder. Written as a pure number, the Strouhal number."),
        dict(term="Cell, grid", meaning="The flow is calculated on a grid of small boxes called cells. More and smaller cells give "
             "a more exact answer but cost more time."),
        dict(term="Cells across the diameter", meaning="How many cells fit across the cylinder's width at the cylinder. It is the "
             "fairest way to compare grids of different shapes."),
        dict(term="OpenFOAM grid", meaning="A grid that bends around the cylinder (grids M1 to M4, each twice as fine as the one before)."),
        dict(term="Equal cells", meaning="gpucfd's first kind of grid: all cells the same size, over the whole channel (G1 to G3)."),
        dict(term="Clustered cells", meaning="gpucfd's second kind: small cells in a band around the cylinder, larger ones far away (C1 to C3)."),
        dict(term="Cut out of the grid", meaning="gpucfd's grids cannot bend, so the round cylinder is cut out of the rows and columns "
             "(an 'immersed body'). Where the cuts fall changes the answer a little."),
        dict(term="Check runs", meaning="Extra runs to test our own result: C1s to C3s (cylinder moved half a cell), C2w (a longer band "
             "of small cells behind the cylinder), and the equal-cell study repeated with a stricter pressure setting."),
        dict(term="Time step", meaning="The calculation moves forward in small jumps of time. Smaller jumps mean more steps and more work."),
        dict(term="Explicit and implicit", meaning="An explicit time scheme makes each jump simple and quick, but it is only stable if the "
             "jump is very small. An implicit one costs more per jump and allows larger jumps. OpenFOAM's is implicit. gpucfd's "
             "first scheme is explicit; it now also has one that treats friction implicitly (IMEX), which works with a body in "
             "the flow (section 5)."),
        dict(term="Pressure setting", meaning="At every step the pressure is worked out approximately. The setting says how closely. "
             "Too loose gives noisy forces; too strict wastes time."),
        dict(term="Error estimate", meaning="What a study says about its own error without looking at the reference. Made from three "
             "grids of growing fineness (the grid convergence index)."),
        dict(term="Order", meaning="How fast the error shrinks when the cells are halved. Order 2 means four times smaller; order 1.5 about 2.8 times."),
        dict(term="Study", meaning="A grid study: the problem solved on several grids, finest last. Its time is the time of all its runs together."),
        dict(term="Periodic", meaning="The flow has settled into a steady repeating pattern; only then are the forces measured."),
        dict(term="Wall time", meaning="Elapsed clock time, as a person would measure it with a stopwatch."),
    ]
    setup = dict(
        intro="The problem: a fluid flows along a channel past a round cylinder (case 4 of this project). Vortices leave the "
              "cylinder alternately from its two sides, so the forces on it swing up and down. We solve it with two programs: "
              "<b>OpenFOAM</b>, the established program, on the processor, and <b>gpucfd</b>, the project's own program, on the "
              "graphics card. Each answer is checked against numbers published by another research group. The list below explains "
              "the words used; Figure {fig:grids} shows gpucfd's two kinds of grid.",
        terms=terms,
        hardware=[["Processor (OpenFOAM)", hw_env.get("CPU", "")], ["Memory", hw_env.get("Memory", "")],
                  ["Graphics card (gpucfd)", gpu_name + ", 12 GB, a consumer card with slow double-precision arithmetic"],
                  ["Machine", "one desktop PC; OpenFOAM and gpucfd both run inside WSL2"]],
        software=[["OpenFOAM", m4meta["openfoam"] + ", " + numerics["solver"] + ", implicit second-order time stepping ("
                   + numerics["schemes"]["ddt"] + "), pressure by multigrid (" + numerics["linear_solvers"]["p"]["solver"] + ")"],
                  ["gpucfd", "explicit three-stage Runge-Kutta time stepping (" + G["scheme"] + "), pressure by conjugate "
                   "gradients with a multigrid preconditioner, cylinder as an immersed body. Also its implicit-friction scheme "
                   "(IMEX), used for the runs ending in i (section 5)"],
                  ["Reference", "FeatFlow, DFG 2D-2, finest level (level " + str(ref["top"]) + ")"],
                  ["Recorded data", "commit " + data_commit() + "; implicit-scheme runs (gpu_imex.json): " + data_commit("gpu_imex.json")
                   + ". No run was made for this report"]],
        fairness=[
            "<b>Same problem, same measurements.</b> Both programs solve the same channel and cylinder. Forces and frequency are "
            "taken by the same code from the last " + str(numerics["time"]["cycles_measured"]) + " cycles and compared with the same reference.",
            "<b>Each tool with its own best settings.</b> OpenFOAM: a grid that follows the cylinder, with the time scheme, "
            "pressure solver and number of threads chosen by tests in the main report. gpucfd: its two kinds of grid, with the "
            "pressure setting chosen for each grid by one rule.",
            "<b>Same procedure.</b> The coarsest grid starts from rest and runs " + m(tf(numerics["time"]["develop_end"], 0))
            + " s of flow. Each finer grid starts from the coarser grid's flow. A run ends when the flow repeats from cycle to cycle.",
            "<b>Rules written before the runs:</b> how accuracy is judged, what counts as the same answer, the target, and how "
            "each check run would be read (<code>numerics.yaml</code>, <code>gpucfd/DESIGN.md</code>, <code>ROADMAP.md</code>).",
            "<b>Time is elapsed time of every run in a study,</b> including sampling the forces. Tuning runs are not counted.",
            "<b>Not like for like: grids.</b> The two kinds of program need different grids, so cell counts cannot be compared "
            "directly, and OpenFOAM's study has four grids where gpucfd's have three.",
            "<b>Not like for like: hardware.</b> gpucfd uses the graphics card and one processor thread. OpenFOAM uses " + threads
            + " processor threads on its four grids.",
            "<b>Not like for like: OpenFOAM's study time.</b> OpenFOAM's runs were recorded with fixed lengths (" + sec(of_rec)
            + "). gpucfd's stop when the flow repeats. To compare the same thing, OpenFOAM's time with that stop was worked out "
            "from its recorded forces: " + sec(of_live) + ". Three of its four runs were then really made with the stop ("
            + live_txt + "); the finest was not.",
            "<b>Not like for like: time-step check.</b> OpenFOAM's time-step error was measured on its second grid. gpucfd's was "
            "measured once, on its middle equal-cell grid (halving the step changed the drag by " + m(tn(cmp["u_time"]["cd_mean"], 1))
            + "); none was made on the clustered grids.",
        ])
    method = dict(
        intro="What was decided, why, and what was changed after the first results.",
        choices=[
            dict(choice="Compare whole studies, not single runs",
                 why="Someone without a reference needs several grids to know how large the error is. So the cost that counts is the "
                     "cost of the whole study.",
                 alternatives="The cheapest single run that reaches an accuracy. Reported too (OpenFOAM's second grid)."),
            dict(choice="The target, fixed in advance: " + "mean and maximum drag within " + pa(tg["want_rel"]) + " of the reference, in "
                        "less time than OpenFOAM's whole study (about " + sec(tg["want_wall"]) + ")",
                 why="OpenFOAM's finest grid is about " + pa(m4["rel"]["cd_mean"]) + " from the reference, so this means 'as accurate "
                     "as OpenFOAM's study'. Its study time is what gpucfd has to beat to be worth using.",
                 alternatives="A target on lift or frequency too (not set; both are reported)."),
            dict(choice="gpucfd's grids: equal cells with " + ", ".join(m(tf(r["D_h"], 1)) for r in uni) + " cells across the "
                        "diameter, then clustered cells with " + ", ".join(m(tf(r["D_h"], 0)) for r in clu),
                 why="With equal cells the whole channel pays for the cell size the cylinder needs. Clustering puts the small cells "
                     "near the cylinder only: the finest clustered grid has twice the resolution at the cylinder for about the same "
                     "number of cells (" + f"{c3['cells']:,}" + " against " + f"{g3['cells']:,}" + ").",
                 alternatives="Equal cells at " + m(tf(c3["D_h"], 0)) + " across: " + f"{grid[-1]['equal_cells']:,.0f}" + " cells, "
                              "not run. A grid that bends around the cylinder: gpucfd does not have one."),
            dict(choice="A run ends when the flow repeats itself, checked while it runs",
                 why="The first gpucfd runs used OpenFOAM's fixed " + m(tf(numerics["time"]["run_length"], 0)) + " s per grid and were "
                     "still drifting at the end, because gpucfd's coarsest grid is far from the answer. Changed after those first runs.",
                 alternatives="Fixed run lengths (OpenFOAM's recorded runs)."),
            dict(choice="One rule for the pressure setting on every grid: the loosest setting whose drag and lift stay within "
                        + p10(PTOL_RULE) + " of the strictest run",
                 why="At first the setting was tuned on the coarsest grid only, and the finest clustered grid then never settled. "
                     "Changed after that run: checked on each grid, and runs that had used a setting the rule rejects were repeated.",
                 alternatives="One strict setting everywhere (" + times(tol_cost, 1) + " the cost per step on the finest grid)."),
            dict(choice="Accuracy and 'same answer' are judged by the case's own rules",
                 why="A result agrees with the reference if it is closer than the two uncertainties added: the study's own error "
                     "estimate plus the reference's. Two programs agree if they differ by less than their two error estimates added. "
                     "Both rules were written before the equal-cell runs; applying the second to the clustered grids is done here.",
                 alternatives="Agreement to a fixed per cent (ignores how well each side knows its own answer)."),
            dict(choice="Three check runs, each with its question and its reading written before it ran",
                 why="The first clustered result was better than the trend of the coarser grids. Before quoting it: is it luck "
                     "(move the cylinder), is the time comparison fair (the pressure rule on the equal-cell grids), and why are lift "
                     "and frequency worse (a longer band).",
                 alternatives="More positions of the cylinder: each costs another whole study, about " + sec(c3s["wall_protocol"])
                              + " (not run)."),
            dict(choice="Try the implicit time scheme, with its pass rule fixed beforehand: the same answer as the explicit run on "
                        "the same grid to within " + sig1(rule_cd, True) + " (drag, frequency) and " + sig1(rule_cl, True) + " (lift)",
                 why="On the finest clustered grid the explicit steps were " + times(c3["below"], 1) + " smaller than the flow needs "
                     "and that grid took " + pa(share_c3, 0) + " of the study's time. Each grid may use whichever scheme is quicker "
                     "on it, because both solve the same equations.",
                 alternatives="Halving the step on a grid that misses the rule, to tell a time error from a different answer ("
                              + ("not needed: no grid missed)." if imex_same else "see the section on the implicit scheme).")),
        ])
    of_t = d["of"]["comp"]["cd_mean"]
    results = dict(
        intro="Differences are in per cent of the reference value; a minus sign means below it. Times are elapsed times on this PC. "
              "Each table is headed by the question it answers; below it, what came out and what it means.",
        tables=[
            dict(title="How close is each grid to the reference?",
                 head=["Solver and grid", "Grid", "Cells", "Cells across the diameter", "Drag (%)", "Lift maximum (%)",
                       "Frequency (%)", "Study time so far (s)"], rows=acc_rows,
                 note="<b>Result.</b> gpucfd needs " + m(tf(c3["D_h"], 0)) + " cells across the cylinder to bring the drag inside "
                      + pa(tg["want_rel"]) + ". OpenFOAM's drag is " + pa(of_drag_lo) + " to " + pa(of_drag_hi) + " off on every grid, "
                      "even the coarsest, but that grid's lift and frequency are " + pa(of[0]["rel"]["cl_max"], 1) + " and "
                      + pa(of[0]["rel"]["St"], 1) + " off. At " + m(tf(g3["D_h"], 0)) + " cells across, OpenFOAM is "
                      + pt(twin["rel"]["cd_mean"]) + " off and gpucfd " + pt(g3["rel"]["cd_mean"]) + " and " + pt(c2["rel"]["cd_mean"])
                      + ". <b>What it means.</b> A grid that follows the cylinder is worth far more per cell than small square cells "
                      "that cut through it. (Study time so far: this grid and the coarser ones of its study.)"),
            dict(title="How well does each study know its own error?",
                 head=["Quantity", "Reference value", "Reference's own uncertainty (%)", "OpenFOAM, finest grid: difference (%)",
                       "OpenFOAM's error estimate (%)", "gpucfd, finest clustered, two positions: difference (%)",
                       "gpucfd's error estimate (%)"], rows=fin_rows,
                 note="<b>Result.</b> OpenFOAM's drag has stopped changing with the grid, so its error estimate is tiny (a few hundredths "
                      "of a per cent). gpucfd's has not: " + pa(gci_lo) + " to " + pa(gci_hi) + " for the drag. <b>What it means.</b> "
                      "gpucfd's drag is close to the reference, but its three grids are too far apart to prove it. The lift maximum "
                      "on the clustered grids does not settle regularly, so its estimate is weak."),
            dict(title="Was the good drag luck? The cylinder moved half a cell",
                 head=["Grids", "Cells across the diameter", "Drag, first position (%)", "Drag, moved (%)",
                       "Lift maximum, first (%)", "Lift maximum, moved (%)", "Frequency, first (%)", "Frequency, moved (%)"],
                 rows=cut_rows,
                 note="<b>Result.</b> On the finest grid the drag moves from " + pt(c3["rel"]["cd_mean"]) + " to "
                      + pt(c3s["rel"]["cd_mean"]) + " and the lift maximum from " + pt(c3["rel"]["cl_max"]) + " to "
                      + pt(c3s["rel"]["cl_max"]) + ". Both are inside the target, in " + sec(c3["wall_protocol"]) + " and "
                      + sec(c3s["wall_protocol"]) + ". <b>What it means.</b> The first position was somewhat favourable, not a fluke. "
                      "This kind of error is invisible to the error estimate, because every grid cuts the cylinder at the same relative "
                      "place. The shifts are different fractions of a cell on each grid, so the rows do not show how the scatter shrinks."),
            dict(title="Why were lift and frequency worse? A longer band of small cells behind the cylinder",
                 head=["Solver and grid", "Grid", "Cells", "Drag (%)", "Lift maximum (%)", "Frequency (%)", "Study time (s)"],
                 rows=band_rows,
                 note="<b>Result.</b> All four have " + m(tf(c2["D_h"], 0)) + " cells across the diameter. With the band reaching "
                      + behind_w + " diameters behind the cylinder instead of " + behind_c + " (" + pa(wide["cells"] / c2["cells"] - 1, 0)
                      + " more cells), the lift maximum goes from " + pt(c2["rel"]["cl_max"]) + " to " + pt(wide["rel"]["cl_max"])
                      + " and the frequency from " + pt(c2["rel"]["St"]) + " to " + pt(wide["rel"]["St"]) + ". Moving the cylinder "
                      "changes them far less (C2s). <b>What it means.</b> The vortices form just behind the cylinder and need small "
                      "cells there. It also changes the cost picture: for the drag alone the clustered study took "
                      + m(tf(c2["wall_protocol"] / g3["wall_protocol"], 2)) + " of the equal-cell study's time, but matching lift and "
                      "frequency too, " + m(tf(wide["wall_protocol"] / g3["wall_protocol"], 2)) + "."),
            dict(title="What does each run cost?",
                 head=["Solver and grid", "Grid", "Cells", "Time steps", "Time (s)", "Time per step (ms)",
                       "Time per cell and step (" + m(r"\mu\mathrm{s}") + ")"], rows=cost_rows,
                 note="<b>Result.</b> Per cell and step gpucfd costs " + num(c3["us"], 3) + " to " + num(g3["us"], 3) + " "
                      + m(r"\mu\mathrm{s}") + " on its finest grids; OpenFOAM " + num(m4["us"], 2) + ". <b>What it means.</b> gpucfd is "
                      + times(m4["us"] / c3["us"], 0) + " to " + times(m4["us"] / g3["us"], 0) + " cheaper per cell and step, and "
                      "spends it on more cells and many more steps. Both use the elapsed time of the run (for gpucfd that includes "
                      "sampling the forces)."),
            dict(title="What does each whole study cost?",
                 head=["Study", "Grids", "Time steps", "Time (s)", "Drag on the last grid (%)", "Lift maximum (%)", "Frequency (%)"],
                 rows=study_rows,
                 note="* An estimate: the stop rule applied afterwards to the recorded forces (section 1). Target time: "
                      + sec(tg["want_wall"]) + ". <b>Result.</b> gpucfd's clustered study takes "
                      + m(tf(c3["wall_protocol"] / tg["want_wall"], 2)) + " of the target time; its equal-cell study "
                      + m(tf(g3["wall_protocol"] / tg["want_wall"], 2)) + " of it, but with the drag " + pa(g3["rel"]["cd_mean"], 1)
                      + " low. OpenFOAM's first two grids take " + m(tf(m2["wall_protocol"] / tg["want_wall"], 2)) + " of it. "
                      "<b>What it means.</b> gpucfd wins the comparison the target sets (whole study against whole study) and "
                      "loses the comparison of the cheapest route to this drag accuracy (Figure {fig:cost})."),
            dict(title="How fast does the error fall as the cells get smaller?",
                 head=["Quantity", "OpenFOAM", "gpucfd, equal cells", "gpucfd, clustered", "gpucfd, clustered, moved"],
                 rows=ord_rows,
                 note="<b>Result.</b> gpucfd's drag converges at order " + m(tf(conv["cd_mean"]["p"], 1)) + " to "
                      + m(tf(convs["cd_mean"]["p"], 1)) + ". 'No clear order' means the three values do not approach a limit from "
                      "one side (for OpenFOAM's drag because it has already converged). <b>What it means.</b> Order 1.5 is the known "
                      "price of cutting a body out of a grid instead of fitting the grid to it."),
        ],
        notes=[
            "<b>How to read Figure {fig:cost}.</b> A study is better the lower and further left its line reaches. OpenFOAM's drag line "
            "is flat at about " + pa(m4["rel"]["cd_mean"]) + ": it has settled, from its first grid on. gpucfd's lines start far off "
            "and fall steeply. The clustered line ends lowest for the drag and above OpenFOAM's for the lift.",
            "<b>The check on the reference.</b> The reference's finest drag is " + m(tf(ref["value"]["cd_mean"], 4)) + ", and its "
            "uncertainty (its two finest levels) is " + pa(uref["cd_mean"]) + ". Its three finest levels point to " + m(tf(ext, 4))
            + " (order " + m(tf(ref["seq"]["cd_mean"]["p"], 2)) + ", so this is itself uncertain), " + pt(ext_gap) + " above. "
            "OpenFOAM's settled drag is " + m(tf(m4["sm"]["cd_mean"], 4)) + ", " + pt(rel_ext(m4)) + " from that. So 'closer to the "
            "reference than OpenFOAM' must not be read as 'more accurate than OpenFOAM'.",
            "<b>Health of the runs.</b> Every run finished without errors. The coarsest grids carry a warning for a brief excess of "
            "the time-step number or the leftover divergence just after the start (" + ", ".join(
                r["level"] for r in uni + clu + shift + [wide] if r["health"] == "warn") + "). The finest grid of each study is "
            "clean and periodic.",
        ])
    bad_g3_txt = (" (not in: " + ", ".join(NAME[k].lower() for k in bad_g3) + ")") if bad_g3 else ""
    same_txt = ("<b>Result.</b> gpucfd's finest clustered grid agrees with OpenFOAM's finest in "
                + str(sum(same[k]["c3_ok"] for k in prim)) + " of " + str(len(prim)) + " outputs"
                + ("" if same_ok else "; the " + ", ".join(NAME[k].lower() for k in bad_c3) + " does not")
                + ". The finest equal-cell grid agrees in " + str(sum(same[k]["g3_ok"] for k in prim)) + " of " + str(len(prim))
                + bad_g3_txt + ". <b>What it means.</b> Where they agree it is partly because gpucfd's error estimate is wide: the "
                "mean drags of the two programs differ by " + pa(same["cd_mean"]["c3_diff"] / ref["value"]["cd_mean"])
                + ", gpucfd below the reference and OpenFOAM above it.")
    same_sec = dict(
        intro="Question: do the two programs give the same answer? We say yes if their results differ by less than their two "
              "error estimates added together. The rule was fixed before the equal-cell runs; here it is also applied to "
              "gpucfd's finest clustered grid. Values are the coefficients themselves, not per cent.",
        head=["Quantity", "OpenFOAM", "gpucfd, finest clustered", "Difference", "Allowed", "Same?"],
        rows=sa_rows, ok=same_ok, verdict=same_txt)
    breakdown = dict(
        intro="Question: gpucfd costs far less per cell and step, so why does it not win by more? Three reasons, each measured here, "
              "a check that the comparison between gpucfd's two kinds of grid was fair, and what the implicit time scheme "
              "changed (last two tables).",
        tables=[
            dict(title="Reason 1: very small time steps. How many steps does each grid need?",
                 head=["Grid type", "Grid", "Share of the study's time (%)", "Time steps", "Time step used (s)",
                       "Largest step the flow alone allows (s)", "How many times smaller"], rows=share_rows,
                 note="gpucfd's time scheme is explicit. It is only stable if each jump in time is small enough for two things: the "
                      "fluid must not cross more than about one cell per step (a limit that shrinks with the cell size), and "
                      "friction must not spread more than about one cell per step (a limit that shrinks with the cell size squared). "
                      "<b>Result.</b> On the finest clustered grid the step is " + times(c3["below"], 1) + " smaller than the flow alone "
                      "needs; on the middle grids " + times(c2["below"], 1) + ". <b>What it means.</b> " + pa(1 - 1 / c3["below"], 0)
                      + " of the finest grid's steps are there for stability, not for accuracy. An implicit treatment of friction "
                      "removes the limit, and gpucfd now has one that works with a body in the flow: it takes " + f"{c3i['steps']:,}"
                      + " steps on the finest grid, but each costs more (last two tables). OpenFOAM, which is implicit, "
                      "takes " + f"{m4['steps']:,}" + " steps on its finest grid where explicit gpucfd takes " + f"{c3['steps']:,}" + "."),
            dict(title="Reason 2: the pressure setting. How exact must it be on each grid?",
                 head=["Grid", "Cells", "Change in forces at " + p10(t_loose), "Change in forces at " + p10(t_mid), "Setting the rule gives",
                       "Setting used", "Time per step at " + p10(t_loose) + " (ms)", "Time per step at " + p10(t_strict) + " (ms)"],
                 rows=tol_rows,
                 note="Each grid was run for one shedding cycle at several settings. 'Change in forces' is the largest change in the "
                      "drag and lift coefficients compared with the strictest run. The rule takes the loosest setting that stays "
                      "within " + p10(PTOL_RULE) + ". ✓: the run in this report used it. <b>Result.</b> The finer the grid, the "
                      "stricter the setting; on the finest clustered grid the strict setting makes a step " + times(tol_cost, 1)
                      + " dearer. <b>What it means.</b> It is tied to reason 1: with an explicit scheme a small error in the pressure "
                      "is divided by the time step when it enters the forces, so very small steps need a very exact pressure."
                      + (" The implicit scheme, with steps " + times(pair3["dt_ratio"], 1) + " larger, still needed the same "
                         "setting on the finest grid. The reason: it splits each step into three unequal parts, and the pressure "
                         "error is divided by the shortest part. That part is only " + times(pair3["dt_ratio"] * (2 / 15) / (1 / 4), 1)
                         + " longer than the explicit scheme's shortest part, too little to allow a looser setting."
                         if imex_tol_same else "")),
            dict(title="Reason 3: the shape of the grid. How many cells does clustering save?",
                 head=["Grid", "Cells", "Cells across the diameter", "Share of cells inside the band around the cylinder (%)",
                       "Cells an equal-cell grid of that small size would need", "Saving"], rows=cell_rows,
                 note="<b>Result.</b> Clustering saves " + times(grid[-1]["equal_cells"] / c3["cells"], 2) + ", and only about "
                      + pa(grid[-1]["share_band"], 0) + " of the clustered cells are in the band where they are wanted. "
                      "<b>What it means.</b> A grid of straight rows and columns cannot be fine in one place only: every small column "
                      "runs across the whole channel, every small row along it (Figure {fig:grids}). The channel is only "
                      + m(tf(g["H"] / g["D"], 1)) + " diameters high, so the saving across it is small. OpenFOAM reaches the same "
                      "drag with " + f"{m2['cells']:,}" + " cells where the finest clustered grid has " + f"{c3['cells']:,}" + "."),
            dict(title="Check: was the equal-cell study's time fair? Rerun with the pressure rule's setting",
                 head=["Finest equal-cell grid", "Pressure setting", "Time per step (ms)", "Time of the run (s)", "Time of the study (s)",
                       "Drag (%)", "Lift maximum (%)", "Frequency (%)"], rows=fair_rows,
                 note="<b>Result.</b> The same coefficients to the digits shown, a step " + times(g3["ms_gpu"] / f3["ms_per_step"], 2)
                      + " dearer, and a study of " + sec(g3["wall_protocol"]) + " instead of " + sec(first_total) + " (the rerun "
                      "became periodic a second of flow earlier, which hid most of the extra cost). <b>What it means.</b> The "
                      "comparison between gpucfd's two kinds of grid now follows one rule, and it barely moved."),
            dict(title="Does an implicit time scheme make it faster? Each grid, run both ways",
                 head=["Grid", "Cells", "Time steps (explicit / implicit)", "Time per step, ms (explicit / implicit)",
                       "Time, s (explicit / implicit)", "Pressure setting (explicit / implicit)",
                       "Largest difference in drag and frequency (%)", "Largest difference in lift (%)"],
                 rows=imex_grid_rows,
                 note="Differences are of the implicit run from the explicit run on the same grid, in per cent of the value. "
                      "<b>Result.</b> The implicit scheme gives the same answer on every grid: drag and frequency within "
                      + sig1(diff_cd, True) + " and lift within " + sig1(diff_cl, True) + " (the rule fixed beforehand: "
                      + sig1(rule_cd, True) + " and " + sig1(rule_cl, True) + "). On the finest grid it takes " + times(pair3["steps_ratio"], 1)
                      + " fewer steps, each " + times(pair3["ms_ratio"], 1) + " dearer: " + sec(c3i["wall_s"]) + " instead of "
                      + sec(ir["C3"]["wall_s"]) + ". On the coarsest grid it is slower (" + sec(c1i["wall_s"]) + " against "
                      + sec(ir["C1"]["wall_s"]) + "), so that grid stays explicit. <b>What it means.</b> Fewer steps, but each is dearer, so the finest run gains only "
                      + times(ir["C3"]["wall_s"] / c3i["wall_s"], 2) + ", not " + times(pair3["steps_ratio"], 1) + ". Each step has "
                      "a pressure solve in every one of its three stages, besides the friction solves."),
            dict(title="What does it do to the whole study? The three grids, with and without it",
                 head=["Study", "Grids", "Time steps", "Time (s)", "Drag on the last grid (%)",
                       "Drag, against the value the reference heads for (%)", "Lift maximum (%)", "Frequency (%)"],
                 rows=imex_study_rows,
                 note="<b>Result.</b> The quickest mix is the implicit scheme on the finest grid only: " + sec(imex_wall) + ", against "
                      + sec(expl_wall) + " explicit and about " + sec(of_live) + " for OpenFOAM's study. The drag does not change ("
                      + pt(imex_rel["cd_mean"]) + " of the published value, " + pt(imex_conv) + " of the value it is heading for). "
                      "The time we expected before building was " + f"{IMEX_EXPECTED_STUDY[0]:.0f} to {IMEX_EXPECTED_STUDY[1]:.0f}"
                      + " s" + ("; it was missed: each step costs more" + (", and the pressure setting did not relax." if imex_tol_same
                                                                            else ".") if imex_missed else ".") + " <b>What it means.</b> The implicit scheme makes gpucfd faster, not more accurate. It "
                      "does not change which tool is cheaper for this accuracy: OpenFOAM's first two grids take "
                      + sec(m2["wall_protocol"]) + "."),
        ],
        notes=[
            "<b>Fixed work per step.</b> A gpucfd step costs " + num(uni[0]["ms_gpu"], 1) + " ms on " + f"{uni[0]['cells']:,}"
            + " cells and " + num(g3["ms_gpu"], 1) + " ms on " + f"{g3['cells']:,}" + ": " + times(g3["cells"] / uni[0]["cells"], 0)
            + " the cells for " + times(g3["ms_gpu"] / uni[0]["ms_gpu"], 1) + " the time (Figure {fig:step}). On grids this small "
            "the card is not kept busy; the cost is set by the number of operations in a step, not by the cells. So fewer cells "
            "bought accuracy per cell, not time.",
            "<b>Where gpucfd's time went</b> (Figure {fig:time}). The finest clustered grid takes " + pa(share_c3, 0) + " of its "
            "study, and both limits bite there: " + times(c3["below"], 1) + " more steps than the flow needs, each "
            + times(tol_cost, 1) + " dearer. Sampling the forces is " + pa(c3["observe_s"] / c3["wall_s"], 0) + " of that run.",
            "<b>What a first attempt cost.</b> The finest clustered grid was first run with the pressure setting tuned on the coarsest "
            "grid. It ran its full " + m(tf(c3f["end_time"] - c3f["start_time"], 0)) + " s of flow (" + sec(c3f["wall_time"])
            + ") without settling: the forces carried noise from the pressure solve. That run is kept in <code>gpu_tuning/</code> "
            "and is in no time quoted here.",
            "<b>One extra run: the longer band of small cells on the finest grid</b> (implicit scheme, one cylinder position; last row "
            "of the last table). The lift maximum is " + pt(wide_rel["cl_max"]) + " instead of " + pt(imex_rel["cl_max"]) + ", the "
            "study " + sec(wide_wall) + " instead of " + sec(imex_wall) + ". The gain, " + pa(wide_gain, 2) + ", is " + ("smaller" if wide_gain < moved_lift else "larger")
            + " than what moving the cylinder changes in the lift maximum (" + pa(moved_lift, 1) + "), so it is a hint, not a result. "
            "Its pressure setting was not checked on this grid.",
        ])
    repro = dict(
        commands=[
            dict(cmd=f"python -m cases.{CASE}.run", what="OpenFOAM: the four grids M1 to M4 (the main report's runs)"),
            dict(cmd=f"python -m cases.{CASE}.analyze", what="OpenFOAM: analysis and the main report (writes analysis.json)"),
            dict(cmd=f"python -m cases.{CASE}.gpu_run production", what="gpucfd, equal cells: G1 from rest, G2, G2 with half the time step, G3"),
            dict(cmd=f"python -m cases.{CASE}.gpu_run ptol [LEVEL]", what="gpucfd: pressure-setting check on one grid"),
            dict(cmd=f"python -m cases.{CASE}.gpu_run clustered", what="gpucfd, clustered cells: C1 from rest, C2, C3"),
            dict(cmd=f"python -m cases.{CASE}.gpu_run clustered C1s C2s C3s", what="check run: the cylinder moved half a cell"),
            dict(cmd=f"python -m cases.{CASE}.gpu_run clustered C2w", what="check run: the longer band of small cells"),
            dict(cmd=f"python -m cases.{CASE}.gpu_run tolcheck C3i C2", what="the implicit scheme: pressure setting of a grid, from the last field of C2 (likewise C1i, C2i)"),
            dict(cmd=f"python -m cases.{CASE}.gpu_run clustered C1i C2i C3i C3wi", what="the implicit scheme on the clustered grids, and the longer band on the finest"),
            dict(cmd=f"python -m cases.{CASE}.gpu_imex", what="the implicit runs against the explicit ones (writes gpu_imex.json)"),
            dict(cmd=f"python -m cases.{CASE}.gpu_compare", what="gpucfd with equal cells against the reference and OpenFOAM (writes gpu_compare.json)"),
            dict(cmd=f"python -m cases.{CASE}.gpu_clustered", what="the clustered grids and the check runs (writes gpu_clustered.json)"),
            dict(cmd=f"python -m cases.{CASE}.gpu_study", what="this report, from the recorded files; runs nothing"),
        ],
        files=[dict(path=f"cases/{CASE}/numerics.yaml", what="every numerical choice, the rules fixed before the runs and the changes made after (block gpucfd)"),
               dict(path=f"cases/{CASE}/gpu_run.py", what="the gpucfd runs"),
               dict(path=f"cases/{CASE}/gpu_compare.py, gpu_clustered.py, gpu_imex.py", what="the analyses this report reads"),
               dict(path=f"cases/{CASE}/gpu_study.py", what="this report"),
               dict(path="gpucfd/DESIGN.md", what="design, pass criteria and cost estimate written before the clustered grids were built; results and check runs"),
               dict(path=f"results/{CASE}/runs/", what="one record per run: G1 to G3, C1 to C3, C1s to C3s, C2w, C1i to C3i, C3wi (gpucfd) and M1 to M4 (OpenFOAM)"),
               dict(path=f"results/{CASE}/gpu_tuning/", what="pressure-setting checks per grid and the first runs they replaced")],
        run_dirs=[dict(path=m4meta["run_root"], what="OpenFOAM cases (outside git)"),
                  dict(path=f"~/gpucfd_runs/{CASE}/", what="gpucfd: final velocity fields of each run (outside git)")],
        environment=report.environment_info(),
        notes=["The numbers of this report are read from recorded files (the commit named in section 1); rebuilding the page "
               "does not rerun anything and does not need the graphics card.",
               "Each time is one measured run. The two finest clustered runs, which differ only in the position of the "
               "cylinder and take the same number of steps, needed " + sec(c3["wall_s"]) + " and " + sec(c3s["wall_s"])
               + ": a difference of " + pa(c3["wall_s"] / c3s["wall_s"] - 1, 0) + ", which shows how far a repeat can move a time."])
    s_ = dict(
        title="gpucfd next to OpenFOAM: a cylinder shedding vortices (DFG 2D-2, Re = 100)",
        date=dt.date.today().isoformat(),
        question="On a cylinder that sheds vortices, does the project's GPU program gpucfd reach the accuracy of OpenFOAM's grid "
                 "study in less time, what does it cost, and what holds it back?",
        verdict=verdict,
        summary=dict(headline=head, answers=answers, findings=findings, limitations=limitations, recommendation=rec),
        setup=setup, method=method, results=results, same_answer=same_sec, breakdown=breakdown, reproducibility=repro,
        figures=figures,
    )
    out = study.render_study(s_, svgs, OUT / "gpu_study.html", insights.load(CASE))
    print("Study:", out)


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def _grids_fig(d, g, c1grid, g1):
    fig, axes = plots.plt.subplots(2, 1, figsize=(plots.COLUMN_IN, 4.9), layout="constrained")
    xu = grids.uniform(g1["n"][0], g["L"])
    yu = grids.uniform(g1["n"][1], g["H"])
    for ax, x, y, title in ((axes[0], xu, yu, "gpucfd, equal cells (coarsest grid, every fourth line)"),
                            (axes[1], c1grid["x"], c1grid["y"], "gpucfd, clustered cells (coarsest grid, every fourth line)")):
        segs = [[(xi, 0.0), (xi, g["H"])] for xi in x[::4]] + [[(0.0, yi), (g["L"], yi)] for yi in y[::4]]
        ax.add_collection(LineCollection(segs, colors=plots.MUTED, linewidths=0.35))
        ax.add_patch(Circle((g["xc"], g["yc"]), g["D"] / 2, facecolor=plots.SERIES[1], edgecolor=plots.INK, linewidth=0.8, zorder=3))
        ax.set_xlim(0, g["L"])
        ax.set_ylim(0, g["H"])
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(title)
        ax.set_ylabel("y (m)")
    b = c1grid["band"]
    axes[1].add_patch(Rectangle((b[0][0], b[1][0]), b[0][1] - b[0][0], b[1][1] - b[1][0], fill=False, edgecolor=plots.INK,
                                linewidth=1.1, linestyle="--", zorder=4))
    axes[1].set_xlabel("x (m), flow from left to right")
    return plots.to_svg(fig, "grids")


def _series(ax, rows, x, y, fam, hollow=False, **kw):
    style = dict(of="o-", uni="s-", clu="^-", shift="^--", wide="D")[fam]
    ax.plot([x(r) for r in rows], [y(r) for r in rows], style, color=COL[fam], ms=6 if fam != "wide" else 7, lw=1.5,
            mfc=plots.SURFACE if hollow else COL[fam], label=FAM[fam], **kw)


def _cost_fig(fam, uref, tg):
    fig, axes = plots.wide(3, 3.5)
    for ax, k, title in zip(axes, ("cd_mean", "cl_max", "St"), ("Mean drag", "Lift maximum", "Shedding frequency")):
        for key in ("of", "uni", "clu", "shift", "wide"):
            _series(ax, fam[key], lambda r: r["wall_protocol"], lambda r, k=k: abs(100 * r["rel"][k]), key, hollow=key == "shift")
        ax.set_xscale("log")
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.axhspan(lo, 100 * uref[k], color=plots.GRID, zorder=0, lw=0)
        ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel("Study wall time so far (s)")
    axes[0].axhline(100 * tg["want_rel"], color=plots.INK2, ls="--", lw=1)
    axes[0].axvline(tg["want_wall"], color=plots.INK2, ls="--", lw=1)
    axes[0].set_ylabel("Difference from the reference (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, fontsize=8)
    return plots.to_svg(fig, "cost")


def _res_fig(fam, conv):
    fig, axes = plots.wide(2, 3.5)
    for ax, k, title in zip(axes, ("cd_mean", "cl_max"), ("Mean drag", "Lift maximum")):
        for key in ("uni", "clu", "shift", "wide"):
            _series(ax, fam[key], lambda r: r["D_h"], lambda r, k=k: abs(100 * r["rel"][k]), key, hollow=key == "shift")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Cells across the cylinder's diameter")
        ax.set_xticks([r["D_h"] for r in fam["uni"]] + [fam["clu"][-1]["D_h"]])
        ax.set_xticklabels([f"{r['D_h']:.0f}" for r in fam["uni"]] + [f"{fam['clu'][-1]['D_h']:.0f}"])
        ax.minorticks_off()
    c = fam["clu"]
    xs = np.array([c[0]["D_h"], c[-1]["D_h"]])
    y0 = abs(100 * c[1]["rel"]["cd_mean"])
    axes[0].plot(xs, 0.5 * y0 * (xs / c[1]["D_h"]) ** (-conv["cd_mean"]["p"]), ":", color=plots.INK2, lw=1.2, label="slope of the observed order")
    axes[0].set_ylabel("Difference from the reference (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, fontsize=8)
    return plots.to_svg(fig, "res")


def _time_fig(fam, tg, imex_rows):
    fig, ax = plots.figure(9.0, 4.3)
    shade = [plots.SERIES[3], plots.SERIES[2], plots.SERIES[1], plots.SERIES[0]]
    def cells(rows):
        """Cells of each grid of a study, e.g. '3k, 13k, 54k, 215k cells'."""
        return ", ".join(f"{r['cells'] / 1000:.0f}k" for r in rows) + " cells"
    bars = [("OpenFOAM, as recorded", cells(fam["of"]), [r["wall_s"] for r in fam["of"]]),
            ("OpenFOAM, with stop rule", cells(fam["of"]) + " (estimate)", [r["live_est"] for r in fam["of"]]),
            ("gpucfd, equal cells", cells(fam["uni"]), [r["wall_s"] for r in fam["uni"]]),
            ("gpucfd, clustered", cells(fam["clu"]), [r["wall_s"] for r in fam["clu"]]),
            ("gpucfd, clustered, implicit finest", cells(imex_rows), [r["wall_s"] for r in imex_rows]),
            ("gpucfd, clustered, moved", cells(fam["shift"]), [r["wall_s"] for r in fam["shift"]])]
    names = ["1st grid of a study", "2nd grid", "3rd grid", "4th grid"]
    ylab = ax.get_yaxis_transform()                 # x in axes units, y in data units (bar number)
    for i, (title, sub, vals) in enumerate(bars):
        left = 0.0
        for j, v in enumerate(vals):
            ax.barh(i, v, left=left, color=shade[j], height=0.62, label=names[j] if i == 0 else None)
            left += v
        ax.text(-0.015, i - 0.14, title, transform=ylab, ha="right", va="center", fontsize=9, fontweight="bold", color=plots.INK)
        ax.text(-0.015, i + 0.16, sub, transform=ylab, ha="right", va="center", fontsize=8.5, color=plots.INK2)
    ax.axvline(tg["want_wall"], color=plots.INK2, ls="--", lw=1)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Wall time of the study (s)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return plots.to_svg(fig, "time")


def _step_fig(fam, tol):
    fig, ax = plots.figure(7.6, 3.6)
    cells = {r["level"]: r["cells"] for r in fam["uni"] + fam["clu"]}
    tols = sorted({t for v in tol.values() for t in v["by"] if t <= 1e-8}, reverse=True)
    for key, mk, ls, lab in (("uni", "o", "-", "equal cells"), ("clu", "s", "--", "clustered")):
        for j, t in enumerate(tols):
            pts = [(cells[r["level"]], tol[r["level"]]["by"][t]["perf"]["ms_per_step"]) for r in fam[key] if t in tol[r["level"]]["by"]]
            exp = int(round(np.log10(t)))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + ls, color=plots.SERIES[1 + j], ms=5, lw=1.4,
                    mfc=plots.SERIES[1 + j] if key == "uni" else plots.SURFACE,
                    label=f"gpucfd, {lab}, tolerance $10^{{{exp}}}$")
    ax.plot([r["cells"] for r in fam["of"]], [r["ms_wall"] for r in fam["of"]], "D-", color=plots.SERIES[0], ms=5, lw=1.4,
            label="OpenFOAM (" + ", ".join(str(r["cores"]) for r in fam["of"]) + " threads)")
    c0 = fam["uni"][0]["cells"]
    y0 = tol[fam["uni"][0]["level"]]["by"][tols[0]]["perf"]["ms_per_step"]
    xs = np.array([c0, fam["uni"][-1]["cells"]])
    ax.plot(xs, y0 * xs / c0, ":", color=plots.INK2, lw=1.2, label="cost proportional to the cells")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cells")
    ax.set_ylabel("Wall time per time step (ms)")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return plots.to_svg(fig, "step")


if __name__ == "__main__":
    main()
