"""Turn the saved FeatFlow DFG 2D-2 page (saved in reference/featflow_dfg2d2_page.html) into the case-4 reference YAML (tables parsed by code)."""
import html, re
from pathlib import Path
import yaml
src = Path("cases/dfg_cylinder_re100/reference/featflow_dfg2d2_page.html").read_text(encoding="utf-8", errors="replace")
tables = re.findall(r"(?s)<table.*?</table>", src)
def rows(t):
    out = []
    for tr in re.findall(r"(?s)<tr.*?</tr>", t):
        cells = [html.unescape(re.sub(r"<[^>]+>", " ", c)).strip() for c in re.findall(r"(?s)<t[dh][^>]*>(.*?)</t[dh]>", tr)]
        out.append([re.sub(r"\s+", " ", c) for c in cells])
    return out
T = [rows(t) for t in tables]
def find(head_word):
    return next(t for t in T if t and any(head_word in c for c in t[0]))
def num(x):
    return float(x.replace("E", "e"))
cd, cl, st = find("min(C_D)"), find("min(C_L)"), find("1/f")
def recs(t, keys):
    return [dict(level=int(r[0]), dt=r[1], **{k: num(v) for k, v in zip(keys, r[2:])}) for r in t[1:]]
ref = dict(
    source=("FeatFlow group (TU Dortmund), 'DFG benchmark 2D-2 (RE100, periodic)', "
            "http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html, retrieved 2026-11-11. "
            "Benchmark defined in: Schaefer, Turek, 'Benchmark computations of laminar flow around a cylinder', "
            "Flow Simulation with High-Performance Computers II, Notes on Numerical Fluid Mechanics 52, 547-566, Vieweg 1996."),
    method=("Finite elements Q2/P1disc in space, no stabilisation; Crank-Nicolson in time with time step k = 1/100, 1/200, 1/400; "
            "mesh levels 3-6 by regular refinement (level 6: 133120 elements, 667264 unknowns). Values from one shedding cycle "
            "[t0, t1] between t = 25 s and 30 s; t0 = time of minimum lift; mean = (max+min)/2, amp = max-min."),
    notes=["The frequency 1/f in these tables is resolved only to about the time step k, so St differs between "
           "rows in steps set by k (e.g. 0.30303 for 1/f = 0.33 s). Treat its spread as part of the reference uncertainty.",
           "No single 'exact' value: the spread between levels and time steps indicates the reference's own uncertainty."],
    drag=recs(cd, ["cd_min", "cd_max", "cd_mean", "cd_amp"]),
    lift=recs(cl, ["cl_min", "cl_max", "cl_mean", "cl_amp"]),
    period=recs(st, ["period", "St"]),
)
out = Path("cases/dfg_cylinder_re100/reference/featflow_dfg2d2_re100.yaml")
out.write_text("# Reference data parsed by code from the FeatFlow benchmark page (tables 1-3). Do not edit by hand.\n"
               + yaml.safe_dump(ref, sort_keys=False, width=110), encoding="utf-8")
print(len(ref["drag"]), len(ref["lift"]), len(ref["period"])); print(ref["drag"][-1], ref["lift"][-1], ref["period"][-1])
