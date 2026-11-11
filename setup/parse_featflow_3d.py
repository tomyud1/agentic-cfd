"""Turn the saved FeatFlow DFG 3D pages (setup/fetch_featflow_3d.sh) into the case-5 reference YAML.

Every number is parsed by code from the results page: the steady Re = 20 table (CFX, OpenFOAM and
FeatFlow on four mesh levels) and the reference values quoted above it.
    python setup/parse_featflow_3d.py        (from the repository folder)
"""
import html
import re
from pathlib import Path

import yaml

REF = Path("cases/dfg_cylinder_3d_re20/reference")
src = (REF / "featflow_dfg3d_results.html").read_text(encoding="utf-8", errors="replace")


def text(c):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()


def rows(t):
    return [[text(c) for c in re.findall(r"(?s)<t[dh][^>]*>(.*?)</t[dh]>", tr)]
            for tr in re.findall(r"(?s)<tr.*?</tr>", t)]


plain = text(src)
m = re.search(r"reference values: c ?D = ([\d.]+) and c ?L = ([\d.]+) \[(\d)\]", plain)
cd_ref, cl_ref, cite = float(m.group(1)), float(m.group(2)), m.group(3)
assert cite == "2", "the reference values are expected to cite [2] (Braack and Richter 2006)"

table = next(rows(t) for t in re.findall(r"(?s)<table.*?</table>", src) if "%Err c D" in text(t) and "Software" in text(t))
records, level, cells = [], None, None
for r in table[1:]:
    first, code, cd, cl, ecd, ecl = r[:6]
    if first.startswith("L"):
        level = int(first[1:])
    elif first:
        cells = int(first)
    records.append(dict(level=level, software=code, cd=float(cd), cl=float(cl), err_cd_pct=float(ecd), err_cl_pct=float(ecl)))
cells_by_level = {}
for r in table[1:]:
    if r[0] and not r[0].startswith("L"):
        cells_by_level[len(cells_by_level) + 1] = int(r[0])
for rec in records:
    rec["cells"] = cells_by_level[rec["level"]]

ref = dict(
    source=("E. Bayraktar, O. Mierka and S. Turek, 'Benchmark computations of 3D laminar flow around a cylinder with "
            "CFX, OpenFOAM and FeatFlow', International Journal of Computational Science and Engineering 7(3), "
            "253-266, 2012. Results page of the FeatFlow group (TU Dortmund): http://wwwold.mathematik.tu-dortmund.de/"
            "~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d/dfg_flow3d_results.html, retrieved 2026-11-11."),
    benchmark=("Steady test case 3D-1Z (Re = 20) of Schaefer and Turek, 'Benchmark computations of laminar flow around "
               "a cylinder', Notes on Numerical Fluid Mechanics 52, 547-566, Vieweg 1996."),
    reference_values=dict(
        cd=cd_ref, cl=cl_ref,
        source=("M. Braack and T. Richter, 'Solutions of 3D Navier-Stokes benchmark problems with adaptive finite "
                "elements', Computers and Fluids 35(4), 372-392, 2006 (quoted as reference [2] on the results page)."),
        note="The page gives the values without an uncertainty; the number of printed digits bounds it."),
    meshes=("Hexahedral meshes: a 2D block mesh extruded in z with 4 cell layers and refined twice gives level 1 "
            "(6144 cells); each further level halves the cell size in all three directions (8 times the cells)."),
    openfoam_settings_in_paper=("OpenFOAM 1.6 with icoFoam (transient, PISO); the Re = 20 case was simulated towards "
                                "the steady state. Gauss linear schemes, GAMG with Gauss-Seidel for p (tolerance 1e-7), "
                                "PBiCG with DILU for U (tolerance 1e-6), as listed on the software page."),
    results=records,
)
out = REF / "featflow_dfg3d_re20.yaml"
out.write_text("# Reference data parsed by code from the FeatFlow results page (setup/parse_featflow_3d.py). "
               "Do not edit by hand.\n" + yaml.safe_dump(ref, sort_keys=False, width=110, allow_unicode=True), encoding="utf-8")
print(cd_ref, cl_ref, len(records), "records;", [r for r in records if r["software"] == "OF"])
