"""Render a results.json + figures into a single self-contained HTML report.

render() runs the contract check first and refuses to write a report that violates it.
"""
import json
import os
import platform
import re
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATES = Path(__file__).resolve().parent / "templates"
LETTERS = "abcdefghijklmnop"

# Shared legend for assumption statuses (same in every report).
ASSUMPTION_LEGEND = [
    dict(status="ok", label="✓ Checked", meaning="Tested with the given data, and it holds."),
    dict(status="standard", label="● Standard", meaning="Accepted physics for this situation; not tested by a calculation."),
    dict(status="imposed", label="◆ Imposed", meaning="True by construction of the model. It cannot fail here, "
                                                      "but it limits where the answer applies."),
    dict(status="violated", label="✗ Violated", meaning="Tested with the given data, and it fails: results are not trustworthy."),
]
CHECK_LEGEND = {"ok": "✓ OK", "warn": "⚠ Warning", "fail": "✗ Fail"}


def m(s):
    """Inline math for report text."""
    return rf"\({s}\)"


def fmt(x, sig=4):
    """Number formatting for HTML (non-math) text."""
    if isinstance(x, bool) or x is None:
        return str(x)
    if isinstance(x, int):
        return f"{x:,}"
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e5:
        return f"{x:.{sig}g}"
    mant, exp = f"{x:.{sig - 1}e}".split("e")
    return f"{mant}×10<sup>{int(exp)}</sup>"


def pct(x, sig=3):
    return fmt(100 * x, sig) + " %"


def tex_num(x, sig=4):
    """Number as LaTeX, e.g. 1.004 \\times 10^{-6}. Used inside \\( \\) math."""
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e5:
        return f"{x:.{sig}g}"
    mant, exp = f"{x:.{sig - 1}e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return rf"{mant} \times 10^{{{int(exp)}}}"


def tex_fix(x, decimals=4):
    """Fixed number of decimals (keeps trailing zeros), for table columns that must line up."""
    return f"{x:.{decimals}f}"


def tex_pct(x, sig=3):
    return tex_num(100 * x, sig) + r"\,\%"


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def environment_info():
    """Software and hardware of this run, read from the machine (for reproducibility.environment)."""
    from . import foam
    cpu = platform.processor()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                   if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    mem = ""
    try:
        kb = int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal")))
        mem = f"{kb / 1024 ** 2:.0f} GB visible to Linux"
    except (OSError, StopIteration):
        pass
    try:
        of = "OpenFOAM " + foam.foam_env("WM_PROJECT_VERSION")
    except Exception:                          # noqa: BLE001 - outside WSL
        of = "OpenFOAM (not found)"
    return [["OpenFOAM", of], ["Python", platform.python_version()],
            ["Operating system", f"{platform.system()} {platform.release()}"],
            ["CPU", f"{cpu}, {os.cpu_count()} logical cores"], ["Memory", mem]]


def fig_numbers(results):
    """{figure key: number} in reading order (the order render() uses). Prefer {fig:key} in text."""
    from .contract import FIGURE_SECTIONS
    order = sorted(results["figures"], key=lambda f: FIGURE_SECTIONS.index(f["section"]))
    return {f["key"]: i + 1 for i, f in enumerate(order)}


def _sub_refs(obj, nums):
    from .contract import FIG_REF
    if isinstance(obj, str):
        return FIG_REF.sub(lambda m: str(nums.get(m.group(1), m.group(0))), obj)
    if isinstance(obj, dict):
        return {k: _sub_refs(v, nums) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sub_refs(v, nums) for v in obj]
    return obj


def _svg_width_px(svg):
    m = re.search(r'<svg[^>]*\swidth="([\d.]+)(pt|px)?"', svg)
    if not m:
        return 880
    w = float(m.group(1))
    return round(w * 4 / 3 if (m.group(2) or "pt") == "pt" else w)


def render(results, figures, out_path, case_dir=None):
    """Check the contract, then write results.json and report.html next to out_path.

    Text anywhere in results may refer to figures as {fig:<key>}; it is replaced by the figure number.
    """
    from .contract import check, lint, FIGURE_SECTIONS, DECISION_KINDS

    errors = check(results, case_dir=case_dir, figures=figures)
    if errors:
        raise ValueError("Report contract violated:\n  - " + "\n  - ".join(errors))
    for w in lint(results)[1]:
        print("report warning:", w)
    nums = fig_numbers(results)
    results = _sub_refs(results, nums)
    out_path = Path(out_path)
    (out_path.parent / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

    # Figures in reading order, each shown at its natural size (never enlarged).
    order = sorted(results["figures"], key=lambda f: FIGURE_SECTIONS.index(f["section"]))
    numbered = [dict(f, number=nums[f["key"]], svg=figures[f["key"]], width_px=_svg_width_px(figures[f["key"]]))
                for f in order]

    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), undefined=StrictUndefined, autoescape=False)
    env.filters.update(fmt=fmt, pct=pct, tnum=tex_num)
    groups = {}                                          # decisions grouped by area, first-seen order
    for i, d in enumerate(results["decisions"]):
        groups.setdefault(d["area"], []).append(dict(d, num=i + 1))
    insights = sorted(results["insights"], key=lambda it: str(it.get("logged", "")))
    html = env.get_template("report.html.j2").render(
        r=results, figs=numbered, decision_groups=list(groups.items()), decision_kinds=DECISION_KINDS,
        insights=insights, assumption_legend=ASSUMPTION_LEGEND, check_legend=CHECK_LEGEND)
    out_path.write_text(html, encoding="utf-8")
    return out_path
