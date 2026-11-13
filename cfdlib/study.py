"""A second, focused report for a performance study (e.g. GPU vs CPU) of a case.

The main report (report.render) answers the physics; this one answers "how fast, on what hardware,
and is it still the same answer?". Same rules: every number comes from code, figures pass the layout
check, text is typeset as LaTeX, and render_study() refuses to write a study that breaks the contract.

    from cfdlib import study
    study.render_study(s, svgs, "results/<case>/gpu_study.html", insights)

Keys of s (all required unless marked optional):
  title, date, question                      the question the study answers, in one sentence
  verdict                                    PASS (fair comparison, same answer), WARN (usable, caveats),
                                             FAIL (comparison not valid: different answers or unfair setup)
  summary: headline, answers[{label, value, note}], findings[], limitations[], recommendation (optional)
  setup: intro, hardware[[name, value]], software[[name, value]], fairness[] (rules that make it fair)
  method: intro, choices[{choice, why, alternatives}]
  results: intro, tables[{title, head[], rows[][], note (optional)}], notes[]
  same_answer: intro, head[], rows[][], ok (bool), verdict (text)
  breakdown: intro, tables[...], notes[]     where the time goes (solver share, copies, assembly)
  reproducibility: commands[{cmd, what}], files[{path, what}], run_dirs[{path, what}], environment, notes (optional)
  figures[{key, section, caption}]           section: setup | results | breakdown; refer to them as {fig:key}
"""
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .contract import FIG_REF, _req, _strings, lint
from .report import TEMPLATES, _svg_width_px, keep_run_record

SECTIONS = ["setup", "results", "breakdown"]
LABELS = {"PASS": "✓ Fair comparison, same answer", "WARN": "⚠ Usable, with caveats", "FAIL": "✗ Not a valid comparison"}
STAGES = ("performance", "gpu")


def check_study(s, figures=None):
    e = []
    _req(s, ["title", "date", "question", "verdict", "summary", "setup", "method", "results", "same_answer",
             "breakdown", "reproducibility", "figures"], "study", e)
    if e:
        return e
    if s["verdict"] not in LABELS:
        e.append(f"verdict must be one of {sorted(LABELS)}")
    _req(s["summary"], ["headline", "answers", "findings", "limitations"], "summary", e)
    for i, a in enumerate(s["summary"].get("answers", [])):
        _req(a, ["label", "value", "note"], f"summary.answers[{i}]", e)
    _req(s["setup"], ["intro", "hardware", "software", "fairness"], "setup", e)
    _req(s["method"], ["intro", "choices"], "method", e)
    for i, c in enumerate(s["method"].get("choices", [])):
        _req(c, ["choice", "why", "alternatives"], f"method.choices[{i}]", e)
    _req(s["results"], ["intro", "tables", "notes"], "results", e)
    for name in ("results", "breakdown"):
        for i, t in enumerate(s[name].get("tables", [])):
            _req(t, ["head", "rows"], f"{name}.tables[{i}]", e)
            if any(len(r) != len(t.get("head", [])) for r in t.get("rows", [])):
                e.append(f"{name}.tables[{i}]: every row needs one cell per column")
    _req(s["same_answer"], ["intro", "head", "rows", "ok", "verdict"], "same_answer", e)
    if s["verdict"] == "PASS" and not s["same_answer"].get("ok"):
        e.append("verdict PASS needs same_answer.ok: a faster run that gives a different answer is not a speed-up")
    _req(s["breakdown"], ["intro", "tables", "notes"], "breakdown", e)
    _req(s["reproducibility"], ["commands", "files", "run_dirs", "environment"], "reproducibility", e)
    keys = set()
    for i, f in enumerate(s["figures"]):
        _req(f, ["key", "section", "caption"], f"figures[{i}]", e)
        if f.get("section") not in SECTIONS:
            e.append(f"figures[{i}].section must be one of {SECTIONS}")
        keys.add(f.get("key"))
        if figures is not None and f.get("key") not in figures:
            e.append(f"figures[{i}]: no SVG passed for key {f.get('key')!r}")
    if not any(f.get("section") == "results" for f in s["figures"]):
        e.append("figures: the results section needs at least one figure (time per step or speed-up versus cells)")
    for where, text in _strings(s):
        for k in FIG_REF.findall(text):
            if k not in keys:
                e.append(f"{where}: {{fig:{k}}} refers to no figure")
    e += [f"typesetting: {x}" for x in lint(s)[0]]
    return e


def render_study(s, figures, out_path, insights=()):
    """Check the study contract, then write study.json and the HTML next to out_path.
    insights: the case's insight entries; those with stage performance or gpu are shown.
    On a re-render (CFD_RERENDER=1) the date and machine record of the replaced JSON are kept."""
    s = keep_run_record(s, Path(out_path).parent / (Path(out_path).stem + ".json"))
    errors = check_study(s, figures)
    if errors:
        raise ValueError("Study contract violated:\n  - " + "\n  - ".join(errors))
    for w in lint(s)[1]:
        print("study warning:", w)
    order = sorted(s["figures"], key=lambda f: SECTIONS.index(f["section"]))
    nums = {f["key"]: i + 1 for i, f in enumerate(order)}

    def sub(o):
        if isinstance(o, str):
            return FIG_REF.sub(lambda m: str(nums.get(m.group(1), m.group(0))), o)
        if isinstance(o, dict):
            return {k: sub(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sub(v) for v in o]
        return o

    s = sub(s)
    out_path = Path(out_path)
    (out_path.parent / (out_path.stem + ".json")).write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    figs = [dict(f, number=nums[f["key"]], svg=figures[f["key"]], width_px=_svg_width_px(figures[f["key"]])) for f in order]
    shown = sorted((it for it in insights if it.get("stage") in STAGES), key=lambda it: str(it.get("logged", "")))
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), undefined=StrictUndefined, autoescape=False)
    html = env.get_template("study.html.j2").render(r=dict(s, verdict_label=LABELS[s["verdict"]]), figs=figs, insights=shown)
    out_path.write_text(html, encoding="utf-8")
    return out_path
