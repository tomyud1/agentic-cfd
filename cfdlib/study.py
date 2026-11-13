"""A second, focused report for a study that compares tools or settings on a solved case: GPU against CPU,
one solver against another, one scheme against another.

The main report (report.render) answers the physics; this one answers "how do the two compare, was the
comparison fair, and is it still the same answer?". Same rules: every number comes from code, figures
pass the layout check, text is typeset as LaTeX, and render_study() refuses to write a study that
breaks the contract.

    from cfdlib import study
    verdict, reasons = study.verdict(fail, warn)          # two lists of plain sentences
    study.render_study(s, svgs, "results/<case>/gpu_study.html", insights)

The keys of s, required and optional, are listed in workflow/references/report-contract.md ("Study
reports"); check_study() below is the rule itself. In short:
  title, date, question, verdict, summary, setup, method, results, same_answer, breakdown,
  reproducibility, figures                                          required, as before
  kicker, verdict_label, sections[] (your own sections), a title per section, summary.reasons[]
  (required when the verdict is not PASS), summary.plain (the headline as plain text, for the case
  catalogue), and per table: question, result, meaning, figure, nowrap       optional

Complete examples: cases/dfg_cylinder_3d_re20/gpu_study.py (one solver, CPU against GPU) and
cases/dfg_cylinder_re100/gpu_study.py (two solvers, with its own sections and section titles).
Run records of either tool in one shape: cfdlib.runrecord. Numbers as table cells: report.m_pct,
m_times, m_pow10, m_fix, m_sig.
"""
import json
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .contract import FIG_REF, _req, _strings, fig_refs, lint
from .report import TEMPLATES, _svg_width_px, keep_run_record

CORE = ["setup", "method", "results", "same_answer", "breakdown"]          # always there, in this order
TITLES = dict(setup="What is compared", method="Method", results="Results", same_answer="Same answer?",
              breakdown="Where the time goes")
IDS = dict(same_answer="same")                                              # anchor of a section when it is not its key
RESERVED = {"summary", "setup", "method", "results", "same", "same_answer", "breakdown", "insights", "repro"}
WITH_TABLES = ["results", "same_answer", "breakdown"]
MARK = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}
LABELS = {"PASS": "Fair comparison, same answer", "WARN": "Usable, with caveats", "FAIL": "Not a valid comparison"}
STAGES = ("performance", "gpu")
KICKER = "performance study"
WIDE_TABLE = 10        # columns; above this a table scrolls sideways in the report column
NOWRAP_CHARS = 34      # a first column whose longest cell is at most this long is kept on one line


def verdict(fail=(), warn=()):
    """The verdict of a study from the reasons its script collected, so that every study decides alike.

    fail: what makes the comparison invalid (a run failed a health check, the answers differ, the setup
    was unfair). warn: what limits it (a rule missed for one output, a check not made). Both are lists of
    plain sentences without a full stop. Returns (verdict, reasons): FAIL if anything is in fail, else
    WARN if anything is in warn, else PASS; reasons = fail + warn, to pass on as summary.reasons, where
    the page shows them under the headline (the contract asks for them when the verdict is not PASS)."""
    fail, warn = [str(x) for x in fail], [str(x) for x in warn]
    return ("FAIL" if fail else "WARN" if warn else "PASS"), fail + warn


def section_order(s):
    """The body sections in reading order: [dict(key, id, kind, title, extra)]. The five core sections,
    with the study's own sections (s['sections']) placed after the core section each names in `after`
    (default: after the last core section)."""
    out = []
    extras = s.get("sections") or []
    for key in CORE:
        out.append(dict(key=key, id=IDS.get(key, key), kind=key, title=(s.get(key) or {}).get("title") or TITLES[key], extra=None))
        out += [dict(key=x["key"], id=x["key"], kind="extra", title=x.get("title", x["key"]), extra=x)
                for x in extras if x.get("after", CORE[-1]) == key]
    return out


def _tables(s):
    """(section key, where, table) for every table of the study, in reading order."""
    for sec in section_order(s):
        d = sec["extra"] if sec["kind"] == "extra" else s.get(sec["key"]) or {}
        if sec["key"] == "same_answer":
            yield sec["key"], "same_answer", d
        for i, t in enumerate(d.get("tables") or []):
            yield sec["key"], f"{sec['key']}.tables[{i}]", t


def figure_order(s):
    """Figure keys in reading order. Within a section: the figures no table claims come first, in the
    order of s['figures']; a figure named by a table (table['figure']) sits right above that table."""
    claimed = {t.get("figure") for _, _, t in _tables(s) if isinstance(t, dict) and t.get("figure")}
    out = []
    for sec in section_order(s):
        out += [f["key"] for f in s["figures"] if f.get("section") == sec["key"] and f.get("key") not in claimed]
        out += [t["figure"] for k, _, t in _tables(s) if k == sec["key"] and isinstance(t, dict) and t.get("figure")]
    return [k for i, k in enumerate(out) if k not in out[:i]]


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
    if s["verdict"] != "PASS" and not s["summary"].get("reasons"):
        e.append("summary.reasons: a verdict other than PASS needs its reasons (one plain sentence each; study.verdict returns them)")
    plain = s["summary"].get("plain")
    if plain is not None and (not isinstance(plain, str) or re.search(r"<|\\\(|\\[a-zA-Z]", plain) or len(plain) > 500):
        e.append("summary.plain: plain text only (no HTML, no LaTeX), at most 500 characters: it is shown in lists outside the page")
    _req(s["setup"], ["intro", "hardware", "software", "fairness"], "setup", e)
    _req(s["method"], ["intro", "choices"], "method", e)
    for i, c in enumerate(s["method"].get("choices", [])):
        _req(c, ["choice", "why", "alternatives"], f"method.choices[{i}]", e)
    _req(s["results"], ["intro", "tables", "notes"], "results", e)
    _req(s["same_answer"], ["intro", "head", "rows", "ok", "verdict"], "same_answer", e)
    if s["verdict"] == "PASS" and not s["same_answer"].get("ok"):
        e.append("verdict PASS needs same_answer.ok: a faster run that gives a different answer is not a speed-up")
    _req(s["breakdown"], ["intro", "tables", "notes"], "breakdown", e)
    seen = set()
    for i, x in enumerate(s.get("sections") or []):
        _req(x, ["key", "title", "intro", "tables", "notes"], f"sections[{i}]", e)
        k = x.get("key")
        if k in RESERVED or k in seen or not re.fullmatch(r"[a-z][a-z0-9_-]*", str(k)):
            e.append(f"sections[{i}].key {k!r}: a new lower-case name (not one of {sorted(RESERVED)})")
        seen.add(k)
        if x.get("after", CORE[-1]) not in CORE:
            e.append(f"sections[{i}].after must be one of {CORE}")
    _req(s["reproducibility"], ["commands", "files", "run_dirs", "environment"], "reproducibility", e)
    if e:
        return e
    places = WITH_TABLES + ["setup"] + [x["key"] for x in s.get("sections") or []]
    keys, where_fig = set(), {}
    for i, f in enumerate(s["figures"]):
        _req(f, ["key", "section", "caption"], f"figures[{i}]", e)
        if f.get("section") not in places:
            e.append(f"figures[{i}].section must be one of {places}")
        keys.add(f.get("key"))
        where_fig[f.get("key")] = f.get("section")
        if figures is not None and f.get("key") not in figures:
            e.append(f"figures[{i}]: no SVG passed for key {f.get('key')!r}")
    if not any(f.get("section") == "results" for f in s["figures"]):
        e.append("figures: the results section needs at least one figure (for example time against cells, or error against cost)")
    used = set()
    for sec, where, t in _tables(s):
        _req(t, ["head", "rows"], where, e)
        if any(len(r) != len(t.get("head", [])) for r in t.get("rows", [])):
            e.append(f"{where}: every row needs one cell per column")
        k = t.get("figure")
        if k is not None:
            if where_fig.get(k) != sec:
                e.append(f"{where}.figure {k!r}: name a figure whose section is {sec!r}")
            if k in used:
                e.append(f"{where}.figure {k!r}: already placed at another table")
            used.add(k)
        if any(not isinstance(c, int) or not 0 <= c < len(t.get("head", [])) for c in t.get("nowrap") or []):
            e.append(f"{where}.nowrap: column numbers, counted from 0")
    for where, text in _strings(s):
        for k in fig_refs(text):
            if k not in keys:
                e.append(f"{where}: {{fig:{k}}} refers to no figure")
    e += [f"typesetting: {x}" for x in lint(s)[0]]
    return e


def warnings(s):
    """What does not break the contract but will read badly (render_study prints these)."""
    w = list(lint(s)[1])
    for _, where, t in _tables(s):
        n = len(t.get("head", []))
        if n > WIDE_TABLE:
            w.append(f"{where}: {n} columns; above about {WIDE_TABLE} a table scrolls sideways in the report column and the "
                     "reader does not see it: split it or drop columns")
        if t.get("note") and re.search(r"<b>\s*(Result|What it means)\.?\s*</b>", str(t["note"])):
            w.append(f"{where}.note: write the result and its meaning in the table's own fields `result` and `meaning`")
    return w


_TAG = re.compile(r"<[^>]+>")
_MATH = re.compile(r"\\\((.*?)\\\)")


def _plain_len(cell):
    """Rough number of characters a cell shows (tags dropped, math counted by its source without commands)."""
    text = _MATH.sub(lambda mm: re.sub(r"\\[a-zA-Z]+|[{}\\,]", "", mm.group(1)), str(cell))
    return max((len(part) for part in re.split(r"<br\s*/?>", _TAG.sub(lambda mm: mm.group(0) if mm.group(0).startswith("<br") else "", text))), default=0)


def _view(s):
    """The study as the template shows it: every table gets the list of columns kept on one line
    (table['nowrap'], or the first column when its cells are short), sections get their numbers."""
    def table(t):
        nw = t.get("nowrap")
        if nw is None:
            nw = [0] if t.get("rows") and max(_plain_len(r[0]) for r in t["rows"]) <= NOWRAP_CHARS else []
        return dict(t, nowrap=list(nw))

    v = dict(s)
    for key in WITH_TABLES:
        v[key] = dict(s[key], tables=[table(t) for t in s[key].get("tables", [])])
    v["same_answer"] = table(v["same_answer"])
    secs = []
    for i, sec in enumerate(section_order(s)):
        x = sec["extra"]
        secs.append(dict(sec, num=i + 1, extra=dict(x, tables=[table(t) for t in x["tables"]]) if x else None))
    v["secs"] = secs
    v["n_insights"] = len(secs) + 1
    return v


def render_study(s, figures, out_path, insights=()):
    """Check the study contract, then write study.json and the HTML next to out_path.
    insights: the insight entries to show, as a list. Entries with stage performance or gpu are kept;
    pass a shorter list to show fewer (filter it in the script). Each is shown folded to its problem
    line, open ones unfolded.
    On a re-render (CFD_RERENDER=1) the date and machine record of the replaced JSON are kept."""
    s = keep_run_record(s, Path(out_path).parent / (Path(out_path).stem + ".json"))
    errors = check_study(s, figures)
    if errors:
        raise ValueError("Study contract violated:\n  - " + "\n  - ".join(errors))
    for w in warnings(s):
        print("study warning:", w)
    order = figure_order(s)
    nums = {k: i + 1 for i, k in enumerate(order)}

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
    by_key = {f["key"]: f for f in s["figures"]}
    claimed = {t["figure"] for _, _, t in _tables(s) if t.get("figure")}
    figs = {k: dict(by_key[k], number=nums[k], svg=figures[k], width_px=_svg_width_px(figures[k]), at_table=k in claimed) for k in order}
    shown = sorted((it for it in insights if it.get("stage") in STAGES), key=lambda it: str(it.get("logged", "")))
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), undefined=StrictUndefined, autoescape=False)
    label = MARK[s["verdict"]] + " " + (s.get("verdict_label") or LABELS[s["verdict"]])
    html = env.get_template("study.html.j2").render(r=dict(_view(s), verdict_label=label, kicker=s.get("kicker") or KICKER),
                                                    figs=figs, insights=shown)
    out_path.write_text(html, encoding="utf-8")
    return out_path
