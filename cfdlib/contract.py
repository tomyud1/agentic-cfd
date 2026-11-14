"""Report contract: what every agentic-CFD report must contain, and a checker that enforces it.

The full human-readable specification is workflow/references/report-contract.md.
report.render() refuses to write a report that fails this check.

CLI:
  python -m cfdlib.contract results/<case>/results.json [<rendered OpenFOAM case dir>]
  python -m cfdlib.contract --decisions cases/<case>/decisions.yaml <rendered OpenFOAM case dir>
      (during setup: are the decisions complete and do they cover every setting in the case?)
  python -m cfdlib.contract --problem cases/<case>/problem.yaml
      (when a problem statement is written: will the report accept it as given?)
"""
import fnmatch
import json
import re
import sys
from pathlib import Path

from . import foamdict

SECTIONS = ["problem", "theory", "decisions", "preflight", "simulation", "comparison", "insights"]
STATUSES = {"ok", "standard", "imposed", "violated"}
CHECK_STATUSES = {"ok", "warn", "fail"}
VERDICTS = {"PASS", "WARN", "FAIL"}
COMPARISON_TYPES = {"verification", "benchmark", "validation", "consistency"}
FIGURE_SECTIONS = ["problem", "theory", "preflight", "simulation", "convergence", "comparison", "performance"]
INSIGHT_STAGES = {"intake", "theory", "setup", "preflight", "production", "analysis", "report", "tooling", "performance", "gpu"}
INSIGHT_STATUSES = {"open", "resolved"}
# What kind of influence a decision has on the answer (shown as a tag in the report).
DECISION_KINDS = {
    "physics": "Defines the problem that is solved",
    "accuracy": "Sets the size of the numerical error",
    "solver": "Changes how the answer is reached, not the converged answer",
    "output": "Decides what is measured or saved",
}
FIG_REF = re.compile(r"\{fig:([\w.-]+)\}")
_CODE = re.compile(r"<code>.*?</code>", re.S)
DESCRIPTION_MAX = 600                          # characters of problem.description (the overview)


def fig_refs(text):
    """Figure keys a text refers to. A placeholder quoted inside <code> is text about the syntax, not a
    reference (case 7: an insight that quoted one made the check look for a figure of that name)."""
    return FIG_REF.findall(_CODE.sub(" ", text))


def check_problem_file(path):
    """Violations in a problem statement (cases/<name>/problem.yaml) that would only show when the
    solver agent renders its report: checked when the problem is written, the agent is not left with a
    given file it may not change and a contract that refuses it (case 7: description of 760 characters)."""
    import yaml
    path = Path(path)
    p = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    e = []
    _req(p, ["name", "title", "description", "setup", "given", "quantities_of_interest"], "problem.yaml", e)
    n = len(str(p.get("description", "")))         # counted as check() counts it in the report
    if n > DESCRIPTION_MAX:
        e.append(f"description: {n} characters; the report's overview allows {DESCRIPTION_MAX} (1-3 sentences). "
                 "Move the details to `setup`")
    if not isinstance(p.get("setup", []), list):
        e.append("setup: a list of short statements (geometry, boundary conditions, definitions)")
    if "quantities_of_interest" in p and not p["quantities_of_interest"]:
        e.append("quantities_of_interest: list the requested outputs")
    ref = (p.get("reference") or {}).get("file") if isinstance(p.get("reference"), dict) else None
    if ref and not (path.parent / ref).exists():
        e.append(f"reference.file: {ref} not found next to the problem statement")
    return e + control_chars(p, "problem.yaml")

# Settings that cannot change the solution or the report; everything else in 0/, constant/
# and system/ must be covered by a decision's `settings` patterns.
INERT = [
    "*/FoamFile/*", "system/controlDict/startFrom", "system/controlDict/stopAt",
    "system/controlDict/purgeWrite", "system/controlDict/writeCompression",
    "system/controlDict/runTimeModifiable", "system/controlDict/writeFormat",
    "system/controlDict/timeFormat", "0/*/dimensions", "system/blockMeshDict/edges",
    "system/blockMeshDict/mergePatchPairs", "system/decomposeParDict/*",
]


def _req(d, keys, where, errors):
    for k in keys:
        if k not in d:
            errors.append(f"{where}: missing '{k}'")


def check_decisions(decisions, e=None):
    e = [] if e is None else e
    if not decisions:
        e.append("decisions: every modelling and numerical choice must be documented")
    for i, d in enumerate(decisions or []):
        _req(d, ["area", "kind", "choice", "settings", "why", "affects", "alternatives"], f"decisions[{i}]", e)
        if "kind" in d and d["kind"] not in DECISION_KINDS:
            e.append(f"decisions[{i}].kind must be one of {sorted(DECISION_KINDS)}")
    return e


def check(results, case_dir=None, figures=None):
    """Return a list of contract violations (empty list = report is acceptable)."""
    e = []
    r = results
    _req(r, ["title", "date", "verdict", "comparison_type", "summary", "problem", "assumptions", "theory",
             "decisions", "preflight", "simulation", "comparison", "insights", "reproducibility", "figures"], "report", e)
    if e:
        return e
    if r["verdict"] not in VERDICTS:
        e.append(f"verdict must be one of {sorted(VERDICTS)}")
    if r["comparison_type"] not in COMPARISON_TYPES:
        e.append(f"comparison_type must be one of {sorted(COMPARISON_TYPES)}")

    s = r["summary"]
    _req(s, ["headline", "answer_columns", "answers", "findings", "limitations"], "summary", e)
    p = r["problem"]
    _req(p, ["description", "given", "wanted"], "problem", e)
    if not isinstance(p.get("setup", []), list):
        e.append("problem.setup: a list of short statements (geometry, boundary conditions, definitions)")
    if len(str(p.get("description", ""))) > DESCRIPTION_MAX:
        e.append("problem.description: keep it to an overview of 1-3 sentences; put the details in problem.setup "
                 "(a list of short statements)")
    if not p.get("wanted"):
        e.append("problem.wanted: list the requested outputs (lettered)")
    wanted = [w.get("letter") for w in p.get("wanted", [])]
    answered = [a.get("letter") for a in s.get("answers", [])]
    if sorted(wanted) != sorted(answered):
        e.append(f"summary.answers letters {answered} must match problem.wanted letters {wanted}")

    a = r["assumptions"]
    for i, row in enumerate(a.get("rows", [])):
        _req(row, ["name", "check", "status"], f"assumptions.rows[{i}]", e)
        if row.get("status") not in STATUSES:
            e.append(f"assumptions.rows[{i}].status must be one of {sorted(STATUSES)}")
    if not a.get("rows"):
        e.append("assumptions.rows: list the assumptions")

    _req(r["theory"], ["intro", "steps", "derived", "self_checks"], "theory", e)
    for i, d in enumerate(r["theory"].get("derived", [])):
        _req(d, ["label", "symbol", "formula", "value_tex", "unit_tex", "why", "params", "check"], f"theory.derived[{i}]", e)

    check_decisions(r["decisions"], e)

    pf = r["preflight"]
    _req(pf, ["purpose", "runs", "checks", "conclusion"], "preflight", e)
    for i, c in enumerate(pf.get("checks", [])):
        _req(c, ["name", "value", "status", "meaning"], f"preflight.checks[{i}]", e)
        if c.get("status") not in CHECK_STATUSES:
            e.append(f"preflight.checks[{i}].status must be one of {sorted(CHECK_STATUSES)}")
    if not pf.get("runs"):
        e.append("preflight.runs: a short test run is required before production runs")

    _req(r["simulation"], ["model", "meshes", "results", "health"], "simulation", e)
    _req(r["comparison"], ["kind", "definitions", "head", "rows", "findings"], "comparison", e)
    for sec in ("simulation", "comparison"):   # optional further tables of a section: [{title, head, rows, note?}]
        for i, x in enumerate(r[sec].get("tables", []) if isinstance(r[sec], dict) else []):
            _req(x, ["title", "head", "rows"], f"{sec}.tables[{i}]", e)
            if any(len(row) != len(x.get("head", [])) for row in x.get("rows", [])):
                e.append(f"{sec}.tables[{i}]: every row needs as many cells as the head")

    if not r["insights"]:
        e.append("insights: record at least one problem met and how it was solved (or what worked first time and why)")
    for i, it in enumerate(r["insights"]):
        _req(it, ["stage", "logged", "problem"], f"insights[{i}]", e)
        if it.get("stage") not in INSIGHT_STAGES:
            e.append(f"insights[{i}].stage must be one of {sorted(INSIGHT_STAGES)}")
        status = it.get("status", "resolved")
        if status not in INSIGHT_STATUSES:
            e.append(f"insights[{i}].status must be one of {sorted(INSIGHT_STATUSES)}")
        if status == "resolved":
            _req(it, ["diagnosis", "fix", "lesson", "guideline"], f"insights[{i}] (resolved)", e)

    if "performance" in r:                     # optional appendix: run-time data for optimisation
        _req(r["performance"], ["intro", "head", "rows", "notes"], "performance", e)
    rp = r["reproducibility"]
    if not isinstance(rp, dict):
        e.append("reproducibility: must be a dict with commands, files, run_dirs, environment")
    else:
        _req(rp, ["commands", "files", "run_dirs", "environment"], "reproducibility", e)
        for k, fields in (("commands", ["cmd", "what"]), ("files", ["path", "what"]), ("run_dirs", ["path", "what"])):
            for i, x in enumerate(rp.get(k, [])):
                _req(x, fields, f"reproducibility.{k}[{i}]", e)

    keys = set()
    for i, f in enumerate(r["figures"]):
        _req(f, ["key", "section", "caption"], f"figures[{i}]", e)
        keys.add(f.get("key"))
        if f.get("section") not in FIGURE_SECTIONS:
            e.append(f"figures[{i}].section must be one of {FIGURE_SECTIONS}")
        if figures is not None and f.get("key") not in figures:
            e.append(f"figures[{i}]: no SVG supplied for key '{f.get('key')}'")
    if not any(f.get("section") == "problem" for f in r["figures"]):
        e.append("figures: the problem section needs a sketch")
    for where, text in _strings(r):
        for k in fig_refs(text):
            if k not in keys:
                e.append(f"{where}: {{fig:{k}}} refers to no figure")

    e += [f"typesetting: {x}" for x in lint(r)[0]]
    if case_dir is not None:
        if Path(case_dir).is_dir():
            e += uncovered_settings(r["decisions"], case_dir)
        else:                                  # re-rendering from saved data on another machine
            print(f"contract: case directory {case_dir} not found; decision coverage of the OpenFOAM "
                  "settings was not re-checked (it was checked when the report was first made)")
    return e


# ---- typesetting lint ---------------------------------------------------------------------------
_SKIP_KEYS = {"settings", "svg", "cmd", "path", "unit", "environment", "data", "key", "section", "status", "letter",
              "latex", "formula", "check", "symbol", "value_tex", "unit_tex", "tex", "logged", "resolved"}
_MATH_OR_CODE = re.compile(r"\\\(.*?\\\)|\\\[.*?\\\]|<code>.*?</code>|<[^>]+>", re.S)
_E_NOTATION = re.compile(r"(?<![\w.])\d+(?:\.\d+)?[eE][-+]?\d+\b")
_ASCII_MATH = re.compile(r"\w\s*\^\s*[\w{(-]|\w\*\w|\b\d+(?:\.\d+)?\s*x\s*10\b")
_UNICODE_MATH = re.compile(r"[√∂∫∑∏≈≤≥±∞∇ατδνρψωμσΔΩ]")
_NUM_SYMBOL = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s+([B-Zbcdefhijklnopqrtuvwyz])\b")
# "Figure", "Figures 3 and", "Fig." ... right before a figure placeholder (which renders as the number only)
_FIG_LEAD = re.compile(r"(?:Figures?|Figs?\.?)(?:\s|&nbsp;|[(,]|and|or|to|\{fig:[\w.-]+\}|[-–])*$", re.I)


def _strings(obj, where="results"):
    """(location, text) for every string a reader sees."""
    if isinstance(obj, str):
        yield where, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in _SKIP_KEYS:
                yield from _strings(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{where}[{i}]")


def lint(results):
    """(errors, warnings) about math written as plain text. Math belongs in \\( \\) (report.m, tex_num)."""
    errors, warnings = [], []
    for where, text in _strings({k: v for k, v in results.items() if k != "figures"} |
                                {"figures": [{"caption": f.get("caption", "")} for f in results.get("figures", [])]}):
        plain = _MATH_OR_CODE.sub(" ", text)
        for m in _E_NOTATION.finditer(plain):
            errors.append(f"{where}: plain-text number '{m.group()}' (use tex_num inside \\( \\))")
        for m in _ASCII_MATH.finditer(plain):
            errors.append(f"{where}: plain-text math '{m.group()}' (write it as LaTeX inside \\( \\))")
        for m in _UNICODE_MATH.finditer(plain):
            warnings.append(f"{where}: math character '{m.group()}' in plain text (write it as LaTeX inside \\( \\))")
        for m in _NUM_SYMBOL.finditer(plain):
            warnings.append(f"{where}: '{m.group()}' looks like a number times a symbol in plain text "
                            "(write \\(0.009\\,U\\) instead)")
        nocode = _CODE.sub(" ", text)
        for m in FIG_REF.finditer(nocode):
            if not _FIG_LEAD.search(nocode[:m.start()]):
                warnings.append(f"{where}: {m.group()} becomes a bare number in the text; write "
                                f"'Figure {m.group()}'")
    summary = results.get("summary")
    for i, a in enumerate(summary.get("answers", []) if isinstance(summary, dict) else []):
        lines = re.split(r"<br\s*/?>|</?div[^>]*>", str(a.get("diff", ""))) if isinstance(a, dict) else []
        longest = max((len(_MATH_OR_CODE.sub(" ", x).split()) for x in lines), default=0)
        if longest > 10:                           # the solved cases have at most 7 words on a line here
            warnings.append(f"results.summary.answers[{i}].diff: a line of {longest} words; this column does not "
                            "wrap and pushes the table off the page. Give one value here and put the words in `sim`")
    errors += control_chars(results)
    return errors, warnings


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|\t")


def control_chars(obj, where="results"):
    """Control characters anywhere in the text (formulas included). They appear when a backslash is
    eaten on the way into a file: '\\t' of '\\times' becomes a TAB, '\\b' of '\\beta' a backspace."""
    if isinstance(obj, str):
        return [f"{where}: control character {m.group()!r} (a LaTeX backslash was eaten, e.g. \\times -> TAB; "
                "edit files that contain LaTeX with the Edit tool)" for m in _CONTROL.finditer(obj)]
    if isinstance(obj, dict):
        return [x for k, v in obj.items() if k != "svg" for x in control_chars(v, f"{where}.{k}")]
    if isinstance(obj, list):
        return [x for i, v in enumerate(obj) for x in control_chars(v, f"{where}[{i}]")]
    return []


def check_code(folder):
    """Python files of a case folder that do not compile cleanly: a list of messages, empty when fine.
    The usual find is an invalid escape sequence: a LaTeX backslash in a normal string, as in "\\(u\\)"
    written without the r prefix or the doubled backslash. Python only warns about it and the report
    still builds, so it goes unnoticed until a letter after the backslash happens to be a real escape
    (case 17).   python -m cfdlib.contract --code cases/<case>"""
    import warnings
    errs = []
    for p in sorted(Path(folder).glob("*.py")):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                compile(p.read_text(encoding="utf-8"), str(p), "exec")
            except (SyntaxError, Warning) as x:
                errs.append(f"{p.name}: {x}")
    return errs


def uncovered_settings(decisions, case_dir):
    """Settings used by the OpenFOAM case that no decision documents."""
    patterns = [pat for d in decisions for pat in d.get("settings", [])]
    missing, unparsed = [], []
    settings = foamdict.case_settings(case_dir, unparsed)
    errs = [f"could not parse {f}: its settings cannot be checked (simplify the file or extend cfdlib.foamdict)"
            for f in unparsed]
    for key in settings:
        if any(fnmatch.fnmatchcase(key, pat) for pat in INERT):
            continue
        if not any(fnmatch.fnmatchcase(key, pat) for pat in patterns):
            missing.append(key)
    return errs + [f"undocumented OpenFOAM setting (add it to a decision's settings): {k}" for k in missing]


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--decisions"]:
        import yaml
        if len(args) != 3:
            sys.exit("usage: python -m cfdlib.contract --decisions cases/<case>/decisions.yaml <case dir>")
        dec = yaml.safe_load(Path(args[1]).read_text(encoding="utf-8"))
        errs = check_decisions(dec) + uncovered_settings(dec, args[2]) + control_chars(dec, "decisions")
        print("\n".join(errs) if errs else f"Decisions OK: {len(dec)} decisions cover every setting in {args[2]}")
        sys.exit(1 if errs else 0)
    if args[:1] == ["--code"]:
        if len(args) != 2:
            sys.exit("usage: python -m cfdlib.contract --code cases/<case>")
        errs = check_code(args[1])
        print("\n".join(errs) if errs else f"Code OK: every Python file in {args[1]} compiles without warnings")
        sys.exit(1 if errs else 0)
    if args[:1] == ["--problem"]:
        if len(args) != 2:
            sys.exit("usage: python -m cfdlib.contract --problem cases/<case>/problem.yaml")
        errs = check_problem_file(args[1])
        print("\n".join(errs) if errs else f"Problem statement OK: {args[1]}")
        sys.exit(1 if errs else 0)
    res = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    errs = check(res, args[1] if len(args) > 1 else None)
    warns = lint(res)[1]
    print("\n".join(errs) if errs else "Contract OK")
    if warns:
        print("Warnings:\n  - " + "\n  - ".join(warns))
    sys.exit(1 if errs else 0)
