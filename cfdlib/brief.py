"""A one-page progress report on a case: the goal, the problem, and the result as two tables side by side,
accuracy and speed, one column per tool.

It is the page to show first; the case report and its study reports hold the detail and are linked from it.
Same rules as there: every number comes from code, text is typeset as LaTeX (contract.lint), and
render_brief() refuses a page that breaks the rule below.

    python -m cfdlib.brief [folder ...]        # the pages of the cases in cases/briefs.yaml (all, or those named)

Two ways to make a page:
  - from a solved case's own records (from_case): the words (goal, story, requirements, what it means, limits)
    are in cases/briefs.yaml; the numbers are cells the case's report and studies already hold, taken from
    their JSON files. No solver runs.
  - by a script of its own when the page needs more, named in cases/briefs.yaml as `script`
    (dev/compare/turbulent_channel_re180/brief.py).

Keys of a page b (render_brief; all required unless marked):
  title, date
  goal            one sentence: what the case is for
  story           one or two sentences: which tool did what, in which order
  problem         dict(requirements[]): what is given and what is asked, as short bullets (at most 8)
  results         dict(headline, meaning[], limits[]): the answer in two or three sentences, what it means,
                  and what the comparison does not show
  accuracy        dict(title, head[], rows[][], note (optional)): one row per requirement of the problem,
                  one column per tool or grid; a cell is text (miss() marks a value outside its limit,
                  wait() a value that is not there)
  speed           dict(title, head[], rows[][], note (optional))
  links[]         dict(href, text): the detailed reports
  verdict         optional: PASS / WARN / FAIL of the case, shown next to the title

A case's entry in cases/briefs.yaml (from_case):
  goal, story, requirements[], meaning[], limits[]          the words
  gpucfd          what to write in gpucfd's cells when it has no numbers ("not run yet", "cannot yet")
  record          optional: the JSON record of gpucfd's runs of this same case (dev/compare/<folder>/gpucfd.json,
                  written by the script next to it; keys below). Its values fill gpucfd's column.
  accuracy        rows: {letter: short name} picks answers of the case's summary (results.json) for OpenFOAM's
                  column; or pick: a ready table (below). Optional title, note.
  speed           performance: [run names] takes those rows of the report's cost table (cells, cores, wall
                  time); or data: {list or files, label, cells, wall, cores} takes them from results.json's data
                  (list) or from run records (files: JSON paths under the repository root); or
                  pick: a ready table; or text: one sentence when nothing was recorded. Optional title, note.
  pick            {file, table, columns[], head[], rows[] (optional)}: columns (by number) of the table with
                  that title in a report or study JSON under results/, with new headings; rows keeps the
                  rows whose first cell is listed.
  sentences       optional: how many sentences of the report's own headline to show (default 2)
  also            optional: a study JSON of the same case whose plain headline is added to the result's headline
  links           optional, [{href, text}]; default: the case's report and study pages

A record of gpucfd's runs (record:), written by code, never by hand:
  answers         {letter: {diff or sim: the finished cell, formed as the case's report forms OpenFOAM's}}
  runs            [{wall_s, cells, ...}]: one per run. With speed data: a run fills gpucfd's cell of the row whose
                  label it gives (the same label pattern, so a run carries the keys the pattern names). A run
                  with a label of its own that matches no row gets a row of its own (key: label).
  plain           optional: gpucfd's result in one sentence, added to the page's headline
  busy_cpu        true if another solver was running while the times were taken: the page then says so
  startup_s       optional: the one-off start of the GPU library, not in the run times: the page says so
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .contract import _req, lint
from .report import TEMPLATES, keep_run_record

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "cases" / "briefs.yaml"
_SUB = re.compile(r"<div class='sub'>.*?</div>", re.S)
_TAG = re.compile(r"<[^>]+>")
DASH = "–"


def miss(cell):
    """A value outside its limit."""
    return f'<span class="miss">{cell}</span>'


def wait(text="running"):
    """A value that is not there (yet)."""
    return f'<span class="wait">{text}</span>'


def check_brief(b):
    e = []
    _req(b, ["title", "date", "goal", "story", "problem", "results", "accuracy", "speed", "links"], "brief", e)
    if e:
        return e
    _req(b["problem"], ["requirements"], "problem", e)
    _req(b["results"], ["headline", "meaning", "limits"], "results", e)
    for key in ("accuracy", "speed"):
        t = b[key]
        _req(t, ["title", "head", "rows"], key, e)
        if any(len(r) != len(t.get("head", [])) for r in t.get("rows", [])):
            e.append(f"{key}: every row needs one cell per column")
        if len(t.get("head", [])) > 6:
            e.append(f"{key}: at most 6 columns; the two tables stand side by side")
    if b.get("verdict") is not None and b["verdict"] not in ("PASS", "WARN", "FAIL"):
        e.append("verdict must be PASS, WARN or FAIL")
    if not b["problem"].get("requirements") or len(b["problem"]["requirements"]) > 8:
        e.append("problem.requirements: one to eight short bullets")
    for where, text in ((k, json.dumps(b[k], ensure_ascii=False)) for k in ("results", "accuracy", "speed", "problem")):
        if "{fig:" in text or re.search(r"\bFigure \d", text):
            e.append(f"{where}: refers to a figure of another report; a one-page report has none")
    e += [f"typesetting: {x}" for x in lint(b)[0]]
    return e


def render_brief(b, sketch_svg, out_path):
    """Check the page, then write <name>.json and the HTML. On a re-render (CFD_RERENDER=1) the date of the
    replaced JSON is kept."""
    out_path = Path(out_path)
    b = keep_run_record(b, out_path.with_suffix(".json"))
    errors = check_brief(b)
    if errors:
        raise ValueError(f"Brief contract violated ({out_path}):\n  - " + "\n  - ".join(errors))
    for w in lint(b)[1]:
        print("brief warning:", w)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(".json").write_text(json.dumps(b, indent=1, ensure_ascii=False), encoding="utf-8")
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), undefined=StrictUndefined, autoescape=False)
    out_path.write_text(env.get_template("brief.html.j2").render(r=b, sketch=sketch_svg), encoding="utf-8")
    return out_path


# ------------------------------------------------------------------------------------------ from a case's records
def short(cell):
    """A cell of a report without its small print."""
    return _SUB.sub("", str(cell)).strip()


def plain(cell):
    return re.sub(r"\s+", " ", _TAG.sub(" ", str(cell))).strip()


def sketch_of(folder):
    """The problem sketch of a case, taken from its rendered report."""
    html = (ROOT / "results" / folder / "report.html").read_text(encoding="utf-8")
    m = re.search(r'<figure class="problem"[^>]*>\s*(<svg.*?</svg>)', html, re.S)
    if not m:
        raise ValueError(f"{folder}: no problem sketch found in report.html")
    return m.group(1)


def _table(data, title):
    """The table with that title anywhere in a report or study JSON ('performance': the report's cost table)."""
    if title == "performance":
        return data["performance"]
    found = []

    def walk(o):
        if isinstance(o, dict):
            if "head" in o and "rows" in o and (o.get("title") == title or o.get("question") == title):
                found.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(data)
    if len(found) != 1:
        raise ValueError(f"table {title!r}: found {len(found)} times")
    return found[0]


def pick(spec):
    """A ready table: (head, rows) from columns of a table in a JSON under the repository root."""
    t = _table(json.loads((ROOT / spec["file"]).read_text(encoding="utf-8")), spec["table"])
    rows = t["rows"]
    if spec.get("rows"):
        by = {plain(r[0]): r for r in rows}
        missing = [n for n in spec["rows"] if n not in by]
        if missing:
            raise ValueError(f"pick {spec['table']!r}: no rows {missing}; there are {list(by)}")
        rows = [by[n] for n in spec["rows"]]
    return list(spec["head"]), [[short(r[c]) for c in spec["columns"]] for r in rows]


def _col(head, *names):
    for i, h in enumerate(head):
        if plain(h) in names:
            return i
    raise ValueError(f"no column named one of {names} in {[plain(h) for h in head]}")


def seconds(w):
    """A run time as text, with two digits that mean something."""
    w = float(w)
    return f"{w:.2f} s" if w < 1 else f"{w:.1f} s" if w < 10 else f"{w:.0f} s"


def from_case(folder, s):
    """The page of a solved case from its records and its entry s of cases/briefs.yaml. Returns (b, sketch)."""
    res = json.loads((ROOT / "results" / folder / "results.json").read_text(encoding="utf-8"))
    no_gpu = wait(s["gpucfd"])
    rec = json.loads((ROOT / s["record"]).read_text(encoding="utf-8")) if s.get("record") else None   # gpucfd's runs of this case
    gpu_runs = list(rec["runs"]) if rec else []
    gpu_time = lambda run: f"{seconds(run['wall_s'])} on the GPU"
    # accuracy
    a = s["accuracy"]
    if "pick" in a:
        head, rows = pick(a["pick"])
    else:
        by = {x["letter"]: x for x in res["summary"]["answers"]}
        mine = (rec or {}).get("answers", {})
        gpu = lambda k: (short(mine[k].get("diff", "")) or short(mine[k]["sim"])) if k in mine else no_gpu
        head = ["Requirement", "OpenFOAM", "gpucfd"] if rec else ["Requirement", "OpenFOAM: difference from the reference", "gpucfd"]
        rows = [[name, short(by[k]["diff"]) or short(by[k]["sim"]), gpu(k)] for k, name in a["rows"].items()]
    accuracy = dict(title=a.get("title", "Accuracy: difference from the reference" if rec else "Accuracy"),
                    head=head, rows=rows, note=a.get("note", ""))
    # speed
    sp = s["speed"]
    if "pick" in sp:
        head, rows = pick(sp["pick"])
    elif "performance" in sp:
        t = res["performance"]
        ic, ik, iw = _col(t["head"], "Cells"), _col(t["head"], "Cores", "Ranks"), _col(t["head"], "Wall (s)")
        by = {plain(r[0]): r for r in t["rows"]}
        head = ["Run", "Cells", "OpenFOAM", "gpucfd"]
        other = DASH if any("label" in run for run in gpu_runs) else no_gpu      # gpucfd ran on grids of its own
        rows = [[n, plain(by[n][ic]), f"{plain(by[n][iw])} s on {plain(by[n][ik])} core{'s' if plain(by[n][ik]) != '1' else ''}", other]
                for n in sp["performance"]]
    elif "data" in sp:
        d = sp["data"]
        head = ["Run"] + (["Cells"] if "cells" in d else []) + ["OpenFOAM", "gpucfd"]
        mine = {d["label"].format(**run): run for run in gpu_runs if "label" not in run}
        source = ([json.loads((ROOT / f).read_text(encoding="utf-8")) for f in d["files"]] if "files" in d
                  else res["data"][d["list"]])
        rows = []
        for r in source:
            label = d["label"].format(**r)
            cells = [f"{int(eval(d['cells'], {}, dict(r))):,}"] if "cells" in d else []            # noqa: S307  (our own spec file)
            rows.append([label] + cells + [f"{seconds(r[d['wall']])} on {d.get('cores', 1)} core",
                                           gpu_time(mine.pop(label)) if label in mine else no_gpu])
        if mine:
            raise ValueError(f"{folder}: gpucfd runs {list(mine)} match no row of the speed table")
    else:
        head, rows = ["", "OpenFOAM", "gpucfd"], [["Run times", sp["text"], no_gpu]]
    for run in gpu_runs:                                          # gpucfd's runs on grids of their own
        if "label" in run:
            rows.append([run["label"]] + ([f"{int(run['cells']):,}"] if "Cells" in head else []) + [DASH, gpu_time(run)])
    notes = [sp.get("note", "")]
    if rec and rec.get("startup_s") is not None:
        notes.append(f"gpucfd's times run from creating the solver to the final field; starting the GPU library "
                     f"adds {seconds(rec['startup_s'])} once.")
    if rec and rec.get("busy_cpu"):
        notes.append("gpucfd's times were taken while another solver was using the CPU, so they are upper bounds.")
    speed = dict(title=sp.get("title", "Speed"), head=head, rows=rows, note=" ".join(n for n in notes if n))
    links = s.get("links") or ([dict(href="report.html", text="the full report of this case")] +
                               [dict(href=p.name, text="its study report") for p in sorted((ROOT / "results" / folder).glob("*_study.html"))])
    sentences = re.split(r"(?:(?<=[.!?])|(?<=[.!?]</b>))\s+(?=[A-Z<])", short(res["summary"]["headline"]))
    headline = " ".join(sentences[:int(s.get("sentences", 2))])
    if s.get("also"):                                             # the plain headline of a study of the same case, added
        headline += " " + json.loads((ROOT / s["also"]).read_text(encoding="utf-8"))["summary"]["plain"].rstrip(".") + "."
    if rec and rec.get("plain"):                                  # gpucfd's result in one sentence, written by its script
        headline += " " + rec["plain"].rstrip(".") + "."
    b = dict(title=res["title"], date=res["date"], verdict=res["verdict"], goal=s["goal"], story=s["story"],
             problem=dict(requirements=list(s["requirements"])),
             results=dict(headline=headline, meaning=list(s["meaning"]), limits=list(s["limits"])),
             accuracy=accuracy, speed=speed, links=links)
    return b, sketch_of(folder)


def build(folders=None):
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    out = []
    for folder, s in spec.items():
        if folders and folder not in folders:
            continue
        if s.get("script"):                                       # a page with a script of its own
            r = subprocess.run([sys.executable, str(ROOT / s["script"])], cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"{s['script']} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
            print(r.stdout.strip().splitlines()[-1])
            out.append(ROOT / "results" / folder / "brief.html")
            continue
        b, sketch = from_case(folder, s)
        page = render_brief(b, sketch, ROOT / "results" / folder / "brief.html")
        print(f"brief: {page.relative_to(ROOT)}  ({b['verdict']})")
        out.append(page)
    return out


if __name__ == "__main__":
    build(sys.argv[1:] or None)
