"""Live workflow log: record problems THE MOMENT you meet them, resolve them later.

Insights written from memory at the end of a long session get lost or distorted, so the log is
kept like a lab notebook: open an entry as soon as something fails, surprises you, or needs a
second attempt (before you start fixing it), then fill in the diagnosis, fix, lesson and guideline
when it is solved. Each entry gets the real time it was logged.

Python:
    from cfdlib import insights
    i = insights.log("lid_driven_cavity_re100", "preflight", "Courant number grows on the 256 mesh")
    ...
    insights.resolve("lid_driven_cavity_re100", i, diagnosis="...", fix="...", lesson="...", guideline="...")

CLI (inside WSL, from the project root):
    python -m cfdlib.insights <case> new <stage> "<problem>"
    python -m cfdlib.insights <case> resolve <id> --diagnosis ".." --fix ".." --lesson ".." --guideline ".."
    python -m cfdlib.insights <case> list

Editing cases/<case>/insights.yaml by hand is also fine; keep the same fields and give `logged`
the real time (run `date "+%Y-%m-%d %H:%M"`).
"""
import argparse
import datetime as dt
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("intake", "theory", "setup", "preflight", "production", "analysis", "report", "tooling", "performance")
RESOLVED_FIELDS = ("diagnosis", "fix", "lesson", "guideline")
HEADER = """\
# Workflow insights, logged LIVE (cfdlib/insights.py). One entry per problem, surprise or retry:
#   id, stage, logged (real time first met), status (open | resolved), problem,
#   then when solved: diagnosis, fix, lesson, guideline (the change to the workflow that would have prevented it).
# stage: intake | theory | setup | preflight | production | analysis | report | tooling | performance
"""


def path(case):
    return ROOT / "cases" / case / "insights.yaml"


def load(case):
    p = path(case)
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or []) if p.exists() else []


def _save(case, entries):
    text = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=110)
    path(case).write_text(HEADER + "\n" + text, encoding="utf-8")


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def _lint(fields):
    """Insights appear in the report, so they obey its typesetting rules: check them now, not at render."""
    from .contract import lint
    errors, warnings = lint({"insight": {k: v for k, v in fields.items() if isinstance(v, str)}})
    if errors:
        raise ValueError("insight text breaks the report's typesetting rules (write math and numbers as "
                         r"\(...\), e.g. \(6.3 \times 10^{-6}\)):" + "\n  - " + "\n  - ".join(errors))
    for w in warnings:
        print("insight warning:", w)


def log(case, stage, problem, **fields):
    """Open a new entry now. Returns its id."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    _lint(dict(problem=problem, **fields))
    entries = load(case)
    new_id = max([e.get("id", 0) for e in entries] + [0]) + 1
    entry = dict(id=new_id, stage=stage, logged=now(), status="open", problem=problem)
    entry.update(fields)
    entries.append(entry)
    if all(entry.get(k) for k in RESOLVED_FIELDS):
        entry["status"] = "resolved"
    _save(case, entries)
    return new_id


def resolve(case, entry_id, overwrite=False, **fields):
    """Add diagnosis / fix / lesson / guideline (any subset) to an entry; resolved once all are there.

    Refuses to change fields that are already filled in unless overwrite=True (protects against a wrong id)."""
    _lint(fields)
    entries = load(case)
    for e in entries:
        if e.get("id") == entry_id:
            clash = [k for k in fields if e.get(k) and e.get(k) != fields[k]]
            if clash and not overwrite:
                raise ValueError(f"insight {entry_id} ('{str(e.get('problem'))[:60]}') already has {clash}; "
                                 "wrong id? pass overwrite=True to replace")
            e.update(fields)
            print(f"insight {entry_id} updated: {str(e.get('problem'))[:70]}")
            if all(e.get(k) for k in RESOLVED_FIELDS):
                e["status"] = "resolved"
                e.setdefault("resolved", now())
            _save(case, entries)
            return e
    raise KeyError(f"no insight with id {entry_id} in {path(case)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Live workflow-insight log")
    ap.add_argument("case")
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("stage", choices=STAGES)
    n.add_argument("problem")
    r = sub.add_parser("resolve")
    r.add_argument("id", type=int)
    for k in RESOLVED_FIELDS:
        r.add_argument(f"--{k}")
    r.add_argument("--overwrite", action="store_true")
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "new":
        print("logged insight", log(a.case, a.stage, a.problem))
    elif a.cmd == "resolve":
        e = resolve(a.case, a.id, overwrite=a.overwrite, **{k: getattr(a, k) for k in RESOLVED_FIELDS if getattr(a, k)})
        print(f"insight {a.id}: {e['status']}")
    else:
        for e in load(a.case):
            print(f"{e.get('id', '?'):>3}  {e.get('logged', '?'):16}  {e.get('stage', '?'):10}  "
                  f"{e.get('status', '?'):8}  {str(e.get('problem', ''))[:70]}")
