"""The case catalogue: one list of every case (cases/catalogue.yaml), rendered into the pages that
show it.

    python -m cfdlib.catalogue           # rewrite the generated parts
    python -m cfdlib.catalogue --check   # report what is out of date, change nothing (exit 1 if so)

Generated parts (between `<!-- catalogue:NAME -->` and `<!-- /catalogue:NAME -->` markers):
  cases/README.md   solved   table of solved cases
  cases/README.md   examples what each solved case folder shows, for agents choosing one to study
  cases/BACKLOG.md  backlog  the tiers and the catalogue of candidates
  index.html        cards    the report gallery served by GitHub Pages

Verdicts are read from each report's JSON file, never typed by hand. The check also fails when a
results/<folder>/results.json exists for a case the catalogue does not list as solved.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "cases" / "catalogue.yaml"
LUMINARY = "https://luminary.ai/verification-and-validation/{}/"
PAGES = "https://tomyud1.github.io/agentic-cfd/"
VERDICT_MD = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL"}
TIERS = [
    ("next", "Tier 1: next (each adds one new capability)"),
    ("harder", "Tier 2: harder versions of what works"),
    ("turbulence", "Tier 3: turbulence beyond the flat plate"),
    ("later", "Later: real geometry and more physics"),
]
GPU = {"yes": "✔", "no": "✘", "maybe": "?"}


def load(path: Path = CATALOGUE) -> list[dict]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]
    for c in cases:
        if c.get("status") not in ("solved", "in_progress", "planned"):
            raise ValueError(f"{c.get('title')}: status must be solved, in_progress or planned")
        if c["status"] == "solved":
            for key in ("id", "folder", "tests", "headline", "reports"):
                if not c.get(key):
                    raise ValueError(f"solved case {c.get('title')}: missing '{key}'")
        elif c.get("tier") not in [t for t, _ in TIERS] + ["catalogue"]:
            raise ValueError(f"{c.get('title')}: tier must be one of next, harder, turbulence, later, catalogue")
    return cases


def verdict(report: dict) -> str:
    """The verdict written by the report's own analysis (its JSON file)."""
    src = report.get("verdict_from") or str(Path(report["path"]).with_name("results.json"))
    data = json.loads((ROOT / src).read_text(encoding="utf-8"))
    v = data.get("verdict")
    if v not in VERDICT_MD:
        raise ValueError(f"{src}: no PASS/WARN/FAIL verdict")
    return v


def cell(text) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ").strip()


def solved_table(cases: list[dict]) -> str:
    rows = ["| # | Case | What it tests | Verdict | Headline | Report |", "|---|---|---|---|---|---|"]
    for c in cases:
        if c["status"] != "solved":
            continue
        for r in c["reports"]:
            rows.append(f"| {c['id']} | {cell(c['title'])} | {cell(c['tests'])} | "
                        f"{VERDICT_MD[verdict(r)]} | {cell(c['headline'])} | [view]({PAGES}{r['path']}) |")
    return "\n".join(rows)


def examples(cases: list[dict]) -> str:
    """One line per solved case that has `shows`: where it is and what can be learned from it."""
    out = []
    for c in cases:
        if c["status"] == "solved" and c.get("shows"):
            out.append(f"- **`cases/{c['folder']}/`** (case {c['id']}, {cell(c['tests'])}). {cell(c['shows'])}.")
    return "\n".join(out)


def source(c: dict) -> str:
    return f"[Luminary]({LUMINARY.format(c['luminary'])})" if c.get("luminary") else ""


def backlog(cases: list[dict]) -> str:
    out = []
    active = [c for c in cases if c["status"] == "in_progress"]
    if active:
        out += ["## In progress", "", "| # | Case | New for the agent | Candidate reference |",
                "|---|---|---|---|"]
        out += [f"| {cell(c.get('id'))} | **{cell(c['title'])}** | {cell(c.get('new'))} | "
                f"{cell(c.get('reference'))} {source(c)} |" for c in active]
        out.append("")
    for tier, heading in TIERS:
        rows = [c for c in cases if c["status"] == "planned" and c.get("tier") == tier]
        if not rows:
            continue
        out += [f"## {heading}", ""]
        if tier == "later":
            out += [f"- **{c['title']}.** {cell(c.get('new'))} {source(c)}".rstrip() for c in rows]
        else:
            out += ["| # | Case | New for the agent | Candidate reference | gpucfd |", "|---|---|---|---|---|"]
            out += [f"| {cell(c.get('id'))} | **{cell(c['title'])}** | {cell(c.get('new'))} | "
                    f"{cell(c.get('reference'))} {source(c)} | {GPU[c.get('gpucfd', 'no')]} |" for c in rows]
        out.append("")
    cat = [c for c in cases if c["status"] == "planned" and c.get("tier") == "catalogue"]
    if cat:
        out += ["## Catalogue: candidates not yet ordered", "",
                "From the Luminary V&V manual; most RANS cases are also NASA TMR cases. A case moves "
                "into a tier, and gets a number, when it is chosen.", ""]
        groups = list(dict.fromkeys(c.get("group", "Other") for c in cat))
        for g in groups:
            out += [f"**{g}**", "", "| Case | Source | gpucfd |", "|---|---|---|"]
            out += [f"| {cell(c['title'])} | {source(c)} | {GPU[c.get('gpucfd', 'no')]} |"
                    for c in cat if c.get("group", "Other") == g]
            out.append("")
    n_solved = sum(1 for c in cases if c["status"] == "solved")
    n_planned = sum(1 for c in cases if c["status"] != "solved")
    out.append(f"*{n_solved} solved (listed in [README.md](README.md)), {n_planned} planned. "
               "Counted by `cfdlib/catalogue.py`.*")
    return "\n".join(out).rstrip()


def cards(cases: list[dict]) -> str:
    out = []
    for c in cases:
        if c["status"] != "solved":
            continue
        for r in c["reports"]:
            v = verdict(r)
            out.append(
                f'  <a class="card" href="{html.escape(r["path"])}"><div class="row"><span class="t">'
                f'{html.escape(c["id"], False)}. {html.escape(c["title"], False)}</span><span class="v {v.lower()}">{v}</span></div>\n'
                f'    <div class="d">{html.escape(c["tests"], False)}. {html.escape(c["headline"], False)}.</div></a>')
    return "\n".join(out)


def replace_block(text: str, name: str, body: str, html_mode: bool = False) -> str:
    start, end = f"<!-- catalogue:{name} -->", f"<!-- /catalogue:{name} -->"
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pat.search(text):
        raise ValueError(f"markers {start} ... {end} not found")
    note = ("<!-- generated by cfdlib/catalogue.py from cases/catalogue.yaml: edit that file, "
            "then run python -m cfdlib.catalogue -->")
    sep = "\n" if html_mode else "\n\n"
    return pat.sub(lambda _: f"{start}\n{note}{sep}{body}{sep}{end}", text)


def targets(cases: list[dict]) -> dict[Path, str]:
    """New contents of every generated file."""
    spec = [(ROOT / "cases/README.md", "solved", solved_table(cases), False),
            (ROOT / "cases/BACKLOG.md", "backlog", backlog(cases), False),
            (ROOT / "cases/README.md", "examples", examples(cases), False),
            (ROOT / "index.html", "cards", cards(cases), True)]
    out: dict[Path, str] = {}
    for p, name, body, h in spec:   # a file may hold several blocks: each builds on the previous one
        out[p] = replace_block(out.get(p) or p.read_text(encoding="utf-8"), name, body, h)
    return out


def unlisted(cases: list[dict]) -> list[str]:
    """Result folders with a report that the catalogue does not list as solved."""
    solved = {c["folder"] for c in cases if c["status"] == "solved"}
    return sorted(p.parent.name for p in (ROOT / "results").glob("*/results.json") if p.parent.name not in solved)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report what is out of date; change nothing")
    args = ap.parse_args(argv)
    cases = load()
    problems = [f"results/{f}/ has a report but cases/catalogue.yaml does not list it as solved"
                for f in unlisted(cases)]
    for path, new in targets(cases).items():
        if path.read_text(encoding="utf-8") != new:
            if args.check:
                problems.append(f"{path.relative_to(ROOT)} is out of date: run python -m cfdlib.catalogue")
            else:
                path.write_text(new, encoding="utf-8", newline="\n")
                print(f"updated {path.relative_to(ROOT)}")
    for p in problems:
        print("CATALOGUE:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
