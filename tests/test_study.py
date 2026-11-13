"""Render two dummy studies to check the study template and contract.
    PYTHONPATH=. python tests/test_study.py <out_dir>
1. The smallest study the contract accepts (required keys only), as a script written before the optional
   keys existed would pass it.
2. A study that uses every optional key: its own kicker, verdict label and section titles, a section of its
   own, a table with question, result, meaning and a figure placed at the table, reasons for a WARN verdict,
   a plain headline. Checks the reading order of the figures and what the page shows.
3. What the contract refuses and what it only warns about.
"""
import copy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from cfdlib import plots, study
from cfdlib.report import m, m_fix, m_pct, m_pow10, m_sig, m_times, tex_num

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
FAILS = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def fig(name):
    f, ax = plots.figure()
    ax.loglog([1e5, 1e6, 3e6], [1.0, 2.0, 3.0], "o-", label="speed-up")
    ax.set_xlabel("Cells"); ax.set_ylabel("Speed-up")
    return plots.to_svg(f, name)


svgs = {"speedup": fig("speedup")}
s = dict(
    title="GPU versus CPU: dummy", date="2026-11-11", question="Does the GPU solve the pressure equation faster?",
    verdict="PASS",
    summary=dict(headline="Dummy headline, see {fig:speedup}.",
                 answers=[dict(label="Pressure solve speed-up", value=m(tex_num(3.0) + r"\times"), note="at 3 million cells")],
                 findings=["one"], limitations=["none"], recommendation="Use it."),
    setup=dict(intro="x", hardware=[["GPU", "RTX 4070"]], software=[["OGL", "dev"]], fairness=["best CPU baseline"]),
    method=dict(intro="x", choices=[dict(choice="GKOCG with BJ", why="w", alternatives="a")]),
    results=dict(intro="x", tables=[dict(title="T", head=["Cells", "CPU", "GPU"], rows=[["1", "2", "3"]])], notes=["n"]),
    same_answer=dict(intro="x", head=["Quantity", "CPU", "GPU"], rows=[[m("c_D"), "6.18", "6.18"]], ok=True, verdict="same"),
    breakdown=dict(intro="x", tables=[], notes=["n"]),
    reproducibility=dict(commands=[dict(cmd="python -m x", what="w")], files=[dict(path="a", what="b")],
                         run_dirs=[dict(path="c", what="d")], environment=[["OS", "Linux"]]),
    figures=[dict(key="speedup", section="results", caption="Dummy.")],
)
ins = [dict(stage="gpu", logged="2026-11-11 12:00", problem="resolved problem", diagnosis="d", fix="f", lesson="l", guideline="g"),
       dict(stage="performance", logged="2026-11-11 13:00", problem="open problem", status="open"),
       dict(stage="setup", logged="2026-11-11 14:00", problem="not a performance entry")]

# 1. required keys only
check("the smallest study passes the contract", not study.check_study(s, svgs), "; ".join(study.check_study(s, svgs)))
page = study.render_study(copy.deepcopy(s), svgs, out / "gpu_study.html", insights=ins).read_text(encoding="utf-8")
check("default kicker, verdict label and section titles",
      "Agentic CFD performance study" in page and "✓ Fair comparison, same answer" in page
      and ">4 Same answer?</a>" in page and ">5 Where the time goes</a>" in page and ">6 Insights</a>" in page)
check("insights: only performance and gpu entries, folded, an open one unfolded",
      page.count("<details class=\"insight") == 2 and "not a performance entry" not in page
      and '<details class="insight open" open>' in page and '<details class="insight resolved">' in page)
check("a short first column is kept on one line", '<td class=" nw">1</td>' in page)

# 2. every optional key
full = copy.deepcopy(s)
full.update(kicker="tool comparison", verdict="WARN", verdict_label="Usable, read the caveats")
full["summary"].update(reasons=["one output differs by more than the two error estimates allow"],
                       plain="Dummy: 3.0 times faster at 3 million cells, same answer in 6 of 7 outputs")
full["results"]["title"] = "Accuracy and cost"
full["same_answer"].update(title="Do the two programs agree?", question="Do they agree?", result="Yes.", meaning="Nothing lost.")
full["breakdown"]["title"] = "What holds it back"
full["results"]["tables"] = [
    dict(title="Cost", question="What does a run cost?", head=["Run with a rather long name to wrap in its cell, certainly", "Time", "Change"],
         rows=[["A", m_fix(1.5, 1), m_pct(0.0123, signed=True)], ["B", m_times(3.3), m_pow10(1e-8)]],
         result="B is " + m_times(3.3) + " dearer.", meaning="Use A.", figure="attached", nowrap=[0, 1]),
    dict(title="Plain", head=["Run", "Time"], rows=[["A", m_sig(4.7e-6)]], note="An ordinary note."),
]
full["sections"] = [dict(key="scheme", title="A section of its own", after="results", intro="x", notes=["n"],
                         tables=[dict(title="Own", head=["a", "b"], rows=[["1", "2"]], meaning="m")])]
full["figures"] = [dict(key="attached", section="results", caption="Placed at its table."),
                   dict(key="speedup", section="results", caption="Free figure of the results."),
                   dict(key="own", section="scheme", caption="In the study's own section."),
                   dict(key="first", section="setup", caption="In the setup.")]
svgs2 = {k: fig(k) for k in ("attached", "speedup", "own", "first")}
errs = study.check_study(full, svgs2)
check("a study with every optional key passes the contract", not errs, "; ".join(errs))
check("figures are numbered in reading order: free figures of a section, then those placed at tables",
      study.figure_order(full) == ["first", "speedup", "attached", "own"], str(study.figure_order(full)))
check("own sections are placed and numbered", [(x["key"], x["title"]) for x in study.section_order(full)][2:4] ==
      [("results", "Accuracy and cost"), ("scheme", "A section of its own")])
page = study.render_study(full, svgs2, out / "tool_study.html", insights=ins).read_text(encoding="utf-8")
check("own kicker, verdict label, titles and section in the page",
      "Agentic CFD tool comparison" in page and "⚠ Usable, read the caveats" in page and ">3 Accuracy and cost</a>" in page
      and ">4 A section of its own</a>" in page and ">5 Do the two programs agree?</a>" in page and ">7 Insights</a>" in page)
check("question, result and meaning are labelled lines; the reasons are shown",
      "<dt>Question</dt><dd>What does a run cost?</dd>" in page and "<dt>Result</dt>" in page and "<dt>What it means</dt><dd>Use A.</dd>" in page
      and "Why this verdict" in page and "one output differs" in page)
i_q, i_fig, i_tab = page.index("What does a run cost?"), page.index("<b>Figure 3.</b> Placed at its table."), page.index("Run with a rather long name")
check("a figure named by a table sits between its question and the table", i_q < i_fig < i_tab)
check("the section title of the same-answer section is not repeated as a table title", page.count("Do the two programs agree?") == 2)
check("the formatters give math or a dash",
      m_pct(0.0024) == r"\(0.24\,\%\)" and m_pct(-0.0024, signed=True, unit=False) == r"\(-0.24\)" and m_pct(-0.0024, size=True) == r"\(0.24\,\%\)"
      and m_pct(None) == "–" and m_times(float("nan")) == "–" and m_pow10(1e-8) == r"\(10^{-8}\)" and m_fix(3.2, 3) == r"\(3.200\)"
      and m_sig(4.7e-6) == r"\(0.0000047\)" and m_sig(0.0005, pct=True, unit=True) == r"\(0.05\,\%\)" and m_times(3.25, 2) == r"\(3.25\times\)")

# 3. refused, and warned about
bad = dict(s, same_answer=dict(s["same_answer"], ok=False))
check("refused: PASS without the same answer", bool(study.check_study(bad, svgs)))
bad = dict(s, verdict="WARN")
check("refused: a verdict other than PASS without reasons", any("reasons" in x for x in study.check_study(bad, svgs)))
bad = copy.deepcopy(full); bad["summary"]["plain"] = "with <b>markup</b>"
check("refused: markup in the plain headline", any("summary.plain" in x for x in study.check_study(bad, svgs2)))
bad = copy.deepcopy(full); bad["results"]["tables"][0]["figure"] = "own"
check("refused: a table naming a figure of another section", any(".figure" in x for x in study.check_study(bad, svgs2)))
bad = copy.deepcopy(full); bad["sections"][0]["key"] = "results"
check("refused: an own section with a reserved name", any("sections[0].key" in x for x in study.check_study(bad, svgs2)))
bad = copy.deepcopy(full); bad["figures"][2]["section"] = "method"
check("refused: a figure in a section that shows none", any("figures[2].section" in x for x in study.check_study(bad, svgs2)))
wide = copy.deepcopy(s); wide["results"]["tables"][0] = dict(head=[str(i) for i in range(12)], rows=[[str(i) for i in range(12)]],
                                                              note="<b>Result.</b> something")
w = study.warnings(wide)
check("warned: a table too wide for the column, and a result written into a note",
      not study.check_study(wide, svgs) and any("12 columns" in x for x in w) and any("result" in x for x in w), str(len(w)))
check("the verdict helper", study.verdict() == ("PASS", []) and study.verdict(warn=["a"]) == ("WARN", ["a"])
      and study.verdict(["f"], ["a"]) == ("FAIL", ["f", "a"]))

print(f"{len(FAILS)} failed" if FAILS else "all passed", "|", out / "gpu_study.html", "|", out / "tool_study.html")
sys.exit(1 if FAILS else 0)
