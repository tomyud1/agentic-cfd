"""Render a dummy GPU study to check the study template and contract (writes to the scratchpad-like tmp dir).
    PYTHONPATH=. python tests/test_study.py <out_dir>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from cfdlib import plots, study
from cfdlib.report import m, tex_num

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
fig, ax = plots.figure()
ax.loglog([1e5, 1e6, 3e6], [1.0, 2.0, 3.0], "o-", label="speed-up")
ax.set_xlabel("Cells"); ax.set_ylabel("Speed-up")
svgs = {"speedup": plots.to_svg(fig, "speedup")}
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
bad = dict(s, verdict="PASS", same_answer=dict(s["same_answer"], ok=False))
print("contract catches PASS without same answer:", bool(study.check_study(bad, svgs)))
print(study.render_study(s, svgs, out / "gpu_study.html", insights=[dict(stage="gpu", logged="2026-11-11 12:00", problem="p")]))
