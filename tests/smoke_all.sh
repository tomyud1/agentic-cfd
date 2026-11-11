#!/bin/bash
cd "/mnt/d/Projects/Agentic CFD"; PY=/home/tomyud/venvs/cfd/bin/python
for c in plane_poiseuille lid_driven_cavity_re100 stokes_first_problem; do echo "== $c"; $PY -m cases.$c.analyze 2>&1 | grep -iE "error|warning|Report:|Traceback|PASS|WARN|FAIL" | head -8; done
echo "== insights guard"; mkdir -p cases/zz_smoke
$PY - <<'P'
from cfdlib import insights
i = insights.log("zz_smoke", "performance", "ok text")
try: insights.log("zz_smoke", "setup", "bad 6.3e-06 number")
except ValueError as e: print("lint guard ok")
insights.resolve("zz_smoke", i, diagnosis="d", fix="f", lesson="l", guideline="g")
try: insights.resolve("zz_smoke", i, fix="other")
except ValueError as e: print("overwrite guard ok")
P
rm -rf cases/zz_smoke
echo "== perf"; $PY -c "from cfdlib import perf; import json; print(json.dumps(perf.summarize('/home/tomyud/OpenFOAM/tomyud-v2606/run/agentic/stokes_first_problem/N2000_dt2.5ms', 'pimpleFoam', 2000), indent=0)); print(perf.machine())"
