#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-$HOME/venvs/cfd/bin/python}"
export RUN="$($PY -c 'from cfdlib import foam; print(foam.foam_env("FOAM_RUN"))')/agentic"   # where case runs live on this machine
export CFD_RERENDER=1   # rebuild the reports without re-stamping their date and machine record
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
echo "== study report (template, contract, formatters)"; T=$(mktemp -d); PYTHONPATH=. $PY tests/test_study.py "$T" 2>&1 | grep -E "^FAIL|failed|all passed|Traceback|Error" | cut -c1-200; rm -rf "$T"
echo "== perf"; $PY -c "from cfdlib import perf; import json, os; print(json.dumps(perf.summarize(os.environ['RUN'] + '/stokes_first_problem/N2000_dt2.5ms', 'pimpleFoam', 2000), indent=0)); print(perf.machine())"
