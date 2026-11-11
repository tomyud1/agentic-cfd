#!/bin/bash
# Re-render every case report from the saved run data in results/<case>/ (no OpenFOAM runs needed),
# then check each report against the contract. Run inside WSL/Linux from anywhere:
#   bash setup/rerender_all.sh
# PY defaults to the project venv; override with  PY=/path/to/python bash setup/rerender_all.sh
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-$HOME/venvs/cfd/bin/python}"
for c in plane_poiseuille lid_driven_cavity_re100 stokes_first_problem dfg_cylinder_re100; do
  echo "== $c"
  $PY -m cases.$c.analyze --report 2>&1 | grep -iE "error|Report:|Traceback|PASS|WARN|FAIL|not found" | head -6
  $PY -m cfdlib.contract results/$c/results.json | head -3
done
