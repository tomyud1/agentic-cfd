#!/bin/bash
# Re-render every case report from the saved run data in results/<case>/ (no OpenFOAM runs needed),
# then check each report against the contract. Run inside WSL/Linux from anywhere:
#   bash setup/rerender_all.sh
# PY defaults to the project venv; override with  PY=/path/to/python bash setup/rerender_all.sh
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-$HOME/venvs/cfd/bin/python}"
export CFD_RERENDER=1   # keep each report's original date and machine record (cfdlib.report.render)
for c in plane_poiseuille lid_driven_cavity_re100 stokes_first_problem dfg_cylinder_re100 dfg_cylinder_3d_re20 heated_pipe_re200 turbulent_flat_plate_sa turbulent_channel_re180; do
  echo "== $c"
  $PY -m cases.$c.analyze --report 2>&1 | grep -iE "error|Report:|Traceback|PASS|WARN|FAIL|not found" | head -6
  $PY -m cfdlib.contract results/$c/results.json | head -3
done
echo "== dfg_cylinder_3d_re20 GPU study"
$PY -m cases.dfg_cylinder_3d_re20.gpu_study 2>&1 | grep -iE "error|Traceback|gpu_study|PASS|WARN|FAIL" | head -4
echo "== dfg_cylinder_re100 gpucfd study (from recorded runs; needs no GPU)"
$PY -m cases.dfg_cylinder_re100.gpu_study 2>&1 | grep -iE "error|Traceback|gpu_study|PASS|WARN|FAIL" | cut -c1-200 | head -4
