#!/bin/bash
cd "/mnt/d/Projects/Agentic CFD"
PY=/home/tomyud/venvs/cfd/bin/python
echo "== contract --decisions (case 2)"; $PY -m cfdlib.contract --decisions cases/lid_driven_cavity_re100/decisions.yaml /home/tomyud/OpenFOAM/tomyud-v2606/run/agentic/lid_driven_cavity_re100/N256
echo "== full contract case 1"; $PY -m cfdlib.contract results/plane_poiseuille/results.json
echo "== foam ps"; $PY -m cfdlib.foam ps
echo "== insights CLI"; mkdir -p cases/zz_smoke
$PY -m cfdlib.insights zz_smoke new setup "test problem with \(x^2\) math"
$PY -m cfdlib.insights zz_smoke resolve 1 --diagnosis d --fix f --lesson l --guideline g
$PY -m cfdlib.insights zz_smoke list; cat cases/zz_smoke/insights.yaml; rm -rf cases/zz_smoke
echo "== monitor residual_targets on case 2 preflight log"
$PY - <<'P'
from cfdlib import monitor
import glob
log = sorted(glob.glob("/home/tomyud/OpenFOAM/tomyud-v2606/run/agentic/lid_driven_cavity_re100/pf*_preflight/log.simpleFoam"))[-1]
a = monitor.assess(log, residual_targets={"Ux": 1e-11, "p": 1e-10})
print(log, a["est_kind"], a["est_full_runtime"], [c["value"] for c in a["checks"]][-2:])
P
