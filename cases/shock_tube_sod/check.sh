#!/bin/bash
# Final checks from Windows: bash cases/shock_tube_sod/check.sh (inside WSL).
cd "/mnt/d/Projects/Agentic CFD"
PY=/home/tomyud/venvs/cfd/bin/python
$PY -m cfdlib.contract --code cases/shock_tube_sod
$PY -m cfdlib.contract results/shock_tube_sod/results.json "$($PY -c 'from cfdlib import foam; print(foam.foam_env("FOAM_RUN"))')/agentic/shock_tube_sod/N12800"
$PY -m cfdlib.contract --decisions cases/shock_tube_sod/decisions.yaml "$($PY -c 'from cfdlib import foam; print(foam.foam_env("FOAM_RUN"))')/agentic/shock_tube_sod/N12800"
$PY -m cfdlib.insights shock_tube_sod list
$PY -m cfdlib.foam ps
