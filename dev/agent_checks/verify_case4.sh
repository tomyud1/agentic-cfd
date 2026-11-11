#!/bin/bash
cd "/mnt/d/Projects/Agentic CFD"; PY=/home/tomyud/venvs/cfd/bin/python
$PY -m cfdlib.foam ps
fin=$(ls -d /home/tomyud/OpenFOAM/tomyud-v2606/run/agentic/dfg_cylinder_re100/*/ | grep -i m4 | head -3); echo "$fin"
$PY -m cfdlib.contract results/dfg_cylinder_re100/results.json $(echo "$fin" | head -1) | head -5
$PY -m cfdlib.insights dfg_cylinder_re100 list | awk '{print $1, $2, $3, $4, $5}'
