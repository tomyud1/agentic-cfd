#!/bin/bash
cd "/mnt/d/Projects/Agentic CFD"
PY=/home/tomyud/venvs/cfd/bin/python
fin=$(ls -d /home/tomyud/OpenFOAM/tomyud-v2606/run/agentic/stokes_first_problem/*/ | grep -v preflight | head -40)
echo "$fin"
$PY -m cfdlib.foam ps
$PY -m cfdlib.insights stokes_first_problem list
$PY - <<'P'
import json
r = json.load(open("results/stokes_first_problem/results.json"))
print(r["verdict"], r["comparison_type"], "| decisions", len(r["decisions"]), "| figs", len(r["figures"]))
for a in r["summary"]["answers"]: print(a["letter"], a["label"][:50], "|", a["diff"][:80])
P
