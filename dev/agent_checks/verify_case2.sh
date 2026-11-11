#!/bin/bash
# Independent checks of the case-2 agent's claims.
cd "/mnt/d/Projects/Agentic CFD"
echo "== contract =="
/home/tomyud/venvs/cfd/bin/python -m cfdlib.contract results/lid_driven_cavity_re100/results.json \
  /home/tomyud/OpenFOAM/tomyud-v2606/run/agentic/lid_driven_cavity_re100/N256
echo "== leftover solver processes =="
ps -eo pid,etime,cmd | grep -E "Foam|foam" | grep -v grep || echo none
echo "== shared files modified by the agent? (newer than case-2 problem.yaml) =="
find cfdlib .claude cases/plane_poiseuille results/plane_poiseuille -newer cases/lid_driven_cavity_re100/problem.yaml -type f \
  -not -path "*/__pycache__/*" | head
echo "== verdict and answers from results.json =="
/home/tomyud/venvs/cfd/bin/python - <<'EOF'
import json
r = json.load(open("results/lid_driven_cavity_re100/results.json"))
print("verdict:", r["verdict"], "| type:", r["comparison_type"])
for a in r["summary"]["answers"]:
    print(a["letter"], "|", a["label"][:60])
print("insights:", len(r["insights"]), "| decisions:", len(r["decisions"]), "| figures:", len(r["figures"]))
EOF
echo "== Ghia reference file unchanged? =="
sha256sum cases/lid_driven_cavity_re100/reference/ghia1982_re100.yaml cases/lid_driven_cavity_re100/problem.yaml
