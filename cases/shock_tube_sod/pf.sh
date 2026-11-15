#!/bin/bash
# Pre-flight from Windows: bash cases/shock_tube_sod/pf.sh (inside WSL). Log: results/shock_tube_sod/preflight.log
cd "/mnt/d/Projects/Agentic CFD"
/home/tomyud/venvs/cfd/bin/python -u -m cases.shock_tube_sod.run "${1:-preflight}" "${@:2}" > "results/shock_tube_sod/${1:-preflight}.log" 2>&1
echo "exit $?" >> "results/shock_tube_sod/${1:-preflight}.log"
