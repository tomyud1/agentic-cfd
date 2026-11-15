#!/bin/bash
# Run a module of this case from Windows: bash cases/shock_diffraction_corner/py.sh <module> [args] (inside WSL).
# The output goes to results/shock_diffraction_corner/logs/<module>[.<args>].log, with "exit <code>" as its last line.
cd "/mnt/d/Projects/Agentic CFD"
mkdir -p results/shock_diffraction_corner/logs
LOG="results/shock_diffraction_corner/logs/$(echo "$*" | tr ' ' '.').log"
/home/tomyud/venvs/cfd/bin/python -u -m cases.shock_diffraction_corner."$1" "${@:2}" > "$LOG" 2>&1
echo "exit $?" >> "$LOG"
