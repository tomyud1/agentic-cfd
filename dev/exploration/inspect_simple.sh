#!/bin/bash
# Residual and iteration counts at a few points of a simpleFoam log.
log="${1:-$HOME/OpenFOAM/tomyud-v2606/run/agentic/plane_poiseuille/Ny128/log.simpleFoam}"
for it in 1 2 3 10 100 1000 5000; do
  awk -v it="$it" '/^Time = /{t=$3} t==it && /Solving for|continuity/' "$log" | head -4 | sed "s/^/[$it] /"
done
grep -m3 -iE "warn|fvOptions|Source|selected" "$log"
