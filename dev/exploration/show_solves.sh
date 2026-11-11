#!/bin/bash
# First and last linear solves of Ux and p in a solver log.
log="${1:-$HOME/OpenFOAM/tomyud-v2606/run/agentic/plane_poiseuille/Ny128/log.simpleFoam}"
grep -E "Solving for (Ux|p)," "$log" | head -6
echo ...
grep -E "Solving for (Ux|p)," "$log" | tail -4
