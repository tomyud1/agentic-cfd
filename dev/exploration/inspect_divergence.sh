#!/bin/bash
# Show how max Courant number and continuity error evolve in a diverged run.
log="${1:-$HOME/OpenFOAM/tomyud-v2606/run/meshStudy_GAMG_Co0.5/N160/log.icoFoam}"
echo "Log: $log"
echo "step  time  maxCo  p-iters(1st,final)  U-iters  contErr(local)"
awk '
/^Time = /{step++; t=$3}
/Courant Number/{co=$NF}
/Solving for Ux/{ux=$NF}
/Solving for p/{ if (pc==0) p1=$NF; else p2=$NF; pc++ }
/continuity errors/{ if (++ce==2) {gsub(",","",$9); cl=$9} }
/^ExecutionTime/{ if (step<=5 || step%2==0) printf "%4d  %s  %s  %s,%s  %s  %s\n", step, t, co, p1, p2, ux, cl; pc=0; ce=0 }
' "$log" | head -40
echo "..."
awk '/^Time = /{step++} /Courant Number/{print step, $NF}' "$log" | tail -8
