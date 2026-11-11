#!/bin/bash
# Print solver settings and early/late iterations for mesh-study runs.
root="$HOME/OpenFOAM/tomyud-v2606/run/meshStudy"
echo "== fvSolution (N320) =="; sed -n '/solvers/,/^}/p;/PISO/,/^}/p' "$root/N320/system/fvSolution"
for N in 160 320; do
  log="$root/N$N/log.icoFoam"
  echo; echo "== N$N: first 3 steps =="
  awk '/^Time = /{c++} c>=1 && c<=3' "$log" | grep -E "Time =|Courant|Solving for"
  echo "== N$N: p iteration counts (first 10 p solves, last 5) =="
  grep "Solving for p" "$log" | head -10 | awk '{print $NF}' | tr '\n' ' '; echo
  grep "Solving for p" "$log" | tail -5 | awk '{print $NF}' | tr '\n' ' '; echo
done
echo; echo "== N320: last 3 steps before crash =="
awk '/^Time = /{t=$0} {buf[NR]=$0} END{}' /dev/null
grep -nE "^Time = " "$root/N320/log.icoFoam" | tail -3
tail -n 40 "$root/N320/log.icoFoam" | grep -E "Time =|Courant|Solving for" | head -20
