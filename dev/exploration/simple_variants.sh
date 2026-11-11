#!/bin/bash
# Compare SIMPLE settings on the 16-cell Poiseuille case: flow rate after N iterations.
# Discrete converged q for this mesh is ~5.026e-4 (exact continuous 4.989e-4).
source /usr/lib/openfoam/openfoam2606/etc/bashrc
set -uo pipefail
base="$FOAM_RUN/agentic/plane_poiseuille/Ny16"
root="$FOAM_RUN/simpleVariants"; rm -rf "$root"; mkdir -p "$root"

variant() {  # label, then foamDictionary edits as "entry=value" pairs
  local label=$1; shift
  local d="$root/$label"; cp -r "$base" "$d"; cd "$d"
  rm -rf [1-9]* postProcessing log.simpleFoam
  foamDictionary -entry endTime -set 200 system/controlDict >/dev/null
  foamDictionary -entry writeInterval -set 200 system/controlDict >/dev/null
  for kv in "$@"; do
    foamDictionary -entry "${kv%%=*}" -set "${kv#*=}" system/fvSolution >/dev/null 2>&1 \
      || foamDictionary -entry "${kv%%=*}" -add "${kv#*=}" system/fvSolution >/dev/null
  done
  simpleFoam > log.simpleFoam 2>&1
  local dat; dat=$(find postProcessing/flowRate -name surfaceFieldValue.dat)
  local L; L=$(awk '/^\(/{print; exit}' constant/polyMesh/points >/dev/null; echo 0.001)
  printf "%-28s" "$label"
  for it in 1 2 5 10 50 200; do
    awk -v it="$it" -v L="$L" '!/^#/ && $1==it {printf " it%-3s q=%.4e", it, $2/L}' "$dat"
  done
  echo
}

variant F_SIMPLE_noRelax       SIMPLE/consistent=no relaxationFactors/equations/U=1 relaxationFactors/fields/p=1 \
                               solvers/U/tolerance=1e-12 solvers/p/tolerance=1e-12
