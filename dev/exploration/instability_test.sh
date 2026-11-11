#!/bin/bash
# Short runs to find what triggers the cavity instability: mesh size N or time step dt.
# Prints the local continuity error at the end of each step; steady growth = unstable.
source /usr/lib/openfoam/openfoam2606/etc/bashrc
set -euo pipefail
src="$FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity"
root="$FOAM_RUN/instabilityTest"; mkdir -p "$root"

run() {  # N dt label [extra fvSchemes edit]
  local N=$1 dt=$2 label=$3 extra=${4:-}
  local d="$root/$label"; rm -rf "$d"; cp -r "$src" "$d"; cd "$d"
  sed -i -E "s/\(20 20 1\)/($N $N 1)/" system/blockMeshDict
  foamDictionary -entry deltaT -set "$dt" system/controlDict > /dev/null
  foamDictionary -entry endTime -set "$(python3 -c "print(30*$dt)")" system/controlDict > /dev/null
  foamDictionary -entry writeControl -set timeStep system/controlDict > /dev/null
  foamDictionary -entry writeInterval -set 1000 system/controlDict > /dev/null
  [ -n "$extra" ] && eval "$extra"
  blockMesh > log.blockMesh 2>&1
  icoFoam > log.icoFoam 2>&1 || true
  local errs; errs=$(grep "continuity errors" log.icoFoam | awk 'NR%2==0{gsub(",","",$9); print $9}')
  local first last co
  first=$(echo "$errs" | sed -n 5p); last=$(echo "$errs" | tail -1)
  co=$(grep "Courant Number" log.icoFoam | awk '{print $NF}' | sort -g | tail -1)
  local dn; dn=$(python3 -c "print(round(0.01*$dt/(0.1/$N)**2,1))")
  printf "%-22s N=%-4s dt=%-10s Co_lid=%-5s diffNum=%-5s contErr step5=%-12s step30=%-12s maxCo=%s\n" \
    "$label" "$N" "$dt" "$(python3 -c "print($dt/(0.1/$N))")" "$dn" "$first" "$last" "$co"
}

gamg() {  # set the listed fvSolution entries to GAMG + GaussSeidel
  for f in "$@"; do
    foamDictionary -entry "solvers/$f/solver" -set GAMG system/fvSolution > /dev/null
    foamDictionary -entry "solvers/$f/preconditioner" -remove system/fvSolution > /dev/null 2>&1 || true
    foamDictionary -entry "solvers/$f/smoother" -set GaussSeidel system/fvSolution > /dev/null
  done
}

MODE="${MODE:-solvers}"
if [ "$MODE" = dt ]; then
  run 80  0.000625   N80_dt625
  run 80  0.0003125  N80_dt3125
  run 160 0.000625   N160_dt625
  run 160 0.0003125  N160_dt3125
elif [ "$MODE" = fix ]; then
  run 160 0.0003125  GAMG_p_DICGS   "gamg p pFinal U; for f in p pFinal; do foamDictionary -entry solvers/\$f/smoother -set DICGaussSeidel system/fvSolution >/dev/null; done"
  run 160 0.0003125  GAMG_p_relTol0 "gamg p pFinal U; foamDictionary -entry solvers/p/relTol -set 0 system/fvSolution >/dev/null"
  run 160 0.0003125  GAMG_p_tol1e8  "gamg p pFinal U; for f in p pFinal; do foamDictionary -entry solvers/\$f/tolerance -set 1e-8 system/fvSolution >/dev/null; done"
else
  run 160 0.0003125  PCG_default
  run 160 0.0003125  GAMG_p_pFinal_U  "gamg p pFinal U"
  run 160 0.0003125  GAMG_p_pFinal    "gamg p pFinal"
  run 160 0.0003125  GAMG_U_only      "gamg U"
  run 160 0.0003125  GAMG_p_only      "gamg p"
fi
