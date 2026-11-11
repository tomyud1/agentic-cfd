#!/bin/bash
# Mesh refinement study of the lid-driven cavity (Re = 10).
# Runs N x N meshes, keeping the Courant number constant by scaling the time step.
# Run from Windows:  wsl -d Ubuntu-24.04 --cd '/mnt/d/Projects/Agentic CFD/setup' -- bash mesh_study.sh
source /usr/lib/openfoam/openfoam2606/etc/bashrc
set -euo pipefail

SIZES="${SIZES:-20 40 80 160 320}"
CO="${CO:-0.5}"            # target lid Courant number: U_lid * dt / dx
LINSOL="${LINSOL:-GAMG}"   # GAMG (multigrid) or tutorial (PCG/smoothSolver defaults)
src="$FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity"
root="$FOAM_RUN/meshStudy_${LINSOL}_Co${CO}"
mkdir -p "$root"
echo "N,cells,deltaT,steps,maxCo,wall_s" > "$root/timings.csv"

for N in $SIZES; do
  case_dir="$root/N$N"
  rm -rf "$case_dir"; cp -r "$src" "$case_dir"; cd "$case_dir"

  dt=$(python3 -c "print($CO*0.1/$N)")   # dx = 0.1/N, U_lid = 1
  sed -i -E "s/\(20 20 1\)/($N $N 1)/" system/blockMeshDict
  grep -q "($N $N 1)" system/blockMeshDict || { echo "Failed to set mesh size"; exit 1; }
  foamDictionary -entry deltaT -set "$dt" system/controlDict > /dev/null
  foamDictionary -entry writeControl -set runTime system/controlDict > /dev/null
  foamDictionary -entry writeInterval -set 0.5 system/controlDict > /dev/null
  if [ "$LINSOL" = GAMG ]; then
    for f in p pFinal U; do   # foamDictionary expands pFinal's $p, so set it explicitly
      foamDictionary -entry "solvers/$f/solver" -set GAMG system/fvSolution > /dev/null
      foamDictionary -entry "solvers/$f/preconditioner" -remove system/fvSolution > /dev/null 2>&1 || true
      foamDictionary -entry "solvers/$f/smoother" -set GaussSeidel system/fvSolution > /dev/null
    done
    # GAMG with the tutorial's loose first pressure solve (relTol 0.05) is unstable here
    # (error grows ~1.4x per step from step 1). Converging every pressure solve fixes it.
    foamDictionary -entry solvers/p/relTol -set 0 system/fvSolution > /dev/null
  fi

  blockMesh > log.blockMesh 2>&1
  start=$(date +%s.%N)
  icoFoam > log.icoFoam 2>&1 || { tail -20 log.icoFoam; exit 1; }
  end=$(date +%s.%N)
  postProcess -func writeCellCentres -latestTime > log.cellCentres 2>&1
  touch "N$N.foam"

  wall=$(python3 -c "print(round($end-$start,2))")
  steps=$(python3 -c "print(round(0.5/$dt))")
  co=$(grep "Courant Number mean" log.icoFoam | awk '{print $NF}' | sort -g | tail -1)
  echo "N=$N cells=$((N*N)) dt=$dt steps=$steps maxCo=$co wall=${wall}s"
  echo "$N,$((N*N)),$dt,$steps,$co,$wall" >> "$root/timings.csv"
done
echo "Results in $root"
