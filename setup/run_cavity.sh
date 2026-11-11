#!/bin/bash
# Runs the lid-driven cavity tutorial (first smoke test of the install).
# Run from Windows:  wsl -d Ubuntu-24.04 --cd '/mnt/d/Projects/Agentic CFD/setup' -- bash run_cavity.sh
# OpenFOAM's bashrc fails under set -e/-u, so source it before strict mode
source /usr/lib/openfoam/openfoam2606/etc/bashrc
set -euo pipefail

echo "OpenFOAM version: $WM_PROJECT_VERSION"
echo "Tutorials: $FOAM_TUTORIALS"
mkdir -p "$FOAM_RUN"
case_src="$FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity"
[ -d "$case_src" ] || { echo "Tutorial not found at $case_src"; ls "$FOAM_TUTORIALS/incompressible" ; exit 1; }

cd "$FOAM_RUN"
rm -rf cavity
cp -r "$case_src" cavity
cd cavity

blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1
/usr/bin/time -f "icoFoam wall time: %e s" icoFoam > log.icoFoam 2>&1 || { tail -30 log.icoFoam; exit 1; }
touch cavity.foam   # lets ParaView open the case

echo "== Mesh =="; grep -E "cells:|Mesh OK|FAILED" log.checkMesh
echo "== Solver =="; grep -E "^Time = " log.icoFoam | tail -1; grep -E "Courant Number mean" log.icoFoam | tail -1
tail -3 log.icoFoam
echo "== Time folders =="; ls -d [0-9]*
echo "Case dir: $PWD"
