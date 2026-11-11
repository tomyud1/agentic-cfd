#!/bin/bash
# Print the key files of the planarPoiseuille tutorial.
source /usr/lib/openfoam/openfoam2606/etc/bashrc
t="$FOAM_TUTORIALS/incompressible/pimpleFoam/laminar/planarPoiseuille"
cd "$t"; find . -type f | sort
strip() { sed -e '/^\/\*/,/\*\//d' -e '/^\s*\/\//d' -e '/^\s*$/d' "$1"; }
for f in Allrun Alltest setups.orig/common/system/blockMeshDict setups.orig/common/constant/fvOptions \
         setups.orig/common/constant/transportProperties setups.orig/common/constant/turbulenceProperties \
         setups.orig/common/0.orig/U setups.orig/common/0.orig/p setups.orig/common/system/controlDict \
         setups.orig/common/system/fvSchemes setups.orig/common/system/fvSolution; do
  [ -f "$f" ] && { echo; echo "######## $f"; strip "$f"; }
done
ls setups.orig
