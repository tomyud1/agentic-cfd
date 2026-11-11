#!/bin/bash
# Print the OpenFOAM build environment that OGL needs.
source /usr/lib/openfoam/openfoam2606/etc/bashrc
echo "version=$WM_PROJECT_VERSION api=${FOAM_API:-} label=${WM_LABEL_SIZE:-} prec=${WM_PRECISION_OPTION:-} mpi=${FOAM_MPI:-}"
echo "FOAM_SRC=$FOAM_SRC"; echo "FOAM_USER_LIBBIN=$FOAM_USER_LIBBIN"; echo "FOAM_LIBBIN=$FOAM_LIBBIN"
ls "$FOAM_SRC/OpenFOAM/lnInclude" | head -2; ls "$FOAM_LIBBIN"/libfiniteVolume.so "$FOAM_LIBBIN/$FOAM_MPI"/libPstream.so
