#!/bin/bash
# Build OGL (OpenFOAM Ginkgo Layer, https://github.com/hpsim/OGL) for ESI OpenFOAM v2606 with CUDA.
# No sudo needed: OGL and the Ginkgo it downloads install into $FOAM_USER_LIBBIN.
# Needs: CUDA toolkit 12 (/usr/local/cuda), cmake >= 3.25, g++, git.
# Usage (inside WSL, from the repository folder):  bash setup/gpu/build_ogl.sh [jobs]   (30-60 min)
source /usr/lib/openfoam/openfoam2606/etc/bashrc     # before strict mode (it fails under set -u)
set -euo pipefail
JOBS="${1:-6}"                                     # nvcc is memory hungry: 6 jobs fit in 15 GB
export PATH=/usr/local/cuda/bin:$PATH
SRC="$HOME/src/OGL"
[ -d "$SRC" ] || git clone https://github.com/hpsim/OGL.git "$SRC"
echo "OGL $(git -C "$SRC" log -1 --format='%h %cd') | OpenFOAM $WM_PROJECT_VERSION (FOAM_API=${FOAM_API:-?}) | label ${WM_LABEL_SIZE:-32} | $(nvcc --version | tail -1)"
ls -d "$FOAM_SRC/OpenFOAM/lnInclude" >/dev/null
mkdir -p "$FOAM_USER_LIBBIN"
# GINKGO_FORCE_GPU_AWARE_MPI OFF: Ubuntu's Open MPI 4.1.6 is not CUDA-aware, so MPI data goes via the host.
# OGL_CUDA_ARCHITECTURES 89: RTX 4070 (Ada).
cmake -S "$SRC" -B "$SRC/build/release" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$FOAM_USER_LIBBIN/.." \
  -DOGL_BUILD_CUDA=ON -DOGL_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_ARCHITECTURES=89 -DGINKGO_CUDA_ARCHITECTURES=89 \
  -DGINKGO_FORCE_GPU_AWARE_MPI=OFF \
  $( [ "${WM_LABEL_SIZE:-32}" = "64" ] && echo -DOGL_DP_LABELS=ON )
cmake --build "$SRC/build/release" -j "$JOBS"
cmake --install "$SRC/build/release"
ls -la "$FOAM_USER_LIBBIN" | grep -iE "ogl|ginkgo" && echo "BUILD_OGL_DONE"
