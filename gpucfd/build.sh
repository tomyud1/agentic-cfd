#!/bin/bash
# Compile the CUDA sources (inside WSL/Linux):  bash gpucfd/build.sh [incompressible|compressible|chemistry]
#   build/libgcfd.so   the incompressible solver (src/gcfd.cu)
#   build/libgcfdc.so  the compressible solver for gas with shocks (src/gcfdc.cu)
#   build/libgcfdk.so  the chemistry of many independent cells (src/gcfdk.cu)
# With no argument all are built; with one, only that library (the other files are not touched).
# ARCH defaults to the RTX 4070 (compute capability 8.9); e.g. ARCH=sm_80 for an A100.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_89}"
WHAT="${1:-all}"
if [ "$WHAT" = all ] || [ "$WHAT" = incompressible ]; then
  "$NVCC" -O3 -std=c++17 -arch="$ARCH" -lineinfo -Xcompiler -fPIC -shared src/gcfd.cu -o build/libgcfd.so
fi
if [ "$WHAT" = all ] || [ "$WHAT" = compressible ]; then
  # --fmad=false: no fused multiply-add. The two cells that share a face each compute its flux; fused
  # differently in the copies of the reconstruction, the two values differed in the last bit and mass leaked
  # (1e-10 with WENO5 in mixed precision). Without fusing they are identical (COMPRESSIBLE.md, rung 3).
  "$NVCC" -O3 -std=c++17 -arch="$ARCH" --fmad=false -lineinfo -Xcompiler -fPIC -shared src/gcfdc.cu -o build/libgcfdc.so.new
  mv build/libgcfdc.so.new build/libgcfdc.so
fi
if [ "$WHAT" = all ] || [ "$WHAT" = chemistry ]; then
  "$NVCC" -O3 -std=c++17 -arch="$ARCH" -lineinfo -Xcompiler -fPIC -shared src/gcfdk.cu -o build/libgcfdk.so.new
  mv build/libgcfdk.so.new build/libgcfdk.so
fi
echo "BUILD_OK $ARCH"
