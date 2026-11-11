#!/bin/bash
# Compile the CUDA sources into build/libgcfd.so (inside WSL/Linux):  bash gpucfd/build.sh
# ARCH defaults to the RTX 4070 (compute capability 8.9); e.g. ARCH=sm_80 for an A100.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_89}"
"$NVCC" -O3 -std=c++17 -arch="$ARCH" -lineinfo -Xcompiler -fPIC -shared src/gcfd.cu -o build/libgcfd.so
echo "BUILD_OK $ARCH"
