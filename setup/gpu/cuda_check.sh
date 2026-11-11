#!/bin/bash
# Compile and run a tiny CUDA program: proves nvcc, the WSL GPU driver and double precision work.
set -e
export PATH=/usr/local/cuda/bin:$PATH
d=$(mktemp -d); cd "$d"
cat > t.cu <<'CU'
#include <cstdio>
__global__ void axpy(int n, double a, const double* x, double* y){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]+=a*x[i];}
int main(){ int n=1<<20; double *x,*y; cudaMallocManaged(&x,n*8); cudaMallocManaged(&y,n*8);
 for(int i=0;i<n;i++){x[i]=1;y[i]=2;} axpy<<<(n+255)/256,256>>>(n,3.0,x,y); cudaDeviceSynchronize();
 cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
 printf("%s, compute %d.%d, %.1f GB, y[0]=%g (expect 5)\n", p.name, p.major, p.minor, p.totalGlobalMem/1e9, y[0]); }
CU
nvcc -o t t.cu && ./t
