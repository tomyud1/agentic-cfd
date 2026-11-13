// gpucfd: GPU pressure Poisson solver (geometric multigrid, alone or as the CG preconditioner).
// Cell-centred structured grid, no ghost cells: boundary conditions are applied inside the stencils.
// Arrays hold the interior cells only, x fastest: index = i + nx * (j + ny * k).
// Builds into a shared library with a plain C API (see the bottom of this file and ../gcfd.py).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      abort();                                                                                 \
    }                                                                                          \
  } while (0)

enum Bc { PERIODIC = 0, NEUMANN = 1, DIRICHLET = 2 };

struct Level {
  int n[3];        // interior cells per direction (n = 1: inactive direction, 2D)
  int bc[6];       // x-, x+, y-, y+, z-, z+
  int coarsen[3];  // 2 if the next coarser level halves this direction, else 1
  double h[3];     // cell size
  double ih[3];    // 1/h and 1/h^2: kernels multiply instead of dividing (a double division costs about
  double ih2[3];   // 10 FP64 operations, and consumer GPUs run FP64 at 1/64 of the float rate)
  // Stretched grids (null: uniform, and every kernel takes its original path). One device array of
  // doubles holding, per direction d with n = n[d] cells, a block of 6 n + 2 values (see GEO_*):
  //   idc[n]    1 / cell width
  //   idm[n+1]  1 / distance between the cell centres on either side of face i (beyond a bounded side
  //             the neighbour is the mirror image of the first cell, so the distance is that cell's width)
  //   fw[n+1]   share of cell i in the flow through a box side centred on face i: dc_i / (dc_(i-1) + dc_i)
  //   ra[n]     restriction: weight of this cell's residual in its parent, and
  //   rb[n]     in the parent's neighbour on this cell's side (the transpose of the prolongation, times
  //             the ratio of the widths; 3/8 and 1/8 on a uniform grid; 1 and 0 if not coarsened)
  //   pw[n]     prolongation: weight of the parent's neighbour (1/4 on a uniform grid)
  const double* geo = nullptr;
  __host__ __device__ size_t stride(int d) const {
    return d == 0 ? size_t(1) : (d == 1 ? size_t(n[0]) : size_t(n[0]) * n[1]);
  }
  __host__ __device__ size_t cells() const { return size_t(n[0]) * n[1] * n[2]; }
};

__host__ __device__ __forceinline__ int geo_off(const Level& L, int d) {
  return d == 0 ? 0 : (d == 1 ? 6 * L.n[0] + 2 : 6 * (L.n[0] + L.n[1]) + 4);
}
__host__ __device__ __forceinline__ int geo_size(const Level& L) { return 6 * (L.n[0] + L.n[1] + L.n[2]) + 6; }
#define GEO_IDC(L, d) ((L).geo + geo_off(L, d))
#define GEO_IDM(L, d) ((L).geo + geo_off(L, d) + (L).n[d])
#define GEO_FW(L, d) ((L).geo + geo_off(L, d) + 2 * (L).n[d] + 1)
#define GEO_RA(L, d) ((L).geo + geo_off(L, d) + 3 * (L).n[d] + 2)
#define GEO_RB(L, d) ((L).geo + geo_off(L, d) + 4 * (L).n[d] + 2)
#define GEO_PW(L, d) ((L).geo + geo_off(L, d) + 5 * (L).n[d] + 2)

template <typename T>
struct Coef {
  T c[3];  // 1/h^2 per active direction, 0 for an inactive one (stretched grids: 1 or 0, a flag only)
  // Stretched grids (null: uniform): idc[n] then idm[n+1] of each direction in turn, in the solver's
  // precision (the multigrid runs in float; see Level::geo for the meaning).
  const T* s = nullptr;
  // Immersed bodies: coefficient of each cell's low face per direction (1 between fluid, 0 where the
  // face lies in a body: zero pressure gradient through it, so the bodies are walls for the pressure).
  // Coarser levels average the fine faces they cover. Null: no bodies, every face 1.
  const T* b[3] = {nullptr, nullptr, nullptr};
};

static const dim3 B3(32, 4, 2);
static dim3 grid3(const Level& L) {
  return dim3((L.n[0] + B3.x - 1) / B3.x, (L.n[1] + B3.y - 1) / B3.y, (L.n[2] + B3.z - 1) / B3.z);
}

#define CELL(L)                                                                                 \
  const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, \
            k = blockIdx.z * blockDim.z + threadIdx.z;                                         \
  if (i >= L.n[0] || j >= L.n[1] || k >= L.n[2]) return;                                      \
  const size_t c = i + size_t(L.n[0]) * (j + size_t(L.n[1]) * k);

// Neighbour of cell c in direction d on side -1/+1. Inside the domain: the neighbour's value.
// Across a periodic boundary: the cell on the other side. Across a Neumann (zero gradient) or
// Dirichlet (zero value at the face) boundary the virtual neighbour is +p[c] or -p[c]: the function
// returns 0 and sets self = +1 or -1 so that the caller can treat it implicitly.
template <typename T>
__device__ __forceinline__ T nbr(const T* p, const Level& L, size_t c, int ii, int d, int side, int& self) {
  const int n = L.n[d];
  const size_t s = L.stride(d);
  self = 0;
  if (side < 0) {
    if (ii > 0) return p[c - s];
    const int b = L.bc[2 * d];
    if (b == PERIODIC) return p[c + size_t(n - 1) * s];
    self = (b == NEUMANN) ? 1 : -1;
    return T(0);
  }
  if (ii < n - 1) return p[c + s];
  const int b = L.bc[2 * d + 1];
  if (b == PERIODIC) return p[c - size_t(n - 1) * s];
  self = (b == NEUMANN) ? 1 : -1;
  return T(0);
}

// Stretched grids: Laplacian(p)[c] = sum over directions of
//   (1 / dc_i) [ (p_(i+1) - p_i) / dm_(i+1) - (p_i - p_(i-1)) / dm_i ]
// the divergence (per cell width) of the gradient (per centre distance). Same boundary rules as below.
template <typename T>
__device__ __forceinline__ void stencil_s(const T* p, const Level& L, const Coef<T>& K, size_t c, int i, int j,
                                          int k, T& off, T& diag) {
  const int ii[3] = {i, j, k};
  off = T(0);
  diag = T(0);
  T full = T(0);  // bodies: the diagonal without them (for a cell with no fluid face)
  const T* blk = K.s;
#pragma unroll
  for (int d = 0; d < 3; ++d) {
    const T* w = blk;
    const T* g = blk + L.n[d];
    blk += 2 * L.n[d] + 1;
    if (K.c[d] == T(0)) continue;
    int sm, sp;
    const T vm = nbr(p, L, c, ii[d], d, -1, sm);
    const T vp = nbr(p, L, c, ii[d], d, +1, sp);
    T gm = g[ii[d]], gp = g[ii[d] + 1];
    const T wc = w[ii[d]];
    if (K.b[d]) {  // face coefficients (bodies), as in stencil(): a non-periodic side keeps its own rule
      const int n = L.n[d];
      const size_t s = L.stride(d);
      full += wc * (gm + gp);
      gm *= (ii[d] > 0 || L.bc[2 * d] == PERIODIC) ? K.b[d][c] : T(1);
      gp *= ii[d] < n - 1 ? K.b[d][c + s] : (L.bc[2 * d + 1] == PERIODIC ? K.b[d][c - size_t(n - 1) * s] : T(1));
    }
    off += wc * (gm * vm + gp * vp);
    diag += wc * (gm * T(1 - sm) + gp * T(1 - sp));
  }
  if (diag == T(0) && full != T(0)) diag = full;  // no fluid face (inside a body): decoupled
}

// Laplacian(p)[c] = off - diag * p[c]
template <typename T>
__device__ __forceinline__ void stencil(const T* p, const Level& L, const Coef<T>& K, size_t c, int i, int j,
                                        int k, T& off, T& diag) {
  if (K.s) {
    stencil_s(p, L, K, c, i, j, k, off, diag);
    return;
  }
  const int ii[3] = {i, j, k};
  off = T(0);
  diag = T(0);
#pragma unroll
  for (int d = 0; d < 3; ++d) {
    if (K.c[d] == T(0)) continue;
    int sm, sp;
    const T vm = nbr(p, L, c, ii[d], d, -1, sm);
    const T vp = nbr(p, L, c, ii[d], d, +1, sp);
    if (K.b[d]) {  // face coefficients (bodies); a non-periodic side keeps its own rule (coefficient 1)
      const int n = L.n[d];
      const size_t s = L.stride(d);
      const T wm = (ii[d] > 0 || L.bc[2 * d] == PERIODIC) ? K.b[d][c] : T(1);
      const T wp = ii[d] < n - 1 ? K.b[d][c + s] : (L.bc[2 * d + 1] == PERIODIC ? K.b[d][c - size_t(n - 1) * s] : T(1));
      off += K.c[d] * (wm * vm + wp * vp);
      diag += K.c[d] * (wm * T(1 - sm) + wp * T(1 - sp));
    } else {
      off += K.c[d] * (vm + vp);
      diag += K.c[d] * T(2 - sm - sp);
    }
  }
  if (diag == T(0)) diag = T(2) * (K.c[0] + K.c[1] + K.c[2]);  // no fluid face (inside a body): decoupled
}

template <typename T>
__global__ void k_apply(const T* p, T* out, Level L, Coef<T> K) {
  CELL(L)
  T off, diag;
  stencil(p, L, K, c, i, j, k, off, diag);
  out[c] = off - diag * p[c];
}

template <typename T>
__global__ void k_residual(const T* x, const T* b, T* r, Level L, Coef<T> K) {
  CELL(L)
  T off, diag;
  stencil(x, L, K, c, i, j, k, off, diag);
  r[c] = b[c] - (off - diag * x[c]);
}

// One colour of red-black Gauss-Seidel: cells with (i+j+k) % 2 == color are solved exactly.
template <typename T>
__global__ void k_rbgs(T* x, const T* b, Level L, Coef<T> K, int color) {
  CELL(L)
  if (((i + j + k) & 1) != color) return;
  T off, diag;
  stencil(x, L, K, c, i, j, k, off, diag);
  x[c] = (off - b[c]) / diag;
}

// Zebra line relaxation (stretched grids). One thread block per line of cells along direction dir;
// the lines whose position in the other two directions has parity `color` are solved exactly, all at
// once (their neighbours are lines of the other colour). In a long thin cell the neighbours across the
// thin direction are coupled (aspect ratio)^2 times more strongly than the others; a point smoother
// then leaves errors that wiggle along the long direction, and multigrid stalls. Solving each line's
// tridiagonal system removes exactly that strong coupling.
// The block's threads build the line's system in parallel (lower, diagonal, upper, right-hand side, in
// shared memory), one thread solves it by the Thomas algorithm, and all threads write the result back.
// Row m of the tridiagonal system of the line through I along dir:  l x_(m-1) + di x_m + h x_(m+1) = rh.
template <typename T>
__device__ __forceinline__ void line_row(const T* x, const T* b, const Level& L, const Coef<T>& K, const T* const* w,
                                         const T* const* g, const int* I, int dir, size_t c, T& l, T& di, T& h, T& rh) {
  const int m = I[dir], n = L.n[dir];
  T off = T(0), diag = T(0), full = T(0);
  l = T(0);
  h = T(0);
#pragma unroll
  for (int d = 0; d < 3; ++d) {
    if (K.c[d] == T(0)) continue;
    const T wc = w[d][I[d]];
    T gm = g[d][I[d]], gp = g[d][I[d] + 1];
    if (K.b[d]) {  // bodies: zero coefficient on a face in a body (same rule as stencil_s)
      const int nd = L.n[d];
      const size_t s = L.stride(d);
      full += wc * (gm + gp);
      gm *= (I[d] > 0 || L.bc[2 * d] == PERIODIC) ? K.b[d][c] : T(1);
      gp *= I[d] < nd - 1 ? K.b[d][c + s] : (L.bc[2 * d + 1] == PERIODIC ? K.b[d][c - size_t(nd - 1) * s] : T(1));
    }
    if (d == dir) {  // neighbours in the line are unknowns; a side keeps its rule (bounded directions only)
      if (m > 0) {
        l = wc * gm;
        diag += l;
      } else {
        diag += wc * gm * T(L.bc[2 * d] == NEUMANN ? 0 : 2);
      }
      if (m < n - 1) {
        h = wc * gp;
        diag += h;
      } else {
        diag += wc * gp * T(L.bc[2 * d + 1] == NEUMANN ? 0 : 2);
      }
    } else {
      int sm, sp;
      const T vm = nbr(x, L, c, I[d], d, -1, sm);
      const T vp = nbr(x, L, c, I[d], d, +1, sp);
      off += wc * (gm * vm + gp * vp);
      diag += wc * (gm * T(1 - sm) + gp * T(1 - sp));
    }
  }
  // A cell with no fluid face is an identity row (l = h = 0 there already), so a line cut by a body
  // falls apart into independent pieces.
  if (diag == T(0) && full != T(0)) diag = full;
  di = -diag;
  rh = b[c] - off;  // l x_(m-1) - diag x_m + h x_(m+1) = b - (the other lines' part)
}

template <typename T>
__global__ void k_line(T* x, const T* b, Level L, Coef<T> K, int dir, int color) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int n = L.n[dir];
  T* lo = reinterpret_cast<T*>(smem);
  T* di = lo + n;
  T* hi = di + n;
  T* rh = hi + n;
  const int a = dir == 0 ? 1 : 0, bq = dir == 2 ? 1 : 2;
  const int ia = int(blockIdx.x % unsigned(L.n[a])), ib = int(blockIdx.x / unsigned(L.n[a]));
  if (((ia + ib) & 1) != color) return;
  int I[3];
  I[a] = ia;
  I[bq] = ib;
  I[dir] = 0;
  const size_t c0 = I[0] + size_t(L.n[0]) * (I[1] + size_t(L.n[1]) * I[2]);
  const size_t st = L.stride(dir);
  const T* w[3];
  const T* g[3];
  {
    const T* blk = K.s;
    for (int d = 0; d < 3; ++d) {
      w[d] = blk;
      g[d] = blk + L.n[d];
      blk += 2 * L.n[d] + 1;
    }
  }
  for (int m = threadIdx.x; m < n; m += blockDim.x) {
    int J[3] = {I[0], I[1], I[2]};
    J[dir] = m;
    line_row(x, b, L, K, w, g, J, dir, c0 + size_t(m) * st, lo[m], di[m], hi[m], rh[m]);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    T cp = hi[0] / di[0], dp = rh[0] / di[0];
    hi[0] = cp;
    rh[0] = dp;
    for (int m = 1; m < n; ++m) {
      const T inv = T(1) / (di[m] - lo[m] * cp);
      cp = hi[m] * inv;
      hi[m] = cp;
      dp = (rh[m] - lo[m] * dp) * inv;
      rh[m] = dp;
    }
    for (int m = n - 2; m >= 0; --m) rh[m] -= hi[m] * rh[m + 1];
  }
  __syncthreads();
  for (int m = threadIdx.x; m < n; m += blockDim.x) x[c0 + size_t(m) * st] = rh[m];
}

// Coarsest level of a stretched grid: x = inv b with the inverse matrix computed once on the CPU (for a
// singular problem the inverse on right-hand sides of zero volume-weighted sum, giving the solution of
// zero volume-weighted mean). Exact, so long thin coarse cells cost nothing.
template <typename T>
__global__ void k_coarse_inv(T* x, const T* b, const T* inv, int n) {
  for (int r = threadIdx.x; r < n; r += blockDim.x) {
    T s = T(0);
    for (int q = 0; q < n; ++q) s += inv[size_t(r) * n + q] * b[q];
    x[r] = s;
  }
}

// Coarse value = average of its fine children.
template <typename T>
__global__ void k_restrict(const T* r, T* bc, Level F, Level C) {
  CELL(C)
  const int cx = F.coarsen[0], cy = F.coarsen[1], cz = F.coarsen[2];
  T s = T(0);
  for (int dz = 0; dz < cz; ++dz)
    for (int dy = 0; dy < cy; ++dy)
      for (int dx = 0; dx < cx; ++dx)
        s += r[(cx * i + dx) + size_t(F.n[0]) * ((cy * j + dy) + size_t(F.n[1]) * (cz * k + dz))];
  bc[c] = s / T(cx * cy * cz);
}

// Residual and restriction in one pass: coarse value = average of the fine children's residuals,
// computed in registers (the residual array is never written and read back). Same arithmetic in the
// same order as k_residual followed by k_restrict, so bit-identical results.
template <typename T>
__device__ __forceinline__ T resres_cell(const T* x, const T* b, const Level& F, const Coef<T>& K, int i, int j,
                                         int k) {
  const int cx = F.coarsen[0], cy = F.coarsen[1], cz = F.coarsen[2];
  T s = T(0);
  for (int dz = 0; dz < cz; ++dz)
    for (int dy = 0; dy < cy; ++dy)
      for (int dx = 0; dx < cx; ++dx) {
        const int fi = cx * i + dx, fj = cy * j + dy, fk = cz * k + dz;
        const size_t fc = fi + size_t(F.n[0]) * (fj + size_t(F.n[1]) * fk);
        T off, diag;
        stencil(x, F, K, fc, fi, fj, fk, off, diag);
        const T r = b[fc] - (off - diag * x[fc]);
        s += r;
      }
  return s / T(cx * cy * cz);
}
template <typename T>
__global__ void k_residual_restrict(const T* x, const T* b, T* bc, Level F, Coef<T> K, Level C) {
  CELL(C)
  bc[c] = resres_cell(x, b, F, K, i, j, k);
}

// Stretched grids: restriction as the exact transpose of the prolongation below (in the volume-weighted
// sum): a coarse cell collects the residual of every fine cell whose interpolated value it enters, its
// two children and the next fine cell on either side, per direction. Then the V-cycle is a symmetric
// operator, which conjugate gradients needs. Plain averaging of the children is not enough with line
// relaxation: a zebra sweep leaves the residual on one row of each coarse cell only, off its centre,
// and the V-cycle's antisymmetric part was measured at 10-17 % (1.5 % with the point smoother); CG stalled.
__device__ __forceinline__ int rstencil(const Level& F, int d, int I, int* idx, double* w) {
  if (F.coarsen[d] != 2) {
    idx[0] = I;
    w[0] = 1.0;
    return 1;
  }
  const int n = F.n[d];
  const bool per = F.bc[2 * d] == PERIODIC;
  const double *ra = GEO_RA(F, d), *rb = GEO_RB(F, d);
  int m = 0;
  idx[m] = 2 * I;
  w[m++] = ra[2 * I];
  idx[m] = 2 * I + 1;
  w[m++] = ra[2 * I + 1];
  int lo = 2 * I - 1, hi = 2 * I + 2;
  if (lo < 0 && per) lo = n - 1;
  if (hi >= n && per) hi = 0;
  if (lo >= 0) {
    idx[m] = lo;
    w[m++] = rb[lo];
  }
  if (hi < n) {
    idx[m] = hi;
    w[m++] = rb[hi];
  }
  return m;
}
template <typename T>
__device__ __forceinline__ T restrict_cell(const T* r, const Level& F, int i, int j, int k) {
  int ix[4], iy[4], iz[4];
  double wx[4], wy[4], wz[4];
  const int mx = rstencil(F, 0, i, ix, wx), my = rstencil(F, 1, j, iy, wy), mz = rstencil(F, 2, k, iz, wz);
  double s = 0.0;
  for (int c3 = 0; c3 < mz; ++c3)
    for (int c2 = 0; c2 < my; ++c2) {
      const double w23 = wy[c2] * wz[c3];
      const size_t row = size_t(F.n[0]) * (iy[c2] + size_t(F.n[1]) * iz[c3]);
      for (int c1 = 0; c1 < mx; ++c1) s += w23 * wx[c1] * double(r[ix[c1] + row]);
    }
  return T(s);
}
template <typename T>
__global__ void k_restrict_s(const T* r, T* bc, Level F, Level C) {
  CELL(C)
  bc[c] = restrict_cell(r, F, i, j, k);
}

// Linear interpolation weights along one direction for a cell-centred fine cell i: 3/4 from its
// parent, 1/4 from the parent's other neighbour (boundary: periodic wrap, or the parent itself with
// +1/4 for Neumann and -1/4 for Dirichlet, which puts the interpolated value to zero at the face).
__device__ __forceinline__ void weights1(int i, int cf, int Nc, int bl, int bh, int* I, double* w) {
  if (cf == 1) {
    I[0] = i; w[0] = 1.0; I[1] = i; w[1] = 0.0;
    return;
  }
  const int Ic = i >> 1;
  int o = (i & 1) ? Ic + 1 : Ic - 1;
  double wo = 0.25;
  if (o < 0) {
    if (bl == PERIODIC) o = Nc - 1;
    else { o = Ic; wo = (bl == NEUMANN) ? 0.25 : -0.25; }
  } else if (o >= Nc) {
    if (bh == PERIODIC) o = 0;
    else { o = Ic; wo = (bh == NEUMANN) ? 0.25 : -0.25; }
  }
  I[0] = Ic; w[0] = 0.75; I[1] = o; w[1] = wo;
}

// Interpolated coarse correction at fine cell (i,j,k) (trilinear, weights1 per direction).
template <typename T>
__device__ __forceinline__ double prolong_cell(const T* e, const Level& F, const Level& C, int i, int j, int k) {
  const int ii[3] = {i, j, k};
  int I[3][2];
  double w[3][2];
#pragma unroll
  for (int d = 0; d < 3; ++d) weights1(ii[d], F.coarsen[d], C.n[d], F.bc[2 * d], F.bc[2 * d + 1], I[d], w[d]);
  if (F.geo) {  // stretched: linear in the true positions of the cell centres (same neighbours and signs)
#pragma unroll
    for (int d = 0; d < 3; ++d) {
      if (F.coarsen[d] != 2) continue;
      const double wo = GEO_PW(F, d)[ii[d]];
      w[d][1] = w[d][1] < 0.0 ? -wo : wo;
      w[d][0] = 1.0 - wo;
    }
  }
  double s = 0.0;
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const double ww = w[0][a] * w[1][b] * w[2][q];
        if (ww != 0.0) s += ww * double(e[I[0][a] + size_t(C.n[0]) * (I[1][b] + size_t(C.n[1]) * I[2][q])]);
      }
  return s;
}
template <typename T>
__global__ void k_prolong_add(T* x, const T* e, Level F, Level C) {
  CELL(F)
  x[c] += T(prolong_cell(e, F, C, i, j, k));
}

__device__ double block_sum(double v) {
  __shared__ double sh[1024];
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
    __syncthreads();
  }
  const double s = sh[0];
  __syncthreads();
  return s;
}

// Loops over all cells of a level by one thread block (cell index c and its i, j, k).
#define BLOCK_CELLS(L, ...)                                                       \
  {                                                                               \
    const unsigned N_ = unsigned((L).cells()), n0_ = unsigned((L).n[0]), n1_ = unsigned((L).n[1]); \
    for (unsigned c = threadIdx.x; c < N_; c += blockDim.x) {                     \
      const unsigned r_ = c / n0_;                                                \
      const int i = int(c - r_ * n0_), j = int(r_ % n1_), k = int(r_ / n1_);      \
      (void)i; (void)j; (void)k;                                                  \
      __VA_ARGS__                                                                 \
    }                                                                             \
    __syncthreads();                                                              \
  }

// With bodies: does cell c have a face with a non-zero coefficient (else it is decoupled)?
template <typename T>
__device__ __forceinline__ bool is_active(const Level& L, const Coef<T>& K, size_t c, int i, int j, int k) {
  const int ii[3] = {i, j, k};
  for (int d = 0; d < 3; ++d) {
    if (K.c[d] == T(0)) continue;
    if (!K.b[d]) return true;
    const int n = L.n[d];
    const size_t s = L.stride(d);
    const bool per = L.bc[2 * d] == PERIODIC;
    const T wm = (ii[d] > 0 || per) ? K.b[d][c] : T(L.bc[2 * d] == DIRICHLET ? 1 : 0);
    const T wp = ii[d] < n - 1 ? K.b[d][c + s] : (per ? K.b[d][c - size_t(n - 1) * s] : T(L.bc[2 * d + 1] == DIRICHLET ? 1 : 0));
    if (wm > T(0) || wp > T(0)) return true;
  }
  return false;
}

// Coarsest level: one thread block, many red-black sweeps from x = 0. For a singular problem the
// right-hand side is shifted to zero mean first and the solution returned with zero mean (with bodies:
// the mean over the cells with fluid faces, the region the constant lives on).
template <typename T>
__device__ void coarse_solve(T* x, const T* b, const Level& L, const Coef<T>& K, int sweeps, int singular) {
  const size_t N = L.cells();
  if (K.b[0] || K.b[1] || K.b[2]) {
    double sb = 0.0, cnt = 0.0, mb = 0.0;
    if (singular) {
      for (unsigned c = threadIdx.x; c < N; c += blockDim.x) {
        const unsigned r = c / unsigned(L.n[0]);
        const int i = int(c - r * unsigned(L.n[0])), j = int(r % unsigned(L.n[1])), k = int(r / unsigned(L.n[1]));
        if (is_active(L, K, c, i, j, k)) {
          sb += double(b[c]);
          cnt += 1.0;
        }
      }
      const double tb = block_sum(sb), tc = block_sum(cnt);
      mb = tc > 0.0 ? tb / tc : 0.0;
    }
    for (size_t c = threadIdx.x; c < N; c += blockDim.x) x[c] = T(0);
    __syncthreads();
    for (int s = 0; s < sweeps; ++s)
      for (int color = 0; color < 2; ++color) BLOCK_CELLS(L, {
          if (((i + j + k) & 1) != color) continue;
          T off, diag;
          stencil(x, L, K, c, i, j, k, off, diag);
          const T bb = is_active(L, K, c, i, j, k) ? b[c] - T(mb) : T(0);
          x[c] = (off - bb) / diag;
        })
    if (singular) {
      double sx = 0.0, cn = 0.0;
      BLOCK_CELLS(L, {
        if (is_active(L, K, c, i, j, k)) {
          sx += double(x[c]);
          cn += 1.0;
        }
      })
      const double tx = block_sum(sx), tn = block_sum(cn);
      const double mx = tn > 0.0 ? tx / tn : 0.0;
      BLOCK_CELLS(L, {
        if (is_active(L, K, c, i, j, k)) x[c] -= T(mx);
      })
    }
    __syncthreads();
    return;
  }
  double mb = 0.0;
  if (singular) {
    double s = 0.0;
    for (size_t c = threadIdx.x; c < N; c += blockDim.x) s += double(b[c]);
    mb = block_sum(s) / double(N);
  }
  for (size_t c = threadIdx.x; c < N; c += blockDim.x) x[c] = T(0);
  __syncthreads();
  for (int s = 0; s < sweeps; ++s)
    for (int color = 0; color < 2; ++color) BLOCK_CELLS(L, {
        if (((i + j + k) & 1) != color) continue;
        T off, diag;
        stencil(x, L, K, c, i, j, k, off, diag);
        x[c] = (off - (b[c] - T(mb))) / diag;
      })
  if (singular) {
    double s = 0.0;
    for (size_t c = threadIdx.x; c < N; c += blockDim.x) s += double(x[c]);
    const double mx = block_sum(s) / double(N);
    for (size_t c = threadIdx.x; c < N; c += blockDim.x) x[c] -= T(mx);
  }
  __syncthreads();
}
static const int COARSE_THREADS = 512;
template <typename T>
__global__ void k_coarse(T* x, const T* b, Level L, Coef<T> K, int sweeps, int singular) {
  coarse_solve(x, b, L, K, sweeps, singular);
}

// The V-cycle below a small level, all inside one thread block and its shared memory (levels of at
// most g_tail_cells cells). On small grids each multigrid pass does little work and waits on memory
// latency; in shared memory that wait is short, and the levels l0..last cost one launch instead of
// about 11 per level. Same per-cell arithmetic as the separate kernels (a red-black colour's cells do
// not depend on each other, so the order within a colour does not matter): bit-identical results.
static size_t g_tail_cells = 1024;          // levels this small or smaller go into the tail (0: none)
static const size_t TAIL_SHARED = 40 * 1024;  // bytes of shared memory the tail may use (x and b per level)
// Levels in the tail. The whole Tail is a kernel argument (at most 4096 bytes). A tail holds at most
// TAIL_SHARED bytes of cells that at least halve from level to level, so it never has more than 13 levels.
static const int MAX_LEVELS = 16;
template <typename T>
struct Tail {
  Level L[MAX_LEVELS];
  Coef<T> K[MAX_LEVELS];
  T* x;        // level l0: solution (in/out)
  const T* b;  // level l0: right-hand side
  int n, nu1, nu2, sweeps, singular;  // levels 0..n-1 here are levels l0..last of the solver
};
template <typename T>
__global__ void k_vtail(Tail<T> t) {
  extern __shared__ __align__(16) unsigned char smem[];
  T* X[MAX_LEVELS];
  T* B[MAX_LEVELS];
  T* m = reinterpret_cast<T*>(smem);
  for (int l = 0; l < t.n; ++l) {
    X[l] = m;
    B[l] = m + t.L[l].cells();
    m += 2 * t.L[l].cells();
  }
  BLOCK_CELLS(t.L[0], {
    X[0][c] = t.x[c];
    B[0][c] = t.b[c];
  })
  const int last = t.n - 1;
  for (int l = 0; l < last; ++l) {  // down: smooth, residual to the coarser level, zero its correction
    const Level& L = t.L[l];
    for (int s = 0; s < t.nu1; ++s)
      for (int color = 0; color < 2; ++color) BLOCK_CELLS(L, {
          if (((i + j + k) & 1) != color) continue;
          T off, diag;
          stencil(X[l], L, t.K[l], c, i, j, k, off, diag);
          X[l][c] = (off - B[l][c]) / diag;
        })
    BLOCK_CELLS(t.L[l + 1], {
      B[l + 1][c] = resres_cell(X[l], B[l], L, t.K[l], i, j, k);
      X[l + 1][c] = T(0);
    })
  }
  coarse_solve(X[last], B[last], t.L[last], t.K[last], t.sweeps, t.singular);
  for (int l = last - 1; l >= 0; --l) {  // up: add the interpolated correction, smooth backwards
    const Level& L = t.L[l];
    BLOCK_CELLS(L, { X[l][c] += T(prolong_cell(X[l + 1], L, t.L[l + 1], i, j, k)); })
    for (int s = 0; s < t.nu2; ++s)
      for (int color = 1; color >= 0; --color) BLOCK_CELLS(L, {
          if (((i + j + k) & 1) != color) continue;
          T off, diag;
          stencil(X[l], L, t.K[l], c, i, j, k, off, diag);
          X[l][c] = (off - B[l][c]) / diag;
        })
  }
  BLOCK_CELLS(t.L[0], { t.x[c] = X[0][c]; })
}

template <typename T>
__global__ void k_dot(const T* a, const T* b, size_t n, double* part) {  // one partial sum per block
  double s = 0.0;
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    s += double(a[q]) * (b ? double(b[q]) : 1.0);
  s = block_sum(s);
  if (threadIdx.x == 0) part[blockIdx.x] = s;
}

// Stretched grids: sum of a b w with w the volume of each unknown's box. The operators there are
// symmetric in this weighted sum (not in the plain one), which is what conjugate gradients needs.
template <typename T>
__global__ void k_dotw(const T* a, const T* b, const T* w, size_t n, double* part) {
  double s = 0.0;
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    s += double(a[q]) * (b ? double(b[q]) : 1.0) * double(w[q]);
  s = block_sum(s);
  if (threadIdx.x == 0) part[blockIdx.x] = s;
}

__global__ void k_sum_parts(const double* part, int np, double* out) {  // one block, fixed order
  double s = 0.0;
  for (int q = threadIdx.x; q < np; q += blockDim.x) s += part[q];
  s = block_sum(s);
  if (threadIdx.x == 0) *out = s;
}


template <typename T>
__global__ void k_axpy(T* y, const T* x, double a, size_t n) {  // y += a x
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    y[q] += T(a) * x[q];
}

template <typename T>
__global__ void k_xpby(T* p, const T* z, double b, size_t n) {  // p = z + b p
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    p[q] = z[q] + T(b) * p[q];
}

template <typename T>
__global__ void k_add_const(T* y, double a, size_t n) {
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    y[q] += T(a);
}

template <typename A, typename B>
__global__ void k_convert(const A* a, B* b, size_t n) {
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    b[q] = B(a[q]);
}

__global__ void k_copy4(const float4* a, float4* b, size_t n) {
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    b[q] = a[q];
}

static int lin_blocks(size_t n) {
  const size_t b = (n + 255) / 256;
  return int(b < 2048 ? (b > 0 ? b : 1) : 2048);
}

static const int DOT_PARTS = 2048;  // partial sums of a dot product (one per block, lin_blocks <= 2048)

// ================================================================================================
// Streams, CUDA graphs and loops decided on the GPU.
// All GPU work goes to one stream (created blocking, so it is also ordered with the synchronous host
// copies on the legacy default stream). Under WSL a kernel launch costs about 25 us and a CPU-GPU round
// trip about 63 us (measured), while on small grids a kernel does a few us of work; the first version
// spent most of a time step on launches and on a round trip per dot product. So a whole pressure solve,
// or a whole Navier-Stokes time step, is recorded once as a CUDA graph and replayed with one launch.
//
// An iterative solve stops on a tolerance, and a recorded graph has no CPU to decide that. So every
// scalar of the conjugate-gradient iterations (dot products, alpha, beta, the stop test) stays on the
// GPU, and each loop is a graph WHILE node (CUDA 12.4+): the GPU decides whether to run the next
// iteration. With graphs off the same kernels run in the same order and the CPU reads the flag before
// each iteration, so results are bit-identical either way (rung 8 checks it).
// ================================================================================================
static bool g_use_graphs = true;
static bool g_capturing = false;       // recording a graph now
static cudaStream_t g_main = nullptr;  // the stream everything runs on
static cudaStream_t g_body = nullptr;  // the stream a loop body is recorded on
static cudaStream_t g_cur = nullptr;   // where launches go: g_main, or g_body while recording a loop body
static cudaStream_t gs() {
  if (!g_main) {
    CK(cudaStreamCreate(&g_main));
    CK(cudaStreamCreate(&g_body));
    g_cur = g_main;
  }
  return g_cur;
}

// The graph-building calls gained an edge-data argument in CUDA 13 (untested here: this machine has 12.9).
static void capture_info(cudaStream_t s, cudaGraph_t* g, const cudaGraphNode_t** deps, size_t* nd) {
  cudaStreamCaptureStatus status;
  unsigned long long id;
#if CUDART_VERSION >= 13000
  CK(cudaStreamGetCaptureInfo(s, &status, &id, g, deps, nullptr, nd));
#else
  CK(cudaStreamGetCaptureInfo(s, &status, &id, g, deps, nd));
#endif
}
static void add_node_after_capture(cudaStream_t s, cudaGraphNode_t* node, cudaGraphNodeParams* p) {
  cudaGraph_t g;
  const cudaGraphNode_t* deps;
  size_t nd;
  capture_info(s, &g, &deps, &nd);
#if CUDART_VERSION >= 13000
  CK(cudaGraphAddNode(node, g, deps, nullptr, nd, p));
  CK(cudaStreamUpdateCaptureDependencies(s, node, nullptr, 1, cudaStreamSetCaptureDependencies));
#else
  CK(cudaGraphAddNode(node, g, deps, nd, p));
  CK(cudaStreamUpdateCaptureDependencies(s, node, 1, cudaStreamSetCaptureDependencies));
#endif
}

__global__ void k_set_cond(const int* go, cudaGraphConditionalHandle h) { cudaGraphSetConditional(h, *go ? 1u : 0u); }

// while (*go) body();  where body updates *go on the GPU. Loops cannot be nested.
template <class Body>
void dev_while(const int* go, Body body) {
  gs();
  if (!g_capturing) {  // run now: the CPU reads the flag before each pass
    for (;;) {
      int h = 0;
      CK(cudaMemcpyAsync(&h, go, sizeof(int), cudaMemcpyDeviceToHost, g_main));
      CK(cudaStreamSynchronize(g_main));
      if (!h) return;
      body();
    }
  }
  if (g_cur != g_main) {
    fprintf(stderr, "gcfd: nested device loops are not supported\n");
    abort();
  }
  cudaGraph_t g;
  const cudaGraphNode_t* deps;
  size_t nd;
  capture_info(g_main, &g, &deps, &nd);
  cudaGraphConditionalHandle h;
  CK(cudaGraphConditionalHandleCreate(&h, g, 0, cudaGraphCondAssignDefault));
  k_set_cond<<<1, 1, 0, g_main>>>(go, h);  // the test before the first pass
  cudaGraphNodeParams p = {};
  p.type = cudaGraphNodeTypeConditional;
  p.conditional.handle = h;
  p.conditional.type = cudaGraphCondTypeWhile;
  p.conditional.size = 1;
  cudaGraphNode_t node;
  add_node_after_capture(g_main, &node, &p);
  CK(cudaStreamBeginCaptureToGraph(g_body, p.conditional.phGraph_out[0], nullptr, nullptr, 0,
                                   cudaStreamCaptureModeThreadLocal));
  g_cur = g_body;
  body();
  k_set_cond<<<1, 1, 0, g_body>>>(go, h);  // the test before the next pass
  g_cur = g_main;
  cudaGraph_t b;
  CK(cudaStreamEndCapture(g_body, &b));
}

// Recorded graphs, keyed by everything their kernels' arguments depend on (scalars, array pointers).
struct GraphCache {
  struct Entry {
    std::vector<unsigned long long> key;
    cudaGraphExec_t exec;
  };
  std::vector<Entry> e;
  ~GraphCache() {
    for (auto& x : e) cudaGraphExecDestroy(x.exec);
  }
  template <class Body>
  cudaGraphExec_t get(const std::vector<unsigned long long>& key, Body body) {  // record if new
    for (auto& x : e)
      if (x.key == key) return x.exec;
    if (e.size() >= 8) {
      CK(cudaGraphExecDestroy(e.front().exec));
      e.erase(e.begin());
    }
    gs();
    cudaGraph_t g;
    g_capturing = true;
    CK(cudaStreamBeginCapture(g_main, cudaStreamCaptureModeThreadLocal));
    body();
    CK(cudaStreamEndCapture(g_main, &g));
    g_capturing = false;
    cudaGraphExec_t x;
    CK(cudaGraphInstantiate(&x, g, 0));
    CK(cudaGraphDestroy(g));
    e.push_back({key, x});
    return x;
  }
  template <class Body>
  void run(const std::vector<unsigned long long>& key, Body body) {
    if (!g_use_graphs || g_capturing) {  // graphs off, or part of a graph being recorded
      body();
      return;
    }
    CK(cudaGraphLaunch(get(key, body), g_main));
  }
};
static unsigned long long kb(double v) {
  unsigned long long u;
  std::memcpy(&u, &v, sizeof u);
  return u;
}

// ------------------------------------------------------------------ conjugate gradients on the GPU
// Scalars of one CG solve, in GPU memory. Updated by one-thread kernels with exactly the arithmetic the
// CPU did before (same operations, same order), so moving them changed no result.
struct CGState {
  double fn, rr, rz, rznew, pq, alpha, beta, tol, stop, s;
  int it, maxit, go, first;
  long long total_its, solves;  // accumulated over solves (iterations per solve = total_its / solves)
  double omega, ts, tt;         // BiCGStab only (k_bi_*)
};

// Dot product (sum if b is null) into *out on the GPU: per-block partial sums, then one block adds them
// in a fixed order. Deterministic (an atomicAdd of block sums made runs differ at 4e-16).
template <typename T>
void dot_dev(const T* a, const T* b, size_t n, double* part, double* out) {
  const int nb = lin_blocks(n);
  k_dot<T><<<nb, 256, 0, gs()>>>(a, b, n, part);
  k_sum_parts<<<1, 1024, 0, gs()>>>(part, nb, out);
}

template <typename T>
void dotw_dev(const T* a, const T* b, const T* w, size_t n, double* part, double* out) {  // w null: plain
  if (!w) {
    dot_dev<T>(a, b, n, part, out);
    return;
  }
  const int nb = lin_blocks(n);
  k_dotw<T><<<nb, 256, 0, gs()>>>(a, b, w, n, part);
  k_sum_parts<<<1, 1024, 0, gs()>>>(part, nb, out);
}
template <typename T>
__global__ void k_sub_wmean(T* a, const double* sum, double denom, size_t n) {  // a -= sum / denom
  const double m = *sum / denom;
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    a[q] += T(-m);
}

template <typename T>
__global__ void k_axpy_d(T* y, const T* x, const double* a, double sign, size_t n) {  // y += sign a x
  const double s = sign * *a;
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    y[q] += T(s) * x[q];
}
template <typename T>
__global__ void k_xpby_d(T* p, const T* z, const double* b, const int* first, size_t n) {  // p = z + b p
  const bool f = *first != 0;
  const double bb = *b;
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    p[q] = f ? z[q] : z[q] + T(bb) * p[q];
}
template <typename T>
__global__ void k_sub_mean(T* a, const double* sum, size_t n) {
  const double m = *sum / double(n);
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    a[q] += T(-m);
}
// Mean over the cells marked in act (1 or 0) removed from those cells (bodies: only the fluid region
// carries the constant that the pressure is undetermined by).
template <typename T>
__global__ void k_sub_mean_m(T* a, const T* act, const double* sum, double nact, size_t n) {
  const double m = *sum / nact;
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    a[q] += T(-m) * act[q];
}

// fmin (optional): a floor under the norm the tolerance is relative to (see NS::project)
__global__ void k_pcg_start(CGState* g, double tol, int maxit, const double* fmin = nullptr) {
  g->fn = (fmin ? fmax(sqrt(g->s), *fmin) : sqrt(g->s)) + 1e-300;
  g->tol = tol;
  g->maxit = maxit;
  g->it = 0;
  g->first = 1;
}
// relative residual of iteration it (after counting it if inc), and whether to go on
__global__ void k_pcg_test(CGState* g, double* hist, int inc) {
  if (inc) ++g->it;
  const double r = sqrt(g->rr) / g->fn;
  hist[g->it] = r;
  g->go = (g->it < g->maxit && r > g->tol) ? 1 : 0;
}
__global__ void k_pcg_final(CGState* g, double* hist) { hist[g->it] = sqrt(g->rr) / g->fn; }
__global__ void k_pcg_beta(CGState* g) { g->beta = g->first ? 0.0 : g->rznew / g->rz; }
__global__ void k_pcg_alpha(CGState* g) {
  g->alpha = g->rznew / g->pq;
  g->rz = g->rznew;
  g->first = 0;
}
__global__ void k_cg_done(CGState* g) {
  g->total_its += g->it;
  g->solves += 1;
}
__global__ void k_cg_reset(CGState* g) {
  g->total_its = 0;
  g->solves = 0;
}
// ------------------------------------------------------------------ BiCGStab on the GPU
// For the two systems of the IMEX scheme with immersed bodies, which are not symmetric (NS::bicgstab):
// the same scalars and the same kind of one-thread kernels as conjugate gradients.
// rz = (r0, r) of the last iteration, rznew = this one's, pq = (r0, v), ts = (t, s), tt = (t, t).
__global__ void k_bi_start(CGState* g, double tol, int maxit, const double* fmin = nullptr) {  // after s = b.b, rr = r.r
  g->fn = (fmin ? fmax(sqrt(g->s), *fmin) : sqrt(g->s)) + 1e-300;
  g->tol = tol;
  g->maxit = maxit;
  g->it = 0;
  g->first = 1;
  g->go = (sqrt(g->rr) / g->fn > tol) ? 1 : 0;
}
__global__ void k_bi_beta(CGState* g) {
  g->beta = (g->first || g->rz == 0.0 || g->omega == 0.0) ? 0.0 : (g->rznew / g->rz) * (g->alpha / g->omega);
  g->rz = g->rznew;
}
__global__ void k_bi_alpha(CGState* g) { g->alpha = g->pq != 0.0 ? g->rz / g->pq : 0.0; }
__global__ void k_bi_omega(CGState* g) { g->omega = g->tt > 0.0 ? g->ts / g->tt : 0.0; }
__global__ void k_bi_test(CGState* g) {  // after rr = r.r of the new residual
  ++g->it;
  g->first = 0;
  g->go = (g->it < g->maxit && sqrt(g->rr) / g->fn > g->tol) ? 1 : 0;  // a NaN residual stops the loop
}
template <typename T>
__global__ void k_bi_p(T* p, const T* r, const T* v, const CGState* g, size_t n) {  // p = r + beta (p - omega v)
  const bool f = g->first != 0;
  const double be = g->beta, om = g->omega;
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    p[q] = f ? r[q] : r[q] + T(be) * (p[q] - T(om) * v[q]);
}

static CGState* new_cg() {
  CGState* g;
  CK(cudaMalloc(&g, sizeof(CGState)));
  CK(cudaMemset(g, 0, sizeof(CGState)));
  return g;
}
static CGState read_cg(const CGState* g) {
  CGState h;
  CK(cudaMemcpy(&h, g, sizeof h, cudaMemcpyDeviceToHost));
  return h;
}

struct Base {
  virtual ~Base() {}
  virtual int levels(int* dims) const = 0;
  virtual void apply(const double* p, double* out) = 0;
  virtual int solve(const double* f, double* x, double tol, int maxit, int method, double* hist, double* ms) = 0;
  virtual double bench_vcycle(int reps) = 0;
  // One V-cycle on the finest level from zero for Laplacian(Z) = R, with R and Z in double precision
  // (converted to this solver's precision inside): the preconditioner of a mixed-precision solve.
  virtual void vcycle_d(const double* R, double* Z) = 0;
};

template <typename T>
struct Solver : Base {
  std::vector<Level> L;
  std::vector<Coef<T>> K;
  std::vector<T*> x, b, r;
  T *p_ = nullptr, *q_ = nullptr, *z_ = nullptr, *rr_ = nullptr;
  bool singular = true;
  int nu1, nu2, coarse_sweeps;
  Base* pre = nullptr;  // mixed precision: single-precision multigrid used as the preconditioner
  CGState* cg = nullptr;
  double* part = nullptr;  // partial sums of dot products
  double* hist = nullptr;  // relative residual per iteration (GPU)
  int hist_n = 0;
  int tail_from = 0;
  GraphCache graphs;
  std::vector<T*> mb;      // face coefficients of all levels (bodies), freed with the solver
  T* act = nullptr;        // finest level: 1 on cells that have a fluid face, 0 inside bodies
  double n_act = 0.0;

  // Bodies: fine[d] = coefficient of each cell's low d-face (1 fluid, 0 in a body; null for an inactive
  // direction). Coarser levels (if mask_levels() > 1) get the average of the fine faces they cover.
  // A closed fluid region (no outflow) makes the problem singular, and its right-hand side has to sum
  // to zero over the region. The divergence there sums to the net flow through the faces in the bodies,
  // which are extrapolated, not projected, so that sum is a small discretization error, not exactly 0.
  // It is removed in the cut cells (active cells with a face in a body), where the error arises, rather
  // than spread over the whole flow as a uniform divergence.
  T* cut = nullptr;
  double n_cut = 0.0;
  // Stretched grids (see Level::geo): device arrays per level, the cell volumes of the finest level
  // (weights of the dot products and of the mean), and the inverse matrix of the coarsest level.
  bool stretched = false;
  std::vector<double*> geo_d;
  std::vector<T*> s_d;
  T* vol = nullptr;
  double vtot = 0.0;
  // Stretched grids with bodies: cell volume on the cells that have a fluid face (0 inside bodies), and
  // its sum there and over the cut cells: the sums and means of a singular problem run over the fluid.
  std::vector<double> vol_h;
  T* vact = nullptr;
  double vact_tot = 0.0, vcut_tot = 0.0;
  T* cinv = nullptr;
  int line_dirs = 0;        // bit d: zebra line relaxation along direction d
  const char* error = nullptr;  // set-up refused (reason), checked by the C API
  // Which directions may use line relaxation (bit mask; experiments: 0 = point smoother on stretched grids).
  static int& line_mask() {
    static int m = 7;
    return m;
  }
  static const int LINE_SHARED = 48 * 1024;  // shared memory of one line: 4 arrays of n values

  // Builds every level's lists from the face positions (xf[d] null: that direction is uniform). Coarser
  // levels keep every second face, so their cells are the unions of two fine cells.
  void build_geo(const double* len, const double* const* xf) {
    std::vector<std::vector<double>> faces(3), nxt(3);
    for (int d = 0; d < 3; ++d) {
      const int n = L[0].n[d];
      faces[d].resize(n + 1);
      for (int i = 0; i <= n; ++i) faces[d][i] = xf[d] ? xf[d][i] : len[d] * i / n;
    }
    std::vector<double> glast;
    for (size_t l = 0; l < L.size(); ++l) {
      Level& F = L[l];
      const bool has_c = l + 1 < L.size();
      std::vector<double> g(geo_size(F), 0.0), st;
      std::vector<std::vector<double>> dcs(3);
      for (int d = 0; d < 3; ++d) {
        const int n = F.n[d];
        const std::vector<double>& x = faces[d];
        const bool per = F.bc[2 * d] == PERIODIC;
        double* idc = g.data() + geo_off(F, d);
        double *idm = idc + n, *fw = idc + 2 * n + 1, *ra = idc + 3 * n + 2, *rb = idc + 4 * n + 2, *pw = idc + 5 * n + 2;
        std::vector<double>& dc = dcs[d];
        dc.resize(n);
        for (int i = 0; i < n; ++i) {
          dc[i] = x[i + 1] - x[i];
          idc[i] = 1.0 / dc[i];
        }
        for (int i = 1; i < n; ++i) {
          idm[i] = 2.0 / (dc[i - 1] + dc[i]);
          fw[i] = dc[i] / (dc[i - 1] + dc[i]);
        }
        if (per) {
          idm[0] = idm[n] = 2.0 / (dc[n - 1] + dc[0]);
          fw[0] = fw[n] = dc[0] / (dc[n - 1] + dc[0]);
        } else {  // mirror cell beyond the side
          idm[0] = 1.0 / dc[0];
          idm[n] = 1.0 / dc[n - 1];
          fw[0] = fw[n] = 0.5;
        }
        const int cf = has_c ? F.coarsen[d] : 1;
        nxt[d].clear();
        if (cf == 2)
          for (int I = 0; I <= n / 2; ++I) nxt[d].push_back(x[2 * I]);
        else
          nxt[d] = x;
        for (int i = 0; i < n; ++i) {
          if (cf != 2) {
            ra[i] = 1.0;
            rb[i] = 0.0;
            pw[i] = 0.0;
            continue;
          }
          const std::vector<double>& X = nxt[d];
          const int Ic = i >> 1, Nc = n / 2;
          const double wc = X[Ic + 1] - X[Ic];
          const double xc = 0.5 * (x[i] + x[i + 1]), Xc = 0.5 * (X[Ic] + X[Ic + 1]);
          int o = (i & 1) ? Ic + 1 : Ic - 1;
          double dist;
          bool outside = false;
          if (o >= 0 && o < Nc) {
            dist = std::fabs(0.5 * (X[o] + X[o + 1]) - Xc);
          } else if (per) {
            o = o < 0 ? Nc - 1 : 0;
            dist = 0.5 * (wc + (X[o + 1] - X[o]));
          } else {
            dist = wc;  // mirror cell
            outside = true;
          }
          pw[i] = std::fabs(xc - Xc) / dist;
          // Transpose of the prolongation, times (fine width / coarse width). Beyond a bounded side the
          // neighbour is the parent itself: weight 1 in all (zero gradient) or 1 - 2 pw (zero value).
          if (outside) {
            const int b = F.bc[2 * d + (o < 0 ? 0 : 1)];
            ra[i] = (b == NEUMANN ? 1.0 : 1.0 - 2.0 * pw[i]) * dc[i] / wc;
            rb[i] = 0.0;
          } else {
            ra[i] = (1.0 - pw[i]) * dc[i] / wc;
            rb[i] = pw[i] * dc[i] / (X[o + 1] - X[o]);
          }
        }
        st.insert(st.end(), idc, idc + n);
        st.insert(st.end(), idm, idm + n + 1);
      }
      double* gd;
      CK(cudaMalloc(&gd, g.size() * sizeof(double)));
      CK(cudaMemcpy(gd, g.data(), g.size() * sizeof(double), cudaMemcpyHostToDevice));
      geo_d.push_back(gd);
      F.geo = gd;
      T* sd;
      CK(cudaMalloc(&sd, st.size() * sizeof(T)));
      upload(st.data(), sd, st.size());
      s_d.push_back(sd);
      if (l == 0) {  // cell volumes of the finest level
        const size_t N = F.cells();
        std::vector<double> v(N);
        vtot = 0.0;
        for (int k = 0; k < F.n[2]; ++k)
          for (int j = 0; j < F.n[1]; ++j)
            for (int i = 0; i < F.n[0]; ++i) {
              const double q = (F.n[0] > 1 ? dcs[0][i] : 1.0) * (F.n[1] > 1 ? dcs[1][j] : 1.0) * (F.n[2] > 1 ? dcs[2][k] : 1.0);
              v[i + size_t(F.n[0]) * (j + size_t(F.n[1]) * k)] = q;
              vtot += q;
            }
        CK(cudaMalloc(&vol, N * sizeof(T)));
        upload(v.data(), vol, N);
        vol_h = v;
      }
      if (!has_c) glast = g;
      faces = nxt;
    }
    build_coarse_inverse(glast);
  }

  // Coarsest level: dense matrix of the stretched Laplacian, inverted on the CPU. Singular problem
  // (no Dirichlet side): with S = V A (symmetric; V the cell volumes) and v the vector of volumes,
  // (S - c v v^T) x = V (b - mean_V(b)) has a unique solution, with zero volume-weighted mean, that
  // solves A x = b - mean_V(b).
  void build_coarse_inverse(const std::vector<double>& g) {
    const Level& C = L.back();
    const size_t N = C.cells();
    if (N > 1024) {
      error = "stretched grid: the coarsest multigrid level has more than 1024 cells (use cell counts that halve "
              "several times, like 2^k or 3 x 2^k)";
      return;
    }
    std::vector<double> A(N * N, 0.0), v(N, 1.0);
    for (int k = 0; k < C.n[2]; ++k)
      for (int j = 0; j < C.n[1]; ++j)
        for (int i = 0; i < C.n[0]; ++i) {
          const int ii[3] = {i, j, k};
          const size_t c = i + size_t(C.n[0]) * (j + size_t(C.n[1]) * k);
          for (int d = 0; d < 3; ++d) {
            const int n = C.n[d];
            if (n == 1) continue;
            const double* idc = g.data() + geo_off(C, d);
            const double* idm = idc + n;
            const size_t st = C.stride(d);
            v[c] *= 1.0 / idc[ii[d]];
            for (int hi = 0; hi < 2; ++hi) {
              const double coef = idc[ii[d]] * idm[ii[d] + hi];
              const int nb = ii[d] + (hi ? 1 : -1);
              if (nb >= 0 && nb < n) {
                A[c * N + (hi ? c + st : c - st)] += coef;
                A[c * N + c] -= coef;
              } else if (C.bc[2 * d + hi] == PERIODIC) {
                A[c * N + (hi ? c - size_t(n - 1) * st : c + size_t(n - 1) * st)] += coef;
                A[c * N + c] -= coef;
              } else if (C.bc[2 * d + hi] == DIRICHLET) {
                A[c * N + c] -= 2.0 * coef;
              }
            }
          }
        }
    std::vector<double> M(N * N), B(N * N, 0.0);
    if (singular) {
      double vt = 0.0, sd = 0.0;
      for (size_t r = 0; r < N; ++r) {
        vt += v[r];
        sd += std::fabs(v[r] * A[r * N + r]);
      }
      const double cc = (sd / N) / (N * (vt / N) * (vt / N));
      for (size_t r = 0; r < N; ++r)
        for (size_t q = 0; q < N; ++q) {
          M[r * N + q] = v[r] * A[r * N + q] - cc * v[r] * v[q];
          B[r * N + q] = v[r] * ((r == q ? 1.0 : 0.0) - v[q] / vt);
        }
    } else {
      M = A;
      for (size_t r = 0; r < N; ++r) B[r * N + r] = 1.0;
    }
    for (size_t col = 0; col < N; ++col) {  // Gauss-Jordan with partial pivoting: B becomes M^-1 B
      size_t piv = col;
      for (size_t r = col + 1; r < N; ++r)
        if (std::fabs(M[r * N + col]) > std::fabs(M[piv * N + col])) piv = r;
      if (piv != col)
        for (size_t q = 0; q < N; ++q) {
          std::swap(M[col * N + q], M[piv * N + q]);
          std::swap(B[col * N + q], B[piv * N + q]);
        }
      const double inv = 1.0 / M[col * N + col];
      for (size_t q = 0; q < N; ++q) {
        M[col * N + q] *= inv;
        B[col * N + q] *= inv;
      }
      for (size_t r = 0; r < N; ++r) {
        if (r == col) continue;
        const double f = M[r * N + col];
        if (f == 0.0) continue;
        for (size_t q = 0; q < N; ++q) {
          M[r * N + q] -= f * M[col * N + q];
          B[r * N + q] -= f * B[col * N + q];
        }
      }
    }
    CK(cudaMalloc(&cinv, N * N * sizeof(T)));
    upload(B.data(), cinv, N * N);
  }
  // How many of the finest levels get the coefficients. Measured (rung 10 set-ups, 128^2 and 256^2):
  // finest level only, 14-15 CG iterations per projection at both sizes (7 without bodies); coefficients
  // averaged onto coarser levels too, CG stalled at 200 iterations for a cylinder in an open channel.
  // Averaged coefficients make coarse cells that are nearly cut off, whose correction overshoots, and the
  // V-cycle stops being a positive definite preconditioner. Coarse levels that see no body are stiffer
  // than the true operator, which only slows the correction down.
  static int& mask_levels() {
    static int m = 1;
    return m;
  }
  void set_mask(const double* const* fine) {
    std::vector<std::vector<double>> cur(3);
    for (int d = 0; d < 3; ++d)
      if (fine[d] && L[0].n[d] > 1) cur[d].assign(fine[d], fine[d] + L[0].cells());
    for (size_t l = 0; l < L.size(); ++l) {
      const Level& F = L[l];
      for (int d = 0; d < 3; ++d) {
        if (cur[d].empty() || int(l) >= (stretched ? 1 : mask_levels())) continue;
        T* a;
        CK(cudaMalloc(&a, F.cells() * sizeof(T)));
        upload(cur[d].data(), a, F.cells());
        mb.push_back(a);
        K[l].b[d] = a;
      }
      if (l == 0) {  // active cells: the stencil's diagonal is not zero (same rule as stencil())
        std::vector<double> ac(F.cells(), 0.0), cu(F.cells(), 0.0);
        n_act = 0.0;
        n_cut = 0.0;
        for (int k = 0; k < F.n[2]; ++k)
          for (int j = 0; j < F.n[1]; ++j)
            for (int i = 0; i < F.n[0]; ++i) {
              const int ii[3] = {i, j, k};
              const size_t c = i + size_t(F.n[0]) * (j + size_t(F.n[1]) * k);
              bool on = false, blocked = false;
              for (int d = 0; d < 3; ++d) {
                if (F.n[d] == 1) continue;
                const size_t s = F.stride(d);
                const int n = F.n[d];
                const bool per = F.bc[2 * d] == PERIODIC;
                const double one = 1.0;
                const double wm = cur[d].empty() ? one : ((ii[d] > 0 || per) ? cur[d][c] : (F.bc[2 * d] == DIRICHLET ? one : 0.0));
                const double wp = cur[d].empty() ? one : (ii[d] < n - 1 ? cur[d][c + s] : (per ? cur[d][c - size_t(n - 1) * s] : (F.bc[2 * d + 1] == DIRICHLET ? one : 0.0)));
                on = on || wm > 0.0 || wp > 0.0;
                if (!cur[d].empty()) {  // a face inside the domain (or across a periodic side) in a body
                  if ((ii[d] > 0 || per) && cur[d][c] == 0.0) blocked = true;
                  if ((ii[d] < n - 1 || per) && wp == 0.0) blocked = true;
                }
              }
              ac[c] = on ? 1.0 : 0.0;
              cu[c] = (on && blocked) ? 1.0 : 0.0;
              n_act += ac[c];
              n_cut += cu[c];
            }
        CK(cudaMalloc(&act, F.cells() * sizeof(T)));
        upload(ac.data(), act, F.cells());
        if (n_cut == 0.0) {  // no cut cell (a body that touches no fluid face): spread over the region
          cu = ac;
          n_cut = n_act;
        }
        CK(cudaMalloc(&cut, F.cells() * sizeof(T)));
        upload(cu.data(), cut, F.cells());
        if (stretched) {
          std::vector<double> va(F.cells());
          vact_tot = vcut_tot = 0.0;
          for (size_t q = 0; q < F.cells(); ++q) {
            va[q] = vol_h[q] * ac[q];
            vact_tot += va[q];
            vcut_tot += vol_h[q] * cu[q];
          }
          cudaFree(vact);
          CK(cudaMalloc(&vact, F.cells() * sizeof(T)));
          upload(va.data(), vact, F.cells());
          // The coarsest level is solved by the inverse of the Laplacian without bodies; with a single
          // level that would be the whole solver.
          if (L.size() < 2) error = "bodies on a stretched grid need at least two multigrid levels (even cell counts)";
        }
      }
      if (l + 1 == L.size()) break;
      const Level& C = L[l + 1];
      std::vector<std::vector<double>> nxt(3);
      for (int d = 0; d < 3; ++d) {
        if (cur[d].empty()) continue;
        nxt[d].assign(C.cells(), 0.0);
        const int cf[3] = {F.coarsen[0], F.coarsen[1], F.coarsen[2]};
        for (int k = 0; k < C.n[2]; ++k)
          for (int j = 0; j < C.n[1]; ++j)
            for (int i = 0; i < C.n[0]; ++i) {
              const int I[3] = {i, j, k};
              int lo[3], cnt[3];
              for (int e = 0; e < 3; ++e) {
                lo[e] = cf[e] * I[e];
                cnt[e] = e == d ? 1 : cf[e];  // the coarse low d-face covers the first child's low face
              }
              double s = 0.0;
              for (int dz = 0; dz < cnt[2]; ++dz)
                for (int dy = 0; dy < cnt[1]; ++dy)
                  for (int dx = 0; dx < cnt[0]; ++dx)
                    s += cur[d][(lo[0] + dx) + size_t(F.n[0]) * ((lo[1] + dy) + size_t(F.n[1]) * (lo[2] + dz))];
              nxt[d][i + size_t(C.n[0]) * (j + size_t(C.n[1]) * k)] = s / double(cnt[0] * cnt[1] * cnt[2]);
            }
      }
      cur = nxt;
    }
    for (auto& x : graphs.e) CK(cudaGraphExecDestroy(x.exec));
    graphs.e.clear();
  }

  Solver(const int* n, const double* len, const int* bc, int nu1_, int nu2_, const double* const* xf = nullptr)
      : nu1(nu1_), nu2(nu2_) {
    Level f;
    for (int d = 0; d < 3; ++d) {
      f.n[d] = n[d];
      f.h[d] = len[d] / n[d];
    }
    for (int s = 0; s < 6; ++s) {
      f.bc[s] = bc[s];
      if (bc[s] == DIRICHLET && n[s / 2] > 1) singular = false;
    }
    while (true) {
      bool any = false;
      for (int d = 0; d < 3; ++d) {
        // periodic directions stay even on every level (red-black ordering across the boundary)
        const int need = f.bc[2 * d] == PERIODIC ? 4 : 2;
        f.coarsen[d] = (f.n[d] > 1 && f.n[d] % need == 0 && f.n[d] >= 4) ? 2 : 1;
        any = any || f.coarsen[d] == 2;
      }
      L.push_back(f);
      if (!any) break;
      Level cl = f;
      for (int d = 0; d < 3; ++d) {
        cl.n[d] = f.n[d] / f.coarsen[d];
        cl.h[d] = f.h[d] * f.coarsen[d];
      }
      f = cl;
    }
    for (auto& l : L)
      for (int d = 0; d < 3; ++d) {
        l.ih[d] = 1.0 / l.h[d];
        l.ih2[d] = 1.0 / (l.h[d] * l.h[d]);
      }
    int maxn = 1;
    for (int d = 0; d < 3; ++d) maxn = L.back().n[d] > maxn ? L.back().n[d] : maxn;
    coarse_sweeps = 2 * maxn * maxn + 20;
    if (coarse_sweeps > 5000) coarse_sweeps = 5000;
    stretched = xf && (xf[0] || xf[1] || xf[2]);
    if (stretched) {
      build_geo(len, xf);
      for (int d = 0; d < 3; ++d) {  // lines along every bounded direction (a periodic one would be cyclic)
        if (L[0].n[d] < 2 || L[0].bc[2 * d] == PERIODIC || !(line_mask() >> d & 1)) continue;
        if (4 * size_t(L[0].n[d]) * sizeof(T) > size_t(LINE_SHARED)) {
          error = "stretched grid: too many cells along a bounded direction for the line smoother (at most 3072; "
                  "1536 with pressure=\"double\")";
          continue;
        }
        line_dirs |= 1 << d;
      }
    }
    for (auto& l : L) {
      Coef<T> k;
      for (int d = 0; d < 3; ++d) k.c[d] = l.n[d] > 1 ? T(1.0 / (l.h[d] * l.h[d])) : T(0);
      if (stretched) {
        for (int d = 0; d < 3; ++d) k.c[d] = l.n[d] > 1 ? T(1) : T(0);
        k.s = s_d[K.size()];
      }
      K.push_back(k);
      T *a, *bb, *cc;
      CK(cudaMalloc(&a, l.cells() * sizeof(T)));
      CK(cudaMalloc(&bb, l.cells() * sizeof(T)));
      CK(cudaMalloc(&cc, l.cells() * sizeof(T)));
      x.push_back(a);
      b.push_back(bb);
      r.push_back(cc);
    }
    const size_t N = L[0].cells();
    CK(cudaMalloc(&p_, N * sizeof(T)));
    CK(cudaMalloc(&q_, N * sizeof(T)));
    CK(cudaMalloc(&z_, N * sizeof(T)));
    CK(cudaMalloc(&rr_, N * sizeof(T)));
    cg = new_cg();
    CK(cudaMalloc(&part, DOT_PARTS * sizeof(double)));
    ensure_hist(200);
  }
  ~Solver() override {
    for (size_t l = 0; l < L.size(); ++l) {
      cudaFree(x[l]);
      cudaFree(b[l]);
      cudaFree(r[l]);
    }
    for (T* a : {p_, q_, z_, rr_}) cudaFree(a);
    for (T* a : mb) cudaFree(a);
    for (double* a : geo_d) cudaFree(a);
    for (T* a : s_d) cudaFree(a);
    cudaFree(vol);
    cudaFree(vact);
    cudaFree(cinv);
    cudaFree(act);
    cudaFree(cut);
    cudaFree(cg);
    cudaFree(part);
    cudaFree(hist);
    delete pre;
  }
  void ensure_hist(int maxit) {  // not while recording a graph (it allocates, and graphs hold the pointer)
    if (maxit + 1 <= hist_n) return;
    if (hist) CK(cudaFree(hist));
    hist_n = maxit + 1;
    CK(cudaMalloc(&hist, hist_n * sizeof(double)));
    for (auto& x : graphs.e) CK(cudaGraphExecDestroy(x.exec));
    graphs.e.clear();
  }

  void vcycle_d(const double* R, double* Z) override {
    const size_t N = L[0].cells();
    k_convert<double, T><<<lin_blocks(N), 256, 0, gs()>>>(R, rr_, N);
    CK(cudaMemsetAsync(z_, 0, N * sizeof(T), gs()));
    vcycle(0, z_, rr_);
    k_convert<T, double><<<lin_blocks(N), 256, 0, gs()>>>(z_, Z, N);
  }

  // Z = M R: one V-cycle from zero, in this precision or (mixed) in the single-precision partner.
  void precondition(T* R, T* Z) {
    if constexpr (std::is_same<T, double>::value) {
      if (pre) {
        pre->vcycle_d(R, Z);
        return;
      }
    }
    CK(cudaMemsetAsync(Z, 0, L[0].cells() * sizeof(T), gs()));
    vcycle(0, Z, R);
  }

  int levels(int* dims) const override {
    for (size_t l = 0; l < L.size(); ++l)
      for (int d = 0; d < 3; ++d) dims[3 * l + d] = L[l].n[d];
    return int(L.size());
  }

  void upload(const double* h, T* d, size_t n) {
    std::vector<T> t(h, h + n);
    CK(cudaMemcpy(d, t.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  void download(const T* d, double* h, size_t n) {
    std::vector<T> t(n);
    CK(cudaMemcpy(t.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    for (size_t q = 0; q < n; ++q) h[q] = double(t[q]);
  }
  void remove_mean(T* a, size_t n) {
    if (vact) {  // stretched, bodies: the volume-weighted mean over the cells with a fluid face
      dot_dev<T>(a, vact, n, part, &cg->s);
      k_sub_mean_m<T><<<lin_blocks(n), 256, 0, gs()>>>(a, act, &cg->s, vact_tot, n);
      return;
    }
    if (vol) {  // stretched: the volume-weighted mean
      dot_dev<T>(a, vol, n, part, &cg->s);
      k_sub_wmean<T><<<lin_blocks(n), 256, 0, gs()>>>(a, &cg->s, vtot, n);
      return;
    }
    if (act) {
      dot_dev<T>(a, act, n, part, &cg->s);
      k_sub_mean_m<T><<<lin_blocks(n), 256, 0, gs()>>>(a, act, &cg->s, n_act, n);
      return;
    }
    dot_dev<T>(a, nullptr, n, part, &cg->s);
    k_sub_mean<T><<<lin_blocks(n), 256, 0, gs()>>>(a, &cg->s, n);
  }

  void smooth(int l, T* X, const T* B, int sweeps, bool forward) {
    const int nact = (L[l].n[0] > 1) + (L[l].n[1] > 1) + (L[l].n[2] > 1);
    if (stretched && line_dirs && nact > 1) {
      // Alternating zebra line relaxation: along each chosen direction in turn, two colours each. The
      // backward sweep runs the same steps in reverse order.
      for (int s = 0; s < sweeps; ++s)
        for (int q = 0; q < 3; ++q) {
          const int d = forward ? q : 2 - q;
          const int n = L[l].n[d];
          if (!(line_dirs >> d & 1) || n < 2) continue;
          const int a = d == 0 ? 1 : 0, bq = d == 2 ? 1 : 2;
          const unsigned lines = unsigned(L[l].n[a]) * unsigned(L[l].n[bq]);
          // At most 128 threads per line: only one thread works during the line solve, and a block of
          // 1024 mostly idle threads fills a whole multiprocessor (measured on 1024^2: 23 ms per V-cycle).
          const int th = n >= 128 ? 128 : ((n + 31) / 32) * 32;
          for (int cq = 0; cq < 2; ++cq)
            k_line<T><<<lines, th, 4 * size_t(n) * sizeof(T), gs()>>>(X, B, L[l], K[l], d, forward ? cq : 1 - cq);
        }
      return;
    }
    for (int s = 0; s < sweeps; ++s) {
      k_rbgs<T><<<grid3(L[l]), B3, 0, gs()>>>(X, B, L[l], K[l], forward ? 0 : 1);
      k_rbgs<T><<<grid3(L[l]), B3, 0, gs()>>>(X, B, L[l], K[l], forward ? 1 : 0);
    }
  }

  // One V-cycle on level l for Laplacian(X) = B, improving X in place.
  // shared memory for levels l..last (x and b each)
  size_t tail_bytes(int l) const {
    size_t s = 0;
    for (size_t m = l; m < L.size(); ++m) s += 2 * L[m].cells() * sizeof(T);
    return s;
  }
  // first level of the one-block tail: at most g_tail_cells cells and fits in TAIL_SHARED
  int first_tail() const {
    const int last = int(L.size()) - 1;
    int l = 0;
    while (l < last && (L[l].cells() > g_tail_cells || tail_bytes(l) > TAIL_SHARED)) ++l;
    return (L[l].cells() <= g_tail_cells && tail_bytes(l) <= TAIL_SHARED) ? l : last + 1;
  }
  void vcycle(int l, T* X, const T* B) {
    const int last = int(L.size()) - 1;
    // Stretched grids: no one-block tail. It was built (each thread solving whole lines in shared memory,
    // bit-identical to the separate kernels) and measured slower: 2.1 against 1.5 ms per solve on 32^2,
    // 7.0 against 6.4 ms on 256^2. A single GPU thread working through a line cell by cell is slow; the
    // separate kernels build each line's system in parallel.
    if (l == 0) tail_from = stretched ? last + 1 : first_tail();
    if (l == last) {
      if (stretched)
        k_coarse_inv<T><<<1, 256, 0, gs()>>>(X, B, cinv, int(L[l].cells()));
      else
        k_coarse<T><<<1, COARSE_THREADS, 0, gs()>>>(X, B, L[l], K[l], coarse_sweeps, singular ? 1 : 0);
      return;
    }
    if (l >= tail_from) {  // this level and all coarser ones: one thread block, in shared memory
      Tail<T> t;
      t.n = last - l + 1;
      if (t.n > MAX_LEVELS) {
        fprintf(stderr, "gcfd: %d levels in the V-cycle tail (at most %d)\n", t.n, MAX_LEVELS);
        abort();
      }
      for (int m = 0; m < t.n; ++m) {
        t.L[m] = L[l + m];
        t.K[m] = K[l + m];  // with bodies the face coefficients are read from global memory
      }
      t.x = X;
      t.b = B;
      t.nu1 = nu1;
      t.nu2 = nu2;
      t.sweeps = coarse_sweeps;
      t.singular = singular ? 1 : 0;
      k_vtail<T><<<1, COARSE_THREADS, tail_bytes(l), gs()>>>(t);
      return;
    }
    smooth(l, X, B, nu1, true);
    if (stretched) {
      k_residual<T><<<grid3(L[l]), B3, 0, gs()>>>(X, B, r[l], L[l], K[l]);
      k_restrict_s<T><<<grid3(L[l + 1]), B3, 0, gs()>>>(r[l], b[l + 1], L[l], L[l + 1]);
    } else {
      k_residual_restrict<T><<<grid3(L[l + 1]), B3, 0, gs()>>>(X, B, b[l + 1], L[l], K[l], L[l + 1]);
    }
    CK(cudaMemsetAsync(x[l + 1], 0, L[l + 1].cells() * sizeof(T), gs()));  // the correction starts from zero
    vcycle(l + 1, x[l + 1], b[l + 1]);
    k_prolong_add<T><<<grid3(L[l]), B3, 0, gs()>>>(X, x[l + 1], L[l], L[l + 1]);
    smooth(l, X, B, nu2, false);
  }

  void apply(const double* ph, double* out) override {
    const size_t N = L[0].cells();
    upload(ph, x[0], N);
    k_apply<T><<<grid3(L[0]), B3, 0, gs()>>>(x[0], r[0], L[0], K[0]);
    download(r[0], out, N);
  }

  // method 0: multigrid V-cycles; method 1: conjugate gradients preconditioned by one V-cycle.
  // hist[it] = ||f - A x|| / ||f|| after it iterations. Returns the iteration count.
  int solve(const double* fh, double* xh, double tol, int maxit, int method, double* hh, double* ms) override {
    const size_t N = L[0].cells();
    ensure_hist(maxit);
    upload(fh, rr_, N);
    upload(xh, x[0], N);
    auto body = [&] { iterate(rr_, x[0], tol, maxit, method); };
    const std::vector<unsigned long long> key = {kb(tol), (unsigned long long)maxit, (unsigned long long)method};
    if (g_use_graphs) graphs.get(key, body);  // record before the clock starts
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, gs()));
    graphs.run(key, body);
    CK(cudaEventRecord(e1, gs()));
    CK(cudaEventSynchronize(e1));
    float t;
    CK(cudaEventElapsedTime(&t, e0, e1));
    *ms = t;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    CK(cudaGetLastError());
    const int it = read_cg(cg).it;
    CK(cudaMemcpy(hh, hist, (it + 1) * sizeof(double), cudaMemcpyDeviceToHost));
    download(x[0], xh, N);
    return it;
  }

  // Solve with right-hand side and solution on the GPU (X: initial guess in, solution out; F is
  // overwritten when the problem is singular: its mean is removed). Queued, no CPU sync; can be recorded.
  void solve_dev(T* F, T* X, double tol, int maxit, const double* fmin = nullptr) { iterate(F, X, tol, maxit, 1, fmin); }

  // The iteration itself, all decisions on the GPU.
  void iterate(T* F, T* X, double tol, int maxit, int method, const double* fmin = nullptr) {
    const size_t N = L[0].cells();
    if (maxit + 1 > hist_n) {
      fprintf(stderr, "gcfd: maxit %d above the prepared %d\n", maxit, hist_n - 1);
      abort();
    }
    T* R = b[0];  // residual (the level-0 rhs array is free: the V-cycle takes its own arguments)
    if (singular && vact) {  // the same on a stretched grid: the volume-weighted sum
      dot_dev<T>(F, vact, N, part, &cg->s);
      k_sub_mean_m<T><<<lin_blocks(N), 256, 0, gs()>>>(F, cut, &cg->s, vcut_tot, N);
    } else if (singular && act) {  // bodies: the sum over the fluid region is taken out in its cut cells
      dot_dev<T>(F, act, N, part, &cg->s);
      k_sub_mean_m<T><<<lin_blocks(N), 256, 0, gs()>>>(F, cut, &cg->s, n_cut, N);
    } else if (singular) {
      remove_mean(F, N);
    }
    dotw_dev<T>(F, F, vol, N, part, &cg->s);
    k_pcg_start<<<1, 1, 0, gs()>>>(cg, tol, maxit, fmin);
    k_residual<T><<<grid3(L[0]), B3, 0, gs()>>>(X, F, R, L[0], K[0]);
    dotw_dev<T>(R, R, vol, N, part, &cg->rr);
    k_pcg_test<<<1, 1, 0, gs()>>>(cg, hist, 0);
    if (method == 0) {
      // multigrid as defect correction: x += M (f - A x). With pre set, M runs in single precision
      // while x and the residual stay in double (iterative refinement).
      dev_while(&cg->go, [&] {
        precondition(R, z_);
        k_axpy<T><<<lin_blocks(N), 256, 0, gs()>>>(X, z_, 1.0, N);
        k_residual<T><<<grid3(L[0]), B3, 0, gs()>>>(X, F, R, L[0], K[0]);
        dotw_dev<T>(R, R, vol, N, part, &cg->rr);
        k_pcg_test<<<1, 1, 0, gs()>>>(cg, hist, 1);
      });
    } else {
      // Preconditioning at the top of the iteration, so the converged last iteration does not pay for
      // a V-cycle it would not use (the first version did; same iterates otherwise).
      dev_while(&cg->go, [&] {
        precondition(R, z_);
        dotw_dev<T>(R, z_, vol, N, part, &cg->rznew);
        k_pcg_beta<<<1, 1, 0, gs()>>>(cg);
        k_xpby_d<T><<<lin_blocks(N), 256, 0, gs()>>>(p_, z_, &cg->beta, &cg->first, N);  // p = z first time
        k_apply<T><<<grid3(L[0]), B3, 0, gs()>>>(p_, q_, L[0], K[0]);
        dotw_dev<T>(p_, q_, vol, N, part, &cg->pq);
        k_pcg_alpha<<<1, 1, 0, gs()>>>(cg);
        k_axpy_d<T><<<lin_blocks(N), 256, 0, gs()>>>(X, p_, &cg->alpha, 1.0, N);
        k_axpy_d<T><<<lin_blocks(N), 256, 0, gs()>>>(R, q_, &cg->alpha, -1.0, N);
        dotw_dev<T>(R, R, vol, N, part, &cg->rr);
        k_pcg_test<<<1, 1, 0, gs()>>>(cg, hist, 1);
      });
      // the recursively updated residual drifts from the true one in single precision: report the true one
      k_residual<T><<<grid3(L[0]), B3, 0, gs()>>>(X, F, R, L[0], K[0]);
      dotw_dev<T>(R, R, vol, N, part, &cg->rr);
      k_pcg_final<<<1, 1, 0, gs()>>>(cg, hist);
    }
    k_cg_done<<<1, 1, 0, gs()>>>(cg);
    if (singular) remove_mean(X, N);
  }

  double bench_vcycle(int reps) override {
    if (pre) return pre->bench_vcycle(reps);  // mixed: the V-cycle runs in the single-precision partner
    const size_t N = L[0].cells();
    CK(cudaMemset(x[0], 0, N * sizeof(T)));
    CK(cudaMemset(rr_, 0, N * sizeof(T)));  // zero right-hand side: same work per cycle, no stale values
    vcycle(0, x[0], rr_);                   // warm-up
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, gs()));
    for (int q = 0; q < reps; ++q) vcycle(0, x[0], rr_);
    CK(cudaEventRecord(e1, gs()));
    CK(cudaEventSynchronize(e1));
    float t;
    CK(cudaEventElapsedTime(&t, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    CK(cudaGetLastError());
    return t / reps;
  }
};

// ================================================================================================
// Incompressible Navier-Stokes on a staggered (MAC) grid.
// Velocity component c lives on the c-faces: q_c[i,j,k] is at the low face of cell (i,j,k) in
// direction c. Convection in divergence form with central averages, diffusion with the 2nd-order
// Laplacian, time stepping with the 3-stage strong-stability-preserving Runge-Kutta scheme, and a
// pressure projection after every stage (Laplacian(phi) = div(u*), u = u* - grad(phi)).
//
// Each direction is periodic or bounded at both ends. A bounded side is one of two kinds:
// - velocity side (a wall, moving or not, or an inflow; L.bc = NEUMANN there for the pressure): the
//   velocity on the side is prescribed. For the normal component, face 0 IS the low side (its stored
//   value is kept at the prescribed one) and face n (not stored) is the high side, whose value comes
//   from the boundary data. A tangential component sees the ghost value 2 U_b - q beyond the side, so
//   that its average on the side equals the prescribed U_b (no slip on a wall).
// - outflow side (L.bc = DIRICHLET: pressure zero on the side, as OpenFOAM's usual outlet): zero normal
//   gradient of the velocity. The normal component on the boundary face is set from the next interior
//   face before each projection and then corrected by the projection like any other face (so the flow
//   out balances the flow in); a tangential component sees the ghost value q (zero gradient).
// The prescribed values can vary along the side (an inflow profile): they are stored per side and per
// component on the side's plane (Bnd), sampled where that component lives.
// ================================================================================================
// Boundary data of each side (x-, x+, y-, y+, z-, z+) and velocity component, sampled on the side's
// plane: index I[a] + n[a] I[b] over the two other directions a < b. Velocity side: the prescribed
// velocity (normal component zero on a wall). Outflow side: only the normal component is used (the
// current value on the high boundary face, which is not stored in the velocity array). Periodic: null.
struct Bnd {
  double* u[6][3];
};
struct Force {
  double g[3];  // constant body force per unit mass (drives channel flows)
};

__device__ __forceinline__ bool wall_dir(const Level& L, int d) { return L.bc[2 * d] != PERIODIC; }
__device__ __forceinline__ bool outflow(const Level& L, int side) { return L.bc[side] == DIRICHLET; }
__host__ __device__ __forceinline__ int plane_idx(const Level& L, int d, int i, int j, int k) {
  return d == 0 ? j + L.n[1] * k : (d == 1 ? i + L.n[0] * k : i + L.n[0] * j);
}

// Normal component a (of direction d) at face (i,j,k), periodic directions wrapped. Face n[d] of a
// bounded direction is the high side, taken from the boundary data (0 on a wall, as before).
template <typename T>
__device__ __forceinline__ T face_n(const T* a, const Level& L, const Bnd& W, int d, int i, int j, int k) {
  int I[3] = {i, j, k};
#pragma unroll
  for (int e = 0; e < 3; ++e) {
    if (I[e] >= 0 && I[e] < L.n[e]) continue;
    if (L.bc[2 * e] == PERIODIC) {
      I[e] = (I[e] + L.n[e]) % L.n[e];
    } else if (e == d && I[e] == L.n[e]) {
      continue;
    } else {
      return T(0);  // not reached: face -1 would only be read by a boundary face, which is not evolved
    }
  }
  if (I[d] == L.n[d]) return T(W.u[2 * d + 1][d][plane_idx(L, d, I[0], I[1], I[2])]);
  return a[I[0] + size_t(L.n[0]) * (I[1] + size_t(L.n[1]) * I[2])];
}

// Ghost value of tangential component comp (value qc at face (i,j,k)) beyond side (d, hi).
// homog = 1 drops the prescribed velocity: the linear part of the operator.
template <typename T>
__device__ __forceinline__ T ghost(T qc, const Level& L, const Bnd& W, int d, int hi, int comp, int i, int j, int k,
                                   int homog) {
  const int s = 2 * d + hi;
  if (outflow(L, s)) return qc;  // zero gradient
  return (homog ? T(0) : T(2.0 * W.u[s][comp][plane_idx(L, d, i, j, k)])) - qc;
}

// Linear index of (i,j,k), wrapped in periodic directions; -1 if it lies beyond a wall.
__device__ __forceinline__ long long at(const Level& L, int i, int j, int k) {
  int I[3] = {i, j, k};
#pragma unroll
  for (int d = 0; d < 3; ++d) {
    if (I[d] < 0 || I[d] >= L.n[d]) {
      if (L.bc[2 * d] != PERIODIC) return -1;
      I[d] = (I[d] + L.n[d]) % L.n[d];
    }
  }
  return I[0] + (long long)L.n[0] * (I[1] + (long long)L.n[1] * I[2]);
}

// A face-normal component at index m: beyond a wall it is the wall itself, where it is zero.
template <typename T>
__device__ __forceinline__ T normal_at(const T* a, long long m) {
  return m < 0 ? T(0) : a[m];
}

// Stretched grids: the same balance over each velocity unknown's box, with the box sizes from the lists
// (Level::geo). Along its own direction the box of component comp runs from cell centre to cell centre
// (width dm at its face); across the other directions it is the cell (width dc).
// - Convection, own direction: the box sides are cell centres, exactly midway between two faces.
// - Convection, other directions: the flow through a box side is the sum of the flows through the two
//   half cells it spans (shares fw and 1 - fw), which makes the box itself divergence-free; the carried
//   velocity is the plain average of the two neighbours (symmetry-preserving form, Verstappen & Veldman
//   2003: convection neither creates nor destroys kinetic energy on any grid).
// - Diffusion: difference of the gradients on the two box sides, each over the distance between the two
//   values it uses, divided by the box width.
template <typename T>
__device__ __forceinline__ T rhs_s(const T* u, const T* v, const T* w, const Level& L, int comp, T nu, const Bnd& W,
                                   const Force& G, int i, int j, int k, size_t c) {
  const int I[3] = {i, j, k};
  const T* V[3] = {u, v, w};
  const T* q = V[comp];
  const int E[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const int* ec = E[comp];
  const T fwc = T(GEO_FW(L, comp)[I[comp]]), fwl = T(1) - fwc;
  T conv = T(0), diff = T(0);
  const T qc = q[c];
  for (int d = 0; d < 3; ++d) {
    if (L.n[d] == 1) continue;
    const int* ed = E[d];
    const int id = I[d], nd = L.n[d];
    const double* idc = GEO_IDC(L, d);
    const double* idm = GEO_IDM(L, d);
    const long long mp = at(L, i + ed[0], j + ed[1], k + ed[2]);
    const long long mm = at(L, i - ed[0], j - ed[1], k - ed[2]);
    T qp, qm;
    if (d == comp) {
      qp = face_n(q, L, W, d, i + ed[0], j + ed[1], k + ed[2]);
      qm = normal_at(q, mm);
      const T Up = T(0.5) * (qc + qp), Um = T(0.5) * (qm + qc);
      const T im = T(idm[id]);
      conv += (Up * Up - Um * Um) * im;
      diff += ((qp - qc) * T(idc[id]) - (qc - qm) * T(idc[id > 0 ? id - 1 : nd - 1])) * im;
    } else {
      qp = mp >= 0 ? q[mp] : ghost(qc, L, W, d, 1, comp, i, j, k, 0);
      qm = mm >= 0 ? q[mm] : ghost(qc, L, W, d, 0, comp, i, j, k, 0);
      const T* a = V[d];
      const T aup = fwc * face_n(a, L, W, d, i + ed[0], j + ed[1], k + ed[2]) +
                    fwl * face_n(a, L, W, d, i + ed[0] - ec[0], j + ed[1] - ec[1], k + ed[2] - ec[2]);
      const T alo = fwc * a[c] + fwl * normal_at(a, at(L, i - ec[0], j - ec[1], k - ec[2]));
      const T ic = T(idc[id]);
      conv += (aup * T(0.5) * (qc + qp) - alo * T(0.5) * (qm + qc)) * ic;
      diff += ((qp - qc) * T(idm[id + 1]) - (qc - qm) * T(idm[id])) * ic;
    }
  }
  return -conv + nu * diff + T(G.g[comp]);
}

template <typename T>
__global__ void k_ns_rhs(const T* __restrict__ u, const T* __restrict__ v, const T* __restrict__ w, T* F,
                         Level L, int comp, T nu, Bnd W, Force G) {
  CELL(L)
  const int I[3] = {i, j, k};
  if (wall_dir(L, comp) && I[comp] == 0) {  // a boundary face: prescribed (or set from the interior), not evolved
    F[c] = T(0);
    return;
  }
  if (L.geo) {
    F[c] = rhs_s(u, v, w, L, comp, nu, W, G, i, j, k, c);
    return;
  }
  const T* V[3] = {u, v, w};
  const T* q = V[comp];
  const int E[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const int* ec = E[comp];
  T conv = T(0), diff = T(0);
  const T qc = q[c];
  for (int d = 0; d < 3; ++d) {
    if (L.n[d] == 1) continue;
    const int* ed = E[d];
    const T ih = T(L.ih[d]), ih2 = T(L.ih2[d]);
    const long long mp = at(L, i + ed[0], j + ed[1], k + ed[2]);
    const long long mm = at(L, i - ed[0], j - ed[1], k - ed[2]);
    T qp, qm;
    if (d == comp) {  // along its own direction: the high boundary face comes from the boundary data
      qp = face_n(q, L, W, d, i + ed[0], j + ed[1], k + ed[2]);
      qm = normal_at(q, mm);
      const T Up = T(0.5) * (qc + qp), Um = T(0.5) * (qm + qc);  // cell-centre values right and left
      conv += (Up * Up - Um * Um) * ih;
    } else {  // across direction d: ghost values beyond a side carry its prescribed velocity
      qp = mp >= 0 ? q[mp] : ghost(qc, L, W, d, 1, comp, i, j, k, 0);
      qm = mm >= 0 ? q[mm] : ghost(qc, L, W, d, 0, comp, i, j, k, 0);
      const T* a = V[d];  // advecting component at the face's edges
      const T aup = T(0.5) * (face_n(a, L, W, d, i + ed[0], j + ed[1], k + ed[2]) +
                              face_n(a, L, W, d, i + ed[0] - ec[0], j + ed[1] - ec[1], k + ed[2] - ec[2]));
      const T alo = T(0.5) * (a[c] + normal_at(a, at(L, i - ec[0], j - ec[1], k - ec[2])));
      conv += (aup * T(0.5) * (qc + qp) - alo * T(0.5) * (qm + qc)) * ih;
    }
    diff += (qp - T(2) * qc + qm) * ih2;
  }
  F[c] = -conv + nu * diff + T(G.g[comp]);
}

template <typename T>
__global__ void k_ns_div(const T* u, const T* v, const T* w, T* dv, Level L, Bnd W) {
  CELL(L)
  const T* V[3] = {u, v, w};
  const int E[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  T s = T(0);
  const int I[3] = {i, j, k};
  for (int d = 0; d < 3; ++d) {
    if (L.n[d] == 1) continue;
    const T ih = L.geo ? T(GEO_IDC(L, d)[I[d]]) : T(L.ih[d]);  // (flow out - flow in) / cell width
    s += (face_n(V[d], L, W, d, i + E[d][0], j + E[d][1], k + E[d][2]) - V[d][c]) * ih;
  }
  dv[c] = s;
}

// q -= scale * d(phi)/dx_comp at the comp-faces (boundary faces: untouched here; outflow faces are
// corrected by k_outflow_correct)
template <typename T>
__global__ void k_ns_correct(T* q, const T* phi, Level L, int comp, double scale = 1.0, const T* fl = nullptr) {
  CELL(L)
  const int I[3] = {i, j, k};
  if (wall_dir(L, comp) && I[comp] == 0) return;  // boundary face (velocity side: zero pressure gradient there)
  if (fl && fl[c] == T(0)) return;                 // a face in a body: set by the body (k_ib_set)
  const int E[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const T ih = L.geo ? T(GEO_IDM(L, comp)[I[comp]]) : T(L.ih[comp]);  // over the distance between the two centres
  q[c] -= T(scale) * (phi[c] - phi[at(L, i - E[comp][0], j - E[comp][1], k - E[comp][2])]) * ih;
}

// One thread per point of side (d, hi)'s plane: its index p and the cell index c next to the side.
#define PLANE(L, d, hi)                                                                 \
  const int a_ = (d) == 0 ? 1 : 0, b_ = (d) == 2 ? 1 : 2;                               \
  const int p = blockIdx.x * blockDim.x + threadIdx.x;                                  \
  if (p >= L.n[a_] * L.n[b_]) return;                                                   \
  int I_[3];                                                                            \
  I_[a_] = p % L.n[a_];                                                                 \
  I_[b_] = p / L.n[a_];                                                                 \
  I_[d] = (hi) ? L.n[d] - 1 : 0;                                                        \
  const size_t c = I_[0] + size_t(L.n[0]) * (I_[1] + size_t(L.n[1]) * I_[2]);
static int plane_blocks(const Level& L, int d) {
  const int a = d == 0 ? 1 : 0, b = d == 2 ? 1 : 2;
  return (L.n[a] * L.n[b] + 255) / 256;
}

// Velocity side, low: the stored boundary face takes the prescribed normal velocity.
template <typename T>
__global__ void k_fix_low(T* q, const double* ub, Level L, int d) {
  PLANE(L, d, 0)
  q[c] = T(ub[p]);
}
// Outflow side: normal velocity on the boundary face = that of the next interior face (zero gradient).
// Low side: face 0 (stored) from face 1. High side: face n (boundary data) from face n-1.
template <typename T>
__global__ void k_outflow(T* q, double* ub, Level L, int d, int hi) {
  PLANE(L, d, hi)
  if (hi) ub[p] = double(q[c]);
  else q[c] = q[c + L.stride(d)];
}
// Projection on an outflow face: the pressure is zero on the side (ghost value -phi beyond it).
template <typename T>
__global__ void k_outflow_correct(T* q, double* ub, const T* phi, Level L, int d, int hi) {
  PLANE(L, d, hi)
  const T ih = L.geo ? T(GEO_IDM(L, d)[hi ? L.n[d] : 0]) : T(L.ih[d]);
  if (hi) ub[p] -= double((-phi[c] - phi[c]) * ih);
  else q[c] -= (phi[c] - (-phi[c])) * ih;
}

// ---------------------------------------------------------------- implicit diffusion (IMEX scheme)
// Laplacian of velocity component comp at its face (i,j,k), with the same wall rules as k_ns_rhs.
// homog = 1 drops the prescribed boundary velocities (ghost -q instead of 2 U_b - q, and 0 for the high
// boundary face of the normal component): the linear part of the operator. The low boundary face is
// part of q itself (kept fixed by the Helmholtz solve, so the operator stays linear on what it changes).
template <typename T>
__device__ __forceinline__ T lap_at(const T* q, const Level& L, int comp, const Bnd& W, int homog, int i, int j,
                                    int k, size_t c) {
  const int E[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const int I[3] = {i, j, k};
  const T qc = q[c];
  T s = T(0);
  if (L.geo) {  // stretched: the diffusion term of rhs_s
    for (int d = 0; d < 3; ++d) {
      if (L.n[d] == 1) continue;
      const int* ed = E[d];
      const int id = I[d], nd = L.n[d];
      const double* idc = GEO_IDC(L, d);
      const double* idm = GEO_IDM(L, d);
      const long long mp = at(L, i + ed[0], j + ed[1], k + ed[2]);
      const long long mm = at(L, i - ed[0], j - ed[1], k - ed[2]);
      T qp, qm;
      if (d == comp) {
        qp = (homog && wall_dir(L, d) && id == nd - 1) ? T(0) : face_n(q, L, W, d, i + ed[0], j + ed[1], k + ed[2]);
        qm = normal_at(q, mm);
        s += ((qp - qc) * T(idc[id]) - (qc - qm) * T(idc[id > 0 ? id - 1 : nd - 1])) * T(idm[id]);
      } else {
        qp = mp >= 0 ? q[mp] : ghost(qc, L, W, d, 1, comp, i, j, k, homog);
        qm = mm >= 0 ? q[mm] : ghost(qc, L, W, d, 0, comp, i, j, k, homog);
        s += ((qp - qc) * T(idm[id + 1]) - (qc - qm) * T(idm[id])) * T(idc[id]);
      }
    }
    return s;
  }
  for (int d = 0; d < 3; ++d) {
    if (L.n[d] == 1) continue;
    const int* ed = E[d];
    const T ih2 = T(L.ih2[d]);
    const long long mp = at(L, i + ed[0], j + ed[1], k + ed[2]);
    const long long mm = at(L, i - ed[0], j - ed[1], k - ed[2]);
    T qp, qm;
    if (d == comp) {
      qp = (homog && wall_dir(L, d) && I[d] == L.n[d] - 1) ? T(0) : face_n(q, L, W, d, i + ed[0], j + ed[1], k + ed[2]);
      qm = normal_at(q, mm);
    } else {
      qp = mp >= 0 ? q[mp] : ghost(qc, L, W, d, 1, comp, i, j, k, homog);
      qm = mm >= 0 ? q[mm] : ghost(qc, L, W, d, 0, comp, i, j, k, homog);
    }
    s += (qp - T(2) * qc + qm) * ih2;
  }
  return s;
}

template <typename T>
__global__ void k_ns_lap(const T* q, T* out, Level L, int comp, Bnd W, int homog) {
  CELL(L)
  const int I[3] = {i, j, k};
  out[c] = (wall_dir(L, comp) && I[comp] == 0) ? T(0) : lap_at(q, L, comp, W, homog, i, j, k, c);
}

// Helmholtz operator A x = x - cnu * Laplacian_homogeneous(x); the identity on the low boundary faces.
template <typename T>
__global__ void k_helm(const T* x, T* Ax, Level L, int comp, Bnd W, double cnu) {
  CELL(L)
  const int I[3] = {i, j, k};
  Ax[c] = (wall_dir(L, comp) && I[comp] == 0) ? x[c] : x[c] - T(cnu) * lap_at(x, L, comp, W, 1, i, j, k, c);
}

// ---------------------------------------------------------------- immersed bodies
// Every face of a velocity component that lies in a body (not fluid) gets its value from a list:
//   q[idx] = a0 + sum_k w_k q[nb_k]   (ns = 4 terms in 2D, 8 in 3D)
// A face near the surface (a ghost face) carries the linear extrapolation along the surface normal
// through the body's velocity at the surface point B and the fluid velocity at the image point
// I = B + s_I n, interpolated (bi- or trilinear) from fluid faces only; faces deeper inside take the
// body's velocity (w = 0). The fluid faces then see the body through their usual stencils (sharp
// interface, ghost-cell method). The lists are built on the CPU (gcfd.py) from signed distances.
// homog = 1 leaves out a0: the rule applied to a correction of the velocity (NS::ghost_op).
template <typename T>
__global__ void k_ib_set(T* q, const long long* idx, const double* a0, const long long* nb, const double* w, int n,
                         int ns, int homog = 0) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) return;
  double s = homog ? 0.0 : a0[g];
  for (int k = 0; k < ns; ++k) s += w[size_t(ns) * g + k] * double(q[nb[size_t(ns) * g + k]]);
  q[idx[g]] = T(s);  // the nb are fluid faces, never written here
}
// Closed fluid region (no outflow): the faces in bodies on its rim carry the extrapolated velocity,
// whose net flow into the region is a small discretization error; incompressibility needs exactly zero.
// So the rim faces are corrected by the least change that removes it: q -= Q * sa, with Q = sum of the
// divergence over the region's cells and sa = sign * area * cell volume / sum(area^2) per rim face
// (sign +1 if the region's cell is on the face's low side, -1 if on its high side).
template <typename T>
__global__ void k_ib_cons(T* q, const long long* idx, const double* sa, int n, const double* Q) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g < n) q[idx[g]] -= T(*Q * sa[g]);
}
// The rule of each face in a body as an equation of the implicit diffusion solve (IMEX with bodies):
// row of face idx: x[idx] - sum_k w_k x[nb_k]  (= a0 on the right-hand side, k_ib_a0).
template <typename T>
__global__ void k_ib_rows(T* Ax, const T* x, const long long* idx, const long long* nb, const double* w, int n, int ns) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) return;
  double s = double(x[idx[g]]);
  for (int k = 0; k < ns; ++k) s -= w[size_t(ns) * g + k] * double(x[nb[size_t(ns) * g + k]]);
  Ax[idx[g]] = T(s);
}
template <typename T>
__global__ void k_ib_a0(T* b, const long long* idx, const double* a0, int n) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g < n) b[idx[g]] = T(a0[g]);
}
// a = 0 where m is not zero (keeps the faces in bodies of a face array: m = the fluid mask)
template <typename T>
__global__ void k_zero_where(T* a, const T* m, size_t n) {
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    if (m[q] != T(0)) a[q] = T(0);
}
__global__ void k_scale(double* a, double f) { *a *= f; }
template <typename T>
__global__ void k_mul(T* a, const T* m, size_t n) {
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    a[q] *= m[q];
}

// Right-hand side of an IMEX stage: b = q + dt (gam N + zeta N_old + alpha nu Lap(q)) + beta dt nu B,
// where B = Laplacian of the zero field with the walls' speeds (the wall terms of the implicit part).
template <typename T>
__global__ void k_imex_b(T* b, const T* q, const T* N, const T* No, const T* Lq, const T* B, double dt, double gam,
                         double zeta, double alpha, double beta, double nu, size_t n) {
  for (size_t s = blockIdx.x * size_t(blockDim.x) + threadIdx.x; s < n; s += size_t(gridDim.x) * blockDim.x)
    b[s] = q[s] + T(dt) * (T(gam) * N[s] + T(zeta) * No[s] + T(alpha * nu) * Lq[s]) + T(beta * dt * nu) * B[s];
}

template <typename T>
__global__ void k_rk(T* q, const T* q0, const T* F, double a, double b, double dt, size_t n) {  // q = a q0 + b (q + dt F)
  for (size_t s = blockIdx.x * size_t(blockDim.x) + threadIdx.x; s < n; s += size_t(gridDim.x) * blockDim.x)
    q[s] = T(a) * q0[s] + T(b) * (q[s] + T(dt) * F[s]);
}

// max that propagates NaN (fmax ignores NaN, which would hide a blown-up solution)
__device__ __forceinline__ double nanmax(double a, double b) { return (a != a || b != b) ? NAN : (a > b ? a : b); }

template <typename T>
__global__ void k_absmax(const T* a, size_t n, double* out) {
  double m = 0.0;
  for (size_t s = blockIdx.x * size_t(blockDim.x) + threadIdx.x; s < n; s += size_t(gridDim.x) * blockDim.x)
    m = nanmax(m, fabs(double(a[s])));
  __shared__ double sh[256];
  sh[threadIdx.x] = m;
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) sh[threadIdx.x] = nanmax(sh[threadIdx.x], sh[threadIdx.x + w]);
    __syncthreads();
  }
  // NaN's bit pattern (0x7ff8...) is larger than any finite non-negative double's, so it wins atomicMax
  if (threadIdx.x == 0) atomicMax(reinterpret_cast<unsigned long long*>(out), __double_as_longlong(sh[0]));
}

__global__ void k_helm_start(CGState* g, double htol) {  // after rr = r.r and s = b.b
  g->stop = htol * htol * fmax(g->s, 1e-300);
  g->it = 0;
  g->first = 0;
  g->go = g->rr > g->stop ? 1 : 0;
}
__global__ void k_helm_alpha(CGState* g) { g->alpha = g->rr / g->pq; }
__global__ void k_helm_beta(CGState* g) {  // after rznew = r.r of the updated residual
  g->beta = g->rznew / g->rr;
  g->rr = g->rznew;
  ++g->it;
  g->go = (g->rr > g->stop && g->it < 500) ? 1 : 0;
}

struct NS {
  using T = double;
  Level L;
  double nu, ptol = 1e-10;
  int ncomp;
  T *q[3] = {nullptr, nullptr, nullptr}, *q0[3] = {nullptr, nullptr, nullptr}, *F[3] = {nullptr, nullptr, nullptr};
  T *phi = nullptr, *dv = nullptr;
  double* dmax = nullptr;
  Solver<double>* P;
  Bnd W = {};  // boundary data (device arrays, one per bounded side and component)
  Bnd W0 = {};  // zeros on the high sides: the boundary data of a velocity CORRECTION (ghost_op), bodies only
  int kind[6];  // 0 periodic, 1 velocity side (wall or inflow), 2 outflow
  bool has_outflow = false;
  Force G;
  // scheme 0: explicit SSP-RK3 for everything. scheme 1 (IMEX): convection and body force by the
  // low-storage RK3 of Spalart, Moser & Rogers (1991), diffusion by Crank-Nicolson within each stage
  // (each velocity component solves (I - beta dt nu Lap) u* = b with conjugate gradients), so the time
  // step is limited by the flow speed only. Second order in time.
  int scheme = 0;
  double htol = 1e-10;
  T *Fo[3] = {nullptr, nullptr, nullptr}, *Bw[3] = {nullptr, nullptr, nullptr};
  T *tmp = nullptr, *cr = nullptr, *cp = nullptr, *cAp = nullptr;
  T* pres = nullptr;  // IMEX: pressure (per unit density), carried between stages (incremental projection)
  CGState* hc = nullptr;  // the Helmholtz solves' scalars
  double* hpart = nullptr;
  // IMEX: F and Fo trade roles every stage (3 per step), so consecutive steps use them the other way
  // round. The parity is passed explicitly, so recording a step changes nothing on the CPU side.
  int parity = 0;
  GraphCache graphs;  // one recorded time step per (dt, parity, tolerances, scheme)
  // Immersed bodies (see k_ib_set). fl: 1 on fluid faces, 0 on faces in a body, per component. With
  // bodies the explicit scheme also carries the pressure between stages (incremental projection, as
  // IMEX does): then the projection increment vanishes at a steady state, so the faces in the bodies,
  // which the projection does not change, stay exactly consistent with the fluid next to them.
  bool has_body = false;
  T* fl[3] = {nullptr, nullptr, nullptr};
  std::vector<double> flh[3];
  struct IBList {
    long long *idx = nullptr, *nb = nullptr;
    double *a0 = nullptr, *w = nullptr;
    int n = 0;
  } ib[3];
  int ib_ns = 4;
  double last_dt = 0.0;
  std::vector<long long> idxh[3];  // host copies of the lists' face indices
  struct Rim {
    long long* idx = nullptr;
    double* sa = nullptr;
    int n = 0;
  } rim[3];  // closed region only (see k_ib_cons)
  bool closed = false;
  double* dsum = nullptr;
  // With bodies the pressure tolerance is relative to max(|div u*|, sqrt(n_fluid) max|u| / h): near a
  // steady state the incremental projection's right-hand side tends to zero, and a tolerance relative
  // to it alone asks for more digits than round-off leaves (measured: 200 iterations, the cap, instead
  // of ~15). The floor still asks for an rms divergence of ptol max|u| / h.
  double* fscale = nullptr;
  double fscale_f = 0.0;
  // Stretched grids: volume of each velocity unknown's box, per component (weights of the dot products
  // of the implicit diffusion solves, whose operator is symmetric in that weighted sum). Null: uniform.
  T* wq[3] = {nullptr, nullptr, nullptr};
  std::vector<double> geo_h;  // host copy of Level::geo
  // Smoothing sweeps before and after the coarse correction on stretched grids. One alternating line
  // sweep each: measured on 256^2 clustered grids, 9-12 CG iterations against 7-10 with two sweeps, at
  // half the cost per iteration (6-8 ms per solve to 1e-10 against 10-13 ms).
  static int& mg_sweeps() {
    static int m = 1;
    return m;
  }

  // kinds: 6 ints (x-, x+, y-, y+, z-, z+): 0 periodic, 1 velocity side, 2 outflow. The boundary data
  // start at zero (walls at rest); set them with set_plane, then refresh.
  NS(const int* n, const double* len, double nu_, int precision, const int* kinds, const double* g,
     const double* const* xf = nullptr)
      : nu(nu_) {
    int bc[6];
    for (int s = 0; s < 6; ++s) {
      kind[s] = kinds[s];
      bc[s] = kinds[s] == 0 ? PERIODIC : (kinds[s] == 2 ? DIRICHLET : NEUMANN);
      has_outflow = has_outflow || kinds[s] == 2;
    }
    for (int d = 0; d < 3; ++d) G.g[d] = g[d];
    const bool str = xf && (xf[0] || xf[1] || xf[2]);
    const int nsm = str ? mg_sweeps() : 2;
    P = new Solver<double>(n, len, bc, nsm, nsm, xf);
    if (precision == 6) P->pre = new Solver<float>(n, len, bc, nsm, nsm, xf);
    L = P->L[0];
    ncomp = n[2] > 1 ? 3 : 2;
    ib_ns = ncomp == 3 ? 8 : 4;
    const size_t N = L.cells();
    for (int s = 0; s < 3; ++s) {
      CK(cudaMalloc(&q[s], N * sizeof(T)));
      CK(cudaMalloc(&q0[s], N * sizeof(T)));
      CK(cudaMalloc(&F[s], N * sizeof(T)));
      CK(cudaMemset(q[s], 0, N * sizeof(T)));
    }
    CK(cudaMalloc(&phi, N * sizeof(T)));
    CK(cudaMalloc(&dv, N * sizeof(T)));
    CK(cudaMemset(phi, 0, N * sizeof(T)));
    CK(cudaMalloc(&dmax, sizeof(double)));
    hc = new_cg();
    CK(cudaMalloc(&hpart, DOT_PARTS * sizeof(double)));
    for (T** a : {&tmp, &cr, &cp, &cAp, &pres}) CK(cudaMalloc(a, N * sizeof(T)));
    CK(cudaMemset(pres, 0, N * sizeof(T)));
    for (int s = 0; s < 3; ++s) {
      CK(cudaMalloc(&Fo[s], N * sizeof(T)));
      CK(cudaMalloc(&Bw[s], N * sizeof(T)));
      CK(cudaMemset(Fo[s], 0, N * sizeof(T)));
    }
    for (int side = 0; side < 6; ++side) {
      if (kind[side] == 0) continue;
      for (int s = 0; s < 3; ++s) {
        CK(cudaMalloc(&W.u[side][s], plane_cells(side / 2) * sizeof(double)));
        CK(cudaMemset(W.u[side][s], 0, plane_cells(side / 2) * sizeof(double)));
      }
    }
    if (L.geo) {  // box volumes of the velocity unknowns: dm along the component's direction, dc across
      std::vector<double> g(geo_size(L));
      CK(cudaMemcpy(g.data(), L.geo, g.size() * sizeof(double), cudaMemcpyDeviceToHost));
      geo_h = g;
      std::vector<double> v(N);
      for (int s = 0; s < ncomp; ++s) {
        for (int k = 0; k < L.n[2]; ++k)
          for (int j = 0; j < L.n[1]; ++j)
            for (int i = 0; i < L.n[0]; ++i) {
              const int I[3] = {i, j, k};
              double q = 1.0;
              for (int d = 0; d < 3; ++d) {
                if (L.n[d] == 1) continue;
                const double* idc = g.data() + geo_off(L, d);
                q *= 1.0 / (d == s ? idc[L.n[d] + I[d]] : idc[I[d]]);
              }
              v[i + size_t(L.n[0]) * (j + size_t(L.n[1]) * k)] = q;
            }
        CK(cudaMalloc(&wq[s], N * sizeof(T)));
        CK(cudaMemcpy(wq[s], v.data(), N * sizeof(T), cudaMemcpyHostToDevice));
      }
    }
    refresh();
  }
  size_t plane_cells(int d) const { return L.cells() / L.n[d]; }
  void set_plane(int side, int comp, const double* h) {
    CK(cudaMemcpy(W.u[side][comp], h, plane_cells(side / 2) * sizeof(double), cudaMemcpyHostToDevice));
  }
  void get_plane(int side, int comp, double* h) {
    CK(cudaMemcpy(h, W.u[side][comp], plane_cells(side / 2) * sizeof(double), cudaMemcpyDeviceToHost));
  }
  // Terms of the implicit Laplacian from the boundary data: Bw = Lap(0) with the prescribed velocities.
  void boundary_terms() {
    CK(cudaMemsetAsync(tmp, 0, L.cells() * sizeof(T), gs()));
    for (int s = 0; s < 3; ++s) k_ns_lap<T><<<grid3(L), B3, 0, gs()>>>(tmp, Bw[s], L, s, W, 0);
  }
  // The stored boundary faces of component comp: prescribed value on a low velocity side, the interior
  // value on an outflow side.
  void boundary_faces(int comp) {
    const int d = comp;
    if (kind[2 * d] == 1) k_fix_low<T><<<plane_blocks(L, d), 256, 0, gs()>>>(q[comp], W.u[2 * d][d], L, d);
    for (int hi = 0; hi < 2; ++hi)
      if (kind[2 * d + hi] == 2) k_outflow<T><<<plane_blocks(L, d), 256, 0, gs()>>>(q[comp], W.u[2 * d + hi][d], L, d, hi);
  }
  void refresh() {  // after the boundary data changed
    boundary_terms();
    for (int s = 0; s < ncomp; ++s) boundary_faces(s);
    CK(cudaDeviceSynchronize());
  }
  ~NS() {
    for (int s = 0; s < 3; ++s) {
      cudaFree(q[s]);
      cudaFree(q0[s]);
      cudaFree(F[s]);
      cudaFree(Fo[s]);
      cudaFree(Bw[s]);
      cudaFree(wq[s]);
    }
    for (int side = 0; side < 6; ++side)
      for (int s = 0; s < 3; ++s) {
        cudaFree(W.u[side][s]);
        cudaFree(W0.u[side][s]);
      }
    cudaFree(dsum);
    cudaFree(fscale);
    for (int s = 0; s < 3; ++s) {
      cudaFree(rim[s].idx);
      cudaFree(rim[s].sa);
      cudaFree(fl[s]);
      cudaFree(ib[s].idx);
      cudaFree(ib[s].nb);
      cudaFree(ib[s].a0);
      cudaFree(ib[s].w);
    }
    for (T* a : {phi, dv, tmp, cr, cp, cAp, pres}) cudaFree(a);
    cudaFree(dmax);
    cudaFree(hc);
    cudaFree(hpart);
    delete P;
  }
  // Bodies, one component at a time: fluid mask (N values) and the list of faces in bodies (see k_ib_set).
  void set_body(int comp, const double* mask, int n, const long long* idx, const double* a0, const long long* nb,
                const double* w) {
    const size_t N = L.cells();
    flh[comp].assign(mask, mask + N);
    idxh[comp].assign(idx, idx + n);
    if (!fl[comp]) CK(cudaMalloc(&fl[comp], N * sizeof(T)));
    CK(cudaMemcpy(fl[comp], mask, N * sizeof(T), cudaMemcpyHostToDevice));
    IBList& b = ib[comp];
    for (void* a : {(void*)b.idx, (void*)b.nb, (void*)b.a0, (void*)b.w}) cudaFree(a);
    b = IBList();
    b.n = n;
    if (n == 0) return;
    CK(cudaMalloc(&b.idx, n * sizeof(long long)));
    CK(cudaMalloc(&b.a0, n * sizeof(double)));
    CK(cudaMalloc(&b.nb, size_t(n) * ib_ns * sizeof(long long)));
    CK(cudaMalloc(&b.w, size_t(n) * ib_ns * sizeof(double)));
    CK(cudaMemcpy(b.idx, idx, n * sizeof(long long), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.a0, a0, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.nb, nb, size_t(n) * ib_ns * sizeof(long long), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.w, w, size_t(n) * ib_ns * sizeof(double), cudaMemcpyHostToDevice));
  }
  void body_done() {  // after set_body for every component: masks into the pressure solver
    const double* f[3] = {nullptr, nullptr, nullptr};
    for (int s = 0; s < ncomp; ++s) f[s] = flh[s].empty() ? nullptr : flh[s].data();
    P->set_mask(f);
    if (P->pre) static_cast<Solver<float>*>(P->pre)->set_mask(f);
    has_body = true;
    if (!fscale) CK(cudaMalloc(&fscale, sizeof(double)));
    {
      double hmin = 1e300;
      for (int d = 0; d < 3; ++d)
        if (L.n[d] > 1) hmin = std::min(hmin, L.h[d]);
      fscale_f = std::sqrt(P->n_act) / hmin;
      if (L.geo) {
        // Stretched: the solver's norms are volume-weighted, so the same floor (a divergence of
        // max|u| / h in every fluid cell, h the cell's smallest width) is sqrt(sum of volume / h^2).
        // With uniform positions that is sqrt(volume) times the uniform floor, as the norms are.
        std::vector<double> ac(L.cells());
        CK(cudaMemcpy(ac.data(), P->act, L.cells() * sizeof(double), cudaMemcpyDeviceToHost));
        double sum = 0.0;
        for (int k = 0; k < L.n[2]; ++k)
          for (int j = 0; j < L.n[1]; ++j)
            for (int i = 0; i < L.n[0]; ++i) {
              const int I[3] = {i, j, k};
              double v = 1.0, hc = 1e300;
              for (int d = 0; d < 3; ++d) {
                if (L.n[d] == 1) continue;
                const double w = 1.0 / geo_h[geo_off(L, d) + I[d]];
                v *= w;
                hc = std::min(hc, w);
              }
              sum += ac[i + size_t(L.n[0]) * (j + size_t(L.n[1]) * k)] * v / (hc * hc);
            }
        fscale_f = std::sqrt(sum);
      }
    }
    closed = P->singular;
    if (closed) {  // the rim faces of the fluid region and their correction weights (k_ib_cons)
      std::vector<double> ac(L.cells());
      CK(cudaMemcpy(ac.data(), P->act, L.cells() * sizeof(double), cudaMemcpyDeviceToHost));
      double vol = 1.0;
      for (int d = 0; d < 3; ++d)
        if (L.n[d] > 1) vol *= L.h[d];
      std::vector<long long> ri[3];
      std::vector<double> rs[3];
      double den = 0.0;
      for (int s = 0; s < ncomp; ++s) {
        double area = vol / L.h[s];
        for (long long f : idxh[s]) {
          const int i = int(f % L.n[0]), j = int((f / L.n[0]) % L.n[1]), k = int(f / (L.n[0] * (long long)L.n[1]));
          const int I[3] = {i, j, k};
          if (L.geo) {  // stretched: the face's own area (the cell widths across its direction)
            area = 1.0;
            for (int d = 0; d < 3; ++d)
              if (d != s && L.n[d] > 1) area *= 1.0 / geo_h[geo_off(L, d) + I[d]];
          }
          if (wall_dir_h(s) && I[s] == 0) continue;  // a boundary face: not part of the region's rim
          int J[3] = {i, j, k};
          J[s] = (I[s] - 1 + L.n[s]) % L.n[s];
          const size_t lo = J[0] + size_t(L.n[0]) * (J[1] + size_t(L.n[1]) * J[2]);
          const double sg = ac[lo] - ac[f];  // +1: region on the low side (this is its cell's high face)
          if (sg == 0.0) continue;
          ri[s].push_back(f);
          rs[s].push_back(sg * area);
          den += area * area;
        }
      }
      if (!dsum) CK(cudaMalloc(&dsum, sizeof(double)));
      for (int s = 0; s < ncomp; ++s) {
        Rim& r = rim[s];
        cudaFree(r.idx);
        cudaFree(r.sa);
        r = Rim();
        r.n = int(ri[s].size());
        if (!r.n) continue;
        for (double& v : rs[s]) v *= (L.geo ? 1.0 : vol) / den;
        CK(cudaMalloc(&r.idx, r.n * sizeof(long long)));
        CK(cudaMalloc(&r.sa, r.n * sizeof(double)));
        CK(cudaMemcpy(r.idx, ri[s].data(), r.n * sizeof(long long), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(r.sa, rs[s].data(), r.n * sizeof(double), cudaMemcpyHostToDevice));
      }
    }
    for (int d = 0; d < 3; ++d)
      if (kind[2 * d + 1] != 0 && !W0.u[2 * d + 1][d]) {
        CK(cudaMalloc(&W0.u[2 * d + 1][d], plane_cells(d) * sizeof(double)));
        CK(cudaMemset(W0.u[2 * d + 1][d], 0, plane_cells(d) * sizeof(double)));
      }
    for (auto& x : graphs.e) CK(cudaGraphExecDestroy(x.exec));
    graphs.e.clear();
    ib_set();
    CK(cudaDeviceSynchronize());
  }
  void ib_set() {
    for (int s = 0; s < ncomp; ++s)
      if (ib[s].n)
        k_ib_set<T><<<(ib[s].n + 255) / 256, 256, 0, gs()>>>(q[s], ib[s].idx, ib[s].a0, ib[s].nb, ib[s].w, ib[s].n, ib_ns);
  }
  void rhs(T* const* Fs) {  // explicit scheme: convection + diffusion + force; IMEX: convection + force only
    const T nu_explicit = scheme == 0 ? T(nu) : T(0);
    for (int s = 0; s < ncomp; ++s)
      k_ns_rhs<T><<<grid3(L), B3, 0, gs()>>>(q[0], q[1], q[2], Fs[s], L, s, nu_explicit, W, G);
  }
  // Solve (I - cnu Lap_homogeneous) x = b for component comp by conjugate gradients (the operator is
  // symmetric positive definite and well conditioned: 1 <= eigenvalues <= 1 + 4 cnu sum(1/h^2)).
  void helmholtz(int comp, T* x, const T* b, double cnu) {
    const size_t n = L.cells();
    k_helm<T><<<grid3(L), B3, 0, gs()>>>(x, cAp, L, comp, W, cnu);
    CK(cudaMemcpyAsync(cr, b, n * sizeof(T), cudaMemcpyDeviceToDevice, gs()));
    k_axpy<T><<<lin_blocks(n), 256, 0, gs()>>>(cr, cAp, -1.0, n);  // r = b - A x
    CK(cudaMemcpyAsync(cp, cr, n * sizeof(T), cudaMemcpyDeviceToDevice, gs()));
    dotw_dev<T>(cr, cr, wq[comp], n, hpart, &hc->rr);
    dotw_dev<T>(b, b, wq[comp], n, hpart, &hc->s);
    k_helm_start<<<1, 1, 0, gs()>>>(hc, htol);
    dev_while(&hc->go, [&] {
      k_helm<T><<<grid3(L), B3, 0, gs()>>>(cp, cAp, L, comp, W, cnu);
      dotw_dev<T>(cp, cAp, wq[comp], n, hpart, &hc->pq);
      k_helm_alpha<<<1, 1, 0, gs()>>>(hc);
      k_axpy_d<T><<<lin_blocks(n), 256, 0, gs()>>>(x, cp, &hc->alpha, 1.0, n);
      k_axpy_d<T><<<lin_blocks(n), 256, 0, gs()>>>(cr, cAp, &hc->alpha, -1.0, n);
      dotw_dev<T>(cr, cr, wq[comp], n, hpart, &hc->rznew);
      k_helm_beta<<<1, 1, 0, gs()>>>(hc);
      k_xpby_d<T><<<lin_blocks(n), 256, 0, gs()>>>(cp, cr, &hc->beta, &hc->first, n);
    });
    k_cg_done<<<1, 1, 0, gs()>>>(hc);
  }
  // ------------------------------------------------- IMEX with immersed bodies (DESIGN.md, stage 3)
  // The projection of the explicit scheme corrects fluid faces only and leaves the faces in bodies at
  // the values extrapolated from u*. Setting them again from the corrected velocity changes the flow
  // through them, so either the ghost rule or the divergence constraint is violated by D_g E G phi, and
  // an incremental pressure feeds that leftover back (unstable with IMEX's substeps once the largest
  // eigenvalue of A^-1 D_g E G passes 0.279). Here the correction of the ghost faces is part of the
  // pressure equation, (A + D_g E G) phi = div u*, so that afterwards the velocity is divergence-free
  // WITH ghost faces that follow their rule. The operator is not symmetric: BiCGStab.
  //
  // out = D_g E G x: the divergence, through the faces in bodies only, of the ghost rule applied to the
  // gradient of x on the fluid faces (in a closed region with the net-flow correction of the rim faces,
  // which is part of the projection). Uses q0 as scratch.
  void ghost_op(const T* x, T* out) {
    const size_t n = L.cells();
    for (int s = 0; s < 3; ++s) {
      CK(cudaMemsetAsync(q0[s], 0, n * sizeof(T), gs()));
      if (s >= ncomp) continue;
      k_ns_correct<T><<<grid3(L), B3, 0, gs()>>>(q0[s], x, L, s, -1.0, fl[s]);  // + grad x on the fluid faces
      if (ib[s].n)
        k_ib_set<T><<<(ib[s].n + 255) / 256, 256, 0, gs()>>>(q0[s], ib[s].idx, ib[s].a0, ib[s].nb, ib[s].w, ib[s].n, ib_ns, 1);
      k_zero_where<T><<<lin_blocks(n), 256, 0, gs()>>>(q0[s], fl[s], n);  // keep the faces in bodies
    }
    k_ns_div<T><<<grid3(L), B3, 0, gs()>>>(q0[0], q0[1], q0[2], out, L, W0);
    if (closed) {
      dot_dev<T>(out, P->vact ? P->vact : P->act, n, hpart, dsum);
      for (int s = 0; s < ncomp; ++s)
        if (rim[s].n) k_ib_cons<T><<<(rim[s].n + 255) / 256, 256, 0, gs()>>>(q0[s], rim[s].idx, rim[s].sa, rim[s].n, dsum);
      k_ns_div<T><<<grid3(L), B3, 0, gs()>>>(q0[0], q0[1], q0[2], out, L, W0);
    }
    k_mul<T><<<lin_blocks(n), 256, 0, gs()>>>(out, P->act, n);
  }
  void op_hat(const T* x, T* out) {  // (A + D_g E G) x
    k_apply<T><<<grid3(L), B3, 0, gs()>>>(x, out, P->L[0], P->K[0]);
    ghost_op(x, tmp);
    k_axpy<T><<<lin_blocks(L.cells()), 256, 0, gs()>>>(out, tmp, 1.0, L.cells());
  }
  // BiCGStab (van der Vorst 1992), right-preconditioned: x += alpha K p + omega K s. A(in, out) applies
  // the operator, K(in, out) the preconditioner (pre false: none, and y, z are never used), dot the
  // inner product. r, r0, p, v, t (and y, z with a preconditioner) are work arrays; r holds b - A x on
  // entry, with g->rr = (r, r) and g->s = (b, b) set and k_bi_start queued.
  template <class Op, class Pre, class Dot>
  void bicgstab(CGState* g, T* x, T* r, T* r0, T* p, T* v, T* t, T* y, T* z, Op A, Pre K, Dot dot, bool pre) {
    const size_t n = L.cells();
    const int nb = lin_blocks(n);
    CK(cudaMemcpyAsync(r0, r, n * sizeof(T), cudaMemcpyDeviceToDevice, gs()));
    T* yy = pre ? y : p;
    T* zz = pre ? z : r;
    dev_while(&g->go, [&] {
      dot(r0, r, &g->rznew);
      k_bi_beta<<<1, 1, 0, gs()>>>(g);
      k_bi_p<T><<<nb, 256, 0, gs()>>>(p, r, v, g, n);
      if (pre) K(p, y);
      A(yy, v);
      dot(r0, v, &g->pq);
      k_bi_alpha<<<1, 1, 0, gs()>>>(g);
      k_axpy_d<T><<<nb, 256, 0, gs()>>>(r, v, &g->alpha, -1.0, n);  // s = r - alpha v, kept in r
      if (pre) K(r, z);
      A(zz, t);
      dot(t, r, &g->ts);
      dot(t, t, &g->tt);
      k_bi_omega<<<1, 1, 0, gs()>>>(g);
      k_axpy_d<T><<<nb, 256, 0, gs()>>>(x, yy, &g->alpha, 1.0, n);
      k_axpy_d<T><<<nb, 256, 0, gs()>>>(x, zz, &g->omega, 1.0, n);
      k_axpy_d<T><<<nb, 256, 0, gs()>>>(r, t, &g->omega, -1.0, n);
      dot(r, r, &g->rr);
      k_bi_test<<<1, 1, 0, gs()>>>(g);
    });
    k_cg_done<<<1, 1, 0, gs()>>>(g);
  }
  // (A + D_g E G) X = F, preconditioned by the multigrid V-cycle of A. X: initial guess in, solution out.
  void solve_hat(T* F, T* X) {
    Solver<double>& S = *P;
    const size_t n = L.cells();
    CGState* g = S.cg;
    T *r = S.b[0], *pp = S.p_, *v = S.q_, *y = S.z_, *r0 = cr, *t = cp, *z = cAp;
    if (S.singular && S.vact) {  // closed region: the sum over the fluid is taken out in its cut cells (as Solver::iterate)
      dot_dev<T>(F, S.vact, n, S.part, &g->s);
      k_sub_mean_m<T><<<lin_blocks(n), 256, 0, gs()>>>(F, S.cut, &g->s, S.vcut_tot, n);
    } else if (S.singular && S.act) {
      dot_dev<T>(F, S.act, n, S.part, &g->s);
      k_sub_mean_m<T><<<lin_blocks(n), 256, 0, gs()>>>(F, S.cut, &g->s, S.n_cut, n);
    }
    auto dot = [&](const T* a, const T* b, double* out) { dotw_dev<T>(a, b, S.vol, n, S.part, out); };
    op_hat(X, v);
    CK(cudaMemcpyAsync(r, F, n * sizeof(T), cudaMemcpyDeviceToDevice, gs()));
    k_axpy<T><<<lin_blocks(n), 256, 0, gs()>>>(r, v, -1.0, n);
    dot(F, F, &g->s);
    dot(r, r, &g->rr);
    k_bi_start<<<1, 1, 0, gs()>>>(g, ptol, 200, fscale);
    bicgstab(g, X, r, r0, pp, v, t, y, z, [&](T* in, T* out) { op_hat(in, out); },
             [&](T* in, T* out) { S.precondition(in, out); }, dot, true);
    if (S.singular) S.remove_mean(X, n);
  }
  // Implicit diffusion with bodies: fluid faces (I - cnu Lap) x = b, and every face in a body with its
  // ghost rule as its equation, x_G - sum w x = a0 (the wall condition holds for the solved velocity,
  // not for the last stage's). The rows of b that belong to faces in bodies are set here.
  void helm_op(int comp, const T* x, T* Ax, double cnu) {
    k_helm<T><<<grid3(L), B3, 0, gs()>>>(x, Ax, L, comp, W, cnu);
    const IBList& bl = ib[comp];
    if (bl.n) k_ib_rows<T><<<(bl.n + 255) / 256, 256, 0, gs()>>>(Ax, x, bl.idx, bl.nb, bl.w, bl.n, ib_ns);
  }
  void helmholtz_body(int comp, T* x, T* b, double cnu) {
    const size_t n = L.cells();
    T *r = cr, *r0 = cp, *pp = cAp, *v = tmp, *t = dv;
    const IBList& bl = ib[comp];
    if (bl.n) k_ib_a0<T><<<(bl.n + 255) / 256, 256, 0, gs()>>>(b, bl.idx, bl.a0, bl.n);
    auto dot = [&](const T* a, const T* c, double* out) { dotw_dev<T>(a, c, wq[comp], n, hpart, out); };
    helm_op(comp, x, v, cnu);
    CK(cudaMemcpyAsync(r, b, n * sizeof(T), cudaMemcpyDeviceToDevice, gs()));
    k_axpy<T><<<lin_blocks(n), 256, 0, gs()>>>(r, v, -1.0, n);
    dot(b, b, &hc->s);
    dot(r, r, &hc->rr);
    k_bi_start<<<1, 1, 0, gs()>>>(hc, htol, 500);
    bicgstab(hc, x, r, r0, pp, v, t, nullptr, nullptr, [&](T* in, T* out) { helm_op(comp, in, out, cnu); },
             [](T*, T*) {}, dot, false);
  }
  void step_imex(double dt, int par) {
    static const double gam[3] = {8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0};
    static const double zeta[3] = {0.0, -17.0 / 60.0, -5.0 / 12.0};
    static const double ab[3] = {4.0 / 15.0, 1.0 / 15.0, 1.0 / 6.0};  // alpha = beta (Crank-Nicolson split)
    // Incremental projection: the predictor includes the last pressure gradient, so the projection
    // only removes the pressure INCREMENT. Without it the implicit diffusion acts on the full gradient
    // part of u*, and near walls Lap(grad p) is not a gradient: an O(dt) error that survives even at
    // steady state (found: the steady cavity differed from the explicit scheme's by up to 6e-4 U).
    const size_t n = L.cells();
    T *Fn[3], *Fold[3];  // this stage's convection term N(u_{k-1}) and the last stage's
    for (int s = 0; s < 3; ++s) {
      Fn[s] = par ? Fo[s] : F[s];
      Fold[s] = par ? F[s] : Fo[s];
    }
    for (int k = 0; k < 3; ++k) {
      const double cdt = 2.0 * ab[k] * dt;  // (alpha + beta) dt: converts the increment phi to pressure
      if (has_outflow) boundary_terms();    // the outflow faces' values changed in the last projection
      rhs(Fn);                              // Fn = N(u_{k-1})
      for (int s = 0; s < ncomp; ++s) {
        k_ns_lap<T><<<grid3(L), B3, 0, gs()>>>(q[s], tmp, L, s, W, 0);
        k_imex_b<T><<<lin_blocks(n), 256, 0, gs()>>>(q0[s], q[s], Fn[s], Fold[s], tmp, Bw[s], dt, gam[k], zeta[k],
                                                     ab[k], ab[k], nu, n);  // q0 holds the stage's right-hand side
        // - (alpha+beta) dt grad p_old; with bodies on fluid faces only (the pressure is not defined in a body)
        k_ns_correct<T><<<grid3(L), B3, 0, gs()>>>(q0[s], pres, L, s, cdt, has_body ? fl[s] : nullptr);
        if (has_body) helmholtz_body(s, q[s], q0[s], ab[k] * dt * nu);     // ghost rules as rows of the solve
        else helmholtz(s, q[s], q0[s], ab[k] * dt * nu);                   // initial guess: the current velocity
        std::swap(Fn[s], Fold[s]);                                           // N(u_{k-1}) becomes N_old next stage
      }
      project();                                                              // u = u* - grad(phi)
      k_axpy<T><<<lin_blocks(n), 256, 0, gs()>>>(pres, phi, 1.0 / cdt, n);  // p += phi / ((alpha+beta) dt)
    }
  }
  bool wall_dir_h(int d) const { return L.bc[2 * d] != PERIODIC; }
  void project() {
    if (has_body) ib_set();  // faces in bodies from the predicted fluid velocity
    if (has_outflow)  // outflow faces from the predicted interior velocity (zero gradient)
      for (int s = 0; s < ncomp; ++s) boundary_faces(s);
    if (closed) {  // zero net flow into the closed fluid region (k_ib_cons)
      k_ns_div<T><<<grid3(L), B3, 0, gs()>>>(q[0], q[1], q[2], dv, L, W);
      dot_dev<T>(dv, P->vact ? P->vact : P->act, L.cells(), hpart, dsum);  // stretched: the net flow itself
      for (int s = 0; s < ncomp; ++s)
        if (rim[s].n) k_ib_cons<T><<<(rim[s].n + 255) / 256, 256, 0, gs()>>>(q[s], rim[s].idx, rim[s].sa, rim[s].n, dsum);
    }
    k_ns_div<T><<<grid3(L), B3, 0, gs()>>>(q[0], q[1], q[2], dv, L, W);
    if (has_body) k_mul<T><<<lin_blocks(L.cells()), 256, 0, gs()>>>(dv, P->act, L.cells());  // cells in bodies: decoupled
    if (has_body) {  // tolerance floor: rms divergence ptol * max|u| / h (see fscale)
      CK(cudaMemsetAsync(fscale, 0, sizeof(double), gs()));
      for (int s = 0; s < ncomp; ++s) k_absmax<T><<<lin_blocks(L.cells()), 256, 0, gs()>>>(q[s], L.cells(), fscale);
      k_scale<<<1, 1, 0, gs()>>>(fscale, fscale_f);
    }
    const bool consistent = has_body && scheme == 1;  // IMEX with bodies: the ghost faces are part of the projection
    if (consistent) solve_hat(dv, phi);
    else P->solve_dev(dv, phi, ptol, 200, has_body ? fscale : nullptr);  // phi from the last projection: initial guess
    for (int s = 0; s < ncomp; ++s) k_ns_correct<T><<<grid3(L), B3, 0, gs()>>>(q[s], phi, L, s, 1.0, fl[s]);
    for (int side = 0; side < 6; ++side)
      if (kind[side] == 2) {
        const int d = side / 2;
        k_outflow_correct<T><<<plane_blocks(L, d), 256, 0, gs()>>>(q[d], W.u[side][d], phi, L, d, side & 1);
      }
    if (consistent) {  // the ghost faces from the corrected velocity: what solve_hat assumed
      ib_set();
      if (closed) {
        k_ns_div<T><<<grid3(L), B3, 0, gs()>>>(q[0], q[1], q[2], dv, L, W);
        dot_dev<T>(dv, P->vact ? P->vact : P->act, L.cells(), hpart, dsum);
        for (int s = 0; s < ncomp; ++s)
          if (rim[s].n) k_ib_cons<T><<<(rim[s].n + 255) / 256, 256, 0, gs()>>>(q[s], rim[s].idx, rim[s].sa, rim[s].n, dsum);
      }
    }
  }
  void stage(double a, double b, double dt) {
    rhs(F);
    for (int s = 0; s < ncomp; ++s)
      k_rk<T><<<lin_blocks(L.cells()), 256, 0, gs()>>>(q[s], q0[s], F[s], a, b, dt, L.cells());
    if (has_body)  // incremental projection (see has_body): the stage includes - b dt grad p
      for (int s = 0; s < ncomp; ++s) k_ns_correct<T><<<grid3(L), B3, 0, gs()>>>(q[s], pres, L, s, b * dt, fl[s]);
    project();
    if (has_body) k_axpy<T><<<lin_blocks(L.cells()), 256, 0, gs()>>>(pres, phi, 1.0 / (b * dt), L.cells());
  }
  void step_body(double dt, int par) {
    if (scheme == 1) {
      step_imex(dt, par);
      return;
    }
    for (int s = 0; s < ncomp; ++s)
      CK(cudaMemcpyAsync(q0[s], q[s], L.cells() * sizeof(T), cudaMemcpyDeviceToDevice, gs()));
    stage(0.0, 1.0, dt);
    stage(0.75, 0.25, dt);
    stage(1.0 / 3.0, 2.0 / 3.0, dt);
  }
  std::vector<unsigned long long> key(double dt, int par) const {
    return {kb(dt), (unsigned long long)par, kb(ptol), kb(htol), (unsigned long long)scheme};
  }
  void prepare(double dt) {  // record the step graphs before the clock starts
    if (!g_use_graphs) return;
    for (int par : {parity, scheme == 1 ? 1 - parity : parity})
      graphs.get(key(dt, par), [&] { step_body(dt, par); });
  }
  void step(double dt) {  // one time step: with graphs on, a single launch and no CPU sync
    const int par = parity;
    graphs.run(key(dt, par), [&] { step_body(dt, par); });
    if (scheme == 1) parity = 1 - parity;
  }
  double max_div() {
    k_ns_div<T><<<grid3(L), B3, 0, gs()>>>(q[0], q[1], q[2], dv, L, W);
    if (has_body) k_mul<T><<<lin_blocks(L.cells()), 256, 0, gs()>>>(dv, P->act, L.cells());  // cells with fluid faces
    CK(cudaMemsetAsync(dmax, 0, sizeof(double), gs()));
    k_absmax<T><<<lin_blocks(L.cells()), 256, 0, gs()>>>(dv, L.cells(), dmax);
    double m;
    CK(cudaMemcpy(&m, dmax, sizeof(double), cudaMemcpyDeviceToHost));
    return m;
  }
};

// Face positions of a stretched direction: n + 1 increasing values from 0 to len. Returns the reason
// they are refused, or null.
static const char* bad_faces(const double* x, int n, double len, bool periodic) {
  if (!x) return nullptr;
  if (periodic) return "a periodic direction must be uniform (no face positions)";
  if (std::fabs(x[0]) > 1e-12 * len || std::fabs(x[n] - len) > 1e-12 * len) return "face positions must run from 0 to the length";
  for (int i = 0; i < n; ++i)
    if (!(x[i + 1] > x[i])) return "face positions must increase";
  return nullptr;
}
template <class S>
static bool refused(S* s) {
  if (!s || !s->error) return false;
  fprintf(stderr, "gcfd: %s\n", s->error);
  return true;
}

extern "C" {

// kinds: 6 ints (x-, x+, y-, y+, z-, z+): 0 periodic (both sides of a direction), 1 velocity side (wall
// or inflow; its velocity set with gcfd_ns_set_bnd, zero until then), 2 outflow (pressure zero, zero
// normal velocity gradient). g: body force (3). Returns null for an invalid set-up (reason on stderr).
void* gcfd_ns_create3(const int* n, const double* len, double nu, int precision, const int* kinds, const double* g,
                      const double* xf, const double* yf, const double* zf);
void* gcfd_ns_create2(const int* n, const double* len, double nu, int precision, const int* kinds, const double* g) {
  return gcfd_ns_create3(n, len, nu, precision, kinds, g, nullptr, nullptr, nullptr);
}
// As gcfd_ns_create2, on a grid stretched along any bounded direction: xf, yf, zf are the face positions
// of that direction (n + 1 values from 0 to len) or null for a uniform direction. All null: the uniform
// solver, unchanged.
void* gcfd_ns_create3(const int* n, const double* len, double nu, int precision, const int* kinds, const double* g,
                      const double* xf, const double* yf, const double* zf) {
  const double* faces[3] = {xf, yf, zf};
  for (int d = 0; d < 3; ++d) {
    if (n[d] >= 1) {
      const char* why = bad_faces(faces[d], n[d], len[d], kinds[2 * d] == 0);
      if (why) {
        fprintf(stderr, "gcfd: direction %c: %s\n", "xyz"[d], why);
        return nullptr;
      }
    }
  }
  for (int d = 0; d < 3; ++d) {
    const char* ax = "xyz";
    if (n[d] < 1) return nullptr;
    if ((kinds[2 * d] == 0) != (kinds[2 * d + 1] == 0)) {
      fprintf(stderr, "gcfd: direction %c: periodic must be set on both sides\n", ax[d]);
      return nullptr;
    }
    if (kinds[2 * d] == 0 && n[d] > 1 && n[d] % 2) {
      fprintf(stderr, "gcfd: direction %c: periodic needs an even cell count\n", ax[d]);
      return nullptr;
    }
    if (kinds[2 * d] != 0 && n[d] < 2) {
      fprintf(stderr, "gcfd: direction %c: a bounded direction needs at least 2 cells\n", ax[d]);
      return nullptr;
    }
    for (int s = 0; s < 2; ++s)
      if (kinds[2 * d + s] < 0 || kinds[2 * d + s] > 2) return nullptr;
  }
  if (n[0] < 2 || n[1] < 2) return nullptr;
  NS* m = new NS(n, len, nu, precision, kinds, g, faces);
  if (refused(m->P) || refused(static_cast<Solver<float>*>(m->P->pre))) {
    delete m;
    return nullptr;
  }
  return m;
}
// Stretched grids, experiments: bit mask of the directions that may use line relaxation (7: every
// bounded direction, the default; 0: the point smoother), and the smoothing sweeps before and after
// the coarse correction in the Navier-Stokes pressure solver. Apply to solvers created afterwards.
void gcfd_set_lines(int mask) {
  Solver<double>::line_mask() = mask;
  Solver<float>::line_mask() = mask;
}
void gcfd_set_ns_sweeps(int sweeps) { NS::mg_sweeps() = sweeps < 1 ? 1 : sweeps; }
int gcfd_ns_stretched(void* s) { return static_cast<NS*>(s)->L.geo ? 1 : 0; }
// Legacy form. walls: 3 ints (1: walls at both ends of that direction, 0: periodic). uwall: 18 doubles,
// the velocity of each wall (x-, x+, y-, y+, z-, z+; only tangential components are used).
void* gcfd_ns_create(const int* n, const double* len, double nu, int precision, const int* walls,
                     const double* uwall, const double* g) {
  int kinds[6];
  for (int d = 0; d < 3; ++d) kinds[2 * d] = kinds[2 * d + 1] = walls[d] ? 1 : 0;
  if (n[2] == 1 && walls[2]) return nullptr;
  NS* m = static_cast<NS*>(gcfd_ns_create2(n, len, nu, precision, kinds, g));
  if (!m) return nullptr;
  for (int side = 0; side < 6; ++side) {
    if (!kinds[side]) continue;
    for (int c = 0; c < 3; ++c) {
      const double v = c == side / 2 ? 0.0 : uwall[3 * side + c];  // a wall carries no flow through it
      std::vector<double> h(m->plane_cells(side / 2), v);
      m->set_plane(side, c, h.data());
    }
  }
  m->refresh();
  return m;
}
void gcfd_ns_destroy(void* s) { delete static_cast<NS*>(s); }
// Boundary data of one side and component (plane array, see Bnd), then the derived terms are updated.
void gcfd_ns_set_bnd(void* s, int side, int comp, const double* h) {
  NS* m = static_cast<NS*>(s);
  if (!m->W.u[side][comp]) return;
  m->set_plane(side, comp, h);
  m->refresh();
}
void gcfd_ns_get_bnd(void* s, int side, int comp, double* h) {
  NS* m = static_cast<NS*>(s);
  if (m->W.u[side][comp]) m->get_plane(side, comp, h);
}
void gcfd_ns_set(void* s, int comp, const double* h) {
  NS* m = static_cast<NS*>(s);
  CK(cudaMemcpy(m->q[comp], h, m->L.cells() * sizeof(double), cudaMemcpyHostToDevice));
  if (comp < m->ncomp) m->boundary_faces(comp);  // boundary faces: prescribed, or from the interior (outflow)
  CK(cudaStreamSynchronize(gs()));
}
void gcfd_ns_get(void* s, int comp, double* h) {
  NS* m = static_cast<NS*>(s);
  CK(cudaMemcpy(h, m->q[comp], m->L.cells() * sizeof(double), cudaMemcpyDeviceToHost));
}
// Advance nsteps steps of size dt. stats: [GPU ms, max |div u| after, mean pressure iterations per
// projection, mean CG iterations per diffusion solve (IMEX)]
void gcfd_ns_run(void* s, double dt, int nsteps, double ptol, double* stats) {
  NS* m = static_cast<NS*>(s);
  m->ptol = ptol;
  m->last_dt = dt;
  m->prepare(dt);
  k_cg_reset<<<1, 1, 0, gs()>>>(m->P->cg);
  k_cg_reset<<<1, 1, 0, gs()>>>(m->hc);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0, gs()));
  for (int q = 0; q < nsteps; ++q) m->step(dt);
  CK(cudaEventRecord(e1, gs()));
  CK(cudaEventSynchronize(e1));
  float t;
  CK(cudaEventElapsedTime(&t, e0, e1));
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  CK(cudaGetLastError());
  stats[0] = t;
  stats[1] = m->max_div();
  const CGState pc = read_cg(m->P->cg), hh = read_cg(m->hc);
  stats[2] = pc.solves ? double(pc.total_its) / pc.solves : 0.0;
  stats[3] = hh.solves ? double(hh.total_its) / hh.solves : 0.0;
}
// CUDA graphs: 1 on (default), 0 off (every kernel launched by the CPU, which reads each loop's stop
// flag). Results are bit-identical either way.
void gcfd_set_graphs(int on) { g_use_graphs = on != 0; }
// Levels with at most this many cells run as one thread block (the V-cycle tail); 0 switches it off.
void gcfd_set_tail(long long cells) { g_tail_cells = size_t(cells < 0 ? 0 : cells); }
// Bodies: how many of the finest multigrid levels carry the body's face coefficients (the coarser ones
// see no body). Applies to solvers given bodies afterwards.
void gcfd_set_mask_levels(int m) {
  Solver<double>::mask_levels() = m;
  Solver<float>::mask_levels() = m;
}

// scheme 0: explicit RK3; 1: IMEX (implicit diffusion). htol: relative tolerance of the diffusion solves.
void gcfd_ns_scheme(void* s, int scheme, double htol) {
  NS* m = static_cast<NS*>(s);
  m->scheme = scheme;
  m->htol = htol;
}
// Immersed bodies, component comp: mask (N values: 1 fluid face, 0 face in a body) and the n faces in
// bodies with their rule q[idx] = a0 + sum_k w[ns g + k] q[nb[ns g + k]] (ns = 4 in 2D, 8 in 3D).
// Call for every component, then gcfd_ns_body_done (before the first run).
void gcfd_ns_body(void* s, int comp, const double* mask, int n, const long long* idx, const double* a0,
                  const long long* nb, const double* w) {
  static_cast<NS*>(s)->set_body(comp, mask, n, idx, a0, nb, w);
}
void gcfd_ns_body_done(void* s) { static_cast<NS*>(s)->body_done(); }
// After gcfd_ns_body_done: 1 if the bodies were accepted, 0 if not (reason on stderr).
int gcfd_ns_body_ok(void* s) {
  NS* m = static_cast<NS*>(s);
  return (refused(m->P) || refused(static_cast<Solver<float>*>(m->P->pre))) ? 0 : 1;
}
// Pressure (per unit density) at the cell centres. IMEX, or with bodies: the pressure carried by the
// incremental projection. Explicit scheme without bodies: from the last projection of the last step.
void gcfd_ns_get_p(void* s, double* h) {
  NS* m = static_cast<NS*>(s);
  const size_t N = m->L.cells();
  CK(cudaMemcpy(h, (m->scheme == 1 || m->has_body) ? m->pres : m->phi, N * sizeof(double), cudaMemcpyDeviceToHost));
  if (!(m->scheme == 1 || m->has_body)) {  // phi = (2/3) dt p in the last stage of SSP-RK3
    const double f = m->last_dt > 0.0 ? 1.5 / m->last_dt : 0.0;
    for (size_t q = 0; q < N; ++q) h[q] *= f;
  }
}
// Sets the pressure carried by the incremental projection (IMEX, or with bodies), to continue from a
// stored state. The explicit scheme without bodies carries no pressure: ignored there.
void gcfd_ns_set_p(void* s, const double* h) {
  NS* m = static_cast<NS*>(s);
  CK(cudaMemcpy(m->pres, h, m->L.cells() * sizeof(double), cudaMemcpyHostToDevice));
}
// Explicit right-hand side of the momentum equation for component comp at the current velocity:
// - convection + nu Laplacian + body force (no pressure), at every face (meaningless on faces in bodies).
void gcfd_ns_rhs(void* s, int comp, double* h) {
  NS* m = static_cast<NS*>(s);
  k_ns_rhs<double><<<grid3(m->L), B3, 0, gs()>>>(m->q[0], m->q[1], m->q[2], m->tmp, m->L, comp, m->nu, m->W, m->G);
  CK(cudaStreamSynchronize(gs()));
  CK(cudaMemcpy(h, m->tmp, m->L.cells() * sizeof(double), cudaMemcpyDeviceToHost));
}
double gcfd_ns_project(void* s) {  // project the current velocity (initial data), return max |div u| after
  NS* m = static_cast<NS*>(s);
  m->project();
  return m->max_div();
}

// bc: 6 ints (x-, x+, y-, y+, z-, z+), 0 periodic, 1 Neumann, 2 Dirichlet. precision: 4 or 8 bytes.
void* gcfd_poisson_create_s(const int* n, const double* len, const int* bc, int precision, int nu1, int nu2,
                            const double* xf, const double* yf, const double* zf);
void* gcfd_poisson_create(const int* n, const double* len, const int* bc, int precision, int nu1, int nu2) {
  return gcfd_poisson_create_s(n, len, bc, precision, nu1, nu2, nullptr, nullptr, nullptr);
}
// As gcfd_poisson_create, on a stretched grid (face positions per direction, or null: see gcfd_ns_create3).
void* gcfd_poisson_create_s(const int* n, const double* len, const int* bc, int precision, int nu1, int nu2,
                            const double* xf, const double* yf, const double* zf) {
  const double* faces[3] = {xf, yf, zf};
  for (int d = 0; d < 3; ++d) {
    if (n[d] < 1) return nullptr;
    {
      const char* why = bad_faces(faces[d], n[d], len[d], bc[2 * d] == PERIODIC || bc[2 * d + 1] == PERIODIC);
      if (why) {
        fprintf(stderr, "gcfd: direction %d: %s\n", d, why);
        return nullptr;
      }
    }
    const bool periodic = bc[2 * d] == PERIODIC || bc[2 * d + 1] == PERIODIC;
    if (periodic && (bc[2 * d] != PERIODIC || bc[2 * d + 1] != PERIODIC)) {
      fprintf(stderr, "gcfd: direction %d: periodic must be set on both sides\n", d);
      return nullptr;
    }
    if (periodic && n[d] > 1 && n[d] % 2) {
      fprintf(stderr, "gcfd: direction %d: periodic needs an even cell count (red-black ordering)\n", d);
      return nullptr;
    }
  }
  if (precision == 4) {
    auto* s = new Solver<float>(n, len, bc, nu1, nu2, faces);
    if (!refused(s)) return s;
    delete s;
    return nullptr;
  }
  if (precision == 8 || precision == 6) {  // 6, mixed: double outer iteration, single-precision multigrid inside
    auto* s = new Solver<double>(n, len, bc, nu1, nu2, faces);
    if (precision == 6) s->pre = new Solver<float>(n, len, bc, nu1, nu2, faces);
    if (!refused(s) && !refused(static_cast<Solver<float>*>(s->pre))) return s;
    delete s;
    return nullptr;
  }
  return nullptr;
}
void gcfd_poisson_destroy(void* s) { delete static_cast<Base*>(s); }
int gcfd_poisson_levels(void* s, int* dims) { return static_cast<Base*>(s)->levels(dims); }
void gcfd_poisson_apply(void* s, const double* p, double* out) { static_cast<Base*>(s)->apply(p, out); }
int gcfd_poisson_solve(void* s, const double* f, double* x, double tol, int maxit, int method, double* hist,
                       double* ms) {
  return static_cast<Base*>(s)->solve(f, x, tol, maxit, method, hist, ms);
}
double gcfd_poisson_bench_vcycle(void* s, int reps) { return static_cast<Base*>(s)->bench_vcycle(reps); }

// Device-to-device copy bandwidth in GB/s (read + write), the ceiling for memory-bound kernels.
double gcfd_bandwidth(double mbytes, int reps) {
  const size_t n = size_t(mbytes * 1048576.0 / sizeof(float4));
  float4 *a, *b;
  CK(cudaMalloc(&a, n * sizeof(float4)));
  CK(cudaMalloc(&b, n * sizeof(float4)));
  CK(cudaMemset(a, 0, n * sizeof(float4)));
  k_copy4<<<1024, 256>>>(a, b, n);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  for (int q = 0; q < reps; ++q) k_copy4<<<4096, 256>>>(a, b, n);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  float t;
  CK(cudaEventElapsedTime(&t, e0, e1));
  cudaFree(a);
  cudaFree(b);
  return 2.0 * n * sizeof(float4) * reps / (t * 1e-3) / 1e9;
}

int gcfd_device(char* name, int len) {
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return -1;
  snprintf(name, len, "%s", p.name);
  return p.major * 10 + p.minor;
}
}
