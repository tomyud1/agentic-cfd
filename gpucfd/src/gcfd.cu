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
  __host__ __device__ size_t stride(int d) const {
    return d == 0 ? size_t(1) : (d == 1 ? size_t(n[0]) : size_t(n[0]) * n[1]);
  }
  __host__ __device__ size_t cells() const { return size_t(n[0]) * n[1] * n[2]; }
};

template <typename T>
struct Coef {
  T c[3];  // 1/h^2 per active direction, 0 for an inactive one
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

// Laplacian(p)[c] = off - diag * p[c]
template <typename T>
__device__ __forceinline__ void stencil(const T* p, const Level& L, const Coef<T>& K, size_t c, int i, int j,
                                        int k, T& off, T& diag) {
  const int ii[3] = {i, j, k};
  off = T(0);
  diag = T(0);
#pragma unroll
  for (int d = 0; d < 3; ++d) {
    if (K.c[d] == T(0)) continue;
    int sm, sp;
    const T vm = nbr(p, L, c, ii[d], d, -1, sm);
    const T vp = nbr(p, L, c, ii[d], d, +1, sp);
    off += K.c[d] * (vm + vp);
    diag += K.c[d] * T(2 - sm - sp);
  }
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

// Coarsest level: one thread block, many red-black sweeps from x = 0. For a singular problem the
// right-hand side is shifted to zero mean first and the solution returned with zero mean.
template <typename T>
__device__ void coarse_solve(T* x, const T* b, const Level& L, const Coef<T>& K, int sweeps, int singular) {
  const size_t N = L.cells();
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
static const int MAX_LEVELS = 24;
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

__global__ void k_pcg_start(CGState* g, double tol, int maxit) {
  g->fn = sqrt(g->s) + 1e-300;
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

  Solver(const int* n, const double* len, const int* bc, int nu1_, int nu2_) : nu1(nu1_), nu2(nu2_) {
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
    for (auto& l : L) {
      Coef<T> k;
      for (int d = 0; d < 3; ++d) k.c[d] = l.n[d] > 1 ? T(1.0 / (l.h[d] * l.h[d])) : T(0);
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
    dot_dev<T>(a, nullptr, n, part, &cg->s);
    k_sub_mean<T><<<lin_blocks(n), 256, 0, gs()>>>(a, &cg->s, n);
  }

  void smooth(int l, T* X, const T* B, int sweeps, bool forward) {
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
    if (l == 0) tail_from = first_tail();
    if (l == last) {
      k_coarse<T><<<1, COARSE_THREADS, 0, gs()>>>(X, B, L[l], K[l], coarse_sweeps, singular ? 1 : 0);
      return;
    }
    if (l >= tail_from) {  // this level and all coarser ones: one thread block, in shared memory
      Tail<T> t;
      t.n = last - l + 1;
      for (int m = 0; m < t.n; ++m) {
        t.L[m] = L[l + m];
        t.K[m] = K[l + m];
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
    k_residual_restrict<T><<<grid3(L[l + 1]), B3, 0, gs()>>>(X, B, b[l + 1], L[l], K[l], L[l + 1]);
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
  void solve_dev(T* F, T* X, double tol, int maxit) { iterate(F, X, tol, maxit, 1); }

  // The iteration itself, all decisions on the GPU.
  void iterate(T* F, T* X, double tol, int maxit, int method) {
    const size_t N = L[0].cells();
    if (maxit + 1 > hist_n) {
      fprintf(stderr, "gcfd: maxit %d above the prepared %d\n", maxit, hist_n - 1);
      abort();
    }
    T* R = b[0];  // residual (the level-0 rhs array is free: the V-cycle takes its own arguments)
    if (singular) remove_mean(F, N);
    dot_dev<T>(F, F, N, part, &cg->s);
    k_pcg_start<<<1, 1, 0, gs()>>>(cg, tol, maxit);
    k_residual<T><<<grid3(L[0]), B3, 0, gs()>>>(X, F, R, L[0], K[0]);
    dot_dev<T>(R, R, N, part, &cg->rr);
    k_pcg_test<<<1, 1, 0, gs()>>>(cg, hist, 0);
    if (method == 0) {
      // multigrid as defect correction: x += M (f - A x). With pre set, M runs in single precision
      // while x and the residual stay in double (iterative refinement).
      dev_while(&cg->go, [&] {
        precondition(R, z_);
        k_axpy<T><<<lin_blocks(N), 256, 0, gs()>>>(X, z_, 1.0, N);
        k_residual<T><<<grid3(L[0]), B3, 0, gs()>>>(X, F, R, L[0], K[0]);
        dot_dev<T>(R, R, N, part, &cg->rr);
        k_pcg_test<<<1, 1, 0, gs()>>>(cg, hist, 1);
      });
    } else {
      // Preconditioning at the top of the iteration, so the converged last iteration does not pay for
      // a V-cycle it would not use (the first version did; same iterates otherwise).
      dev_while(&cg->go, [&] {
        precondition(R, z_);
        dot_dev<T>(R, z_, N, part, &cg->rznew);
        k_pcg_beta<<<1, 1, 0, gs()>>>(cg);
        k_xpby_d<T><<<lin_blocks(N), 256, 0, gs()>>>(p_, z_, &cg->beta, &cg->first, N);  // p = z first time
        k_apply<T><<<grid3(L[0]), B3, 0, gs()>>>(p_, q_, L[0], K[0]);
        dot_dev<T>(p_, q_, N, part, &cg->pq);
        k_pcg_alpha<<<1, 1, 0, gs()>>>(cg);
        k_axpy_d<T><<<lin_blocks(N), 256, 0, gs()>>>(X, p_, &cg->alpha, 1.0, N);
        k_axpy_d<T><<<lin_blocks(N), 256, 0, gs()>>>(R, q_, &cg->alpha, -1.0, N);
        dot_dev<T>(R, R, N, part, &cg->rr);
        k_pcg_test<<<1, 1, 0, gs()>>>(cg, hist, 1);
      });
      // the recursively updated residual drifts from the true one in single precision: report the true one
      k_residual<T><<<grid3(L[0]), B3, 0, gs()>>>(X, F, R, L[0], K[0]);
      dot_dev<T>(R, R, N, part, &cg->rr);
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
// Each direction is periodic or bounded by a wall at both ends (L.bc = NEUMANN there, for the
// pressure). On a wall the normal velocity is zero: for the normal component, face 0 IS the low wall
// (kept at zero) and face n (not stored) is the high wall. A tangential component sees the ghost value
// 2 U_wall - q beyond a wall, so that its average on the wall equals the wall's speed (no slip; a
// moving lid has U_wall != 0).
// ================================================================================================
struct Walls {
  double U[6][3];  // velocity of each wall: x-, x+, y-, y+, z-, z+
};
struct Force {
  double g[3];  // constant body force per unit mass (drives channel flows)
};

__device__ __forceinline__ bool wall_dir(const Level& L, int d) { return L.bc[2 * d] != PERIODIC; }

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

template <typename T>
__global__ void k_ns_rhs(const T* __restrict__ u, const T* __restrict__ v, const T* __restrict__ w, T* F,
                         Level L, int comp, T nu, Walls W, Force G) {
  CELL(L)
  const int I[3] = {i, j, k};
  if (wall_dir(L, comp) && I[comp] == 0) {  // a wall face: the normal velocity stays zero
    F[c] = T(0);
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
    if (d == comp) {  // along its own direction: the wall faces are zero
      qp = normal_at(q, mp);
      qm = normal_at(q, mm);
      const T Up = T(0.5) * (qc + qp), Um = T(0.5) * (qm + qc);  // cell-centre values right and left
      conv += (Up * Up - Um * Um) * ih;
    } else {  // across direction d: ghost values beyond walls carry the wall speed
      qp = mp >= 0 ? q[mp] : T(2.0 * W.U[2 * d + 1][comp]) - qc;
      qm = mm >= 0 ? q[mm] : T(2.0 * W.U[2 * d][comp]) - qc;
      const T* a = V[d];  // advecting component at the face's edges; zero on a wall (no flow through it)
      const T aup = T(0.5) * (normal_at(a, mp) + normal_at(a, at(L, i + ed[0] - ec[0], j + ed[1] - ec[1], k + ed[2] - ec[2])));
      const T alo = T(0.5) * (a[c] + normal_at(a, at(L, i - ec[0], j - ec[1], k - ec[2])));
      conv += (aup * T(0.5) * (qc + qp) - alo * T(0.5) * (qm + qc)) * ih;
    }
    diff += (qp - T(2) * qc + qm) * ih2;
  }
  F[c] = -conv + nu * diff + T(G.g[comp]);
}

template <typename T>
__global__ void k_ns_div(const T* u, const T* v, const T* w, T* dv, Level L) {
  CELL(L)
  const T* V[3] = {u, v, w};
  const int E[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  T s = T(0);
  for (int d = 0; d < 3; ++d) {
    if (L.n[d] == 1) continue;
    s += (normal_at(V[d], at(L, i + E[d][0], j + E[d][1], k + E[d][2])) - V[d][c]) * T(L.ih[d]);
  }
  dv[c] = s;
}

// q -= scale * d(phi)/dx_comp at the comp-faces (wall faces: untouched)
template <typename T>
__global__ void k_ns_correct(T* q, const T* phi, Level L, int comp, double scale = 1.0) {
  CELL(L)
  const int I[3] = {i, j, k};
  if (wall_dir(L, comp) && I[comp] == 0) return;  // wall face: no correction (zero pressure gradient there)
  const int E[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  q[c] -= T(scale) * (phi[c] - phi[at(L, i - E[comp][0], j - E[comp][1], k - E[comp][2])]) * T(L.ih[comp]);
}

template <typename T>
__global__ void k_zero_wall(T* q, Level L, int comp) {
  CELL(L)
  const int I[3] = {i, j, k};
  if (wall_dir(L, comp) && I[comp] == 0) q[c] = T(0);
}

// ---------------------------------------------------------------- implicit diffusion (IMEX scheme)
// Laplacian of velocity component comp at its face (i,j,k), with the same wall rules as k_ns_rhs.
// homog = 1 drops the wall speed (ghost -q instead of 2 U_wall - q): the linear part of the operator.
template <typename T>
__device__ __forceinline__ T lap_at(const T* q, const Level& L, int comp, const Walls& W, int homog, int i, int j,
                                    int k, size_t c) {
  const int E[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const T qc = q[c];
  T s = T(0);
  for (int d = 0; d < 3; ++d) {
    if (L.n[d] == 1) continue;
    const int* ed = E[d];
    const T ih = T(L.ih[d]), ih2 = T(L.ih2[d]);
    const long long mp = at(L, i + ed[0], j + ed[1], k + ed[2]);
    const long long mm = at(L, i - ed[0], j - ed[1], k - ed[2]);
    T qp, qm;
    if (d == comp) {
      qp = normal_at(q, mp);
      qm = normal_at(q, mm);
    } else {
      qp = mp >= 0 ? q[mp] : (homog ? T(0) : T(2.0 * W.U[2 * d + 1][comp])) - qc;
      qm = mm >= 0 ? q[mm] : (homog ? T(0) : T(2.0 * W.U[2 * d][comp])) - qc;
    }
    s += (qp - T(2) * qc + qm) * ih2;
  }
  return s;
}

template <typename T>
__global__ void k_ns_lap(const T* q, T* out, Level L, int comp, Walls W, int homog) {
  CELL(L)
  const int I[3] = {i, j, k};
  out[c] = (wall_dir(L, comp) && I[comp] == 0) ? T(0) : lap_at(q, L, comp, W, homog, i, j, k, c);
}

// Helmholtz operator A x = x - cnu * Laplacian_homogeneous(x); the identity on wall faces (kept at zero).
template <typename T>
__global__ void k_helm(const T* x, T* Ax, Level L, int comp, Walls W, double cnu) {
  CELL(L)
  const int I[3] = {i, j, k};
  Ax[c] = (wall_dir(L, comp) && I[comp] == 0) ? x[c] : x[c] - T(cnu) * lap_at(x, L, comp, W, 1, i, j, k, c);
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
  Walls W;
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

  NS(const int* n, const double* len, double nu_, int precision, const int* walls, const double* uwall,
     const double* g)
      : nu(nu_) {
    int bc[6];
    for (int d = 0; d < 3; ++d) bc[2 * d] = bc[2 * d + 1] = walls[d] ? NEUMANN : PERIODIC;
    for (int s = 0; s < 6; ++s)
      for (int d = 0; d < 3; ++d) W.U[s][d] = uwall[3 * s + d];
    for (int d = 0; d < 3; ++d) G.g[d] = g[d];
    P = new Solver<double>(n, len, bc, 2, 2);
    if (precision == 6) P->pre = new Solver<float>(n, len, bc, 2, 2);
    L = P->L[0];
    ncomp = n[2] > 1 ? 3 : 2;
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
      // wall terms of the Laplacian: Lap(0) with the walls' speeds
      CK(cudaMemset(tmp, 0, N * sizeof(T)));
      k_ns_lap<T><<<grid3(L), B3, 0, gs()>>>(tmp, Bw[s], L, s, W, 0);
    }
    CK(cudaDeviceSynchronize());
  }
  ~NS() {
    for (int s = 0; s < 3; ++s) {
      cudaFree(q[s]);
      cudaFree(q0[s]);
      cudaFree(F[s]);
      cudaFree(Fo[s]);
      cudaFree(Bw[s]);
    }
    for (T* a : {phi, dv, tmp, cr, cp, cAp, pres}) cudaFree(a);
    cudaFree(dmax);
    cudaFree(hc);
    cudaFree(hpart);
    delete P;
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
    dot_dev<T>(cr, cr, n, hpart, &hc->rr);
    dot_dev<T>(b, b, n, hpart, &hc->s);
    k_helm_start<<<1, 1, 0, gs()>>>(hc, htol);
    dev_while(&hc->go, [&] {
      k_helm<T><<<grid3(L), B3, 0, gs()>>>(cp, cAp, L, comp, W, cnu);
      dot_dev<T>(cp, cAp, n, hpart, &hc->pq);
      k_helm_alpha<<<1, 1, 0, gs()>>>(hc);
      k_axpy_d<T><<<lin_blocks(n), 256, 0, gs()>>>(x, cp, &hc->alpha, 1.0, n);
      k_axpy_d<T><<<lin_blocks(n), 256, 0, gs()>>>(cr, cAp, &hc->alpha, -1.0, n);
      dot_dev<T>(cr, cr, n, hpart, &hc->rznew);
      k_helm_beta<<<1, 1, 0, gs()>>>(hc);
      k_xpby_d<T><<<lin_blocks(n), 256, 0, gs()>>>(cp, cr, &hc->beta, &hc->first, n);
    });
    k_cg_done<<<1, 1, 0, gs()>>>(hc);
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
      rhs(Fn);                              // Fn = N(u_{k-1})
      for (int s = 0; s < ncomp; ++s) {
        k_ns_lap<T><<<grid3(L), B3, 0, gs()>>>(q[s], tmp, L, s, W, 0);
        k_imex_b<T><<<lin_blocks(n), 256, 0, gs()>>>(q0[s], q[s], Fn[s], Fold[s], tmp, Bw[s], dt, gam[k], zeta[k],
                                                     ab[k], ab[k], nu, n);  // q0 holds the stage's right-hand side
        k_ns_correct<T><<<grid3(L), B3, 0, gs()>>>(q0[s], pres, L, s, cdt);  // - (alpha+beta) dt grad p_old
        helmholtz(s, q[s], q0[s], ab[k] * dt * nu);                         // initial guess: the current velocity
        std::swap(Fn[s], Fold[s]);                                           // N(u_{k-1}) becomes N_old next stage
      }
      project();                                                              // u = u* - grad(phi)
      k_axpy<T><<<lin_blocks(n), 256, 0, gs()>>>(pres, phi, 1.0 / cdt, n);  // p += phi / ((alpha+beta) dt)
    }
  }
  void project() {
    k_ns_div<T><<<grid3(L), B3, 0, gs()>>>(q[0], q[1], q[2], dv, L);
    P->solve_dev(dv, phi, ptol, 200);  // phi from the last projection is the initial guess
    for (int s = 0; s < ncomp; ++s) k_ns_correct<T><<<grid3(L), B3, 0, gs()>>>(q[s], phi, L, s);
  }
  void stage(double a, double b, double dt) {
    rhs(F);
    for (int s = 0; s < ncomp; ++s)
      k_rk<T><<<lin_blocks(L.cells()), 256, 0, gs()>>>(q[s], q0[s], F[s], a, b, dt, L.cells());
    project();
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
    k_ns_div<T><<<grid3(L), B3, 0, gs()>>>(q[0], q[1], q[2], dv, L);
    CK(cudaMemsetAsync(dmax, 0, sizeof(double), gs()));
    k_absmax<T><<<lin_blocks(L.cells()), 256, 0, gs()>>>(dv, L.cells(), dmax);
    double m;
    CK(cudaMemcpy(&m, dmax, sizeof(double), cudaMemcpyDeviceToHost));
    return m;
  }
};

extern "C" {

// walls: 3 ints (1: walls at both ends of that direction, 0: periodic). uwall: 18 doubles, the velocity
// of each wall (x-, x+, y-, y+, z-, z+; only tangential components are used). g: body force (3).
void* gcfd_ns_create(const int* n, const double* len, double nu, int precision, const int* walls,
                     const double* uwall, const double* g) {
  for (int d = 0; d < 3; ++d) {
    if (n[d] < 1) return nullptr;
    if (!walls[d] && n[d] > 1 && n[d] % 2) return nullptr;  // periodic: even (red-black ordering)
  }
  if (n[0] < 2 || n[1] < 2 || (n[2] == 1 && walls[2])) return nullptr;
  return new NS(n, len, nu, precision, walls, uwall, g);
}
void gcfd_ns_destroy(void* s) { delete static_cast<NS*>(s); }
void gcfd_ns_set(void* s, int comp, const double* h) {
  NS* m = static_cast<NS*>(s);
  CK(cudaMemcpy(m->q[comp], h, m->L.cells() * sizeof(double), cudaMemcpyHostToDevice));
  k_zero_wall<double><<<grid3(m->L), B3, 0, gs()>>>(m->q[comp], m->L, comp);  // a wall face carries no flow
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

// scheme 0: explicit RK3; 1: IMEX (implicit diffusion). htol: relative tolerance of the diffusion solves.
void gcfd_ns_scheme(void* s, int scheme, double htol) {
  NS* m = static_cast<NS*>(s);
  m->scheme = scheme;
  m->htol = htol;
}
double gcfd_ns_project(void* s) {  // project the current velocity (initial data), return max |div u| after
  NS* m = static_cast<NS*>(s);
  m->project();
  return m->max_div();
}

// bc: 6 ints (x-, x+, y-, y+, z-, z+), 0 periodic, 1 Neumann, 2 Dirichlet. precision: 4 or 8 bytes.
void* gcfd_poisson_create(const int* n, const double* len, const int* bc, int precision, int nu1, int nu2) {
  for (int d = 0; d < 3; ++d) {
    if (n[d] < 1) return nullptr;
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
  if (precision == 4) return new Solver<float>(n, len, bc, nu1, nu2);
  if (precision == 8) return new Solver<double>(n, len, bc, nu1, nu2);
  if (precision == 6) {  // mixed: double outer iteration, single-precision multigrid inside
    auto* s = new Solver<double>(n, len, bc, nu1, nu2);
    s->pre = new Solver<float>(n, len, bc, nu1, nu2);
    return s;
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
