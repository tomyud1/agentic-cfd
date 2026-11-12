// gpucfd: GPU pressure Poisson solver (geometric multigrid, alone or as the CG preconditioner).
// Cell-centred structured grid, no ghost cells: boundary conditions are applied inside the stencils.
// Arrays hold the interior cells only, x fastest: index = i + nx * (j + ny * k).
// Builds into a shared library with a plain C API (see the bottom of this file and ../gcfd.py).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

template <typename T>
__global__ void k_prolong_add(T* x, const T* e, Level F, Level C) {
  CELL(F)
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
  x[c] += T(s);
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

// Coarsest level: one thread block, many red-black sweeps from x = 0. For a singular problem the
// right-hand side is shifted to zero mean first and the solution returned with zero mean.
template <typename T>
__global__ void k_coarse(T* x, const T* b, Level L, Coef<T> K, int sweeps, int singular) {
  const size_t N = L.cells();
  const int n0 = L.n[0], n1 = L.n[1];
  double mb = 0.0;
  if (singular) {
    double s = 0.0;
    for (size_t c = threadIdx.x; c < N; c += blockDim.x) s += double(b[c]);
    mb = block_sum(s) / double(N);
  }
  for (size_t c = threadIdx.x; c < N; c += blockDim.x) x[c] = T(0);
  __syncthreads();
  for (int s = 0; s < sweeps; ++s)
    for (int color = 0; color < 2; ++color) {
      for (size_t c = threadIdx.x; c < N; c += blockDim.x) {
        const int i = int(c % n0), j = int((c / n0) % n1), k = int(c / (size_t(n0) * n1));
        if (((i + j + k) & 1) != color) continue;
        T off, diag;
        stencil(x, L, K, c, i, j, k, off, diag);
        x[c] = (off - (b[c] - T(mb))) / diag;
      }
      __syncthreads();
    }
  if (singular) {
    double s = 0.0;
    for (size_t c = threadIdx.x; c < N; c += blockDim.x) s += double(x[c]);
    const double mx = block_sum(s) / double(N);
    for (size_t c = threadIdx.x; c < N; c += blockDim.x) x[c] -= T(mx);
  }
}

template <typename T>
__global__ void k_dot(const T* a, const T* b, size_t n, double* out) {
  double s = 0.0;
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < n; q += size_t(gridDim.x) * blockDim.x)
    s += double(a[q]) * (b ? double(b[q]) : 1.0);
  s = block_sum(s);
  if (threadIdx.x == 0) atomicAdd(out, s);
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
  double* dsum = nullptr;
  bool singular = true;
  int nu1, nu2, coarse_sweeps;
  Base* pre = nullptr;  // mixed precision: single-precision multigrid used as the preconditioner

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
    CK(cudaMalloc(&dsum, sizeof(double)));
  }
  ~Solver() override {
    for (size_t l = 0; l < L.size(); ++l) {
      cudaFree(x[l]);
      cudaFree(b[l]);
      cudaFree(r[l]);
    }
    cudaFree(p_);
    cudaFree(q_);
    cudaFree(z_);
    cudaFree(rr_);
    cudaFree(dsum);
    delete pre;
  }

  void vcycle_d(const double* R, double* Z) override {
    const size_t N = L[0].cells();
    k_convert<double, T><<<lin_blocks(N), 256>>>(R, rr_, N);
    CK(cudaMemsetAsync(z_, 0, N * sizeof(T)));
    vcycle(0, z_, rr_);
    k_convert<T, double><<<lin_blocks(N), 256>>>(z_, Z, N);
  }

  // Z = M R: one V-cycle from zero, in this precision or (mixed) in the single-precision partner.
  void precondition(T* R, T* Z) {
    if constexpr (std::is_same<T, double>::value) {
      if (pre) {
        pre->vcycle_d(R, Z);
        return;
      }
    }
    CK(cudaMemsetAsync(Z, 0, L[0].cells() * sizeof(T)));
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
  double dot(const T* a, const T* bb, size_t n) {
    CK(cudaMemset(dsum, 0, sizeof(double)));
    k_dot<T><<<lin_blocks(n), 256>>>(a, bb, n, dsum);
    double s;
    CK(cudaMemcpy(&s, dsum, sizeof(double), cudaMemcpyDeviceToHost));
    return s;
  }
  void remove_mean(T* a, size_t n) {
    const double m = dot(a, nullptr, n) / double(n);
    k_add_const<T><<<lin_blocks(n), 256>>>(a, -m, n);
  }

  void smooth(int l, T* X, const T* B, int sweeps, bool forward) {
    for (int s = 0; s < sweeps; ++s) {
      k_rbgs<T><<<grid3(L[l]), B3>>>(X, B, L[l], K[l], forward ? 0 : 1);
      k_rbgs<T><<<grid3(L[l]), B3>>>(X, B, L[l], K[l], forward ? 1 : 0);
    }
  }

  // One V-cycle on level l for Laplacian(X) = B, improving X in place.
  void vcycle(int l, T* X, const T* B) {
    const int last = int(L.size()) - 1;
    if (l == last) {
      k_coarse<T><<<1, 512>>>(X, B, L[l], K[l], coarse_sweeps, singular ? 1 : 0);
      return;
    }
    smooth(l, X, B, nu1, true);
    k_residual<T><<<grid3(L[l]), B3>>>(X, B, r[l], L[l], K[l]);
    k_restrict<T><<<grid3(L[l + 1]), B3>>>(r[l], b[l + 1], L[l], L[l + 1]);
    CK(cudaMemsetAsync(x[l + 1], 0, L[l + 1].cells() * sizeof(T)));  // the correction starts from zero
    vcycle(l + 1, x[l + 1], b[l + 1]);
    k_prolong_add<T><<<grid3(L[l]), B3>>>(X, x[l + 1], L[l], L[l + 1]);
    smooth(l, X, B, nu2, false);
  }

  void apply(const double* ph, double* out) override {
    const size_t N = L[0].cells();
    upload(ph, x[0], N);
    k_apply<T><<<grid3(L[0]), B3>>>(x[0], r[0], L[0], K[0]);
    download(r[0], out, N);
  }

  // method 0: multigrid V-cycles; method 1: conjugate gradients preconditioned by one V-cycle.
  // hist[it] = ||f - A x|| / ||f|| after it iterations. Returns the iteration count.
  int solve(const double* fh, double* xh, double tol, int maxit, int method, double* hist, double* ms) override {
    const size_t N = L[0].cells();
    upload(fh, rr_, N);
    upload(xh, x[0], N);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    const int it = iterate(tol, maxit, method, hist);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float t;
    CK(cudaEventElapsedTime(&t, e0, e1));
    *ms = t;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    CK(cudaGetLastError());
    download(x[0], xh, N);
    return it;
  }

  // Solve with right-hand side and solution already on the GPU (x_dev: initial guess in, solution out).
  int solve_dev(const T* f_dev, T* x_dev, double tol, int maxit, double* hist) {
    const size_t N = L[0].cells();
    CK(cudaMemcpy(rr_, f_dev, N * sizeof(T), cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(x[0], x_dev, N * sizeof(T), cudaMemcpyDeviceToDevice));
    const int it = iterate(tol, maxit, 1, hist);
    CK(cudaMemcpy(x_dev, x[0], N * sizeof(T), cudaMemcpyDeviceToDevice));
    return it;
  }

  // The iteration itself: right-hand side in rr_, initial guess and result in x[0].
  int iterate(double tol, int maxit, int method, double* hist) {
    const size_t N = L[0].cells();
    T* X = x[0];
    T* F = rr_;  // keep the right-hand side apart from the level-0 multigrid arrays
    if (singular) remove_mean(F, N);
    const double fn = std::sqrt(dot(F, F, N)) + 1e-300;
    int it = 0;
    T* R = b[0];  // residual (the level-0 rhs array is free: the V-cycle takes its own arguments)
    if (method == 0) {
      // multigrid as defect correction: x += M (f - A x). With pre set, M runs in single precision
      // while x and the residual stay in double (iterative refinement).
      k_residual<T><<<grid3(L[0]), B3>>>(X, F, R, L[0], K[0]);
      hist[0] = std::sqrt(dot(R, R, N)) / fn;
      while (it < maxit && hist[it] > tol) {
        precondition(R, z_);
        k_axpy<T><<<lin_blocks(N), 256>>>(X, z_, 1.0, N);
        ++it;
        k_residual<T><<<grid3(L[0]), B3>>>(X, F, R, L[0], K[0]);
        hist[it] = std::sqrt(dot(R, R, N)) / fn;
      }
    } else {
      k_residual<T><<<grid3(L[0]), B3>>>(X, F, R, L[0], K[0]);
      hist[0] = std::sqrt(dot(R, R, N)) / fn;
      precondition(R, z_);
      CK(cudaMemcpy(p_, z_, N * sizeof(T), cudaMemcpyDeviceToDevice));
      double rz = dot(R, z_, N);
      while (it < maxit && hist[it] > tol) {
        k_apply<T><<<grid3(L[0]), B3>>>(p_, q_, L[0], K[0]);
        const double alpha = rz / dot(p_, q_, N);
        k_axpy<T><<<lin_blocks(N), 256>>>(X, p_, alpha, N);
        k_axpy<T><<<lin_blocks(N), 256>>>(R, q_, -alpha, N);
        ++it;
        hist[it] = std::sqrt(dot(R, R, N)) / fn;
        if (hist[it] <= tol) break;
        precondition(R, z_);
        const double rz2 = dot(R, z_, N);
        k_xpby<T><<<lin_blocks(N), 256>>>(p_, z_, rz2 / rz, N);
        rz = rz2;
      }
      // the recursively updated residual drifts from the true one in single precision: report the true one
      k_residual<T><<<grid3(L[0]), B3>>>(X, F, R, L[0], K[0]);
      hist[it] = std::sqrt(dot(R, R, N)) / fn;
    }
    if (singular) remove_mean(X, N);
    return it;
  }

  double bench_vcycle(int reps) override {
    if (pre) return pre->bench_vcycle(reps);  // mixed: the V-cycle runs in the single-precision partner
    const size_t N = L[0].cells();
    CK(cudaMemset(x[0], 0, N * sizeof(T)));
    CK(cudaMemset(rr_, 0, N * sizeof(T)));  // zero right-hand side: same work per cycle, no stale values
    vcycle(0, x[0], rr_);  // warm-up
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int q = 0; q < reps; ++q) vcycle(0, x[0], rr_);
    CK(cudaEventRecord(e1));
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
    const T h = T(L.h[d]);
    const long long mp = at(L, i + ed[0], j + ed[1], k + ed[2]);
    const long long mm = at(L, i - ed[0], j - ed[1], k - ed[2]);
    T qp, qm;
    if (d == comp) {  // along its own direction: the wall faces are zero
      qp = normal_at(q, mp);
      qm = normal_at(q, mm);
      const T Up = T(0.5) * (qc + qp), Um = T(0.5) * (qm + qc);  // cell-centre values right and left
      conv += (Up * Up - Um * Um) / h;
    } else {  // across direction d: ghost values beyond walls carry the wall speed
      qp = mp >= 0 ? q[mp] : T(2.0 * W.U[2 * d + 1][comp]) - qc;
      qm = mm >= 0 ? q[mm] : T(2.0 * W.U[2 * d][comp]) - qc;
      const T* a = V[d];  // advecting component at the face's edges; zero on a wall (no flow through it)
      const T aup = T(0.5) * (normal_at(a, mp) + normal_at(a, at(L, i + ed[0] - ec[0], j + ed[1] - ec[1], k + ed[2] - ec[2])));
      const T alo = T(0.5) * (a[c] + normal_at(a, at(L, i - ec[0], j - ec[1], k - ec[2])));
      conv += (aup * T(0.5) * (qc + qp) - alo * T(0.5) * (qm + qc)) / h;
    }
    diff += (qp - T(2) * qc + qm) / (h * h);
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
    s += (normal_at(V[d], at(L, i + E[d][0], j + E[d][1], k + E[d][2])) - V[d][c]) / T(L.h[d]);
  }
  dv[c] = s;
}

template <typename T>
__global__ void k_ns_correct(T* q, const T* phi, Level L, int comp) {
  CELL(L)
  const int I[3] = {i, j, k};
  if (wall_dir(L, comp) && I[comp] == 0) return;  // wall face: no correction (zero pressure gradient there)
  const int E[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  q[c] -= (phi[c] - phi[at(L, i - E[comp][0], j - E[comp][1], k - E[comp][2])]) / T(L.h[comp]);
}

template <typename T>
__global__ void k_zero_wall(T* q, Level L, int comp) {
  CELL(L)
  const int I[3] = {i, j, k};
  if (wall_dir(L, comp) && I[comp] == 0) q[c] = T(0);
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

struct NS {
  using T = double;
  Level L;
  double nu, ptol = 1e-10;
  int ncomp;
  T *q[3] = {nullptr, nullptr, nullptr}, *q0[3] = {nullptr, nullptr, nullptr}, *F[3] = {nullptr, nullptr, nullptr};
  T *phi = nullptr, *dv = nullptr;
  double* dmax = nullptr;
  Solver<double>* P;
  long poisson_its = 0, projections = 0;
  Walls W;
  Force G;

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
  }
  ~NS() {
    for (int s = 0; s < 3; ++s) {
      cudaFree(q[s]);
      cudaFree(q0[s]);
      cudaFree(F[s]);
    }
    cudaFree(phi);
    cudaFree(dv);
    cudaFree(dmax);
    delete P;
  }
  void rhs() {
    for (int s = 0; s < ncomp; ++s) k_ns_rhs<T><<<grid3(L), B3>>>(q[0], q[1], q[2], F[s], L, s, T(nu), W, G);
  }
  void project() {
    k_ns_div<T><<<grid3(L), B3>>>(q[0], q[1], q[2], dv, L);
    std::vector<double> hist(201);
    poisson_its += P->solve_dev(dv, phi, ptol, 200, hist.data());
    ++projections;
    for (int s = 0; s < ncomp; ++s) k_ns_correct<T><<<grid3(L), B3>>>(q[s], phi, L, s);
  }
  void stage(double a, double b, double dt) {
    rhs();
    for (int s = 0; s < ncomp; ++s) k_rk<T><<<lin_blocks(L.cells()), 256>>>(q[s], q0[s], F[s], a, b, dt, L.cells());
    project();
  }
  void step(double dt) {
    for (int s = 0; s < ncomp; ++s) CK(cudaMemcpyAsync(q0[s], q[s], L.cells() * sizeof(T), cudaMemcpyDeviceToDevice));
    stage(0.0, 1.0, dt);
    stage(0.75, 0.25, dt);
    stage(1.0 / 3.0, 2.0 / 3.0, dt);
  }
  double max_div() {
    k_ns_div<T><<<grid3(L), B3>>>(q[0], q[1], q[2], dv, L);
    CK(cudaMemset(dmax, 0, sizeof(double)));
    k_absmax<T><<<lin_blocks(L.cells()), 256>>>(dv, L.cells(), dmax);
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
  k_zero_wall<double><<<grid3(m->L), B3>>>(m->q[comp], m->L, comp);  // a wall face carries no flow
}
void gcfd_ns_get(void* s, int comp, double* h) {
  NS* m = static_cast<NS*>(s);
  CK(cudaMemcpy(h, m->q[comp], m->L.cells() * sizeof(double), cudaMemcpyDeviceToHost));
}
// Advance nsteps steps of size dt. stats: [GPU ms, max |div u| after, mean pressure iterations per projection]
void gcfd_ns_run(void* s, double dt, int nsteps, double ptol, double* stats) {
  NS* m = static_cast<NS*>(s);
  m->ptol = ptol;
  m->poisson_its = 0;
  m->projections = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  for (int q = 0; q < nsteps; ++q) m->step(dt);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  float t;
  CK(cudaEventElapsedTime(&t, e0, e1));
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  CK(cudaGetLastError());
  stats[0] = t;
  stats[1] = m->max_div();
  stats[2] = m->projections ? double(m->poisson_its) / m->projections : 0.0;
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
