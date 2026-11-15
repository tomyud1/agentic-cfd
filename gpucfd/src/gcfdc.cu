// gpucfd, compressible part: the Euler equations for an ideal gas with shocks, on the GPU.
// Design and verification: gpucfd/COMPRESSIBLE.md.  Build: bash gpucfd/build.sh  (build/libgcfdc.so).
//
// Conservative finite volumes on a uniform Cartesian grid. Per cell five numbers, each in its own array
// (x fastest): density, momentum x, y, z, total energy. A direction with one cell is switched off.
// One kernel launch per stage of the time scheme: every cell reads its neighbours along each direction,
// builds the states on its two faces, solves the two Riemann problems and writes the stage result
//     out = (1 - b) U0 + b (in + dt L(in)).
// The flux of a face is computed by both cells that share it, with the same code and the same numbers,
// so both get the same value and the scheme conserves to round-off.
// Boundaries are applied inside the stencil (no ghost cells are stored).
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define MAXSEG 4
#define MAXSTATE 8
#define NRED 8
#define BLOCK 256

enum { BC_PERIODIC = 0, BC_WALL = 1, BC_INFLOW = 2, BC_OUTFLOW = 3, BC_SUPERSONIC = 4 };
enum { RC_FIRST = 0, RC_MINMOD = 1, RC_VANLEER = 2, RC_MC = 3, RC_CENTRAL = 4, RC_WENO5 = 5 };
enum { FX_RUSANOV = 0, FX_HLL = 1, FX_HLLC = 2, FX_HYBRID = 3 };

template <class T> struct Par {
    int n[3], act[3];
    T gamma, a, b, sensor, weps;
    double dth[3];                             // dt / h per direction
    int recon, flux;
    int side_type[6], side_state[6], nseg[6];
    int seg_lo[6][MAXSEG][2], seg_hi[6][MAXSEG][2], seg_type[6][MAXSEG], seg_state[6][MAXSEG];
    T state[MAXSTATE][5];                      // rho, u, v, w, p
};

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)

// ---- boundaries: what a stencil sees beyond a side -----------------------------------------------------
// Primitive state (rho, normal velocity, two tangential velocities, p) of the cell k steps along direction d
// from cell c. m is the cell it was read from (-1: a prescribed state); kind is the boundary type used (-1 inside).
template <class S, class T> __device__ __forceinline__
void load(const Par<T>& P, const S* q, long N, const int* c, int d, int k, T* w, long& m, int& kind) {
    const int t1 = (d + 1) % 3, t2 = (d + 2) % 3, n = P.n[d];
    int idx = c[d] + k;
    T sgn = 1;
    kind = -1;
    if (idx < 0 || idx >= n) {
        const int side = 2 * d + (idx >= n);
        int type = P.side_type[side], st = P.side_state[side];
        for (int s = 0; s < P.nseg[side]; s++)
            if (c[t1] >= P.seg_lo[side][s][0] && c[t1] < P.seg_hi[side][s][0] &&
                c[t2] >= P.seg_lo[side][s][1] && c[t2] < P.seg_hi[side][s][1]) {
                type = P.seg_type[side][s]; st = P.seg_state[side][s]; break;
            }
        kind = type;
        if (type == BC_PERIODIC) idx = ((idx % n) + n) % n;
        else if (type == BC_WALL) { idx = idx < 0 ? -1 - idx : 2 * n - 1 - idx; sgn = -1; }   // mirror image
        else if (type == BC_OUTFLOW) idx = idx < 0 ? 0 : n - 1;                                  // zero gradient
        else {                                                                                   // prescribed state
            w[0] = P.state[st][0]; w[1] = P.state[st][1 + d]; w[2] = P.state[st][1 + t1]; w[3] = P.state[st][1 + t2];
            w[4] = P.state[st][4]; m = -1; return;
        }
    }
    int cc[3] = {c[0], c[1], c[2]};
    cc[d] = idx;
    m = cc[0] + (long)P.n[0] * (cc[1] + (long)P.n[1] * cc[2]);
    const T r = (T)q[m], ir = T(1) / r;
    const T mom[3] = {(T)q[N + m], (T)q[2 * N + m], (T)q[3 * N + m]};
    const T E = (T)q[4 * N + m];
    w[0] = r; w[1] = sgn * mom[d] * ir; w[2] = mom[t1] * ir; w[3] = mom[t2] * ir;
    w[4] = (P.gamma - 1) * (E - T(0.5) * (mom[0] * mom[0] + mom[1] * mom[1] + mom[2] * mom[2]) * ir);
}

// ---- reconstruction: the two face values of a cell from its neighbours ---------------------------------
template <class T> __device__ __forceinline__ T slope(int rc, T dm, T dp) {
    if (rc == RC_CENTRAL) return T(0.5) * (dm + dp);
    if (!(dm * dp > 0)) return 0;
    if (rc == RC_MINMOD) return fabs(dm) < fabs(dp) ? dm : dp;
    if (rc == RC_VANLEER) return 2 * dm * dp / (dm + dp);
    const T x = 2 * fabs(dm), y = 2 * fabs(dp), z = T(0.5) * fabs(dm + dp);                       // MC
    const T s = x < y ? (x < z ? x : z) : (y < z ? y : z);
    return dm > 0 ? s : -s;
}

// WENO5-Z value at the face between c and d, from the side of c (Borges et al. 2008)
template <class T> __device__ __forceinline__ T weno(T a, T b, T c, T d, T e, T tiny, T weps) {
    const T b0 = T(13.0 / 12.0) * (a - 2 * b + c) * (a - 2 * b + c) + T(0.25) * (a - 4 * b + 3 * c) * (a - 4 * b + 3 * c);
    const T b1 = T(13.0 / 12.0) * (b - 2 * c + d) * (b - 2 * c + d) + T(0.25) * (b - d) * (b - d);
    const T b2 = T(13.0 / 12.0) * (c - 2 * d + e) * (c - 2 * d + e) + T(0.25) * (3 * c - 4 * d + e) * (3 * c - 4 * d + e);
    const T eps = weps * (a * a + b * b + c * c + d * d + e * e) + tiny;
    const T tau = fabs(b0 - b2);
    const T a0 = T(0.1) * (1 + tau / (b0 + eps)), a1 = T(0.6) * (1 + tau / (b1 + eps)), a2 = T(0.3) * (1 + tau / (b2 + eps));
    return (a0 * (2 * a - 7 * b + 11 * c) + a1 * (-b + 5 * c + 2 * d) + a2 * (2 * c + 5 * d - e)) / (6 * (a0 + a1 + a2));
}

// w: the stencil, w[j] the cell itself. wm: value on its low face, wp: on its high face.
template <class T> __device__ __forceinline__ void recon_cell(int rc, const T (*w)[5], int j, T* wm, T* wp, T tiny, T weps) {
    for (int v = 0; v < 5; v++) {
        const T c = w[j][v];
        if (rc == RC_FIRST) { wm[v] = c; wp[v] = c; }
        else if (rc == RC_WENO5) {
            wp[v] = weno(w[j - 2][v], w[j - 1][v], c, w[j + 1][v], w[j + 2][v], tiny, weps);
            wm[v] = weno(w[j + 2][v], w[j + 1][v], c, w[j - 1][v], w[j - 2][v], tiny, weps);
        } else {
            const T s = T(0.5) * slope(rc, c - w[j - 1][v], w[j + 1][v] - c);
            wm[v] = c - s; wp[v] = c + s;
        }
    }
    // a face value with no gas in it: fall back to the cell's own value (first order in this cell)
    if (!(wm[0] > 0 && wm[4] > 0 && wp[0] > 0 && wp[4] > 0))
        for (int v = 0; v < 5; v++) { wm[v] = w[j][v]; wp[v] = w[j][v]; }
}

// ---- fluxes -------------------------------------------------------------------------------------------
template <class T> __device__ __forceinline__ void physflux(T g, const T* w, T* F, T* U) {
    const T r = w[0], u = w[1], p = w[4];
    const T E = p / (g - 1) + T(0.5) * r * (u * u + w[2] * w[2] + w[3] * w[3]);
    U[0] = r; U[1] = r * u; U[2] = r * w[2]; U[3] = r * w[3]; U[4] = E;
    F[0] = r * u; F[1] = r * u * u + p; F[2] = r * u * w[2]; F[3] = r * u * w[3]; F[4] = u * (E + p);
}

// Flux through a face from the states on its two sides (normal velocity in w[1]).
template <class T> __device__ __forceinline__ void riemann(int fx, T g, const T* wl, const T* wr, T* F) {
    T Fl[5], Fr[5], Ul[5], Ur[5];
    physflux(g, wl, Fl, Ul);
    physflux(g, wr, Fr, Ur);
    const T cl = sqrt(g * wl[4] / wl[0]), cr = sqrt(g * wr[4] / wr[0]);
    if (fx == FX_RUSANOV) {
        const T sl = fabs(wl[1]) + cl, sr = fabs(wr[1]) + cr, s = sl > sr ? sl : sr;
        for (int v = 0; v < 5; v++) F[v] = T(0.5) * (Fl[v] + Fr[v]) - T(0.5) * s * (Ur[v] - Ul[v]);
        return;
    }
    // fastest waves to the left and right: the two states' own speeds and those of their Roe average (Einfeldt)
    const T ql = sqrt(wl[0]), qr = sqrt(wr[0]), iq = T(1) / (ql + qr);
    const T ut = (ql * wl[1] + qr * wr[1]) * iq, vt = (ql * wl[2] + qr * wr[2]) * iq, wt = (ql * wl[3] + qr * wr[3]) * iq;
    const T Ht = (ql * (Ul[4] + wl[4]) / wl[0] + qr * (Ur[4] + wr[4]) / wr[0]) * iq;
    const T ct = sqrt((g - 1) * (Ht - T(0.5) * (ut * ut + vt * vt + wt * wt)));
    T SL = wl[1] - cl, SR = wr[1] + cr;
    if (ut - ct < SL) SL = ut - ct;
    if (ut + ct > SR) SR = ut + ct;
    if (SL >= 0) { for (int v = 0; v < 5; v++) F[v] = Fl[v]; return; }
    if (SR <= 0) { for (int v = 0; v < 5; v++) F[v] = Fr[v]; return; }
    if (fx == FX_HLL) {
        const T i = T(1) / (SR - SL);
        for (int v = 0; v < 5; v++) F[v] = (SR * Fl[v] - SL * Fr[v] + SL * SR * (Ur[v] - Ul[v])) * i;
        return;
    }
    // HLLC (Toro): the middle wave (contact) restored; written so that a wall gets exactly zero mass and energy flux
    const T ml = wl[0] * (SL - wl[1]), mr = wr[0] * (SR - wr[1]);
    const T Ss = (wr[4] - wl[4] + ml * wl[1] - mr * wr[1]) / (ml - mr);
    const T plr = T(0.5) * (wl[4] + wr[4] + ml * (Ss - wl[1]) + mr * (Ss - wr[1]));
    const T D[5] = {0, 1, 0, 0, Ss};
    // F = [Ss (SK UK - FK) + SK plr D] / (SK - Ss), written as plr D + Ss (...) / (SK - Ss): with Ss = 0 (a wall, gas
    // at rest) the flux is then exactly (0, p, 0, 0, 0), in any precision
    if (Ss >= 0) { const T i = Ss / (SL - Ss); for (int v = 0; v < 5; v++) F[v] = plr * D[v] + (SL * Ul[v] - Fl[v] + plr * D[v]) * i; }
    else         { const T i = Ss / (SR - Ss); for (int v = 0; v < 5; v++) F[v] = plr * D[v] + (SR * Ur[v] - Fr[v] + plr * D[v]) * i; }
}

// ---- one stage of the time scheme ----------------------------------------------------------------------
// S: the type the state is stored in; T: the type the fluxes are computed in (S = double with T = float is "mixed")
template <class S, class T> __global__ void k_stage(Par<T> P, const S* q, const S* q0, S* qo, const unsigned char* fl, long N, T tiny) {
    const long m = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;
    const int c[3] = {(int)(m % P.n[0]), (int)((m / P.n[0]) % P.n[1]), (int)(m / ((long)P.n[0] * P.n[1]))};
    const int reach = P.recon == RC_WENO5 ? 3 : 2;
    S R[5] = {0, 0, 0, 0, 0};                  // the fluxes are T, their differences are summed in S (see the end)
    for (int d = 0; d < 3; d++) {
        if (!P.act[d]) continue;
        const int t1 = (d + 1) % 3, t2 = (d + 2) % 3;
        T w[7][5];
        long mi[7];
        int kind[7];
        for (int k = 0; k < 7; k++) { mi[k] = -1; kind[k] = -1; for (int v = 0; v < 5; v++) w[k][v] = 1; }
        for (int k = -reach; k <= reach; k++) load(P, q, N, c, d, k, w[k + 3], mi[k + 3], kind[k + 3]);
        T wm[3][5], wp[3][5];
        for (int j = -1; j <= 1; j++) recon_cell(P.recon, w, j + 3, wm[j + 1], wp[j + 1], tiny, P.weps);
        int fa = P.flux, fb = P.flux;                                  // low face, high face
        if (P.flux == FX_HYBRID) {
            // HLLC everywhere, except HLL on a face that lies along a strong shock (a shock found in ANOTHER
            // direction in one of the two cells): that is where HLLC lets a flat shock break up
            const int other = ~(1 << d) & 7;
            const int f0 = fl[m], fm = mi[2] >= 0 ? fl[mi[2]] : 0, fp = mi[4] >= 0 ? fl[mi[4]] : 0;
            fa = ((f0 | fm) & other) ? FX_HLL : FX_HLLC;
            fb = ((f0 | fp) & other) ? FX_HLL : FX_HLLC;
        }
        T Fa[5], Fb[5], U[5];
        if (kind[2] == BC_SUPERSONIC) physflux(P.gamma, w[2], Fa, U); else riemann(fa, P.gamma, wp[0], wm[1], Fa);
        if (kind[4] == BC_SUPERSONIC) physflux(P.gamma, w[4], Fb, U); else riemann(fb, P.gamma, wp[1], wm[2], Fb);
        const S s = (S)P.dth[d];
        R[0] -= s * ((S)Fb[0] - (S)Fa[0]);
        R[1 + d] -= s * ((S)Fb[1] - (S)Fa[1]);
        R[1 + t1] -= s * ((S)Fb[2] - (S)Fa[2]);
        R[1 + t2] -= s * ((S)Fb[3] - (S)Fa[3]);
        R[4] -= s * ((S)Fb[4] - (S)Fa[4]);
    }
    // a U0 + b (in + dt L(in)) with a = 1 - b, written as U0 + b ((in - U0) + dt L): in single precision the
    // rounded weights 1/3 and 2/3 add up to 1 + 3e-8, which made every step create that share of mass (rung 2)
    // Mixed precision (S double, T float): both cells of a face get the same float flux, and everything after it
    // is done in double, so what leaves one cell enters the next to double round-off.
    for (int v = 0; v < 5; v++) qo[v * N + m] = q0[v * N + m] + (S)P.b * ((q[v * N + m] - q0[v * N + m]) + R[v]);
}

// One bit per direction: is there a strong pressure jump between this cell and a neighbour along it?
template <class S, class T> __global__ void k_sensor(Par<T> P, const S* q, unsigned char* fl, long N) {
    const long m = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;
    const int c[3] = {(int)(m % P.n[0]), (int)((m / P.n[0]) % P.n[1]), (int)(m / ((long)P.n[0] * P.n[1]))};
    T w[5];
    long mm;
    int kind;
    load(P, q, N, c, 0, 0, w, mm, kind);
    const T p0 = w[4];
    int f = 0;
    for (int d = 0; d < 3; d++) {
        if (!P.act[d]) continue;
        for (int k = -1; k <= 1; k += 2) {
            load(P, q, N, c, d, k, w, mm, kind);
            const T p1 = w[4];
            if (fabs(p1 - p0) > P.sensor * (p1 < p0 ? p1 : p0)) f |= 1 << d;
        }
    }
    fl[m] = (unsigned char)f;
}

// Per block of cells: sums of the five conserved numbers, largest signal speed over cell size, lowest density
// and pressure. Summed in a fixed order, in double, so the totals are the same in every run.
template <class S, class T> __global__ void k_reduce(Par<T> P, const S* q, long N, double* out, double ih0, double ih1, double ih2) {
    __shared__ double sh[NRED][BLOCK];
    const int t = threadIdx.x;
    const long m = (long)blockIdx.x * blockDim.x + t;
    double v[NRED] = {0, 0, 0, 0, 0, 0, 1e300, 1e300};
    if (m < N) {
        const double r = q[m], mx = q[N + m], my = q[2 * N + m], mz = q[3 * N + m], E = q[4 * N + m];
        const double p = ((double)P.gamma - 1) * (E - 0.5 * (mx * mx + my * my + mz * mz) / r);
        const double cs = sqrt((double)P.gamma * p / r);
        v[0] = r; v[1] = mx; v[2] = my; v[3] = mz; v[4] = E;
        v[5] = P.act[0] * (fabs(mx / r) + cs) * ih0 + P.act[1] * (fabs(my / r) + cs) * ih1 + P.act[2] * (fabs(mz / r) + cs) * ih2;
        v[6] = r; v[7] = p;
    }
    for (int k = 0; k < NRED; k++) sh[k][t] = v[k];
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) {
            for (int k = 0; k < 5; k++) sh[k][t] += sh[k][t + s];
            double a = sh[5][t], b = sh[5][t + s];
            sh[5][t] = (a > b || a != a) ? a : b;                      // a NaN wins, so a broken run is seen
            for (int k = 6; k < 8; k++) { a = sh[k][t]; b = sh[k][t + s]; sh[k][t] = (a < b || a != a) ? a : b; }
        }
        __syncthreads();
    }
    if (t == 0) for (int k = 0; k < NRED; k++) out[(long)blockIdx.x * NRED + k] = sh[k][0];
}

// ---- host side ----------------------------------------------------------------------------------------
struct Base {
    virtual ~Base() {}
    virtual void set(const double* U) = 0;
    virtual void get(double* U) = 0;
    virtual void scheme(int recon, int flux, int stages, double sensor) = 0;
    virtual int add_state(const double* s) = 0;
    virtual void side(int side, int type, int state) = 0;
    virtual int segment(int side, const int* lo, const int* hi, int type, int state) = 0;
    virtual void reduce(double* r) = 0;
    virtual void step(double dt) = 0;
    virtual double bytes() = 0;
    virtual double state_speed() = 0;
    virtual void weno_eps(double e) = 0;
    double t = 0;
};

template <class S, class T> struct Solver : Base {
    Par<T> P;
    long N;
    int nblk, stages = 3, nstate = 0;
    double h[3];
    S *A, *B, *C;
    unsigned char* fl;
    double* red;
    std::vector<double> hred;
    std::vector<S> buf;

    Solver(const int* n, const double* hh, double gamma) {
        memset(&P, 0, sizeof(P));
        N = 1;
        for (int d = 0; d < 3; d++) { P.n[d] = n[d]; P.act[d] = n[d] > 1; h[d] = hh[d]; N *= n[d]; }
        P.gamma = (T)gamma; P.recon = RC_VANLEER; P.flux = FX_HYBRID; P.sensor = (T)0.5;
        P.weps = sizeof(T) == 4 ? (T)1e-10 : (T)1e-20;
        nblk = (int)((N + BLOCK - 1) / BLOCK);
        CHECK(cudaMalloc(&A, 5 * N * sizeof(S)));
        CHECK(cudaMalloc(&B, 5 * N * sizeof(S)));
        CHECK(cudaMalloc(&C, 5 * N * sizeof(S)));
        CHECK(cudaMalloc(&fl, N));
        CHECK(cudaMalloc(&red, (size_t)nblk * NRED * sizeof(double)));
        CHECK(cudaMemset(fl, 0, N));
        hred.resize((size_t)nblk * NRED);
        buf.resize(5 * N);
    }
    ~Solver() { cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(fl); cudaFree(red); }
    void weno_eps(double e) override { P.weps = (T)e; }
    // Largest signal speed over cell size among the prescribed boundary states: the gas that is about to enter
    // must limit the time step too (at the start of an inflow into still gas it is the only fast gas there is).
    double state_speed() override {
        double m = 0;
        for (int k = 0; k < nstate; k++) {
            const double r = P.state[k][0], c = sqrt((double)P.gamma * P.state[k][4] / r);
            double v = 0;
            for (int d = 0; d < 3; d++) v += P.act[d] * (fabs((double)P.state[k][1 + d]) + c) / h[d];
            if (v > m) m = v;
        }
        return m;
    }
    double bytes() override { return 15.0 * N * sizeof(S) + N + (double)nblk * NRED * 8; }
    void set(const double* U) override {
        for (long i = 0; i < 5 * N; i++) buf[i] = (S)U[i];
        CHECK(cudaMemcpy(A, buf.data(), 5 * N * sizeof(S), cudaMemcpyHostToDevice));
    }
    void get(double* U) override {
        CHECK(cudaMemcpy(buf.data(), A, 5 * N * sizeof(S), cudaMemcpyDeviceToHost));
        for (long i = 0; i < 5 * N; i++) U[i] = (double)buf[i];
    }
    void scheme(int recon, int flux, int st, double sensor) override { P.recon = recon; P.flux = flux; stages = st; P.sensor = (T)sensor; }
    int add_state(const double* s) override {
        if (nstate >= MAXSTATE) return -1;
        for (int v = 0; v < 5; v++) P.state[nstate][v] = (T)s[v];
        return nstate++;
    }
    void side(int sd, int type, int state) override { P.side_type[sd] = type; P.side_state[sd] = state; }
    int segment(int sd, const int* lo, const int* hi, int type, int state) override {
        int k = P.nseg[sd];
        if (k >= MAXSEG) return -1;
        for (int i = 0; i < 2; i++) { P.seg_lo[sd][k][i] = lo[i]; P.seg_hi[sd][k][i] = hi[i]; }
        P.seg_type[sd][k] = type; P.seg_state[sd][k] = state;
        return P.nseg[sd]++;
    }
    void reduce(double* r) override {
        k_reduce<S, T><<<nblk, BLOCK>>>(P, A, N, red, P.act[0] / h[0], P.act[1] / h[1], P.act[2] / h[2]);
        CHECK(cudaMemcpy(hred.data(), red, (size_t)nblk * NRED * sizeof(double), cudaMemcpyDeviceToHost));
        for (int k = 0; k < NRED; k++) r[k] = hred[k];
        for (long b = 1; b < nblk; b++) {
            const double* x = &hred[b * NRED];
            for (int k = 0; k < 5; k++) r[k] += x[k];
            if (x[5] > r[5] || x[5] != x[5]) r[5] = x[5];
            for (int k = 6; k < 8; k++) if (x[k] < r[k] || x[k] != x[k]) r[k] = x[k];
        }
    }
    void stage(const S* in, S* out, double a, double b) {
        P.a = (T)a; P.b = (T)b;
        if (P.flux == FX_HYBRID && P.act[0] + P.act[1] + P.act[2] > 1) k_sensor<S, T><<<nblk, BLOCK>>>(P, in, fl, N);
        k_stage<S, T><<<nblk, BLOCK>>>(P, in, A, out, fl, N, sizeof(T) == 4 ? (T)1e-37 : (T)1e-300);
    }
    void step(double dt) override {
        for (int d = 0; d < 3; d++) P.dth[d] = dt / h[d];
        if (stages == 1) { stage(A, B, 0, 1); std::swap(A, B); }
        else if (stages == 2) { stage(A, B, 0, 1); stage(B, C, 0.5, 0.5); std::swap(A, C); }                  // Heun (SSP-RK2)
        else { stage(A, B, 0, 1); stage(B, C, 0.75, 0.25); stage(C, B, 1.0 / 3.0, 2.0 / 3.0); std::swap(A, B); }   // SSP-RK3
        t += dt;
    }
};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

extern "C" {

// precision: 4 = float, 8 = double, 6 = mixed (state stored in double, fluxes computed in float)
void* gcfdc_create(const int* n, const double* h, double gamma, int precision) {
    if (precision == 4) return (Base*)new Solver<float, float>(n, h, gamma);
    if (precision == 6) return (Base*)new Solver<double, float>(n, h, gamma);
    return (Base*)new Solver<double, double>(n, h, gamma);
}
void gcfdc_destroy(void* s) { delete (Base*)s; }
void gcfdc_set(void* s, const double* U) { ((Base*)s)->set(U); }
void gcfdc_get(void* s, double* U) { ((Base*)s)->get(U); }
void gcfdc_scheme(void* s, int recon, int flux, int stages, double sensor) { ((Base*)s)->scheme(recon, flux, stages, sensor); }
int gcfdc_add_state(void* s, const double* st) { return ((Base*)s)->add_state(st); }
void gcfdc_side(void* s, int side, int type, int state) { ((Base*)s)->side(side, type, state); }
int gcfdc_segment(void* s, int side, const int* lo, const int* hi, int type, int state) { return ((Base*)s)->segment(side, lo, hi, type, state); }
// r[8]: sums of rho, mx, my, mz, E over the cells; largest sum over directions of (|u| + c) / h; lowest rho; lowest p
void gcfdc_reduce(void* s, double* r) { ((Base*)s)->reduce(r); }
double gcfdc_bytes(void* s) { return ((Base*)s)->bytes(); }
double gcfdc_time(void* s) { return ((Base*)s)->t; }
// WENO5: differences below sqrt(eps) of the values themselves count as smooth (default 1e-20 double, 1e-10 float)
void gcfdc_weno_eps(void* s, double eps) { ((Base*)s)->weno_eps(eps); }

// nsteps steps of size dt with no copy to the CPU. hist (or NULL): 8 numbers of gcfdc_reduce after every step.
// Returns the milliseconds taken (GPU finished).
double gcfdc_run_fixed(void* sv, double dt, long nsteps, double* hist) {
    Base* s = (Base*)sv;
    CHECK(cudaDeviceSynchronize());
    const double t0 = now_ms();
    for (long k = 0; k < nsteps; k++) {
        s->step(dt);
        if (hist) s->reduce(hist + NRED * k);
    }
    CHECK(cudaDeviceSynchronize());
    return now_ms() - t0;
}

// Advance by tspan with dt = cfl / max(sum over directions of (|u| + c) / h), found again every `every` steps;
// the remaining time is cut into equal steps so that the end is met exactly.
// stats[8]: steps, time advanced, last dt, milliseconds, last max speed over h, lowest rho, lowest p, status
// (0 ok, 1 not a number or no gas left, 2 step limit reached).
void gcfdc_advance(void* sv, double tspan, double cfl, int every, long maxsteps, double* stats) {
    Base* s = (Base*)sv;
    CHECK(cudaDeviceSynchronize());
    const double t0 = now_ms();
    double done = 0, dt = 0, r[NRED] = {0};
    long k = 0;
    int status = 0;
    while (done < tspan * (1 - 1e-13)) {
        if (k >= maxsteps) { status = 2; break; }
        if (k % every == 0) {
            s->reduce(r);
            if (!(r[5] == r[5]) || !(r[6] > 0) || !(r[7] > 0) || !(r[5] < 1e300)) { status = 1; break; }
            const double bs = s->state_speed();
            if (bs > r[5]) r[5] = bs;
            const double left = tspan - done;
            const double nrem = ceil(left * r[5] / cfl - 1e-9);
            dt = left / (nrem < 1 ? 1 : nrem);
        }
        if (done + dt > tspan) dt = tspan - done;
        s->step(dt);
        done += dt;
        k++;
    }
    if (status == 0) {
        s->reduce(r);
        if (!(r[5] == r[5]) || !(r[6] > 0) || !(r[7] > 0)) status = 1;
    }
    CHECK(cudaDeviceSynchronize());
    stats[0] = (double)k; stats[1] = done; stats[2] = dt; stats[3] = now_ms() - t0;
    stats[4] = r[5]; stats[5] = r[6]; stats[6] = r[7]; stats[7] = status;
}

}  // extern "C"
