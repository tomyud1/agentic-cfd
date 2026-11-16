// gpucfd, chemistry: the reactions of many independent cells, advanced on the GPU.
// Design and verification: gpucfd/CHEMISTRY.md.  Build: bash gpucfd/build.sh chemistry  (build/libgcfdk.so).
//
// One GPU thread per cell. A cell holds its species (moles per kg of gas), and its enthalpy (constant
// pressure) or internal energy (constant volume), chemical energy included. Chemistry does not change that
// energy, so the temperature is never integrated: it is found from the energy by Newton's method whenever
// rates are needed, and the energy is kept exactly.
// Over a step dt each cell takes its own sub-steps with its own error control (explicit Runge-Kutta pairs or
// a Rosenbrock method) and remembers its sub-step size for the next call.
// Species change only by whole reactions: the progress of each reaction over a sub-step is added up and then
// applied to the species, so atoms are kept to round-off whatever the error of the rates.
// The reaction scheme (CHEMKIN form: Arrhenius rates, third bodies, fall-off with or without Troe's form,
// reverse rates from the equilibrium constants, NASA polynomials) arrives as arrays from the wrapper, which
// takes them from cfdlib.chem's reader. No rate constant is written in this file.
// Units: mol, m, s, K, J. Three precisions: S is the type the cell's state is stored in, T the type the rates
// are computed in (S = double with T = float is "mixed").
#include "gcfdk_core.h"
#include <chrono>
#include <vector>

#define BLOCK 128
#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// The reaction scheme, the thermodynamics, the rates and the integrators of one cell are in gcfdk_core.h, shared
// with the reacting flow solver (gcfdr.cu). This file holds the kernels over this library's own arrays.

// The stored energy (over R) of every cell from its temperature and species, in the storage precision.
template <class S> __global__ void k_init(const Mech<S>* Mp, const S* sig, const S* Tc, S* En, long N) {
    const long m = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (m >= N) return;
    const Mech<S>& M = *Mp;
    S s[NSMAX], ssum = 0, f, d;
    for (int i = 0; i < M.ns; i++) { s[i] = sig[i * N + m]; ssum += s[i]; }
    energy(M, s, ssum, Tc[m], f, d);
    En[m] = f;
}

// For the tests: per cell the rates and thermodynamic properties as the kernel computes them, in double.
// out, per cell (nq = 2 nr + 3 ns + 2 numbers): qf[nr], qr[nr] (mol/(m3 s)), d(species)/dt [ns] (mol/(kg s)),
// enthalpy [ns] (J/mol), heat capacity [ns] (J/(mol K)), temperature found from the energy, specific volume.
template <class S, class T> __global__ void k_rates(const Mech<T>* Mp, const S* sig, const S* Tc, const S* En, const S* PV, double* out, long N) {
    const long m = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (m >= N) return;
    const Mech<T>& M = *Mp;
    const int ns = M.ns, nr = M.nr;
    T s[NSMAX], r[NRMAX], k[NSMAX], qf[NRMAX], qr[NRMAX], v;
    for (int i = 0; i < ns; i++) s[i] = (T)sig[i * N + m];
    T Tk = (T)Tc[m] * (T)1.02;                                                // a guess off by 2 %: Newton must find it back
    rhs(M, s, (T)En[m], (T)PV[m], Tk, r, k, qf, qr, &v);
    const long nq = 2 * nr + 3 * ns + 2;
    double* o = out + m * nq;
    for (int j = 0; j < nr; j++) { o[j] = qf[j]; o[nr + j] = qr[j]; }
    for (int i = 0; i < ns; i++) {
        const int rg = Tk >= M.Tmid[i];
        const T* a = M.a[i][rg];
        const T* ah = M.ah[i][rg];
        o[2 * nr + i] = k[i];
        o[2 * nr + ns + i] = M.R * (Tk * (a[0] + Tk * (ah[0] + Tk * (ah[1] + Tk * (ah[2] + Tk * ah[3])))) + a[5]);
        o[2 * nr + 2 * ns + i] = M.R * (a[0] + Tk * (a[1] + Tk * (a[2] + Tk * (a[3] + Tk * a[4]))));
    }
    o[2 * nr + 3 * ns] = Tk;
    o[2 * nr + 3 * ns + 1] = v;
}

template <class T> __device__ __forceinline__ void cross(double* tcross, long m, double level, T Told, T Tnew, double t0, double dt) {
    if (level > 0 && tcross[m] < 0 && Told < level && Tnew >= level)          // the reading rule of case 20: a straight line over the step
        tcross[m] = t0 + (level - (double)Told) / ((double)Tnew - (double)Told) * dt;
}

// One step dt of every cell with an explicit Runge-Kutta pair (erk_cell).
template <class S, class T> __global__
void k_erk(const Mech<T>* Mp, const Tab<T>* Tp, Ctl<T> C, S* sig, S* Tc, const S* En, const S* PV, float* hs, double* tcross,
           int* nsub, int* nrej, unsigned char* stat, long N, double t0, double dtd, double level, S* hrow) {
    const long m = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (m >= N) return;
    const Mech<T>& M = *Mp;
    const int ns = M.ns;
    S s0[NSMAX];
    for (int i = 0; i < ns; i++) s0[i] = sig[i * N + m];
    const T Told = (T)Tc[m], dt = (T)dtd;
    T Tk = Told, hctl = hs[m] > 0 ? (T)hs[m] : dt;
    int na = 0, nj = 0;
    int status = erk_cell<S, T>(M, *Tp, C, s0, (T)En[m], (T)PV[m], Tk, dt, hctl, na, nj);
    for (int i = 0; i < ns; i++) sig[i * N + m] = s0[i];
    Tc[m] = (S)Tk;
    hs[m] = (float)hctl;
    nsub[m] += na; nrej[m] += nj;
    if (!(Tk == Tk)) status |= 2;
    stat[m] |= (unsigned char)status;
    cross(tcross, m, level, Told, Tk, t0, dtd);
    if (hrow) {                                                               // a row of the history: temperature, then the species
        hrow[m] = (S)Tk;
        for (int i = 0; i < ns; i++) hrow[(1 + i) * N + m] = s0[i];
    }
}

// One step dt of every cell with the Rosenbrock method (ros_cell); exact: its matrix written out, not differenced.
template <class S, class T> __global__
void k_ros(const Mech<T>* Mp, Ctl<T> C, int exact, S* sig, S* Tc, const S* En, const S* PV, float* hs, double* tcross,
           int* nsub, int* nrej, unsigned char* stat, long N, double t0, double dtd, double level, S* hrow) {
    const long m = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (m >= N) return;
    const Mech<T>& M = *Mp;
    const int ns = M.ns;
    S s0[NSMAX];
    for (int i = 0; i < ns; i++) s0[i] = sig[i * N + m];
    const T Told = (T)Tc[m], dt = (T)dtd;
    T Tk = Told, hctl = hs[m] > 0 ? (T)hs[m] : dt;
    int na = 0, nj = 0;
    int status = ros_cell<S, T>(M, C, exact != 0, s0, (T)En[m], (T)PV[m], Tk, dt, hctl, na, nj);
    for (int i = 0; i < ns; i++) sig[i * N + m] = s0[i];
    Tc[m] = (S)Tk;
    hs[m] = (float)hctl;
    nsub[m] += na; nrej[m] += nj;
    if (!(Tk == Tk)) status |= 2;
    stat[m] |= (unsigned char)status;
    cross(tcross, m, level, Told, Tk, t0, dtd);
    if (hrow) {
        hrow[m] = (S)Tk;
        for (int i = 0; i < ns; i++) hrow[(1 + i) * N + m] = s0[i];
    }
}

// For the tests: per cell the matrix of rhs_jac folded onto the species, J[a][b] = d(rate of species act[a]) /
// d(species act[b]) with energy and volume (or pressure) held; nact x nact numbers per cell, row by row.
template <class S, class T> __global__ void k_jac(const Mech<T>* Mp, const S* sig, const S* Tc, const S* En, const S* PV, double* out, long N) {
    const long m = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (m >= N) return;
    const Mech<T>& M = *Mp;
    const int ns = M.ns, n = M.nact;
    T s[NSMAX], r[NRMAX], k[NSMAX], G[NRMAX][NSMAX];
    int inv[NSMAX];
    for (int i = 0; i < ns; i++) { s[i] = (T)sig[i * N + m]; inv[i] = -1; }
    for (int a = 0; a < n; a++) inv[M.act[a]] = a;
    T Tk = (T)Tc[m];
    rhs_jac(M, s, (T)En[m], (T)PV[m], Tk, r, k, G);
    double* o = out + m * (long)n * n;
    for (int q = 0; q < n * n; q++) o[q] = 0;
    for (int j = 0; j < M.nr; j++)
        for (int q = 0; q < 3; q++) {
            const int a = M.reac[j][q] >= 0 ? inv[M.reac[j][q]] : -1, b = M.prod[j][q] >= 0 ? inv[M.prod[j][q]] : -1;
            if (a >= 0) for (int c = 0; c < n; c++) o[a * n + c] -= G[j][c];
            if (b >= 0) for (int c = 0; c < n; c++) o[b * n + c] += G[j][c];
        }
}

// ---------------------------------------------------------------------------------------------- host side
static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Base {
    virtual ~Base() {}
    virtual void set(const double* T, const double* pv, const double* sigma) = 0;
    virtual void get(double* T, double* energy, double* sigma, double* h) = 0;
    virtual void control(int method, double rtol, double atol, int nfix, int maxsub, double level, double quiet) = 0;
    virtual double run(double dt, long nsteps, double* hist) = 0;
    virtual void stats(int* nsub, int* nrej, int* status, double* tcross) = 0;
    virtual void rates(double* out) = 0;
    virtual void jacobian(double* out) = 0;
    virtual double bytes() = 0;
    double t = 0;
    long N = 0;
    int ns = 0, nr = 0;
};

template <class S, class T> struct Chem : Base {
    Mech<double> H;
    Mech<T>* dM; Mech<S>* dMs; Tab<T>* dTab;
    S *sig, *Tc, *En, *PV;
    float* hs; double* tcross; int *nsub, *nrej; unsigned char* stat;
    Ctl<T> C; int method = ME_BS23; double level = 0;
    std::vector<S> buf;

    Chem(long n, const Mech<double>& D) : H(D) {
        N = n; ns = D.ns; nr = D.nr;
        Mech<T> mt; Mech<S> ms;
        convert(D, mt); convert(D, ms);
        CHECK(cudaMalloc(&dM, sizeof(mt))); CHECK(cudaMemcpy(dM, &mt, sizeof(mt), cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&dMs, sizeof(ms))); CHECK(cudaMemcpy(dMs, &ms, sizeof(ms), cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&dTab, sizeof(Tab<T>)));
        CHECK(cudaMalloc(&sig, (size_t)ns * N * sizeof(S)));
        CHECK(cudaMalloc(&Tc, N * sizeof(S))); CHECK(cudaMalloc(&En, N * sizeof(S))); CHECK(cudaMalloc(&PV, N * sizeof(S)));
        CHECK(cudaMalloc(&hs, N * sizeof(float))); CHECK(cudaMalloc(&tcross, N * sizeof(double)));
        CHECK(cudaMalloc(&nsub, N * sizeof(int))); CHECK(cudaMalloc(&nrej, N * sizeof(int))); CHECK(cudaMalloc(&stat, N));
        buf.resize((size_t)ns * N);
        control(ME_BS23, 1e-6, 1e-12, 0, 100000, 0, 0);
    }
    ~Chem() override {
        cudaFree(dM); cudaFree(dMs); cudaFree(dTab); cudaFree(sig); cudaFree(Tc); cudaFree(En); cudaFree(PV);
        cudaFree(hs); cudaFree(tcross); cudaFree(nsub); cudaFree(nrej); cudaFree(stat);
    }
    double bytes() override { return (double)N * ((ns + 3) * sizeof(S) + 4 + 8 + 4 + 4 + 1); }
    int grid() const { return (int)((N + BLOCK - 1) / BLOCK); }
    void up(S* dev, const double* src, size_t n) {
        for (size_t i = 0; i < n; i++) buf[i] = (S)src[i];
        CHECK(cudaMemcpy(dev, buf.data(), n * sizeof(S), cudaMemcpyHostToDevice));
    }
    void down(double* dst, const S* dev, size_t n) {
        CHECK(cudaMemcpy(buf.data(), dev, n * sizeof(S), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; i++) dst[i] = (double)buf[i];
    }
    void set(const double* T_, const double* pv, const double* sigma) override {
        up(Tc, T_, N); up(PV, pv, N); up(sig, sigma, (size_t)ns * N);
        k_init<S><<<grid(), BLOCK>>>(dMs, sig, Tc, En, N);
        CHECK(cudaMemset(hs, 0, N * sizeof(float))); CHECK(cudaMemset(nsub, 0, N * sizeof(int)));
        CHECK(cudaMemset(nrej, 0, N * sizeof(int))); CHECK(cudaMemset(stat, 0, N));
        std::vector<double> neg(N, -1.0);
        CHECK(cudaMemcpy(tcross, neg.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaDeviceSynchronize());
        t = 0;
    }
    void get(double* T_, double* energy, double* sigma, double* h) override {
        down(T_, Tc, N); down(energy, En, N); down(sigma, sig, (size_t)ns * N);
        for (long i = 0; i < N; i++) energy[i] *= H.R;
        std::vector<float> hb(N);
        CHECK(cudaMemcpy(hb.data(), hs, N * sizeof(float), cudaMemcpyDeviceToHost));
        for (long i = 0; i < N; i++) h[i] = hb[i];
    }
    void control(int me, double rtol, double atol, int nfix, int maxsub, double lev, double quiet) override {
        method = me; level = lev;
        C.rtol = (T)rtol; C.atol = (T)atol; C.nfix = nfix; C.maxsub = maxsub; C.quiet = (T)quiet;
        Tab<T> B;
        tableau(me == ME_BS23 ? ME_BS23 : ME_DP45, B);
        CHECK(cudaMemcpy(dTab, &B, sizeof(B), cudaMemcpyHostToDevice));
    }
    void step(double dt, S* hrow) {
        if (method == ME_ROS34 || method == ME_ROS34A) k_ros<S, T><<<grid(), BLOCK>>>(dM, C, method == ME_ROS34A, sig, Tc, En, PV, hs, tcross, nsub, nrej, stat, N, t, dt, level, hrow);
        else k_erk<S, T><<<grid(), BLOCK>>>(dM, dTab, C, sig, Tc, En, PV, hs, tcross, nsub, nrej, stat, N, t, dt, level, hrow);
        t += dt;
    }
    // nsteps steps. hist (or NULL): the temperature and the species of every cell after every step, kept on the
    // GPU during the run and copied once at the end. Returns the milliseconds taken, that copy included.
    double run(double dt, long nsteps, double* hist) override {
        const size_t row = (size_t)(1 + ns) * N;
        S* dh = nullptr;
        if (hist) CHECK(cudaMalloc(&dh, row * nsteps * sizeof(S)));
        CHECK(cudaDeviceSynchronize());
        const double t0 = now_ms();
        for (long k = 0; k < nsteps; k++) step(dt, dh ? dh + k * row : nullptr);
        if (hist) {
            std::vector<S> hb(row * nsteps);
            CHECK(cudaMemcpy(hb.data(), dh, row * nsteps * sizeof(S), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < row * nsteps; i++) hist[i] = (double)hb[i];
            cudaFree(dh);
        }
        CHECK(cudaDeviceSynchronize());
        return now_ms() - t0;
    }
    void stats(int* a, int* b, int* c, double* tc) override {
        CHECK(cudaMemcpy(a, nsub, N * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b, nrej, N * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<unsigned char> sb(N);
        CHECK(cudaMemcpy(sb.data(), stat, N, cudaMemcpyDeviceToHost));
        for (long i = 0; i < N; i++) c[i] = sb[i];
        CHECK(cudaMemcpy(tc, tcross, N * sizeof(double), cudaMemcpyDeviceToHost));
    }
    void rates(double* out) override {
        const long nq = 2 * nr + 3 * ns + 2;
        double* d;
        CHECK(cudaMalloc(&d, (size_t)nq * N * sizeof(double)));
        k_rates<S, T><<<grid(), BLOCK>>>(dM, sig, Tc, En, PV, d, N);
        CHECK(cudaMemcpy(out, d, (size_t)nq * N * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(d);
    }
    void jacobian(double* out) override {
        const size_t nq = (size_t)H.nact * H.nact;
        double* d;
        CHECK(cudaMalloc(&d, nq * N * sizeof(double)));
        k_jac<S, T><<<grid(), BLOCK>>>(dM, sig, Tc, En, PV, d, N);
        CHECK(cudaMemcpy(out, d, nq * N * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(d);
    }
};

extern "C" {

// The reaction scheme as two arrays, packed by gpucfd/chemistry.py from a cfdlib.chem.Mechanism (layout:
// unpack_mech in gcfdk_core.h). precision: 4 = float, 8 = double, 6 = mixed (state in double, rates in float).
// mode: 0 constant pressure, 1 constant volume. Returns NULL if the scheme is larger than the compiled limits.
void* gcfdk_create(long N, int precision, int mode, const int* im, const double* dm) {
    static Mech<double> D;
    if (!unpack_mech(im, dm, mode, D)) return nullptr;
    if (precision == 4) return (Base*)new Chem<float, float>(N, D);
    if (precision == 6) return (Base*)new Chem<double, float>(N, D);
    return (Base*)new Chem<double, double>(N, D);
}
void gcfdk_destroy(void* s) { delete (Base*)s; }
// T[N]; pv[N]: pressure (mode 0) or specific volume (mode 1); sigma[ns N]: mol/kg, species by species.
// Computes each cell's stored energy from them and clears time, sub-step sizes, counters and crossing times.
void gcfdk_set(void* s, const double* T, const double* pv, const double* sigma) { ((Base*)s)->set(T, pv, sigma); }
// energy[N]: J/kg (enthalpy or internal energy, as stored); h[N]: the sub-step size each cell would take next
void gcfdk_get(void* s, double* T, double* energy, double* sigma, double* h) { ((Base*)s)->get(T, energy, sigma, h); }
// method: 0 Bogacki-Shampine 3(2), 1 Dormand-Prince 5(4), 2 Rosenbrock 4(3) with its matrix by differences,
// 3 the same with its matrix written out exactly. nfix > 0: that many equal sub-steps per step, no error control.
// level > 0: record when each cell's temperature first reaches it. quiet > 0: a cell whose species would change
// over the step by less than quiet times the allowed error takes one plain step (quiet_cell in gcfdk_core.h).
void gcfdk_control(void* s, int method, double rtol, double atol, int nfix, int maxsub, double level, double quiet) {
    ((Base*)s)->control(method, rtol, atol, nfix, maxsub, level, quiet);
}
// nsteps steps of size dt. hist (or NULL): after every step the temperature and the species of every cell,
// (1 + ns) N numbers per step. Returns the milliseconds taken (GPU finished).
double gcfdk_advance(void* s, double dt, long nsteps, double* hist) { return ((Base*)s)->run(dt, nsteps, hist); }
// per cell: accepted and rejected sub-steps since gcfdk_set, status (0 ok, 1 sub-step limit reached, 2 broke
// down), and the time of the first crossing of the level (-1: not yet)
void gcfdk_stats(void* s, int* nsub, int* nrej, int* status, double* tcross) { ((Base*)s)->stats(nsub, nrej, status, tcross); }
// per cell 2 nr + 3 ns + 2 numbers: see k_rates
void gcfdk_rates(void* s, double* out) { ((Base*)s)->rates(out); }
// per cell nact x nact numbers: see k_jac
void gcfdk_jacobian(void* s, double* out) { ((Base*)s)->jacobian(out); }
double gcfdk_bytes(void* s) { return ((Base*)s)->bytes(); }
double gcfdk_time(void* s) { return ((Base*)s)->t; }

}  // extern "C"
