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
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define NSMAX 12          // species
#define NRMAX 32          // reactions
#define NEMAX 12          // reactions with a collision partner (rows of efficiencies)
#define NSTAGE 7
#define BLOCK 128
#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

enum { KD_THIRD = 1, KD_FALL = 2, KD_TROE = 4, KD_TROE4 = 8 };
enum { ME_BS23 = 0, ME_DP45 = 1, ME_ROS34 = 2 };

template <class T> struct Mech {
    int ns, nr, nact, mode;                      // mode 0: constant pressure, 1: constant volume
    int act[NSMAX];                              // species that some reaction changes
    int reac[NRMAX][3], prod[NRMAX][3];          // species on each side (repeated for a count of 2 or 3), -1: none
    int kind[NRMAX], effrow[NRMAX];
    T R, lnpR;                                   // gas constant; ln(standard pressure / R)
    T Tmid[NSMAX];
    T a[NSMAX][2][7];                            // NASA coefficients, low and high range
    T ah[NSMAX][2][4], as[NSMAX][2][3];          // a1/2, a2/3, a3/4, a4/5 (enthalpy) and a2/2, a3/3, a4/4 (entropy)
    T lnA[NRMAX], beta[NRMAX], Ta[NRMAX], lnA0[NRMAX], beta0[NRMAX], Ta0[NRMAX], troe[NRMAX][4], dnu[NRMAX];
    T eff[NEMAX][NSMAX];
};

template <class T> struct Tab {                  // an explicit Runge-Kutta pair whose last stage is the new state
    int S, q;                                    // stages; the error estimate falls as h^q
    T a[NSTAGE][NSTAGE], b[NSTAGE], e[NSTAGE];
};

template <class T> struct Ctl {
    T rtol, atol;                                // error per sub-step: atol (mol/kg) + rtol |species|
    int nfix, maxsub;                            // nfix > 0: that many equal sub-steps and no error control
};

template <class T> __device__ __forceinline__ T eps_of() { return sizeof(T) == 4 ? (T)1.1920929e-7 : (T)2.220446049250313e-16; }
template <class T> __device__ __forceinline__ T tiny_of() { return sizeof(T) == 4 ? (T)1e-37 : (T)1e-300; }

// Energy over R of the mixture s (mol/kg) at temperature Tk, and its derivative: enthalpy, or internal energy.
template <class T> __device__ __forceinline__ void energy(const Mech<T>& M, const T* s, T ssum, T Tk, T& f, T& d) {
    f = 0; d = 0;
    for (int i = 0; i < M.ns; i++) {
        const int rg = Tk >= M.Tmid[i];
        const T* a = M.a[i][rg];
        const T* ah = M.ah[i][rg];
        f += s[i] * (Tk * (a[0] + Tk * (ah[0] + Tk * (ah[1] + Tk * (ah[2] + Tk * ah[3])))) + a[5]);
        d += s[i] * (a[0] + Tk * (a[1] + Tk * (a[2] + Tk * (a[3] + Tk * a[4]))));
    }
    if (M.mode) { f -= Tk * ssum; d -= ssum; }
}

// The temperature at which the mixture s has the energy ER (energy over R), from the guess Tk.
template <class T> __device__ __forceinline__ T temperature(const Mech<T>& M, const T* s, T ssum, T ER, T Tk) {
    const T tol = 32 * eps_of<T>();
    T f, d, Tp = Tk;
    for (int it = 0; it < 30; it++) {
        energy(M, s, ssum, Tk, f, d);
        const T dT = (ER - f) / d;
        Tp = Tk;
        Tk = fmin(fmax(Tk + dT, (T)50), (T)20000);
        if (fabs(dT) <= tol * Tk) return Tk;
    }
    // Not converged: the two polynomials of the data do not meet exactly where they join, so for a narrow band
    // of energies no temperature exists and Newton's method steps back and forth across the joint. The
    // temperature is then the joint itself, found by halving between the last two tries.
    T lo = fmin(Tp, Tk), hi = fmax(Tp, Tk);
    for (int it = 0; it < 64; it++) {
        const T mid = (lo + hi) / 2;
        energy(M, s, ssum, mid, f, d);
        if (f < ER) lo = mid; else hi = mid;
    }
    return hi;
}

// Rates of the mixture s (mol/kg) with energy ER: r[j] = net rate of reaction j and k[i] = rate of change of
// species i, both per kg of gas (mol/(kg s)). Tk: guess in, temperature out. pv: the pressure (mode 0) or the
// specific volume (mode 1). Optional: forward and reverse rate of each reaction per volume, and the volume.
template <class T> __device__ __forceinline__
void rhs(const Mech<T>& M, const T* s, T ER, T pv, T& Tk, T* r, T* k, T* qfo = nullptr, T* qro = nullptr, T* vo = nullptr) {
    const int ns = M.ns;
    T ssum = 0;
    for (int i = 0; i < ns; i++) ssum += s[i];
    Tk = temperature(M, s, ssum, ER, Tk);
    const T v = M.mode ? pv : ssum * M.R * Tk / pv;
    const T iv = 1 / v, lnT = log(Tk), iT = 1 / Tk;
    T c[NSMAX], g[NSMAX];
    for (int i = 0; i < ns; i++) {
        const int rg = Tk >= M.Tmid[i];
        const T* a = M.a[i][rg];
        const T* ah = M.ah[i][rg];
        const T* as = M.as[i][rg];
        c[i] = s[i] * iv;                                                     // mol/m3
        g[i] = (a[0] + Tk * (ah[0] + Tk * (ah[1] + Tk * (ah[2] + Tk * ah[3]))) + a[5] * iT)
             - (a[0] * lnT + Tk * (a[1] + Tk * (as[0] + Tk * (as[1] + Tk * as[2]))) + a[6]);   // Gibbs energy / (R T)
        k[i] = 0;
    }
    const T lnC = M.lnpR - lnT;                                               // ln(p_std / (R T))
    for (int j = 0; j < M.nr; j++) {
        const T lnkf = M.lnA[j] + M.beta[j] * lnT - M.Ta[j] * iT;
        T dg = 0, cf = 1, cr = 1;
        for (int q = 0; q < 3; q++) {
            const int a = M.reac[j][q], b = M.prod[j][q];
            if (a >= 0) { dg -= g[a]; cf *= c[a]; }
            if (b >= 0) { dg += g[b]; cr *= c[b]; }
        }
        const int kd = M.kind[j];
        T fac = 1;
        if (kd & (KD_THIRD | KD_FALL)) {
            const T* ef = M.eff[M.effrow[j]];
            T Mc = 0;
            for (int i = 0; i < ns; i++) Mc += ef[i] * c[i];
            if (kd & KD_FALL) {                                               // between the low- and high-pressure limits
                const T Pr = fmax(exp(M.lnA0[j] + M.beta0[j] * lnT - M.Ta0[j] * iT - lnkf) * Mc, tiny_of<T>());
                T F = 1;
                if (kd & KD_TROE) {
                    const T* tr = M.troe[j];
                    T Fc = (1 - tr[0]) * exp(-Tk / tr[1]) + tr[0] * exp(-Tk / tr[2]);
                    if (kd & KD_TROE4) Fc += exp(-tr[3] * iT);
                    const T lF = log10(Fc), lP = log10(Pr);
                    const T cc = (T)-0.4 - (T)0.67 * lF, nn = (T)0.75 - (T)1.27 * lF;
                    const T f1 = (lP + cc) / (nn - (T)0.14 * (lP + cc));
                    F = pow((T)10, lF / (1 + f1 * f1));
                }
                fac = Pr / (1 + Pr) * F;
            } else fac = Mc;
        }
        // reverse rate constant = forward / equilibrium constant, ln Kc = -dg + dnu ln(p_std / (R T))
        const T qf = exp(lnkf) * fac * cf, qr = exp(lnkf + dg - M.dnu[j] * lnC) * fac * cr;
        const T net = (qf - qr) * v;
        r[j] = net;
        for (int q = 0; q < 3; q++) {
            const int a = M.reac[j][q], b = M.prod[j][q];
            if (a >= 0) k[a] -= net;
            if (b >= 0) k[b] += net;
        }
        if (qfo) { qfo[j] = qf; qro[j] = qr; }
    }
    if (vo) *vo = v;
}

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

// Sub-step control shared by the integrators: the factor by which the next sub-step may change.
template <class T> __device__ __forceinline__ T step_factor(T err, int q) {
    if (!(err > 0)) return (T)5;
    return fmin((T)5, fmax((T)0.2, (T)0.9 * pow(err, (T)-1 / (T)q)));
}

template <class T> __device__ __forceinline__ void cross(double* tcross, long m, double level, T Told, T Tnew, double t0, double dt) {
    if (level > 0 && tcross[m] < 0 && Told < level && Tnew >= level)          // the reading rule of case 20: a straight line over the step
        tcross[m] = t0 + (level - (double)Told) / ((double)Tnew - (double)Told) * dt;
}

// One step dt of every cell with an explicit Runge-Kutta pair.
template <class S, class T> __global__
void k_erk(const Mech<T>* Mp, const Tab<T>* Tp, Ctl<T> C, S* sig, S* Tc, const S* En, const S* PV, float* hs, double* tcross,
           int* nsub, int* nrej, unsigned char* stat, long N, double t0, double dtd, double level, S* hrow) {
    const long m = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (m >= N) return;
    const Mech<T>& M = *Mp;
    const Tab<T>& B = *Tp;
    const int ns = M.ns, nr = M.nr, St = B.S;
    S s0[NSMAX];
    for (int i = 0; i < ns; i++) s0[i] = sig[i * N + m];
    const T ER = (T)En[m], pv = (T)PV[m], Told = (T)Tc[m], dt = (T)dtd;
    T Tk = Told, trem = dt, hctl = hs[m] > 0 ? (T)hs[m] : dt;
    T y0[NSMAX], y[NSMAX], k[NSTAGE][NSMAX], r0[NRMAX], r[NRMAX], xi[NRMAX];
    int na = 0, nj = 0, status = 0, left = C.nfix;
    bool first = true;
    const T hfix = dt / (T)(C.nfix > 0 ? C.nfix : 1);
    while (C.nfix > 0 ? left > 0 : trem > 4 * eps_of<T>() * dt) {
        if (na + nj >= C.maxsub) { status |= 1; break; }
        const bool clipped = hctl >= trem;
        const T h = C.nfix > 0 ? hfix : (clipped ? trem : hctl);
        for (int i = 0; i < ns; i++) y0[i] = (T)s0[i];
        if (first) { rhs(M, y0, ER, pv, Tk, r0, k[0]); first = false; }
        for (int j = 0; j < nr; j++) xi[j] = B.b[0] * r0[j];
        T Ts = Tk;
        for (int st = 1; st < St; st++) {
            for (int i = 0; i < ns; i++) {
                T acc = 0;
                for (int q = 0; q < st; q++) acc += B.a[st][q] * k[q][i];
                y[i] = y0[i] + h * acc;
            }
            rhs(M, y, ER, pv, Ts, r, k[st]);
            for (int j = 0; j < nr; j++) xi[j] += B.b[st] * r[j];
        }
        T err = 0;                                                            // the last stage state y is the new state
        if (C.nfix <= 0) {
            for (int a = 0; a < M.nact; a++) {
                const int i = M.act[a];
                T e = 0;
                for (int st = 0; st < St; st++) e += B.e[st] * k[st][i];
                const T w = h * e / (C.atol + C.rtol * fmax(fabs(y0[i]), fabs(y[i])));
                err += w * w;
            }
            err = sqrt(err / (T)M.nact);
        }
        if (C.nfix > 0 || err <= 1) {                                         // accept (not a number: reject)
            S ds[NSMAX];
            for (int i = 0; i < ns; i++) ds[i] = 0;
            for (int j = 0; j < nr; j++) {                                    // whole reactions, applied in the storage precision
                const S x = (S)xi[j];
                for (int q = 0; q < 3; q++) {
                    const int a = M.reac[j][q], b = M.prod[j][q];
                    if (a >= 0) ds[a] -= x;
                    if (b >= 0) ds[b] += x;
                }
            }
            for (int i = 0; i < ns; i++) s0[i] += (S)h * ds[i];
            Tk = Ts;
            for (int i = 0; i < ns; i++) k[0][i] = k[St - 1][i];
            for (int j = 0; j < nr; j++) r0[j] = r[j];
            trem -= h; left--; na++;
            if (C.nfix <= 0) {
                const T hn = h * step_factor(err, B.q);
                hctl = clipped ? fmax(hctl, hn) : hn;
            }
        } else {
            nj++;
            hctl = h * (err == err ? step_factor(err, B.q) : (T)0.2);
            if (!(hctl > (T)1e-30)) { status |= 2; break; }
        }
    }
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

// LU factors of the n x n matrix A (row-major, rows of length NSMAX) with partial pivoting, and a solve.
template <class T> __device__ __forceinline__ void lu(T (*A)[NSMAX], int* piv, int n) {
    for (int c = 0; c < n; c++) {
        int p = c;
        T big = fabs(A[c][c]);
        for (int i = c + 1; i < n; i++) if (fabs(A[i][c]) > big) { big = fabs(A[i][c]); p = i; }
        piv[c] = p;
        if (p != c) for (int j = 0; j < n; j++) { const T t = A[c][j]; A[c][j] = A[p][j]; A[p][j] = t; }
        const T id = 1 / A[c][c];
        for (int i = c + 1; i < n; i++) {
            const T f = A[i][c] * id;
            A[i][c] = f;
            for (int j = c + 1; j < n; j++) A[i][j] -= f * A[c][j];
        }
    }
}
template <class T> __device__ __forceinline__ void lusolve(T (*A)[NSMAX], const int* piv, int n, T* x) {
    for (int c = 0; c < n; c++) {
        const int p = piv[c];
        if (p != c) { const T t = x[c]; x[c] = x[p]; x[p] = t; }
        for (int i = c + 1; i < n; i++) x[i] -= A[i][c] * x[c];
    }
    for (int c = n - 1; c >= 0; c--) {
        for (int j = c + 1; j < n; j++) x[c] -= A[c][j] * x[j];
        x[c] /= A[c][c];
    }
}

// One step dt of every cell with the Rosenbrock method of order 4 with an estimate of order 3, in the form and
// with the constants of OpenFOAM's Rosenbrock34 (the Kaps-Rentrop method with Shampine's constants; Hairer and
// Wanner, Solving Ordinary Differential Equations II, section IV.7). Implicit, and linear: one Jacobian (by
// finite differences over the species that react) and one LU factorisation per sub-step, four solves.
template <class S, class T> __global__
void k_ros(const Mech<T>* Mp, Ctl<T> C, S* sig, S* Tc, const S* En, const S* PV, float* hs, double* tcross,
           int* nsub, int* nrej, unsigned char* stat, long N, double t0, double dtd, double level, S* hrow) {
    const long m = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (m >= N) return;
    const Mech<T>& M = *Mp;
    const int ns = M.ns, n = M.nact;
    const T a21 = 2, a31 = (T)(48.0 / 25.0), a32 = (T)(6.0 / 25.0), c21 = -8, c31 = (T)(372.0 / 25.0), c32 = (T)(12.0 / 5.0),
            c41 = (T)(-112.0 / 125.0), c42 = (T)(-54.0 / 125.0), c43 = (T)(-2.0 / 5.0),
            b1 = (T)(19.0 / 9.0), b2 = (T)0.5, b3 = (T)(25.0 / 108.0), b4 = (T)(125.0 / 108.0),
            e1 = (T)(34.0 / 108.0), e2 = (T)(7.0 / 36.0), e3 = 0, e4 = (T)(125.0 / 108.0), gam = (T)0.5;
    S s0[NSMAX];
    for (int i = 0; i < ns; i++) s0[i] = sig[i * N + m];
    const T ER = (T)En[m], pv = (T)PV[m], Told = (T)Tc[m], dt = (T)dtd, sq = sqrt(eps_of<T>());
    T Tk = Told, trem = dt, hctl = hs[m] > 0 ? (T)hs[m] : dt;
    T y0[NSMAX], y[NSMAX], f0[NSMAX], f[NSMAX], r[NRMAX], J[NSMAX][NSMAX], A[NSMAX][NSMAX], k1[NSMAX], k2[NSMAX], k3[NSMAX], k4[NSMAX];
    int piv[NSMAX], na = 0, nj = 0, status = 0, left = C.nfix;
    bool fresh = true;                                                        // f0 and J belong to the present state
    const T hfix = dt / (T)(C.nfix > 0 ? C.nfix : 1);
    while (C.nfix > 0 ? left > 0 : trem > 4 * eps_of<T>() * dt) {
        if (na + nj >= C.maxsub) { status |= 1; break; }
        const bool clipped = hctl >= trem;
        const T h = C.nfix > 0 ? hfix : (clipped ? trem : hctl);
        if (fresh) {
            T ssum = 0;
            for (int i = 0; i < ns; i++) { y0[i] = (T)s0[i]; ssum += y0[i]; }
            rhs(M, y0, ER, pv, Tk, r, f0);
            for (int i = 0; i < ns; i++) y[i] = y0[i];
            for (int a = 0; a < n; a++) {
                const int i = M.act[a];
                const T d = sq * fmax(fabs(y0[i]), (T)1e-7 * ssum);
                T Ts = Tk;
                y[i] = y0[i] + d;
                rhs(M, y, ER, pv, Ts, r, f);
                y[i] = y0[i];
                for (int b = 0; b < n; b++) J[b][a] = (f[M.act[b]] - f0[M.act[b]]) / d;
            }
            fresh = false;
        }
        const T ih = 1 / h;
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < n; b++) A[a][b] = -J[a][b];
            A[a][a] += ih / gam;
        }
        lu(A, piv, n);
        T Ts = Tk;
        for (int a = 0; a < n; a++) k1[a] = f0[M.act[a]];
        lusolve(A, piv, n, k1);
        for (int a = 0; a < n; a++) y[M.act[a]] = y0[M.act[a]] + a21 * k1[a];
        rhs(M, y, ER, pv, Ts, r, f);
        for (int a = 0; a < n; a++) k2[a] = f[M.act[a]] + c21 * k1[a] * ih;
        lusolve(A, piv, n, k2);
        for (int a = 0; a < n; a++) y[M.act[a]] = y0[M.act[a]] + a31 * k1[a] + a32 * k2[a];
        rhs(M, y, ER, pv, Ts, r, f);
        for (int a = 0; a < n; a++) k3[a] = f[M.act[a]] + (c31 * k1[a] + c32 * k2[a]) * ih;
        lusolve(A, piv, n, k3);
        for (int a = 0; a < n; a++) k4[a] = f[M.act[a]] + (c41 * k1[a] + c42 * k2[a] + c43 * k3[a]) * ih;
        lusolve(A, piv, n, k4);
        T err = 0;
        for (int a = 0; a < n; a++) {
            const int i = M.act[a];
            const T dy = b1 * k1[a] + b2 * k2[a] + b3 * k3[a] + b4 * k4[a];
            const T e = e1 * k1[a] + e2 * k2[a] + e3 * k3[a] + e4 * k4[a];
            const T w = e / (C.atol + C.rtol * fmax(fabs(y0[i]), fabs(y0[i] + dy)));
            err += w * w;
            k1[a] = dy;                                                       // k1 now holds the change of the species
        }
        err = sqrt(err / (T)n);
        for (int a = 0; a < n; a++) y[M.act[a]] = y0[M.act[a]];
        if (C.nfix > 0 || err <= 1) {
            for (int a = 0; a < n; a++) s0[M.act[a]] += (S)k1[a];
            trem -= h; left--; na++;
            fresh = true;
            if (C.nfix <= 0) {
                const T hn = h * step_factor(err, 4);
                hctl = clipped ? fmax(hctl, hn) : hn;
            }
        } else {
            nj++;
            hctl = h * (err == err ? step_factor(err, 4) : (T)0.2);
            if (!(hctl > (T)1e-30)) { status |= 2; break; }
        }
    }
    T ssum = 0;                                                               // the temperature of the new state
    for (int i = 0; i < ns; i++) { y0[i] = (T)s0[i]; ssum += y0[i]; }
    Tk = temperature(M, y0, ssum, ER, Tk);
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

// ---------------------------------------------------------------------------------------------- host side
static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class T> static void convert(const Mech<double>& D, Mech<T>& M) {
    memset(&M, 0, sizeof(M));
    M.ns = D.ns; M.nr = D.nr; M.nact = D.nact; M.mode = D.mode;
    memcpy(M.act, D.act, sizeof(M.act)); memcpy(M.reac, D.reac, sizeof(M.reac)); memcpy(M.prod, D.prod, sizeof(M.prod));
    memcpy(M.kind, D.kind, sizeof(M.kind)); memcpy(M.effrow, D.effrow, sizeof(M.effrow));
    M.R = (T)D.R; M.lnpR = (T)D.lnpR;
    for (int i = 0; i < NSMAX; i++) {
        M.Tmid[i] = (T)D.Tmid[i];
        for (int g = 0; g < 2; g++) {
            for (int q = 0; q < 7; q++) M.a[i][g][q] = (T)D.a[i][g][q];
            for (int q = 0; q < 4; q++) M.ah[i][g][q] = (T)D.ah[i][g][q];
            for (int q = 0; q < 3; q++) M.as[i][g][q] = (T)D.as[i][g][q];
        }
    }
    for (int j = 0; j < NRMAX; j++) {
        M.lnA[j] = (T)D.lnA[j]; M.beta[j] = (T)D.beta[j]; M.Ta[j] = (T)D.Ta[j]; M.dnu[j] = (T)D.dnu[j];
        M.lnA0[j] = (T)D.lnA0[j]; M.beta0[j] = (T)D.beta0[j]; M.Ta0[j] = (T)D.Ta0[j];
        for (int q = 0; q < 4; q++) M.troe[j][q] = (T)D.troe[j][q];
    }
    for (int e = 0; e < NEMAX; e++) for (int i = 0; i < NSMAX; i++) M.eff[e][i] = (T)D.eff[e][i];
}

template <class T> static void tableau(int method, Tab<T>& B) {
    double a[NSTAGE][NSTAGE] = {{0}}, b[NSTAGE] = {0}, e[NSTAGE] = {0};
    memset(&B, 0, sizeof(B));
    if (method == ME_BS23) {             // Bogacki and Shampine 1989: order 3, estimate from an order 2 formula
        B.S = 4; B.q = 3;
        a[1][0] = 1.0 / 2; a[2][1] = 3.0 / 4;
        a[3][0] = 2.0 / 9; a[3][1] = 1.0 / 3; a[3][2] = 4.0 / 9;
        const double lo[4] = {7.0 / 24, 1.0 / 4, 1.0 / 3, 1.0 / 8};
        for (int q = 0; q < 4; q++) { b[q] = a[3][q]; e[q] = b[q] - lo[q]; }
    } else {                             // Dormand and Prince 1980: order 5, estimate from an order 4 formula
        B.S = 7; B.q = 5;
        a[1][0] = 1.0 / 5;
        a[2][0] = 3.0 / 40; a[2][1] = 9.0 / 40;
        a[3][0] = 44.0 / 45; a[3][1] = -56.0 / 15; a[3][2] = 32.0 / 9;
        a[4][0] = 19372.0 / 6561; a[4][1] = -25360.0 / 2187; a[4][2] = 64448.0 / 6561; a[4][3] = -212.0 / 729;
        a[5][0] = 9017.0 / 3168; a[5][1] = -355.0 / 33; a[5][2] = 46732.0 / 5247; a[5][3] = 49.0 / 176; a[5][4] = -5103.0 / 18656;
        a[6][0] = 35.0 / 384; a[6][2] = 500.0 / 1113; a[6][3] = 125.0 / 192; a[6][4] = -2187.0 / 6784; a[6][5] = 11.0 / 84;
        const double lo[7] = {5179.0 / 57600, 0, 7571.0 / 16695, 393.0 / 640, -92097.0 / 339200, 187.0 / 2100, 1.0 / 40};
        for (int q = 0; q < 7; q++) { b[q] = a[6][q]; e[q] = b[q] - lo[q]; }
    }
    for (int s = 0; s < NSTAGE; s++) {
        B.b[s] = (T)b[s]; B.e[s] = (T)e[s];
        for (int q = 0; q < NSTAGE; q++) B.a[s][q] = (T)a[s][q];
    }
}

struct Base {
    virtual ~Base() {}
    virtual void set(const double* T, const double* pv, const double* sigma) = 0;
    virtual void get(double* T, double* energy, double* sigma, double* h) = 0;
    virtual void control(int method, double rtol, double atol, int nfix, int maxsub, double level) = 0;
    virtual double run(double dt, long nsteps, double* hist) = 0;
    virtual void stats(int* nsub, int* nrej, int* status, double* tcross) = 0;
    virtual void rates(double* out) = 0;
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
        control(ME_BS23, 1e-6, 1e-12, 0, 100000, 0);
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
    void control(int me, double rtol, double atol, int nfix, int maxsub, double lev) override {
        method = me; level = lev;
        C.rtol = (T)rtol; C.atol = (T)atol; C.nfix = nfix; C.maxsub = maxsub;
        Tab<T> B;
        tableau(me == ME_BS23 ? ME_BS23 : ME_DP45, B);
        CHECK(cudaMemcpy(dTab, &B, sizeof(B), cudaMemcpyHostToDevice));
    }
    void step(double dt, S* hrow) {
        if (method == ME_ROS34) k_ros<S, T><<<grid(), BLOCK>>>(dM, C, sig, Tc, En, PV, hs, tcross, nsub, nrej, stat, N, t, dt, level, hrow);
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
};

extern "C" {

// The reaction scheme as two arrays (packed by gpucfd/chemistry.py from a cfdlib.chem.Mechanism):
//   im: ns, nr, nact, neff, act[nact], then per reaction reac[3], prod[3], kind, effrow
//   dm: R, p_std, then per species Tmid, low[7], high[7]; per reaction lnA, beta, Ta, lnA0, beta0, Ta0, troe[4],
//       sum of dnu; then neff rows of ns efficiencies
// precision: 4 = float, 8 = double, 6 = mixed (state in double, rates in float). mode: 0 constant pressure,
// 1 constant volume. Returns NULL if the scheme is larger than the compiled limits.
void* gcfdk_create(long N, int precision, int mode, const int* im, const double* dm) {
    static Mech<double> D;
    memset(&D, 0, sizeof(D));
    D.ns = im[0]; D.nr = im[1]; D.nact = im[2]; D.mode = mode;
    const int neff = im[3];
    if (D.ns > NSMAX || D.nr > NRMAX || neff > NEMAX) {
        fprintf(stderr, "gcfdk: scheme too large (%d species, %d reactions, %d with collision partners; limits %d, %d, %d)\n",
                D.ns, D.nr, neff, NSMAX, NRMAX, NEMAX);
        return nullptr;
    }
    const int* ip = im + 4;
    for (int a = 0; a < D.nact; a++) D.act[a] = *ip++;
    for (int j = 0; j < D.nr; j++) {
        for (int q = 0; q < 3; q++) D.reac[j][q] = *ip++;
        for (int q = 0; q < 3; q++) D.prod[j][q] = *ip++;
        D.kind[j] = *ip++; D.effrow[j] = *ip++;
    }
    const double* dp = dm;
    D.R = *dp++;
    D.lnpR = log(*dp++ / D.R);
    for (int i = 0; i < D.ns; i++) {
        D.Tmid[i] = *dp++;
        for (int g = 0; g < 2; g++) {
            double* a = D.a[i][g];
            for (int q = 0; q < 7; q++) a[q] = *dp++;
            D.ah[i][g][0] = a[1] / 2; D.ah[i][g][1] = a[2] / 3; D.ah[i][g][2] = a[3] / 4; D.ah[i][g][3] = a[4] / 5;
            D.as[i][g][0] = a[2] / 2; D.as[i][g][1] = a[3] / 3; D.as[i][g][2] = a[4] / 4;
        }
    }
    for (int j = 0; j < D.nr; j++) {
        D.lnA[j] = *dp++; D.beta[j] = *dp++; D.Ta[j] = *dp++; D.lnA0[j] = *dp++; D.beta0[j] = *dp++; D.Ta0[j] = *dp++;
        for (int q = 0; q < 4; q++) D.troe[j][q] = *dp++;
        D.dnu[j] = *dp++;
    }
    for (int e = 0; e < neff; e++) for (int i = 0; i < D.ns; i++) D.eff[e][i] = *dp++;
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
// method: 0 Bogacki-Shampine 3(2), 1 Dormand-Prince 5(4), 2 Rosenbrock 4(3). nfix > 0: that many equal sub-steps
// per step, no error control. level > 0: record when each cell's temperature first reaches it.
void gcfdk_control(void* s, int method, double rtol, double atol, int nfix, int maxsub, double level) {
    ((Base*)s)->control(method, rtol, atol, nfix, maxsub, level);
}
// nsteps steps of size dt. hist (or NULL): after every step the temperature and the species of every cell,
// (1 + ns) N numbers per step. Returns the milliseconds taken (GPU finished).
double gcfdk_advance(void* s, double dt, long nsteps, double* hist) { return ((Base*)s)->run(dt, nsteps, hist); }
// per cell: accepted and rejected sub-steps since gcfdk_set, status (0 ok, 1 sub-step limit reached, 2 broke
// down), and the time of the first crossing of the level (-1: not yet)
void gcfdk_stats(void* s, int* nsub, int* nrej, int* status, double* tcross) { ((Base*)s)->stats(nsub, nrej, status, tcross); }
// per cell 2 nr + 3 ns + 2 numbers: see k_rates
void gcfdk_rates(void* s, double* out) { ((Base*)s)->rates(out); }
double gcfdk_bytes(void* s) { return ((Base*)s)->bytes(); }
double gcfdk_time(void* s) { return ((Base*)s)->t; }

}  // extern "C"
