// gpucfd, chemistry core: what the chemistry kernel (gcfdk.cu) and the reacting flow solver (gcfdr.cu) share.
// The reaction scheme as arrays, the thermodynamic data, the temperature search, the reaction rates, the
// matrix of their derivatives written out exactly, and the integrators as functions of ONE cell. Each of the
// two files wraps them in kernels over its own arrays, so there is one copy of every formula.
// Design and verification: gpucfd/CHEMISTRY.md (the kernel) and gpucfd/REACTING.md (the joined solver).
// Units: mol, m, s, K, J. S is the type a cell's state is stored in, T the type the rates are computed in.
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NSMAX 12          // species
#define NRMAX 32          // reactions
#define NEMAX 12          // reactions with a collision partner (rows of efficiencies)
#define NSTAGE 7

enum { KD_THIRD = 1, KD_FALL = 2, KD_TROE = 4, KD_TROE4 = 8 };
// ME_ROS34: the matrix by differences of the rates; ME_ROS34A: the matrix written out exactly (rhs_jac)
enum { ME_BS23 = 0, ME_DP45 = 1, ME_ROS34 = 2, ME_ROS34A = 3 };

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
    T W[NSMAX], iW[NSMAX];                       // molar mass (kg/mol) and one over it (the flow solver's species are kg/m3)
};

template <class T> struct Tab {                  // an explicit Runge-Kutta pair whose last stage is the new state
    int S, q;                                    // stages; the error estimate falls as h^q
    T a[NSTAGE][NSTAGE], b[NSTAGE], e[NSTAGE];
};

template <class T> struct Ctl {
    T rtol, atol;                                // error per sub-step: atol (mol/kg) + rtol |species|
    int nfix, maxsub;                            // nfix > 0: that many equal sub-steps and no error control
    T quiet;                                     // > 0: a cell whose species would change by less than `quiet` times
                                                 // that error over the whole step takes one plain step (see quiet_cell)
};

template <class T> __device__ __forceinline__ T eps_of() { return sizeof(T) == 4 ? (T)1.1920929e-7 : (T)2.220446049250313e-16; }
template <class T> __device__ __forceinline__ T tiny_of() { return sizeof(T) == 4 ? (T)1e-37 : (T)1e-300; }

// Energy over R of the mixture s (mol/kg) at temperature Tk, and its derivative: enthalpy, or internal energy.
// (s may also be given per volume, mol/m3: then the energy is per volume too.)
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

// The rates as rhs() gives them, and G[j][b] = d r[j] / d s[act[b]]: how the rate of each reaction changes with
// each reacting species while the cell's energy and its volume (or pressure) are held, so with the temperature
// following the species. Written out exactly, term by term; nothing is found by differences.
//   r_j = v q_j(c, T), c = s / v. With dT_b = dT/ds_b = -(energy of species b) / (heat capacity of the mixture)
//   and dv_b = dv/ds_b (zero at constant volume):
//   G_jb = dq_j/dc_b + dv_b (q_j - sum_m c_m dq_j/dc_m) + v dq_j/dT dT_b.
template <class T> __device__ __forceinline__
void rhs_jac(const Mech<T>& M, const T* s, T ER, T pv, T& Tk, T* r, T* k, T (*G)[NSMAX]) {
    const int ns = M.ns, n = M.nact;
    T ssum = 0;
    for (int i = 0; i < ns; i++) ssum += s[i];
    Tk = temperature(M, s, ssum, ER, Tk);
    const T v = M.mode ? pv : ssum * M.R * Tk / pv;
    const T iv = 1 / v, lnT = log(Tk), iT = 1 / Tk;
    T c[NSMAX], g[NSMAX], hR[NSMAX], dTs[NSMAX], dvs[NSMAX];
    T dE = 0;                                                                 // heat capacity of the mixture over R
    for (int i = 0; i < ns; i++) {
        const int rg = Tk >= M.Tmid[i];
        const T* a = M.a[i][rg];
        const T* ah = M.ah[i][rg];
        const T* as = M.as[i][rg];
        c[i] = s[i] * iv;
        hR[i] = Tk * (a[0] + Tk * (ah[0] + Tk * (ah[1] + Tk * (ah[2] + Tk * ah[3])))) + a[5];   // enthalpy / R
        g[i] = (a[0] + Tk * (ah[0] + Tk * (ah[1] + Tk * (ah[2] + Tk * ah[3]))) + a[5] * iT)
             - (a[0] * lnT + Tk * (a[1] + Tk * (as[0] + Tk * (as[1] + Tk * as[2]))) + a[6]);
        dE += s[i] * (a[0] + Tk * (a[1] + Tk * (a[2] + Tk * (a[3] + Tk * a[4]))));
        k[i] = 0;
    }
    if (M.mode) dE -= ssum;
    for (int i = 0; i < ns; i++) {
        dTs[i] = -(hR[i] - (M.mode ? Tk : (T)0)) / dE;
        dvs[i] = M.mode ? (T)0 : v / ssum + v * iT * dTs[i];
    }
    const T lnC = M.lnpR - lnT, ln10 = (T)2.302585092994046;
    for (int j = 0; j < M.nr; j++) {
        const T lnkf = M.lnA[j] + M.beta[j] * lnT - M.Ta[j] * iT;
        const T dkf = (M.beta[j] + M.Ta[j] * iT) * iT;                        // d ln kf / dT
        T dg = 0, dh = 0, cf = 1, cr = 1;
        int nf = 0, nb = 0;
        for (int q = 0; q < 3; q++) {
            const int a = M.reac[j][q], b = M.prod[j][q];
            if (a >= 0) { dg -= g[a]; dh -= hR[a]; cf *= c[a]; nf++; }
            if (b >= 0) { dg += g[b]; dh += hR[b]; cr *= c[b]; nb++; }
        }
        const int kd = M.kind[j];
        const T* ef = nullptr;
        T fac = 1, Mc = 0, dfT = 0, dfM = 0;                                  // d ln fac / dT; d fac / d Mc
        if (kd & (KD_THIRD | KD_FALL)) {
            ef = M.eff[M.effrow[j]];
            for (int i = 0; i < ns; i++) Mc += ef[i] * c[i];
            if (kd & KD_FALL) {
                const T Pr = fmax(exp(M.lnA0[j] + M.beta0[j] * lnT - M.Ta0[j] * iT - lnkf) * Mc, tiny_of<T>());
                T F = 1, dLP = 0, dFT = 0;                                    // d log10 F / d log10 Pr; d ln F / dT at fixed Pr
                if (kd & KD_TROE) {
                    const T* tr = M.troe[j];
                    const T e1 = (1 - tr[0]) * exp(-Tk / tr[1]), e2 = tr[0] * exp(-Tk / tr[2]);
                    T Fc = e1 + e2, dFc = -e1 / tr[1] - e2 / tr[2];
                    if (kd & KD_TROE4) { const T e3 = exp(-tr[3] * iT); Fc += e3; dFc += e3 * tr[3] * iT * iT; }
                    const T lF = log10(Fc), lP = log10(Pr);
                    const T cc = (T)-0.4 - (T)0.67 * lF, nn = (T)0.75 - (T)1.27 * lF;
                    const T x = lP + cc, den = nn - (T)0.14 * x, f1 = x / den, w = 1 / (1 + f1 * f1);
                    F = pow((T)10, lF * w);
                    const T s2 = lF * 2 * f1 * w * w;                         // -d(lF w)/d f1
                    dLP = -s2 * nn / (den * den);
                    const T dLF = w - s2 * ((T)-0.67 * nn + (T)1.27 * x) / (den * den);
                    dFT = ln10 * dLF * dFc / (Fc * ln10);
                }
                fac = Pr / (1 + Pr) * F;
                const T Aa = 1 / (1 + Pr) + dLP;                              // d ln fac / d ln Pr
                dfT = Aa * ((M.beta0[j] + M.Ta0[j] * iT) * iT - dkf) + dFT;
                dfM = Mc > 0 ? fac * Aa / Mc : (T)0;
            } else { fac = Mc; dfM = 1; }
        }
        const T kf = exp(lnkf), kr = exp(lnkf + dg - M.dnu[j] * lnC);
        const T qf = kf * fac * cf, qr = kr * fac * cr;
        const T net = (qf - qr) * v;
        r[j] = net;
        T dq[NSMAX];
        for (int i = 0; i < ns; i++) dq[i] = 0;
        for (int q = 0; q < 3; q++) {                                         // each molecule of each side in turn
            const int a = M.reac[j][q], b = M.prod[j][q];
            if (a >= 0) {
                k[a] -= net;
                T pr = kf * fac;
                for (int p = 0; p < 3; p++) if (p != q && M.reac[j][p] >= 0) pr *= c[M.reac[j][p]];
                dq[a] += pr;
            }
            if (b >= 0) {
                k[b] += net;
                T pr = kr * fac;
                for (int p = 0; p < 3; p++) if (p != q && M.prod[j][p] >= 0) pr *= c[M.prod[j][p]];
                dq[b] -= pr;
            }
        }
        const T base = kf * cf - kr * cr;                                     // the rate without its collision partner
        T Ej = nf * qf - nb * qr;                                             // sum over species of c dq/dc
        if (ef) {
            for (int i = 0; i < ns; i++) dq[i] += base * dfM * ef[i];
            Ej += base * dfM * Mc;
        }
        const T dkr = dkf - (dh * iT * iT - M.dnu[j] * iT);                   // d ln kr / dT
        const T dqT = qf * (dkf + dfT) - qr * (dkr + dfT);
        const T qE = (qf - qr) - Ej;
        for (int b = 0; b < n; b++) {
            const int i = M.act[b];
            G[j][b] = dq[i] + dvs[i] * qE + v * dqT * dTs[i];
        }
    }
}

// Sub-step control shared by the integrators: the factor by which the next sub-step may change.
template <class T> __device__ __forceinline__ T step_factor(T err, int q) {
    if (!(err > 0)) return (T)5;
    return fmin((T)5, fmax((T)0.2, (T)0.9 * pow(err, (T)-1 / (T)q)));
}

// A quiet cell: over the whole step dt its species would change, at their present rates k, by less than `quiet`
// times the error the integrator allows itself. Cold gas ahead of a flame is quiet (its rates are zero in
// practice). Such a cell takes one plain step (all of its error is below the allowed error) and no more rates.
template <class T> __device__ __forceinline__ bool quiet_cell(const Mech<T>& M, const Ctl<T>& C, const T* y0, const T* k, T dt) {
    T err = 0;
    for (int a = 0; a < M.nact; a++) {
        const int i = M.act[a];
        const T w = dt * k[i] / (C.atol + C.rtol * fabs(y0[i]));
        err += w * w;
    }
    return sqrt(err / (T)M.nact) <= C.quiet;                                  // not a number: not quiet
}

// Whole reactions applied to the species in the storage precision: s += h sum_j xi_j (products - reactants).
template <class S, class T> __device__ __forceinline__ void apply_reactions(const Mech<T>& M, S* s0, const T* xi, T h) {
    S ds[NSMAX];
    for (int i = 0; i < M.ns; i++) ds[i] = 0;
    for (int j = 0; j < M.nr; j++) {
        const S x = (S)xi[j];
        for (int q = 0; q < 3; q++) {
            const int a = M.reac[j][q], b = M.prod[j][q];
            if (a >= 0) ds[a] -= x;
            if (b >= 0) ds[b] += x;
        }
    }
    for (int i = 0; i < M.ns; i++) s0[i] += (S)h * ds[i];
}

// One step dt of ONE cell with an explicit Runge-Kutta pair. s0: the species (mol/kg), in and out. ER: the
// cell's energy over R; pv: pressure or specific volume. Tk: guess in, temperature out. hctl: the cell's
// sub-step size, in and out. na, nj: accepted and rejected sub-steps are added. Returns the status bits
// (1: sub-step limit reached, 2: broke down).
template <class S, class T> __device__ __forceinline__
int erk_cell(const Mech<T>& M, const Tab<T>& B, const Ctl<T>& C, S* s0, T ER, T pv, T& Tk, T dt, T& hctl, int& na, int& nj) {
    const int ns = M.ns, nr = M.nr, St = B.S;
    T trem = dt;
    T y0[NSMAX], y[NSMAX], k[NSTAGE][NSMAX], r0[NRMAX], r[NRMAX], xi[NRMAX];
    int status = 0, left = C.nfix, done = 0;
    bool first = true;
    const T hfix = dt / (T)(C.nfix > 0 ? C.nfix : 1);
    while (C.nfix > 0 ? left > 0 : trem > 4 * eps_of<T>() * dt) {
        if (done >= C.maxsub) { status |= 1; break; }
        const bool clipped = hctl >= trem;
        const T h = C.nfix > 0 ? hfix : (clipped ? trem : hctl);
        for (int i = 0; i < ns; i++) y0[i] = (T)s0[i];
        if (first) {
            rhs(M, y0, ER, pv, Tk, r0, k[0]);
            first = false;
            if (C.quiet > 0 && C.nfix <= 0 && quiet_cell(M, C, y0, k[0], dt)) { apply_reactions(M, s0, r0, dt); return status; }
        }
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
        done++;
        if (C.nfix > 0 || err <= 1) {                                         // accept (not a number: reject)
            apply_reactions(M, s0, xi, h);                                    // whole reactions, in the storage precision
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
    return status;
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

// x += g (r + G k): the progress of every reaction that belongs to one stage of the implicit method
template <class T> __device__ __forceinline__ void stage_progress(const Mech<T>& M, T (*G)[NSMAX], const T* r, const T* k, T g, T* x) {
    for (int j = 0; j < M.nr; j++) {
        T acc = r[j];
        for (int b = 0; b < M.nact; b++) acc += G[j][b] * k[b];
        x[j] += g * acc;
    }
}

// One step dt of ONE cell with the Rosenbrock method of order 4 with an estimate of order 3, in the form and
// with the constants of OpenFOAM's Rosenbrock34 (the Kaps-Rentrop method with Shampine's constants; Hairer and
// Wanner, Solving Ordinary Differential Equations II, section IV.7). Implicit, and linear: one matrix and one LU
// factorisation per sub-step, four solves. Arguments as erk_cell.
// exact = false: the matrix by differences of the rates (one rate evaluation per reacting species); the species
//   are changed by the solution of the linear systems.
// exact = true: the matrix from rhs_jac. Every stage k_s solves (1/(gam h) - J) k_s = (rates) + ..., and J k is a
//   sum of whole reactions, so k_s is a sum of whole reactions too: their progress x_s is rebuilt from k_s
//   (stage_progress) and the species are changed by whole reactions in the storage precision. Atoms are then
//   kept to round-off of the storage precision whatever the error of the solves (which matters in mixed precision).
template <class S, class T> __device__ __forceinline__
int ros_cell(const Mech<T>& M, const Ctl<T>& C, bool exact, S* s0, T ER, T pv, T& Tk, T dt, T& hctl, int& na, int& nj) {
    const int ns = M.ns, nr = M.nr, n = M.nact;
    const T a21 = 2, a31 = (T)(48.0 / 25.0), a32 = (T)(6.0 / 25.0), c21 = -8, c31 = (T)(372.0 / 25.0), c32 = (T)(12.0 / 5.0),
            c41 = (T)(-112.0 / 125.0), c42 = (T)(-54.0 / 125.0), c43 = (T)(-2.0 / 5.0),
            b1 = (T)(19.0 / 9.0), b2 = (T)0.5, b3 = (T)(25.0 / 108.0), b4 = (T)(125.0 / 108.0),
            e1 = (T)(34.0 / 108.0), e2 = (T)(7.0 / 36.0), e3 = 0, e4 = (T)(125.0 / 108.0), gam = (T)0.5;
    const T sq = sqrt(eps_of<T>());
    T trem = dt;
    T y0[NSMAX], y[NSMAX], f0[NSMAX], f[NSMAX], r0[NRMAX], r[NRMAX], J[NSMAX][NSMAX], A[NSMAX][NSMAX], k1[NSMAX], k2[NSMAX], k3[NSMAX], k4[NSMAX];
    T G[NRMAX][NSMAX], x1[NRMAX], x2[NRMAX], x3[NRMAX], x4[NRMAX];
    int piv[NSMAX], inv[NSMAX], status = 0, left = C.nfix, done = 0;
    for (int i = 0; i < ns; i++) inv[i] = -1;
    for (int a = 0; a < n; a++) inv[M.act[a]] = a;
    bool fresh = true, start = true;                                          // fresh: f0 and J belong to the present state
    const T hfix = dt / (T)(C.nfix > 0 ? C.nfix : 1);
    while (C.nfix > 0 ? left > 0 : trem > 4 * eps_of<T>() * dt) {
        if (done >= C.maxsub) { status |= 1; break; }
        const bool clipped = hctl >= trem;
        const T h = C.nfix > 0 ? hfix : (clipped ? trem : hctl);
        if (fresh) {
            T ssum = 0;
            for (int i = 0; i < ns; i++) { y0[i] = (T)s0[i]; ssum += y0[i]; }
            const bool ask = start && C.quiet > 0 && C.nfix <= 0;
            if (!exact || ask) {
                rhs(M, y0, ER, pv, Tk, r0, f0);
                if (ask && quiet_cell(M, C, y0, f0, dt)) { apply_reactions(M, s0, r0, dt); return status; }
            }
            if (exact) {
                rhs_jac(M, y0, ER, pv, Tk, r0, f0, G);
                for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) J[a][b] = 0;
                for (int j = 0; j < nr; j++)
                    for (int q = 0; q < 3; q++) {
                        const int a = M.reac[j][q] >= 0 ? inv[M.reac[j][q]] : -1, b = M.prod[j][q] >= 0 ? inv[M.prod[j][q]] : -1;
                        if (a >= 0) for (int c = 0; c < n; c++) J[a][c] -= G[j][c];
                        if (b >= 0) for (int c = 0; c < n; c++) J[b][c] += G[j][c];
                    }
                for (int i = 0; i < ns; i++) y[i] = y0[i];
            } else {
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
            }
            start = false; fresh = false;
        }
        const T ih = 1 / h, gh = gam * h;
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < n; b++) A[a][b] = -J[a][b];
            A[a][a] += ih / gam;
        }
        lu(A, piv, n);
        T Ts = Tk;
        for (int a = 0; a < n; a++) k1[a] = f0[M.act[a]];
        lusolve(A, piv, n, k1);
        if (exact) { for (int j = 0; j < nr; j++) x1[j] = 0; stage_progress(M, G, r0, k1, gh, x1); }
        for (int a = 0; a < n; a++) y[M.act[a]] = y0[M.act[a]] + a21 * k1[a];
        rhs(M, y, ER, pv, Ts, r, f);
        for (int a = 0; a < n; a++) k2[a] = f[M.act[a]] + c21 * k1[a] * ih;
        lusolve(A, piv, n, k2);
        if (exact) { for (int j = 0; j < nr; j++) x2[j] = gam * c21 * x1[j]; stage_progress(M, G, r, k2, gh, x2); }
        for (int a = 0; a < n; a++) y[M.act[a]] = y0[M.act[a]] + a31 * k1[a] + a32 * k2[a];
        rhs(M, y, ER, pv, Ts, r, f);
        for (int a = 0; a < n; a++) k3[a] = f[M.act[a]] + (c31 * k1[a] + c32 * k2[a]) * ih;
        lusolve(A, piv, n, k3);
        for (int a = 0; a < n; a++) k4[a] = f[M.act[a]] + (c41 * k1[a] + c42 * k2[a] + c43 * k3[a]) * ih;
        lusolve(A, piv, n, k4);
        if (exact) {
            for (int j = 0; j < nr; j++) { x3[j] = gam * (c31 * x1[j] + c32 * x2[j]); x4[j] = 0; }
            stage_progress(M, G, r, k3, gh, x3);
            for (int j = 0; j < nr; j++) x4[j] = gam * (c41 * x1[j] + c42 * x2[j] + c43 * x3[j]);
            stage_progress(M, G, r, k4, gh, x4);
        }
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
        done++;
        if (C.nfix > 0 || err <= 1) {
            if (exact) {
                for (int j = 0; j < nr; j++) x1[j] = b1 * x1[j] + b2 * x2[j] + b3 * x3[j] + b4 * x4[j];
                apply_reactions(M, s0, x1, (T)1);
            } else for (int a = 0; a < n; a++) s0[M.act[a]] += (S)k1[a];
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
    return status;
}

// ---------------------------------------------------------------------------------------------- host side
template <class T> static void convert(const Mech<double>& D, Mech<T>& M) {
    memset(&M, 0, sizeof(M));
    M.ns = D.ns; M.nr = D.nr; M.nact = D.nact; M.mode = D.mode;
    memcpy(M.act, D.act, sizeof(M.act)); memcpy(M.reac, D.reac, sizeof(M.reac)); memcpy(M.prod, D.prod, sizeof(M.prod));
    memcpy(M.kind, D.kind, sizeof(M.kind)); memcpy(M.effrow, D.effrow, sizeof(M.effrow));
    M.R = (T)D.R; M.lnpR = (T)D.lnpR;
    for (int i = 0; i < NSMAX; i++) {
        M.Tmid[i] = (T)D.Tmid[i]; M.W[i] = (T)D.W[i]; M.iW[i] = (T)D.iW[i];
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

// The reaction scheme from the two arrays packed by gpucfd/chemistry.py (pack) from a cfdlib.chem.Mechanism:
//   im: ns, nr, nact, neff, act[nact], then per reaction reac[3], prod[3], kind, effrow
//   dm: R, p_std, then per species Tmid, low[7], high[7]; per reaction lnA, beta, Ta, lnA0, beta0, Ta0, troe[4],
//       sum of dnu; then neff rows of ns efficiencies; then the molar mass of each species (kg/mol)
// Returns false if the scheme is larger than the compiled limits.
static bool unpack_mech(const int* im, const double* dm, int mode, Mech<double>& D) {
    memset(&D, 0, sizeof(D));
    D.ns = im[0]; D.nr = im[1]; D.nact = im[2]; D.mode = mode;
    const int neff = im[3];
    if (D.ns > NSMAX || D.nr > NRMAX || neff > NEMAX) {
        fprintf(stderr, "gpucfd chemistry: scheme too large (%d species, %d reactions, %d with collision partners; limits %d, %d, %d)\n",
                D.ns, D.nr, neff, NSMAX, NRMAX, NEMAX);
        return false;
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
    for (int i = 0; i < D.ns; i++) { D.W[i] = *dp++; D.iW[i] = 1.0 / D.W[i]; }
    return true;
}
