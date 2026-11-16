// gpucfd, reacting part: gas with shock waves that burns. The shock solver (gcfdc.cu) joined to the chemistry
// kernel (gcfdk.cu). Design and verification: gpucfd/REACTING.md.  Build: bash gpucfd/build.sh reacting
// (build/libgcfdr.so).
//
// Conservative finite volumes on a uniform Cartesian grid, as in gcfdc.cu. Per cell ns + 4 numbers, each in its
// own array (x fastest): the mass of every species per volume (kg/m3), momentum x, y, z, and the total energy
// per volume, which includes the chemical energy of the species. The density is the sum of the species, so the
// mass fractions add up to one by construction. Next to the state every cell keeps its temperature, pressure
// and speed of sound, found from the state when the state is written.
// Thermodynamics: a mixture of ideal gases whose heat capacities change with temperature (NASA polynomials),
// with the data, the energy formula and the temperature search of the chemistry kernel (gcfdk_core.h).
// Fluxes: both cells of a face compute its flux from the same numbers with the same code (gcfdc.cu). A species
// is carried by the mass flux of the flux formula. The energy of a face state and its speed of sound come from
// the mixture at that face.
// One time step: flow, then chemistry (or the other orders, see step()). Chemistry: every cell burns at
// constant volume with its density and internal energy held; the integrators of gcfdk_core.h work directly on
// this solver's arrays.
// Guard, three parts, every use counted: a cell whose reconstructed face values hold a negative species or no
// pressure uses its own average on its faces (first order in that cell); a species that ends a stage below
// zero is set to zero and the other species scaled so that the cell keeps its mass; a cell that ends a stage
// colder than a floor gets the energy of the floor temperature.
// Three precisions: S is the type the state is stored in, T the type fluxes, temperatures and rates are computed
// in (S = double with T = float is "mixed").
#include "gcfdk_core.h"
#include <chrono>
#include <vector>

#define MAXSEG 4
#define MAXSTATE 8
#define NV (NSMAX + 4)                         // most conserved numbers per cell
#define NEXT 7                                 // numbers of a reduction after the sums (see gcfdr_reduce)
#define NRED (NV + NEXT)
#define BLOCK 256
#define RBLOCK 128

enum { BC_PERIODIC = 0, BC_WALL = 1, BC_INFLOW = 2, BC_OUTFLOW = 3, BC_SUPERSONIC = 4 };
enum { RC_FIRST = 0, RC_MINMOD = 1, RC_VANLEER = 2, RC_MC = 3, RC_CENTRAL = 4, RC_WENO5 = 5 };
enum { FX_RUSANOV = 0, FX_HLL = 1, FX_HLLC = 2, FX_HYBRID = 3 };
enum { SP_FLOW_CHEM = 0, SP_CHEM_FLOW = 1, SP_STRANG = 2 };
enum { G_FIRST = 0, G_SPECIES = 1, G_COLD = 2, G_FLAT = 3, G_N = 4 };

template <class T> struct Par {
    int n[3], act[3], ns;
    T b, sensor, weps, Tfloor;
    double dth[3];                             // dt / h per direction
    int recon, flux;
    int side_type[6], side_state[6], nseg[6];
    int seg_lo[6][MAXSEG][2], seg_hi[6][MAXSEG][2], seg_type[6][MAXSEG], seg_state[6][MAXSEG];
    T state[MAXSTATE][NV];                     // species (kg/m3), u, v, w, p
};

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)

// ---- boundaries: what a stencil sees beyond a side -----------------------------------------------------
// State (species, normal velocity, two tangential velocities, p) of the cell k steps along direction d from
// cell c. m is the cell it was read from (-1: a prescribed state); kind is the boundary type used (-1 inside).
// pr: the cells' temperature, pressure and speed of sound (three arrays).
template <class S, class T> __device__ __forceinline__
void load(const Par<T>& P, const S* q, const S* pr, long N, const int* c, int d, int k, T* w, long& m, int& kind) {
    const int t1 = (d + 1) % 3, t2 = (d + 2) % 3, n = P.n[d], ns = P.ns;
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
            for (int i = 0; i < ns; i++) w[i] = P.state[st][i];
            w[ns] = P.state[st][ns + d]; w[ns + 1] = P.state[st][ns + t1]; w[ns + 2] = P.state[st][ns + t2];
            w[ns + 3] = P.state[st][ns + 3]; m = -1; return;
        }
    }
    int cc[3] = {c[0], c[1], c[2]};
    cc[d] = idx;
    m = cc[0] + (long)P.n[0] * (cc[1] + (long)P.n[1] * cc[2]);
    T r = 0;
    for (int i = 0; i < ns; i++) { w[i] = (T)q[i * N + m]; r += w[i]; }
    const T ir = T(1) / r;
    const T mom[3] = {(T)q[ns * N + m], (T)q[(ns + 1) * N + m], (T)q[(ns + 2) * N + m]};
    w[ns] = sgn * mom[d] * ir; w[ns + 1] = mom[t1] * ir; w[ns + 2] = mom[t2] * ir;
    w[ns + 3] = (T)pr[N + m];
}

// ---- reconstruction: the two face values of a cell from its neighbours ---------------------------------
template <class T> __device__ __forceinline__ T slope(int rc, T dm, T dp) {
    if (rc == RC_CENTRAL) return T(0.5) * (dm + dp);
    if (!(dm * dp > 0)) return 0;
    if (rc == RC_MINMOD) return fabs(dm) < fabs(dp) ? dm : dp;
    if (rc == RC_VANLEER) {
        // never more than twice the smaller difference, also after rounding: a face value then cannot leave
        // the range of the cell and its neighbour (a vanishing species would come out negative)
        const T s = 2 * dm * (dp / (dm + dp)), lim = 2 * (fabs(dm) < fabs(dp) ? fabs(dm) : fabs(dp));
        return fabs(s) > lim ? (dm > 0 ? lim : -lim) : s;
    }
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

// The two face values (lo, hi) of the middle cell c of five in a row.
template <class T> __device__ __forceinline__ void faces(int rc, T a, T b, T c, T d, T e, T& lo, T& hi, T tiny, T weps) {
    if (rc == RC_WENO5) { hi = weno(a, b, c, d, e, tiny, weps); lo = weno(e, d, c, b, a, tiny, weps); }
    else { const T s = T(0.5) * slope(rc, c - b, d - c); lo = c - s; hi = c + s; }
}

// w: the stencil (species densities, velocity, pressure), w[j] the cell itself; rs: the density of each stencil
// cell; y: its mass fractions. wm: values on the cell's low face, wp: on its high face (species densities again).
// What is reconstructed: density, velocity, pressure and the mass fraction of every species. The mass fractions
// at a face are then scaled to add up to one and multiplied by the face's density. So the density does not
// depend on how the gas is divided into species (limiters are not linear: the sum of limited species is not
// the limited sum), and a mass fraction at a face stays between its neighbours' values.
// The guard, returned as bits: 2 = a mass fraction came out negative at a face (WENO5 does that next to a
// front) and was put back to the cell's own value on both faces (first order for that species in this cell);
// 1 = no pressure or no gas at a face, the whole cell falls back to its own average on both faces.
template <class T> __device__ __forceinline__
int recon_cell(int rc, int ns, const T (*w)[NV], const T (*y)[NSMAX], const T* rs, int j, T* wm, T* wp, T tiny, T weps) {
    const int nv = ns + 4;
    if (rc == RC_FIRST) {
        for (int v = 0; v < nv; v++) { wm[v] = w[j][v]; wp[v] = w[j][v]; }
        return 0;
    }
    T rm, rp;
    faces(rc, rs[j - 2], rs[j - 1], rs[j], rs[j + 1], rs[j + 2], rm, rp, tiny, weps);
    for (int v = ns; v < nv; v++) faces(rc, w[j - 2][v], w[j - 1][v], w[j][v], w[j + 1][v], w[j + 2][v], wm[v], wp[v], tiny, weps);
    int used = 0;
    bool ok = wm[ns + 3] > 0 && wp[ns + 3] > 0 && rm > 0 && rp > 0;
    T sm = 0, sp = 0;
    if (ok) {
        for (int i = 0; i < ns; i++) {
            faces(rc, y[j - 2][i], y[j - 1][i], y[j][i], y[j + 1][i], y[j + 2][i], wm[i], wp[i], tiny, weps);
            if (!(wm[i] >= 0 && wp[i] >= 0)) { wm[i] = y[j][i]; wp[i] = y[j][i]; used = 2; }
            sm += wm[i]; sp += wp[i];
        }
        ok = sm > 0 && sp > 0;
    }
    if (!ok) {
        for (int v = 0; v < nv; v++) { wm[v] = w[j][v]; wp[v] = w[j][v]; }
        return 1;
    }
    const T fm = rm / sm, fp = rp / sp;
    for (int i = 0; i < ns; i++) { wm[i] *= fm; wp[i] *= fp; }
    return used;
}

// ---- fluxes -------------------------------------------------------------------------------------------
// From a face state w (species, normal velocity, two tangential, p): the conserved numbers U, their flux F, the
// density, the speed of sound and the ratio of heat capacities of the mixture at that face. The temperature
// follows from the gas law, the energy (chemical energy included) from the thermodynamic data.
template <class T> __device__ __forceinline__
void facestate(const Mech<T>& M, int ns, const T* w, T* U, T* F, T& rho, T& c, T& g) {
    T cc[NSMAX], csum = 0;
    rho = 0;
    for (int i = 0; i < ns; i++) { rho += w[i]; cc[i] = w[i] * M.iW[i]; csum += cc[i]; }     // mol/m3
    const T u = w[ns], p = w[ns + 3];
    T f, d;
    energy(M, cc, csum, p / (M.R * csum), f, d);                 // internal energy / R and heat capacity / R, per volume
    g = (d + csum) / d;
    c = sqrt(g * p / rho);
    const T E = M.R * f + T(0.5) * rho * (u * u + w[ns + 1] * w[ns + 1] + w[ns + 2] * w[ns + 2]);
    for (int i = 0; i < ns; i++) { U[i] = w[i]; F[i] = w[i] * u; }
    U[ns] = rho * u; U[ns + 1] = rho * w[ns + 1]; U[ns + 2] = rho * w[ns + 2]; U[ns + 3] = E;
    F[ns] = rho * u * u + p; F[ns + 1] = rho * u * w[ns + 1]; F[ns + 2] = rho * u * w[ns + 2]; F[ns + 3] = u * (E + p);
}

// Flux through a face from the states on its two sides (normal velocity in w[ns]).
template <class T> __device__ __forceinline__ void riemann(const Mech<T>& M, int ns, int fx, const T* wl, const T* wr, T* F) {
    const int nv = ns + 4;
    T Fl[NV], Fr[NV], Ul[NV], Ur[NV], rl, rr, cl, cr, gl, gr;
    facestate(M, ns, wl, Ul, Fl, rl, cl, gl);
    facestate(M, ns, wr, Ur, Fr, rr, cr, gr);
    const T ul = wl[ns], ur = wr[ns], pl = wl[ns + 3], pr = wr[ns + 3];
    if (fx == FX_RUSANOV) {
        const T sl = fabs(ul) + cl, sr = fabs(ur) + cr, s = sl > sr ? sl : sr;
        for (int v = 0; v < nv; v++) F[v] = T(0.5) * (Fl[v] + Fr[v]) - T(0.5) * s * (Ur[v] - Ul[v]);
        return;
    }
    // fastest waves to the left and right: the two states' own speeds and those of their Roe average (Einfeldt).
    // The average's speed of sound in the form that needs no energy: for one gas with a constant ratio of heat
    // capacities it is the same number as gcfdc.cu's; with two ratios the larger is used.
    const T ql = sqrt(rl), qr = sqrt(rr), iq = T(1) / (ql + qr);
    const T ut = (ql * ul + qr * ur) * iq;
    const T du2 = (ur - ul) * (ur - ul) + (wr[ns + 1] - wl[ns + 1]) * (wr[ns + 1] - wl[ns + 1]) + (wr[ns + 2] - wl[ns + 2]) * (wr[ns + 2] - wl[ns + 2]);
    const T gh = gl > gr ? gl : gr;
    const T ct = sqrt((ql * cl * cl + qr * cr * cr) * iq + T(0.5) * (gh - 1) * ql * qr * iq * iq * du2);
    T SL = ul - cl, SR = ur + cr;
    if (ut - ct < SL) SL = ut - ct;
    if (ut + ct > SR) SR = ut + ct;
    if (SL >= 0) { for (int v = 0; v < nv; v++) F[v] = Fl[v]; return; }
    if (SR <= 0) { for (int v = 0; v < nv; v++) F[v] = Fr[v]; return; }
    if (fx == FX_HLL) {
        const T i = T(1) / (SR - SL);
        for (int v = 0; v < nv; v++) F[v] = (SR * Fl[v] - SL * Fr[v] + SL * SR * (Ur[v] - Ul[v])) * i;
        return;
    }
    // HLLC (Toro): the middle wave (contact) restored; written so that a wall gets exactly zero mass and energy
    // flux (gcfdc.cu). A species crosses with the gas on the side the middle wave comes from.
    const T ml = rl * (SL - ul), mr = rr * (SR - ur);
    const T Ss = (pr - pl + ml * ul - mr * ur) / (ml - mr);
    const T plr = T(0.5) * (pl + pr + ml * (Ss - ul) + mr * (Ss - ur));
    T D[NV];
    for (int v = 0; v < nv; v++) D[v] = 0;
    D[ns] = 1; D[ns + 3] = Ss;
    if (Ss >= 0) { const T i = Ss / (SL - Ss); for (int v = 0; v < nv; v++) F[v] = plr * D[v] + (SL * Ul[v] - Fl[v] + plr * D[v]) * i; }
    else         { const T i = Ss / (SR - Ss); for (int v = 0; v < nv; v++) F[v] = plr * D[v] + (SR * Ur[v] - Fr[v] + plr * D[v]) * i; }
}

// ---- the new state of a cell: the guard, then temperature, pressure and speed of sound, then the write ----
// U: species, momentum, energy (changed by the guard). Tguess: the cell's last temperature.
template <class S, class T> __device__ __forceinline__
void finish(const Par<T>& P, const Mech<T>& M, S* U, T Tguess, long m, long N, S* qo, S* po, int* gcount, double* gsum) {
    const int ns = P.ns, nv = ns + 4;
    S rho = 0, neg = 0;
    for (int i = 0; i < ns; i++) {
        if (U[i] < 0) { neg -= U[i]; U[i] = 0; }
        rho += U[i];
    }
    if (neg > 0) {                                               // species below zero: to zero, the rest scaled, mass kept
        const S keep = (rho - neg) / rho;
        for (int i = 0; i < ns; i++) U[i] *= keep;
        rho -= neg;
        atomicAdd(&gcount[G_SPECIES], 1);
        atomicAdd(&gsum[0], (double)neg);
    }
    const S rhoe = U[ns + 3] - S(0.5) * (U[ns] * U[ns] + U[ns + 1] * U[ns + 1] + U[ns + 2] * U[ns + 2]) / rho;
    T cc[NSMAX], csum = 0, f, d;
    for (int i = 0; i < ns; i++) { cc[i] = (T)U[i] * M.iW[i]; csum += cc[i]; }
    T Tk = temperature(M, cc, csum, (T)rhoe / M.R, Tguess);
    if (!(rhoe == rhoe)) Tk = (T)rhoe;                           // not a number stays not a number, so that it is seen
    else if (!(Tk > P.Tfloor)) {                                 // colder than the floor: the energy of the floor
        Tk = P.Tfloor;
        energy(M, cc, csum, Tk, f, d);
        const S add = (S)(M.R * f) - rhoe;
        U[ns + 3] += add;
        atomicAdd(&gcount[G_COLD], 1);
        atomicAdd(&gsum[1], (double)add);
    }
    energy(M, cc, csum, Tk, f, d);
    const T p = M.R * csum * Tk;
    for (int v = 0; v < nv; v++) qo[v * N + m] = U[v];
    po[m] = (S)Tk; po[N + m] = (S)p; po[2 * N + m] = (S)sqrt((d + csum) / d * p / (T)rho);
}

// ---- one stage of the time scheme ----------------------------------------------------------------------
// out = U0 + b ((in - U0) + dt L(in)); pr, po: temperature, pressure, speed of sound of `in` and of `out`.
template <class S, class T> __global__
void k_stage(Par<T> P, const Mech<T>* Mp, const S* q, const S* pr, const S* q0, S* qo, S* po, const unsigned char* fl, long N, T tiny,
             int* gcount, double* gsum) {
    const long m = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;
    const Mech<T>& M = *Mp;
    const int ns = P.ns, nv = ns + 4;
    const int c[3] = {(int)(m % P.n[0]), (int)((m / P.n[0]) % P.n[1]), (int)(m / ((long)P.n[0] * P.n[1]))};
    const int reach = P.recon == RC_WENO5 ? 3 : 2;
    S R[NV];                                   // the fluxes are T, their differences are summed in S
    for (int v = 0; v < nv; v++) R[v] = 0;
    int nfirst = 0, nflat = 0;
    for (int d = 0; d < 3; d++) {
        if (!P.act[d]) continue;
        const int t1 = (d + 1) % 3, t2 = (d + 2) % 3;
        T w[7][NV];
        long mi[7];
        int kind[7];
        for (int k = 0; k < 7; k++) { mi[k] = -1; kind[k] = -1; for (int v = 0; v < nv; v++) w[k][v] = 1; }
        for (int k = -reach; k <= reach; k++) load(P, q, pr, N, c, d, k, w[k + 3], mi[k + 3], kind[k + 3]);
        T wm[3][NV], wp[3][NV], rs[7], y[7][NSMAX];
        for (int k = 0; k < 7; k++) {
            rs[k] = 0;
            for (int i = 0; i < ns; i++) rs[k] += w[k][i];
            for (int i = 0; i < ns; i++) y[k][i] = w[k][i] / rs[k];
        }
        for (int j = -1; j <= 1; j++) {
            const int used = recon_cell(P.recon, ns, w, y, rs, j + 3, wm[j + 1], wp[j + 1], tiny, P.weps);
            if (j == 0) { nfirst += used & 1; nflat += (used >> 1) & 1; }
        }
        int fa = P.flux, fb = P.flux;                                  // low face, high face
        if (P.flux == FX_HYBRID) {
            // HLLC everywhere, except HLL on a face that lies along a strong shock (a shock found in ANOTHER
            // direction in one of the two cells): that is where HLLC lets a flat shock break up
            const int other = ~(1 << d) & 7;
            const int f0 = fl[m], fm = mi[2] >= 0 ? fl[mi[2]] : 0, fp = mi[4] >= 0 ? fl[mi[4]] : 0;
            fa = ((f0 | fm) & other) ? FX_HLL : FX_HLLC;
            fb = ((f0 | fp) & other) ? FX_HLL : FX_HLLC;
        }
        T Fa[NV], Fb[NV], U[NV], rr, cs, gg;
        if (kind[2] == BC_SUPERSONIC) facestate(M, ns, w[2], U, Fa, rr, cs, gg); else riemann(M, ns, fa, wp[0], wm[1], Fa);
        if (kind[4] == BC_SUPERSONIC) facestate(M, ns, w[4], U, Fb, rr, cs, gg); else riemann(M, ns, fb, wp[1], wm[2], Fb);
        // through a wall nothing flows: only the pressure pushes. The flux formula gives that from the mirror
        // image up to rounding; here the rounding is taken out, so that mass and energy are kept exactly.
        if (kind[2] == BC_WALL) { for (int v = 0; v < nv; v++) if (v != ns) Fa[v] = 0; }
        if (kind[4] == BC_WALL) { for (int v = 0; v < nv; v++) if (v != ns) Fb[v] = 0; }
        const S s = (S)P.dth[d];
        for (int i = 0; i < ns; i++) R[i] -= s * ((S)Fb[i] - (S)Fa[i]);
        R[ns + d] -= s * ((S)Fb[ns] - (S)Fa[ns]);
        R[ns + t1] -= s * ((S)Fb[ns + 1] - (S)Fa[ns + 1]);
        R[ns + t2] -= s * ((S)Fb[ns + 2] - (S)Fa[ns + 2]);
        R[ns + 3] -= s * ((S)Fb[ns + 3] - (S)Fa[ns + 3]);
    }
    S U[NV];                                   // U0 + b ((in - U0) + dt L): see gcfdc.cu for why it is written so
    for (int v = 0; v < nv; v++) U[v] = q0[v * N + m] + (S)P.b * ((q[v * N + m] - q0[v * N + m]) + R[v]);
    if (nfirst) atomicAdd(&gcount[G_FIRST], nfirst);
    if (nflat) atomicAdd(&gcount[G_FLAT], nflat);
    finish<S, T>(P, M, U, (T)pr[m], m, N, qo, po, gcount, gsum);
}

// One bit per direction: is there a strong pressure jump between this cell and a neighbour along it?
template <class S, class T> __global__ void k_sensor(Par<T> P, const S* q, const S* pr, unsigned char* fl, long N) {
    const long m = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;
    const int c[3] = {(int)(m % P.n[0]), (int)((m / P.n[0]) % P.n[1]), (int)(m / ((long)P.n[0] * P.n[1]))};
    T w[NV];
    long mm;
    int kind;
    const T p0 = (T)pr[N + m];
    int f = 0;
    for (int d = 0; d < 3; d++) {
        if (!P.act[d]) continue;
        for (int k = -1; k <= 1; k += 2) {
            load(P, q, pr, N, c, d, k, w, mm, kind);
            const T p1 = w[P.ns + 3];
            if (fabs(p1 - p0) > P.sensor * (p1 < p0 ? p1 : p0)) f |= 1 << d;
        }
    }
    fl[m] = (unsigned char)f;
}

// The state from what the user gives: species (kg/m3), velocity (in the momentum arrays) and temperature (in
// the first of the three property arrays). Energy and properties are formed in the storage precision.
template <class S> __global__ void k_init(int ns, const Mech<S>* Mp, S* q, S* pr, long N) {
    const long m = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;
    const Mech<S>& M = *Mp;
    S cc[NSMAX], csum = 0, rho = 0, f, d;
    for (int i = 0; i < ns; i++) { const S r = q[i * N + m]; rho += r; cc[i] = r * M.iW[i]; csum += cc[i]; }
    const S u = q[ns * N + m], v = q[(ns + 1) * N + m], w = q[(ns + 2) * N + m], Tk = pr[m];
    energy(M, cc, csum, Tk, f, d);
    q[ns * N + m] = rho * u; q[(ns + 1) * N + m] = rho * v; q[(ns + 2) * N + m] = rho * w;
    q[(ns + 3) * N + m] = M.R * f + S(0.5) * rho * (u * u + v * v + w * w);
    const S p = M.R * csum * Tk;
    pr[N + m] = p; pr[2 * N + m] = sqrt((d + csum) / d * p / rho);
}

// ---- chemistry of every cell over dt, at constant volume, on the flow solver's own arrays --------------
template <class S, class T> __global__
void k_chem(int ns, const Mech<T>* Mp, const Mech<S>* Msp, const Tab<T>* Tp, Ctl<T> C, int method, S* q, S* pr, float* hs,
            int* nsub, int* nrej, unsigned char* stat, long N, double dtd) {
    const long m = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;
    const Mech<T>& M = *Mp;
    const Mech<S>& MS = *Msp;
    S U[NSMAX], s0[NSMAX], s1[NSMAX], rho = 0;
    for (int i = 0; i < ns; i++) { U[i] = q[i * N + m]; rho += U[i]; }
    if (!(rho > 0)) { stat[m] |= 2; return; }
    const S ir = 1 / rho;
    const S mx = q[ns * N + m], my = q[(ns + 1) * N + m], mz = q[(ns + 2) * N + m];
    const S e = (q[(ns + 3) * N + m] - S(0.5) * (mx * mx + my * my + mz * mz) * ir) * ir;      // internal energy, J/kg
    for (int i = 0; i < ns; i++) { s0[i] = U[i] * MS.iW[i] * ir; s1[i] = s0[i]; }              // mol/kg
    const T dt = (T)dtd;
    T Tk = (T)pr[m], hctl = hs[m] > 0 ? (T)hs[m] : dt;
    int na = 0, nj = 0, status;
    if (method == ME_ROS34 || method == ME_ROS34A) status = ros_cell<S, T>(M, C, method == ME_ROS34A, s1, (T)(e / MS.R), (T)ir, Tk, dt, hctl, na, nj);
    else status = erk_cell<S, T>(M, *Tp, C, s1, (T)(e / MS.R), (T)ir, Tk, dt, hctl, na, nj);
    bool changed = false;
    for (int i = 0; i < ns; i++) {
        const S ds = s1[i] - s0[i];
        if (ds != 0) { U[i] += rho * MS.W[i] * ds; q[i * N + m] = U[i]; changed = true; }
    }
    hs[m] = (float)hctl;
    nsub[m] += na; nrej[m] += nj;
    if (!(Tk == Tk)) status |= 2;
    stat[m] |= (unsigned char)status;
    if (!changed) return;                                        // nothing burnt: the flow's own numbers stay
    T cc[NSMAX], csum = 0, f, d;
    for (int i = 0; i < ns; i++) { cc[i] = (T)U[i] * M.iW[i]; csum += cc[i]; }
    energy(M, cc, csum, Tk, f, d);
    const T p = M.R * csum * Tk;
    pr[m] = (S)Tk; pr[N + m] = (S)p; pr[2 * N + m] = (S)sqrt((d + csum) / d * p / (T)rho);
}

// Per block of cells: sums of the conserved numbers, then (NEXT numbers) the largest signal speed over cell
// size, lowest density, lowest pressure, lowest and highest temperature, highest pressure, highest speed of the
// gas. Summed in a fixed order, in double, so the totals are the same in every run.
template <class S, class T> __global__ void k_reduce(Par<T> P, const S* q, const S* pr, long N, double* out, double ih0, double ih1, double ih2) {
    __shared__ double sh[NRED][RBLOCK];
    const int t = threadIdx.x, ns = P.ns, nv = ns + 4, nred = nv + NEXT;
    const long m = (long)blockIdx.x * blockDim.x + t;
    double v[NRED];
    for (int k = 0; k < nred; k++) v[k] = 0;
    v[nv + 1] = v[nv + 2] = v[nv + 3] = 1e300;
    if (m < N) {
        double r = 0;
        for (int k = 0; k < nv; k++) { v[k] = q[k * N + m]; if (k < ns) r += v[k]; }
        const double ux = v[ns] / r, uy = v[ns + 1] / r, uz = v[ns + 2] / r, cs = pr[2 * N + m];
        v[nv] = P.act[0] * (fabs(ux) + cs) * ih0 + P.act[1] * (fabs(uy) + cs) * ih1 + P.act[2] * (fabs(uz) + cs) * ih2;
        v[nv + 1] = r; v[nv + 2] = pr[N + m]; v[nv + 3] = pr[m]; v[nv + 4] = pr[m]; v[nv + 5] = pr[N + m];
        v[nv + 6] = sqrt(ux * ux + uy * uy + uz * uz);
    }
    for (int k = 0; k < nred; k++) sh[k][t] = v[k];
    __syncthreads();
    for (int s = RBLOCK / 2; s > 0; s >>= 1) {
        if (t < s) {
            for (int k = 0; k < nv; k++) sh[k][t] += sh[k][t + s];
            for (int k = nv; k < nred; k++) {                          // a NaN wins, so a broken run is seen
                const double a = sh[k][t], b = sh[k][t + s];
                const bool low = k >= nv + 1 && k <= nv + 3;
                sh[k][t] = ((low ? a < b : a > b) || a != a) ? a : b;
            }
        }
        __syncthreads();
    }
    if (t == 0) for (int k = 0; k < nred; k++) out[(long)blockIdx.x * NRED + k] = sh[k][0];
}

// ---- host side ----------------------------------------------------------------------------------------
static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Base {
    virtual ~Base() {}
    virtual void set(const double* rho, const double* vel, const double* T) = 0;
    virtual void get(double* U, double* prop) = 0;
    virtual void scheme(int recon, int flux, int stages, double sensor) = 0;
    virtual void chemistry(int on, int method, double rtol, double atol, int maxsub, double quiet, int split) = 0;
    virtual int add_state(const double* s, double c) = 0;
    virtual void side(int side, int type, int state) = 0;
    virtual int segment(int side, const int* lo, const int* hi, int type, int state) = 0;
    virtual void reduce(double* r) = 0;
    virtual void step(double dt) = 0;
    virtual double bytes() = 0;
    virtual double state_speed() = 0;
    virtual void weno_eps(double e) = 0;
    virtual void floor(double T) = 0;
    virtual void counts(int* g, double* a, double* ms, int clear) = 0;
    virtual void chem_stats(int* nsub, int* nrej, int* status) = 0;
    double t = 0, ms_flow = 0, ms_chem = 0;
    bool timed = false;
    int nred = 0;
};

template <class S, class T> struct Solver : Base {
    Par<T> P;
    Mech<double> H;
    Mech<T>* dM; Mech<S>* dMs; Tab<T>* dTab;
    Ctl<T> C;
    long N;
    int ns, nv, nblk, nrblk, stages = 3, nstate = 0, method = ME_BS23, split = SP_FLOW_CHEM, chem_on = 0;
    double h[3], state_c[MAXSTATE];
    S *A, *B, *D, *PA, *PB, *PD;               // three copies of the state, and of temperature, pressure, speed of sound
    unsigned char *fl, *stat;
    double *red, *gsum;
    float* hs;
    int *nsub, *nrej, *gcount;
    std::vector<double> hred;
    std::vector<S> buf;

    Solver(const int* n, const double* hh, const Mech<double>& Dm) : H(Dm) {
        memset(&P, 0, sizeof(P));
        N = 1;
        for (int d = 0; d < 3; d++) { P.n[d] = n[d]; P.act[d] = n[d] > 1; h[d] = hh[d]; N *= n[d]; }
        ns = P.ns = H.ns; nv = ns + 4; nred = nv + NEXT;
        P.recon = RC_VANLEER; P.flux = FX_HYBRID; P.sensor = (T)0.5; P.Tfloor = (T)60;
        P.weps = sizeof(T) == 4 ? (T)1e-10 : (T)1e-20;
        nblk = (int)((N + BLOCK - 1) / BLOCK); nrblk = (int)((N + RBLOCK - 1) / RBLOCK);
        Mech<T> mt; Mech<S> ms;
        convert(H, mt); convert(H, ms);
        CHECK(cudaMalloc(&dM, sizeof(mt))); CHECK(cudaMemcpy(dM, &mt, sizeof(mt), cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&dMs, sizeof(ms))); CHECK(cudaMemcpy(dMs, &ms, sizeof(ms), cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&dTab, sizeof(Tab<T>)));
        for (S** a : {&A, &B, &D}) CHECK(cudaMalloc(a, (size_t)nv * N * sizeof(S)));
        for (S** a : {&PA, &PB, &PD}) CHECK(cudaMalloc(a, (size_t)3 * N * sizeof(S)));
        CHECK(cudaMalloc(&fl, N)); CHECK(cudaMalloc(&stat, N));
        CHECK(cudaMalloc(&red, (size_t)nrblk * NRED * sizeof(double)));
        CHECK(cudaMalloc(&hs, N * sizeof(float)));
        CHECK(cudaMalloc(&nsub, N * sizeof(int))); CHECK(cudaMalloc(&nrej, N * sizeof(int)));
        CHECK(cudaMalloc(&gcount, G_N * sizeof(int))); CHECK(cudaMalloc(&gsum, 2 * sizeof(double)));
        CHECK(cudaMemset(fl, 0, N));
        hred.resize((size_t)nrblk * NRED);
        buf.resize((size_t)nv * N);
        chemistry(0, ME_BS23, 1e-6, 1e-14, 100000, 0, SP_FLOW_CHEM);
        clear();
    }
    ~Solver() {
        for (void* a : {(void*)A, (void*)B, (void*)D, (void*)PA, (void*)PB, (void*)PD, (void*)fl, (void*)stat, (void*)red, (void*)gsum, (void*)hs,
                        (void*)nsub, (void*)nrej, (void*)gcount, (void*)dM, (void*)dMs, (void*)dTab}) cudaFree(a);
    }
    void clear() {
        CHECK(cudaMemset(hs, 0, N * sizeof(float))); CHECK(cudaMemset(nsub, 0, N * sizeof(int)));
        CHECK(cudaMemset(nrej, 0, N * sizeof(int))); CHECK(cudaMemset(stat, 0, N));
        CHECK(cudaMemset(gcount, 0, G_N * sizeof(int))); CHECK(cudaMemset(gsum, 0, 2 * sizeof(double)));
        ms_flow = ms_chem = 0;
    }
    void weno_eps(double e) override { P.weps = (T)e; }
    void floor(double Tf) override { P.Tfloor = (T)Tf; }
    // Largest signal speed over cell size among the prescribed boundary states (see gcfdc.cu)
    double state_speed() override {
        double m = 0;
        for (int k = 0; k < nstate; k++) {
            double v = 0;
            for (int d = 0; d < 3; d++) v += P.act[d] * (fabs((double)P.state[k][ns + d]) + state_c[k]) / h[d];
            if (v > m) m = v;
        }
        return m;
    }
    double bytes() override { return (3.0 * (nv + 3) * sizeof(S) + 2 + 4 + 4 + 4) * N + (double)nrblk * NRED * 8; }
    void up(S* dev, const double* src, size_t n) {
        for (size_t i = 0; i < n; i++) buf[i] = (S)src[i];
        CHECK(cudaMemcpy(dev, buf.data(), n * sizeof(S), cudaMemcpyHostToDevice));
    }
    void down(double* dst, const S* dev, size_t n) {
        CHECK(cudaMemcpy(buf.data(), dev, n * sizeof(S), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; i++) dst[i] = (double)buf[i];
    }
    // rho[ns N]: species, kg/m3; vel[3 N]; T[N]
    void set(const double* rho, const double* vel, const double* T_) override {
        up(A, rho, (size_t)ns * N); up(A + (size_t)ns * N, vel, (size_t)3 * N); up(PA, T_, N);
        k_init<S><<<nblk, BLOCK>>>(ns, dMs, A, PA, N);
        clear();
        CHECK(cudaDeviceSynchronize());
        t = 0;
    }
    // U[(ns + 4) N]: species, momentum, energy; prop[3 N]: temperature, pressure, speed of sound
    void get(double* U, double* prop) override { down(U, A, (size_t)nv * N); down(prop, PA, (size_t)3 * N); }
    void scheme(int recon, int flux, int st, double sensor) override { P.recon = recon; P.flux = flux; stages = st; P.sensor = (T)sensor; }
    void chemistry(int on, int me, double rtol, double atol, int maxsub, double quiet, int sp) override {
        chem_on = on && H.nr > 0; method = me; split = sp;
        C.rtol = (T)rtol; C.atol = (T)atol; C.nfix = 0; C.maxsub = maxsub; C.quiet = (T)quiet;
        Tab<T> Bt;
        tableau(me == ME_BS23 ? ME_BS23 : ME_DP45, Bt);
        CHECK(cudaMemcpy(dTab, &Bt, sizeof(Bt), cudaMemcpyHostToDevice));
    }
    int add_state(const double* s, double c) override {
        if (nstate >= MAXSTATE) return -1;
        for (int v = 0; v < nv; v++) P.state[nstate][v] = (T)s[v];
        state_c[nstate] = c;
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
        k_reduce<S, T><<<nrblk, RBLOCK>>>(P, A, PA, N, red, P.act[0] / h[0], P.act[1] / h[1], P.act[2] / h[2]);
        CHECK(cudaMemcpy(hred.data(), red, (size_t)nrblk * NRED * sizeof(double), cudaMemcpyDeviceToHost));
        for (int k = 0; k < nred; k++) r[k] = hred[k];
        for (long b = 1; b < nrblk; b++) {
            const double* x = &hred[b * NRED];
            for (int k = 0; k < nv; k++) r[k] += x[k];
            for (int k = nv; k < nred; k++) {
                const bool low = k >= nv + 1 && k <= nv + 3;
                if ((low ? x[k] < r[k] : x[k] > r[k]) || x[k] != x[k]) r[k] = x[k];
            }
        }
    }
    void stage(const S* in, const S* pin, S* out, S* pout, double b) {
        P.b = (T)b;
        if (P.flux == FX_HYBRID && P.act[0] + P.act[1] + P.act[2] > 1) k_sensor<S, T><<<nblk, BLOCK>>>(P, in, pin, fl, N);
        k_stage<S, T><<<nblk, BLOCK>>>(P, dM, in, pin, A, out, pout, fl, N, sizeof(T) == 4 ? (T)1e-37 : (T)1e-300, gcount, gsum);
    }
    void flow(double dt) {
        for (int d = 0; d < 3; d++) P.dth[d] = dt / h[d];
        if (stages == 1) { stage(A, PA, B, PB, 1); std::swap(A, B); std::swap(PA, PB); }
        else if (stages == 2) { stage(A, PA, B, PB, 1); stage(B, PB, D, PD, 0.5); std::swap(A, D); std::swap(PA, PD); }      // Heun (SSP-RK2)
        else { stage(A, PA, B, PB, 1); stage(B, PB, D, PD, 0.25); stage(D, PD, B, PB, 2.0 / 3.0); std::swap(A, B); std::swap(PA, PB); }   // SSP-RK3
    }
    void chem(double dt) {
        k_chem<S, T><<<nblk, BLOCK>>>(ns, dM, dMs, dTab, C, method, A, PA, hs, nsub, nrej, stat, N, dt);
    }
    void part(bool is_chem, double dt) {
        if (is_chem && !chem_on) return;
        if (!timed) { if (is_chem) chem(dt); else flow(dt); return; }
        CHECK(cudaDeviceSynchronize());
        const double t0 = now_ms();
        if (is_chem) chem(dt); else flow(dt);
        CHECK(cudaDeviceSynchronize());
        (is_chem ? ms_chem : ms_flow) += now_ms() - t0;
    }
    // One time step. The order of its two parts: flow then chemistry (the default), chemistry then flow, or half
    // the chemistry, the flow, the other half (Strang).
    void step(double dt) override {
        if (split == SP_CHEM_FLOW) { part(true, dt); part(false, dt); }
        else if (split == SP_STRANG) { part(true, 0.5 * dt); part(false, dt); part(true, 0.5 * dt); }
        else { part(false, dt); part(true, dt); }
        t += dt;
    }
    void counts(int* g, double* a, double* ms, int clr) override {
        CHECK(cudaMemcpy(g, gcount, G_N * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(a, gsum, 2 * sizeof(double), cudaMemcpyDeviceToHost));
        ms[0] = ms_flow; ms[1] = ms_chem;
        if (clr) {
            CHECK(cudaMemset(gcount, 0, G_N * sizeof(int))); CHECK(cudaMemset(gsum, 0, 2 * sizeof(double)));
            ms_flow = ms_chem = 0;
        }
    }
    void chem_stats(int* a, int* b, int* c) override {
        CHECK(cudaMemcpy(a, nsub, N * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b, nrej, N * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<unsigned char> sb(N);
        CHECK(cudaMemcpy(sb.data(), stat, N, cudaMemcpyDeviceToHost));
        for (long i = 0; i < N; i++) c[i] = sb[i];
    }
};

extern "C" {

// The gases and the reaction scheme as two arrays (packed by gpucfd/chemistry.py, layout: unpack_mech in
// gcfdk_core.h). precision: 4 = float, 8 = double, 6 = mixed (state stored in double; fluxes, temperatures and
// rates computed in float). Returns NULL if the scheme is larger than the compiled limits.
void* gcfdr_create(const int* n, const double* h, int precision, const int* im, const double* dm) {
    static Mech<double> D;
    if (!unpack_mech(im, dm, 1, D)) return nullptr;
    if (precision == 4) return (Base*)new Solver<float, float>(n, h, D);
    if (precision == 6) return (Base*)new Solver<double, float>(n, h, D);
    return (Base*)new Solver<double, double>(n, h, D);
}
void gcfdr_destroy(void* s) { delete (Base*)s; }
// rho[ns N]: mass of each species per volume, species by species; vel[3 N]: u, v, w; T[N]. Clears the time, the
// counters of the guard and of the chemistry, and the timers.
void gcfdr_set(void* s, const double* rho, const double* vel, const double* T) { ((Base*)s)->set(rho, vel, T); }
// U[(ns + 4) N]: species (kg/m3), momentum x y z, total energy; prop[3 N]: temperature, pressure, speed of sound
void gcfdr_get(void* s, double* U, double* prop) { ((Base*)s)->get(U, prop); }
void gcfdr_scheme(void* s, int recon, int flux, int stages, double sensor) { ((Base*)s)->scheme(recon, flux, stages, sensor); }
// on: 0 no reactions. method, rtol, atol, maxsub, quiet: as gcfdk_control. split: 0 flow then chemistry,
// 1 chemistry then flow, 2 half chemistry, flow, half chemistry.
void gcfdr_chemistry(void* s, int on, int method, double rtol, double atol, int maxsub, double quiet, int split) {
    ((Base*)s)->chemistry(on, method, rtol, atol, maxsub, quiet, split);
}
// st: species (kg/m3), u, v, w, p of a prescribed boundary state; c: its speed of sound
int gcfdr_add_state(void* s, const double* st, double c) { return ((Base*)s)->add_state(st, c); }
void gcfdr_side(void* s, int side, int type, int state) { ((Base*)s)->side(side, type, state); }
int gcfdr_segment(void* s, int side, const int* lo, const int* hi, int type, int state) { return ((Base*)s)->segment(side, lo, hi, type, state); }
// r[ns + 4 + 7]: sums over the cells of species, momentum x y z, energy; then largest sum over directions of
// (|u| + c) / h, lowest density, lowest pressure, lowest temperature, highest temperature, highest pressure,
// highest speed of the gas
void gcfdr_reduce(void* s, double* r) { ((Base*)s)->reduce(r); }
int gcfdr_nred(void* s) { return ((Base*)s)->nred; }
double gcfdr_bytes(void* s) { return ((Base*)s)->bytes(); }
double gcfdr_time(void* s) { return ((Base*)s)->t; }
void gcfdr_weno_eps(void* s, double eps) { ((Base*)s)->weno_eps(eps); }
// The guard's temperature floor (default 60 K)
void gcfdr_floor(void* s, double T) { ((Base*)s)->floor(T); }
// timed != 0: the GPU is made to finish after each part of a step, and the time of flow and chemistry is added up
void gcfdr_timed(void* s, int timed) { ((Base*)s)->timed = timed != 0; }
// g[3]: uses of the guard since the last clear: cells x stages x directions that fell back to first order,
// cells x stages with a species set to zero, cells x stages raised to the temperature floor. a[2]: mass moved by
// the second (kg/m3, summed), energy added by the third (J/m3, summed). ms[2]: milliseconds in flow and in
// chemistry (when timed). clear != 0: start counting again.
void gcfdr_counts(void* s, int* g, double* a, double* ms, int clear) { ((Base*)s)->counts(g, a, ms, clear); }
// per cell: accepted and rejected chemistry sub-steps since gcfdr_set, and the status (0 ok, 1 sub-step limit
// reached, 2 broke down)
void gcfdr_chem_stats(void* s, int* nsub, int* nrej, int* status) { ((Base*)s)->chem_stats(nsub, nrej, status); }

// nsteps steps of size dt with no copy to the CPU. hist (or NULL): the numbers of gcfdr_reduce after every step.
// Returns the milliseconds taken (GPU finished).
double gcfdr_run_fixed(void* sv, double dt, long nsteps, double* hist) {
    Base* s = (Base*)sv;
    CHECK(cudaDeviceSynchronize());
    const double t0 = now_ms();
    for (long k = 0; k < nsteps; k++) {
        s->step(dt);
        if (hist) s->reduce(hist + (long)s->nred * k);
    }
    CHECK(cudaDeviceSynchronize());
    return now_ms() - t0;
}

// Advance by tspan with dt = cfl / max(sum over directions of (|u| + c) / h), found again every `every` steps;
// the remaining time is cut into equal steps so that the end is met exactly.
// stats[9]: steps, time advanced, last dt, milliseconds, last max speed over h, lowest density, lowest pressure,
// status (0 ok, 1 not a number or no gas left, 2 step limit reached), the highest pressure at any of the looks.
void gcfdr_advance(void* sv, double tspan, double cfl, int every, long maxsteps, double* stats) {
    Base* s = (Base*)sv;
    CHECK(cudaDeviceSynchronize());
    const double t0 = now_ms();
    const int o = s->nred - NEXT;
    double done = 0, dt = 0, r[NRED] = {0}, peak = 0;
    long k = 0;
    int status = 0;
    while (done < tspan * (1 - 1e-13)) {
        if (k >= maxsteps) { status = 2; break; }
        if (k % every == 0) {
            s->reduce(r);
            if (r[o + 5] > peak) peak = r[o + 5];
            if (!(r[o] == r[o]) || !(r[o + 1] > 0) || !(r[o + 2] > 0) || !(r[o] < 1e300)) { status = 1; break; }
            const double bs = s->state_speed();
            if (bs > r[o]) r[o] = bs;
            const double left = tspan - done;
            const double nrem = ceil(left * r[o] / cfl - 1e-9);
            dt = left / (nrem < 1 ? 1 : nrem);
        }
        if (done + dt > tspan) dt = tspan - done;
        s->step(dt);
        done += dt;
        k++;
    }
    if (status == 0) {
        s->reduce(r);
        if (r[o + 5] > peak) peak = r[o + 5];
        if (!(r[o] == r[o]) || !(r[o + 1] > 0) || !(r[o + 2] > 0)) status = 1;
    }
    CHECK(cudaDeviceSynchronize());
    stats[8] = peak;
    stats[0] = (double)k; stats[1] = done; stats[2] = dt; stats[3] = now_ms() - t0;
    stats[4] = r[o]; stats[5] = r[o + 1]; stats[6] = r[o + 2]; stats[7] = status;
}

}  // extern "C"
