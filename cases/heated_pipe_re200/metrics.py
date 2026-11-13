"""Quantities compared with the exact solution, computed from one run's saved arrays (runs/<name>.npz/.json).

Used by analyze.py and, before production, on the coarsest-mesh pre-flight run to size the tolerances.
Nu, T_w, T_b follow the problem's definitions: T_b is the flow-weighted mean over a cross-section
(cell column: sum u T V / sum u V, the midpoint rule), T_w the wall face value, Nu = q_w D / (k (T_w - T_b)).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "results" / HERE.name / "runs"
X_MIN_D = 0.5                 # comparison of Nu(x) from 0.5 D on (the series is converged there; theory.X_MIN_D)
X_MAX_D = 100.0               # comparisons end here: the outlet's zero-gradient condition shifts the last two
                              # cells (insight: on L1 they reach back to 108 D); fixed in metres, same on every mesh
X_TAIL_D = (60.0, 100.0)      # fit window for the simulation's own fully developed value (and its outlet values)
X_MAX_ERR2_D = 5.0            # a second max-error window that excludes the start of heating


def load(name):
    rec = json.loads((RUNS / f"{name}.json").read_text())
    arr = dict(np.load(RUNS / f"{name}.npz"))
    return rec, arr


def faces_from_centres(xc, x_start):
    f = np.empty(len(xc) + 1)
    f[0] = x_start
    for i, x in enumerate(xc):
        f[i + 1] = 2 * x - f[i]
    return f


def tail_fit(x, dT):
    """dT = A + B exp(-c x) by least squares (no theory input). Returns A, B, c."""
    p0 = (dT[-1], dT[0] - dT[-1], 1.0 / (0.5 * (x[-1] - x[0])))
    p, _ = optimize.curve_fit(lambda xx, A, B, c: A + B * np.exp(-c * (xx - x[0])), x, dT, p0=p0, maxfev=20000)
    return float(p[0]), float(p[1]), float(p[2])


def wedge(rec, arr, ex, G, half_angle_deg):
    D, q, k = G["D"], G["q_w"], G["k"]
    a = D / 2
    x, Tw, Tb = arr["x"], arr["Tw"], arr["Tb"]
    h = x > 0
    xh, Twh, Tbh = x[h], Tw[h], Tb[h]
    fh = faces_from_centres(xh, 0.0)
    assert abs(fh[-1] - G["Lh"]) < 1e-9 * G["Lh"], "heated faces do not end at L_h"
    dx = np.diff(fh)
    dT = Twh - Tbh
    nu_s = q * D / (k * dT)
    nu_e = ex.nu(xh)
    w = (xh >= X_MIN_D * D) & (xh <= X_MAX_D * D)
    rel = nu_s / nu_e - 1
    e_rms = float(math.sqrt(np.sum(dx[w] * rel[w] ** 2) / np.sum(dx[w])))
    e_max = float(np.max(np.abs(rel[w])))
    def rms(a, b):
        ww = (xh >= a * D) & (xh <= b * D)
        return float(math.sqrt(np.sum(dx[ww] * rel[ww] ** 2) / np.sum(dx[ww])))
    e_rms_entr, e_rms_down = rms(X_MIN_D, 10.0), rms(40.0, X_MAX_D)
    w5 = (xh >= X_MAX_ERR2_D * D) & (xh <= X_MAX_D * D)
    e_max5 = float(np.max(np.abs(rel[w5])))
    # Fully developed value of the simulation itself: tail fit (no theory input)
    wt = (xh >= X_TAIL_D[0] * D) & (xh <= X_TAIL_D[1] * D)
    A, B, c = tail_fit(xh[wt], dT[wt])
    nu_fd_s = q * D / (k * A)
    # Outlet: the wall-to-bulk difference from the same fit at x = L_h (the last cells carry the outlet
    # condition's shift), the bulk temperature from the outlet patch (exact energy flux)
    dT_out = A + B * math.exp(-c * (G["Lh"] - xh[wt][0]))
    nu_out_s, nu_out_e = q * D / (k * dT_out), float(ex.nu(G["Lh"])[0])
    last = dict(nu=float(nu_s[-1]), nu_ex_xc=float(nu_e[-1]), Tw=float(Twh[-1]), Tb=float(Tbh[-1]),
                Tw_ex_xc=float(ex.T_w(xh[-1])[0]), Tb_ex_xc=float(ex.T_b(xh[-1])), x=float(xh[-1]), dx=float(dx[-1]))
    # Entrance length: first crossing of 1.05 Nu_fd (own fd value), linear interpolation
    def crossing(nu, ref):
        target = 1.05 * ref
        i = int(np.argmax(nu < target))
        return float(xh[i - 1] + (target - nu[i - 1]) * (xh[i] - xh[i - 1]) / (nu[i] - nu[i - 1]))
    l_th_s = crossing(nu_s, nu_fd_s)
    l_th_s_exfd = crossing(nu_s, ex.nu_fd)
    # The same post-processing applied to the EXACT solution at the same cell positions: its result minus
    # the exact value is the bias of the method (fit, crossing), not of the simulation
    dTe = ex.T_w(xh) - ex.T_b(xh)
    Ae, Be, ce = tail_fit(xh[wt], dTe[wt])
    pp_nu_fd = (q * D / (k * Ae)) / ex.nu_fd - 1
    pp_nu_out = (q * D / (k * (Ae + Be * math.exp(-ce * (G["Lh"] - xh[wt][0]))))) / float(ex.nu(G["Lh"])[0]) - 1
    pp_l_th = crossing(nu_e, q * D / (k * Ae)) / ex.l_th() - 1
    # Temperatures at the outlet: bulk from the outlet patch, wall = bulk + fitted wall-to-bulk difference
    Tb_out_patch = rec["phiT_out"] / rec["phi_out"]
    Tw_out = float(Tb_out_patch + dT_out)
    dT_fd = q * D / (k * ex.nu_fd)
    Tw_e, Tb_e = ex.T_w(xh), ex.T_b(xh)
    e_Tw = float(np.max(np.abs(Twh[w] - Tw_e[w]))) / dT_fd
    e_Tb = float(np.max(np.abs(Tbh[w] - Tb_e[w]))) / dT_fd
    # Energy and mass balance (wedge: heated wall area = L_h x chord)
    th = math.radians(half_angle_deg)
    A_h = G["Lh"] * 2 * a * math.sin(th)
    Q_in = q * A_h
    Q_out = G["rho"] * G["cp"] * (rec["phiT_out"] + rec["phiT_in"])
    energy = Q_out / Q_in - 1
    mass = (rec["phi_out"] + rec["phi_in"]) / (-rec["phi_in"])
    # Velocity profile at x = 0: mean of the two columns either side (equal lengths), against the parabola
    i0 = int(np.argmax(x > 0))
    u0 = 0.5 * (arr["ux"][:, i0 - 1] + arr["ux"][:, i0])
    r = arr["r"]
    u_par = 2 * G["U"] * (1 - (r / a) ** 2)
    # centreline: quadratic through the 3 innermost cells, evaluated at r = 0
    cc = np.polyfit(r[:3] ** 2, u0[:3], 1)
    u_c0 = float(cc[1])
    return dict(
        x=xh, dx=dx, nu=nu_s, nu_ex=nu_e, rel=rel, Tw=Twh, Tb=Tbh, Tw_ex=Tw_e, Tb_ex=Tb_e,
        e_rms=e_rms, e_max=e_max, e_max5=e_max5, e_rms_entr=e_rms_entr, e_rms_down=e_rms_down,
        nu_out=nu_out_s, nu_out_ex=nu_out_e, e_nu_out=nu_out_s / nu_out_e - 1, last=last,
        e_last_Tb=(last["Tb"] - last["Tb_ex_xc"]) / (q * D / (k * ex.nu_fd)),
        nu_fd=nu_fd_s, e_nu_fd=nu_fd_s / ex.nu_fd - 1, fit=(A, B, c),
        pp_nu_fd=pp_nu_fd, pp_nu_out=pp_nu_out, pp_l_th=pp_l_th,
        l_th=l_th_s, l_th_exfd=l_th_s_exfd, e_l_th=l_th_s / ex.l_th() - 1,
        Tw_out=Tw_out, Tw_out_ex=float(ex.T_w(G["Lh"])[0]), Tb_out=Tb_out_patch, Tb_out_ex=float(ex.T_b(G["Lh"])),
        e_Tw=e_Tw, e_Tb=e_Tb, dT_fd=dT_fd,
        energy=energy, mass=mass, u0=u0, r=r, u_par=u_par, u_c0=u_c0, e_uc0=u_c0 / (2 * G["U"]) - 1,
        e_u0=float(np.max(np.abs(u0 - u_par))) / (2 * G["U"]),
    )


def half3d(rec, arr, G):
    """3D half tube: per axial layer the bulk temperature, the peripherally averaged wall temperature (wall
    faces are equally spaced around the half circle), top and bottom wall temperatures, and Nu."""
    D, q, k = G["D"], G["q_w"], G["k"]
    C, U, T, V = arr["C"], arr["U"], arr["T"], arr["V"]
    key = np.round(C[:, 0], 9)
    xs, inv = np.unique(key, return_inverse=True)
    wsum = np.bincount(inv, U[:, 0] * V)
    Tb = np.bincount(inv, U[:, 0] * T * V) / wsum
    sec = np.sqrt(U[:, 1] ** 2 + U[:, 2] ** 2)
    sec_max = np.zeros(len(xs))
    np.maximum.at(sec_max, inv, sec)
    kw = np.round(arr["Cw"][:, 0], 9)
    xw, invw = np.unique(kw, return_inverse=True)
    Tw_avg = np.bincount(invw, arr["Tw"]) / np.bincount(invw)
    top = np.full(len(xw), np.nan)
    bot = np.full(len(xw), np.nan)
    for j in range(len(xw)):
        sel = invw == j
        y = arr["Cw"][sel, 1]
        top[j] = arr["Tw"][sel][np.argmax(y)]
        bot[j] = arr["Tw"][sel][np.argmin(y)]
    h = xs > 0
    assert np.allclose(xs[h], xw, rtol=0, atol=1e-9), "wall faces and cell layers do not line up"
    nu = q * D / (k * (Tw_avg - Tb[h]))
    return dict(x=xw, Tb=Tb[h], Tw=Tw_avg, Tw_top=top, Tw_bot=bot, nu=nu, sec_max=sec_max[h],
                energy=None)


if __name__ == "__main__":
    import sys
    import yaml
    from . import theory
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    num = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    G = {k: v["value"] for k, v in problem["given"].items()}
    ex = theory.Exact(problem["given"])
    for name in sys.argv[1:]:
        rec, arr = load(name)
        mt = wedge(rec, arr, ex, G, num["geometry"]["wedge_half_angle_deg"])
        print(name, rec["cells"], "cells", rec["iterations"], "its")
        for k, v in mt.items():
            if np.isscalar(v) or isinstance(v, tuple):
                print(f"   {k:12s} {v}")
