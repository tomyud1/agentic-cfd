"""The requested outputs and the physical checks, computed from a stored run (runs/<name>.npz).

All arrays of a run are on NASA's grid: xs, ys are the grid lines, fields are [j (y), i (x)] at cell centres.
"""
import json

import numpy as np

from .common import RUNS


def load_run(name):
    rec = json.loads((RUNS / f"{name}.json").read_text())
    a = dict(np.load(RUNS / f"{name}.npz"))
    a["xc"] = 0.5 * (a["xs"][1:] + a["xs"][:-1])
    a["yc"] = 0.5 * (a["ys"][1:] + a["ys"][:-1])
    a["i0"] = len(a["xc"]) - len(a["tau_x"])                  # first cell column on the plate
    return rec, a


def cf_along(a, G):
    """(x of the wall-face centres, Cf) on the plate."""
    return a["xc"][a["i0"]:], a["tau_x"] / (0.5 * G["U_inf"] ** 2)


def cf_at(a, G, x):
    """Cf at x by linear interpolation between the two wall faces next to it (the stations are grid points,
    so no wall face is centred on them)."""
    xf, cf = cf_along(a, G)
    return float(np.interp(x, xf, cf))


def cf_wall_cell(a, G):
    """Cf rebuilt from the velocity in the wall cell, nu u_P / y_P: what the solver's wall face uses."""
    return G["nu"] * a["ux"][0, a["i0"]:] / a["yc"][0] / (0.5 * G["U_inf"] ** 2)


def cd_integral(a, G):
    """Drag coefficient from the wall shear stress, summed face by face (reference area: the plate length)."""
    dx = np.diff(a["xs"])[a["i0"]:]
    return float(np.sum(a["tau_x"] * dx) / (0.5 * G["U_inf"] ** 2 * G["x_out"]))


def balances(a, G):
    """Mass and x-momentum balance over the whole domain with the solver's face fluxes.

    Momentum (per unit density and depth): the friction force on the plate must equal the x-momentum the
    flow loses through the outer boundaries plus the pressure force on the inflow side (the outflow and
    the top are at the free-stream pressure, 0). Left out: the viscous normal stress on the inflow and
    outflow sides."""
    dy, dx = np.diff(a["ys"]), np.diff(a["xs"])
    friction = float(np.sum(a["tau_x"] * dx[a["i0"]:]))
    u_in = -a["phi_inlet"] / dy                              # inflow velocity of each face, from its own flux
    conv = float(np.sum(a["phi_inlet"] * u_in) + np.sum(a["phi_outlet"] * a["u_outlet"])
                 + np.sum(a["phi_top"] * a["u_top"]))
    p_in = float(np.sum(a["p"][:, 0] * dy))                   # zero-gradient inflow: face pressure = cell pressure
    lost = -conv + p_in
    inflow = float(-np.sum(a["phi_inlet"]))
    mass = float(np.sum(a["phi_inlet"]) + np.sum(a["phi_outlet"]) + np.sum(a["phi_top"])) / inflow
    return dict(friction=friction, momentum_lost=lost, momentum=lost / friction - 1.0, mass=mass,
                top_outflow=float(np.sum(a["phi_top"])) / inflow, pressure_term=p_in / friction)


def column(a, field, x):
    """A field's wall-normal profile at x, by linear interpolation between the two cell columns next to x."""
    xc = a["xc"]
    k = int(np.clip(np.searchsorted(xc, x), 1, len(xc) - 1))
    w = (x - xc[k - 1]) / (xc[k] - xc[k - 1])
    return (1 - w) * a[field][:, k - 1] + w * a[field][:, k]


def profile(a, G, x):
    """Velocity and eddy viscosity profiles at x, in outer and in wall units."""
    u = column(a, "ux", x)
    cf = cf_at(a, G, x)
    u_tau = G["U_inf"] * np.sqrt(cf / 2)
    return dict(y=a["yc"], u=u / G["U_inf"], yplus=a["yc"] * u_tau / G["nu"], uplus=u / u_tau, u_tau=float(u_tau), cf=cf,
                nut=column(a, "nut", x) / G["nu"], nuTilda=column(a, "nuTilda", x) / G["nu"])


def nut_max_along(a, G):
    """(x of the cell columns, largest nu_t/nu in each column)."""
    return a["xc"], a["nut"].max(axis=0) / G["nu"]


def thicknesses(a, G, x):
    """Boundary layer thickness (u = 0.99 U_e), displacement and momentum thickness at x, with the edge velocity
    U_e taken as the largest velocity of the profile."""
    u = column(a, "ux", x)
    ue = float(u.max())
    dy = np.diff(a["ys"])
    r = u / ue
    j_edge = int(np.argmax(u))
    d99 = float(np.interp(0.99, r[:j_edge + 1], a["yc"][:j_edge + 1]))
    m = a["yc"] <= 5 * d99
    return dict(delta99=d99, delta_star=float(np.sum((1 - r[m]) * dy[m])), theta=float(np.sum(r[m] * (1 - r[m]) * dy[m])),
                u_edge=ue / G["U_inf"])


def limiter_activity(a, G, kappa=0.41, cv1=7.1, cs=0.3):
    """Does OpenFOAM's limiter on the model's S-tilde (S-tilde >= 0.3 Omega) ever act in the solution?
    Returns the smallest S-tilde / Omega over the cells above the plate, before limiting (a value above 0.3 means
    the limiter changed nothing), how many cells fall below 0.3 and how far along the plate the last of them is. Omega from central differences
    of the cell values."""
    i0 = a["i0"]
    yc, xc = a["yc"], a["xc"]
    dudy = np.gradient(a["ux"], yc, axis=0)
    dvdx = np.gradient(a["uy"], xc, axis=1)
    omega = np.abs(dudy - dvdx)[:, i0:]
    chi = a["nuTilda"][:, i0:] / G["nu"]
    fv1 = chi ** 3 / (chi ** 3 + cv1 ** 3)
    fv2 = 1 - chi / (1 + chi * fv1)
    d = yc[:, None]
    s = omega + fv2 * a["nuTilda"][:, i0:] / (kappa * d) ** 2
    ratio = s / np.maximum(omega, 1e-300)
    inside = (chi > 1.0) & (omega > 1e-3 * omega.max())       # where the model is active and the shear is not negligible
    low = inside & (ratio < cs)
    xcol = np.broadcast_to(xc[i0:][None, :], ratio.shape)
    return dict(min_ratio=float(ratio[inside].min()), cells_limited=int(low.sum()), cells=int(inside.sum()),
                x_last=float(xcol[low].max()) if low.any() else None)
