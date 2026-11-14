"""Turn the statistics files of Vreman and Kuerten (2014), saved by setup/fetch_vk2014_channel.sh in
cases/turbulent_channel_re180/reference/vk2014/, into the case's reference YAML. Every number in the
YAML is read or computed here; nothing is typed.

The two simulations are of the same flow in the same box by two independent codes (S2: spectral, FD2:
finite differences on a staggered grid), each averaged over 200 time units. What one differs from the
other is the reference's own uncertainty, and it is written into the YAML as numbers.
"""
import re
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

DIR = Path("cases/turbulent_channel_re180/reference")
RAW = DIR / "vk2014"
RE_TAU = 180.0
# What the paper itself states. Typed from the paper (Tables I and II, Sections II and III), not parsed from a file.
PAPER = dict(
    _note="Typed by a person from the paper's Tables I and II and Sections II and III; everything else in this file is parsed.",
    driving="a constant force (pressure gradient) in both simulations",
    finite_difference_code="staggered grid; fourth order along x and z, second order along y",
    largest_cell_sizes_in_wall_units=dict(spectral=dict(x=5.9, y=2.9, z=3.9), finite_difference=dict(x=4.4, y=2.2, z=2.9)),
    statistical_error_percent=dict(
        _note=("Largest relative statistical error along a profile for 200 time units of averaging, estimated by the authors "
               "from a coarser finite-difference run that they averaged over 1300 time units."),
        mean_velocity=0.1, u_rms=0.4, v_rms=0.4, w_rms=0.3),
    largest_difference_between_the_two_codes_percent=dict(mean_velocity=0.1, u_rms=0.3, v_rms=0.6, w_rms=0.4),
    authors_bound_on_largest_relative_error_percent=dict(mean_velocity=0.2, rms_fluctuations=1.0),
    coarser_runs_for_scale=dict(
        _note=("Largest relative difference from the spectral simulation along the mean-velocity profile, for the authors' "
               "runs on coarser grids and for the older database of Moser, Kim and Mansour (1999). Cell sizes in wall "
               "units along x, largest along y, along z."),
        finite_difference_coarser=dict(cells_wall_units=[8.8, 4.4, 5.9], mean_velocity_percent=0.8, u_rms_percent=1.0),
        spectral_coarser=dict(cells_wall_units=[17.7, 4.4, 5.9], mean_velocity_percent=0.4, u_rms_percent=0.8),
        moser_kim_mansour_1999=dict(cells_wall_units=[17.7, 4.4, 5.9], mean_velocity_percent=0.3, u_rms_percent=2.7)),
    centreline_velocity_table=dict(spectral=18.28, finite_difference=18.28, moser_kim_mansour_1999=18.30),
)


def read(name, columns):
    """Header (the lines that start with %) and the table of one file. columns: {number: text the header
    must give for that column}, so a changed file layout stops the script instead of shifting a column."""
    text = (RAW / name).read_text(encoding="ascii")
    head = [ln.strip().lstrip("%").strip() for ln in text.splitlines() if ln.strip().startswith("%")]
    rows = [ln.split() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("%")]
    data = np.array(rows, float)
    for col, want in columns.items():
        line = next((h for h in head if re.match(rf"Column\s+{col}:", h)), "")
        if want not in line:
            raise SystemExit(f"{name}: column {col} is described as {line!r}, expected {want!r}")
    if data.shape[1] < max(columns):
        raise SystemExit(f"{name}: {data.shape[1]} columns")

    def field(key, pattern, conv=str):
        m = next((re.search(pattern, h) for h in head if h.startswith(key)), None)
        if not m:
            raise SystemExit(f"{name}: header line {key!r} not found or not understood")
        return conv(m.group(1))
    info = dict(parameters=field("Parameters:", r"Parameters:\s*(.+)"), domain=field("Domain:", r"Domain:\s*(.+)"),
                resolution=field("Resolution:", r"Resolution:\s*(.+)"), averaging_time=field("Averaging time:", r"([\d.]+)", float),
                computed_u_tau=field("Computed u_tau:", r"([\d.]+)", float))
    if re.sub(r"\s+", "", info["parameters"]) != "H=1;nu=1/180;forcing=1" or re.sub(r"\s+", "", info["domain"]) != "4pix2x4pi/3":
        raise SystemExit(f"{name}: not the set-up of the case: {info}")
    return info, data


def lst(a):
    return [float(f"{v:.10g}") for v in np.asarray(a)]


def scalars(y, u, urms, yv, uv, faces=None):
    """Numbers of one simulation. y, yv in units of the half height (0 = wall, 1 = centre plane).
    Bulk velocity: integral of the mean velocity over the half channel, by a cubic spline through the
    points with the wall (0, 0) added and a symmetric centre plane; for the staggered code also by its
    own rule (cell values times cell heights), to show what the choice of rule is worth."""
    yy, uu = np.concatenate(([0.0], y)), np.concatenate(([0.0], u))
    if yy[-1] < 1.0 - 1e-9:                              # cell centres: mirror the last cells about the centre plane
        yy, uu = np.concatenate((yy, 2.0 - yy[::-1][:3])), np.concatenate((uu, uu[::-1][:3]))
        sp = CubicSpline(yy, uu)
    else:
        sp = CubicSpline(yy, uu, bc_type=("not-a-knot", (1, 0.0)))
    bulk = float(sp.integrate(0.0, 1.0))
    out = dict(bulk_velocity=bulk, centreline_velocity=float(sp(1.0)), friction_coefficient=2.0 / bulk ** 2,
               bulk_reynolds_number=2.0 * RE_TAU * bulk)
    if faces is not None:
        out["bulk_velocity_cell_sum"] = float(np.sum(u * np.diff(faces)))
    k = int(np.argmax(urms))
    pk = CubicSpline(y[k - 3:k + 4], urms[k - 3:k + 4])
    ypk = float(pk.derivative().roots(extrapolate=False)[0])
    out.update(u_rms_peak=float(pk(ypk)), u_rms_peak_y_plus=ypk * RE_TAU)
    k = int(np.argmin(uv))
    pk = CubicSpline(yv[k - 3:k + 4], uv[k - 3:k + 4])
    ypk = float(pk.derivative().roots(extrapolate=False)[0])
    out.update(reynolds_shear_stress_peak=float(-pk(ypk)), reynolds_shear_stress_peak_y_plus=ypk * RE_TAU)
    return {k: float(f"{v:.8g}") for k, v in out.items()}


def main():
    s_info, S = read("Chan180_S2a_basic_few.txt", {1: "y+", 2: "<u1>", 5: "RMS(u1)", 6: "RMS(u2)", 7: "RMS(u3)", 9: "<u1' u2'>"})
    u_info, FU = read("Chan180_FD2_basic_u.txt", {1: "y+", 2: "<u>", 3: "rms(u)"})
    v_info, FV = read("Chan180_FD2_basic_v.txt", {1: "y+", 3: "rms(v)", 6: "<u'v'>"})
    w_info, FW = read("Chan180_FD2_basic_w.txt", {1: "y+", 3: "rms(w)"})
    if not (u_info == v_info == w_info):
        raise SystemExit("the three FD2 files do not describe the same simulation")
    if not np.allclose(FU[:, 0], FW[:, 0]):
        raise SystemExit("FD2: u and w are not on the same points")
    ys, yc, yf = S[:, 0] / RE_TAU, FU[:, 0] / RE_TAU, FV[:, 0] / RE_TAU
    if not (ys.max() <= 1.0 + 1e-9 and yc.max() < 1.0 and yf.max() <= 1.0 + 1e-9 and np.all(np.diff(ys) > 0)):
        raise SystemExit("profiles are expected over one half of the channel, wall to centre plane")
    faces = np.concatenate(([0.0], yf))                  # the staggered points are the cell faces of the u points
    if len(faces) != len(yc) + 1 or not np.allclose(0.5 * (faces[:-1] + faces[1:]), yc, atol=2e-4):
        raise SystemExit("FD2: the staggered points are not the faces of the cells of u")
    spec = scalars(ys, S[:, 1], S[:, 4], ys, S[:, 8])
    fd = scalars(yc, FU[:, 1], FU[:, 2], yf, FV[:, 5], faces)
    names = ("bulk_velocity", "centreline_velocity", "friction_coefficient", "u_rms_peak", "reynolds_shear_stress_peak")
    between = {k: dict(difference=float(f"{fd[k] - spec[k]:.3g}"), percent=float(f"{100 * (fd[k] / spec[k] - 1):.3g}")) for k in names}
    # the largest difference between the two codes along the profile (FD2 put on the spectral points)
    on = lambda y, f: CubicSpline(y, f)(ys[(ys > y[0]) & (ys < y[-1])])          # noqa: E731
    inside = lambda y: (ys > y[0]) & (ys < y[-1])                                # noqa: E731
    prof = dict(mean_velocity=float(np.abs(on(yc, FU[:, 1]) - S[inside(yc), 1]).max()),
                u_rms=float(np.abs(on(yc, FU[:, 2]) - S[inside(yc), 4]).max()),
                v_rms=float(np.abs(on(yf, FV[:, 2]) - S[inside(yf), 5]).max()),
                w_rms=float(np.abs(on(yc, FW[:, 2]) - S[inside(yc), 6]).max()),
                reynolds_shear_stress=float(np.abs(on(yf, FV[:, 5]) - S[inside(yf), 8]).max()))
    ref = dict(
        source=("A.W. Vreman and J.G.M. Kuerten, 'Comparison of direct numerical simulation databases of turbulent channel "
                "flow at Re_tau = 180', Physics of Fluids 26, 015102 (2014). Statistics files as published by the first "
                "author at http://www.vremanresearch.nl/channel.html (databases Chan180_S2 and Chan180_FD2), retrieved "
                "2026-11-14. The flow is that of Kim, Moin and Moser, J. Fluid Mech. 177 (1987), 133-166, and Moser, Kim "
                "and Mansour, Phys. Fluids 11 (1999), 943-945, simulated again with higher resolution and longer averaging."),
        saved_files="reference/vk2014/ holds the four files exactly as downloaded (setup/fetch_vk2014_channel.sh).",
        units=("Lengths in units of the half height H of the channel, velocities in units of the friction velocity that "
               "the driving force fixes, sqrt(force x H) = 1 (the files' header: H=1; nu=1/180; forcing=1). y_plus = 180 y. "
               "Profiles cover one half of the channel, from the wall (y = 0) to the centre plane (y = 1)."),
        setup=dict(friction_reynolds_number=RE_TAU, domain=s_info["domain"],
                   spectral=dict(database="Chan180_S2 (file Chan180_S2a_basic_few.txt)", method="Fourier-Chebyshev spectral",
                                 resolution=s_info["resolution"], averaging_time=s_info["averaging_time"],
                                 computed_u_tau=s_info["computed_u_tau"]),
                   finite_difference=dict(database="Chan180_FD2 (files Chan180_FD2_basic_u/v/w.txt)",
                                          method="finite differences on a staggered grid",
                                          resolution=u_info["resolution"], averaging_time=u_info["averaging_time"],
                                          computed_u_tau=u_info["computed_u_tau"])),
        how_computed=("Scalars computed by setup/parse_vk2014_channel.py from the profiles: bulk velocity = integral of the "
                      "mean velocity over the half channel (cubic spline through the points, wall added); "
                      "bulk_velocity_cell_sum = the staggered code's own rule, cell value times cell height; friction "
                      "coefficient = 2 / bulk^2 (wall shear stress over half the density times the bulk velocity squared, "
                      "with the exact wall shear of the settled flow); bulk Reynolds number = bulk velocity x 2 H / nu; "
                      "peaks from a cubic spline through the seven points around the largest value."),
        spectral=spec, finite_difference=fd,
        uncertainty=dict(
            note=("The reference's own uncertainty, as numbers. (1) Two independent codes: finite_difference minus spectral, "
                  "for the scalars and as the largest difference along each profile (velocity units). (2) Statistical: "
                  "the mean wall shear of a settled flow equals the driving force exactly, so computed_u_tau differs "
                  "from 1 only by what 200 time units of averaging leave. Neither is a bound on the error of the reference."),
            between_codes=between, largest_profile_difference={k: float(f"{v:.3g}") for k, v in prof.items()},
            wall_shear_deficit=dict(spectral=float(f"{1 - s_info['computed_u_tau'] ** 2:.3g}"),
                                    finite_difference=float(f"{1 - u_info['computed_u_tau'] ** 2:.3g}")),
            stated_in_paper=PAPER),
        profiles=dict(
            spectral=dict(y_plus=lst(S[:, 0]), mean_velocity=lst(S[:, 1]), u_rms=lst(S[:, 4]), v_rms=lst(S[:, 5]),
                          w_rms=lst(S[:, 6]), reynolds_shear_stress=lst(S[:, 8])),
            finite_difference=dict(y_plus=lst(FU[:, 0]), mean_velocity=lst(FU[:, 1]), u_rms=lst(FU[:, 2]), w_rms=lst(FW[:, 2]),
                                   y_plus_staggered=lst(FV[:, 0]), v_rms=lst(FV[:, 2]), reynolds_shear_stress=lst(FV[:, 5])),
            note="reynolds_shear_stress is the mean of u'v' (negative in this half of the channel).")
    )
    out = DIR / "vk2014_chan180.yaml"
    out.write_text("# Reference data of Vreman and Kuerten (2014), parsed by code (setup/parse_vk2014_channel.py) from the files\n"
                   "# in reference/vk2014/. Do not edit by hand.\n"
                   + yaml.safe_dump(ref, sort_keys=False, width=120, default_flow_style=None), encoding="utf-8", newline="\n")
    print("points: spectral", len(ys), " finite difference", len(yc), "+", len(yf), "staggered")
    print("spectral", spec)
    print("finite difference", fd)
    print("between codes", between)
    print("largest profile differences", prof)
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
