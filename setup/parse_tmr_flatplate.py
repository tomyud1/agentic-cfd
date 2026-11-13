"""Turn the files saved by setup/fetch_tmr_flatplate.sh into the reference YAML of case 7 (turbulent flat
plate, Spalart-Allmaras, NASA TMR case VERIF/2DZP). Every number comes from the saved files; the script
stops if a page no longer says what the YAML states about the conditions.

    python -m setup.parse_tmr_flatplate          (repo root; needs numpy and yaml)
"""
import gzip
import html
import re
from pathlib import Path

import numpy as np
import yaml

REF = Path("cases/turbulent_flat_plate_sa/reference")
TMR = REF / "tmr"
BASE = "https://tmbwg.github.io/turbmodels"
RETRIEVED = "2026-11-13"


def text(name):
    return (TMR / name).read_text(encoding="utf-8", errors="replace")


def plain(name):
    t = re.sub(r'<img[^>]*alt="([^"]*)"[^>]*>', r" [\1] ", text(name))
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))


def zones(name):
    """A Tecplot-style .dat file: {zone title: rows of floats} (a file without zones: {'': rows})."""
    out, cur = {}, ""
    for line in text(name).splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.lower().startswith("variables"):
            continue
        m = re.match(r'(?i)zone.*?t\s*=\s*"([^"]*)"', s)
        if m:
            cur = m.group(1).strip()
            continue
        out.setdefault(cur, []).append([float(v.replace("D", "E")) for v in s.split()])
    return out


def convergence(name, key):
    return {z: [dict(cells=int(r[0]), h=r[2], **{key: r[3]}) for r in rows] for z, rows in zones(name).items()}


def gci_tables():
    """The four uncertainty tables of the SA page, in page order: Cf and drag (CFL3D, FUN3D), then the
    two tables of FUN3D on triangles."""
    out = []
    for t in re.findall(r"(?s)<table.*?</table>", text("flatplate_sa.html")):
        if "apparent order" not in t:
            continue
        rows = []
        for tr in re.findall(r"(?s)<tr.*?</tr>", t)[1:]:
            c = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", x))).strip()
                 for x in re.findall(r"(?s)<t[dh][^>]*>(.*?)</t[dh]>", tr)]
            if len(c) == 5:
                rows.append(dict(code=c[0], order=float(c[1]), ea21_percent=float(c[2].rstrip("%")),
                                 eext21_percent=float(c[3].rstrip("%")), gci_fine_percent=float(c[4].rstrip("%"))))
        out.append(rows)
    return out


def grid(path):
    t = gzip.open(path, "rt").read().split()
    ni, nj = int(t[1]), int(t[2])
    a = np.array(t[3:], float)
    assert int(t[0]) == 1 and a.size == 2 * ni * nj, path
    return a[:ni * nj].reshape(nj, ni), a[ni * nj:].reshape(nj, ni)


def main():
    intro, sa, grids_page, spalart = (plain(n) for n in ("flatplate.html", "flatplate_sa.html", "flatplate_grids.html",
                                                          "spalart.html"))
    # the statements the YAML makes about the case must still be on the saved pages
    for page, phrase in ((intro, "M = 0.2"), (intro, "Re = 5 million based on length \"1\" of the grid"),
                         (intro, "solid wall of the grid extended from x = 0 to x = 2"),
                         (intro, "maximum boundary layer thickness is about 0.03 L"),
                         (intro, "this is a compressible flow verification case"),
                         (sa, r"[\tilde \nu_{farfield} \geq 3 \nu_{\infty}]"),
                         (sa, "first-order upwinding for the advective terms of the turbulence model"),
                         (sa, "incorrect reference area (of 1 instead of 2)"),
                         (sa, "SA-noft2 yields essentially identical results for this case as SA"),
                         (grids_page, "Each coarser grid is exactly every-other-point of the next finer grid"),
                         (spalart, r"[\tilde \nu_{farfield} = 3 \nu_{\infty}]")):
        assert phrase in page, f"not found on the saved page: {phrase}"
    assert "tur10 3.0" in text("cfl3d_typical.inp") and "0.2000" in text("cfl3d_typical.inp")

    cf, cd, cf_inc = convergence("cf_convergence.dat", "cf"), convergence("drag_convergence.dat", "cd"), \
        convergence("cf_incomp_results_sa.dat", "cf")
    g = gci_tables()
    assert len(g) == 4 and [r["code"] for r in g[0]] == ["CFL3D", "FUN3D"], g
    station = float(re.search(r"x=([0-9.]+) with grid size", text("cf_convergence.dat")).group(1))

    levels, prev = [], None
    for p in sorted((REF / "grids").glob("*.p2dfmt.gz"), key=lambda q: q.stat().st_size):
        x, y = grid(p)
        # a rectangle of straight lines (to the 12 digits of the file): x depends on i only, y on j only
        assert np.abs(x - x[0]).max() < 1e-10 and np.abs(y - y[:, :1]).max() < 1e-10, "grid lines are not straight"
        if prev is not None:      # nested: the coarser grid is every other point of this one
            assert np.abs(x[::2, ::2] - prev[0]).max() < 1e-12 and np.abs(y[::2, ::2] - prev[1]).max() < 1e-12, p
        prev = (x, y)
        xs, ys = x[0], y[:, 0]
        plate = xs >= -1e-12
        i = int(np.argmin(np.abs(xs - station)))
        # the data files label the station 0.970084071; the grid point is 0.970084048 (2e-8 apart)
        assert abs(xs[i] - station) < 1e-6, f"{p.name}: the station is not a grid point (nearest {xs[i]})"
        levels.append(dict(file=f"grids/{p.name}", points=[int(x.shape[1]), int(x.shape[0])],
                           cells=int((x.shape[1] - 1) * (x.shape[0] - 1)), points_on_plate=int(plate.sum()),
                           x_min=float(xs[0]), x_max=float(xs[-1]), y_max=float(ys[-1]),
                           first_cell_height=float(ys[1] - ys[0]),
                           cell_length_at_plate_start=float(xs[plate][1] - xs[plate][0]),
                           grid_point_at_station=float(xs[i]),
                           wall_face_centres_next_to_station=[float(0.5 * (xs[i - 1] + xs[i])), float(0.5 * (xs[i] + xs[i + 1]))]))
    assert [lv["cells"] for lv in levels] == sorted(r["cells"] for r in cf["CFL3D"]), "grid sizes differ from the data files"

    fine = lambda rows: next(r for r in rows if r["cells"] == levels[-1]["cells"])
    inc_name = next(k for k in cf_inc if "FUN3D" in k)
    ref = dict(
        source=(f"NASA Langley Turbulence Modeling Resource (run by the AIAA Turbulence Model Benchmarking Working "
                f"Group), case VERIF/2DZP '2D Zero Pressure Gradient Flat Plate Verification Case', pages "
                f"{BASE}/flatplate.html, {BASE}/flatplate_sa.html, {BASE}/flatplate_grids.html, retrieved {RETRIEVED}. "
                f"Model: Spalart, Allmaras, 'A One-Equation Turbulence Model for Aerodynamic Flows', Recherche "
                f"Aerospatiale 1, 1994, 5-21, in the form on {BASE}/spalart.html. The data is 'provided as-is'."),
        saved_files="reference/tmr/ holds the pages and data files exactly as downloaded (setup/fetch_tmr_flatplate.sh).",
        conditions=dict(
            mach=0.2, reynolds_per_unit_length=5.0e6, reference_length=1.0, plate=[0.0, 2.0],
            reference_area_for_drag=2.0, reference_temperature_rankine=540.0,
            domain=dict(x=[levels[-1]["x_min"], levels[-1]["x_max"]], y=[0.0, levels[-1]["y_max"]]),
            boundaries_from_figure=dict(
                _note="read from the figure plateBCpic.jpg by a person, not parsed (the page gives the boundary conditions only there)",
                inflow="total pressure Pt/Pref = 1.02828, total temperature Tt/Tref = 1.008, 1 quantity from interior",
                top="farfield Riemann BC", outflow="P/Pref = 1.0, other quantities from interior",
                bottom_ahead_of_plate="symmetry (from the inflow to x = 0)", plate="adiabatic solid wall"),
            sa_inflow="nu_tilde = 3 nu_infinity (page: 'nu_tilde_farfield >= 3 nu_inf'; CFL3D input: tur10 3.0)",
            intro_note="The page says variations of the inflow, top and outflow conditions may also work and give similar results; "
                       "moving the top from y = L to y = 0.48 L changed drag and Cf at x = 0.97 by less than 0.2 %."),
        reference_codes=("CFL3D (cell-centred, structured) and FUN3D (node-centred, unstructured, same hexahedral grids). Both "
                         "compressible, Roe flux-difference splitting with UMUSCL, full Navier-Stokes, first-order upwind for the "
                         "advective terms of the turbulence model. Not converged to machine zero (CFL3D density residual below 1e-13)."),
        station_x=station,
        station_note=("The station is a grid point of all five grids, so a cell-centred code has no wall face centred on it: the "
                      "two nearest wall-face centres are listed per grid. The page does not say how CFL3D evaluated Cf there."),
        definitions=dict(cf="wall shear stress / (0.5 rho_inf U_inf^2) at the station", cd="drag of the plate / (0.5 rho_inf U_inf^2 "
                         "times the reference area 2), friction only"),
        grids=levels,
        compressible=dict(
            _note="M = 0.2. The values the page presents as the expected results.",
            cf_at_station=cf, cd=cd,
            uncertainty_finest_three_grids=dict(cf_at_station=g[0], cd=g[1]),
            cf_between_codes_finest_grid_percent=100 * (fine(cf["FUN3D"])["cf"] / fine(cf["CFL3D"])["cf"] - 1),
            cd_between_codes_finest_grid_percent=100 * (fine(cd["FUN3D"])["cd"] / fine(cd["CFL3D"])["cd"] - 1)),
        incompressible=dict(
            _note=("From cf_incomp_results_sa.dat, which the SA page offers 'to give an indication of the magnitude of the "
                   "difference between compressible (M=0.2) and incompressible results'. Cf at the station only: no drag, no "
                   "profiles, no uncertainty table."),
            cf_at_station=cf_inc,
            fun3d_incompressible_over_cfl3d_compressible_finest_grid_percent=100 * (fine(cf_inc[inc_name])["cf"]
                                                                                    / fine(cf["CFL3D"])["cf"] - 1),
            between_incompressible_codes_finest_grid_percent=100 * (
                fine(next(v for k, v in cf_inc.items() if k != inc_name))["cf"] / fine(cf_inc[inc_name])["cf"] - 1)),
        profiles=dict(
            _note="Finest grid (545 x 385), compressible. Read the files in reference/tmr/ with a Tecplot-zone reader.",
            files={"cf_plate.dat": "Cf along the plate, CFL3D and FUN3D",
                   "flatplate_u.dat": "u / U_inf against y at x = 0.97008 and x = 1.90334 (CFL3D)",
                   "flatplate_u+y+.dat": "u+ against log10(y+) at the same two stations (CFL3D)",
                   "u+y+theory.dat": "law of the wall, kappa = 0.41, B = 5.0 (White, Viscous Fluid Flow, 1974, p. 472)",
                   "mut_0.97.dat": "eddy viscosity / freestream laminar viscosity against y at x = 0.97, CFL3D and FUN3D",
                   "find_peak_mut.dat": "largest eddy viscosity ratio against x (CFL3D)"}),
        notes=[
            "The page's main results are compressible (M = 0.2): 'if you run this case with an incompressible code, your results "
            "may be close - but not quite the same - as the grid is refined'.",
            "Drag: 'formal order-property convergence may not be generally achievable', because the skin friction is singular at "
            "the leading edge; the two codes also differ locally at the leading and the trailing edge of the plate.",
            "OpenFOAM: 'As of 2014, the implementation of SA, SST, and Wilcox2006 in the standard version of OpenFOAM did not match "
            "the equations given on this website' (openfoam_issues.html; the page does not track later versions).",
            "Results of SA without the ft2 term (SA-noft2) are 'essentially identical' to SA for this case.",
            "The files give more digits than the codes' convergence supports (SA page).",
        ])
    out = REF / "tmr_2dzp_sa.yaml"
    out.write_text("# Reference data of NASA TMR case VERIF/2DZP with the Spalart-Allmaras model, parsed by code\n"
                   "# (setup/parse_tmr_flatplate.py) from the files in reference/tmr/ and reference/grids/. Do not edit by hand.\n"
                   + yaml.safe_dump(ref, sort_keys=False, width=110), encoding="utf-8")
    print(f"written {out}")
    for lv in levels:
        print(f"  {lv['points'][0]:4d} x {lv['points'][1]:3d}  cells {lv['cells']:6d}  on plate {lv['points_on_plate']:3d}  "
              f"first cell {lv['first_cell_height']:.3e}  wall faces next to the station "
              f"{lv['wall_face_centres_next_to_station'][0]:.4f}, {lv['wall_face_centres_next_to_station'][1]:.4f}")
    print(f"  Cf at x = {station}: CFL3D {fine(cf['CFL3D'])['cf']:.6e}, FUN3D {fine(cf['FUN3D'])['cf']:.6e}, "
          f"FUN3D incompressible {fine(cf_inc[inc_name])['cf']:.6e} "
          f"({ref['incompressible']['fun3d_incompressible_over_cfl3d_compressible_finest_grid_percent']:+.2f} % of CFL3D)")


if __name__ == "__main__":
    main()
