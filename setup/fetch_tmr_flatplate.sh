#!/bin/bash
# Fetch the reference data of case 7 (turbulent flat plate, Spalart-Allmaras) from NASA's Turbulence
# Modeling Resource, case VERIF/2DZP. Run from anywhere:  bash setup/fetch_tmr_flatplate.sh [grids]
# Saves the pages and data files as published into cases/turbulent_flat_plate_sa/reference/tmr/, then
# parses them into reference/tmr_2dzp_sa.yaml (setup/parse_tmr_flatplate.py).
# With the argument "grids" it also downloads NASA's grid archive (all formats) and keeps the five 2D
# PLOT3D grids, gzipped, in reference/grids/.
cd "$(dirname "$0")/.." || exit 1
BASE="https://tmbwg.github.io/turbmodels"
OUT=cases/turbulent_flat_plate_sa/reference/tmr
mkdir -p "$OUT"
get() { curl -sSLf -m 60 -A "curl" -o "$OUT/$2" "$BASE/$1" && echo "$(stat -c %s "$OUT/$2") $2" || echo "FAILED $1"; }
get flatplate.html flatplate.html
get flatplate_sa.html flatplate_sa.html
get flatplate_grids.html flatplate_grids.html
get spalart.html spalart.html
get openfoam_issues.html openfoam_issues.html
get FlatPlate/Grids/plateBCpic.jpg plateBCpic.jpg
for f in cf_convergence.dat drag_convergence.dat cf_plate.dat mut_0.97.dat find_peak_mut.dat "flatplate_u+y+.dat" \
         "u+y+theory.dat" flatplate_u.dat cfl3d_typical.inp fun3d.nml_typical; do
  get "FlatPlate/SA/$f" "$f"
done
get FlatPlate_validation/cf_incomp_results_sa.dat cf_incomp_results_sa.dat
if [ "$1" = "grids" ]; then
  G=cases/turbulent_flat_plate_sa/reference/grids
  T=$(mktemp -d)
  curl -sSLf -m 600 -A "curl" -o "$T/grids.zip" "https://www.nasa.gov/wp-content/uploads/2026/02/flatplategrids-grids.zip" \
    && echo "$(stat -c %s "$T/grids.zip") grids.zip" || { echo "FAILED grids.zip"; exit 1; }
  mkdir -p "$G" "$T/x"
  unzip -q -o "$T/grids.zip" -d "$T/x"
  find "$T/x" -name "*.p2dfmt*" | while read -r f; do
    b=$(basename "$f")
    case "$b" in *.gz) cp "$f" "$G/$b" ;; *) gzip -9 -n -c "$f" > "$G/$b.gz" ;; esac
  done
  ls -l "$G"
  rm -rf "$T"
fi
