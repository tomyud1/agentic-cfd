#!/bin/bash
# Run the verification ladders (inside WSL, from anywhere): bash gpucfd/run_ladder.sh [incompressible|compressible]
# With no argument both are run. Prints one summary line per test file; details and JSON results in
# gpucfd/tests/results/.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-$HOME/venvs/cfd/bin/python}"
WHAT="${1:-all}"
fails=0
one() {   # test module, then its arguments
  local t="$1"; shift
  out=$($PY -m gpucfd.tests.$t "$@" 2>&1)
  echo "$out" | grep -E "^FAIL" || true
  echo "$t${1:+ $*}: $(echo "$out" | tail -1)"
  echo "$out" | tail -1 | grep -q "^0 failed" || fails=$((fails + 1))
}
if [ "$WHAT" = all ] || [ "$WHAT" = incompressible ]; then
  for t in test_poisson test_ns test_walls test_open test_bodies test_wall_forces test_steady test_stretched test_stretched_bodies test_imex_bodies test_reproducible; do
    one $t
  done
fi
if [ "$WHAT" = all ] || [ "$WHAT" = compressible ]; then
  # The compressible solver (gpucfd/COMPRESSIBLE.md). The corner runs to 400 x 400 here (about 15 s; result file
  # test_euler_corner_to400.json); the recorded run to 1600 x 1600 is  bash gpucfd/tests/run_corner.sh 1600.
  one test_euler_kernels
  one test_euler_smooth
  one test_euler_sod
  one test_euler_corner 400
  one test_euler_speed
fi
exit $fails
