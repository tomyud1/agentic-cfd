#!/bin/bash
# Run the whole verification ladder (inside WSL, from anywhere): bash gpucfd/run_ladder.sh
# Prints one summary line per test file; details and JSON results in gpucfd/tests/results/.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-$HOME/venvs/cfd/bin/python}"
fails=0
for t in test_poisson test_ns test_walls test_open test_bodies test_steady test_reproducible; do
  out=$($PY -m gpucfd.tests.$t 2>&1)
  echo "$out" | grep -E "^FAIL" || true
  echo "$t: $(echo "$out" | tail -1)"
  echo "$out" | tail -1 | grep -q "^0 failed" || fails=$((fails + 1))
done
exit $fails
