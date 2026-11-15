#!/bin/bash
# Run rung 4 of the compressible ladder with its output in a log that can be watched while it runs:
#   bash gpucfd/tests/run_corner.sh [arguments of gpucfd.tests.test_euler_corner]
cd "$(dirname "$0")/../.." || exit 1
tag=$(echo "$*" | tr ' ' '_')
log="gpucfd/tests/results/test_euler_corner${tag:+_$tag}.log"
"${PY:-$HOME/venvs/cfd/bin/python}" -u -m gpucfd.tests.test_euler_corner "$@" > "$log" 2>&1
echo "exit $?" >> "$log"
