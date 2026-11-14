#!/bin/bash
# Run a command of openfoam.py inside WSL from the repository root (so that it can be started from Windows):
#   bash dev/compare/turbulent_channel_re180/openfoam.sh tune
cd "$(dirname "$0")/../../.." || exit 1
for c in "$@"; do
  PYTHONPATH=. "${PY:-$HOME/venvs/cfd/bin/python}" -u dev/compare/turbulent_channel_re180/openfoam.py $c 2>&1
  code=$?
  echo "EXIT $code ($c)"
  [ $code -eq 0 ] || exit $code
done
