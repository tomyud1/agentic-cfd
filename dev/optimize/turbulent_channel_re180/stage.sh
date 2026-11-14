#!/bin/bash
# Run a command of stage.py inside WSL from the repository root (so that it can be started from Windows):
#   bash dev/optimize/turbulent_channel_re180/stage.sh pre z15
# The script logs to dev/optimize/turbulent_channel_re180/logs/<variant>.log itself.
cd "$(dirname "$0")/../../.." || exit 1
PYTHONPATH=. "${PY:-$HOME/venvs/cfd/bin/python}" -u dev/optimize/turbulent_channel_re180/stage.py "$@" 2>&1
echo "EXIT $?"
