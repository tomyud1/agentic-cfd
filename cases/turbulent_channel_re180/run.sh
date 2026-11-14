#!/bin/bash
# Launch a command of run.py inside WSL from the repository root, with its output in a log file of the
# project (so that it can be watched from Windows):  bash cases/turbulent_channel_re180/run.sh preflight
cd "/mnt/d/Projects/Agentic CFD" || exit 1
mkdir -p results/turbulent_channel_re180/logs
/home/tomyud/venvs/cfd/bin/python -u -m cases.turbulent_channel_re180.run "$@" \
    > "results/turbulent_channel_re180/logs/console_$1.log" 2>&1
echo "EXIT $?" >> "results/turbulent_channel_re180/logs/console_$1.log"
