#!/bin/bash
# Run a module of this case inside WSL from the repository root:  bash cases/h2_air_ignition_box/py.sh run preflight
# (from Windows: wsl -d Ubuntu-24.04 -- bash "/mnt/d/Projects/Agentic CFD/cases/h2_air_ignition_box/py.sh" <module> [args])
cd "$(dirname "$0")/../.." || exit 1
mod=$1; shift
mkdir -p results/h2_air_ignition_box/logs
~/venvs/cfd/bin/python -u -m cases.h2_air_ignition_box.$mod "$@" 2>&1 | tee results/h2_air_ignition_box/logs/$mod.log
