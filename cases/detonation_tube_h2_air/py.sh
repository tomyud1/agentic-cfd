#!/bin/bash
# Run a module of this case inside WSL from the repository root:  bash cases/detonation_tube_h2_air/py.sh run preflight
# (from Windows: wsl -d Ubuntu-24.04 -- bash "/mnt/d/Projects/Agentic CFD/cases/detonation_tube_h2_air/py.sh" <module> [args])
# LOG=<name> names the log file (default: the module's name), so that two calls of one module do not share a log.
cd "$(dirname "$0")/../.." || exit 1
mod=$1; shift
mkdir -p results/detonation_tube_h2_air/logs
~/venvs/cfd/bin/python -u -m cases.detonation_tube_h2_air.$mod "$@" 2>&1 | tee results/detonation_tube_h2_air/logs/${LOG:-$mod}.log
