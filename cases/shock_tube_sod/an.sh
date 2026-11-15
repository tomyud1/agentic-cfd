#!/bin/bash
# Analysis and report from Windows: bash cases/shock_tube_sod/an.sh [png dir] (inside WSL).
# Log: results/shock_tube_sod/analyze.log. With a directory, every figure is also written there as a PNG.
cd "/mnt/d/Projects/Agentic CFD"
if [ -n "$1" ]; then export CFD_FIG_PNG_DIR="$1"; fi
/home/tomyud/venvs/cfd/bin/python -u -m cases.shock_tube_sod.analyze > results/shock_tube_sod/analyze.log 2>&1
echo "exit $?" >> results/shock_tube_sod/analyze.log
