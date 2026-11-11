#!/bin/bash
# Pressure solves and OGL timings of a test run: bash setup/gpu/show_log.sh <run name>
c="$HOME/OpenFOAM/tomyud-v2606/run/ogl_test/$1"
grep "Solving for p" "$c/log.icoFoam" | head -4; grep "Solving for p" "$c/log.icoFoam" | tail -2
grep "OGL LOG\]\[Proc: 0\]" "$c/log.icoFoam" | tail -12
grep -E "^ExecutionTime" "$c/log.icoFoam" | tail -1
