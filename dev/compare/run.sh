#!/bin/bash
# Run one comparison script inside WSL with the pipeline's Python, log next to it.
#   bash dev/compare/run.sh <folder> <script.py> [args]      (log: dev/compare/<folder>/<script>.log)
cd "$(dirname "$0")/../.." || exit 1
folder="$1"; script="$2"; shift 2
log="dev/compare/$folder/${script%.py}${1:+_$1}.log"
PYTHONPATH=. "$HOME/venvs/cfd/bin/python" -u "dev/compare/$folder/$script" "$@" > "$log" 2>&1
echo "exit $?" >> "$log"
tail -n 40 "$log"
