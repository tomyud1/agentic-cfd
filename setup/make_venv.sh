#!/bin/bash
# Create the Python environment for the pipeline (inside WSL, on the Linux filesystem).
# Needs python3-venv:  sudo apt-get install -y python3-venv
set -euo pipefail
VENV="$HOME/venvs/cfd"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -q -r "$(dirname "$0")/../requirements.txt"
"$VENV/bin/python" -c "import numpy, scipy, sympy, matplotlib, yaml, jinja2; print('venv OK:', numpy.__version__, sympy.__version__, matplotlib.__version__)"
