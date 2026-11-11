#!/bin/bash
# Report which Python packages and tools are available, and find Poiseuille-like tutorials.
source /usr/lib/openfoam/openfoam2606/etc/bashrc
echo "== Python =="; python3 --version
for m in numpy scipy sympy matplotlib yaml jinja2 pandas; do
  python3 -c "import $m; print('$m', getattr($m,'__version__','ok'))" 2>/dev/null || echo "$m MISSING"
done
python3 -c "import venv, ensurepip" 2>/dev/null && echo "venv+ensurepip OK" || echo "venv/ensurepip MISSING"
for t in pip3 uv quarto pdflatex pandoc; do printf "%-9s %s\n" "$t" "$(command -v $t || echo MISSING)"; done
echo; echo "== Poiseuille-like tutorials =="
find "$FOAM_TUTORIALS" -maxdepth 4 -type d -iname "*poiseuille*"
echo; echo "== Tutorials using a momentum source =="
grep -rl "SemiImplicitSource\|meanVelocityForce" "$FOAM_TUTORIALS" --include=fvOptions --include=fvModels 2>/dev/null | head -10
