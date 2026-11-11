"""Sanity check of the cavity result: Reynolds number and the vertical-centreline u profile.

At Re=10 the flow is close to Stokes (creeping) flow. Known reference for a Stokes
lid-driven cavity: min u/U on the vertical centreline is about -0.21, near y/L ~ 0.53.
"""
import re
import sys
from pathlib import Path

case = Path(sys.argv[1])
latest = max((d for d in case.iterdir() if re.fullmatch(r"[0-9.]+", d.name) and d.name != "0"),
             key=lambda d: float(d.name))


def read_vectors(path):
    text = path.read_text()
    body = text[text.index("internalField"):]
    n = int(re.search(r"\n(\d+)\n\(", body).group(1))
    vecs = re.findall(r"\(([-0-9.eE+]+) ([-0-9.eE+]+) ([-0-9.eE+]+)\)", body)[:n]
    return [tuple(map(float, v)) for v in vecs]


props = (case / "constant/transportProperties").read_text() if (case / "constant/transportProperties").exists() \
    else (case / "constant/physicalProperties").read_text()
nu = float(re.search(r"nu\s+(?:\[[^\]]*\]\s*)?([0-9.eE+-]+)", props).group(1))
U_lid, L = 1.0, 0.1
print(f"Latest time: {latest.name}   nu = {nu}   Re = U*L/nu = {U_lid * L / nu:.0f}")

U = read_vectors(latest / "U")
C = read_vectors(latest / "C")

# Cells closest to the vertical centreline x = L/2
xs = sorted({round(c[0], 6) for c in C})
xc = min(xs, key=lambda x: abs(x - L / 2))
line = sorted((c[1] / L, u[0] / U_lid) for c, u in zip(C, U) if abs(c[0] - xc) < 1e-6)

print(f"\nVertical line at x/L = {xc / L:.3f}")
print("  y/L     u/U")
for y, u in line:
    print(f"  {y:.3f}  {u:+.4f}")
ymin, umin = min(line, key=lambda p: p[1])
print(f"\nMin u/U = {umin:+.3f} at y/L = {ymin:.3f}   (Stokes reference: about -0.21 at y/L ~ 0.53)")
