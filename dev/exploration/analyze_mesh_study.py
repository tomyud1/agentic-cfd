"""Mesh-convergence analysis of the cavity study.

For each N x N mesh: u/U on the vertical centreline x = L/2 (average of the two cell
columns either side of it), its minimum (parabolic fit through the 3 lowest points),
and the net flow across the line (mass conservation check).
Then Richardson extrapolation on the three finest meshes estimates the mesh-independent
value and the observed order of accuracy.

Usage: python3 -I analyze_mesh_study.py <meshStudy dir>
"""
import json
import math
import re
import sys
from pathlib import Path

L, U_LID = 0.1, 1.0


def read_vectors(path):
    text = path.read_text()
    body = text[text.index("internalField"):]
    n = int(re.search(r"\n(\d+)\n\(", body).group(1))
    vecs = re.findall(r"\(([-0-9.eE+]+) ([-0-9.eE+]+) ([-0-9.eE+]+)\)", body)[:n]
    return [tuple(map(float, v)) for v in vecs]


def centreline(case, N):
    t = max((d for d in case.iterdir() if re.fullmatch(r"[0-9.]+", d.name) and d.name != "0"),
            key=lambda d: float(d.name))
    C, U = read_vectors(t / "C"), read_vectors(t / "U")
    assert len(C) == len(U) == N * N, f"{case}: {len(C)} centres, {len(U)} velocities, expected {N*N}"
    h = L / N
    # Coordinates are written to only ~6 digits, so locate cells by grid index, not exact x.
    cols = {}
    for c, u in zip(C, U):
        i, j = int(c[0] / h), int(c[1] / h)
        if i in (N // 2 - 1, N // 2):
            cols.setdefault(j, []).append(u[0])
    assert len(cols) == N and all(len(v) == 2 for v in cols.values()), f"{case}: bad centreline"
    return float(t.name), [((j + 0.5) / N, sum(v) / 2 / U_LID) for j, v in sorted(cols.items())]


def parabolic_min(line):
    i = min(range(1, len(line) - 1), key=lambda k: line[k][1])
    (y0, f0), (y1, f1), (y2, f2) = line[i - 1:i + 2]
    h = y1 - y0
    denom = f0 - 2 * f1 + f2
    shift = 0.5 * h * (f0 - f2) / denom
    return y1 + shift, f1 - 0.25 * (f0 - f2) * shift / h


def richardson(f_coarse, f_mid, f_fine, r=2.0):
    p = math.log(abs((f_coarse - f_mid) / (f_mid - f_fine))) / math.log(r)
    f_exact = f_fine + (f_fine - f_mid) / (r ** p - 1)
    return p, f_exact


root = Path(sys.argv[1])
rows = []
for case in sorted(root.glob("N*"), key=lambda d: int(d.name[1:])):
    if not (case / "log.cellCentres").exists():
        continue
    N = int(case.name[1:])
    t, line = centreline(case, N)
    y_min, u_min = parabolic_min(line)
    net = sum(u for _, u in line) / N            # integral of u/U d(y/L)
    rows.append(dict(N=N, t_end=t, u_min=u_min, y_min=y_min, net_flow=net))

print(f"{'N':>5} {'cells':>7} {'min u/U':>9} {'at y/L':>7} {'net flow':>10}")
for r in rows:
    print(f"{r['N']:>5} {r['N']**2:>7} {r['u_min']:+9.5f} {r['y_min']:7.4f} {r['net_flow']:+10.2e}")

result = dict(rows=rows)
if len(rows) >= 3:
    a, b, c = rows[-3:]
    p, u_ex = richardson(a["u_min"], b["u_min"], c["u_min"])
    print(f"\nRichardson ({a['N']}/{b['N']}/{c['N']}): observed order p = {p:.2f}, "
          f"mesh-independent min u/U = {u_ex:+.5f}")
    if not 0.5 < p < 4:
        print("  WARNING: observed order outside 0.5-4, extrapolated value is unreliable")
    print("Error of each mesh vs that estimate:")
    for r in rows:
        print(f"  N={r['N']:>4}: {abs(r['u_min'] - u_ex) / abs(u_ex) * 100:6.2f} %")
    result.update(order=p, u_min_extrapolated=u_ex)

(root / "convergence.json").write_text(json.dumps(result, indent=2))
