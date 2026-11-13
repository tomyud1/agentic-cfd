"""A published structured grid as an OpenFOAM mesh, point for point.

Benchmarks such as NASA's Turbulence Modeling Resource publish their grid families (PLOT3D files), so
that every code runs on the same points and the grid stops being a difference between codes.
blockMesh cannot reproduce such a grid (its grading is geometric); this module writes the mesh files
directly from the grid's points:

    x, y = read_plot3d_2d("reference/grids/flatplate_clust2_4levelsdown_35x25.p2dfmt.gz")
    info = write_polymesh(case_dir, x, y, patches={
        "inlet":    ("i-", None),              # the whole side i = 0
        "outlet":   ("i+", None),
        "top":      ("j+", None),
        "symmetry": ("j-", (0, 4)),            # cells 0..3 along that side
        "plate":    ("j-", (4, None)),         # cells 4..end
    }, types={"plate": "wall", "symmetry": "symmetryPlane"})

x[j, i], y[j, i] are the points of one block, i along the first index of the file, j along the second.
The mesh is one cell thick in z with an `empty` patch front and back (a 2D case). Cell (i, j) has the
number i + ni * j, so fields can be reshaped to (nj, ni) without searching.

Two things to know:
- Small wall cells need many digits. OpenFOAM writes 6 by default; a wall cell of 5e-7 on a domain of
  size 2 needs 8 just to exist and more to be accurate. write_polymesh writes 17 digits and reports
  `digits_needed`; give `writePrecision` in controlDict at least that (anything that writes the mesh
  again, such as decomposePar or renumberMesh, uses it), and check with `checkMesh` after such a step.
- Check the result with `checkMesh` (run by cfdlib.foam.run) before the first solver run: it reports
  aspect ratio and non-orthogonality, which for these grids are large by design near the wall.
"""
import gzip
from pathlib import Path

import numpy as np

SIDES = ("i-", "i+", "j-", "j+")
HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "constant/polyMesh";
    object      {obj};
}}
"""


def read_plot3d_2d(path):
    """Points of a formatted, single-block 2D PLOT3D grid (.p2dfmt, optionally gzipped): arrays x[j, i], y[j, i]."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        t = f.read().split()
    if int(t[0]) != 1:
        raise ValueError(f"{path}: {t[0]} blocks; only single-block grids are read")
    ni, nj = int(t[1]), int(t[2])
    a = np.array([v.replace("D", "E").replace("d", "e") for v in t[3:]], float)
    if a.size != 2 * ni * nj:
        raise ValueError(f"{path}: {a.size} numbers for a {ni} x {nj} grid (expected {2 * ni * nj}); is it a 3D file?")
    return a[:ni * nj].reshape(nj, ni), a[ni * nj:].reshape(nj, ni)


def index_range(coordinate, lo=None, hi=None, tol=1e-9):
    """The cell range (start, stop) along a side whose cells lie between two values of a point coordinate
    along that side, for the `patches` of write_polymesh. Both values must be points of the grid.
    Example: the plate from x = 0 on: index_range(x[0], lo=0.0)."""
    c = np.asarray(coordinate, float)

    def at(v):
        k = int(np.argmin(np.abs(c - v)))
        if abs(c[k] - v) > tol * max(1.0, abs(v)):
            raise ValueError(f"no grid point at {v}: the nearest is {c[k]}")
        return k
    return (0 if lo is None else at(lo), len(c) - 1 if hi is None else at(hi))


def write_polymesh(case_dir, x, y, patches, types=None, thickness=1.0, empty_name="frontAndBack"):
    """Write constant/polyMesh (points, faces, owner, neighbour, boundary) for the block x[j, i], y[j, i].

    patches: {name: (side, cell range)} with side one of "i-", "i+", "j-", "j+" and the range (start, stop)
      in cells along that side (None: the whole side; stop None: to the end). Every boundary face must
      belong to exactly one patch. types: {name: OpenFOAM patch type}, default "patch".
    Returns dict(cells, ni, nj, patches {name: faces}, min_edge, digits_needed, max_aspect_ratio).
    The grid must be right-handed: j to the left of i (x growing with i and y with j, for example)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 2 or min(x.shape) < 2:
        raise ValueError("x and y must be 2D arrays of the same shape, at least 2 x 2 points")
    npj, npi = x.shape
    ni, nj = npi - 1, npj - 1
    types = dict(types or {})

    # signed cell areas: positive for a right-handed grid, and no folded cell
    area = 0.5 * ((x[1:, 1:] - x[:-1, :-1]) * (y[1:, :-1] - y[:-1, 1:]) - (x[1:, :-1] - x[:-1, 1:]) * (y[1:, 1:] - y[:-1, :-1]))
    if not (area > 0).all():
        raise ValueError("cells with zero or negative area: the grid is left-handed (swap or reverse an index) or folded")

    def p(i, j, k):
        return i + npi * j + npi * npj * k

    def c(i, j):
        return i + ni * j

    faces, owner, neighbour = [], [], []
    for j in range(nj):                                     # internal faces, in OpenFOAM's order: by owner, then by neighbour
        for i in range(ni):
            if i < ni - 1:
                faces.append((p(i + 1, j, 0), p(i + 1, j + 1, 0), p(i + 1, j + 1, 1), p(i + 1, j, 1)))
                owner.append(c(i, j))
                neighbour.append(c(i + 1, j))
            if j < nj - 1:
                faces.append((p(i, j + 1, 0), p(i, j + 1, 1), p(i + 1, j + 1, 1), p(i + 1, j + 1, 0)))
                owner.append(c(i, j))
                neighbour.append(c(i, j + 1))

    def side_face(side, k):
        if side == "i-":
            return (p(0, k, 0), p(0, k, 1), p(0, k + 1, 1), p(0, k + 1, 0)), c(0, k)
        if side == "i+":
            return (p(ni, k, 0), p(ni, k + 1, 0), p(ni, k + 1, 1), p(ni, k, 1)), c(ni - 1, k)
        if side == "j-":
            return (p(k, 0, 0), p(k + 1, 0, 0), p(k + 1, 0, 1), p(k, 0, 1)), c(k, 0)
        return (p(k, nj, 0), p(k, nj, 1), p(k + 1, nj, 1), p(k + 1, nj, 0)), c(k, nj - 1)

    covered = {s: np.zeros(nj if s[0] == "i" else ni, int) for s in SIDES}
    boundary = []
    for name, (side, rng) in patches.items():
        if side not in SIDES:
            raise ValueError(f"patch {name}: side must be one of {SIDES}")
        n = len(covered[side])
        a, b = (0, n) if rng is None else (rng[0], n if rng[1] is None else rng[1])
        if not 0 <= a < b <= n:
            raise ValueError(f"patch {name}: cell range {a}..{b} is outside 0..{n} on side {side}")
        start = len(faces)
        for k in range(a, b):
            f, o = side_face(side, k)
            faces.append(f)
            owner.append(o)
        covered[side][a:b] += 1
        boundary.append((name, types.get(name, "patch"), b - a, start))
    for s, cov in covered.items():
        if (cov != 1).any():
            k = int(np.argmax(cov != 1))
            raise ValueError(f"side {s}: cell {k} is in {int(cov[k])} patches; every boundary face needs exactly one")
    start = len(faces)
    for j in range(nj):
        for i in range(ni):
            faces.append((p(i, j, 0), p(i, j + 1, 0), p(i + 1, j + 1, 0), p(i + 1, j, 0)))        # back, z = 0
            owner.append(c(i, j))
    for j in range(nj):
        for i in range(ni):
            faces.append((p(i, j, 1), p(i + 1, j, 1), p(i + 1, j + 1, 1), p(i, j + 1, 1)))        # front, z = thickness
            owner.append(c(i, j))
    boundary.append((empty_name, "empty", 2 * ni * nj, start))

    d = Path(case_dir) / "constant" / "polyMesh"
    d.mkdir(parents=True, exist_ok=True)
    pts = [f"({xx:.17g} {yy:.17g} {z:.17g})" for z in (0.0, float(thickness)) for xx, yy in zip(x.ravel(), y.ravel())]
    (d / "points").write_text(HEADER.format(cls="vectorField", obj="points") + f"\n{len(pts)}\n(\n" + "\n".join(pts) + "\n)\n")
    (d / "faces").write_text(HEADER.format(cls="faceList", obj="faces") + f"\n{len(faces)}\n(\n"
                             + "\n".join("4(%d %d %d %d)" % f for f in faces) + "\n)\n")
    note = f"nPoints: {len(pts)} nCells: {ni * nj} nFaces: {len(faces)} nInternalFaces: {len(neighbour)}"
    for obj, arr in (("owner", owner), ("neighbour", neighbour)):
        head = HEADER.format(cls="labelList", obj=obj).replace("    location", f'    note        "{note}";\n    location')
        (d / obj).write_text(head + f"\n{len(arr)}\n(\n" + "\n".join(map(str, arr)) + "\n)\n")
    blocks = []
    for name, typ, n, st in boundary:
        extra = "        inGroups        1(wall);\n" if typ == "wall" else ""
        blocks.append(f"    {name}\n    {{\n        type            {typ};\n{extra}        nFaces          {n};\n"
                      f"        startFace       {st};\n    }}")
    (d / "boundary").write_text(HEADER.format(cls="polyBoundaryMesh", obj="boundary") + f"\n{len(boundary)}\n(\n"
                                + "\n".join(blocks) + "\n)\n")

    ex = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))    # edges along i, shape (npj, ni)
    ey = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))    # edges along j, shape (nj, npi)
    li, lj = 0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[:, 1:] + ey[:, :-1])
    min_edge = float(min(ex.min(), ey.min()))
    size = float(max(np.abs(x).max(), np.abs(y).max(), 1e-300))
    # digits for the smallest edge to keep 3 significant digits when coordinates are written
    digits = int(np.ceil(np.log10(size / min_edge))) + 3
    return dict(cells=ni * nj, ni=ni, nj=nj, patches={b[0]: b[2] for b in boundary}, min_edge=min_edge,
                digits_needed=digits, max_aspect_ratio=float(np.maximum(li / lj, lj / li).max()))
