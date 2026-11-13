"""Tests of cfdlib/structgrid.py: a published structured grid written as an OpenFOAM mesh.

    PYTHONPATH=. python tests/test_structgrid.py          (inside WSL from the project root; needs OpenFOAM)

1. A small made-up grid: face counts, refused inputs (a boundary face in no patch or in two, a
   left-handed grid).
2. NASA's flat-plate grids (case 7, reference/grids/), coarsest, middle and finest: OpenFOAM's checkMesh
   accepts the topology, the cell count and the patch sizes are right, the mesh is exactly orthogonal,
   and the points OpenFOAM reads back are the file's to 15 digits (the first cell is 5e-7 high).
   checkMesh's one complaint, the aspect ratio of the wall cells, is the grid's design and is expected.
"""
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from cfdlib import foam, structgrid

FAILS = []
GRIDS = Path("cases/turbulent_flat_plate_sa/reference/grids")
CONTROL = """FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 16; timeFormat general; timePrecision 6;
"""
MINIMAL = {
    "fvSchemes": "ddtSchemes { default steadyState; } gradSchemes { default Gauss linear; } divSchemes { default none; } "
                 "laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; } "
                 "snGradSchemes { default corrected; }",
    "fvSolution": "solvers { }",
}


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def raises(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    return None


# 1. a small made-up grid, 4 x 3 cells, stretched and sheared a little
xi, yj = np.array([0.0, 1.0, 2.5, 3.0, 4.0]), np.array([0.0, 0.1, 0.5, 1.0])
X, Y = np.meshgrid(xi, yj)
Y = Y + 0.05 * X
allp = {"inlet": ("i-", None), "outlet": ("i+", None), "top": ("j+", None), "sym": ("j-", (0, 1)), "plate": ("j-", (1, None))}
tmp = Path(tempfile.mkdtemp(prefix="structgrid_", dir=Path.home()))
try:
    info = structgrid.write_polymesh(tmp / "small", X, Y, allp, types={"plate": "wall"})
    check("small grid: cells and patch sizes", info["cells"] == 12 and info["patches"] ==
          dict(inlet=3, outlet=3, top=4, sym=1, plate=3, frontAndBack=24), str(info["patches"]))
    check("index_range finds the cells from a coordinate", structgrid.index_range(xi, lo=1.0) == (1, 4)
          and structgrid.index_range(xi, hi=2.5) == (0, 2) and raises(lambda: structgrid.index_range(xi, lo=1.2)) is not None)
    e1 = raises(lambda: structgrid.write_polymesh(tmp / "bad", X, Y, {k: v for k, v in allp.items() if k != "sym"}))
    e2 = raises(lambda: structgrid.write_polymesh(tmp / "bad", X, Y, dict(allp, sym=("j-", (0, 2)))))
    e3 = raises(lambda: structgrid.write_polymesh(tmp / "bad", X[:, ::-1], Y[:, ::-1], allp))
    check("refused: a boundary face in no patch, in two patches, a left-handed grid",
          bool(e1 and "0 patches" in e1 and e2 and "2 patches" in e2 and e3 and "left-handed" in e3), f"{e1} | {e2} | {e3}")

    # 2. NASA's grids through OpenFOAM's checkMesh
    files = sorted(GRIDS.glob("*.p2dfmt.gz"), key=lambda q: q.stat().st_size)
    for f in (files[0], files[2], files[-1]):
        x, y = structgrid.read_plot3d_2d(f)
        case = tmp / f.name.split(".")[0]
        (case / "system").mkdir(parents=True)
        (case / "system" / "controlDict").write_text(CONTROL)
        for name, body in MINIMAL.items():                  # checkMesh wants them to exist; nothing here is used
            (case / "system" / name).write_text(CONTROL.splitlines()[0].replace("controlDict", name) + "\n" + body + "\n")
        plate = structgrid.index_range(x[0], lo=0.0)
        info = structgrid.write_polymesh(case, x, y, {"inlet": ("i-", None), "outlet": ("i+", None), "top": ("j+", None),
                                                      "symmetry": ("j-", (0, plate[0])), "plate": ("j-", plate)},
                                         types={"plate": "wall", "symmetry": "symmetryPlane"})
        try:
            foam.run("checkMesh", case)
            failed = False
        except RuntimeError:
            failed = True                                   # a crash; a failed quality check still exits with 0
        log = (case / "log.checkMesh").read_text()
        tag = f"{x.shape[1]} x {x.shape[0]}"
        cells = int(re.search(r"cells:\s+(\d+)", log).group(1))
        nplate = int(re.search(r"plate\s+(\d+)\s+\d+", log).group(1))
        topo = all(s in log for s in ("Boundary definition OK.", "Point usage OK.", "Upper triangular ordering OK.",
                                      "Face vertices OK.", "Number of regions: 1 (OK)."))
        check(f"{tag}: checkMesh accepts the topology; cells and plate faces as in the file",
              topo and cells == (x.shape[1] - 1) * (x.shape[0] - 1) == info["cells"] and nplate == int((x[0] >= -1e-12).sum()) - 1,
              f"cells {cells}, plate faces {nplate}")
        northo = float(re.search(r"Mesh non-orthogonality Max: ([-+0-9.eE]+)", log).group(1))
        skew = float(re.search(r"Max skewness = ([-+0-9.eE]+)", log).group(1))
        complaints = re.findall(r"\*\*\*(.*)", log)
        only_aspect = all("aspect ratio" in c.lower() for c in complaints)
        check(f"{tag}: orthogonal, not skewed, and the only complaint (if any) is the aspect ratio of the wall cells",
              northo < 1e-3 and skew < 1e-6 and only_aspect and not failed,   # degrees; the file has 12 digits
              f"non-orthogonality {northo:g}, skewness {skew:g}, complaints {complaints}, aspect ratio "
              f"{info['max_aspect_ratio']:.0f}, digits needed {info['digits_needed']}")
        # points as OpenFOAM stores them, against the file
        nums = re.findall(r"^\(([-+0-9.eE]+) ([-+0-9.eE]+) ([-+0-9.eE]+)\)$", (case / "constant/polyMesh/points").read_text(), re.M)
        pts = np.array(nums, float)[: x.size]
        check(f"{tag}: the points on disk are the file's (first cell height kept to 15 digits)",
              np.array_equal(pts[:, 0], x.ravel()) and np.array_equal(pts[:, 1], y.ravel())
              and abs(info["min_edge"] / (y[1] - y[0]).min() - 1) < 1e-12, f"smallest edge {info['min_edge']:.6e}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
