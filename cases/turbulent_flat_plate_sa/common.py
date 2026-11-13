"""Shared loaders for the turbulent flat plate case: problem, numerical plan, NASA's grids and data files."""
import re
from pathlib import Path

import numpy as np
import yaml

from cfdlib import structgrid

HERE = Path(__file__).resolve().parent
NAME = HERE.name
OUT = HERE.parents[1] / "results" / NAME
RUNS = OUT / "runs"
REF = HERE / "reference"


def load():
    problem = yaml.safe_load((HERE / "problem.yaml").read_text(encoding="utf-8"))
    numerics = yaml.safe_load((HERE / "numerics.yaml").read_text(encoding="utf-8"))
    return problem, numerics


def given(problem):
    return {k: v["value"] for k, v in problem["given"].items()}


def reference():
    return yaml.safe_load((REF / "tmr_2dzp_sa.yaml").read_text(encoding="utf-8"))


def grid_files():
    """NASA's grid files, coarse to fine (the reference YAML lists them coarse to fine)."""
    return [REF / g["file"] for g in reference()["grids"]]


def read_grid(level_index):
    """Points x[j, i], y[j, i] of grid level 0 (coarsest) .. 4 (finest)."""
    return structgrid.read_plot3d_2d(grid_files()[level_index])


def tecplot_zones(path):
    """Zones of a Tecplot ASCII file as {zone title: array (rows, columns)}, plus the variable names.
    Reads NASA's small data files (one 'variables=' line, 'zone t="..."' lines, rows of numbers)."""
    names, zones, cur = None, {}, None
    for line in Path(path).read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        low = s.lower()
        if low.startswith("variables"):
            names = re.findall(r'"([^"]*)"', s)
            continue
        if low.startswith("zone"):
            m = re.search(r't\s*=\s*"([^"]*)"', s, re.I)
            cur = re.sub(r"\s+", " ", m.group(1)).strip() if m else f"zone{len(zones)}"
            zones[cur] = []
            continue
        if low.startswith("title"):
            continue
        try:
            row = [float(v.replace("D", "E").replace("d", "e")) for v in s.replace(",", " ").split()]
        except ValueError:
            continue
        if cur is None:
            cur = "zone0"
            zones[cur] = []
        zones[cur].append(row)
    return names, {k: np.array(v, float) for k, v in zones.items()}
