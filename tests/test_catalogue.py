"""The generated case tables are up to date with cases/catalogue.yaml (run inside WSL from the
project root):

    PYTHONPATH=. python tests/test_catalogue.py
"""
from pathlib import Path

import yaml

from cfdlib import catalogue, contract

assert catalogue.main(["--check"]) == 0, "catalogue out of date: run python -m cfdlib.catalogue"
print("catalogue: OK")

# A problem statement that is prepared but not solved yet must be one the report will accept as given:
# the solver agent may not change it (case 7's description was longer than the report allows).
ROOT = Path(__file__).resolve().parents[1]
waiting = [c["folder"] for c in yaml.safe_load((ROOT / "cases/catalogue.yaml").read_text(encoding="utf-8"))["cases"]
           if c.get("folder") and c.get("status") != "solved" and (ROOT / "cases" / c["folder"] / "problem.yaml").exists()]
for folder in waiting:
    errs = contract.check_problem_file(ROOT / "cases" / folder / "problem.yaml")
    assert not errs, (folder, errs)
print(f"problem statements waiting for an agent: {len(waiting)} checked")
