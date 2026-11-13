"""The generated case tables are up to date with cases/catalogue.yaml (run inside WSL from the
project root):

    PYTHONPATH=. python tests/test_catalogue.py
"""
from cfdlib import catalogue

assert catalogue.main(["--check"]) == 0, "catalogue out of date: run python -m cfdlib.catalogue"
print("catalogue: OK")
