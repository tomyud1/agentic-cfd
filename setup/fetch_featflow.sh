#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
curl -sSL -m 30 -o cases/dfg_cylinder_re100/reference/featflow_dfg2d2_page.html "http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html"
stat -c %s cases/dfg_cylinder_re100/reference/featflow_dfg2d2_page.html
"${PY:-$HOME/venvs/cfd/bin/python}" -m setup.parse_featflow
