#!/bin/bash
# Download the FeatFlow DFG 3D benchmark pages (Bayraktar, Mierka & Turek 2012) for case 5, then run
# setup/parse_featflow_3d.py to turn them into the reference YAML.
# The FeatFlow site works over plain http (its https handshake failed from this machine).
cd "$(dirname "$0")/../cases/dfg_cylinder_3d_re20/reference" || exit 1
B=http://wwwold.mathematik.tu-dortmund.de
curl -sSL -m 60 -o featflow_dfg3d_page.html    "$B/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d.html"
curl -sSL -m 60 -o featflow_dfg3d_results.html "$B/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d/dfg_flow3d_results.html"
curl -sSL -m 60 -o featflow_dfg3d_software.html "$B/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d/dfg_flow3d_software.html"
curl -sSL -m 60 -o featflow_dfg3d_configuration.html "$B/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d/dfg_flow3d_configuration.html"
curl -sSL -m 60 -o featflow_dfg3d_reference.html "$B/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d/dfg_flow3d_reference.html"
curl -sSL -m 60 -o featflow_dfg3d_conclusion.html "$B/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d/dfg_flow3d_conclusion.html"
ls -la
curl -sSL -m 60 -o featflow_dfg3d_geometry.png "$B/~featflow/media/dfg_flow3d/fac_geo_3d.png"
curl -sSL -m 60 -o featflow_dfg3d_mesh2d.png "$B/~featflow/media/dfg_flow3d/meshV3_side_new.png"
curl -sSL -m 60 -o BenchValues.txt "$B/~featflow/media/dfg_flow3d/BenchValues.txt"
file featflow_dfg3d_geometry.png featflow_dfg3d_mesh2d.png BenchValues.txt
