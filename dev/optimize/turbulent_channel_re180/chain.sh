#!/bin/bash
# Directional runs one after the other (one GPU job at a time): for each variant given, the pre-phase and
# the averaging of plan.yaml. Stops at the first command that fails.
#   bash dev/optimize/turbulent_channel_re180/chain.sh z15 [y15 x15]
here="$(dirname "$0")"
for v in "$@"; do
  bash "$here/stage.sh" pre "$v" | tail -3 | cut -c1-400
  bash "$here/stage.sh" avg "$v" | tail -4 | cut -c1-400 | tee /tmp/chain_last.txt
  grep -q "EXIT 0" /tmp/chain_last.txt || { echo "CHAIN STOPPED at $v"; exit 1; }
done
echo "CHAIN DONE $*"
