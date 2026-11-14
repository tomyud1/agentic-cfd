#!/bin/bash
# The final run of the stage for the grid the rule picked (one GPU job): pre-phase from a source variant's
# field, then averaging until the stop rule of plan.yaml holds or the final run's GPU time is used up.
#   bash dev/optimize/turbulent_channel_re180/final.sh x15z15 x15
here="$(dirname "$0")"
bash "$here/stage.sh" pre "$1" "$2" | tail -3 | cut -c1-400 | tee /tmp/final_last.txt
grep -q "EXIT 0" /tmp/final_last.txt || { echo "FINAL STOPPED in the pre-phase"; exit 1; }
bash "$here/stage.sh" avg "$1" stop | tail -6 | cut -c1-500
echo "FINAL DONE $1"
