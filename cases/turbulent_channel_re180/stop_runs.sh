#!/bin/bash
# Stop a running run.py of this case (and nothing else). Its own command line does not hold the pattern.
pat="cases.turbulent_channel_re180.ru"
pids=$(pgrep -f "${pat}n ")
echo "stopping: ${pids:-none}"
[ -n "$pids" ] && kill $pids
sleep 3
pgrep -af "${pat}n " || echo "no run left"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
