#!/bin/bash
# Reference data of case 17 (turbulent channel at friction Reynolds number 180): the statistics files of
# Vreman and Kuerten (2014) as their author publishes them, then parsed into the case's reference YAML.
# Run inside WSL from anywhere:  bash setup/fetch_vk2014_channel.sh
cd "$(dirname "$0")/.." || exit 1
DIR=cases/turbulent_channel_re180/reference/vk2014
mkdir -p "$DIR"
for f in Chan180_S2a_basic_few.txt Chan180_FD2_basic_u.txt Chan180_FD2_basic_v.txt Chan180_FD2_basic_w.txt; do
  curl -sSL --compressed -m 60 -o "$DIR/$f" "http://www.vremanresearch.nl/$f" || exit 1
  echo "$f $(stat -c %s "$DIR/$f") bytes"
done
"${PY:-$HOME/venvs/cfd/bin/python}" -m setup.parse_vk2014_channel
