#!/bin/bash
# Installs ESI OpenFOAM v2606 in WSL2 Ubuntu 24.04.
# Run from Windows:  wsl -d Ubuntu-24.04 --cd '/mnt/d/Projects/Agentic CFD/setup' -- bash install_openfoam.sh
set -euo pipefail

VER=2606

# Add the official openfoam.com apt repository
curl -fsSL https://dl.openfoam.com/add-debian-repo.sh -o /tmp/add-openfoam-repo.sh
sudo bash /tmp/add-openfoam-repo.sh

sudo apt-get update
sudo apt-get install -y "openfoam${VER}-default"

# Shortcut: type "of${VER}" in a new shell to load the OpenFOAM environment
if ! grep -q "alias of${VER}=" ~/.bashrc; then
  echo "alias of${VER}='source /usr/lib/openfoam/openfoam${VER}/etc/bashrc'" >> ~/.bashrc
fi

source "/usr/lib/openfoam/openfoam${VER}/etc/bashrc" || true
echo
echo "== Installed =="
foamVersion -h >/dev/null 2>&1 || true
echo "WM_PROJECT_VERSION=${WM_PROJECT_VERSION:-unknown}"
which blockMesh icoFoam
echo "Done. Open a new Ubuntu shell and run: of${VER}"
