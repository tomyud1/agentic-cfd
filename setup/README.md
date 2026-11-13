# Setup: start here

How to install everything, re-create the reports, re-run the simulations and build the GPU solver,
on a fresh machine. After installing, describe the machine in one file in
[`workflow/machines/`](../workflow/machines/README.md) (hardware, paths, launch commands): agents
read it before running anything. `dev-pc.md` there is the development PC.

**Tested on:** Windows 11 with WSL2 (Ubuntu 24.04), ESI OpenFOAM v2606 and Python 3.12, an AMD
Ryzen 7 7800X3D (8 cores, 16 threads) and an NVIDIA RTX 4070. A native Ubuntu 24.04 machine works
the same way: run the commands below in its terminal.

## 1. Install OpenFOAM and Python

On Windows, install WSL2 and Ubuntu 24.04 first (PowerShell as administrator):

```bash
wsl --install -d Ubuntu-24.04
```

Then, in the Ubuntu terminal, from the repository folder:

```bash
bash setup/install_openfoam.sh
sudo apt-get install -y python3-venv && bash setup/make_venv.sh
```

The first script adds the openfoam.com package repository and installs `openfoam2606-default`
(needs sudo). The second creates the Python environment in `~/venvs/cfd` with the pinned packages
from [`requirements.txt`](../requirements.txt). Check the OpenFOAM install with
`bash setup/run_cavity.sh` (runs the lid-driven cavity tutorial).

## 2. Re-create the reports from the saved data (minutes, no OpenFOAM needed)

The data extracted from every run is stored in `results/<case>/runs/`, so all reports can be
rebuilt, including theory, figures and checks:

```bash
bash setup/rerender_all.sh
```

A re-render keeps each report's original date and machine record (`CFD_RERENDER=1`): it rebuilds
the page, not the history of when and where the case was run.

## 3. Re-run the simulations

Each report's **Reproducibility** section lists the exact commands, in order; run them from the
repository folder. For example, case 3:

```bash
~/venvs/cfd/bin/python -m cases.stokes_first_problem.run preflight
~/venvs/cfd/bin/python -m cases.stokes_first_problem.run production --jobs 4
~/venvs/cfd/bin/python -m cases.stokes_first_problem.analyze
```

Runs go to `$FOAM_RUN/agentic/<case>/` (on the Linux file system, which is much faster than a
Windows drive). Cases 1-3 take minutes; case 4 takes a few hours (its finest run alone takes about
35 minutes on 16 threads).

## 4. View a report

Open `results/<case>/report.html` in a browser. It is one self-contained file; the formulas load
MathJax from a CDN. The published reports are at
[tomyud1.github.io/agentic-cfd](https://tomyud1.github.io/agentic-cfd/).

## 5. GPU (optional)

Needs an NVIDIA GPU and the CUDA 12 toolkit at `/usr/local/cuda`.

- **gpucfd**, our GPU solver: compile it, then run its verification ladder.

  ```bash
  bash gpucfd/build.sh
  bash gpucfd/run_ladder.sh
  ```

  `build.sh` targets the RTX 4070 (`sm_89`). On another GPU, set its architecture, for example
  `ARCH=sm_80 bash gpucfd/build.sh` for an A100. Details: [`gpucfd/DESIGN.md`](../gpucfd/DESIGN.md).
- **OpenFOAM's OGL plugin** (GPU linear solvers, used in the case 5 GPU study):
  `bash setup/gpu/build_ogl.sh` (30-60 minutes). The other scripts in [`gpu/`](gpu) check CUDA and
  the plugin.

## Other scripts here

- `fetch_featflow.sh`, `fetch_featflow_3d.sh`, `parse_featflow*.py`: download and parse the FeatFlow
  benchmark data for cases 4 and 5.
- `probe_env.sh`: lists the Python packages and tools that are available.
