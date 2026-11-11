# Environment (this machine)

- Windows 11 host; OpenFOAM runs in **WSL2 Ubuntu 24.04** (distro `Ubuntu-24.04`, user `tomyud`).
- OpenFOAM: ESI **v2606**, `/usr/lib/openfoam/openfoam2606`. Run directory `$FOAM_RUN` =
  `/home/tomyud/OpenFOAM/tomyud-v2606/run`; put case runs under `$FOAM_RUN/agentic/<case>/`
  (Linux filesystem: much faster than /mnt/d).
- Project on Windows: `D:\Projects\Agentic CFD` = `/mnt/d/Projects/Agentic CFD` in WSL.
- Python for the pipeline (numpy, scipy, sympy, matplotlib, pyyaml, jinja2):
  `/home/tomyud/venvs/cfd/bin/python` (3.12). No sudo is available to you: you cannot install system packages.
- CPU: AMD Ryzen 7 7800X3D (8 cores / 16 threads, 96 MB L3 cache), 15 GB RAM visible to WSL.
  Open MPI 4.1.6 (`mpirun`) works with OpenFOAM; more than 8 ranks needs `mpirun --use-hwthread-cpus`.
  16 ranks were fastest for a 215 thousand cell pimpleFoam case (5.4 times one rank). GPU (RTX 4070)
  not usable yet: no GPU solver plugin is installed, so do not try; record what a GPU test would need.

## How to run things

From **PowerShell**:
```
wsl -d Ubuntu-24.04 --cd '/mnt/d/Projects/Agentic CFD' -- /home/tomyud/venvs/cfd/bin/python -m cases.<name>.run
```
- PowerShell mangles inline bash (quotes, `|`, `(`, `\|`): never pass `bash -c '...'` with anything
  complex. Write a `.sh` file and run `wsl -d Ubuntu-24.04 -- bash '/mnt/d/.../x.sh'`, or do the
  work in Python with `cfdlib.foam.run()`.

From **Git Bash** (the Bash tool on this machine is Git Bash/MSYS, not Linux):
```
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 --cd "/mnt/d/Projects/Agentic CFD" -- /home/tomyud/venvs/cfd/bin/python -m cases.<name>.run 2>&1 | tr -d '\000'
```
- Without `MSYS_NO_PATHCONV=1`, MSYS rewrites `/mnt/d/...` into `C:/Program Files/Git/mnt/d/...`.
- `wsl.exe` prints its own messages in UTF-16, so the output contains NUL bytes (grep then says
  "Binary file matches"): pipe through `tr -d '\000'`.
- Environment variables reach WSL only through `WSLENV`, e.g.
  `CFD_FIG_PNG_DIR=/mnt/c/.../png WSLENV=CFD_FIG_PNG_DIR MSYS_NO_PATHCONV=1 wsl ...`.

Other commands (inside WSL, from the project root):
```
python -m cfdlib.contract --decisions cases/<name>/decisions.yaml <case dir>   # decisions cover every setting?
python -m cfdlib.contract results/<name>/results.json <finest case dir>        # full report contract
python -m cfdlib.insights <name> list                                          # the live insight log
python -m cfdlib.foam ps                                                       # running solvers + their case dirs
python -m cfdlib.foam stop <case dir>                                          # kill a run (and its solver)
```

## Pitfalls
- The WSL warning "Failed to detect nested virtualization support" is harmless.
- Never name a script after a Python standard-library module (`inspect.py`, `test.py`, `types.py`,
  `io.py`...): Python puts the script's folder first on the path and numpy then imports your file.
- **Edit files that contain LaTeX (decisions.yaml, analyze.py, theory.py) only with the Edit/Write
  tools.** The Bash tool collapses `\\` to `\` inside heredocs and inline scripts, so `\times` turns
  into a TAB and `\beta` into a backspace (cases 3 and 4 both hit this). The contract rejects control
  characters, but the Edit tool avoids the problem.
- Long runs: run them in the background and keep working; use `cfdlib.monitor.assess` on the log.
  For a sequence of long runs, use one background script that writes a completion marker and record
  per run. After any interruption (including an API pause), check those files before re-running
  anything: in case 4 a queued chain finished on its own while the agent thought it had died.
  Stopping the background Python process does NOT stop the solver (it runs in its own process group,
  so a timeout can kill it cleanly): use `python -m cfdlib.foam stop <case dir>` and check with
  `python -m cfdlib.foam ps` before restarting a case.
- Independent runs may run concurrently (up to ~6). Parallel runs of one case (`decomposePar` +
  `mpirun -np N`) are possible for large meshes; document either as a decision.

## Looking at the report
The built-in browser cannot open `file://` pages. A local server serves `results/`:
`http://localhost:8765/<name>/report.html` (configuration `reports` in `.claude/launch.json`; start it
with the preview tool if the page does not load). Add `?v=<n>` to the URL after re-rendering so the
browser does not show a cached copy. Without a browser, `CFD_FIG_PNG_DIR` makes `plots.to_svg` also
write each figure as a PNG.
