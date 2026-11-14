# Machine: development PC (Windows 11 + WSL2, RTX 4070)

The student's desktop, where cases 1-6 were solved. Facts that belong to this computer only. What
holds on every machine is in [`../references/environment.md`](../references/environment.md).

## Hardware and limits

- CPU: AMD Ryzen 7 7800X3D (8 cores / 16 threads, 96 MB L3 cache), 24 GB RAM visible to WSL
  (`C:\Users\tomyu\.wslconfig`; check with `free -g`; roughly 1-1.5 GB per million cells for a 3D
  incompressible solver, so about 15 million cells at most).
- GPU: NVIDIA RTX 4070 (12 GB, compute capability 8.9; consumer card: fast memory, about 504 GB/s,
  but slow double precision). CUDA 12.9 at `/usr/local/cuda`. gpucfd builds with the default
  `sm_89`.
- Open MPI 4.1.6 (`mpirun`) works with OpenFOAM; more than 8 ranks needs `mpirun --use-hwthread-cpus`.
  16 ranks were fastest for a 215 thousand cell pimpleFoam case (5.4 times one rank).
- Independent runs at the same time: up to about 6. While all 16 threads run a solver, your own work
  slows it: leave 1-2 threads free or budget 1.5 times the projection.
- Run-time budget: a single run projected above about 15 minutes is a reason to coarsen the plan.
  The exception is a run whose outputs are time averages (turbulence without a model): its length is
  set by the flow, not by the grid. Budget such a case as a whole, about 2 GPU hours, with the
  finest run at most about one hour, and say so in the decisions (case 17 used 1.8 hours).

## Software and paths

- Windows 11 host; OpenFOAM runs in **WSL2 Ubuntu 24.04** (distro `Ubuntu-24.04`, user `tomyud`).
- OpenFOAM: ESI **v2606**, `/usr/lib/openfoam/openfoam2606`. `$FOAM_RUN` =
  `/home/tomyud/OpenFOAM/tomyud-v2606/run`, on the Linux filesystem (much faster than /mnt/d).
- Repository on Windows: `D:\Projects\Agentic CFD` = `/mnt/d/Projects/Agentic CFD` in WSL.
- Python for the pipeline: `/home/tomyud/venvs/cfd/bin/python` (3.12). No sudo is available to you:
  you cannot install system packages.
- OGL plugin: built into `$FOAM_USER_LIBBIN`; source in `~/src/OGL`.

## How to run things here

The agent's shell is on Windows, the tools are in WSL, so every command crosses that boundary.

From **PowerShell**:
```
wsl -d Ubuntu-24.04 --cd '/mnt/d/Projects/Agentic CFD' -- /home/tomyud/venvs/cfd/bin/python -m cases.<name>.run
```
- PowerShell mangles inline bash (quotes, `|`, `(`, `\|`): never pass `bash -c '...'` with anything
  complex. Write a `.sh` file and run `wsl -d Ubuntu-24.04 -- bash '/mnt/d/.../x.sh'`, or do the
  work in Python with `cfdlib.foam.run()`.

From **Git Bash** (Claude Code's Bash tool on this machine is Git Bash/MSYS, not Linux):
```
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 --cd "/mnt/d/Projects/Agentic CFD" -- /home/tomyud/venvs/cfd/bin/python -m cases.<name>.run 2>&1 | tr -d '\000'
```
- Without `MSYS_NO_PATHCONV=1`, MSYS rewrites `/mnt/d/...` into `C:/Program Files/Git/mnt/d/...`.
- `wsl.exe` prints its own messages in UTF-16, so the output contains NUL bytes (grep then says
  "Binary file matches"): pipe through `tr -d '\000'`.
- Environment variables reach WSL only through `WSLENV`, e.g.
  `CFD_FIG_PNG_DIR=/mnt/c/.../png WSLENV=CFD_FIG_PNG_DIR MSYS_NO_PATHCONV=1 wsl ...`.
- Git Bash expands `$var` inside an inline `bash -c` before WSL sees it: put loops in a `.sh` file.
- Never call `python` from Git Bash itself, not even for one line: it is the Windows launcher, not
  the pipeline's Python, and fed a here-document it hangs until the tool's time limit.
- A shell heredoc that writes or patches code eats backslashes (a `\\n` in a Python string arrives as
  a real line break, a double backslash as a single one): change code with your file-editing tool.
- Windows Python's `Path.write_text` turns every line end into CRLF: a script that patches repository
  files from the Windows side writes bytes (`write_bytes(text.encode("utf-8"))`) or runs inside WSL.
- In a live watch on a log, every stage of the pipe must write line by line, or nothing arrives until
  the run ends: `grep --line-buffered`, `sed -u`, `stdbuf -oL cut ...` (`tr` and `cut` buffer whole
  blocks by default). Simplest for a run started from Windows: inside the `.sh` that WSL runs, write
  the log to a file in the project (`python -u ... > run.log 2>&1`) and watch that file.

## Pitfalls of this machine

- The WSL warning "Failed to detect nested virtualization support" is harmless.
- `/tmp` inside WSL is wiped when the WSL virtual machine idles between calls: keep files you need
  under `$FOAM_RUN` or the repository.
- OpenFOAM cannot use paths with spaces, and the repository path has one: keep every case run under
  `$FOAM_RUN/agentic/<case>/`, never inside the repository.

## Looking at the report

Claude Code's built-in browser cannot open `file://` pages. A local server serves `results/`:
`http://localhost:8765/<name>/report.html` (configuration `reports` in `.claude/launch.json`; start it
with the preview tool if the page does not load). If another session already serves `results/` on
8765, open that address: a second server gets another port and its tab may not load.
