# Environment (this machine)

- Windows 11 host; OpenFOAM runs in **WSL2 Ubuntu 24.04** (distro `Ubuntu-24.04`, user `tomyud`).
- OpenFOAM: ESI **v2606**, `/usr/lib/openfoam/openfoam2606`. Run directory `$FOAM_RUN` =
  `/home/tomyud/OpenFOAM/tomyud-v2606/run`; put case runs under `$FOAM_RUN/agentic/<case>/`
  (Linux filesystem: much faster than /mnt/d).
- Project on Windows: `D:\Projects\Agentic CFD` = `/mnt/d/Projects/Agentic CFD` in WSL.
- Python for the pipeline (numpy, scipy, sympy, matplotlib, pyyaml, jinja2):
  `/home/tomyud/venvs/cfd/bin/python` (3.12). No sudo is available to you: you cannot install system packages.
- CPU: AMD Ryzen 7 7800X3D (8 cores / 16 threads, 96 MB L3 cache), 24 GB RAM visible to WSL
  (`C:\Users\tomyu\.wslconfig`; check with `free -g`; roughly 1-1.5 GB per million cells for a 3D
  incompressible solver, so about 15 million cells at most).
  Open MPI 4.1.6 (`mpirun`) works with OpenFOAM; more than 8 ranks needs `mpirun --use-hwthread-cpus`.
  16 ranks were fastest for a 215 thousand cell pimpleFoam case (5.4 times one rank).
- GPU: NVIDIA RTX 4070 (12 GB, compute capability 8.9; consumer card: fast memory, about 504 GB/s,
  but slow double precision). CUDA 12.9 at `/usr/local/cuda`. See "GPU linear solvers (OGL)" below.

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

## GPU linear solvers (OGL)
OGL (OpenFOAM Ginkgo Layer, https://github.com/hpsim/OGL, commit 7c3ced4, Ginkgo 2.0.0 develop) is
built with CUDA into `$FOAM_USER_LIBBIN` (`setup/gpu/build_ogl.sh`). It replaces only the LINEAR
SOLVER of a field (usually `p`); assembly, momentum and everything else stay on the CPU. Smoke test:
`setup/gpu/test_ogl.py` (2D cavity, 160 thousand cells). What works and what does not (2026-11-11):
- `libs ("libOGL.so");` in `system/controlDict`. In `fvSolution`, start from the settings tuned in
  case 5 (3D steady, 0.1-4.75 million cells):
  ```
  p
  {
      solver GKOCG; executor cuda; scaling -1; forceHostBuffer true; verbose 1;
      matrixFormat Coo; adaptMinIter false; tolerance 1e-06; relTol 0.05;   // your tolerances
      preconditioner
      {
          preconditioner Multigrid;
          minCoarseRows 1000;       // default 64000: far too large a coarsest level
          coarseSolver CG; maxIterCoarse 20; relTolCoarse 0.01;   // default: ONE Jacobi sweep
          maxIterSmoother 2;        // default 1
          caching 20;               // reuse the preconditioner for 20 solves (set-up is expensive)
      }
  }
  ```
  **Preconditioner options are read only inside the `preconditioner { }` sub-dictionary**, in lower
  camel case (`maxLevels`, `minCoarseRows`, `smoother`, `maxIterSmoother`, `coarseSolver`,
  `maxIterCoarse`, `relTolCoarse`, `cycle`, `reuseHierarchy`, `caching`). A misspelt or misplaced
  option is ignored without a warning: read the names from the OGL source (`~/src/OGL/include/OGL/
  Preconditioner/`) before adding one.
- **`adaptMinIter`** (default true) forces every solve to do at least 0.6 times the iterations of the
  previous one; switch it off for partial solves (`relTol` > 0, SIMPLE). Default multigrid: 132
  iterations per solve, without adaptMinIter 97, with the settings above 15 (case 5, 1 million cells).
- **Set-up cost**: without `caching`, building the multigrid hierarchy cost about 0.9 s per solve on 1
  million cells, several times the solve itself. `reuseHierarchy` helped less than `caching`.
- **Only parallel runs**: serial runs stop with "Only parallel runs are supported for OGL", and
  OpenFOAM refuses `-parallel` on 1 rank, so use `decomposePar` and `mpirun -np N` with N >= 2.
  `ranksPerGPU N` gathers the N ranks' matrices onto the one GPU.
- **`forceHostBuffer true` is required**: Ubuntu's Open MPI is not CUDA-aware; without it OGL passes
  GPU memory to MPI and crashes (segfault in MPI_Alltoallv).
- **Preconditioner and sign**: OpenFOAM's pressure matrix has a negative diagonal. CG with `BJ` or
  `none` did not converge within the default 1000 iterations (residual 0.2 to 1) and the run then
  blew up (floating point exception). CG with `preconditioner Multigrid` and `scaling -1` converged
  and matched the CPU pressure to \(1.6 \times 10^{-6}\) relative (with default multigrid settings,
  which are weak: see above). Block Jacobi (631 iterations), ISAI (353) and IC (207) were all poor in
  case 5; SSOR smoothing, `matrixFormat Csr` and a W cycle did not help.
- **Measured gain (case 5, steady 3D, best CPU = GAMG DICGaussSeidel on 8 ranks, GPU on 8 ranks)**:
  time per SIMPLE iteration CPU/GPU 0.23 at 90 thousand cells, 0.55 at 0.3 million, 0.90 at 1
  million, 1.03 at 2.4 million, 1.13 at 4.75 million (break-even about 2 million cells). Pressure solve
  alone: 1.65 at 4.75 million. The pressure solve is only 30-37 % of an iteration on the CPU, so the
  gain can never exceed about 1.5; matrices also pass through host memory every solve. Peak GPU
  memory 4.85 GB at 4.75 million cells.
- `verbose 1` prints per-solve timings: lines `[OGL LOG][Proc: 0]p: solve: <ms>`, `init_precond`,
  `call_update` (matrix copy), `copy_x_back`. Peak GPU memory: `nvidia-smi --query-gpu=memory.used --format=csv`.
- OGL's own CI tests only OpenFOAM v2512; v2606 built and ran without changes, but record anything odd
  as an insight (stage `gpu`).

## Looking at the report
The built-in browser cannot open `file://` pages. A local server serves `results/`:
`http://localhost:8765/<name>/report.html` (configuration `reports` in `.claude/launch.json`; start it
with the preview tool if the page does not load). Add `?v=<n>` to the URL after re-rendering so the
browser does not show a cached copy. Without a browser, `CFD_FIG_PNG_DIR` makes `plots.to_svg` also
write each figure as a PNG.
