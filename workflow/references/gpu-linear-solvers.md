# GPU linear solvers for OpenFOAM (OGL)

For a performance study that puts OpenFOAM's pressure solve on a GPU. Everything here was measured
in case 5 on the development PC (RTX 4070, [`../machines/dev-pc.md`](../machines/dev-pc.md)); on
another GPU the settings are a starting point and the timings must be measured again. For a solver
that runs the whole time step on the GPU, see gpucfd ([`gpucfd/DESIGN.md`](../../gpucfd/DESIGN.md)).

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
