# gpucfd: a GPU-resident incompressible flow solver

Goal: a solver that keeps the whole time step on the GPU and is trusted because every part is
verified before it is made fast. Case 5 showed why: OpenFOAM's GPU plugin only moves the pressure
solve to the GPU, copies matrices through host memory every solve, and is capped near 1.5 times.

## Scope (first version)
- Incompressible, single phase, laminar (constant density and viscosity).
- Structured Cartesian grids (uniform first, stretched later), 2D or 3D (2D = one cell in z).
- Staggered (MAC) grid: pressure at cell centres, each velocity component on its faces. Discretely
  divergence-free velocity, no pressure checkerboarding.
- Projection method: predict velocity, solve a pressure Poisson equation, correct.
- Pressure Poisson equation: geometric multigrid (red-black Gauss-Seidel smoother, cell averaging
  restriction, (bi/tri)linear prolongation), alone or as the preconditioner of conjugate gradients.
- Boundaries: periodic, walls (no-slip, moving lid), later inflow/outflow.
- Curved bodies (the DFG cylinders) need a later step (immersed boundary or curvilinear blocks).

Not in scope: everything else OpenFOAM does. OpenFOAM stays the reference to check against.

## How trust is built: the verification ladder
Each rung must pass before the next one is built, and every rung stays in the test suite, so an
optimisation that breaks anything is caught at once.

| Rung | What | Check | Pass rule |
|---|---|---|---|
| 0 | Hardware | measured memory bandwidth (copy kernel) | recorded; sets the speed ceiling |
| 1 | Discrete operators (Laplacian, ghost cells for each boundary type) | against an independent numpy implementation | equal to round-off (double: 1e-12 relative, float: 1e-5) |
| 2 | Poisson solver | against scipy's direct sparse solve on small grids | equal to the solver tolerance |
| 3 | Poisson discretisation | manufactured solutions on 4 grids | observed order 2 (1.9 to 2.1) |
| 4 | Multigrid quality | residual reduction per cycle on growing grids | factor below 0.2 and iterations independent of grid size |
| 5 | Single precision | float against double on the same grid | difference far below the discretisation error |
| 6 | Navier-Stokes, periodic | Taylor-Green vortex (exact solution) | order 2 in space and time; divergence at round-off |
| 7 | Walls, steady | lid-driven cavity Re 100 against Ghia et al. and our case 2 OpenFOAM results | within case 2's bands |
| 8 | Transient with walls | Stokes' first problem (case 3), Poiseuille (case 1) | exact solutions, order 2 |
| 9 | 3D | duct flow, then the DFG cylinders once curved bodies exist | benchmark bands of cases 4 and 5 |

The numpy references are written separately from the CUDA code, not translated from it, so a shared
mistake is unlikely.

## How speed is pursued
- Measure first: every benchmark reports time, cells per second and the achieved fraction of the
  measured memory bandwidth (rung 0). Memory traffic, not arithmetic, limits this kind of code, so
  that fraction says how much room is left.
- Every optimisation (single precision, fused kernels, CUDA graphs against launch overhead,
  matrix-free stencils, better smoothers) is a separate change, re-run against the whole ladder,
  and kept only if the ladder still passes and the benchmark improves.
- Compare against OpenFOAM on the same problem and machine with its best settings (GAMG, 8 or 16
  ranks), the same tolerance, and the same answer, as in the case 5 study.

## Layout
```
gpucfd/
  DESIGN.md        this file
  build.sh         compiles src/ into build/libgcfd.so (nvcc, run inside WSL)
  src/             CUDA C++ (header-only pieces + one .cu with the C API)
  gcfd.py          Python wrapper (ctypes); numpy arrays in and out
  tests/           the verification ladder (python -m gpucfd.tests.<name>)
  bench/           benchmarks; results as JSON in bench/results/
```

## Conventions
- Arrays hold interior cells only, x fastest (index i + nx (j + ny k)); the Python side sees shape
  (nz, ny, nx). There are no ghost cells: boundary conditions are applied inside each stencil
  (periodic wraps; Neumann/zero gradient uses the cell itself; Dirichlet/zero at the face uses minus
  the cell). The numpy references in the tests use ghost-cell padding instead, on purpose, so the two
  implementations are independent.
- Staggered velocity: component c is stored on the low c-face of each cell.
- The Poisson equation is solved as Laplacian(p) = f. With no Dirichlet boundary the problem is
  singular: f is shifted to zero mean and p is returned with zero mean.
- Reductions (norms, dot products) accumulate in double precision, even when fields are float.

## Status (2026-11-11)

Built and verified: the pressure Poisson solver (rungs 0-5), Navier-Stokes on periodic boxes
(rung 6), walls (rung 7), reproducibility (rung 8), and two time schemes: explicit RK3 and IMEX
(implicit diffusion). Run the whole ladder with `bash gpucfd/run_ladder.sh` (or
`python -m gpucfd.tests.test_poisson | test_ns | test_walls | test_reproducible`); results in
`gpucfd/tests/results/`.

| Rung | Result |
|---|---|
| 0 | RTX 4070, measured copy bandwidth about 390-410 GB/s |
| 1 | Laplacian equals the numpy reference to 2e-16 (double), 7e-8 (float), all boundary sets, 2D and 3D |
| 2 | Solutions equal a dense direct solve to 2e-12 |
| 3 | Order of accuracy 1.99-2.00 for Neumann, Dirichlet, periodic and mixed boundaries, 2D and 3D |
| 4 | Multigrid: 12 V-cycles (2D) / 13-14 (3D) to 1e-10 at every grid size from 32^2 to 1024^2 and 16^3 to 128^3; with CG 8-9 iterations; reduction 0.06-0.17 per iteration |
| 5 | Mixed precision (double outer iteration, single-precision multigrid) equals double to 1e-14 with the same iteration count. Pure single precision is NOT safe: on a 256^2 grid its error was 6 times the discretisation error (round-off grows like 1/h^2), so it is not used |
| 6 | Taylor-Green (2D) and ABC flow (3D): spatial order 2.00, temporal order 3 (RK3) and 2.0 (IMEX, mean 1.99 / 2.07), divergence 1e-13 or below, mixed = double to 1e-15 |
| 7a | Walls against exact solutions: Couette flow (moving wall) stays exact to 1e-16 in 2D and 3D; decaying channel mode order 2.00 (2D, and 3D with walls in another direction); Poiseuille flow from rest with a body force order 2.00; zero flow through wall faces, divergence 0 |
| 7b | Lid-driven cavity Re 100 (`python -m gpucfd.compare.cavity [imex] 32 64 128`): GPU and OpenFOAM (case 2) solutions converge to each other at second order (centreline difference 0.0027, 0.0007, 0.0002 U on 32^2, 64^2, 128^2); both stay 0.009 U from Ghia et al., as case 2 found (Ghia's own error). Both time schemes (all wall tests are run with each) |
| 7c | Both time schemes reach the same steady cavity: 5e-9 U apart on 16^2 and 32^2 |
| 8 | Reproducibility: two identical runs are bit-identical, and CUDA graphs on/off are bit-identical (Poisson double/mixed, cavity and 3D periodic flow with each scheme) |

Bugs the ladder caught while building (each would have produced plausible-looking wrong results):
1. Multigrid corrections on coarse levels started from leftover memory (rung 2 and 4 failures).
2. Pure single precision silently lost accuracy on fine grids (rung 5): led to mixed precision.
3. A blown-up run reported "divergence 0": `fmax` ignores NaN. The maximum now propagates NaN and
   `NavierStokes.run` raises with the stability limits.
4. The first IMEX version reached a different steady cavity than the explicit scheme (up to 6e-4 U):
   with implicit diffusion, a non-incremental projection leaves an O(dt) error near walls, because
   the Laplacian of a pressure gradient is not a gradient there. Fixed with the incremental projection
   (predictor uses the last pressure; the projection only removes the increment). Rung 7c now checks it.
5. Results differed between identical runs at 4e-16: dot products summed block results with
   atomicAdd, whose order varies. Now deterministic (fixed-order sums), which also makes "this
   optimisation changed nothing" checkable bit for bit (rung 8).

Speed (RTX 4070, `python -m gpucfd.bench.bench_poisson`, numbers in `bench/results/`):
- Pressure solve to 1e-6, mixed precision, 3D: 9.6 ms at 2.1 million cells, 100 ms at 16.8 million.
- Large grids reach 50-60 % of the measured bandwidth; small grids are dominated by kernel launches
  (about 100 per V-cycle).
- Navier-Stokes (RK3, 3 projections per step, about 7 CG iterations each): 62 ms per step at 2.1
  million cells, 480 ms at 16.8 million (about 28 ns per cell per step). The pressure solve is nearly
  all of it.
- Compared with OpenFOAM on the same flow: see "Comparison with OpenFOAM" below.
- **Where the GPU solver is slow (cavity, rung 7b)**: the steady 2D cavity on 128^2 took 396 s of
  GPU time against 46 s for OpenFOAM's steady SIMPLE solver (case 2). Two reasons, both measured:
  (1) diffusion is explicit, so the time step is limited by 0.15 h^2 / nu instead of the flow speed:
  29 511 steps on 128^2 where the convective limit would allow about 4 300; (2) small grids are
  dominated by launch overhead: about 10 ms per time step even on 32^2 (1024 cells), roughly 1000
  kernel launches per step. Neither is a GPU limit; both are the next speed work.
- A loose relative pressure tolerance (1e-4) does not change the cavity answer (2.6e-7 U against
  1e-3 U between grids) and halves the time.

### Speed work, round 1 (2026-11-12)
Two changes, each gated by the full ladder:
1. **IMEX time scheme** (`scheme="imex"`): convection by the low-storage RK3 of Spalart, Moser & Rogers
   (1991), diffusion by Crank-Nicolson inside each stage (one conjugate-gradient Helmholtz solve per
   velocity component, about 3-10 iterations), with an incremental pressure projection. The time step
   is limited by the flow speed only. Second order in time (RK3 explicit stays available: third order).
2. **CUDA graphs** for the multigrid V-cycles (about 100 launches recorded once, replayed as one).

| Case | Before | After | Speed-up |
|---|---|---|---|
| Cavity 32^2 to steady state | 19.5 s (rk3) | 10.9 s (imex + graphs) | 1.8 |
| Cavity 64^2 | 89.6 s | 19.4 s | 4.6 |
| Cavity 128^2 | 396 s, 29 511 steps | 42.8 s, 4 320 steps | 9.3 |
| ABC flow 64^3 (comparison flow) | 0.30 s | 0.18 s | 1.7 |
| ABC flow 128^3 | 1.73 s | 1.57 s | 1.1 |
| ABC flow 256^3 | 32.3 s | 31.4 s | 1.0 |

Answers unchanged: cavity centrelines identical to the earlier runs' accuracy (same differences from
Ghia and OpenFOAM), ABC errors identical. The cavity on 128^2 now takes about as long as case 2's
OpenFOAM run (46 s, steady SIMPLE as case 2 configured it; not tuned for speed, so not a fair speed
comparison). Large 3D grids barely change: they were already limited by memory bandwidth, not
launches.

Still left on small grids: about 10 ms per time step on 128^2. The Helmholtz and pressure CG loops
synchronise with the CPU for every dot product, and the time-step kernels outside the V-cycle are
not in a graph.

### Speed work, round 2 (2026-11-12)
Measured first (`python -m gpucfd.bench.bench_step`): 10.5 ms per time step even on a 32^2 cavity
(1024 cells), so the cost was fixed overhead, not work. Under WSL a kernel launch costs about 25 us
and a CPU-GPU round trip about 63 us, and a step made about 115 round trips (one per dot product)
and about 1200 launches. Changes, each gated by the ladder:
1. **Every CG scalar on the GPU, loops decided by the GPU.** Dot products, alpha, beta and the stop
   test live in GPU memory (one-thread kernels doing exactly the CPU's former arithmetic), and each
   CG loop is a CUDA graph WHILE node (CUDA 12.4+). A whole pressure solve, and a whole time step,
   is recorded once and replayed with one launch and no CPU sync. With graphs off, the CPU launches
   the same kernels and reads the stop flag; rung 8 checks that both give bit-identical results.
2. **Fused residual + restriction** in the V-cycle (the residual is never stored), and the coarse
   end of each V-cycle (levels of at most 1024 cells) in **one thread block in shared memory**.
3. **Multiply by 1/h and 1/h^2 instead of dividing** in the Navier-Stokes kernels. This GPU runs
   double-precision arithmetic at 1/64 of the float rate, and a double division costs about 10 of
   those operations, so these kernels were limited by arithmetic, not memory (12 % faster in 3D).
4. **Larger time steps for steady problems**: the steady answer does not depend on the time step
   (rung 7c), and the IMEX scheme on the Re 100 cavity is stable up to CFL about 4 (blows up at 6;
   cell Reynolds number 0.8, so the implicit diffusion damps the grid-scale modes that limit explicit
   convection). `compare.cavity imex cfl=2` uses half the measured limit. The default stays 0.8: in
   convection-dominated flows the limit is the usual one (about 1.7).

**Same answers**: changes 1 and 2 are bit-identical to the round-1 build (checked with
`tests/same_as_build.py` against a build of the old source: 18 of 18 results equal to the last bit,
Poisson, cavity and 3D flow, both schemes). Change 3 is bit-identical where h is a power of two (the
cavity) and differs at 1e-10 on the 3D flow (a pressure solve near its 1e-4 tolerance took a
different iteration count); the 3D errors against the exact solution are unchanged to 3 digits.
Change 4 moves the steady cavity by at most 3e-10 U.

Per time step (`bench/results/bench_step.json`):

| Case | Round 1 | Round 2 | Speed-up |
|---|---|---|---|
| Cavity 32^2, IMEX | 10.5 ms | 2.1 ms | 5.0 |
| Cavity 128^2, IMEX | 11.6 ms | 2.9 ms | 4.0 |
| ABC 64^3, IMEX | 15.2 ms | 8.5 ms | 1.8 |
| ABC 128^3, IMEX | 63.6 ms | 49.8 ms | 1.3 |

Whole runs (`python -m gpucfd.bench.round2`, `bench/results/round2.json`):

| Case | Before round 1 | Round 1 | Round 2 | vs OpenFOAM |
|---|---|---|---|---|
| Cavity 128^2 to steady state | 396 s | 42.8 s | 11.1 s (CFL 0.8), 5.1 s (CFL 2) | case 2: 46 s (steady SIMPLE, not tuned for speed) |
| Cavity 256^2 | | | 36.0 s (CFL 0.8), 17.7 s (CFL 2) | |
| ABC 32^3 (comparison flow, RK3) | 0.088 s | | 0.020 s | 18 times faster (0.36 s) |
| ABC 64^3 | 0.30 s | 0.18 s | 0.107 s | 47 times (4.98 s) |
| ABC 128^3 | 1.73 s | 1.57 s | 1.24 s | 235 times (291 s) |
| ABC 256^3 | 32.3 s | 31.4 s | 26.5 s | (OpenFOAM does not fit in 23 GB) |

What did not pay: the one-block V-cycle tail sped a 32^2 mixed-precision V-cycle up 27 % (110 to
80 us) but a whole step only about 5 %, within run-to-run noise (about 10 %, so timings are best of
several runs). Small grids are now limited by latency: about 1200 dependent GPU operations per step
at 1-2 us each. Nsight Systems under WSL records API calls but not kernel times, which made this
harder to see; the step costs were measured with microbenchmarks instead.

## Limitations (2026-11-12)

**Problems it cannot solve yet**
- Geometry: a box only. Uniform Cartesian grids (no stretching, so a boundary layer forces a fine
  grid everywhere), no bodies inside the domain (no cylinders, no curved walls).
- Boundaries: periodic, or no-slip walls that can slide tangentially (a moving lid). No inflow,
  outflow, symmetry, slip or pressure boundaries, so no channel with an inlet, no external flow.
- Physics: incompressible, single phase, constant density and viscosity, laminar. No turbulence
  model (only resolved simulations at low Reynolds number), no heat transfer or transported scalars,
  no buoyancy, no free surface, no compressibility. Body force constant in space and time.
- Numerics: central differences without upwinding or limiters, so the grid must resolve the flow
  (cell Reynolds number U h / nu around 2 or below for clean results); at higher Reynolds numbers it
  will oscillate rather than smear.
- Grid sizes: periodic directions need even cell counts, and multigrid works well only when each
  direction halves several times (sizes like 2^k or 3 x 2^k). An awkward size leaves a large coarsest
  grid, which is solved slowly by one thread block.
- Tooling: no field output for ParaView, no restart files, no probes or force coefficients; data
  comes out through Python (`NavierStokes.get`). One GPU, no MPI.

**Speed limits**
- Small grids are latency-bound: a time step is about 1200 dependent GPU operations of a few
  microseconds each, so it costs about 2 ms whatever the grid size below about 128^2.
- Double-precision arithmetic: consumer GPUs (the RTX 4070 here, and the RTX Spark-class machines)
  run FP64 at 1/64 of the float rate. The velocity, the diffusion solves and all reductions are
  double; only the pressure multigrid runs in float (mixed precision).
- Large grids are limited by memory bandwidth, and the pressure is solved three times per time
  step (once per Runge-Kutta stage).
- Steady problems are time-marched: the number of time units to settle is set by the physics (27
  for the Re 100 cavity), and convection is explicit, so the time step is bounded by the flow speed
  (CFL about 1.7 for convection-dominated flows; up to about 4 when the cell Reynolds number is
  below 1, as in the cavity, because the implicit diffusion damps the grid-scale modes).
- Memory: 12 GB holds about 17 million cells (256^3) with this layout.

## Next steps (in order)
1. ~~Fair OpenFOAM comparison~~ (done 2026-11-12, below). Open question from it: why OpenFOAM's
   error converges faster than second order on this flow (one more grid would show it).
2. ~~Walls~~ (done 2026-11-12, rung 7).
3. ~~Speed, round 2~~ (done 2026-11-12, above). Left for later: a true steady solver (implicit
   convection, Newton), fewer pressure solves per step, fused CG kernels, and running on a GPU with
   full-rate FP64 or on native Linux.
4. Stretched grids, inflow/outflow, then curved bodies (immersed boundary) for the DFG cylinders.

## Comparison with OpenFOAM (2026-11-12)

`python -m gpucfd.compare.abc tune|tune2|tune3|run` (numbers in `compare/results/abc.json`).

**Problem**: decaying ABC flow in a periodic box [0, 2π]^3, ν = 0.01, to t = 1; an exact
solution of the Navier-Stokes equations, so each solver's answer is judged by its error against it.
Same uniform grids on both sides. OpenFOAM v2606 `pimpleFoam` (collocated finite volume, `backward`
in time, `Gauss linear` everywhere, GAMG with DIC-Gauss-Seidel for pressure) on 8 MPI ranks of the
Ryzen 7 7800X3D; gpucfd on the RTX 4070.

**Rules fixed before the runs**: each solver uses its cheapest settings whose error stays within 10 %
of its own strict-setting error on that grid; time = the solver's own time loop (OpenFOAM:
profiling of `time.run()`; GPU: CUDA events), excluding start-up and output.

**Tuning** (each side got the same search):
- OpenFOAM: Courant number 0.8 / 0.4 / 0.2 / 0.1, 2 or 3 pressure correctors, 1 or 2 outer
  correctors, pressure tolerance 1e-8 to 1e-4, momentum tolerance 1e-10 to 1e-4, 4 / 8 / 16 ranks.
  Best: Courant 0.4, 3 correctors, 1 outer, 8 ranks; pressure tolerance 1e-4 on 32^3 and 64^3 but
  1e-6 on 128^3 (OpenFOAM's tolerances are absolute: 1e-4 raised the 128^3 error by 49 %).
- gpucfd: CFL 0.8 / 0.4 / 0.2 and relative pressure tolerance 1e-10 / 1e-6 / 1e-4. Its error did not
  change with either (RK3 time error is tiny; a relative tolerance scales with the grid). Best: CFL
  0.8, tolerance 1e-4 (error within 3 % of the strict setting on every grid).

**Results** (loop time, seconds; error = max |u - exact|):

| Grid | OpenFOAM best | OpenFOAM strict | gpucfd best | gpucfd strict | Speed-up, same grid (best/best) |
|---|---|---|---|---|---|
| 32^3 (33 k cells) | 0.36 s, err 3.9e-3 | | 0.09 s, err 5.3e-5 | 0.27 s | 4 |
| 64^3 (262 k) | 4.98 s, err 3.6e-4 | | 0.30 s, err 1.3e-5 | 0.62 s | 17 |
| 128^3 (2.1 M) | 291 s, err 3.0e-5 | 514 s, err 2.7e-5 | 1.73 s, err 3.3e-6 | 3.35 s | 168 |
| 256^3 (16.8 M) | (too large for 23 GB) | | 32.3 s, err 8.5e-7 | | |

(Times as first measured. After the speed work the gpucfd times are 0.020, 0.107, 1.24 and 26.5 s
with the same errors: 18, 47 and 235 times faster than OpenFOAM's best on the same grids.)

What the numbers say:
1. **Same grid, same flow: the GPU solver is 4 to 170 times faster**, and the gap grows with the
   grid. OpenFOAM slows down faster than its work grows (64^3 to 128^3: 16 times the work, 58 times
   the time): the 64^3 case fits in the CPU's 96 MB cache, the 128^3 case does not, and its tighter
   pressure tolerance costs more iterations.
2. **Where OpenFOAM's time goes** (128^3, best): pressure solve 46 %, momentum solve 10 %, the
   rest (assembling equations, interpolation, MPI) 44 %. A GPU linear solver can only touch the
   first part, which is why case 5's plugin was capped near 1.5 times.
3. **Accuracy differs too, for a different reason.** At the same grid gpucfd's error is 74 (32^3)
   to 9 (128^3) times smaller. That comes from the discretisation (staggered grid vs OpenFOAM's
   collocated one), not from the GPU. The ratio shrinks on finer grids because OpenFOAM's error falls
   faster than second order here (observed 3.4 to 3.6, not yet explained). Measured at equal
   accuracy instead of equal grid, the gap is larger still (gpucfd on 64^3 is more accurate than
   OpenFOAM on 128^3, in 0.3 s against 291 s), but that number mixes method and hardware, so the
   same-grid speed-up is the one to quote.

**Limits of this comparison** (also what keeps it honest):
- gpucfd currently solves only uniform periodic boxes; OpenFOAM solves any geometry on any mesh.
  This measures what a structured, GPU-resident design gains on problems it can handle. Walls and
  other boundaries will cost some of that gain; the next rungs will show how much.
- One flow, laminar, one machine (RTX 4070 against an 8-core Ryzen 7 7800X3D).
- Total wall times (start-up, decomposition, output) favour the GPU even more (128^3: 2.4 s against
  354 s) and are recorded in the JSON.
