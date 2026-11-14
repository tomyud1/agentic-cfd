# gpucfd: a GPU-resident incompressible flow solver

Goal: a solver that keeps the whole time step on the GPU and is trusted because every part is
verified before it is made fast. Case 5 showed why: OpenFOAM's GPU plugin only moves the pressure
solve to the GPU, copies matrices through host memory every solve, and is capped near 1.5 times.

## Scope (first version)
- Incompressible, single phase, laminar (constant density and viscosity).
- Structured Cartesian grids, uniform or stretched per direction (cells clustered toward walls or
  around a body; 2026-11-13), 2D or 3D (2D = one cell in z).
- Staggered (MAC) grid: pressure at cell centres, each velocity component on its faces. Discretely
  divergence-free velocity, no pressure checkerboarding.
- Projection method: predict velocity, solve a pressure Poisson equation, correct.
- Pressure Poisson equation: geometric multigrid (red-black Gauss-Seidel smoother, cell averaging
  restriction, (bi/tri)linear prolongation), alone or as the preconditioner of conjugate gradients.
  On stretched grids: zebra line relaxation and the transposed prolongation as restriction.
- Boundaries: periodic, walls (no-slip, moving lid), inflow (prescribed velocity profile), outflow
  (pressure zero, zero velocity gradient).
- Immersed bodies of any shape given by a signed distance (ghost-cell method), with forces, on
  uniform and on stretched grids, with either time scheme (IMEX with bodies: 2026-11-13).

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
| 9 | Open boundaries | Poiseuille channel with inflow and outflow in every direction; Kovasznay flow (exact Navier-Stokes) with inflow profiles on all sides | order 2; the discrete fully developed flow kept to round-off; flow out = flow in |
| 10 | Immersed bodies | circular Couette flow between immersed cylinders and a tilted channel (exact), 3D extrusions, DFG benchmark 2D-1 | velocity order 2 (mean over the grids); 3D = 2D; drag, pressure difference and lift converge to the benchmark; forces independent of the box |
| 11 | Steady-state tooling | iteration-error estimator on sequences with a known limit; prolongation of a converged Couette flow; three-grid Couette study sequenced and from rest | estimate exact for one mode, never too small otherwise, noise floors recognised; prolongation keeps the coarse accuracy; sequenced = from rest within the stop tolerances, and cheaper |
| 12 | Stretched grids (cells clustered toward walls) | the new code path with uniform face positions against the uniform code; Laplacian against numpy; direct solve; manufactured solutions; multigrid against the cell aspect ratio; kinetic energy of convection; Poiseuille, channel mode, Kovasznay, open channel; cavity Re 100 | equal to round-off; order 2; iterations independent of the grid and within twice the uniform count up to aspect ratio 64; energy conserved to round-off; cavity within 0.01 U of Ghia and converging to the uniform solution |
| 13 | Immersed bodies on stretched grids | the stretched code path with uniform face positions against the uniform code with bodies; force balance of a random field; circular Couette flow on grids whose cell size varies by 3; 3D extrusion; DFG 2D-1 on grids clustered around the cylinder | equal to round-off; force zero to round-off; velocity order 2 (mean over the grids); 3D = 2D; drag converges to the benchmark and equals the uniform grid's at the same cells per diameter within its error; forces independent of the box; reproducible bit for bit |
| 14 | IMEX (implicit diffusion) with immersed bodies | growth of the linear step on the channel with a cylinder at five time steps, two positions and a clustered grid; the Re 100 flow at CFL 0.8; ghost rule and divergence after a step; circular Couette flow; steady states against the explicit scheme (Couette, DFG 2D-1); order in time on the Re 100 channel; 3D extrusion | no growth; ghost faces follow their rule to round-off and the velocity is divergence-free with them; velocity order 2; steady state equal to the explicit scheme's (1e-7) and independent of the time step; order 2 in time; 3D = 2D; reproducible bit for bit |

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
- When a scheme is unstable, model the linear part of the step as matrices and compute its growth
  before trying fixes in CUDA (`tests/imex_body_model.py`: seconds per variant, growth rates equal
  to the solver's to 3 digits; it takes any time scheme or ghost rule). Stage 3: trying variants in
  the solver had ended in "not understood"; the model found two causes in an afternoon.
- Before writing down an expected speed-up, time about 200 steps of a prototype. Stage 3's estimate
  (study 550 to 800 s, measured 1005 s) assumed a cost per step and a pressure tolerance that a
  build already on disk would have corrected.
- After a change that is meant to alter results, save a new reference for `tests/same_as_build`
  with the new build, so that the next speed-only change is compared against it.

## Layout
```
gpucfd/
  DESIGN.md        this file
  build.sh         compiles src/ into build/libgcfd.so (nvcc, run inside WSL)
  src/             CUDA C++ (header-only pieces + one .cu with the C API)
  gcfd.py          Python wrapper (ctypes); numpy arrays in and out
  grids.py         face positions for stretched grids (tanh toward walls, a fine band around a body)
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
(rung 6), walls (rung 7), reproducibility (rung 8), inflow and outflow (rung 9), immersed bodies
(rung 10), and two time schemes: explicit RK3 and IMEX (implicit diffusion). Run the whole ladder
with `bash gpucfd/run_ladder.sh` (or `python -m gpucfd.tests.test_poisson | test_ns | test_walls |
test_open | test_bodies | test_steady | test_stretched | test_stretched_bodies | test_imex_bodies |
test_reproducible`); results in `gpucfd/tests/results/`. Stretched grids (rung 12), bodies on
them (rung 13) and IMEX with bodies (rung 14) were added on 2026-11-13.

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
| 8 | Reproducibility: two identical runs are bit-identical, and CUDA graphs on/off are bit-identical (Poisson double/mixed; cavity, 3D periodic flow and a channel with inflow/outflow, each with both schemes; immersed bodies in an open channel and a closed Couette cell) |
| 9 | Open boundaries (`test_open`), each check with both schemes: Poiseuille channel with parabolic inflow and outflow along +x, -x, +y and z (3D) converges at order 1.84, 1.93 (N = 16, 32, 64; error = 0.99 h^2 - 2.84 h^3, so coarse pairs read low); fed the discrete equations' own fully developed profile (computed separately in numpy), the whole channel keeps it to 6e-16 (RK3) / 1e-14 (IMEX, diffusion solves to 1e-14); flow out = flow in exactly; the -x channel equals the mirrored +x channel to 2e-14; started from rest it reaches the same state. Kovasznay flow (Re 40, exact Navier-Stokes with convection; velocity profiles prescribed on all four sides): L2 order 2.02, 2.01 (32^2-128^2); the max error, in the first cell row along a side, converges at 1.90-1.92 out to 256^2 while the interior converges at 2.00 |
| 10 | Immersed bodies (`test_bodies`, explicit scheme): circular Couette flow between immersed cylinders, velocity order 2.50, 2.10 (rms), pressure away from the walls 2.37, 1.48; channel at atan(1/2) to the grid, mean order 2.24; 3D extrusions equal 2D; DFG 2D-1 drag -2.26 % / -0.65 % (D/h 17.6 / 35), pressure difference -0.23 % at D/h 35 for this position of the cylinder in the grid (-0.2 to -2.4 % over four positions, drag -0.62 to -0.85 %: stage 2, check runs); forces exact discrete momentum balance (box-independent to 2e-18). Details: Immersed bodies, below |
| 11 | Steady-state tooling (`test_steady`, `gpucfd/steady.py`): iteration-error estimator exact for one geometric mode, conservative (9x) for an oscillating one, never too small while a fast mode dies out once the slow rate is known, noise floors recognised in 97 % of noisy histories and sized (covers the true error 97 % of the time); prolongation 32^2 -> 64^2 of the Couette flow keeps the coarse error (5.5e-3 vs 6.0e-3); Couette 32/64/128 sequenced vs from rest: same answers (worst 0.29 of the allowance), 7.1 s vs 14.2 s GPU on the refined grids. DFG 2D-1 study (one-off): 20x at D/h 35. Details: Steady flows, below |
| 12 | Stretched grids (`test_stretched`, 2026-11-13): with uniform face positions the new code path equals the uniform code (Laplacian exactly, solutions 1.5e-11, Navier-Stokes 1e-14 / 7e-16); stretched Laplacian equals the numpy reference to 6e-15, solutions equal a direct solve to 2e-12; manufactured solutions order 1.99-2.01 (2D and 3D, cell ratio 8); pressure iterations 9 / 10-11 / 11-12 at wall aspect ratio 4 / 16 / 64 on 64^2 to 512^2 (uniform 8-9); convection conserves kinetic energy to 1e-17; Poiseuille from rest order 2.03 (IMEX) and 2.05 (RK3), decaying channel mode 2.03-2.04 (2D and 3D, both schemes), Kovasznay L2 order 2.07, 2.02, open channel 2.04 with flow out = flow in exactly; both schemes reach the same steady state (cavity 5e-9 U, Kovasznay 2e-8, open channel 1e-9); cavity Re 100 on clustered grids within 0.005 (u) / 0.009 (v) U of Ghia and 0.0052, 0.0013, 0.00025 U from the uniform 256^2 solution at N = 32, 64, 128; reproducible bit for bit. Details: Stretched grids, below |
| 13 | Immersed bodies on stretched grids (`test_stretched_bodies`, 2026-11-13): force balance of a random field 7e-16; with uniform face positions the stretched code path has identical ghost rules and equals the uniform code after 100 steps (velocity 5e-13 closed cell, 5e-11 open channel; force 2e-13); circular Couette flow with cell size varying by 3, velocity order 2.69, 2.34, divergence 3e-13, 3D = 2D to 6e-13; 5 pressure iterations per projection with a body (uniform: 7.7); DFG 2D-1 drag -2.23 % / -0.56 % at D/h 17.5 / 35 with 3.85 times fewer cells than the uniform grids at -2.26 % / -0.65 %; forces box-independent to 5e-18; reproducible bit for bit. The surface pressure difference depends on where the grid cuts the cylinder (-0.2 to -2.4 % at D/h 35, both grid types). Details: Stretched grids, stage 2, below |
| 14 | IMEX with immersed bodies (`test_imex_bodies`, 2026-11-13): no growth in 15 linear runs (nu dt / h^2 from 0.01 to 4, two positions of the cylinder, a clustered grid; before: 1.17 and 1.59 per step) nor in the Re 100 flow at CFL 0.8; after a step the ghost faces follow their rule to 3e-17 (explicit scheme: 1e-4) with max divergence h / U 2e-8; circular Couette flow rms error 6.00e-3, 1.06e-3, 2.49e-4, the explicit scheme's; steady state equal to RK3's to 8e-13 (uniform) and 4.6e-8 (stretched), independent of the time step to 6e-13; DFG 2D-1 drag equal to 3e-8; order in time 2.07, 2.03; 3D = 2D to 5e-13; reproducible bit for bit. Details: IMEX with immersed bodies, below |

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

### Open boundaries (2026-11-12)
Each bounded side is a **velocity side** (wall or inflow: velocity prescribed, pressure gradient
zero) or an **outflow** (pressure zero, zero normal gradient of the velocity: OpenFOAM's usual
outlet). Prescribed values can vary along a side; they are stored per side and component on the
side's plane, sampled where that component lives (`NavierStokes.side_coords`).

On the staggered grid the normal velocity of the low side is face 0 of the stored array, and that of
the high side is face n, which is not stored: it lives in the side's plane array. Boundary faces are
never evolved by the momentum equation. On an outflow side the boundary face is set from the next
interior face before every projection and then corrected by the projection like any interior face,
with the pressure's ghost value -phi beyond the side. That correction is exactly the one the
Dirichlet pressure stencil assumes, so the projected velocity is divergence-free and the flow out
equals the flow in to round-off with no rescaling. With an outflow the pressure problem is no longer
singular; without one, the prescribed velocities must carry zero net flow (`NavierStokes` refuses
otherwise). For the implicit diffusion (IMEX), the high side's boundary values enter the
right-hand side (the operator stays linear and symmetric) and are refreshed every stage when there
is an outflow.

The wall-only paths are unchanged: all 18 results of `tests/same_as_build.py` are bit-identical
to the build before this change.

### Immersed bodies (2026-11-12)
`NavierStokes(..., bodies=[...])` with shapes from `gpucfd/bodies.py` (Cylinder, Sphere, Box,
HalfSpace, Inside for fluid inside a shape, SDF for any exact signed distance), each optionally with
a velocity field (`bodies.rotation`). The geometry is fixed in time; surfaces may move tangentially.

**Velocity: sharp-interface ghost-cell method.** Every face of a velocity component where the signed
distance is <= 0 is not fluid. A ghost face G (within 1.5 h of the surface) takes the value on the
straight line along the surface normal through the body's velocity at the closest surface point B and
the fluid velocity at the image point I = B + s_I n:  u_G = U_B + (u_I - U_B) phi_G / s_I. u_I is
interpolated (bi- or trilinear) from fluid faces only, which s_I >= sqrt(dim) h guarantees for an exact
signed distance (the build checks every stencil and moves I out if needed). Deeper faces take the
body's velocity. The rules are built once in Python (`bodies.build`) as lists of (face, a0, 4 or 8
weighted neighbours) and applied on the GPU before every projection (`k_ib_set`). The fluid faces then
see the bodies through their usual stencils: nothing else in the momentum equation changes.

**Pressure.** The projection solves the Poisson equation with a zero coefficient on every face in a
body (zero pressure gradient through it: a staircase wall for the pressure, the usual choice of
ghost-cell methods) and corrects only fluid faces. So the faces in bodies keep their rules after the
projection, and with an incremental projection (the explicit scheme carries the pressure between
stages when bodies are present) the increment vanishes at a steady state: the ghost rules and the
divergence constraint then hold exactly together. Cells with no fluid face are decoupled.

Found while building (each a measured failure first):
1. With only the finest multigrid level masked and the one-block tail ignoring the masks, a 32^2 grid
   (which lives entirely in the tail) took 200 CG iterations: the tail and the coarsest solve now
   honour masks (and remove the mean over the fluid cells only).
2. Masks averaged onto coarser levels made the preconditioner fail outright for a cylinder in an
   open channel (200 iterations, the cap): averaged coefficients create coarse cells that are nearly
   cut off, whose corrections overshoot. Only the finest level is masked now: 14-15 iterations per
   projection at 128^2 and 256^2 (7 without bodies), independent of the grid.
3. A closed fluid region (Couette cell) is singular, and the extrapolated rim velocities carry a small
   net flow into it, so the leftover divergence was spread uniformly over the flow (5e-5 after one
   step) and the CG tolerance became unreachable. The rim faces now get the least correction that
   makes the net flow exactly zero (`k_ib_cons`): divergence at the solver tolerance.
4. Near a steady state the projection's right-hand side tends to zero and a tolerance relative to it
   asks for more digits than round-off leaves (200 iterations instead of 2). With bodies the tolerance
   is relative to max(|div u*|, sqrt(n) max|u| / h): an rms divergence of ptol max|u| / h.
5. **IMEX with bodies is unstable and refused** (`scheme="rk3"` is required). Measured: about 25 %
   growth per step, also with convection negligible, with the diffusion fully implicit, with the wall
   rules as exact rows of the diffusion solve (BiCGStab), with or without re-extrapolating after the
   projection; it disappears when the incremental pressure is switched off, which would leave a
   time-step-dependent slip at the walls. RK3, with the same incremental pressure, is stable. Not
   understood yet; the cost is that diffusion limits RK3's time step on fine grids (DFG at D/h 70:
   8 times below the convective limit). *Resolved 2026-11-13: two causes, a pressure gradient
   applied inside the body and the ghost faces' answer to the pressure correction fed back into
   the pressure; see "IMEX with immersed bodies, stage 3" below. IMEX with bodies is no longer
   refused.*

**Forces.** `forces(box)`: summed over the fluid faces in a box around the bodies, the solver's own
discrete momentum equation telescopes to the fluxes through the box surface plus those into the
bodies, so F = -[sum_box (rhs - grad p - g) vol + sum_surface (flux out) area] with the solver's
fluxes is the exact discrete force, including the unsteady term. For a random field and no body it
returns 6e-16; two different boxes around the DFG cylinder agree to 2e-18. *Added 2026-11-14 for case 5
(a cylinder from wall to wall):* the box may reach a wall. Before, `forces` refused a box at a
non-periodic side and a box whose faces cut a body, and for a body that touches a wall every box does
one or the other. Now the friction on the piece of wall inside the box enters the balance (the
solver's own wall rule, the mirror value 2 u_wall - u at the mirror distance), so the result is still
the force on the bodies alone. The component normal to that wall is not computed (NaN). Walls only,
not inflow or outflow sides; Python only, no change to the solver. Checks
(`tests/test_wall_forces.py`, in `run_ladder.sh`): no body and a random field, boxes from wall to
wall, uniform and stretched grids, a moving wall: 2e-16; a cylinder from wall to wall, two boxes:
equal to 1e-16. `surface_pressure(points)`
extrapolates the pressure to the surface along the normal (quadratic through samples at 1.5, 2.5,
3.5 h in the fluid).

**Rung 10 results** (`test_bodies`, RK3):

| Check | Result |
|---|---|
| Circular Couette flow between a rotating inner and a fixed outer immersed cylinder (closed region), exact | velocity rms error 6.0e-3, 1.1e-3, 2.5e-4 (N = 32, 64, 128 across the box): order 2.50, 2.10; max error order 2.79, 1.71 |
| Couette pressure | rms error beyond 1.5 h of the walls: order 2.37, 1.48 (mean 1.9). The max error sits in the cells next to the walls (within 0.02-0.3 h) and does not converge cleanly: 1.9e-2, 6.8e-3, 9.8e-3 |
| Poiseuille flow in a channel at atan(1/2) to the grid (flat walls, generic angle) | rms error 2.8e-3, 3.3e-4, 1.3e-4: pairs 3.12, 1.36, mean 2.24 |
| 3D: Couette flow extruded along z and along x | equals the 2D solution to 8e-13 / 1e-12 (with the 3D image distance in both: it is sqrt(dim) h, so the default 2D and 3D discretisations differ, by 9e-3 at N = 32) |
| Started from rest | same steady state to 1.4e-12 |
| DFG 2D-1 (Re 20), reference cD 5.57954, cL 0.010619, dP 0.117520 (Nabh 1998) | D/h 17.6: cD -2.26 %, dP -3.24 %, cL 4.3 times the reference; D/h 35: cD -0.65 % (order 1.8), dP -0.23 %, cL +15 %. One position of the cylinder in the grid: over four positions at D/h 35, cD -0.62 to -0.85 %, dP -0.23 to -2.39 %, cL +15 to +175 % (`tests/shift_study.py`) |

Immersed-boundary errors fluctuate with how the surface cuts each grid (a single grid pair can read
1.4 or 3), so the orders are judged on the mean over the range. The lift in 2D-1 is 0.2 % of the drag,
so its relative error is large on coarse grids; it falls with refinement.

### Steady flows: grid sequencing and an error-based stop (2026-11-12)
Steady flows are reached by marching in time. With bodies that march had to use RK3 until 2026-11-13
(IMEX was unstable there, above), so on fine grids the time step is small (diffusion limit 0.15 h^2 / nu). DFG 2D-1 at
D/h 35 spent 398 GPU-s settling from rest; the D/h 70 run from rest was stopped unfinished after more
than 9400 GPU-s. Two things cut that, both in `gpucfd/steady.py` (the estimator is in
`cfdlib/verification.py`, so OpenFOAM runs can use it too):

- **Error-based stop (`march`).** Close to steady state the slowest mode dominates, and each change of
  a monitored quantity (drag, lift, a pressure difference) is about rho times the one before, so the
  error still left is |last change| rho / (1 - rho) (`iteration_error`). The run stops when that is
  below a tolerance on two consecutive chunks. A study needs the iteration error only well below the
  grid's discretisation error. On grid k that is known while it runs, about |f_k - f_(k-1)| / (r^p - 1),
  and the tolerance is 1 % of it (p observed from three grids, else 2). The old rule (change below
  1e-6 per time unit) kept marching long after the answer was good to the grid's accuracy.
- **Grid sequencing (`prolong`, `sequence`).** Each finer grid starts from the coarser solution,
  interpolated (bi/trilinear) and projected, instead of from rest. `sequence(target=...)` also stops
  refining once every quantity's discretisation error estimate is below a target.

Pitfalls found while building it (each would have stopped runs too early, or never):
1. A slow mode hidden under a fast one is invisible in the history: while the fast one dominates, the
   estimate is up to 11x too small (synthetic two-mode test). Fixes: the estimate is trusted only once
   the ratio has settled (1 - rho agrees within 1.5x over the window); finer grids take the coarser
   grid's measured slow rate as a floor for rho (same physics, same chunk length); the first grid gets
   a minimum time from a physical time scale (`t_min`: 30 for DFG 2D-1, about 3 flow-through times).
2. Solver noise. Monitored quantities do not settle below a floor set by the solver tolerances (the
   RK3 pressure is updated by phi / (b dt), so the projection tolerance is amplified as dt shrinks:
   DFG lift noise 2e-6 at D/h 17.6, 4e-5 at D/h 70). There the changes flip sign without shrinking.
   The estimator recognises that (at least 3 sign flips in 8 changes, the last 4 not smaller than 0.2
   of the first 4) and returns the noise size as the error; the march stops (more marching cannot
   help) and flags the quantity noise-limited if the floor is above its tolerance. Without this the
   first DFG run marched 200 time units past convergence. The rate floor handed to the next grid must
   come from a settled estimate, not from noise (rho 2.8 there would have stopped nothing).
3. `cfdlib.verification.level_sequence` returned a Richardson value for any observed order. A Couette
   pressure probe less than one cell from the wall on 16^2 gave order 0.006 and a "Richardson" value
   11 times the exact one. It now refuses below order 0.5 (`plausible` False). The rung-11 probes sit
   at least 2.6 cells from both walls.
4. Rung 11's divergence bound after prolongation was first set at 1e-10 without checking what
   `project()` achieves. It reaches max |div u| h / U = 3.6e-10, the projection tolerance; the bound was
   changed to 1e-8 after seeing that number.

**DFG 2D-1 on three grids** (Re 20; one-off run, `t_min` 30, chunks of 1 time unit):

| D/h | start | GPU-s | t | cD | cL | dP |
|---|---|---|---|---|---|---|
| 17.6 | rest, old stop | 15 | 14 | 5.45362 (-2.26 %) | 0.04618 | 0.11371 (-3.24 %) |
| 17.6 | rest, new stop | 23 | 31 (t_min) | 5.453616 | 0.046178 (noise-limited) | 0.1137133 |
| 35 | rest, old stop | 398 | 128 | 5.54312 (-0.65 %) | 0.01222 | 0.11725 (-0.23 %) |
| 35 | rest, new stop | 105 | 31 (t_min) | 5.543124 | 0.012219 | 0.1172475 |
| 35 | sequenced, new stop | **19.6** | 6 | 5.543128 | 0.012228 | 0.1172476 |
| 70 | rest, old stop | > 9400, stopped | - | - | - | - |
| 70 | sequenced, new stop | **331** | 12 | 5.571607 (-0.14 %) | 0.010084 (-5.0 %, noise-limited) | 0.116970 (-0.47 %) |

At D/h 35 the new stop gives 3.8x and sequencing another 5.4x (20x together; the from-rest run with
the new stop ended at t_min, so part of its time is that choice). All runs agree within their stop
tolerances. The whole three-grid study took 374 GPU-s.

What the third grid shows (reference cD 5.57954, cL 0.010619, dP 0.117520):
- Drag converges at observed order 1.65. Richardson gives 5.5849 (+0.10 %), no better than the finest
  grid (-0.14 %).
- Lift has overshot the reference (+335 %, +15 %, -5 %). The three-grid "order 4" is spurious and its
  Richardson value (-6.4 %) is worse than the finest grid. Extrapolated values are estimates to report
  next to the finest grid, never the answer.
- **Open:** the pressure difference gets worse on the finest grid (-0.23 % at D/h 35, -0.47 % at
  D/h 70). It comes from the surface pressure, extrapolated along the normal from 1.5-3.5 h, and the
  immersed-boundary pressure next to a body converges irregularly (rung 10: max error near the walls
  1.9e-2, 6.8e-3, 9.8e-3). Drag and lift come from the momentum balance away from the body and do not
  show it. Rung 10 runs only D/h 17.6 and 35, so it does not see this.

### gpucfd as the agent's solver: case 4, DFG 2D-2 (2026-11-12)

`cfdlib/gpu.py` (check_fit, run_transient, write_run), `cases/dfg_cylinder_re100/gpu_run.py` and
`gpu_compare.py` (numbers in `results/dfg_cylinder_re100/gpu_compare.json`). The unsteady cylinder at
Re 100 that case 4 solved with OpenFOAM, run by gpucfd and judged by case 4's own rules.

**Set-up**: uniform grids over the whole channel, n = (384, 72) x 2^k (D/h 17.6, 35, 70), cylinder as an
immersed body, RK3 (bodies need it), dt = min(0.8 h / 1.5 Um, 0.15 h^2 / nu); forces from the momentum
balance on a box around the cylinder, sampled every 2-2.5 ms (about 150 samples per period, peaks
parabola-refined as in case 4). The same protocol as OpenFOAM: G1 from rest for 15 s, each finer level from
the coarser one's last field (prolonged). Rules fixed before the runs (numerics.yaml, gpucfd block).

**Two changes after the first runs, both recorded in numerics.yaml and the case's insights 17 and 18**:
- Pressure tolerance tuned like OpenFOAM's GAMG: 1e-8 instead of 1e-10 (drag and lift within 2.6e-5 of the
  strict run over 1 s, 1.7x less time per step; 1e-6 is off by 7.9e-3).
- Each level stops on case 4's periodicity rule, checked live every second, instead of after OpenFOAM's
  fixed 3 s. G1's drag is 16 % low, so G2 and G3 were still drifting at 3 s (G3 periodic at 5 s).

**Result** (FeatFlow level 6 reference; OpenFOAM M4 = case 4's finest O-grid):

| Quantity | gpucfd G3 | vs FeatFlow | OpenFOAM M4 | vs FeatFlow | G3 - M4 within both bands? |
|---|---|---|---|---|---|
| cd max | 3.1866 | -1.04 % | 3.2280 | +0.25 % | yes (bands wide) |
| cd mean | 3.1569 | -0.99 % | 3.1964 | +0.25 % | yes (bands wide) |
| cl max | 0.9528 | -3.36 % | 0.9900 | +0.41 % | **no** |
| cl amplitude | 1.9446 | -3.16 % | 2.0147 | +0.33 % | yes (bands wide) |
| St | 0.30276 | +0.29 % | 0.30150 | -0.12 % | yes |

- Verdict by case 4's rules: WARN (lift maximum more than one band off; lift minimum order 0.78). Every
  primary quantity is inside two bands, but only because the bands are wide: they are the GCI of G1-G3,
  and G1 is far from converged (drag -16 %), so the "ok" statuses prove little. The expectation set
  before the runs (drag within 0.3 %) was missed: -1.0 %.
- Drag converges at order 1.5 (as on DFG 2D-1: 1.65), the Strouhal number at 1.9. Richardson from
  G1-G3 overshoots (drag +1.3 %): G1 is outside the asymptotic range.
- Time error: G2 against G2 with dt / 2 differs by 2e-5 in the drag peaks; negligible.
- Mass: in the developed flow, inflow and outflow differ by 2.7e-3 of the flow on G1 and 4e-5 on G3
  (flow into the cells cut by the body; it shrinks with the grid). The largest values, right after each
  start, are 2.5e-2 and 3.3e-4.

**Cost** (wall time for each tool's whole protocol on this PC):

| | gpucfd (RTX 4070) | OpenFOAM (Ryzen 7 7800X3D) |
|---|---|---|
| levels | G1, G2, G3: 27648, 110592, 442368 cells | M1-M4: 3358 to 214912 cells, up to 16 ranks |
| total | 280 s GPU, 328 s wall (first runs: 255 s, 315 s) | 2548 s wall (M4 alone 2110 s on 16 ranks) |
| per cell and step | 0.033 microseconds (G3; first run 0.023) | 1.64 microseconds (M4, wall, 16 ranks) |
| finest drag error | -1.0 % | +0.25 % (M2 already +0.24 %, 97 s on one core) |

(Updated 2026-11-13: G3 was rerun with the pressure tolerance its own rule gives, 1e-9 instead of the 1e-8
tuned on G1; the coefficients are unchanged to the digits shown, a step costs 14.7 ms instead of 10.3. See
"Stretched grids, stage 2", check run 2.)

So gpucfd ran the protocol 8 times faster and is about 50 times faster per cell and step (70 with the first,
looser tolerance; corrected 2026-11-13: that sets gpucfd's GPU time against OpenFOAM's wall time. With wall
time on both sides it is 45 times on G3 and 25 times on the clustered C3), but it is not
faster at equal accuracy: OpenFOAM's O-grid reaches 0.25 % at 13432 cells in 97 s, which gpucfd would
need roughly 140-280 cells per diameter for (estimated from order 1.5; not run: D/h 140 is 1.8 million
cells with a viscous time-step limit, about 40 GPU-minutes). The cost is in the grid, not the solver:
a uniform grid has to resolve the boundary layer everywhere, and the immersed body converges at order
1.5. Stretched grids (next steps) attack exactly that. (Done 2026-11-13: with cells clustered around the
cylinder, 140 cells per diameter fit in 459 thousand cells and give the drag within 0.11 % of the benchmark's
finest tabulated level (0.31 % below the value it converges to) in 1 421 s for three levels. See "Stretched
grids, stage 2", and the study report `results/dfg_cylinder_re100/gpu_study.html`.)

### gpucfd on case 5, the 3D cylinder from wall to wall (2026-11-14)

Plan, script and records: `dev/compare/dfg_cylinder_3d_re20/` (the plan was written before the runs).
DFG 3D-1Z, Re 20, steady: drag and lift of a cylinder that spans a square channel.

- **What had to be added:** `forces` for a box that reaches a wall (Immersed bodies, Forces, above).
- **Grid:** uniform in y and z, clustered along x only (cells of size h = H / ny from one diameter
  ahead of the cylinder to four behind, growing to 3.3 h away from it). Clustering in two directions
  would need plane relaxation in the pressure solver (stage 1's open item); with one direction the
  pressure took 3 iterations per projection. IMEX at CFL 0.8, pressure tolerance 1e-9, each grid
  started from the one before (`steady.march`, `steady.prolong`).

| Grid | Cells | D/h | Steps | GPU s | cD (reference 6.18533) | cL (reference 0.009401) |
|---|---|---|---|---|---|---|
| G1 | 192 x 64 x 64 = 0.79 M | 15.6 | 5148 | 237 | 6.03409 (-2.44 %) | +0.01251 |
| G2 | 288 x 96 x 96 = 2.65 M | 23.4 | 1386 | 220 | 6.09622 (-1.44 %) | -0.00684 |
| G3 | 384 x 128 x 128 = 6.29 M | 31.2 | 1584 | 600 | 6.12926 (-0.91 %) | +0.02210 |

- **Drag:** within 1 % on G3, as the 2D version of the flow suggested (-0.65 % at D/h 35). OpenFOAM's
  body-fitted grid: -0.08 % with 2.4 M cells. The three values give an observed order of 0.83 and a
  wide error band (2.5 %); against the reference the last pair converges at order 1.6.
- **Lift: not converged.** It changes sign from grid to grid. The lift is 0.15 % of the drag, below
  the error the ghost-cell surface makes in the forces on these grids (rung 10 saw the same in 2D: +335 %,
  +15 %, -5 % at D/h 17.6, 35, 70).
- **Cost:** about 60 ns of GPU time per cell and step (46 ms per step on G1, 380 ms on G3), 2.5 GB for
  G3. 12 GB would hold about four times G3's cells in this family. Matching OpenFOAM's drag in this
  family would need about 140 cells per diameter, 600 million cells: it needs fine cells around the
  cylinder in y and z too, so plane relaxation (or semi-coarsening), or a more accurate surface rule.
- **Seen on the way:** G1's forces had settled after one flow-through time (12 s of 37.5 s marched:
  the minimum time of three flow-throughs was too cautious). Flow out differs from flow in by 1.6e-3
  (G1), 4e-4 (G2 and G3): the cells cut by the body, and it did not shrink from G2 to G3. One position
  of the cylinder in the grid only; times taken while another solver used 8 CPU cores.

### Stretched grids, stage 1: boxes with walls, no bodies (2026-11-13)

**Why.** On a uniform grid one thin layer at a wall sets the cell size for the whole domain. A
stretched grid keeps the same number of cells per direction but lets their widths vary: small at
the walls, large in the middle. This is stage 1 of three (ROADMAP step 2): the discretisation and
the pressure solver on such grids, for boxes with walls, inflow/outflow and periodic sides. Bodies
on stretched grids and IMEX with bodies are stages 2 and 3 and were refused at this stage (stage 2:
below).

**Design, written before any code was run.**

*The grid.* Still a box of nx x ny x nz cells in rows and columns (a "rectilinear" or
"tensor-product" grid): direction d has n+1 face positions x_0 = 0 < x_1 < ... < x_n = L, and cell i
lies between faces i and i+1. Every row of cells has the same widths, so the grid is described by
three short lists, not by a mesh file. Two lengths per direction appear everywhere below:
- the cell width dc_i = x_(i+1) - x_i;
- the distance between neighbouring cell centres dm_i = (dc_(i-1) + dc_i) / 2, which belongs to
  face i. On a bounded side the neighbour beyond the wall is the mirror image of the first cell,
  so dm = dc there.

On a uniform grid both are h and every formula below turns into the existing one.

*Python API.* `NavierStokes(..., grid=(gx, gy, gz))` and `Poisson(..., grid=...)`: each entry is
`None` (uniform) or an array of the n+1 face positions. `gpucfd/grids.py` makes the usual ones, with
the parameter an engineer thinks in: `grids.tanh(n, L, ratio)` clusters toward both ends with
`ratio` = largest cell / smallest cell (hyperbolic-tangent spacing, the standard for channel
flows), `grids.tanh(n, L, ratio, side="low")` toward one end. `grids.describe(faces)` returns the
smallest and largest cell and the largest step between neighbouring cells. Periodic directions must
stay uniform. `s.dt_cfl(cfl)` gives the convective time-step limit from the actual velocity in each
cell (small cells at a no-slip wall carry small velocities, so the smallest cell times the largest
velocity would be far too strict).

*How the cell sizes enter the equations (the "metric terms").* The staggered layout does not
change: pressure in the cell centre, each velocity component in the middle of its faces. Each
unknown still owns a small box (its control volume), and every term is still a balance of what
crosses that box's sides, divided by the box's size. Only the sizes now come from the lists:
- Divergence of a cell: (flow out - flow in) / dc per direction. Exact for any widths.
- Pressure gradient at a face: (p_i - p_(i-1)) / dm_i.
- The pressure Laplacian is the divergence of that gradient, so the projection still makes the
  velocity divergence-free to round-off.
- Diffusion of a velocity component: across its own direction its box runs from cell centre to cell
  centre (width dm), and the gradients on the box's sides use dc; across the other directions the
  box is the cell (width dc) and the gradients use dm. Beyond a wall the ghost value sits at the
  mirror distance, as before.
- Convection: the momentum flux through a box side is (flow through the side) x (velocity carried).
  The flow through a side that spans two half cells is the sum of the two half-cell flows, so the
  box of every velocity unknown is itself exactly divergence-free (mass-conserving); the carried
  velocity is the plain average of the two neighbours. This is the "symmetry-preserving" form of
  Verstappen & Veldman (2003): like the uniform scheme, convection neither creates nor destroys
  kinetic energy, on any grid.

Accuracy: the face is no longer midway between the two cell centres, so a single difference is
formally first order. On a *smooth* grid, where neighbouring widths differ by O(h) (the tanh grids
do), the errors of neighbouring cells cancel and the solution is second order. The ladder checks
that by measurement.

*Symmetry and the iterative solvers.* Conjugate gradients needs a symmetric matrix. Row i of the
Laplacian is divided by dc_i, so with unequal cells the matrix is no longer symmetric; multiplied
by the cell volume it is. So the solvers keep the plain operator and weight every dot product by the
volume (cell volume for the pressure, the velocity's box volume for the implicit diffusion): the
same thing, done without changing the operator. A singular pressure problem needs the
volume-weighted sum of its right-hand side to vanish (it does: that sum is the net flow through the
boundary) and the pressure is returned with zero volume-weighted mean.

*The pressure solver on long thin cells (the hard part).* Multigrid works by letting a cheap
"smoother" remove the error that changes from cell to cell and leaving the smooth remainder to
coarser grids. The red-black Gauss-Seidel smoother updates one cell at a time from its neighbours.
In a cell 64 times longer than it is thin, the two neighbours across the thin direction are
(64)^2 = 4096 times more strongly coupled than the other two. The point smoother then only smooths
along the thin direction; errors that wiggle along the long direction survive, the coarse grid
cannot represent them, and convergence stalls. The two classic cures:
1. **Line relaxation.** Update a whole line of cells along the strongly coupled direction at once,
   by solving the small tridiagonal system that couples them exactly. The strong coupling is then
   solved, not iterated. Alternate lines are coloured like a zebra so that all lines of one colour
   are independent and can run in parallel.
2. **Semi-coarsening.** Keep the point smoother but coarsen only along the strongly coupled
   direction, until the cells are roughly square.

**Choice: alternating zebra line relaxation with the existing full coarsening.** Reason: on a
wall-clustered grid the strong direction changes from place to place (thin in y at the bottom wall,
thin in x at the side walls, square in the corners and the middle), and case 4 will cluster in both
directions around the cylinder. One coarsening rule for the whole grid cannot follow that;
semi-coarsening would need several coarse grids per level. Line relaxation along x and then along y
handles any 2D mix, and it keeps the grid hierarchy unchanged. Lines run along every bounded
direction (a periodic direction is uniform and would need a cyclic line solve; not built).

On the GPU: one thread block per line. Its threads build the line's tridiagonal system in parallel
(the memory traffic), one thread solves it by the Thomas algorithm in shared memory (a short
sequential job on fast memory), and all threads write the result back. The coarsest level (a few
cells) is solved exactly: its inverse matrix is computed once on the CPU. The one-block V-cycle
tail is not used on stretched grids yet.

Known weak spot, stated before measuring: in 3D a cell that is thin in two directions (along an
edge of a duct) is strongly coupled in a plane, which needs plane relaxation; lines are then not
enough. If line relaxation fails the criteria, the second strategy to try is semi-coarsening; if
both fail after a serious attempt, the work stops and what was measured is written down. (Outcome: it did not fail; see Results.)

*Uniform grids.* Every kernel keeps its uniform code path and takes the new one only when face
positions were given, so uniform results must stay bit-identical.

**Pass criteria, fixed in advance.**
1. Uniform grids stay bit-identical to the build before this change (`tests/same_as_build.py`
   against the old library; `tests/test_reproducible.py`), and the whole existing ladder still
   passes (`bash gpucfd/run_ladder.sh`).
2. New rung 12 (`tests/test_stretched.py`) on wall-clustered grids:
   - the stretched code path given uniform face positions equals the uniform code to round-off;
   - the Laplacian equals an independent numpy reference to round-off; solutions equal a direct
     solve; manufactured solutions converge at order 2 (1.9 to 2.1 on the finest pair);
   - plane Poiseuille flow from rest (walls clustered, both time schemes): order 2; Kovasznay flow
     (exact, with convection, clustered in both directions): order 2 in L2; channel with inflow and
     outflow: flow out = flow in to round-off;
   - lid-driven cavity at Re 100 on clustered grids: centreline velocities within 0.01 U of Ghia
     (the uniform rule) and converging to the uniform-grid solution;
   - two identical runs bit-identical, CUDA graphs on = off.
3. Pressure-solver cost against the cell aspect ratio at the wall (1, 4, 16, 64; cavity grids):
   iterations per solve and time per time step, as a table here, for the point smoother and the
   line smoother. Wanted: iterations independent of grid size and within 2x of the uniform count
   at every aspect ratio. Where it degrades is stated.
4. A first fair number for the benefit. Test: the Re 100 cavity (the ladder's wall benchmark).
   Error = largest difference of the two centreline velocity profiles from a fine uniform
   reference (512^2, interpolated with cubic splines), at the grid's own points. Reported: the
   cells and GPU wall time the best clustered grid needs to reach the error of the uniform 256^2
   grid, both with IMEX at the same CFL number. Whatever comes out is reported, also if the
   stretched grid loses.

Each run's stop: the tests stop on their own steady/time criteria as the existing rungs do; the
cavity runs stop when the velocity changes by less than 1e-7 U per time unit (the rung 7b rule) or
at t = 80.

**Results (2026-11-13).** All four criteria were measured; 1 to 3 pass, 4 gave an honest "not here".

*What changed from the plan while building, each from a measurement:*
1. **The restriction had to become the exact transpose of the interpolation.** With line relaxation
   and the usual averaging of the four children, plain multigrid converged (factor 0.1 per cycle)
   but conjugate gradients stalled (30 iterations for what multigrid alone did in 10). Measured as a
   matrix: the V-cycle was not symmetric, antisymmetric part 10-17 % and growing with the grid,
   against 1.5 % with the point smoother. Cause: a zebra sweep leaves the residual on one row of
   each coarse cell only, a quarter cell off its centre, and averaging assigns it to that coarse
   cell alone. Now a coarse cell collects the residual of every fine cell whose interpolated value
   it enters (weights 3/8 and 1/8 per direction on a uniform grid). Antisymmetric part: zero to
   round-off; eigenvalues of the preconditioned operator within 0.85-1.0.
2. **One alternating line sweep before and after the coarse correction, not two:** 9-12 iterations
   against 7-10, at half the cost per iteration (6-8 ms per solve to 1e-10 on 256^2, against 10-13).
3. **At most 128 threads per line.** With one block of 1024 threads per line only one thread works
   during the line solve and each block fills a whole multiprocessor: 23 ms per V-cycle on 1024^2.
4. **A one-block tail for the small levels was built and removed.** Bit-identical to the separate
   kernels but slower (2.1 against 1.5 ms per solve on 32^2, 7.0 against 6.4 ms on 256^2): a single
   GPU thread working through a line cell by cell is slow.
5. **Test rules.** The Kovasznay and open-channel studies with the explicit scheme were replaced
   by "the explicit scheme reaches the IMEX steady state on one grid" (2e-8, 1e-9). As first
   written (N = 16, 32, 64) the explicit Kovasznay study took 131 s and failed my extra rule on the
   max-norm order of the coarsest pair (1.69 at N = 16; L2 orders 2.16, 2.07; the errors equal the
   IMEX ones on the same grids to 3 digits). The pre-set criterion (order 2 in L2) is judged on the
   IMEX study. With velocities prescribed on every side of a clustered grid, the exact profiles
   balance the discrete flow only to O(h^2) (1.2e-3 on 32^2), which the solver refuses; the test
   shifts the outlet by the missing flow.

*Criterion 1: uniform grids unchanged.* `same_as_build`: 18 of 18 arrays bit-identical to the
library built before this change (Poisson double and mixed, cavity and 3D flow, both schemes). The
whole ladder passes, `test_reproducible` included.

*Criterion 2: rung 12,* in the status table above. Not in the pre-set list but added: convection
conserves kinetic energy to round-off on a clustered grid (1e-17 relative, 2D and 3D), which
pins down the weights of the convective fluxes; an order test alone would not.

*Criterion 3: pressure-solver cost against the cell aspect ratio* (`python -m
gpucfd.bench.bench_stretched cost`, numbers in `bench/results/bench_stretched.json`). Lid-driven
cavity, Re 100, IMEX, cells clustered toward all four walls; ratio = largest / smallest cell = the
aspect ratio of the cells at the middle of each wall. Flow developed for 2 time units, then 200
steps timed (best of 3), pressure tolerance 1e-8, time step at CFL 0.8 of the local velocities.

| Grid | Aspect ratio | Pressure iterations per projection, lines | same, point smoother | ms per step, lines | ms per step, point |
|---|---|---|---|---|---|
| 128^2 | uniform code | 7.0 | | 4.3 | |
| 128^2 | 1 (stretched code, uniform faces) | 7.0 | | 8.5 | |
| 128^2 | 4 | 7.0 | 20.7 | 9.0 | 8.0 |
| 128^2 | 16 | 8.0 | 60.2 | 10.1 | 21.1 |
| 128^2 | 64 | 9.9 | 157 | 12.2 | 50.8 |
| 256^2 | uniform code | 7.0 | | 6.7 | |
| 256^2 | 1 (stretched code, uniform faces) | 7.0 | | 15.6 | |
| 256^2 | 4 | 7.3 | 21.7 | 17.1 | 13.8 |
| 256^2 | 16 | 8.7 | 67.0 | 20.1 | 35.7 |
| 256^2 | 64 | 9.7 | 192 | 22.9 | 95.7 |

- **Iterations:** with line relaxation 7 to 10 per projection from aspect ratio 1 to 64, the same on
  both grids. The wanted "within twice the uniform count, independent of the grid" holds. The
  Poisson test shows the same out to 512^2 (9 / 10-11 / 11-12 to 1e-10, uniform 8-9) and 14-15 at
  aspect ratio 256 (one check, 64^2 to 1024^2). The point smoother needs 3, 8 and 16-20 times more.
- **Where it degrades, 1: cost per step.** The stretched code path costs about twice the uniform
  one at the same cell count before any stretching (8.5 against 4.3 ms, 15.6 against 6.7 ms): a
  line-relaxation V-cycle is 2.5 to 3 times the price of a point-smoother one (one thread block per
  line, a 16-point restriction, no one-block tail). So at aspect ratio 4 the point smoother is
  still cheaper per step with 3 times the iterations; lines pay from about 8 up. The diffusion
  solves also need more iterations on thin cells (4 at ratio 1, 11-12 at 64; plain CG, no
  preconditioner). From aspect ratio 1 to 64 a step grows by another 45 %.
- **Where it degrades, 2: 3D grids clustered in two directions.** A box clustered toward all its
  walls (thin in two directions along the edges; the weak spot named in the design): 11-14
  iterations at ratio 4 but 23, 31, 38 at ratio 16 on 32^3, 64^3, 128^3, growing with the grid.
  Lines are not enough there; it needs plane relaxation or semi-coarsening. 3D grids clustered in
  one direction only are not affected by this (not measured separately).
- Single-precision line solves were not tested beyond aspect ratio 256.

*Criterion 4: a first fair number for the benefit* (`bench_stretched benefit`, `match`). Lid-driven
cavity at Re 100 to steady state, IMEX at CFL 2 on every grid, error against the uniform 512^2
solution (97.6 GPU-s) as fixed above.

| Grid | Cells | Error (U) | Steps | GPU time |
|---|---|---|---|---|
| uniform 64^2 | 4 096 | 1.53e-3 | 864 | 2.1 s |
| uniform 128^2 | 16 384 | 3.47e-4 | 1 728 | 5.5 s |
| uniform 256^2 | 65 536 | 6.82e-5 | 3 456 | 19.2 s |
| ratio 2, 64^2 | 4 096 | 9.46e-4 | 1 092 | 3.9 s |
| ratio 2, 128^2 | 16 384 | 2.23e-4 | 2 235 | 12.5 s |
| ratio 2, 224^2 | 50 176 | 5.99e-5 | 4 080 | 36.6 s |
| ratio 4, 128^2 | 16 384 | 2.29e-4 | 3 496 | 20.2 s |
| ratio 8, 128^2 | 16 384 | 2.78e-4 | 5 689 | 33.8 s |

**To reach the accuracy of the uniform 256^2 grid (6.8e-5 U), the best clustered grid (ratio 2)
needs 224^2 = 50 thousand cells instead of 66 thousand (23 % fewer) and 36.6 GPU-s instead of 19.2
(1.9 times longer).** On this problem stretching does not pay. The reasons, all measured:
- The gain in accuracy is small: at the same cell count the error falls to 0.6 of the uniform
  one with ratio 2 or 4, and less with ratio 8. At Re 100 the wall layers are a tenth of the box
  or more; there is no thin layer to concentrate cells in, and clustering coarsens the middle.
- Each step costs about twice as much (criterion 3), and grids this small are latency-bound, so
  fewer cells do not make a step cheaper.
- More steps: the smallest cells sit in the lid corners, where the velocity is the lid's, so the
  convective time step shrinks with the smallest cell (1.3 to 3.3 times more steps for ratio 2 to 8).

*Added after seeing that result, and chosen for it: where stretching does pay* (`bench_stretched
layer`). Stokes' first problem (case 3: a wall set in motion under fluid at rest, exact solution),
run until the layer's 1 % depth is a tenth of the domain height, 2D with 8 cells along the wall,
same time step on every grid (halving it changes nothing to 3 digits).

| Grid across the layer | Error (U) | Cells in the layer |
|---|---|---|
| uniform 128 / 256 / 512 | 2.6e-3 / 6.6e-4 / 1.6e-4 | 12 / 25 / 51 |
| clustered to the wall, ratio 16: 32 / 64 / 128 | 1.5e-3 / 3.8e-4 / 9.6e-5 | 11 / 22 / 45 |
| clustered to the wall, ratio 64: 32 / 64 / 128 | 7.7e-4 / 1.9e-4 / 4.9e-5 | 15 / 31 / 61 |

For the error of the uniform 256-cell grid the ratio-64 grid needs about 35 cells, 7 times fewer
(ratio 16: about 48, 5 times fewer; both read off at order 2 between the measured grids). This is a property of the problem (a layer a tenth of the domain
thick), not a general factor, and it says nothing about time: the grids are too small for that.
It is the regime of a body's boundary layer in a large domain, which stage 2 has to measure on
case 4.

### Stretched grids, stage 2: immersed bodies (2026-11-13)

**Why.** Stage 1 left bodies refused on stretched grids. Case 4 showed what that costs: the
cylinder needs small cells (its boundary layer is about a tenth of its diameter), and a uniform
grid then has small cells in the whole channel. Stage 2 lets the cells be small around the body
only, and measures on case 4 what that is worth.

**Design, written before any solver code was changed.**

*What stays as it is.* The method of "Immersed bodies" above: faces of a velocity component inside
a body get their value from a rule (ghost-cell method), the pressure sees the bodies as walls (zero
gradient through a face in a body), forces come from the solver's own momentum balance over a box.
The stretched discretisation of stage 1 reads the faces in bodies through its usual stencils, so
the momentum equation needs no change. Four things need the cell sizes:

1. **The ghost rules** (`bodies.build`). Formulation A, the one built first: the same construction
   as on a uniform grid. A ghost face G, its closest surface point B, the image point
   I = B + s_I n in the fluid, and u_G = U_B + (u_I - U_B) phi_G / s_I (a straight line along the
   normal through the wall value and the fluid value). On unequal cells:
   - u_I is interpolated (bi- or trilinear) between the component's true node positions, so the
     weights follow the local cell sizes;
   - "near the surface" and the image distance use the *local* cell size h_loc at G (the largest
     distance from G to a neighbouring node of its component, over the directions) in place of the
     one global h: ghost band 1.5 h_loc, s_I = (sqrt(dim) + 0.05) h_loc, moved outward in steps of
     0.25 h_loc until the interpolation stencil holds fluid faces only (the existing check).
   With uniform positions this is the existing rule, number for number. A line over a distance of
   a few local cells still has an error of order h_loc^2.
   Formulation B, named now as the second attempt if A does not converge or is unstable: no image
   point; a least-squares linear fit through the wall value at B and the nearest fluid faces, which
   has no distance to choose on cells of very different size.
2. **The pressure equation.** The face coefficient b (1 between fluid cells, 0 for a face in a
   body) multiplies the face's gradient in the stretched Laplacian:
   (1 / dc_i) [ b_(i+1) (p_(i+1) - p_i) / dm_(i+1) - b_i (p_i - p_(i-1)) / dm_i ] per direction.
   Multiplied by the cell volume it stays symmetric, so the volume-weighted conjugate gradients of
   stage 1 apply unchanged. The line smoother puts the same coefficients into each line's
   tridiagonal system: a line cut by a body falls apart into independent pieces, and a cell with
   no fluid face is an identity row. As on uniform grids, only the finest multigrid level sees the
   bodies (averaged coefficients on coarse levels broke the preconditioner there). A closed fluid
   region (singular problem) uses volume-weighted sums over the fluid cells and face areas for the
   rim correction.
3. **Forces.** The box balance with the true sizes: every velocity unknown's box has width dm
   along its own direction and dc across, and the fluxes are the ones of stage 1's stencil. The
   sum must still telescope exactly, which a random field tests (zero force with no body).
4. **Pressure at a point** (`pressure_at`, `surface_pressure`): interpolation between the true
   cell centres; the surface samples at 1.5, 2.5, 3.5 local cells.

*The grid for a body in a large domain* (`grids.clustered`, new): cells of one size h in a band
around the body, growing linearly with the distance from the band up to a far-field factor c that
is solved so that n cells fill the length. Doubling n and halving h refines every cell by two.
The grid is still a tensor product: the fine band in x runs across the whole channel height and
the fine band in y along its whole length.

*Uniform grids.* All new code sits behind the stretched switch, in C and in Python, so uniform
results must stay bit-identical.

**Pass criteria, fixed in advance.**
1. Uniform grids bit-identical to the build before this change (`tests/same_as_build.py`), and
   the whole ladder passes.
2. New rung 13 (`tests/test_stretched_bodies.py`, results in
   `tests/results/test_stretched_bodies.json`):
   - no body, random field on a stretched grid: `forces` returns zero to round-off (1e-12);
   - uniform face positions through the stretched code path: the ghost rules equal the uniform
     ones (1e-12); after the same number of steps the velocity, the force and the surface pressure
     equal the uniform code's to 1e-9 (closed Couette cell and the open DFG channel);
   - circular Couette flow (exact) on grids whose cell size varies smoothly by a factor 3 across
     both cylinders: velocity order 2 by rung 10's rule (mean order at least 1.8, no pair below
     1.2); divergence at the solver tolerance (closed region); the 3D extrusion equals 2D;
   - pressure iterations per projection with bodies on a clustered grid: reported; wanted within
     twice the uniform-grid count with bodies, and not growing with the grid;
   - DFG 2D-1 (Re 20, steady) on grids clustered around the cylinder, D/h 17.5 and 35 at the body
     in the ladder (D/h 70 as a one-off run): drag converges to the benchmark at order 1.5 or
     better; at equal D/h the drag and the pressure difference agree with the uniform grid's
     within the uniform grid's own error against the benchmark; forces from two boxes agree to
     round-off; the error against the cell count is tabulated next to rung 10's uniform grids;
   - two identical runs bit-identical, CUDA graphs on = off.
3. Case 4 (DFG 2D-2, Re 100) on clustered grids C1, C2, C3 (below), by the rules of the gpucfd
   block of `cases/dfg_cylinder_re100/numerics.yaml`: same scheme, same time-step rule
   (`max_dt`: 0.8 h / 2.25 and 0.15 h^2 / nu with the smallest cell), same pressure tolerance,
   same force box and sampling, C1 from rest for 15 s, each finer level from the coarser one's
   last field, run for at least 3 s and until the case's periodicity rule holds (checked live every
   second; at most 15 s). Reported: drag, lift, Strouhal number against FeatFlow, cells, time
   step, wall time, next to uniform G3 and OpenFOAM.
4. A plain statement against the target.

**The grids for case 4 and the cost estimate** (cell counts from `grids.clustered`; no flow run
yet). Band of fine cells: x from 0.10 to 0.35 (one diameter ahead of the cylinder's centre, one
and a half behind), y from 0.10 to 0.30; growth over 0.10 in x and 0.05 in y. The far-field
factor comes out at 4.43 in x and 2.10 in y at every level.

| Level | Cells | D/h at the body | D/h far wake | Uniform grid with this D/h | Time step: viscous / convective limit |
|---|---|---|---|---|---|
| C1 | 256 x 112 = 28 672 | 35 | 7.9 | 110 880 | 1.22e-3 / 1.02e-3 |
| C2 | 512 x 224 = 114 688 | 70 | 15.8 | 441 980 (G3) | 3.06e-4 / 5.08e-4 |
| C3 | 1024 x 448 = 458 752 | 140 | 31.6 | 1 767 920 | 7.65e-5 / 2.54e-4 |
| C4 | 2048 x 896 = 1 835 008 | 280 | 63 | 7 071 680 | 1.91e-5 / 1.27e-4 |

So the clustered grid has 3.85 times fewer cells than the uniform grid with the same cells at the
body, not 30 times: the channel is only 4.1 diameters high and the fine bands cross it.

Cost per step, assumed from what is measured: case 4's uniform runs took 6.4, 7.8 and 10.3 ms per
step at 28, 111 and 442 thousand cells (RK3, bodies), and stage 1 measured the stretched code
path at twice the uniform one. That gives 12.8, 15.7, 20.6 and about 84 ms per step for C1 to C4.
Note what the uniform numbers say: 16 times the cells cost only 1.6 times per step, because these
grids are still latency-bound. Fewer cells therefore buy little time per step here.

| Run | Steps | Projected GPU time |
|---|---|---|
| C1, 15 s from rest | 14 800 | 189 s |
| C2, about 5 s to periodic | 16 300 | 257 s |
| C3, about 5 s | 65 400 | 1 346 s |
| C4, about 5 s | 261 000 | about 6 hours |

Projected protocol times (wall = GPU plus about 25 s per level for sampling, as in the uniform
runs), and the drag error expected from the measured order 1.5 if only the cells at the body
matter (which the runs have to show):

| Up to | Expected drag error | Projected wall time | Against |
|---|---|---|---|
| C2 (D/h 70) | -1.0 % (as uniform G3) | about 490 s | uniform G1-G3: 315 s |
| C3 (D/h 140) | -0.35 % | about 1 870 s | target: 0.25 % in under 1 740 s |
| D/h 176 (0.25 % by order 1.5) | -0.25 % | about 3 600 s | the same target |
| C4 (D/h 280) | -0.12 % | about 6 hours | the same target |

**What the estimate says, before building:**
- *The cheap question* (same drag error as uniform G3, less wall time?) is expected to come out
  **no**: about 490 s against 315 s. The clustered grid has a quarter of the cells, but a step
  costs twice as much per cell and small grids are latency-bound.
- *The target (0.25 % in under 1 740 s) is not expected from stage 2 alone.* With the explicit
  scheme the time step follows the smallest cell squared: at D/h 140 it is 3.3 times below the
  convective limit, at D/h 280 6.6 times. The level that should give 0.25 % (D/h about 176) is
  projected at twice the target time. It needs stage 3 (IMEX with bodies: 4 times fewer steps
  there), and even then the projection lands near the target, not clearly under it (1 550 to
  1 950 s if an IMEX step costs 1.5 to 2 times an explicit one), so the speed of the stretched
  code path matters as well.
- **Stage 2's own target is therefore:** the correctness criteria 1 and 2; the cheap question
  answered by measurement, whichever way it falls; and one run of C3 (D/h 140 at the body, the
  cell count of uniform G3): drag within 0.5 % of FeatFlow, half the error of uniform G3, with its
  wall time stated against the 1 740 s. C4 is not run (6 hours).
- Stops, fixed now: every level as in criterion 3. C3 is timed over 200 steps first; it is not
  started if its projection exceeds 45 minutes, and it is stopped if the drag has not settled
  into a cycle after 8 s of flow time. If C2's drag error is more than twice uniform G3's, the
  far wake or the band is too coarse: one variant with a wider band is run on C2 before C3.

**Results (2026-11-13).** Criteria 1 and 2 pass, with one check of criterion 2 failed as written
and replaced (below). On case 4 the cheap question comes out yes for drag, and the level C3 meets
the step's target: drag within 0.25 % of FeatFlow in less wall time than OpenFOAM's study. The
limits of that statement are listed with it.

*What changed from the plan while building, each from a measurement:*
1. **Formulation A worked at the first attempt**; formulation B was not needed and not built.
2. **A tolerance floor was in the wrong norm.** The first iteration count looked too good: 2
   pressure iterations per projection on the clustered grid against 8 on the uniform one. With
   bodies the pressure tolerance has a floor (item 4 of "Immersed bodies"), written for plain
   sums; the stretched solver's norms are volume-weighted, so the floor was about 100 times too
   high and the solve stopped early. Now the floor is the same quantity in the weighted norm.
   After the fix: 5 iterations against 7.7, with a smaller remaining divergence.
3. **The pressure-difference check failed as written, and the cause was the check.** At D/h 35
   the clustered grid gave the DFG 2D-1 pressure difference to -1.94 % where rung 10's uniform
   grid has -0.23 %; the criterion asked for agreement within the uniform grid's own error. A
   separate study (`tests/shift_study.py`, 8 runs) moved the cylinder along x by 0, 1/4, 1/2 and
   3/4 of a cell on both grids:

   | Grid, D/h 35 | Drag error over the four positions | Pressure difference error | Lift (reference 0.0106) |
   |---|---|---|---|
   | uniform, 110 592 cells | -0.65, -0.62, -0.69, -0.85 % | -0.23, -1.90, -2.39, -0.75 % | 0.012, 0.029, 0.016, 0.021 |
   | clustered, 28 672 cells | -0.56, -0.70, -0.85, -0.83 % | -1.94, -2.44, -0.71, -0.54 % | 0.024, 0.011, 0.010, 0.014 |

   So rung 10's -0.23 % was one lucky position of the surface between the grid lines. At 35
   cells per diameter the surface pressure difference is good to about 2.4 %, the drag to about
   0.25 % around its mean error, and the lift (0.2 % of the drag) to a factor 2 to 3, on either
   grid. The check now compares the clustered value with the uniform grid's band over the four
   positions; the test output still states the first failure. This also corrects what "Steady
   flows" above calls open (the pressure difference "getting worse" at D/h 70): it is scatter.
4. **The pressure tolerance has to be checked on the level it is used on.** Case 4's first C3 run
   used the 1e-8 tuned on the coarsest uniform grid and never passed the periodicity rule in 15 s
   of flow, although the flow had settled after 2 s: the drag peaks jittered by 0.07 % from cycle
   to cycle (uniform G3: 0.001 %), with 1.5 pressure iterations per projection. The case's own
   tolerance rule (the loosest tolerance whose drag and lift stay within 1e-4 of the 1e-10 run),
   applied on each level over one shedding cycle:

   | Level | 1e-8 against 1e-10: drag / lift | 1e-9 | Rule gives | ms per step at 1e-8 / 1e-9 / 1e-10 |
   |---|---|---|---|---|
   | C1 | 1.4e-5 / 1.4e-5 | | 1e-8 | 6.4 / 8.1 / 10.2 |
   | C2 | 5.2e-5 / 6.8e-4 | 1.3e-6 / 3.1e-6 | 1e-9 | 9.4 / 12.5 / 15.4 |
   | C3 | 6.0e-3 / 6.2e-3 | 5.3e-5 / 2.5e-4 | 1e-10 | 13.5 / 19.7 / 28.7 |
   | uniform G3 | 7.3e-5 / 3.9e-4 | 1.3e-5 / 1.0e-5 | 1e-9 | 10.6 / 15.1 / 18.6 |

   Why: the explicit scheme updates the pressure by (projection result) / (stage time step), and
   the tolerance floor is relative to max|u| / h, so with a time step that shrinks like h^2 a
   fixed relative tolerance leaves more and more pressure noise. C2 and C3 were rerun with the
   tolerances the rule gives; the first runs are kept in the case's `gpu_tuning/`. The uniform G3
   fails the same check with the 1e-8 it was run with; it was rerun with 1e-9 afterwards (check
   run 2 below), and the tables here show that rerun.
5. **A stop rule was broken once, and why.** The first C3 run was to be stopped at 8 s of flow if
   not periodic. A run only writes its record when it returns, so stopping it would have thrown
   away 24 minutes of recorded forces; it was left to its own limit of 15 s (2 680 s of wall
   time). That record is what showed the jitter. The limit is now part of the run
   (`max_length` 8 s), so a run stops by itself and keeps its record.
6. **The cost estimate was too pessimistic,** by 1.3 to 2 times per step. It assumed a stretched
   step at twice a uniform one (stage 1, no bodies). With a body the line smoother needs fewer
   iterations than the point smoother (5.0 against 7.7 on the same grid: one line solve couples a
   cell to everything along its row, also around the body), which pays for most of its price:
   8.9 against 8.5 ms per step on the same 27 648 cells. At equal cell count and tolerance the
   clustered levels cost 1.2 to 1.5 times the uniform ones per step (table above, C3 against G3).
   The finer levels also became periodic after 3 s where the uniform ones needed 5.

*Criterion 1: uniform grids unchanged.* `same_as_build`: 26 of 26 arrays bit-identical to the
library and the Python files before this change. Two body cases were added to that check first
(a closed Couette cell and an open channel with a cylinder: velocity, pressure, force and
surface pressure). The whole ladder passes: nine test files, rung 13
included, 0 failed (its rerun changed only timings in the older result files, which were restored).

*Criterion 2: rung 13* (`test_stretched_bodies`):

| Check | Result |
|---|---|
| No body, random field on a stretched grid: force from the box balance | 7e-16 (2D), 1e-16 (3D, clustered in x and y) |
| Uniform face positions through the stretched code path | the ghost rules are identical; after 100 steps the closed Couette cell differs by 5e-13 (velocity) and 4e-12 (pressure), the open channel with a cylinder by 5e-11 (velocity), 2e-13 (force), 7e-12 (surface pressure) |
| Circular Couette flow, cell size varying smoothly by 3 across both cylinders (N = 32, 64, 128) | velocity rms error 1.24e-2, 1.93e-3, 3.81e-4: order 2.69, 2.34 (mean 2.51); max error order 2.79, 1.72. Uniform grids at the same N (rung 10): 6.0e-3, 1.1e-3, 2.5e-4; the clustered grids are 1.5 to 2 times less accurate here because their largest cells, 1.5 times the mean, sit on the outer cylinder |
| Couette pressure, rms beyond 1.5 local cells of the walls | 2.93e-3, 6.15e-4, 2.22e-4: order 2.25, 1.47 (mean 1.86) |
| Closed region on a stretched grid | max divergence h_min / U 3e-13 (the rim correction with face areas) |
| 3D: the Couette flow extruded along z, clustered in x and y | equals 2D to 6e-13 |
| Pressure iterations per projection, DFG channel, tolerance 1e-10 | clustered 5.0, 5.0, 4.6 at D/h 17.5, 35, 70; uniform 7.7, 7.5: fewer, and not growing with the grid |
| DFG 2D-1 (Re 20) drag, clustered grids | -2.23 % at D/h 17.5 (7 168 cells), -0.56 % at D/h 35 (28 672 cells): order 2.0. One-off third grid: -0.17 % at D/h 70 (114 688 cells), order 1.7; the three grids together took 250 GPU-s |
| The same against the uniform grids of rung 10 at equal D/h | -2.26 % (27 648 cells), -0.65 % (110 592 cells): the same drag with 3.85 times fewer cells; the differences (0.03 and 0.09 points) are inside the scatter of item 3. D/h 70 (one-off): -0.14 % with 442 368 cells, 374 GPU-s for its three grids |
| DFG 2D-1 pressure difference | D/h 35: -1.94 %, inside the uniform grid's band over cut positions (-0.23 to -2.39 %); as first written this check failed (item 3). D/h 70 (one-off): +0.08 % (uniform: -0.47 %) |
| Forces from two boxes, one reaching into the stretched cells | agree to 5e-18 |
| Reproducibility | two runs bit-identical, CUDA graphs on = off |

Not tested: the tilted channel of rung 10 (its box is periodic, and periodic directions must stay
uniform); 3D bodies on grids clustered in three directions; `steady.prolong` between clustered
grids is used by the two studies but has no check of its own (it only sets a start field).

*Criterion 3: case 4 (DFG 2D-2, Re 100) on clustered grids* (`gpu_run.py clustered`,
`gpu_clustered.py`, numbers in `results/dfg_cylinder_re100/gpu_clustered.json`). Errors against
FeatFlow level 6; wall time includes sampling the forces.

| Level | Cells | D/h at the body | Time step | Pressure tolerance | Flow time | ms per step | Wall time | Protocol so far | Drag max / mean | Lift max / amplitude | St |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 28 672 | 35 | 1.02e-3 | 1e-8 | 15 s from rest | 6.4 | 112 s | 112 s | -3.79 / -3.67 % | -6.3 / -13.0 % | +1.90 % |
| C2 | 114 688 | 70 | 3.06e-4 | 1e-9 | 3.0 s (periodic) | 12.6 | 129 s | 241 s | -1.12 / -1.07 % | -5.15 / -4.11 % | +0.54 % |
| C3 | 458 752 | 140 | 7.65e-5 | 1e-10 | 3.0 s (periodic) | 29.5 | 1 180 s | **1 421 s** | **-0.11 / -0.09 %** | -0.85 / -1.08 % | +0.13 % |
| uniform G1 | 27 648 | 17.6 | 2.02e-3 | 1e-8 | 15 s from rest | 6.4 | 64 s | 64 s | -16.8 / -16.5 % | -27 / -38 % | +5.56 % |
| uniform G2 | 110 592 | 35 | 1.01e-3 | 1e-8 | 5 s | 7.8 | 50 s | 114 s | -5.19 / -5.06 % | -8.5 / -14.3 % | +1.42 % |
| uniform G3 | 442 368 | 70 | 3.04e-4 | 1e-9 (rerun; first run 1e-8: 10.3 ms, 201 s, 315 s) | 4 s | 14.7 | 214 s | 328 s | -1.04 / -0.99 % | -3.35 / -3.16 % | +0.29 % |
| OpenFOAM M2 | 13 432 | body-fitted | | | 3 s | | 97 s | 153 s | mean +0.24 % | | |
| OpenFOAM M4 | 214 912 | body-fitted | | | 3 s | | 2 110 s | 2 548 s (about 1 740 s with the live stop) | +0.25 / +0.25 % | +0.41 / +0.33 % | -0.12 % |

- **The cheap question: does the clustered grid reach uniform G3's drag error in less wall time?
  Yes, with two qualifications.** C2 has G3's cells at the body and a quarter of its cells in
  all: mean drag -1.07 % against -0.99 %, in 241 s against 328 s (0.74 times). The
  qualifications: C2's drag error is 0.08 points larger (the scatter with the position of the
  surface in the grid was 0.2 points at D/h 35 and is not measured at D/h 70), so it is "about
  the same", not "no larger"; and its lift and Strouhal number are
  worse (-5.2 % and +0.54 % against -3.4 % and +0.29 %), which points at the wake, where C2's
  cells are 4.4 times larger than at the body. Check run 3 below confirms that, and shows that
  matching the lift as well costs the time advantage (308 s against 328 s).
- **C3: drag within 0.11 % of FeatFlow, 1 421 s for the whole protocol.** Stage 2's own target
  (0.5 %) and the step's target (0.25 % in less than OpenFOAM's 1 740 s) are both met, the time
  with a margin of 18 %. The estimate above expected -0.35 % in about 1 870 s.
- The uniform grid with C3's body resolution would have 1.77 million cells and the same time
  step; it was not run (the estimate in "case 4" above was 40 GPU-minutes for one level).

**Against the target, and what limits the statement.**
*Corrected 2026-11-13 (found while writing the study report `results/dfg_cylinder_re100/gpu_study.html`):*
the benchmark's finest tabulated level is itself about 0.2 % below the value its levels converge to
(mean drag 3.1952 extrapolated from FeatFlow's three finest levels, observed order 1.08; level 6 has
3.1884). Against that converged value C3 and its shifted twin are **0.31 % and 0.38 % low**, and
OpenFOAM's M4 is at +0.04 %. Every "0.1 to 0.2 %" below is against level 6, as the target was written;
gpucfd did not reach OpenFOAM's accuracy on this case.
*Met as it was defined:* the finest clustered level's drag is 0.11 % (maximum) and 0.09 % (mean)
from FeatFlow, inside 0.25 %, and the three levels took 1 421 s of wall time on this PC against
about 1 740 s for OpenFOAM's four-level study. What a reader has to know with it:
1. **The three levels do not prove 0.25 % by themselves.** The drag converges at observed order
   1.4 (errors -3.67, -1.07, -0.09 %), and the grid convergence index of C3, the case's own
   uncertainty measure, is 0.75 %. The last step is better than the trend (order 1.5 from C2
   gives about -0.4 %), so part of the 0.1 % may be a favourable position of the surface in the
   grid. A
   user without the reference could claim about 0.75 %, not 0.25 %. FeatFlow's own uncertainty
   for the drag (level 6 against level 5) is 0.24 %. *Answered by check run 1 below:* with the
   cylinder shifted by half a cell the drag is 0.16 to 0.19 % off, so the first position was
   somewhat favourable and the number to quote is 0.1 to 0.2 %; the 0.75 % stays as the study's
   own estimate.
2. **It is drag.** Lift is 0.85 to 1.1 % low and the Strouhal number 0.13 % high; OpenFOAM's M4
   has +0.41 %, +0.33 % and -0.12 %. By the case's rule applied to C1 to C3 (band = grid
   convergence index + FeatFlow's uncertainty) every primary quantity is inside its band except
   the lift minimum (1.16 % off, band 0.64 %: WARN). The bands are wide, 1.0 % for the drag,
   because the levels are still far apart.
3. **The time is one measured run per level,** after a first attempt that failed on the pressure
   tolerance (item 4). The tolerance studies (about 7 minutes of GPU time) and the failed C3 run
   (45 minutes) are not in the 1 421 s, as OpenFOAM's tuning is not in its 1 740 s.
4. **OpenFOAM is still far cheaper for this accuracy on this problem.** Its second grid has the
   drag at +0.24 % after 153 s (M1 and M2), nine times less than 1 421 s. The target compares
   with its whole four-level study, which is what a grid-convergence study costs each tool. A
   body-fitted grid needs 13 thousand cells where the clustered Cartesian grid needs 459 thousand.
5. **The explicit scheme pays twice at the finest level.** C3's time step is 3.3 times below the
   convective limit, and the tight pressure tolerance it forces doubles the cost per step (29.5
   against 13.2 ms). C3 is 83 % of the protocol's time. Both go away with an implicit viscous
   term (stage 3); that is where the next factor is, not in the grid.
6. One flow, 2D, one PC, Re 100. In 3D the same clustering needs plane relaxation (stage 1's
   open item) before it can be tried on case 5.

**Check runs (2026-11-13, after the first results; three questions the results left open).**
Stops and readings were written into `numerics.yaml` (gpucfd.clustered) before each run.

*1. Is C3's 0.1 % typical, or a lucky position of the cylinder in the grid?* The whole protocol
again with the cylinder moved along x by half a C3 cell (0.36 mm; a quarter of a C2 cell, an
eighth of a C1 cell). Same flow, same grids, the surface cut differently.

| Level | Drag max / mean, first position | Shifted | Lift maximum, first / shifted | St, first / shifted | Wall time, first / shifted |
|---|---|---|---|---|---|
| C1 (D/h 35) | -3.79 / -3.67 % | -4.28 / -4.15 % | -6.3 / -10.0 % | +1.90 / +2.12 % | 112 / 109 s |
| C2 (D/h 70) | -1.12 / -1.07 % | -1.21 / -1.16 % | -5.15 / -5.33 % | +0.54 / +0.52 % | 129 / 125 s |
| C3 (D/h 140) | -0.11 / -0.09 % | -0.19 / -0.16 % | -0.85 / -1.15 % | +0.13 / +0.10 % | 1 180 / 1 141 s |
| protocol | | | | | 1 421 / 1 375 s |

- Both positions are inside 0.25 % and under 1 740 s; both C3 runs were periodic after 3 s.
  **The target holds on both.** The first position was the more favourable one, by 0.07 points:
  a little more than the 0.05 expected beforehand.
- The grid-to-grid evidence is the same for both sets: observed order 1.4 and 1.6, grid
  convergence index of C3 0.75 % and 0.62 %.
- **What can be quoted for the drag at 140 cells per diameter: 0.1 to 0.2 % below FeatFlow's
  finest tabulated level and 0.31 to 0.38 % below the value it converges to
  (mean -0.09 and -0.16 %, maximum -0.11 and -0.19 % of level 6), in 23 to 24 minutes for the three-grid
  study.** With it: this is two positions, so a range and not a statistic; FeatFlow's own
  uncertainty is 0.24 %; the study's own error estimate, made without the reference, is 0.6 to
  0.75 %. Lift maximum: 0.85 to 1.15 % low. Strouhal number: 0.10 to 0.13 % high.
- The shifts are different fractions of a cell on each level, so the three rows do not show how
  the scatter falls with refinement.

*2. What does the uniform study cost when it follows the same tolerance rule?* The tolerance
check on every uniform level: G1 1e-8 (by construction), G2 1e-8 passes (drag 3.1e-5, lift
3.9e-5), G3 needs 1e-9. G3 rerun with 1e-9 from G2's last field: the same coefficients (drag
-1.04 / -0.99 %, lift maximum -3.35 %, St +0.29 %), 14.7 ms per step, periodic after 4 s of flow,
214 s of wall time. The uniform protocol is 328 s (280 GPU-s), not the 385 s I estimated: the
rerun settled in 4 s where the first run needed 5 (it started from a G2 field 2 s older). The
first G3 record is kept in the case's `gpu_tuning/`; `gpu_compare.json` now holds the rerun,
with the same verdict (WARN).

*3. Are C2's worse lift and Strouhal number caused by the larger cells behind the cylinder?*
C2w: the band of fine cells reaches 4 diameters behind the cylinder's centre where C2's reaches
1.5 (640 x 224 = 143 360 cells, 25 % more), started from C1's field.

| Run | Cells | Drag max / mean | Lift max / amplitude | St | Wall time, protocol |
|---|---|---|---|---|---|
| C2 | 114 688 | -1.12 / -1.07 % | -5.15 / -4.11 % | +0.54 % | 241 s |
| C2, shifted | 114 688 | -1.21 / -1.16 % | -5.33 / -4.54 % | +0.52 % | 234 s |
| C2w, longer band | 143 360 | -0.97 / -0.94 % | -3.68 / -3.13 % | +0.25 % | 308 s |
| uniform G3 | 442 368 | -1.04 / -0.99 % | -3.35 / -3.16 % | +0.29 % | 328 s |

- Yes: "recovered" by the rule set beforehand (lift maximum within 3.7 %, St within 0.35 %). The
  change is far larger than what the position of the cylinder does at this level (0.2 points in
  the lift, 0.02 in St). The near wake needs the body's cell size for more than 1.5 diameters.
- It changes the answer to the cheap question. With every quantity matched, the clustered grid
  has a third of the uniform grid's cells but takes nearly the same time: 308 s against 328 s.
  C2w needed 4 s of flow to become periodic (C2: 3 s) and costs 14.2 ms per step, as much as
  uniform G3. The 0.74 times of C2 holds for the drag alone.
- Not run: the longer band on the finest level. From C2w's cost (25 % more cells) a C3w would
  take about 1 750 to 1 780 s for the protocol, at the edge of the target's time, and should bring
  the lift error below the 0.85 to 1.15 % of C3.

### IMEX with immersed bodies, stage 3 (2026-11-13)

**Why.** With a body the solver had to use the explicit scheme, whose time step follows the smallest
cell squared (0.15 h_min^2 / nu). On case 4's finest clustered level that is 3.3 times below what the
flow speed needs, and the level is 83 % of the study's time. IMEX with bodies was refused as unstable
("Immersed bodies", item 5), and the cause was not known. Stage 3 finds the cause, removes it, and
measures on case 4 what it is worth.

**The cause, found before any solver code was changed.** Two separate mechanisms. Both were isolated
with a numpy model of the time step (2D, the linear part: no convection; `tests/imex_body_model.py`),
which reproduces the solver's growth rates to three digits in every variant tried, and each was then
confirmed on the solver itself with a diagnostic build.

1. *A pressure gradient inside the body (fast: the error triples every step).* The implicit
   predictor subtracted the pressure gradient on every face, also on the faces in a body, where the
   pressure is not defined (the cells there are switched off and hold zero, so a face between a cut
   cell and a dead cell saw the whole pressure over one cell width). The faces in bodies were unknowns
   of the diffusion solve, so this force spread to the fluid faces next to the body; the projection
   answered with more pressure, which made the force larger. The explicit scheme masks that term
   (`fl`), the implicit one did not: an oversight of the first version, not a property of the method.
   Measured, uniform 384 x 72 channel with the DFG cylinder, nu dt / h^2 = 0.31, convection
   negligible: growth 3.03 per step in the solver, 3.05 in the model. With the mask it is gone.
2. *The ghost faces answer the pressure correction, and the leftover is fed back into the pressure
   (slow: 10 to 60 % per step, sign flipping every step).* This one stays when everything else is
   done right: mask in place, the ghost rule as exact rows of the diffusion solve, diffusion fully
   implicit, ghost faces set again after the projection. It is the instability the earlier note
   describes. The mechanism:
   - The projection corrects fluid faces only (u = u* - grad phi) and leaves the ghost faces at the
     values extrapolated from u*. So after it either the velocity is divergence-free and the ghost
     faces no longer follow their rule (they belong to u*, not to u), or the ghost faces are set
     again from u and the velocity is no longer divergence-free. The difference is the ghost rule
     applied to grad phi: a leftover divergence D_g E G phi in the cut cells (G the gradient on fluid
     faces, E the ghost rule, D_g the divergence through the faces in bodies).
   - The next projection removes that leftover, and because the pressure is incremental, adds it to
     the pressure divided by the substep: p += phi / c. Written for one pressure mode with
     M = A^-1 D_g E G (A the pressure operator) and no viscosity, the pressure after each stage obeys
     p_new = -M [ (1 + r) p - r p_old ],  r = previous substep / this substep.
   - M's largest eigenvalue mu measures how strongly the ghost rule answers a pressure gradient. It
     depends only on the grid and the body: 0.26 for the cylinder at 8 cells per diameter, 0.30 on
     the 384 x 72 DFG grid, 0.34 with the cylinder moved 0.37 cell.
   - The recurrence is stable for mu below a limit set by the sequence of substeps. IMEX (substeps
     8/15, 2/15, 1/3 of the step: the second is four times shorter than the first, so a leftover
     sized for the first is multiplied by four when it becomes pressure): **mu < 0.279**. Equal
     substeps: mu < 1/3. The explicit SSP-RK3 with the same incremental pressure: **mu < 0.683**.
   - Prediction against measurement, 384 x 72, convection negligible, nu dt / h^2 = 0.01: predicted
     growth 1.175 per step, solver 1.174. With viscosity the near-wall cells are tied to the wall and
     the growth falls: 1.148, 1.106 at nu dt / h^2 = 0.15, 0.31, stable from about 1. So the scheme is
     unstable exactly where it is wanted, at time steps a few times the explicit limit. It also
     explains every earlier observation: unstable with exact wall rows and fully implicit diffusion
     (they do not touch the feedback), stable without the incremental pressure (no feedback), and RK3
     stable (0.30 and 0.34 are below 0.683). The closed Couette cell of rung 10 has mu = 0.33 and
     grows as well (1.37 per step at nu dt / h^2 = 0.01 in the solver; 1.49 predicted without
     viscosity). *(This sentence first said the Couette cell was below the limit: I had seen it stay
     quiet for 40 steps from the exact solution and assumed so. The model's number and a run with a
     perturbed start corrected it, before the results below.)*
   - **A limit of the explicit scheme that was not known:** RK3 with bodies is stable only while
     mu < 0.683. Every body run so far is far below it, but nothing in the ghost rule guarantees it.

**Design.** The projection must leave a velocity that is divergence-free *with ghost faces that
follow their rule*. Then there is no leftover, and nothing to feed back.

1. *Consistent projection.* Solve for phi with the correction of the ghost faces included:
   (A + D_g E G) phi = div(u*, ghost faces from u*), then u = u* - grad phi on fluid faces and the
   ghost faces set again from u. (In a closed region the net-flow correction of the rim faces is part
   of the operator, as it is part of the projection.) Ikeno & Kajishima (2007) call this "consistent
   with the wall condition". In the model the growth is below 1 at every time step tried (nu dt / h^2
   from 0.02 to 16), also for the cylinder position with mu = 0.34 where the present projection
   grows 1.59 per step.
   The operator is no longer symmetric, so conjugate gradients do not apply. It is A (1 + M) with
   M's eigenvalues around 0.3 at most, so the multigrid V-cycle of A stays a good preconditioner:
   BiCGStab, right-preconditioned with the V-cycle (mixed precision as now), all scalars on the GPU
   and the loop in the step's graph, like the present solver. The extra term costs a gradient, the
   ghost rule and a divergence per application (kernels that exist), two applications and two
   V-cycles per iteration.
2. *The ghost rule as rows of the diffusion solve.* Not needed for stability (the model is as stable
   with ghost faces frozen during the solve), but otherwise the wall condition lags the solve by one
   stage, which costs accuracy in time. Each ghost face is an unknown with its rule as its equation.
   Not symmetric either: the same BiCGStab, without preconditioner (the system is well conditioned:
   eigenvalues between 1 and 1 + 8 nu dt / (3.75 h^2)).
3. *Pressure gradient on fluid faces only* in the predictor (cause 1).
4. *Scope.* Only IMEX with bodies takes the new path. The explicit scheme with bodies keeps its
   projection (verified results must stay bit-identical); IMEX without bodies is untouched.
5. *Steady state.* When the flow is steady phi is zero, the ghost faces follow their rule and the
   velocity is divergence-free in both schemes, and the momentum balance of an IMEX stage reduces to
   the explicit one (the substep weights cancel). So IMEX and RK3 solve the same discrete steady
   equations and the steady state cannot depend on the time step.
6. *Second attempt, named now:* if BiCGStab with the V-cycle does not converge well, two projections
   per stage with the present symmetric solver: the first with the old ghost faces (it alone updates
   the pressure), then the ghost faces, then a second one that only cleans the velocity. Stable in
   the model at every time step tried (it needs mu < 1), at the price of a second solve.

**Pass criteria, fixed in advance.**
1. Nothing verified changes: `tests/same_as_build.py`, 38 of 38 arrays bit-identical to the build
   before this change (12 stretched-grid arrays, with and without bodies, were added to that check
   first), and the whole ladder passes.
2. New rung 14 (`tests/test_imex_bodies.py`, results in `tests/results/test_imex_bodies.json`):
   - *No growth.* Open channel with the cylinder, convection negligible, 200 steps at
     nu dt / h^2 = 0.01, 0.15, 0.31, 1 and 4: the change per step does not grow over the last 100
     steps (factor at most 1.0). On the uniform 384 x 72 grid at two positions of the cylinder
     (the second moved 0.37 cell: mu 0.34) and on the coarsest clustered grid of case 4's family.
   - *No growth with the flow.* The same channel at Re 100 from rest at the convective time step
     (CFL 0.8), 2 s of flow on the uniform and on the clustered grid: velocities stay bounded
     (below 3 times the inflow maximum).
   - *The projection keeps its promise.* After a step, the ghost faces equal their rule applied to
     the stored velocity (1e-12) and the divergence is at the solver tolerance.
   - *Accuracy, exact solution.* Circular Couette flow of rung 10, N = 32, 64, 128, IMEX at the
     convective time step: velocity order by rung 10's rule (mean at least 1.8, no pair below 1.2).
   - *Same steady state as the explicit scheme, and no dependence on the time step.* Couette at
     N = 32 and 64 (uniform, and N = 32 with cell sizes varying by 3): IMEX at two time steps a
     factor 4 apart and RK3, all marched until the change per time unit is below 1e-10. Tolerance:
     max difference over the fluid faces 1e-7 of the wall speed (the marches stop at about 1e-9; a
     slip that grows with the time step would be 1e-4 or more, as the first IMEX had: 6e-4).
     Open channel, DFG 2D-1 (Re 20) at 17.6 cells per diameter, uniform and clustered: drag and lift
     of IMEX and RK3 agree to 1e-6 of the drag.
   - *Second order in time with a body.* The DFG channel at Re 100 on the 384 x 72 grid, from a
     developed unsteady state: 0.2 s of flow with time steps dt, dt/2, dt/4, dt/8 (dt at CFL 0.8).
     Observed order of the velocity (L2 over fluid faces, from successive differences) at least
     1.8, and the IMEX solutions converge to the explicit scheme's at its own small step.
   - *3D = 2D* for the Couette flow extruded along z (1e-9), and two identical runs bit-identical,
     CUDA graphs on = off.
   - *Cost.* Iterations per pressure solve and per diffusion solve reported.
3. Case 4 on the clustered grids, by the rules of the gpucfd block of `numerics.yaml` (pressure
   tolerance per level from `cfdlib.gpu.tolerance_check`, each level stopped on the periodicity
   check, started from the same fields as the explicit runs).
   - *Same answer, rule fixed now:* against the recorded explicit run on the same grid, mean and
     maximum drag and the Strouhal number within 0.05 % of the value, lift maximum and amplitude
     within 0.3 %. (For scale: moving the cylinder half a cell moves the drag by 0.07 % and the
     lift by 0.3 %.) If a level misses, it is rerun with half the time step to tell a time error
     from a different answer, and both are reported.
   - *Wall time* of the three-level study against 1 421 s (explicit) and about 1 740 s (OpenFOAM).
     Each level may use the scheme that is faster on it: both solve the same steady equations.
4. A plain statement of the outcome.

**What it should save on case 4's finest clustered level (C3, 458 752 cells), estimated before
building.** Explicit: time step 7.65e-5 (viscous limit), 29.5 ms per step at the tolerance 1e-10 the
small step forces, 39 200 steps for 3 s of flow, 1 157 s. IMEX: time step 2.54e-4 (CFL 0.8), so
11 800 steps, 3.3 times fewer. Per step: three stages, each with two diffusion solves (cheap: a
handful of iterations of a 5-point stencil) and one pressure solve by BiCGStab, which I expect at
1.5 to 2 times a conjugate-gradient solve at the same tolerance. The pressure update divides by a
substep 3.3 times longer, so the tolerance the level needs should relax from 1e-10 toward 1e-9
(explicit: 19.7 ms per step at 1e-9, 13.5 at 1e-8). That gives 25 to 45 ms per step and
**300 to 530 s for C3, against 1 157 s: 2.2 to 3.9 times less.** On C1 and C2 the viscous limit is
above or near the convective one (1.2 and 0.6 times), so IMEX cannot win there and they keep the
explicit scheme: study 241 s + C3 = **about 550 to 800 s against 1 421 s.** At 280 cells per
diameter (C4, not planned) the step ratio is 6.6 and the explicit scheme's 6 hours would become
about 1 to 1.5 hours.

**Results (2026-11-13).** Criteria 1 and 2 pass. On case 4 IMEX gives the explicit scheme's answer
on every level, and the three-level study takes 1 005 s instead of 1 421 s. That is less than the
estimate above (550 to 800 s), for two reasons that are measured below.

*What changed from the plan while building, each from a measurement:*
1. **The design worked in the first build**; the second attempt (two projections per stage) was not
   needed and was not built in the solver (it exists in the model, where it is stable).
2. **The pressure tolerance did not relax.** By the case's rule (drag and lift within 1e-4 of the
   1e-10 run over one cycle) IMEX needs 1e-9, 1e-9, 1e-10 on C1, C2, C3; the explicit scheme
   1e-8, 1e-9, 1e-10. The estimate assumed the tolerance follows the time step. It follows the
   shortest substep, because that is what the pressure update divides by, and IMEX's shortest
   substep (2/15 of the step) is only 1.8 times the explicit scheme's (1/4 of a step that is 3.3
   times shorter). On C3 1e-9 misses the rule by a factor 2 (2.2e-4).
3. **A step costs 2.1 times an explicit one, not 1.5.** C3: 62.5 ms against 29.5 ms at the same
   tolerance. Measured split of the IMEX step: 23.5 ms that do not depend on the pressure
   iterations (six diffusion solves, right-hand sides, the first residuals) and 13.6 ms per
   pressure iteration, of which a projection needs 2.8 (each is two V-cycles and two applications
   of the operator with its ghost part). The explicit step is 29.5 ms in all.
4. **Tried and removed:** starting each pressure solve from the last projection's phi rescaled to the
   new substep (by the ratio of the substeps, or its square). More iterations, not fewer: 2.98 and
   3.08 against 2.75 on C3. The plain last phi stays the initial guess.
5. `tests/same_as_build.py` got twelve stretched-grid arrays before the change (it covered uniform
   grids only) and nine IMEX-with-body arrays after it, for the next change.

*Criterion 1: nothing verified changes.* `same_as_build`: 38 of 38 arrays bit-identical to the
build before this change (Poisson, cavity and 3D flow with both schemes, bodies with the explicit
scheme, on uniform and on stretched grids). The whole ladder passes: ten test files, rung 14
included, 0 failed.

*Criterion 2: rung 14* (`test_imex_bodies`):

| Check | Result |
|---|---|
| No growth, convection negligible: open channel with the cylinder, 200 steps, nu dt / h^2 = 0.01, 0.15, 0.31, 1, 4; uniform 384 x 72 at two positions of the cylinder, clustered 128 x 56 | the change per step falls in all 15 runs; largest of the last 100 steps against the 10 before them: at most 0.88. Before: 1.17 and 1.59 per step at the two positions |
| No growth with the flow: Re 100 from rest at CFL 0.8, 2 s | max velocity 0.438 (uniform) and 0.436 (clustered 256 x 112), inflow maximum 0.3 |
| The projection's promise, after those runs | the ghost faces equal their rule applied to the stored velocity to 3e-17 (the explicit scheme after 20 steps: 1.6e-4 and 6.6e-5 of velocity, the leftover described above); max divergence h / U 2e-8 and 8e-9 at tolerance 1e-9 |
| Circular Couette flow (exact), N = 32, 64, 128, IMEX at the convective time step (nu dt / h^2 = 0.64, 1.28, 2.56) | rms error 6.00e-3, 1.06e-3, 2.49e-4: order 2.50, 2.10. The explicit scheme's errors to all three digits (rung 10) |
| Same steady state as the explicit scheme; no dependence on the time step (tolerance 1e-7 of the wall speed) | Couette, uniform N = 32 and 64: IMEX - RK3 7e-13 and 8e-13, IMEX at dt against dt / 4 3e-13 and 5e-13. Cell sizes varying by 3, N = 32: IMEX - RK3 4.6e-8 (inside the tolerance; the two IMEX runs agree to 6e-13, so the 4.6e-8 is where the explicit march stopped), steps to the steady state 220 to 480 against 935 to 4 044 |
| DFG 2D-1 (Re 20), steady drag and lift, IMEX against RK3 (tolerance 1e-6 of the drag) | uniform 384 x 72: 2.9e-8 and 4e-10; clustered 128 x 56: 8e-9 and 6e-10. GPU time to the steady state 18 s against 50 s, and 8 s against 14 s |
| Second order in time with a body: Re 100 channel, 384 x 72, from a state 1 s after the start, 0.2 s with dt, dt/2, dt/4, dt/8 (dt = 0.01, CFL 0.79, nu dt / h^2 = 0.30) | successive differences (rms over the fluid faces) 2.39e-8, 5.70e-9, 1.40e-9: order 2.07, 2.03. Against RK3 at dt / 16: 3.1e-8, 7.6e-9, 2.0e-9, 8.9e-10 (it converges to the explicit solution). The flow itself changes by 3.7e-3 over the 0.2 s: the time error is small at this state, the order is what the check shows |
| Couette extruded along z, IMEX | equals 2D to 5e-13 |
| Reproducibility | two runs bit-identical, CUDA graphs on = off |
| Iterations | pressure (BiCGStab) 1.7 to 5.7 per projection at tolerance 1e-12 in the growth runs, 1.8 to 2.7 at 1e-9 with the flow; diffusion (BiCGStab, tolerance 1e-10) 2 to 4 |

Not tested: a body on a 3D grid clustered in all three directions; the tilted periodic channel of
rung 10 with IMEX; time steps above CFL 0.8 with a body (the cavity ran IMEX at CFL 2 to 4).

*Criterion 3: case 4 (DFG 2D-2, Re 100) on the clustered grids* (`gpu_run.py clustered C1i C2i C3i`,
`gpu_imex.py`, numbers in `results/dfg_cylinder_re100/gpu_imex.json`; tolerances from
`gpu_run.py tolcheck`, `gpu_tuning/tol_C*i.json`). A trailing i is the level with IMEX. Every IMEX
level started from the field its explicit twin started from.

| Level | Scheme | Time step | Pressure tolerance | Steps | ms per step | Pressure iterations | Wall time | IMEX against explicit: drag mean and max, Strouhal / lift max and amplitude |
|---|---|---|---|---|---|---|---|---|
| C1 | RK3 | 1.02e-3 | 1e-8 | 14 766 | 6.4 | 4.0 | 112 s | |
| C1i | IMEX | 1.02e-3 | 1e-9 | 14 766 | 11.4 | 2.9 | 187 s | 0.001 % / 0.014 % |
| C2 | RK3 | 3.06e-4 | 1e-9 | 9 808 | 12.6 | 4.0 | 129 s | |
| C2i | IMEX | 5.08e-4 | 1e-9 | 5 908 | 21.1 | 2.4 | 134 s | 0.001 % / 0.003 % |
| C3 | RK3 | 7.65e-5 | 1e-10 | 39 232 | 29.5 | 3.9 | 1 180 s | |
| C3i | IMEX | 2.54e-4 | 1e-10 | 11 817 | 62.5 | 2.8 | **764 s** | 0.001 % / 0.002 % |
| C3wi (longer band, 573 440 cells, from C2w) | IMEX | 2.54e-4 | 1e-10 (C3i's) | 11 817 | 82.4 | 2.7 | 1 001 s | no explicit twin |

- **Same answer: yes, on all three levels,** by a wide margin (rule: 0.05 % for drag and Strouhal
  number, 0.3 % for lift). C3i against FeatFlow level 6: drag maximum -0.115 %, mean -0.093 %, lift
  maximum -0.85 %, amplitude -1.08 %, Strouhal number +0.125 %, the numbers of C3 to every digit
  shown. Against the value the benchmark converges to (3.1952) the mean drag is -0.31 %, as C3's.
  All three IMEX levels became periodic when their twins did (C2i and C3i after 3 s).
- **Wall time of the study: 1 005 s** (C1 and C2 explicit, C3 with IMEX), against 1 421 s
  explicit and about 1 740 s for OpenFOAM's study (partly an estimate: its finest grid, 81 % of
  it, was not run with the live stop). All three levels with IMEX: 1 085 s. The study report
  `results/dfg_cylinder_re100/gpu_study.html` has the explicit runs next to OpenFOAM; it does not
  hold the IMEX runs yet.
- **The longer band of fine cells on the finest level** (C3wi: fine cells to 4 diameters behind
  the cylinder, the run stage 2 left open because the explicit scheme would have taken about
  1 780 s for its study). Lift maximum -0.69 % (C3: -0.85 %), amplitude -0.94 % (-1.08 %),
  Strouhal number +0.07 % (+0.13 %), drag -0.12 / -0.10 % of level 6 (unchanged; -0.31 % of the
  converged value). Study C1 + C2w + C3wi: 1 308 s. So the lift is below 1 % as expected, by a
  small step: 0.16 points, where moving the cylinder half a cell moved C3's lift by 0.3 points.
  One position only; its tolerance is C3i's, not checked on this grid.
- **Where IMEX pays:** when the viscous limit is well below the convective time step. A step costs
  about twice an explicit one, so the gain is the ratio of the two limits divided by about 2:
  C1 (ratio 0.83, the convective limit already binds) 0.6 times, slower; C2 (1.66) 0.96 times,
  even; C3 (3.3) 1.55 times. At 280 cells per diameter (C4, ratio 6.6; not run) that rule gives
  about 3 times: about 2 hours where the explicit scheme needs about 6.

**The outcome, and what limits it.**
*What is now possible:* a flow around a body can be advanced with implicit diffusion, at a time
step set by the flow speed alone, with the explicit scheme's answer: the same steady states to
1e-12, the same errors against the exact Couette solution, second order in time, and on case 4
drag, lift and shedding frequency within 0.014 % on every level. The cause of the old instability is
known as a formula that predicts the growth rates the solver shows.
*What it costs:* a time step about twice as dear as an explicit one, and no relief in the pressure
tolerance. On case 4 the finest level takes 764 s instead of 1 180 s and the study 1 005 s instead
of 1 421 s (0.71 times). The estimate made before building (0.4 to 0.55 times) was too hopeful.
*Limits:*
1. It does not change the accuracy. gpucfd's mean drag on case 4 is still 0.31 % below the
   benchmark's converged value where OpenFOAM is within 0.04 %, and OpenFOAM's second grid has
   0.24 % (against level 6) after 153 s. The comparison with OpenFOAM is unchanged except for the
   study time.
2. One flow, 2D, Re 100, one PC, one run per level. The times are single measurements; the same
   run twice differed by 3 % earlier today (C3 and its shifted twin).
3. Not run: the shifted cylinder with IMEX. C3i's numbers equal C3's, so the spread over
   positions measured for C3 (drag 0.07 %, lift 0.3 %) is the best estimate for C3i too, but it
   is not a measurement of it. The longer band (C3wi) was run at one position only.
4. The explicit scheme with bodies keeps its projection, and with it the hidden stability limit
   (mu < 0.683) and a ghost-face mismatch of about 1e-4 of the velocity after each step. Giving
   it the consistent projection would remove both, change every explicit result with a body at
   about that level, and cost BiCGStab's dearer iterations: a decision for later, not taken here.
5. BiCGStab can break down (a division by a vanishing inner product). It did not in any run; a
   breakdown would show as NaN and stop the run with the usual message.
6. In principle the operator A (1 + M) could be singular for a body whose mu reaches -1. Every mu
   computed so far is positive (0.26 to 0.34).

**What I would do next,** in order of expected gain per effort:
1. *One pressure solve per step.* The scheme solves the pressure three times per step (once per
   substep) and its shortest substep sets the tolerance. A second-order scheme with one projection
   per step (Crank-Nicolson with Adams-Bashforth convection, or a two-stage IMEX Runge-Kutta scheme)
   would cut the pressure work per step by two to three, at a somewhat smaller time step, and
   relax the tolerance. It needs its own stability and accuracy rung.
2. *Cheaper diffusion solves:* a third of the IMEX step does not depend on the pressure. A Jacobi
   preconditioner, or one solve for both components' common part, are candidates.
3. *The consistent projection for the explicit scheme* (limit 4), with the student's decision.
4. *C3wi with the cylinder moved half a cell* (about 17 minutes), so that the lift of the longer
   band is quoted with its spread.

### Turbulence resolved without a model: the plane channel (2026-11-14)

gpucfd has no turbulence model. This section records what it does with a turbulent flow as it is,
measured before building anything, to decide which way to go for turbulence. The scripts and raw
numbers are in `dev/exploration/les_channel/`.

**The flow.** Fluid between two parallel plates, driven by a constant force, at friction Reynolds
number 180: half the plate distance 1, driving force 1 (so the friction velocity is 1), viscosity
1/180, box 4 pi x 2 x 4 pi / 3, periodic along the flow (x) and across it (z). The standard first
test of a code that resolves turbulence. Reference: Vreman and Kuerten (2014), the same box and
forcing with two independent codes, bulk velocity 15.696 and 15.701, centre-plane velocity 18.28
(parsed from their files into `cases/turbulent_channel_re180/reference/`). Time unit: half height
over friction velocity.

**It runs today, with no new solver code.** The scheme conserves kinetic energy (the
symmetry-preserving form of the convective terms), so the turbulent flow runs stably on cells far
larger than the laminar rule (cell Reynolds number about 2) allows, and the grid is judged in wall
units. Wall cell 1 wall unit high, IMEX scheme, pressure tolerance 1e-8:

| Grid (cells along the flow in wall units, across) | Bulk velocity | Against 15.70 | GPU time |
|---|---|---|---|
| 64 x 64 x 64 (35, 12) | 14.53, averaged over 20 time units | 7.5 % low | 19 min for 40 units |
| 192 x 64 x 64 (12, 12) | 15.57, averaged over 15 time units | 0.8 % low, still rising | 34 min for 30 units |

The second average is not settled, and the tool written afterwards says so
(`cfdlib.signals.stationary` on the recorded series: 15.568 +- 0.018, the second half 2.9 standard
errors above the first, 7 independent samples). Over its last 5 time units the bulk velocity was
15.63 (0.4 % low). The peak of the streamwise fluctuation is 2.64 at 13 wall units on both grids
(reference 2.67 at 14.4). These two runs are an exploration, not a verified result: case 17 is
the verification.

*Corrected on the day:* the first write-up compared with 15.63, a value quoted from memory of the
1987 paper; against the transcribed reference the two runs are 7.5 % and 0.8 % low, not 7.1 % and
0.4 %.

**Case 17, the verified run (2026-11-14): verdict FAIL.** A fresh agent ran three grids of one
family: cells square in the wall plane, all three directions refined together, IMEX, pressure
tolerance 1e-6, the step re-chosen every 5 steps at a Courant number of 0.75. Each was averaged
over about 37 time units after its start-up. Report: `results/turbulent_channel_re180/`.

| Grid (dx+ = dz+, wall cell) | Bulk velocity | Against 15.70 | Largest gap of the mean velocity | Largest gap of u rms | Wall time per time unit |
|---|---|---|---|---|---|
| 96 x 32 x 32 (24, 2.0) | 15.94 +- 0.03 | +1.5 % | +0.58, in the core | +0.30 at y+ 6 | 5.3 s |
| 144 x 48 x 48 (16, 1.3) | 15.44 +- 0.04 | -1.6 % | -0.64 at y+ 26 | +0.26 at y+ 6 | 24.8 s |
| 192 x 64 x 64 (12, 1.0) | 15.65 +- 0.02 | -0.34 % | -0.31 at y+ 22 | +0.14 at y+ 4 | 62 s |

Gaps are in friction velocities, against the spectral reference. What this says:
- The bulk velocity does not approach the reference from one side, so the three grids give no
  order of accuracy and no extrapolated value. Its good value on the finest grid is partly
  cancellation: the profile is 0.31 low in the buffer layer and 0.03 high at the centre.
- The profiles do improve from grid to grid (the last two rows), but on the finest grid they are
  still outside the limits the agent set before its runs (0.30 for the mean velocity, 0.08 for
  u rms). The fluctuation along the flow is too strong near the wall and the two across it too
  weak: the usual sign of cells too large for the streaks at the wall. Which direction limits the
  accuracy was not separated.
- The reference used cells two to four times smaller in each direction. The next level of this
  family (288 x 96 x 96, 2.7 million cells) needs about four hours for the same averaging on this
  PC and was not run.
- Requested single numbers on the finest grid: centre velocity +0.17 %, friction coefficient
  +0.68 %, peak of u rms -0.6 %, total shear stress within 0.004 of its exact straight line.

**Speed against OpenFOAM**, same grid, same time step, start from gpucfd's developed field.
OpenFOAM v2606, `pimpleFoam` without a model, the settings of its channel tutorial (backward
differencing, linear interpolation, two pressure corrections, GAMG), best of 8 and 4 cores:

| Cells | gpucfd per step (pressure iterations) | OpenFOAM per step | Faster by | 30 time units: gpucfd, OpenFOAM |
|---|---|---|---|---|
| 64 x 64 x 64, 0.26 million | 46 ms (20) | 104 ms | 2.3 | 15 min, 0.6 h |
| 192 x 64 x 64, 0.79 million | 68 ms (9) | 449 ms | 6.7 | 29 min, 3.2 h |
| 128 x 128 x 128, 2.1 million | 322 ms (21) | 1784 ms | 5.5 | 2.5 h, 13.6 h |
| 256 x 128 x 128, 4.2 million | 428 ms (11) | 3880 ms | 9.1 | 3.6 h, 32.6 h |

The OpenFOAM times for 30 units are projected from 40 to 60 timed steps; OpenFOAM was not run
that long, and its accuracy on these grids was not measured. Its settings are the tutorial's, not
tuned, and its two-correction PISO is first order in time (case 4's lesson).

**OpenFOAM's accuracy on 192 x 64 x 64, measured later (2026-11-14,
`dev/compare/turbulent_channel_re180/`, plan written before the run).** One full run with the
same cells and time step, started from gpucfd's field: 12 time units of start-up, 30 averaged,
5.2 hours on 8 cores (464 ms per step; gpucfd 65 ms, 7.2 times faster). OpenFOAM meets 1 of the
case's 10 limits, gpucfd 5 on this grid and 10 on 288 x 64 x 96. OpenFOAM's bulk velocity is
+4.5 % from the reference (gpucfd -0.34 %), its friction coefficient -8.4 %; v rms and w rms are
too low by 0.11 and 0.10 (gpucfd 0.07 and 0.08). Limits: its bulk velocity was still rising
(+1.0 % from the first ten averaged time units to the last ten, wall shear 1.0 % below the
driving force), so the settled value is somewhat higher; its numerics were the tutorial's and
only the pressure solver was tuned, for speed. Why it is less accurate on the same cells was not
tested (a guess: the backward time scheme and the co-located layout damp the small eddies more).
On 288 x 64 x 96 OpenFOAM was timed only: 1424 ms per step against gpucfd's 171 ms (8.3 times),
about 14 hours projected for gpucfd's 1.7-hour run.

**Where the time goes, and what this flow taught.**
- A step costs about four times more per cell than on the periodic box (6 to 15 million
  cell-steps per second against about 36): walls, the stretched code path, the implicit diffusion.
  So the 47 to 235 times of the box does not carry over.
- Cells three times longer than wide double the pressure work: 20 iterations with dx = 3 dz, 9 with
  dx = dz.
- The time-step limit moves with the flow: over 6 time units the largest local Courant rate was
  between 340 and 440 at the logged times, and a fixed step set at 0.64 of the limit of the start field reached a Courant
  number of 1.23. `run_transient(adapt_cfl=...)` re-chooses the step as the run goes. It overshoots
  its target by however much the rate grows between two looks, and the rate is an extreme of the
  field: in case 17 it rose by up to 35 % within 10 steps over 40 time units (20 % in a 4-unit
  pre-flight), so a target of 0.8 with a look every 10 steps reached 1.08. A look every 5 steps
  with a target of 0.75 stayed at 0.89 or below in runs of up to 62 time units. `cfl_steps` sets the interval
  of the looks independently of the sampling.
- The largest velocity over the smallest cell overstates the Courant number about 7 times on
  the 64 x 64 x 64 grid (the wall cells carry the smallest velocities). `NavierStokes.courant_rate()` uses each
  cell's own size; `run_transient` takes it from `observe`.
- The bulk velocity settles slowly, with a time of about 9 time units. Its rate of change is the
  driving force minus the mean wall shear, and the wall shear follows the bulk velocity roughly as
  its power 1.75, so a disturbance decays with the time U_b / (1.75 u_tau^2 / delta). *Corrected
  after case 17:* this section first said "remembers about one time unit". That number came from
  batch means on records of 15 to 20 time units, whose blocks are shorter than the true memory and
  so cannot see it. Two consequences, both measured in case 17:
  - a run of 30 to 40 time units holds only about two independent values of the bulk velocity.
    Plain batch means gave an error three times too small (two runs of one grid: 15.978 and
    15.904, each "+- 0.010"). The error from the exact balance of the box
    (`gpucfd.turbulence.bulk_average`) gives 0.15 to 0.25 %, and four runs of one grid scatter by
    0.20 %. So about 40 time units of averaging fix the bulk velocity to 0.2 %; 0.1 % would need
    about 300.
  - a run started from a coarser grid's field needs about two settling times (20 time units)
    before averaging can start, because the levels differ by 1.3 to 3 % in bulk velocity. A third
    of every such run in case 17 was start-up.
- Without a body the pressure tolerance is a cost setting: over 0.3 time units the bulk velocity
  and the wall shear with 1e-6 differ from those with 1e-10 by 1e-10 or less, the divergence left
  is four orders below the health limit, and a step is 9 % (finest grid) to 20 % (coarsest)
  cheaper than with 1e-8. Over a long run two turbulent flows part ways whatever the tolerance.
- A time-step study by averages costs as much as the averages: half the step on the coarsest grid
  changed the bulk velocity by -0.11 % +- 0.26 %. `cfdlib.gpu.step_check` compares a short stretch
  from the same field with the step and half of it, before the two runs part.

**Tools that came out of it** (tests in `tests/test_gpu_tools.py`, part 8, and `tests/test_tools.py`,
part 12):
- `gpucfd.turbulence.channel_start`: a start field that turns turbulent within a few time units.
- `gpucfd.turbulence.PlaneStats`: mean profile, fluctuation intensities, turbulent shear stress
  and wall shear, formed like the solver's own fluxes. The balance of mean momentum then holds to
  round-off for any single field (largest difference from the solver's own right-hand side 4e-12
  against terms of order 10, both schemes), so what is left of it after averaging measures the
  averaging, not the post-processing.
- `gpucfd.grids.tanh_wall`: a wall grid from the height of its wall cell.
- `cfdlib.signals.stationary`, `settle_time`, `stationary_watch`: the time average with its
  statistical error from batch means, the end of the start-up, a live stop. On signals whose true
  error is known the estimate is 12 to 17 % high on average (it keeps the largest of several
  block sizes), never below 0.85 of the truth in 36 trials. That holds for signals whose memory is
  short against the record / 8. For a slowly settling signal it gives 0.4 of the true error; with
  the slow time passed as `memory` 0.9, and `signals.relaxing` (the error from the signal's own
  recorded rate of change) 1.2 and reliable (`tests/test_tools.py`, part 13).
- Added after case 17: `gpucfd.turbulence.bulk_average` and `channel_watch` (the error of the bulk
  velocity and a stop rule built on it; replayed on case 17's finest run it stops at the same
  time, 58.0), and in `cfdlib.gpu`: `preflight`, `step_check`, `save_state` / `load_state` with
  checkpoints in `run_transient`, `sample_steps` and `cfl_steps`, thinned records, `peek`.

**The optimize stage of case 17 (2026-11-14): which grid resolves the profiles.** Run by the dev
agent after the solve, with target, budget and stop rule fixed first (`ROADMAP.md` 3b; plan,
script and records in `dev/optimize/turbulent_channel_re180/`; the page with the profiles side by
side is `results/turbulent_channel_re180/optimize_study.html`). The case's verdict stays FAIL:
that is the solve. Same numerics in every run, only the grid differs. First one direction refined
alone by 1.5, each run 10 time units from the finest grid's field; then one run on the grid the
rule picked. "Gap" is the largest difference from the reference along a profile, in friction
velocities, with the limits the solver agent set before its runs.

| Grid | Cells in wall units, x / wall / z | Averaged | Bulk velocity | Mean velocity (0.30) | u (0.08) | v (0.05) | w (0.06) | Shear stress (0.03) | Seconds per time unit |
|---|---|---|---|---|---|---|---|---|---|
| 192 x 64 x 64, case 17's finest | 11.8 / 1.0 / 11.8 | 38 | -0.34 % | -0.31 | +0.14 | -0.07 | -0.08 | +0.034 | 62 |
| 192 x 64 x 96, finer across the flow | 11.8 / 1.0 / 7.9 | 10 | -0.60 % | -0.17 | -0.10 | -0.04 | -0.04 | +0.012 | 81 |
| 192 x 96 x 64, finer toward the wall | 11.8 / 0.67 / 11.8 | 10 | -0.54 % | -0.36 | +0.15 | -0.08 | -0.09 | +0.036 | 111 |
| 288 x 64 x 64, finer along the flow | 7.9 / 1.0 / 11.8 | 10 | +0.77 %, rising | +0.26 | +0.11 | -0.08 | -0.08 | +0.018 | 118 |
| **288 x 64 x 96, the final grid** | 7.9 / 1.0 / 7.9 | 30 | +0.17 % +- 0.19 % | +0.10 | -0.04 | -0.03 | -0.03 | +0.007 | 164 |

- *The final grid meets every pass / fail limit of case 17:* the five profiles, the bulk velocity,
  the centre velocity (+0.53 %, limit 1), the friction coefficient (-0.34 %, limit 2), the peak
  of the streamwise fluctuation (-0.69 %, limit 2) and its place (0.4 wall units off, limit 1.5).
  Its run ended on the stop rule after 30 time units; the two plates agree (0.1 standard errors)
  and the total shear stress lies within 0.0006 of its exact straight line. Cost: 1.7 GPU hours
  with start-up (2.25 times the cells of case 17's finest grid, 2.6 times the time per time unit).
- *What limits the accuracy:* the cells across the flow first (they must resolve the streaks at
  the wall), the cells along it second. Toward the wall 64 cells with a wall cell of one wall
  unit are enough: refining there changed nothing beyond the scatter.
- *The bulk velocity of the first grid was right by cancellation.* Refining across the flow lowers
  it, refining along the flow raises it by about 1 %.
- *Limits of this result.* It is one grid, with no grid study at its level. The last step in each
  direction still moved the bulk velocity by 0.6 to 0.8 % in opposite directions, so by a
  second-order estimate about 0.5 % of grid error is left from each, and the agreement to 0.17 %
  again includes cancellation. The time-step check was not repeated (same step control as
  case 17). One Reynolds number.
- *The cheap option:* 192 x 64 x 96 costs 1.3 times the first grid and brings four of the five
  profiles inside (the streamwise fluctuation stays out: -0.10, its peak 2.7 % low).
- *How the grid was picked, and what did not work.* The plan was to split each gap into shares
  of the three directions, assuming second order, and to predict candidate grids from them.
  (1) On the gap as one number this fails: the place of the largest difference moves from run
  to run and its sign with it. The model has to be applied at each height of the profile.
  (2) Its own check failed even so: with all directions refined without end it leaves +0.24 in
  the mean velocity, several times the scatter, so the grids are not yet in the range where the
  error falls with the square of the cell size. It ranks the directions and no more. (3) What
  held was the plainest form: the grid refined in two directions is the first grid plus the two
  measured changes. Predicted against measured: fluctuations and shear stress within 0.015, the
  mean velocity +0.20 against +0.10.
- *Start-up.* Each run started from a field of the nearest grid, ran 4 time units, had its
  velocity scaled once to where the wall shear pointed (`gpucfd.turbulence.rescale_to_balance`),
  and settled for 3 more. The wall shear over the averaging was then within 0.2 % of the driving
  force in three of four runs: 7 time units of start-up where case 17's levels needed about 20.
  In the fourth (finer along the flow) it was 0.7 % off and the bulk velocity still rising: its
  field was still adapting when it was scaled.
- *Cost notes.* Steps per time unit at the same Courant target: 960 on the first grid, 808 with z
  refined (the better resolved flow has milder extremes), 1136 with x refined, 962 on the final
  grid. Pressure iterations per step rose from 7.3 to 9.7. The cost projected from the
  directional runs was 139 s per time unit, measured 164.
- GPU time of the stage: 3.1 of the 5 hours budgeted.

**The other route, a model as in case 7 (RANS), predicted to lose.** On case 7's finest grid
(208,896 cells, aspect ratio 21,479) the thin cells allow a convective time step of 1.7e-6 s,
measured on OpenFOAM's solution, and one pass of the flow over the plate takes 34 thousand such
steps. With implicit diffusion at about 30 ms per step that is 1.5 to 3 hours where OpenFOAM's
steady solver took 10 minutes. It would also need the model's equation, a viscosity that varies
in space, a pressure solver for aspect ratios of 10^4 (tested to 64) and implicit convection. Not
built.

**Open.**
- One flow, one Reynolds number. One grid resolves its profiles (288 x 64 x 96, the optimize
  stage above), with no grid study at that level: about 0.5 % of grid error is left in the bulk
  velocity from each of two directions, of opposite sign.
- Used in case 17 and so no longer open: cells with dx = dz, and the pressure tolerance of 1e-6.
- Still to try, for accuracy per cost:
  - a finer step across the flow on the final grid (288 x 64 x 128, 2.4 million cells), to see
    whether the profiles have stopped moving;
  - several shorter runs from decorrelated fields in place of one long one: the same statistics,
    and the error straight from their scatter;
  - the Courant rate computed on the GPU, so the step can be re-chosen every step at no cost
    (today the field is read back for it);
  - one pressure solve per step in place of three, and the stretched code path.
- What a model adds. At this Reynolds number none is needed. At friction Reynolds number 590 the
  same resolution in wall units needs about 17 million cells (an estimate by scaling, not a
  measurement), the limit of this card's memory; there a model for the smallest eddies on a
  coarser grid (large-eddy simulation) would pay. Its first brick, a viscosity that varies from
  cell to cell, is also the first brick of any other model.

## Limitations (2026-11-12)

**Problems it cannot solve yet**
- Geometry: a box with immersed bodies of fixed shape (any exact signed distance; surfaces may move
  tangentially, bodies may not translate). Around a body the grid can be clustered (stage 2,
  2026-11-13), but it stays rows and columns: the fine band crosses the whole domain in each
  direction, so the saving is bounded (case 4: 3.85 times fewer cells, where a body-fitted grid
  has 30 times fewer). With the explicit scheme the smallest cell sets the time step
  (0.15 h_min^2 / nu); IMEX with bodies (2026-11-13) removes that limit at about twice the cost
  per step, so it pays when the limit is more than about twice below the convective step (case 4:
  1.55 times faster at 140 cells per diameter, slower at 35). Either way the pressure tolerance
  that keeps the forces clean tightens with the grid (case 4: 1e-8, 1e-9, 1e-10 explicit and 1e-9,
  1e-9, 1e-10 IMEX at 35, 70, 140 cells per diameter; check it on the finest level). The explicit
  scheme with bodies is stable only while the ghost feedback mu stays below 0.683 (0.26 to 0.34
  in every body run so far; `tests/imex_body_model.py` computes it for a 2D case); IMEX with
  bodies has no such limit. The pressure next to a body surface converges
  irregularly in the max norm, and the surface pressure depends on where the grid cuts the body
  (DFG 2D-1 pressure difference at 35 cells per diameter: -0.2 to -2.4 % over four positions);
  forces come from the momentum balance away from the surface and scatter far less (0.25 %).
- Boundaries: periodic; no-slip walls that can slide tangentially (a moving lid); inflow with any
  steady velocity profile; outflow with zero pressure and zero velocity gradient. No symmetry/slip
  boundaries, no prescribed-pressure inlets, no time-varying inflow (the profile is fixed per run;
  it can be changed between runs), no convective outflow condition (vortices leaving the domain may
  reflect more than with one).
- Physics: incompressible, single phase, constant density and viscosity; laminar, or turbulent
  with every eddy resolved on the grid (measured on one flow, the channel at friction Reynolds
  number 180, section above; the largest grid this PC affords gave the mean flow rate to 0.3 %
  but not the profiles). No turbulence model, RANS or LES, no heat transfer or transported scalars,
  no buoyancy, no free surface, no compressibility. Body force constant in space and time.
- Numerics: central differences without upwinding or limiters, so the grid must resolve the flow
  (cell Reynolds number U h / nu around 2 or below for clean results); at higher Reynolds numbers it
  will oscillate rather than smear.
- Grid sizes: periodic directions need even cell counts, and multigrid works well only when each
  direction halves several times (sizes like 2^k or 3 x 2^k). An awkward size leaves a large coarsest
  grid, which is solved slowly by one thread block.
- Stretched grids (2026-11-13): bounded directions only (a periodic direction must be uniform); at
  most 3072 cells along a stretched direction (1536 with a double-precision pressure solver); cell
  counts must coarsen to at most 1024 cells; with bodies, at least two multigrid levels; a 3D
  grid clustered in two directions converges slowly (23-38 pressure iterations at ratio 16, growing
  with the grid), and bodies on 3D grids clustered in all three directions are untested; grid
  sequencing (`steady.prolong`) works between stretched grids in the two cylinder studies but has
  no check of its own. With the explicit scheme the smallest cell sets the viscous time step for
  the whole domain (0.15 h_min^2 / nu): use IMEX where there is no body, and with a body when that
  limit is more than about twice below the convective step.
- Tooling: no field output for ParaView, no restart files; forces on bodies and surface pressure
  come from Python (`NavierStokes.forces`, `surface_pressure`), fields through `get` and `pressure`.
  One GPU, no MPI.

**Speed limits**
- Small grids are latency-bound: a time step is about 1200 dependent GPU operations of a few
  microseconds each, so it costs about 2 ms whatever the grid size below about 128^2.
- Double-precision arithmetic: consumer GPUs (the RTX 4070 here, and the RTX Spark-class machines)
  run FP64 at 1/64 of the float rate. The velocity, the diffusion solves and all reductions are
  double; only the pressure multigrid runs in float (mixed precision).
- Large grids are limited by memory bandwidth, and the pressure is solved three times per time
  step (once per Runge-Kutta stage).
- IMEX with bodies: the two systems of a stage are not symmetric and are solved by BiCGStab (two
  V-cycles and two operator applications per pressure iteration). A step costs about twice an
  explicit one at the same tolerance (case 4, 459 thousand cells: 62.5 against 29.5 ms).
- Stretched grids cost about twice the time per step of a uniform grid with the same cells (line
  relaxation; 8.5 against 4.3 ms on 128^2), and another 45 % at wall aspect ratio 64. With a body
  the gap is smaller, 1.2 to 1.5 times at the same cells and tolerance (the line smoother needs 5
  pressure iterations where the point smoother needs 7.7).
- Steady problems are time-marched: the number of time units to settle is set by the physics (27
  for the Re 100 cavity; in a grid study, sequencing and the error-based stop of `steady.py` cut the
  finer grids to a few time units, 20x at DFG D/h 35), and convection is explicit, so the time step is bounded by the flow speed
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
4. ~~Inflow/outflow~~ (done 2026-11-12, rung 9, below).
5. ~~Curved bodies (immersed boundary) with force coefficients~~ (done 2026-11-12, rung 10, above).
   ~~Open: IMEX with bodies (instability with the incremental pressure)~~ (solved, item 8).
   Steady studies: grid sequencing and an error-based stop (done 2026-11-12, rung 11, above). The
   surface pressure difference that seemed to get worse on the finest DFG grid is scatter with the
   position of the surface in the grid (stage 2, item 3). Open: a surface pressure that does not
   depend on the cut.
6. ~~gpucfd as a backend of the agent's pipeline (cfdlib)~~ (done 2026-11-12, case 4, above): 8x faster
   than OpenFOAM over the protocol, but less accurate per cell (drag -1.0 % against +0.25 %).
7. Stretched grids (boundary layers without refining everywhere). Case 4 shows this is what decides
   whether gpucfd beats OpenFOAM on bodies: the solver is 25 to 45x faster per cell and step (wall time on both
   sides; first written as 50 to 70x, which set GPU time against wall time), the uniform grid
   needs about 30x more cells. Measure of success: case 4's drag within 0.25 % of FeatFlow (OpenFOAM's
   M4) in less wall time than OpenFOAM's study.
   - ~~Stage 1: grids with walls, no bodies~~ (done 2026-11-13, rung 12, above). The pressure solver
     holds 7-10 iterations up to aspect ratio 64 in 2D.
   - ~~Stage 2: bodies on stretched grids~~ (done 2026-11-13, rung 13, above). Case 4 with cells
     clustered around the cylinder: drag within 0.11 % of FeatFlow's finest tabulated level (0.31 %
     below the value it converges to, where OpenFOAM is within 0.04 %) at 140 cells per diameter, three
     levels in 1 421 s of wall time (OpenFOAM's study: about 1 740 s). The measure of success is
     met as defined; the levels' own uncertainty estimate is 0.75 %, see the limits there.
   - Speed of the stretched code path, 1.2 to 1.5 times the uniform one per step with a body (2 times
     without). Candidates, each to be measured: both zebra colours and both directions in fewer
     kernels, parallel cyclic reduction in place of one thread per line solve, a diagonal
     preconditioner for the diffusion solves.
   - 3D grids clustered in two directions: plane relaxation or semi-coarsening (needed for case 5).
   - Answered by three check runs (2026-11-13, stage 2 section): the drag at 140 cells per diameter
     is 0.1 to 0.2 % off the tabulated benchmark (0.3 to 0.4 % off its converged value) over two
     positions of the cylinder in the grid; the uniform study costs
     328 s with its rule's tolerance; a longer fine band behind the cylinder recovers lift and
     Strouhal number and uses up the time advantage at 70 cells per diameter. The longer band on
     the finest level was run with IMEX in stage 3 (lift maximum 0.69 % low instead of 0.85 %, study
     1 308 s; one position, and the gain is smaller than the 0.3 % that moving the cylinder changes).
     Still open: more than two positions, and that run with the cylinder shifted.
8. ~~IMEX with bodies~~ (done 2026-11-13, rung 14, above). Case 4's study on the clustered grids:
   1 005 s instead of 1 421 s with the same answer; the finest level 1.55 times faster. Less than
   hoped: a step costs twice an explicit one and the tolerance did not relax. Next for speed with
   bodies, each to be measured: a second-order scheme with one pressure solve per step in place
   of three; cheaper diffusion solves; and, as a question of method, the consistent projection for
   the explicit scheme too (it would remove that scheme's stability limit mu < 0.683).
9. 3D cylinder (DFG 3D, case 5) once stretched grids work.
10. Turbulence. The channel was run as a verified case (case 17, 2026-11-14, verdict FAIL: mean
    flow rate within 0.3 %, profiles not resolved on 192 x 64 x 64). The optimize stage on the same
    case found the grid that meets every limit (288 x 64 x 96, 1.7 GPU hours; channel section).
    Next, with a target at a higher Reynolds number, a viscosity that varies in space and a model
    for the smallest eddies.

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
