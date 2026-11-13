# gpucfd: a GPU-resident incompressible flow solver

Goal: a solver that keeps the whole time step on the GPU and is trusted because every part is
verified before it is made fast. Case 5 showed why: OpenFOAM's GPU plugin only moves the pressure
solve to the GPU, copies matrices through host memory every solve, and is capped near 1.5 times.

## Scope (first version)
- Incompressible, single phase, laminar (constant density and viscosity).
- Structured Cartesian grids, uniform or stretched per direction (cells clustered toward walls;
  stage 1, 2026-11-13: without bodies), 2D or 3D (2D = one cell in z).
- Staggered (MAC) grid: pressure at cell centres, each velocity component on its faces. Discretely
  divergence-free velocity, no pressure checkerboarding.
- Projection method: predict velocity, solve a pressure Poisson equation, correct.
- Pressure Poisson equation: geometric multigrid (red-black Gauss-Seidel smoother, cell averaging
  restriction, (bi/tri)linear prolongation), alone or as the preconditioner of conjugate gradients.
  On stretched grids: zebra line relaxation and the transposed prolongation as restriction.
- Boundaries: periodic, walls (no-slip, moving lid), inflow (prescribed velocity profile), outflow
  (pressure zero, zero velocity gradient).
- Immersed bodies of any shape given by a signed distance (ghost-cell method), with forces.

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
  grids.py         face positions for stretched grids (tanh clustering)
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
test_open | test_bodies | test_steady | test_stretched | test_reproducible`); results in
`gpucfd/tests/results/`. Stretched grids (rung 12) were added on 2026-11-13.

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
| 10 | Immersed bodies (`test_bodies`, explicit scheme): circular Couette flow between immersed cylinders, velocity order 2.50, 2.10 (rms), pressure away from the walls 2.37, 1.48; channel at atan(1/2) to the grid, mean order 2.24; 3D extrusions equal 2D; DFG 2D-1 drag -2.26 % / -0.65 % (D/h 17.6 / 35), pressure difference -0.23 % at D/h 35; forces exact discrete momentum balance (box-independent to 2e-18). Details: Immersed bodies, below |
| 11 | Steady-state tooling (`test_steady`, `gpucfd/steady.py`): iteration-error estimator exact for one geometric mode, conservative (9x) for an oscillating one, never too small while a fast mode dies out once the slow rate is known, noise floors recognised in 97 % of noisy histories and sized (covers the true error 97 % of the time); prolongation 32^2 -> 64^2 of the Couette flow keeps the coarse error (5.5e-3 vs 6.0e-3); Couette 32/64/128 sequenced vs from rest: same answers (worst 0.29 of the allowance), 7.1 s vs 14.2 s GPU on the refined grids. DFG 2D-1 study (one-off): 20x at D/h 35. Details: Steady flows, below |
| 12 | Stretched grids (`test_stretched`, 2026-11-13): with uniform face positions the new code path equals the uniform code (Laplacian exactly, solutions 1.5e-11, Navier-Stokes 1e-14 / 7e-16); stretched Laplacian equals the numpy reference to 6e-15, solutions equal a direct solve to 2e-12; manufactured solutions order 1.99-2.01 (2D and 3D, cell ratio 8); pressure iterations 9 / 10-11 / 11-12 at wall aspect ratio 4 / 16 / 64 on 64^2 to 512^2 (uniform 8-9); convection conserves kinetic energy to 1e-17; Poiseuille from rest order 2.03 (IMEX) and 2.05 (RK3), decaying channel mode 2.03-2.04 (2D and 3D, both schemes), Kovasznay L2 order 2.07, 2.02, open channel 2.04 with flow out = flow in exactly; both schemes reach the same steady state (cavity 5e-9 U, Kovasznay 2e-8, open channel 1e-9); cavity Re 100 on clustered grids within 0.005 (u) / 0.009 (v) U of Ghia and 0.0052, 0.0013, 0.00025 U from the uniform 256^2 solution at N = 32, 64, 128; reproducible bit for bit. Details: Stretched grids, below |

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
   8 times below the convective limit).

**Forces.** `forces(box)`: summed over the fluid faces in a box around the bodies, the solver's own
discrete momentum equation telescopes to the fluxes through the box surface plus those into the
bodies, so F = -[sum_box (rhs - grad p - g) vol + sum_surface (flux out) area] with the solver's
fluxes is the exact discrete force, including the unsteady term. For a random field and no body it
returns 6e-16; two different boxes around the DFG cylinder agree to 2e-18. `surface_pressure(points)`
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
| DFG 2D-1 (Re 20), reference cD 5.57954, cL 0.010619, dP 0.117520 (Nabh 1998) | D/h 17.6: cD -2.26 %, dP -3.24 %, cL 4.3 times the reference; D/h 35: cD -0.65 % (order 1.8), dP -0.23 %, cL +15 % |

Immersed-boundary errors fluctuate with how the surface cuts each grid (a single grid pair can read
1.4 or 3), so the orders are judged on the mean over the range. The lift in 2D-1 is 0.2 % of the drag,
so its relative error is large on coarse grids; it falls with refinement.

### Steady flows: grid sequencing and an error-based stop (2026-11-12)
Steady flows are reached by marching in time. With bodies that march must use RK3 (IMEX is unstable
there, above), so on fine grids the time step is small (diffusion limit 0.15 h^2 / nu). DFG 2D-1 at
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
| St | 0.30276 | +0.29 % | 0.30150 | -0.13 % | yes |

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
| total | 255 s GPU, 315 s wall | 2548 s wall (M4 alone 2110 s on 16 ranks) |
| per cell and step | 0.023 microseconds (G3) | 1.64 microseconds (M4, wall, 16 ranks) |
| finest drag error | -1.0 % | +0.25 % (M2 already +0.24 %, 97 s on one core) |

So gpucfd ran the protocol 8 times faster and is about 70 times faster per cell and step, but it is not
faster at equal accuracy: OpenFOAM's O-grid reaches 0.25 % at 13432 cells in 97 s, which gpucfd would
need roughly 140-280 cells per diameter for (estimated from order 1.5; not run: D/h 140 is 1.8 million
cells with a viscous time-step limit, about 40 GPU-minutes). The cost is in the grid, not the solver:
a uniform grid has to resolve the boundary layer everywhere, and the immersed body converges at order
1.5. Stretched grids (next steps) attack exactly that.

### Stretched grids, stage 1: boxes with walls, no bodies (2026-11-13)

**Why.** On a uniform grid one thin layer at a wall sets the cell size for the whole domain. A
stretched grid keeps the same number of cells per direction but lets their widths vary: small at
the walls, large in the middle. This is stage 1 of three (ROADMAP step 2): the discretisation and
the pressure solver on such grids, for boxes with walls, inflow/outflow and periodic sides. Bodies
on stretched grids and IMEX with bodies are stages 2 and 3 and are refused for now.

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

## Limitations (2026-11-12)

**Problems it cannot solve yet**
- Geometry: a box with immersed bodies of fixed shape (any exact signed distance; surfaces may move
  tangentially, bodies may not translate). Stretched grids only without bodies (stage 1, 2026-11-13),
  so around a body the grid is still uniform and a boundary layer forces it fine everywhere. Bodies need the explicit scheme (IMEX with bodies
  is unstable, see above), and the pressure next to a body surface converges irregularly in the max
  norm (the cells next to it; forces come from the momentum balance away from them).
- Boundaries: periodic; no-slip walls that can slide tangentially (a moving lid); inflow with any
  steady velocity profile; outflow with zero pressure and zero velocity gradient. No symmetry/slip
  boundaries, no prescribed-pressure inlets, no time-varying inflow (the profile is fixed per run;
  it can be changed between runs), no convective outflow condition (vortices leaving the domain may
  reflect more than with one).
- Physics: incompressible, single phase, constant density and viscosity, laminar. No turbulence
  model (only resolved simulations at low Reynolds number), no heat transfer or transported scalars,
  no buoyancy, no free surface, no compressibility. Body force constant in space and time.
- Numerics: central differences without upwinding or limiters, so the grid must resolve the flow
  (cell Reynolds number U h / nu around 2 or below for clean results); at higher Reynolds numbers it
  will oscillate rather than smear.
- Grid sizes: periodic directions need even cell counts, and multigrid works well only when each
  direction halves several times (sizes like 2^k or 3 x 2^k). An awkward size leaves a large coarsest
  grid, which is solved slowly by one thread block.
- Stretched grids (2026-11-13): bounded directions only (a periodic direction must be uniform); at
  most 3072 cells along a stretched direction (1536 with a double-precision pressure solver); cell
  counts must coarsen to at most 1024 cells; no bodies, and `forces`, `pressure_at` and
  `surface_pressure` refuse; a 3D grid clustered in two directions converges slowly (23-38 pressure
  iterations at ratio 16, growing with the grid); grid sequencing (`steady.prolong`) reads the
  stretched positions but has not been tested on them. With the explicit scheme the smallest cell
  sets the viscous time step for the whole domain (0.15 h_min^2 / nu): use IMEX.
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
- Stretched grids cost about twice the time per step of a uniform grid with the same cells (line
  relaxation; 8.5 against 4.3 ms on 128^2), and another 45 % at wall aspect ratio 64.
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
   Open: IMEX with bodies (instability with the incremental pressure).
   Steady studies: grid sequencing and an error-based stop (done 2026-11-12, rung 11, above). Open:
   the surface pressure difference on the finest DFG grid.
6. ~~gpucfd as a backend of the agent's pipeline (cfdlib)~~ (done 2026-11-12, case 4, above): 8x faster
   than OpenFOAM over the protocol, but less accurate per cell (drag -1.0 % against +0.25 %).
7. Stretched grids (boundary layers without refining everywhere). Case 4 shows this is what decides
   whether gpucfd beats OpenFOAM on bodies: the solver is 70x faster per cell and step, the uniform grid
   needs about 30x more cells. Measure of success: case 4's drag within 0.25 % of FeatFlow (OpenFOAM's
   M4) in less wall time than OpenFOAM's study.
   - ~~Stage 1: grids with walls, no bodies~~ (done 2026-11-13, rung 12, above). The pressure solver
     holds 7-10 iterations up to aspect ratio 64 in 2D.
   - Stage 2: bodies on stretched grids. Needs: the ghost-cell rules of `bodies.py` on unequal cells
     (image distance and interpolation weights from the local cell sizes), the body's face
     coefficients in the stretched stencil and in the line solves, the force balance and the
     surface pressure with the cell sizes. Then case 4 with cells clustered around the cylinder.
   - Speed of the stretched code path, now 2 times the uniform one per step. Candidates, each to be
     measured: both zebra colours and both directions in fewer kernels, parallel cyclic reduction
     in place of one thread per line solve, a diagonal preconditioner for the diffusion solves.
   - 3D grids clustered in two directions: plane relaxation or semi-coarsening (needed for case 5).
8. IMEX with bodies (the time step on fine grids is set by the explicit viscous limit; on case 4's G3
   it is 0.6 times the convective one). On a stretched grid with a body this is no longer optional:
   the explicit limit follows the smallest cell squared.
9. 3D cylinder (DFG 3D, case 5) once stretched grids work.

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
