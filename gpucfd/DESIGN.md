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

Built and verified: the pressure Poisson solver (rungs 0-5) and Navier-Stokes on periodic boxes
(rung 6). Run the ladder with `python -m gpucfd.tests.test_poisson` and `python -m gpucfd.tests.test_ns`;
results in `gpucfd/tests/results/`.

| Rung | Result |
|---|---|
| 0 | RTX 4070, measured copy bandwidth about 390-410 GB/s |
| 1 | Laplacian equals the numpy reference to 2e-16 (double), 7e-8 (float), all boundary sets, 2D and 3D |
| 2 | Solutions equal a dense direct solve to 2e-12 |
| 3 | Order of accuracy 1.99-2.00 for Neumann, Dirichlet, periodic and mixed boundaries, 2D and 3D |
| 4 | Multigrid: 12 V-cycles (2D) / 13-14 (3D) to 1e-10 at every grid size from 32^2 to 1024^2 and 16^3 to 128^3; with CG 8-9 iterations; reduction 0.06-0.17 per iteration |
| 5 | Mixed precision (double outer iteration, single-precision multigrid) equals double to 1e-14 with the same iteration count. Pure single precision is NOT safe: on a 256^2 grid its error was 6 times the discretisation error (round-off grows like 1/h^2), so it is not used |
| 6 | Taylor-Green (2D) and ABC flow (3D): spatial order 2.00, temporal order 3 (RK3), divergence 1e-13, mixed = double to 1e-15 |

Bugs the ladder caught while building (each would have produced plausible-looking wrong results):
1. Multigrid corrections on coarse levels started from leftover memory (rung 2 and 4 failures).
2. Pure single precision silently lost accuracy on fine grids (rung 5): led to mixed precision.
3. A blown-up run reported "divergence 0": `fmax` ignores NaN. The maximum now propagates NaN and
   `NavierStokes.run` raises with the stability limits.

Speed (RTX 4070, `python -m gpucfd.bench.bench_poisson`, numbers in `bench/results/`):
- Pressure solve to 1e-6, mixed precision, 3D: 9.6 ms at 2.1 million cells, 100 ms at 16.8 million.
- Large grids reach 50-60 % of the measured bandwidth; small grids are dominated by kernel launches
  (about 100 per V-cycle).
- Navier-Stokes (RK3, 3 projections per step, about 7 CG iterations each): 62 ms per step at 2.1
  million cells, 480 ms at 16.8 million (about 28 ns per cell per step). The pressure solve is nearly
  all of it.
- Not yet compared with OpenFOAM on the same problem: that needs the same equations, grid and
  tolerance on both sides (next).

## Next steps (in order)
1. **Fair OpenFOAM comparison** on the same problem (periodic 3D flow or the cavity): same grid,
   same tolerance, same answer; OpenFOAM with its best setup (GAMG, 8 or 16 ranks).
2. **Walls** (no-slip, moving lid) on the staggered grid, then rung 7: lid-driven cavity Re 100
   against Ghia et al. and case 2.
3. **Speed, each change gated by the ladder**: projection tolerance tied to the divergence target
   instead of 1e-10 relative; CUDA graphs against launch overhead; red-black kernels that launch only
   the cells of one colour; fused residual + restriction; on-device CG scalars (no host sync per dot).
4. Stretched grids, inflow/outflow, then curved bodies (immersed boundary) for the DFG cylinders.
