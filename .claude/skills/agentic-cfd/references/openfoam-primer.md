# OpenFOAM primer and known pitfalls (ESI OpenFOAM v2606)

## What each file controls (each is a decision to document)

| File | Controls | Affects |
|---|---|---|
| `system/controlDict` | `application` (solver), endTime, deltaT, write settings, function objects (outputs) | Physics solved; time accuracy; what is measured |
| `system/blockMeshDict` | Geometry, cell counts, grading, boundary patches and their types | Discretisation error; which boundaries exist |
| `0/<field>` | Initial field and boundary conditions per patch | Defines the problem (wrong BCs converge to a wrong answer) |
| `constant/transportProperties` | Viscosity model and value (kinematic `nu`) | Physics |
| `constant/turbulenceProperties` | `simulationType laminar` or RAS/LES model | Modelling error mesh refinement cannot remove |
| `constant/fvOptions` | Sources (body forces, porosity, ...) | Physics |
| `system/fvSchemes` | Time, gradient, convection (div), Laplacian, interpolation schemes | Order of accuracy, numerical diffusion, boundedness |
| `system/fvSolution` | Linear solvers and tolerances, pressure-velocity algorithm (SIMPLE/PISO/PIMPLE), relaxation | Convergence, stability, speed; iteration error |

Incompressible solvers use **kinematic** pressure `p/rho` [m²/s²] and kinematic viscosity `nu`.

## Common solvers (incompressible)
- `icoFoam`: transient, laminar, PISO. Simple and robust; no fvOptions support.
- `simpleFoam`: steady, SIMPLE; laminar or RANS. Needs relaxation for nonlinear problems.
- `pimpleFoam`: transient, PIMPLE (PISO + outer iterations); laminar/RANS/LES; supports fvOptions.
Tutorials live in `$FOAM_TUTORIALS/incompressible/<solver>/...` (e.g. `icoFoam/cavity/cavity`).
- 1D transient wall-bounded flows: `pimpleFoam/laminar/planarPoiseuille` (cyclic + empty, one cell
  across). For a LINEAR problem, `momentumPredictor yes` with 1 outer corrector is enough; the
  tutorial's settings (15 outer correctors, predictor off) are for its non-Newtonian model.
- Time schemes: `backward` (second order, BDF2) takes its first step with Euler (first order); a
  per-step balance must use the same formula step by step. `CrankNicolson 0.9` is the other common
  second-order choice.

- **Time accuracy of PISO/PIMPLE**: with `nOuterCorrectors 1` (PISO mode) the scheme is only
  FIRST order in time for nonlinear flows, whatever `ddtSchemes` says (case 4: lift amplitude 10 %
  off at Courant 0.8; changes halved with the time step). Use 2 outer correctors (PIMPLE) when forces
  or frequencies must be time-accurate, and confirm the order with a 3-point time-step study.

## Non-uniform inlet profile (no compilation)
`exprFixedValue` with `valueExpr` (v2606 keyword; `expression` is rejected) and `pos()` for the face
centre, e.g. a parabola: `valueExpr "vector(4*Um*pos().y()*(H - pos().y())/(H*H), 0, 0)";` plus a
`value uniform (0 0 0);` line. See `cases/dfg_cylinder_re100/template/0/U`. Check the inlet flow
rate in the pre-flight (`surfaceFieldValue`, `operation sum`, field `phi`) against the exact value.

## Curved bodies with blockMesh (O-grid)
A cylinder (or any round body) in a channel: put a square box around the body, split the box into
four O-grid blocks between the box sides and the body (cells graded towards the wall), fill the rest
of the channel with ordinary blocks, and make the body's edges `arc`s. Generate vertices, blocks,
arcs and patches in Python and fill them into the template with Jinja loops (doubling every cell
count per mesh level is then trivial). Worked example: `mesh_spec()` in
`cases/dfg_cylinder_re100/run.py` and its `template/system/blockMeshDict` (checkMesh max
non-orthogonality about 41-43 degrees at the O-grid corners: fine). 3D: extrude the same blocking in
z with real cells instead of one `empty` layer.

## Function-object outputs in v2606 (check every file in the pre-flight)
- `forces` writes `force.dat` as NINE scalar columns (total, pressure, viscous; x y z each), not
  bracketed vectors.
- `surfaceFieldValue` with `areaNormalIntegrate` of a SCALAR (e.g. `p`) silently writes 0; use
  `areaIntegrate` and multiply by the patch normal yourself.
- Print the header and first rows of every `postProcessing` file of the pre-flight run and check each
  is non-trivial before relying on it.
- The solver log names every pressure solve of a PISO/PIMPLE step `p` (no `pFinal` in the log); count
  the final solve of a step by position (`cfdlib.perf` does).

## Parallel runs (MPI)
`system/decomposeParDict` (`numberOfSubdomains N; method scotch;`), then `decomposePar`,
`mpirun -np N <solver> -parallel` (through `cfdlib.foam.run`), `reconstructPar` (all times or
`-latestTime`). Function-object output is written by the master. Small meshes do not scale (too few
cells per core, communication dominates): measure; a common rule of thumb is that below roughly
10-20 thousand cells per core scaling flattens, but find this machine's value.
- More than 8 ranks needs `mpirun --use-hwthread-cpus -np N` (Open MPI counts only the 8 physical
  cores as slots; without it: "not enough slots available").
- Measured on this machine (case 4, pimpleFoam, 215 thousand cells, GAMG pressure): speed-up 1.6, 2.8,
  4.2 and 5.4 on 2, 4, 8 and 16 ranks; 16 (all hardware threads) was fastest. The limit is memory
  bandwidth, and GAMG needs more pressure iterations on more subdomains (5.5 on 4, 8.2 on 8).
- Measured on this machine (case 5, simpleFoam, steady 3D, 0.3-4.75 million cells): 8 ranks beat 16
  on every mesh (2.4 million cells: 1.72 against 1.90 s per iteration). A steady 3D SIMPLE iteration
  is memory-bandwidth bound; hyper-threads add MPI traffic without adding bandwidth. So the best rank
  count depends on the case: measure 8 and 16.
- Pressure solver: GAMG was 4 times faster than PCG with DIC (182 iterations per solve) on 54
  thousand cells. GAMG's `smoother DICGaussSeidel` needed 5 instead of 13 iterations per solve in case 5
  (6 % faster per iteration than `GaussSeidel`). Never benchmark a GPU solver against PCG on one core.
- Momentum balance through an inlet whose profile is not the developed one: the viscous normal stress
  at the inlet integrates to zero in the continuum but not discretely (a first-order remainder that
  shrinks only with the first cell size). Include or measure it (case 5: three quarters of the imbalance).

## Pitfalls already met (each cost real time)
1. **Courant number**: transient PISO solvers become unstable when max Co exceeds ~1. Refining the
   mesh at a fixed time step raises Co; scale `deltaT` with the cell size. The tutorial cavity uses
   Co ≈ 1 at the lid: any refinement without changing deltaT diverges.
2. **GAMG pressure with loose first solve in TRANSIENT PISO**: icoFoam cavity with GAMG for p and
   the tutorial's `relTol 0.05` diverged (error grew ~1.4x per step) while PCG was fine. `relTol 0`
   for p fixed it. This applies to transient PISO/PIMPLE pressure solves. In steady SIMPLE(C),
   `relTol 0.05-0.1` is normal (the outer iterations remove the partial-solve error; case 2 used it).
   After switching linear solvers, always run a pre-flight.
3. **Normalised residuals of ~zero fields are noise** (e.g. v and p in Poiseuille flow). They never
   fall, so `residualControl` never triggers and plots look alarming. Use a physical check.
4. **Under-relaxation is a pseudo time step**: on diffusion-dominated problems its cost grows like
   N² (thousands of iterations at 128 cells). Linear problems: relaxation 1. Nonlinear: needed
   (typical SIMPLE: U 0.7, p 0.3; SIMPLEC `consistent yes`: U 0.9, p 1). The textbook values are
   cautious: in case 2, SIMPLEC with U 0.97-0.99 reached the same answer 3-9× sooner. See "Steady runs:
   getting to the answer fast".
5. **SIMPLEC without relaxation divides by zero** for pure diffusion (floating-point exception).
6. **Output precision**: default `writePrecision 6` truncates coordinates and values; use 12 for
   analysis, and locate cells by index or with sampling function objects, not by exact coordinates.
7. **Sourcing the environment**: `source /usr/lib/openfoam/openfoam2606/etc/bashrc` fails under
   `set -eu`; source it before enabling strict mode (cfdlib.foam.run does this for you).
8. **foamDictionary expands macros** (`$p`) when it edits a file: an entry defined as `{ $p; relTol 0; }`
   becomes a full copy. Edit both entries or write the file from a template instead.
9. **The log banner contains "Floating point exception trapping enabled"**: that is not an error.
10. **Residual targets are not a solution-error target** (case 2): on finer meshes SIMPLE needs more
   iterations (about 3.5x per refinement at Re 100) and the same residual target leaves a larger
   change per iteration. Choose the targets from the physical check, verify them on the coarse
   meshes, expect roughly one more decade per refinement.
11. **Conservation checks must use the conservative quantities**: a stream function rebuilt from
   cell-centre velocities differs from the face fluxes near singular corners (Rhie-Chow term), so a
   "mass" check built on it tests post-processing, not the solver. Use face fluxes (`phi`), OpenFOAM's
   `streamFunction`, or evaluate away from singularities.
12. **Pre-flight and production in the same directory**: `render_case` deletes the target directory
   first, so re-running a pre-flight there wipes a production run. Use separate directories.
13. **PISO is first order in time** for nonlinear flows (see "Time accuracy" above): it cost case 4 a
   full re-run of its mesh study.
14. **A pre-flight from rest underestimates the cost of a developed unsteady flow**: in case 4 GAMG
   needed 7-12 pressure iterations per solve in the shedding flow versus fewer at start-up, and the
   finest run took twice the projection. Start the pre-flight from a developed field
   (`mapFields` from a coarser run) or budget twice the projection.

## Steady runs: getting to the answer fast (measured on case 2's cavity)
A steady SIMPLE run is a march in pseudo-time. Under-relaxation sets the pseudo time step of each
cell, about α/(1−α) divided by its matrix diagonal. Where viscosity dominates (most cells of a fine
laminar mesh) that diagonal grows like ν/h², so the step shrinks like h² and the iteration count
grows like N² (case 2: 3.6× per refinement). Three levers, measured on the Re 100 cavity, one core,
each counted to the same answer: the probes within 1e-6 U of the residual-converged run.

| Mesh | Case 2 as run (SIMPLEC, U 0.9, from rest) | Faster settings found | Time saved |
|---|---|---|---|
| 128² | 1931 it, 27 s | U 0.98: 411 it, 9.3 s; start from 64², U 0.97: 471 it, 7.1 s | 3-4× |
| 256² | 7491 it, 437 s (residual targets: 10864 it, 627 s) | U 0.99: 801 it, 111 s; start from 128², U 0.97 or 0.99: 981-1241 it, 70-77 s | 6× (9× against the run as made) |
| 512² | not run; about 27000 it, about 1.9 h by extrapolation | start from 256², U 0.99: 1441 it, 581 s (sharing the CPU with another run) | about 11× |

- **Stop on the iteration error, not on residuals.** `monitor.steady_decider` (LiveStop) stops once
  the estimated error left in every monitored value is below your tolerance. Residual targets say
  nothing about the answer's error (pitfall 10); case 2 iterated 1.5× longer than its 1e-6 answer
  needed. Pick the tolerance from the physics, for example a small fraction of the mesh change
  between levels. `monitor.steady_replay` tests a tolerance on a recorded history first.
- **Relax less: a larger pseudo step.** SIMPLEC (`consistent yes`) with U 0.97-0.99 instead of 0.9
  cut the iterations 3-9×. It is not monotone: on 128², 0.98 was best and 0.99 took 1.8× more
  iterations; on 256², 0.99 took the fewest and 0.995 1.8× more, in about the same wall time. Each
  iteration also costs more at first (the linear solvers do more sweeps), so compare wall time. A short trial ranks the settings well: a few hundred iterations
  per setting, side by side on separate cores, comparing how fast the probe changes shrink
  (iterations per decade over iterations 200-400). The ranking matched the full runs on 256² and
  got the best two right on 128².
- **Grid sequencing.** Start each mesh from the coarser mesh's answer (`foam.map_fields`). From rest
  the fine mesh spends most of its iterations spinning up the flow (512² from rest at U 0.99 had not
  converged after 2500 iterations; from 256² it took 1441). The gain was uneven on 256² (2.2× at
  U 0.97, none at 0.99), so measure. mapFields cannot use paths with a space.
- **What did not work:** pimpleFoam as a pseudo-transient march (implicit Euler, Courant 5-50, one
  outer corrector) diverged within ten steps. The segregated pressure-velocity coupling needs outer
  iterations with relaxation at large time steps, which brings back the cost.
- MPI adds its own factor on top (Parallel runs, above). None of this changes the converged answer,
  only how fast you reach it, so a speed setting needs a check of the answer, not a re-derivation.

Scripts and raw numbers: `dev/steady_speed/`.

## Stopping on a physical criterion: runTimeControl
Function object `runTimeControl` (`libs (utilityFunctionObjects)`) ends the run when conditions hold.
Condition types in v2606: `average` (a function-object result has settled: `functionObject`,
`fields`, `tolerance`, `window`), `equationInitialResidual`, `equationMaxIter`, `maxDuration`,
`minMax`, `minTimeStep`, `none`. Example: `$FOAM_TUTORIALS/incompressible/simpleFoam/simpleCar/system/runTimeControls`
(stops when the drag coefficient's running average settles). Which function-object results a
condition can watch (e.g. probes vs. `fieldMinMax`/`surfaceFieldValue`) must be checked in a short
test; log what works as an insight.

## Useful utilities
`blockMesh`, `checkMesh` (must report "Mesh OK."), `postProcess -func writeCellCentres -latestTime`,
`foamDictionary -entry <a/b> -set <v> <file>`, function objects `sample`/`singleGraph` (line
sampling), `wallShearStress`, `surfaceFieldValue`, `fieldMinMax`, `probes`, `streamFunction`
(`postProcess -func streamFunction`: point field in m²/s built from the face fluxes, zero on the
walls of a closed cavity, same sign convention as Ghia et al.; read it by point index).
