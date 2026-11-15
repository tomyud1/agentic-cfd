# gpucfd, compressible part: a GPU solver for gas with shock waves

Built 2026-11-15 as the foundation of gpucfd's own way to the detonation engine (ROADMAP, "Now: the
rotating detonation engine"). It stands next to the incompressible solver ([`DESIGN.md`](DESIGN.md)) and
shares nothing with it but the folder, the build script and the house rules: every part is verified
before it is trusted, rung by rung, and every rung stays in the test suite.

**In short.** The solver passes its five rungs. On the shock tube (case 18) its errors are 1.5 to 3
times smaller than OpenFOAM's on the same grids, in half the steps and 20 to 70 times less time. On
the shock round a corner (case 19) it meets the problem's three requirements on every grid up to
1600 x 1600, with the same accuracy of the flat shock as OpenFOAM and 50 times less time than
OpenFOAM's 8-core run on 800 x 800. Two things are worse than OpenFOAM and are said below: the
pressure ripple left by the start of the shock, and that the sharpest reconstruction breaks down in
the near-vacuum behind the corner.

Words used below:
- **Cell average**: the solver stores, per cell, the mean density, momentum and energy in that cell.
- **Flux**: what crosses a face between two cells per second. What leaves one cell enters the next,
  so mass, momentum and energy are conserved by construction.
- **Reconstruction**: guessing the gas state at a face from the cell averages around it.
- **Flux formula (Riemann solver)**: the rule that turns the two guesses on a face (from its left and
  from its right) into one flux.
- **Courant number**: how many cells the fastest signal crosses in one time step.

## Scope (first version)

- The Euler equations: gas with no viscosity and no heat conduction; one ideal gas with a constant
  ratio of specific heats.
- Uniform Cartesian grids in one, two or three dimensions (a direction with one cell is switched off).
- Explicit in time with one, two or three stages per step.
- Sides: periodic, frictionless wall, outflow (zero gradient), inflow with a fixed state. One side can
  be split into up to four parts with different types (the corner problem has a wall below the corner
  and an inflow above it on the same side).
- The whole time step runs on the GPU. A few numbers (largest signal speed, lowest density and
  pressure) come back to the CPU when the time step is chosen from the Courant number.

Not in scope yet: viscosity, several gases, heat capacities that change with temperature, burning,
stretched grids, bodies inside the grid. See "What the next step needs".

## Build and run

```
bash gpucfd/build.sh compressible          # only this library: build/libgcfdc.so (the other library is not touched)
bash gpucfd/run_ladder.sh compressible     # rungs 1 to 5, about five minutes (the corner to 400 x 400)
bash gpucfd/tests/run_corner.sh 1600       # the recorded corner run, about six minutes, log in tests/results/
bash dev/compare/run.sh shock_tube_sod gpucfd_run.py      # the shock tube next to OpenFOAM, writes gpucfd.json
```

```python
from gpucfd.euler import Euler
s = Euler((400, 400, 1), (1.0, 1.0, 1.0), gamma=1.4, sides={k: "wall" for k in ("x-", "x+", "y-", "y+")},
          recon="vanleer", flux="hybrid", precision="mixed")
s.segment("x-", {"y": (200, 400)}, ("supersonic", dict(rho=5.84, u=1416.0, p=3.0e6)))   # part of a side
s.set(rho=1.16, p=1.0e5)
st = s.run(4.5e-4, cfl=0.4)       # or s.run_fixed(dt, steps)
f = s.get()                       # rho, u, v, w, p, E as arrays (nz, ny, nx)
```

Files: `src/gcfdc.cu` (CUDA), `euler.py` (wrapper), `exact_riemann.py` (exact shock-tube solution),
`tests/euler_ref.py` (the independent NumPy version of the scheme), `tests/test_euler_*.py` (the rungs).

## The method in plain words

1. **Per cell five numbers**, each in its own array: density, momentum in x, y, z, total energy.
2. **One GPU kernel launch per stage.** Every cell reads its neighbours along each direction, turns
   them into density, velocity and pressure, and reconstructs the state on its two faces. Choices:
   first order (no reconstruction), three slope limiters (minmod, van Leer, MC), no limiter, and WENO5
   (a weighted blend of three parabolas that avoids the one crossing a jump). If a reconstructed face
   value has no gas or no pressure, that cell falls back to its own average.
3. **The flux formula** on each face. Four choices: Rusanov (simplest, smears most), HLL (two waves),
   HLLC (three waves: it keeps the border between two gases sharp) and a **hybrid**: HLLC everywhere,
   but HLL on faces that lie *along* a strong shock (a pressure jump of more than half between
   neighbours in another direction). Why: see rung 4.
4. **Both cells of a face compute its flux**, with the same numbers and the same code, so they get the
   same value. This costs arithmetic twice but needs no flux arrays and one launch per stage.
5. **Time**: the stages are combined as `U0 + b ((in - U0) + dt L(in))`. Three stages is the
   third-order "strong stability preserving" Runge-Kutta scheme, the default.
6. **Boundaries are applied inside the stencil**: beyond a wall the cell sees its own mirror image with
   the normal velocity reversed; beyond an inflow the fixed state. An inflow marked "supersonic" sends
   exactly the state's own flux through the side, which is right when the gas enters faster than sound.
7. **Three precisions.** Single, double, and **mixed** (the default): the state is stored in double,
   the fluxes are computed in single precision, and their differences are added up in double.

Stored per cell: 15 numbers (the state, and two stage copies) plus one byte: 61 bytes in single, 121
in mixed and double precision.

## The verification ladder

| Rung | What | Result |
|---|---|---|
| 1 | The kernels against an independent NumPy implementation of the same scheme (`test_euler_kernels`) | Pass. 192 combinations of boundary type, reconstruction, flux formula and stages, each in 1D, 2D and 3D: equal to 4e-16 in double, 1e-7 in single and mixed. A 1D problem along y and along z equals the one along x exactly |
| 2 | Smooth flows with exact solutions (`test_euler_smooth`) | Pass. A density wave carried by a stream: order 2.00 without limiter, 2.0 (van Leer, MC), 1.9 (minmod), 5.00 (WENO5), in 1D and 2D; time scheme 3.00 (three stages), 2.00 (two). A vortex carried across a box (nonlinear, 2D): 2.0 for every reconstruction |
| 3 | The shock tube of case 18 against its exact solution, 100 to 12800 cells (`test_euler_sod`) | Pass. Details and OpenFOAM's numbers below. One of the case's warning limits is not kept: density wiggle 5.3 % on 100 cells, limit 5 % (OpenFOAM: 5.3 % on 12800 cells) |
| 4 | A flat Mach 5.09 shock in a duct, and the corner problem of case 19, 100 x 100 to 1600 x 1600 (`test_euler_corner`) | Pass: the problem's three requirements on every grid, with two settings. Details below |
| 5 | Speed and memory (`test_euler_speed`) | Pass. 160 to 190 million cells x steps per second in mixed precision on 2D grids of 4 million cells and more, 130 million in 3D; 121 bytes per cell |

The NumPy reference is written separately and built differently (ghost cells, one flux per face,
velocity as a vector), so a shared mistake is unlikely. The exact shock-tube solution is this
solver's own (`exact_riemann.py`); scoring OpenFOAM's stored profiles with it reproduces the errors
OpenFOAM's report recorded to 1e-9, which checks both.

### Rung 2: what "order" means here

WENO5 is fifth order only for the linear density wave. For a real (nonlinear) flow the scheme is
second order with every reconstruction, because it reconstructs density, velocity and pressure from
cell averages and uses one flux per face. WENO5 then lowers the error (4 to 5 times on the vortex) but not
the order. This is the designed behaviour, not a defect.

### Rung 3: the shock tube next to OpenFOAM

Same problem, same grids, same reading rules (the case's `measure.py`), the same exact solution.
Plan written before the runs: `dev/compare/shock_tube_sod/plan.yaml`; record: `gpucfd.json` there.

**What each side was given.**
- OpenFOAM: `rhoCentralFoam` as the solver agent of case 18 set it up. Kurganov flux, van Leer
  limiter, its only time scheme (one stage), 1.8 steps per cell (Courant number 0.19), chosen by that
  agent from four time steps on 400 cells because larger steps ring. One CPU core. Not rerun.
- gpucfd, tuned: three stages; reconstruction, flux formula and Courant number chosen by a budget of
  twelve runs on 400 cells (van Leer, MC, WENO5; HLL, HLLC; Courant 0.4, 0.8). The rule, fixed
  before: lowest sum of the three errors among the runs that keep the case's ripple and wiggle limits.
  It chose WENO5 with HLLC at Courant 0.4. (At 0.8 the errors are 4 % larger in half the steps; the
  rule did not weigh cost.)
- gpucfd, like for like: van Leer and HLL, the closest to OpenFOAM's limiter and flux. Not tuned.

Mean error against the exact cell averages, as a share of the left density, the speed between the
waves and the left pressure; times are wall time from creating the solver to the final field:

| Cells | OpenFOAM: density / velocity / pressure | steps | time | gpucfd tuned | steps | time | gpucfd like for like |
|---|---|---|---|---|---|---|---|
| 100 | 0.45 / 0.82 / 0.34 % | 180 | 0.40 s | 0.30 / 0.59 / 0.23 % | 90 | 0.007 s | 0.44 / 0.87 / 0.34 % |
| 400 | 0.127 / 0.230 / 0.092 % | 720 | 0.71 s | 0.078 / 0.160 / 0.059 % | 350 | 0.03 s | 0.134 / 0.241 / 0.089 % |
| 1600 | 0.043 / 0.063 / 0.034 % | 2880 | 2.3 s | 0.022 / 0.031 / 0.014 % | 1390 | 0.10 s | 0.043 / 0.051 / 0.022 % |
| 12800 | 0.0085 / 0.0117 / 0.0068 % | 23040 | 84 s | 0.0041 / 0.0053 / 0.0021 % | 11090 | 1.2 s | 0.0090 / 0.0075 / 0.0029 % |

| On every grid | OpenFOAM | gpucfd tuned | gpucfd like for like |
|---|---|---|---|
| Order with which the error falls (density / velocity / pressure, fitted over all grids) | 0.81 / 0.85 / 0.78 | 0.89 / 0.95 / 0.96 | 0.81 / 0.98 / 0.98 |
| Width of the shock (10 to 90 % of the jump) | 1.8 to 2.2 cells | 1.8 to 2.1 cells | 2.3 to 2.5 cells |
| Width of the border between the two gases, 100 and 12800 cells | 4.4 and 6.3 cells | 3.0 and 6.3 cells | 4.1 and 16.9 cells |
| Ripple of the pressure between fan and shock, finest grid | 0.6 % | 0.008 % | below 0.005 % |
| Mass and energy, largest change in any step | 4e-14, 5e-14 | 1e-15 (mixed and double) | 1e-15 |
| Momentum against the push of the two end walls | 1e-13 | 5e-8 in mixed, 5e-16 in double | the same |

What this says:
- **Tuned, gpucfd is 1.5 times (coarse) to 2 to 3 times (fine) more accurate on the same grid**, and
  20 to 70 times faster. In one dimension the GPU is idle most of the time: a step costs about 0.1 ms
  whatever the grid, so these times measure launch overhead, not the card.
- **Like for like, the density error is the same as OpenFOAM's** (ratio 0.94 to 1.04). So the two
  codes agree where they use the same ingredients, and the gain in density comes from WENO5 and HLLC.
- **Velocity and pressure converge faster in gpucfd even like for like** (order 0.98 against 0.85 and
  0.78). Case 18 left open why OpenFOAM's error in the expansion falls slowly; a second solver with
  the same limiter does not show it, so it belongs to `rhoCentralFoam`, not to the limiter.
- **The ringing of OpenFOAM is not only its one-stage time step.** gpucfd with ONE stage, van Leer and
  HLL ripples 0.03 % at Courant 0.19 and 0.13 % at 0.39; OpenFOAM 0.6 % and 3.1 %. With three stages
  gpucfd ripples 0.01 % up to Courant 1.0. Case 18's insight 5 blamed the time step alone.
- **Expected and found** (plan): errors 1.5 to 3 times smaller: found 1.4 to 3.2. Like for like about
  equal: found for the density; pressure up to 2.4 times better. "About a quarter of OpenFOAM's
  steps": found half (the rule chose Courant 0.4). 12800 cells under 2 s: 1.2 s. Momentum to 1e-9 in
  mixed precision: found 5e-8 (see "Precision"). End cells undisturbed to 1e-10: found 3e-7 in mixed
  precision with WENO5, exactly 0 in double.

Every reconstruction and flux formula on 400 cells (Courant 0.8, double), density error and widths:

| Reconstruction | Rusanov | HLL | HLLC | Border width with HLLC |
|---|---|---|---|---|
| first order | 1.07 % | 0.79 % | 0.76 % | 19.8 cells |
| minmod | 0.27 % | 0.21 % | 0.21 % | 8.3 |
| van Leer | 0.17 % | 0.14 % | 0.13 % | 5.8 |
| MC | 0.16 % | 0.12 % | 0.11 % | 5.0 |
| WENO5 | 0.097 % | 0.081 % | 0.080 % | 3.4 |

### Rung 4: a Mach 5.09 shock, flat and round a corner

Plan with pass rules and expectations, written before the runs:
`dev/compare/shock_diffraction_corner/plan.yaml`. Outputs read by case 19's own rules
(`cases/shock_diffraction_corner/measure.py`), so they stand next to OpenFOAM's.

**Which flux formula keeps a flat strong shock flat.** A flat shock in a duct of 400 x 100 cells, the
still air disturbed by noise of one part in 10 000:

| Flux formula | van Leer: shock off a straight line | WENO5 | Pressure variation along the shock |
|---|---|---|---|
| HLLC | 0.50 cells | 0.79 cells | 6 % |
| hybrid (HLLC, HLL along the shock) | 0.0001 | 0.0006 | 0.002 to 0.007 % |
| HLL | 0.0001 | 0.0004 | 0.001 to 0.005 % |
| Rusanov | 0.0001 | 0.0034 | 0.002 to 0.05 % |

Plain HLLC breaks the shock up. This is the known "carbuncle" of flux formulas that keep the middle
wave: along the shock nothing damps a disturbance. The hybrid keeps HLLC's sharp border everywhere
else and is as straight as HLL. **The hybrid is the solver's default flux.**

**Which settings survive the corner** (200 x 200). Behind the corner the gas expands to 5 % of the
still air's density. WENO5 with HLL, HLLC or the hybrid produces a cell without pressure after about
90 steps and stops; WENO5 with Rusanov runs, and van Leer runs with every flux. The plan's rule
(sharpest that is straight and survives) chose **WENO5 with Rusanov**. Because that pairs the sharpest
reconstruction with the most smearing flux, **van Leer with the hybrid** was run as a second setting.

**The problem's requirements** (mixed precision, Courant number 0.4 summed over both directions):

| Cells per side | Setting | Speed of the flat shock | State behind it (p / rho / u) | Mass, energy against the inflow | Flat part off a straight line | Steps | Time |
|---|---|---|---|---|---|---|---|
| 100 | WENO5, Rusanov | +0.056 % | -0.19 / -0.31 / +0.08 % | 6e-8 | 3e-6 cells | 477 | 0.1 s |
| 200 | | -0.099 % | -0.10 / -0.17 / +0.04 % | 6e-8 | 5e-6 | 933 | 0.4 s |
| 400 | | +0.017 % | -0.05 / -0.08 / +0.02 % | 6e-8 | 3e-5 | 1840 | 2.3 s |
| 800 | | +0.005 % | -0.02 / -0.04 / +0.01 % | 6e-8 | 4e-3 | 3680 | 16 s |
| 1600 | | +0.0003 % | -0.011 / -0.019 / +0.005 % | 6e-8 | 8e-3 | 7308 | 129 s |
| 100 | van Leer, hybrid | +0.065 % | -0.28 / -0.52 / +0.12 % | 6e-8 | 3e-6 | 454 | 0.1 s |
| 200 | | -0.001 % | -0.13 / -0.23 / +0.06 % | 6e-8 | 3e-5 | 904 | 0.3 s |
| 400 | | +0.025 % | -0.06 / -0.11 / +0.03 % | 6e-8 | 7e-5 | 1807 | 2.0 s |
| 800 | | +0.004 % | -0.03 / -0.05 / +0.01 % | 6e-8 | 7e-5 | 3620 | 15 s |
| 1600 | | -0.0005 % | -0.015 / -0.026 / +0.006 % | 6e-8 | 6e-5 | 7241 | 114 s |
| required | | within 0.5 % (finest) | within 1 % (finest) | within 1e-4 | within 1 cell | | |

All three requirements are met on every grid with both settings, and no wave has reached the floor
or the right side. A picture of the density on the finest grid: `tests/results/euler_corner_density.png`
(and `..._vanleer_hybrid.png`).

**Next to OpenFOAM** (case 19: `rhoCentralFoam`, Kurganov, van Leer, one stage, Courant 0.2 along x,
as its solver agent set it up; one core up to 400 x 400, eight cores on 800 x 800; its finest grid is
800 x 800). gpucfd: the two settings above, RTX 4070.

| 800 x 800 | OpenFOAM | gpucfd WENO5, Rusanov | gpucfd van Leer, hybrid |
|---|---|---|---|
| Speed of the flat shock | +0.006 % | +0.005 % | +0.004 % |
| Pressure, density behind it | -0.020, -0.036 % | -0.023, -0.038 % | -0.030, -0.053 % |
| Pressure ripple behind the flat shock | 2.0 % | 4.4 % | 2.7 % |
| How far the bent shock has run down the step face | 0.3373 m | 0.3375 m | 0.3373 m |
| The same picture at two times: difference along the step face, along the corner's line | 3.2 %, 0.63 % | 3.8 %, 0.51 % | 6.1 %, 0.43 % |
| Lowest density in the field | 0.131 kg/m3 | 0.070 | 0.062 |
| Steps | 4080 | 3680 | 3620 |
| Time | 742 s on 8 cores | 16 s | 15 s |

| Time per grid | 100 | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|---|
| OpenFOAM | 3.3 s (1 core) | 21 s (1 core) | 187 s (1 core) | 742 s (8 cores) | not run |
| gpucfd (van Leer, hybrid) | 0.1 s | 0.3 s | 2.0 s | 15 s | 114 s |

What this says:
- **The flat shock is the same in both codes** (speed and state agree to a few thousandths of a per
  cent on each grid), and the bent shock on the step face lands within 0.2 mm on 800 x 800.
- **gpucfd is about 50 times faster** than OpenFOAM's 8-core run on 800 x 800 and about 90 times
  faster than its one-core run on 400 x 400, with about the same number of steps. (Times of the
  recorded run, no OpenFOAM process on the CPU; an earlier run of the same grids was 15 % faster.)
- **gpucfd's pressure ripple behind the shock is larger** (2.7 to 4.4 % against 2.0 %). It is the
  trace of the start: the shock begins as a sharp jump on the inlet, and the two waves this leaves are
  a few cells wide and travel with the gas. A scheme that smears less keeps them taller. It falls
  slowly with the grid (3.3 to 2.3 % from 100 to 1600 cells, van Leer). Not a requirement; not fixed.
- **gpucfd resolves a deeper expansion behind the corner** (lowest density 0.06 against 0.13 kg/m3).
  Whether 0.06 is right is not known: there is no exact value, and it has not settled with the grid
  for WENO5 (0.05 to 0.07), while van Leer with the hybrid gives 0.062 on every grid from 200 on.
- **Expected and found** (plan): speed within 0.1 % and state within 0.2 % from 400 x 400 on: found.
  Mass and energy to 1e-12: found 6e-8, because in mixed precision the inflow flux is formed in single
  precision (requirement: 1e-4). HLLC alone and WENO5 may fail: both did, as described. The two times
  differ by a few per cent on 400 x 400: found 6 to 8 %, 3 % on 1600 x 1600. 1600 x 1600 in about one
  minute: found 114 to 129 s (the speed was overestimated by two).

### Rung 5: speed and memory

Fixed time step, three stages, best of three timings, nothing else on the GPU, no OpenFOAM process on
the CPU. "Cells x steps per second" counts whole steps (three stages each).

| Grid | Setting | Single | Mixed | Double |
|---|---|---|---|---|
| 12 800 (1D) | van Leer, hybrid | 250 million | 198 | 39 |
| 512 x 512 | van Leer, hybrid | 203 | 184 | 39 |
| 2048 x 2048 | van Leer, hybrid | 223 | 177 | 39 |
| 4096 x 4096 (16.8 million cells) | van Leer, hybrid | 214 | 192 | 40 |
| 4096 x 4096 | WENO5, Rusanov | 181 | 170 | 8 |
| 128^3 | van Leer, hybrid | 145 | 131 | 26 |
| 256^3 (16.8 million cells) | van Leer, hybrid | 145 | 134 | 28 |

- Mixed precision costs 5 to 20 % against single; double is about 5 times slower than mixed with van
  Leer and about 20 times slower with WENO5. Two runs of this table differed by up to 10 %. The card's double arithmetic is the limit, not its memory.
- About 5.5 ns per cell and step in 2D and 7.5 ns in 3D (mixed). For scale: OpenFOAM's
  `rhoCentralFoam` took 300 ns per cell and step on one core in case 18 and 280 ns on eight cores in
  case 19 (wall time), with one stage per step.
- Small grids are limited by launches: 0.07 to 0.1 ms per step at any size below about 20 000 cells.
- Memory: 61 bytes per cell in single, 121 in mixed and double. The 12 GB card holds about 90
  million cells in mixed precision (9500 x 9500, or 450^3). The corner on 1600 x 1600 used 296 MB;
  time, not memory, set that grid: 3200 x 3200 would take about 15 minutes.

## Precision: single, mixed, double

Tested on rungs 2, 3 and 5. **Decision: mixed precision is the default; single is not recommended.**

- **Single precision loses mass**: about 3e-9 of the mass per step on a fine smooth grid (1e-4 after
  24 000 steps, 2e-7 on the shock tube with 3200 cells). It happens with every scheme, also the
  simplest, and without fused multiply-add; one step alone shows no bias. **The cause was not found.**
- **Mixed precision** keeps mass and energy to 1e-15 like double, at 80 to 85 % of single's speed and
  twice its memory. Its error equals double's down to about 1e-6 of the solution (random noise of
  the single-precision fluxes), far below the error of any grid with a shock.
- **What mixed cannot do**: forces on boundaries are formed in single precision. The momentum follows
  the exact push of the walls to 5e-8, the inflow of the corner problem is exact to 6e-8, and with
  WENO5 noise of 1e-7 reaches cells that should be untouched. A limit of 1e-11 on these needs double.
- **Double** is for verification: 5 to 20 times slower on this card.

## What the ladder caught while building

Each would have given plausible wrong results.
1. In single precision the stage weights 1/3 and 2/3 add up to 1 + 3e-8, so every step created that
   share of mass (rung 2). The stage is now written `U0 + b ((in - U0) + dt L)`.
2. Single precision loses mass even so (above): led to mixed precision.
3. Mixed precision first lost energy at 1e-9: each cell rounded its own flux difference in single
   precision. The differences are now formed in double (rung 3).
4. The HLLC formula multiplied and divided the wall pressure by the same wave speed, so a wall at
   rest did not feel exactly the pressure. Rewritten so that it does, in any precision (rung 3).
5. With WENO5 the two cells of a face got fluxes that differed in the last bit: the compiler fused
   multiply-adds differently in the three copies of the reconstruction. Mass leaked at 1e-10. The
   library is built without fused multiply-add (`--fmad=false`, rung 3).
6. The time step ignored the gas that is about to enter: at the start of the corner problem the still
   air set a step six times too large for the inflow. The fixed states now limit the step (rung 4).
   Before the fix the start ripple was 6.8 % instead of 4.9 %.

## Limits

- One gas, constant ratio of specific heats, no viscosity, no heat conduction, uniform grids, no bodies.
- **No positivity limiter.** WENO5 with HLL, HLLC or the hybrid stops in the near-vacuum behind the
  corner. Until one is built, the safe settings for strong expansions are van Leer with any flux, or
  WENO5 with Rusanov.
- The pressure ripple left by an impulsively started shock is 2 to 5 %, larger than OpenFOAM's 2 %.
- "inflow" is a fixed state beyond the side. It is exact for gas entering faster than sound; it is
  not a proper boundary condition for a slow inflow.
- At most four parts per side and eight fixed states.
- The hybrid flux's threshold (pressure jump above one half) and the 2D Courant number (0.4) were
  chosen once and not tuned. Courant 0.8 would halve the corner's times if it is stable there.
- Small grids pay about 0.03 ms per kernel launch; the launches are not recorded as CUDA graphs.
- Single precision loses mass, cause unknown.
- The comparison times: gpucfd on an RTX 4070 against OpenFOAM on 1 or 8 cores of the same PC, each
  as its agent set it up. OpenFOAM's runs were not tuned for speed and include its start-up.

## What the next step needs: several gases, real heat capacities, stiff burning

The target is a detonation: a shock that ignites the gas behind it (steps 3 and 4 of the ladder,
`results/rde/plan.html`). Three things must be added. How the present layout allows each:

1. **Several species.** The state is a set of separate arrays, one per conserved quantity, and the
   kernels loop over "5". Species are more arrays of the same kind: the density of each species
   (rho times its mass fraction), carried by the same mass flux the flux formula already computes
   (upwind with the middle wave of HLLC). The loops over 5 become loops over 5 + the number of
   species; the boundary rules, the stencil, the stage formula and the mixed-precision rule stay as
   they are. Memory: 3 x (5 + species) numbers per cell; hydrogen-air with 9 species is 336 bytes per
   cell in mixed precision, so the card holds about 30 million cells.
2. **Heat capacities that change with temperature.** Today the ratio of specific heats appears in
   two places only: where a cell's stored numbers are turned into pressure (`load`), and in the flux
   (`physflux`, `riemann`: energy from pressure, speed of sound). With real gases the first becomes a
   small Newton iteration per cell for the temperature from the energy (polynomial heat capacities),
   and the second uses each state's own ratio. A known trap to plan for: at a border between gases
   with different ratios a conservative scheme makes pressure wiggles; the usual cures (the
   "double flux" method, or carrying one extra quantity) should be chosen and tested on a rung of its
   own before any burning is switched on.
3. **Stiff chemistry in every cell.** Burning couples nothing between cells, so it is a kernel of its
   own that advances each cell's species and temperature over the flow's time step, between two half
   flow steps (splitting). One thread per cell solves a small stiff system (10 unknowns for hydrogen)
   with an implicit method and its own sub-steps. This fits the layout: it reads and writes the same
   arrays. Two costs must be measured before sizing a case: this chemistry needs double precision
   (rates depend exponentially on temperature), which this card runs 5 to 20 times slower, and cells in
   the reaction zone need many sub-steps while the rest need none, so the work per cell is uneven.
4. **Also needed first:** a positivity limiter (the corner showed it), and a decision on the reaction
   zone's resolution: a shock takes 2 cells here, the border between gases 3 to 6.

Rungs for that work, each with a known answer: mixing of two gases at rest and a moving border
between them (no pressure wiggles); ignition at constant volume against a reference kinetics code
(case 20); the speed and the structure of a flat detonation against the Chapman-Jouguet and ZND
theory (case 21).

## Open

- Why single precision loses mass.
- The start ripple behind the shock (a smoother start, or more smearing in the first steps).
- 3200 x 3200 and finer on the corner (the card holds them; about 15 minutes for 3200 x 3200).
- `cases/briefs.yaml` (outside this work's limits) labels the shock tube's speed rows by cells and
  steps; gpucfd's step counts differ, so its runs carry labels of their own. A label by cells only,
  and a `record:` line, would put the two tools side by side on the one-page report.
