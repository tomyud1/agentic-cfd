# Changelog

What changed in the project and when, newest first, in plain language. Read this to catch up on
recent work. Details live in the linked commits, reports and [`gpucfd/DESIGN.md`](gpucfd/DESIGN.md).

Each entry says **what** changed, **why it matters**, **how it was verified**, the key numbers, and
the commit. Open problems are marked **Open**. Work not yet committed goes under "In progress", and
moves to a dated entry when it is committed.

## In progress

- **Faster, self-stopping grid studies for steady flows** (`gpucfd/steady.py`,
  `gpucfd/tests/test_steady.py`, `iteration_error` in `cfdlib/verification.py`; not committed yet).
  - **Not a direct steady solver:** gpucfd still marches in time until the flow settles. Two changes
    make that cheaper:
    - **Grid sequencing:** each finer grid starts from the coarser grid's solution, interpolated, so
      it doesn't redo the whole start-up from rest.
    - **An error-based stop:** the run ends once the estimated error still left from marching is
      below 1% of the grid's own discretisation error. It no longer runs until the forces stop
      changing in the sixth digit, which is far tighter than the grid's accuracy.
  - **Logging and stopping refinement:** every time chunk logs the forces with that error estimate,
    so runs can be watched live. A study can stop adding grids once the extrapolated answer has
    settled.
  - **A known limit, handled:** a slowly decaying part of the solution can hide under a faster one,
    and then the estimate is up to 11 times too small (synthetic test). The fix: finer grids reuse
    the decay rate the coarser grid measured, and the first grid gets a minimum run time set by
    physics.
  - **Status:** the estimator tests pass. The GPU tests (rung 11: Couette 16, 32 and 64 cells, both
    sequenced and from rest) and the DFG 2D-1 comparison have not run yet.
  - **Motivation:** with bodies, gpucfd must use the explicit scheme (IMEX is unstable there). This
    made fine-grid steady runs slow: DFG 2D-1 at 35 cells per diameter took 400 GPU-seconds, mostly
    spent settling.
  - **Next:** connect gpucfd to the agent through cfdlib, starting with case 4 (DFG 2D-2), against
    OpenFOAM.

## 2026-11-12

### Case backlog (`cases/BACKLOG.md`) and project docs, `ba626b8` and later
- **What:**
  - [`cases/BACKLOG.md`](cases/BACKLOG.md) lists the next problems in order: heat transfer (heated
    pipe, buoyancy cavity), the first RANS case (NASA TMR turbulent flat plate), harder laminar
    cases that gpucfd can also run, and TMR turbulence cases.
  - It also proposes adopting manufactured solutions (MMS) for verification.
  - The changelog, AGENTS.md (the starting point for every coding agent) and the changelog rule in
    CLAUDE.md came in `ba626b8`.
  - The README now shows only the current state. gpucfd has its own short section next to the
    results, and the roadmap points to the backlog. A new rule in CLAUDE.md says to replace README
    text instead of adding to it.
- **Why it matters:** each next case adds one new capability. The sources (NASA TMR, and Luminary's
  V&V catalogue, suggested by Dhanush from Prof. Frankel's lab) come with grid families and
  reference results, and some with a commercial result to compare against.

### gpucfd: bodies inside the flow (rung 10), `a043ae6`
- **What:** shapes defined by a signed distance (cylinders, any shape you can describe that way) are
  placed inside the grid with a sharp-interface (ghost-cell) immersed boundary. Drag, lift and
  surface pressure come from the exact discrete momentum balance.
- **Why it matters:** gpucfd can now run flow around objects, which is the core of the DFG cylinder
  benchmarks and of most external-flow problems.
- **Verified:**
  - Exact solutions: flow between two immersed rotating cylinders (velocity order 2.1–2.5), and a
    channel tilted to the grid (order 2.2).
  - 3D extrusions give the same answer as 2D.
  - The forces don't depend on the box they are measured over (to 2e-18).
  - Runs without bodies are bit-identical to the previous build.
- **DFG 2D-1 benchmark (Re 20)**, with 35 cells per diameter: drag −0.65%, pressure difference
  −0.23%, lift +15%. On the coarser grid (17.6 cells per diameter) lift was 4.3 times the reference,
  so lift converges, but needs fine grids. Lift is only 0.2% of the drag in this case.
- **Open:**
  - The implicit (IMEX) scheme is unstable with bodies, so bodies use the explicit scheme only.
  - The largest pressure error, in the cells right next to a body, does not converge cleanly.

### gpucfd: inflow and outflow boundaries (rung 9), `a043ae6`
- **What:**
  - Any side can be an inflow with any velocity profile.
  - Any side can be an outflow: pressure zero and zero velocity gradient, the same as OpenFOAM's
    usual outlet.
- **Why it matters:** channels and flows past objects need an inlet and an outlet. Before this,
  gpucfd only handled closed or periodic boxes.
- **Verified:**
  - Poiseuille channel flowing in every direction, in 2D and 3D: second order.
  - Kovasznay flow, an exact Navier–Stokes solution with convection: second order.
  - The flow out equals the flow in to round-off.
  - Wall and periodic runs are bit-identical to the previous build.

### Project direction for agents (`CLAUDE.md`), `b702d6a`
- **What:** [`CLAUDE.md`](CLAUDE.md) states the goal (automate the CFD engineer's work and build
  the agent's tools), the bottlenecks (compute, usage budget, expertise) and the working rules. It is
  loaded by every Claude Code session in the repository.

### gpucfd: speed work, round 2, `d9d7090`
- **What:**
  - Every solver loop is now decided on the GPU, and a whole time step runs as one recorded CUDA
    graph, so the CPU no longer sits in the loop.
  - Fused kernels, and a 1/h precomputation (the RTX 4070 does double precision 64 times slower than
    single).
- **Why it matters:** most of each time step had been spent waiting on the CPU (about 115 round
  trips per step under WSL).
- **Numbers:**
  - Cavity 128²: 43 s → 5 s (OpenFOAM on the same case: 46 s).
  - 3D flow: 47× (64³) and 235× (128³) faster than OpenFOAM on the same grid.
- **Verified:** every speed-only change gives bit-identical results (`tests/same_as_build.py`).

### gpucfd: speed work, round 1, `c305b60`
- **What:**
  - An IMEX time scheme: convection explicit (RK3), diffusion implicit (Crank–Nicolson). The time
    step is then limited by the flow speed only.
  - CUDA graphs for the multigrid cycles.
  - Deterministic sums, so runs are bit-for-bit reproducible.
- **Numbers:** cavity 128²: 396 s → 42.8 s, with the same answer.
- **Bugs caught by the checks:**
  - The first IMEX version left a small error near walls in the steady state.
  - Atomic additions made identical runs differ at the 4e-16 level.

### gpucfd: walls, body force and the lid-driven cavity (rung 7), `698b197`
- **What:** no-slip walls (including a moving lid) and a constant body force.
- **Verified:**
  - Couette flow is exact to 1e-16, and Poiseuille flow converges at order 2.00.
  - On the Re 100 cavity, gpucfd's and OpenFOAM's solutions converge to each other at second order.
- **Found:** on that small steady case gpucfd was 9× slower than OpenFOAM, which motivated both
  rounds of speed work.

### gpucfd: like-for-like comparison with OpenFOAM, `aa3d28b`
- **What:** the same 3D flow (decaying ABC flow, an exact solution), the same grids, and each
  solver tuned by the same rule.
- **Numbers:** gpucfd was 4–168× faster than OpenFOAM on 8 CPU cores (later 18–235×, after the
  speed work), with a smaller error on every grid.
- **Why OpenFOAM's GPU plugins are capped:** 44% of OpenFOAM's time is outside its linear solvers,
  where a GPU plugin can't help.
- **Limits:** periodic boxes only at that point, one laminar flow, one machine.

## 2026-11-11

### gpucfd: first verified stage, `1d0b29b`
- **What:** a solver that keeps the whole time step on the GPU:
  - pressure solved with geometric multigrid, alone or as the preconditioner of conjugate gradients;
  - mixed precision (double outer iteration, single-precision multigrid);
  - Navier–Stokes on periodic boxes, on a staggered grid with RK3.
- **Why:** case 5 showed that offloading only OpenFOAM's linear solver is capped near 1.5×.
- **Verified:** a ladder of checks, each built on the last (rungs 0–6):
  - operators equal an independent numpy reference;
  - order 2 in space and 3 in time;
  - multigrid iterations don't grow with grid size;
  - mixed precision gives the same answer as double.

### Case 5: 3D cylinder (DFG 3D-1Z, Re 20) and GPU vs CPU study, `86e95aa`
- **Physics report:**
  - FAIL: lift is 1.7% off, outside its band. Drag is within 0.08%.
  - A second mesh family showed the lift depends on how the cells are graded around the cylinder.
  - [Report](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_3d_re20/report.html)
- **GPU study:**
  - OpenFOAM with the OGL plugin on the RTX 4070 gives the same answer as the CPU.
  - The GPU breaks even at about 2 million cells and is 1.13× faster at 4.75 million.
  - It is capped near 1.5× because the pressure solve is only a third of each iteration.
  - [Report](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_3d_re20/gpu_study.html)
- **Workflow:**
  - Study-report renderer and high-resolution profiling in `cfdlib`.
  - Guidelines: OGL settings, check the Amdahl bound first, compare against the best CPU setup, and
    use a second mesh family for sensitive outputs.

### Repository published, `d7f61bc`
- Pushed to GitHub ([tomyud1/agentic-cfd](https://github.com/tomyud1/agentic-cfd)) with GitHub
  Pages for the reports. This first commit contains everything from 2026-11-10 and 2026-11-11 below.

### Case 4: cylinder in a channel (DFG 2D-2, Re 100, vortex shedding)
- **Result:** WARN. Drag, lift and Strouhal number are all inside their bands (St 0.3015 vs 0.3019).
  The report includes an MPI scaling study.
  [Report](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_re100/report.html)
- **Found:** OpenFOAM's PISO algorithm was only first-order accurate in time on this flow, which
  made the lift 10% wrong. Every unsteady nonlinear problem now needs a time-step study in its
  pre-flight.
- **Workflow:** signal analysis (`cfdlib/signals.py`), linear-solver iteration counts in the
  performance data, more checks in the report contract. 16 insights logged.

### Case 3: Stokes' first problem
- **Result:** PASS in 24 minutes. Velocity within 0.002% of U, second-order convergence.
  [Report](https://tomyud1.github.io/agentic-cfd/results/stokes_first_problem/report.html)
- **Workflow:**
  - Performance measurement for every run (`cfdlib/perf.py`) and an optional performance appendix.
  - Fixes to the round-off floor in run monitoring and to the insight log.
  - 11 insights logged.
- **Found:** a subagent could not load the project skill by name, so launch prompts now give the
  path to `SKILL.md`.

### Workflow fixes after case 2
- **What:**
  - Live, timestamped insight log (`cfdlib/insights.py`).
  - A figure checker that refuses overlapping text.
  - Each decision is tagged with a kind.
  - LaTeX checks.
  - Structured reproducibility information.
  - A visual review of every report before it counts as done.

## 2026-11-10

### Case 2: lid-driven cavity, Re 100 (first test on a fresh agent)
- **Result:** WARN. Velocity profiles are within 0.009 U of Ghia et al. (1982). The gap doesn't
  shrink with refinement, and the agent traced it to the benchmark's own mesh. 15 insights logged.
  [Report](https://tomyud1.github.io/agentic-cfd/results/lid_driven_cavity_re100/report.html)

### The three-layer design
- **What:**
  - **Tools:** `cfdlib`. Code computes every number.
  - **Guidelines:** the `agentic-cfd` skill, an OpenFOAM primer and the report contract.
  - **Worked examples:** the solved cases.
  - A contract checker refuses to render a report if any OpenFOAM setting is undocumented.
- **Why:** so a fresh agent can solve a problem it has never seen.

### Case 1: plane Poiseuille flow
- **Result:** PASS. Velocity profile within 0.0061% of the exact solution, second-order convergence.
  [Report](https://tomyud1.github.io/agentic-cfd/results/plane_poiseuille/report.html)

### Setup
- WSL2 with Ubuntu 24.04, ESI OpenFOAM v2606 and a Python environment, on a Ryzen 7 7800X3D with an
  RTX 4070.
- **Found:** the cavity case diverged on fine meshes with GAMG at pressure `relTol 0.05`.
  `relTol 0` fixes it.
