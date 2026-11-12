# Changelog

What changed in the project and when, newest first, in plain language. Read this to catch up on
recent work. Details live in the linked commits, reports and [`gpucfd/DESIGN.md`](gpucfd/DESIGN.md).

Each entry says **what** changed, **why it matters**, **how it was verified**, the key numbers, and
the commit. Open problems are marked **Open**. Work not yet committed goes under "In progress", and
moves to a dated entry when it is committed.

## In progress

Nothing uncommitted. Next, in this order:
1. Stretched grids for gpucfd: case 4 showed they decide whether it can beat OpenFOAM on bodies.
   Success: case 4 drag within 0.25% of the benchmark in less wall time than OpenFOAM's study.
2. Then the IMEX fix for bodies, a 3D cylinder, and more OpenFOAM problem types
   ([cases/BACKLOG.md](cases/BACKLOG.md)).

## 2026-11-12

### Peek gives the agent numbers; a live page shows the human engineer the figure, commit hash to be filled in
- **What:**
  - `peek` now gives the history as numbers by default. Residuals, Courant number and each
    monitored series appear as min / mean / max over 8 equal time windows (`--windows N`).
  - `--csv DIR` writes the selected series at full resolution for the agent's own analysis.
  - The figure is now opt-in (`--plot`).
  - New `python -m cfdlib.monitor watch <case> --every 60` keeps `<case>/peek.html` up to date
    until the run ends. The page holds the figure plus the text and reloads itself in a browser, so
    the human engineer can follow the run.
- **Why it matters:** an agent reads exact numbers more reliably than a picture, and a table costs
  fewer tokens than an image. A figure is still the fastest way for a person to see a run. The
  agent can still peek whenever it judges it useful.
- **How it was verified:**
  - On case 4's `live_M1`, the table shows the shedding develop in numbers: the lift amplitude grows
    from 4e-5 to 5.2e-4 by t ≈ 3.9 s and then holds, and the residuals level off at the same time.
  - On a running scratch run, the watch page was refreshed while the run went (state "running") and
    left at "finished" when it ended.
  - The CSV export holds every time step.

### Peek: the agent can look into a running simulation whenever it chooses (`57c45a3`)
- **What:** `python -m cfdlib.monitor peek <case dir>` gives a snapshot of an OpenFOAM run from its
  first step to now:
  - the state (running, finished, crashed) and progress, with the time left;
  - health so far;
  - the live stop's last verdict;
  - every monitored series in one line each.

  It also writes a figure of the whole history: residuals, Courant number, forces, probes and other
  postProcessing output. `--files` and `--from` narrow it down. SKILL.md stage 4 says how to use it.
- **Why it matters:** the automatic live stop runs in Python beside the solver and costs no agent
  tokens. Peek is the other half: the agent decides when to look, after a minute or after hours.
  Between looks nothing runs, so a long simulation does not fill the agent's session. Everything
  is read from files the run writes anyway, so it works on any OpenFOAM case.
- **How it was verified:**
  - A scratch run was peeked twice while running and once after the end. It was seen as running,
    and the time-left estimate was right: 15 s left at 8 s; the run ended at 23 s.
  - A finished case: 1.6 s for an 8 MB log.
  - A third live-stop run, `live_M2`, stopped at 17.74 s, where the replay predicted 17.70 s. It took
    90 s instead of 97 s, with numbers within 9e-5 of the full run.
  - The live stop's progress lines now only print when a verdict changes.
- **Open:** gpucfd runs have no peek yet; they run inside Python and report through
  `cfdlib.gpu.run_transient`'s log lines.

### OpenFOAM runs stop live on their physical check (`5a064ef`)
- **What:** `cfdlib/monitor.py` gained `LiveStop`. While OpenFOAM runs, it reads the run's own output
  every 10 s and applies the stopping rule written down before the run. Once the rule holds, it
  triggers OpenFOAM's `abort` function object, which writes the current step and ends the run
  normally, in serial and parallel. The planned end time becomes only a maximum. `monitor.judge`
  and `monitor.settled` turn per-cycle values into a verdict:
  - **settled:** below tolerance, with no drift left;
  - **floor:** the changes stopped shrinking above tolerance; the run stops with a warning.

  Case 4 uses it in `run.py live` and checks it in `run.py live_check`. SKILL.md stage 4 now says to
  stop every run this way.
- **Why it matters:** a fixed end time is a guess. Too short gives a wrong answer (gpucfd G1 was 16%
  low after 3 s). Too long wastes the most time on the finest, most expensive grid.
- **How it was verified:**
  - **Replay:** the rule was first replayed on case 4's recorded OpenFOAM histories.
    - It found that **M1 never meets the periodicity rule**: its drag peaks jitter by about 7e-4 per
      cycle at 9, 12 and 15 s alike, above the 5e-4 tolerance. A plain "change below tolerance" rule
      would never stop it, hence the floor state.
    - M2, M3 and M4 meet the rule after 2.7, 2.35 and 2.0 s of their 3 s.
  - **Live runs:** two live runs passed the rules set before them.
    - **`live_M1`** (serial): stopped at 7.85 s at its floor, in 30 s instead of 56 s.
    - **`live_M3`** (4 ranks): stopped at 20.33 s with everything settled, in 211 s instead of 285 s.
      `reconstructPar` worked after the stop.
  - **Tests:** `tests/test_tools.py` checks three cases: a converging series is settled, jitter above
    tolerance is a floor, and equal small one-way steps keep running. That test caught a hole in the
    first version: a steady drift with every step below tolerance counted as settled. The fixed rule
    extrapolates a one-way trend at its average decay rate. Its replay gives the same stop times as
    the first version, which the live runs used.
- **Key numbers:**
  - Peaks and frequency of the stopped runs differ from the full-length runs by at most 1.1e-4.
  - Estimated case 4 OpenFOAM study: about 1740 s instead of 2548 s, about a third less, mostly from
    M4 (2.0 s of 3 s).
  - Insight 20 in case 4. Records: `results/dfg_cylinder_re100/live_stop.json`.
- **Open:**
  - Steady SIMPLE cases still stop on residual targets. Their per-mesh check that the residuals meet
    the physical check is unchanged.
  - The full study was not rerun with the live stop; the saving is the replay's estimate.

### gpucfd connected to the agent; case 4 run on gpucfd and OpenFOAM (`dac661a`)
- **What:**
  - `cfdlib/gpu.py` lets the agent use gpucfd as a solver:
    - `check_fit` says whether a problem fits gpucfd.
    - `run_transient` runs it with live health checks and a live stopping rule.
    - `write_run` records the run like an OpenFOAM run.
  - The skill (SKILL.md, stage 2) now says when to choose gpucfd and how to verify it.
  - Case 4 (cylinder at Re 100, vortex shedding) was run on gpucfd (`gpu_run.py`) and compared with
    the OpenFOAM result by case 4's own rules (`gpu_compare.py`).
- **Why it matters:** this is the first time the agent can choose between two solvers, with
  measured numbers on the same problem.
- **How it was verified:** three grids (17.6, 35 and 70 cells per diameter) and a time-step check,
  with all rules set before the runs.
  - **Pressure tolerance:** tuned with a same-answer rule, as OpenFOAM's solver settings were (1.7×
    less time per step).
  - **Stopping:** each grid stops once case 4's periodicity rule holds, checked live.
- **Key numbers:**
  - **Time:** gpucfd ran the whole study in 315 s against OpenFOAM's 2548 s (8×). Per cell and
    time step it is about 70× faster.
  - **Accuracy:** gpucfd's drag is 1.0% and its lift maximum 3.4% below the benchmark, against
    0.25% and 0.4% for OpenFOAM. The Strouhal number is within 0.3% on both.
  - **Same answer?** The two tools agree within their uncertainties on everything except the lift
    maximum. Those uncertainty bands are wide, though, because gpucfd's coarsest grid is far off.
  - **Verdict:** WARN by case 4's rules. The drag target set before the runs (within 0.3%) was
    missed.
- **Problems found:**
  - A tight solver tolerance copied from the tests made each step 1.7× slower than needed.
  - OpenFOAM's fixed 3 s per grid was too short for gpucfd: its coarse grid starts 16% low, so the
    finer grids need longer to settle. They are now stopped by the periodicity check instead.
- **Open:** at equal accuracy OpenFOAM is faster here. Its body-fitted grid reaches 0.25% with 13
  thousand cells in 97 s. gpucfd's uniform grid would need roughly 140 to 280 cells per diameter
  (estimated, not run). Stretched grids are the next gpucfd step.
- **Next:** the same live stop for OpenFOAM runs; then stretched grids for gpucfd.

### gpucfd: grid studies for steady flows, 20× faster (rung 11), `507ebeb`
- **What:** `gpucfd/steady.py` makes steady grid studies faster and self-stopping. gpucfd still
  marches in time to steady state; it is not a direct steady solver. Two changes make the march
  cheaper:
  - **Grid sequencing:** each finer grid starts from the coarser grid's solution instead of from rest.
  - **An error-based stop:** each run stops once the error left from marching is below 1% of the
    grid's own discretisation error, which is estimated while it runs from the difference to the
    coarser grid. Every time chunk logs the forces with that estimate, so runs can be watched live.
  - A study can also stop adding grids once the answer has settled.
  - The estimator (`iteration_error`) is in `cfdlib/verification.py`, so OpenFOAM runs can use it
    next.
- **Why it matters:** with bodies, gpucfd must use the explicit scheme (IMEX is unstable there), so
  fine-grid steady runs were slow. DFG 2D-1 at 70 cells per diameter had not finished after
  2.6 GPU-hours from rest.
- **How it was verified:** rung 11 (`gpucfd/tests/test_steady.py`, 6 of 6 pass) and a one-off DFG
  2D-1 study.
  - **Estimator:** tested on sequences with a known limit. It is exact for a single decaying mode,
    never too small in the harder cases, and it recognises solver noise.
  - **Same answers:** a Couette flow with an exact solution, on three grids, gives the same answers
    sequenced and from rest.
  - **DFG 2D-1:** the same drag, lift and pressure difference as runs started from rest.
- **Key numbers (DFG 2D-1, Re 20):**
  - **35 cells per diameter:** 19.6 GPU-seconds against 398 from rest (20×). The new stop gives
    3.8×, sequencing another 5.4×.
  - **70 cells per diameter:** 331 GPU-seconds against more than 9,400 from rest. Drag is within
    0.14% of the benchmark, lift within 5%. The three-grid study takes 6 GPU-minutes in all.
  - **Couette (exact solution):** sequencing halves the GPU time (7.1 s against 14.2 s).
- **Problems found while building it, all fixed:**
  - A slow decaying mode hidden under a fast one made the error estimate up to 11× too small.
  - Solver noise made a run march 200 time units past convergence.
  - The Richardson tool gave a value 11× off for a probe too close to a wall. It now refuses to
    extrapolate when the observed order is below 0.5.
- **Open:**
  - **Pressure difference:** it gets slightly worse on the finest grid (−0.23% at 35 cells per
    diameter, −0.47% at 70). The surface pressure next to an immersed body converges irregularly; drag
    and lift are not affected.
  - **Richardson extrapolation:** it does not improve drag or lift here. Report it as an estimate,
    never as the answer.
- **Next:** connect gpucfd to the agent through cfdlib, starting with case 4 (DFG 2D-2), against
  OpenFOAM.

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
