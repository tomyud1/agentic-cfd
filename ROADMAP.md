# Roadmap

The next steps toward the goal in [`CLAUDE.md`](CLAUDE.md): an agent that solves the CFD problems
engineers get, with tools that are better and faster than what exists. Solving verified problems
and learning from each one ([`dev/LOOP.md`](dev/LOOP.md)) is the method, not the goal. History is in
[`CHANGELOG.md`](CHANGELOG.md); the cases are in [`cases/BACKLOG.md`](cases/BACKLOG.md).

Agreed with the student on 2026-11-13. The meeting with Prof. Frankel in the week of 2026-11-16 may
change the order.

## Where the project stands

| Part | Works today | Missing |
|---|---|---|
| Workflow | Problem statement to checked report, on laminar flow (steady, unsteady, 2D, 3D), heat transfer, and one turbulent flow with a turbulence model (flat plate) | Turbulent flows with pressure gradients and separation, real geometry, a problem with no reference answer |
| Tools | OpenFOAM runs with live stop and monitoring, verification maths, report contract. gpucfd: 47-235× faster than OpenFOAM on a periodic box; runs a turbulent channel without a model, 2-9× faster per step than OpenFOAM on 8 cores (case 17, verdict FAIL: mean speed 0.34 % off on the finest grid, but the profiles are not resolved there) | gpucfd's grids are rows and columns of cells (stretched per direction, also around bodies), so on bodies it still needs about 30× more cells than OpenFOAM's body-fitted grid for the same accuracy, and its drag on case 4 stops 0.3 % short of the converged benchmark. No meshing beyond `blockMesh` |
| Agent | Claude at high effort (case 7: Fable 5.2), on one PC | Never tried with another agent or on another machine |

## Now: until the meeting (development PC)

### 1. Ready for an outside tester

A PhD student of Prof. Frankel will install and test the repository next week. **Done
(2026-11-13):** the restructure is committed and pushed, and a clean copy in a different folder
rebuilt all six reports and passed the tests without this PC's paths.

### 2. Tools: stretched grids for gpucfd

Better tools are a main goal, and a GPU solver is the tool with the most interest behind it. Today
gpucfd's cells all have one size, so a thin boundary layer forces a fine grid everywhere. A
stretched grid puts small cells at the walls and the body and large ones elsewhere. Case 4 measured
what that is worth: gpucfd is 25 to 45× faster per cell and step, but its uniform grid needed
442 thousand cells for 1.0 % in drag where OpenFOAM's body-fitted grid reached 0.25 % with
13 thousand.

In stages, each one verified on the ladder before the next (plan in
[`gpucfd/DESIGN.md`](gpucfd/DESIGN.md)):

1. **Stretched grids with walls, no bodies. Done (2026-11-13).** Second order on four exact
   solutions; with line relaxation the pressure solver holds 7 to 10 iterations up to cell aspect
   ratio 64. Not a speed win yet: a stretched step costs about twice a uniform one, and on the
   Re 100 cavity (thick wall layers) stretching took 1.9× the time. Open: grids clustered in two
   directions in 3D converge slowly.
2. **Bodies on stretched grids. Done (2026-11-13).** Verified on the ladder (rung 13). On case 4
   with cells clustered around the cylinder: at the uniform grid's drag error (about 1.0 %) the
   study takes 241 s against 328 s, and about the same time once lift and shedding frequency have
   to match too. At 140 cells per diameter the drag is 0.1 to 0.2 % from the benchmark (two
   positions of the cylinder in the grid) and the three-level study takes about 1400 s. Clustering
   saves 3.85× in cells, not more: the fine bands cross the whole channel.
3. **The IMEX scheme (implicit diffusion) with bodies. Done (2026-11-13).** The cause of its
   instability was found (the ghost faces of the body answer the pressure correction, and the
   leftover was fed back into the pressure) and removed with a projection that includes the ghost
   faces. Verified on the ladder (rung 14): same steady states as the explicit scheme, second
   order in time. On case 4 the finest level gives the same answer 1.55 times faster and the
   three-level study takes 1005 s instead of about 1400 s. Less than hoped: an implicit step costs
   twice an explicit one. It does not change the accuracy.

*Success, fixed in advance:* case 4's drag within 0.25 % of the benchmark in less wall time than
OpenFOAM's whole study (about 1740 s with the live stop). Speed-only changes stay bit-identical on
uniform grids. The stop rule for stage 2 was: if, at the same drag error, the clustered grid's
study does not take less wall time than gpucfd's own uniform-grid study after two working
sessions, write down what was found and move to step 3 first. It was not needed.

*Outcome (2026-11-13): met as defined, but the definition was too easy.* Drag 0.1 to 0.2 % below
the benchmark's finest tabulated value in about 1400 s, repeated with the cylinder moved half a
cell in the grid. That tabulated value is itself about 0.2 % below the value the benchmark's own
levels converge to. Against the converged value gpucfd is 0.3 to 0.4 % low, and OpenFOAM is
within 0.04 %. The other limits: the three levels by themselves support an uncertainty of 0.6 to
0.75 %; lift is about 1 % low; and OpenFOAM's second grid alone has the drag at 0.24 % after
153 s. All of it is in the
[study report](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_re100/gpu_study.html).
A future target is set against the reference's converged value, with the reference's own
uncertainty stated. Stage 3 cut the study to 1005 s with the same drag, 0.3 % below the converged
benchmark: faster, not more accurate. One more run had the fine cells reach further behind the
cylinder on the finest level: lift 0.7 % low in place of 0.85 %, study 1308 s. That is one position
of the cylinder, and moving the cylinder changes the lift by 0.3 %, so it is a hint, not a result.
Details and limits: [`gpucfd/DESIGN.md`](gpucfd/DESIGN.md), "Stretched grids, stage 2" and "IMEX
with immersed bodies, stage 3".

*Left open on purpose* (each is in `gpucfd/DESIGN.md`; none is next before the meeting, because
case 4's gap to OpenFOAM is now accuracy per cell, not time):
- One pressure solve per time step in place of three: the largest speed lever left with bodies.
- The explicit scheme with bodies has a stability limit nobody knew before stage 3. Every run so far
  is at half of it or below, and a run that crossed it would blow up visibly, not give a quiet wrong
  answer. Giving the explicit scheme the new projection would remove the limit, but it changes every
  explicit result with a body in about the fourth digit, so those records would have to be made
  again. Decision for the student; recommended only if a case comes near the limit.
- The longer-band run with the cylinder shifted (17 GPU-minutes), so its lift has a spread.

### 3. Case 7: turbulent flat plate (RANS, Spalart-Allmaras, NASA TMR)

**Done (2026-11-13).** A fresh agent solved the first turbulence case in 52 minutes: skin friction
0.028 % from NASA's incompressible value, verdict WARN by a rule that counted grid error only
(0.024 %). The project keeps solving known cases until the capabilities are there; case 8
(buoyancy cavity) follows.

*Decided after case 7 (2026-11-14):* the next step for turbulence is gpucfd, not tuning OpenFOAM
(a tuned setting would be specific to one case). Two routes were measured before building
anything ([`gpucfd/DESIGN.md`](gpucfd/DESIGN.md), "Turbulence resolved without a model"):
- *A turbulence model as in case 7 (RANS) in gpucfd:* predicted to lose, 1.5 to 3 hours against
  OpenFOAM's 10 minutes on case 7's finest grid, because gpucfd marches in time and the thin wall
  cells force a tiny time step. Not built.
- *Resolving the eddies in time:* works today with no model. On a channel at friction Reynolds
  number 180 the mean speed was 0.8 % below the reference after 34 minutes (not yet settled), at
  2 to 9 times OpenFOAM's speed per step.

### 3b. Case 17: turbulent channel without a model (gpucfd's first turbulent flow)

**Solved by a fresh agent (2026-11-14), verdict FAIL.** Reference: Vreman and Kuerten 2014, two
independent codes within 0.03 % on the mean speed.

*Fixed in advance, and what came out.* Success for gpucfd was: mean speed through the channel
within 1 % of the reference, with a statistical error of at most 0.3 % and more than one grid to
show the trend, in under about 5 GPU hours on this PC. Outcome: met on the finest grid only
(192 x 64 x 64: 0.34 % low, error 0.16 %, 1.8 GPU hours in all). The two coarser grids are 1.5 %
high and 1.6 % low, so there is no trend, and the velocity and fluctuation profiles miss the
limits the agent set before its runs. The grid that fits this PC does not resolve the flow at
the wall.

*The agent's 19 insights are answered* (2026-11-14, [`CHANGELOG.md`](CHANGELOG.md)). Left open
from them: a contract check of a gpucfd case's decisions against its recorded solver settings, and
gpucfd records in `cfdlib.monitor`'s own live view (`cfdlib.gpu.peek` is a separate command).

*Optimize stage (step 8 of the loop), done 2026-11-14: target met.* The question was: what is the
cheapest grid on this PC that resolves the profiles?
- Fixed in advance: one grid on which all five profile limits of case 17 hold and the mean speed
  is within 1 % with a statistical error of at most 0.25 %, in a final run of at most 2 GPU
  hours; budget 5 GPU hours; if no grid of at most 3 million cells can do it, stop and report
  which direction limits the accuracy.
- Outcome: 288 x 64 x 96 (1.8 million cells) meets every pass / fail limit of the case: mean
  speed +0.17 % with an error of 0.19 %, the five profile gaps at a quarter to two thirds of
  their limits. Final run 1.7 GPU hours, the stage 3.1. Three short runs, each refining one
  direction, showed what matters: the cells across the flow first, along it second, toward the
  wall not at all.
- Its limits: one grid, no grid study at its level, and about 0.5 % of grid error left in the
  mean speed from each of two directions, with opposite signs. The verdict of the case stays
  FAIL, because that is the solve. Details: [`gpucfd/DESIGN.md`](gpucfd/DESIGN.md), channel
  section; records in `dev/optimize/turbulent_channel_re180/`.
- Not done: a page that shows the improved profiles next to the first ones (a study report of
  the case), and one finer step across the flow to see whether the profiles have stopped moving.

*After it,* with a target at a higher Reynolds number fixed in advance: a viscosity that varies
from cell to cell (the first brick of any model), then a model for the smallest eddies (large-eddy
simulation). Whether that is the right direction is question 1 for Prof. Frankel.

### 4. Workflow: offer, do not force

The instruction to read every solved case is gone: the solved cases are now a library
(`cases/README.md` lists what each one shows) that the agent consults as it sees fit. Still to do:

- Go through `WORKFLOW.md` with the same rule: keep the firm rules that protect correctness, turn
  the rest into offers with their trade-offs.
- Split it into a short core plus topic references opened when the physics calls for them (heat
  transfer, periodic flows, steady speed-up, performance study). Case stories move to their topic.
- Record what each case cost (agent, minutes, tokens) in the catalogue. Case 7's cost is in the
  changelog only (Fable 5.2 at high effort, 52 minutes, 26 of them solver runs; tokens not recorded).
- The same diet for `gpucfd/DESIGN.md` (about 1500 lines): one file per stage, with DESIGN.md as
  the index, status table and limits. Less urgent: no agent reads it on every case.

## Next week

- **Meeting with Prof. Frankel.** The questions below.
- **The PhD student's test.** Treat what they report like a solver agent's insights: each one gets
  a tool fix, a guidance change or a stated reason for neither.
- **Second machine, second agent.** Describe the lab machine in `workflow/machines/`, rebuild the
  reports there, run gpucfd's ladder. The other student's local agent (DGX Spark) solves cases 1
  and 3 from scratch; what it cannot follow should become tool checks, not more text.

## After that

- **More known cases**, each adding a capability (tiers 1 to 3 of the backlog).
- **gpucfd beyond stretched grids.** Which way depends on the meeting: turbulence (large-eddy
  simulation suits a fast explicit GPU solver), heat transfer, 3D bodies. Compressible flow would
  be a different solver, not an extension.
- **Meshing from real geometry** (`snappyHexMesh` or Gmsh): engineers' problems arrive as geometry.
- **Measuring the method:** what became of each insight; a test that re-rendering changes no
  published report (figures are byte-identical from run to run since 2026-11-13); a small generated
  file of each case's headline numbers, with a test that the numbers quoted by hand in README,
  ROADMAP and the design notes still match it; and replaying an old case with a fresh agent after
  large workflow changes.
- **A blind case** (the agent does not see the reference and must give an uncertainty band), once
  the capabilities above exist. Then Prof. Frankel's problem with no published answer.

## Decided

- **Required accuracy belongs to the problem.** `problem.yaml` should state it, and a small miss can
  still be acceptable depending on what the result is used for. How an engineer sets and judges it
  is a question for Prof. Frankel; the verdict rules change after that.
- **No open-source license for now.** The work may become commercial. Without a license nobody may
  reuse the code, which is the safe position; testers get permission personally.
- **Case sources.** The other student's links are NASA TMR and the Luminary V&V manual, both in
  `cases/catalogue.yaml`.
- **Not moving `cfdlib/`, `gpucfd/` or `cases/`.** The commands printed in the published reports
  depend on them.

## Questions for Prof. Frankel

1. Which flows matter for a GPU solver, to him and to industry: incompressible or compressible,
   RANS or LES, which kinds of geometry and grid? This decides gpucfd's direction after stretched
   grids.
2. How does an engineer set the required accuracy of a simulation, and when is a small miss
   acceptable?
   And a narrower one: when three grids do not show a clear order of convergence, which error
   band is honest? Today the tools assume second order with a safety factor of 1.25 and mark the
   number as a placeholder; common practice (Roache) is a factor of 3. Switching would widen the
   bands of cases 4 and 5 and can change their verdicts.
   Case 7 adds an example: its friction is 0.028 % from NASA's value and got a WARN, because the
   band held only the grid error of the two solutions (0.024 %), while a boundary choice alone
   moves the result by 0.26 %. Is such a band too strict to be useful?
3. May the repository stay public? Who owns the code under the Technion's rules, and should gpucfd
   move to a private repository?
4. His criticism of one report (the failed case 5 is the most instructive).
5. A problem he cares about with no published answer, for later.
6. The lab machine: hardware, access, and whether an agent may run on it.
