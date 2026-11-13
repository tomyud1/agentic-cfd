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
| Workflow | Problem statement to checked report, on laminar flow (steady, unsteady, 2D, 3D) and heat transfer | Turbulence, real geometry, a problem with no reference answer |
| Tools | OpenFOAM runs with live stop and monitoring, verification maths, report contract. gpucfd: 47-235× faster than OpenFOAM on a periodic box | gpucfd's grids are rows and columns of cells (stretched per direction, also around bodies), so on bodies it still needs about 30× more cells than OpenFOAM's body-fitted grid for the same accuracy; with bodies its time scheme is explicit. No meshing beyond `blockMesh` |
| Agent | Claude Opus 5.5, high effort, on one PC | Never tried with another agent or on another machine |

## Now: until the meeting (development PC)

### 1. Ready for an outside tester

A PhD student of Prof. Frankel will install and test the repository next week. **Done
(2026-11-13):** the restructure is committed and pushed, and a clean copy in a different folder
rebuilt all six reports and passed the tests without this PC's paths.

### 2. Tools: stretched grids for gpucfd

Better tools are a main goal, and a GPU solver is the tool with the most interest behind it. Today
gpucfd's cells all have one size, so a thin boundary layer forces a fine grid everywhere. A
stretched grid puts small cells at the walls and the body and large ones elsewhere. Case 4 measured
what that is worth: gpucfd is about 50× faster per cell and step, but its uniform grid needed
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
3. **The IMEX scheme with bodies** (unstable today). On a stretched grid the smallest cell sets the
   explicit viscous time step for the whole domain, so without this the speed is lost again.

*Success, fixed in advance:* case 4's drag within 0.25 % of the benchmark in less wall time than
OpenFOAM's whole study (about 1740 s with the live stop). Speed-only changes stay bit-identical on
uniform grids. The stop rule for stage 2 was: if, at the same drag error, the clustered grid's
study does not take less wall time than gpucfd's own uniform-grid study after two working
sessions, write down what was found and move to step 3 first. It was not needed.

*Outcome (2026-11-13): met as defined, with limits.* Drag 0.1 to 0.2 % below the benchmark in
about 1400 s, repeated with the cylinder moved half a cell in the grid. The limits: the three
levels by themselves support an uncertainty of 0.6 to 0.75 %, not 0.25 %; lift is about 1 % low;
and OpenFOAM's second grid alone has the drag at 0.24 % after 153 s. Still open: the finest
level with the fine cells reaching further behind the cylinder (about 30 GPU-minutes), which
should bring the lift below 1 % at a study time close to the 1740 s limit. Stage 3 is the next
step: the finest level is 83 % of the time, held back by the explicit time step and the pressure
tolerance it forces. Details and limits: [`gpucfd/DESIGN.md`](gpucfd/DESIGN.md), "Stretched grids,
stage 2".

### 3. Case 7: turbulent flat plate (RANS, Spalart-Allmaras, NASA TMR)

The first turbulence case, with a fresh agent. Preparation as in `dev/LOOP.md`: TMR's data in
`reference/`, and a tool-gap check (turbulence dictionaries in the contract, y+ extraction). The
project keeps solving known cases until the capabilities are there; case 8 (buoyancy cavity)
follows.

### 4. Workflow: offer, do not force

The instruction to read every solved case is gone: the solved cases are now a library
(`cases/README.md` lists what each one shows) that the agent consults as it sees fit. Still to do:

- Go through `WORKFLOW.md` with the same rule: keep the firm rules that protect correctness, turn
  the rest into offers with their trade-offs.
- Split it into a short core plus topic references opened when the physics calls for them (heat
  transfer, periodic flows, steady speed-up, performance study). Case stories move to their topic.
- From case 7 on, record what each case cost (agent, minutes, tokens) in the catalogue.

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
- **Measuring the method:** what became of each insight, byte-identical re-renders as a regression
  test, and replaying an old case with a fresh agent after large workflow changes.
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
3. May the repository stay public? Who owns the code under the Technion's rules, and should gpucfd
   move to a private repository?
4. His criticism of one report (the failed case 5 is the most instructive).
5. A problem he cares about with no published answer, for later.
6. The lab machine: hardware, access, and whether an agent may run on it.
