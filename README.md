# Agentic CFD

An AI agent that takes a fluid-flow problem, solves it with [OpenFOAM](https://www.openfoam.com/),
and writes a report an engineer can check line by line.

You give it a short problem statement (geometry, fluid, boundary conditions, what you want to know,
and optionally reference data). The agent derives what theory can say, sets up and tests the
simulation, runs mesh and time-step studies, compares the result with the exact solution or a
published benchmark, and explains every choice it made in an HTML report.

Final-year project, Faculty of Mechanical Engineering, Technion, supervised by Prof. Steven Frankel.
The agent here is [Claude Code](https://claude.com/claude-code); the workflow in
[`workflow/`](workflow/WORKFLOW.md) is written for any coding agent. The project also builds faster
tools for the agent, starting with a GPU solver ([below](#gpu-solver-gpucfd)). Every problem it
solves is used to improve it ([how it works](#how-it-works)).

## Start here

| You want to | Go to |
|---|---|
| See what the agent solved: every case, its verdict and its report | [cases/README.md](cases/README.md) (or the [report gallery](https://tomyud1.github.io/agentic-cfd/)) |
| Install it, re-create the reports, re-run a case | [setup/README.md](setup/README.md) |
| Solve your own CFD problem with a coding agent | [AGENTS.md](AGENTS.md) (for the agent) and [cases/README.md](cases/README.md#give-the-agent-a-new-problem) (for you) |
| Develop the project itself (tools, gpucfd, the workflow) | [CLAUDE.md](CLAUDE.md), then [dev/LOOP.md](dev/LOOP.md) |
| See what changed recently, and what is planned | [CHANGELOG.md](CHANGELOG.md), [ROADMAP.md](ROADMAP.md) |
| See which problems come next, and every candidate | [cases/BACKLOG.md](cases/BACKLOG.md) |
| Read how the GPU solver works and how it is verified | [gpucfd/DESIGN.md](gpucfd/DESIGN.md) |

## Results

Each case is given to a **fresh agent** that has never seen it, with only the problem statement, the
tools and the guidelines. So far that covers laminar flow, steady and unsteady, in 2D and 3D, heat
transfer, and a first turbulent flow with a turbulence model (RANS), each checked against an exact
solution or a published benchmark (Ghia et al., FeatFlow, NASA's Turbulence Modeling Resource). Every report ends in a verdict, PASS, WARN or FAIL, set by accuracy rules the agent fixed
before seeing its results. Verdicts are never softened: the failures are listed in
[cases/README.md](cases/README.md) next to the passes.

## GPU solver (gpucfd)

Case 5 showed that OpenFOAM's GPU plugins can't go past about 1.5×, because only the linear
solver runs on the GPU. [`gpucfd/`](gpucfd) is our own solver that keeps the whole time step on the
GPU.

- **Speed:** on the same 3D flow and grid as OpenFOAM, each with its best settings, it is 47× faster
  at 262 thousand cells and 235× faster at 2.1 million, with a smaller error (RTX 4070 against
  8 CPU cores).
- **Trust:** every feature passes a verification ladder before the next is built: exact solutions,
  an independent reference code and the order of accuracy. Speed-only changes must give
  bit-identical results.
- **What it handles:** incompressible laminar flow in a box with walls, inflow and outflow, and
  bodies of any shape inside the flow, on uniform grids or with cells clustered toward walls and
  around bodies. On the DFG cylinder benchmark at Re 100 (case 4), with cells clustered around the
  cylinder, drag at 140 cells per diameter is 0.1 to 0.2% below the benchmark's finest tabulated
  value and 0.3 to 0.4% below the value the benchmark's own levels converge to (OpenFOAM: 0.04%);
  lift is about 1% low. The three-grid study takes 17 minutes against 29 for OpenFOAM's, with
  implicit diffusion on the finest grid (24 with the explicit scheme;
  [study report](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_re100/gpu_study.html)).
- **Not yet:** turbulence, heat transfer. OpenFOAM's
  body-fitted grid is still the cheaper way to a given accuracy for bodies at Re 100 (case 4: drag
  within 0.25% at 13 thousand cells in under 3 minutes; gpucfd needs 459 thousand cells).

## How it works

Three parts, and a loop that improves them:

1. **Workflow** ([`workflow/`](workflow)): the stages an agent follows (intake, theory, setup,
   pre-flight, production, analysis, report, visual review), an OpenFOAM primer with every pitfall
   met so far, and the report contract. The solved [`cases/`](cases) are its worked examples: the
   agent copies their structure, not their numbers.
2. **Tools** ([`cfdlib/`](cfdlib), OpenFOAM, [`gpucfd/`](gpucfd)): deterministic code that runs the
   solvers, reads their output, computes errors, observed order, Richardson extrapolation and GCI,
   draws figures (and refuses ones with overlapping text), and renders the report. Rule: **code
   computes every number**; the agent never types a result. A report contract
   ([`cfdlib/contract.py`](cfdlib/contract.py)) is checked before any report is written.
3. **Agent**: any coding agent that can read files and run commands. What differs between computers
   is kept in one file per machine ([`workflow/machines/`](workflow/machines)).

**The loop** ([`dev/LOOP.md`](dev/LOOP.md)): a fresh agent solves a problem with a known answer and
logs every limitation it meets; a second agent then reviews the report and the log, and improves
the workflow and the tools before the next problem. Example: in case 4 the agent found that
OpenFOAM's PISO algorithm was only first-order accurate in time for this flow, which made the lift
10 % wrong; the workflow now puts a time-step study in the pre-flight of every unsteady nonlinear
problem.

## Repository map

```
AGENTS.md                     for an agent solving a user's CFD problem with this repository
CLAUDE.md, ROADMAP.md         the project's direction and rules, for agents developing it; the next steps
workflow/                     WORKFLOW.md (the stages), references/ (primer, contract), machines/ (one file per computer)
cfdlib/                       shared tools: run OpenFOAM and gpucfd, verification, plots, report, contract
gpucfd/                       the GPU solver: CUDA source, Python wrapper, verification ladder, DESIGN.md
cases/                        one folder per problem; catalogue.yaml (every case), README.md (solved), BACKLOG.md (next)
results/                      one report per case, with the extracted run data
dev/                          LOOP.md (how a solved case improves the project), review script, old experiments
setup/, tests/                install, re-render all reports, GPU setup, benchmark data; tool tests
index.html                    the report gallery served by GitHub Pages (generated from cases/catalogue.yaml)
```

## Roadmap

The ordered plan is in [ROADMAP.md](ROADMAP.md). In short:
more problem types (buoyancy next, then harder turbulent flows); a shorter workflow that offers its
records to the agent instead of prescribing them; then a second machine and a second agent.

## Background papers

The project started from three papers on LLM agents for OpenFOAM (not included here, see the
sources):

- *A Preliminary Assessment of Coding Agents for CFD Workflows*, arXiv:2602.11689 (2026).
- *OpenFOAMGPT 2.0: End-to-end, trustworthy automation for CFD*, International Journal of Heat and
  Fluid Flow (2026).
- *IteraSim RAG*, arXiv:2607.20346 (2026).

## Credits

Benchmark data in `cases/*/reference/` belongs to its authors (cited in each file). OpenFOAM is a
trademark of OpenCFD Ltd; this project is not approved or endorsed by OpenCFD.
