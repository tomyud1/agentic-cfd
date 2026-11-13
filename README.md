# Agentic CFD

An AI agent that takes a fluid-flow problem, solves it with [OpenFOAM](https://www.openfoam.com/),
and writes a report an engineer can check line by line.

You give it a short problem statement (geometry, fluid, boundary conditions, what you want to know,
and optionally reference data). The agent derives what theory can say, sets up and tests the
simulation, runs mesh and time-step studies, compares the result with the exact solution or a
published benchmark, and explains every choice it made in an HTML report.

Final-year project, Faculty of Mechanical Engineering, Technion, supervised by Prof. Steven Frankel.
The agent is [Claude Code](https://claude.com/claude-code) following the workflow in
[`.claude/skills/agentic-cfd/`](.claude/skills/agentic-cfd/SKILL.md). The project also builds
faster tools for the agent, starting with a GPU solver ([below](#gpu-solver-gpucfd)).

## Start here

| You want to | Go to |
|---|---|
| See what the agent solved: every case, its verdict and its report | [cases/README.md](cases/README.md) (or the [report gallery](https://tomyud1.github.io/agentic-cfd/)) |
| Install it, re-create the reports, re-run a case | [setup/README.md](setup/README.md) |
| Solve your own CFD problem with a coding agent | [AGENTS.md](AGENTS.md) (for the agent) and [cases/README.md](cases/README.md#give-the-agent-a-new-problem) (for you) |
| Develop the project itself (tools, gpucfd, the workflow) | [CLAUDE.md](CLAUDE.md) |
| See what changed recently | [CHANGELOG.md](CHANGELOG.md) |
| See which problems come next, and every candidate | [cases/BACKLOG.md](cases/BACKLOG.md) |
| Read how the GPU solver works and how it is verified | [gpucfd/DESIGN.md](gpucfd/DESIGN.md) |

## Results

Each case is given to a **fresh agent** that has never seen it, with only the problem statement, the
tools and the guidelines. So far that covers laminar flow, steady and unsteady, in 2D and 3D, and
heat transfer, each checked against an exact solution or a published benchmark (Ghia et al.,
FeatFlow). Every report ends in a verdict, PASS, WARN or FAIL, set by accuracy rules the agent fixed
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
  bodies of any shape inside the flow. On the DFG cylinder benchmark (Re 20), drag is within 0.14%
  at 70 cells per diameter; the three-grid study takes 6 GPU-minutes.
- **Not yet:** turbulence, heat transfer, stretched grids. Without stretched grids, OpenFOAM's
  body-fitted grid is the better choice for bodies at Re 100 (case 4: drag within 0.25% at 13 thousand
  cells, against 1.0% at 442 thousand for gpucfd).

## How it works

Three layers, so the agent can solve problems it has not seen before:

1. **Tools** ([`cfdlib/`](cfdlib)): deterministic Python that runs OpenFOAM or gpucfd, reads their
   output, computes errors, observed order, Richardson extrapolation and GCI, draws figures (and
   refuses ones with overlapping text), and renders the report. Rule: **code computes every
   number**; the agent never types a result.
2. **Guidelines** ([`.claude/skills/agentic-cfd/`](.claude/skills/agentic-cfd)): the workflow
   (intake, theory, setup, pre-flight, production, analysis, report, visual review), an OpenFOAM
   primer with every pitfall met so far, the report contract and the machine setup.
3. **Worked examples** ([`cases/`](cases)): the solved cases. The agent copies their structure, not
   their numbers.

A **report contract** ([`cfdlib/contract.py`](cfdlib/contract.py)) is checked before any report is
written: required sections, every setting covered by a decision, math typeset as LaTeX, figure
references valid.

The **insight log** is how the workflow improves: agents record problems as they meet them, and the
lessons go into the guidelines for the next agent. Example: in case 4 the agent found that
OpenFOAM's PISO algorithm was only first-order accurate in time for this flow, which made the lift
10 % wrong; the guidelines now require a time-step study in the pre-flight of every unsteady
nonlinear problem.

## Repository map

```
AGENTS.md                     for an agent solving a user's CFD problem with this repository
CLAUDE.md                     the project's direction and rules, for agents developing it
.claude/skills/agentic-cfd/   the agent's workflow (SKILL.md) and references
cfdlib/                       shared tools: run OpenFOAM and gpucfd, verification, plots, report, contract
cases/                        one folder per problem; catalogue.yaml (every case), README.md (solved), BACKLOG.md (next)
results/                      one report per case, with the extracted run data
gpucfd/                       the GPU solver: CUDA source, Python wrapper, verification ladder, DESIGN.md
setup/                        install, re-render all reports, GPU setup, benchmark data (README.md)
tests/, dev/                  smoke tests; early exploration and independent checks of the agents' claims
index.html                    the report gallery served by GitHub Pages (generated from cases/catalogue.yaml)
```

## Roadmap

- **More problem types:** turbulence (RANS) next: [cases/BACKLOG.md](cases/BACKLOG.md).
- **gpucfd as a tool for the agent:** connected (`cfdlib/gpu.py`, guidance in the skill). On case 4
  it ran the whole grid study 8× faster than OpenFOAM, but its uniform grid needs about 30× more cells
  around the body for the same accuracy. Next: stretched grids.
- **Smarter runs:** runs stop live on their physical check, not at a guessed end time (case 4: a
  third shorter). Steady runs reach the same answer sooner: OpenFOAM 3-11× (case 2, 128² to 512²),
  gpucfd 20×.

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
