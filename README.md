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

**What's new:** [CHANGELOG.md](CHANGELOG.md). **What's next:** [cases/BACKLOG.md](cases/BACKLOG.md).

## Results so far

Each case was given to a **fresh agent** that had never seen it, with only the problem statement,
the tools and the guidelines. Its lessons were then folded back into the guidelines before the next case.

| # | Case | What it tests | Verdict | Headline | Report |
|---|---|---|---|---|---|
| 1 | Plane Poiseuille flow | Steady, linear, exact solution | ✅ PASS | Velocity profile within 0.0061 % of the exact solution, 2nd-order convergence | [view](https://tomyud1.github.io/agentic-cfd/results/plane_poiseuille/report.html) |
| 2 | Lid-driven cavity, Re = 100 | Steady, nonlinear, benchmark (Ghia et al. 1982) | ⚠️ WARN | Velocity profiles within 0.009 U of the benchmark; the remaining gap is traced to the benchmark's own mesh | [view](https://tomyud1.github.io/agentic-cfd/results/lid_driven_cavity_re100/report.html) |
| 3 | Stokes' first problem | Transient, linear, exact solution | ✅ PASS | Velocity within 0.002 % of U, 2nd-order convergence | [view](https://tomyud1.github.io/agentic-cfd/results/stokes_first_problem/report.html) |
| 4 | Cylinder in a channel, Re = 100 (DFG 2D-2) | Transient, nonlinear, vortex shedding, benchmark (FeatFlow) | ⚠️ WARN | Drag, lift and Strouhal number all inside their uncertainty bands (St 0.3015 vs 0.3019); MPI scaling study | [view](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_re100/report.html) |
| 5 | Cylinder in a square duct, Re = 20 (DFG 3D-1Z) | Steady, 3D, up to 2.4 million cells, benchmark (FeatFlow) | ❌ FAIL | Drag within 0.08 % of the reference; lift 1.7 % off, outside its band. A second mesh family shows the lift depends on the cell grading around the cylinder | [view](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_3d_re20/report.html) |
| 5b | GPU vs CPU study (same case) | OpenFOAM pressure solve on an RTX 4070 (OGL plugin) vs the best CPU setup | ✅ PASS | Same answer on both sides; GPU breaks even at about 2 million cells and is 1.13 times faster at 4.75 million. Capped near 1.5 times because the pressure solve is only a third of each iteration | [view](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_3d_re20/gpu_study.html) |

**WARN** means the answer is usable but a check the agent set for itself, before seeing the results,
did not fully pass. **FAIL** means a requested output missed the accuracy rule fixed before the runs.
The report says which check and why. Verdicts are never softened.

### What is in every report

| Section | Content |
|---|---|
| Summary | The answer to each requested output (lettered a, b, c...), key findings, limitations |
| 1 Problem | The problem in short, what we want to find, the setup, a sketch, the given data |
| 2 Assumptions | Each one with how it was checked (tested, standard, imposed by the model, or violated) |
| 3 Theory | What can be known without simulating: exact solution (SymPy) or scaling, limits and conservation laws |
| 4 Decisions | Every OpenFOAM setting, grouped into decisions with why, effect and alternatives. A checker refuses to render the report if any setting is undocumented |
| 5 Pre-flight | Short test runs before the long ones: crashes, stability, projected run time |
| 6 Simulation | The model, mesh and time-step studies, run health |
| 7 Comparison | Verification (exact solution) or validation (benchmark) with error bands and observed order of accuracy |
| 8 Insights | A live lab notebook: every problem the agent met, how it solved it and what should change in the workflow |
| 9 Reproducibility | The exact commands, files and software versions |
| Appendix | Performance (case 4): MPI scaling, linear solvers, what limits speed |

Case 5 also has a separate **performance study** report (GPU vs CPU): what is compared, the method,
results per mesh size, the same-answer check, where the time goes and the insights.

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
- **Not yet:** turbulence, heat transfer, stretched grids.

Details, status and limits: [`gpucfd/DESIGN.md`](gpucfd/DESIGN.md). History:
[CHANGELOG.md](CHANGELOG.md).

## How it works

Three layers, so the agent can solve problems it has not seen before:

1. **Tools** ([`cfdlib/`](cfdlib)): deterministic Python that runs OpenFOAM, reads its output,
   computes errors, observed order, Richardson extrapolation and GCI, draws figures (and refuses ones
   with overlapping text), and renders the report. Rule: **code computes every number**; the agent
   never types a result.
2. **Guidelines** ([`.claude/skills/agentic-cfd/`](.claude/skills/agentic-cfd)): the workflow
   (intake, theory, setup, pre-flight, production, analysis, report, visual review), an OpenFOAM
   primer with every pitfall met so far, the report contract and the machine setup.
3. **Worked examples** ([`cases/`](cases)): the five solved cases. The agent copies their structure,
   not their numbers.

A **report contract** ([`cfdlib/contract.py`](cfdlib/contract.py)) is checked before any report is
written: required sections, every setting covered by a decision, math typeset as LaTeX, figure
references valid.

The **insight log** is how the workflow improves: agents record problems as they meet them
(61 entries over cases 1-5), and the lessons go into the guidelines for the next agent. Example: in
case 4 the agent found that OpenFOAM's PISO algorithm was only first-order accurate in time for this
flow, which made the lift 10 % wrong; the guidelines now require a time-step study in the pre-flight
of every unsteady nonlinear problem.

## Repository map

```
.claude/skills/agentic-cfd/   the agent's workflow (SKILL.md) and references (OpenFOAM primer,
                              report contract, environment)
cfdlib/                       shared tools: foam.py (run OpenFOAM), monitor.py (pre-flight, health),
                              verification.py, signals.py, perf.py, plots.py, report.py,
                              contract.py, insights.py, templates/report.html.j2
cases/<case>/                 one folder per problem
  problem.yaml                the problem statement (the only input the agent gets)
  reference/                  benchmark data with citations (cases 2 and 4)
  numerics.yaml               the agent's numerical plan (meshes, time steps, tolerances)
  decisions.yaml              every choice, with why / effect / alternatives
  insights.yaml               the live insight log
  template/                   the OpenFOAM case (0/, constant/, system/) with {{ placeholders }}
  theory.py / analytical.py   theory (SymPy)
  run.py, analyze.py          run the simulations; analyse and write the report
  sketch.py                   the problem sketch
cases/BACKLOG.md              the next problems, in order
results/<case>/               report.html, results.json and the extracted run data (runs/)
gpucfd/                       the GPU solver: CUDA source, Python wrapper, verification ladder, DESIGN.md
setup/                        install OpenFOAM and Python, re-render all reports, fetch benchmark data
tests/                        smoke tests of the tools
dev/                          early exploration scripts and independent checks of the agents' claims
```

## Reproduce it

Everything ran on Windows 11 with WSL2 (Ubuntu 24.04), ESI OpenFOAM v2606 and Python 3.12, on an
AMD Ryzen 7 7800X3D (8 cores, 16 threads). A native Ubuntu 24.04 machine works the same way; run
the commands below in its terminal.

### 1. Install

On Windows, install WSL2 and Ubuntu 24.04 first (PowerShell as administrator):

```bash
wsl --install -d Ubuntu-24.04
```

Then, in the Ubuntu terminal, from the repository folder:

```bash
bash setup/install_openfoam.sh
sudo apt-get install -y python3-venv && bash setup/make_venv.sh
```

The first script adds the openfoam.com package repository and installs `openfoam2606-default`
(needs sudo). The second creates the Python environment in `~/venvs/cfd` with the pinned packages
from [`requirements.txt`](requirements.txt). Check the OpenFOAM install with
`bash setup/run_cavity.sh` (runs the lid-driven cavity tutorial).

### 2. Re-create the reports from the saved data (minutes, no OpenFOAM needed)

The data extracted from every run is stored in `results/<case>/runs/`, so all reports can be
rebuilt, including theory, figures and checks:

```bash
bash setup/rerender_all.sh
```

### 3. Re-run the simulations

Each report's **Reproducibility** section lists the exact commands, in order; run them from the repository folder. For example, case 3:

```bash
~/venvs/cfd/bin/python -m cases.stokes_first_problem.run preflight
~/venvs/cfd/bin/python -m cases.stokes_first_problem.run production --jobs 4
~/venvs/cfd/bin/python -m cases.stokes_first_problem.analyze
```

Runs go to `$FOAM_RUN/agentic/<case>/` (on the Linux file system, which is much faster than a
Windows drive). Cases 1-3 take minutes; case 4 takes a few hours (its finest run alone takes about
35 minutes on 16 threads).

### 4. View a report

Open `results/<case>/report.html` in a browser. It is one self-contained file; the formulas load
MathJax from a CDN.

### 5. Give the agent a new problem

1. Open this repository in a coding agent ([Claude Code](https://claude.com/claude-code), Codex or
   any other) and have it start from [AGENTS.md](AGENTS.md).
2. Write `cases/<new_case>/problem.yaml` in the same format as the existing cases: a short
   description, a `setup` list, the `given` data, the `quantities_of_interest`, and optionally a
   `reference/` folder with benchmark data and its citation.
3. Ask: *"Use the agentic-cfd workflow in `.claude/skills/agentic-cfd/SKILL.md` to solve
   `cases/<new_case>/problem.yaml`."*

The agent reads the guidelines and the worked examples, logs insights as it goes, and finishes
with `results/<new_case>/report.html`. The cases here used Claude Opus with high effort; a case took
from about 25 minutes (case 3) to about 2 hours (case 5, including the GPU study).

## Roadmap

- **More problem types:** heat transfer and turbulence (RANS) first. The ordered list, with sources:
  [cases/BACKLOG.md](cases/BACKLOG.md).
- **gpucfd as a tool for the agent:** the agent picks it when a problem fits, and documents that
  choice like any other decision.
- **Smarter runs:** done for gpucfd steady flows (each finer grid starts from the coarser one and stops
  once its answer has settled: 20× faster). Next: the same live stop for OpenFOAM runs.

GPU setup instructions for OpenFOAM's OGL plugin are in [`setup/gpu/`](setup/gpu).

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
