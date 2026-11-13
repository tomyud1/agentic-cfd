# Solved cases: start here

Every problem the agent has solved, with its verdict and report. The same reports, as a gallery:
[tomyud1.github.io/agentic-cfd](https://tomyud1.github.io/agentic-cfd/). The next problems are in
[BACKLOG.md](BACKLOG.md).

Each case was given to a **fresh agent** that had never seen it, with only the problem statement,
the tools and the guidelines. Its lessons were then folded back into the guidelines before the next
case.

## Results

| # | Case | What it tests | Verdict | Headline | Report |
|---|---|---|---|---|---|
| 1 | Plane Poiseuille flow | Steady, linear, exact solution | ✅ PASS | Velocity profile within 0.0061 % of the exact solution, 2nd-order convergence | [view](https://tomyud1.github.io/agentic-cfd/results/plane_poiseuille/report.html) |
| 2 | Lid-driven cavity, Re = 100 | Steady, nonlinear, benchmark (Ghia et al. 1982) | ⚠️ WARN | Velocity profiles within 0.009 U of the benchmark; the remaining gap is traced to the benchmark's own mesh | [view](https://tomyud1.github.io/agentic-cfd/results/lid_driven_cavity_re100/report.html) |
| 3 | Stokes' first problem | Transient, linear, exact solution | ✅ PASS | Velocity within 0.002 % of U, 2nd-order convergence | [view](https://tomyud1.github.io/agentic-cfd/results/stokes_first_problem/report.html) |
| 4 | Cylinder in a channel, Re = 100 (DFG 2D-2) | Transient, nonlinear, vortex shedding, benchmark (FeatFlow) | ⚠️ WARN | Drag, lift and Strouhal number all inside their uncertainty bands (St 0.3015 vs 0.3019); MPI scaling study | [view](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_re100/report.html) |
| 5 | Cylinder in a square duct, Re = 20 (DFG 3D-1Z) | Steady, 3D, up to 2.4 million cells, benchmark (FeatFlow) | ❌ FAIL | Drag within 0.08 % of the reference; lift 1.7 % off, outside its band. A second mesh family shows the lift depends on the cell grading around the cylinder | [view](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_3d_re20/report.html) |
| 5b | GPU vs CPU study (same case) | OpenFOAM pressure solve on an RTX 4070 (OGL plugin) vs the best CPU setup | ✅ PASS | Same answer on both sides; GPU breaks even at about 2 million cells and is 1.13 times faster at 4.75 million. Capped near 1.5 times because the pressure solve is only a third of each iteration | [view](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_3d_re20/gpu_study.html) |
| 6 | Heated pipe, Re = 200, uniform wall heat flux | Steady, first heat transfer, exact solution (thermal entrance series) | ⚠️ WARN | Local Nusselt number within 0.005 % RMS of the exact solution; fully developed Nu 0.011 % off, just outside the agent's own 0.01 % rule; gravity tested with a 3D run | [view](https://tomyud1.github.io/agentic-cfd/results/heated_pipe_re200/report.html) |

**PASS** means every check passed. **WARN** means the answer is usable but a check the agent set for
itself, before seeing the results, did not fully pass. **FAIL** means a requested output missed the
accuracy rule fixed before the runs. The report says which check and why. Verdicts are never
softened.

When a case is solved, add one row here and one card to the gallery ([`index.html`](../index.html)),
remove it from [BACKLOG.md](BACKLOG.md), and note it in the [CHANGELOG](../CHANGELOG.md). Once the
table passes about 20 rows, split it into sections by physics (laminar, heat transfer, turbulence,
...).

## What is in every report

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
| Appendix | Optional studies, such as performance (case 4: MPI scaling, linear solvers, what limits speed) |

Some cases add a separate study report, such as the GPU vs CPU study of case 5 (what is compared,
the method, results per mesh size, the same-answer check, where the time goes, the insights).

## What is in a case folder

```
cases/<case>/
  problem.yaml                the problem statement (the only input the agent gets)
  reference/                  benchmark data with citations (when the case has a benchmark)
  numerics.yaml               the agent's numerical plan (meshes, time steps, tolerances)
  decisions.yaml              every choice, with why / effect / alternatives
  insights.yaml               the live insight log
  template/                   the OpenFOAM case (0/, constant/, system/) with {{ placeholders }}
  theory.py / analytical.py   theory (SymPy)
  run.py, analyze.py          run the simulations; analyse and write the report
  sketch.py                   the problem sketch
results/<case>/               report.html, results.json and the extracted run data (runs/)
```

The agent copies the structure of these folders, not their numbers.

## Give the agent a new problem

1. Open this repository in a coding agent ([Claude Code](https://claude.com/claude-code), Codex or
   any other) and have it start from [AGENTS.md](../AGENTS.md).
2. Write `cases/<new_case>/problem.yaml` in the same format as the existing cases: a short
   description, a `setup` list, the `given` data, the `quantities_of_interest`, and optionally a
   `reference/` folder with benchmark data and its citation.
3. Ask: *"Use the agentic-cfd workflow in `.claude/skills/agentic-cfd/SKILL.md` to solve
   `cases/<new_case>/problem.yaml`."*

The agent reads the guidelines and the worked examples, logs insights as it goes, and finishes with
`results/<new_case>/report.html`. The cases here used Claude Opus with high effort; a case took from
about 25 minutes (case 3) to about 2 hours (case 5, including the GPU study).

To re-run a solved case, follow the commands in its report's **Reproducibility** section (machine
setup: [setup/README.md](../setup/README.md)).
