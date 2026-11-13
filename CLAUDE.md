# Agentic CFD: project direction

Read this first in every session. It says what the project is for, so that work on any part of the
repository pulls in the same direction. It is for developing the repository; an agent that only uses
the workflow to solve a user's CFD problem starts from [`AGENTS.md`](AGENTS.md).

## The goal

**Automate the work of a CFD engineer as far as possible with an AI agent, and build the tools that
agent needs.**

Final-year project, Faculty of Mechanical Engineering, Technion, supervised by Prof. Steven Frankel.
The student is Tomer (GitHub `tomyud1`).

Three layers, in order of importance:

1. **The agent workflow** ([`.claude/skills/agentic-cfd/`](.claude/skills/agentic-cfd/SKILL.md)).
   It takes the problems CFD engineers actually get, and goes from problem statement to a verified,
   checkable report. Every choice is documented, every number is computed by code, and verdicts are
   never softened. This is the product.
2. **The tool arsenal** ([`cfdlib/`](cfdlib), OpenFOAM, [`gpucfd/`](gpucfd), and future tools). The
   agent picks whichever tool fits the problem. OpenFOAM gives breadth (any geometry, turbulence,
   heat transfer, multiphase). A tool is judged by how much it helps the agent, and possibly human
   engineers, solve real problems faster and more reliably.
3. **gpucfd** is one such tool: a GPU-resident solver that is very fast on the problems it covers.
   It should become a fast, verified backend for the agent. It is not a general replacement for
   OpenFOAM. Grow it where it gives the agent a real advantage, and wire it into `cfdlib` so the
   agent can choose it, and document that choice like any other decision.

Making the process itself faster and better is part of the project: speeding up simulations, the
workflow and the tools (gpucfd started this way), and building new tools where the existing ones
fall short.

The project began with the supervisor's question of whether OpenFOAM can run on GPUs. He is now
enthusiastic about all of it: the workflow, the reports and gpucfd. Measure every comparison fairly
(same problem, same grid, each side with its best settings) and state its limits.

## Bottlenecks to work around

- **Compute.** Development and tests run on one desktop: an RTX 4070 with 12 GB and FP64 at 1/64 of
  the float rate, a Ryzen 7 7800X3D, and WSL2. Long runs slow everything down. Prof. Frankel's lab
  is expected to give access to a bigger machine with its own local agent in the week after
  2026-11-12 (specs not known yet). Work will then move between machines through git, so keep setup
  steps and paths portable, not tied to this PC.
- **Usage budget.** The student's Claude plan has a small weekly limit (65% of a week's usage went in
  two days with one agent), so only one agent runs at a time. Wasted turns cost days of progress, not
  just minutes.
- **Expertise.** The student is new to CFD and is learning as the project goes. The supervisor is
  the domain expert. His tests and criticism of the workflow are the most valuable input the project
  can get.

## Working rules that follow from this

- **Decide when a run ends before you start it.** Know the stopping criterion, the expected result
  and what counts as pass or fail. Write them down, then run.
- **Watch long runs live.** Stream their output (for example with a Monitor on the log) and check
  residuals, forces and conservation as they go. Stop early when the answer has settled or is
  clearly wrong. Don't wait for a run to finish before looking at it.
- **Every rerun should answer a new question.** Speed work reruns the same case on purpose: the
  same numbers, faster, is the proof that the change is right. A rerun that changes nothing, or
  whose answer is already known, is waste. Before killing a running job, check whether its result is
  still needed. Run one GPU job at a time.
- **Keep verified work safe.** Each verified milestone should get its own commit, separate from
  work in progress. Commit and push only when the student asks.
- **Keep the changelog current.** Every commit that adds or changes something meaningful also
  updates [`CHANGELOG.md`](CHANGELOG.md) in that same commit. Add a dated entry, newest first, in the
  same format: what, why it matters, how it was verified, key numbers, open problems. Fill in the
  commit hash on the next change. List significant work that isn't committed yet under "In
  progress". Write it in plain language: the student uses it to report progress to the supervisor.
- **Keep the README a front door, not a history.** It shows the current state only: what the project
  is, what it can do, and where to go next. It has no table of cases. When something changes,
  replace the affected line or bullet instead of adding sentences to it. No counts that need
  updating by hand ("five cases", "61 insights"). Detail goes in the linked pages: setup in
  `setup/README.md`, history in `CHANGELOG.md`, gpucfd details in `gpucfd/DESIGN.md`. Keep it under
  about 150 lines: a new section must replace or shorten something.
- **Every case lives in one list,** [`cases/catalogue.yaml`](cases/catalogue.yaml): solved, in
  progress and planned. The solved-case table, the backlog and the report gallery are generated
  from it by `python -m cfdlib.catalogue`; never edit those tables by hand. When a case is solved,
  set `status: solved` with its headline and report, run the script, and commit the result with the
  case. `tests/test_catalogue.py` fails if a table is out of date or a report is missing from the
  list.
- **Explain as you go.** Explain the physics and numerics behind each choice in plain terms, so the
  student can follow, check and present the work.
- **Be honest.** Report failures, fair comparisons and limits in full. The supervisor is a CFD
  expert, and one overstated claim costs more trust than a FAIL verdict.

## Where things are

- GitHub: [github.com/tomyud1/agentic-cfd](https://github.com/tomyud1/agentic-cfd) (public; the
  reports are served by GitHub Pages at
  [tomyud1.github.io/agentic-cfd](https://tomyud1.github.io/agentic-cfd/)).
- [`CHANGELOG.md`](CHANGELOG.md): what changed and when. Read it to catch up on recent work.
- [`README.md`](README.md): the front door, with a "Start here" table of links.
- [`cases/catalogue.yaml`](cases/catalogue.yaml): every case, solved and planned; the source of the
  tables in `cases/README.md` (solved cases, verdicts, reports) and `cases/BACKLOG.md`.
- [`setup/README.md`](setup/README.md): install, re-create the reports, re-run cases, build gpucfd.
- [`cases/BACKLOG.md`](cases/BACKLOG.md): the next problems to give the agent, in order, and the
  full catalogue of candidates (NASA TMR, the Luminary V&V manual).
- [`.claude/skills/agentic-cfd/SKILL.md`](.claude/skills/agentic-cfd/SKILL.md): the workflow the agent
  follows for each problem.
- [`gpucfd/DESIGN.md`](gpucfd/DESIGN.md): the GPU solver's plan, verification ladder, status,
  limits and comparison with OpenFOAM.
- Machine setup and WSL details:
  [`.claude/skills/agentic-cfd/references/environment.md`](.claude/skills/agentic-cfd/references/environment.md).
