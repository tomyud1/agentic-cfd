# Agentic CFD: project direction

Read this first in every session. It says what the project is for, so that work on any part of the
repository pulls in the same direction. It is for developing the repository; an agent that only uses
the workflow to solve a user's CFD problem starts from [`AGENTS.md`](AGENTS.md).

## The goal

**Automate the work of a CFD engineer as far as possible with an AI agent, and build the tools that
agent needs.**

Final-year project, Faculty of Mechanical Engineering, Technion, supervised by Prof. Steven Frankel.
The student is Tomer (GitHub `tomyud1`).

Three parts:

1. **The workflow** ([`workflow/`](workflow/WORKFLOW.md)). A CFD engineer gets a problem and must
   test it and write a report. The workflow takes an agent from the problem statement to a verified,
   checkable report: every choice documented, every number computed by code, verdicts never
   softened. It carries the experience of every agent that solved a case before. This is the
   product.
2. **The tools** ([`cfdlib/`](cfdlib), OpenFOAM, [`gpucfd/`](gpucfd), and future tools). Everything
   the agent uses between the problem and the report. We extend OpenFOAM's tools, tune them, or
   build new ones, aiming for the best in quality and speed. OpenFOAM gives breadth (any geometry,
   turbulence, heat transfer, multiphase). gpucfd is one tool: a GPU-resident solver, very fast on
   the problems it covers, not a general replacement for OpenFOAM. Better tools are a main goal
   in their own right, not only a service to the workflow: open-source GPU flow solvers are scarce,
   and the supervisor and industry both take a strong interest in them. A tool is judged by
   measured quality and speed on real problems, compared fairly.
3. **The agent.** Whichever coding agent the user chooses: Claude here, a local model on another
   student's DGX Spark, something else later. So the workflow lives in an agent-neutral folder,
   machine facts live in one file per machine ([`workflow/machines/`](workflow/machines/README.md)),
   and nothing may depend on one agent's harness or one computer's paths.

**How they improve: the loop** ([`dev/LOOP.md`](dev/LOOP.md)). The loop is a method, not the
goal. The goal is to solve CFD problems well and to have tools that are better and faster. The
method is to throw verified CFD problems at the project: a fresh solver agent takes a problem with a
known answer and solves it with the workflow and tools, logging every limitation it meets. The dev
agent (usually Claude) then reviews the problem, the report, its verification and the
insights, and improves the workflow and the tools. In a session you are one of the two: the solver
agent starts from [`AGENTS.md`](AGENTS.md), the dev agent from this file.

Where the project is going next, in order: [`ROADMAP.md`](ROADMAP.md).

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
- **Give the agent agency.** Do not force a solver agent to behave in a set way or to digest
  information unless correctness truly needs it. Offer tools, records and trade-offs, say where
  they are, and let the agent decide what to use and what to read. The solved cases are a library
  to consult, not a reading list. Firm rules are for correctness only (code computes every number,
  stopping is physical, verdicts are never softened).
- **Keep the workflow lean.** Every case adds lessons, and an agent reads the workflow on every
  case. Turn a lesson into a tool or a check when you can; otherwise put the text in the most
  specific reference that fits. Nothing machine-specific or Claude-specific goes in
  `workflow/WORKFLOW.md`.
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
- [`ROADMAP.md`](ROADMAP.md): the next steps, in order, and what is deliberately not next.
- [`dev/LOOP.md`](dev/LOOP.md): the improvement loop, step by step: prepare a case, launch a fresh
  agent, review, improve, regress, record.
- [`workflow/WORKFLOW.md`](workflow/WORKFLOW.md): the workflow the solver agent follows for each
  problem, with its references in `workflow/references/`. `.claude/skills/agentic-cfd/SKILL.md` is
  only a pointer to it, so that Claude Code finds it as a skill.
- [`gpucfd/DESIGN.md`](gpucfd/DESIGN.md): the GPU solver's plan, verification ladder, status,
  limits and comparison with OpenFOAM.
- Machines: [`workflow/machines/`](workflow/machines/README.md), one file per computer
  (`dev-pc.md` has this PC's paths and WSL details).
