---
name: agentic-cfd
description: Solve a fluid-flow problem end to end with OpenFOAM (or the GPU solver gpucfd) and produce a verified HTML report (problem, assumptions, theory, decisions, pre-flight test, simulation, comparison, insights). Use when given a CFD problem statement in cases/<name>/problem.yaml, or when a user describes a flow problem in their own words and wants it simulated, analysed or reported on (the problem statement is then written with them first).
---

# Agentic CFD workflow (pointer)

The workflow is not specific to Claude, so it lives in an agent-neutral folder that any coding agent
opens by path. This file only lets Claude Code find it as a skill.

Read these now, in full, and follow them:

1. [`workflow/WORKFLOW.md`](../../../workflow/WORKFLOW.md): the workflow, stage by stage.
2. [`workflow/references/environment.md`](../../../workflow/references/environment.md): how to run
   things, and which machine file in `workflow/machines/` describes the computer you are on.

The other references (`openfoam-primer.md`, `report-contract.md`, `gpu-linear-solvers.md`) are named
in the workflow at the stage where they are needed.
