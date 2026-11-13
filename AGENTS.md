# Solve a CFD problem with this repository

For any coding agent (Claude Code, Codex or another) that a user asks to solve a fluid-flow problem
with this workflow and these tools.

1. **Know the machine:** read its file in [`workflow/machines/`](workflow/machines/README.md)
   (hardware, paths, how to launch commands). If the machine has none, install with
   [`setup/README.md`](setup/README.md) and write one; `dev-pc.md` belongs to the development PC.
2. **Follow the workflow** in [`workflow/WORKFLOW.md`](workflow/WORKFLOW.md) and its `references/`
   folder. Read it in full before starting. Claude Code also finds it as the `agentic-cfd` skill.
3. **Use the tools** in [`cfdlib/`](cfdlib): OpenFOAM, the GPU solver gpucfd (`cfdlib/gpu.py`),
   verification, figures and the report. The solved cases in [`cases/`](cases/README.md) are worked
   examples: copy their structure, not their numbers.
4. **Get the problem from the user.** If they describe it in words, write
   `cases/<name>/problem.yaml` from their description, in the format of the solved cases: physical
   inputs only, no numerical choices. Ask for anything essential that is missing, show them the file,
   and start only once they confirm it; from then on it is the given problem. A problem needs:
   - what to find (the outputs wanted, and how accurate they must be, if known);
   - the geometry and its dimensions;
   - the fluid and its properties (density, viscosity; for heat transfer also conductivity and
     specific heat);
   - the conditions at every boundary (inlet, outlet, walls) and, for a flow that changes in time,
     the starting state;
   - optionally, reference data to compare against (an exact solution, a benchmark, measurements)
     with its source.
5. **Deliver** `results/<name>/report.html`, with every choice documented and every number computed
   by code.
6. **Leave your experience behind.** Log every problem, surprise and retry as an insight while you
   work (`cfdlib.insights`). Do not change the shared tools or the workflow yourself: say what
   should change in the insight. That log is how the next agent starts better off than you did.

Developing this repository itself (its tools, gpucfd, the workflow)? Read [`CLAUDE.md`](CLAUDE.md),
the project's direction and working rules, and [`dev/LOOP.md`](dev/LOOP.md), how a solved case turns
into improvements.
