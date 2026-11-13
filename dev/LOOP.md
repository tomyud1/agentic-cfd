# The improvement loop

How one CFD problem makes the project better. Every case goes through this loop, so it is written
down once here. For the agent that develops the repository (the **dev agent**); the agent that
solves the problem (the **solver agent**) never needs this file.

```
            problem with a known answer
                       │
   dev agent ── 1 choose ── 2 prepare ── 3 launch ──▶ solver agent (fresh, any model)
       ▲                                                   │ follows workflow/, uses the tools,
       │                                                   │ logs insights live
       │                                                   ▼
   7 record ◀── 6 regress ◀── 5 improve ◀── 4 review ◀── report + insights
   (catalogue,   (old reports   (tools first,   (is the report right?
    changelog)    still build)   then guidance)   what cost time?)
```

## Two roles, kept apart

- **Solver agent.** Starts with no memory of the project. It gets one problem, the workflow
  ([`workflow/WORKFLOW.md`](../workflow/WORKFLOW.md)) and the tools. It does not change `cfdlib/`,
  `workflow/`, `gpucfd/` or other cases: what it needs changed, it logs as an insight. This is what
  makes a case a fair test of the workflow.
- **Dev agent.** Knows the project. It prepares the problem, reviews the result and changes the
  workflow and tools. It does not solve the case, and it does not edit the solver agent's case folder
  afterwards: that folder is the record of what a fresh agent did with the workflow of that day.

## The steps

1. **Choose** the next case from [`cases/catalogue.yaml`](../cases/catalogue.yaml). It should test
   one thing the workflow has not faced. Improve a tool first if you already know the case will be
   slow or stuck without it: a fresh-agent run is the expensive step.
2. **Prepare.**
   - Write `cases/<name>/problem.yaml`: physical inputs only, no numerical choices.
   - Put the reference data in `cases/<name>/reference/`, transcribed from the original source,
     with the citation. Never from memory, never from a second-hand page.
   - Tool-gap check: can the tools read this solver's log and dictionaries, and extract the
     requested outputs? Fix gaps now.
   - Commit the problem before launching, so the review can show it was not changed.
3. **Launch** a fresh solver agent with only this (other agents cannot load Claude's skill, so the
   prompt names files by path):

   > Solve one CFD problem with this repository. Start from `AGENTS.md`, then read
   > `workflow/WORKFLOW.md` in full and follow it. The machine is `workflow/machines/<machine>.md`.
   > The problem is `cases/<name>/problem.yaml`; it is given, do not change it. Do not modify
   > `cfdlib/`, `workflow/`, `gpucfd/` or other cases: log what should change as insights. Deliver
   > `results/<name>/report.html`. When done, report the verdict, the wall-clock time, and anything
   > you could not do.

   No hints about the method or the answer. Note the model, its effort setting and the start time.

   The same holds when you hand part of the development work to another agent (a tool to build, a
   study to run): if you limit what it may change, ask it to send back what it learned and what it
   would change outside those limits. Each item then gets an answer as in step 5.
4. **Review** the result. First `bash dev/review_case.sh <name>` (contract, leftover processes,
   problem unchanged, shared files untouched, verdict, insight counts). Then read the report as the
   supervisor would:
   - Is the problem solved the problem given? Are departures flagged as assumptions?
   - Do the reference values in the report match `reference/`?
   - Recompute two or three headline numbers yourself from the raw data in `results/<name>/runs/`.
   - Verification: is the observed order believable, and were the accuracy rules fixed before the
     production runs? Do they discriminate (not 100× looser than what was reached, not tighter than
     the method can show)?
   - Is each explanation checked against the data (sign, size, scaling)?
   - Is the verdict fair? Never soften it. If the review finds the verdict wrong, say so in the
     changelog; do not edit the report.
   - Open the report in a browser and look at every section.
   - Note the cost: wall-clock minutes, tokens, how often a human had to step in, and where the
     time went (case 6: about 40 of 96 minutes went to tool problems).
5. **Improve.** Every insight gets an answer, in this order of preference:
   1. **A tool or a check.** A lesson that code enforces costs the next agent nothing to read and
      cannot be forgotten (case 6: `foam check` now reports a silently ignored SIMPLEC).
   2. **Guidance**, in the most specific file that fits: a topic reference before the workflow
      itself. Offer the tool and its trade-off, and leave the choice to the agent; keep firm rules
      for correctness only. Do not oblige future agents to read or do something because one case
      needed it.
   3. **Nothing new**: the case stays as a worked example, or the insight is declined, with the
      reason.

   Test each tool change against recorded runs (replays and trap tests in `tests/test_tools.py`; the gpucfd run tools in
   `tests/test_gpu_tools.py`, which needs the GPU).
6. **Regress.** `bash setup/rerender_all.sh` rebuilds every report from saved data and checks its
   contract; run `tests/`. A change that breaks an old report is not finished.
7. **Record.** Two commits, so the case stays visible exactly as the fresh agent left it:
   - the case: set `status: solved` in `catalogue.yaml`, run `python -m cfdlib.catalogue`;
   - the improvements: tools, guidance, the case's `shows` line in the catalogue (what a later
     agent can learn from its folder), and the
     [`CHANGELOG.md`](../CHANGELOG.md) entry saying what each insight led to.

## What the loop does not do yet

Planned in [`ROADMAP.md`](../ROADMAP.md): a record of what became of each insight, cost and agent
per case in the catalogue, replaying an old case to measure whether the workflow got better, and
blind cases where the solver agent does not see the reference.

## Other things in `dev/`

- `review_case.sh`: the scripted part of step 4.
- `exploration/`, `steady_speed/`, `agent_checks/`: one-off experiments and migrations behind
  earlier changes, kept as a record. Their paths are those of the development PC.
