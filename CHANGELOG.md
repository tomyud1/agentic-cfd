# Changelog

What changed in the project and when, newest first, in plain language. Read this to catch up on
recent work. Details live in the linked commits, reports and [`gpucfd/DESIGN.md`](gpucfd/DESIGN.md).

Each entry says **what** changed, **why it matters**, **how it was verified**, the key numbers, and
the commit. Open problems are marked **Open**. Work not yet committed goes under "In progress", and
moves to a dated entry when it is committed.

## In progress

What comes next, in order, is in [`ROADMAP.md`](ROADMAP.md).

## 2026-11-13

### Repository aligned with the goal: workflow, tools, agent, and the loop, commit hash to be filled in
- **What:**
  - **The workflow moved to an agent-neutral folder.** `.claude/skills/agentic-cfd/SKILL.md` is now
    [`workflow/WORKFLOW.md`](workflow/WORKFLOW.md), with its references in `workflow/references/`.
    The skill file that Claude Code loads is a short pointer to it. The text of the workflow is
    unchanged except for the machine-specific phrases.
  - **Machine facts are split from the workflow.** The old `environment.md` mixed three things. They
    are now three files:
    - [`workflow/machines/dev-pc.md`](workflow/machines/dev-pc.md): this PC's hardware, paths, limits
      and the WSL launch commands;
    - [`workflow/references/environment.md`](workflow/references/environment.md): what holds on every
      machine;
    - [`workflow/references/gpu-linear-solvers.md`](workflow/references/gpu-linear-solvers.md): the
      OGL notes from case 5.

    [`workflow/machines/README.md`](workflow/machines/README.md) says how to describe a new machine.
  - **The improvement loop is written down:** [`dev/LOOP.md`](dev/LOOP.md). It has the two roles
    (solver agent, dev agent), the seven steps from choosing a case to recording it, the launch
    prompt, and the review checklist. Until now it existed only in the changelog's history.
  - `dev/review_case.sh <case>` replaces the three per-case review scripts: contract, leftover
    solver processes, problem and reference unchanged, shared files untouched, verdict, insights.
  - Tests and setup scripts no longer contain this PC's paths (`tests/smoke_*.sh`,
    `tests/test_tools.py`, `setup/fetch_featflow.sh`): they find the repository from their own
    location and the run folder from OpenFOAM's `$FOAM_RUN`.
  - CLAUDE.md states the goal as three parts (workflow, tools, agent) and the loop, and has a new
    rule: keep the workflow lean. README, AGENTS.md and `setup/README.md` follow.
  - **Solved cases are a library, not a reading list.** The workflow no longer tells the agent to
    read all files of every solved case. Each solved case has a `shows` line in
    `cases/catalogue.yaml` (what its folder teaches), and `cfdlib/catalogue.py` generates the list
    "What each solved case shows" in [`cases/README.md`](cases/README.md). The agent decides what
    to open. CLAUDE.md has the rule behind it: give the agent agency.
  - New [`ROADMAP.md`](ROADMAP.md): the next steps in order, what was decided, and the questions
    for Prof. Frankel.
- **Why it matters:** the lab machine and a second student's local agent are about to use this
  repository. Before this change the workflow sat in a Claude-only folder, told every agent to use
  one user's home directory, and the way a solved case turns into improvements was not written
  anywhere.
- **Not changed:** `cfdlib/`, `gpucfd/`, `cases/` and `results/` stay where they are. The commands
  printed in the published reports depend on them.
- **How it was verified:**
  - `tests/test_tools.py`, `test_catalogue.py`, `test_layout.py`, `test_study.py` and both smoke
    scripts pass with the portable paths.
  - Every relative link in the Markdown files resolves (19 files, 0 broken).
  - `dev/review_case.sh heated_pipe_re200` runs clean: contract OK, no leftover solver, WARN, 16
    insights, none unresolved.
  - **Clean copy:** the files a `git clone` would contain (613) were copied to a new folder with a
    different path. There `setup/rerender_all.sh` rebuilt all six reports and the GPU study with
    every contract OK and the same verdicts, and the tool, catalogue and layout tests passed.
  - No simulation was re-run and no report changed.
- **Open:**
  - The loop in `dev/LOOP.md` describes how cases 2-6 were handled. No fresh agent has been launched
    with the new launch prompt or the pointer skill file yet; case 7 will be the first.
  - `tests/smoke_all.sh` rebuilds three reports and leaves them changed in git, although their
    content is the same: matplotlib stamps each figure with the time and random ids. Restore them
    with `git checkout -- results` after running it (a fix is listed in ROADMAP under "Measuring
    the method").

### Everything case 6 exposed, fixed: steady stop, process safety, solver check, guidance (`3cf874b`)
- **What:** all 16 insights of case 6 were acted on.
  - **Steady stop (insights 9, 15):** `monitor.steady_check` now handles values converged to their
    printed digits. When the fitted geometric estimate has no answer, two conservative fallbacks are
    tried:
    - a block bound on summed changes, with a safety factor of 10;
    - the output precision, for a value that stopped drifting after it had moved.
  - **Two older weaknesses, found by trap tests while doing this:**
    - The estimate is multiplied by 10 while the decay is still slowing down. A fast mode dying out
      had made it 18× too small.
    - A value that has never changed no longer decides the stop. A probe the solution has not
      reached yet had counted as converged.
    - `verification.iteration_error` itself is unchanged, so gpucfd's steady runs and case 4's
      periodicity rule behave as before.
  - **`foam.map_fields` (insight 7):** it deletes the fields mapFields adds (C, Cx, V ...), which had
    made the contract refuse the report.
  - **Process safety (insight 10):**
    - `foam.stop` also kills a solver whose Python wrapper died, found by its working directory.
    - `foam.render_case` refuses to delete a case directory with a live solver in it.
  - **Ignored settings (insight 3):** new `foam.ignored_settings` and `python -m cfdlib.foam check`
    look in the solver's source code for whether it reads SIMPLE/PIMPLE `consistent`. The pre-flight
    reports it, so the silent SIMPLEC of `buoyantBoussinesqSimpleFoam` is now caught.
  - **Report template (insight 16):** inline math in the "What we want to find" list keeps its
    spaces.
  - **Re-rendering:** `CFD_RERENDER=1`, set by `setup/rerender_all.sh`, keeps each report's original
    date and machine record. Before, a re-render stamped today's date and today's WSL memory on old
    cases. All six reports were re-rendered with the template fix. Case 4's report now includes its
    insights 17-19, which were logged after it was first rendered.
  - **Guidance:**
    - The primer has a new heat-transfer section: solver choices, no SIMPLEC in the buoyant solvers,
      wall-flux face values, the zero-gradient outlet artifact, and the wedge bias. It also has
      pitfalls 15 (absolute tolerances below round-off) and 16 (mapFields), and the pressure-solver
      ranking flip.
    - SKILL.md has case 6 as a worked example, plus notes on series truncation, outputs that are
      limits, error budgets with extraction bias, same-mesh on/off tests, refinement before
      attribution, safe process control, a steady pre-flight to convergence, and wide equations in
      the visual review.
    - SKILL.md stage 0 (asked for by the docs session): a user who describes a problem in words gets
      a `problem.yaml` written with them and confirmed before work starts. The skill's description
      now says so.
- **Why it matters:** about 40 of case 6's 96 minutes were lost to these problems. On case 6's own
  recorded runs, the fixed stop would have ended:

  | Run | Stop now | Rows the run actually went to |
  |---|---|---|
  | L1 | 91 | 3736 |
  | L2 | 181 | 3690 |
  | L3 | 441 | 1319 |
  | L4 | 1611 | 3291 |

  Every stop has a true remaining error below the tolerance.
- **How it was verified:**
  - `tests/test_tools.py`:
    - replays case 2's four and case 6's four recorded meshes, asserting the stop row and the true
      error;
    - trap tests for two decay modes, a one-way drift at the last digit, flicker, and a probe
      reached late;
    - a fake solver process for the render guard and for `stop`;
    - the SIMPLEC check on both solvers;
    - `map_fields` on case 2's runs.
  - `test_study.py` and `test_layout.py` pass. All six contracts pass after the re-render.
  - In the browser, the space before each equation in the requested list renders (4.4 px, was 0).
- **Open:**
  - A slow mode hidden under a faster one can still fool any stop rule that only looks at the
    history so far (documented in `iteration_error`). The 10× factor and `hold` reduce the risk;
    a coarser mesh's rate (`rho_min`) removes it.
  - The case-local workaround in `cases/heated_pipe_re200/run.py` stays, as the record of what the
    agent did.

### One case catalogue generates the case tables; AGENTS.md is for users' own problems (`84037c5`)
- **What:**
  - New [`cases/catalogue.yaml`](cases/catalogue.yaml): one list of every case, solved, in progress
    and planned. It holds the 7 solved reports, the 10 tiered cases (7-16), 4 "later" items and 40
    more candidates from the Luminary V&V manual (most RANS ones are also NASA TMR cases), grouped
    by physics.
  - New [`cfdlib/catalogue.py`](cfdlib/catalogue.py) generates the solved-case table in
    `cases/README.md`, the tables in `cases/BACKLOG.md` and the cards of the report gallery
    (`index.html`). Verdicts are read from each report's own JSON file, never typed by hand.
    `--check` and `tests/test_catalogue.py` fail if a table is out of date, or if a results folder
    has a report that the catalogue does not list as solved.
  - AGENTS.md is now for an agent that a user asks to solve their own CFD problem: set up, follow the
    skill, use the tools, write `problem.yaml` from the user's description and have the user confirm
    it, deliver the report. Agents developing the repository read CLAUDE.md.
  - "Give the agent a new problem" (`cases/README.md`): describe the problem in words and list what
    to give; the agent writes `problem.yaml` itself. The model is named: Claude Opus 5.5, high effort.
- **Why it matters:** with 50 or more cases planned, three hand-kept lists (the case table, the
  backlog, the gallery) would drift apart. Now a solved case is one edit to `catalogue.yaml` and one
  command, and the test catches a forgotten one.
- **Verified:** the generator reproduces the existing tables and cards with the verdicts read from
  the result files (2 PASS, 3 WARN, 1 FAIL, plus the 5b study PASS); `--check` passes after a run.
- **Open:** the workflow's intake step (SKILL.md) should also say how to write `problem.yaml` from a
  description in words; it is in AGENTS.md for now.

### README split into a front door and two "start here" pages (`f52a4f1`)
- **What:**
  - The README no longer has a table of cases. It now has a "Start here" table of links, a short
    results paragraph, the gpucfd summary, how it works and a top-level repository map (222 to 117
    lines).
  - New [`cases/README.md`](cases/README.md): every solved case with its verdict, headline and
    report, what is in every report, the case folder layout, and how to give the agent a new problem.
  - New [`setup/README.md`](setup/README.md): install, re-create the reports, re-run a case, view a
    report, and build gpucfd and the OGL plugin (with a note on setting the GPU architecture).
  - CLAUDE.md's README rule: no case table and no hand-kept counts in the README, and a budget of
    about 150 lines. AGENTS.md and BACKLOG.md point to the new pages.
- **Why it matters:** with 50 or more cases planned, a table in the README would bury the project's
  front page. Now a new case adds one row to `cases/README.md` and one card to the gallery, and the
  README stays the same. `setup/README.md` is also the single file to follow when setting up the lab
  machine.
- **Verified:** every link target exists; the old README had already drifted ("the five solved
  cases" and "61 entries over cases 1-5" after case 6), and those counts are gone.

### Case 6 solved by a fresh agent: heated pipe, WARN (`9f11a67`)
- **What:** the first heat-transfer case. A fresh agent got only the problem statement, the tools
  and the guidelines.
  - It derived the exact solution itself: the thermal-entrance series with 80 eigenfunctions,
    checked against its own finite-volume solution to 4e-6.
  - It solved the case with `buoyantBoussinesqSimpleFoam` on axisymmetric wedge meshes L1-L4
    (up to 80 × 640 cells).
  - It ran a 3D half-tube pair, with and without gravity, to test buoyancy.
  - It logged 16 insights.
- **Result:** WARN, because two of the agent's own rules, fixed before production, missed narrowly:
  - fully developed Nu is +0.011 % from 48/11, against a 0.01 % limit (+0.008 % of that is the
    bias of its own tail fit, measured on the exact solution);
  - the observed order of the Nu(x) error is 1.42 from L3 to L4, against 1.5 to 3. It is 1.6
    downstream and 1.25 near the start of heating, where the inflow profile and axial conduction
    differ from the idealised exact solution; that was not separated by a run.
- **Key numbers** (finest mesh):
  - Nu(x) within 0.005 % RMS of the exact solution.
  - Entrance length 60.07 D against 60.11 D (x⁺ = 0.0431).
  - Outlet wall temperature within 1.3e-5 K.
  - Energy balance better than 1e-9.
  - Gravity changes Nu by at most 0.15 %, but makes the top of the outlet wall 0.034 K warmer than
    the bottom (11 % of the wall-to-bulk difference).
- **How it was checked (dev review):**
  - The contract passes.
  - Only the case and results folders changed.
  - The outlet bulk and wall temperatures and x⁺ agree with an independent hand calculation.
- **Cost:** 96 minutes of agent time.
  - Simulations took 29.5 min.
  - About 40 min went on runs it discarded and on the steady stop holding runs past convergence
    (insights 8-10 and 15). Those are fixed next.
- **Open:** in the problem design, the outlet at 120 D is still 0.53 % above the fully developed
  Nu, so the agent had to extrapolate it (insight 1). That is a fair test, but worth knowing.

### Case 6 problem statement; the contract checker reports files it cannot read (`e0fd8c0`)
- **What:**
  - `cases/heated_pipe_re200/problem.yaml`: water at 20 °C in a 4 mm tube, Re 200 (Pr 7, so
    Pe 1400). It has a 20 D unheated section, then 120 D with a uniform wall heat flux of
    200 W/m². It asks for the local and fully developed Nusselt number, the thermal entrance
    length, and the wall and bulk temperatures. It is a verification case: the agent derives the
    exact solution itself.
  - `contract.uncovered_settings` now reports an OpenFOAM dictionary it cannot parse. Before, such
    a file was skipped silently, so its settings were never checked against the decisions.
- **Why it matters:** case 6 is the first heat-transfer problem, and its new files (`0/T`,
  thermophysical properties, gravity) must be covered by the checker like every other setting.
- **How the problem was chosen:**
  - The numbers are computed by code. The wall-to-bulk temperature difference is 0.31 K and the
    bulk temperature rises 0.38 K, so Ri ≈ 1e-3, Ra ≈ 280 and the viscosity changes by about 1.6 %.
    Constant properties and no buoyancy should therefore hold, but the agent has to show it.
  - The calming section is twice the hydrodynamic entry length (about 10 D), so the flow should be
    developed where the heating starts.
  - The heated length is about 1.7 times the usual thermal-entry estimate (about 70 D), which
    leaves a fully developed stretch.
- **How it was verified:**
  - The parser reads all 35 heat-transfer and scalar-transport tutorials of OpenFOAM v2606, every
    file.
  - The log parser and the steady stop do not depend on the field names.
  - The contracts of cases 1-5 still pass. A new test checks that an unreadable file is reported.

## 2026-11-12

### Steady OpenFOAM runs reach the same answer 3 to 11 times sooner (`26d0d82`)
- **What:**
  - `monitor.steady_decider` is a LiveStop rule for steady runs. It stops once the estimated
    iteration error of every monitored value is below a tolerance, instead of on residual targets.
    `monitor.steady_check` and `monitor.steady_replay` are its parts, for live use and for testing
    on recorded runs.
  - `foam.map_fields` starts a mesh from a coarser mesh's answer (grid sequencing). It refuses
    paths with a space, which OpenFOAM cannot use.
  - The primer has a new section with the measurements, and SKILL.md stage 4 offers the tools.
- **Why it matters:** the next cases (heated pipe, buoyant cavity, the steady cylinder, the
  backward-facing step) are steady. In case 2, iterations grew 3.6× per mesh refinement, so a
  512² mesh would have taken about 2 hours on one core. The cause is under-relaxation: it acts as a
  pseudo time step that shrinks like h² where viscosity dominates.
- **How it was verified:**
  - Replay on case 2's four recorded meshes, 16 combinations: the true remaining error was always
    below the estimate, at 0.6 to 0.97 of the tolerance.
  - 14 dev runs of case 2 with other relaxation factors and starting fields, plus 3 pimpleFoam runs
    that diverged. Every converging run's answer is within 1e-6 U of the residual-converged run, or
    of the run's own long tail on 512².
  - A live run through the tool on 256² stopped by itself at iteration 815 (the replay predicted
    801), with fields written there, 7e-7 U from the converged answer.
  - `tests/test_tools.py` replays all four meshes and grows a probes file in pieces.
- **Key numbers** (one core, wall time to 1e-6 U):

  | Mesh | Case 2's settings | Best found |
  |---|---|---|
  | 128² | 27 s | 7.1 s |
  | 256² | 437 s (627 s as run) | 70 s |
  | 512² | about 1.9 h (extrapolated) | 581 s |

  The three levers: the error-based stop (1.5×), SIMPLEC with velocity relaxation 0.97-0.99
  instead of 0.9 (3-9× fewer iterations), and grid sequencing (up to 2.2×). Details:
  [`dev/steady_speed/`](dev/steady_speed/README.md).
- **Open:**
  - The best relaxation depends on the mesh and is not monotone (0.98 best on 128², 0.99 on 256²).
    A short side-by-side trial ranks the settings well, so the guidance offers that instead of a
    fixed number.
  - At higher relaxation each early iteration costs more (more linear-solver sweeps), so the
    time saved is smaller than the iterations saved.
  - pimpleFoam as a pseudo-transient march (Courant 5-50, one outer corrector) diverged and was
    dropped.
  - Tested on one laminar 2D flow only. The heated pipe will be the first other case.

### Monitoring guidance written as options, not rules (`1acade2`)
- **What:** SKILL.md stage 4 now offers the live stop, peek and watch as tools the agent may use as
  it judges best. It says plainly that the saved history in `<case>/peek/` is always complete and
  can be read in any part, in any way. Only the updating is incremental. How often to look, how
  much history a stopping rule uses, and whether to open the figure are the agent's call. The one
  firm rule stays: a run stops on its physical check, with the end time only a maximum.
- **Why it matters:** the agent should be guided, not boxed in. Fixed rhythms and "only read what is
  new" would take away judgement the agent needs on problems we have not seen yet.

### The live stop's check costs the same at step one million as at step one thousand (`af88528`)
- **What:**
  - New `monitor.DatTail` reads a postProcessing file incrementally. Each call parses only the lines
    added since the last one and keeps the history in a growing array.
  - Case 4's stopping rule analyses only the last 14 shedding periods. The verdict depends only on
    the last 9 cycles: 4 for the spread, 5 for the drift test, 9 for the floor test.
  - Fixed: changes that are exactly zero no longer count as a one-way drift (a division by zero on
    perfectly periodic signals).
- **Why it matters:** the check runs every 10 s beside the solver. Before, it re-read the whole force
  file and re-analysed every cycle each time, which costs more and more as the run grows.
- **How it was verified:**
  - Synthetic force history of 1,000,000 steps (80 MB): the old check took 7.2 s each time. The new
    one took 0.14 s after 20,000 new steps, with the same verdict.
  - The replay on case 4's recorded runs gives the same stop times on all four grids. Only the
    number of cycles analysed changed (M1: 18 → 13).
  - A unit test grows a file in pieces, cut mid-line, and compares with a one-shot read.

### Peek appends to the raw history instead of rewriting it, for very long runs (`f60a49e`)
- **What:** `cfdlib.monitor.sync` keeps a byte offset per source file (solver log, each
  postProcessing file) in `<case>/peek/.state.json`.
  - Each peek reads only what the run wrote since the last one, and appends those rows to the CSVs.
    While the solver runs, only complete time steps are taken.
  - Next to each CSV is a float64 twin (`.f64`) for fast loading. The summary, health and figure are
    computed from these files, not by re-parsing the log.
  - `monitor.assess` now has an array core (`assess_arrays`); its results are unchanged.
- **Why it matters:** many CFD runs are huge. Rewriting the files at every peek would make each look
  cost more as the run grows. Now the cost of a peek follows what is new, plus a fast binary load.
- **How it was verified:**
  - `assess` gives identical output, old against new, on 6 real logs: steady and transient, up to
    10,864 steps.
  - A run grown in 13 random chunks, cut mid-line and mid-step, gives the same rows as a one-shot
    parse: solver log, forces and probes, 2330 rows each.
  - A real running simulation was peeked and watched to its end.
- **Key numbers:**
  - Peek on case 4's `live_M3`: 0.35 s the first time, 0.01 s when nothing is new.
  - Loading a million rows × 10 columns: 0.034 s from `.f64`, against 2.85 s from the CSV text.
- **Open:** the live stop's own check (`read_dat_live` plus the cycle analysis) still re-reads its
  force file every 10 s. That costs a few seconds at a million steps, so it should use the same
  incremental reader when a run gets that long.

### Peek always saves the raw run history for the agent to search (`7991d62`)
- **What:** every `peek` writes the full history to `<case>/peek/`, every time step, whatever
  `--files` or `--from` show:
  - `solver_log.csv`: residuals, Courant number, continuity error and solver time;
  - one CSV per postProcessing file: forces, probes, fluxes.

  It lists the files with row counts and columns.
- **Why it matters:** the agent can grep or load exactly the values it needs instead of reading them
  off small plots. The window table stays as a quick overview. The figure (`--plot`, `watch`) is for
  when the shape matters and for showing the engineer how the run is going.
- **How it was verified:** on case 4's `live_M3`, 10 files of 2330 rows each (1.6 MB), checked
  against the log and force.dat.

### Peek gives the agent numbers; a live page shows the human engineer the figure (`b149395`)
- **What:**
  - `peek` now gives the history as numbers by default. Residuals, Courant number and each
    monitored series appear as min / mean / max over 8 equal time windows (`--windows N`).
  - `--csv DIR` writes the selected series at full resolution for the agent's own analysis.
  - The figure is now opt-in (`--plot`).
  - New `python -m cfdlib.monitor watch <case> --every 60` keeps `<case>/peek.html` up to date
    until the run ends. The page holds the figure plus the text and reloads itself in a browser, so
    the human engineer can follow the run.
- **Why it matters:** an agent reads exact numbers more reliably than a picture, and a table costs
  fewer tokens than an image. A figure is still the fastest way for a person to see a run. The
  agent can still peek whenever it judges it useful.
- **How it was verified:**
  - On case 4's `live_M1`, the table shows the shedding develop in numbers: the lift amplitude grows
    from 4e-5 to 5.2e-4 by t ≈ 3.9 s and then holds, and the residuals level off at the same time.
  - On a running scratch run, the watch page was refreshed while the run went (state "running") and
    left at "finished" when it ended.
  - The CSV export holds every time step.

### Peek: the agent can look into a running simulation whenever it chooses (`57c45a3`)
- **What:** `python -m cfdlib.monitor peek <case dir>` gives a snapshot of an OpenFOAM run from its
  first step to now:
  - the state (running, finished, crashed) and progress, with the time left;
  - health so far;
  - the live stop's last verdict;
  - every monitored series in one line each.

  It also writes a figure of the whole history: residuals, Courant number, forces, probes and other
  postProcessing output. `--files` and `--from` narrow it down. SKILL.md stage 4 says how to use it.
- **Why it matters:** the automatic live stop runs in Python beside the solver and costs no agent
  tokens. Peek is the other half: the agent decides when to look, after a minute or after hours.
  Between looks nothing runs, so a long simulation does not fill the agent's session. Everything
  is read from files the run writes anyway, so it works on any OpenFOAM case.
- **How it was verified:**
  - A scratch run was peeked twice while running and once after the end. It was seen as running,
    and the time-left estimate was right: 15 s left at 8 s; the run ended at 23 s.
  - A finished case: 1.6 s for an 8 MB log.
  - A third live-stop run, `live_M2`, stopped at 17.74 s, where the replay predicted 17.70 s. It took
    90 s instead of 97 s, with numbers within 9e-5 of the full run.
  - The live stop's progress lines now only print when a verdict changes.
- **Open:** gpucfd runs have no peek yet; they run inside Python and report through
  `cfdlib.gpu.run_transient`'s log lines.

### OpenFOAM runs stop live on their physical check (`5a064ef`)
- **What:** `cfdlib/monitor.py` gained `LiveStop`. While OpenFOAM runs, it reads the run's own output
  every 10 s and applies the stopping rule written down before the run. Once the rule holds, it
  triggers OpenFOAM's `abort` function object, which writes the current step and ends the run
  normally, in serial and parallel. The planned end time becomes only a maximum. `monitor.judge`
  and `monitor.settled` turn per-cycle values into a verdict:
  - **settled:** below tolerance, with no drift left;
  - **floor:** the changes stopped shrinking above tolerance; the run stops with a warning.

  Case 4 uses it in `run.py live` and checks it in `run.py live_check`. SKILL.md stage 4 now says to
  stop every run this way.
- **Why it matters:** a fixed end time is a guess. Too short gives a wrong answer (gpucfd G1 was 16%
  low after 3 s). Too long wastes the most time on the finest, most expensive grid.
- **How it was verified:**
  - **Replay:** the rule was first replayed on case 4's recorded OpenFOAM histories.
    - It found that **M1 never meets the periodicity rule**: its drag peaks jitter by about 7e-4 per
      cycle at 9, 12 and 15 s alike, above the 5e-4 tolerance. A plain "change below tolerance" rule
      would never stop it, hence the floor state.
    - M2, M3 and M4 meet the rule after 2.7, 2.35 and 2.0 s of their 3 s.
  - **Live runs:** two live runs passed the rules set before them.
    - **`live_M1`** (serial): stopped at 7.85 s at its floor, in 30 s instead of 56 s.
    - **`live_M3`** (4 ranks): stopped at 20.33 s with everything settled, in 211 s instead of 285 s.
      `reconstructPar` worked after the stop.
  - **Tests:** `tests/test_tools.py` checks three cases: a converging series is settled, jitter above
    tolerance is a floor, and equal small one-way steps keep running. That test caught a hole in the
    first version: a steady drift with every step below tolerance counted as settled. The fixed rule
    extrapolates a one-way trend at its average decay rate. Its replay gives the same stop times as
    the first version, which the live runs used.
- **Key numbers:**
  - Peaks and frequency of the stopped runs differ from the full-length runs by at most 1.1e-4.
  - Estimated case 4 OpenFOAM study: about 1740 s instead of 2548 s, about a third less, mostly from
    M4 (2.0 s of 3 s).
  - Insight 20 in case 4. Records: `results/dfg_cylinder_re100/live_stop.json`.
- **Open:**
  - Steady SIMPLE cases still stop on residual targets. Their per-mesh check that the residuals meet
    the physical check is unchanged.
  - The full study was not rerun with the live stop; the saving is the replay's estimate.

### gpucfd connected to the agent; case 4 run on gpucfd and OpenFOAM (`dac661a`)
- **What:**
  - `cfdlib/gpu.py` lets the agent use gpucfd as a solver:
    - `check_fit` says whether a problem fits gpucfd.
    - `run_transient` runs it with live health checks and a live stopping rule.
    - `write_run` records the run like an OpenFOAM run.
  - The skill (SKILL.md, stage 2) now says when to choose gpucfd and how to verify it.
  - Case 4 (cylinder at Re 100, vortex shedding) was run on gpucfd (`gpu_run.py`) and compared with
    the OpenFOAM result by case 4's own rules (`gpu_compare.py`).
- **Why it matters:** this is the first time the agent can choose between two solvers, with
  measured numbers on the same problem.
- **How it was verified:** three grids (17.6, 35 and 70 cells per diameter) and a time-step check,
  with all rules set before the runs.
  - **Pressure tolerance:** tuned with a same-answer rule, as OpenFOAM's solver settings were (1.7×
    less time per step).
  - **Stopping:** each grid stops once case 4's periodicity rule holds, checked live.
- **Key numbers:**
  - **Time:** gpucfd ran the whole study in 315 s against OpenFOAM's 2548 s (8×). Per cell and
    time step it is about 70× faster.
  - **Accuracy:** gpucfd's drag is 1.0% and its lift maximum 3.4% below the benchmark, against
    0.25% and 0.4% for OpenFOAM. The Strouhal number is within 0.3% on both.
  - **Same answer?** The two tools agree within their uncertainties on everything except the lift
    maximum. Those uncertainty bands are wide, though, because gpucfd's coarsest grid is far off.
  - **Verdict:** WARN by case 4's rules. The drag target set before the runs (within 0.3%) was
    missed.
- **Problems found:**
  - A tight solver tolerance copied from the tests made each step 1.7× slower than needed.
  - OpenFOAM's fixed 3 s per grid was too short for gpucfd: its coarse grid starts 16% low, so the
    finer grids need longer to settle. They are now stopped by the periodicity check instead.
- **Open:** at equal accuracy OpenFOAM is faster here. Its body-fitted grid reaches 0.25% with 13
  thousand cells in 97 s. gpucfd's uniform grid would need roughly 140 to 280 cells per diameter
  (estimated, not run). Stretched grids are the next gpucfd step.
- **Next:** the same live stop for OpenFOAM runs; then stretched grids for gpucfd.

### gpucfd: grid studies for steady flows, 20× faster (rung 11), `507ebeb`
- **What:** `gpucfd/steady.py` makes steady grid studies faster and self-stopping. gpucfd still
  marches in time to steady state; it is not a direct steady solver. Two changes make the march
  cheaper:
  - **Grid sequencing:** each finer grid starts from the coarser grid's solution instead of from rest.
  - **An error-based stop:** each run stops once the error left from marching is below 1% of the
    grid's own discretisation error, which is estimated while it runs from the difference to the
    coarser grid. Every time chunk logs the forces with that estimate, so runs can be watched live.
  - A study can also stop adding grids once the answer has settled.
  - The estimator (`iteration_error`) is in `cfdlib/verification.py`, so OpenFOAM runs can use it
    next.
- **Why it matters:** with bodies, gpucfd must use the explicit scheme (IMEX is unstable there), so
  fine-grid steady runs were slow. DFG 2D-1 at 70 cells per diameter had not finished after
  2.6 GPU-hours from rest.
- **How it was verified:** rung 11 (`gpucfd/tests/test_steady.py`, 6 of 6 pass) and a one-off DFG
  2D-1 study.
  - **Estimator:** tested on sequences with a known limit. It is exact for a single decaying mode,
    never too small in the harder cases, and it recognises solver noise.
  - **Same answers:** a Couette flow with an exact solution, on three grids, gives the same answers
    sequenced and from rest.
  - **DFG 2D-1:** the same drag, lift and pressure difference as runs started from rest.
- **Key numbers (DFG 2D-1, Re 20):**
  - **35 cells per diameter:** 19.6 GPU-seconds against 398 from rest (20×). The new stop gives
    3.8×, sequencing another 5.4×.
  - **70 cells per diameter:** 331 GPU-seconds against more than 9,400 from rest. Drag is within
    0.14% of the benchmark, lift within 5%. The three-grid study takes 6 GPU-minutes in all.
  - **Couette (exact solution):** sequencing halves the GPU time (7.1 s against 14.2 s).
- **Problems found while building it, all fixed:**
  - A slow decaying mode hidden under a fast one made the error estimate up to 11× too small.
  - Solver noise made a run march 200 time units past convergence.
  - The Richardson tool gave a value 11× off for a probe too close to a wall. It now refuses to
    extrapolate when the observed order is below 0.5.
- **Open:**
  - **Pressure difference:** it gets slightly worse on the finest grid (−0.23% at 35 cells per
    diameter, −0.47% at 70). The surface pressure next to an immersed body converges irregularly; drag
    and lift are not affected.
  - **Richardson extrapolation:** it does not improve drag or lift here. Report it as an estimate,
    never as the answer.
- **Next:** connect gpucfd to the agent through cfdlib, starting with case 4 (DFG 2D-2), against
  OpenFOAM.

### Case backlog (`cases/BACKLOG.md`) and project docs, `ba626b8` and later
- **What:**
  - [`cases/BACKLOG.md`](cases/BACKLOG.md) lists the next problems in order: heat transfer (heated
    pipe, buoyancy cavity), the first RANS case (NASA TMR turbulent flat plate), harder laminar
    cases that gpucfd can also run, and TMR turbulence cases.
  - It also proposes adopting manufactured solutions (MMS) for verification.
  - The changelog, AGENTS.md (the starting point for every coding agent) and the changelog rule in
    CLAUDE.md came in `ba626b8`.
  - The README now shows only the current state. gpucfd has its own short section next to the
    results, and the roadmap points to the backlog. A new rule in CLAUDE.md says to replace README
    text instead of adding to it.
- **Why it matters:** each next case adds one new capability. The sources (NASA TMR, and Luminary's
  V&V catalogue, suggested by Dhanush from Prof. Frankel's lab) come with grid families and
  reference results, and some with a commercial result to compare against.

### gpucfd: bodies inside the flow (rung 10), `a043ae6`
- **What:** shapes defined by a signed distance (cylinders, any shape you can describe that way) are
  placed inside the grid with a sharp-interface (ghost-cell) immersed boundary. Drag, lift and
  surface pressure come from the exact discrete momentum balance.
- **Why it matters:** gpucfd can now run flow around objects, which is the core of the DFG cylinder
  benchmarks and of most external-flow problems.
- **Verified:**
  - Exact solutions: flow between two immersed rotating cylinders (velocity order 2.1–2.5), and a
    channel tilted to the grid (order 2.2).
  - 3D extrusions give the same answer as 2D.
  - The forces don't depend on the box they are measured over (to 2e-18).
  - Runs without bodies are bit-identical to the previous build.
- **DFG 2D-1 benchmark (Re 20)**, with 35 cells per diameter: drag −0.65%, pressure difference
  −0.23%, lift +15%. On the coarser grid (17.6 cells per diameter) lift was 4.3 times the reference,
  so lift converges, but needs fine grids. Lift is only 0.2% of the drag in this case.
- **Open:**
  - The implicit (IMEX) scheme is unstable with bodies, so bodies use the explicit scheme only.
  - The largest pressure error, in the cells right next to a body, does not converge cleanly.

### gpucfd: inflow and outflow boundaries (rung 9), `a043ae6`
- **What:**
  - Any side can be an inflow with any velocity profile.
  - Any side can be an outflow: pressure zero and zero velocity gradient, the same as OpenFOAM's
    usual outlet.
- **Why it matters:** channels and flows past objects need an inlet and an outlet. Before this,
  gpucfd only handled closed or periodic boxes.
- **Verified:**
  - Poiseuille channel flowing in every direction, in 2D and 3D: second order.
  - Kovasznay flow, an exact Navier–Stokes solution with convection: second order.
  - The flow out equals the flow in to round-off.
  - Wall and periodic runs are bit-identical to the previous build.

### Project direction for agents (`CLAUDE.md`), `b702d6a`
- **What:** [`CLAUDE.md`](CLAUDE.md) states the goal (automate the CFD engineer's work and build
  the agent's tools), the bottlenecks (compute, usage budget, expertise) and the working rules. It is
  loaded by every Claude Code session in the repository.

### gpucfd: speed work, round 2, `d9d7090`
- **What:**
  - Every solver loop is now decided on the GPU, and a whole time step runs as one recorded CUDA
    graph, so the CPU no longer sits in the loop.
  - Fused kernels, and a 1/h precomputation (the RTX 4070 does double precision 64 times slower than
    single).
- **Why it matters:** most of each time step had been spent waiting on the CPU (about 115 round
  trips per step under WSL).
- **Numbers:**
  - Cavity 128²: 43 s → 5 s (OpenFOAM on the same case: 46 s).
  - 3D flow: 47× (64³) and 235× (128³) faster than OpenFOAM on the same grid.
- **Verified:** every speed-only change gives bit-identical results (`tests/same_as_build.py`).

### gpucfd: speed work, round 1, `c305b60`
- **What:**
  - An IMEX time scheme: convection explicit (RK3), diffusion implicit (Crank–Nicolson). The time
    step is then limited by the flow speed only.
  - CUDA graphs for the multigrid cycles.
  - Deterministic sums, so runs are bit-for-bit reproducible.
- **Numbers:** cavity 128²: 396 s → 42.8 s, with the same answer.
- **Bugs caught by the checks:**
  - The first IMEX version left a small error near walls in the steady state.
  - Atomic additions made identical runs differ at the 4e-16 level.

### gpucfd: walls, body force and the lid-driven cavity (rung 7), `698b197`
- **What:** no-slip walls (including a moving lid) and a constant body force.
- **Verified:**
  - Couette flow is exact to 1e-16, and Poiseuille flow converges at order 2.00.
  - On the Re 100 cavity, gpucfd's and OpenFOAM's solutions converge to each other at second order.
- **Found:** on that small steady case gpucfd was 9× slower than OpenFOAM, which motivated both
  rounds of speed work.

### gpucfd: like-for-like comparison with OpenFOAM, `aa3d28b`
- **What:** the same 3D flow (decaying ABC flow, an exact solution), the same grids, and each
  solver tuned by the same rule.
- **Numbers:** gpucfd was 4–168× faster than OpenFOAM on 8 CPU cores (later 18–235×, after the
  speed work), with a smaller error on every grid.
- **Why OpenFOAM's GPU plugins are capped:** 44% of OpenFOAM's time is outside its linear solvers,
  where a GPU plugin can't help.
- **Limits:** periodic boxes only at that point, one laminar flow, one machine.

## 2026-11-11

### gpucfd: first verified stage, `1d0b29b`
- **What:** a solver that keeps the whole time step on the GPU:
  - pressure solved with geometric multigrid, alone or as the preconditioner of conjugate gradients;
  - mixed precision (double outer iteration, single-precision multigrid);
  - Navier–Stokes on periodic boxes, on a staggered grid with RK3.
- **Why:** case 5 showed that offloading only OpenFOAM's linear solver is capped near 1.5×.
- **Verified:** a ladder of checks, each built on the last (rungs 0–6):
  - operators equal an independent numpy reference;
  - order 2 in space and 3 in time;
  - multigrid iterations don't grow with grid size;
  - mixed precision gives the same answer as double.

### Case 5: 3D cylinder (DFG 3D-1Z, Re 20) and GPU vs CPU study, `86e95aa`
- **Physics report:**
  - FAIL: lift is 1.7% off, outside its band. Drag is within 0.08%.
  - A second mesh family showed the lift depends on how the cells are graded around the cylinder.
  - [Report](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_3d_re20/report.html)
- **GPU study:**
  - OpenFOAM with the OGL plugin on the RTX 4070 gives the same answer as the CPU.
  - The GPU breaks even at about 2 million cells and is 1.13× faster at 4.75 million.
  - It is capped near 1.5× because the pressure solve is only a third of each iteration.
  - [Report](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_3d_re20/gpu_study.html)
- **Workflow:**
  - Study-report renderer and high-resolution profiling in `cfdlib`.
  - Guidelines: OGL settings, check the Amdahl bound first, compare against the best CPU setup, and
    use a second mesh family for sensitive outputs.

### Repository published, `d7f61bc`
- Pushed to GitHub ([tomyud1/agentic-cfd](https://github.com/tomyud1/agentic-cfd)) with GitHub
  Pages for the reports. This first commit contains everything from 2026-11-10 and 2026-11-11 below.

### Case 4: cylinder in a channel (DFG 2D-2, Re 100, vortex shedding)
- **Result:** WARN. Drag, lift and Strouhal number are all inside their bands (St 0.3015 vs 0.3019).
  The report includes an MPI scaling study.
  [Report](https://tomyud1.github.io/agentic-cfd/results/dfg_cylinder_re100/report.html)
- **Found:** OpenFOAM's PISO algorithm was only first-order accurate in time on this flow, which
  made the lift 10% wrong. Every unsteady nonlinear problem now needs a time-step study in its
  pre-flight.
- **Workflow:** signal analysis (`cfdlib/signals.py`), linear-solver iteration counts in the
  performance data, more checks in the report contract. 16 insights logged.

### Case 3: Stokes' first problem
- **Result:** PASS in 24 minutes. Velocity within 0.002% of U, second-order convergence.
  [Report](https://tomyud1.github.io/agentic-cfd/results/stokes_first_problem/report.html)
- **Workflow:**
  - Performance measurement for every run (`cfdlib/perf.py`) and an optional performance appendix.
  - Fixes to the round-off floor in run monitoring and to the insight log.
  - 11 insights logged.
- **Found:** a subagent could not load the project skill by name, so launch prompts now give the
  path to `SKILL.md`.

### Workflow fixes after case 2
- **What:**
  - Live, timestamped insight log (`cfdlib/insights.py`).
  - A figure checker that refuses overlapping text.
  - Each decision is tagged with a kind.
  - LaTeX checks.
  - Structured reproducibility information.
  - A visual review of every report before it counts as done.

## 2026-11-10

### Case 2: lid-driven cavity, Re 100 (first test on a fresh agent)
- **Result:** WARN. Velocity profiles are within 0.009 U of Ghia et al. (1982). The gap doesn't
  shrink with refinement, and the agent traced it to the benchmark's own mesh. 15 insights logged.
  [Report](https://tomyud1.github.io/agentic-cfd/results/lid_driven_cavity_re100/report.html)

### The three-layer design
- **What:**
  - **Tools:** `cfdlib`. Code computes every number.
  - **Guidelines:** the `agentic-cfd` skill, an OpenFOAM primer and the report contract.
  - **Worked examples:** the solved cases.
  - A contract checker refuses to render a report if any OpenFOAM setting is undocumented.
- **Why:** so a fresh agent can solve a problem it has never seen.

### Case 1: plane Poiseuille flow
- **Result:** PASS. Velocity profile within 0.0061% of the exact solution, second-order convergence.
  [Report](https://tomyud1.github.io/agentic-cfd/results/plane_poiseuille/report.html)

### Setup
- WSL2 with Ubuntu 24.04, ESI OpenFOAM v2606 and a Python environment, on a Ryzen 7 7800X3D with an
  RTX 4070.
- **Found:** the cavity case diverged on fine meshes with GAMG at pressure `relTol 0.05`.
  `relTol 0` fixes it.
