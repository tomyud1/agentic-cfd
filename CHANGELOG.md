# Changelog

What changed in the project and when, newest first, in plain language. Read this to catch up on
recent work. Details live in the linked commits, reports and [`gpucfd/DESIGN.md`](gpucfd/DESIGN.md).

Each entry says **what** changed, **why it matters**, **how it was verified**, the key numbers, and
the commit. Open problems are marked **Open**. Work not yet committed goes under "In progress", and
moves to a dated entry when it is committed.

## In progress

Nothing is waiting to be committed.

What comes next, in order, is in [`ROADMAP.md`](ROADMAP.md).

## 2026-11-14

### A page for case 17's optimize stage, commit hash to be filled in
`results/turbulent_channel_re180/optimize_study.html`: the new grid's profiles next to the first
grid's, in simple words, with a list of the words used.
- **What it shows.** The curves themselves, their difference from the reference with the pass
  band, which grid direction mattered, all ten limits of the case for both grids, how sure the
  result is, and what each run cost.
- **Its verdict is WARN,** set by rules in the script: every limit is met, but on one grid with
  no finer one to confirm it, and the time-step check was not repeated.
- **How it is built.** `dev/optimize/turbulent_channel_re180/study.py` forms every number from
  the recorded runs; it needs no GPU and builds the same file twice. It is in the case tables
  as "17b", in the report gallery, and in `setup/rerender_all.sh`.

### Case 17, optimize stage: a grid that meets every limit (`e8b9054`)
The first use of the loop's new step 8. Case 17 ended with a FAIL: the mean speed was right but
the velocity and fluctuation profiles missed their limits. The question for this stage, fixed
before any run: what is the cheapest grid on this PC that resolves the profiles?
- **What was done.** Three short runs from the finest grid's field, each with one direction 1.5
  times finer (along the flow, across it, toward the wall). Then one full run on the grid a rule
  written beforehand picked from them.
- **Result.** The grid 288 x 64 x 96 (1.8 million cells; the first was 192 x 64 x 64) meets every
  pass / fail limit of the case. Mean speed +0.17 % from the reference with a statistical error
  of 0.19 %. The five profile gaps (largest difference along the profile, in friction
  velocities; first grid, new grid, limit): mean velocity 0.31, 0.10, 0.30. Fluctuation along
  the flow 0.14, 0.04, 0.08. Toward the wall 0.07, 0.03, 0.05. Across the flow 0.08, 0.03, 0.06.
  Shear stress 0.034, 0.007, 0.03.
- **What matters.** The cells across the flow are the main cause, the cells along it the second.
  Toward the wall the first grid was already fine enough. The first grid's good mean speed was
  two grid errors cancelling: refining along the flow alone moved it from -0.3 % to +0.8 %.
- **Cost.** The final run took 1.7 GPU hours with start-up (limit 2), the whole stage 3.1 of
  the 5 budgeted. A time unit costs 2.6 times that of the first grid.
- **How it was checked.** The script reproduces the report's numbers for the first grid before
  anything is compared. The final run ended on the stop rule after 30 time units; its two
  plates agree, and its total shear stress lies on the exact straight line within 0.0006.
- **What did not work, recorded.** The planned model (each direction's share of a gap, assuming
  the error falls with the square of the cell size) failed its own check: these grids are too
  coarse for that. It only ranks the directions. What held was simpler: the new grid's result
  is the first grid's plus the two measured changes (mean velocity predicted 0.20, measured
  0.10; the others within 0.015).
- **New tool.** `gpucfd.turbulence.rescale_to_balance`: after a few time units on a new grid,
  move the flow's speed once to where the wall friction says it is heading. Start-up fell from
  about 20 time units to 7. Tested in `tests/test_gpu_tools.py`.
- **Limits.** One grid, no grid study at its level: the last refinement in each direction still
  moved the mean speed by 0.6 to 0.8 %, in opposite directions, so the 0.17 % again includes
  some cancellation. The case's verdict stays FAIL; this stage is recorded next to it, in
  `gpucfd/DESIGN.md` (channel section) and `dev/optimize/turbulent_channel_re180/`.
- **Open.** One finer step across the flow would show whether the profiles have stopped moving.
  (The page that shows the new profiles next to the first ones came after this commit.)

### What case 17 taught the tools and the workflow (`0aa0d12`)
The agent's 19 insights each got an answer, tools first.
- *The statistical error of a slowly settling quantity (insights 10, 17).* The flaw: the tool
  cuts the record into blocks, and a quantity whose memory is longer than a block looks more
  certain than it is. Now `cfdlib.signals.stationary` takes the slow time as `memory` and
  corrects for it, and says "not reliable" when the record holds few memories.
  `signals.relaxing` gives a reliable error from a short record when the quantity's rate of
  change is recorded too; `gpucfd.turbulence.bulk_average` applies it to a channel's mean speed,
  and `channel_watch` is a ready stop rule built on it. `settle_time` can wait for the slow
  approach as well. *Checked:* on 120 test signals with a known error the old estimate gives
  0.40 of the truth, the corrected one 0.89, the new one 1.16. On case 17's recorded runs the
  new tools give the agent's own numbers (15.6457 ± 0.0244), and the stop rule replayed on the
  finest run stops at the same time, 58.0.
- *The time step that follows the flow (3, 5, 9).* `run_transient` now takes steps as settings
  (`sample_steps`, no largest step needed) and re-chooses the step at its own interval
  (`cfl_steps`), separate from the sampling. Its text states how far the target is overshot.
  On the test box a look every 2 steps reached 0.92 where a look every 5 reached 0.97.
- *Long runs (4, 7, 16, 18).* Restart files (`save_state`, `load_state`, checkpoints inside
  `run_transient`; a continued run matches the original to 2e-14), a pre-flight for gpucfd
  (`cfdlib.gpu.preflight`: health, cost per step, seconds per time unit, projected time), a
  short time-step check from one field (`step_check`), a look into a running case
  (`python -m cfdlib.gpu peek`), and stop rules that say at each check what is still missing.
  Records can keep profiles at a coarser interval than the scalars (`write_run(store_every=)`).
- *Grid study with unequal steps (2).* `verification.three_grid_study(h=...)` takes the cell
  sizes of the three grids; it recovers a known order with ratios 1.5 and 4/3 and equals the
  old result for equal ratios.
- *Reports (11, 12, 19).* Sections 6 and 7 can carry further tables (`tables`). The environment
  table can show the GPU, and a run record the solver's settings. `cfdlib.contract --code`
  finds Python files that only compile with a warning (a LaTeX backslash in a normal string);
  the review script and the tests run it on every case.
- *Guidance, each in the most specific file (1, 6, 8, 13, 14).* A time budget for averaged runs
  and "never call python from Git Bash" in the machine file. In the workflow, four short
  additions: unequal grid ratios are fine, size a profile's limit on the coarsest grid, show an
  integral next to its profile, project an averaged run as seconds per time unit. What to
  compare in a pressure-tolerance check without a body is in that tool's text.
- *Records corrected (10, 13, 15).* `gpucfd/DESIGN.md` now has case 17's three grids, the
  settling time of 9 time units in place of "about one", the measured overshoot of the time
  step, and the list of things to try next. The fit check's note carries the three grids.
- *Not built:* a contract check of a gpucfd case's decisions against its recorded settings
  (11), and gpucfd records in `cfdlib.monitor`'s own live view (16; `peek` is a separate
  command). Both are in `ROADMAP.md`.
- *The loop itself* (`dev/LOOP.md`): a new step 8, **optimize**: after a case is recorded, make
  it more accurate and faster against the verified solve, with target, budget and stop rule
  fixed first. Before a solve, fix only what would stop it or make it wrong. The review now
  checks when the limits were written and whether repeat runs support the error claimed.
- *Old work still holds:* all tests pass (GPU tool tests included, 9 new checks), and all eight
  reports and both study pages rebuild. No old report changes. Case 17's page changes in the
  sixth decimal of its figure coordinates only; its numbers file is identical.
- *Proposed here, done since:* the optimize stage for case 17 (entry above).

### Case 17 solved: turbulent channel without a model, verdict FAIL (`b431e6d`)
A fresh agent (Claude Fable 5.2, high effort) solved the channel on gpucfd in 2 h 15 min with no
human help. Report: `results/turbulent_channel_re180/report.html`.
- *The result:* on the finest grid (192 x 64 x 64 cells) the mean speed through the channel is
  15.65 ± 0.02 against 15.70 in the reference, 0.34 % low. The centre speed is 0.17 % high, the
  friction coefficient 0.68 % high, the peak of the fluctuation along the flow 0.6 % low. These
  four are inside the limits the agent set before the runs.
- *Why FAIL:* before the runs the agent also set limits on the whole profiles, and all five are
  missed. The mean velocity is up to 0.31 friction velocities low, 22 wall units from the plate
  (limit 0.30). The fluctuations along the flow are too strong near the wall (0.14 against 0.08),
  those across it too weak (0.07 and 0.08 against 0.05 and 0.06), and the shear stress is 0.034
  off against 0.03. Two of the five lie within two statistical errors of their limit.
- *What it means:* the good mean speed is partly luck. The profile is too low near the wall and
  slightly too high in the middle, and the two cancel in the average. On the three grids the mean
  speed is 15.94, 15.44 and 15.65, so it does not approach the answer from one side and the grid
  study gives no order of accuracy. The cells on the finest grid (12 wall units along and across
  the flow) are too large for the smallest eddies at the wall; the reference used cells two to
  four times smaller. gpucfd computes this flow, but this grid does not resolve it. The next
  finer grid (288 x 96 x 96) needs about four hours for the same averaging and was not run.
- *A flaw in a tool prepared for this case, found by the agent:* `cfdlib.signals.stationary` gave
  a statistical error for the mean speed that was about three times too small, and called it
  reliable. Two runs of the coarsest grid from the same start gave 15.978 and 15.904, each
  claimed to ± 0.010. The reason: wall friction pulls the mean speed back to its level slowly,
  in about 9 time units, and a run of 30 to 40 time units is too short for the tool to see so
  long a memory. The statement in `gpucfd/DESIGN.md` that the memory is about one time unit is
  wrong for the same reason. The agent wrote a corrected estimate in its case folder (from the
  exact balance of the box) and changed its stop rule to use it.
- *Also found:* the time step that follows the flow overshoots its target (Courant number 1.08
  with a target of 0.8 and a look every 10 steps). The agent ended at a look every 5 steps and a
  target of 0.75. The two findings forced two restarts, about 12 minutes of GPU time.
- *How the review checked it:* the scripted check is clean (report contract, problem and
  reference unchanged, no shared file touched). The agent's transcript shows the pass and fail
  limits written ten minutes before the first production run and never edited. Recomputed from
  the raw run records with separate code: the mean speed (15.6457), the five largest profile
  differences (0.313, 0.136, 0.072, 0.083, 0.034) and the total shear stress (within 0.004 of
  the exact line) all match the report. Four runs of the coarsest grid scatter by 0.20 %, which
  fits the agent's corrected error (0.15 to 0.25 %) and not the tool's (0.06 %). The verdict is
  fair and stands.
- *Against what was fixed in advance for gpucfd* (entry below): a mean speed within 1 % with a
  statistical error of at most 0.3 % is met on the finest grid only. The two coarser grids are
  1.5 % high and 1.6 % low, so "shown on more than one grid" is not met. GPU time was 1.8 of the
  5 hours allowed.
- *Cost:* 2 h 15 min in all, 110 minutes of it on the GPU (60 on the finest grid). About a
  third of every run is start-up that is cut from the averages. The agent logged 19 insights,
  each with a proposed change.
- **Open:** no grid here resolves the profiles, and OpenFOAM's accuracy on this flow is still not
  measured. (The 19 insights were answered afterwards: the entry above.)

### Case 17 prepared: turbulent channel computed without a model (`9f060f3`)
The next verified case, and the first turbulent flow for gpucfd. A fresh agent gets it next.
- *The problem* (`cases/turbulent_channel_re180/problem.yaml`): flow between two plates driven
  by a constant force, friction Reynolds number 180, in a box that repeats along and across the
  flow. No turbulence model: the eddies are computed in time and the answers are time averages.
  Asked for: mean speed through the channel, friction coefficient, velocity profile, strength of
  the fluctuations, the shear stress against its exact straight line, and how well the average
  has settled.
- *The reference:* Vreman and Kuerten (2014) computed exactly this set-up with two independent
  codes, each averaged over 200 time units. Their files were downloaded from the author's site
  and parsed by code (`setup/fetch_vk2014_channel.sh`, `setup/parse_vk2014_channel.py`). The two
  codes give a mean speed of 15.696 and 15.701 (0.03 % apart); the paper puts the error of
  either below 0.2 % for the mean velocity and below 1 % for the fluctuations. The reference
  file holds these uncertainties as numbers. One check of the parsing: the centre-plane velocity
  computed from the files is 18.28 for both codes, the value printed in the paper's table.
- *Tools a fresh agent would have been stuck without, found by a dry run on the GPU:*
  - the fit check called gpucfd "laminar only" and would have refused this grid outright (its
    rule for laminar flows does not apply here). It now takes the cell sizes in wall units and
    answers with what was measured;
  - no tool gave the average of a signal that never settles, with its statistical error.
    `cfdlib.signals.stationary` does (also where the start-up ends, and a live stop);
  - `gpucfd.turbulence`: a start field that turns turbulent, and the plane averages (profile,
    fluctuations, shear stress, wall shear) formed exactly like the solver's own fluxes;
  - the run loop's Courant check used the largest velocity over the smallest cell, which
    overstates it about 7 times on a wall grid, and a fixed time step overshot the real limit
    (1.23). The loop can now follow the limit (`adapt_cfl`).
- *How it was checked:* the plane averages reproduce the solver's own balance of mean momentum
  to round-off (4e-12 against terms of order 10); the bulk velocity changes by driving force
  minus wall shear as it must; the statistical error is 12 to 17 % on the safe side on test
  signals whose true error is known; a 6-time-unit run on 64 x 64 x 64 went through the whole
  chain with a clean health record. All tool tests pass.
- *Correction to the prediction in the next entry:* it compared with a mean speed of 15.63 quoted from
  memory. Against the transcribed reference (15.70) the two exploration runs are 7.5 % and 0.8 %
  low, not 7.1 % and 0.4 %. The new statistics tool also confirms that the finer run had not
  settled (its second half lies 2.9 standard errors above its first).
- *Fixed in advance for the case* (also in `ROADMAP.md`): success for gpucfd is a mean speed
  within 1 % of the reference with a statistical error of at most 0.3 %, shown on more than one
  grid, in under about 5 GPU hours on this PC. If no settled average exists on a grid at least
  as fine as 192 x 64 x 64 after that time, the run is stopped and reviewed.
- *Old work still holds:* all tests pass (the GPU tool tests and gpucfd's wall and
  stretched-grid tests included), and all seven reports and both study pages rebuild from saved
  data with no file changed. The rebuild exposed one old flaw: rebuilt on a later day, case 6's
  results file took that day's date, because its script writes the file itself after the
  renderer. The renderer now hands the kept date back to the script.
- *Open:* OpenFOAM's accuracy on this flow is not measured. The agent's run will take hours of
  GPU time, the longest case so far.

### Which way for turbulence in gpucfd: a measured prediction (`9f060f3`)
Before
building any turbulence model, two routes were compared by measurement. Scripts and numbers:
`dev/exploration/les_channel/` (`predict.py`, `openfoam_time.py`, `results.json`); the write-up
is in [`gpucfd/DESIGN.md`](gpucfd/DESIGN.md), "Turbulence resolved without a model".
- *Route 1, a turbulence model as in case 7 (RANS):* predicted to lose. On case 7's finest grid
  the thin wall cells allow a time step of 1.7 millionths of a second (measured on OpenFOAM's
  solution), so gpucfd, which marches in time, would need an estimated 1.5 to 3 hours where
  OpenFOAM's steady solver took 10 minutes. Not built.
- *Route 2, resolving the turbulent eddies in time (the plane channel at friction Reynolds number
  180):* gpucfd runs this flow today, with no model and no new solver code. On 0.79 million
  cells the mean speed through the channel is 0.8 % below the reference after 34 minutes on the
  GPU, and was still rising when the run ended. On a grid three times coarser along the flow it
  is 7.5 % low (19 minutes).
- *Speed against OpenFOAM, same grid and time step, per step:* 2.3 times faster at 0.26 million
  cells, 5.5 to 9.1 times at 0.8 to 4.2 million (OpenFOAM on 8 cores with its tutorial's
  settings, not tuned). Far less than the 47 to 235 times of the periodic box: with walls and a
  stretched grid a step costs about four times more per cell.
- *Open:* OpenFOAM's accuracy on these grids was not measured; the finer run's average had not
  settled; what a model adds at higher Reynolds numbers is not measured at all.

## 2026-11-13

### What case 7 taught the tools and the workflow (`9085126`)
The agent that solved case 7 logged 16 problems and surprises. Each one gets an answer here: a tool
fix where code can prevent it next time, a line of guidance where it cannot, or a stated reason for
doing nothing. No simulation was run for this entry.

- **The stop rule no longer accepts a solution that flips between states.**
  - *Problem:* with one setting the solution alternated between two states on every iteration.
    The stop rule looked at every 20th iteration, saw the same state each time, and ended the run
    as converged (insights 3 and 4).
  - *Now:* the rule also looks at neighbouring iterations. A regular cycle of 2 to 6 iterations is
    reported as "oscillating" and the run is not stopped (`monitor.steady_check`).
  - *Checked:* on the recorded run that fooled it, the rule now refuses to stop at every point of
    the history, for even and for odd sampling. The same grid's good run still stops where it did
    (iteration 351 of 8433). The stop points of the 12 recorded histories of cases 2 and 6 are
    unchanged.
- **Starting a run from another solution keeps the run's own boundary values.**
  - *Problem:* a test run with a 0.3 % faster inflow was started from the main solution. OpenFOAM's
    mapping tool copied the old inflow speed over the new one without a word, so the run repeated
    the main answer (insight 16).
  - *Now:* `foam.map_fields` writes the run's own fixed values back after mapping and prints each
    one that had been replaced.
  - *Checked:* on case 2's recorded runs with the lid speed set to 1.25: the plain mapping leaves 1
    (the source's value), the tool leaves 1.25.
  - *Limit:* it covers constant fixed values. A boundary value given as a list, or by a special
    boundary type, is still the agent's to check; the tool's text and the primer say so.
- **The short test run works on parallel cases** (`monitor.preflight`, insight 2). It used to drop
  the pieces a parallel run needs. *Checked:* 30 iterations on 2 processor cores on case 7's
  coarsest grid; a parallel command on a case that is not split yet is refused with the reason.
- **How fast did a grid really converge?** The stop rule is checked every few seconds, so a tiny
  grid runs thousands of iterations past the point where it first held (8433 run, 351 needed). The
  stop record now also holds that point (`t_first_held`, insight 7). *Not done:* checking by
  iteration count instead of by the clock. The overshoot cost 10 seconds, and it would change
  where every run stops.
- **A problem statement is checked when it is written** (insight 9). My statement for case 7 had
  an overview of 650 characters; the report allows 600, and the agent may not edit the statement.
  New: `python -m cfdlib.contract --problem <file>`, a test on every prepared case that is not
  solved yet, and a line in the loop's preparation step. Case 7's statement stays as it was given.
- **Three report slips now get a warning** (insights 10 and 11): a sentence in the summary's last
  column (it does not wrap and pushed the table off the page), a figure reference that would show
  as a bare number, and a figure placeholder quoted as an example, which the checker took for a
  real reference and refused.
- **Positions of cells and wall faces from a published grid** (`structgrid.centres`, insight 8),
  so an analysis does not rebuild them by hand.
- **Guidance, in the most specific place.**
  - *OpenFOAM primer, new section "Turbulence models":* check which published variant a solver's
    default is, with the Spalart-Allmaras switches found here (insights 1 and 6); the setting that
    caused the flipping; why the error numbers of the solver fall long before the answer has
    settled (insight 15). One caution added to the steady-run section: the converged answer
    depends very slightly on one solver setting, so keep it fixed within a grid study (insight 5).
  - *Workflow:* a reference's uncertainty is more than its grid error (insight 13); an explanation
    that needs a sensitivity should measure it with a test that changes one thing only
    (insight 12). With the pointers to the tool changes above, the workflow grew by 12 lines.
  - *Loop, preparation step:* record what the source says about its own uncertainty, and check
    which reference speed its coefficients use (insight 14). The same note is on the next NASA
    case in the case list.
- **Declined, with the reason.**
  - *A warning when the solver's error numbers stay flat* (insight 3): they are flat in every
    unsteady run, so it would warn falsely there. The new check names the cycle directly.
  - *A helper for trial runs* (insight 4) and *a general tool that evaluates a turbulence model's
    switches on a solution* (insight 6): one case needed each; case 7's folder is the worked
    example.
  - *"Write no finding before its numbers exist"* (insight 12): already the firm rule that code
    computes every number.
  - *Changing case 7's reference file* (insights 13 and 14): it is the record of what the agent
    was given.
- **Queued in the roadmap:** making this kind of flow faster (insight 15), as the student's open
  decision after case 7: first by tuning OpenFOAM (a day of trials), or with a turbulence model in
  gpucfd (weeks). And one more example for the accuracy question to Prof. Frankel (insight 13).
- **Re-check of the old work.** All tests pass (the gpucfd run tools were not re-tested: nothing
  they use changed). All seven main reports and both study pages were rebuilt from saved data. The
  seven main reports come out byte for byte as committed. The 4b study page changed in one place:
  the commit it names for its data, which was one commit behind (a page committed together with
  its data cannot know that commit's name).
- **Housekeeping.** Case 7 is in the rebuild script; the dry-run script from the preparation is
  now in `dev/agent_checks/`; the case list says what a later agent can learn from case 7's folder.

### Case 7 solved: turbulent flat plate, verdict WARN (0.028 % from NASA) (`fcda82a`)
The project's first case with a turbulence model. A fresh agent (Claude Fable 5.2, high effort)
solved it with the workflow and OpenFOAM in 52 minutes, without help. This commit holds the case
exactly as that agent left it.
[Report](https://tomyud1.github.io/agentic-cfd/results/turbulent_flat_plate_sa/report.html).

Words: *skin friction* is the drag force per area that the flow puts on the wall, as a
coefficient. A *turbulence model* (here Spalart-Allmaras) replaces the fast swirling motion of a
turbulent flow by one extra equation, so that a steady solution can be computed. *Grid-converged*
means the value the solutions approach as the grid gets finer, estimated from the three finest
grids. *Mach 0.2* means the reference flow was slightly compressible (speed one fifth of the
speed of sound).

- **The result.**

  | Quantity | This solution | NASA | Difference |
  |---|---|---|---|
  | Skin friction at NASA's point, finest grid (545 × 385 points) | 0.0027285 | 0.0027291 (incompressible) | −0.023 % |
  | Same, grid-converged | 0.0027286 | 0.0027294 | −0.028 % |
  | Drag of the plate, finest grid | 0.0028831 | 0.0028599 (Mach 0.2) | +0.81 %, context only |

- **Why WARN.** Before the runs the agent fixed its own rule: pass only if the grid-converged
  friction is inside the grid uncertainty of the two solutions added together (0.024 %). The result
  is 0.028 % away: just outside. No check failed. The rule was not changed afterwards.
- **Is the rule too strict?** Probably. It counts grid error only. The agent's own test shows that
  the way the top boundary is set moves the friction by 0.26 %, ten times the remaining difference,
  and NASA says in words that boundary choices move its results by up to 0.2 %. The verdict stays
  as it is; what the workflow learned from this is in the improvements that follow this commit.
- **OpenFOAM's model is NASA's model.** NASA warned in 2014 that OpenFOAM's version did not match.
  In the version used here (v2606) the constants are NASA's. One small term is off by default; the
  agent switched it on, and measured that it changes the friction by less than 0.0001 %.
- **How it was verified.**
  - *By the agent:* five grids with the friction approaching its limit at order 1.74 (NASA's own
    code: 1.53); the momentum the flow loses equals the drag to 0.002 % on every grid; the velocity
    near the wall follows the model's own exact wall law within 0.24 %; a second numerical scheme
    gives a grid-converged friction 0.003 % from the first.
  - *By the dev agent:* friction on all five grids, the grid-converged value, both uncertainty
    bands and the drag recomputed from the raw run data with a separate script: every headline
    number matches. The problem statement and reference data are unchanged, and no shared tool was
    touched.
- **A solver that looked converged but was not.** With one setting the solution flipped between
  two states on every iteration. Our stop rule sampled every 20 iterations, always saw the same
  state, and declared the run settled. The agent caught it with a physical check: the drag did not
  match the momentum the flow had lost (8.6 % off). The guard for the stop rule is in the
  improvements that follow this commit.
- **A hint about NASA's data.** NASA calls the 0.87 % between its Mach 0.2 and its incompressible
  friction a compressibility effect. The agent estimates that most of it (0.61 of 0.84 % at the
  point) comes from a slightly different flow speed above the plate in the two set-ups, not from
  density. This rests on two points along the plate: a hint, not a conclusion.
- **Cost.** 52 minutes, 26 of them solver runs; the finest grid took 10 minutes on 8 processor
  cores (11,062 iterations). No person stepped in.
- **Open:**
  - Only the friction at one point has a like-for-like reference. Drag, friction along the plate
    and velocity profiles are compared with NASA's Mach 0.2 results, as context.
  - A second WARN-only check did not pass: the ratio of our friction to the Mach 0.2 friction
    varies by 0.42 % along the plate (limit 0.3 %); the flow-speed difference above explains the
    size.
  - How strict an accuracy rule should be when the reference states part of its uncertainty only
    in words is a question for Prof. Frankel.

### Case 7 prepared: turbulent flat plate with the Spalart-Allmaras model (`ed6c3e1`)
- **What:** everything a fresh agent needs to solve the project's first turbulence case is now in the
  repository. The case is NASA's standard check of a turbulence model: air flowing along a flat
  plate, with the turbulence replaced by a model (Spalart-Allmaras). Nothing was solved; this is the
  preparation step of the loop.
- **The pieces.**
  - *The problem statement* (`cases/turbulent_flat_plate_sa/problem.yaml`): geometry, flow, model,
    and what to find (friction at one point of the plate on each grid, drag, velocity profiles).
  - *The reference data* (`reference/`): NASA's pages and data files saved as published, and one
    file built from them by a script that stops if a page no longer says what the file states
    (`setup/fetch_tmr_flatplate.sh`, `setup/parse_tmr_flatplate.py`). NASA's five grids are stored too.
  - *A tool to use NASA's grids* (`cfdlib/structgrid.py`): it writes a published grid as an OpenFOAM
    mesh, point for point, with named boundaries. Until now the workflow could only build grids with
    OpenFOAM's own simple mesher, which cannot reproduce NASA's spacing. Using NASA's grids removes
    the grid as a difference between our result and theirs.
  - *A tool to read values on a wall* (`foam.read_boundary_field`): friction and the wall-cell
    measure (y+) face by face along the plate.
- **A trap found in the reference, before any run.** NASA's main results are for slightly
  compressible air (Mach 0.2); OpenFOAM's standard solver for this case treats the flow as
  incompressible. NASA publishes both for the friction at one point, and on the finest grid they
  differ by 0.87 %: about 30 times the uncertainty NASA gives for either. So the problem statement
  names the incompressible values as the like-for-like reference. Drag and the profiles exist only
  for the compressible flow, and the statement says so. (The same kind of mistake as on case 4,
  where a target was set against the wrong reference value; this time it was caught first.)
- **Also from NASA's pages:** as of 2014 OpenFOAM's version of this model did not match NASA's
  equations, and NASA does not track later versions. A small mismatch is therefore a possible
  honest outcome; the problem statement asks the agent to say what any remaining difference is
  made of.
- **How it was verified.**
  - *Are the grids loaded correctly?* OpenFOAM's own mesh checker accepts the coarsest, the middle
    and the finest grid (cell counts and plate faces as in NASA's files); the points on disk equal
    NASA's to the last digit, including the first cell at the wall, which is 0.0000005 high. The
    checker's one complaint is the shape of the thin wall cells, which is how these grids are meant
    to be (`tests/test_structgrid.py`).
  - *Can our tools handle a turbulence case?* A throwaway setup on the coarsest grid, run for 300
    iterations (one second): the dictionary reader, the report checker, the health monitor, and the
    reading of friction, y+, profiles and drag all work (10 checks, 0 gaps). That setup is not a
    solution. Its script is kept outside the repository until the case is solved, so that the
    fresh agent finds no ready-made solver settings; it then goes to `dev/agent_checks/`.
  - The reference script checks that the five grids are nested, straight and of the sizes the data
    files name, and that the point where NASA tabulates friction is a grid point of all five.
- **Open:** the case is not launched. A fresh agent solves it next (`dev/LOOP.md`, step 3). The
  guard against too few digits when a mesh is written again is a note in the primer and a number
  the grid tool reports, not yet an automatic check.

### Tools for comparing two solvers: study reports and run records (`f6364c4`)
The two agents that compared gpucfd with OpenFOAM on case 4 said where the tooling slowed them down.
This entry answers their list. No simulation was run. Both study pages (4b and 5b) were rebuilt and
show the same numbers and the same verdicts as before (checked number by number), and all six main
reports rebuild unchanged.

Words: a *study report* is the second page of a case, which compares two tools (4b: gpucfd and
OpenFOAM; 5b: OpenFOAM on the processor and on the graphics card). A *run record* is the small
file each run leaves behind (grid size, time step, how long it took). *Wall time* is elapsed time,
as on a stopwatch.

- **Each table explains itself.**
  - *Problem:* a table's question, result and meaning had to be squeezed into its title and a long
    note under it.
  - *Now:* a table can carry a **question** (shown above it), a **result** and **what it means**
    (shown below it, each on its own labelled line). Every table of 4b uses them.
  - *For the next report:* the writer fills three fields per table, and a reader can follow a page
    by reading only those lines.
- **The page fits a comparison of two solvers, not only "processor against graphics card".**
  - *Problem:* section titles and the line above the title were fixed for 5b's kind of study.
  - *Now:* a study can rename its sections, add sections of its own, and place a figure right
    above the table it explains. 4b has a section of its own for the implicit time scheme. Short
    labels in the first column stay on one line by themselves, a table too wide for the page gives
    a warning (5b's widest table was split in two), and the reasons for a verdict other than PASS
    are listed under the headline. The contract now demands those reasons.
  - *For the next report:* no more workarounds inside table cells.
- **The list of insights no longer fills the page.**
  - *Problem:* every performance insight was printed in full; on 4b that was a sixth of the page.
  - *Now:* each insight shows its problem; a click opens the rest. Open ones start unfolded. The
    section is about 2.5 times shorter on 4b.
- **One way to read a run record, with one clock.**
  - *Problem:* OpenFOAM's and gpucfd's records name things differently, and one name ("cost per
    cell and step") meant elapsed time in one and time inside the solver in the other. That mix
    produced the wrong "50 times faster" corrected in the entry below.
  - *Now:* `cfdlib.runrecord.read` returns either record with the same names, and every cost says
    which clock it is on. New gpucfd records store both costs, the two limits on the time step
    and which one set it. Older records still load. A record can also hold repeated timings and
    reports how far they are apart.
  - *For the next report:* a speed comparison cannot mix the two clocks by accident.
- **Case 4's rules and its list of runs are written once.**
  - *Problem:* the target (drag within 0.25 %, in less than 1740 s), the pressure rule and the
    rule for "same answer" were typed in several scripts. Which run was which was read from the
    letters of its name (C3wi = finest grid, longer band, implicit scheme).
  - *Now:* the rules are named entries in `numerics.yaml`, and so is a table of all 16 gpucfd
    runs (grid, variant, time scheme, the run it starts from). The run script, the two analyses
    and the report read them. The run script has one way to check a pressure setting, not two.
  - *Checked:* the two analyses give the same numbers as before, and the run script builds all
    15 recorded runs with the recorded grid, scheme and time step (built only, not run).
  - *For the next run:* a new run is a new line in the table.
- **An error estimate that is not really an estimate is marked.**
  - *Problem:* the usual formula for a grid study's error needs three grids that approach a limit
    from one side. When they do not, the code quietly assumed a typical behaviour and still
    printed a number.
  - *Now:* that number carries a flag and the reason (`cfdlib.verification.three_grid_study`), and
    4b marks it with a star (gpucfd's lift maximum on the clustered grids).
  - *For the next report:* a placeholder cannot pass for a measured error.
- **A study's one-line summary in the case list cannot go stale.**
  - *Problem:* it was typed by hand in the catalogue. 5b's said "capped near 1.5 times";
    the numbers say 1.57.
  - *Now:* the study's script writes the sentence from its numbers and the catalogue reads it,
    as it already reads the verdict.
- **Smaller things:** shared number formats for report tables (per cent with a sign, "3.3 times",
  powers of ten); the study contract is described in `workflow/references/report-contract.md`; the
  smoke test now renders a study.
- **A warning explained.** The coarsest gpucfd runs of case 4 carry a warning that the time step was
  too large for the flow speed. It was thought to come from the developed flow. The recorded data
  say otherwise: the step is too large only in the first 0.02 to 0.04 seconds after the start from
  rest, and never afterwards. The warning now says when it happened, so the right fix (a smaller
  step at the very start, not for the whole run) is clear. The recorded runs are unchanged.
- **Open:** not done, each with its reason in the hand-over: taking the time step from the flow
  itself is offered but not tried on a case; no timing has been repeated yet, so no time on any page
  has a spread; the same marking of weak error estimates in cases 2 and 5; printing a study page does
  not unfold the insights.
- **Checked a second time by the dev agent.** The six main reports are word for word what was
  committed; every formula on the two study pages is the same as before (591 and 571 of them); all
  report rules pass; the tests pass (study, catalogue, layout, tools, the GPU tools, the grid
  loader); the new reader opens all 71 stored run records. Not checked from end to end, as the
  agent said: the commands of the rewritten run script that would repeat or overwrite recorded
  runs. Watch their first real use.
- **What became of the agent's 16 notes.**
  - *Done now.* (1) **Reports no longer change when nothing changed.** Every re-render used to
    rewrite every report, because each figure carried a date and random labels. Figures are now
    written the same way every time; two full renders in a row gave identical files. "Did a
    published report change?" is now a plain file comparison. (2) **A misleading sentence on the
    4b page is corrected.** It called OpenFOAM's error estimate for the drag "tiny". That number is
    a placeholder too (the drag has stopped moving in one direction between the grids, so the
    formula has nothing to measure); the page now marks it with a star and says what it means. The
    same point is logged in case 4's own record (insight 26); the main report's table was left as
    the solver agent wrote it. (3) The loop's notes say how to commit when several sessions share
    the folder. (4) The workflow's new paragraph on "one clock on both sides" was cut to two lines.
  - *A correction of the dev agent's own statement.* The warning on the coarsest gpucfd runs was
    passed on as "the flow gets faster than the speed the time step was planned for". The recorded
    data show it happens only in the first hundredths of a second after the start (entry above).
  - *Put to the supervisor* (`ROADMAP.md`, questions): which error band is honest when three grids
    show no clear order. Today a factor of 1.25 with a mark; common practice is 3, which can
    change the verdicts of cases 4 and 5.
  - *Written down for later* (`ROADMAP.md`): a small generated file of each case's headline
    numbers and a test that the numbers quoted by hand elsewhere still match it.
  - *Not done, with the reason.* Requiring a question on every table of a study: some tables are
    plain listings. Moving the verdict rules into the shared helper: wait until a third study
    repeats them. Repeating the 13-minute run that carries 4b's headline time, to give it a
    spread: the times compared differ by far more than the few per cent a repeat would show.

### Correction: gpucfd's drag on case 4 is 0.3 to 0.4 % from the converged benchmark (`d10a2c1`)
- **What:** the claim "drag 0.1 to 0.2 % from the benchmark" in the two entries below it is
  true only against the benchmark's finest tabulated number. Found by the agent that wrote the
  study report (next entry), and recomputed by the dev agent.
- **The check: question, result, meaning.**
  - *Is the benchmark number itself converged?* The benchmark (FeatFlow) is a simulation on a
    series of finer and finer grids. Its last three grids are still moving: they head for a drag
    0.21 % above the last tabulated one. Against that converged value OpenFOAM is within 0.04 %,
    and gpucfd is 0.31 % and 0.38 % low (two positions of the cylinder).
  - *What it means:* gpucfd did not reach OpenFOAM's accuracy on this case. It came within
    0.4 %, in less time than OpenFOAM's full study. The target ("within 0.25 % of the benchmark")
    was met as written, but it was written against a number that is 0.2 % off, so OpenFOAM's own
    finest grid would narrowly have failed it. gpucfd's own error estimate (0.6 to 0.75 %) had
    been saying that 0.1 % was too good.
- **A second correction:** "about 50 times faster than OpenFOAM per cell and time step" compared
  two different clocks. With wall time on both sides it is 45 times on the grid with equal cells
  and 25 times on the clustered grid.
- **Changed:** README, ROADMAP and the workflow's gpucfd advice quote both drag numbers and the
  corrected speed factor. Case 4's insight log has the lesson (insight 25). The agent building the
  implicit scheme was told to quote its results the same way.
- **The lesson for setting targets:** the workflow already tells a solver agent to compare with
  a benchmark's converged value as well as its finest level. The target for the GPU solver was
  set by the dev agent without that rule. Future targets use it.
- **What happened to the report agent's 30 findings and suggestions:**
  - *Acted on now:* the two corrections above; the catalogue check now catches a study report
    that is not listed, and counts studies apart from cases; the machine file notes the report
    server's port; case 4's report gets a link to the study; the study's headline and its
    catalogue line get the correction (the last two by the report agent).
  - *Passed to the agent building the implicit scheme:* the corrections in `gpucfd/DESIGN.md`,
    and that the grid helper must stay usable without the graphics card.
  - *Queued as one follow-up task, after the solver work:* a more flexible study-report
    template (question, result and meaning per table; free sections; figures next to their
    tables), shared number formatters and a verdict helper, the study contract in the contract
    reference, one common shape for OpenFOAM and gpucfd run records (same clock, time-step limits
    stored), rule constants in `numerics.yaml`, not in comments or code, a flag for an
    uncertainty estimate that is not reliable, and reading a study's headline from its data.
  - *Left for the student to decide:* one CPU run of OpenFOAM's finest grid with the live stop
    (about 23 minutes), which would turn the 1740 s target time from an estimate into a
    measurement.
- **How it was verified:** the dev agent recomputed the extrapolation by hand from the benchmark's
  table and got the same picture (converged drag 3.195, order about 1.1); the numbers quoted here
  are the ones the study script computes (3.1952, order 1.08). The catalogue, study, layout and
  tool tests pass, and case 4's report was rebuilt with the new insight and the link (contract
  OK, verdict unchanged).

### Case 4b: a study report that puts gpucfd next to OpenFOAM (`d10a2c1`)
- **What:** a second report for case 4 (the cylinder shedding vortices, Re 100):
  `results/dfg_cylinder_re100/gpu_study.html`, written by the new script
  `cases/dfg_cylinder_re100/gpu_study.py`. It shows OpenFOAM's four grids, gpucfd's three grids
  with equal cells, gpucfd's three grids with cells clustered around the cylinder, and the three
  check runs, side by side: accuracy, cost, what limits gpucfd, and the outcome against the
  target. It is listed in the catalogue as case 4b (verdict WARN: usable, with caveats).
- **Why it matters:** until now these results were only in `gpucfd/DESIGN.md` and in data files.
  The supervisor can now read them in one page, with the limits stated as plainly as the result.
- **No run was made.** The page is built from the files recorded on 2026-11-12 and 2026-11-13,
  and every number in it is computed by the script from those files. Rebuilding it does not need
  the graphics card.
- **The page, as question, result, meaning:**
  1. *Does gpucfd reach the target (drag within 0.25 % of the benchmark in less time than
     OpenFOAM's whole study)?* Drag 0.09 % and 0.16 % below the benchmark (two positions of the
     cylinder in the grid), study time 1421 s and 1375 s against about 1743 s. Yes, as defined.
     The page's headline and first key number say in the same breath that the benchmark's finest
     value is itself about 0.2 % low, and what the drag is against the value it is heading for
     (the correction above).
  2. *How well does gpucfd know its own answer?* Its three grids alone give an error estimate of
     0.6 to 0.75 % for the drag; OpenFOAM's give 0.02 %. So gpucfd is close to the benchmark, but
     could not prove it without the benchmark.
  3. *What does it cost?* Per cell and time step gpucfd is 25 to 45 times cheaper than OpenFOAM.
     It spends that on more cells and many more time steps. OpenFOAM has the same drag accuracy
     after 153 s, on its second grid.
  4. *What holds gpucfd back?* On its finest grid 70 % of the time steps are there only to keep
     the calculation stable, a finer grid needs a stricter pressure setting (twice the cost per
     step on the finest), and only 21 % of a clustered grid's cells are in the band around the
     cylinder, because the band crosses the whole channel.
- **Three things the page shows that were not written down before:**
  - *The benchmark itself is not exact at this level.* Its own three finest levels point to a
    drag 0.21 % above the value we compare with, and OpenFOAM's converged drag agrees with that
    to 0.04 %. Measured against it, gpucfd's drag would be 0.31 % and 0.38 % low, not 0.09 % and
    0.16 %. "Closer to the benchmark than OpenFOAM" does not mean "more accurate than OpenFOAM":
    OpenFOAM's finest grid, at +0.252 %, would itself narrowly fail the 0.25 % test.
  - *"About 50 times faster per cell and step" compared two different clocks:* gpucfd's solver
    time without the force output against OpenFOAM's whole elapsed time. With the whole elapsed
    time on both sides it is 45 times with equal cells and 25 times with clustered cells.
  - *gpucfd and OpenFOAM do not give the same answer in every output.* By the rule fixed before
    the runs (the two results differ by less than their two error estimates added), the finest
    clustered grid agrees with OpenFOAM in 6 of 7 outputs; the lift minimum does not.
- **Rewritten in plain words after the student said the first version was too hard to read.**
  The summary now opens with three short sentences and seven short key numbers (before: a ten-line
  paragraph of percentages). A list of the words used (reference, drag, lift, cells across the
  diameter, explicit and implicit time step, pressure setting, error estimate, the grid names)
  comes first in section 1. Every table is headed by the question it answers and ends in "Result"
  and "What it means"; tables have fewer columns. The same facts are in it, and the limits are
  still stated as plainly. The study template gained an optional list of terms.
- **The implicit scheme's result was added the same day** (stage 3, the entry above). The page now
  also answers "does an implicit time scheme make it faster?": two tables (each grid with both
  schemes; the study with and without it), a new bar in the time figure, and the study time a
  reader should take away: 1005 s, with 1421 s for the explicit scheme. It says that the estimate
  made beforehand (550 to 800 s) was missed, and why the finest grid still needs the strictest
  pressure setting (the implicit scheme splits a step into three unequal parts, and the shortest
  part decides). The verdict is still WARN.
- **Case 4's main report now links to this page** (one new line among its key findings). It was
  rebuilt for that: its contract check passes and its verdict (WARN) and date are unchanged.
- **How it was verified:** the page passes the contract check for study reports and the figure
  layout check; a second build gives the same data file; the catalogue and its test pass; every
  section was looked at in a browser. The numbers checked against `gpucfd/DESIGN.md` agree with
  it (the drag, lift and frequency errors, the study times, the time-step and cell-count
  factors), except the per-cell speed factor above and one rounding (OpenFOAM's frequency error
  is -0.12 %, not -0.13 %).
- **Open:** the report takes OpenFOAM's study time with a live stop (1743 s) from a replay of
  the recorded runs; the finest OpenFOAM grid, 81 % of that time, was never run with the stop.
  The finest clustered grid with the longer band of small cells is still not run.

### gpucfd: the implicit time scheme now works around bodies (stage 3) (`5bec14d`)
- **What:** the GPU solver has two ways of stepping in time. The explicit one must take very small
  steps on fine grids (the step shrinks with the cell size squared, because of friction in the
  fluid). The implicit one handles friction differently and its step only has to follow the flow
  speed. Until now the implicit one blew up as soon as there was a body in the flow, and nobody
  knew why, so every flow around a body used the slow small steps. The cause is found and
  removed; the implicit scheme is verified with bodies; and its worth is measured on case 4 (the
  cylinder shedding vortices).
- **Why the old scheme blew up.** Two separate faults.
  - *A plain oversight:* the scheme pushed on the grid points inside the body with a pressure
    that does not exist there. The error tripled every step. The explicit scheme already skipped
    those points.
  - *The real one:* the solver makes the flow obey two rules near a body. Rule 1: no fluid
    appears or vanishes (mass is conserved). Rule 2: the fluid sticks to the wall, which the
    solver imposes through helper values just inside the body ("ghost" values, copied from the
    fluid next to them). Each step it fixed rule 1 by nudging the fluid, which broke rule 2 a
    little; then it refreshed the ghost values, which broke rule 1 a little; and that small
    leftover was added to the pressure. The implicit scheme splits a step into three unequal
    parts, and going from a long part to a short one multiplied the leftover by four. Above a
    certain strength of this feedback the error grows every step.
  - *How sure:* a small separate program that imitates one time step predicts how fast the error
    grows (17.5 % per step on case 4's grid); the solver showed 17.4 %. It also explains why the
    explicit scheme never blew up: its limit is higher (and that limit was not known before).
- **The fix:** fix both rules at once. The pressure equation now knows that the ghost values will
  move with the fluid, so after each step mass is conserved *and* the fluid sticks to the wall.
  Then there is no leftover to feed back. The equation is harder to solve than before (it lost a
  symmetry the old solver relied on), so it got a new solver loop. The explicit scheme was not
  touched.
- **The checks: question, result, meaning.**
  - *Does it still blow up?* 15 runs of the situation that used to blow up, with small and large
    time steps, two positions of the cylinder, equal and clustered cells; and the real flow at
    the large time step. Result: the error shrinks every time. Before, it grew 17 to 59 % per step.
  - *Are both rules really kept?* After a step the ghost values match the fluid to the last digit
    the computer holds, and mass is conserved to the solver's tolerance. In the explicit scheme
    the ghost values are off by about one part in ten thousand: small, known now, left alone.
  - *Is it as accurate as the explicit scheme?* On the flow between two cylinders with an exact
    textbook answer the errors are the same to three digits on all three grids, and halving the
    cell size still cuts the error by four or more.
  - *Does the settled flow depend on the time step, or on the scheme?* No: a step four times
    smaller gives the same flow to twelve digits, and so does the explicit scheme (to seven
    digits on the one grid with unequal cells, where the explicit run stopped earlier). On the
    slow-flow cylinder (Re 20) the drag agrees to seven digits and arrives in a third to a half
    of the time.
  - *Is it accurate in time?* Halving the time step cuts the error by four, as designed, on the
    flow around the cylinder while it is still changing; and it lands on the explicit scheme's
    flow.
  - *Did anything old change?* 38 stored results of the earlier solver (with and without bodies,
    equal and clustered cells) are identical to the last bit, and the whole test ladder passes.
- **What it is worth on case 4** (cells clustered around the cylinder, three grids).
  - *Same answer?* Yes, on every grid: drag and shedding frequency agree with the explicit runs
    to 0.001 %, lift to 0.014 %. The rule fixed beforehand allowed 0.05 % and 0.3 %.
  - *How much faster?* The finest grid takes 764 s instead of 1180 s. The whole three-grid study
    takes 1005 s (17 minutes) instead of 1421 s; OpenFOAM's study takes about 1740 s.
  - *What it means:* a real gain, 1.5 times on the grid that dominates, but less than the 2 to 4
    times I estimated before building. The implicit scheme takes 3.3 times fewer steps there, but
    each step costs twice as much (the harder equation), and the pressure still has to be solved
    as tightly as before. On the coarsest grid the implicit scheme is slower, so the study keeps
    the explicit scheme there: both give the same answer, so each grid can use the faster one.
- **Limits, to say with it.** It makes the solver faster, not more accurate: the drag is still
  0.3 % below the value the benchmark converges to, where OpenFOAM is within 0.04 %, and
  OpenFOAM's coarse body-fitted grid is still far cheaper for this accuracy. One flow, in 2D, one
  run per grid.
- **One extra run the new scheme made affordable:** the finest grid with the small cells reaching
  further behind the cylinder (the wake needs them). *Question:* does the lift come closer?
  *Result:* lift 0.7 % low instead of 0.85 %, the shedding frequency error halves, drag
  unchanged; the study with it takes 1308 s, still under OpenFOAM's 1740 s. *Meaning:* a small
  step in the right direction, smaller than what moving the cylinder half a cell does to the
  lift (0.3 %), so it is a hint, not a result, until it is repeated at a second position. That
  repeat, and the implicit run with the cylinder moved, were not made.
- **Open, in order of expected gain:** a scheme that solves the pressure once per step where
  this one solves it three times (that is where the time goes now); cheaper friction solves;
  and whether to give the explicit scheme the same two-rules-at-once fix, which would remove its
  hidden limit but change all its body results in the fourth digit (a decision for the student).
- **Where:** the cause, the formula, the design and every number are in
  [`gpucfd/DESIGN.md`](gpucfd/DESIGN.md), "IMEX with immersed bodies, stage 3". Tests:
  `gpucfd/tests/test_imex_bodies.py` (rung 14), the imitation program
  `gpucfd/tests/imex_body_model.py`, case 4's runs `gpu_run.py clustered C1i C2i C3i` and their
  comparison `gpu_imex.py`.
- **Checked a second time by the dev agent**, to be sure the source in the repository is what
  produced the numbers. The solver was built again from the source files; the 38 stored results
  of the older solver are still identical to the last bit; the new tests (rung 14) pass, and every
  number they write equals the agent's to the last digit (only the timings differ); the imitation
  program gives the same limits and growth rates. Case 4's runs were not repeated (25 GPU-minutes
  that would answer nothing new): their numbers were read again from the stored records.
- **What became of the agent's 22 suggestions.**
  - *Done now.* A small tool, `cfdlib.gpu.pick_scheme`, compares the two time-step limits of a grid
    and says which scheme should be faster; on case 4's three grids it names the scheme that was
    measured faster. The workflow's paragraph on this shrank to one line that points to the tool.
    The solver's notes for developers gained three habits: when a scheme blows up, imitate one
    step in a small program and compute its growth before changing the solver; time 200 steps of
    a prototype before writing down an expected speed-up; save a new comparison reference after a
    change that alters results (saved for this one). The machine notes say how to watch a running
    job from Windows. The guidance says that steady flows around bodies settle sooner with the
    implicit scheme (one case: same drag, 18 s against 50 s). The roadmap's status table and two
    stale lines in the design file were corrected.
  - *Passed on.* The study report (case 4b) got a section with these results from the session that
    wrote it (checked: it rebuilds, passes its contract, and the page list and links are in order). Four tool items (one table of case 4's runs in place of names with suffix letters,
    one way to check a pressure tolerance where the run script now has two, a time step taken from
    the measured flow speed so the coarsest grid stops raising a warning, and repeated timings with
    their spread) joined the waiting task "Study-report tooling and run records".
  - *Written down, not done* (in `ROADMAP.md` and the design file's limits): one pressure solve per
    step in place of three, with the smaller solver savings; the explicit scheme's hidden stability
    limit and the choice whether to give it the new fix (the student's decision; recommended only
    if a case comes near the limit, because a run that crossed it would blow up visibly and the fix
    means making every explicit result with a body again); the limit's value on clustered and 3D
    grids, which the small program cannot compute yet; a slowly fading wobble at the outflow in
    one test without flow; the explicit scheme stopping a hair short of steady on one grid (inside
    its tolerance); splitting the 1500-line design file into one file per stage.
  - *Not done, with the reason.* Timings were not repeated: every time quoted differs from its
    neighbour by far more than the 3 % seen between two runs of one grid.

### gpucfd: the stage 2 result checked, and the tools and corrections it called for (`4b12f39`)
- **What:** three check runs on case 4 (the cylinder shedding vortices, Re 100), and every
  suggestion the agent of stage 2 left behind was acted on: new tools, corrected claims, updated
  guidance.
- **Why it matters:** the stage 2 result (drag 0.1 % from the benchmark) looked almost too good,
  and that agent had found that an earlier quoted number was a lucky one. Before a number goes to
  the supervisor it should survive a second try.
- **The check runs: the question each one asks, what came out, what it means.**
  1. *Was the 0.1 % luck?* The solver lays a grid of rows and columns over the flow, and the round
     cylinder cuts through its cells; where the cuts fall changes the answer a little. The whole
     study was repeated with the cylinder moved by half a cell. Drag came out 0.16 % from the
     benchmark, against 0.09 % the first time, in 1375 s against 1421 s. The target (0.25 %, under
     1740 s) holds both times. What can be quoted: **drag 0.1 to 0.2 % below the benchmark**.
  2. *Was the time comparison fair?* The older run with equal cells had used a looser setting
     inside the pressure solver than the new runs. Rerun with the rule the new runs follow: the
     same drag, lift and frequency, and 328 s for its study where 315 s was recorded. The
     comparison is now like for like, and it barely moved.
  3. *Why were lift and shedding frequency worse on the clustered grid?* The suspicion was the
     larger cells in the wake, where the vortices form. With the small cells reaching 4 cylinder
     diameters behind the cylinder instead of 1.5 (25 % more cells), lift and frequency are as good
     as on the grid with equal cells. So the wake needs small cells for longer than first given.
- **What this changes in the stage 2 story:**
  - *Fewer cells did not buy much time.* Matching drag alone, the clustered grid is a quarter
    faster than equal cells (241 s against 328 s). Matching lift and frequency too, it takes
    nearly the same time (308 s) with a third of the cells. Grids this small are limited by fixed
    work per time step, not by the number of cells.
  - *gpucfd is about 50 times faster than OpenFOAM per cell and time step on this case,* not 70:
    the fair pressure setting costs that much.
  - Still true: OpenFOAM reaches the same drag accuracy in about 3 minutes with a small grid
    fitted to the cylinder. And the drag range is from two positions; lift is about 1 % low.
- **New tools, so the next case does not repeat this by hand** (`cfdlib/gpu.py`):
  - `tolerance_check`: finds the pressure setting a grid level may use. Stage 2 lost 45 minutes
    to a setting tuned on the coarsest grid only.
  - A run now saves what it has recorded as it goes (`partial`). Before, stopping a run from
    outside threw its data away, which is why the agent could not apply its own stop rule.
  - The live check that ends a run can be made as often as wanted (`watch_every`), not only at
    each log line.
  - `position_spread`: reports how much the outputs move with the position of a body in the grid.
  - `gpucfd.grids.body_band`: the measured default for the band of small cells around a body.
  - `check_fit` now warns that on small grids fewer cells do not mean less time.
- **Corrected claims:**
  - A test of the immersed-body code passed only because of a lucky grid position (the pressure
    difference across the cylinder, 0.2 % off where other positions give up to 2.4 %). It is now
    judged against the measured spread; the lift check beside it had the same weakness.
  - The README no longer quotes single-position numbers as accuracies.
- **Guidance and records:** the workflow's advice on choosing gpucfd is rewritten (what
  clustering buys, checking the pressure setting on every grid, quoting a body's result with its
  spread over positions, saying which outputs "the same accuracy" covers). Case 4's insight log
  has four new entries, marked as coming from this work and not from the agent that solved the
  case. The machine file has two new pitfalls. `dev/LOOP.md`: an agent given limited work is
  asked to send back what it would change outside its limits.
- **How it was verified:**
  - The new run tools have their own test on the GPU, `tests/test_gpu_tools.py` (8 checks, all
    pass): the new options do not change a run's samples; the saved-as-it-goes record holds
    exactly what the run had sampled; a run stopped from the keyboard returns its record; the
    tolerance check picks the loosest setting that gives the same answer.
  - The immersed-body test was rerun with the two reworded checks: 12 of 12 pass, and its numbers
    are the ones recorded before (only the judging changed).
  - The solver itself was not touched, so the bit-identity check and the full ladder were not
    rerun.
  - The tool, catalogue and layout tests pass; all Markdown links resolve; case 4's report was
    rebuilt with the four new insights and passes its contract (verdict unchanged, WARN).
  - The check runs' numbers were computed by the stage 2 agent's analysis script; the dev agent
    read them from `gpu_clustered.json` and did not recompute them independently.
- **Open:**
  - The finest grid with the longer band of small cells has not been run (about 30 GPU-minutes).
    It should bring the lift error below 1 %, at a study time close to the 1740 s limit.
  - Two positions give a range, not a statistic. Four positions on the middle grid take 8
    minutes; on the finest, an hour.
  - Case 4's report does not show the gpucfd runs. They are in `gpucfd/DESIGN.md` and
    `results/dfg_cylinder_re100/gpu_clustered.json`. A separate study report, like the GPU study
    of case 5, is proposed.
  - Stage 3 (an implicit time scheme with bodies) is the next large step.

### gpucfd: cells clustered around a body (stretched grids, stage 2) (`1f14ce2`)
- **What:** the GPU solver can now use small cells around a body and larger ones far from it
  (ROADMAP step 2, stage 2 of 3). Until now, with a body in the flow, every cell had to have the
  size the body needs.
  - `grids.clustered(n, length, band, h, grow)` makes such a grid: one cell size inside a band
    around the body, growing with the distance from it.
  - The solver parts that deal with a body now use the true cell sizes: the rules that put the
    body into the flow, the pressure solver, and the force and surface-pressure output.
  - Case 4 (cylinder shedding vortices, Re 100) has three clustered grids, C1 to C3, with 35, 70
    and 140 cells across the cylinder's diameter (`gpu_run.py clustered`, `gpu_clustered.py`).
- **Why it matters:** near a body the flow changes quickly and needs small cells. With one cell
  size, that meant small cells in the whole channel. This is the stage that tests, on case 4,
  whether clustering makes gpucfd competitive around bodies. The target was fixed in advance: drag
  within 0.25 % of the benchmark in less wall time than OpenFOAM's study (about 1740 s).
- **The tests: the question each one asks, what came out, what it means.** The design, the pass
  rules and a cost estimate were written in `gpucfd/DESIGN.md` before the solver was changed.
  1. *Did the change break what already worked?* On grids with equal cells the new build gives
     the same numbers as the old one to the last digit (26 of 26 stored results, two of them with
     bodies), and every older test passes. Everything verified earlier still holds.
  2. *Is the new code the old method, or a different one?* Fed a grid with equal cells, the new
     code path behaves like the old code: same rules for the body, results equal to about one part
     in a trillion. So it is the old method plus uneven cells.
  3. *Is the force output right on uneven cells?* With no body in the flow the computed force is
     zero (7e-16), as it must be. Drag and lift can be trusted as arithmetic.
  4. *Is the flow around a body still accurate on uneven cells?* This was the risky part. On a
     flow with a known exact answer (fluid between a rotating and a fixed cylinder), with the cell
     size varying by a factor 3, halving the cells cuts the error by 5 to 6; a factor 4 is the
     goal. The method works; the backup method was not needed.
  5. *Same answer with fewer cells?* On the steady cylinder (Re 20) the clustered grid gives the
     drag of the equal-cell grid (-0.56 % against -0.65 % from the benchmark) with a quarter of
     the cells. Clustering pays, but by 4 times, not 30: the grid is still rows and columns, so
     the band of small cells crosses the whole channel.
  6. *Why did one check fail?* The pressure difference between the front and the back of the
     cylinder came out 1.9 % off where the equal-cell grid had 0.2 %. Moving the cylinder by a
     fraction of a cell on the equal-cell grid moved its own answer between 0.2 % and 2.4 %. So
     the earlier 0.2 % was a lucky position, not the grid's accuracy, and the new code is not at
     fault. Pressure read at the surface is good to about 2 % at this resolution; drag moves by
     only 0.25 %. The check now compares against that range and still reports the first failure.
  7. *Is the clustered grid faster at the same accuracy?* Case 4, same cells at the cylinder as
     the finest equal-cell grid: drag error 1.07 % against 0.99 %, in 241 s against 315 s. Yes,
     about a quarter faster. Lift and shedding frequency are a little worse, because the cells
     behind the cylinder are larger.
  8. *Does it reach the target?* With 140 cells across the diameter the drag is 0.1 % from the
     benchmark and the three grids together take 1421 s. The target is met as it was defined.
- **Key numbers, case 4, differences from the FeatFlow benchmark:**

  | Run | Cells | Drag | Lift maximum | Shedding frequency | Wall time, all grids |
  |---|---|---|---|---|---|
  | gpucfd, clustered, 70 cells per diameter | 115 thousand | -1.07 % | -5.2 % | +0.54 % | 241 s |
  | gpucfd, equal cells, 70 cells per diameter | 442 thousand | -0.99 % | -3.4 % | +0.29 % | 315 s |
  | gpucfd, clustered, 140 cells per diameter | 459 thousand | -0.09 % | -0.85 % | +0.13 % | 1421 s |
  | OpenFOAM, four grids fitted to the cylinder | 215 thousand | +0.25 % | +0.41 % | -0.13 % | about 1740 s |

- **Read the 0.1 % with these cautions:**
  - It may be partly luck. Judged only by how the answer changes from grid to grid, without the
    benchmark, the three grids support about 0.75 %. The finest grid is better than the trend,
    and check 6 shows that the position of the cylinder in the grid matters.
  - It is drag. Lift is about 1 % low.
  - OpenFOAM is still the cheaper tool for this accuracy on this problem: its second grid has the
    drag at 0.24 % after about 3 minutes. We beat the time of its whole four-grid study.
  - 1421 s is the second attempt (next point). Tuning runs are not counted, on either side.
- **Found while building:**
  - *The pressure solve has to be stricter on finer grids.* With small cells and the present time
    scheme the time step is tiny, and a tolerance that was fine on the coarse grid then leaves
    noise in the forces. The first finest-grid run never settled into a repeating cycle: its drag
    peaks jittered by 0.07 %. Case 4's own tolerance rule, applied on each grid, gives 1e-8, 1e-9
    and 1e-10 for C1, C2, C3; the rerun settled in 3 s of flow. The stricter setting doubles the
    cost of a step. The finest equal-cell run of 2026-11-12 fails the same check slightly (lift
    off by 4e-4); it was not rerun, and by that rule its 315 s would be about 385 s.
  - *A mistake of mine, caught by a number that looked too good.* On the new grids the pressure
    solver stopped too early (2 iterations where 8 were expected), because a limit inside it was
    written for equal cells. Fixed before any result was used; the count is now 5.
  - *A stop rule I did not follow.* The first finest-grid run was to be stopped at 8 s of flow if
    not settled. Stopping it would have thrown its record away, so it ran to 15 s (45 minutes).
    The limit is now part of the run, which then stops by itself and keeps its record.
  - *The estimate made beforehand was too pessimistic.* It expected the clustered grid to lose
    check 7 and to miss the target (0.35 % in 1870 s). A step costs less than assumed, because
    with a body the new pressure smoother needs fewer iterations (5 against 7.7).
- **Open when this was committed, answered by the check runs in the entry above:** whether 0.1 %
  is typical (it is 0.1 to 0.2 %), the lift on the clustered grids (a longer band of small cells
  fixes it), and the out-of-date wording in `workflow/WORKFLOW.md` (rewritten). Two numbers in
  this entry changed with those runs: the equal-cell study takes 328 s, not 315 s, once it is held
  to the same pressure rule, and the estimate of 385 s for it was too high.
- **Still open:** stage 3, an implicit time scheme with bodies. The finest grid is 83 % of the
  time; its time step is 3.3 times smaller than the flow speed alone would need, and the strict
  pressure tolerance comes from the same cause.

### gpucfd: stretched grids, stage 1 (walls, no bodies) (`bdee4b7`)
- **What:** the GPU solver can now use cells of unequal size, for example small at the walls and
  large in the middle (ROADMAP step 2, stage 1 of 3).
  - A direction is described by its face positions: `NavierStokes(..., grid=(None, y, None))` with
    `y = grids.tanh(64, 1.0, ratio=8)` from the new `gpucfd/grids.py` (ratio = largest / smallest
    cell). Works with walls, inflow, outflow and both time schemes; periodic directions stay
    uniform. `dt_cfl()` gives the time step from the local velocities.
  - The equations use the true cell sizes in every term. Convection is written so that it neither
    creates nor destroys kinetic energy on any grid.
  - **Pressure solver.** On long thin cells the old smoother stalls (157-192 iterations per solve
    at aspect ratio 64). It now solves whole lines of cells at once (zebra line relaxation, along
    x and y in turn), and the restriction is the exact transpose of the interpolation.
  - Immersed bodies on stretched grids are refused for now (stage 2).
- **Why it matters:** this is the first of the three stages toward case 4's target (drag within
  0.25 % in less time than OpenFOAM). It shows the hard part, the pressure solver, works.
- **How it was verified** (design and pass criteria were written in `gpucfd/DESIGN.md` before any
  run):
  - Uniform grids are bit-identical to the build before the change (18 of 18 arrays), and the whole
    ladder passes.
  - New rung 12 (`gpucfd/tests/test_stretched.py`, 48 checks): the new code fed uniform positions
    equals the old code to round-off; Laplacian against an independent numpy reference (6e-15);
    direct solve; order 1.99-2.01 for the pressure equation; Poiseuille, a decaying channel mode,
    Kovasznay flow and an open channel at order 2.02-2.07; kinetic energy of convection conserved
    to 1e-17; cavity Re 100 on clustered grids within 0.009 U of Ghia and converging to the
    uniform solution; runs reproducible bit for bit.
- **Key numbers:**
  - Pressure iterations per projection (cavity): 7.0 uniform, 7.0 / 8.0-8.7 / 9.7-9.9 at wall
    aspect ratio 4 / 16 / 64, the same on 128^2 and 256^2. The old smoother: 21 / 60-67 / 157-192.
  - Cost: a step on the stretched code path takes about twice a uniform step with the same cells
    (8.5 against 4.3 ms on 128^2), and 45 % more at aspect ratio 64.
  - Benefit, first fair number (cavity Re 100, same accuracy as uniform 256^2): the best clustered
    grid needs 23 % fewer cells (50 against 66 thousand) but 1.9 times the time (36.6 against
    19.2 GPU-s). On this flow stretching does not pay: its wall layers are not thin.
  - Where it does pay (added afterwards, exact solution, a wall layer a tenth of the domain thick):
    7 times fewer cells across the layer for the same error.
- **Found while building:** conjugate gradients stalled with line relaxation although plain
  multigrid converged. The multigrid cycle was not symmetric (10-17 %); averaging the residual
  onto the coarse grid is the cause. Fixed with the transposed interpolation. A one-block version
  of the small levels was built, measured slower, and removed.
- **Open:**
  - The benefit for bodies is not measured yet: that is stage 2 (bodies on stretched grids), then
    stage 3 (IMEX with bodies).
  - The stretched code path is 2 times slower per step than the uniform one.
  - 3D grids clustered in two directions converge slowly (23-38 iterations at ratio 16, growing
    with the grid); they need plane relaxation. 3D clustering in one direction is fine in the
    tests that ran (channel mode), but its solver cost was not measured.

### Repository aligned with the goal: workflow, tools, agent, and the loop (`96553e7`)
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
