# Agentic CFD workflow

For any coding agent. Paths in this file are relative to the repository root, except `references/`
and `machines/`, which are next to this file in `workflow/`.

You turn a problem statement into a trustworthy simulation report. The reader is an engineer who
will mostly read the summary and must be able to check every claim. Four rules:

1. **Code computes every number.** You write the code; you never type a result, reference value
   or formula output into the report by hand. Theory is solved symbolically (SymPy) and self-checked.
2. **Every choice is documented.** Each OpenFOAM setting belongs to a decision with why / effect /
   alternatives. A checker refuses to render the report otherwise.
3. **Nothing from memory.** Reference data, correlations and constants come from the problem's
   `reference/` files or a cited source. If you must use an approximate value, say so in the report.
4. **Log insights live, like a lab notebook.** Part of your job is to improve this workflow for the
   agents after you. The moment something fails, surprises you, needs a second attempt, or is unclear
   in these guidelines, open an insight (before you start fixing it):
   `insights.log(case, stage, problem)`; fill in diagnosis, fix, lesson and guideline with
   `insights.resolve(...)` once solved. Do not save them up for the end: memories of a long session
   get lost and distorted. Expect to log 5-15 entries; a run with none was not looked at closely.

## Folder layout

```
cfdlib/                       shared tools (do NOT modify; log needed changes as insights, stage "tooling")
  foam.py        render_case, run (timeout, PID file), stop, running; read fields (cell values, values on a
                 boundary patch) and postProcessing .dat files
  structgrid.py  a published structured grid (PLOT3D file) as an OpenFOAM mesh, point for point, with named
                 patches (read_plot3d_2d, index_range, write_polymesh): for benchmarks that provide their grids;
                 centres gives the cell and wall-face positions, so a profile is a column, without sampling
  monitor.py     parse_log, assess (health checks), preflight (short test run + run-time projection),
                 LiveStop / judge / settled (end a run live once its physical check holds),
                 peek (look into a run at any moment: progress, health, verdict, history as numbers),
                 watch (a self-refreshing page with the history figure, for the human engineer)
  verification.py error norms, observed order, Richardson extrapolation, GCI, level_sequence
                 (convergence of any sequence of levels, including a reference's own)
  signals.py     periodic signals: cycles, refined peaks, periodicity check, FFT frequency; signals that
                 fluctuate about a level (turbulence): stationary (mean, statistical error, drift),
                 settle_time (where the start-up ends), stationary_watch (a live stop on the error)
  plots.py       figure style and sizes (figure, wide, sketch); to_svg checks for overlapping text
  report.py      m(), tex_num(), tex_fix() (fixed decimals for table columns), tex_pct(), LETTERS, load_yaml,
                 environment_info(), render(); ready table cells m_fix, m_pct, m_times, m_pow10, m_sig
  perf.py        performance numbers per run (summarize, machine) for the performance appendix
  contract.py    the report contract checker (also --decisions mode)
  study.py       a second, focused report that compares tools or settings (render_study, verdict), e.g. GPU
                 vs CPU, or one solver against another
  runrecord.py   a run record of either tool in one shape (read): wall time and cost per cell and step
                 under the same names for OpenFOAM and gpucfd
  gpu.py         gpucfd as a solver: check_fit (does the problem fit it?), pick_scheme (time scheme per
                 level), time_step (the step and which limit set it), run_transient (sampled run with
                 health checks and a live watch), write_run (runs/<name>.json + .npz like OpenFOAM runs)
  insights.py    the live insight log (checks typesetting when you log; refuses to overwrite a filled entry)
  foamdict.py    OpenFOAM dictionary parser (used by the checker)
cases/<name>/                 one folder per problem
  problem.yaml   GIVEN by the user: description (short overview), setup (list of facts: geometry,
                 boundary conditions, definitions), given data, requested outputs [+ reference/].
                 Do not change the physics. Pass description and setup to the report unchanged.
  reference/     user-supplied reference data with citations
  numerics.yaml  your numerical plan (meshes, run lengths, tolerances)
  decisions.yaml every choice, written WHILE you set up the case
  insights.yaml  the live log (cfdlib.insights)
  template/      the OpenFOAM case (0/, constant/, system/) with {{ placeholders }}
  theory.py / analytical.py, sketch.py, run.py, analyze.py
results/<name>/  raw data, results.json, report.html
```

**Solved cases are there to study, when you want to.** `cases/README.md` lists every solved case
with what its folder shows (a mesh generated in Python, a live stop rule, a time-step study, a heat
transfer setup, a GPU study...). Each folder is a complete record: the numerical plan, the
decisions, the code and the insight log of the agent that solved it. You decide which ones are worth
your time, if any: the ones close to your problem, or one that shows a technique you need. Searching
`cases/*/insights.yaml` for a keyword shows whether a problem you just hit was met before. Copy
structure and technique, not numerical settings. `cases/plane_poiseuille/` is the smallest complete
case, if you want to see the full set of files once.

## Stages

### 0. Intake
- If the user describes the problem in words and there is no `problem.yaml` yet, write
  `cases/<name>/problem.yaml` from the description in the worked examples' format: physical inputs
  only, no numerics. Ask for anything essential that is missing, show the file to the user, and
  start only after they confirm. From then on it is the given problem.
  `python -m cfdlib.contract --problem <file>` says whether the report will accept it as written.
- Read `problem.yaml` and any `reference/` files. List the requested outputs (they get letters a, b, c...).
- Compute the governing dimensionless numbers (Re, Ma, ...) in code. Decide which physics apply.
- If data is missing or contradictory, choose a documented assumption and flag it in the report's
  limitations. Never silently invent inputs.
- If a reference gives no uncertainty, estimate one (its grid, its printed digits, a second source)
  and record how; you will need it in stage 5. A reference's uncertainty is more than its grid
  error: count what its authors say about boundary conditions and iteration too (case 7's band held
  the grid error of both sides only, 0.024 %, while NASA's page states 0.2 % for boundary choices).

### 1. Theory: what can be known without the simulation
- Exact solution available: derive it with SymPy, self-check it (satisfies equations and boundary
  conditions), evaluate the requested quantities. See `cases/plane_poiseuille/analytical.py`.
- No exact solution (usual case): say so in `theory.intro`, and give what theory CAN say: limiting
  cases, scaling, conservation laws, symmetry, sign of quantities, expected flow structure.
  See `cases/lid_driven_cavity_re100/theory.py`.
- Every derived quantity gets: why it matters, the meaning and value of each symbol in this
  problem, and the formula with the numbers substituted (hand-check).
- A series solution converges slowly near where it starts (case 6: the thermal-entrance series still
  needed more than 40 terms at 0.5 D). Check the truncation at the smallest position you use, and
  check the series against an independent method (case 6: a method-of-lines solution, 4e-6 apart).
- When a requested output is a limit (fully developed, far field, long time), work out how close the
  domain gets to it. Case 6's outlet at 120 D was still 0.53 % above the fully developed Nusselt
  number. Either compare at the same position or extrapolate the simulation and the exact solution
  the same way.

### 2. Setup and decisions
- **Choose the solver first, and record it as a decision (kind `solver`).** OpenFOAM is the default. Consider
  gpucfd (`cfdlib.gpu.check_fit`, limits in `gpucfd/DESIGN.md`) when the problem is incompressible, in a box
  (bodies inside it allowed), and laminar or turbulent with the eddies resolved on the grid (gpucfd has no
  turbulence model). What decides is the cost to reach the required accuracy, not speed
  per cell. Case 4 (DFG 2D-2, `gpu_run.py`, `gpu_compare.py`) measured both (study report
  `results/dfg_cylinder_re100/gpu_study.html`): gpucfd is 25 to 45 times faster per cell and step, but its
  grid is rows and columns with the body immersed in it. Cells can be clustered around a body
  (`gpucfd.grids`), which saves about 4 times in cells, not the 30 of a body-fitted grid, and with a body
  the explicit scheme's time step follows the smallest cell squared (`scheme="imex"` removes that limit at
  about twice the cost per step). OpenFOAM's O-grid had the drag within 0.25 % of the benchmark at 13432
  cells after about 3 minutes; gpucfd at 140 cells per diameter, after about 17 minutes (24 with the
  explicit scheme), was still 0.3 to 0.4 % below the value the benchmark converges to. So: OpenFOAM for
  bodies with thin boundary layers (Re about 100 and up on a cylinder), where gpucfd's implicit scheme
  shortens its study but does not change its accuracy per cell; gpucfd for flows its grid resolves well
  (periodic boxes, channels, cavities, low Reynolds numbers) and for steady flows with `gpucfd.steady`
  (grid sequencing).
  When you compare tools "at the same accuracy", say which outputs that covers: on case 4 the answer
  differed between drag alone and drag with lift and frequency.
- When gpucfd is the solver, `cfdlib/gpu.py` has the tools; what case 4 taught:
  - Grids n0 times 2^k with multigrid-friendly sizes (powers of 2 times 3 or 9); time step from
    `NavierStokes.max_dt`. Around a body, `gpucfd.grids.body_band` gives a measured default for the band
    of fine cells (the wake needs them for longer than the body does).
  - The time scheme is a choice per level (`pick_scheme`): implicit diffusion (`scheme="imex"`) pays
    on fine grids, and both schemes give the same answer, so the levels of one study may differ.
  - Tune `pressure_tol` with a same-answer rule on every level, the finest included
    (`tolerance_check`): with bodies the pressure noise grows as the time step shrinks (case 4:
    \(10^{-8}\) on the coarsest level, \(10^{-10}\) on the finest).
  - An immersed body's result shifts with where the grid cuts it, and a grid study does not see that
    error. Run the finest level again with the body moved half a cell and quote the range next to the
    grid-convergence index (`position_spread`).
  - `run_transient` samples, checks health and records the run like an OpenFOAM run. Stop each level on
    the physical check (`stop_on_watch`, with `watch_every` about one period), not after a fixed time: a
    coarse gpucfd level can be far from the fine answer (case 4: G1 drag 16 % low), and the next level
    then takes several flow-through times to settle. Give it `partial`, so that a run you have to kill
    keeps its record.
  - A turbulent flow computed without a model never becomes steady or periodic: every output is a time
    average with a statistical error, which no grid refinement removes. The tools: `gpucfd.turbulence`
    (a start field that turns turbulent, plane averages formed like the solver's own fluxes),
    `run_transient(adapt_cfl=...)` (the time-step limit moves with the flow) and
    `cfdlib.signals.stationary` (mean, standard error, drift). What was measured on a channel, with
    costs: `gpucfd/DESIGN.md`, "Turbulence resolved without a model".
- Find the closest OpenFOAM tutorial (`$FOAM_TUTORIALS`) and start from it; make minimal changes.
- Read `references/openfoam-primer.md` for what each file controls and the known pitfalls.
- Templates may contain Jinja loops: generating a whole `blockMeshDict` from Python is fine and is
  the way to mesh curved bodies (example: `mesh_spec()` in `cases/dfg_cylinder_re100/run.py`).
- **Edit files that contain LaTeX only with your file-editing tool** (Edit/Write in Claude Code),
  never through shell heredocs or inline Python: the shell eats backslashes (`\times` becomes a TAB). Cases 3 and 4 both lost
  time to this. Small one-off patch scripts piped through Bash count too (case 5 had a near miss).
- Write `decisions.yaml` as you go: one entry per choice with `kind` (physics / accuracy / solver /
  output) and `settings` patterns covering every setting in the case files. Check coverage as soon as
  one case is rendered: `python -m cfdlib.contract --decisions cases/<name>/decisions.yaml <case dir>`.
- Plan a mesh study: at least 3 meshes, refinement ratio 2 (needed for observed order and GCI).
  Transient problems also need a time-step study (or an argument, with numbers, that time error is
  negligible).
- Define a **physical** convergence/steadiness check for your problem (a balance, a conserved
  quantity, a monitored value that must stop changing) and its tolerance BEFORE seeing results.
  Normalised residuals alone are not enough. Transient problems: a per-step balance (change of a
  conserved quantity = boundary fluxes) is a strong check; it must use the solver's own time scheme.
- Pass/fail tolerances must discriminate: size them from the coarse pre-flight error and the expected
  order (what the finest mesh should reach), not from round numbers. A tolerance 100x looser than what
  is achieved proves nothing (case 3). Make the observed order of accuracy a rule of its own.
- At very small tolerances, budget for three separate errors:
  - the mesh error;
  - the bias of how an output is extracted (a fit, an extrapolation, a threshold crossing);
  - the differences between the simulated problem and the exact solution's idealisation (inflow
    profile, axial conduction).

  Measure the extraction bias by applying the same extraction to the exact solution. In case 6 the
  tail fit alone gave Nu_fd +8e-5 and the 5 % entrance length −7e-4. Both WARNs of case 6 sat at
  that level.
- When an assumption's dimensionless indicator is ambiguous (buoyancy at Ri 1e-3 and Ra 280), you can
  measure its effect with a same-mesh pair, with and without the effect. Report it on the local
  requested quantities as well as on the means: in case 6, gravity changed Nu by 0.15 % but made the
  top of the wall 11 % of ΔT warmer than the bottom.

### 3. Pre-flight test (mandatory, before any long run)
- `cfdlib.monitor.preflight()` on the coarsest AND finest meshes with a short run (a few % of the
  full length). It works in a separate `<case>_preflight` copy; render pre-flight and production
  cases into separate directories too, so re-running one never deletes the other. It takes the
  command as you would run it, so an MPI command works on a case that is already decomposed.
- It reports crashes, NaN, Courant number, residual/Courant/continuity trends and projects run time.
  Transient runs: the projection is reliable. Steady runs: pass `residual_targets`; the result is a
  **lower bound** (SIMPLE slows down later and on finer meshes).
- Print the header and first rows of every `postProcessing` file the pre-flight wrote and check
  each is non-trivial (case 4: a pressure integral wrote 0 at every step, silently).
- **Transient nonlinear problems: a short time-step study (3 time steps, ratio 2, on a coarse mesh)
  belongs in the pre-flight**, before the mesh study. It shows the observed order in time: PISO
  (1 outer corrector) turned out first order in case 4 and cost a full re-run.
- A pre-flight that starts from rest underestimates the cost of a developed unsteady flow (case 4:
  twice). Start it from a developed field (`mapFields` from a coarse run) or budget twice the projection.
- The pre-flight also lists settings the solver silently ignores (`foam.ignored_settings`, or
  `python -m cfdlib.foam check <case>`). Today that covers SIMPLEC (`consistent yes`) for solvers that
  have none, such as `buoyantBoussinesqSimpleFoam`.
- Steady runs: a short pre-flight ends long before convergence, so it cannot show what a converged
  iteration costs. Letting the coarsest mesh run to convergence once shows the linear-solver iteration
  counts at the end. A solver sitting at its `maxIter` there means its absolute tolerance is below
  round-off. In case 6 that made converged iterations 20× dearer and cost about 9 minutes of
  discarded runs.
- If it fails or warns, fix the setup and log an insight. No production runs on a failing pre-flight.
  If the projected time is excessive (the budget is in your machine file; about 15 min per run on
  the development PC), coarsen the plan and say so.

### 4. Production runs
- Run meshes coarse to fine. For steady runs, extrapolate the iteration count of the finest mesh from
  the coarser ones (it often grows 3-4x per refinement) before launching it.
- Stopping must be physical: make the physical check of stage 2 the stopping rule, and the planned
  end time only a maximum. How you apply it is your call; the tools are there to help:
  - `monitor.LiveStop(case, decide)`, passed as `foam.run(..., watch=...)`, applies your rule beside
    the solver every few seconds (no tokens), and ends the run cleanly once it holds (OpenFOAM's
    `abort` function object, serial and parallel).
  - `monitor.judge` / `monitor.settled` classify per-cycle or per-chunk values: settled, still
    changing, or at a floor. A floor means the changes stopped shrinking above the tolerance; it
    stops with a warning, and the spread becomes the iteration uncertainty. Case 4's M1 drag peaks
    jitter by 7e-4 against a 5e-4 rule, and waiting longer never helps.
  - For long runs, `monitor.DatTail` reads only what is new at each check. Looking at only the recent
    stretch the verdict depends on keeps a check cheap: 7.2 s → 0.14 s at a million steps in case 4.
    Use a longer stretch, or the whole history, whenever your rule needs it.
  - Example: `cases/dfg_cylinder_re100/run.py` (`periodic_rule`, `live_decider`, `do_live`).
    Replaying a rule on recorded histories (`do_live_check`) is a cheap way to test it before you
    rely on it.

  For steady SIMPLE runs that stop on residual targets, check per mesh that the targets actually
  satisfy the physical check, on the coarse meshes first.
- Steady runs can usually reach the same answer several times sooner (case 2 cavity: 3-4× on
  128², 6× on 256², about 11× on 512²). The primer's "Steady runs: getting to the answer fast"
  has the measurements and the trade-offs. The tools:
  - `monitor.steady_decider(fo, file, tol, ...)` as the LiveStop rule. It stops once the estimated
    iteration error of every monitored value (probes, forces, a wall heat flux) is below `tol`,
    instead of stopping on residual targets. `monitor.steady_replay` tests a tolerance on a recorded
    history.
    - It handles values converged to their printed digits: on case 6's recorded runs it would have
      stopped L4 at iteration 1611 instead of 3289.
    - It is cautious while the decay is still slowing down.
    - It does not stop on a solution that flips between states from one iteration to the next
      (reported as "oscillating"; case 7). For that it needs the monitored values of every iteration.
    - Its checks come at intervals of wall time, so a small grid runs far past the point where the
      rule first held. The decision records that point (`t_first_held`).
    - A value that has never changed is listed as "constant so far" and does not decide the stop. A
      probe far downstream of a moving front looks constant until the front arrives, so put probes
      where the answer changes.
  - `foam.map_fields(case, coarser_case)` starts a mesh from the coarser mesh's answer (grid
    sequencing), or a variant from the main run. It keeps the case's own fixed boundary values.
  - Less relaxation (SIMPLEC with U 0.97-0.99) is a larger pseudo time step. The best value
    depends on the mesh, so a short side-by-side trial is a cheap way to pick it.

  Whether to use these, and how, is your call. Record what you chose as a decision.
- Independent runs (different meshes) may run at the same time (how many: your machine file; 6 on
  the 8 cores of the development PC). Say so in
  the reproducibility notes, since it inflates per-run wall times.
- You can look into a running case whenever you judge it useful, and read whatever part of its
  history you want:
  - `python -m cfdlib.monitor peek <case dir>` prints the state, progress and time left, health so
    far, and the live stop's last verdict. It adds a short overview of the history: min / mean / max
    over a few time windows. Options: `--files`, `--from`, `--windows`, `--plot`.
  - Each peek also brings the raw history in `<case>/peek/` up to date, every step from the start:
    - `solver_log.csv`: residuals, Courant number, continuity error and solver time;
    - one CSV per postProcessing file.

    These files are yours to search and load in any way: grep, slices, your own analysis code, or
    `monitor.load_csv`, which reads the float64 twin `<name>.f64` fast. Only the updating is
    incremental; the files always hold the complete history.
  - `--plot` (or `watch`) gives the same history as a figure. Look at it when a shape matters, and
    use it to show the engineer how the run is going.
  - Between peeks nothing runs, so a long run costs no tokens while you are not looking. How often
    you look is up to you. A Monitor on the run's own log works well for the stop, the end and
    errors.
- For the human engineer: `python -m cfdlib.monitor watch <case dir> --every 60` keeps
  `<case>/peek.html` up to date until the run ends: a figure of the whole history plus the same text,
  reloading itself in a browser. Start it in the background and give the engineer the printed path.
- Long runs go in the background. `python -m cfdlib.foam ps` lists solver processes, and
  `python -m cfdlib.foam stop <case dir>` kills a run, including an orphan whose Python wrapper is
  gone. `foam.render_case` refuses to delete a case directory with a live solver in it.
  - Do not use `pkill -f <pattern>` when the pattern also appears in your own command line: it kills
    its own shell first. In case 6 that left orphan solvers writing into re-created case directories,
    and every run had to be repeated.
  - Check with `foam ps` in a separate call.

  Record per run: health (`monitor.assess`), the physical check, wall time.
- Run a sequence of long runs as ONE background script that writes a completion marker and a record
  per run. After any interruption (an API pause too), read those files before re-running anything.
- While all hardware threads run a solver, your own work slows it: leave 1-2 threads free or budget
  1.5 times the projection.

### 5. Analysis
- All comparisons in code. With an exact solution: errors and observed order per mesh.
  Without: observed order and Richardson extrapolation from 3 meshes, GCI on the finest mesh
  (`verification.py`), then compare the extrapolated value with the reference.
- A benchmark is a simulation too: judge agreement against (your GCI band + the reference's own
  uncertainty), not against zero. If the difference does not shrink with refinement, it is not your
  discretisation error; test robustness (another scheme, tighter iteration) before blaming either
  side, and recommend a second independent reference.
- When the reference gives several mesh levels, it is still converging: compute ITS observed order
  and Richardson value (`verification.level_sequence`) and compare your result with both its finest
  level and its extrapolated value. Decide the pass rule before looking (case 4 kept the finest-level
  rule and reported the extrapolation as context).
- Periodic outputs (shedding, oscillations): use `cfdlib.signals` (cycles between upward crossings,
  parabola-refined peaks, periodicity check, FFT cross-check) instead of writing your own.
- **Small or sensitive outputs** (lift of a nearly symmetric body, a small difference of large
  terms): one mesh family's GCI can be far too optimistic. Plan a second family (different grading,
  same cell counts) on at least two levels and report the spread. In case 5 the lift's GCI band was
  0.6 %, but two families on the finest level differed by 1.9 % and bracketed the reference.
- Before attributing a difference to an idealisation of the reference or the exact solution, show it
  does not shrink with refinement. In case 6 the centreline velocity at the start of heating was
  0.53 % low on L1. Most of that was mesh error (−0.53, −0.25, −0.12, −0.08 % on L1-L4); only about
  −0.06 % was real.
- A coarse-mesh match can be luck (error cancellation). Report differences honestly, where and why.
- An explanation is a claim: check it against the data (sign, size, scaling) before writing it. In
  case 3 the first explanation of an error had the wrong sign and a third of the size. If it needs
  a sensitivity, measure that with a variant that changes one thing only (case 7: a closed top
  boundary changed the stream's speed and its pressure gradient together, and still fitted the data
  to two digits).

### 6. Report
- Build `results.json` in `analyze.py` and call `report.render(results, svgs, out, case_dir=<finest case>)`.
  The contract is in `references/report-contract.md`; `render` refuses to write a report that breaks it.
- Math: every formula, symbol and scientific number in text as LaTeX inside `\( \)` (`m()`,
  `tex_num()`): write `\(0.009\,U\)`, never `0.009 U`, `6.3e-06` or `u*H/nu`. The checker rejects
  e-notation and ASCII math, and warns on number-times-symbol in plain text.
- Captions and findings are results too: generate them from the data (a caption saying "shrinks"
  must be computed). Refer to figures as `{fig:<key>}`; render() inserts the number.
- Figures (`cfdlib.plots`): `plots.figure()` for one panel, `plots.wide(n)` for 2-3 panels,
  `plots.sketch()` for the problem sketch (required). `to_svg()` refuses figures whose text overlaps
  text, lines, arrows or shapes, or whose legend covers data, or that are wider than the column.
  Fix the layout (labels away from lines, `annotate` with a leader line, legend outside); `allow=`
  only for overlaps that are truly intended, and log an insight if you need it.
- Verdict: PASS only if every check you defined passes; WARN if results are usable with caveats;
  FAIL otherwise. Never soften a failed check.

### 7. Visual review (mandatory)
The contract checks structure; only looking catches what a reader will see. Open the report in the
browser through a local server (file:// usually does not work in an agent's browser):
`http://localhost:8765/<name>/report.html` (how to start the server: `references/environment.md`; in
Claude Code it is the `reports` configuration of `.claude/launch.json`). Add `?v=<n>` after
re-rendering to avoid a cached copy.
- Screenshot every section, top to bottom. Check: figures readable (text not tiny, nothing
  overlapping, leader lines not crossing curves), tables not cut off, math rendered (no raw `\(`),
  table columns aligned, the summary answers each request.
- Run in the page (waits for MathJax, opens every ⓘ panel, then lists every element whose content
  is cut off; should be empty; run right after loading, before MathJax finishes, it reports un-typeset
  TeX as overflow):
  `await MathJax.startup.promise; document.querySelectorAll('details').forEach(d => d.open = true); [...document.querySelectorAll('section *')].filter(x => !x.tagName.startsWith('MJX') && x.scrollWidth > x.clientWidth + 2 && getComputedStyle(x).overflowX !== 'visible').map(x => x.closest('section').id + ': ' + x.className)`
  then screenshot the opened ⓘ panels too.
- A wide equation scrolls sideways inside its box, so a screenshot shows it cut at the edge. Scroll
  it, or keep hand checks short enough to fit (case 6 shortened several).
- **After any change of plan mid-course** (scheme, rank count, run length, mesh levels), search
  decisions.yaml, analyze.py and the rendered text for the superseded names and numbers. Case 4's
  final review found a decision, reproducibility commands and a limitation still describing the old
  plan; no automatic check catches this.
- Render a draft report from partial results while the finest run is still going; layout problems
  show up early.
- Fix what you find, re-render, look again. Log what you found as insights (stage "report").
- Without a browser: run analyze with `CFD_FIG_PNG_DIR=<dir>` to get PNGs of every figure and look at them.

## Side mission: performance
Cases 1-3 used one CPU core; case 4 measured MPI scaling and linear solvers (numbers in the primer).
Larger problems need speed, so every report should leave data for optimising it (CPU cores and MPI
now; GPU linear solvers next).
- `foam.run` writes `<case>/perf.<name>.json` (wall time, CPU time, peak memory) for every command.
  `perf.summarize(case_dir, solver, cells, cores)` adds steps, wall microseconds per cell per step,
  CPU efficiency, mean linear-solver iterations per field (pressure usually dominates: that is what a
  GPU linear solver would accelerate) and the time outside the solver loop.
- Measure before you optimise. Typical levers, in order of effort: run independent cases at the same
  time; MPI decomposition of the large runs (`decomposePar`, `mpirun -np N <solver> -parallel`,
  `reconstructPar`; on the development PC 8 physical cores / 16 threads, Open MPI 4.1.6,
  `--use-hwthread-cpus` above 8 ranks) with a scaling test (1, 2, 4, 8, 16 ranks on the finest mesh, recording pressure iterations
  per decomposition); linear-solver choice and tolerances for `p` (GAMG was 4 times faster than PCG
  in case 4); write interval and function objects (`forces` every iteration cost 6 % of a 2.4
  million cell iteration in case 5; use `executeInterval` when the stopping rule allows).
  `perf.summarize` gives `lin_iters_by_solve` (iterations of the 1st, 2nd, ... and last pressure
  solve of a step). A setting tuned on a coarse level (a tolerance, a relaxation factor) is a guess
  for the fine one: confirm it there with the same same-answer rule before the production run.
- **Time with profiling, not the log clock.** `ClockTime` in the solver log (and `timeInfo`) has 1 s
  resolution; `ExecutionTime` is rank 0's CPU time, which undercounts wall time when ranks wait (16
  ranks on 16 threads: half). Switch on `profiling { active true; }` in controlDict and read
  `perf.profiling(case, cores)`: high-resolution wall time per section (`fvMatrix::solve.p`,
  `fvMatrix::solve.U`, each function object).
- A GPU test must compare against the BEST CPU setup, measured per case: GAMG smoother (`DICGaussSeidel`
  needed 5 instead of 13 iterations in case 5) and rank count (8 ranks beat 16 on steady 3D, case 5;
  16 won on transient 2D, case 4). Never PCG on one core. Report time and iterations per pressure solve
  separately from the total.
- Report it in the optional `performance` appendix (see report-contract.md) and log what you learn as
  insights with stage `performance`: what limited speed, what helped by how much, what you would try next.

## Performance study (when problem.yaml has a `study` block)
Some cases ask a second question with its own report, e.g. "how much faster is the GPU?". Answer it
with `cfdlib.study.render_study()` (keys and rules: `references/report-contract.md`, "Study reports") into the file named in
`study.output`; the main report answers the physics as usual and links to it in its notes.
- **Amdahl bound first.** Before any GPU run, measure the pressure share of an iteration with
  profiling on the CPU baseline and state the bound 1 / (1 - share): a solver that only accelerates
  the pressure solve can never beat it (case 5: share 30-37 %, bound about 1.5; measured 1.13 at 4.75
  million cells). It sets expectations and the time budget for tuning.
- **Same answer first.** A faster run that gives a different answer is not a speed-up. Compare the
  requested outputs (forces etc.) of CPU and GPU runs on the same mesh and state the tolerance before
  measuring. `verdict PASS` requires it.
- **Fair baseline.** The CPU side uses its best measured setup on this machine (GAMG, all useful
  ranks). Same mesh, same schemes, same tolerances on both sides; only the linear solver and where it
  runs change. Say which pressure solve tolerance each side reaches (`relTol`, `tolerance`).
- **One clock on both sides.** Compare tools on the wall time of the whole run; `cfdlib.runrecord.read`
  returns any run record with each cost named by its clock.
- **Measure, do not project.** Several mesh sizes (factor about 2-8 in cells each), up to the largest
  that fits the memory, at least 2 repeats of the timing run on the largest; time per iteration after
  start-up, iterations and time per pressure solve, total wall time, peak GPU memory (`nvidia-smi
  --query-gpu=memory.used --format=csv` while it runs).
- **Where the time goes**: solve time of p and U, matrix assembly, host-device copies (OGL with
  `verbose 1` prints per-solve timings `[OGL LOG]`), everything else. A GPU linear solver only
  accelerates the solve; the rest still runs on the CPU and limits the gain (Amdahl).
- Log what you learn as insights with stage `gpu` (GPU-specific) or `performance`; the study report
  shows both.
- GPU linear solvers (OGL plugin, solver names, tuned options, pitfalls):
  `references/gpu-linear-solvers.md`. Start from the settings found in case 5, not from OGL's defaults.
- An optimisation that can change the answer is a decision (document it) and must be shown to give
  the same result as the baseline (to round-off or within the stated tolerance).

## Insight entries
Fields: `stage`, `problem` (when logged), then `diagnosis`, `fix`, `lesson`, `guideline` (the change
to THESE guidelines or tools that would have prevented it). Also log things that worked first time
only because of a specific guideline or example, and anything in these guidelines that was unclear,
wrong or missing. Unresolved entries are shown as unresolved in the report; that is allowed and honest.

## Machine and environment
Two files, read both before running anything:
- `references/environment.md`: what holds on every machine (where runs go, the tool commands,
  pitfalls, how to look at a report).
- `machines/<your machine>.md`: hardware, paths, limits and the exact launch commands of the computer
  you are on (`machines/dev-pc.md` is the development PC: Windows with WSL, where commands cross
  from PowerShell or Git Bash into Linux). On a machine with no file yet, write one first:
  `machines/README.md` says how.

Short version: run `~/venvs/cfd/bin/python -m cases.<name>.<module>` from the repository root, on
the Linux side; put shell work in `.sh` files.
