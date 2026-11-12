---
name: agentic-cfd
description: Solve a fluid-flow problem end to end with OpenFOAM and produce a verified HTML report (problem, assumptions, theory, decisions, pre-flight test, simulation, comparison, insights). Use when given a CFD problem statement in cases/<name>/problem.yaml or asked to simulate a flow and report on it.
---

# Agentic CFD workflow

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
  foam.py        render_case, run (timeout, PID file), stop, running; read fields and postProcessing .dat files
  monitor.py     parse_log, assess (health checks), preflight (short test run + run-time projection),
                 LiveStop / judge / settled (end a run live once its physical check holds),
                 peek (look into a run at any moment: progress, health, verdict, history as numbers),
                 watch (a self-refreshing page with the history figure, for the human engineer)
  verification.py error norms, observed order, Richardson extrapolation, GCI, level_sequence
                 (convergence of any sequence of levels, including a reference's own)
  signals.py     periodic signals: cycles, refined peaks, periodicity check, FFT frequency
  plots.py       figure style and sizes (figure, wide, sketch); to_svg checks for overlapping text
  report.py      m(), tex_num(), tex_fix() (fixed decimals for table columns), tex_pct(), LETTERS, load_yaml,
                 environment_info(), render()
  perf.py        performance numbers per run (summarize, machine) for the performance appendix
  contract.py    the report contract checker (also --decisions mode)
  study.py       a second, focused report for a performance study (render_study), e.g. GPU vs CPU
  gpu.py         gpucfd as a solver: check_fit (does the problem fit it?), run_transient (sampled run with
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

**Worked examples** (read all their files before starting; copy the structure, not the numerical
settings):
- `cases/plane_poiseuille/`: linear, steady, exact solution (verification). Report:
  `results/plane_poiseuille/report.html`.
- `cases/lid_driven_cavity_re100/`: nonlinear, steady, no exact solution, published benchmark
  (benchmark comparison, Richardson extrapolation, residual-stopped SIMPLEC, separate pre-flight and
  production commands). Report: `results/lid_driven_cavity_re100/report.html`. Its insights.yaml
  holds 15 lessons from the agent that built it.
- `cases/stokes_first_problem/`: linear, TRANSIENT, exact solution (combined and separate mesh and
  time-step studies, per-step momentum balance, concurrent runs). Report:
  `results/stokes_first_problem/report.html`.
- `cases/dfg_cylinder_re100/`: nonlinear, TRANSIENT and periodic (vortex shedding), benchmark with
  several reference levels, curved-body O-grid mesh generated in Python, MPI runs and a performance
  appendix. Report: `results/dfg_cylinder_re100/report.html`. Its insights.yaml holds 16 lessons
  (the most important: PISO was first order in time).
- `cases/dfg_cylinder_3d_re20/`: nonlinear, STEADY, 3D (O-grid extruded with wall grading), benchmark
  with levels 1-4, a second mesh family for the sensitive lift, and a GPU vs CPU performance study
  (`gpu_study.py`, OGL tuning, profiling-based timing). Reports: `results/dfg_cylinder_3d_re20/report.html`
  and `gpu_study.html`. Its insights.yaml holds 14 lessons (lift needed a second mesh family; the GPU
  gain is capped by the pressure share).

## Stages

### 0. Intake
- Read `problem.yaml` and any `reference/` files. List the requested outputs (they get letters a, b, c...).
- Compute the governing dimensionless numbers (Re, Ma, ...) in code. Decide which physics apply.
- If data is missing or contradictory, choose a documented assumption and flag it in the report's
  limitations. Never silently invent inputs.
- If a reference gives no uncertainty, estimate one (its grid, its printed digits, a second source)
  and record how; you will need it in stage 5.

### 1. Theory: what can be known without the simulation
- Exact solution available: derive it with SymPy, self-check it (satisfies equations and boundary
  conditions), evaluate the requested quantities. See `cases/plane_poiseuille/analytical.py`.
- No exact solution (usual case): say so in `theory.intro`, and give what theory CAN say: limiting
  cases, scaling, conservation laws, symmetry, sign of quantities, expected flow structure.
  See `cases/lid_driven_cavity_re100/theory.py`.
- Every derived quantity gets: why it matters, the meaning and value of each symbol in this
  problem, and the formula with the numbers substituted (hand-check).

### 2. Setup and decisions
- **Choose the solver first, and record it as a decision (kind `solver`).** OpenFOAM is the default. Consider
  gpucfd (`cfdlib.gpu.check_fit`, limits in `gpucfd/DESIGN.md`) when the problem is incompressible, laminar,
  in a box (bodies inside it allowed) and a uniform grid can resolve it. What decides is the cost to reach
  the required accuracy, not speed per cell. Case 4 (DFG 2D-2, `gpu_run.py`, `gpu_compare.py`) measured both:
  gpucfd is about 70 times faster per cell and step, but its uniform grid with an immersed cylinder needed
  442368 cells for drag within 1.0 % of the benchmark, while OpenFOAM's O-grid had it within 0.25 % at
  13432 cells (97 s). So: OpenFOAM for bodies with thin boundary layers (Re about 100 and up on a cylinder)
  until gpucfd has stretched grids; gpucfd for flows a uniform grid resolves well (periodic boxes,
  channels, cavities, low Reynolds numbers) and for steady flows with `gpucfd.steady` (grid sequencing).
- When gpucfd is the solver: grids n0 times 2^k with multigrid-friendly sizes (powers of 2 times 3 or 9);
  time step from `NavierStokes.max_dt`; tune `pressure_tol` in the pre-flight with a same-answer rule (case
  4: \(10^{-8}\), 1.7 times faster than \(10^{-10}\)); run with `cfdlib.gpu.run_transient`, which samples,
  checks health and records the run like an OpenFOAM run; stop each level on the physical check
  (`stop_on_watch`), not after a fixed time: a coarse gpucfd level can be far from the fine answer (case 4:
  G1 drag 16 % low), and the next level then takes several flow-through times to settle.
- Find the closest OpenFOAM tutorial (`$FOAM_TUTORIALS`) and start from it; make minimal changes.
- Read `references/openfoam-primer.md` for what each file controls and the known pitfalls.
- Templates may contain Jinja loops: generating a whole `blockMeshDict` from Python is fine and is
  the way to mesh curved bodies (example: `mesh_spec()` in `cases/dfg_cylinder_re100/run.py`).
- **Edit files that contain LaTeX only with the Edit/Write tools**, never through Bash heredocs or
  inline Python: the Bash tool eats backslashes (`\times` becomes a TAB). Cases 3 and 4 both lost
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

### 3. Pre-flight test (mandatory, before any long run)
- `cfdlib.monitor.preflight()` on the coarsest AND finest meshes with a short run (a few % of the
  full length). It works in a separate `<case>_preflight` copy; render pre-flight and production
  cases into separate directories too, so re-running one never deletes the other.
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
- If it fails or warns, fix the setup and log an insight. No production runs on a failing pre-flight.
  If the projected time is excessive (> ~15 min per run here), coarsen the plan and say so.

### 4. Production runs
- Run meshes coarse to fine. For steady runs, extrapolate the iteration count of the finest mesh from
  the coarser ones (it often grows 3-4x per refinement) before launching it.
- Stopping must be physical: make the physical check of stage 2 the stopping rule, and the planned
  end time only a maximum. Use `monitor.LiveStop(case, decide)` as `foam.run(..., watch=...)`:
  `decide(case)` reads the run's output so far (`monitor.read_dat_live`) and returns
  `monitor.judge({quantity: per-cycle or per-chunk values}, {quantity: tol})`; once every quantity is
  settled the solver writes the current step and ends normally (OpenFOAM's `abort` function object,
  serial and parallel). A quantity whose changes stop shrinking above its tolerance is at a floor:
  the run stops with a warning, and that spread becomes its iteration uncertainty (case 4's M1 drag
  peaks jitter by 7e-4 against a 5e-4 rule; waiting longer never helps). Example:
  `cases/dfg_cylinder_re100/run.py` (`periodic_rule`, `live_decider`, `do_live`). Replay the rule
  on recorded histories before trusting it (`do_live_check`). For steady SIMPLE runs, check per mesh
  that the residual targets actually satisfy the physical check, on the coarse meshes first.
- Independent runs (different meshes) may run at the same time, up to 6 at once (8 cores). Say so in
  the reproducibility notes, since it inflates per-run wall times.
- Look into a running case whenever you judge it useful:
  `python -m cfdlib.monitor peek <case dir> [--files forcesCyl,probes] [--from T] [--windows N] [--csv DIR]`.
  It prints the state, progress and time left, health so far, and the live stop's last verdict. It
  then gives the whole history as numbers: residuals, Courant number and each monitored series as
  min / mean / max over N equal time windows (default 8). Rising or falling means, a growing or
  steady oscillation and the start-up transient all show in the rows. Narrow it with `--files` and
  `--from`; use `--windows` for finer rows. Use `--csv` to read the raw series at full resolution
  with your own code. Between peeks nothing runs, so a run of many hours costs no tokens. A sensible
  rhythm: early (is the start healthy?), then at intervals matched to the projected run time, then at
  the end; peek more often when something looks wrong. A Monitor on the run's own log should only
  report the stop, the end and errors.
- For the human engineer: `python -m cfdlib.monitor watch <case dir> --every 60` keeps
  `<case>/peek.html` up to date until the run ends: a figure of the whole history plus the same text,
  reloading itself in a browser. Start it in the background and give the engineer the printed
  path. `peek --plot` writes the figure once.
- Long runs go in the background; `python -m cfdlib.foam ps` lists solver processes, and
  `python -m cfdlib.foam stop <case dir>` kills one (stopping the Python wrapper alone leaves an
  orphan solver). Record per run: health (`monitor.assess`), the physical check, wall time.
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
- A coarse-mesh match can be luck (error cancellation). Report differences honestly, where and why.
- An explanation is a claim: check it against the data (sign, size, scaling) before writing it. In
  case 3 the first explanation of an error had the wrong sign and a third of the size.

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
built-in browser through the local server (file:// does not work):
`http://localhost:8765/<name>/report.html` (if it does not load, start the `reports` server from
`.claude/launch.json` with the preview tool). Add `?v=<n>` after re-rendering to avoid a cached copy.
- Screenshot every section, top to bottom. Check: figures readable (text not tiny, nothing
  overlapping, leader lines not crossing curves), tables not cut off, math rendered (no raw `\(`),
  table columns aligned, the summary answers each request.
- Run in the page (waits for MathJax, opens every ⓘ panel, then lists every element whose content
  is cut off; should be empty; run right after loading, before MathJax finishes, it reports un-typeset
  TeX as overflow):
  `await MathJax.startup.promise; document.querySelectorAll('details').forEach(d => d.open = true); [...document.querySelectorAll('section *')].filter(x => !x.tagName.startsWith('MJX') && x.scrollWidth > x.clientWidth + 2 && getComputedStyle(x).overflowX !== 'visible').map(x => x.closest('section').id + ': ' + x.className)`
  then screenshot the opened ⓘ panels too.
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
  `reconstructPar`; 8 physical cores / 16 threads, Open MPI 4.1.6, `--use-hwthread-cpus` above 8
  ranks) with a scaling test (1, 2, 4, 8, 16 ranks on the finest mesh, recording pressure iterations
  per decomposition); linear-solver choice and tolerances for `p` (GAMG was 4 times faster than PCG
  in case 4); write interval and function objects (`forces` every iteration cost 6 % of a 2.4
  million cell iteration in case 5; use `executeInterval` when the stopping rule allows).
  `perf.summarize` gives `lin_iters_by_solve` (iterations of the 1st, 2nd, ... and last pressure
  solve of a step).
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
with `cfdlib.study.render_study()` (contract and keys in its docstring) into the file named in
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
- **Measure, do not project.** Several mesh sizes (factor about 2-8 in cells each), up to the largest
  that fits the memory, at least 2 repeats of the timing run on the largest; time per iteration after
  start-up, iterations and time per pressure solve, total wall time, peak GPU memory (`nvidia-smi
  --query-gpu=memory.used --format=csv` while it runs).
- **Where the time goes**: solve time of p and U, matrix assembly, host-device copies (OGL with
  `verbose 1` prints per-solve timings `[OGL LOG]`), everything else. A GPU linear solver only
  accelerates the solve; the rest still runs on the CPU and limits the gain (Amdahl).
- Log what you learn as insights with stage `gpu` (GPU-specific) or `performance`; the study report
  shows both.
- GPU details for this machine (OGL plugin, solver names, tuned options, pitfalls):
  `references/environment.md`. Start from the settings found in case 5, not from OGL's defaults.
- An optimisation that can change the answer is a decision (document it) and must be shown to give
  the same result as the baseline (to round-off or within the stated tolerance).

## Insight entries
Fields: `stage`, `problem` (when logged), then `diagnosis`, `fix`, `lesson`, `guideline` (the change
to THESE guidelines or tools that would have prevented it). Also log things that worked first time
only because of a specific guideline or example, and anything in these guidelines that was unclear,
wrong or missing. Unresolved entries are shown as unresolved in the report; that is allowed and honest.

## Machine and environment
See `references/environment.md` (WSL commands from PowerShell AND from Git Bash, paths, quoting,
the report server). Short version: run Python inside WSL with
`/home/tomyud/venvs/cfd/bin/python -m cases.<name>.<module>` from `/mnt/d/Projects/Agentic CFD`;
put shell work in `.sh` files; never pass complex inline bash through PowerShell.
