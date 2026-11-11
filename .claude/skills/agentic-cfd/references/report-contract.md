# Report contract

`results.json` must have the structure below. `cfdlib.contract.check()` enforces it, and
`cfdlib.report.render()` refuses to write a report that fails. Text fields are HTML and may contain
inline math `\( ... \)` (build it with `cfdlib.report.m()`). See `cases/plane_poiseuille/analyze.py`
and `cases/lid_driven_cavity_re100/analyze.py` for complete, passing examples.

Any text field may refer to a figure as `{fig:<key>}`; `render()` replaces it with the figure's number
(unknown keys fail the check). Typesetting is checked too: e-notation (`6.3e-06`) and ASCII math
(`m^2`, `u*H`) outside `\( \)` fail; a number followed by a one-letter symbol in plain text
(`0.009 U`) or a math character (√, τ, ≈, Greek letters) in plain text gives a warning. Control
characters anywhere (a TAB where `\times` was) fail. Insight text is checked when it is logged. Use `tex_fix(x, decimals)` for table columns that must line up.

| Key | Content |
|---|---|
| `title`, `date` | Problem title; ISO date |
| `verdict` | `PASS` / `WARN` / `FAIL` (rule-based in code; list the rules in findings when not PASS) |
| `comparison_type` | `verification` (exact solution), `benchmark` (trusted reference simulation), `validation` (experiment), `consistency` (no reference: physical checks only) |
| `summary.headline` | One sentence: verdict and the most important number |
| `summary.answer_columns` | 4 column titles, e.g. `["Requested", "Reference (Ghia 1982)", "Simulation (finest mesh)", "Difference"]` |
| `summary.answers[]` | One per requested output: `letter`, `label`, `theory` (reference/theory value), `sim`, `diff`. Letters must match `problem.wanted` |
| `summary.findings[]` | Short bullets: accuracy, order, convergence, key physics |
| `summary.limitations[]` | What the result does not cover |
| `problem.description`, `setup[]`, `given[]`, `wanted[]` | `description`: short overview (1-3 sentences, at most 600 characters). `setup`: list of short facts (geometry, coordinates, boundary conditions, definitions of output quantities), copied from problem.yaml. `given`: label, symbol (math), value (math), unit. `wanted`: letter, key, text. Optional `reference`: one sentence naming supplied reference data and its source. Shown as overview, "What we want to find", setup bullets, sketch, given data |
| `assumptions.rows[]` | name, check (justification, with computed numbers where possible), status: `ok` (checked with data) / `standard` / `imposed` / `violated` |
| `theory.intro` | What theory can and cannot give here |
| `theory.steps[]` | label, latex (display math, no delimiters), note. May be empty |
| `theory.derived[]` | label, symbol, formula, value_tex, unit_tex, why, params[] (tex, meaning, value, unit), check (formula with numbers substituted). May be empty |
| `theory.self_checks[]` | name, ok. May be empty |
| `decisions[]` | area, kind, choice, settings[] (path patterns), why, affects, alternatives. `kind`: `physics` (defines the problem), `accuracy` (sets numerical error: mesh, schemes, time step, iteration tolerance), `solver` (how the answer is reached, not the converged answer: algorithm, relaxation, linear solvers, initial guess of a steady run), `output` (what is measured or saved). Every non-inert setting in the finest case must match a pattern. Shown in the report as an at-a-glance table plus one card per decision |
| `preflight` | purpose, runs[] (name, setup, outcome), checks[] (name, value, status ok/warn/fail, meaning), conclusion |
| `simulation.model[]` | aspect, real, model: how the real problem maps to the simulation |
| `simulation.meshes` | intro, head[], rows[][] |
| `simulation.results` | head[], rows[][] |
| `simulation.health` | head[], rows[][], notes[]: per-run convergence and health evidence |
| `comparison` | kind (sentence), definitions[] (name, latex, meaning), head[], rows[][], findings[] |
| `insights[]` | Load from `insights.yaml` (`cfdlib.insights.load`). stage (intake/theory/setup/preflight/production/analysis/report/tooling), logged (time first met), status (`open`/`resolved`), problem; resolved entries also diagnosis, fix, lesson, guideline. At least one. Shown in logged order; open ones are marked unresolved |
| `figures[]` | key, section (problem/theory/preflight/simulation/convergence/comparison), caption. A `problem` sketch is required. SVGs passed to `render()` keyed by `key` |
| `reproducibility` | dict: `commands[]` (cmd, what) in the order to run them; `files[]` (path, what) the case files; `run_dirs[]` (path, what); `environment` = `report.environment_info()` (read from the machine, not typed); optional `notes[]` (e.g. runs executed concurrently) |
| `performance` | optional appendix B (recommended for every case with runs over a minute): intro, head[], rows[][] (per run: cells, cores, steps, wall time, wall microseconds per cell per step, CPU efficiency, mean pressure linear-solver iterations...), notes[] (what limited speed, what was tried, what to try next). Figures with section `performance` appear there |
| `data` | optional: raw numbers behind the tables |

## Inert settings (do not need a decision)
`FoamFile` headers, field `dimensions`, and controlDict `startFrom, stopAt, purgeWrite,
writeCompression, runTimeModifiable, writeFormat, timeFormat`; blockMesh `edges, mergePatchPairs`;
`decomposeParDict`. Everything else must be covered.
