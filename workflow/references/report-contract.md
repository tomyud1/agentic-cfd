# Report contract

`results.json` must have the structure below. `cfdlib.contract.check()` enforces it, and
`cfdlib.report.render()` refuses to write a report that fails. Text fields are HTML and may contain
inline math `\( ... \)` (build it with `cfdlib.report.m()`). See `cases/plane_poiseuille/analyze.py`
and `cases/lid_driven_cavity_re100/analyze.py` for complete, passing examples.

Any text field may refer to a figure as `{fig:<key>}`; `render()` replaces it with the figure's number
(unknown keys fail the check). It inserts the number only, so write the word: `Figure {fig:<key>}`
(a bare placeholder gets a warning). A placeholder quoted inside `<code>` is not a reference. Typesetting is checked too: e-notation (`6.3e-06`) and ASCII math
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
| `summary.answers[]` | One per requested output: `letter`, `label`, `theory` (reference/theory value), `sim`, `diff`. Letters must match `problem.wanted`. `diff` is one value (or a few short lines separated by `<br>`): its column does not wrap, so a sentence there pushes the table off the page (put the words in `sim`) |
| `summary.findings[]` | Short bullets: accuracy, order, convergence, key physics |
| `summary.limitations[]` | What the result does not cover |
| `problem.description`, `setup[]`, `given[]`, `wanted[]` | `description`: short overview (1-3 sentences, at most 600 characters). `setup`: list of short facts (geometry, coordinates, boundary conditions, definitions of output quantities), copied from problem.yaml. `given`: label, symbol (math), value (math), unit. `wanted`: letter, key, text. Optional `reference`: one sentence naming supplied reference data and its source. Shown as overview, "What we want to find", setup bullets, sketch, given data |
| `assumptions.rows[]` | name, check (justification, with computed numbers where possible), status: `ok` (checked with data) / `standard` / `imposed` / `violated` |
| `theory.intro` | What theory can and cannot give here |
| `theory.steps[]` | label, latex (display math, no delimiters), note. May be empty |
| `theory.derived[]` | label, symbol, formula, value_tex, unit_tex, why, params[] (tex, meaning, value: a number, which the template formats, unit), check (formula with numbers substituted). May be empty |
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

## Study reports

A second report on a solved case that compares tools or settings: GPU against CPU, one solver against
another, one scheme against another. `cfdlib.study.render_study(s, svgs, out, insights)` writes
`<name>.html` and `<name>.json`; `cfdlib.study.check_study()` is the rule and refuses a study that
breaks it; `cfdlib.study.warnings()` lists what will read badly (a table of more than about ten
columns scrolls sideways unseen). Text and typesetting rules are those of the main report, and
`{fig:<key>}` works the same way. Complete examples: `cases/dfg_cylinder_3d_re20/gpu_study.py`
(one solver, CPU against GPU) and `cases/dfg_cylinder_re100/gpu_study.py` (two solvers, with its
own section and section titles). `tests/test_study.py` renders a study with every key.

Firm rules: `verdict` PASS needs `same_answer.ok` (a faster run with a different answer is not a
speed-up); any other verdict needs `summary.reasons`; the results section has at least one figure.
Everything else below is there to use when it makes the page clearer.

| Key | Content |
|---|---|
| `title`, `date`, `question` | The question the study answers, in one sentence |
| `verdict` | `PASS` / `WARN` / `FAIL`, from rules in the script. `study.verdict(fail, warn)` turns two lists of plain sentences into the verdict and its reasons, so every study decides alike: anything in `fail` (a run failed a health check, the answers differ, the setup was unfair) gives FAIL, anything in `warn` gives WARN |
| `kicker`, `verdict_label` | optional: the line above the title (default "performance study") and the words next to the verdict mark (defaults: "Fair comparison, same answer", "Usable, with caveats", "Not a valid comparison") |
| `summary` | `headline`, `answers[]` (label, value, note), `findings[]`, `limitations[]`; optional `recommendation`; `reasons[]` (why the verdict is what it is, shown under the headline; required unless PASS); optional `plain`: the headline as plain text without HTML or LaTeX, which `cases/catalogue.yaml` shows for the study (a study entry has no typed headline) |
| `setup` | `intro`, `hardware[[name, value]]`, `software[[name, value]]`, `fairness[]` (the rules that make the comparison fair, and what is not like for like); optional `terms[]` (term, meaning): the words the page uses, for a reader new to the field |
| `method` | `intro`, `choices[]` (choice, why, alternatives) |
| `results` | `intro`, `tables[]`, `notes[]` |
| `same_answer` | `intro`, `head[]`, `rows[][]`, `ok` (bool), `verdict` (one sentence); it is a table, so it also takes `question`, `result`, `meaning`, `note` |
| `breakdown` | `intro`, `tables[]`, `notes[]`: where the time goes, or what limits the tool |
| `sections[]` | optional sections of your own: `key` (a new lower-case name), `title`, `intro`, `tables[]`, `notes[]`, and `after` (the core section it follows; default: after `breakdown`) |
| a `title` in `setup`, `method`, `results`, `same_answer`, `breakdown` | optional: replaces the section's default title ("What is compared", "Method", "Results", "Same answer?", "Where the time goes") |
| a table | `head[]`, `rows[][]` (one cell per column); optional `title`; optional `question` (shown above the table), `result` and `meaning` (shown below it, each on its own labelled line): what the table was made to answer, what came out, and what follows from it; optional `note` (small print: how to read a column, a footnote); optional `figure` (the key of a figure of the same section, drawn right above this table); optional `nowrap` (column numbers kept on one line; by default the first column when its cells are short) |
| `figures[]` | `key`, `section` (`setup`, `results`, `same_answer`, `breakdown` or the key of your own section), `caption`. A figure that no table names is drawn at the top of its section. Numbers follow the reading order |
| `reproducibility` | as in the main report: `commands[]`, `files[]`, `run_dirs[]`, `environment`, optional `notes[]` |
| insights (argument of `render_study`) | the case's insight entries; those with stage `performance` or `gpu` are shown, each folded to its problem line. Pass a shorter list to show fewer |

Numbers for the tables: `cfdlib.report.m_fix`, `m_pct` (per cent, signed or not), `m_times`,
`m_pow10`, `m_sig` return a finished cell (math, or a dash for a missing value). Run records of
OpenFOAM and gpucfd in one shape, with wall time and wall-based cost under the same names:
`cfdlib.runrecord.read`. A grid-convergence index that is a placeholder, not an estimate, is
flagged by `cfdlib.verification.three_grid_study` (`gci_reliable`): mark it in the table.
