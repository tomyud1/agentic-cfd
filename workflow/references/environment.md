# Running things (every machine)

Hardware, paths and launch commands differ per computer: read the file for yours in
[`../machines/`](../machines/README.md) first (`dev-pc.md` is the development PC). If there is none,
follow `machines/README.md` to write one. This page holds what is the same everywhere.

## Layout of a working machine

- OpenFOAM: the ESI line (v2606 on the development PC). Source its environment before any `set -e`.
- Case runs go under `$FOAM_RUN/agentic/<case>/`, never inside the repository: a Linux filesystem
  is much faster than a mounted Windows drive, and OpenFOAM cannot use a path with spaces.
- Python for the pipeline (numpy, scipy, sympy, matplotlib, pyyaml, jinja2): the environment made
  by `setup/make_venv.sh`, `~/venvs/cfd/bin/python`. Run modules from the repository root:
  `~/venvs/cfd/bin/python -m cases.<name>.<module>`.

## Commands (from the repository root)

```
python -m cfdlib.contract --decisions cases/<name>/decisions.yaml <case dir>   # decisions cover every setting?
python -m cfdlib.contract results/<name>/results.json <finest case dir>        # full report contract
python -m cfdlib.insights <name> list                                          # the live insight log
python -m cfdlib.foam ps                                                       # running solvers + their case dirs
python -m cfdlib.foam stop <case dir>                                          # kill a run (and its solver)
python -m cfdlib.foam check <case dir>                                         # settings the solver ignores
python -m cfdlib.monitor peek <case dir>                                       # look into a running case
```

## Pitfalls

- Never name a script after a Python standard-library module (`inspect.py`, `test.py`, `types.py`,
  `io.py`...): Python puts the script's folder first on the path and numpy then imports your file.
- **Edit files that contain LaTeX (decisions.yaml, analyze.py, theory.py) with your file-editing
  tool, not through a shell.** Heredocs and inline scripts collapse `\\` to `\`, so `\times` turns
  into a TAB and `\beta` into a backspace (cases 3 and 4 both hit this with Claude Code's Bash tool).
  The contract rejects control characters, but editing the file directly avoids the problem.
- Put shell work in `.sh` files instead of long inline commands, above all when the command passes
  through a second shell (WSL from Windows, ssh).
- Long runs: run them in the background and keep working; use `cfdlib.monitor.assess` on the log.
  For a sequence of long runs, use one background script that writes a completion marker and record
  per run. After any interruption (including an API pause), check those files before re-running
  anything: in case 4 a queued chain finished on its own while the agent thought it had died.
  Stopping the background Python process does NOT stop the solver (it runs in its own process group,
  so a timeout can kill it cleanly): use `python -m cfdlib.foam stop <case dir>`, which also finds a
  solver whose wrapper already died, and check with `python -m cfdlib.foam ps` in a separate call
  before restarting a case. Do not use `pkill -f <pattern>` when the pattern is in your own command
  line: it kills its own shell (case 6 lost every run to the orphans this left).
- Independent runs may run concurrently (how many: your machine file). Parallel runs of one case
  (`decomposePar` + `mpirun -np N`) are possible for large meshes; document either as a decision.

## Looking at the report

`results/<name>/report.html` is one self-contained file. An agent's browser tool usually cannot open
`file://` pages, so serve the folder and open `http://localhost:8765/<name>/report.html`:

```
python -m http.server 8765 --bind 127.0.0.1 --directory results
```

Add `?v=<n>` to the URL after re-rendering so the browser does not show a cached copy. Without a
browser, `CFD_FIG_PNG_DIR=<dir>` makes `plots.to_svg` also write each figure as a PNG.
