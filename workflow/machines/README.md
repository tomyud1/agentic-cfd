# Machine files

The workflow is the same on every computer. What changes is in one file per machine here: hardware,
paths, how commands are launched, and limits. An agent reads the file for the machine it is on
before running anything.

| File | Machine |
|---|---|
| [`dev-pc.md`](dev-pc.md) | The student's desktop: Windows 11 + WSL2, Ryzen 7 7800X3D, RTX 4070. Cases 1-6 were solved here |

## On a new machine

1. Install with [`setup/README.md`](../../setup/README.md).
2. Run `bash setup/probe_env.sh` and write `<machine>.md` here from what it prints. State:
   - **Hardware:** CPU cores and threads, RAM, GPU model and memory. From these: how many
     independent runs fit at once, the largest mesh that fits, and (for gpucfd) the `ARCH` to build
     with.
   - **Software and paths:** the OpenFOAM version, `$FOAM_RUN`, the Python to use
     (`~/venvs/cfd/bin/python` if you used `setup/make_venv.sh`), and whether the agent has sudo.
   - **How to run things:** the exact command line that works from the agent's shell. A native Linux
     machine needs none of the WSL lines in `dev-pc.md`.
   - **Run-time budget:** how long one run may take before the plan should be coarsened.
   - **Pitfalls** met on this machine, as they are found.
3. Check the machine with the tests that need no solved case: `bash setup/rerender_all.sh` (all
   reports rebuild from saved data) and, with a GPU, `bash gpucfd/run_ladder.sh`.
4. Add a row to the table above.

Timings in the primer and in the solved cases were measured on `dev-pc`. On another machine they
are a starting point: measure before relying on them (best rank count, linear-solver ranking, GPU
break-even).
