# Steady-run speed study (2026-11-12)

Dev experiment behind the primer section "Steady runs: getting to the answer fast": how much sooner
can OpenFOAM's steady SIMPLE runs reach the same answer? Measured on case 2 (lid-driven cavity,
Re 100), whose recorded runs (residual targets 1e-11, SIMPLEC, U 0.9) are the reference.

Run inside WSL. Paths point to `$FOAM_RUN/agentic/lid_driven_cavity_re100` (case 2's runs) and
`$FOAM_RUN/agentic/dev_steady` (these runs); the scratch paths in the scripts are those of the
session that made them and need adjusting.

- `replay2.py`: the error-based stop replayed on case 2's recorded probe histories.
- `exp.sh NAME simple|pimple RELAX|COURANT [rest|N64...] [N] [steps]`: one variant; `batch1-3.sh` ran them.
- `an1.py N names`: rows to the 1e-6 stop and the error there against the converged answer.
- `rate.py`: does a short run rank the relaxation settings? (iterations per decade over rows 200-400).
- `live.py`: the tool itself (`monitor.steady_decider` in LiveStop) on a real run of 256².
- `et2.sh`: solver time to each stop row, from the logs' ExecutionTime.

Results (one core; several dev runs shared the CPU, so their times are upper bounds):

| Run | Rows to 1e-6 U | Solver time | Error at stop |
|---|---|---|---|
| 128², U 0.9 (case 2) | 1931 | 27.3 s | 9.3e-7 |
| 128², U 0.95 / 0.97 / 0.98 / 0.99 / 0.995 | 931 / 571 / 411 / 751 / 1091 | 16.6 / 12.4 / 9.3 / 10.3 / ~12 s | < 8.6e-7 |
| 128² from 64², U 0.97 | 471 | 7.1 s | 4.4e-7 |
| 256², U 0.9 (case 2) | 7491 (run as made: 10864) | 437 s (627 s) | 9.7e-7 |
| 256², U 0.97 / 0.99 / 0.995 | 2121 / 801 / 1411 | 198 / 111 / 102 s | < 9.1e-7 |
| 256² from 128², U 0.97 / 0.99 | 981 / 1241 | 77 / 70 s | 9.4e-7 / 1.1e-7 |
| 256², U 0.99, live stop through cfdlib | 815 (replay: 801) | 107 s | 7.0e-7 |
| 512² from rest, U 0.99 | not reached in 2500 | | |
| 512² from 256², U 0.99 | 1441 | 581 s (2 runs sharing) | 8.2e-7 vs its own row 2500 |
| pimpleFoam, Euler, Courant 5 / 20 / 50, 1 outer corrector | diverged within 10 steps | | |
