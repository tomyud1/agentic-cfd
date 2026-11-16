# gpucfd, chemistry: the reactions of many cells on the GPU

**Status (2026-11-15): stage 1 is finished and every rung passes.** A kernel that advances the
chemistry of many independent cells. It is not yet coupled to the flow solver. The developing agent
reviewed it the same day: the library was rebuilt and the ladder rerun (all pass), and one check was
added at the state behind a detonation's shock (its own section below). That check changes the
speed forecast for a detonation.

| Rung | What | State |
|---|---|---|
| 1 | Rates and thermodynamics against `cfdlib.chem` | **Pass** (`tests/test_chem_rates.py`) |
| 2 | One parcel: case 20 at constant pressure, the same at constant volume, order of each integrator | **Pass** (`tests/test_chem_parcel.py`) |
| tuning | The integrator and tolerances, 48 runs | **Done** (`tests/test_chem_batch.py tune`): explicit Bogacki-Shampine, relative tolerance 1e-6, absolute 1e-14 mol/kg |
| 3 | Many cells at once; speed | **Pass** (`tests/test_chem_batch.py`, `tests/test_chem_speed.py`) |
| 4 | Precision: single, mixed, double; the default | **Pass** (`tests/test_chem_precision.py`): mixed is the default |
| record | Case 20 next to OpenFOAM, for the one-page report | **Written** (`dev/compare/h2_air_ignition_box/gpucfd.json`) |

**Added on 2026-11-16 by stage 2** (the coupling to the flow solver, [`REACTING.md`](REACTING.md)):
the kernel's formulas moved to `src/gcfdk_core.h`, which the joined solver shares, and this page's
ladder was run again on it (all pass). Two additions, both off by default here: the method `ros34a`
(the implicit method with its matrix written out exactly; atoms kept to rounding in mixed precision,
but not faster), and `control(quiet=1)` (a cell that hardly changes takes one plain step). The joined
solver's own default is `bs23` at tolerance 1e-4 with `quiet=1`.

To continue from here: the next stage is the coupling to the flow solver (last sections of this
page). The pass rules, expectations and tuning budget were written before the runs in
[`dev/compare/h2_air_ignition_box/plan.yaml`](../dev/compare/h2_air_ignition_box/plan.yaml), with
notes added after each rung. Every number on this page is in `tests/results/test_chem_*.json` or in
`gpucfd.json`.

## In short

- The kernel gives the same answer as the project's own chemistry tool (`cfdlib.chem`) and as
  OpenFOAM. On case 20 the ignition time is 217.7223 microseconds and the temperature at 1 ms is
  2220.4498 K: 0.022 % and 0.0023 % from the reference, where 1 % and 0.1 % are required. Atoms are
  kept to 1.3e-14, where 1e-5 is required.
- It is fast when there are many cells: about 200 million cells times steps per second at the step a
  flow solver takes (1e-8 s), on the RTX 4070. OpenFOAM's measured cost in case 20 is about 10 000
  steps per second for its one cell on one core. That comparison has limits, stated below.
- It is slow for one cell. A single parcel takes 14.2 s for 80 000 steps; OpenFOAM takes 11.0 s. A
  GPU pays off only with thousands of cells.
- What I expected and did not find: an implicit method does **not** win clearly at large steps. The
  explicit method is the default at every step size.

## Words used

- **Cell, parcel**: a small amount of gas with one temperature, one pressure and one mixture.
- **Scheme**: the list of reactions with their rate constants. Here OpenFOAM's hydrogen scheme: 10
  species and 27 reaction lines.
- **Ignition time**: the time at which the temperature first passes a set level (1200 K in case 20).
- **Step**: the time the caller asks for (the flow solver's time step). **Sub-step**: a smaller step
  the kernel takes inside it, chosen for each cell by itself.
- **Explicit / implicit**: an explicit method uses only the rates at the current state; it is cheap
  but must take sub-steps as short as the fastest reaction. An implicit method solves a small system
  of equations in each sub-step; it costs more but can take long sub-steps.
- **Tolerance**: the error the kernel allows itself in one sub-step. Relative tolerance is a share of
  each species' amount; absolute tolerance (mol/kg) is the floor for species that are nearly absent.
- **Single, double, mixed**: numbers with 7 digits, with 16 digits, or the state kept with 16 digits
  and the rates computed with 7.
- **Constant pressure / constant volume**: what is held fixed while the gas burns. A free parcel
  burns at constant pressure (case 20). A cell of a flow solver burns at constant volume during the
  chemistry part of its time step.

## What it is

- Source: [`src/gcfdk.cu`](src/gcfdk.cu) (one file, builds to `build/libgcfdk.so` with
  `bash gpucfd/build.sh chemistry`). Python wrapper: [`chemistry.py`](chemistry.py).
- The reaction data comes from OpenFOAM's CHEMKIN files through `cfdlib.chem`'s reader. No rate
  constant is typed by hand.
- Tests: `bash gpucfd/run_ladder.sh chemistry` (about a quarter of an hour, needs OpenFOAM installed
  for the scheme files).
- `cfdlib/chem.py` gained `VolumeParcel`, the same parcel at constant volume, as the reference for
  the constant-volume mode.

```python
from cfdlib import chem
from gpucfd.chemistry import Chemistry

mech = chem.build()                              # OpenFOAM's hydrogen scheme
c = Chemistry(mech, n, constant="volume")        # or "pressure"; default setting unless given
c.set(T=T, rho=rho, Y=Y)                         # arrays of n cells (or p=... at constant pressure)
c.advance(1e-8)                                  # one step for all cells
s = c.get()                                      # T, p, rho, Y, ...
bad = c.stats()["status"]                        # per cell: 0 is healthy
```

## The method in plain words

One GPU thread works on one cell. Cells do not talk to each other, so each takes as many sub-steps
as it needs.

1. **Temperature is never integrated.** Each cell stores its energy (enthalpy at constant pressure,
   internal energy at constant volume). Reactions move energy between "chemical" and "heat" but do
   not change the total, so the stored number stays fixed. The temperature is found from it whenever
   the rates are needed. Energy is therefore kept exactly, not approximately.
2. **Species change only by whole reactions.** The kernel adds up how far each reaction has gone and
   applies that to the species. One reaction never creates or destroys an atom, so atoms are kept to
   round-off.
3. **Rates** follow the CHEMKIN rules of the scheme: Arrhenius rates, third bodies (a collision
   partner that carries energy away), fall-off with pressure (Lindemann and Troe), and backward rates
   from equilibrium constants.
4. **Three integrators** are built in, and the default was chosen by measurement: two explicit ones
   (Bogacki-Shampine of order 3, called `bs23`; Dormand-Prince of order 5, `dp45`) and one implicit
   (Rosenbrock of order 4, `ros34`, with the same constants OpenFOAM's `Rosenbrock34` uses).
5. **Error control**: each sub-step gives two answers of different order; their difference estimates
   the error. Too large: the sub-step is repeated shorter. The cell remembers its sub-step size for
   the next step.
6. Each cell records when its temperature first crossed a set level, so ignition times need no
   history.

## The ladder and what it found

| Rung | Check | Found |
|---|---|---|
| 1 | Forward rates against `cfdlib.chem`, 2000 random states, double | differ by 7.8e-15 (constant pressure), 7.2e-15 (constant volume) |
| 1 | Backward rates | 7.5e-14, 6.1e-14 |
| 1 | Species rates, as a share of the total turnover | 5.7e-14, 5.6e-14 |
| 1 | Enthalpy, heat capacity, stored energy | 1.1e-14, 1.3e-15, 6.0e-16 |
| 1 | Temperature found back from the energy | within 2.7e-12 K, 6.8e-12 K |
| 2 | Case 20 against `cfdlib.chem` (100 000 steps of 1e-8 s) | ignition time +3.4e-7, final temperature −9.1e-7 K |
| 2 | **Required**: ignition time within 1 % of the reference | 217.722 against 217.77 microseconds: −0.022 % |
| 2 | **Required**: temperature at 1 ms within 0.1 % | 2220.450 against 2220.5 K: −0.0023 % |
| 2 | **Required**: atoms constant within 0.001 % | 1.3e-14 |
| 2 | Energy, as a share of the heat released (1.7789e6 J/kg) | 5.9e-16; 8.0e-10 while the gas sits at exactly 1000 K (see "caught") |
| 2 | Answer does not depend on the step (1e-6 s against 1e-8 s) | ignition time +3.8e-7, temperature at common times within 2.1e-10 K |
| 2 | With OpenFOAM's constants, against OpenFOAM taken to a step of zero | ignition time +1.0e-7, final temperature −6.4e-7 K |
| 2 | Constant volume against `VolumeParcel` | ignition 214.1823 microseconds (+3.6e-7), final 2477.0407 K (+1.6e-11 K), pressure 460 486.47 Pa = 2.2723 times the start (+1.1e-15); atoms 1.6e-14 |
| 2 | Order of each integrator with fixed sub-steps | `bs23` 3.00 (should be 3), `dp45` 5.18 (5), `ros34` 3.98 (4) |
| 3 | 4096 cells, start temperatures 1000 to 1400 K: each cell alone, in a batch of 16 in another order, and in the reversed batch | identical bit for bit, in double, mixed and single |
| 3 | Five of those cells against `cfdlib.chem` | ignition between 9.48 and 223.15 microseconds; worst difference 3.5e-7, and 9.0e-5 K |
| 3 | Default setting, every cell, steps of 1e-8 s | worst cell 0.0005 % in ignition time, 0.00005 % in final temperature (mixed) |
| 3 | No cell broke in any of 117 timings; memory as stated | 125 bytes per cell (73 in single) |
| 4 | Atoms in double and mixed within 1e-12 | 1.2e-14 and 2.6e-14; single 1.1e-4 |

"OpenFOAM's constants": OpenFOAM uses slightly different values of the gas constant and standard
pressure than CHEMKIN. The scheme is the same; the ignition time moves from 217.7223 to 217.7099
microseconds. The kernel takes either set from `cfdlib.chem`.

The kernel and OpenFOAM (taken to a step of zero) agree to 1e-7. Both differ from `cfdlib.chem` by
3e-7. So `cfdlib.chem`'s own solution, at its tolerance of 1e-10, is the least exact of the three.
This is why my expectation "within 1e-7 of `cfdlib.chem`" was missed (found 3.4e-7) while the pass
rule (1e-6) was met.

## Choosing the integrator

Budget fixed before the runs: 3 methods × 4 relative tolerances (1e-3 to 1e-8) × 2 absolute
tolerances × 2 step sizes = 48 runs on 4096 cells in double, each to 0.3 ms. The rule: among the
settings whose worst cell stays within 0.1 % in ignition time and 0.01 % in final temperature, take
the one with the least GPU time at steps of 1e-8 s.

- **At steps of 1e-8 s** (what a flow solver with shock waves takes) every method needs exactly one
  sub-step per step. The fastest reactions of this scheme act in about 1e-8 s, so the step is already
  short enough. The cheapest method per sub-step then wins: `bs23` 10.8 s (4 rate evaluations), `dp45`
  18.4 s (7), `ros34` 27.7 s (11, because it also builds a matrix). Errors in ignition time: 1.3e-7,
  9.6e-12, 4.7e-9. The rule chose **`bs23`, relative tolerance 1e-6, absolute 1e-14 mol/kg**.
- **At steps of 1e-6 s** the explicit methods take 11 to 25 sub-steps, the implicit one 4 to 26
  depending on tolerance. With a loose tolerance (1e-3) `ros34` was fastest on 4096 cells (2.3 s
  against 3.2 s). On a million cells in double it is still 18 % faster (1.35 against 1.14 million
  cells × steps per second). In mixed precision it is slower by more than half (4.9 against 11.0
  million): its matrix is built from differences of rates, and with 7-digit rates those differences
  are noisy.

So the default is `bs23` at every step size. I expected hundreds of explicit sub-steps at 1e-6 s and
a clear win for the implicit method. Found: 7 sub-steps in burnt gas, 24 to 58 in the temperature
rise, 18.5 on average. This scheme at 2 atm is only mildly stiff.

Decided while the student was away: I kept the tolerance the rule gave (1e-6). At steps of 1e-6 s in
mixed precision a tolerance of 1e-4 or 1e-5 costs 0.7 s and 1.0 s against 1.2 s, with ignition time
errors of 0.012 % and 0.0014 %. That is a possible later saving, not taken.

## Precision

Default setting, case 20 and the 4096 cells:

| | Ignition time, from double | Atoms | Million cells × steps per second (1e-8 s; 1e-6 s) |
|---|---|---|---|
| single | 1.6e-4 | 1.1e-4 | 232; 9.1 |
| mixed | 3.3e-6 | 2.6e-14 | 180; 7.6 |
| double | — | 1.2e-14 | 20; 1.05 |

- **Single precision fails the problem's atom requirement** (1.1e-4 against 1e-5). It is not offered
  as a default.
- **Mixed precision is the default.** It keeps atoms as well as double, moves the ignition time by
  0.0003 %, and is 9 times faster than double on this card (which computes in double at 1/64 of its
  single rate).
- Mixed precision cannot use a relative tolerance below about 1e-5: the error stops improving near
  6e-6 and the kernel rejects more and more sub-steps (364 sub-steps per step at tolerance 1e-8,
  against 15.6 at 1e-6). Use double for tight tolerances.

## Speed

Default setting, RTX 4070, nothing else on the GPU. Million cells × steps per second:

| Cells | Mixed, step 1e-8 s | Double, step 1e-8 s |
|---|---|---|
| 1000 | 14 to 15 (0.07 ms per step) | (0.32 ms per step) |
| 10 000 | 99 | |
| 100 000 | 175 | |
| 1 to 16 million | 180 to 207 | 18 to 21 |

- At a step of 1e-8 s the speed is the same in every state of the gas (before ignition, in the rise,
  burnt): about 200 million, one sub-step each.
- At a step of 1e-6 s, a million cells in mixed precision: 59 to 67 million before ignition, 28
  million in burnt gas, 3.1 to 7.5 million in the temperature rise, 7.4 to 7.9 million for a mixed
  population (17 % not started, 18 % igniting, 65 % burnt).
- Small batches are slow: below about 10 000 cells a step costs the same 0.07 ms whatever the count.
  That is the fixed cost of starting work on the GPU.
- Memory: 125 bytes per cell. 16 million cells were tested (2 GB); by that count the 12 GB card would
  hold about 90 million, not tested.
- Putting equal cells next to each other gains 20 % at large steps and nothing at 1e-8 s.

## Next to OpenFOAM on case 20

One parcel, constant pressure, 1 ms. gpucfd in double with relative tolerance 1e-8 (the tolerance
OpenFOAM's runs used), OpenFOAM's `chemFoam` on one core.

| Steps | gpucfd, wall time | OpenFOAM, wall time | gpucfd, ignition (microseconds) | OpenFOAM |
|---|---|---|---|---|
| 2500 | 1.7 s | 4.4 s | 217.7170 | 217.6711 |
| 10 000 | 2.4 s | 4.9 s | 217.7220 | 217.7011 |
| 40 000 | 7.3 s | 7.8 s | 217.7222 | 217.7077 |
| 80 000 | 14.2 s | 11.0 s | 217.7223 | 217.7088 |

The gpucfd column uses CHEMKIN's constants, OpenFOAM its own; with OpenFOAM's constants gpucfd gives
217.7099 microseconds and 2220.5360 K, and OpenFOAM taken to a step of zero gives the same to 8e-8
and 6.4e-7 K. Read from the table: gpucfd's answer hardly moves with the number of steps, because
energy is kept exactly; OpenFOAM's moves in the fifth digit.

4096 parcels at once, 80 000 steps, default setting: 5.3 s. That is less than half of what OpenFOAM
needs for one parcel.

Cost per cell and step, gpucfd on a million cells (mixed, 1e-8 s) against OpenFOAM's recorded cost
(one cell, one core, 1.25e-8 s):

| State | gpucfd | OpenFOAM | Ratio |
|---|---|---|---|
| before ignition | 209.5 million per second | 102 microseconds per step (9793 per second) | 21 000 |
| temperature rise | 204.4 million | 660 microseconds (1516 per second) | 135 000 |
| burnt gas | 206.7 million | 105 microseconds (9540 per second) | 22 000 |

**Limits of this comparison. Do not quote the ratio without them.**

- OpenFOAM's number is the cost of a whole step of a one-cell run: chemistry, but also its
  thermodynamics, bookkeeping and output. How much of the 100 microseconds is chemistry was not
  measured. On a mesh of many cells its cost per cell is probably much lower. The ratio compares "a
  GPU full of cells" with "the only OpenFOAM chemistry run we have".
- One CPU core against a whole GPU.
- Different tolerances: gpucfd's default (1e-6, mixed precision) against OpenFOAM's 1e-8. In double
  at 1e-8 s gpucfd is 10 times slower (table above), so like for like the ratios are 10 times
  smaller.
- For one parcel the GPU is the wrong tool: 14.2 s against 11.0 s.

## Expected and found

Written in `plan.yaml` before the runs; none of these is a requirement of the problem.

| Expected | Found |
|---|---|
| Within 1e-7 of `cfdlib.chem` | 3.4e-7; the difference is `cfdlib.chem`'s own accuracy |
| Constant volume: near 2500 K at about 2.4 times the pressure | 2477 K, 2.27 times |
| Small batches: 0.03 to 0.1 ms per step | 0.07 ms in mixed, 0.32 ms in double |
| At least 10 million cells × steps per second; double 5 to 20 times slower | 200 million; double 10 times slower |
| Hundreds of explicit sub-steps at 1e-6 s; implicit wins clearly | 7 to 58 sub-steps; implicit wins by 18 % in double at a loose tolerance and loses in mixed |
| A thousand times OpenFOAM's cost or more | 21 000 to 135 000, with the limits above |
| Single precision loses atoms at 1e-7 to 1e-4 | 1.1e-4: outside the requirement |
| Mixed moves the ignition time by less than 1e-4, and cannot use a tolerance below about 1e-5 | 3.3e-6; confirmed |

## What the ladder caught while building

- **A rate of exactly zero.** When a reaction's collision partner is absent, `cfdlib.chem` keeps a
  tiny floor and the kernel gave zero. Rung 1 showed it; the kernel now uses the same floor.
- **The data has a small step at 1000 K.** The heat-capacity data is two curves that meet at 1000 K,
  and for this mixture they differ there by 0.058 J/kg. Case 20 starts at exactly 1000 K and the first
  reactions absorb a little heat, so for the first 6.6 microseconds the gas has an energy for which
  no temperature exists. The search stopped on the wrong side. The kernel now settles on 1000 K
  there. The energy check allows the size of the data's step on those steps (found 8.0e-10, step
  3.2e-8) and round-off everywhere else (found 5.9e-16).
- **Explicit methods blow up with fixed sub-steps of 5e-8 s or more** at the ignition state; `bs23`
  is stable at 2.5e-8 s. The order test was moved to shorter sub-steps. With error control this
  cannot happen: a sub-step that is too long is rejected.
- **One cell on a GPU is slow.** 100 000 steps of one cell took 43 s; moving the history onto the GPU
  brought 32 s. The rest is one GPU thread working alone.

## What is worse or limited

- One scheme, one mixture. The ladder covers 2 atm and start temperatures of 1000 to 1400 K; the
  review added one state behind a detonation's shock (next section). There the answer is right but
  the default method needs 5 to 20 sub-steps per step, so the choice of integrator must be measured
  again for detonations.
- Slower than one CPU core for one cell, and no gain below about 10 000 cells.
- Built-in size limits: 12 species, 32 reactions, 3 species on each side of a reaction. Only the rate
  laws this scheme uses. A larger scheme (methane, kerosene) needs the limits raised and more rate
  laws, and will be stiffer, so the implicit method matters more there.
- The implicit method builds its matrix by differences, which is costly (11 rate evaluations) and
  poor in mixed precision.
- At large steps the kernel waits for its slowest cells: a cell in the temperature rise takes 8 to 20
  times as long as one that has not started.
- The wrapper copies the state between Python and the GPU on `set` and `get`. A flow solver must
  hand over its GPU arrays directly.
- Single precision breaks the atom requirement.
- The tests read the scheme from an OpenFOAM installation (`$FOAM_TUTORIALS`).

## At the state behind a detonation's shock (review check)

Added by the developing agent: [`dev/agent_checks/chem_kernel_review.py`](../dev/agent_checks/chem_kernel_review.py),
record `chem_kernel_review.json` next to it; pass rules written before the run. The gas right behind
a shock of the theoretical detonation speed in hydrogen and air (27.7 atm, 1542 K), burning at
constant volume. Reference: `cfdlib.chem.VolumeParcel`.

| | Reference | Kernel, default setting, steps of 2.5e-9 s |
|---|---|---|
| Time to reach 2000 K | 0.41869 microseconds | −0.005 % (rule: 0.1 %) |
| Temperature after 2 microseconds | 3381.36 K | the same to 1e-7 (rule: 0.01 %) |
| Pressure after 2 microseconds | 1.951 times the start (54 atm) | the same to 1e-7 (rule: 0.01 %) |
| Atoms | | kept to 1e-15; no cell reported trouble |

The same holds in double with tolerance 1e-8. With steps of 1e-8 s the time to 2000 K is off by
0.04 %: the kernel reads that time from a straight line across one step, and here the rise takes only
a few steps.

**What it costs there (expected 50 to 200 million cells × steps per second; found 16 to 19).**

- One sub-step per step is not enough at this pressure. The default method takes 5.5 sub-steps per
  step of 2.5e-9 s and 20 per step of 1e-8 s.
- A million cells igniting together, steps of 5e-9 s (the step OpenFOAM's trial of case 21 took on
  0.1 mm cells): 18 to 19 million cells × steps per second, 10 sub-steps per step.
- The same cells after they have burnt (3381 K, 54 atm): 16 million, 11 sub-steps per step. So burnt
  gas is not cheap here. Its fast reactions run forward and backward in balance, and an explicit
  method must still take sub-steps as short as those reactions.
- In mixed precision about a quarter of the attempted sub-steps were rejected (8 % in double).
- Limit of this number: 54 atm is the constant-volume end state. Behind a real detonation the burnt
  gas expands to about 16 atm, where the chemistry is slower. That state was not timed.

**Is another built-in method cheaper there?** Timed with a budget of 12 runs fixed in advance
([`dev/agent_checks/chem_detonation_methods.py`](../dev/agent_checks/chem_detonation_methods.py)):
3 methods × mixed and double × tolerances 1e-4 and 1e-6, on 250 000 cells for 2 microseconds in steps
of 5e-9 s. All 12 keep the accuracy rule. Million cells × steps per second, mixed precision:

| Method | Tolerance 1e-4 | Tolerance 1e-6 |
|---|---|---|
| `bs23` (explicit, the default) | 25 (8.8 sub-steps per step) | 19 (10.5), the default |
| `dp45` (explicit) | 18 (6.5) | 12 (8.8) |
| `ros34` (implicit) | 29 (2.8) | 5.6 (15) |

- The best setting is only 1.6 times cheaper than the default. No setting brings back the 200 million
  of the ignition problem.
- The explicit methods take 9 to 10 sub-steps whatever the tolerance. Their sub-step is set by
  stability (the fastest reaction), not by accuracy. That is what "stiff" means.
- The implicit method needs only 2 to 3 sub-steps, but each costs 11 rate evaluations because its
  matrix is built by differences. With a matrix written out exactly it would cost about a third of
  that. That is the most promising saving, and it is not built.
- In double every method is about ten times slower than in mixed (1.5 to 3.4 million).

What follows: in a detonation the chemistry will cost several times the flow step in every hot
cell, not about the same. Before step 4 on the GPU: give the implicit method an exact matrix, or
give burnt cells near equilibrium a cheaper treatment, and measure again.

## What coupling to the flow solver needs next (not in this stage)

1. **Species transport.** The flow solver ([`COMPRESSIBLE.md`](COMPRESSIBLE.md)) carries density,
   momentum and energy. It must also carry the mass of each species (9 more fields here), moved with
   the same fluxes, so that the fractions stay between 0 and 1 and add up to one.
2. **Heat capacities that change with temperature.** The flow solver assumes a fixed ratio of heat
   capacities. With reactions, temperature and pressure must come from the energy and the mixture
   through the same data the kernel uses (its `temperature()` function can be shared), and the shock
   treatment needs the mixture's own speed of sound. The flow's energy must include the chemical
   energy of the species, as the kernel's does.
3. **A guard against negative pressure.** After a flow step near a strong shock a cell can end with
   too little internal energy or a slightly negative species. The kernel reports such a cell
   (`status`), it does not repair it. The flow solver needs a rule: limit the fluxes, or repair and
   count.
4. **Splitting.** Each time step: flow, then chemistry at constant volume with density and internal
   energy held. That mode exists and is verified (rung 2). To be decided and measured: the order of
   the two parts (flow-chemistry, or half-chemistry, flow, half-chemistry).
5. **No copies.** The kernel should take pointers to the flow solver's GPU arrays.
6. **Skip cold cells.** In a detonation most cells are cold unburnt gas or fully burnt. A cheap test
   could skip the rates there. At steps of 1e-8 s every cell costs the same today.

What to expect in speed: at 2 atm the chemistry needs one cheap sub-step per cell at the flow
solver's step, about 200 million cells × steps per second. The flow solver itself does 160 to 190
million in 2D (`COMPRESSIBLE.md`). There chemistry costs about as much as the flow step. At
detonation pressure it does not (section above): hot cells need about ten sub-steps, so with the
default method a detonation run would spend most of its time in chemistry. Cold gas ahead of the
wave is cheap. All of this is before the cost of carrying 9 more species fields.

## Open

- OpenFOAM's chemistry timed on a mesh of many cells, for a fair cost per cell. One number exists:
  the trial for case 21 (`reactingFoam`, 1200 cells, the whole solver, chemistry tolerance 1e-3)
  cost 57 microseconds per cell and step on one core. Case 21 asks its solver agent for the share of
  the chemistry in that.
- An exact matrix for the implicit method, then the choice of integrator at detonation states again
  (section above). A second scheme.
- A cheaper explicit method with two stages for flow-sized steps; exact (not differenced) matrix for
  the implicit method; sorting cells by work at large steps.
