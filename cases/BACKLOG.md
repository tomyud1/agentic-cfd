# Case backlog

Problems to give the agent next, in order. Each one should test something the agent hasn't faced
yet. Solved cases are listed in [cases/README.md](README.md). When a case from here is solved, move it
there and note it in the [CHANGELOG](../CHANGELOG.md).

**Sources.** The list draws on two catalogues:
- The [Luminary V&V manual](https://luminary.ai/verification-and-validation/): 56 cases from a
  commercial solver. We use it as a catalogue of well-posed problems and as a commercial result to
  compare against. It has no downloads and no stated license, so reference data comes from the
  original sources it cites.
- NASA's [Turbulence Modeling Resource](https://tmbwg.github.io/turbmodels/) (TMR). It moved to
  GitHub in February 2026 and is run by the AIAA Turbulence Model Benchmarking Working Group. It is
  free and public, and provides nested grid families plus reference results from NASA's CFL3D and
  FUN3D codes. Its data is "provided as-is" (home page).

**Rule (from the workflow):** reference values are never taken from memory or from this file. The
"candidate reference" column is a starting point. Confirm the source and put its data, with the
citation, in the case's `reference/` folder before solving.

**gpucfd column:** whether the GPU solver can run the case today. The flow must be laminar and
incompressible, in a box or around immersed bodies. A ✔ case is a chance to put gpucfd next to
OpenFOAM inside the agent workflow.

## Tier 1: next (each adds one new capability)

| # | Case | New for the agent | Candidate reference | gpucfd |
|---|---|---|---|---|
| 7 | **Turbulent flat plate, SA model** (TMR verification case 2DZP) | First RANS case; wall-resolved grid (y+ ≈ 0.1 to 1.7) | TMR: 5 nested grids, CFL3D/FUN3D results for Cf at x = 0.97 and drag, with GCI. There is an incompressible data file (`cf_incomp_results_sa.dat`); TMR's main case is compressible at M = 0.2 | ✘ |
| 8 | **Differentially heated square cavity** (buoyancy cavity) | Buoyancy (Boussinesq) and coupled heat and flow | de Vahl Davis (1983) benchmark (confirm) | ✘ |

## Tier 2: harder versions of what works

| # | Case | New for the agent | Candidate reference | gpucfd |
|---|---|---|---|---|
| 9 | **Backward-facing step, laminar, Re 800** | Separation and reattachment; long domain; sensitivity to the outflow condition | Gartling (1990) benchmark; Armaly et al. (1983) experiment (confirm) | ✔ (inflow, outflow and a step as an immersed body) |
| 10 | **Lid-driven cavity, Re 1000** | Stronger nonlinearity than case 2, secondary vortices | Ghia et al. (1982), Re 1000 columns. Case 2 has only the Re 100 table (`ghia1982_re100.yaml`), so transcribe the rest | ✔ |
| 11 | **Cylinder, Re 40** (steady, separated) | Recirculation length and drag, steady wake | Literature values cited by Luminary's case page (confirm) | ✔ |
| 12 | **Sphere, Re 100** | 3D external flow and 3D meshing (snappyHexMesh) | Literature values cited by Luminary's case page (confirm) | ✔ (3D immersed body) |

## Tier 3: turbulence beyond the flat plate

| # | Case | New for the agent | Source |
|---|---|---|---|
| 13 | Bump in a channel, SA / SST (TMR 2DB) | Pressure gradients; comparing two turbulence models | TMR verification |
| 14 | NACA 0012, SA (TMR 2DN00) | Airfoil, far field, lift and drag against experiment | TMR validation; Luminary publishes CL 1.0907 and CD 0.012453 at 10°, a commercial result to compare against |
| 15 | Backward-facing step, turbulent (TMR 2DBFS, Driver & Seegmiller) | Turbulent separation, where RANS models are known to struggle | TMR validation |
| 16 | NASA wall-mounted hump (TMR 2DWMH) | Separation and reattachment: a known hard case for RANS | TMR validation |

## Later: real geometry and more physics

- **Geometry from an STL file**, meshed with snappyHexMesh. The sphere (case 12) is the first step.
- **Conjugate heat transfer:** a fin in cross-flow (Luminary `fin_effectiveness`).
- **Unsteady turbulence (DES/LES):** a cylinder at Re 3900. This is expensive, so it waits for lab
  hardware.
- **A problem with no benchmark at all**, from Prof. Frankel. This is the real test of the workflow:
  the agent has to judge its own answer without a reference.

## Verification method to adopt: manufactured solutions

Luminary verifies its solvers with the method of manufactured solutions (MMS): choose a smooth
solution, add the source terms that make it exact, and check the order of accuracy. Two uses:
- **In gpucfd's verification ladder:** checks code paths that have no natural exact solution, such
  as bodies combined with outflow.
- **In the agent workflow:** a check that the solver setup (schemes, boundary conditions) reaches
  its design order before the agent trusts it on a problem with no exact answer.
