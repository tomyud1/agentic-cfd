"""A file of the developing agent. An agent that solves the case does not need it and should not open it.

Why case 21 has no detonation on fine grids: an independent check of the reason, done in the review of gpucfd's
joined solver (2026-11-16). The statement to check (dev/compare/detonation_tube_h2_air/why_no_detonation.py, part
"cpu", which uses cfdlib.chem and OpenFOAM's hydrogen scheme): the shock that the case's hot layer drives heats
the fresh gas to 931 K at 1.366 MPa, and gas in that state does not ignite within the 50 microseconds of the run.

Here the same question with Cantera and Cantera's own hydrogen scheme (h2o2.yaml: other reactions and other
thermodynamic data): how long does stoichiometric hydrogen and air take to ignite at constant volume
(temperature 400 K above its start)? Cantera is not part of the project's environment.

    <python with cantera> dev/agent_checks/case21_ignition_cantera.py

Found on 2026-11-16 with Cantera 3.2.0 (next to it: cfdlib.chem with OpenFOAM's scheme, from why_no_detonation.json):
    1.366 MPa,  931 K   no ignition within 2000 microseconds     (no ignition within 200)
    1.650 MPa, 1050 K   no ignition within 2000 microseconds     (no ignition within 200)
    2.000 MPa, 1180 K   159.5 microseconds                       (142.7)
    2.433 MPa, 1388 K   2.60 microseconds                        (1.10)
    2.810 MPa, 1543 K   0.56 microseconds                        (0.41)
The first row is the state behind the hot layer's shock; the second is what the fine-grid runs show behind the
shock at first; the last is the state at the front of a Chapman-Jouguet detonation.
"""
import cantera as ct

WAIT = 2e-3                                                      # s
print("cantera", ct.__version__)
for p, T in ((1.366e6, 931.0), (1.65e6, 1050.0), (2.0e6, 1180.0), (2.433e6, 1388.0), (2.81e6, 1543.0)):
    g = ct.Solution("h2o2.yaml")
    g.TPX = T, p, "H2:2, O2:1, N2:3.76"
    r = ct.IdealGasReactor(g)
    net = ct.ReactorNet([r])
    t_ign, t = None, 0.0
    while t < WAIT:
        t = net.step()
        if r.T >= T + 400.0:
            t_ign = t
            break
    print(f"{p / 1e6:.3f} MPa, {T:.0f} K: " + (f"ignites after {1e6 * t_ign:.2f} microseconds" if t_ign else f"no ignition within {1e6 * WAIT:.0f} microseconds"))
