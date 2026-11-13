# Live check of cfdlib's steady stop on a real run: case 2 N256, SIMPLEC U 0.99, from rest.
import sys, shutil, json, numpy as np
sys.path.insert(0, "/mnt/d/Projects/Agentic CFD")
from pathlib import Path
from cfdlib import foam, monitor
R = Path("/home/tomyud/OpenFOAM/tomyud-v2606/run/agentic")
src, case = R / "dev_steady/s99_256", R / "dev_steady/live_s99_256"
shutil.rmtree(case, ignore_errors=True); case.mkdir()
for d in ("system", "constant", "0"):
    shutil.copytree(src / d, case / d)
ls = monitor.LiveStop(case, monitor.steady_decider("probes", "U", 1e-6, columns=[0, 1, 3, 4, 6, 7, 9, 10]),
                      log=lambda m: print(m, flush=True))
wall = foam.run("simpleFoam", case, watch=ls, watch_every=1.0)
rows = foam.read_dat(sorted((case / "postProcessing/probes").rglob("U"))[0])
v = np.array([[x for vec in r[1:] for x in vec[:2]] for r in rows])
ref = np.load("/mnt/d/Projects/Agentic CFD/results/lid_driven_cavity_re100/runs/N256.npz")["probe_vals"][-1]
last = sorted(foam.time_dirs(case), key=lambda p: float(p.name))[-1].name
print(f"wall {wall:.1f} s, {len(v)} iterations, last written time {last}, max |probe - converged N256| {np.max(np.abs(v[-1]-ref)):.2e}")
print(json.dumps(ls.summary()["decision"]["reason"]))
