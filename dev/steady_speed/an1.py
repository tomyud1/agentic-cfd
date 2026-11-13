import sys, glob, os, numpy as np
sys.path.insert(0, "/mnt/d/Projects/Agentic CFD")
from cfdlib import foam
from cfdlib.verification import iteration_error
R = "/home/tomyud/OpenFOAM/tomyud-v2606/run/agentic"
N = sys.argv[1] if len(sys.argv) > 1 else "128"
def load(c):
    rows = foam.read_dat(sorted(glob.glob(f"{c}/postProcessing/probes/*/U"))[0])
    return np.array([[x for vec in r[1:] for x in vec[:2]] for r in rows])
try:
    ref = load(f"{R}/lid_driven_cavity_re100/N{N}")[-1]
except Exception:
    ref = None   # no converged run on this mesh: judge against each run's own last value
times = dict(l.split()[:2] for l in open(f"{R}/dev_steady/times.txt") if " s" in l)
names = sys.argv[2].split(",")
for name in names:
    c = f"{R}/dev_steady/{name}"
    try: v = load(c)
    except Exception as e: print(name, "no data", e); continue
    r = v[-1] if ref is None else ref
    stop, hold, ch, tol = None, 0, 10, 1e-6
    for k in range(ch, len(v), ch):
        hist = v[:k + 1][::-1][::ch][::-1]
        if max(iteration_error(hist[:, j])["error"] for j in range(v.shape[1])) < tol:
            hold += 1
            if hold >= 2: stop = k; break
        else: hold = 0
    wall = float(times.get(name, "nan")); per = wall / len(v)
    endd = np.max(np.abs(v[-1] - r))
    if stop is None:
        print(f"{name:10s} {len(v)} steps, no stop at 1e-6; end diff vs simpleFoam {endd:.1e}; {per*1e3:.1f} ms/step"); continue
    print(f"{name:10s} stop at {stop+1:5d} steps ({(stop+1)*per:6.1f} s at {per*1e3:.1f} ms/step); error at stop vs converged simpleFoam {np.max(np.abs(v[stop]-r)):.1e}; end diff {endd:.1e}")
