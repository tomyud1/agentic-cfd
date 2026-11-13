# Does a short run rank the relaxation settings? Iterations per decade of the slowest probe, from rows [a, b).
import sys, glob, math, numpy as np
sys.path.insert(0, "/mnt/d/Projects/Agentic CFD")
from cfdlib import foam
R = "/home/tomyud/OpenFOAM/tomyud-v2606/run/agentic"
def load(c):
    rows = foam.read_dat(sorted(glob.glob(f"{c}/postProcessing/probes/*/U"))[0])
    return np.array([[x for vec in r[1:] for x in vec[:2]] for r in rows])
def per_decade(v, a, b):
    # change size per row, max over probes, fit log10 slope over [a, b)
    d = np.max(np.abs(np.diff(v[a:b], axis=0)), axis=1)
    i = np.arange(len(d)); m = d > 0
    s = np.polyfit(i[m], np.log10(d[m]), 1)[0]
    return -1 / s if s < 0 else math.inf
for name, src in [("N128 a0.9", "lid_driven_cavity_re100/N128"), ("N128 a0.95", "dev_steady/s95"), ("N128 a0.97", "dev_steady/s97"),
                  ("N128 a0.98", "dev_steady/s98"), ("N128 a0.99", "dev_steady/s99"), ("N128 a0.995", "dev_steady/s995"),
                  ("N256 a0.9", "lid_driven_cavity_re100/N256"), ("N256 a0.97", "dev_steady/s97_256"), ("N256 a0.99", "dev_steady/s99_256"),
                  ("N256 a0.995", "dev_steady/s995_256"), ("N512 a0.99", "dev_steady/s99_512"), ("N512 seq a0.99", "dev_steady/seq_s99_512")]:
    v = load(f"{R}/{src}")
    print(f"{name:15s} it/decade: rows 100-300 {per_decade(v,100,300):7.0f}  200-400 {per_decade(v,200,400):7.0f}  last 500 {per_decade(v,len(v)-500,len(v)):7.0f}  (rows {len(v)}, last change {np.max(np.abs(v[-1]-v[-2])):.1e})")
