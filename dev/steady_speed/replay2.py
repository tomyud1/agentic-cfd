# Replay an error-based stop (cfdlib.verification.iteration_error) on case 2's recorded probe histories.
import sys, math, numpy as np
sys.path.insert(0, "/mnt/d/Projects/Agentic CFD")
from cfdlib import foam
from cfdlib.verification import iteration_error
R = "/home/tomyud/OpenFOAM/tomyud-v2606/run/agentic/lid_driven_cavity_re100"
U = float(sys.argv[1]); tols = [float(x) for x in sys.argv[2].split(",")]; chunks = [int(x) for x in sys.argv[3].split(",")]
for N in (32, 64, 128, 256):
    rows = foam.read_dat(f"{R}/N{N}/postProcessing/probes/0/U")
    it = np.array([r[0] for r in rows]); v = np.array([[c for vec in r[1:] for c in vec[:2]] for r in rows]) / U
    final = v[-1]
    for tol in tols:
        for ch in chunks:
            stop = None; hold = 0
            for k in range(ch, len(it), ch):
                hist = v[ch - 1:k + 1:ch] if False else v[:k + 1][::-1][::ch][::-1]
                errs = [iteration_error(hist[:, j])["error"] for j in range(v.shape[1])]
                if max(errs) < tol:
                    hold += 1
                    if hold >= 2: stop = k; break
                else: hold = 0
            if stop is None: print(f"N{N} tol {tol:.0e} chunk {ch}: never (recorded {int(it[-1])})"); continue
            true = np.max(np.abs(v[stop] - final))
            print(f"N{N} tol {tol:.0e} chunk {ch}: stop at {int(it[stop])} of {int(it[-1])} ({it[stop]/it[-1]:.0%}); true error {true:.2e} (est<{tol:.0e}) {'OK' if true < tol else 'UNDER-ESTIMATE'}")
