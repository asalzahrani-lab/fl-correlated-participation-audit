"""Revision round 10 (internal review), M7: where the observation model fails.
Re-runs the configuration in which the Model-1 diagnostics failed (DP-FedAvg, softmax regression on 50 standardised
principal components of Fashion-MNIST, d = 510 parameters, RANDOM canary directions) next to the control used since
(centred raw pixels, d = 7,850, canary directions drawn from the low-variance principal subspace), with the settings of
fl_sim.py (T = 200, q = 0.1, sigma = 2, C = 0.5, 200 real clients, 400 canaries, lr = 0.05, Markov lambda = 0.9).
Per run: the erasure ratio (present canaries' mean sum score over its Model-1 prediction, as in fl_mlp*.py) and the
absent canaries' noise scale (noise units) and mean lag-1 autocorrelation.
Output rev_round10_erasure.csv."""
import csv, sys, time
import numpy as np
from fl_sim import prepare, run_once

T, q, SIG, C, NC, M = 200, 0.1, 2.0, 0.5, 200, 400
R = {"pca50_random": int(sys.argv[1]) if len(sys.argv) > 1 else 10, "raw_lowvar": int(sys.argv[2]) if len(sys.argv) > 2 else 4}
t0 = time.time(); log = lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True)
rows = []
for name, data in [("pca50_random", prepare(dim=50, raw=False)), ("raw_lowvar", prepare(raw=True, null_from=150))]:
    rng = np.random.default_rng(1013)
    for rep in range(R[name]):
        res = run_once("markov", 0.9, T, q, SIG, C, NC, M, data, rng, lr=0.05)
        X, present, counts = res["X"] / SIG, res["present"], res["counts"]   # X in noise units
        S = res["X"].sum(axis=1) / np.sqrt(T)                                 # sum score in units of C (as in fl_sim.py)
        erasure = S[present].mean() / (counts[present].mean() / np.sqrt(T))
        Xa = X[~present]; scale = float(Xa.std()); lag1 = float(np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in Xa]))
        rows.append([name, res["d"], rep, f"{res['acc']:.4f}", f"{erasure:.3f}", f"{scale:.3f}", f"{lag1:.4f}"])
        log(f"{name} (d={res['d']}) rep {rep}: acc {res['acc']:.3f} erasure {erasure:.3f} absent scale {scale:.3f} lag-1 {lag1:+.4f}")
with open("rev_round10_erasure.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["config", "d", "rep", "acc", "erasure_ratio", "absent_scale", "absent_lag1"]); w.writerows(rows)
for name in R:
    e = np.array([float(r[4]) for r in rows if r[0] == name])
    log(f"{name}: erasure ratio mean {e.mean():.3f} (min {e.min():.3f}, max {e.max():.3f}) over {len(e)} runs")
log("done")
