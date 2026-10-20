"""Revision B4: validity of the reported audits against the LOWER 99% confidence limit of each reference (the
strictest reading of 'never exceeds the reference').  Reads step19_audits.npz (T = 200 simulated audits),
step16_results_*_T1000.csv per-audit values if available, and the rev_train_*.csv / fl_mlp* training results.
Prints counts of certified and uncertified estimates above the reference value and above the lower limit."""
import numpy as np, csv, glob
d = np.load("step19_audits.npz", allow_pickle=True)
LIM = {("markov", "1e-4"): (11.869, 11.736), ("markov", "1e-5"): (15.053, 14.762),
       ("trace", "1e-4"): (10.685, 10.174), ("trace", "1e-5"): (14.97, 14.213), ("iid", "1e-4"): (2.866, 2.866)}
for (law, dl), (ref, lo) in LIM.items():
    for est in ["cert", "v2"]:
        k = f"{law}_{est}_{dl}"
        if k not in d: continue
        v = np.asarray(d[k], float)
        print(f"{law:6s} {dl} {est:4s}: n={len(v)} max {v.max():.3f} mean {v.mean():.3f} | above ref {ref}: {(v > ref).sum()} | above lower limit {lo}: {(v > lo).sum()} | recovery vs ref {v.mean()/ref:.3f}, vs re-run point")
