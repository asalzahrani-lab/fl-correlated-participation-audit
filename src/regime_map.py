"""Poisson-error map over (lambda, sigma, T): true eps of the per-round release under a two-state Markov
participation process vs Poisson accounting, at delta=1e-4, q=0.1, C=1."""
import numpy as np, csv, time
from corraudit import *
q, C, DELTA = 0.1, 1.0, 1e-4
rng = np.random.default_rng(99)
out = open("regime_map.csv", "a", newline=""); w = csv.writer(out)
import os
if os.path.getsize("regime_map.csv") == 0:
    w.writerow(["T", "sigma", "lambda", "eps_true_mc", "eps_poisson", "ratio"])
t0 = time.time()
for T in [200, 1000]:
    for sigma in [1.0, 2.0, 4.0, 8.0]:
        ep = eps_poisson_pld(q, T, C, sigma, DELTA)
        for lam in [0.0, 0.3, 0.6, 0.9]:
            P, pi, emit = two_state_chain(q, lam)
            n = 300_000 if T == 200 else 150_000
            e = eps_full_mc(P, pi, emit, T, C, sigma, DELTA, n=n, rng=rng)
            w.writerow([T, sigma, lam, f"{e:.4f}", f"{ep:.4f}", f"{e/ep:.3f}"]); out.flush()
            print(f"T={T} sigma={sigma} lam={lam}: true={e:.3f} poisson={ep:.3f} ratio={e/ep:.2f} [{time.time()-t0:.0f}s]", flush=True)
out.close()
