"""Per-round references with 99% intervals for the renewal laws b = 2, 5 and the cycle b = 9 (q = 0.1, T = 200)."""
import csv, numpy as np
from corraudit import minsep_chain
from certified import ChainLaw
from rev_common import eps_interval
T, q, a, n = 200, 0.1, 0.5, 1_000_000
rows = []
for b in [2, 5, 9]:
    P, pi, emit = minsep_chain(q, b); law = ChainLaw(P, pi, emit, T); rng = np.random.default_rng(1300 + b)
    lp = np.empty(n); lq = np.empty(n)
    for s in range(0, n, 25_000):
        B = law.sample(25_000, rng).astype(float)
        lp[s:s + 25_000] = law.loglr(a * B + rng.standard_normal((25_000, T)), a)
        lq[s:s + 25_000] = law.loglr(rng.standard_normal((25_000, T)), a)
    for d, pois in [(1e-4, 2.8662), (1e-5, 3.3597)]:
        e, lo, hi = eps_interval(lp, lq, d, alpha=0.01)
        rows.append([b, d, n, f"{e:.4f}", f"{lo:.4f}", f"{hi:.4f}", pois, f"{e/pois:.3f}", f"{lo/pois:.3f}", f"{hi/pois:.3f}"])
        print(f"b={b} delta={d:g}: eps {e:.4f} [{lo:.4f}, {hi:.4f}] ratio {e/pois:.3f} [{lo/pois:.3f}, {hi/pois:.3f}]", flush=True)
with open("rev_renewal_ci.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["b", "delta", "n_mc", "eps", "ci_lo", "ci_hi", "poisson", "ratio", "ratio_lo", "ratio_hi"]); w.writerows(rows)
