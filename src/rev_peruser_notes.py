"""Revision B1 supporting numbers for the per-device subsection and the tail-device explanation (R4).
Reads rev_peruser_devices.csv; recomputes the running-example policy (fixed cohort 0.6, seed 0)."""
import csv
import numpy as np
from rev_common import window, apply_policy

A, _, _ = window(10, 3, verbose=False); N, Tw = A.shape
B = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0))
pool = A.sum(axis=0); K = int(round(0.6 * pool.mean())); rt = np.minimum(1.0, K / np.maximum(pool, 1))
full = pool <= K
print(f"K={K}; rounds where the whole pool is taken: {full.mean():.3f} ({full.sum()} of {Tw}); pool range {pool.min()}-{pool.max()}")
avail = A.mean(axis=1); ever = avail > 0
hit_full = (A & full[None, :]).any(axis=1)
print(f"ever-available devices {ever.sum()} ({ever.mean():.3f}); of them available in at least one take-all round: {hit_full[ever].mean():.3f}")
rows = list(csv.DictReader(open("rev_peruser_devices.csv")))
bpr = np.array([float(r["b_pr_0.0001"]) for r in rows]); afm = np.array([float(r["a_fm_0.0001"]) for r in rows]); athm = np.array([float(r["a_thm3_0.0001"]) for r in rows])
bfm = np.array([float(r["b_fm_0.0001"]) for r in rows]); emp = np.array([float(r["emp_fm_0.0001"]) for r in rows])
print(f"b_pr among ever-available: median {np.median(bpr[ever]):.3f} min {bpr[ever].min():.3f}; share > 1.219: {(bpr[ever] > 1.219).mean():.3f}")
print(f"b_pr among ever-available and never in a take-all round: n={(ever & ~hit_full).sum()} median {np.median(bpr[ever & ~hit_full]) if (ever & ~hit_full).any() else float('nan'):.3f}")
# single Gaussian round at mu = 0.5
from scipy.stats import norm
from scipy.optimize import brentq
mu = 0.5; dl = lambda e: norm.cdf(-e / mu + mu / 2) - np.exp(e) * norm.cdf(-e / mu - mu / 2)
print(f"one unsubsampled round (mu=0.5): eps(1e-4) = {brentq(lambda e: dl(e) - 1e-4, 0.01, 20):.3f}, eps(1e-5) = {brentq(lambda e: dl(e) - 1e-5, 0.01, 20):.3f}")
# correlation of per-device eps with availability
from scipy.stats import spearmanr
print(f"Spearman(avail, b_pr) = {spearmanr(avail, bpr)[0]:.3f}; Spearman(avail, a_fm) = {spearmanr(avail, afm)[0]:.3f}; Spearman(avail, emp_fm) = {spearmanr(avail, emp)[0]:.3f}")
# quantiles among ever-available devices
for nm, v in [("a_fm", afm), ("emp_fm", emp), ("b_fm", bfm), ("a_thm3", athm), ("b_pr", bpr)]:
    print(nm, "ever-available: median %.3f p90 %.3f max %.3f" % (np.median(v[ever]), np.quantile(v[ever], .9), v.max()))
# tail devices of the running example (R4)
maxN = np.array([max(B[i, s0:s0 + 200].sum() for s0 in range(0, Tw - 200 + 1, 10)) for i in range(N)])
tail = np.flatnonzero(maxN >= 60)
rank = (avail[:, None] < avail[None, :]).sum(axis=1)  # number of devices with higher availability
incl = np.array([B[i][A[i]].mean() for i in tail])
print(f"tail devices (N>=60): {len(tail)}; availability {avail[tail].min():.3f}-{avail[tail].max():.3f}; top share {(rank[tail].max()+1)/N:.3f}; inclusion | available median {np.median(incl):.3f} range {incl.min():.3f}-{incl.max():.3f}; population inclusion|available {B[A].mean():.3f}")
# per-device budget: which devices top each knowledge level
for nm, v in [("b_pr", bpr), ("a_thm3", athm), ("a_fm", afm)]:
    top = np.argsort(-v)[:5]
    print(nm, "top5:", [(int(i), round(float(avail[i]), 3), round(float(v[i]), 2)) for i in top])
