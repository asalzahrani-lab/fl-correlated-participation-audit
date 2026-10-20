"""Revision round 7 (new reviewer, question Q7): two further sensitivity checks of the trace results.

(1) Round length at a fixed wall-clock horizon.  The same 3-day window of the paper (the 10-minute analysis window,
    same absolute start), the same running policy (fixed cohort of 0.6 times the mean pool, seed 0), horizons of about
    33 hours: 5-minute rounds (T = 400, stride 20), 10-minute rounds (T = 200, stride 10: the paper's running example)
    and 30-minute rounds (T = 67, stride 3).  For each: q, D, and the exact final-model eps (Theorem 1) of the empirical
    count law against the Binomial(T, q) value at delta = 1e-4 and 1e-5.
(2) Device subsampling.  Random subsets of 50% and 80% of the 572 devices (20 subsets each, subset seeds 0..19); the
    policy is re-applied to each subset (cohort 0.6 times the subset's mean pool, seed 0); final-model ratio at T = 200.
Output rev_round7_trace_sens.csv and the printed log."""
import csv
import numpy as np
from scipy.stats import binom
from rev_common import window, apply_policy, windows, log
from corraudit import eps_sum_exact

SIGMA, C = 2.0, 1.0
DELTAS = [1e-4, 1e-5]


def fm_ratio(B, T, stride):
    counts, _ = windows(B, T, stride); pn = np.bincount(counts, minlength=T + 1) / len(counts)
    q = counts.mean() / T; D = counts.var() / counts.mean(); out = [q, D, len(counts)]
    for d in DELTAS:
        e = eps_sum_exact(pn, C, SIGMA, T, d); eb = eps_sum_exact(binom.pmf(np.arange(T + 1), T, q), C, SIGMA, T, d)
        out += [e, eb, e / eb]
    return out


rows = []
A10, sel10, start10 = window(10, 3)
log(f"10-min analysis window: start index {start10}, {A10.shape[0]} devices")
for rmin, T, stride in [(5, 400, 20), (10, 200, 10), (30, 67, 3)]:
    start = int(round(start10 * 10 / rmin)); W = int(3 * 24 * 60 / rmin)
    A, sel, _ = window(rmin, 3, start=start, verbose=False)
    B = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0))
    q, D, nw, e4, b4, r4, e5, b5, r5 = fm_ratio(B, T, stride)
    log(f"round {rmin} min, T={T} ({T*rmin/60:.1f} h), {A.shape[0]} devices, K={int(round(0.6*A.sum(axis=0).mean()))}: q={q:.4f} D={D:.2f} "
        f"final-model eps {e4:.3f} vs Binomial {b4:.3f} (ratio {r4:.2f}) at 1e-4, {e5:.3f} vs {b5:.3f} ({r5:.2f}) at 1e-5")
    rows.append(["round_length", rmin, T, stride, A.shape[0], "", f"{q:.4f}", f"{D:.2f}", nw, f"{e4:.3f}", f"{b4:.3f}", f"{r4:.3f}", f"{e5:.3f}", f"{b5:.3f}", f"{r5:.3f}"])

N = A10.shape[0]
for frac in (0.5, 0.8):
    r4s, r5s = [], []
    for s in range(20):
        keep = np.sort(np.random.default_rng(s).choice(N, int(round(frac * N)), replace=False))
        B = apply_policy(A10[keep], "fixed", 0.6, 0, np.random.default_rng(0))
        q, D, nw, e4, b4, r4, e5, b5, r5 = fm_ratio(B, 200, 10); r4s.append(r4); r5s.append(r5)
        rows.append(["device_subsample", 10, 200, 10, len(keep), s, f"{q:.4f}", f"{D:.2f}", nw, f"{e4:.3f}", f"{b4:.3f}", f"{r4:.3f}", f"{e5:.3f}", f"{b5:.3f}", f"{r5:.3f}"])
    log(f"{int(frac*100)}% of devices ({int(round(frac*N))}), 20 subsets: final-model ratio median {np.median(r4s):.2f} [{min(r4s):.2f}, {max(r4s):.2f}] at 1e-4, "
        f"{np.median(r5s):.2f} [{min(r5s):.2f}, {max(r5s):.2f}] at 1e-5")
with open("rev_round7_trace_sens.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["check", "round_min", "T", "stride", "devices", "subset_seed", "q", "D", "n_windows", "fm_eps_1e-4", "fm_binom_1e-4",
                                   "fm_ratio_1e-4", "fm_eps_1e-5", "fm_binom_1e-5", "fm_ratio_1e-5"]); w.writerows(rows)
