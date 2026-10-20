"""Revision round 3, B2: how much of a device's participation under the timer policies is determined by its availability.
Under a timer the eligible pool exceeds the cohort target only in the first rounds (2 of 432 under both timers), so
given the availability record a device's path is determined up to the server's draws in those rounds, which propagate
through the timers.  We run the policy under 100 seeds (same availability, different draws) and report, per device,
the number of distinct paths, the range of the largest window count N_i (T = 200, stride 10) and of the known-path
loss eps(a sqrt N_i), and the count-averaged value Sum_n p_n h_eps(a sqrt n) over the seed law of N_i (exact for an
adversary who sees the path, Proposition 3).  Output rev_timer_seeds.csv and a printed summary."""
import csv
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from rev_common import window, apply_policy
from corraudit import eps_poisson_pld

T, A_SIG, DELTAS, SEEDS = 200, 0.5, [1e-4, 1e-5], 100
def h(mu, e): return norm.cdf(mu / 2 - e / mu) - np.exp(e) * norm.cdf(-mu / 2 - e / mu) if mu > 0 else 0.0
def eps_gauss(mu, d):
    if mu <= 0 or h(mu, 0.0) <= d: return 0.0
    return brentq(lambda e: h(mu, e) - d, 0.0, 200.0)
def eps_count_avg(pn, d):
    f = lambda e: sum(p * h(A_SIG * np.sqrt(n), e) for n, p in pn.items() if n > 0) - d
    if f(0.0) <= 0: return 0.0
    return brentq(f, 0.0, 200.0)

A, _, _ = window(10, 3, verbose=False); N, Tw = A.shape
starts = list(range(0, Tw - T + 1, 10)); rows = []
for name, b in [("fixed cohort 0.6, timer 20", 20), ("fixed cohort 0.6, timer 5", 5)]:
    paths = np.zeros((SEEDS, N, Tw), dtype=bool); maxN = np.zeros((SEEDS, N), dtype=int)
    for s in range(SEEDS):
        B = apply_policy(A, "fixed", 0.6, b, np.random.default_rng(s)); paths[s] = B
        maxN[s] = np.array([max(B[i, s0:s0 + T].sum() for s0 in starts) for i in range(N)])
    q = paths[0].mean(); claim = {d: eps_poisson_pld(q, T, 1.0, 2.0, d) for d in DELTAS}
    ever = maxN.max(axis=0) > 0
    n_distinct = np.array([len({paths[s, i].tobytes() for s in range(SEEDS)}) for i in range(N)])
    invariant = (n_distinct == 1) & ever
    rngN = maxN.max(axis=0) - maxN.min(axis=0)
    print(f"{name}: participating devices {ever.sum()}; path invariant across {SEEDS} seeds: {invariant.sum()} of {ever.sum()}; "
          f"devices with N range >= 1: {(rngN[ever] >= 1).sum()}, >= 2: {(rngN[ever] >= 2).sum()}, max range {rngN.max()}")
    for d in DELTAS:
        e = np.array([[eps_gauss(A_SIG * np.sqrt(n), d) for n in maxN[s]] for s in range(SEEDS)])   # seeds x devices
        e_seed0 = e[0]; e_min = e.min(axis=0); e_max = e.max(axis=0)
        ca = np.array([eps_count_avg({int(n): c / SEEDS for n, c in zip(*np.unique(maxN[:, i], return_counts=True))}, d) if ever[i] else 0.0 for i in range(N)])
        worst = int(np.argmax(e_seed0))
        rows.append([name, d, f"{claim[d]:.3f}", int(ever.sum()), int(invariant.sum()), f"{np.median(e_seed0[ever]):.2f}", f"{e_seed0.max():.2f}", worst,
                     f"{e_min[worst]:.2f}", f"{e_max[worst]:.2f}", int(n_distinct[worst]), f"{np.median(e_min[ever]):.2f}", f"{np.median(e_max[ever]):.2f}",
                     f"{np.median(ca[ever]):.2f}", f"{ca.max():.2f}", f"{(e_min[ever] > claim[d]).mean():.3f}"])
        print(f"  delta {d:g}: claim {claim[d]:.3f}; seed-0 median {np.median(e_seed0[ever]):.2f} max {e_seed0.max():.2f} (device {worst}, over seeds "
              f"[{e_min[worst]:.2f}, {e_max[worst]:.2f}], {n_distinct[worst]} distinct paths); median over devices of the seed-min {np.median(e_min[ever]):.2f} "
              f"and seed-max {np.median(e_max[ever]):.2f}; count-averaged (Prop. 3) median {np.median(ca[ever]):.2f} max {ca.max():.2f}; "
              f"share of devices whose seed-min exceeds the claim {(e_min[ever] > claim[d]).mean():.3f}")
with open("rev_timer_seeds.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["policy", "delta", "poisson_claim", "participating_devices", "path_invariant_devices", "seed0_median", "seed0_max", "worst_device",
                "worst_min_over_seeds", "worst_max_over_seeds", "worst_distinct_paths", "median_seed_min", "median_seed_max", "count_avg_median", "count_avg_max",
                "share_seed_min_above_claim"]); w.writerows(rows)
