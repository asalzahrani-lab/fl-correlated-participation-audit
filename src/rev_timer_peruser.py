"""Reviewer-check follow-up: per-device privacy under the timer policies against an adversary who knows availability
(position (b) of the per-device section).  Under a timer the cohort target is rarely binding, so given the availability
path a device's participation is (almost) deterministic and the server's draws give no amplification.  For each device
we take its T = 200 window (stride 10) with the largest participation count N_i under the seed-0 policy and report
eps(delta) of the Gaussian pair with shift a*sqrt(N_i), i.e. the loss when the participations are known; the rounds in
which the cohort target binds (when some eligible devices are left out) are counted and reported.
Output rev_timer_peruser.csv and printed summary."""
import csv
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from rev_common import window, apply_policy
from corraudit import eps_poisson_pld

T, A_SIG, DELTAS = 200, 0.5, [1e-4, 1e-5]
def h(mu, e): return norm.cdf(mu / 2 - e / mu) - np.exp(e) * norm.cdf(-mu / 2 - e / mu) if mu > 0 else 0.0
def eps_gauss(mu, d):
    if mu <= 0 or h(mu, 0.0) <= d: return 0.0
    return brentq(lambda e: h(mu, e) - d, 0.0, 200.0)

A, _, _ = window(10, 3, verbose=False); N, Tw = A.shape
rows = []
for name, b in [("fixed cohort 0.6, timer 5", 5), ("fixed cohort 0.6, timer 20", 20), ("fixed cohort 0.6", 0)]:
    B = apply_policy(A, "fixed", 0.6, b, np.random.default_rng(0))
    K = int(round(0.6 * A.sum(axis=0).mean()))
    # eligible pool after the timer, recomputed: a round binds when more devices were eligible than taken
    last = np.full(N, -10**9); binding = 0
    for t in range(Tw):
        elig = A[:, t] & (t - last > b)
        if elig.sum() > K: binding += 1
        last[B[:, t]] = t
    q = B.mean(); claim = {d: eps_poisson_pld(q, T, 1.0, 2.0, d) for d in DELTAS}
    maxN = np.array([max(B[i, s:s + T].sum() for s in range(0, Tw - T + 1, 10)) for i in range(N)])
    part = maxN > 0
    for d in DELTAS:
        e = np.array([eps_gauss(A_SIG * np.sqrt(n), d) for n in maxN])
        rows.append([name, d, binding, f"{q:.4f}", f"{claim[d]:.3f}", int(part.sum()), int(np.median(maxN[part])), int(maxN.max()),
                     f"{np.median(e[part]):.2f}", f"{e.max():.2f}", f"{np.median(e[part])/claim[d]:.1f}", f"{e.max()/claim[d]:.1f}", f"{(e[part] > claim[d]).mean():.3f}"])
        print(f"{name}, delta {d:g}: binding rounds {binding}/{Tw}; q {q:.4f}; claim {claim[d]:.3f}; participating devices {part.sum()}; "
              f"median max-count {np.median(maxN[part]):.0f}, max {maxN.max()}; eps median {np.median(e[part]):.2f} max {e.max():.2f} "
              f"(x{np.median(e[part])/claim[d]:.1f}, x{e.max()/claim[d]:.1f}); share above claim {(e[part] > claim[d]).mean():.3f}")
with open("rev_timer_peruser.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["policy", "delta", "binding_rounds", "q", "poisson_claim", "participating_devices", "median_maxN", "max_maxN",
                                   "eps_known_median", "eps_known_max", "median_over_claim", "max_over_claim", "share_above_claim"]); w.writerows(rows)
