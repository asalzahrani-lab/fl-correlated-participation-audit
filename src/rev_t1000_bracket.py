"""Revision round 3, B3: the upper end of Proposition 2's bracket at T = 1,000 for the running example (5-minute rounds,
4-day window, fixed cohort 0.6 x E[pool], seed 0, stride 5), i.e. Theorem 3 on the empirical count law of the T = 1,000
windows, next to the Poisson per-round value; the lower end is the Monte-Carlo reference of step16 (51.26 / 58.24).
Also the schema bound at the largest count.  Output rev_t1000_bracket.csv."""
import csv
import numpy as np
from rev_common import window, apply_policy
from theory import h_gauss
from corraudit import eps_poisson_pld, eps_from_delta_curve

T, A_SIG = 1000, 0.5
REF = {1e-4: 51.2579, 1e-5: 58.2427}            # step16_accounting_trace_T1000.csv (2M samples)
A, _, _ = window(5, 4)
B = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0)); q = B.mean(); N, Tw = B.shape
counts = np.array([B[i, s0:s0 + T].sum() for i in range(N) for s0 in range(0, Tw - T + 1, 5)])
pn = np.bincount(counts, minlength=T + 1) / len(counts); n = np.arange(len(pn)); mu = np.sqrt(n) * A_SIG
rows = []
for d in [1e-4, 1e-5]:
    thm3 = eps_from_delta_curve(lambda e: float(np.sum(pn * h_gauss(mu, e))), d, 0.0, 400.0)
    schema = eps_from_delta_curve(lambda e: float(h_gauss(np.sqrt(counts.max()) * A_SIG, e)), d, 0.0, 400.0)
    pois = eps_poisson_pld(q, T, 1.0, 2.0, d)
    rows.append([d, f"{q:.4f}", f"{pois:.3f}", f"{REF[d]:.2f}", f"{REF[d]/pois:.2f}", f"{thm3:.2f}", f"{thm3/pois:.2f}", int(counts.max()), f"{schema:.2f}"])
    print(f"T={T} delta={d:g}: q {q:.4f}; Poisson {pois:.3f}; reference {REF[d]:.2f} ({REF[d]/pois:.2f}x); Theorem 3 {thm3:.2f} ({thm3/pois:.2f}x); max N {counts.max()}, schema {schema:.2f}")
with open("rev_t1000_bracket.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["delta", "q", "poisson_perround", "reference_mc", "ratio_lower", "thm3_upper", "ratio_upper", "max_N", "schema"]); w.writerows(rows)
