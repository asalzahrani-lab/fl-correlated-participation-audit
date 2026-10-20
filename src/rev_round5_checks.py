"""Revision round 5: the two counterexamples of the fifth-round review, recomputed with the paper's own tools.

(1) Dispersion does not determine the final-model curve.  Sum-statistic model, T = 200, q = 0.1, a = 0.5 (sigma = 2,
    C = 1).  The count law P(N=19) = 18/181, P(N=20) = 0.9, P(N=200) = 1/1810 has mean 20, variance 18 and dispersion
    D = 0.9 = 1 - q, exactly those of Binomial(200, 0.1), yet Theorem 1 gives a final-model eps of 23.0 against 2.69 at
    delta = 1e-4: the curve is set by the upper tail of the count law, not by its first two moments.
(2) A deterministic count does not by itself dominate rate-matched Poisson sampling.  Full per-round transcript, T = 2,
    a = 0.5, q = 1/2: random allocation (exactly one of the two rounds, uniformly) against independent participation
    (each round with probability 1/2), absent law N(0, I_2).  Hockey-stick divergences in both directions on a 2-D grid
    (step 0.005 on [-10, 10]^2), at eps = 0 (total variation), 0.5, 1 and 2.
Output rev_round5_checks.log (printed) and rev_round5_checks.csv."""
import csv
import numpy as np
from scipy.stats import norm, binom
from corraudit import eps_sum_exact

T, A, SIGMA, C = 200, 0.5, 2.0, 1.0
rows = []
pn = np.zeros(T + 1); pn[19] = 18 / 181; pn[20] = 0.9; pn[200] = 1 / 1810
n = np.arange(T + 1); mean = float((n * pn).sum()); var = float((n ** 2 * pn).sum() - mean ** 2)
pb = binom.pmf(n, T, 0.1); mb = float((n * pb).sum()); vb = float((n ** 2 * pb).sum() - mb ** 2)
print(f"(1) count law: sum {pn.sum():.12f}, mean {mean:.6f}, variance {var:.6f}, D {var/mean:.6f}; Binomial: mean {mb:.6f}, variance {vb:.6f}, D {vb/mb:.6f}")
for d in (1e-4, 1e-5):
    e_c = eps_sum_exact(pn, C, SIGMA, T, d); e_b = eps_sum_exact(pb, C, SIGMA, T, d)
    print(f"    delta={d:g}: final-model eps, equal-dispersion law {e_c:.4f}; Binomial {e_b:.4f}; ratio {e_c/e_b:.2f}")
    rows.append(["dispersion", d, "", f"{e_c:.4f}", f"{e_b:.4f}", f"{e_c/e_b:.3f}"])
x = np.linspace(-10, 10, 4001); X, Y = np.meshgrid(x, x, indexing="ij"); dx = x[1] - x[0]
f = lambda mx, my: norm.pdf(X, mx) * norm.pdf(Y, my)
Q = f(0, 0); Pal = .5 * f(A, 0) + .5 * f(0, A); Pii = .25 * (f(0, 0) + f(A, 0) + f(0, A) + f(A, A))
hs = lambda P, R, e: float(np.sum(np.maximum(P - np.exp(e) * R, 0)) * dx * dx)
print("(2) T = 2, q = 1/2, a = 0.5: hockey-stick divergences, random allocation vs independent participation")
for e in (0.0, 0.5, 1.0, 2.0):
    v = [hs(Pal, Q, e), hs(Q, Pal, e), hs(Pii, Q, e), hs(Q, Pii, e)]
    print(f"    eps={e:.1f}: allocation H(P||Q) {v[0]:.7f}  H(Q||P) {v[1]:.3e} | independent H(P||Q) {v[2]:.7f}  H(Q||P) {v[3]:.3e}")
    rows.append(["allocation_vs_iid", "", e, f"{v[0]:.8g};{v[1]:.8g}", f"{v[2]:.8g};{v[3]:.8g}", ""])
with open("rev_round5_checks.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["check", "delta", "eps", "value_A", "value_B", "ratio"]); w.writerows(rows)
