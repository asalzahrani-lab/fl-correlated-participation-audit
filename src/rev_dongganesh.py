"""Revision B6: numerical agreement with the Monte-Carlo accountant of Dong and Ganesh (arXiv:2602.09338, Thm 5.1).

Their b-min-sep sampler (exclude the previous b'-1 rounds after a participation, then include with probability p,
warm start from the stationary law) is our minimum-separation renewal law with b = b' - 1 and the same p = q/(1 - q b).
For DP-SGD (identity strategy matrix) their recursion reads, in noise units,
    f_i = (1 - p) f_{i+1} + p exp(a y_i - a^2/2) f_{i+b'},   f_i = 1 for i > T,
    L(y) = (f_1 + p sum_{i=2}^{b'} f_i) / (1 + (b' - 1) p).
We implement it independently (log domain), evaluate it on the same Monte-Carlo samples as our forward algorithm over
the hidden-Markov representation, and compare log L sample by sample and the resulting eps(delta).  p = 1 (b = 9 at
q = 0.1) is their balls-in-bins case, i.e. our deterministic cycle.  Output rev_dongganesh.csv."""
import csv
import numpy as np
from corraudit import minsep_chain
from certified import ChainLaw, plain_eps

T, q, a, n = 200, 0.1, 0.5, 100_000


def loglr_dg(Y, bprime, p):
    m = Y.shape[0]; logf = np.zeros((m, T + bprime + 1))            # index i = 1..T, logf[:, i]; beyond T -> 0
    l1p = np.log1p(-p) if p < 1 else -np.inf; lp = np.log(p)
    for i in range(T, 0, -1):
        stay = l1p + logf[:, i + 1] if p < 1 else np.full(m, -np.inf)
        part = lp + a * Y[:, i - 1] - a * a / 2 + (logf[:, i + bprime] if i + bprime <= T else 0.0)
        logf[:, i] = np.logaddexp(stay, part)
    terms = [logf[:, 1]] + [lp + logf[:, i] for i in range(2, bprime + 1)]
    return np.logaddexp.reduce(np.stack(terms), axis=0) - np.log1p((bprime - 1) * p)


rows = []
for b in [2, 5, 9]:
    P, pi, emit = minsep_chain(q, b); law = ChainLaw(P, pi, emit, T)
    p = q / (1 - q * b); rng = np.random.default_rng(1200 + b)
    Bp = law.sample(n, rng).astype(float)
    Yp = a * Bp + rng.standard_normal((n, T)); Yq = rng.standard_normal((n, T))
    ours_p, ours_q = law.loglr(Yp, a), law.loglr(Yq, a)
    dg_p, dg_q = loglr_dg(Yp, b + 1, min(p, 1.0)), loglr_dg(Yq, b + 1, min(p, 1.0))
    diff = max(np.abs(ours_p - dg_p).max(), np.abs(ours_q - dg_q).max())
    e_ours = {d: plain_eps(ours_p, ours_q, d) for d in (1e-4, 1e-5)}
    e_dg = {d: plain_eps(dg_p, dg_q, d) for d in (1e-4, 1e-5)}
    rows.append([b, b + 1, f"{p:.4f}", n, f"{diff:.2e}", f"{e_ours[1e-4]:.4f}", f"{e_dg[1e-4]:.4f}", f"{e_ours[1e-5]:.4f}", f"{e_dg[1e-5]:.4f}"])
    print(f"b={b} (their b'={b+1}, p={p:.3f}): max |log L_ours - log L_DG| = {diff:.2e}; eps(1e-4) {e_ours[1e-4]:.4f} vs {e_dg[1e-4]:.4f}; eps(1e-5) {e_ours[1e-5]:.4f} vs {e_dg[1e-5]:.4f}", flush=True)
with open("rev_dongganesh.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["b_ours", "b_theirs", "p", "n_mc", "max_abs_diff_logLR", "eps_1e-4_ours", "eps_1e-4_DG", "eps_1e-5_ours", "eps_1e-5_DG"]); w.writerows(rows)
