"""Revision round 5: an independent recomputation of the upper end of the per-device bracket (position (b)).

The published upper end (Theorem 2 of the journal version = Theorem 3 of the extended version) is computed in
rev_peruser_exact2.py from the Poisson-binomial count law of each device window by an in-place recursion
(poisson_binomial) and theory.eps_jensen_full (Brent's method on delta(eps) = sum_n p_n h_eps(a sqrt n)).
This script recomputes it with different code paths:
  * the Poisson-binomial law by the discrete Fourier transform of its characteristic function
    (p_n = (T+1)^-1 sum_k exp(-i w k n) prod_j (1 - r_j + r_j exp(i w k)), w = 2 pi / (T + 1));
  * h_eps(mu) = Phi(mu/2 - eps/mu) - exp(eps) Phi(-mu/2 - eps/mu) in log space (scipy.special.log_ndtr);
  * eps(delta) by plain bisection on [0, 400] to 1e-10;
on every one of the 24 windows of every one of the 402 ever-eligible devices, at delta = 1e-4 and 1e-5, and compares
the per-device maximum over windows with the column thm3_<delta> of results/rev_peruser_exact2_devices.csv.
Output: rev_round5_upper_check.csv (per device) and the log printed to stdout."""
import csv
import numpy as np
from scipy.special import log_ndtr, ndtr
from rev_common import window

T, SIGMA, C = 200, 2.0, 1.0
A_SIG = C / SIGMA
DELTAS = [1e-4, 1e-5]


def pb_dft(rates, T_):
    r = np.asarray(rates, float)
    k = np.arange(T_ + 1); w = 2 * np.pi / (T_ + 1)
    z = np.exp(1j * w * k)
    phi = np.prod(1 - r[:, None] + r[:, None] * z[None, :], axis=0)       # characteristic function at w k
    n = np.arange(T_ + 1)
    p = np.real(np.exp(-1j * w * np.outer(n, k)) @ phi) / (T_ + 1)
    p = np.clip(p, 0.0, None)
    return p / p.sum()


def h(mu, eps):
    mu = np.asarray(mu, float); out = np.zeros_like(mu)
    m = mu > 0
    x1 = mu[m] / 2 - eps / mu[m]; x2 = -mu[m] / 2 - eps / mu[m]
    out[m] = ndtr(x1) - np.exp(eps + log_ndtr(x2))
    return np.clip(out, 0.0, None)


def eps_upper(pn, delta):
    mu = A_SIG * np.sqrt(np.arange(len(pn)))
    f = lambda e: float(np.sum(pn * h(mu, e))) - delta
    if f(0.0) <= 0: return 0.0
    lo, hi = 0.0, 400.0
    while hi - lo > 1e-10:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0: lo = mid
        else: hi = mid
    return hi


if __name__ == "__main__":
    A, _, _ = window(10, 3, verbose=False); N, Tw = A.shape
    pool = A.sum(axis=0); K = int(round(0.6 * pool.mean())); rt = np.minimum(1.0, K / np.maximum(pool, 1))
    starts = list(range(0, Tw - T + 1, 10))
    pub = {int(r["device"]): r for r in csv.DictReader(open("rev_peruser_exact2_devices.csv"))}
    rows = []; diffs = {d: [] for d in DELTAS}
    for i in range(N):
        if A[i].sum() == 0: continue
        vals = np.array([[eps_upper(pb_dft(A[i, s0:s0 + T] * rt[s0:s0 + T], T), d) for d in DELTAS] for s0 in starts])
        row = [i]
        for k, d in enumerate(DELTAS):
            mine = float(vals[:, k].max()); theirs = float(pub[i][f"thm3_{d:g}"])
            diffs[d].append(abs(mine - theirs)); row += [f"{mine:.6f}", f"{theirs:.6f}", int(np.argmax(vals[:, k]))]
        rows.append(row)
    with open("rev_round5_upper_check.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["device", "upper_1e-4_independent", "upper_1e-4_published", "window_1e-4",
                    "upper_1e-5_independent", "upper_1e-5_published", "window_1e-5"])
        w.writerows(rows)
    print(f"K = {K}; devices ever eligible: {len(rows)}; windows per device: {len(starts)}")
    for k, d in enumerate(DELTAS):
        col = np.array([float(r[1 + 3 * k]) for r in rows])
        top = rows[int(np.argmax(col))]
        print(f"delta={d:g}: independent upper end median {np.median(col):.3f}, max {col.max():.4f} (device {top[0]}, window {top[3 + 3 * k]});"
              f" max |independent - published| over devices {max(diffs[d]):.2e}")
