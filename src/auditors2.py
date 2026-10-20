"""Step 4-5: dependence-robust one-run auditors.

All auditors receive per-canary data from ONE run with m canaries, each present with prob 1/2 (auditor's coin),
each following a KNOWN participation process (the auditor chose it), and the per-round observations
X[i, t] = <noisy aggregate_t, u_i> / C_claimed  (all-iterates threat model).

  gauss_sum_audit        : Agrawal-style Gaussian fit on S_i = sum_t X[i,t]/sqrt(T), Bonferroni box.
  count_cond_audit       : present law = exact count mixture (known p_n) with estimated (mu0, s0, beta), Bonferroni box.
  path_aware_audit       : estimate a = C/sigma with a CI, report eps_full(a_lo) from a precomputed table
                           (Monte-Carlo on the HMM likelihood ratio under the known process).
  lr_threshold_audit     : nonparametric: score = HMM log-likelihood ratio at the CLAIMED (C, sigma);
                           delta(eps) = CP_lo[P_present(score > t)] - e^eps CP_hi[P_absent(score > t)] over a fixed
                           threshold grid with Bonferroni; valid for any mechanism given independent canaries.
"""
import numpy as np
from scipy.stats import norm, chi2, beta as beta_dist
from scipy.stats import t as student_t
from corraudit import count_distribution, loglr_hmm, eps_from_delta_curve, _bonf_box, sample_paths
from vecgauss import eps_gauss_pair_vec


def scores_sum(X):
    return X.sum(axis=1) / np.sqrt(X.shape[1])


def gauss_sum_audit(X, present, delta, alpha=0.05, ngrid=5):
    S = scores_sum(X)
    a = alpha / 4
    (m0l, m0h), (s0l, s0h) = _bonf_box(S[~present], a)
    (m1l, m1h), (s1l, s1h) = _bonf_box(S[present], a)
    g = np.meshgrid(np.linspace(m0l, m0h, ngrid), np.linspace(s0l, s0h, ngrid),
                    np.linspace(m1l, m1h, ngrid), np.linspace(s1l, s1h, ngrid), indexing="ij")
    e = eps_gauss_pair_vec(g[0].ravel(), g[1].ravel(), g[2].ravel(), g[3].ravel(), delta)
    return float(e.min())


def _eps_mixture(means, w, s, mu0, s0, delta, grid=6001):
    keep = w > 1e-14
    means, w = means[keep], w[keep] / w[keep].sum()
    lo = min(means.min(), mu0) - 10 * max(s, s0); hi = max(means.max(), mu0) + 10 * max(s, s0)
    x = np.linspace(lo, hi, grid); dx = x[1] - x[0]
    p1 = (w[None, :] * norm.pdf(x[:, None], means[None, :], s)).sum(axis=1)
    p0 = norm.pdf(x, mu0, s0)
    f = lambda e: max(np.sum(np.clip(p1 - np.exp(e) * p0, 0, None)) * dx, np.sum(np.clip(p0 - np.exp(e) * p1, 0, None)) * dx)
    return eps_from_delta_curve(f, delta)


def estimate_shift(X, present, counts, alpha):
    """Per-participation shift beta (in sum-statistic units) from the present-vs-absent mean gap / mean count,
    with a t-interval; and (mu0, s0) box from the absent group."""
    S = scores_sum(X)
    Sp, Sa = S[present], S[~present]
    gap = Sp.mean() - Sa.mean()
    se = np.sqrt(Sp.var(ddof=1) / len(Sp) + Sa.var(ddof=1) / len(Sa))
    tq = student_t.ppf(1 - alpha / 2, min(len(Sp), len(Sa)) - 1)
    nbar = counts[present].mean()
    return gap / nbar, (gap - tq * se) / nbar, (gap + tq * se) / nbar


def count_cond_audit(X, present, counts, pn, delta, alpha=0.05, ngrid=3):
    """Known count law pn (auditor's process).  Box over (mu0, s0, beta), alpha/3 each."""
    a = alpha / 3
    S = scores_sum(X)
    (m0l, m0h), (s0l, s0h) = _bonf_box(S[~present], a)
    _, bl, bh = estimate_shift(X, present, counts, a)
    bl, bh = max(bl, 0.0), max(bh, 0.0)
    n = np.arange(len(pn))
    best = np.inf
    for mu0 in np.linspace(m0l, m0h, ngrid):
        for s0 in np.linspace(s0l, s0h, ngrid):
            for b in np.linspace(bl, bh, ngrid):
                best = min(best, _eps_mixture(mu0 + b * n, pn, s0, mu0, s0, delta))
    return best


class PathAwareTable:
    """eps_full as a function of a = C/sigma for a fixed process and T (Monte Carlo on the HMM LR)."""

    def __init__(self, P, pi, emit, T, a_grid, delta, n=100_000, seed=0, chunk=25_000):
        self.a_grid = np.array(a_grid); self.delta = delta
        rng = np.random.default_rng(seed)
        # sample once, reuse across a (LR depends on a through C X - C^2/2 with sigma=1 units: set sigma=1, C=a)
        lps, lqs = {a: [] for a in a_grid}, {a: [] for a in a_grid}
        done = 0
        while done < n:
            k = min(chunk, n - done)
            B = sample_paths(P, pi, emit, T, k, rng)
            Z1 = rng.standard_normal((k, T)); Z0 = rng.standard_normal((k, T))
            for a in a_grid:
                Xp = B * a + Z1; Xq = Z0
                lps[a].append(loglr_hmm(Xp, P, pi, emit, a, 1.0))
                lqs[a].append(loglr_hmm(Xq, P, pi, emit, a, 1.0))
            done += k
        self.eps = []
        for a in a_grid:
            lp = np.concatenate(lps[a]); lq = np.concatenate(lqs[a])
            f = lambda e: max(np.mean(np.clip(1 - np.exp(e - lp), 0, None)), np.mean(np.clip(1 - np.exp(e + lq), 0, None)))
            self.eps.append(eps_from_delta_curve(f, delta))
        self.eps = np.array(self.eps)

    def __call__(self, a):
        return float(np.interp(a, self.a_grid, self.eps, left=0.0))


def path_aware_audit(X, present, counts, T, table, alpha=0.05):
    """a = C/sigma: C from the shift (beta = C/sqrt(T)), sigma from the absent-score sd (sigma_sum = sigma);
    a_lo = C_lo / sigma_hi (Bonferroni alpha/2 each).  Reports eps_full(a_lo)."""
    a = alpha / 2
    S = scores_sum(X)
    _, bl, _ = estimate_shift(X, present, counts, a)
    C_lo = max(bl, 0.0) * np.sqrt(T)
    _, (s0l, s0h) = _bonf_box(S[~present], a)
    return table(C_lo / s0h)


def lr_threshold_audit(X, present, P, pi, emit, C_claim, sigma_claim, delta, alpha=0.05, thresholds=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)):
    """Nonparametric audit at the mechanism's own likelihood ratio (claimed parameters, known process)."""
    ell = loglr_hmm(X, P, pi, emit, C_claim, sigma_claim)
    ep, ea = ell[present], ell[~present]
    npres, nabs = len(ep), len(ea)
    a = alpha / (2 * len(thresholds))
    best = 0.0
    for t in thresholds:
        kp = int((ep > t).sum()); ka = int((ea > t).sum())
        p_lo = beta_dist.ppf(a, kp, npres - kp + 1) if kp > 0 else 0.0           # CP lower bound on P_present(A)
        q_hi = beta_dist.ppf(1 - a, ka + 1, nabs - ka) if ka < nabs else 1.0      # CP upper bound on P_absent(A)
        # largest eps with p_lo - e^eps q_hi > delta, also the reverse direction (A^c) for completeness
        if p_lo - delta > 0 and q_hi > 0:
            e1 = np.log((p_lo - delta) / q_hi)
        elif p_lo - delta > 0 and q_hi == 0:
            e1 = 40.0
        else:
            e1 = 0.0
        best = max(best, e1)
    return float(best)


def path_aware_audit_v2(X, present, B, T, table, alpha=0.05):
    """Path-regression version: the auditor knows each canary's participation indicators B (m x T).
    a_hat = sum_{present} sum_t B_it X_it / sum_{present} sum_t B_it  (unbiased for a under Model 1 in noise units
    once X is divided by the noise sd), with sd estimated from the absent group's per-round observations.
    a_lo = (a_hat - z sigma_hi/sqrt(n_obs)) / sigma_hi  in units where X = a B + sigma Z.  Bonferroni alpha/2."""
    a2 = alpha / 2
    Bp = B[present].astype(float); Xp = X[present]
    nobs = Bp.sum()
    shift = (Bp * Xp).sum() / nobs                        # estimate of the per-round shift (= a*sigma in X units)
    xa = X[~present].ravel()
    n0 = len(xa); sd = xa.std(ddof=1)
    sd_hi = sd * np.sqrt((n0 - 1) / chi2.ppf(a2 / 2, n0 - 1))
    z = norm.ppf(1 - a2 / 2)
    shift_lo = max(shift - z * sd_hi / np.sqrt(nobs), 0.0)
    return table(shift_lo / sd_hi)
