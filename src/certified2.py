"""Merged certified path-aware auditor (step 14).

Same estimator as path-aware v2 (a_lo from the path regression), but the map a_lo -> eps is certified with
three improvements over certified.py:
  (1) the hockey-stick divergence delta(eps) = E_P[(1 - e^eps / LR)_+]  (and the reverse direction) is lower-bounded
      DIRECTLY by a confidence bound on the mean of a [0,1]-valued variable.  The DEFAULT (used for every certified
      number in the paper) is the Chernoff-Hoeffding bound in its Kullback-Leibler form (kl_lo; Hoeffding 1963, Thm 1):
      mu_lo = min{mu : n kl(m, mu) <= ln(1/alpha)}.  An empirical-Bernstein variant (bernstein_lo; Maurer & Pontil
      2009, Thm 4, with the sample variance replaced by its upper bound n/(n-1) m) is kept as an option.  Both are
      monotone in eps, so eps_cert = sup{eps : delta_lo(eps) > delta} is well defined;
  (2) a fine grid (step 0.01) with floor interpolation (eps_full(a) is nondecreasing in a);
  (3) alpha split 0.045 (estimation of a_lo) + 0.005 (Monte Carlo, 0.0025 per direction).
No Bonferroni over grid points is needed: the grid entry is selected by a_lo, which is independent of the Monte-Carlo
samples, so P[failure] <= alpha_mc whichever entry is used.

Shared samples across a: one draw of (B, Z_P, Z_Q) per chunk serves every a (X_P = aB + Z_P, X_Q = Z_Q).  For the
empirical path law the two matmuls B_s B^T and Z B^T are computed once per chunk and reused for every a.
"""
import numpy as np
from scipy.stats import norm, chi2
from corraudit import sample_paths, loglr_hmm
from certified import EmpiricalPathLaw, ChainLaw, delta_lo_curve, eps_from_lower_curve, plain_eps


def bernstein_lo(m, n, alpha):
    """Lower confidence bound on E[X] for X in [0,1] from sample mean m of n i.i.d. samples, using the empirical
    Bernstein inequality with the variance proxy V' = n/(n-1) m >= sample variance."""
    L = np.log(2.0 / alpha)
    V = m * n / (n - 1)
    return m - np.sqrt(2 * V * L / n) - 7 * L / (3 * (n - 1))


def _kl(p, q):
    p = min(max(p, 1e-300), 1 - 1e-16); q = min(max(q, 1e-300), 1 - 1e-16)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def kl_lo(m, n, alpha, iters=60):
    """Lower confidence bound on E[X] for X in [0,1] from the sample mean m of n i.i.d. samples, via the
    Chernoff-Hoeffding bound P(mean >= m) <= exp(-n kl(m, mu)) for mu < m (Hoeffding 1963, Thm 1, KL form), which
    holds for all [0,1]-valued variables because E[e^{sX}] <= 1 - mu + mu e^s.  Returns min{mu : n kl(m, mu) <= ln(1/alpha)}."""
    if m <= 0.0:
        return 0.0
    L = np.log(1.0 / alpha) / n
    lo, hi = 0.0, m
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _kl(m, mid) > L: lo = mid
        else: hi = mid
    return hi


def delta_lo_bern(lp, lq, eps, alpha, bound=kl_lo):
    """Rigorous lower bound (confidence 1 - alpha, Bonferroni over the two directions) on
    max(H_eps(P||Q), H_eps(Q||P)) from samples lp = log LR under P, lq = log LR under Q.
    bound = kl_lo (default, Chernoff-Hoeffding KL form) or bernstein_lo."""
    a = alpha / 2
    mp = float(np.mean(np.clip(1.0 - np.exp(eps - lp), 0.0, 1.0)))
    mq = float(np.mean(np.clip(1.0 - np.exp(eps + lq), 0.0, 1.0)))
    return max(bound(mp, len(lp), a), bound(mq, len(lq), a), 0.0)


def eps_cert_bern(lp, lq, delta, alpha, hi=60.0, iters=50):
    f = lambda e: delta_lo_bern(lp, lq, e, alpha)
    if f(0.0) <= delta:
        return 0.0
    lo, h = 0.0, hi
    if f(h) > delta:
        return h
    for _ in range(iters):
        mid = 0.5 * (lo + h)
        if f(mid) > delta: lo = mid
        else: h = mid
    return lo


class CertifiedTable2:
    """eps_cert_bern(a), eps_cert_step(a) (old method) and eps_plain(a) on a grid, from shared samples."""

    def __init__(self, law, a_grid, delta, n_mc, alpha_mc, rng, chunk=4000, log=None, keep_samples=False):
        self.a_grid = np.array(a_grid, float); self.delta = delta; self.n_mc = n_mc; self.alpha_mc = alpha_mc
        G = len(self.a_grid)
        lps = np.empty((G, n_mc), np.float32); lqs = np.empty((G, n_mc), np.float32)
        done = 0
        is_emp = isinstance(law, EmpiricalPathLaw)
        if is_emp:
            BT = law.B.T                                   # T x M
            const0 = law.logw.astype(np.float32)           # M
            npath = law.n_path.astype(np.float32)          # M
        while done < n_mc:
            k = min(chunk, n_mc - done)
            B = law.sample(k, rng)                          # k x T int8
            Zp = rng.standard_normal((k, law.T)).astype(np.float32)
            Zq = rng.standard_normal((k, law.T)).astype(np.float32)
            if is_emp:
                Gp = B.astype(np.float32) @ BT             # k x M  (<B_s, B_path>)
                ZpB = Zp @ BT; ZqB = Zq @ BT               # k x M
                for gi, a in enumerate(self.a_grid):
                    const = const0 - 0.5 * a * a * npath
                    Zm = a * (a * Gp + ZpB) + const[None, :]
                    m = Zm.max(axis=1, keepdims=True)
                    lps[gi, done:done + k] = (m + np.log(np.exp(Zm - m).sum(axis=1, keepdims=True)))[:, 0]
                    Zm = a * ZqB + const[None, :]
                    m = Zm.max(axis=1, keepdims=True)
                    lqs[gi, done:done + k] = (m + np.log(np.exp(Zm - m).sum(axis=1, keepdims=True)))[:, 0]
            else:
                Bf = B.astype(np.float64); Zpd = Zp.astype(np.float64); Zqd = Zq.astype(np.float64)
                for gi, a in enumerate(self.a_grid):
                    lps[gi, done:done + k] = loglr_hmm(Bf * a + Zpd, law.P, law.pi, law.emit, a, 1.0)
                    lqs[gi, done:done + k] = loglr_hmm(Zqd, law.P, law.pi, law.emit, a, 1.0)
            done += k
            if log and (done // chunk) % 25 == 0:
                log(f"  samples {done}/{n_mc}")
        self.eps_bern, self.eps_step, self.eps_plain = [], [], []
        for gi, a in enumerate(self.a_grid):
            lp = lps[gi].astype(np.float64); lq = lqs[gi].astype(np.float64)
            eb = eps_cert_bern(lp, lq, delta, alpha_mc)
            es = eps_from_lower_curve(delta_lo_curve(lp, lq, alpha_mc), delta)
            ep = plain_eps(lp, lq, delta)
            self.eps_bern.append(eb); self.eps_step.append(es); self.eps_plain.append(ep)
            if log: log(f"  a={a:.2f}: plain={ep:.3f} cert_step={es:.3f} cert_bern={eb:.3f}")
        self.eps_bern = np.array(self.eps_bern); self.eps_step = np.array(self.eps_step); self.eps_plain = np.array(self.eps_plain)
        if keep_samples: self.lps, self.lqs = lps, lqs

    def _floor(self, a):
        if not np.isfinite(a):                       # a non-finite estimate reads as "below the grid": certificate 0
            return -1
        idx = np.searchsorted(self.a_grid - 1e-9, a, side="right") - 1
        return idx

    def cert(self, a):
        i = self._floor(a); return 0.0 if i < 0 else float(self.eps_bern[i])

    def cert_step(self, a):
        i = self._floor(a); return 0.0 if i < 0 else float(self.eps_step[i])

    def plain(self, a):
        return float(np.interp(a, self.a_grid, self.eps_plain, left=0.0))


def snr_lower_bound(X, present, B, alpha):
    """Lower confidence limit on a = shift / noise sd at level 1 - alpha: two one-sided limits (normal for the
    regression shift, chi-square for the absent-group sd), each at level alpha/2 (Bonferroni).
    Returns 0 (the paper's convention, Section V-C) when there is no participating present-canary observation
    (n_obs = 0) or fewer than two absent-canary observations (n_0 < 2); the checks come before any division, so
    no NaN can reach the table lookup (round 12)."""
    present = np.asarray(present, dtype=bool)
    Bp = B[present].astype(float); Xp = X[present].astype(float)
    nobs = Bp.sum()
    xa = X[~present].astype(float).ravel(); n0 = len(xa)
    if nobs <= 0 or n0 < 2:
        return 0.0
    shift = (Bp * Xp).sum() / nobs
    sd = xa.std(ddof=1)
    sd_hi = sd * np.sqrt((n0 - 1) / chi2.ppf(alpha / 2, n0 - 1))     # one-sided upper limit on the noise sd
    z = norm.ppf(1 - alpha / 2)                                         # one-sided lower limit on the shift
    if not np.isfinite(sd_hi) or sd_hi <= 0:
        return 0.0
    shift_lo = max(shift - z * sd_hi / np.sqrt(nobs), 0.0)
    a_lo = shift_lo / sd_hi
    return float(a_lo) if np.isfinite(a_lo) else 0.0
