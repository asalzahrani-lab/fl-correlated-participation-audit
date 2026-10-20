"""Certified path-aware auditing: (i) participation law given as an EMPIRICAL set of paths (e.g. from the FLASH
trace under a cohort policy) or as a finite-state chain; (ii) the Monte-Carlo error of the eps_full(a) table is folded
into delta through a rigorous lower confidence bound on the hockey-stick divergence.

Lower bound used:  delta(eps) = E_P[(1 - e^{eps}/LR)_+] >= (1 - e^{-c}) P_P(log LR > eps + c)  for every c > 0,
and P_P(.) is a binomial proportion -> Clopper-Pearson lower bound.  Max over a small grid of c with Bonferroni.
Same for the reverse direction with samples from Q and LR^{-1}.  eps_cert = sup{eps : delta_lo(eps) > delta}.
"""
import numpy as np
from scipy.stats import beta as beta_dist, norm, chi2
from corraudit import sample_paths, loglr_hmm


class EmpiricalPathLaw:
    """Uniform (or weighted) mixture over a finite set of observed participation paths (M x T, 0/1)."""

    def __init__(self, paths, weights=None):
        self.B = np.asarray(paths, dtype=np.float32)
        self.M, self.T = self.B.shape
        self.w = np.full(self.M, 1.0 / self.M) if weights is None else np.asarray(weights, float) / np.sum(weights)
        self.logw = np.log(np.clip(self.w, 1e-300, None)).astype(np.float32)
        self.n_path = self.B.sum(axis=1)

    def sample(self, n, rng):
        idx = rng.choice(self.M, size=n, p=self.w)
        return self.B[idx].astype(np.int8)

    def loglr(self, X, a, chunk=4000):
        """log sum_paths w exp(a <B_path, X> - a^2 n_path / 2)  (noise sd = 1 units)."""
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float64)
        const = (self.logw - 0.5 * a * a * self.n_path).astype(np.float32)   # M
        for s in range(0, len(X), chunk):
            S = X[s:s + chunk] @ self.B.T                                       # n x M
            Z = a * S + const[None, :]
            m = Z.max(axis=1, keepdims=True)
            out[s:s + chunk] = (m + np.log(np.exp(Z - m).sum(axis=1, keepdims=True)))[:, 0]
        return out


class ChainLaw:
    def __init__(self, P, pi, emit, T):
        self.P, self.pi, self.emit, self.T = P, pi, emit, T

    def sample(self, n, rng):
        return sample_paths(self.P, self.pi, self.emit, self.T, n, rng)

    def loglr(self, X, a):
        return loglr_hmm(np.asarray(X, float), self.P, self.pi, self.emit, a, 1.0)


def delta_lo_curve(lp, lq, alpha, cs=(0.25, 0.5, 1.0, 2.0)):
    """Return a function eps -> rigorous lower bound on max(H_eps(P||Q), H_eps(Q||P)) at confidence 1-alpha,
    from samples lp = log LR under P (present) and lq = log LR under Q (absent)."""
    n_p, n_q = len(lp), len(lq)
    a = alpha / (2 * len(cs))   # two directions x |cs| thresholds

    def cp_lo(k, n):
        return beta_dist.ppf(a, k, n - k + 1) if k > 0 else 0.0

    def f(eps):
        best = 0.0
        for c in cs:
            kp = int(np.sum(lp > eps + c)); best = max(best, (1 - np.exp(-c)) * cp_lo(kp, n_p))
            kq = int(np.sum(-lq > eps + c)); best = max(best, (1 - np.exp(-c)) * cp_lo(kq, n_q))
        return best
    return f


def eps_from_lower_curve(f, delta, hi=60.0, iters=50):
    """sup{eps : f(eps) > delta} (f nonincreasing)."""
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


def plain_eps(lp, lq, delta, hi=60.0):
    def d(e):
        return max(np.mean(np.clip(1 - np.exp(e - lp), 0, None)), np.mean(np.clip(1 - np.exp(e + lq), 0, None)))
    if d(0.0) <= delta: return 0.0
    lo, h = 0.0, hi
    for _ in range(50):
        mid = 0.5 * (lo + h)
        if d(mid) > delta: lo = mid
        else: h = mid
    return lo


class CertifiedTable:
    """eps_cert(a) and eps_plain(a) on a grid of a; conservative interpolation (value at the grid point <= a)."""

    def __init__(self, law, a_grid, delta, n_mc, alpha_mc, rng, chunk=25_000, log=None):
        self.a_grid = np.array(a_grid); self.delta = delta
        self.eps_cert, self.eps_plain = [], []
        for a in self.a_grid:
            lps, lqs = [], []
            done = 0
            while done < n_mc:
                k = min(chunk, n_mc - done)
                B = law.sample(k, rng)
                Xp = B * a + rng.standard_normal((k, law.T)).astype(np.float32)
                Xq = rng.standard_normal((k, law.T)).astype(np.float32)
                lps.append(law.loglr(Xp, a)); lqs.append(law.loglr(Xq, a)); done += k
            lp, lq = np.concatenate(lps), np.concatenate(lqs)
            f = delta_lo_curve(lp, lq, alpha_mc)
            ec = eps_from_lower_curve(f, delta); ep = plain_eps(lp, lq, delta)
            self.eps_cert.append(ec); self.eps_plain.append(ep)
            if log: log(f"  a={a:.3f}: eps_plain={ep:.3f} eps_cert={ec:.3f}")
        self.eps_cert = np.array(self.eps_cert); self.eps_plain = np.array(self.eps_plain)

    def cert(self, a):
        idx = np.searchsorted(self.a_grid, a, side="right") - 1
        return 0.0 if idx < 0 else float(self.eps_cert[idx])

    def plain(self, a):
        return float(np.interp(a, self.a_grid, self.eps_plain, left=0.0))


def snr_lower_bound(X, present, B, alpha):
    """a_lo from the path regression (shift) and the absent-group sd, Bonferroni alpha/2 each."""
    a2 = alpha / 2
    Bp = B[present].astype(float); Xp = X[present].astype(float)
    nobs = Bp.sum(); shift = (Bp * Xp).sum() / nobs
    xa = X[~present].astype(float).ravel(); n0 = len(xa); sd = xa.std(ddof=1)
    sd_hi = sd * np.sqrt((n0 - 1) / chi2.ppf(a2 / 2, n0 - 1))
    z = norm.ppf(1 - a2 / 2)
    shift_lo = max(shift - z * sd_hi / np.sqrt(nobs), 0.0)
    return shift_lo / sd_hi


def certified_audit(X, present, B, table, alpha_est):
    return table.cert(snr_lower_bound(X, present, B, alpha_est))
