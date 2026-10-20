"""Vectorised eps(delta) between N(mu0,s0^2) and N(mu1,s1^2) over arrays of parameters.

Release note: upper tails are evaluated as ndtr(-z) instead of 1 - ndtr(z).  This changes nothing for eps below about 12
(every Gaussian-fit and count-conditioned audit value in the paper is below 5) and removes a cancellation error at
large eps (e.g. the no-subsampling pair with shift 0.5*sqrt(200): 50.50, as the PLD accountant gives, not 51.30)."""
import numpy as np
from scipy.special import ndtr


def _hs_vec(mu0, s0, mu1, s1, eps):
    """H_eps(P1||P0) for arrays (broadcast) of parameters and eps."""
    a = -0.5 / s1**2 + 0.5 / s0**2
    b = mu1 / s1**2 - mu0 / s0**2
    c = -0.5 * mu1**2 / s1**2 + 0.5 * mu0**2 / s0**2 + np.log(s0 / s1) - eps
    out = np.zeros(np.broadcast(a, b, c).shape)
    a, b, c = np.broadcast_arrays(a, b, c)
    mu0, s0, mu1, s1 = np.broadcast_arrays(mu0, s0, mu1, s1)
    mu0 = np.broadcast_to(mu0, a.shape); s0 = np.broadcast_to(s0, a.shape)
    mu1 = np.broadcast_to(mu1, a.shape); s1 = np.broadcast_to(s1, a.shape)
    eps_b = np.broadcast_to(eps, a.shape)
    lin = np.abs(a) < 1e-14
    # --- linear case (equal variances): region x > x0 if b>0 else x < x0
    if lin.any():
        bb = np.where(lin, b, 1.0)
        x0 = -c / bb
        up = bb > 0
        p1 = np.where(up, ndtr(-(x0 - mu1) / s1), ndtr((x0 - mu1) / s1))
        p0 = np.where(up, ndtr(-(x0 - mu0) / s0), ndtr((x0 - mu0) / s0))
        out = np.where(lin, p1 - np.exp(eps_b) * p0, out)
    # --- quadratic case
    quad = ~lin
    aa = np.where(quad, a, 1.0)
    disc = b * b - 4 * aa * c
    pos = disc > 0
    sq = np.sqrt(np.clip(disc, 0, None))
    r1 = (-b - sq) / (2 * aa)
    r2 = (-b + sq) / (2 * aa)
    lo = np.minimum(r1, r2); hi = np.maximum(r1, r2)
    # a > 0: region = (-inf, lo) U (hi, inf);  a < 0: region = (lo, hi)
    cdf1_lo = ndtr((lo - mu1) / s1); cdf1_hi = ndtr((hi - mu1) / s1)
    cdf0_lo = ndtr((lo - mu0) / s0); cdf0_hi = ndtr((hi - mu0) / s0)
    outer1 = cdf1_lo + ndtr(-(hi - mu1) / s1); outer0 = cdf0_lo + ndtr(-(hi - mu0) / s0)
    inner1 = cdf1_hi - cdf1_lo;       inner0 = cdf0_hi - cdf0_lo
    val_pos = np.where(aa > 0, outer1 - np.exp(eps_b) * outer0, inner1 - np.exp(eps_b) * inner0)
    # disc <= 0: a>0 -> whole line (value 1 - e^eps < 0 -> 0); a<0 -> empty -> 0
    val = np.where(pos, val_pos, 0.0)
    out = np.where(quad, val, out)
    return np.clip(out, 0.0, None)


def eps_gauss_pair_vec(mu0, s0, mu1, s1, delta, hi=60.0, iters=60):
    """Bisection for eps(delta) elementwise; returns array."""
    mu0, s0, mu1, s1 = np.broadcast_arrays(np.asarray(mu0, float), np.asarray(s0, float),
                                           np.asarray(mu1, float), np.asarray(s1, float))
    lo_e = np.zeros(mu0.shape); hi_e = np.full(mu0.shape, hi)

    def d(e):
        return np.maximum(_hs_vec(mu0, s0, mu1, s1, e), _hs_vec(mu1, s1, mu0, s0, e))

    at0 = d(lo_e) <= delta
    for _ in range(iters):
        mid = 0.5 * (lo_e + hi_e)
        ok = d(mid) <= delta  # eps large enough
        hi_e = np.where(ok, mid, hi_e)
        lo_e = np.where(ok, lo_e, mid)
    e = hi_e
    return np.where(at0, 0.0, e)
