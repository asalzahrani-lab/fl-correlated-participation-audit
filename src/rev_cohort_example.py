"""Revision round 2, A1-A2: the two small examples of Section III-C and Section VI-E of the full version v8.

(1) Fixed-size cohort, one round, two eligible clients, K = 1 drawn uniformly, sigma = 2, C = 1 (a = C/sigma = 0.5),
    zero-out adjacency, target update +C, the other client's update -C, cohort hidden.
      actual transcript pair   P = .5 N(a,1) + .5 N(-a,1)   vs   Q = .5 N(0,1) + .5 N(-a,1)
      scalar observation model P0 = .5 N(a,1) + .5 N(0,1)   vs   Q0 = N(0,1)            (lower end of Proposition 2)
      cohort known to the adversary: Sum_n p_n h_eps(a sqrt n) = delta with p_1 = 1/2   (upper end, Theorem 3)
    The scalar model understates the transcript's eps; the cohort-known value overstates it (Proposition 2).
(2) Worst window is not the largest-expected-count window: two rounds with inclusion rates (1, 0) against (0.51, 0.51),
    a = 0.5, observation model (Poisson-binomial count, Theorem 1 on the per-round PLD = product of the two rounds).
(3) The lower end of Proposition 2 is not attained by orthogonal other-client updates in general: example (1) with the
    other client's update rotated orthogonal to u (norm C, 3C, 0).  The projection on u is the scalar pair, but the
    orthogonal coordinate reveals whether the other client was drawn, i.e. whether the target took a seat; only an
    uninformative other coordinate (update 0) attains the scalar value.
(4) Why participation must be exogenous to the training noise (reviewer example, round 4): two rounds, no other clients,
    B_1 = 0 and B_2 = 1{Z_1 > 0}, i.e. the server selects the target in round 2 from the sign of the first released
    output.  Transcript pair (Z_1, a 1{Z_1>0} + Z_2) against (Z_1, Z_2): both hockey-stick divergences equal (1/2) h_eps(a),
    the joint-release value, while the scalar model with the same marginal law (an independent hidden coin) gives the
    secret-path value of the mixture .5 N(a,1) + .5 N(0,1) against N(0,1).
Prints the eps at delta = 1e-4 and 1e-5 for the four examples.  Pure NumPy/SciPy; runs in a second."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

a = 0.5
DELTAS = [1e-4, 1e-5]


def hs_mixture(P, Q, eps, lo=-12.0, hi=12.0, n=400001):
    """Hockey-stick divergence H_eps(P||Q) = int (p - e^eps q)_+ for two Gaussian mixtures [(mu, w), ...]."""
    x = np.linspace(lo, hi, n)
    p = sum(w * norm.pdf(x, mu) for mu, w in P)
    q = sum(w * norm.pdf(x, mu) for mu, w in Q)
    return np.trapezoid(np.maximum(p - np.exp(eps) * q, 0.0), x)


def eps_of(P, Q, delta):
    f = lambda e: max(hs_mixture(P, Q, e), hs_mixture(Q, P, e)) - delta
    return brentq(f, 1e-6, 20.0, xtol=1e-9)


def h_gauss(mu, eps):
    """H_eps(N(mu,1) || N(0,1))."""
    return norm.cdf(mu / 2 - eps / mu) - np.exp(eps) * norm.cdf(-mu / 2 - eps / mu)


def eps_count_avg(pn, delta):
    """Sum_n p_n h_eps(a sqrt n) = delta (Theorem 3 / Proposition 3)."""
    f = lambda e: sum(p * (h_gauss(a * np.sqrt(n), e) if n > 0 else 0.0) for n, p in pn.items()) - delta
    return brentq(f, 1e-6, 20.0, xtol=1e-9)


if __name__ == "__main__":
    print("(1) fixed-size cohort, two clients, K = 1, a = 0.5")
    P = [(a, .5), (-a, .5)]; Q = [(0.0, .5), (-a, .5)]
    P0 = [(a, .5), (0.0, .5)]; Q0 = [(0.0, 1.0)]
    for d in DELTAS:
        e_act = eps_of(P, Q, d); e_sc = eps_of(P0, Q0, d); e_ck = eps_count_avg({0: .5, 1: .5}, d)
        print(f"  delta={d:g}: transcript (cohort hidden) {e_act:.3f}   scalar model {e_sc:.3f}   cohort known {e_ck:.3f}"
              f"   -> {e_sc:.3f} <= {e_act:.3f} <= {e_ck:.3f}: {e_sc <= e_act <= e_ck}")
    print("(2) worst window: rates (1, 0) against (0.51, 0.51), a = 0.5, observation model")
    for d in DELTAS:
        e_sure = eps_of([(a, 1.0)], [(0.0, 1.0)], d)                      # one sure round: N(a,1) vs N(0,1)
        # two rounds at rate r each: product of two one-round pairs (r N(a,1) + (1-r) N(0,1)) vs N(0,1)
        r = 0.51
        # exact two-round PLD by 2-D quadrature of the product distribution
        x = np.linspace(-10, 10, 2001); X, Y = np.meshgrid(x, x, indexing="ij")
        p1 = lambda z: r * norm.pdf(z, a) + (1 - r) * norm.pdf(z); q1 = norm.pdf
        Pxy = p1(X) * p1(Y); Qxy = q1(X) * q1(Y); dx = x[1] - x[0]
        hs2 = lambda e: max(np.sum(np.maximum(Pxy - np.exp(e) * Qxy, 0)) * dx * dx,
                            np.sum(np.maximum(Qxy - np.exp(e) * Pxy, 0)) * dx * dx)
        e_two = brentq(lambda e: hs2(e) - d, 1e-6, 20.0, xtol=1e-7)
        print(f"  delta={d:g}: (1,0) -> {e_sure:.3f} (expected count 1.00)   (0.51,0.51) -> {e_two:.3f} (expected count 1.02)")
    print("(3) example (1) with the other client's update orthogonal to u: two-coordinate transcript")
    x = np.linspace(-9, 9, 1801); X, Y = np.meshgrid(x, x, indexing="ij"); dx = x[1] - x[0]
    mix = lambda comps: sum(w * norm.pdf(X, mx) * norm.pdf(Y, my) for (mx, my), w in comps)
    for b in [a, 3 * a, 0.0]:
        Pxy = mix([((a, 0), .5), ((0, b), .5)]); Qxy = mix([((0, 0), .5), ((0, b), .5)])
        hs = lambda e: max(np.sum(np.maximum(Pxy - np.exp(e) * Qxy, 0)), np.sum(np.maximum(Qxy - np.exp(e) * Pxy, 0))) * dx * dx
        vals = [brentq(lambda e: hs(e) - d, 1e-6, 15.0, xtol=1e-7) for d in DELTAS]
        print(f"  other update norm {b / a:.0f}C orthogonal to u: eps = {vals[0]:.3f} (1e-4), {vals[1]:.3f} (1e-5)"
              f"{'   = scalar pair (attained)' if b == 0 else '   > scalar pair 1.091 / 1.354'}")
    print("(4) adaptive selection from the released noise: B_2 = 1{Z_1 > 0}")
    x = np.linspace(-9, 9, 1801); X, Y = np.meshgrid(x, x, indexing="ij"); dx = x[1] - x[0]
    Pxy = norm.pdf(X) * np.where(X > 0, norm.pdf(Y, a), norm.pdf(Y)); Qxy = norm.pdf(X) * norm.pdf(Y)
    hs = lambda e: max(np.sum(np.maximum(Pxy - np.exp(e) * Qxy, 0)), np.sum(np.maximum(Qxy - np.exp(e) * Pxy, 0))) * dx * dx
    for d in DELTAS:
        e_ad = brentq(lambda e: hs(e) - d, 1e-6, 15.0, xtol=1e-7); e_half = eps_count_avg({0: .5, 1: .5}, d)
        e_sc = eps_of([(a, .5), (0.0, .5)], [(0.0, 1.0)], d)
        print(f"  delta={d:g}: adaptive transcript {e_ad:.3f} (closed form (1/2) h(a): {e_half:.3f}); exogenous hidden coin (scalar model) {e_sc:.3f}")
