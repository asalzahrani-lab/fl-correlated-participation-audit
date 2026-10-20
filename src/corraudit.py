"""
Pilot: privacy auditing under temporally correlated canary participation.

Mechanism (scalar projection of the Gauss "Model 1"):
    round t = 1..T releases  X_t = B_t * C + sigma * Z_t,   Z_t ~ N(0,1) i.i.d.
    B_t in {0,1} is the canary's participation indicator.
    Present world: B follows a participation process (i.i.d. Bernoulli(q), two-state
    Markov chain with stationary q and autocorrelation lam, or a b-min-separation
    renewal process with stationary q).  Absent world: B == 0.

Quantities:
    eps_full_mc   : (eps, delta) of the full per-round release, Monte-Carlo on the
                    HMM likelihood ratio (adversary does not know the path).
    eps_sum_exact : (eps, delta) of the post-processed statistic S = sum_t X_t / sqrt(T),
                    exact (1-D integration over the count mixture).
    eps_poisson   : i.i.d. Bernoulli(q) accounting via dp_accounting PLD.
    eps_schema    : worst-case k participations (k-fold Gaussian composition, no amplification).
Auditors (one run, m canaries, each present with prob 1/2):
    gauss_audit            : fit N(mu0,s0) / N(mu1,s1) to absent/present scores, eps between the
                             fitted Gaussians minimised over a Bonferroni t/chi2 box  (Agrawal et al. 2026).
    count_conditioned_audit: present law = empirical count mixture of Gaussians (uses the canaries'
                             observed participation counts), eps minimised over a Bonferroni box.
    steinke_audit          : guess-counting bound of Steinke, Nasr, Jagielski (2023).
"""
import numpy as np
from scipy.stats import norm, t as student_t, chi2, binom
from scipy.optimize import brentq

# ----------------------------------------------------------------------------- processes

def two_state_chain(q, lam):
    """Stationary two-state chain with P(B_t=1)=q and lag-1 autocorrelation lam."""
    lo = -min(q / (1 - q), (1 - q) / q)
    assert lo <= lam < 1, f"lam must be in [{lo:.3f},1)"
    P = np.array([[1 - q * (1 - lam), q * (1 - lam)],
                  [(1 - q) * (1 - lam), q + (1 - q) * lam]])
    pi = np.array([1 - q, q])
    emit = np.array([0, 1])
    return P, pi, emit


def iid_chain(q):
    return two_state_chain(q, 0.0)


def minsep_chain(q, b):
    """Renewal process: after a participation the client is ineligible for b rounds, then
    participates with prob p each round.  Stationary rate q = p/(1+p b)."""
    p = q / (1 - q * b)
    assert p > 0, "q*b must be < 1"
    p = min(p, 1.0)  # p == 1: deterministic cyclic participation with period b+1
    S = b + 1  # states: 0 = eligible, j>0 = j rounds of refractory period left
    P = np.zeros((S, S))
    P[0, 0] = 1 - p
    P[0, b] = p if b > 0 else P[0, 0] + p  # participate -> go to state b (refractory)
    if b == 0:
        P[0, 0] = 1.0
    for j in range(1, S):
        P[j, j - 1] = 1.0
    # emission: participation happens on the transition into refractory state b from state 0.
    # We model it as the state entered: state b means "participated this round".
    # For b == 0 the chain is i.i.d. Bernoulli(p) -> handle separately.
    if b == 0:
        return iid_chain(q)
    emit = np.zeros(S, dtype=int)
    emit[b] = 1
    # stationary distribution
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi = pi / pi.sum()
    return P, pi, emit


def mixing_time(P, pi, eps=0.25):
    """Smallest t with max_s TV(P^t(s,.), pi) <= eps."""
    S = P.shape[0]
    Pt = np.eye(S)
    for t in range(1, 5000):
        Pt = Pt @ P
        tv = 0.5 * np.abs(Pt - pi[None, :]).sum(axis=1).max()
        if tv <= eps:
            return t
    return np.inf


def sample_paths(P, pi, emit, T, n, rng):
    """n sample paths of participation indicators (n x T) and the counts."""
    S = P.shape[0]
    cum = np.cumsum(P, axis=1)
    state = rng.choice(S, size=n, p=pi)
    B = np.empty((n, T), dtype=np.int8)
    for t in range(T):
        # The participation indicator at round t is emit[state_t]; start from stationary state_0
        # and transition before reading, so that stationary statistics hold for every t.
        u = rng.random(n)
        state = (u[:, None] > cum[state]).sum(axis=1)
        B[:, t] = emit[state]
    return B


def count_distribution(P, pi, emit, T):
    """Exact law of N_T = sum_t B_t (same convention as sample_paths)."""
    S = P.shape[0]
    # f[s, n] = P(state_t = s, N_t = n)
    f = np.zeros((S, T + 1))
    f[:, 0] = pi
    for t in range(T):
        g = np.zeros_like(f)
        # transition
        h = P.T @ f  # h[s', n] = sum_s P[s,s'] f[s,n]
        # emission shifts count for states with emit==1
        for s in range(S):
            if emit[s] == 1:
                g[s, 1:] = h[s, :-1]
            else:
                g[s, :] = h[s, :]
        f = g
    pn = f.sum(axis=0)
    pn = np.clip(pn, 0, None)
    return pn / pn.sum()

# ----------------------------------------------------------------------------- hockey-stick helpers

def eps_from_delta_curve(delta_of_eps, delta, lo=0.0, hi=60.0):
    """Invert a decreasing delta(eps) curve."""
    if delta_of_eps(lo) <= delta:
        return 0.0
    if delta_of_eps(hi) > delta:
        return hi
    return brentq(lambda e: delta_of_eps(e) - delta, lo, hi, xtol=1e-6)


def _gauss_hs(mu0, s0, mu1, s1, eps):
    """Hockey-stick H_eps(P1 || P0) for P0=N(mu0,s0^2), P1=N(mu1,s1^2), computed from the
    set {x : p1(x) > e^eps p0(x)} (quadratic in x)."""
    # log p1 - log p0 = a x^2 + b x + c
    a = -0.5 / s1**2 + 0.5 / s0**2
    b = mu1 / s1**2 - mu0 / s0**2
    c = -0.5 * mu1**2 / s1**2 + 0.5 * mu0**2 / s0**2 + np.log(s0 / s1)
    cc = c - eps
    if abs(a) < 1e-14:
        if abs(b) < 1e-14:
            return max(0.0, (np.exp(cc) - 1.0)) if cc > 0 else 0.0
        x0 = -cc / b
        if b > 0:
            A = (x0, np.inf)
        else:
            A = (-np.inf, x0)
        intervals = [A]
    else:
        disc = b * b - 4 * a * cc
        if disc <= 0:
            intervals = [(-np.inf, np.inf)] if a > 0 else []
        else:
            r1 = (-b - np.sqrt(disc)) / (2 * a)
            r2 = (-b + np.sqrt(disc)) / (2 * a)
            lo, hi = min(r1, r2), max(r1, r2)
            if a > 0:
                intervals = [(-np.inf, lo), (hi, np.inf)]
            else:
                intervals = [(lo, hi)]
    tot = 0.0
    for (l, h) in intervals:
        p1 = norm.cdf(h, mu1, s1) - norm.cdf(l, mu1, s1)
        p0 = norm.cdf(h, mu0, s0) - norm.cdf(l, mu0, s0)
        tot += p1 - np.exp(eps) * p0
    return max(tot, 0.0)


def eps_gauss_pair(mu0, s0, mu1, s1, delta):
    """eps(delta) between N(mu0,s0^2) and N(mu1,s1^2), both directions."""
    f = lambda e: max(_gauss_hs(mu0, s0, mu1, s1, e), _gauss_hs(mu1, s1, mu0, s0, e))
    return eps_from_delta_curve(f, delta)


def eps_schema(k, C, sigma, delta):
    """k Gaussian releases each with sensitivity C and noise sigma -> mu-GDP with mu = sqrt(k) C/sigma."""
    mu = np.sqrt(k) * C / sigma
    return eps_gauss_pair(0.0, 1.0, mu, 1.0, delta)

# ----------------------------------------------------------------------------- sum statistic (exact)

def sum_laws(pn, C, sigma, T):
    """Present law of S = sum X_t / sqrt(T): mixture_n pn N(C n / sqrt(T), sigma^2); absent N(0, sigma^2)."""
    n = np.arange(len(pn))
    means = C * n / np.sqrt(T)
    return means, pn


def _mix_pdf(x, means, w, s):
    return (w[None, :] * norm.pdf(x[:, None], means[None, :], s)).sum(axis=1)


def _mix_sf(x, means, w, s):
    return (w[None, :] * norm.sf(x[:, None], means[None, :], s)).sum(axis=1)


def _mix_cdf(x, means, w, s):
    return (w[None, :] * norm.cdf(x[:, None], means[None, :], s)).sum(axis=1)


def eps_sum_exact(pn, C, sigma, T, delta, grid=20001):
    """Exact eps(delta) for the sum statistic.  Present = count mixture (all shifts >= 0, so the
    likelihood ratio is monotone in s), absent = N(0, sigma^2)."""
    means, w = sum_laws(pn, C, sigma, T)
    keep = w > 1e-16
    means, w = means[keep], w[keep] / w[keep].sum()
    hi = max(means.max(), 0) + 12 * sigma
    lo = -12 * sigma
    x = np.linspace(lo, hi, grid)
    logp1 = np.log(np.clip(_mix_pdf(x, means, w, sigma), 1e-300, None))
    logp0 = norm.logpdf(x, 0, sigma)
    llr = logp1 - logp0  # increasing in x

    def delta_of_eps(e):
        # direction P1 vs P0: region llr > e is an upper tail x > x*
        idx = np.searchsorted(llr, e)
        if idx >= len(x):
            d1 = 0.0
        else:
            xs = x[idx]
            d1 = _mix_sf(np.array([xs]), means, w, sigma)[0] - np.exp(e) * norm.sf(xs, 0, sigma)
        # direction P0 vs P1: region llr < -e is a lower tail x < x**
        idx2 = np.searchsorted(llr, -e)
        if idx2 <= 0:
            d2 = 0.0
        else:
            xs2 = x[idx2 - 1]
            d2 = norm.cdf(xs2, 0, sigma) - np.exp(e) * _mix_cdf(np.array([xs2]), means, w, sigma)[0]
        return max(d1, d2, 0.0)

    return eps_from_delta_curve(delta_of_eps, delta)

# ----------------------------------------------------------------------------- full release (Monte Carlo on HMM LR)

def loglr_hmm(X, P, pi, emit, C, sigma):
    """log [ sum_paths P(path) prod_t exp(b_t (C x_t - C^2/2)/sigma^2) ]  for each row of X."""
    n, T = X.shape
    S = P.shape[0]
    la = np.log(np.clip(pi, 1e-300, None))[None, :].repeat(n, axis=0)  # log alpha (n x S)
    logP = np.log(np.clip(P, 1e-300, None))
    e_on = (C * X - C * C / 2) / sigma**2  # n x T, log emission factor when b=1
    emit_mask = (emit == 1)[None, :]
    for t in range(T):
        # transition in log space: la'[s'] = logsumexp_s (la[s] + logP[s,s'])
        m = la.max(axis=1, keepdims=True)
        la = m + np.log(np.exp(la - m) @ np.exp(logP))
        la = la + np.where(emit_mask, e_on[:, t][:, None], 0.0)
    m = la.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(la - m).sum(axis=1, keepdims=True)))[:, 0]


def eps_full_mc(P, pi, emit, T, C, sigma, delta, n=400_000, rng=None, return_curve=False, chunk=50_000):
    """eps(delta) of the full release via Monte Carlo on the likelihood ratio (chunked to bound memory).
    delta(eps) = max( E_P[(1 - e^eps / LR)_+],  E_Q[(1 - e^eps LR)_+] )."""
    rng = np.random.default_rng(0) if rng is None else rng
    lps, lqs = [], []
    done = 0
    while done < n:
        k = min(chunk, n - done)
        B = sample_paths(P, pi, emit, T, k, rng)
        Xp = B * C + sigma * rng.standard_normal((k, T))
        lps.append(loglr_hmm(Xp, P, pi, emit, C, sigma))
        del Xp, B
        Xq = sigma * rng.standard_normal((k, T))
        lqs.append(loglr_hmm(Xq, P, pi, emit, C, sigma))
        del Xq
        done += k
    lp = np.concatenate(lps); lq = np.concatenate(lqs)

    def delta_of_eps(e):
        d1 = np.mean(np.clip(1.0 - np.exp(e - lp), 0, None))
        d2 = np.mean(np.clip(1.0 - np.exp(e + lq), 0, None))
        return max(d1, d2)

    e = eps_from_delta_curve(delta_of_eps, delta)
    if return_curve:
        return e, lp, lq
    return e


def eps_poisson_pld(q, T, C, sigma, delta):
    from dp_accounting import pld
    from dp_accounting.pld import privacy_loss_distribution as pldlib
    acc = pld.PLDAccountant()
    from dp_accounting import dp_event
    ev = dp_event.SelfComposedDpEvent(
        dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma / C)), T)
    acc.compose(ev)
    return acc.get_epsilon(delta)

# ----------------------------------------------------------------------------- one-run auditors

def simulate_scores(P, pi, emit, T, C, sigma, m, rng):
    """m canaries; each present with prob 1/2.  Returns scores S_i, present flags, counts n_i."""
    present = rng.random(m) < 0.5
    B = sample_paths(P, pi, emit, T, m, rng)
    counts = B.sum(axis=1)
    Bp = B * present[:, None]
    X = Bp * C + sigma * rng.standard_normal((m, T))
    S = X.sum(axis=1) / np.sqrt(T)
    return S, present, counts


def _bonf_box(x, alpha):
    """(mean interval, sd interval) at level alpha (each)."""
    n = len(x)
    mu, sd = x.mean(), x.std(ddof=1)
    tq = student_t.ppf(1 - alpha / 2, n - 1)
    mu_lo, mu_hi = mu - tq * sd / np.sqrt(n), mu + tq * sd / np.sqrt(n)
    sd_lo = sd * np.sqrt((n - 1) / chi2.ppf(1 - alpha / 2, n - 1))
    sd_hi = sd * np.sqrt((n - 1) / chi2.ppf(alpha / 2, n - 1))
    return (mu_lo, mu_hi), (sd_lo, sd_hi)


def gauss_audit(S, present, delta, alpha=0.05, ngrid=5):
    """Agrawal et al.-style: Gaussian fits to absent/present scores; eps minimised over a
    Bonferroni box (4 parameters, alpha/4 each)."""
    a = alpha / 4
    (m0l, m0h), (s0l, s0h) = _bonf_box(S[~present], a)
    (m1l, m1h), (s1l, s1h) = _bonf_box(S[present], a)
    best = np.inf
    for mu0 in np.linspace(m0l, m0h, ngrid):
        for s0 in np.linspace(s0l, s0h, ngrid):
            for mu1 in np.linspace(m1l, m1h, ngrid):
                for s1 in np.linspace(s1l, s1h, ngrid):
                    best = min(best, eps_gauss_pair(mu0, s0, mu1, s1, delta))
    point = eps_gauss_pair(S[~present].mean(), S[~present].std(ddof=1),
                           S[present].mean(), S[present].std(ddof=1), delta)
    return best, point


def count_conditioned_audit(S, present, counts, T, delta, alpha=0.05, ngrid=5):
    """Present law modelled as the empirical count mixture sum_n p_hat_n N(mu0 + beta n, s0^2),
    with (mu0, s0) from the absent group and beta from regressing present scores on counts.
    eps minimised over a Bonferroni box on (mu0, s0, beta) (alpha/3 each)."""
    a = alpha / 3
    (m0l, m0h), (s0l, s0h) = _bonf_box(S[~present], a)
    Sp, nn = S[present], counts[present].astype(float)
    # regression S = mu + beta n  (mu free): use absent mean as intercept proxy is biased; fit both
    A = np.vstack([np.ones_like(nn), nn]).T
    coef, res, _, _ = np.linalg.lstsq(A, Sp, rcond=None)
    beta = coef[1]
    resid = Sp - A @ coef
    s_res = resid.std(ddof=2)
    sxx = ((nn - nn.mean()) ** 2).sum()
    se_beta = s_res / np.sqrt(sxx) if sxx > 0 else np.inf
    tq = student_t.ppf(1 - a / 2, len(Sp) - 2)
    bl, bh = beta - tq * se_beta, beta + tq * se_beta
    # empirical count law from all canaries' paths (auditor drew every path)
    nmax = counts.max()
    pn = np.bincount(counts, minlength=nmax + 1) / len(counts)
    n = np.arange(nmax + 1)
    best = np.inf
    for mu0 in np.linspace(m0l, m0h, ngrid):
        for s0 in np.linspace(s0l, s0h, ngrid):
            for b in np.linspace(max(bl, 0.0), max(bh, 0.0), ngrid):
                means = mu0 + b * n
                e = _eps_mixture_vs_gauss(means, pn, s0, mu0, s0, delta)
                best = min(best, e)
    return best, beta


def _eps_mixture_vs_gauss(means, w, s, mu0, s0, delta, grid=8001):
    """eps(delta) between mixture_n w_n N(means_n, s^2) (present) and N(mu0, s0^2) (absent);
    generic 1-D integration (no monotonicity assumption)."""
    keep = w > 1e-14
    means, w = means[keep], w[keep] / w[keep].sum()
    lo = min(means.min(), mu0) - 12 * max(s, s0)
    hi = max(means.max(), mu0) + 12 * max(s, s0)
    x = np.linspace(lo, hi, grid)
    dx = x[1] - x[0]
    p1 = _mix_pdf(x, means, w, s)
    p0 = norm.pdf(x, mu0, s0)

    def delta_of_eps(e):
        d1 = np.sum(np.clip(p1 - np.exp(e) * p0, 0, None)) * dx
        d2 = np.sum(np.clip(p0 - np.exp(e) * p1, 0, None)) * dx
        return max(d1, d2)

    return eps_from_delta_curve(delta_of_eps, delta)


def steinke_audit(S, present, delta, alpha=0.05, k=None):
    """Steinke-Nasr-Jagielski one-run bound, Corollary 5.4 (ternary guesses): guess the top-k scores
    'present' and the bottom-k 'absent' (r = 2k guesses, v correct).  For (eps,delta)-DP,
        P[W >= v] <= f(v) + 2 m delta * max_i (f(v-i) - f(v))/i,   f(v) = P[Bin(r, e^eps/(1+e^eps)) >= v].
    Returns the largest eps whose bound is still <= alpha."""
    m = len(S)
    order = np.argsort(S)
    if k is None:
        k = max(1, m // 10)
    top = order[-k:]
    bot = order[:k]
    v = int(present[top].sum() + (~present[bot]).sum())
    r = 2 * k
    ii = np.arange(1, m + 1)

    def pval(e):
        p = np.exp(e) / (1 + np.exp(e))
        f = lambda x: binom.sf(x - 1, r, p)  # P[Bin >= x]
        fv = f(v)
        fvi = np.where(v - ii <= 0, 1.0, f(np.maximum(v - ii, 0)))
        return fv + 2 * m * delta * np.max((fvi - fv) / ii)

    if pval(0.0) > alpha:
        return 0.0
    hi = 40.0
    if pval(hi) <= alpha:
        return hi
    return brentq(lambda e: pval(e) - alpha, 0.0, hi, xtol=1e-5)
