"""Shared helpers for the revision (after the two pre-submission reviews).

Trace loading, analysis windows, selection policies and sliding windows are the same as in step21b_consistent.py
(copied here so that the revision scripts do not execute step21b on import).  Also: two-sided Chernoff-Hoeffding
(KL form) confidence limits on the hockey-stick curve from Monte-Carlo samples of log LR, and their inversion into a
confidence interval for eps(delta)."""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.environ.get("P3_TRACE", os.path.join(HERE, "..", "trace"))
sys.path.insert(0, TRACE_DIR)

_t0 = time.time()
def log(s):
    print(f"{s} [{time.time()-_t0:.0f}s]", flush=True)

_DEVS = None
def devices():
    """Devices with at least two days of trace: list of (guid, availability intervals, first ts, last ts)."""
    global _DEVS
    if _DEVS is None:
        from trace_analysis import ready_intervals
        d = json.load(open(os.path.join(TRACE_DIR, "state_traces.json")))
        out = []
        for k, v in d.items():
            iv, tf, tl = ready_intervals(v["messages"])
            if tf is None or tl - tf < 2 * 86400:
                continue
            out.append((k, iv, tf, tl))
        _DEVS = out
    return _DEVS


def grid(round_min):
    """Availability and coverage matrices on the global round grid (all devices x all rounds)."""
    from trace_analysis import availability_grid
    devs = devices(); step = round_min * 60
    t0_ = min(tf for (_, _, tf, _) in devs); t1_ = max(tl for (_, _, _, tl) in devs)
    times = np.arange(t0_, t1_, step)
    Amat = np.zeros((len(devs), len(times)), dtype=bool); cover = np.zeros_like(Amat)
    for i, (k, iv, tf, tl) in enumerate(devs):
        Amat[i] = availability_grid(iv, t0_, t1_, step); cover[i] = (times >= tf) & (times <= tl)
    return Amat, cover, times


def window_candidates(round_min, win_days, stride=36):
    Amat, cover, times = grid(round_min)
    W = int(win_days * 24 * 60 / round_min)
    cands = list(range(0, len(times) - W, stride))
    scores = np.array([(cover[:, s0:s0 + W].all(axis=1)).sum() for s0 in cands])
    return Amat, cover, times, W, cands, scores


def window(round_min, win_days, start=None, verbose=True):
    """The analysis window of the paper (start=None: maximal number of fully covered devices) or a given start."""
    Amat, cover, times, W, cands, scores = window_candidates(round_min, win_days)
    best = cands[int(np.argmax(scores))] if start is None else start
    sel = cover[:, best:best + W].all(axis=1)
    if verbose:
        log(f"{round_min}-min/{win_days}-day window start {best}: {sel.sum()} devices")
    return Amat[sel, best:best + W], np.flatnonzero(sel), best


def apply_policy(A, kind, par, b, rng):
    """Same as step21b: thinning at rate par, or fixed cohort of round(par * mean pool); timer b (ineligible for b
    rounds after a participation)."""
    N, Tw = A.shape; B = np.zeros_like(A); last = np.full(N, -10**9); Epool = A.sum(axis=0).mean()
    for t in range(Tw):
        idx = np.flatnonzero(A[:, t] & (t - last > b))
        if kind == "thin":
            chosen = idx[rng.random(len(idx)) < par]
        else:
            K = int(round(par * Epool)); chosen = idx if len(idx) <= K else rng.choice(idx, K, replace=False)
        B[chosen, t] = True; last[chosen] = t
    return B


def windows(B, T, stride):
    N, Tw = B.shape
    dev = np.repeat(np.arange(N), (Tw - T) // stride + 1)
    counts = np.array([B[i, s0:s0 + T].sum() for i in range(N) for s0 in range(0, Tw - T + 1, stride)])
    return counts, dev


def window_paths(B, T, stride):
    N, Tw = B.shape
    return np.array([B[i, s0:s0 + T] for i in range(N) for s0 in range(0, Tw - T + 1, stride)], dtype=np.int8)


# ---------------------------------------------------------------- two-sided KL confidence limits
def _kl(p, q):
    p = min(max(p, 1e-300), 1 - 1e-16); q = min(max(q, 1e-300), 1 - 1e-16)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def kl_lo(m, n, alpha, iters=80):
    if m <= 0.0:
        return 0.0
    L = np.log(1.0 / alpha) / n; lo, hi = 0.0, m
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _kl(m, mid) > L: lo = mid
        else: hi = mid
    return hi


def kl_hi(m, n, alpha, iters=80):
    L = np.log(1.0 / alpha) / n; lo, hi = m, 1.0
    if m >= 1.0:
        return 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _kl(m, mid) > L: hi = mid
        else: lo = mid
    return lo


def hs_means(lp, lq, eps):
    mp = float(np.mean(np.clip(1.0 - np.exp(eps - lp), 0.0, 1.0)))
    mq = float(np.mean(np.clip(1.0 - np.exp(eps + lq), 0.0, 1.0)))
    return mp, mq


def eps_interval(lp, lq, delta, alpha=0.01, hi=80.0, iters=60):
    """Plain estimate and a (1 - alpha) confidence interval for eps(delta) = inf{eps : max(H(P||Q), H(Q||P)) <= delta}
    from n i.i.d. samples of log LR under each world.  Each of the four one-sided limits (two directions x two sides)
    is at level alpha/4, so all four hold jointly with probability >= 1 - alpha; the curve is nonincreasing in eps,
    so the limits invert by bisection."""
    lp = np.asarray(lp, np.float64); lq = np.asarray(lq, np.float64); n_p, n_q = len(lp), len(lq); a4 = alpha / 4

    def d_plain(e):
        mp, mq = hs_means(lp, lq, e); return max(mp, mq)

    def d_lo(e):
        mp, mq = hs_means(lp, lq, e); return max(kl_lo(mp, n_p, a4), kl_lo(mq, n_q, a4))

    def d_hi(e):
        mp, mq = hs_means(lp, lq, e); return max(kl_hi(mp, n_p, a4), kl_hi(mq, n_q, a4))

    def inv(f):
        if f(0.0) <= delta:
            return 0.0
        lo_, hi_ = 0.0, hi
        for _ in range(iters):
            mid = 0.5 * (lo_ + hi_)
            if f(mid) > delta: lo_ = mid
            else: hi_ = mid
        return 0.5 * (lo_ + hi_)
    return inv(d_plain), inv(d_lo), inv(d_hi)
