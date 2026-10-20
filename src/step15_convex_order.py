"""Step 15: the dispersion law for the final-model (sum) release, tested as a mathematical statement.

The sum statistic S = sum_t X_t / sqrt(T) has present law  M_p = sum_n p_n N(a n / sqrt(T), 1)  and absent law N(0,1),
so eps_sum(delta) is a functional of the count law p alone (Theorem 1).  Two candidate orderings:
  (CX)  p <=_cx p'  (convex order, same mean)      =>  eps_sum(p) <= eps_sum(p')  for all delta?
  (D)   D(p) <= D(p') (dispersion index, same mean) =>  eps_sum(p) <= eps_sum(p')  for all delta?
Tests: (1) chains of random mean-preserving spreads (each step is a convex-order increase); (2) the standard families
(Binomial, Markov, min-sep, beta-binomial, all-or-none, deterministic) compared pairwise by D, with a check of whether
each pair is convex-ordered; (3) adversarial pairs: same mean and variance, different upper tail.
Outputs: step15_chains.csv, step15_families.csv, step15_adversarial.csv and a printed summary."""
import numpy as np, csv, time, itertools
from scipy.stats import binom, betabinom
from corraudit import two_state_chain, minsep_chain, count_distribution, eps_sum_exact

T, q, a = 200, 0.1, 0.5
C, SIGMA = 1.0, 1.0 / a
DELTAS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
rng = np.random.default_rng(15)
t0 = time.time()
log = lambda s: print(f"{s} [{time.time()-t0:.0f}s]", flush=True)
ns = np.arange(T + 1)


def moments(p):
    m = (p * ns).sum(); v = (p * ns ** 2).sum() - m * m
    return m, v, v / m


def eps_all(p):
    return [eps_sum_exact(p, C, SIGMA, T, d) for d in DELTAS]


def is_cx_leq(p, p2, tol=1e-12):
    """p <=_cx p2 on integers with equal means  <=>  equal means and
    sum_{n>=k} (n-k)^+ weights: E_p[(N-k)_+] <= E_p2[(N-k)_+] for all k (stop-loss order)."""
    m1, _, _ = moments(p); m2, _, _ = moments(p2)
    if abs(m1 - m2) > 1e-9: return False
    for k in range(T + 1):
        if (p * np.clip(ns - k, 0, None)).sum() > (p2 * np.clip(ns - k, 0, None)).sum() + tol: return False
    return True


def mps(p, rng, max_k=None):
    """One random mean-preserving spread: take mass eta at n and move half to n-k, half to n+k."""
    support = np.flatnonzero(p > 1e-10)
    for _ in range(100):
        n = rng.choice(support)
        kmax = min(n, T - n) if max_k is None else min(n, T - n, max_k)
        if kmax < 1: continue
        k = int(rng.integers(1, kmax + 1))
        eta = p[n] * rng.uniform(0.1, 0.8)
        p2 = p.copy(); p2[n] -= eta; p2[n - k] += eta / 2; p2[n + k] += eta / 2
        return p2
    return p.copy()


# ---------------------------------------------------------------- base laws
base = {}
base["binomial"] = binom.pmf(ns, T, q)
for lam in [0.3, 0.6, 0.9, 0.97]:
    P, pi, emit = two_state_chain(q, lam); base[f"markov{lam}"] = count_distribution(P, pi, emit, T)
for b in [2, 5, 9]:
    P, pi, emit = minsep_chain(q, b); base[f"minsep{b}"] = count_distribution(P, pi, emit, T)
for rho in [0.02, 0.1, 0.3]:   # beta-binomial with mean q and intra-class correlation rho
    ab = (1 - rho) / rho; base[f"betabin_rho{rho}"] = betabinom.pmf(ns, T, q * ab, (1 - q) * ab)
pz = np.zeros(T + 1); pz[0] = 1 - q; pz[T] = q; base["all_or_none"] = pz
pd_ = np.zeros(T + 1); pd_[int(round(q * T))] = 1.0; base["deterministic"] = pd_

if __name__ == "__main__":
    # ---------------------------------------------------------------- (2) families
    log("== families")
    fam = {}
    with open("step15_families.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["law", "mean", "var", "D"] + [f"eps_delta{d:g}" for d in DELTAS])
        for name, p in base.items():
            m, v, D = moments(p); e = eps_all(p); fam[name] = (m, v, D, e)
            w.writerow([name, f"{m:.4f}", f"{v:.4f}", f"{D:.4f}"] + [f"{x:.4f}" for x in e])
            log(f"{name:16s} mean {m:6.2f} D {D:7.3f} eps " + " ".join(f"{x:7.3f}" for x in e))
    # pairwise: D-order vs eps-order, and convex-order check
    names = list(base.keys()); viol = {d: [] for d in DELTAS}; cx_pairs = []; ncomp = 0
    for n1, n2 in itertools.combinations(names, 2):
        m1, v1, D1, e1 = fam[n1]; m2, v2, D2, e2 = fam[n2]
        if abs(m1 - m2) > 0.5: continue   # only same-mean comparisons (minsep laws have slightly different means)
        ncomp += 1
        lo, hi = (n1, n2) if D1 <= D2 else (n2, n1)
        plo, phi = base[lo], base[hi]
        cx = is_cx_leq(plo, phi, tol=1e-9) if abs(m1 - m2) < 1e-6 else False
        cx_pairs.append((lo, hi, cx))
        for i, d in enumerate(DELTAS):
            elo, ehi = (e1[i], e2[i]) if D1 <= D2 else (e2[i], e1[i])
            if elo > ehi * (1 + 1e-6): viol[d].append((lo, hi, elo, ehi, cx))
    log(f"same-mean pairs compared: {ncomp}; convex-ordered pairs: {sum(1 for _,_,c in cx_pairs if c)}")
    for d in DELTAS:
        log(f"delta={d:g}: D-order violations {len(viol[d])}: " + "; ".join(f"{lo}(D-low, eps {elo:.2f}) > {hi}(D-high, eps {ehi:.2f}) [cx={cx}]" for lo, hi, elo, ehi, cx in viol[d][:6]))

    # ---------------------------------------------------------------- (1) convex-order chains
    log("== convex-order chains")
    chain_rows = []; cx_viol = {d: 0 for d in DELTAS}; steps_total = 0; worst = {d: 0.0 for d in DELTAS}
    for start in ["binomial", "markov0.6", "minsep5"]:
        for c in range(12):
            p = base[start].copy(); e_prev = eps_all(p); D_prev = moments(p)[2]
            for s in range(25):
                p2 = mps(p, rng, max_k=30 if s < 15 else None)
                assert is_cx_leq(p, p2, tol=1e-9)
                e2 = eps_all(p2); D2 = moments(p2)[2]; steps_total += 1
                for i, d in enumerate(DELTAS):
                    if e2[i] < e_prev[i] * (1 - 1e-6):
                        cx_viol[d] += 1; worst[d] = max(worst[d], e_prev[i] - e2[i])
                chain_rows.append([start, c, s, f"{D_prev:.4f}", f"{D2:.4f}"] + [f"{x:.4f}" for x in e_prev] + [f"{x:.4f}" for x in e2])
                p, e_prev, D_prev = p2, e2, D2
            log(f"  chain {start} #{c}: final D {D_prev:.2f}, eps(1e-4) {e_prev[2]:.3f}")
    with open("step15_chains.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["start", "chain", "step", "D_before", "D_after"] + [f"eps_before_{d:g}" for d in DELTAS] + [f"eps_after_{d:g}" for d in DELTAS]); w.writerows(chain_rows)
    log(f"convex-order steps tested: {steps_total}")
    for d in DELTAS:
        log(f"delta={d:g}: violations of monotonicity under a mean-preserving spread: {cx_viol[d]} (worst drop {worst[d]:.4f})")

    # ---------------------------------------------------------------- (3) adversarial: same mean AND variance, different tails
    log("== adversarial pairs (same mean and variance, heavier upper tail vs lighter)")
    adv_rows = []
    m0 = q * T
    for trial in range(40):
        # law A: binomial-like core + small mass far in the upper tail; law B: same mean/variance via a symmetric two-point spread near the mean
        base_p = binom.pmf(ns, T, q)
        nfar = int(rng.integers(60, 120)); eta = rng.uniform(0.0005, 0.01)
        pA = base_p.copy() * (1 - eta)
        # add mass eta at nfar and compensate the mean by moving mass to lower counts
        pA[nfar] += eta
        mA = (pA * ns).sum(); shift = mA - m0                      # positive: need to lower the mean
        # remove mass from counts above mean proportionally and put it at 0 to restore the mean
        hi_mask = ns > m0
        need = shift / m0 if m0 > 0 else 0
        # simple fix: scale mass at n0 = round(m0) region down and move to 0
        k = 0; p_try = pA.copy()
        while (p_try * ns).sum() > m0 + 1e-9 and k < 200:
            nn = int(rng.integers(1, int(m0) + 1)); mv = min(p_try[nn], 0.002)
            p_try[nn] -= mv; p_try[0] += mv; k += 1
        if abs((p_try * ns).sum() - m0) > 0.05: continue
        pA = p_try / p_try.sum()
        mA, vA, DA = moments(pA)
        # law B: binomial spread symmetrically around the mean until its variance matches vA (mean preserved)
        pB = base_p.copy(); vB = moments(pB)[1]; it = 0
        while vB < vA and it < 5000:
            nn = int(round(mA)); kk = int(rng.integers(1, 12)); eta2 = min(pB[nn], 0.001)
            if eta2 <= 0: break
            pB[nn] -= eta2; pB[nn - kk] += eta2 / 2; pB[nn + kk] += eta2 / 2; vB = moments(pB)[1]; it += 1
        mB, vB, DB = moments(pB)
        if abs(mA - mB) > 0.05 or abs(vA - vB) / vA > 0.05: continue
        eA, eB = eps_all(pA), eps_all(pB)
        tailA = (pA * np.clip(ns - 50, 0, None)).sum(); tailB = (pB * np.clip(ns - 50, 0, None)).sum()
        adv_rows.append([trial, f"{mA:.3f}", f"{vA:.3f}", f"{DA:.3f}", f"{mB:.3f}", f"{vB:.3f}", f"{DB:.3f}", f"{tailA:.4f}", f"{tailB:.4f}"] + [f"{x:.4f}" for x in eA] + [f"{x:.4f}" for x in eB])
        log(f"  trial {trial}: D_A {DA:.3f} D_B {DB:.3f} (A has the heavy upper tail, stop-loss@50: {tailA:.4f} vs {tailB:.4f}); eps_A " + " ".join(f"{x:6.2f}" for x in eA) + " | eps_B " + " ".join(f"{x:6.2f}" for x in eB))
    with open("step15_adversarial.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["trial", "mean_A", "var_A", "D_A", "mean_B", "var_B", "D_B", "stoploss50_A", "stoploss50_B"] + [f"epsA_{d:g}" for d in DELTAS] + [f"epsB_{d:g}" for d in DELTAS]); w.writerows(adv_rows)
    log("done")
