"""Revision A1/B: the threshold criterion of the sharpened Proposition 1 on the 900 replayed mean-preserving spreads.

For each spread p -> p' (same replay as step15c: rng seed 15, 3 starts x 12 chains x 25 steps) and each delta,
s* = s*_p(eps_p(delta)) solves log L_p(s*) = eps_p(delta); the forward inequality H(M_p') >= H(M_p) at eps_p is
certified when E_p'[Phi_bar(s* - c N)] >= E_p[Phi_bar(s* - c N)] (Proposition 1(i)), with no integration grid.
eps_p(delta) is read from step15c_lrcheck.csv (same rows, same order).  Also records whether s* >= cT (Prop. 1(ii)).
Outputs rev_threshold.csv and prints the counts per delta, next to the old pointwise-LR certificate counts."""
import csv
import numpy as np
from scipy.stats import binom, norm
from corraudit import two_state_chain, minsep_chain, count_distribution

T, q, a = 200, 0.1, 0.5
c = a / np.sqrt(T); ns = np.arange(T + 1)
DELTAS = ["0.01", "0.001", "0.0001", "1e-05", "1e-06"]
rng = np.random.default_rng(15)


def mps(p, rng, max_k=None):                    # identical to step15_convex_order.mps
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


def logL(p, s):
    z = c * ns * s - (c * ns) ** 2 / 2; keep = p > 0; m = z[keep].max()
    return m + np.log(np.sum(p[keep] * np.exp(z[keep] - m)))


def s_star(p, eps):
    lo, hi = -60.0, 400.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if logL(p, mid) > eps: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


base = {"binomial": binom.pmf(ns, T, q)}
P, pi, emit = two_state_chain(q, 0.6); base["markov0.6"] = count_distribution(P, pi, emit, T)
P, pi, emit = minsep_chain(q, 5); base["minsep5"] = count_distribution(P, pi, emit, T)
old = list(csv.DictReader(open("step15c_lrcheck.csv")))
rows = []; i = 0
cnt_new = {d: 0 for d in DELTAS}; cnt_old = {d: 0 for d in DELTAS}; cnt_far = {d: 0 for d in DELTAS}
for start in ["binomial", "markov0.6", "minsep5"]:
    for ch in range(12):
        p = base[start].copy()
        for st in range(25):
            p2 = mps(p, rng, max_k=30 if st < 15 else None)
            r = old[i]; assert r["start"] == start and int(r["chain"]) == ch and int(r["step"]) == st
            row = [start, ch, st]
            for d in DELTAS:
                e = float(r[f"eps_p_{d}"]); s = s_star(p, e)
                diff = float(np.sum((p2 - p) * norm.sf(s - c * ns)))
                ok = diff >= -1e-15; far = s >= c * T
                cnt_new[d] += ok; cnt_old[d] += int(r[f"certified_{d}"]); cnt_far[d] += far
                row += [f"{e:.4f}", f"{s:.4f}", f"{diff:.3e}", int(ok), int(far)]
            rows.append(row); p = p2; i += 1
with open("rev_threshold.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["start", "chain", "step"] + sum(([f"eps_p_{d}", f"s_star_{d}", f"criterion_{d}", f"certified_{d}", f"far_tail_{d}"] for d in DELTAS), []))
    w.writerows(rows)
print(f"spreads: {i}")
for d in DELTAS:
    print(f"delta {d}: threshold criterion certifies {cnt_new[d]}/{i}; old pointwise LR certificate {cnt_old[d]}/{i}; s* >= cT in {cnt_far[d]}/{i}")
