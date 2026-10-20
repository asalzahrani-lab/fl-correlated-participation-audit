"""Replay the convex-order chains of step 15 and re-examine every step whose eps decreased (at any delta) with a 20x
finer integration grid, to separate numerical tolerance from genuine violations."""
import numpy as np, csv, time
from scipy.stats import binom
from corraudit import two_state_chain, minsep_chain, count_distribution, eps_sum_exact
from step15_convex_order import T, q, C, SIGMA, DELTAS, moments, mps, is_cx_leq
rng = np.random.default_rng(15)
t0 = time.time(); log = lambda s: print(f"{s} [{time.time()-t0:.0f}s]", flush=True)
ns = np.arange(T + 1)
base = {"binomial": binom.pmf(ns, T, q)}
P, pi, emit = two_state_chain(q, 0.6); base["markov0.6"] = count_distribution(P, pi, emit, T)
P, pi, emit = minsep_chain(q, 5); base["minsep5"] = count_distribution(P, pi, emit, T)
flag = []
for start in ["binomial", "markov0.6", "minsep5"]:
    for c in range(12):
        p = base[start].copy(); e_prev = [eps_sum_exact(p, C, SIGMA, T, d) for d in DELTAS]
        for s in range(25):
            p2 = mps(p, rng, max_k=30 if s < 15 else None)
            e2 = [eps_sum_exact(p2, C, SIGMA, T, d) for d in DELTAS]
            for i, d in enumerate(DELTAS):
                if e2[i] < e_prev[i]:
                    flag.append((start, c, s, d, e_prev[i], e2[i], p.copy(), p2.copy()))
            p, e_prev = p2, e2
        log(f"replayed {start} #{c}; flagged so far {len(flag)}")
log(f"flagged steps (any decrease at default grid): {len(flag)}")
with open("step15b_flagged.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["start", "chain", "step", "delta", "eps_before_default", "eps_after_default", "eps_before_fine", "eps_after_fine", "drop_fine"])
    for (start, c, s, d, eb, ea, p, p2) in flag:
        ebf = eps_sum_exact(p, C, SIGMA, T, d, grid=400001); eaf = eps_sum_exact(p2, C, SIGMA, T, d, grid=400001)
        w.writerow([start, c, s, f"{d:g}", f"{eb:.6f}", f"{ea:.6f}", f"{ebf:.6f}", f"{eaf:.6f}", f"{ebf-eaf:.2e}"])
        log(f"{start} #{c} step {s} delta={d:g}: default {eb:.6f} -> {ea:.6f} (drop {eb-ea:.2e}); fine grid {ebf:.6f} -> {eaf:.6f} (drop {ebf-eaf:.2e})")
log("done")
