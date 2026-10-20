"""Pilot A: reference eps under correlated inclusion vs Poisson accounting and schema worst case."""
import numpy as np, time, csv, sys
from scipy.stats import norm
from corraudit import *

C, sigma, q = 1.0, 2.0, 0.1
DELTAS = [1e-3, 1e-4]
NMC = 300_000
rng = np.random.default_rng(2026)

configs = []
for T in [200, 1000]:
    configs.append((T, "iid", 0.0))
    for lam in ([0.3, 0.6, 0.9, 0.97] if T == 200 else [0.6, 0.9, 0.99]):
        configs.append((T, "markov", lam))
    for b in ([2, 5, 9] if T == 200 else [5, 9]):
        configs.append((T, "minsep", b))

import os
done = set()
if os.path.exists("pilot_a_results.csv"):
    for row in csv.DictReader(open("pilot_a_results.csv")):
        done.add((int(row["T"]), row["process"], float(row["param"])))
out = open("pilot_a_results.csv", "a", newline="")
w = csv.writer(out)
if not done: w.writerow(["T", "process", "param", "tmix", "count_mean", "count_var", "count_max_q999",
            "dK_sum_gauss", "delta", "eps_full_mc", "eps_sum_exact", "eps_poisson_pld", "eps_schema_maxp"])
t0 = time.time()
for (T, proc, par) in configs:
    if (T, proc, float(par)) in done:
        continue
    if proc == "iid":
        P, pi, emit = iid_chain(q)
    elif proc == "markov":
        P, pi, emit = two_state_chain(q, par)
    else:
        P, pi, emit = minsep_chain(q, int(par))
    tmix = mixing_time(P, pi)
    pn = count_distribution(P, pi, emit, T)
    n = np.arange(T + 1)
    cm = float(np.sum(n * pn)); cv = float(np.sum(n * n * pn) - cm**2)
    cq = int(n[np.searchsorted(np.cumsum(pn), 0.999)])
    # Kolmogorov distance between the present sum law (mixture) and its moment-matched Gaussian
    means = C * n / np.sqrt(T)
    mu1 = float(np.sum(pn * means)); v1 = sigma**2 + float(np.sum(pn * means**2) - mu1**2)
    xs = np.linspace(-10 * sigma, means.max() + 10 * sigma, 20001)
    Fmix = (pn[None, :] * norm.cdf(xs[:, None], means[None, :], sigma)).sum(axis=1)
    dK = float(np.abs(Fmix - norm.cdf(xs, mu1, np.sqrt(v1))).max())
    e_full, lp, lq = eps_full_mc(P, pi, emit, T, C, sigma, DELTAS[0], n=NMC, rng=rng, return_curve=True)

    def delta_of_eps(e):
        return max(np.mean(np.clip(1.0 - np.exp(e - lp), 0, None)), np.mean(np.clip(1.0 - np.exp(e + lq), 0, None)))

    for d in DELTAS:
        ef = eps_from_delta_curve(delta_of_eps, d)
        es = eps_sum_exact(pn, C, sigma, T, d)
        ep = eps_poisson_pld(q, T, C, sigma, d)
        esch = eps_schema(cq, C, sigma, d)
        w.writerow([T, proc, par, tmix, f"{cm:.3f}", f"{cv:.3f}", cq, f"{dK:.2e}", d,
                    f"{ef:.4f}", f"{es:.4f}", f"{ep:.4f}", f"{esch:.4f}"])
        out.flush()
        print(f"T={T} {proc} {par} tmix={tmix} var={cv:.1f} dK={dK:.1e} delta={d}: full={ef:.3f} sum={es:.3f} poisson={ep:.3f} schema(k={cq})={esch:.3f}  [{time.time()-t0:.0f}s]", flush=True)
out.close()
