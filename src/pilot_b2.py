"""Pilot B2: final-model (sum) release, where the mechanism's eps is exact at any delta.
Coverage of the Gaussian-fit audit (Bonferroni box) and of the count-conditioned audit vs the exact eps,
as delta shrinks, under i.i.d., Markov and min-sep participation."""
import numpy as np, time, csv
from scipy.stats import norm
from corraudit import *
from pilot_b import gauss_audit_fast, count_conditioned_fast

C, sigma, q, T = 1.0, 2.0, 0.1, 200
M, R, ALPHA = 1000, 300, 0.05
DELTAS = [1e-4, 1e-5, 1e-6, 1e-8]
rng = np.random.default_rng(11)
cells = [("iid", 0.0), ("markov", 0.6), ("markov", 0.9), ("markov", 0.97), ("minsep", 5)]

out = open("pilot_b2_results.csv", "w", newline="")
w = csv.writer(out)
w.writerow(["T", "process", "param", "delta", "eps_exact_sum", "eps_gaussfit_population", "estimator",
            "mean_hat", "sd_hat", "coverage", "ratio"])
t0 = time.time()
for (proc, par) in cells:
    if proc == "iid":
        P, pi, emit = iid_chain(q)
    elif proc == "markov":
        P, pi, emit = two_state_chain(q, par)
    else:
        P, pi, emit = minsep_chain(q, int(par))
    pn = count_distribution(P, pi, emit, T)
    n = np.arange(T + 1)
    means = C * n / np.sqrt(T)
    mu1 = float(np.sum(pn * means)); s1 = float(np.sqrt(sigma**2 + np.sum(pn * means**2) - mu1**2))
    # population-level (infinite-sample) Gaussian-fit eps: what the fit converges to
    exact = {d: eps_sum_exact(pn, C, sigma, T, d) for d in DELTAS}
    popfit = {d: eps_gauss_pair(0.0, sigma, mu1, s1, d) for d in DELTAS}
    sims = [simulate_scores(P, pi, emit, T, C, sigma, M, rng) for _ in range(R)]
    for d in DELTAS:
        gb, cc = [], []
        for (S, present, counts) in sims:
            gb.append(gauss_audit_fast(S, present, d)[0])
            cc.append(count_conditioned_fast(S, present, counts, d)[0])
        for name, v in (("gauss_box", np.array(gb)), ("countcond", np.array(cc))):
            w.writerow([T, proc, par, d, f"{exact[d]:.4f}", f"{popfit[d]:.4f}", name, f"{v.mean():.3f}",
                        f"{v.std():.3f}", f"{np.mean(v <= exact[d]):.3f}", f"{v.mean()/exact[d]:.3f}"])
            print(f"{proc} {par} delta={d:g}: exact={exact[d]:.3f} popfit={popfit[d]:.3f} | {name:10s} mean={v.mean():.3f}±{v.std():.3f} coverage={np.mean(v <= exact[d]):.3f} [{time.time()-t0:.0f}s]", flush=True)
        out.flush()
out.close()
