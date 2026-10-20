"""Step 15c: pointwise certificate for the convex-order law on the replayed chains.
For p <=_cx p', if L_{p'}(x) >= L_p(x) for every x on the rejection region {x > x*_p(eps)} then
H_eps(M_{p'}||Q) >= H_eps(M_p||Q) exactly (pointwise argument of Proposition 9), with no integration error.
We check the inequality on a fine x-grid for x >= x*_p(eps_p(1e-2)) (the smallest rejection threshold among the
tested deltas) and record the minimum of log L_{p'} - log L_p there, and the largest x below the threshold where the
inequality fails (the non-convex window)."""
import numpy as np, csv, time, sys
from scipy.stats import binom, norm
from corraudit import two_state_chain, minsep_chain, count_distribution, eps_sum_exact, sum_laws, _mix_pdf
import step15_convex_order as s15   # NOTE: importing re-runs its module body once (families + chains); acceptable
from step15_convex_order import T, q, C, SIGMA, DELTAS, moments, mps
rng = np.random.default_rng(15)
t0 = time.time(); log = lambda s: print(f"{s} [{time.time()-t0:.0f}s]", flush=True)
ns = np.arange(T + 1)
base = {"binomial": binom.pmf(ns, T, q)}
P, pi, emit = two_state_chain(q, 0.6); base["markov0.6"] = count_distribution(P, pi, emit, T)
P, pi, emit = minsep_chain(q, 5); base["minsep5"] = count_distribution(P, pi, emit, T)
X = np.linspace(-12 * SIGMA, C * T / np.sqrt(T) + 12 * SIGMA, 400001)
logq = norm.logpdf(X, 0, SIGMA)
def llr(p):
    means, w = sum_laws(p, C, SIGMA, T); keep = w > 1e-16
    return np.log(np.clip(_mix_pdf(X, means[keep], w[keep] / w[keep].sum(), SIGMA), 1e-300, None)) - logq
rows = []; nfail = {d: 0 for d in DELTAS}; nsteps = 0
for start in ["binomial", "markov0.6", "minsep5"]:
    for c in range(12):
        p = base[start].copy(); l_p = llr(p)
        for s in range(25):
            p2 = mps(p, rng, max_k=30 if s < 15 else None); l_2 = llr(p2); nsteps += 1
            bad = np.flatnonzero(l_2 < l_p - 1e-12)
            xfail = float(X[bad.max()]) if len(bad) else -np.inf     # largest x where L_{p'} < L_p
            row = [start, c, s, f"{xfail:.4f}", f"{moments(p2)[2]:.4f}"]
            for d in DELTAS:
                e = eps_sum_exact(p, C, SIGMA, T, d)
                xs = float(X[min(np.searchsorted(l_p, e), len(X) - 1)])
                ok = xfail < xs                                       # certificate: no crossing on the rejection region
                nfail[d] += (not ok)
                row += [f"{e:.4f}", f"{xs:.4f}", int(ok)]
            rows.append(row)
            p, l_p = p2, l_2
        log(f"{start} #{c}: steps so far {nsteps}, certificate failures per delta {nfail}")
with open("step15c_lrcheck.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["start", "chain", "step", "largest_x_with_L2<Lp", "D_after"] + sum(([f"eps_p_{d:g}", f"x_star_{d:g}", f"certified_{d:g}"] for d in DELTAS), [])); w.writerows(rows)
log(f"steps {nsteps}; certificate failures per delta: {nfail}")
