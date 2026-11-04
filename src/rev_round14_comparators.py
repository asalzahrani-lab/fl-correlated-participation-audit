"""Round 14: comparators on the same information as the path-aware audit (the reviews of TIFS v13 asked that the
contribution of the path be separated from that of knowing the law and of the certification method).

Four auditors on the same 120 audit data sets per law (m = 1,000, T = 200, a = 0.5, delta = 1e-4, seed 14800 + law):
  path       the certified path-aware audit of Section V (regression on the known paths, a_lo at alpha_est = 0.045,
             the law's certified table);
  sum-law    the sum statistic S_i = sum_t X_t / sqrt(T) with the SAME law and the SAME certified per-round table: the
             auditor knows each present canary's count N_i, regresses S_i on N_i / sqrt(T) through the origin (the
             model gives S_i ~ N(a N_i / sqrt(T), s^2)), bounds a from below with the same two one-sided limits
             (normal for the slope, chi-square for the absent canaries' scale; variance of the slope s^2 T / sum N_i^2
             against s^2 / sum N_i for the path regression) and maps the limit through the table -- this isolates what
             the path adds over the count: the information ratio (sum_i N_i^2 / T) / sum_i N_i;
  count-cond the count-conditioned sum audit of Table III (the sum release's own privacy under the exact count law,
             Theorem 1), for reference;
  max-fit    a maximum-over-rounds score in the manner of Algorithm 3 of Andrew et al. (main-text ref. [27]): each
             canary scored by its largest projection over the rounds, in declared-noise units (their algorithm scores
             each round by the cosine with that round's model delta, normalised by the delta's norm; here the
             normalisation is the fixed noise scale), fitted by Gaussians to the present and absent scores as the
             Gaussian-fit baseline of Table III does with the sum score (Bonferroni box over the four fitted
             parameters, alpha = 0.05), reported against the per-round reference.
Laws: Markov lambda = 0.9 (reference 11.869), i.i.d. (2.866), the FLASH trace law (10.69) and the cyclic law (7.94), each with the
certified table of Table III (rev_round14_common.py).
Reported per auditor and law: mean +- sd, recovery, estimates above the reference, flags of the Poisson claim (2.866;
1.219 for the trace law), and for sum-law the mean information ratio.

    cd results && P3_TRACE=../trace python3 ../src/rev_round14_comparators.py       # a few minutes
Output rev_round14_comparators.csv.
"""
import csv, time
import numpy as np
from scipy.stats import norm, chi2
from corraudit import two_state_chain, iid_chain, eps_poisson_pld, count_distribution, _bonf_box
from certified import ChainLaw, EmpiricalPathLaw
from certified2 import snr_lower_bound
from auditors2 import count_cond_audit
from vecgauss import eps_gauss_pair_vec
from rev_common import window, apply_policy
from rev_round14_common import paper_table

T, A_TRUE, DELTA, Q, M, R = 200, 0.5, 1e-4, 0.1, 1000, 120
A_EST = 0.045
t0 = time.time()
log = lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True)



def sum_law_lower_bound(X, present, B, alpha):
    """Lower limit on a from the sum statistic and the known counts (slope of S_i on N_i / sqrt(T) through the origin)."""
    S = X.sum(axis=1) / np.sqrt(T); N = B.sum(axis=1).astype(float)
    Sp, Np = S[present], N[present] / np.sqrt(T); xa = X[~present].astype(float).ravel(); n0 = len(xa)
    if Np.sum() <= 0 or n0 < 2: return 0.0, 0.0
    slope = (Sp * Np).sum() / (Np * Np).sum()
    sd = xa.std(ddof=1); sd_hi = sd * np.sqrt((n0 - 1) / chi2.ppf(alpha / 2, n0 - 1)); z = norm.ppf(1 - alpha / 2)
    a_lo = max(slope - z * sd_hi / np.sqrt((Np * Np).sum()), 0.0) / sd_hi
    info = (Np * Np).sum() / N[present].sum()                       # (sum N^2 / T) / sum N
    return float(a_lo), float(info)


def max_fit_audit(X, present, alpha=0.05, ngrid=5):
    y = X.max(axis=1); a = alpha / 4
    (m0l, m0h), (s0l, s0h) = _bonf_box(y[~present], a); (m1l, m1h), (s1l, s1h) = _bonf_box(y[present], a)
    g = np.meshgrid(np.linspace(m0l, m0h, ngrid), np.linspace(s0l, s0h, ngrid), np.linspace(m1l, m1h, ngrid), np.linspace(s1l, s1h, ngrid), indexing="ij")
    return float(eps_gauss_pair_vec(g[0].ravel(), g[1].ravel(), g[2].ravel(), g[3].ravel(), DELTA).min())


if __name__ == "__main__":
    laws = {}
    P, pi, emit = two_state_chain(Q, 0.9); laws["markov"] = (ChainLaw(P, pi, emit, T), 11.869, paper_table("markov"), count_distribution(P, pi, emit, T), eps_poisson_pld(Q, T, 1.0, 2.0, DELTA))
    P, pi, emit = iid_chain(Q); laws["iid"] = (ChainLaw(P, pi, emit, T), 2.866, paper_table("iid"), count_distribution(P, pi, emit, T), eps_poisson_pld(Q, T, 1.0, 2.0, DELTA))
    A, _, _ = window(10, 3); Bt = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0))
    starts = np.arange(0, Bt.shape[1] - T + 1, 10); W = np.concatenate([Bt[:, s:s + T] for s in starts], axis=0).astype(np.int8)
    pn_trace = np.bincount(W.sum(axis=1), minlength=T + 1) / len(W)
    laws["trace"] = (EmpiricalPathLaw(W), 10.69, paper_table("trace"), pn_trace, eps_poisson_pld(float(W.mean()), T, 1.0, 2.0, DELTA))
    cyc = np.zeros((10, T), np.int8)
    for ph in range(10): cyc[ph, ph::10] = 1
    laws["cyclic"] = (EmpiricalPathLaw(cyc), 7.94, paper_table("cyclic"), np.bincount(cyc.sum(axis=1), minlength=T + 1) / 10.0, eps_poisson_pld(Q, T, 1.0, 2.0, DELTA))
    rows = []
    for li, name in enumerate(("markov", "iid", "trace", "cyclic")):
        L, ref, tab, pn, pois = laws[name]; rng = np.random.default_rng(14800 + li)
        est = {k: [] for k in ("path", "sum-law", "count-cond", "max-fit")}; infos = []
        for _ in range(R):
            present = rng.random(M) < 0.5; B = L.sample(M, rng)
            X = (B * present[:, None]) * A_TRUE + rng.standard_normal((M, T)).astype(np.float32)
            est["path"].append(tab.cert(snr_lower_bound(X, present, B, A_EST)))
            alo, info = sum_law_lower_bound(X, present, B, A_EST); est["sum-law"].append(tab.cert(alo)); infos.append(info)
            est["count-cond"].append(count_cond_audit(X, present, B.sum(axis=1), pn, DELTA))
            est["max-fit"].append(max_fit_audit(X, present))
        for k, v in est.items():
            v = np.array(v)
            rows.append([name, k, R, f"{ref:.3f}", f"{pois:.3f}", f"{v.mean():.4f}", f"{v.std():.4f}", f"{v.mean() / ref:.4f}", int((v > ref).sum()), int((v > pois).sum()),
                         f"{np.mean(infos):.4f}" if k == "sum-law" else ""])
            log(f"{name} {k:10s}: {v.mean():.3f} +- {v.std():.3f} ({v.mean() / ref:.1%} of {ref}); above the reference {int((v > ref).sum())}/{R}; flags {int((v > pois).sum())}/{R}"
                + (f"; information ratio {np.mean(infos):.3f}" if k == "sum-law" else ""))
    with open("rev_round14_comparators.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["law", "auditor", "R", "reference", "poisson", "mean", "sd", "recovery", "above_ref", "flags", "info_ratio"]); w.writerows(rows)
    log("done")
