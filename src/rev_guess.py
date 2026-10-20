"""Revision B5: the guess-counting one-run audit of Steinke, Nasr and Jagielski (2023, Theorem 5.2 / Corollary 5.4)
given the SAME information as the path-aware audit (aggregates + the canaries' known paths), so that the effect of the
information can be separated from the effect of the parametric, certified map of Section V.

Scores per canary: (a) sum statistic S_i = sum_t X_it / sqrt(T) (the final-model score of the published instantiations);
(b) path-aware log-likelihood ratio  a * sum_t B_it X_it - a^2 N_i / 2  with a = 1/sigma (the claimed noise).
Guesses: 'present' for the k highest scores and 'absent' for the k lowest (r = 2k guesses), k in K_SET fixed in
advance; the reported lower bound is the best over K_SET at confidence 1 - beta/|K_SET| each (Bonferroni), so the
report is valid at level beta.  Bound (Cor. 5.4): reject eps if
    f(v) + 2 m delta max_i (f(v - i) - f(v)) / i <= beta',   f(v) = P[Binomial(r, e^eps/(1+e^eps)) >= v].
Outputs rev_guess.csv (one row per law x score: mean, sd, recovery against the paper's reference)."""
import csv, sys
import numpy as np
from scipy.stats import binom
from corraudit import two_state_chain, sample_paths
from certified import EmpiricalPathLaw

T, a, m, R, DELTA, BETA = 200, 0.5, 1000, 120, 1e-4, 0.05
K_SET = [10, 25, 50, 100, 200, 300, 400, 500]
REF = {"markov": 11.869, "iid": 2.869, "trace": 10.685}
POIS = {"markov": 2.866, "iid": 2.866, "trace": 1.218}


def bound_ok(eps, r, v, beta):
    if v <= 0:
        return False
    p = np.exp(eps) / (1 + np.exp(eps))
    xs = np.arange(0, v + 1)
    fx = np.where(xs > 0, binom.sf(xs - 1, r, p), 1.0)                      # f(x) = P[Bin >= x], x = 0..v
    fv = fx[v]; i = np.arange(1, min(m, v) + 1)
    slack = float(np.max((fx[v - i] - fv) / i))                              # exact max over i = 1..min(m, v)
    return fv + 2 * m * DELTA * slack <= beta


def eps_lower(r, v, beta):
    if not bound_ok(0.0, r, v, beta):
        return 0.0
    lo, hi = 0.0, 30.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if bound_ok(mid, r, v, beta): lo = mid
        else: hi = mid
    return lo


def audit(scores, present):
    order = np.argsort(scores); best = 0.0
    for k in K_SET:
        hi_idx, lo_idx = order[-k:], order[:k]
        v = int(present[hi_idx].sum() + (~present[lo_idx]).sum())
        best = max(best, eps_lower(2 * k, v, BETA / len(K_SET)))
    return best


def paths_for(law_name, n, rng, trace_law=None):
    if law_name == "trace":
        return trace_law.sample(n, rng)
    P, pi, emit = two_state_chain(0.1, 0.9 if law_name == "markov" else 0.0)
    return sample_paths(P, pi, emit, T, n, rng)


if __name__ == "__main__":
    laws = sys.argv[1:] or ["markov", "iid", "trace"]
    trace_law = None
    if "trace" in laws:
        from rev_common import window, apply_policy, window_paths
        A, _, _ = window(10, 3)
        trace_law = EmpiricalPathLaw(window_paths(apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0)), T, 10))
    rows = []
    for law_name in laws:
        rng = np.random.default_rng({"markov": 8101, "iid": 8102, "trace": 8103}[law_name])
        res = {"sum": [], "path": []}
        for rep in range(R):
            B = paths_for(law_name, m, rng, trace_law).astype(np.float32)
            present = rng.random(m) < 0.5
            X = a * present[:, None] * B + rng.standard_normal((m, T)).astype(np.float32)
            res["sum"].append(audit(X.sum(axis=1) / np.sqrt(T), present))
            res["path"].append(audit(a * (B * X).sum(axis=1) - a * a * B.sum(axis=1) / 2, present))
        for sc in ["sum", "path"]:
            v = np.array(res[sc])
            rows.append([law_name, sc, R, f"{v.mean():.3f}", f"{v.std():.3f}", f"{v.min():.3f}", f"{v.max():.3f}",
                         REF[law_name], f"{v.mean()/REF[law_name]:.3f}", int((v > REF[law_name]).sum()), int((v > POIS[law_name]).sum())])
            print(f"{law_name} {sc}: guess-counting eps {v.mean():.2f} ± {v.std():.2f} (recovery {v.mean()/REF[law_name]:.0%}); above ref {(v > REF[law_name]).sum()}/{R}; flags {(v > POIS[law_name]).sum()}/{R}", flush=True)
    with open("rev_guess.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["law", "score", "R", "mean", "sd", "min", "max", "reference", "recovery", "above_ref", "flags"]); w.writerows(rows)
