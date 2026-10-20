"""Revision round 2, B3: the guess-counting audit of Steinke, Nasr and Jagielski with THREE scores, each matched to
the release it audits:
  sum       S_i = T^{-1/2} sum_t X_it                       -> a function of X only: audits the secret-path release;
  marginal  log L_Pi(X_i) (likelihood ratio of the secret-path pair, forward algorithm / log-sum-exp over paths)
                                                           -> a function of X only: audits the secret-path release,
                                                              and is the Neyman-Pearson statistic for it;
  path      a sum_t B_it X_it - a^2 N_i / 2  (likelihood ratio given the canary's own path)
                                                           -> uses the path B_i, which is part of the mechanism's
                                                              randomness: audits the JOINT release (B, X), whose loss is
                                                              Sum_n p_n h_eps(a sqrt n) (Proposition 3 / Theorem 3 of the full version v8).
Same audit data as rev_guess.py (same seeds), 120 audits per law, m = 1,000, delta = 1e-4, beta = 0.05 with
Bonferroni over K_SET.  Output rev_guess2.csv: mean, sd, recovery against the matched target, the Poisson claim of the
release tested (per-round PLD value for the secret path; count-averaged Binomial value for the joint release) and the
flags against it, and the cap (the eps certified by r = m correct guesses out of m, at the level used)."""
import csv, sys
import numpy as np
from scipy.stats import binom
from corraudit import two_state_chain, sample_paths, count_distribution
from certified import EmpiricalPathLaw, ChainLaw
from theory import eps_jensen_full
from rev_guess import K_SET, eps_lower, audit, T, a, m, R, DELTA, BETA, REF, POIS

if __name__ == "__main__":
    laws = sys.argv[1:] or ["markov", "iid", "trace"]
    from rev_common import window, apply_policy, window_paths
    A, _, _ = window(10, 3)
    trace_paths = window_paths(apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0)), T, 10)
    trace_law = EmpiricalPathLaw(trace_paths)
    pn_trace = np.bincount(trace_paths.sum(axis=1), minlength=T + 1) / len(trace_paths)
    JOINT = {}
    for name in ["markov", "iid"]:
        P, pi, emit = two_state_chain(0.1, 0.9 if name == "markov" else 0.0)
        JOINT[name] = eps_jensen_full(count_distribution(P, pi, emit, T), 1.0, 2.0, DELTA)
    JOINT["trace"] = eps_jensen_full(pn_trace, 1.0, 2.0, DELTA)
    # Poisson claim of each release: secret-path release -> the per-round Poisson PLD value (POIS); joint release ->
    # the count-averaged value under Binomial counts at the same mean rate (q = 0.1 for markov/iid, 0.0471 for trace).
    from scipy.stats import binom as _binom
    from corraudit import eps_poisson_pld as _epp
    from scipy.optimize import brentq as _brentq
    POIS_JOINT = {}
    for name in laws:
        q = _brentq(lambda q: _epp(q, T, 1.0, 2.0, DELTA) - POIS[name], 1e-3, 0.9)
        POIS_JOINT[name] = eps_jensen_full(_binom.pmf(np.arange(T + 1), T, q), 1.0, 2.0, DELTA)
    print(f"Poisson claims: secret-path {POIS}; joint release {POIS_JOINT}", flush=True)
    cap = eps_lower(2 * 500, 1000, BETA / len(K_SET)); cap1 = eps_lower(2 * 500, 1000, BETA)
    print(f"joint-release targets: {JOINT}; cap with 1,000 correct guesses: {cap:.3f} (Bonferroni over {len(K_SET)}), {cap1:.3f} (single k)", flush=True)
    rows = []
    for law_name in laws:
        rng = np.random.default_rng({"markov": 8101, "iid": 8102, "trace": 8103}[law_name])
        if law_name == "trace": law = trace_law
        else:
            P, pi, emit = two_state_chain(0.1, 0.9 if law_name == "markov" else 0.0); law = ChainLaw(P, pi, emit, T)
        res = {"sum": [], "marginal": [], "path": []}
        for rep in range(R):
            if law_name == "trace": B = trace_law.sample(m, rng)
            else: B = sample_paths(P, pi, emit, T, m, rng)
            B = B.astype(np.float32)
            present = rng.random(m) < 0.5
            X = a * present[:, None] * B + rng.standard_normal((m, T)).astype(np.float32)
            res["sum"].append(audit(X.sum(axis=1) / np.sqrt(T), present))
            res["marginal"].append(audit(law.loglr(X.astype(np.float64), a), present))
            res["path"].append(audit(a * (B * X).sum(axis=1) - a * a * B.sum(axis=1) / 2, present))
            if rep % 20 == 0: print(f"{law_name} rep {rep}: sum {res['sum'][-1]:.2f} marginal {res['marginal'][-1]:.2f} path {res['path'][-1]:.2f}", flush=True)
        for sc in ["sum", "marginal", "path"]:
            v = np.array(res[sc]); target = JOINT[law_name] if sc == "path" else REF[law_name]
            claim = POIS_JOINT[law_name] if sc == "path" else POIS[law_name]
            rows.append([law_name, sc, "joint (B,X)" if sc == "path" else "secret path X", R, f"{v.mean():.3f}", f"{v.std():.3f}", f"{v.min():.3f}", f"{v.max():.3f}",
                         f"{target:.3f}", f"{v.mean()/target:.3f}", int((v > target).sum()), f"{claim:.3f}", int((v > claim).sum())])
            print(f"{law_name} {sc}: {v.mean():.3f} ± {v.std():.3f}; target {target:.2f} (recovery {v.mean()/target:.0%}); above target {(v > target).sum()}/{R}; claim {claim:.2f}, flags {(v > claim).sum()}/{R}", flush=True)
    with open("rev_guess2.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["law", "score", "audited_release", "R", "mean", "sd", "min", "max", "target", "recovery", "above_target", "poisson_claim", "flags"]); w.writerows(rows)
        w.writerow(["cap_1000_correct_bonferroni8", f"{cap:.3f}"]); w.writerow(["cap_1000_correct_single_k", f"{cap1:.3f}"])
