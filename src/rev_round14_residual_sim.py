"""Round 14: the operating region of the observation model -- controlled residuals added to the canary observations.

Model 1 says a canary observes X_t = a S B_t + Z_t with Z_t white standard normal.  In training the real clients'
updates add a residual along the canary direction.  Here a residual E_t is added to every observation of every canary
in the simulation of Section V (Markov law lambda = 0.9, q = 0.1, T = 200, a = 0.5, m = 1,000, table of
Table III, rev_round14_common.py), in six classes at relative sizes rho in {0.05, 0.10, 0.25} of the noise scale:
  none        E = 0 (the model; the null for the diagnostic rule)
  white       E_t iid N(0, rho^2): in the model with noise scale sqrt(1 + rho^2); reference eps(a / sqrt(1 + rho^2))
  mean        E_t = rho for all canaries and rounds: a constant offset, which the adversary subtracts, so the release's
              privacy is unchanged (reference 11.869); the regression estimate of a is biased by +rho
  ar          E_t = AR(1) with lag-1 autocorrelation phi in {0.5, 0.9} and marginal sd rho (plus the white noise): the
              noise is no longer white, the scalar reference does not describe the release; classed as sensitivity,
              exceedance counted against the nominal 11.869
  presence+   E_t = +rho on a present canary's participating rounds, 0 elsewhere: the real clients reinforce the
              canary direction when the canary is in, so the transcript carries signal a + rho; reference eps(a + rho)
  presence-   E_t = -rho on those rounds: the pull-back of Section V-F; reference eps(a - rho) (a loss, not a failure)
200 audits per cell (seed 14700 + cell), alpha_est = 0.045.  A diagnostic rule is frozen on the null cell before the
other cells are run: reject the audit (report 0) when any of the absent canaries' noise scale, lag-1 autocorrelation,
mean or KS distance, or the present canaries' erasure ratio, lies outside the null's mean +- 4 standard deviations
(KS: above the null's maximum).  A centred variant of the estimator (the absent canaries' mean subtracted from the shift, with the
variance term 1/n_0 in the limit) is certified alongside.  Reported per cell: the diagnostic rejection rate, exceedances of the cell's reference
and of the nominal 11.869 among all audits and among the audits the rule passes, the mean certificate and recovery
against the cell's reference, and flags of the Poisson claim 2.866.

    cd results && python3 ../src/rev_round14_residual_sim.py        # a few minutes
Outputs rev_round14_residual_sim.csv and rev_round14_residual_rule.csv (the frozen thresholds).
"""
import csv, time
import numpy as np
from scipy.stats import kstest
from corraudit import two_state_chain, eps_poisson_pld
from certified import ChainLaw
from certified2 import snr_lower_bound
from rev_round14_common import paper_table
from scipy.stats import norm, chi2


def snr_lower_bound_centred(X, present, B, alpha):
    """The estimator of eq. (9) with the absent canaries' mean subtracted from the shift (a constant offset along the
    canary direction then cancels); the normal limit carries the extra variance term 1/n_0.  Same zero-return convention."""
    present = np.asarray(present, dtype=bool); Bp = B[present].astype(float); Xp = X[present].astype(float); nobs = Bp.sum()
    xa = X[~present].astype(float).ravel(); n0 = len(xa)
    if nobs <= 0 or n0 < 2: return 0.0
    shift = (Bp * Xp).sum() / nobs - xa.mean(); sd = xa.std(ddof=1)
    sd_hi = sd * np.sqrt((n0 - 1) / chi2.ppf(alpha / 2, n0 - 1)); z = norm.ppf(1 - alpha / 2)
    if not np.isfinite(sd_hi) or sd_hi <= 0: return 0.0
    a_lo = max(shift - z * sd_hi * np.sqrt(1.0 / nobs + 1.0 / n0), 0.0) / sd_hi
    return float(a_lo) if np.isfinite(a_lo) else 0.0

T, A_TRUE, DELTA, Q, M, R = 200, 0.5, 1e-4, 0.1, 1000, 200
A_EST = 0.045; NOMINAL = 11.869; POIS = eps_poisson_pld(Q, T, 1.0, 1.0 / A_TRUE, DELTA)
RHOS = (0.05, 0.10, 0.25)
t0 = time.time()
log = lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True)



def diagnostics(X, present, B):
    Xa = X[~present]
    s_hat = float(Xa.std()); mean_a = float(Xa.mean())
    lag1 = float(np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in Xa[:200]]))
    ks = float(kstest(((Xa - mean_a) / s_hat).ravel(), "norm").statistic)
    S = X.sum(axis=1) / np.sqrt(T); counts = B.sum(axis=1)
    erasure = float(S[present].mean() / (A_TRUE * counts[present].mean() / np.sqrt(T)))
    return dict(s_hat=s_hat, lag1=lag1, mean_a=mean_a, ks=ks, erasure=erasure)


def residual(kind, rho, phi, present, B, rng):
    m = B.shape[0]
    if kind == "none": return np.zeros((m, T), np.float32)
    if kind == "white": return (rho * rng.standard_normal((m, T))).astype(np.float32)
    if kind == "mean": return np.full((m, T), rho, np.float32)
    if kind == "ar":
        E = np.empty((m, T)); E[:, 0] = rng.standard_normal(m)
        for t in range(1, T): E[:, t] = phi * E[:, t - 1] + np.sqrt(1 - phi * phi) * rng.standard_normal(m)
        return (rho * E).astype(np.float32)
    sign = 1.0 if kind == "presence+" else -1.0
    return (sign * rho * (B * present[:, None])).astype(np.float32)


if __name__ == "__main__":
    P, pi, emit = two_state_chain(Q, 0.9); L = ChainLaw(P, pi, emit, T); tab = paper_table("markov")
    cells = [("none", 0.0, 0.0)] + [(k, rho, 0.0) for k in ("white", "mean", "presence+", "presence-") for rho in RHOS] \
        + [("ar", rho, phi) for phi in (0.5, 0.9) for rho in RHOS]
    rule = None; rows = []
    for ci, (kind, rho, phi) in enumerate(cells):
        rng = np.random.default_rng(14700 + ci); ests, dgs, ests_c = [], [], []
        for _ in range(R):
            present = rng.random(M) < 0.5; B = L.sample(M, rng)
            X = (B * present[:, None]) * A_TRUE + rng.standard_normal((M, T)).astype(np.float32) + residual(kind, rho, phi, present, B, rng)
            ests.append(tab.cert(snr_lower_bound(X, present, B, A_EST))); ests_c.append(tab.cert(snr_lower_bound_centred(X, present, B, A_EST))); dgs.append(diagnostics(X, present, B))
        e = np.array(ests); ec = np.array(ests_c); keys = ("s_hat", "lag1", "mean_a", "erasure")
        if kind == "none":
            rule = {k: (np.mean([d[k] for d in dgs]), np.std([d[k] for d in dgs])) for k in keys}; rule["ks_max"] = max(d["ks"] for d in dgs)
            with open("rev_round14_residual_rule.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["statistic", "null_mean", "null_sd", "lower", "upper"])
                for k in keys: w.writerow([k, f"{rule[k][0]:.5f}", f"{rule[k][1]:.5f}", f"{rule[k][0] - 4 * rule[k][1]:.5f}", f"{rule[k][0] + 4 * rule[k][1]:.5f}"])
                w.writerow(["ks", "", "", "", f"{rule['ks_max']:.5f}"])
            log(f"rule frozen on the null: " + ", ".join(f"{k} {rule[k][0]:.4f}+-{rule[k][1]:.4f}" for k in keys) + f", ks max {rule['ks_max']:.4f}")
        passed = np.array([all(abs(d[k] - rule[k][0]) <= 4 * rule[k][1] for k in keys) and d["ks"] <= rule["ks_max"] for d in dgs])
        if kind in ("none", "mean", "ar"): ref = NOMINAL
        elif kind == "white": ref = tab.plain(A_TRUE / np.sqrt(1 + rho * rho))
        elif kind == "presence+": ref = tab.plain(min(A_TRUE + rho, tab.a[-1]))
        else: ref = tab.plain(max(A_TRUE - rho, tab.a[0]))
        ab_ref, ab_nom = int((e > ref).sum()), int((e > NOMINAL).sum())
        ab_ref_p, ab_nom_p = int((e[passed] > ref).sum()), int((e[passed] > NOMINAL).sum())
        rows.append([kind, rho, phi, R, f"{ref:.3f}", f"{e.mean():.4f}", f"{e.std():.4f}", f"{e.mean() / ref:.4f}", int((~passed).sum()), ab_ref, ab_nom,
                     int(passed.sum()), ab_ref_p, ab_nom_p, int((e > POIS).sum()), f"{np.mean([d['erasure'] for d in dgs]):.4f}", f"{np.mean([d['s_hat'] for d in dgs]):.4f}",
                     f"{np.mean([d['lag1'] for d in dgs]):.4f}", f"{np.mean([d['mean_a'] for d in dgs]):.4f}", f"{np.mean([d['ks'] for d in dgs]):.5f}",
                     f"{ec.mean():.4f}", f"{ec.mean() / ref:.4f}", int((ec > ref).sum()), int((ec > NOMINAL).sum())])
        log(f"{kind} rho={rho} phi={phi}: reference {ref:.2f}; certified {e.mean():.2f} +- {e.std():.2f} ({e.mean() / ref:.1%}); rejected by the rule {int((~passed).sum())}/{R}; "
            f"above reference {ab_ref} (passers {ab_ref_p}/{int(passed.sum())}); above nominal 11.869 {ab_nom} (passers {ab_nom_p}); flags {int((e > POIS).sum())}")
    with open("rev_round14_residual_sim.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class", "rho", "phi", "R", "reference", "mean_cert", "sd_cert", "recovery", "rejected", "above_ref_all", "above_nominal_all",
                                       "passed", "above_ref_passed", "above_nominal_passed", "flags", "erasure_mean", "s_hat_mean", "lag1_mean", "absent_mean", "ks_mean",
                                       "mean_cert_centred", "recovery_centred", "above_ref_centred", "above_nominal_centred"]); w.writerows(rows)
    log("done")
