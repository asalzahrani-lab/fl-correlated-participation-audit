"""Revision round 7 (new reviewer): power of the certified audit at small violations, and its sensitivity to a
misspecified participation law.

Setting as in Section V of the paper: two-state chain with stationary rate q = 0.1, T = 200, a = 0.5 (sigma = 2),
delta = 1e-4, m = 1,000 canaries with fair presence coins, alpha_est = 0.045, alpha_mc = 0.005, KL-certified table on a
0.01 grid, Poisson claim 2.866 (dp_accounting PLD at q = 0.1).

  power LAM     canaries follow the chain with lambda = LAM (0.3, 0.6, 0.8): reference eps_true at a = 0.5 with a
                two-sided 99% interval (2M samples, separate stream), certified table for the same law (1M samples per
                grid point, 0.43..0.51), 120 audits; flags = estimates above the Poisson claim.
  misspec LHAT  canaries follow lambda = 0.9 (the paper's law) but the auditor's table is built for lambda = LHAT
                (0.85, 0.89, 0.91, 0.95), as if the law had been estimated with that error; the 120 audit data sets are
                the same for every LHAT (one stream), so the rows differ only by the table.  Reports eps_true(LHAT), the
                estimates, and how many exceed the true eps_true(0.9) = 11.87 (a validity failure for the deployment's
                law) and the claim.
  lamse         sampling error of the maximum-likelihood lambda from the participation logs of m = 1,000 devices over
                T = 200 rounds (200 replicates).
  combine       collects the CSV parts into rev_round7_audit_sens.csv and prints the summary.
Per-audit estimates are saved as rev_round7_power_<LAM>_est.npy and rev_round7_misspec_<LHAT>_est.npy (the release
keeps those of the three power runs and of the correctly specified table, LHAT = 0.9).
Seeds: tables 9100 + k, references 9200 + k, power audits 9300 + k, misspecification audits 9399, lamse 9400; the table
and reference streams are disjoint from the audit streams."""
import sys, os, csv, glob, time
import numpy as np
from corraudit import two_state_chain, eps_poisson_pld
from certified import ChainLaw
from certified2 import CertifiedTable2, snr_lower_bound
from rev_common import eps_interval

T, A_TRUE, DELTA, Q, M = 200, 0.5, 1e-4, 0.1, 1000
R = int(os.environ.get("R_AUD", 120))
A_EST, A_MC = 0.045, 0.005
GRID = np.round(np.arange(0.43, 0.5101, 0.01), 2)
N_TABLE, N_REF = int(os.environ.get("N_TABLE", 1_000_000)), int(os.environ.get("N_REF", 2_000_000))
EPS_TRUE_09 = 11.869                     # the paper's 10M-sample reference for lambda = 0.9 (Table II)
t0 = time.time()
log = lambda s: print(f"{s} [{time.time()-t0:.0f}s]", flush=True)
POIS = eps_poisson_pld(Q, T, 1.0, 1.0 / A_TRUE, DELTA)
K = {0.3: 1, 0.6: 2, 0.8: 3, 0.85: 4, 0.89: 5, 0.91: 6, 0.95: 7, 0.9: 8}


def law(lam):
    P, pi, emit = two_state_chain(Q, lam); return ChainLaw(P, pi, emit, T)


def reference(lam):
    L = law(lam); rng = np.random.default_rng(9200 + K[lam]); lp = np.empty(N_REF); lq = np.empty(N_REF); ch = 25_000
    for s in range(0, N_REF, ch):
        B = L.sample(ch, rng).astype(float)
        lp[s:s + ch] = L.loglr(A_TRUE * B + rng.standard_normal((ch, T)), A_TRUE)
        lq[s:s + ch] = L.loglr(rng.standard_normal((ch, T)), A_TRUE)
    return eps_interval(lp, lq, DELTA)


def table(lam):
    return CertifiedTable2(law(lam), GRID, DELTA, N_TABLE, A_MC, np.random.default_rng(9100 + K[lam]), chunk=25_000, log=log)


def audits(lam_canary, seed):
    L = law(lam_canary); rng = np.random.default_rng(seed); out = []
    for _ in range(R):
        present = rng.random(M) < 0.5
        B = L.sample(M, rng)
        X = (B * present[:, None]) * A_TRUE + rng.standard_normal((M, T)).astype(np.float32)
        out.append(snr_lower_bound(X, present, B, A_EST))
    return np.array(out)


def write(name, rows):
    with open(name, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["mode", "lambda_canaries", "lambda_table", "eps_true_table_law", "ci99_lo", "ci99_hi",
                                       "eps_true_canary_law", "poisson", "ratio_true_over_poisson", "R", "mean_eps_hat", "sd_eps_hat",
                                       "recovery", "above_true", "flags", "a_lo_mean"]); w.writerows(rows)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "power":
        lam = float(sys.argv[2]); ref, lo, hi = reference(lam); log(f"lambda={lam}: eps_true {ref:.3f} [{lo:.3f}, {hi:.3f}], Poisson {POIS:.3f}, ratio {ref/POIS:.3f}")
        tab = table(lam); alo = audits(lam, 9300 + K[lam]); e = np.array([tab.cert(x) for x in alo])
        np.save(f"rev_round7_power_{lam}_est.npy", e); log(f"largest estimate {e.max():.3f}; above the point reference {int((e > ref).sum())}/{R}")
        row = ["power", lam, lam, f"{ref:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{ref:.4f}", f"{POIS:.4f}", f"{ref/POIS:.3f}", R,
               f"{e.mean():.4f}", f"{e.std():.4f}", f"{e.mean()/ref:.3f}", int((e > hi).sum()), int((e > POIS).sum()), f"{alo.mean():.4f}"]
        log(f"lambda={lam}: certified {e.mean():.3f} +- {e.std():.3f} (recovery {e.mean()/ref:.3f}); above upper limit of reference {int((e > hi).sum())}; flags {int((e > POIS).sum())}/{R}")
        write(f"rev_round7_power_{lam}.csv", [row])
    elif mode == "misspec":
        lh = float(sys.argv[2]); ref, lo, hi = reference(lh); log(f"table law lambda={lh}: eps_true {ref:.3f} [{lo:.3f}, {hi:.3f}]")
        tab = table(lh); alo = audits(0.9, 9399); e = np.array([tab.cert(x) for x in alo])
        np.save(f"rev_round7_misspec_{lh}_est.npy", e); log(f"largest estimate {e.max():.3f}; above the upper 99% limit 12.03 of the 10M re-run: {int((e > 12.033).sum())}/{R}")
        row = ["misspec", 0.9, lh, f"{ref:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{EPS_TRUE_09:.4f}", f"{POIS:.4f}", f"{EPS_TRUE_09/POIS:.3f}", R,
               f"{e.mean():.4f}", f"{e.std():.4f}", f"{e.mean()/EPS_TRUE_09:.3f}", int((e > EPS_TRUE_09).sum()), int((e > POIS).sum()), f"{alo.mean():.4f}"]
        log(f"canaries lambda=0.9, table lambda={lh}: certified {e.mean():.3f} +- {e.std():.3f} (vs true 11.869: {e.mean()/EPS_TRUE_09:.3f}); above 11.869: {int((e > EPS_TRUE_09).sum())}/{R}; flags {int((e > POIS).sum())}/{R}")
        write(f"rev_round7_misspec_{lh}.csv", [row])
    elif mode == "lamse":
        rng = np.random.default_rng(9400); P, pi, emit = two_state_chain(Q, 0.9); L = ChainLaw(P, pi, emit, T); est = []
        for _ in range(200):
            B = L.sample(M, rng).astype(int)
            n01 = ((B[:, :-1] == 0) & (B[:, 1:] == 1)).sum(); n00 = ((B[:, :-1] == 0) & (B[:, 1:] == 0)).sum()
            n11 = ((B[:, :-1] == 1) & (B[:, 1:] == 1)).sum(); n10 = ((B[:, :-1] == 1) & (B[:, 1:] == 0)).sum()
            p01 = n01 / (n00 + n01); p11 = n11 / (n10 + n11); est.append(p11 - p01)   # lambda = P(1->1) - P(0->1)
        est = np.array(est)
        log(f"MLE of lambda from {M} devices x {T} rounds: mean {est.mean():.4f}, sd {est.std():.4f}, 99% range [{np.quantile(est, .005):.4f}, {np.quantile(est, .995):.4f}]")
        with open("rev_round7_lamse.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["devices", "T", "replicates", "lambda", "mean_hat", "sd_hat", "q005", "q995"])
            w.writerow([M, T, 200, 0.9, f"{est.mean():.5f}", f"{est.std():.5f}", f"{np.quantile(est, .005):.5f}", f"{np.quantile(est, .995):.5f}"])
    elif mode == "combine":
        rows = []
        for f in sorted(glob.glob("rev_round7_power_*.csv")) + sorted(glob.glob("rev_round7_misspec_*.csv")):
            rows += list(csv.reader(open(f)))[1:]
        write("rev_round7_audit_sens.csv", rows)
        for r in rows: print(", ".join(r))
