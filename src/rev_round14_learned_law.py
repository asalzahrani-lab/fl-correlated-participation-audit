"""Round 14: the audit with a participation law estimated from independent logs (the reviews of TIFS v13 asked what
happens to the certificate when the table is built from a law learned from finite logs, and what log size makes the
procedure safe).  A sensitivity study, not a guarantee: the conservative rule below rests on the monotonicity of the
certified curve in lambda that every table we built shows (a table for less dependence errs on the safe side,
Section V-F), which is observed, not proved.

Setting (Section V): canaries follow the two-state chain with rate q = 0.1 and lag-1 autocorrelation lambda = 0.9,
T = 200, a = 0.5, delta = 1e-4, m = 1,000 canaries, alpha_mc = 0.005.  The rate q is the cohort rate the operator sets
and is taken as known; only the dependence lambda is learned.

  table LAM      certified table for the chain with lambda = LAM (1M samples per grid point on 0.43..0.51, as in
                 rev_round7_audit_sens.py; seed 14100 + 1000*LAM), written to rev_round14_table_lam_<LAM>.csv.
  estimate       maximum-likelihood lambda = P(1->1) - P(0->1) from the logs of N in {100, 500, 2000} devices over
                 T rounds, 1,000 replicates each (seed 14200), with its delta-method standard error; written to
                 rev_round14_lamhat.csv.
  audit          one fresh audit data set per replicate (seed 14300 + replicate + 10 N), certified three ways with the tables
                 of the lambda grid: the oracle table (lambda = 0.9), the plug-in table (the grid point nearest to
                 lambda_hat; alpha_est = 0.045) and the conservative table (the largest grid point at or below
                 lambda_hat - z_{0.99} se, alpha_est = 0.035, so that alpha_law + alpha_est + alpha_mc = 0.05; the
                 auditor abstains, reporting 0, when that point lies below the grid).  Exceedances of the true loss
                 11.869 and of the upper 99% limit 12.033 of its 10M re-run, flags of the Poisson claim 2.866, recovery
                 and abstentions, per log size; written to rev_round14_learned_law.csv (summary) and
                 rev_round14_learned_law_audits.csv (one row per replicate).
  outfamily      out-of-family check: canaries follow the FLASH trace law (the running example, Section VI) while the
                 table is built for a two-state chain fitted to the trace logs (pooled rate and lag-1 autocorrelation of
                 the 572 devices' participation); 120 audits (seed 14400); written to rev_round14_outfamily.csv.
Run from results/: python3 ../src/rev_round14_learned_law.py table 0.9 ; ... ; estimate ; audit ; P3_TRACE=../trace
python3 ../src/rev_round14_learned_law.py outfamily
"""
import sys, os, csv, time
import numpy as np
from scipy.stats import norm
from corraudit import two_state_chain, eps_poisson_pld
from certified import ChainLaw
from certified2 import CertifiedTable2, snr_lower_bound

T, A_TRUE, DELTA, Q, M = 200, 0.5, 1e-4, 0.1, 1000
A_MC, A_EST_PLUGIN, A_EST_CONS, A_LAW = 0.005, 0.045, 0.035, 0.01
GRID = np.round(np.arange(0.43, 0.5101, 0.01), 2)
LAM_GRID = [0.86, 0.87, 0.875, 0.88, 0.885, 0.89, 0.895, 0.90, 0.905, 0.91, 0.915, 0.92, 0.93]
N_TABLE = int(os.environ.get("N_TABLE", 1_000_000))
EPS_TRUE, EPS_HI = 11.869, 12.033          # Table II: 10M reference and the upper 99% limit of the independent re-run
SIZES, R = (100, 500, 2000), 1000
t0 = time.time()
log = lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True)


def law(lam, q=Q):
    P, pi, emit = two_state_chain(q, lam); return ChainLaw(P, pi, emit, T)


def table_path(lam): return f"rev_round14_table_lam_{lam:.3f}.csv"


def build_table(lam, q=Q, grid=GRID, seed=None, path=None):
    seed = int(14100 + round(1000 * lam)) if seed is None else seed
    tab = CertifiedTable2(law(lam, q), grid, DELTA, N_TABLE, A_MC, np.random.default_rng(seed), chunk=25_000, log=log)
    with open(path or table_path(lam), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["a", "eps_plain", "eps_cert_step_0.005", "eps_cert_bern_0.005"])
        for a, ep, es, eb in zip(tab.a_grid, tab.eps_plain, tab.eps_step, tab.eps_bern):
            w.writerow([f"{a:.2f}", f"{ep:.4f}", f"{es:.4f}", f"{eb:.4f}"])
    return tab


class Table:
    def __init__(self, path):
        rows = list(csv.DictReader(open(path)))
        self.a = np.array([float(r["a"]) for r in rows]); self.c = np.array([float(r["eps_cert_bern_0.005"]) for r in rows])

    def cert(self, a):
        if not np.isfinite(a): return 0.0
        i = np.searchsorted(self.a - 1e-9, a, side="right") - 1
        return 0.0 if i < 0 else float(self.c[i])


def mle_lambda(B):
    B = B.astype(np.int8)
    n01 = int(((B[:, :-1] == 0) & (B[:, 1:] == 1)).sum()); n00 = int(((B[:, :-1] == 0) & (B[:, 1:] == 0)).sum())
    n11 = int(((B[:, :-1] == 1) & (B[:, 1:] == 1)).sum()); n10 = int(((B[:, :-1] == 1) & (B[:, 1:] == 0)).sum())
    n0, n1 = n00 + n01, n10 + n11
    p01, p11 = n01 / n0, n11 / n1
    se = float(np.sqrt(p01 * (1 - p01) / n0 + p11 * (1 - p11) / n1))
    return p11 - p01, se, n0, n1


def audit_data(L, rng):
    present = rng.random(M) < 0.5
    B = L.sample(M, rng)
    X = (B * present[:, None]) * A_TRUE + rng.standard_normal((M, T)).astype(np.float32)
    return X, present, B


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "table":
        lam = float(sys.argv[2]); log(f"table for lambda={lam} (q={Q}), {N_TABLE} samples per grid point")
        tab = build_table(lam); log(f"written {table_path(lam)}: cert at a=0.47 {tab.cert(0.47):.3f}, plain at 0.5 {tab.plain(0.5):.3f}")
    elif mode == "estimate":
        rng = np.random.default_rng(14200); L = law(0.9)
        with open("rev_round14_lamhat.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["devices", "replicate", "lambda_hat", "se", "n0", "n1"])
            for N in SIZES:
                ests = []
                for r in range(R):
                    lh, se, n0, n1 = mle_lambda(L.sample(N, rng)); ests.append(lh)
                    w.writerow([N, r, f"{lh:.6f}", f"{se:.6f}", n0, n1])
                ests = np.array(ests)
                log(f"N={N}: lambda_hat mean {ests.mean():.4f} sd {ests.std():.4f} (delta-method se of the first replicate {se:.4f}); "
                    f"P(lambda_hat > 0.91) = {(ests > 0.91).mean():.3f}, P(> 0.905) = {(ests > 0.905).mean():.3f}")
    elif mode == "audit":
        tabs = {lam: Table(table_path(lam)) for lam in LAM_GRID}
        ests = list(csv.DictReader(open("rev_round14_lamhat.csv")))
        pois = eps_poisson_pld(Q, T, 1.0, 1.0 / A_TRUE, DELTA); L = law(0.9); z = norm.ppf(1 - A_LAW)
        rows, summ = [], []
        for N in SIZES:
            sub = [e for e in ests if int(e["devices"]) == N]; assert len(sub) == R
            rec = {k: [] for k in ("oracle", "plugin", "cons")}; cnt = {k: dict(above=0, above_hi=0, flag=0, abstain=0) for k in rec}
            for e in sub:
                r = int(e["replicate"]); lh, se = float(e["lambda_hat"]), float(e["se"])
                X, present, B = audit_data(L, np.random.default_rng(14300 + r + 10 * N))
                alo_p = snr_lower_bound(X, present, B, A_EST_PLUGIN); alo_c = snr_lower_bound(X, present, B, A_EST_CONS)
                lam_near = float(LAM_GRID[int(np.argmin(np.abs(np.array(LAM_GRID) - lh)))])
                lam_lo = lh - z * se; below = [g for g in LAM_GRID if g <= lam_lo + 1e-12]
                est = {"oracle": tabs[0.9].cert(alo_p), "plugin": tabs[lam_near].cert(alo_p),
                       "cons": tabs[max(below)].cert(alo_c) if below else 0.0}
                if not below: cnt["cons"]["abstain"] += 1
                for k, v in est.items():
                    rec[k].append(v); cnt[k]["above"] += v > EPS_TRUE; cnt[k]["above_hi"] += v > EPS_HI; cnt[k]["flag"] += v > pois
                rows.append([N, r, f"{lh:.5f}", f"{se:.5f}", f"{lam_near:.3f}", f"{lam_lo:.5f}", f"{max(below):.3f}" if below else "",
                             f"{alo_p:.5f}", f"{alo_c:.5f}", f"{est['oracle']:.4f}", f"{est['plugin']:.4f}", f"{est['cons']:.4f}"])
            for k in rec:
                v = np.array(rec[k]); c = cnt[k]
                summ.append([N, k, R, f"{v.mean():.4f}", f"{v.std():.4f}", f"{v.mean() / EPS_TRUE:.4f}", c["above"], c["above_hi"], c["flag"], c["abstain"]])
                log(f"N={N} {k}: certified {v.mean():.2f} +- {v.std():.2f} (recovery {v.mean() / EPS_TRUE:.1%}); above 11.869: {c['above']}/{R}; "
                    f"above 12.033: {c['above_hi']}/{R}; flags {c['flag']}/{R}; abstentions {c['abstain']}")
        with open("rev_round14_learned_law_audits.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["devices", "replicate", "lambda_hat", "se", "lambda_plugin_table", "lambda_lo", "lambda_cons_table",
                                           "a_lo_0.045", "a_lo_0.035", "cert_oracle", "cert_plugin", "cert_cons"]); w.writerows(rows)
        with open("rev_round14_learned_law.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["devices", "rule", "replicates", "mean_cert", "sd_cert", "recovery", "above_true", "above_hi", "flags", "abstentions"]); w.writerows(summ)
    elif mode == "outfamily":
        from rev_common import window, apply_policy
        from certified import EmpiricalPathLaw
        A, _, _ = window(10, 3); B = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0))
        starts = np.arange(0, B.shape[1] - T + 1, 10); W = np.concatenate([B[:, s:s + T] for s in starts], axis=0).astype(np.int8)
        q_hat = W.mean(); lag1 = np.corrcoef(W[:, :-1].ravel(), W[:, 1:].ravel())[0, 1]
        lh, se, n0, n1 = mle_lambda(W)
        log(f"trace law: {W.shape[0]} windows, rate {q_hat:.4f}, pooled lag-1 {lag1:.4f}, chain MLE lambda {lh:.4f} (se {se:.4f})")
        grid = np.round(np.arange(0.36, 0.5501, 0.01), 2)
        tab = build_table(lh, q=q_hat, grid=grid, seed=14401, path="rev_round14_table_tracefit.csv")
        tab = Table("rev_round14_table_tracefit.csv")
        L = EmpiricalPathLaw(W); rng = np.random.default_rng(14400); pois = eps_poisson_pld(q_hat, T, 1.0, 1.0 / A_TRUE, DELTA)
        ref = 10.69; ests = []
        for _ in range(120):
            X, present, Bc = audit_data(L, rng); ests.append(tab.cert(snr_lower_bound(X, present, Bc, A_EST_PLUGIN)))
        e = np.array(ests)
        log(f"canaries: trace law (reference {ref}); table: fitted chain (q={q_hat:.4f}, lambda={lh:.3f}); certified {e.mean():.3f} +- {e.std():.3f} "
            f"(recovery {e.mean() / ref:.1%}); above the reference {int((e > ref).sum())}/120; flags of the Poisson claim {pois:.3f}: {int((e > pois).sum())}/120")
        with open("rev_round14_outfamily.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["canary_law", "table_law", "q_hat", "lambda_hat", "reference", "poisson", "R", "mean_cert", "sd_cert", "recovery", "above_ref", "flags"])
            w.writerow(["FLASH trace (fixed cohort 0.6)", "two-state chain fitted to the trace logs", f"{q_hat:.5f}", f"{lh:.5f}", ref, f"{pois:.4f}", 120,
                        f"{e.mean():.4f}", f"{e.std():.4f}", f"{e.mean() / ref:.4f}", int((e > ref).sum()), int((e > pois).sum())])
