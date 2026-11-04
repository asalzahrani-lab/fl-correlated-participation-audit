"""Round 15: one record per audit for the noise-error and residual simulations of round 14, and independent null
calibration cells -- the inputs of rev_round15_screening.py.

The round-14 scripts stored per-cell aggregates only.  This script regenerates every audit of
rev_round14_noise_sim.py (24 cells x 120 audits, seeds 14500 + cell, the same random stream in the same order) and of
rev_round14_residual_sim.py (19 cells x 200 audits, seeds 14700 + cell) and writes one row per audit:

  identity      experiment, cell, audit_id, rep, law, m, T, noise_r, residual class, rho, phi, cell seed, table_id
                (sha256 of the merged certified table's grid and certificates, rev_round14_common.paper_table)
  claim         the correct accountant's value at the declared noise (11.869 / 2.866 / 10.69), its documented upper
                99% limit where one exists (12.033 for the Markov law; the trace re-run's upper limit equals its
                claim), and the cell's reference (the realised loss eps(a / r) of rev_round14_noise_sim.csv, copied
                from that file; the residual cell's reference as rev_round14_residual_sim.py defines it)
  estimates     a_lo and the certificate (eq. (11)), the centred variant, all at full precision; the raw flag of the
                claim (strict exceedance), the flag of the upper limit, the exceedance of the reference
  diagnostics   the five statistics of the frozen rule with the definitions of rev_round14_residual_sim.diagnostics
                (s_hat: population sd of the pooled absent-canary observations in declared-noise units; lag1_200: mean
                of the per-canary Pearson lag-1 correlation over the first 200 absent canaries; mean_a: pooled absent
                mean; ks: Kolmogorov-Smirnov distance of (x - mean_a) / s_hat to N(0, 1); erasure: mean present-canary
                sum score over a N-bar / sqrt T), plus lag1_all (over every absent canary, the definition of
                rev_train.diagnostics) and mean_a_norm = mean_a / s_hat (the scale-normalised mean of the rule-B form)
  sufficient    n0 (absent observations), n_abs (absent canaries), sum_a, sumsq_a, lagsum_a (pooled sum of x_t x_{t+1}
  statistics    over absent canaries), n_obs (present canaries' participating observations), sum_obs (their sum),
                n_pres, sum_counts_pres, sum_S_pres (sum of the present canaries' sum scores S_i).  From these a_lo,
                the centred a_lo, s_hat, mean_a and erasure are recomputed exactly (rev_round15_screening.py checks
                this on every row); lag1 and ks need the full sample and are reproduced by re-running with the seed.

Null calibration cells (experiment "null"): 200 audits at the declared noise with no residual for (Markov, 1,000),
(Markov, 200), (i.i.d., 1,000) and (trace, 1,000), seeds 15000 to 15003, generated after the test cells but never
used by them: the per-configuration rules of rev_round15_screening.py are frozen on these cells alone.

    cd results && P3_TRACE=../trace python3 ../src/rev_round15_records.py          # about 8 minutes
Output rev_round15_records.csv (2,880 + 3,800 + 800 rows) and rev_round15_records_cells.csv (per-cell mean and sd of
the certificate, the number of certificates above the cell's claim -- the flags of rev_round14_noise_sim.csv for the
noise cells, the exceedances of the nominal 11.869 for the residual cells -- and above the cell's reference, which must
equal the corresponding columns of rev_round14_noise_sim.csv / rev_round14_residual_sim.csv).
"""
import csv, hashlib, time, os
import numpy as np
from scipy.stats import kstest
from corraudit import two_state_chain, iid_chain, eps_poisson_pld
from certified import ChainLaw, EmpiricalPathLaw
from certified2 import snr_lower_bound
from rev_common import window, apply_policy
from rev_round14_common import paper_table
from rev_round14_residual_sim import snr_lower_bound_centred, residual, RHOS

T, A_TRUE, DELTA, Q = 200, 0.5, 1e-4, 0.1
A_EST = 0.045
RS = [1.10, 1.00, 0.98, 0.95, 0.90, 0.80]
CLAIM = {"markov": 11.869, "iid": 2.866, "trace": 10.69}
UPPER = {"markov": 12.033, "iid": float("nan"), "trace": 10.69}
NOMINAL = 11.869
t0 = time.time()
log = lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True)

COLS = ["experiment", "cell", "audit_id", "rep", "law", "m", "T", "noise_r", "residual", "rho", "phi", "seed", "table_id",
        "claim", "claim_upper", "reference", "a_lo", "cert", "a_lo_c", "cert_c", "flag_claim", "flag_upper", "above_reference",
        "s_hat", "lag1_200", "lag1_all", "mean_a", "mean_a_norm", "ks", "erasure",
        "n0", "n_abs", "sum_a", "sumsq_a", "lagsum_a", "n_obs", "sum_obs", "n_pres", "sum_counts_pres", "sum_S_pres"]


def table_id(tab):
    h = hashlib.sha256(); h.update(np.ascontiguousarray(tab.a, np.float64).tobytes()); h.update(np.ascontiguousarray(tab.c, np.float64).tobytes())
    return h.hexdigest()[:12]


def record(X, present, B):
    """Estimates, diagnostics and sufficient statistics of one audit (X in declared-noise units)."""
    Xa = X[~present].astype(np.float64); xa = Xa.ravel(); n0 = xa.size
    s_hat = float(Xa.std()); mean_a = float(Xa.mean())
    lag1_200 = float(np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in Xa[:200]]))
    lag1_all = float(np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in Xa]))
    ks = float(kstest((xa - mean_a) / s_hat, "norm").statistic)
    S = X.astype(np.float64).sum(axis=1) / np.sqrt(T); counts = B.sum(axis=1)
    erasure = float(S[present].mean() / (A_TRUE * counts[present].mean() / np.sqrt(T)))
    Bp = B[present].astype(np.float64); Xp = X[present].astype(np.float64)
    a_lo = snr_lower_bound(X, present, B, A_EST); a_lo_c = snr_lower_bound_centred(X, present, B, A_EST)
    return dict(a_lo=a_lo, a_lo_c=a_lo_c, s_hat=s_hat, lag1_200=lag1_200, lag1_all=lag1_all, mean_a=mean_a, mean_a_norm=mean_a / s_hat, ks=ks, erasure=erasure,
                n0=n0, n_abs=int(Xa.shape[0]), sum_a=float(xa.sum()), sumsq_a=float((xa * xa).sum()), lagsum_a=float((Xa[:, :-1] * Xa[:, 1:]).sum()),
                n_obs=int(Bp.sum()), sum_obs=float((Bp * Xp).sum()), n_pres=int(present.sum()), sum_counts_pres=int(counts[present].sum()),
                sum_S_pres=float(S[present].sum()))


def row(exp, cell, rep, law, m, r, kind, rho, phi, seed, tid, claim, upper, ref, tab, rec):
    cert = tab.cert(rec["a_lo"]); cert_c = tab.cert(rec["a_lo_c"])
    aid = f"{exp}-{cell:02d}-{rep:03d}"
    return [exp, cell, aid, rep, law, m, T, f"{r:.2f}", kind, rho, phi, seed, tid, f"{claim:.3f}", f"{upper:.3f}", f"{ref:.4f}",
            repr(rec["a_lo"]), repr(cert), repr(rec["a_lo_c"]), repr(cert_c), int(cert > claim), int(cert > upper) if np.isfinite(upper) else "", int(cert > ref),
            repr(rec["s_hat"]), repr(rec["lag1_200"]), repr(rec["lag1_all"]), repr(rec["mean_a"]), repr(rec["mean_a_norm"]), repr(rec["ks"]), repr(rec["erasure"]),
            rec["n0"], rec["n_abs"], repr(rec["sum_a"]), repr(rec["sumsq_a"]), repr(rec["lagsum_a"]), rec["n_obs"], repr(rec["sum_obs"]), rec["n_pres"],
            rec["sum_counts_pres"], repr(rec["sum_S_pres"])]


if __name__ == "__main__":
    laws = {}
    P, pi, emit = two_state_chain(Q, 0.9); laws["markov"] = ChainLaw(P, pi, emit, T)
    P, pi, emit = iid_chain(Q); laws["iid"] = ChainLaw(P, pi, emit, T)
    A, _, _ = window(10, 3); Bt = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0))
    starts = np.arange(0, Bt.shape[1] - T + 1, 10); W = np.concatenate([Bt[:, s:s + T] for s in starts], axis=0).astype(np.int8)
    laws["trace"] = EmpiricalPathLaw(W)
    tabs = {k: paper_table(k) for k in laws}; tids = {k: table_id(tabs[k]) for k in laws}
    log("tables: " + ", ".join(f"{k} {tids[k]} ({len(tabs[k].a)} grid points)" for k in tabs))
    realised = {(r_["law"], int(r_["m"]), float(r_["noise_r"])): float(r_["realised_loss"]) for r_ in csv.DictReader(open("rev_round14_noise_sim.csv"))}
    out = open("rev_round15_records.csv", "w", newline=""); w = csv.writer(out); w.writerow(COLS)
    cells_out = []

    # ---- (1) the noise-error simulation, seeds 14500 + cell, the stream of rev_round14_noise_sim.py
    cell = 0
    for name in ("markov", "iid", "trace"):
        for m in ((1000, 200) if name == "markov" else (1000,)):
            for r in RS:
                L, tab = laws[name], tabs[name]; seed = 14500 + cell; rng = np.random.default_rng(seed)
                ref = realised[(name, m, r)]; recs = []
                for rep in range(120):
                    present = rng.random(m) < 0.5; B = L.sample(m, rng)
                    X = (B * present[:, None]) * A_TRUE + r * rng.standard_normal((m, T)).astype(np.float32)
                    rec = record(X, present, B); recs.append(rec)
                    w.writerow(row("noise", cell, rep, name, m, r, "none", 0.0, 0.0, seed, tids[name], CLAIM[name], UPPER[name], ref, tab, rec))
                e = np.array([tab.cert(x["a_lo"]) for x in recs])
                cells_out.append(["noise", cell, name, m, r, "none", 0, 0, 120, f"{e.mean():.4f}", f"{e.std():.4f}", int((e > CLAIM[name]).sum()), int((e > ref).sum())])
                log(f"noise cell {cell} {name} m={m} r={r}: certified {e.mean():.4f} +- {e.std():.4f}; flags {int((e > CLAIM[name]).sum())}; above realised {int((e > ref).sum())}; "
                    f"s_hat mean {np.mean([x['s_hat'] for x in recs]):.4f}; ks mean {np.mean([x['ks'] for x in recs]):.5f}")
                out.flush(); cell += 1

    # ---- (2) the residual simulation, seeds 14700 + cell, the stream of rev_round14_residual_sim.py (Markov, m = 1,000)
    L, tab = laws["markov"], tabs["markov"]
    cells = [("none", 0.0, 0.0)] + [(k, rho, 0.0) for k in ("white", "mean", "presence+", "presence-") for rho in RHOS] \
        + [("ar", rho, phi) for phi in (0.5, 0.9) for rho in RHOS]
    for ci, (kind, rho, phi) in enumerate(cells):
        seed = 14700 + ci; rng = np.random.default_rng(seed); recs = []
        if kind in ("none", "mean", "ar"): ref = NOMINAL
        elif kind == "white": ref = tab.plain(A_TRUE / np.sqrt(1 + rho * rho))
        elif kind == "presence+": ref = tab.plain(min(A_TRUE + rho, tab.a[-1]))
        else: ref = tab.plain(max(A_TRUE - rho, tab.a[0]))
        for rep in range(200):
            present = rng.random(1000) < 0.5; B = L.sample(1000, rng)
            X = (B * present[:, None]) * A_TRUE + rng.standard_normal((1000, T)).astype(np.float32) + residual(kind, rho, phi, present, B, rng)
            rec = record(X, present, B); recs.append(rec)
            w.writerow(row("residual", ci, rep, "markov", 1000, 1.0, kind, rho, phi, seed, tids["markov"], NOMINAL, UPPER["markov"], ref, tab, rec))
        e = np.array([tab.cert(x["a_lo"]) for x in recs]); ec = np.array([tab.cert(x["a_lo_c"]) for x in recs])
        cells_out.append(["residual", ci, "markov", 1000, 1.0, kind, rho, phi, 200, f"{e.mean():.4f}", f"{e.std():.4f}", int((e > NOMINAL).sum()), int((e > ref).sum())])
        log(f"residual cell {ci} {kind} rho={rho} phi={phi}: reference {ref:.3f}; certified {e.mean():.4f} +- {e.std():.4f}; above reference {int((e > ref).sum())}; "
            f"centred {ec.mean():.4f}, above {int((ec > ref).sum())}")
        out.flush()

    # ---- (3) independent null calibration cells, seeds 15000 + k
    for k, (name, m) in enumerate((("markov", 1000), ("markov", 200), ("iid", 1000), ("trace", 1000))):
        L, tab = laws[name], tabs[name]; seed = 15000 + k; rng = np.random.default_rng(seed); recs = []
        for rep in range(200):
            present = rng.random(m) < 0.5; B = L.sample(m, rng)
            X = (B * present[:, None]) * A_TRUE + rng.standard_normal((m, T)).astype(np.float32)
            rec = record(X, present, B); recs.append(rec)
            w.writerow(row("null", k, rep, name, m, 1.0, "none", 0.0, 0.0, seed, tids[name], CLAIM[name], UPPER[name], CLAIM[name], tab, rec))
        e = np.array([tab.cert(x["a_lo"]) for x in recs])
        cells_out.append(["null", k, name, m, 1.0, "none", 0, 0, 200, f"{e.mean():.4f}", f"{e.std():.4f}", int((e > CLAIM[name]).sum()), int((e > CLAIM[name]).sum())])
        log(f"null cell {k} {name} m={m}: certified {e.mean():.4f} +- {e.std():.4f}; s_hat {np.mean([x['s_hat'] for x in recs]):.5f} +- {np.std([x['s_hat'] for x in recs]):.5f}; "
            f"ks max {max(x['ks'] for x in recs):.5f}")
        out.flush()
    out.close()
    with open("rev_round15_records_cells.csv", "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["experiment", "cell", "law", "m", "noise_r", "residual", "rho", "phi", "R", "mean_cert", "sd_cert", "above_claim", "above_reference"]); cw.writerows(cells_out)
    log("done")
