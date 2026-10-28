"""Revision round 10 (internal review), M6: audit power as a function of the number of canaries.
Simulated certified path-aware audits (delta = 1e-4, T = 200, a = 0.5, 120 audits per cell) with m = 100, 200, 500,
1,000, 2,000 and 5,000 canaries under the Markov law (lambda = 0.9, q = 0.1) and the trace law of the running example.
Certified tables: the published ones (step14 + step16 entries, as in step19_rerun_audits.py) extended downwards so that
small audits do not fall below the grid: Markov a = 0.30..0.39 and trace a = 0.25..0.35, KL-certified at
alpha_mc = 0.005 from N_EXT shared samples (each entry is individually certified, so mixing sample sizes is valid).
Reported per cell: mean certified estimate, recovery against the reference (11.869 Markov, 10.685 trace), flags of the
Poisson claim, estimates above the reference, mean a_lo and its gap below a, mean n_obs, and the predicted gap
z_{1-alpha_est/2} / sqrt(n_obs) of the sizing rule.
Outputs rev_round10_canaries.csv and rev_round10_canaries_ext_tables.csv."""
import csv, sys, time
import numpy as np
from scipy.stats import norm
from corraudit import two_state_chain, eps_poisson_pld
from certified import ChainLaw, EmpiricalPathLaw
from certified2 import CertifiedTable2, snr_lower_bound
from rev_common import window, apply_policy, window_paths

T, a_true, R = 200, 0.5, 120
A_EST, A_MC = 0.045, 0.005
N_EXT = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
N_EXT_TRACE = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
MS = [100, 200, 500, 1000, 2000, 5000]
rng = np.random.default_rng(1011)
t0 = time.time(); log = lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True)


def read_table(path, col_cert):
    rows = list(csv.DictReader(open(path)))
    return {round(float(r["a"]), 2): float(r[col_cert]) for r in rows}


P, pi, emit = two_state_chain(0.1, 0.9)
A, _, _ = window(10, 3, verbose=False)
B = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0))
paths = window_paths(B, T, 10)
laws = {"markov": ChainLaw(P, pi, emit, T), "trace": EmpiricalPathLaw(paths)}
q = {"markov": 0.1, "trace": float(paths.mean())}
claim = {k: eps_poisson_pld(q[k], T, 1.0, 2.0, 1e-4) for k in laws}
refs = {"markov": 11.869, "trace": 10.685}
log(f"trace law: {len(paths)} paths, q {q['trace']:.4f}; claims {claim}")

tables = {}
for k, files, ext, n_ext, chunk in [("markov", [("step14_table_markov.csv", "eps_cert_bern_0.005"), ("step16_table_markov.csv", "eps_cert_1e-4")],
                                      np.round(np.arange(0.30, 0.3901, 0.01), 2), N_EXT, 25_000),
                                     ("trace", [("step14_table_trace.csv", "eps_cert_bern_0.005"), ("step16_table_trace.csv", "eps_cert_1e-4")],
                                      np.round(np.arange(0.25, 0.3501, 0.01), 2), N_EXT_TRACE, 4000)]:
    d = {}
    for path, col in files: d.update(read_table(path, col))
    log(f"{k}: building the extension a = {ext[0]:.2f}..{ext[-1]:.2f} from {n_ext} samples")
    tab = CertifiedTable2(laws[k], ext, 1e-4, n_ext, A_MC, rng, chunk=chunk, log=None)
    for a, e in zip(ext, tab.eps_bern):
        if round(float(a), 2) not in d: d[round(float(a), 2)] = float(e)
    with open("rev_round10_canaries_ext_tables.csv", "a", newline="") as f:
        w = csv.writer(f)
        for a, e, p in zip(ext, tab.eps_bern, tab.eps_plain): w.writerow([k, f"{a:.2f}", n_ext, f"{p:.4f}", f"{e:.4f}"])
    ag = np.array(sorted(d)); ce = np.array([d[x] for x in ag])
    if not np.all(np.diff(ce) >= -1e-9): log(f"{k}: note: certified entries not monotone across sample sizes (validity is per entry)")
    tables[k] = (ag, ce)
    log(f"{k}: table a = {ag[0]:.2f}..{ag[-1]:.2f} ({len(ag)} points), cert {ce[0]:.3f}..{ce[-1]:.3f}")

z = norm.ppf(1 - A_EST / 2)
rows = []
for k in ["markov", "trace"]:
    ag, ce = tables[k]
    for m in MS:
        est, alo, nobs = [], [], []
        for rep in range(R):
            present = rng.random(m) < 0.5
            Bc = laws[k].sample(m, rng)
            X = (Bc * present[:, None]) * a_true + rng.standard_normal((m, T)).astype(np.float32)
            a_c = snr_lower_bound(X, present, Bc, A_EST)
            i = np.searchsorted(ag - 1e-9, a_c, side="right") - 1 if np.isfinite(a_c) else -1
            est.append(0.0 if i < 0 else float(ce[i])); alo.append(a_c); nobs.append(float(Bc[present].sum()))
        est, alo, nobs = map(np.array, (est, alo, nobs))
        rows.append([k, m, R, f"{est.mean():.4f}", f"{est.std():.4f}", f"{est.mean() / refs[k]:.4f}", int(np.sum(est > claim[k])), int(np.sum(est > refs[k])),
                     int(np.sum(est == 0)), f"{alo.mean():.4f}", f"{1 - alo.mean() / a_true:.4f}", f"{nobs.mean():.0f}", f"{z / np.sqrt(nobs.mean()) / a_true:.4f}"])
        log(f"{k} m={m}: {est.mean():.3f}+-{est.std():.3f} recovery {est.mean() / refs[k]:.3f} flags {int(np.sum(est > claim[k]))}/{R} above {int(np.sum(est > refs[k]))} "
            f"zero {int(np.sum(est == 0))} | a_lo {alo.mean():.4f} ({1 - alo.mean() / a_true:.3f} below a; rule {z / np.sqrt(nobs.mean()) / a_true:.3f}) n_obs {nobs.mean():.0f}")
with open("rev_round10_canaries.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["law", "m", "R", "mean", "sd", "recovery", "flags", "above_reference", "zero_reports", "alo_mean", "alo_gap_rel",
                                   "nobs_mean", "rule_gap_rel"]); w.writerows(rows)
log("done")
