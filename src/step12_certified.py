"""Step 12: certified path-aware auditor on (A) the two-state Markov law (lambda=0.9) and (B) the empirical path law
from the FLASH trace (fixed cohort K = 0.6 x mean pool, 10-min rounds), T = 200, a = 0.5, delta = 1e-4.
Null runs: >= 100 audits; false flag = reported eps above the high-precision reference."""
import numpy as np, csv, time, sys, json
import os; sys.path.insert(0, os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")))
from corraudit import two_state_chain, eps_poisson_pld
from certified import *

T, a_true, DELTA, q = 200, 0.5, 1e-4, 0.1
ALPHA_EST, ALPHA_MC = 0.025, 0.025      # total 0.05
N_TABLE = 500_000
R = 120
m = 1000
rng = np.random.default_rng(2027)
a_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
t0 = time.time()
log = lambda s: print(f"{s} [{time.time()-t0:.0f}s]", flush=True)

# ---------- (B) empirical path law from the trace: rebuild the policy paths
from trace_analysis import ready_intervals, availability_grid
d = json.load(open(os.path.join(os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")), "state_traces.json")))
devs = []
for k, v in d.items():
    iv, tf, tl = ready_intervals(v["messages"])
    if tf is None or tl - tf < 2 * 86400: continue
    devs.append((k, iv, tf, tl))
step = 600
t0_ = min(tf for (_, _, tf, _) in devs); t1_ = max(tl for (_, _, _, tl) in devs)
times = np.arange(t0_, t1_, step); Rn = len(times)
Amat = np.zeros((len(devs), Rn), dtype=bool); cover = np.zeros((len(devs), Rn), dtype=bool)
for i, (k, iv, tf, tl) in enumerate(devs):
    Amat[i] = availability_grid(iv, t0_, t1_, step); cover[i] = (times >= tf) & (times <= tl)
W = 3 * 144
best = max(range(0, Rn - W, 36), key=lambda s0: (cover[:, s0:s0 + W].all(axis=1)).sum())
sel = cover[:, best:best + W].all(axis=1)
A = Amat[sel, best:best + W]; N, Tw = A.shape
rng_pol = np.random.default_rng(0)
Bpol = np.zeros_like(A); Epool = A.sum(axis=0).mean(); K = int(round(0.6 * Epool))
for t in range(Tw):
    idx = np.flatnonzero(A[:, t])
    chosen = idx if len(idx) <= K else rng_pol.choice(idx, K, replace=False)
    Bpol[chosen, t] = True
paths = []
for i in range(N):
    for s0 in range(0, Tw - T + 1, 10):
        paths.append(Bpol[i, s0:s0 + T])
paths = np.array(paths, dtype=np.int8)
keep = paths.sum(axis=1) > 0
q_emp = paths.mean()
log(f"trace law: {len(paths)} paths ({keep.sum()} with >=1 participation), mean rate q={q_emp:.4f}, count mean {paths.sum(axis=1).mean():.2f} var {paths.sum(axis=1).var():.2f}")
law_trace = EmpiricalPathLaw(paths)
P, pi, emit = two_state_chain(q, 0.9)
law_markov = ChainLaw(P, pi, emit, T)

out = open("step12_results.csv", "w", newline=""); w = csv.writer(out)
w.writerow(["law", "reference_plain_eps_2M", "reference_cert_eps_2M", "poisson_eps_same_q", "estimator", "R", "mean", "sd", "false_flag_rate_vs_ref", "flag_poisson"])
for name, law, qq in [("markov0.9", law_markov, q), ("trace_cohort0.6", law_trace, q_emp)]:
    log(f"== {name}: building certified table (n={N_TABLE} per a)")
    table = CertifiedTable(law, a_grid, DELTA, N_TABLE, ALPHA_MC, rng, log=log)
    # high-precision reference at the true a
    lps, lqs = [], []; done = 0
    while done < 2_000_000:
        k = min(25_000, 2_000_000 - done)
        B = law.sample(k, rng)
        lps.append(law.loglr(B * a_true + rng.standard_normal((k, T)).astype(np.float32), a_true))
        lqs.append(law.loglr(rng.standard_normal((k, T)).astype(np.float32), a_true)); done += k
    lp, lq = np.concatenate(lps), np.concatenate(lqs)
    ref_plain = plain_eps(lp, lq, DELTA); ref_cert = eps_from_lower_curve(delta_lo_curve(lp, lq, 0.05), DELTA)
    ep = eps_poisson_pld(qq, T, 1.0, 1.0 / a_true, DELTA)
    log(f"{name}: reference eps (2M samples) plain={ref_plain:.3f} certified-lower={ref_cert:.3f}; Poisson at q={qq:.4f}: {ep:.3f}; table plain(0.5)={table.plain(0.5):.3f} cert(0.5)={table.cert(0.5):.3f}")
    res = {"certified": [], "plain_interp": []}
    for rep in range(R):
        present = rng.random(m) < 0.5
        B = law.sample(m, rng)
        X = (B * present[:, None]) * a_true + rng.standard_normal((m, T)).astype(np.float32)
        a_lo = snr_lower_bound(X, present, B, ALPHA_EST)
        res["certified"].append(table.cert(a_lo)); res["plain_interp"].append(table.plain(a_lo))
    for est, v in res.items():
        v = np.array(v)
        w.writerow([name, f"{ref_plain:.4f}", f"{ref_cert:.4f}", f"{ep:.4f}", est, R, f"{v.mean():.4f}", f"{v.std():.4f}", f"{np.mean(v > ref_plain):.3f}", f"{np.mean(v > ep):.3f}"])
        log(f"{name} {est:12s}: {v.mean():.3f}±{v.std():.3f} (ref {ref_plain:.3f}) false-flag {np.mean(v > ref_plain):.3f} flag-Poisson {np.mean(v > ep):.2f} ratio {v.mean()/ref_plain:.2f}")
    out.flush()
out.close()
