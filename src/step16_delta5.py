"""Step 16: deployment-grade delta.  Everything at delta = 1e-5 (and 1e-4 from the same samples for continuity):
Poisson PLD eps, exact final-model (sum) eps from the count law (Theorem 1), plain Monte-Carlo reference of the
per-round release, KL-certified table, and 120 merged-certified audits.
usage: python step16_delta5.py markov|trace [n_mc] [R]"""
import numpy as np, csv, time, sys, json
import os; sys.path.insert(0, os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")))
from scipy.stats import binom
from corraudit import two_state_chain, eps_poisson_pld, eps_sum_exact, count_distribution
from certified import EmpiricalPathLaw, ChainLaw, plain_eps
from certified2 import CertifiedTable2, snr_lower_bound, eps_cert_bern

which = sys.argv[1]
T, a_true, q = 200, 0.5, 0.1
N_MC = int(sys.argv[2]) if len(sys.argv) > 2 else (10_000_000 if which == "markov" else 3_000_000)
R = int(sys.argv[3]) if len(sys.argv) > 3 else 120
A_EST, A_MC = 0.045, 0.005
m = 1000
rng = np.random.default_rng(3031 if which == "markov" else 3032)
a_grid = np.round(np.arange(0.44, 0.5201, 0.01), 2) if which == "markov" else (np.round(np.arange(0.38, 0.5201, 0.01), 2) if len(sys.argv) > 4 and sys.argv[4] == "fine" else np.array([0.42, 0.44, 0.46, 0.48, 0.50]))
DELTAS = [1e-5, 1e-4]
t0 = time.time()
log = lambda s: print(f"{s} [{time.time()-t0:.0f}s]", flush=True)

if which == "trace":
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
    qq = paths.mean()
    law = EmpiricalPathLaw(paths); name = "trace_cohort0.6"
    counts = paths.sum(axis=1); pn = np.bincount(counts, minlength=T + 1) / len(counts)
    log(f"trace law: {len(paths)} paths, q={qq:.4f}, count mean {counts.mean():.2f} var {counts.var():.2f}")
else:
    P, pi, emit = two_state_chain(q, 0.9); law = ChainLaw(P, pi, emit, T); name = "markov0.9"; qq = q
    pn = count_distribution(P, pi, emit, T)

# ---- accounting at both deltas (instant)
rows_acc = []
for DELTA in DELTAS:
    ep = eps_poisson_pld(qq, T, 1.0, 1.0 / a_true, DELTA)
    e_sum = eps_sum_exact(pn, 1.0, 1.0 / a_true, T, DELTA)
    pn_pois = binom.pmf(np.arange(T + 1), T, qq)
    e_sum_pois = eps_sum_exact(pn_pois, 1.0, 1.0 / a_true, T, DELTA)
    rows_acc.append([name, DELTA, ep, e_sum, e_sum_pois, e_sum / e_sum_pois])
    log(f"delta={DELTA:g}: Poisson per-round eps={ep:.3f}; final-model exact: law {e_sum:.3f} vs Poisson-count {e_sum_pois:.3f} (ratio {e_sum/e_sum_pois:.2f})")

# ---- per-round reference + certified tables at both deltas from shared samples
log(f"== {name}: sampling {N_MC} per grid point ({len(a_grid)} points)")
table5 = CertifiedTable2(law, a_grid, DELTAS[0], N_MC, A_MC, rng, chunk=(4000 if which == "trace" else 25_000), log=log, keep_samples=True)
# recompute at delta=1e-4 from the same samples
eps_plain4, eps_bern4 = [], []
for gi, a in enumerate(a_grid):
    lp = table5.lps[gi].astype(np.float64); lq = table5.lqs[gi].astype(np.float64)
    eps_plain4.append(plain_eps(lp, lq, DELTAS[1])); eps_bern4.append(eps_cert_bern(lp, lq, DELTAS[1], A_MC))
    log(f"  a={a:.2f}: delta=1e-4 plain={eps_plain4[-1]:.3f} cert={eps_bern4[-1]:.3f} | delta=1e-5 plain={table5.eps_plain[gi]:.3f} cert={table5.eps_bern[gi]:.3f}")
SUF = "_fine" if len(sys.argv) > 4 and sys.argv[4] == "fine" else ""
with open(f"step16_table_{which}{SUF}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["a", "eps_plain_1e-5", "eps_cert_1e-5", "eps_plain_1e-4", "eps_cert_1e-4"])
    for gi, a in enumerate(a_grid):
        w.writerow([f"{a:.2f}", f"{table5.eps_plain[gi]:.4f}", f"{table5.eps_bern[gi]:.4f}", f"{eps_plain4[gi]:.4f}", f"{eps_bern4[gi]:.4f}"])
i50 = int(np.argmin(np.abs(a_grid - 0.5)))
ref5, ref4 = table5.eps_plain[i50], eps_plain4[i50]
ep5 = [r for r in rows_acc if r[1] == 1e-5][0][2]; ep4 = [r for r in rows_acc if r[1] == 1e-4][0][2]
log(f"per-round reference at a=0.5: delta=1e-5 {ref5:.3f} (Poisson {ep5:.3f}, ratio {ref5/ep5:.2f}); delta=1e-4 {ref4:.3f} (Poisson {ep4:.3f}, ratio {ref4/ep4:.2f})")
with open(f"step16_accounting_{which}{SUF}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["law", "delta", "poisson_perround_eps", "final_model_exact_eps", "final_model_poisson_count_eps", "final_model_ratio", "perround_reference_eps", "perround_ratio"])
    for r in rows_acc:
        ref = ref5 if r[1] == 1e-5 else ref4
        w.writerow([r[0], f"{r[1]:g}", f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[5]:.3f}", f"{ref:.4f}", f"{ref/r[2]:.3f}"])

# ---- audits at delta = 1e-5 (and 1e-4) with the merged certified auditor
def floor_idx(a):
    if not np.isfinite(a): return -1                   # non-finite estimate: below the grid, certificate 0 (round 12)
    i = np.searchsorted(a_grid - 1e-9, a, side="right") - 1
    return i
res = {"cert_1e-5": [], "v2_1e-5": [], "cert_1e-4": [], "v2_1e-4": []}
for rep in range(R):
    present = rng.random(m) < 0.5
    B = law.sample(m, rng)
    X = (B * present[:, None]) * a_true + rng.standard_normal((m, T)).astype(np.float32)
    a_c = snr_lower_bound(X, present, B, A_EST); a_v = snr_lower_bound(X, present, B, 0.05)
    i = floor_idx(a_c)
    res["cert_1e-5"].append(0.0 if i < 0 else float(table5.eps_bern[i])); res["cert_1e-4"].append(0.0 if i < 0 else float(eps_bern4[i]))
    res["v2_1e-5"].append(float(np.interp(a_v, a_grid, table5.eps_plain, left=0.0))); res["v2_1e-4"].append(float(np.interp(a_v, a_grid, eps_plain4, left=0.0)))
with open(f"step16_results_{which}{SUF}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["law", "delta", "estimator", "R", "mean", "sd", "reference", "poisson", "recovery", "false_flag_rate", "flag_poisson"])
    for est, v in res.items():
        v = np.array(v); DELTA = 1e-5 if est.endswith("1e-5") else 1e-4
        ref = ref5 if DELTA == 1e-5 else ref4; ep = ep5 if DELTA == 1e-5 else ep4
        w.writerow([name, f"{DELTA:g}", est, R, f"{v.mean():.4f}", f"{v.std():.4f}", f"{ref:.4f}", f"{ep:.4f}", f"{v.mean()/ref:.3f}", f"{np.mean(v > ref):.3f}", f"{np.mean(v > ep):.3f}"])
        log(f"{name} {est:10s}: {v.mean():.3f}±{v.std():.3f} ref {ref:.3f} recovery {v.mean()/ref:.2f} false-flag {np.mean(v > ref):.3f} flag-Poisson {np.mean(v > ep):.2f}")
log("done")
