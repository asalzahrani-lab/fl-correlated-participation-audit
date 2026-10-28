"""Step 14: merged certified path-aware auditor (Bernstein certification, 0.01 grid, 0.045/0.005 alpha split)
on (A) the two-state Markov law (lambda=0.9) and (B) the FLASH empirical path law (fixed cohort 0.6 x pool).
usage: python step14_merged.py markov|trace [n_mc] [R]"""
import numpy as np, csv, time, sys, json
import os; sys.path.insert(0, os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")))
from corraudit import two_state_chain, eps_poisson_pld
from certified import EmpiricalPathLaw, ChainLaw
from certified2 import CertifiedTable2, snr_lower_bound

which = sys.argv[1]
T, a_true, DELTA, q = 200, 0.5, 1e-4, 0.1
N_MC = int(sys.argv[2]) if len(sys.argv) > 2 else (2_000_000 if which == "markov" else 1_000_000)
R = int(sys.argv[3]) if len(sys.argv) > 3 else 120
A_EST_NEW, A_MC_NEW = 0.045, 0.005
A_EST_OLD, A_MC_OLD = 0.025, 0.025
m = 1000
rng = np.random.default_rng({"markov": 2027, "trace": 2028, "iid": 2029}[which])
a_grid = np.round(np.arange({"markov": 0.40, "trace": 0.36, "iid": 0.40}[which], 0.5501, 0.01), 2)
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
    log(f"trace law: {len(paths)} paths, q={qq:.4f}")
    law = EmpiricalPathLaw(paths); name = "trace_cohort0.6"
    ref_plain = 10.3656
elif which == "iid":
    from corraudit import iid_chain
    P, pi, emit = iid_chain(q); law = ChainLaw(P, pi, emit, T); name = "iid"; qq = q
    ref_plain = 2.8690
else:
    P, pi, emit = two_state_chain(q, 0.9); law = ChainLaw(P, pi, emit, T); name = "markov0.9"; qq = q
    ref_plain = 11.8174
ep = eps_poisson_pld(qq, T, 1.0, 1.0 / a_true, DELTA)

log(f"== {name}: building table (n={N_MC}, {len(a_grid)} grid points)")
table = CertifiedTable2(law, a_grid, DELTA, N_MC, A_MC_NEW, rng, chunk=(4000 if which == "trace" else 25_000), log=log, keep_samples=True)
# old-protocol certification on the same samples: step method at alpha 0.025, coarse 0.05 grid floor
from certified import delta_lo_curve, eps_from_lower_curve
coarse = np.round(np.arange(a_grid[0], 0.5501, 0.05), 2)
old_vals = {}
for a in coarse:
    gi = int(np.argmin(np.abs(a_grid - a)))
    lp = table.lps[gi].astype(np.float64); lq = table.lqs[gi].astype(np.float64)
    old_vals[a] = eps_from_lower_curve(delta_lo_curve(lp, lq, A_MC_OLD), DELTA)
def cert_old(a):
    if not np.isfinite(a): return 0.0                  # non-finite estimate: certificate 0 (round 12)
    i = np.searchsorted(coarse - 1e-9, a, side="right") - 1
    return 0.0 if i < 0 else float(old_vals[coarse[i]])
log(f"table at a=0.50: plain={table.plain(0.5):.3f} cert_bern={table.cert(0.5):.3f} cert_step={table.cert_step(0.5):.3f} old(0.025,0.05grid)={cert_old(0.5):.3f}; Poisson={ep:.3f}")
with open(f"step14_table_{which}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["a", "eps_plain", "eps_cert_step_0.005", "eps_cert_bern_0.005"])
    for a, p_, s_, b_ in zip(a_grid, table.eps_plain, table.eps_step, table.eps_bern):
        w.writerow([f"{a:.2f}", f"{p_:.4f}", f"{s_:.4f}", f"{b_:.4f}"])

res = {"merged_bern_0.01grid": [], "step_0.01grid_a045": [], "old_certified": [], "plain_v2": []}
alos = []
for rep in range(R):
    present = rng.random(m) < 0.5
    B = law.sample(m, rng)
    X = (B * present[:, None]) * a_true + rng.standard_normal((m, T)).astype(np.float32)
    a_new = snr_lower_bound(X, present, B, A_EST_NEW)
    a_old = snr_lower_bound(X, present, B, A_EST_OLD)
    a_v2 = snr_lower_bound(X, present, B, 0.05)
    alos.append(a_new)
    res["merged_bern_0.01grid"].append(table.cert(a_new))
    res["step_0.01grid_a045"].append(table.cert_step(a_new))
    res["old_certified"].append(cert_old(a_old))
    res["plain_v2"].append(table.plain(a_v2))
alos = np.array(alos)
log(f"a_lo (alpha 0.045): mean {alos.mean():.4f} sd {alos.std():.4f} min {alos.min():.4f} max {alos.max():.4f}")
with open(f"step14_results_{which}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["law", "reference_plain_eps_2M", "poisson_eps_same_q", "estimator", "R", "mean", "sd", "recovery", "false_flag_rate_vs_ref", "flag_poisson"])
    for est, v in res.items():
        v = np.array(v)
        w.writerow([name, f"{ref_plain:.4f}", f"{ep:.4f}", est, R, f"{v.mean():.4f}", f"{v.std():.4f}", f"{v.mean()/ref_plain:.3f}", f"{np.mean(v > ref_plain):.3f}", f"{np.mean(v > ep):.3f}"])
        log(f"{name} {est:22s}: {v.mean():.3f}±{v.std():.3f} recovery {v.mean()/ref_plain:.2f} false-flag {np.mean(v > ref_plain):.3f} flag-Poisson {np.mean(v > ep):.2f}")
log("done")
