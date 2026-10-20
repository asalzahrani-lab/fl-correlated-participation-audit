"""Step 21b: consistency pass on the trace pipeline after the Section-6 review.
(A) Policy table recomputed with a FRESH random stream (seed 0) per policy, so that the fixed-cohort-0.6 row is the
    same realisation as the horizon curves, step 18, step 19 and the audits; adds the Theorem-3 count-averaged
    per-round bound at both deltas, the per-round Poisson value, and the number of devices with a defined lag-1.
(B) Running example (K = 27, seed 0, stride 10): tail support of the count law (distinct devices contributing windows
    with N >= k), between-device share of Var(N_T) and Var(p_i)/E[p_i] against the fitted slope, for both runs.
(C) Descriptive statistics of the 5-minute / 4-day window.
(D) Tail mass of the step-18 model count laws.  (E) Wall-clock of Theorem 1 and Theorem 3 on the count law.
Outputs step21b_policies.csv, step21b_tails.csv; prints the rest."""
import numpy as np, csv, json, sys, time
import os; sys.path.insert(0, os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")))
from scipy.stats import binom
from trace_analysis import ready_intervals, availability_grid, lag1
from corraudit import eps_sum_exact, eps_poisson_pld
from theory import eps_jensen_full

C, SIGMA = 1.0, 2.0
DELTAS = [1e-4, 1e-5]
t0 = time.time(); log = lambda s: print(f"{s} [{time.time()-t0:.0f}s]", flush=True)
d = json.load(open(os.path.join(os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")), "state_traces.json")))
devs = []
for k, v in d.items():
    iv, tf, tl = ready_intervals(v["messages"])
    if tf is None or tl - tf < 2 * 86400: continue
    devs.append((k, iv, tf, tl))


def window(round_min, win_days):
    step = round_min * 60
    t0_ = min(tf for (_, _, tf, _) in devs); t1_ = max(tl for (_, _, _, tl) in devs)
    times = np.arange(t0_, t1_, step); Rn = len(times)
    Amat = np.zeros((len(devs), Rn), dtype=bool); cover = np.zeros((len(devs), Rn), dtype=bool)
    for i, (k, iv, tf, tl) in enumerate(devs):
        Amat[i] = availability_grid(iv, t0_, t1_, step); cover[i] = (times >= tf) & (times <= tl)
    W = int(win_days * 24 * 60 / round_min)
    cands = list(range(0, Rn - W, 36)); scores = [(cover[:, s0:s0 + W].all(axis=1)).sum() for s0 in cands]
    best = cands[int(np.argmax(scores))]
    sel = cover[:, best:best + W].all(axis=1)
    log(f"{round_min}-min/{win_days}-day window: search grid {len(cands)} starts (stride 36 rounds); best {max(scores)} devices, next {sorted(scores)[-2]}, {sorted(scores)[-3]}")
    return Amat[sel, best:best + W]


def apply_policy(A, kind, par, b, rng):
    N, Tw = A.shape; B = np.zeros_like(A); last = np.full(N, -10**9); Epool = A.sum(axis=0).mean()
    for t in range(Tw):
        idx = np.flatnonzero(A[:, t] & (t - last > b))
        if kind == "thin": chosen = idx[rng.random(len(idx)) < par]
        else:
            K = int(round(par * Epool)); chosen = idx if len(idx) <= K else rng.choice(idx, K, replace=False)
        B[chosen, t] = True; last[chosen] = t
    return B


def windows(B, T, stride):
    N, Tw = B.shape
    dev = np.repeat(np.arange(N), (Tw - T) // stride + 1)
    counts = np.array([B[i, s0:s0 + T].sum() for i in range(N) for s0 in range(0, Tw - T + 1, stride)])
    return counts, dev


# ---------------- (A) policy table, fresh seed per policy
A3 = window(10, 3); N3, Tw3 = A3.shape
log(f"3-day window: {N3} devices, mean availability {A3.mean():.4f}, E[pool] {A3.sum(axis=0).mean():.1f}")
policies = [("thinning 0.05", "thin", 0.05, 0), ("thinning 0.20", "thin", 0.20, 0), ("thinning 0.50", "thin", 0.50, 0), ("take all available", "thin", 1.0, 0),
            ("fixed cohort 0.1 x pool", "fixed", 0.1, 0), ("fixed cohort 0.3 x pool", "fixed", 0.3, 0), ("fixed cohort 0.6 x pool", "fixed", 0.6, 0), ("fixed cohort 1.0 x pool", "fixed", 1.0, 0),
            ("fixed cohort 0.6, timer 5 rounds", "fixed", 0.6, 5), ("fixed cohort 0.6, timer 20 rounds", "fixed", 0.6, 20)]
T = 200
with open("step21b_policies.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["policy", "q", "lag1_median", "n_lag1_defined", "D", "n_paths", "eps_1e-4", "eps_pois_1e-4", "ratio_1e-4", "eps_1e-5", "eps_pois_1e-5", "ratio_1e-5",
                                   "perround_poisson_1e-4", "thm3_bound_1e-4", "perround_poisson_1e-5", "thm3_bound_1e-5"])
    for name, kind, par, b in policies:
        B = apply_policy(A3, kind, par, b, np.random.default_rng(0))
        counts, _ = windows(B, T, 10); pn = np.bincount(counts, minlength=T + 1) / len(counts)
        q = counts.mean() / T; D = counts.var() / counts.mean()
        lam = np.array([lag1(B[i]) for i in range(N3)]); lam_med = np.nanmedian(lam); n_def = int(np.sum(~np.isnan(lam)))
        row = [name, f"{q:.4f}", f"{lam_med:.3f}", n_def, f"{D:.2f}", len(counts)]
        extra = []
        for dl in DELTAS:
            e = eps_sum_exact(pn, C, SIGMA, T, dl); ep = eps_sum_exact(binom.pmf(np.arange(T + 1), T, q), C, SIGMA, T, dl)
            row += [f"{e:.3f}", f"{ep:.3f}", f"{e/ep:.2f}"]
            extra += [f"{eps_poisson_pld(q, T, C, SIGMA, dl):.3f}", f"{eps_jensen_full(pn, C, SIGMA, dl):.3f}"]
        w.writerow(row + extra)
        log(f"{name:34s} q={q:.4f} lag1={lam_med:.3f} ({n_def} defined) D={D:.2f} final-model ratio {row[8]} / {row[11]}; per-round Poisson {extra[0]} / {extra[2]}, Thm3 bound {extra[1]} / {extra[3]}")

# ---------------- (B) tail support and variance decomposition
with open("step21b_tails.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["round_min", "T", "n_windows", "n_devices", "max_N", "k", "P(N>=k)", "windows_N>=k", "devices_N>=k", "between_share", "VarP_over_EP", "D"])
    for round_min, win_days, Ts, stride, ks in [(10, 3, [50, 100, 200, 300, 400], 10, [40, 60, 80, 100]), (5, 4, [100, 200, 400, 600, 800, 1000], 5, [100, 150, 200, 300])]:
        A = window(round_min, win_days) if round_min == 5 else A3
        B = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0)); N = B.shape[0]
        p_dev = B.mean(axis=1); VE = p_dev.var() / p_dev.mean()
        pa = A.mean(axis=1)
        log(f"{round_min}-min window: {N} devices, mean availability {A.mean():.4f}, never available {np.mean(pa==0):.3f}, >0.5 {np.mean(pa>0.5):.4f}, pool mean {A.sum(axis=0).mean():.1f} min {A.sum(axis=0).min()} max {A.sum(axis=0).max()}, lambda_A median {np.nanmedian([lag1(A[i]) for i in range(N)]):.3f}, q {B.mean():.4f}, Var(p_i)/E[p_i] {VE:.4f}")
        for T in Ts:
            counts, dev = windows(B, T, stride); D = counts.var() / counts.mean()
            mean_by_dev = np.array([counts[dev == i].mean() for i in range(N)]); between = mean_by_dev.var() / counts.var()
            for k in ks:
                sel = counts >= k
                w.writerow([round_min, T, len(counts), N, int(counts.max()), k, f"{sel.mean():.5f}", int(sel.sum()), int(len(np.unique(dev[sel]))), f"{between:.3f}", f"{VE:.4f}", f"{D:.2f}"])
            tails = "; ".join(f"N>={k}: {np.mean(counts>=k):.4f} ({len(np.unique(dev[counts>=k]))} devices)" for k in ks)
            log(f"  T={T}: D={D:.2f} max N {counts.max()} between-device share {between:.3f} | {tails}")

# ---------------- (D) model tails
emp = np.load("step18_pn_empirical.npy")
for m in ["empirical", "M1", "M2", "M3", "M4", "M5", "M6"]:
    pn = np.load(f"step18_pn_{m}.npy"); n = np.arange(len(pn))
    log(f"step18 {m:9s}: P(N>=60)={pn[n>=60].sum():.4f} P(N>=80)={pn[n>=80].sum():.4f} P(N>=100)={pn[n>=100].sum():.5f} max support {n[pn>0].max()}")

# ---------------- (E) timing
B = apply_policy(A3, "fixed", 0.6, 0, np.random.default_rng(0)); counts, _ = windows(B, 200, 10); pn = np.bincount(counts, minlength=201) / len(counts)
t1 = time.time(); e1 = eps_sum_exact(pn, C, SIGMA, 200, 1e-4); t2 = time.time(); e3 = eps_jensen_full(pn, C, SIGMA, 1e-4); t3 = time.time()
log(f"timing on the count law: Theorem 1 exact {e1:.3f} in {t2-t1:.3f} s; Theorem 3 bound {e3:.3f} in {t3-t2:.3f} s")
log("done")
