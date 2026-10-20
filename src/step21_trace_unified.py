"""Step 21: one trace pipeline for Section 3.6 / Section 6.
(A) Policy table on the 3-day, 10-minute window (572 fully covered devices among those with >= 2 days of trace):
    each policy -> participation matrix -> all T=200 sub-windows at stride 10 -> count law -> Theorem-1 exact
    final-model eps at delta 1e-4 and 1e-5, vs the Binomial law of the same rate; D; lag-1 autocorrelation.
(B) Horizon curves at fixed round length and cohort: 10-min rounds, K = 0.6 x mean pool of the 3-day window,
    T in {50,100,200,300,400}; and 5-min rounds, 4-day window (486 devices), K = 0.6 x its mean pool,
    T in {100,200,400,600,800,1000}; stride 10 (10-min) / 5 (5-min).
(C) Delta-method standard error of a Monte-Carlo eps(delta) reference: se = (d eps/d ln delta) * sqrt((1-delta)/(n delta)).
Outputs step21_policies.csv, step21_horizon.csv."""
import numpy as np, csv, json, sys, time
import os; sys.path.insert(0, os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")))
from scipy.stats import binom
from trace_analysis import ready_intervals, availability_grid, lag1
from corraudit import eps_sum_exact

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
    best = max(range(0, Rn - W, 36), key=lambda s0: (cover[:, s0:s0 + W].all(axis=1)).sum())
    sel = cover[:, best:best + W].all(axis=1)
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


def count_law(B, T, stride):
    N, Tw = B.shape
    counts = np.array([B[i, s0:s0 + T].sum() for i in range(N) for s0 in range(0, Tw - T + 1, stride)])
    pn = np.bincount(counts, minlength=T + 1) / len(counts)
    return counts, pn


def stats(B, T, stride):
    counts, pn = count_law(B, T, stride)
    q = counts.mean() / T; D = counts.var() / counts.mean() if counts.mean() > 0 else float("nan")
    lam = np.nanmedian([lag1(B[i]) for i in range(B.shape[0])])
    out = {"q": q, "D": D, "lam": lam, "n_paths": len(counts)}
    for dl in DELTAS:
        e = eps_sum_exact(pn, C, SIGMA, T, dl); ep = eps_sum_exact(binom.pmf(np.arange(T + 1), T, q), C, SIGMA, T, dl)
        out[f"eps_{dl:g}"] = e; out[f"eps_pois_{dl:g}"] = ep; out[f"ratio_{dl:g}"] = e / ep
    return out


# ---------------- (A) policies, 3-day 10-min window, T = 200, stride 10
A3 = window(10, 3); log(f"3-day window: {A3.shape[0]} devices, {A3.shape[1]} rounds, mean availability {A3.mean():.4f}, E[pool] {A3.sum(axis=0).mean():.1f}")
rng = np.random.default_rng(0)
policies = [("thinning 0.05", "thin", 0.05, 0), ("thinning 0.20", "thin", 0.20, 0), ("thinning 0.50", "thin", 0.50, 0), ("take all available", "thin", 1.0, 0),
            ("fixed cohort 0.1 x pool", "fixed", 0.1, 0), ("fixed cohort 0.3 x pool", "fixed", 0.3, 0), ("fixed cohort 0.6 x pool", "fixed", 0.6, 0), ("fixed cohort 1.0 x pool", "fixed", 1.0, 0),
            ("fixed cohort 0.6, timer 5 rounds", "fixed", 0.6, 5), ("fixed cohort 0.6, timer 20 rounds", "fixed", 0.6, 20)]
with open("step21_policies.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["policy", "q", "lag1_median", "D", "n_paths", "eps_1e-4", "eps_pois_1e-4", "ratio_1e-4", "eps_1e-5", "eps_pois_1e-5", "ratio_1e-5"])
    for name, kind, par, b in policies:
        B = apply_policy(A3, kind, par, b, rng); s = stats(B, 200, 10)
        w.writerow([name, f"{s['q']:.4f}", f"{s['lam']:.3f}", f"{s['D']:.2f}", s["n_paths"], f"{s['eps_0.0001']:.3f}", f"{s['eps_pois_0.0001']:.3f}", f"{s['ratio_0.0001']:.2f}", f"{s['eps_1e-05']:.3f}", f"{s['eps_pois_1e-05']:.3f}", f"{s['ratio_1e-05']:.2f}"])
        log(f"{name:34s} q={s['q']:.4f} lag1={s['lam']:.3f} D={s['D']:.2f} eps(1e-4)={s['eps_0.0001']:.3f} ratio={s['ratio_0.0001']:.2f} | 1e-5 ratio={s['ratio_1e-05']:.2f}")

# ---------------- (B) horizon curves
with open("step21_horizon.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["round_min", "window_days", "devices", "K", "T", "hours", "q", "D", "n_paths", "eps_1e-4", "ratio_1e-4", "eps_1e-5", "ratio_1e-5"])
    for round_min, win_days, Ts, stride in [(10, 3, [50, 100, 200, 300, 400], 10), (5, 4, [100, 200, 400, 600, 800, 1000], 5)]:
        A = window(round_min, win_days); rng = np.random.default_rng(0)
        B = apply_policy(A, "fixed", 0.6, 0, rng); K = int(round(0.6 * A.sum(axis=0).mean()))
        for T in Ts:
            s = stats(B, T, stride)
            w.writerow([round_min, win_days, A.shape[0], K, T, T * round_min / 60, f"{s['q']:.4f}", f"{s['D']:.2f}", s["n_paths"], f"{s['eps_0.0001']:.3f}", f"{s['ratio_0.0001']:.2f}", f"{s['eps_1e-05']:.3f}", f"{s['ratio_1e-05']:.2f}"])
            log(f"{round_min}-min rounds, {win_days}-day window, K={K}, T={T} ({T*round_min/60:.0f} h): q={s['q']:.4f} D={s['D']:.2f} ratio(1e-4)={s['ratio_0.0001']:.2f} ratio(1e-5)={s['ratio_1e-05']:.2f} [{s['n_paths']} paths]")
log("done")
