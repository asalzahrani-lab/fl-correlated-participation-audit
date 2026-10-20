"""Step 18: can the accountant replace the raw trace by a fitted availability model?
Fit four generative models of device availability to the FLASH window (572 devices, 432 ten-minute rounds), push each
through the SAME fixed-cohort policy (K = 0.6 x mean empirical pool), and compare the resulting participation-count
law and the exact final-model eps (Theorem 1) with the empirical path law.
  M1 markov-hetero   : two-state Markov chain per device with its own (p_i, lambda_i)            [Rodio et al. + heterogeneity]
  M2 renewal-pooled  : alternating renewal, pooled empirical on/off run-length laws, homogeneous
  M3 renewal-hetero  : pooled run-length shapes, off-law rescaled per device to match p_i
  M4 diurnal-markov  : time-inhomogeneous two-state chain, hour-of-day transition probabilities pooled over devices
  M5 diurnal-hetero  : M4 with a per-device factor on the off->on probability so that the mean availability matches p_i
Outputs step18_results.csv (one row per model, averaged over replicate population simulations)."""
import numpy as np, csv, json, sys, time, datetime
import os; sys.path.insert(0, os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")))
from scipy.stats import binom
from trace_analysis import ready_intervals, availability_grid, lag1
from corraudit import eps_sum_exact

T, DELTAS = 200, [1e-4, 1e-5]
C, SIGMA = 1.0, 2.0
REPS = 8
rng = np.random.default_rng(18)
t0 = time.time(); log = lambda s: print(f"{s} [{time.time()-t0:.0f}s]", flush=True)

# ---------------------------------------------------------------- empirical window (as in steps 11/12/16)
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
hour0 = datetime.datetime.fromtimestamp(times[best]).hour
hours = ((np.arange(Tw) * 10 / 60) + hour0) % 24
Epool = A.sum(axis=0).mean(); K = int(round(0.6 * Epool))
log(f"window: {N} devices, {Tw} rounds, start hour {hour0}, mean availability {A.mean():.4f}, E[pool] {Epool:.1f}, K {K}")


def policy_paths(Aw, rng_pol):
    """Fixed-cohort policy on an availability matrix -> participation matrix and the list of T-round windows."""
    B = np.zeros_like(Aw)
    for t in range(Aw.shape[1]):
        idx = np.flatnonzero(Aw[:, t])
        chosen = idx if len(idx) <= K else rng_pol.choice(idx, K, replace=False)
        B[chosen, t] = True
    paths = np.array([B[i, s0:s0 + T] for i in range(Aw.shape[0]) for s0 in range(0, Aw.shape[1] - T + 1, 10)], dtype=np.int8)
    return B, paths


def summarize(paths, B):
    counts = paths.sum(axis=1); pn = np.bincount(counts, minlength=T + 1) / len(counts)
    q = paths.mean(); D = counts.var() / counts.mean()
    lam = np.nanmedian([lag1(B[i]) for i in range(B.shape[0])])
    e = {dl: eps_sum_exact(pn, C, SIGMA, T, dl) for dl in DELTAS}
    ep = {dl: eps_sum_exact(binom.pmf(np.arange(T + 1), T, q), C, SIGMA, T, dl) for dl in DELTAS}
    return dict(q=q, D=D, lam=lam, pn=pn, eps=e, eps_pois=ep, pool=B.sum(axis=0))


Bemp, Pemp = policy_paths(A, np.random.default_rng(0))
S_emp = summarize(Pemp, Bemp)
log(f"empirical: q={S_emp['q']:.4f} D={S_emp['D']:.2f} lambda_B={S_emp['lam']:.3f} eps_sum(1e-4)={S_emp['eps'][1e-4]:.3f} (Poisson-count {S_emp['eps_pois'][1e-4]:.3f}) eps_sum(1e-5)={S_emp['eps'][1e-5]:.3f}")

# ---------------------------------------------------------------- fitted ingredients
p_dev = A.mean(axis=1)
lam_dev = np.array([lag1(A[i]) if 0 < p_dev[i] < 1 else np.nan for i in range(N)])
on_runs, off_runs = [], []
for i in range(N):
    a = A[i].astype(int); ch = np.flatnonzero(np.diff(a) != 0) + 1
    for s in np.split(a, ch):
        (on_runs if s[0] == 1 else off_runs).append(len(s))
on_runs, off_runs = np.array(on_runs), np.array(off_runs)
# hour-of-day transition probabilities pooled over devices
h_idx = (hours).astype(int)
n_on_off = np.zeros(24); n_on = np.zeros(24); n_off_on = np.zeros(24); n_off = np.zeros(24)
for i in range(N):
    a = A[i]
    for t in range(Tw - 1):
        h = h_idx[t]
        if a[t]: n_on[h] += 1; n_on_off[h] += (not a[t + 1])
        else: n_off[h] += 1; n_off_on[h] += a[t + 1]
p_on_off = (n_on_off + 0.5) / (n_on + 1.0); p_off_on = (n_off_on + 0.5) / (n_off + 1.0)
log("hourly P(off->on): " + " ".join(f"{x:.3f}" for x in p_off_on))
log("hourly P(on->off): " + " ".join(f"{x:.3f}" for x in p_on_off))

BURN = 2 * 144


def sim_markov_hetero(rng):
    Aw = np.zeros((N, Tw), bool)
    for i in range(N):
        p = p_dev[i]
        if p <= 0 or p >= 1: Aw[i] = bool(p >= 1); continue
        lam = lam_dev[i] if np.isfinite(lam_dev[i]) else 0.0
        lam = min(max(lam, 0.0), 0.999)
        a01 = (1 - lam) * p; a10 = (1 - lam) * (1 - p)        # off->on, on->off
        s = rng.random() < p
        for t in range(Tw):
            Aw[i, t] = s
            s = (rng.random() < a01) if not s else (rng.random() >= a10)
    return Aw


def sim_renewal(rng, hetero):
    Aw = np.zeros((N, Tw), bool)
    mean_on = on_runs.mean(); mean_off = off_runs.mean()
    for i in range(N):
        p = p_dev[i]
        if p <= 0 or p >= 1: Aw[i] = bool(p >= 1); continue
        if hetero:
            # scale the off-law so that mean_on / (mean_on + c*mean_off) = p_i
            c = mean_on * (1 - p) / (p * mean_off)
        else:
            c = 1.0
        seq = []; s = rng.random() < (mean_on / (mean_on + c * mean_off))
        total = 0
        while total < Tw + BURN:
            L = int(rng.choice(on_runs)) if s else max(1, int(round(c * rng.choice(off_runs))))
            seq.append((s, L)); total += L; s = not s
        a = np.concatenate([np.full(L, st, bool) for st, L in seq])
        Aw[i] = a[BURN:BURN + Tw]
    return Aw


def sim_diurnal(rng, hetero):
    Aw = np.zeros((N, Tw), bool)
    hh = ((np.arange(Tw + BURN) * 10 / 60 + hour0 - BURN * 10 / 60) % 24).astype(int)
    pbar = A.mean()
    for i in range(N):
        p = p_dev[i]
        if p <= 0 or p >= 1: Aw[i] = bool(p >= 1); continue
        f = (p / pbar) if hetero else 1.0
        po = np.clip(p_off_on * f, 0, 1); pf = p_on_off
        s = rng.random() < p; a = np.zeros(Tw + BURN, bool)
        for t in range(Tw + BURN):
            a[t] = s; h = hh[t]
            s = (rng.random() < po[h]) if not s else (rng.random() >= pf[h])
        Aw[i] = a[BURN:]
    return Aw


def calib_factor(p_target, iters=30):
    """Per-device factor f on P(off->on | hour) such that the periodic chain's mean availability equals p_target."""
    hh24 = ((np.arange(144) * 10 / 60) % 24).astype(int)
    def mean_avail(f):
        po = np.clip(p_off_on * f, 0, 1); x = p_target
        for _ in range(4 * 144):             # 4 days of iteration -> periodic steady state
            h = hh24[_ % 144]; x = x * (1 - p_on_off[h]) + (1 - x) * po[h]
        acc = 0.0
        for t in range(144):
            h = hh24[t]; x = x * (1 - p_on_off[h]) + (1 - x) * po[h]; acc += x
        return acc / 144
    lo, hi = 1e-3, 1e3
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if mean_avail(mid) < p_target: lo = mid
        else: hi = mid
    return np.sqrt(lo * hi)


def sim_diurnal_calibrated(rng):
    Aw = np.zeros((N, Tw), bool)
    hh = ((np.arange(Tw + BURN) * 10 / 60 + hour0 - BURN * 10 / 60) % 24).astype(int)
    for i in range(N):
        p = p_dev[i]
        if p <= 0 or p >= 1: Aw[i] = bool(p >= 1); continue
        po = np.clip(p_off_on * calib_factor(p), 0, 1); pf = p_on_off
        s = rng.random() < p; a = np.zeros(Tw + BURN, bool)
        for t in range(Tw + BURN):
            a[t] = s; h = hh[t]
            s = (rng.random() < po[h]) if not s else (rng.random() >= pf[h])
        Aw[i] = a[BURN:]
    return Aw


models = {"M1 markov-hetero": lambda r: sim_markov_hetero(r),
          "M2 renewal-pooled": lambda r: sim_renewal(r, False),
          "M3 renewal-hetero": lambda r: sim_renewal(r, True),
          "M4 diurnal-markov": lambda r: sim_diurnal(r, False),
          "M5 diurnal-hetero": lambda r: sim_diurnal(r, True),
          "M6 diurnal-calibrated": lambda r: sim_diurnal_calibrated(r)}

rows = [["empirical", f"{A.mean():.4f}", f"{S_emp['q']:.4f}", f"{S_emp['D']:.2f}", f"{S_emp['lam']:.3f}", "0", f"{S_emp['eps'][1e-4]:.3f}", f"{S_emp['eps'][1e-5]:.3f}", f"{S_emp['eps'][1e-4]/S_emp['eps_pois'][1e-4]:.2f}", f"{S_emp['eps'][1e-5]/S_emp['eps_pois'][1e-5]:.2f}", "1.000", "1.000"]]
prof_emp = np.array([S_emp["pool"][(hours >= h) & (hours < h + 3)].mean() for h in range(0, 24, 3)])
for name, sim in models.items():
    acc = {k: [] for k in ["avail", "q", "D", "lam", "ks", "e4", "e5", "r4", "r5", "pcorr"]}
    pn_sum = np.zeros(T + 1)
    for rep in range(REPS):
        Aw = sim(rng); Bw, Pw = policy_paths(Aw, rng); S = summarize(Pw, Bw)
        cdf_e = np.cumsum(S_emp["pn"]); cdf_m = np.cumsum(S["pn"])
        prof = np.array([S["pool"][(hours >= h) & (hours < h + 3)].mean() for h in range(0, 24, 3)])
        acc["avail"].append(Aw.mean()); acc["q"].append(S["q"]); acc["D"].append(S["D"]); acc["lam"].append(S["lam"])
        acc["ks"].append(np.abs(cdf_e - cdf_m).max()); acc["e4"].append(S["eps"][1e-4]); acc["e5"].append(S["eps"][1e-5])
        acc["r4"].append(S["eps"][1e-4] / S["eps_pois"][1e-4]); acc["r5"].append(S["eps"][1e-5] / S["eps_pois"][1e-5])
        acc["pcorr"].append(np.corrcoef(prof, prof_emp)[0, 1]); pn_sum += S["pn"]
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    np.save(f"step18_pn_{name.split()[0]}.npy", pn_sum / REPS)
    rows.append([name, f"{m['avail']:.4f}", f"{m['q']:.4f}", f"{m['D']:.2f}", f"{m['lam']:.3f}", f"{m['ks']:.3f}", f"{m['e4']:.3f}", f"{m['e5']:.3f}", f"{m['r4']:.2f}", f"{m['r5']:.2f}", f"{m['e4']/S_emp['eps'][1e-4]:.3f}", f"{m['pcorr']:.3f}"])
    log(f"{name:18s}: avail {m['avail']:.4f} q {m['q']:.4f} D {m['D']:.2f} lamB {m['lam']:.3f} KS {m['ks']:.3f} eps(1e-4) {m['e4']:.3f} ({m['e4']/S_emp['eps'][1e-4]:.2f} of empirical) eps(1e-5) {m['e5']:.3f} ratio-vs-Poisson {m['r4']:.2f}/{m['r5']:.2f} diurnal-corr {m['pcorr']:.2f}")
with open("step18_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "mean_availability", "q", "D", "lambda_B_median", "KS_count_law_vs_empirical", "eps_sum_1e-4", "eps_sum_1e-5", "ratio_vs_poisson_1e-4", "ratio_vs_poisson_1e-5", "eps_1e-4_over_empirical", "diurnal_pool_profile_corr"]); w.writerows(rows)
np.save("step18_pn_empirical.npy", S_emp["pn"])
log("done")
