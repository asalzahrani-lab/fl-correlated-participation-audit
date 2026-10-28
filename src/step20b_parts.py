"""Step 20b: restart-safe version of step20_T1000.py / step16_delta5.py.
The Monte-Carlo samples of log LR are produced in independent PARTS (one process each, own seed), saved to disk, and
combined afterwards; a container restart loses at most the part in progress.

usage:
  python step20b_parts.py sample  which part n_part [fine] [T] [round_min] [win_days] [stride]
  python step20b_parts.py combine which [fine] [T] [round_min] [win_days] [stride] [R]
which = markov | trace.  Parts land in step20_parts/{which}{SUF}_part{k}.npz (lps, lqs: G x n float32; a_grid).
combine writes the same files as step20_T1000.py: step16_table_{which}{SUF}.csv, step16_accounting_{which}{SUF}.csv,
step16_results_{which}{SUF}.csv (SUF = _fine and/or _T{T}), plus step16_combine_{which}{SUF}.log lines on stdout."""
import numpy as np, csv, time, sys, json, os, glob
import os; sys.path.insert(0, os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")))
from scipy.stats import binom
from corraudit import two_state_chain, eps_poisson_pld, eps_sum_exact, count_distribution
from certified import EmpiricalPathLaw, ChainLaw, plain_eps
from certified2 import CertifiedTable2, snr_lower_bound, eps_cert_bern

mode, which = sys.argv[1], sys.argv[2]
if mode == "sample":
    part, n_part = int(sys.argv[3]), int(sys.argv[4]); rest = sys.argv[5:]
else:
    part, n_part = None, None; rest = sys.argv[3:]
fine = len(rest) > 0 and rest[0] == "fine"
if fine: rest = rest[1:]
T = int(rest[0]) if len(rest) > 0 else 200
ROUND_MIN = int(rest[1]) if len(rest) > 1 else 10
WIN_DAYS = float(rest[2]) if len(rest) > 2 else 3
STRIDE = max(1, int(rest[3])) if len(rest) > 3 else 10
R = int(rest[4]) if len(rest) > 4 else 120
a_true, q = 0.5, 0.1
A_EST, A_MC = 0.045, 0.005
m = 1000
if T != 200: a_grid = np.array([0.44, 0.46, 0.48, 0.50])
elif which == "markov": a_grid = np.round(np.arange(0.44, 0.5201, 0.01), 2)
elif fine: a_grid = np.round(np.arange(0.38, 0.5201, 0.01), 2)
else: a_grid = np.array([0.42, 0.44, 0.46, 0.48, 0.50])
DELTAS = [1e-5, 1e-4]
SUF = ("_fine" if fine else "") + (f"_T{T}" if T != 200 else "")
os.makedirs("step20_parts", exist_ok=True)
t0 = time.time()
log = lambda s: print(f"{s} [{time.time()-t0:.0f}s]", flush=True)


def build_law():
    if which == "trace":
        from trace_analysis import ready_intervals, availability_grid
        d = json.load(open(os.path.join(os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")), "state_traces.json")))
        devs = []
        for k, v in d.items():
            iv, tf, tl = ready_intervals(v["messages"])
            if tf is None or tl - tf < 2 * 86400: continue
            devs.append((k, iv, tf, tl))
        step = ROUND_MIN * 60
        t0_ = min(tf for (_, _, tf, _) in devs); t1_ = max(tl for (_, _, _, tl) in devs)
        times = np.arange(t0_, t1_, step); Rn = len(times)
        Amat = np.zeros((len(devs), Rn), dtype=bool); cover = np.zeros((len(devs), Rn), dtype=bool)
        for i, (k, iv, tf, tl) in enumerate(devs):
            Amat[i] = availability_grid(iv, t0_, t1_, step); cover[i] = (times >= tf) & (times <= tl)
        W = int(WIN_DAYS * 24 * 60 / ROUND_MIN)
        best = max(range(0, Rn - W, 36), key=lambda s0: (cover[:, s0:s0 + W].all(axis=1)).sum())
        sel = cover[:, best:best + W].all(axis=1)
        A = Amat[sel, best:best + W]; N, Tw = A.shape
        rng_pol = np.random.default_rng(0)
        Bpol = np.zeros_like(A); Epool = A.sum(axis=0).mean(); K = int(round(0.6 * Epool))
        for t in range(Tw):
            idx = np.flatnonzero(A[:, t])
            chosen = idx if len(idx) <= K else rng_pol.choice(idx, K, replace=False)
            Bpol[chosen, t] = True
        paths = np.array([Bpol[i, s0:s0 + T] for i in range(N) for s0 in range(0, Tw - T + 1, STRIDE)], dtype=np.int8)
        qq = paths.mean(); counts = paths.sum(axis=1)
        pn = np.bincount(counts, minlength=T + 1) / len(counts)
        log(f"trace law: {len(paths)} paths from {N} devices, window {Tw} rounds of {ROUND_MIN} min, K={K}, q={qq:.4f}, count mean {counts.mean():.2f} var {counts.var():.2f} D {counts.var()/counts.mean():.2f}")
        return EmpiricalPathLaw(paths), "trace_cohort0.6", qq, pn
    P, pi, emit = two_state_chain(q, 0.9)
    return ChainLaw(P, pi, emit, T), "markov0.9", q, count_distribution(P, pi, emit, T)


law, name, qq, pn = build_law()

if mode == "sample":
    out = f"step20_parts/{which}{SUF}_part{part}.npz"
    if os.path.exists(out):
        log(f"{out} exists; nothing to do"); sys.exit(0)
    seed = (3031 if which == "markov" else 3032) * 1000 + 17 * T + part
    rng = np.random.default_rng(seed)
    log(f"== {name}{SUF}: part {part}, {n_part} samples per grid point ({len(a_grid)} points), seed {seed}")
    tab = CertifiedTable2(law, a_grid, DELTAS[0], n_part, A_MC, rng, chunk=(4000 if which == "trace" else 25_000), log=log, keep_samples=True)
    np.savez(out + ".tmp.npz", lps=tab.lps, lqs=tab.lqs, a_grid=a_grid, seed=seed)
    os.replace(out + ".tmp.npz", out)
    log(f"saved {out}")
    sys.exit(0)

# ---------------- combine
files = sorted(glob.glob(f"step20_parts/{which}{SUF}_part*.npz"))
assert files, "no parts found"
lps = np.concatenate([np.load(f)["lps"] for f in files], axis=1); lqs = np.concatenate([np.load(f)["lqs"] for f in files], axis=1)
for f in files: assert np.allclose(np.load(f)["a_grid"], a_grid)
N_MC = lps.shape[1]
log(f"combined {len(files)} parts: {N_MC} samples per grid point")

rows_acc = []
for DELTA in DELTAS:
    ep = eps_poisson_pld(qq, T, 1.0, 1.0 / a_true, DELTA)
    e_sum = eps_sum_exact(pn, 1.0, 1.0 / a_true, T, DELTA)
    e_sum_pois = eps_sum_exact(binom.pmf(np.arange(T + 1), T, qq), 1.0, 1.0 / a_true, T, DELTA)
    rows_acc.append([name, DELTA, ep, e_sum, e_sum_pois, e_sum / e_sum_pois])
    log(f"delta={DELTA:g}: Poisson per-round eps={ep:.3f}; final-model exact: law {e_sum:.3f} vs Poisson-count {e_sum_pois:.3f} (ratio {e_sum/e_sum_pois:.2f})")

tab = {d: {"plain": [], "cert": []} for d in DELTAS}
for gi, a in enumerate(a_grid):
    lp = lps[gi].astype(np.float64); lq = lqs[gi].astype(np.float64)
    for d in DELTAS:
        tab[d]["plain"].append(plain_eps(lp, lq, d)); tab[d]["cert"].append(eps_cert_bern(lp, lq, d, A_MC))
    log(f"  a={a:.2f}: delta=1e-4 plain={tab[1e-4]['plain'][-1]:.3f} cert={tab[1e-4]['cert'][-1]:.3f} | delta=1e-5 plain={tab[1e-5]['plain'][-1]:.3f} cert={tab[1e-5]['cert'][-1]:.3f}")
with open(f"step16_table_{which}{SUF}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["a", "eps_plain_1e-5", "eps_cert_1e-5", "eps_plain_1e-4", "eps_cert_1e-4"])
    for gi, a in enumerate(a_grid):
        w.writerow([f"{a:.2f}", f"{tab[1e-5]['plain'][gi]:.4f}", f"{tab[1e-5]['cert'][gi]:.4f}", f"{tab[1e-4]['plain'][gi]:.4f}", f"{tab[1e-4]['cert'][gi]:.4f}"])
i50 = int(np.argmin(np.abs(a_grid - 0.5)))
ref = {d: tab[d]["plain"][i50] for d in DELTAS}
epois = {r[1]: r[2] for r in rows_acc}
slope = (ref[1e-5] - ref[1e-4]) / np.log(10)
for d in DELTAS:
    se = slope * np.sqrt((1 - d) / (N_MC * d))
    log(f"per-round reference at a=0.5, delta={d:g}: {ref[d]:.3f} +- {se:.3f} (Poisson {epois[d]:.3f}, ratio {ref[d]/epois[d]:.2f})")
with open(f"step16_accounting_{which}{SUF}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["law", "delta", "poisson_perround_eps", "final_model_exact_eps", "final_model_poisson_count_eps", "final_model_ratio", "perround_reference_eps", "perround_reference_se", "perround_ratio", "n_mc"])
    for r in rows_acc:
        se = slope * np.sqrt((1 - r[1]) / (N_MC * r[1]))
        w.writerow([r[0], f"{r[1]:g}", f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[5]:.3f}", f"{ref[r[1]]:.4f}", f"{se:.4f}", f"{ref[r[1]]/r[2]:.3f}", N_MC])

rng = np.random.default_rng(4041 if which == "markov" else 4042)
res = {f"{k}_{d}": [] for k in ["cert", "v2"] for d in ["1e-5", "1e-4"]}
for rep in range(R):
    present = rng.random(m) < 0.5
    B = law.sample(m, rng)
    X = (B * present[:, None]) * a_true + rng.standard_normal((m, T)).astype(np.float32)
    a_c = snr_lower_bound(X, present, B, A_EST); a_v = snr_lower_bound(X, present, B, 0.05)
    i = np.searchsorted(a_grid - 1e-9, a_c, side="right") - 1 if np.isfinite(a_c) else -1   # non-finite: certificate 0 (round 12)
    for d, key in [(1e-5, "1e-5"), (1e-4, "1e-4")]:
        res[f"cert_{key}"].append(0.0 if i < 0 else float(tab[d]["cert"][i]))
        res[f"v2_{key}"].append(float(np.interp(a_v, a_grid, tab[d]["plain"], left=0.0)))
with open(f"step16_results_{which}{SUF}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["law", "delta", "estimator", "R", "mean", "sd", "min", "max", "reference", "reference_se", "poisson", "recovery", "n_above_reference", "n_above_poisson"])
    for est, v in res.items():
        v = np.array(v); d = 1e-5 if est.endswith("1e-5") else 1e-4
        se = slope * np.sqrt((1 - d) / (N_MC * d))
        w.writerow([name, f"{d:g}", est, R, f"{v.mean():.4f}", f"{v.std():.4f}", f"{v.min():.4f}", f"{v.max():.4f}", f"{ref[d]:.4f}", f"{se:.4f}", f"{epois[d]:.4f}", f"{v.mean()/ref[d]:.3f}", int(np.sum(v > ref[d])), int(np.sum(v > epois[d]))])
        log(f"{name} {est:10s}: {v.mean():.3f}+-{v.std():.3f} [{v.min():.2f},{v.max():.2f}] ref {ref[d]:.3f} recovery {v.mean()/ref[d]:.3f} above-ref {int(np.sum(v > ref[d]))}/{R} flags {int(np.sum(v > epois[d]))}/{R}")
log("done")
