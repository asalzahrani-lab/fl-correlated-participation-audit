"""Revision B3: robustness of the trace results.
 (1) Twenty independent draws (seeds 0..19) of each of the ten policies on the 3-day window: q, lag-1, D, exact
     final-model eps under the empirical count law and under the Binomial law, ratios at 1e-4 and 1e-5, the
     Theorem-3 bound -> median, IQR, min, max per policy (rev_robust_policies.csv).
 (2) Leave-out of tail devices for the running example (seed 0): each device that reaches N >= 60 removed in turn,
     and all of them together (rev_robust_leaveout.csv).
 (3) Non-overlapping windows: T = 200 at 10-min (windows starting at 0 and 200) and the 5-min horizon run with
     non-overlapping windows for T in {100, 200, 300, 400, 500, 576, 1000} (rev_robust_nonoverlap.csv).
 (4) Out-of-window replication: the best 3-day 10-min window disjoint from the analysis window, the ten policies
     (seed 0) there; and the per-device two-state availability model (M1) fitted on the analysis window and evaluated,
     for the devices present in both windows, against the empirical law of the other window (rev_robust_window.csv).
"""
import csv
import numpy as np
from scipy.stats import binom
from rev_common import log, window, window_candidates, apply_policy, windows
from corraudit import eps_sum_exact
from theory import eps_jensen_full
from trace_analysis import lag1

T, C, SIGMA = 200, 1.0, 2.0
DELTAS = [1e-4, 1e-5]
POLICIES = [("thinning 0.05", "thin", 0.05, 0), ("thinning 0.20", "thin", 0.20, 0), ("thinning 0.50", "thin", 0.50, 0),
            ("take all available", "thin", 1.0, 0), ("fixed cohort 0.1", "fixed", 0.1, 0), ("fixed cohort 0.3", "fixed", 0.3, 0),
            ("fixed cohort 0.6", "fixed", 0.6, 0), ("fixed cohort 1.0", "fixed", 1.0, 0), ("fixed cohort 0.6, timer 5", "fixed", 0.6, 5),
            ("fixed cohort 0.6, timer 20", "fixed", 0.6, 20)]


def summarize(B, T_=T, stride=10, thm3=False):
    counts, dev = windows(B, T_, stride); pn = np.bincount(counts, minlength=T_ + 1) / len(counts)
    q = counts.mean() / T_; D = counts.var() / counts.mean() if counts.mean() > 0 else np.nan
    out = {"q": q, "D": D, "maxN": int(counts.max())}
    pb = binom.pmf(np.arange(T_ + 1), T_, q)
    for d in DELTAS:
        e = eps_sum_exact(pn, C, SIGMA, T_, d); eb = eps_sum_exact(pb, C, SIGMA, T_, d)
        out[f"fm_{d:g}"] = e; out[f"fmb_{d:g}"] = eb; out[f"ratio_{d:g}"] = e / eb if eb > 0 else np.nan
        if thm3: out[f"thm3_{d:g}"] = eps_jensen_full(pn, C, SIGMA, d)
    return out


if __name__ == "__main__":
    A, idxA, startA = window(10, 3)
    # ---------------- (1) twenty draws per policy
    rows = []
    for name, kind, par, b in POLICIES:
        vals = []
        for seed in range(20):
            B = apply_policy(A, kind, par, b, np.random.default_rng(seed))
            s = summarize(B); lam = np.array([lag1(B[i]) for i in range(B.shape[0])]); s["lag1"] = np.nanmedian(lam)
            s["part_per_device"] = B.sum(axis=1).mean(); vals.append(s)
        for key in ["q", "lag1", "D", "fm_0.0001", "ratio_0.0001", "ratio_1e-05", "part_per_device"]:
            v = np.array([x[key] for x in vals])
            rows.append([name, key, f"{np.median(v):.4f}", f"{np.quantile(v, .25):.4f}", f"{np.quantile(v, .75):.4f}", f"{v.min():.4f}", f"{v.max():.4f}", f"{vals[0][key]:.4f}"])
        log(f"{name}: D median {np.median([x['D'] for x in vals]):.2f} [{min(x['D'] for x in vals):.2f}, {max(x['D'] for x in vals):.2f}]; ratio 1e-4 median {np.median([x['ratio_0.0001'] for x in vals]):.2f} [{min(x['ratio_0.0001'] for x in vals):.2f}, {max(x['ratio_0.0001'] for x in vals):.2f}]; 1e-5 median {np.median([x['ratio_1e-05'] for x in vals]):.2f} [{min(x['ratio_1e-05'] for x in vals):.2f}, {max(x['ratio_1e-05'] for x in vals):.2f}]")
    with open("rev_robust_policies.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["policy", "quantity", "median", "q25", "q75", "min", "max", "seed0"]); w.writerows(rows)
    # ---------------- (2) leave-out of tail devices, running example
    B0 = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0)); N, Tw = B0.shape
    maxN = np.array([max(B0[i, s0:s0 + T].sum() for s0 in range(0, Tw - T + 1, 10)) for i in range(N)])
    tail = np.flatnonzero(maxN >= 60); base = summarize(B0, thm3=True)
    lrows = [["none", 0, f"{base['D']:.2f}", f"{base['ratio_0.0001']:.3f}", f"{base['ratio_1e-05']:.3f}", f"{base['thm3_0.0001']:.2f}", base["maxN"]]]
    for i in tail:
        keep = np.setdiff1d(np.arange(N), [i]); s = summarize(B0[keep], thm3=True)
        lrows.append([f"device {i}", 1, f"{s['D']:.2f}", f"{s['ratio_0.0001']:.3f}", f"{s['ratio_1e-05']:.3f}", f"{s['thm3_0.0001']:.2f}", s["maxN"]])
    for k in [5, 10, len(tail)]:
        drop = tail[np.argsort(-maxN[tail])][:k]; keep = np.setdiff1d(np.arange(N), drop); s = summarize(B0[keep], thm3=True)
        lrows.append([f"top {k} tail devices", k, f"{s['D']:.2f}", f"{s['ratio_0.0001']:.3f}", f"{s['ratio_1e-05']:.3f}", f"{s['thm3_0.0001']:.2f}", s["maxN"]])
        log(f"leave out top {k} tail devices: D {s['D']:.2f} ratio 1e-4 {s['ratio_0.0001']:.2f} 1e-5 {s['ratio_1e-05']:.2f} Thm3 {s['thm3_0.0001']:.2f} maxN {s['maxN']}")
    one = np.array([float(r[3]) for r in lrows[1:1 + len(tail)]])
    log(f"leave-one-out over {len(tail)} tail devices: ratio 1e-4 in [{one.min():.3f}, {one.max():.3f}] (all devices {base['ratio_0.0001']:.3f})")
    with open("rev_robust_leaveout.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["removed", "n_removed", "D", "ratio_1e-4", "ratio_1e-5", "thm3_1e-4", "maxN"]); w.writerows(lrows)
    # ---------------- (3) non-overlapping windows
    nrows = []
    s = summarize(B0, stride=200); nrows.append([10, 200, "non-overlapping", f"{s['D']:.2f}", f"{s['ratio_0.0001']:.3f}", f"{s['ratio_1e-05']:.3f}", 2 * N])
    s = summarize(B0, stride=10); nrows.append([10, 200, "stride 10", f"{s['D']:.2f}", f"{s['ratio_0.0001']:.3f}", f"{s['ratio_1e-05']:.3f}", 24 * N])
    A5, _, _ = window(5, 4); B5 = apply_policy(A5, "fixed", 0.6, 0, np.random.default_rng(0)); N5, Tw5 = B5.shape
    for T_ in [100, 200, 300, 400, 500, 576, 1000]:
        s = summarize(B5, T_=T_, stride=T_)
        nrows.append([5, T_, "non-overlapping", f"{s['D']:.2f}", f"{s['ratio_0.0001']:.3f}", f"{s['ratio_1e-05']:.3f}", N5 * (Tw5 // T_)])
        log(f"5-min T={T_} non-overlapping: windows {N5 * (Tw5 // T_)} D {s['D']:.2f} ratio 1e-4 {s['ratio_0.0001']:.2f}")
    with open("rev_robust_nonoverlap.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["round_min", "T", "windows", "D", "ratio_1e-4", "ratio_1e-5", "n_windows"]); w.writerows(nrows)
    # ---------------- (4) out-of-window replication
    Amat, cover, times, W, cands, scores = window_candidates(10, 3)
    disjoint = [(sc, s0) for sc, s0 in zip(scores, cands) if s0 + W <= startA or s0 >= startA + W]
    sc2, start2 = max(disjoint)
    A2, idx2, _ = window(10, 3, start=start2)
    wrows = []
    for name, kind, par, b in POLICIES:
        s = summarize(apply_policy(A2, kind, par, b, np.random.default_rng(0)))
        wrows.append(["other window", name, start2, len(idx2), f"{s['q']:.4f}", f"{s['D']:.2f}", f"{s['ratio_0.0001']:.3f}", f"{s['ratio_1e-05']:.3f}"])
        log(f"other window (start {start2}, {len(idx2)} devices) {name}: D {s['D']:.2f} ratio 1e-4 {s['ratio_0.0001']:.2f} 1e-5 {s['ratio_1e-05']:.2f}")
    # per-device two-state model fitted on window A, evaluated on window 2 for common devices
    common = np.intersect1d(idxA, idx2); ia = np.searchsorted(idxA, common); i2 = np.searchsorted(idx2, common)
    A_fit, A_test = A[ia], A2[i2]
    rng = np.random.default_rng(11); sims = []
    for rep in range(8):
        Asim = np.zeros_like(A_test)
        for k in range(len(common)):
            a = A_fit[k].astype(int); n0 = (a[:-1] == 0).sum(); n1 = (a[:-1] == 1).sum()
            p01 = (((a[:-1] == 0) & (a[1:] == 1)).sum() + 0.5) / (n0 + 1.0); p11 = (((a[:-1] == 1) & (a[1:] == 1)).sum() + 0.5) / (n1 + 1.0)
            st = rng.random() < a.mean()
            for t in range(A_test.shape[1]):
                Asim[k, t] = st; st = rng.random() < (p11 if st else p01)
        sims.append(apply_policy(Asim, "fixed", 0.6, 0, rng))
    emp = summarize(apply_policy(A_test, "fixed", 0.6, 0, np.random.default_rng(0)))
    fit_in = summarize(apply_policy(A_fit, "fixed", 0.6, 0, np.random.default_rng(0)))
    cnts = [windows(Bs, T, 10)[0] for Bs in sims]; allc = np.concatenate(cnts)
    pn = np.bincount(allc, minlength=T + 1) / len(allc); qm = allc.mean() / T
    m1 = {d: eps_sum_exact(pn, C, SIGMA, T, d) for d in DELTAS}
    for d in DELTAS:
        wrows.append(["M1 fitted on analysis window", f"delta {d:g}", start2, len(common), f"{qm:.4f}", f"{allc.var()/allc.mean():.2f}",
                      f"model {m1[d]:.3f} / empirical other window {emp[f'fm_{d:g}']:.3f} = {m1[d]/emp[f'fm_{d:g}']:.3f}",
                      f"empirical analysis window {fit_in[f'fm_{d:g}']:.3f}"])
        log(f"M1 out-of-window, {len(common)} common devices, delta {d:g}: model FM eps {m1[d]:.3f} vs empirical other window {emp[f'fm_{d:g}']:.3f} (ratio {m1[d]/emp[f'fm_{d:g}']:.2f}); analysis window empirical {fit_in[f'fm_{d:g}']:.3f}")
    with open("rev_robust_window.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["what", "policy_or_delta", "start", "devices", "q", "D", "ratio_1e-4_or_model", "ratio_1e-5_or_note"]); w.writerows(wrows)
