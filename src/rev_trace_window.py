"""Revision B3 (4), replacing the last block of rev_trace_robust.py: out-of-window replication.

The trace's device coverage is concentrated in rounds ~650-1650 of the 10-minute grid, so no 3-day window disjoint
from the analysis window [972, 1404) has more than 9 fully covered devices.  The replication therefore uses the best
2-day window that starts at or after the end of the analysis window (start 1404, 288 rounds): the ten policies (seed 0
and 20 seeds) are run there with T = 200 at stride 10, and the per-device two-state availability model (M1) fitted on
the analysis window is evaluated, for the devices present in both windows, against the empirical count law of the later
window (a temporal hold-out).  Output rev_robust_window.csv."""
import csv
import numpy as np
from rev_common import log, window, window_candidates, apply_policy, windows
from corraudit import eps_sum_exact
from rev_trace_robust import POLICIES, summarize, T, C, SIGMA, DELTAS

if __name__ == "__main__":
    A, idxA, startA = window(10, 3)
    Amat, cover, times, W2, cands, scores = window_candidates(10, 2)
    later = [(sc, s0) for sc, s0 in zip(scores, cands) if s0 >= startA + A.shape[1]]
    sc2, start2 = max(later)
    A2, idx2, _ = window(10, 2, start=start2)
    wrows = []
    for name, kind, par, b in POLICIES:
        vals = [summarize(apply_policy(A2, kind, par, b, np.random.default_rng(s))) for s in range(20)]
        r4 = np.array([v["ratio_0.0001"] for v in vals]); r5 = np.array([v["ratio_1e-05"] for v in vals]); D = np.array([v["D"] for v in vals])
        wrows.append(["later window", name, start2, len(idx2), f"{np.median([v['q'] for v in vals]):.4f}", f"{np.median(D):.2f}",
                      f"{np.median(r4):.3f} [{r4.min():.3f}, {r4.max():.3f}]", f"{np.median(r5):.3f} [{r5.min():.3f}, {r5.max():.3f}]"])
        log(f"later window (start {start2}, {len(idx2)} devices, 2 days) {name}: q {wrows[-1][4]} D {np.median(D):.2f} ratio 1e-4 {np.median(r4):.2f} [{r4.min():.2f}, {r4.max():.2f}] 1e-5 {np.median(r5):.2f}")
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
    emp_vals = [summarize(apply_policy(A_test, "fixed", 0.6, 0, np.random.default_rng(s))) for s in range(20)]
    fit_in = summarize(apply_policy(A_fit, "fixed", 0.6, 0, np.random.default_rng(0)))
    allc = np.concatenate([windows(Bs, T, 10)[0] for Bs in sims])
    pn = np.bincount(allc, minlength=T + 1) / len(allc); qm = allc.mean() / T
    for d in DELTAS:
        m1 = eps_sum_exact(pn, C, SIGMA, T, d); emp = np.array([v[f"fm_{d:g}"] for v in emp_vals])
        wrows.append(["M1 fitted on analysis window, tested on later window", f"delta {d:g}", start2, len(common), f"{qm:.4f}",
                      f"{allc.var()/allc.mean():.2f}", f"model {m1:.3f} / empirical later window median {np.median(emp):.3f} [{emp.min():.3f}, {emp.max():.3f}] = {m1/np.median(emp):.3f}",
                      f"empirical analysis window (same devices) {fit_in[f'fm_{d:g}']:.3f}"])
        log(f"M1 hold-out, {len(common)} common devices, delta {d:g}: model FM eps {m1:.3f} vs empirical later window median {np.median(emp):.3f} [{emp.min():.3f}, {emp.max():.3f}] (ratio {m1/np.median(emp):.2f}); analysis window empirical {fit_in[f'fm_{d:g}']:.3f}")
    with open("rev_robust_window.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["what", "policy_or_delta", "start", "devices", "q", "D", "ratio_1e-4_or_model", "ratio_1e-5_or_note"]); w.writerows(wrows)
