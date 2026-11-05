"""Seventeenth round: the per-round Monte-Carlo reference of the running example on non-overlapping windows.

The trace law of the paper (Table II, Sec. VI-B) is the uniform mixture over the 13,728 windows of 200 ten-minute
rounds taken at stride 10 from the 572 devices of the 3-day window under the running policy (fixed cohort 0.6,
seed 0).  Adjacent windows of one device overlap in 190 of 200 rounds.  rev_trace_robust.py checked the final-model
ratio on the 1,144 non-overlapping windows (starts 0 and 200; D = 21.83, ratios 6.11 / 7.42); this script computes
the per-round reference on the same non-overlapping law: eps(delta; a = 0.5) of the secret-path release with its
two-sided 99 % Monte-Carlo interval (rev_common.eps_interval, as for the stride-10 re-run of rev_refs.py), the Poisson
value at the law's rate, the ratio, and the count-averaged upper end (Theorem 2) on the law's count distribution.

usage:
  python rev_round17_nonoverlap_ref.py sample PART N   -> rev_parts/nonoverlap200_partK.npz (lp, lq at a = 0.5)
  python rev_round17_nonoverlap_ref.py combine         -> results/rev_round17_nonoverlap_ref.csv
Seeds 7106 * 1000 + part.  Restart-safe: existing parts are skipped."""
import os, sys, glob, csv
import numpy as np
from rev_common import log, window, apply_policy, window_paths, windows, eps_interval
from certified import EmpiricalPathLaw
from corraudit import eps_poisson_pld
from theory import eps_jensen_full

A_SIG, T, STRIDE = 0.5, 200, 200
DELTAS = [1e-4, 1e-5]
SEED = 7106
os.makedirs("rev_parts", exist_ok=True)


def build():
    A, _, _ = window(10, 3)
    B = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0))
    paths = window_paths(B, T, STRIDE)
    c = paths.sum(axis=1)
    log(f"nonoverlap200: {len(paths)} paths ({B.shape[0]} devices x {paths.shape[0] // B.shape[0]}), q = {paths.mean():.4f}, "
        f"D = {c.var()/c.mean():.2f}, max N = {c.max()}")
    return EmpiricalPathLaw(paths), float(paths.mean()), c


def sample(part, n):
    out = f"rev_parts/nonoverlap200_part{part}.npz"
    if os.path.exists(out):
        log(f"{out} exists"); return
    law, q, _ = build()
    rng = np.random.default_rng(SEED * 1000 + part)
    chunk = 4000
    lp = np.empty(n); lq = np.empty(n)
    for s in range(0, n, chunk):
        k = min(chunk, n - s)
        Bs = law.sample(k, rng).astype(np.float32)
        Z = rng.standard_normal((k, T)).astype(np.float32)
        lp[s:s + k] = law.loglr(A_SIG * Bs + Z, A_SIG)
        lq[s:s + k] = law.loglr(rng.standard_normal((k, T)).astype(np.float32), A_SIG)
        if (s // chunk) % 50 == 0:
            log(f"  part {part}: {s + k}/{n}")
    np.savez(out + ".tmp.npz", lp=lp, lq=lq, q=q)
    os.replace(out + ".tmp.npz", out)
    log(f"saved {out}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sample":
        sample(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "combine":
        files = sorted(glob.glob("rev_parts/nonoverlap200_part*.npz"))
        lp = np.concatenate([np.load(f)["lp"] for f in files]); lq = np.concatenate([np.load(f)["lq"] for f in files])
        _, q, counts = build()
        pn = np.bincount(counts, minlength=T + 1) / len(counts)
        rows = []
        for d in DELTAS:
            e, lo, hi = eps_interval(lp, lq, d, alpha=0.01)
            ep = eps_poisson_pld(q, T, 1.0, 1.0 / A_SIG, d)
            up = eps_jensen_full(pn, 1.0, 2.0, d)
            rows.append(["nonoverlap200", T, d, len(lp), len(counts), f"{q:.4f}", f"{counts.var()/counts.mean():.2f}", f"{e:.3f}", f"{lo:.3f}", f"{hi:.3f}",
                         f"{ep:.3f}", f"{e/ep:.2f}", f"{lo/ep:.2f}", f"{hi/ep:.2f}", f"{up:.3f}", f"{up/ep:.2f}"])
            log(f"nonoverlap200 delta={d:g}: eps {e:.3f} 99% CI [{lo:.3f}, {hi:.3f}]  Poisson {ep:.3f}  ratio {e/ep:.2f} [{lo/ep:.2f}, {hi/ep:.2f}]  "
                f"Theorem 2 {up:.3f} ({up/ep:.2f}x)  (n={len(lp)})")
        os.makedirs("../results", exist_ok=True)
        with open("../results/rev_round17_nonoverlap_ref.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["law", "T", "delta", "n_mc", "n_windows", "q", "D", "eps_plain", "ci99_lo", "ci99_hi", "eps_poisson", "ratio", "ratio_lo", "ratio_hi", "thm2_upper", "thm2_ratio"])
            w.writerows(rows)
