"""Round 13: sampling uncertainty of the trace numbers that come from the finite device population (the reviews of
TIFS v12 asked how much the exact accountant's output moves with the estimation of the participation law).

Nonparametric bootstrap over devices.  The 572 devices of the 3-day, 10-minute window are resampled with replacement
R times (replicate r uses seed 13001 + r); the policy is re-applied to the resampled population with its own cohort
size (fixed cohort: K = round(0.6 x the resampled population's mean pool), policy seed 0, as in the device-subsampling
check of round 7); the count law over the 24 windows of T = 200 at stride 10 gives, at delta = 1e-4 and 1e-5,
  (i)   the exact final-model eps of the empirical law (Theorem 1) and of Binomial(T, q) at the replicate's q, and
        their ratio (the paper's final-model ratio, 5.16 for the running example);
  (ii)  the count-averaged bound (Theorem 2) and its ratio to the Poisson per-round eps at the replicate's q
        (PLD accountant, discretisation 1e-4), the paper's upper end of the pooled per-round bracket (19.94, 16.4x).
Two policies: the running example (fixed cohort 0.6) and take-all (the 9.44 of Table V).  Percentile intervals
(2.5 %, 97.5 %) over replicates are reported next to the analysis population's own values.  The device-level laws
(Section VI-F) are not resampled: a device's bracket is a property of its own windows.

    cd results && P3_TRACE=../trace python3 ../src/rev_round13_bootstrap.py [R]      # R = 500 by default; the published run used R = 1000 (~22 min)
Outputs rev_round13_bootstrap.csv (one row per replicate and policy) and rev_round13_bootstrap_summary.csv.
"""
import csv, sys, time
import numpy as np
from scipy.stats import binom
from rev_common import log, window, apply_policy
from corraudit import eps_sum_exact
from theory import eps_jensen_full

T, C, SIGMA, STRIDE = 200, 1.0, 2.0, 10
DELTAS = [1e-4, 1e-5]
POLICIES = [("fixed cohort 0.6", "fixed", 0.6), ("take all available", "thin", 1.0)]


def eps_poisson(q, delta, disc=1e-4):
    from dp_accounting import dp_event, pld
    acc = pld.PLDAccountant(value_discretization_interval=disc)
    acc.compose(dp_event.SelfComposedDpEvent(dp_event.PoissonSampledDpEvent(float(q), dp_event.GaussianDpEvent(SIGMA / C)), T))
    return acc.get_epsilon(delta)


def window_counts(B):
    cs = np.concatenate([np.zeros((B.shape[0], 1), np.int64), np.cumsum(B, axis=1, dtype=np.int64)], axis=1)
    starts = np.arange(0, B.shape[1] - T + 1, STRIDE)
    return (cs[:, starts + T] - cs[:, starts]).ravel()


def quantities(B):
    counts = window_counts(B); pn = np.bincount(counts, minlength=T + 1) / len(counts)
    q = counts.mean() / T; D = counts.var() / counts.mean()
    out = {"q": q, "D": D, "maxN": int(counts.max())}
    pb = binom.pmf(np.arange(T + 1), T, q)
    for d in DELTAS:
        fm = eps_sum_exact(pn, C, SIGMA, T, d); fmb = eps_sum_exact(pb, C, SIGMA, T, d)
        th = eps_jensen_full(pn, C, SIGMA, d); po = eps_poisson(q, d)
        out.update({f"fm_{d:g}": fm, f"fm_binom_{d:g}": fmb, f"fm_ratio_{d:g}": fm / fmb,
                    f"thm2_{d:g}": th, f"poisson_{d:g}": po, f"thm2_ratio_{d:g}": th / po})
    return out


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    A, _, _ = window(10, 3); N = A.shape[0]
    keys = None; rows = []; t0 = time.time()
    for name, kind, par in POLICIES:
        point = quantities(apply_policy(A, kind, par, 0, np.random.default_rng(0)))
        log(f"{name}, analysis population ({N} devices): q={point['q']:.4f} D={point['D']:.2f} final-model ratio {point['fm_ratio_0.0001']:.2f} / "
            f"{point['fm_ratio_1e-05']:.2f}; Theorem 2 {point['thm2_0.0001']:.2f} = {point['thm2_ratio_0.0001']:.2f}x Poisson {point['poisson_0.0001']:.3f}")
        rows.append(dict(policy=name, replicate=-1, **point))
        for r in range(R):
            idx = np.random.default_rng(13001 + r).integers(0, N, N)
            out = quantities(apply_policy(A[idx], kind, par, 0, np.random.default_rng(0)))
            rows.append(dict(policy=name, replicate=r, **out))
            if (r + 1) % 50 == 0:
                log(f"  {name}: {r + 1}/{R} replicates [{time.time() - t0:.0f}s]")
    keys = list(rows[0].keys())
    with open("rev_round13_bootstrap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(x) for x in rows]
    summ = []
    for name, _, _ in POLICIES:
        pt = [x for x in rows if x["policy"] == name and x["replicate"] == -1][0]
        reps = [x for x in rows if x["policy"] == name and x["replicate"] >= 0]
        for key in [k for k in keys if k not in ("policy", "replicate")]:
            v = np.array([x[key] for x in reps]); lo, md, hi = np.quantile(v, [0.025, 0.5, 0.975])
            summ.append([name, key, len(v), f"{pt[key]:.4f}", f"{md:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{v.min():.4f}", f"{v.max():.4f}"])
        log(f"{name}: final-model ratio 1e-4 point {pt['fm_ratio_0.0001']:.2f}, bootstrap 95% [{np.quantile([x['fm_ratio_0.0001'] for x in reps], .025):.2f}, "
            f"{np.quantile([x['fm_ratio_0.0001'] for x in reps], .975):.2f}]; Theorem 2 ratio 1e-4 point {pt['thm2_ratio_0.0001']:.2f}, 95% "
            f"[{np.quantile([x['thm2_ratio_0.0001'] for x in reps], .025):.2f}, {np.quantile([x['thm2_ratio_0.0001'] for x in reps], .975):.2f}]; "
            f"D point {pt['D']:.1f}, 95% [{np.quantile([x['D'] for x in reps], .025):.1f}, {np.quantile([x['D'] for x in reps], .975):.1f}]")
    with open("rev_round13_bootstrap_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["policy", "quantity", "replicates", "point", "median", "q025", "q975", "min", "max"]); w.writerows(summ)
    log(f"done in {time.time() - t0:.0f}s")
