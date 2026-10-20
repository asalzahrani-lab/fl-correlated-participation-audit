"""Revision round 2, B1-B2: per-device privacy with the availability realisation known (position (b)), computed
 (i) with the EXACT per-round inclusion probabilities A_it * min(1, K/pool_t) (no rounding; the distinct values of
     K/pool_t are few, so the events are grouped exactly), and
 (ii) as the MAXIMUM over the device's 24 windows of T = 200 (stride 10), not the window with the largest expected
     count: a coarse PLD (discretisation 1e-2) screens all windows, the best is recomputed at 1e-3.
For each device and window we also compute the count-averaged value of Theorem 3 on the Poisson-binomial count law,
Sum_n p_n h_eps(a sqrt n), which is exact for an adversary who knows availability AND the cohort draws (Proposition 2(i) of the full version v8)
and an upper bound on the loss of any weaker adversary; the per-round PLD value is the loss of the observation model
(other clients' projected contribution independent of the target's inclusion) and a lower bound on the mechanism's
worst-case loss (Section III-C).  The final-model value (Theorem 1) is exact under the Poisson-binomial law.
Outputs rev_peruser_exact_devices.csv (per device: both maxima, the windows that attain them, the expected-count
window values) and rev_peruser_exact_summary.csv."""
import csv, time
import numpy as np
from scipy.stats import binom
from rev_common import log, window, apply_policy
from corraudit import eps_sum_exact, eps_poisson_pld
from theory import eps_jensen_full
from dp_accounting import pld, dp_event

T, SIGMA, C = 200, 2.0, 1.0
DELTAS = [1e-4, 1e-5]


def pld_exact(rates, deltas, disc):
    r = np.asarray(rates, float); r = r[r > 0]
    if len(r) == 0: return [0.0 for _ in deltas]
    evs = []
    for val, cnt in zip(*np.unique(np.round(r, 9), return_counts=True)):
        g = dp_event.GaussianDpEvent(SIGMA / C)
        evs.append(dp_event.SelfComposedDpEvent(g if val >= 1.0 else dp_event.PoissonSampledDpEvent(float(val), g), int(cnt)))
    acc = pld.PLDAccountant(value_discretization_interval=disc); acc.compose(dp_event.ComposedDpEvent(evs))
    return [acc.get_epsilon(d) for d in deltas]


def poisson_binomial(rates, T_):
    pn = np.zeros(T_ + 1); pn[0] = 1.0
    for r in rates:
        if r <= 0: continue
        pn[1:] = pn[1:] * (1 - r) + pn[:-1] * r; pn[0] *= (1 - r)
    return pn


if __name__ == "__main__":
    A, _, _ = window(10, 3); N, Tw = A.shape
    B = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0))
    pool = A.sum(axis=0); K = int(round(0.6 * pool.mean())); rt = np.minimum(1.0, K / np.maximum(pool, 1))
    q_pop = 0.0471
    claim = {d: eps_poisson_pld(q_pop, T, C, SIGMA, d) for d in DELTAS}
    fm_binom = {d: eps_sum_exact(binom.pmf(np.arange(T + 1), T, q_pop), C, SIGMA, T, d) for d in DELTAS}
    log(f"K={K} claim {claim} FM Binomial {fm_binom}")
    starts = list(range(0, Tw - T + 1, 10)); rows = []; t0 = time.time()
    for i in range(N):
        if A[i].sum() == 0:
            rows.append(dict(device=i, avail=0.0, **{f"{k}_{d:g}": 0.0 for k in ("pr", "pr_expw", "thm3", "fm") for d in DELTAS},
                             **{"w_pr": -1, "w_thm3": -1, "w_fm": -1, "w_exp": -1, "n_windows_pr_within_1pct": 0})); continue
        exp_cnt = [float((A[i, s0:s0 + T] * rt[s0:s0 + T]).sum()) for s0 in starts]; w_exp = int(np.argmax(exp_cnt))
        coarse = []; thm3 = []; fm = []
        for s0 in starts:
            rates = A[i, s0:s0 + T] * rt[s0:s0 + T]
            if rates.sum() == 0:
                coarse.append([0.0, 0.0]); thm3.append([0.0, 0.0]); fm.append([0.0, 0.0]); continue
            coarse.append(pld_exact(rates, DELTAS, 1e-2))
            pn = poisson_binomial(rates, T)
            thm3.append([eps_jensen_full(pn, C, SIGMA, d) for d in DELTAS])
            fm.append([eps_sum_exact(pn, C, SIGMA, T, d) for d in DELTAS])
        coarse = np.array(coarse); thm3 = np.array(thm3); fm = np.array(fm)
        # refine: every window whose coarse value is within 2% of the coarse maximum (at either delta)
        cand = set()
        for k, d in enumerate(DELTAS):
            cand |= set(np.flatnonzero(coarse[:, k] >= 0.98 * coarse[:, k].max()).tolist())
        cand.add(w_exp)
        fine = {w: pld_exact(A[i, starts[w]:starts[w] + T] * rt[starts[w]:starts[w] + T], DELTAS, 1e-3) for w in cand}
        row = dict(device=i, avail=float(A[i].mean()), w_exp=w_exp, n_windows_pr_within_1pct=int((coarse[:, 0] >= 0.99 * coarse[:, 0].max()).sum()))
        for k, d in enumerate(DELTAS):
            wbest = max(cand, key=lambda w: fine[w][k]); row[f"pr_{d:g}"] = fine[wbest][k]; row[f"w_pr"] = wbest
            row[f"pr_expw_{d:g}"] = fine[w_exp][k]
            row[f"thm3_{d:g}"] = float(thm3[:, k].max()); row["w_thm3"] = int(np.argmax(thm3[:, k]))
            row[f"fm_{d:g}"] = float(fm[:, k].max()); row["w_fm"] = int(np.argmax(fm[:, k]))
        rows.append(row)
        if i % 25 == 0:
            log(f"device {i}/{N}: pr max {row['pr_0.0001']:.2f} (expected-count window {row['pr_expw_0.0001']:.2f}), thm3 {row['thm3_0.0001']:.2f}, fm {row['fm_0.0001']:.2f}")
    keys = list(rows[-1].keys())
    with open("rev_peruser_exact_devices.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    ever = np.array([r["avail"] > 0 for r in rows]); summ = []
    for col, ref in [("pr", claim), ("pr_expw", claim), ("thm3", claim), ("fm", fm_binom)]:
        for d in DELTAS:
            v = np.array([r[f"{col}_{d:g}"] for r in rows])[ever]
            qs = np.quantile(v, [0.5, 0.9, 0.99])
            summ.append([col, d, len(v), f"{qs[0]:.3f}", f"{qs[1]:.3f}", f"{qs[2]:.3f}", f"{v.max():.3f}", int(np.array([r['device'] for r in rows])[ever][np.argmax(v)]),
                         f"{ref[d]:.3f}", f"{v.max()/ref[d]:.2f}", f"{(v > ref[d]).mean():.3f}", f"{np.median(v)/ref[d]:.2f}"])
            log(f"{col} delta={d:g}: median {qs[0]:.3f} p90 {qs[1]:.3f} p99 {qs[2]:.3f} max {v.max():.3f} (device {summ[-1][7]}) | ref {ref[d]:.3f} max ratio {v.max()/ref[d]:.2f} median ratio {np.median(v)/ref[d]:.2f} share above {(v > ref[d]).mean():.3f}")
    moved = sum(1 for r in rows if r["avail"] > 0 and r["w_pr"] != r["w_exp"])
    gain = np.array([r["pr_0.0001"] - r["pr_expw_0.0001"] for r in rows if r["avail"] > 0])
    log(f"devices whose worst window differs from the expected-count window: {moved} of {ever.sum()}; max gain {gain.max():.3f}, mean gain {gain.mean():.4f}")
    with open("rev_peruser_exact_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["quantity", "delta", "devices", "median", "p90", "p99", "max", "argmax_device", "reference", "max_over_reference", "share_above_reference", "median_over_reference"]); w.writerows(summ)
    log(f"done in {time.time()-t0:.0f}s")
