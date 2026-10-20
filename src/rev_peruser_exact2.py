"""Revision round 3, B1: per-device privacy with the availability realisation known (position (b)), as in
rev_peruser_exact.py, but with a CERTIFIED numerical bracket and a certified window search.

The per-round value of the observation model is the eps(delta) of the composition over the window of the scalar pairs
with the exact inclusion rates A_it * min(1, K/pool_t).  dp_accounting's PLD gives two estimates of that value:
  pessimistic (connect-the-dots, rounding up, truncated mass added to delta)   -> an UPPER estimate of eps(delta);
  optimistic  (privacy buckets, rounding down, truncated mass dropped)         -> a LOWER estimate.
The published lower end of Proposition 2's bracket is therefore the OPTIMISTIC estimate at discretisation 1e-3
(pr_lo); the pessimistic estimate (pr_hi) bounds the same quantity from above, and the gap is the numerical error.

Window search.  Every one of the 24 windows (T = 200, stride 10) is screened with the pessimistic estimate at
discretisation 1e-2, which is an upper bound on its true value.  Candidates (within 2% of the coarse maximum, plus the
expected-count window) are computed at 1e-3 with both estimates; the worst window w* maximises the fine optimistic
value.  Certificate: if pr_lo(w*) >= coarse pessimistic value of every window not refined, no unrefined window can
exceed w*; otherwise the offending windows are refined too (loop until the certificate holds).  For the 20 devices with
the largest pr_lo all 24 windows are additionally computed at 1e-3 as a direct check, and for the worst device the
chosen window is recomputed at 1e-4.

Theorem 3 (count-averaged, upper end) and Theorem 1 (final model) are exact on the Poisson-binomial count law.
Outputs rev_peruser_exact2_devices.csv, rev_peruser_exact2_summary.csv, rev_peruser_exact2_top20.csv."""
import csv, time, logging
import numpy as np
from scipy.stats import binom
from rev_common import log, window, apply_policy
from corraudit import eps_sum_exact, eps_poisson_pld
from theory import eps_jensen_full
from dp_accounting.pld import privacy_loss_distribution as PLD

logging.getLogger().setLevel(logging.ERROR)   # silence the privacy-buckets notice of the optimistic estimate
T, SIGMA, C = 200, 2.0, 1.0
DELTAS = [1e-4, 1e-5]
TOP = 20


def pld_eps(rates, deltas, disc, pessimistic):
    """eps(delta) of the composed scalar pairs; pessimistic -> upper estimate, optimistic -> lower estimate."""
    r = np.asarray(rates, float); r = r[r > 0]
    if len(r) == 0: return [0.0 for _ in deltas]
    pld = None
    for val, cnt in zip(*np.unique(np.round(r, 9), return_counts=True)):
        p = PLD.from_gaussian_mechanism(standard_deviation=SIGMA / C, sensitivity=1.0, pessimistic_estimate=pessimistic,
                                        value_discretization_interval=disc, sampling_prob=float(min(val, 1.0)),
                                        use_connect_dots=pessimistic).self_compose(int(cnt))
        pld = p if pld is None else pld.compose(p)
    return [pld.get_epsilon_for_delta(d) for d in deltas]


def poisson_binomial(rates, T_):
    pn = np.zeros(T_ + 1); pn[0] = 1.0
    for r in rates:
        if r <= 0: continue
        pn[1:] = pn[1:] * (1 - r) + pn[:-1] * r; pn[0] *= (1 - r)
    return pn

G = {}


def compute_device(i):
    A, rt, starts, W = G["A"], G["rt"], G["starts"], G["W"]
    if A[i].sum() == 0:
        return dict(device=i, avail=0.0, w_exp=-1, w_pr=-1, **{f"w_pr_{d:g}": -1 for d in DELTAS}, w_thm3=-1, w_fm=-1, n_refined=0, certified=1,
                    **{f"{k}_{d:g}": 0.0 for k in ("pr_lo", "pr_hi", "pr_expw_lo", "thm3", "fm", "max_coarse_unrefined") for d in DELTAS})
    rates_w = [A[i, s0:s0 + T] * rt[s0:s0 + T] for s0 in starts]
    exp_cnt = [float(r.sum()) for r in rates_w]; w_exp = int(np.argmax(exp_cnt))
    coarse = np.array([pld_eps(r, DELTAS, 1e-2, True) for r in rates_w])         # upper bounds, every window
    pn_w = [poisson_binomial(r, T) for r in rates_w]
    thm3 = np.array([[eps_jensen_full(pn, C, SIGMA, d) for d in DELTAS] for pn in pn_w])
    fm = np.array([[eps_sum_exact(pn, C, SIGMA, T, d) for d in DELTAS] for pn in pn_w])
    cand = {w_exp}
    for k in range(len(DELTAS)):
        cand |= set(np.flatnonzero(coarse[:, k] >= 0.98 * coarse[:, k].max()).tolist())
    fine_lo, fine_hi = {}, {}
    def refine(w):
        fine_lo[w] = pld_eps(rates_w[w], DELTAS, 1e-3, False); fine_hi[w] = pld_eps(rates_w[w], DELTAS, 1e-3, True)
    for w in cand: refine(w)
    for k in range(len(DELTAS)):
        while True:
            wbest = max(fine_lo, key=lambda w: fine_lo[w][k]); lo = fine_lo[wbest][k]
            offenders = [w for w in range(W) if w not in fine_lo and coarse[w, k] > lo]
            if not offenders: break
            for w in offenders: refine(w)
    row = dict(device=i, avail=float(A[i].mean()), w_exp=w_exp, n_refined=len(fine_lo), certified=1)
    for k, d in enumerate(DELTAS):
        wbest = max(fine_lo, key=lambda w: fine_lo[w][k])
        row[f"pr_lo_{d:g}"] = fine_lo[wbest][k]; row[f"w_pr_{d:g}"] = wbest
        unref = [coarse[w, k] for w in range(W) if w not in fine_lo]
        row[f"max_coarse_unrefined_{d:g}"] = max(unref) if unref else 0.0
        row[f"pr_hi_{d:g}"] = max([fine_hi[w][k] for w in fine_hi] + unref)
        row[f"pr_expw_lo_{d:g}"] = fine_lo[w_exp][k]
        row[f"thm3_{d:g}"] = float(thm3[:, k].max()); row["w_thm3"] = int(np.argmax(thm3[:, k]))
        row[f"fm_{d:g}"] = float(fm[:, k].max()); row["w_fm"] = int(np.argmax(fm[:, k]))
        if row[f"max_coarse_unrefined_{d:g}"] > row[f"pr_lo_{d:g}"]: row["certified"] = 0
    row["w_pr"] = row["w_pr_0.0001"]
    return row


if __name__ == "__main__":
    A, _, _ = window(10, 3); N, Tw = A.shape
    B = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0))
    pool = A.sum(axis=0); K = int(round(0.6 * pool.mean())); rt = np.minimum(1.0, K / np.maximum(pool, 1))
    q_pop = 0.0471
    claim = {d: eps_poisson_pld(q_pop, T, C, SIGMA, d) for d in DELTAS}
    fm_binom = {d: eps_sum_exact(binom.pmf(np.arange(T + 1), T, q_pop), C, SIGMA, T, d) for d in DELTAS}
    log(f"K={K} claim {claim} FM Binomial {fm_binom}")
    starts = list(range(0, Tw - T + 1, 10)); W = len(starts); t0 = time.time()
    G.update(A=A, rt=rt, starts=starts, W=W)
    from multiprocessing import Pool
    rows = []; n_refined_total = 0; cert_fail = 0
    with Pool(2) as pool_:
        for row in pool_.imap(compute_device, range(N), chunksize=4):
            rows.append(row); n_refined_total += row["n_refined"]; cert_fail += 1 - row["certified"]
            i = row["device"]
            if i % 25 == 0 and row["avail"] > 0:
                log(f"device {i}/{N}: pr [{row['pr_lo_0.0001']:.3f}, {row['pr_hi_0.0001']:.3f}] window {row['w_pr']} (refined {row['n_refined']} of {W}), "
                    f"thm3 {row['thm3_0.0001']:.2f}, fm {row['fm_0.0001']:.2f}  [{time.time()-t0:.0f}s]")
    keys = list(rows[-1].keys())
    with open("rev_peruser_exact2_devices.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    ever = np.array([r["avail"] > 0 for r in rows]); devs = np.array([r["device"] for r in rows])
    log(f"windows refined at 1e-3: {n_refined_total} of {int(ever.sum()) * W}; devices with a failed certificate: {cert_fail}")
    summ = []
    for col, ref in [("pr_lo", claim), ("pr_hi", claim), ("pr_expw_lo", claim), ("thm3", claim), ("fm", fm_binom)]:
        for d in DELTAS:
            v = np.array([r[f"{col}_{d:g}"] for r in rows])[ever]; qs = np.quantile(v, [0.5, 0.9, 0.99])
            summ.append([col, d, len(v), f"{qs[0]:.3f}", f"{qs[1]:.3f}", f"{qs[2]:.3f}", f"{v.max():.3f}", int(devs[ever][np.argmax(v)]),
                         f"{ref[d]:.3f}", f"{v.max()/ref[d]:.2f}", f"{(v > ref[d]).mean():.4f}", int((v > ref[d]).sum()), f"{np.median(v)/ref[d]:.2f}"])
            log(f"{col} delta={d:g}: median {qs[0]:.3f} p90 {qs[1]:.3f} p99 {qs[2]:.3f} max {v.max():.3f} (device {summ[-1][7]}) | ref {ref[d]:.3f} "
                f"max ratio {v.max()/ref[d]:.2f} median ratio {np.median(v)/ref[d]:.2f} above {(v > ref[d]).sum()} of {len(v)}")
    gap = np.array([r["pr_hi_0.0001"] - r["pr_lo_0.0001"] for r in rows])[ever]
    log(f"numerical gap pr_hi - pr_lo at 1e-4: median {np.median(gap):.4f}, max {gap.max():.4f}; relative max {(gap / np.maximum(np.array([r['pr_lo_0.0001'] for r in rows])[ever], 1e-9)).max():.4f}")
    moved = sum(1 for r in rows if r["avail"] > 0 and r["w_pr"] != r["w_exp"])
    gain = np.array([r["pr_lo_0.0001"] - r["pr_expw_lo_0.0001"] for r in rows if r["avail"] > 0])
    log(f"devices whose worst window differs from the expected-count window: {moved} of {ever.sum()}; max gain {gain.max():.3f}")
    with open("rev_peruser_exact2_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["quantity", "delta", "devices", "median", "p90", "p99", "max", "argmax_device", "reference", "max_over_reference",
                                       "share_above_reference", "n_above_reference", "median_over_reference"]); w.writerows(summ)
    # direct check: all 24 windows at 1e-3 for the TOP devices by pr_lo, and 1e-4 for the worst device's window
    order = sorted([r for r in rows if r["avail"] > 0], key=lambda r: -r["pr_lo_0.0001"])[:TOP]; top = []
    for r in order:
        i = r["device"]; rates_w = [A[i, s0:s0 + T] * rt[s0:s0 + T] for s0 in starts]
        lo = np.array([pld_eps(rw, DELTAS, 1e-3, False) for rw in rates_w]); hi = np.array([pld_eps(rw, DELTAS, 1e-3, True) for rw in rates_w])
        wl = int(np.argmax(lo[:, 0])); same = int(wl == r["w_pr"])
        top.append([i, r["w_pr"], wl, same, f"{lo[:, 0].max():.4f}", f"{hi[:, 0].max():.4f}", f"{lo[:, 1].max():.4f}", f"{hi[:, 1].max():.4f}", f"{r['pr_lo_0.0001']:.4f}"])
        log(f"top device {i}: all-window max lo {lo[:, 0].max():.4f} hi {hi[:, 0].max():.4f} at window {wl} (certified search chose {r['w_pr']}, {r['pr_lo_0.0001']:.4f}) same={same}")
    with open("rev_peruser_exact2_top20.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["device", "w_certified", "w_all_windows", "same", "lo_1e-4", "hi_1e-4", "lo_1e-5", "hi_1e-5", "certified_lo_1e-4"]); w.writerows(top)
    i = order[0]["device"]; s0 = starts[order[0]["w_pr"]]; rw = A[i, s0:s0 + T] * rt[s0:s0 + T]
    try:
        lo4 = pld_eps(rw, DELTAS, 1e-4, False); hi4 = pld_eps(rw, DELTAS, 1e-4, True)
        log(f"worst device {i} at discretisation 1e-4: [{lo4[0]:.4f}, {hi4[0]:.4f}] at 1e-4, [{lo4[1]:.4f}, {hi4[1]:.4f}] at 1e-5")
    except Exception as e:
        log(f"1e-4 refinement failed: {e}")
    log(f"done in {time.time()-t0:.0f}s")
