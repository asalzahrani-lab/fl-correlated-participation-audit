"""Revision round 10 (internal review), M3: rev_peruser_exact2.py at delta = 1e-6 and 1e-8, unchanged otherwise
(position (b): availability known; certified PLD bracket lower end, Theorem 2 upper end and exact sum statistic, each the
maximum over the device's 24 windows of T = 200; certified window search; top-20 all-window check; worst device at a
1e-4 discretisation).  The Poisson claim and the Binomial sum-statistic value are computed at the population rate
0.0471, as in rev_peruser_exact2.py.  Inversions of Theorem 2 use an upper search limit of 400.
Optional arguments: two deltas (default 1e-6 1e-8); other deltas add a suffix to the output names.
Outputs rev_round10_peruser_devices.csv, rev_round10_peruser_summary.csv, rev_round10_peruser_top20.csv."""
import csv, sys, time, logging
import numpy as np
from scipy.stats import binom
from rev_common import log, window, apply_policy
from corraudit import eps_sum_exact, eps_poisson_pld
from theory import h_gauss
from corraudit import eps_from_delta_curve


def eps_jensen_full(pn, C, sigma, delta):
    n = np.arange(len(pn)); mu = np.sqrt(n) * C / sigma
    e = eps_from_delta_curve(lambda x: float(np.sum(pn * h_gauss(mu, x))), delta, 0.0, 400.0)
    assert e < 400.0 - 1e-6
    return e
from dp_accounting.pld import privacy_loss_distribution as PLD

logging.getLogger().setLevel(logging.ERROR)   # silence the privacy-buckets notice of the optimistic estimate
T, SIGMA, C = 200, 2.0, 1.0
DELTAS = [float(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1e-6, 1e-8]   # e.g. python rev_round10_peruser_delta.py 1e-10
TAG = "" if DELTAS == [1e-6, 1e-8] else "_" + "_".join(f"{d:g}" for d in DELTAS)
TOP = 20


def pld_eps(rates, deltas, disc, pessimistic):
    """eps(delta) of the composed scalar pairs; pessimistic -> upper estimate, optimistic -> lower estimate."""
    r = np.asarray(rates, float); r = r[r > 0]
    if len(r) == 0: return [0.0 for _ in deltas]
    pld = None
    for val, cnt in zip(*np.unique(r, return_counts=True)):
        p = PLD.from_gaussian_mechanism(standard_deviation=SIGMA / C, sensitivity=1.0, pessimistic_estimate=pessimistic,
                                        value_discretization_interval=disc, sampling_prob=float(min(val, 1.0)),
                                        use_connect_dots=pessimistic)
        # optimistic: no tail truncation in any composition, so the PMF is a lower estimate without an added infinity
        # mass; pessimistic: the library default (truncated tail mass added to the infinity mass) is conservative.
        p = p.self_compose(int(cnt), tail_mass_truncation=0.0) if not pessimistic else p.self_compose(int(cnt))
        pld = p if pld is None else (pld.compose(p, tail_mass_truncation=0.0) if not pessimistic else pld.compose(p))
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
    row["w_pr"] = row[f"w_pr_{DELTAS[0]:g}"]
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
                log(f"device {i}/{N}: pr [{row[f'pr_lo_{DELTAS[0]:g}']:.3f}, {row[f'pr_hi_{DELTAS[0]:g}']:.3f}] window {row['w_pr']} (refined {row['n_refined']} of {W}), "
                    f"thm3 {row[f'thm3_{DELTAS[0]:g}']:.2f}, fm {row[f'fm_{DELTAS[0]:g}']:.2f}  [{time.time()-t0:.0f}s]")
    keys = list(rows[-1].keys())
    with open(f"rev_round10_peruser_devices{TAG}.csv", "w", newline="") as f:
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
    gap = np.array([r[f"pr_hi_{DELTAS[0]:g}"] - r[f"pr_lo_{DELTAS[0]:g}"] for r in rows])[ever]
    log(f"numerical gap pr_hi - pr_lo at {DELTAS[0]:g}: median {np.median(gap):.4f}, max {gap.max():.4f}; relative max {(gap / np.maximum(np.array([r[f'pr_lo_{DELTAS[0]:g}'] for r in rows])[ever], 1e-9)).max():.4f}")
    moved = sum(1 for r in rows if r["avail"] > 0 and r["w_pr"] != r["w_exp"])
    gain = np.array([r[f"pr_lo_{DELTAS[0]:g}"] - r[f"pr_expw_lo_{DELTAS[0]:g}"] for r in rows if r["avail"] > 0])
    log(f"devices whose worst window differs from the expected-count window: {moved} of {ever.sum()}; max gain {gain.max():.3f}")
    with open(f"rev_round10_peruser_summary{TAG}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["quantity", "delta", "devices", "median", "p90", "p99", "max", "argmax_device", "reference", "max_over_reference",
                                       "share_above_reference", "n_above_reference", "median_over_reference"]); w.writerows(summ)
    # direct check: all 24 windows at 1e-3 for the TOP devices by pr_lo, and 1e-4 for the worst device's window
    order = sorted([r for r in rows if r["avail"] > 0], key=lambda r: -r[f"pr_lo_{DELTAS[0]:g}"])[:TOP]; top = []
    for r in order:
        i = r["device"]; rates_w = [A[i, s0:s0 + T] * rt[s0:s0 + T] for s0 in starts]
        lo = np.array([pld_eps(rw, DELTAS, 1e-3, False) for rw in rates_w]); hi = np.array([pld_eps(rw, DELTAS, 1e-3, True) for rw in rates_w])
        wl = int(np.argmax(lo[:, 0])); same = int(wl == r["w_pr"])
        top.append([i, r["w_pr"], wl, same, f"{lo[:, 0].max():.4f}", f"{hi[:, 0].max():.4f}", f"{lo[:, 1].max():.4f}", f"{hi[:, 1].max():.4f}", f"{r[f'pr_lo_{DELTAS[0]:g}']:.4f}"])
        log(f"top device {i}: all-window max lo {lo[:, 0].max():.4f} hi {hi[:, 0].max():.4f} at window {wl} (certified search chose {r['w_pr']}, {r[f'pr_lo_{DELTAS[0]:g}']:.4f}) same={same}")
    with open(f"rev_round10_peruser_top20{TAG}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["device", "w_certified", "w_all_windows", "same", f"lo_{DELTAS[0]:g}", f"hi_{DELTAS[0]:g}", f"lo_{DELTAS[1]:g}", f"hi_{DELTAS[1]:g}", f"certified_lo_{DELTAS[0]:g}"]); w.writerows(top)
    i = order[0]["device"]; s0 = starts[order[0]["w_pr"]]; rw = A[i, s0:s0 + T] * rt[s0:s0 + T]
    try:
        lo4 = pld_eps(rw, DELTAS, 1e-4, False); hi4 = pld_eps(rw, DELTAS, 1e-4, True)
        log(f"worst device {i} at discretisation 1e-4: [{lo4[0]:.4f}, {hi4[0]:.4f}] at {DELTAS[0]:g}, [{lo4[1]:.4f}, {hi4[1]:.4f}] at {DELTAS[1]:g}")
    except Exception as e:
        log(f"1e-4 refinement failed: {e}")
    log(f"done in {time.time()-t0:.0f}s")
