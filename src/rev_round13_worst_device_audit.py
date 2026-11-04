"""Round 13: the path-aware audit pointed at the worst device's own law (the reviews of TIFS v12 asked what the audit
recovers for the device that matters most, rather than for the pooled law the canaries follow in Section V).

Law.  Position (b) of Section VI-F for the device with the largest certified lower end (device 335, window 21):
given its realised availability A_i(t) and the pool sizes, the device joins round t independently with probability
p_t = A_i(t) min(1, K / pool(t)), K = 27.  An auditor who holds the logs can give a canary group exactly this law.
Because the rounds are independent given the rates, the per-round curve eps(delta; a) of this law is computed without
Monte Carlo, by the PLD accountant (composition of the subsampled Gaussian pairs with sampling probability p_t and noise 1/a):
the optimistic estimate (rounded down, composed without tail truncation) is a lower estimate and the pessimistic
(connect-the-dots) one an upper estimate, so no Monte-Carlo certificate is needed.  The audit table is the optimistic
estimate on the 0.01 grid of a used by every other table (floor rule), at discretisation 1e-3; the reference is the
optimistic estimate at a = 0.5 and discretisation 1e-4 (the 27.07 of Section VI-F), bracketed by the pessimistic one.

Audits.  120 independent data sets (seed 13002): m = 1,000 canaries, T = 200, a = 0.5, present with probability 1/2,
paths drawn from the law above; the lower confidence limit a_lo at level 0.05 (no Monte-Carlo allowance is needed, so
the whole level goes to the estimate), mapped through the table.  Reported: certified mean +- sd, recovery against the
reference, flags against the Poisson claim at the population rate (1.218), and the number of certified values above the
reference (the audit's error rate; the nominal level is 0.05).

    cd results && P3_TRACE=../trace python3 ../src/rev_round13_worst_device_audit.py     # a few minutes
Outputs rev_round13_worst_table.csv (the table), rev_round13_worst_audits.csv (one row per audit) and the printed log.
"""
import csv, time
import numpy as np
from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism as PLD_gauss
from rev_common import log, window, apply_policy
from certified2 import snr_lower_bound
from corraudit import eps_poisson_pld

T, C, SIGMA = 200, 1.0, 2.0
DELTAS = [1e-4, 1e-5]
DEVICE, A_TRUE, M, R, ALPHA, SEED = 335, 0.5, 1000, 120, 0.05, 13002
GRID = np.round(np.arange(0.40, 0.5201, 0.01), 2)


def pld_eps(rates, a, deltas, disc, pessimistic):
    """eps(delta) of the per-round release of a law with independent rounds (rates p_t) and shift a in unit noise."""
    r = np.asarray(rates, float); r = r[r > 0]; pld = None
    for val, cnt in zip(*np.unique(r, return_counts=True)):
        p = PLD_gauss(standard_deviation=1.0 / a, sensitivity=1.0, pessimistic_estimate=pessimistic, value_discretization_interval=disc,
                      sampling_prob=float(min(val, 1.0)), use_connect_dots=pessimistic)
        p = p.self_compose(int(cnt), tail_mass_truncation=0.0) if not pessimistic else p.self_compose(int(cnt))
        pld = p if pld is None else (pld.compose(p, tail_mass_truncation=0.0) if not pessimistic else pld.compose(p))
    return [pld.get_epsilon_for_delta(d) for d in deltas]


def floor_cert(table, a):
    if not np.isfinite(a): return 0.0
    i = np.searchsorted(GRID - 1e-9, a, side="right") - 1
    return 0.0 if i < 0 else float(table[i])


if __name__ == "__main__":
    t0 = time.time()
    A, _, _ = window(10, 3); N, Tw = A.shape
    apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0))             # the running policy; only its K and pool enter below
    pool = A.sum(axis=0); K = int(round(0.6 * pool.mean())); rt = np.minimum(1.0, K / np.maximum(pool, 1))
    rows = list(csv.DictReader(open("rev_peruser_exact2_devices.csv")))
    dev = [r for r in rows if int(r["device"]) == DEVICE][0]; w = int(dev["w_pr_0.0001"]); s0 = 10 * w
    rates = A[DEVICE, s0:s0 + T] * rt[s0:s0 + T]
    log(f"device {DEVICE}: availability {A[DEVICE].mean():.3f}, window {w} (rounds {s0} to {s0 + T - 1}), K={K}, expected participations {rates.sum():.1f}, "
        f"stored lower end {float(dev['pr_lo_0.0001']):.3f} [{float(dev['pr_hi_0.0001']):.3f}] at 1e-4")
    ref_lo = pld_eps(rates, A_TRUE, DELTAS, 1e-4, False); ref_hi = pld_eps(rates, A_TRUE, DELTAS, 1e-4, True)
    check = pld_eps(rates, A_TRUE, DELTAS, 1e-3, False)
    log(f"reference at a=0.5: optimistic {ref_lo[0]:.4f} / {ref_lo[1]:.4f}, pessimistic {ref_hi[0]:.4f} / {ref_hi[1]:.4f} (1e-4); "
        f"optimistic at 1e-3 {check[0]:.4f} (stored {float(dev['pr_lo_0.0001']):.4f}) [{time.time() - t0:.0f}s]")
    claim = {d: eps_poisson_pld(0.0471, T, C, SIGMA, d) for d in DELTAS}
    log(f"Poisson claim at the population rate: {claim[1e-4]:.3f} / {claim[1e-5]:.3f}")
    table = {d: [] for d in DELTAS}; table_hi = {d: [] for d in DELTAS}
    with open("rev_round13_worst_table.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["a", "eps_lo_1e-4", "eps_hi_1e-4", "eps_lo_1e-5", "eps_hi_1e-5"])
        for a in GRID:
            lo = pld_eps(rates, a, DELTAS, 1e-3, False); hi = pld_eps(rates, a, DELTAS, 1e-3, True)
            for k, d in enumerate(DELTAS): table[d].append(lo[k]); table_hi[d].append(hi[k])
            wr.writerow([f"{a:.2f}", f"{lo[0]:.4f}", f"{hi[0]:.4f}", f"{lo[1]:.4f}", f"{hi[1]:.4f}"])
    for d in DELTAS:
        assert all(np.diff(table[d]) > 0), "table not increasing in a"
        log(f"table at {d:g}: {table[d][0]:.3f} (a=0.40) ... {table[d][-1]:.3f} (a=0.52); pessimistic exceeds optimistic by at most "
            f"{max(h - l for h, l in zip(table_hi[d], table[d])):.3f}")
    rng = np.random.default_rng(SEED); out = []
    for r in range(R):
        present = rng.random(M) < 0.5
        B = (rng.random((M, T)) < rates[None, :]).astype(np.int8)
        X = (B * present[:, None]) * A_TRUE + rng.standard_normal((M, T)).astype(np.float32)
        a_lo = snr_lower_bound(X, present, B, ALPHA)
        row = {"audit": r, "n_obs": int(B[present].sum()), "a_lo": a_lo}
        for d in DELTAS:
            row[f"cert_{d:g}"] = floor_cert(table[d], a_lo)
            row[f"flag_{d:g}"] = int(row[f"cert_{d:g}"] > claim[d]); row[f"above_ref_{d:g}"] = int(row[f"cert_{d:g}"] > ref_lo[DELTAS.index(d)])
        out.append(row)
    with open("rev_round13_worst_audits.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys())); wr.writeheader(); [wr.writerow(x) for x in out]
    al = np.array([x["a_lo"] for x in out])
    log(f"{R} audits, m={M}: n_obs mean {np.mean([x['n_obs'] for x in out]):.0f}; a_lo mean {al.mean():.4f} sd {al.std():.4f} min {al.min():.4f}, "
        f"coverage of a_lo <= a: {(al <= A_TRUE).mean():.3f}")
    for k, d in enumerate(DELTAS):
        c = np.array([x[f"cert_{d:g}"] for x in out])
        log(f"delta={d:g}: certified {c.mean():.2f} +- {c.std():.2f} (recovery {c.mean() / ref_lo[k]:.1%} of {ref_lo[k]:.2f}); flags {int(sum(x[f'flag_{d:g}'] for x in out))}/{R} "
            f"against the claim {claim[d]:.3f}; above the reference {int(sum(x[f'above_ref_{d:g}'] for x in out))}/{R}; min {c.min():.2f} max {c.max():.2f}")
    log(f"done in {time.time() - t0:.0f}s")
