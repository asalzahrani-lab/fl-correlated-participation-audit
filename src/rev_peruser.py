"""Revision B1: per-device (per-user) privacy on the 3-day window under the running policy (fixed cohort 0.6,
K = 27, seed 0), against the pooled law and the Poisson claim at the population rate.

Three per-device laws, by what the adversary knows about the device:
 (a)  availability MODEL known: a two-state availability chain fitted to the device's own 432-round availability path
      (Laplace-smoothed transition counts), thinned by the device's mean inclusion probability given availability
      s_i = mean_t min(1, K / pool_t) over its available rounds; a 3-state hidden Markov law with deterministic emission.
 (a') the device's own empirical window law (its 24 windows of T = 200 at stride 10); the pooled law is the mixture
      of these, so Proposition 'pooled laws' applies exactly.
 (b)  availability REALISATION known: the server's cohort draws are the only randomness, so B_t are independent
      Bernoulli(A_it * min(1, K / pool_t)); per-round privacy is the composition of subsampled Gaussians with these
      rates (PLD, dp_accounting), the count law is Poisson-binomial.  Window: the device's T = 200 window with the
      largest expected count.
Outputs rev_peruser_devices.csv (one row per device) and rev_peruser_summary.csv; per-round Monte-Carlo references
with 99% intervals for the ten devices with the largest Theorem-3 bound under (a)."""
import csv, sys, time
import numpy as np
from scipy.stats import binom
from rev_common import log, window, apply_policy, eps_interval
from corraudit import count_distribution, eps_sum_exact, eps_poisson_pld
from theory import eps_jensen_full
from certified import ChainLaw

T, SIGMA, C, A_SIG = 200, 2.0, 1.0, 0.5
DELTAS = [1e-4, 1e-5]


def pld_hetero(rates, deltas):
    """eps at each delta of the composition of Poisson-subsampled Gaussians with the given per-round rates; rates are
    rounded UP to multiples of 0.01 (a conservative coarsening) and the PLD uses a 1e-3 discretisation."""
    from dp_accounting import pld, dp_event
    r = np.asarray(rates, float); r = r[r > 0]
    if len(r) == 0:
        return [0.0 for _ in deltas]
    r = np.minimum(1.0, np.ceil(np.round(r / 0.01, 9)) * 0.01)
    evs = []
    for val, cnt in zip(*np.unique(np.round(r, 4), return_counts=True)):
        g = dp_event.GaussianDpEvent(SIGMA / C)
        evs.append(dp_event.SelfComposedDpEvent(g if val >= 1.0 else dp_event.PoissonSampledDpEvent(float(val), g), int(cnt)))
    acc = pld.PLDAccountant(value_discretization_interval=1e-3); acc.compose(dp_event.ComposedDpEvent(evs))
    return [acc.get_epsilon(d) for d in deltas]


def poisson_binomial(rates, T_):
    pn = np.zeros(T_ + 1); pn[0] = 1.0
    for r in rates:
        if r <= 0: continue
        pn[1:] = pn[1:] * (1 - r) + pn[:-1] * r; pn[0] *= (1 - r)
    return pn


def hmm3(p01, p11, s):
    """States: 0 = unavailable, 1 = available not chosen, 2 = available and chosen. Emission B = [0, 0, 1]."""
    P = np.zeros((3, 3))
    for st, pa in [(0, p01), (1, p11), (2, p11)]:
        P[st, 0] = 1 - pa; P[st, 1] = pa * (1 - s); P[st, 2] = pa * s
    stat1 = p01 / (1 - p11 + p01) if (1 - p11 + p01) > 0 else 1.0
    pi = np.array([1 - stat1, stat1 * (1 - s), stat1 * s])
    return P, pi, np.array([0, 0, 1])


if __name__ == "__main__":
    A, _, _ = window(10, 3)
    N, Tw = A.shape
    B = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0))
    pool = A.sum(axis=0); K = int(round(0.6 * pool.mean())); rt = np.minimum(1.0, K / np.maximum(pool, 1))
    q_pop = B[:, :].mean()
    ep_claim = {d: eps_poisson_pld(0.0471, T, C, SIGMA, d) for d in DELTAS}
    fm_binom = {d: eps_sum_exact(binom.pmf(np.arange(T + 1), T, 0.0471), C, SIGMA, T, d) for d in DELTAS}
    log(f"N={N} K={K} q_pop={q_pop:.4f} Poisson claim {ep_claim} FM Binomial {fm_binom}")
    rows = []; chains = {}
    t0 = time.time()
    for i in range(N):
        Ai = A[i].astype(int); pA = Ai.mean()
        row = {"device": i, "avail": pA, "part_rate": B[i].mean()}
        # ---- (a) fitted availability model
        n0 = (Ai[:-1] == 0).sum(); n1 = (Ai[:-1] == 1).sum()
        n01 = ((Ai[:-1] == 0) & (Ai[1:] == 1)).sum(); n11 = ((Ai[:-1] == 1) & (Ai[1:] == 1)).sum()
        if pA == 0:
            for d in DELTAS:
                row[f"a_fm_{d:g}"] = 0.0; row[f"a_thm3_{d:g}"] = 0.0
            row["s_i"] = 0.0
        else:
            p01 = (n01 + 0.5) / (n0 + 1.0); p11 = (n11 + 0.5) / (n1 + 1.0)
            s_i = float(rt[Ai == 1].mean()); row["s_i"] = s_i
            P, pi, emit = hmm3(p01, p11, s_i); chains[i] = (P, pi, emit)
            pn = count_distribution(P, pi, emit, T)
            for d in DELTAS:
                row[f"a_fm_{d:g}"] = eps_sum_exact(pn, C, SIGMA, T, d)
                row[f"a_thm3_{d:g}"] = eps_jensen_full(pn, C, SIGMA, d)
        # ---- (a') own empirical window law
        counts = np.array([B[i, s0:s0 + T].sum() for s0 in range(0, Tw - T + 1, 10)])
        pn_emp = np.bincount(counts, minlength=T + 1) / len(counts)
        for d in DELTAS:
            row[f"emp_fm_{d:g}"] = eps_sum_exact(pn_emp, C, SIGMA, T, d) if counts.max() > 0 else 0.0
        # ---- (b) availability realisation known
        starts = list(range(0, Tw - T + 1, 10))
        exp_cnt = [float((A[i, s0:s0 + T] * rt[s0:s0 + T]).sum()) for s0 in starts]
        s0 = starts[int(np.argmax(exp_cnt))]; rates = A[i, s0:s0 + T] * rt[s0:s0 + T]
        row["b_expected_count"] = max(exp_cnt)
        pn_b = poisson_binomial(rates, T)
        prs = pld_hetero(rates, DELTAS)
        for d, pr in zip(DELTAS, prs):
            row[f"b_fm_{d:g}"] = eps_sum_exact(pn_b, C, SIGMA, T, d) if rates.sum() > 0 else 0.0
            row[f"b_pr_{d:g}"] = pr
        rows.append(row)
        if i % 50 == 0:
            log(f"device {i}/{N}")
    keys = list(rows[0].keys())
    with open("rev_peruser_devices.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    # ---- per-round Monte-Carlo for the 10 devices with the largest Theorem-3 bound under (a)
    top = sorted([r for r in rows if r["device"] in chains], key=lambda r: -r["a_thm3_0.0001"])[:10]
    mc_rows = []
    for r in top:
        i = r["device"]; P, pi, emit = chains[i]; law = ChainLaw(P, pi, emit, T); rng = np.random.default_rng(9100 + i)
        n = 300_000; lp = np.empty(n); lq = np.empty(n)
        for s in range(0, n, 25_000):
            Bp = law.sample(25_000, rng).astype(float)
            lp[s:s + 25_000] = law.loglr(A_SIG * Bp + rng.standard_normal((25_000, T)), A_SIG)
            lq[s:s + 25_000] = law.loglr(rng.standard_normal((25_000, T)), A_SIG)
        out = [i, f"{r['avail']:.3f}", f"{r['s_i']:.3f}"]
        for d in DELTAS:
            e, lo, hi = eps_interval(lp, lq, d, alpha=0.01)
            out += [f"{e:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{r[f'a_thm3_{d:g}']:.3f}", f"{r[f'b_pr_{d:g}']:.3f}"]
        mc_rows.append(out); log(f"device {i}: avail {r['avail']:.3f} per-round (a) eps {out[3]} [{out[4]}, {out[5]}], Thm3 {out[6]}, (b) {out[7]}")
    with open("rev_peruser_mc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "avail", "s_i"] + sum(([f"a_pr_{d:g}", f"a_pr_lo_{d:g}", f"a_pr_hi_{d:g}", f"a_thm3_{d:g}", f"b_pr_{d:g}"] for d in DELTAS), []))
        w.writerows(mc_rows)
    # ---- summary
    summ = []
    for col, ref in [("a_fm", fm_binom), ("emp_fm", fm_binom), ("b_fm", fm_binom), ("a_thm3", ep_claim), ("b_pr", ep_claim)]:
        for d in DELTAS:
            v = np.array([r[f"{col}_{d:g}"] for r in rows])
            qs = np.quantile(v, [0.5, 0.9, 0.99]); mx = v.max(); frac = float((v > ref[d]).mean())
            summ.append([col, d, f"{qs[0]:.3f}", f"{qs[1]:.3f}", f"{qs[2]:.3f}", f"{mx:.3f}", f"{ref[d]:.3f}", f"{mx/ref[d]:.2f}", f"{frac:.3f}"])
            log(f"{col} delta={d:g}: median {qs[0]:.3f} p90 {qs[1]:.3f} p99 {qs[2]:.3f} max {mx:.3f} | reference {ref[d]:.3f} (max ratio {mx/ref[d]:.2f}); share above {frac:.3f}")
    with open("rev_peruser_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["quantity", "delta", "median", "p90", "p99", "max", "reference", "max_over_reference", "share_above_reference"]); w.writerows(summ)
