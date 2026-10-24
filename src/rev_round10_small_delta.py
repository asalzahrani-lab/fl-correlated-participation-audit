"""Revision round 10 (internal review), M3: the pooled ratios at deployment-scale delta.
For the two-state chain (lambda = 0.9, q = 0.1) and the running example of the trace (fixed cohort 0.6, seed 0;
T = 200 on the 10-minute 3-day window, stride 10; T = 1,000 on the 5-minute 4-day window, stride 5), at
delta in {1e-4, 1e-5, 1e-6, 1e-8, 1e-10} (Gboard reports at 1e-10):
  Poisson per-round eps (PLD accountant of dp-accounting at the law's rate),
  exact sum-statistic eps under the law and under the Binomial count law of equal rate (Theorem 1) and their ratio,
  Theorem 2 (count-averaged upper bound) on the law's count law and its ratio to the Poisson value.
All are closed-form or PLD computations; the Monte-Carlo per-round references are not extended below 1e-5 because the
tail mass that sets them is not resolved at affordable sample sizes.
Numerical checks: Theorem 1 recomputed on a 4x finer grid; Poisson PLD recomputed at a 10x finer discretisation;
every inversion uses an upper search limit of 400 and asserts that it is not reached.
Also the sum-statistic ratio of the ten policies of Table V at the same values of delta.
Outputs rev_round10_small_delta.csv and rev_round10_small_delta_policies.csv."""
import csv, time
import numpy as np
from scipy.stats import binom
from corraudit import two_state_chain, count_distribution, eps_sum_exact, eps_from_delta_curve
from theory import h_gauss
from rev_common import window, apply_policy

DELTAS = [1e-4, 1e-5, 1e-6, 1e-8, 1e-10]
SIGMA, C = 2.0, 1.0
HI = 400.0
t0 = time.time(); log = lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True)


def eps_poisson(q, T, delta, disc=1e-4):
    from dp_accounting import dp_event, pld
    acc = pld.PLDAccountant(value_discretization_interval=disc)
    acc.compose(dp_event.SelfComposedDpEvent(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(SIGMA / C)), T))
    return acc.get_epsilon(delta)


def eps_thm2(pn, delta):
    n = np.arange(len(pn)); mu = np.sqrt(n) * C / SIGMA
    e = eps_from_delta_curve(lambda x: float(np.sum(pn * h_gauss(mu, x))), delta, 0.0, HI)
    assert e < HI - 1e-6, "search limit reached"
    return e


def eps_fm(pn, T, delta, grid=20001):
    e = eps_sum_exact(pn, C, SIGMA, T, delta, grid=grid)
    assert e < 60.0 - 1e-6, "search limit of eps_sum_exact reached"
    return e


laws = {}
P, pi, emit = two_state_chain(0.1, 0.9)
for T in (200, 1000):
    laws[f"chain0.9_T{T}"] = (count_distribution(P, pi, emit, T), 0.1, T)
A, _, _ = window(10, 3)
B = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0)); N, Tw = B.shape
c200 = np.array([B[i, s0:s0 + 200].sum() for i in range(N) for s0 in range(0, Tw - 200 + 1, 10)])
laws["trace_T200"] = (np.bincount(c200, minlength=201) / len(c200), float(c200.mean()) / 200, 200)   # q as in Table V (step21b)
A5, _, _ = window(5, 4)
B5 = apply_policy(A5, "fixed", 0.6, 0, np.random.default_rng(0)); N5, Tw5 = B5.shape
c1000 = np.array([B5[i, s0:s0 + 1000].sum() for i in range(N5) for s0 in range(0, Tw5 - 1000 + 1, 5)])
laws["trace_T1000"] = (np.bincount(c1000, minlength=1001) / len(c1000), float(B5.mean()), 1000)   # q as in rev_t1000_bracket.py
for k, (pn, q, T) in laws.items():
    n = np.arange(len(pn)); m = float(np.sum(n * pn)); v = float(np.sum(n * n * pn)) - m * m
    log(f"{k}: q {q:.4f} (count mean / T {m / T:.4f}), D {v / m:.2f}")

rows = []
for k, (pn, q, T) in laws.items():
    pb = binom.pmf(np.arange(T + 1), T, q)
    for d in DELTAS:
        ep = eps_poisson(q, T, d); ep_f = eps_poisson(q, T, d, 1e-5)
        fm = eps_fm(pn, T, d); fm_f = eps_fm(pn, T, d, 80001)
        fb = eps_fm(pb, T, d); fb_f = eps_fm(pb, T, d, 80001)
        up = eps_thm2(pn, d)
        rows.append([k, T, f"{q:.5f}", f"{d:g}", f"{ep:.4f}", f"{fm:.4f}", f"{fb:.4f}", f"{fm / fb:.3f}", f"{up:.4f}", f"{up / ep:.3f}",
                     f"{abs(ep_f - ep):.2e}", f"{max(abs(fm_f - fm), abs(fb_f - fb)):.2e}"])
        log(f"{k} delta={d:g}: Poisson {ep:.3f} | sum statistic {fm:.3f} vs Binomial {fb:.3f} (ratio {fm / fb:.2f}) | "
            f"Thm 2 {up:.3f} ({up / ep:.2f}x Poisson) | checks: PLD {abs(ep_f - ep):.1e}, grid {max(abs(fm_f - fm), abs(fb_f - fb)):.1e}")
with open("rev_round10_small_delta.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["law", "T", "q", "delta", "poisson_perround", "sum_stat_law", "sum_stat_binomial", "sum_stat_ratio", "thm2_upper", "thm2_over_poisson",
                "check_pld_finer", "check_grid_finer"])
    w.writerows(rows)

# sum-statistic ratio of the ten policies of Table V (seed 0, T = 200, stride 10, q = count mean / T as in step21b)
policies = [("thinning 0.05", "thin", 0.05, 0), ("thinning 0.20", "thin", 0.20, 0), ("thinning 0.50", "thin", 0.50, 0), ("take all available", "thin", 1.0, 0),
            ("fixed cohort 0.1 x pool", "fixed", 0.1, 0), ("fixed cohort 0.3 x pool", "fixed", 0.3, 0), ("fixed cohort 0.6 x pool", "fixed", 0.6, 0),
            ("fixed cohort 1.0 x pool", "fixed", 1.0, 0), ("fixed cohort 0.6, timer 5 rounds", "fixed", 0.6, 5), ("fixed cohort 0.6, timer 20 rounds", "fixed", 0.6, 20)]
prow = []
for name, kind, par, b in policies:
    Bp = apply_policy(A, kind, par, b, np.random.default_rng(0)); Np, Twp = Bp.shape
    cnt = np.array([Bp[i, s0:s0 + 200].sum() for i in range(Np) for s0 in range(0, Twp - 200 + 1, 10)])
    pn = np.bincount(cnt, minlength=201) / len(cnt); q = cnt.mean() / 200; pb = binom.pmf(np.arange(201), 200, q)
    r = [name, f"{q:.4f}"]
    for d in DELTAS:
        r.append(f"{eps_fm(pn, 200, d) / eps_fm(pb, 200, d):.3f}")
    prow.append(r); log(f"{name:34s} q={q:.4f} sum-statistic ratio at {DELTAS}: {r[2:]}")
with open("rev_round10_small_delta_policies.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["policy", "q"] + [f"sum_stat_ratio_{d:g}" for d in DELTAS]); w.writerows(prow)
log("done")
