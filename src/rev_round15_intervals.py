"""Round 15: independent two-sided 99% intervals for the realised loss eps(delta; a / r) of the noise-error cells.

rev_round14_noise_sim.py compared each certificate with a point value of the realised loss (the merged table's plain
Monte-Carlo value at a / r, or a 400k-sample estimate beyond the grid).  Here every (law, r) point gets its own
estimate from 400,000 fresh samples per world (seed 14900 + index, never used elsewhere) with the two-sided
Chernoff-Hoeffding interval of rev_common.eps_interval (level 0.01, as the references of Table II), so that
rev_round15_screening.py can class each certificate as below the interval (covered), inside it (undecided) or above
its upper end (an exceedance against the interval).  The i.i.d. law's loss is the exact PLD value (no interval).

    cd results && P3_TRACE=../trace python3 ../src/rev_round15_intervals.py        # about 20 minutes
Output rev_round15_intervals.csv.
"""
import csv, time
import numpy as np
from corraudit import two_state_chain, eps_poisson_pld
from certified import ChainLaw, EmpiricalPathLaw
from rev_common import window, apply_policy, eps_interval

T, A_TRUE, DELTA, Q = 200, 0.5, 1e-4, 0.1
RS = [1.10, 1.00, 0.98, 0.95, 0.90, 0.80]
N = 400_000
t0 = time.time()
log = lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True)


def samples(L, a, n, rng, chunk=20_000):
    lp = np.empty(n); lq = np.empty(n)
    for s in range(0, n, chunk):
        k = min(chunk, n - s); B = L.sample(k, rng).astype(np.float64)
        lp[s:s + k] = L.loglr(a * B + rng.standard_normal((k, T)), a); lq[s:s + k] = L.loglr(rng.standard_normal((k, T)), a)
    return lp, lq


if __name__ == "__main__":
    laws = {}
    P, pi, emit = two_state_chain(Q, 0.9); laws["markov"] = ChainLaw(P, pi, emit, T)
    A, _, _ = window(10, 3); Bt = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0))
    starts = np.arange(0, Bt.shape[1] - T + 1, 10); W = np.concatenate([Bt[:, s:s + T] for s in starts], axis=0).astype(np.int8)
    laws["trace"] = EmpiricalPathLaw(W)
    rows = []; idx = 0
    for r in RS:
        a = A_TRUE / r; e = eps_poisson_pld(Q, T, 1.0, 1.0 / a, DELTA)
        rows.append(["iid", f"{r:.2f}", f"{a:.4f}", "exact", f"{e:.4f}", f"{e:.4f}", f"{e:.4f}"]); log(f"iid r={r}: exact PLD {e:.4f}")
    for name in ("markov", "trace"):
        for r in RS:
            a = A_TRUE / r; rng = np.random.default_rng(14900 + idx); idx += 1
            lp, lq = samples(laws[name], a, N, rng); e, lo, hi = eps_interval(lp, lq, DELTA, alpha=0.01)
            rows.append([name, f"{r:.2f}", f"{a:.4f}", N, f"{e:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
            log(f"{name} r={r}: a/r={a:.4f}; eps {e:.4f} [{lo:.4f}, {hi:.4f}] from {N} samples (seed {14900 + idx - 1})")
    with open("rev_round15_intervals.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["law", "noise_r", "a_over_r", "samples", "eps", "lo99", "hi99"]); w.writerows(rows)
    log("done")
