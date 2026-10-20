"""Revision B2/B4: single-signal (a = 0.5) Monte-Carlo references with two-sided confidence intervals.

Laws (all T = 200 unless stated):
  markov200   two-state chain, q = 0.1, lambda = 0.9 (the paper's main simulated law)
  trace200    running example: 3-day 10-min window, fixed cohort 0.6, seed 0, stride 10 (the paper's trace law)
  timer5      same window, fixed cohort 0.6 with a 5-round timer
  timer20     same window, fixed cohort 0.6 with a 20-round timer
  droptail    trace200 with the windows of the devices that ever reach N >= 60 removed (tail-support sensitivity)

usage:
  python rev_refs.py sample LAW PART N        -> rev_parts/LAW_partK.npz  (lp, lq float64 at a = 0.5)
  python rev_refs.py combine LAW              -> rev_refs_LAW.csv (plain eps and 99% CI at 1e-4, 1e-5, Poisson)
  python rev_refs.py stored                   -> CIs from the stored step20 parts (T = 1000 laws, trace fine)
Restart-safe: existing parts are skipped."""
import os, sys, glob, csv
import numpy as np
from rev_common import log, window, apply_policy, window_paths, eps_interval
from certified import EmpiricalPathLaw, ChainLaw
from corraudit import two_state_chain, eps_poisson_pld

A_SIG, T = 0.5, 200
DELTAS = [1e-4, 1e-5]
SEEDS = {"markov200": 7101, "trace200": 7102, "timer5": 7103, "timer20": 7104, "droptail": 7105}
os.makedirs("rev_parts", exist_ok=True)


def build(law_name):
    if law_name == "markov200":
        P, pi, emit = two_state_chain(0.1, 0.9)
        return ChainLaw(P, pi, emit, T), 0.1
    A, _, _ = window(10, 3)
    b = {"trace200": 0, "droptail": 0, "timer5": 5, "timer20": 20}[law_name]
    B = apply_policy(A, "fixed", 0.6, b, np.random.default_rng(0))
    if law_name == "droptail":
        N, Tw = B.shape
        keep = [i for i in range(N) if max(B[i, s0:s0 + T].sum() for s0 in range(0, Tw - T + 1, 10)) < 60]
        log(f"droptail: removing {N - len(keep)} devices that reach N >= 60")
        B = B[keep]
    paths = window_paths(B, T, 10)
    q = paths.mean(); c = paths.sum(axis=1)
    log(f"{law_name}: {len(paths)} paths, q = {q:.4f}, D = {c.var()/c.mean():.2f}, max N = {c.max()}")
    return EmpiricalPathLaw(paths), q


def sample(law_name, part, n):
    out = f"rev_parts/{law_name}_part{part}.npz"
    if os.path.exists(out):
        log(f"{out} exists"); return
    law, q = build(law_name)
    rng = np.random.default_rng(SEEDS[law_name] * 1000 + part)
    chunk = 25_000 if law_name == "markov200" else 4000
    lp = np.empty(n); lq = np.empty(n)
    for s in range(0, n, chunk):
        k = min(chunk, n - s)
        B = law.sample(k, rng).astype(np.float32)
        Z = rng.standard_normal((k, T)).astype(np.float32)
        lp[s:s + k] = law.loglr(A_SIG * B + Z, A_SIG)
        lq[s:s + k] = law.loglr(rng.standard_normal((k, T)).astype(np.float32), A_SIG)
    np.savez(out + ".tmp.npz", lp=lp, lq=lq, q=q)
    os.replace(out + ".tmp.npz", out)
    log(f"saved {out}")


def report(name, lp, lq, q, T_=T, extra=None, pois=None):
    rows = []
    for d in DELTAS:
        e, lo, hi = eps_interval(lp, lq, d, alpha=0.01)
        ep = pois[d] if pois else eps_poisson_pld(q, T_, 1.0, 1.0 / A_SIG, d)
        rows.append([name, T_, d, len(lp), f"{q:.4f}", f"{e:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{ep:.3f}",
                     f"{e/ep:.2f}", f"{lo/ep:.2f}", f"{hi/ep:.2f}"] + ([extra] if extra else []))
        log(f"{name} T={T_} delta={d:g}: eps {e:.3f}  99% CI [{lo:.3f}, {hi:.3f}]  Poisson {ep:.3f}  ratio {e/ep:.2f} [{lo/ep:.2f}, {hi/ep:.2f}]  (n={len(lp)})")
    return rows


HEADER = ["law", "T", "delta", "n_mc", "q", "eps_plain", "ci99_lo", "ci99_hi", "eps_poisson", "ratio", "ratio_lo", "ratio_hi"]

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sample":
        sample(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "combine":
        law_name = sys.argv[2]
        files = sorted(glob.glob(f"rev_parts/{law_name}_part*.npz"))
        lp = np.concatenate([np.load(f)["lp"] for f in files]); lq = np.concatenate([np.load(f)["lq"] for f in files])
        q = float(np.load(files[0])["q"])
        rows = report(law_name, lp, lq, q)
        with open(f"rev_refs_{law_name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(HEADER); w.writerows(rows)
    elif mode == "stored":
        rows = []
        for which, suf, T_ in [("markov", "_T1000", 1000), ("trace", "_T1000", 1000), ("trace", "_fine", 200)]:
            files = sorted(glob.glob(f"step20_parts/{which}{suf}_part*.npz"))
            if not files:
                continue
            ag = np.load(files[0])["a_grid"]; gi = int(np.argmin(np.abs(ag - 0.5)))
            lp = np.concatenate([np.load(f)["lps"][gi] for f in files]).astype(np.float64)
            lq = np.concatenate([np.load(f)["lqs"][gi] for f in files]).astype(np.float64)
            acc = list(csv.DictReader(open(f"step16_accounting_{which}{suf}.csv")))
            pois = {float(r["delta"]): float(r["poisson_perround_eps"]) for r in acc}
            q = 0.1 if which == "markov" else float("nan")
            rows += report(f"{which}{suf}", lp, lq, q, T_, pois=pois)
        with open("rev_refs_stored.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(HEADER); w.writerows(rows)
