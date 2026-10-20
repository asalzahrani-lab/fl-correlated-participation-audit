"""Revision (reviewer point on the timer): what the timer costs in utility.

DP-FedAvg on Fashion-MNIST (IID shards) with the 572 devices of the analysis window as clients and their participation
taken from the trace under three policies: fixed cohort 0.6 without a timer (the running example, mean cohort 27.0),
with a 5-round timer (mean cohort 9.8) and with a 20-round timer (4.5), plus fixed cohort 0.1 without a timer (K = 5,
mean cohort 3.8), which has about the timer-20 cohort size without the timer.  Rounds 0..199 of the 432-round window,
seeds 0..4 of the policy and of the training.  Clipping and local training as in rev_train.py (784-64-10, lr 0.05,
batch 30, one epoch, C = 0.7 x median initial update norm); server: theta += (clipped sum + N(0, sigma^2 C^2 I)) / E[cohort],
sigma = 2.  Output rev_timer_utility.csv (test accuracy per policy and seed, and the mean cohort)."""
import csv, sys
import numpy as np
from rev_common import log, window, apply_policy
from rev_train import init_params, local_update, flat, unflat, accuracy
from fl_sim import load_idx, DATA

T, SIGMA = 200, 2.0
POL = [("fixed cohort 0.6", "fixed", 0.6, 0), ("fixed cohort 0.6, timer 5", "fixed", 0.6, 5),
       ("fixed cohort 0.6, timer 20", "fixed", 0.6, 20), ("fixed cohort 0.1", "fixed", 0.1, 0)]
if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    A, _, _ = window(10, 3); N = A.shape[0]
    Xtr = load_idx(f"{DATA}/train-images-idx3-ubyte.gz"); ytr = load_idx(f"{DATA}/train-labels-idx1-ubyte.gz")
    Xte = load_idx(f"{DATA}/t10k-images-idx3-ubyte.gz"); yte = load_idx(f"{DATA}/t10k-labels-idx1-ubyte.gz")
    mu = Xtr.mean(axis=0); Xtr = Xtr - mu; Xte = Xte - mu
    rows = []
    for seed in range(R):
        rng0 = np.random.default_rng(5000 + seed)
        perm = rng0.permutation(len(ytr)); per = len(ytr) // N; shards = [perm[i * per:(i + 1) * per] for i in range(N)]
        p0 = init_params(rng0, (64,)); norms = []
        for k in range(20):
            idx = rng0.permutation(len(ytr))[:per]; pk = local_update(p0, Xtr[idx], ytr[idx], 0.05, 30, rng0)
            norms.append(np.linalg.norm(flat(pk) - flat(p0)))
        C = 0.7 * float(np.median(norms))
        for name, kind, par, b in POL:
            B = apply_policy(A, kind, par, b, np.random.default_rng(seed))[:, :T]
            denom = max(B.sum(axis=0).mean(), 1e-9); rng = np.random.default_rng(6000 + seed)
            p = [x.copy() for x in p0]; d = flat(p).size
            for t in range(T):
                agg = np.zeros(d, np.float32)
                for k in np.flatnonzero(B[:, t]):
                    delta = flat(local_update(p, Xtr[shards[k]], ytr[shards[k]], 0.05, 30, rng)) - flat(p)
                    nrm = np.linalg.norm(delta)
                    if nrm > C: delta *= C / nrm
                    agg += delta
                agg += (SIGMA * C * rng.standard_normal(d)).astype(np.float32)
                p = unflat(flat(p) + agg / denom, p)
            acc = accuracy(p, Xte, yte)
            rows.append([name, seed, f"{denom:.2f}", f"{B.mean():.4f}", f"{C:.4f}", f"{acc:.4f}"])
            log(f"seed {seed} {name}: mean cohort {denom:.1f}, q {B.mean():.4f}, accuracy {acc:.3f}")
    with open("rev_timer_utility.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["policy", "seed", "mean_cohort", "q", "C_clip", "test_acc"]); w.writerows(rows)
