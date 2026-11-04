"""Round 14: two training experiments on the pipeline of rev_train.py (DP-FedAvg with canary clients, Fashion-MNIST).

(1) Implementation noise error.  The server adds Gaussian noise of scale r * sigma * C while the accountant and the
    auditor are told the nominal sigma = 2.  The claim is the correct accountant's value for the canaries' own law at
    the nominal noise (11.869 for the Markov law of Section V, the 10M-sample reference of Table II; for i.i.d.
    participation rev_train.REF_IID = 2.869, the Monte-Carlo reference, against the exact PLD value 2.866: no flag
    count differs), so at r = 1 the claim is true and at r < 1 the realised loss exceeds it.  An accountant that
    reads the logs and the configuration cannot see r; the audit estimates the signal-to-noise ratio from the released
    aggregates and can.  Configurations noise_r1.00, noise_r0.95, noise_r0.90, noise_r0.80 (Markov law, the base_orth
    setting: 784-64-10, 200 real clients, 1,000 canaries, orthonormal directions) and noise_iid_r1.00, noise_iid_r0.90.
(2) Heterogeneous partition.  The 235k-parameter network of the medium setting (784-256-128-10, 500 real clients) with
    a label-skewed Dirichlet(0.1) partition or IID shards, and 100 or 500 canaries (present canaries per round over real
    clients per round 0.1 or 0.5): configurations dir01_m100, dir01_m500, iid_m100, iid_m500 (Markov law, r = 1).
Everything else (local SGD, clipping calibration at 0.7 times the median local-update norm, canaries injected after
clipping, Hadamard directions, the diagnostics, the certified table step14_table_markov.csv / step14_table_iid.csv) is
that of rev_train.py.  Each configuration has its own seed (1401 to 1406 for the noise configurations, 1411 to 1414 for the partitions), ten runs by default.

usage (from results/):  python3 ../src/rev_round14_train.py CONFIG [R]
Output: rev_round14_train_<CONFIG>.csv (one row per run; columns as rev_train.py plus noise_r, partition and the
flag against the correct-accountant claim), appended on resume.  The column true_loss_r is the step-14 table's value at
min(a/r, 0.55), a lower estimate of the realised loss beyond the grid's end (the 400k-sample estimates of
rev_round14_noise_sim.py are 15.02 at r = 0.9 and 19.65 at r = 0.8), so above_true_r = 0 implies no run above the
realised loss.
"""
import sys, csv, time, os
import numpy as np
from rev_train import (init_params, flat, unflat, local_update, accuracy, HadamardCanaries, diagnostics, CsvTable,
                       T, Q, SIGMA, A_TRUE, REF_MARKOV, REF_IID, POIS)
from corraudit import two_state_chain, iid_chain, sample_paths
from certified2 import snr_lower_bound
from fl_sim import load_idx, DATA

#          name:          hidden      n_clients  m    law       partition    noise_r  seed
CONFIGS = {
    "noise_r1.00":     ((64,),      200, 1000, "markov", "iid",       1.00, 1401),
    "noise_r0.95":     ((64,),      200, 1000, "markov", "iid",       0.95, 1402),
    "noise_r0.90":     ((64,),      200, 1000, "markov", "iid",       0.90, 1403),
    "noise_r0.80":     ((64,),      200, 1000, "markov", "iid",       0.80, 1404),
    "noise_iid_r1.00": ((64,),      200, 1000, "iid",    "iid",       1.00, 1405),
    "noise_iid_r0.90": ((64,),      200, 1000, "iid",    "iid",       0.90, 1406),
    "dir01_m100":      ((256, 128), 500, 100,  "markov", "dirichlet", 1.00, 1411),
    "dir01_m500":      ((256, 128), 500, 500,  "markov", "dirichlet", 1.00, 1412),
    "iid_m100":        ((256, 128), 500, 100,  "markov", "iid",       1.00, 1413),
    "iid_m500":        ((256, 128), 500, 500,  "markov", "iid",       1.00, 1414),
}
DIR_ALPHA = 0.1


def partition(ytr, n_clients, kind, rng):
    if kind == "iid":
        perm = rng.permutation(len(ytr)); per = len(ytr) // n_clients
        return [perm[i * per:(i + 1) * per] for i in range(n_clients)]
    shards = [[] for _ in range(n_clients)]
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(ytr == c)); prop = rng.dirichlet(np.full(n_clients, DIR_ALPHA))
        cuts = (np.cumsum(prop) * len(idx)).astype(int)[:-1]
        for k, part in enumerate(np.split(idx, cuts)): shards[k].extend(part.tolist())
    return [np.array(s, dtype=np.int64) for s in shards]


def run_once(cfg, data, C_clip, rng, lr=0.05, batch=30):
    hidden, n_clients, m, law, kind, r, _ = cfg
    Xtr, ytr, Xte, yte = data
    P, pi, emit = iid_chain(Q) if law == "iid" else two_state_chain(Q, 0.9)
    shards = partition(ytr, n_clients, kind, rng)
    Breal = sample_paths(P, pi, emit, T, n_clients, rng); Bcan = sample_paths(P, pi, emit, T, m, rng)
    present = rng.random(m) < 0.5
    p = init_params(rng, hidden); d = flat(p).size
    can = HadamardCanaries(d, m, rng)
    X = np.zeros((m, T), np.float32); denom = n_clients * Q; clipf = []
    for t in range(T):
        agg = np.zeros(d, np.float32); nclip = nsel = 0
        for k in np.flatnonzero(Breal[:, t]):
            if len(shards[k]) == 0: continue
            pk = local_update(p, Xtr[shards[k]], ytr[shards[k]], lr, batch, rng)
            delta = flat(pk) - flat(p); nrm = np.linalg.norm(delta); nsel += 1
            if nrm > C_clip: delta *= C_clip / nrm; nclip += 1
            agg += delta
        clipf.append(nclip / max(nsel, 1))
        active = np.flatnonzero(Bcan[:, t] & present)
        if len(active): agg += C_clip * can.sum_dirs(active)
        agg += (r * SIGMA * C_clip * rng.standard_normal(d)).astype(np.float32)      # the implemented noise: r times the nominal
        X[:, t] = can.project(agg) / C_clip
        p = unflat(flat(p) + agg / denom, p)
    sizes = np.array([len(s) for s in shards])
    return dict(X=X, present=present, B=Bcan, acc=accuracy(p, Xte, yte), clip=float(np.mean(clipf)), d=d,
                shard_min=int(sizes.min()), shard_max=int(sizes.max()))


if __name__ == "__main__":
    name = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 10; cfg = CONFIGS[name]
    hidden, n_clients, m, law, kind, r, seed = cfg
    rng = np.random.default_rng(seed)
    Xtr = load_idx(f"{DATA}/train-images-idx3-ubyte.gz"); ytr = load_idx(f"{DATA}/train-labels-idx1-ubyte.gz")
    Xte = load_idx(f"{DATA}/t10k-images-idx3-ubyte.gz"); yte = load_idx(f"{DATA}/t10k-labels-idx1-ubyte.gz")
    mu = Xtr.mean(axis=0); data = (Xtr - mu, ytr, Xte - mu, yte)
    per = len(ytr) // n_clients
    p0 = init_params(rng, hidden); norms = []
    for k in range(20):
        idx = rng.permutation(len(ytr))[:per]; pk = local_update(p0, data[0][idx], ytr[idx], 0.05, 30, rng)
        norms.append(np.linalg.norm(flat(pk) - flat(p0)))
    C_clip = float(np.median(norms)) * 0.7
    tab = CsvTable("step14_table_iid.csv" if law == "iid" else "step14_table_markov.csv")
    ref = REF_IID if law == "iid" else REF_MARKOV                 # the correct accountant's claim at the nominal noise
    out_path = f"rev_round14_train_{name}.csv"; new = not os.path.exists(out_path)
    done = 0 if new else len(list(csv.DictReader(open(out_path))))
    f = open(out_path, "a", newline=""); w = csv.writer(f)
    if new:
        w.writerow(["config", "rep", "d", "n_clients", "m_canary", "law", "partition", "noise_r", "C_clip", "acc", "clip_frac", "shard_min", "shard_max",
                    "a_lo", "covered_nominal", "a_hat", "s_hat", "lag1", "xcorr_mean", "xcorr_sd", "xcorr_maxabs", "ks", "res_part", "res_nonpart",
                    "erasure", "cert", "uncert", "claim", "true_loss_r", "poisson", "above_true_r", "flag_claim", "flag_poisson"])
    t0 = time.time()
    print(f"{name}: hidden={hidden} clients={n_clients} canaries={m} law={law} partition={kind} noise_r={r} C_clip={C_clip:.3f}; resuming at rep {done}", flush=True)
    for rep in range(R):
        res = run_once(cfg, data, C_clip, rng)
        if rep < done: continue
        X, present, B = res["X"], res["present"], res["B"]
        a_lo = snr_lower_bound(X, present, B, 0.045); a_v2 = snr_lower_bound(X, present, B, 0.05)
        dg = diagnostics(X, present, B)
        S = X.sum(axis=1) / np.sqrt(T); counts = B.sum(axis=1)
        er = float(S[present].mean() / (counts[present].mean() / np.sqrt(T)))
        cert = tab.cert(a_lo); unc = tab.plain_interp(a_v2)
        true_r = tab.plain_interp(min(A_TRUE / r, tab.a[-1]))        # the realised loss at signal-to-noise a/r (capped at the grid)
        w.writerow([name, rep, res["d"], n_clients, m, law, kind, r, f"{C_clip:.4f}", f"{res['acc']:.4f}", f"{res['clip']:.3f}", res["shard_min"], res["shard_max"],
                    f"{a_lo:.4f}", int(a_lo <= A_TRUE), f"{dg['a_hat']:.4f}", f"{dg['s_hat']:.4f}", f"{dg['lag1']:.4f}",
                    f"{dg['xcorr_mean']:.5f}", f"{dg['xcorr_sd']:.5f}", f"{dg['xcorr_maxabs']:.4f}", f"{dg['ks']:.5f}",
                    f"{dg['res_part']:.4f}", f"{dg['res_nonpart']:.4f}", f"{er:.3f}", f"{cert:.3f}", f"{unc:.3f}",
                    f"{ref:.3f}", f"{true_r:.3f}", f"{POIS:.3f}", int(cert > true_r), int(cert > ref), int(cert > POIS)]); f.flush()
        print(f"  rep {rep}: acc={res['acc']:.3f} clip={res['clip']:.2f} shards[{res['shard_min']},{res['shard_max']}] a_lo={a_lo:.3f} a_hat={dg['a_hat']:.3f} "
              f"s_hat={dg['s_hat']:.3f} lag1={dg['lag1']:.3f} ks={dg['ks']:.4f} erasure={er:.2f} cert={cert:.2f} (claim {ref:.2f}, loss at r {true_r:.2f}) [{time.time()-t0:.0f}s]", flush=True)
    f.close()
