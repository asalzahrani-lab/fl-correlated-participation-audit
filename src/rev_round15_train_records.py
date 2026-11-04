"""Round 15: the noise-error training runs of rev_round14_train.py regenerated with a full per-run record.

rev_round14_train.py stored s_hat, lag-1 (over every absent canary), the normalised KS distance and the erasure ratio
of each run but not the absent canaries' mean, so the frozen diagnostic rule of rev_round14_residual_sim.py (which
also uses that mean) could not be applied to the training runs from the stored files.  This script re-runs the six
noise configurations with their documented seeds (1401 to 1406; the same random stream in the same order: the
clipping calibration first, then ten runs) and writes one row per run with every column of rev_round14_train.py plus
the absent mean (mean_a, in declared-noise units), lag1_200 (the rule's definition: the first 200 absent canaries),
mean_a_norm = mean_a / s_hat and the pooled sufficient statistics of rev_round15_records.py.  The certificate, a_lo,
s_hat, accuracy and the stored diagnostics must agree with rev_round14_train_<CONFIG>.csv (rev_round15_screening.py
checks every row to the stored precision).

usage (from results/):  python3 ../src/rev_round15_train_records.py CONFIG       # 2 to 3 minutes per configuration
Output rev_round15_train_records_<CONFIG>.csv.
"""
import sys, csv, time, os
import numpy as np
from scipy.stats import kstest
from rev_train import init_params, flat, local_update, diagnostics, CsvTable, T, SIGMA, A_TRUE, REF_MARKOV, REF_IID, POIS
from rev_round14_train import CONFIGS, run_once
from certified2 import snr_lower_bound
from rev_round14_residual_sim import snr_lower_bound_centred
from fl_sim import load_idx, DATA

COLS = ["config", "rep", "d", "n_clients", "m_canary", "law", "partition", "noise_r", "C_clip", "acc", "clip_frac", "shard_min", "shard_max",
        "a_lo", "covered_nominal", "a_hat", "s_hat", "lag1", "xcorr_mean", "xcorr_sd", "xcorr_maxabs", "ks", "res_part", "res_nonpart",
        "erasure", "cert", "uncert", "claim", "true_loss_r", "poisson", "above_true_r", "flag_claim", "flag_poisson",
        "mean_a", "mean_a_norm", "lag1_200", "a_lo_c", "cert_c", "n0", "n_abs", "sum_a", "sumsq_a", "lagsum_a", "n_obs", "sum_obs", "n_pres", "sum_counts_pres", "sum_S_pres"]


def extra(X, present, B):
    """The absent mean and the sufficient statistics, in declared-noise units (X is in units of C; divide by sigma)."""
    Xa = (X[~present] / SIGMA).astype(np.float64); xa = Xa.ravel(); n0 = xa.size
    s_hat = float(Xa.std()); mean_a = float(Xa.mean())
    lag1_200 = float(np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in Xa[:200]]))
    Xn = (X / SIGMA).astype(np.float32)
    S = Xn.astype(np.float64).sum(axis=1) / np.sqrt(T); counts = B.sum(axis=1)
    Bp = B[present].astype(np.float64); Xp = Xn[present].astype(np.float64)
    a_lo_c = snr_lower_bound_centred(Xn, present, B, 0.045)
    return dict(mean_a=mean_a, mean_a_norm=mean_a / s_hat, lag1_200=lag1_200, a_lo_c=a_lo_c, n0=n0, n_abs=int(Xa.shape[0]), sum_a=float(xa.sum()),
                sumsq_a=float((xa * xa).sum()), lagsum_a=float((Xa[:, :-1] * Xa[:, 1:]).sum()), n_obs=int(Bp.sum()), sum_obs=float((Bp * Xp).sum()),
                n_pres=int(present.sum()), sum_counts_pres=int(counts[present].sum()), sum_S_pres=float(S[present].sum()))


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
    ref = REF_IID if law == "iid" else REF_MARKOV
    out_path = f"rev_round15_train_records_{name}.csv"; new = not os.path.exists(out_path)
    done = 0 if new else len(list(csv.DictReader(open(out_path))))
    f = open(out_path, "a", newline=""); w = csv.writer(f)
    if new: w.writerow(COLS)
    t0 = time.time()
    print(f"{name}: hidden={hidden} clients={n_clients} canaries={m} law={law} partition={kind} noise_r={r} C_clip={C_clip:.3f}; resuming at rep {done}", flush=True)
    for rep in range(R):
        res = run_once(cfg, data, C_clip, rng)
        if rep < done: continue
        X, present, B = res["X"], res["present"], res["B"]
        a_lo = snr_lower_bound(X, present, B, 0.045); a_v2 = snr_lower_bound(X, present, B, 0.05)
        dg = diagnostics(X, present, B); ex = extra(X, present, B)
        S = X.sum(axis=1) / np.sqrt(T); counts = B.sum(axis=1)
        er = float(S[present].mean() / (counts[present].mean() / np.sqrt(T)))
        cert = tab.cert(a_lo); unc = tab.plain_interp(a_v2); cert_c = tab.cert(ex["a_lo_c"])
        true_r = tab.plain_interp(min(A_TRUE / r, tab.a[-1]))
        w.writerow([name, rep, res["d"], n_clients, m, law, kind, r, f"{C_clip:.4f}", f"{res['acc']:.4f}", f"{res['clip']:.3f}", res["shard_min"], res["shard_max"],
                    f"{a_lo:.4f}", int(a_lo <= A_TRUE), f"{dg['a_hat']:.4f}", f"{dg['s_hat']:.4f}", f"{dg['lag1']:.4f}",
                    f"{dg['xcorr_mean']:.5f}", f"{dg['xcorr_sd']:.5f}", f"{dg['xcorr_maxabs']:.4f}", f"{dg['ks']:.5f}",
                    f"{dg['res_part']:.4f}", f"{dg['res_nonpart']:.4f}", f"{er:.3f}", f"{cert:.3f}", f"{unc:.3f}",
                    f"{ref:.3f}", f"{true_r:.3f}", f"{POIS:.3f}", int(cert > true_r), int(cert > ref), int(cert > POIS),
                    repr(ex["mean_a"]), repr(ex["mean_a_norm"]), repr(ex["lag1_200"]), repr(ex["a_lo_c"]), repr(cert_c), ex["n0"], ex["n_abs"], repr(ex["sum_a"]), repr(ex["sumsq_a"]),
                    repr(ex["lagsum_a"]), ex["n_obs"], repr(ex["sum_obs"]), ex["n_pres"], ex["sum_counts_pres"], repr(ex["sum_S_pres"])]); f.flush()
        print(f"  rep {rep}: acc={res['acc']:.3f} clip={res['clip']:.2f} a_lo={a_lo:.4f} s_hat={dg['s_hat']:.4f} mean_a={ex['mean_a']:.5f} lag1={dg['lag1']:.4f}/{ex['lag1_200']:.4f} "
              f"ks={dg['ks']:.5f} erasure={er:.3f} cert={cert:.3f} cert_c={cert_c:.3f} [{time.time()-t0:.0f}s]", flush=True)
    f.close()
