"""Round 15: the CIFAR-10 runs of rev_round14_cnn.py regenerated with a full per-run record (the absent canaries'
mean, lag1_200, the centred estimator and the pooled sufficient statistics, as rev_round15_train_records.py adds them
to the Fashion-MNIST runs).  Same seeds (1431 to 1433, per-run streams seed + 1000 (rep + 1)), same thread count
(P3_THREADS, default 2).  PyTorch's CPU reductions are not bit-reproducible across runs: a_lo, s_hat, lag-1, the
certificate and every flag agree with rev_round14_cnn_<CONFIG>.csv exactly, while accuracy (within 0.003), erasure,
KS, the uncertified value and the 200-canary correlation sample (a few units of the last printed digit) differ
between runs (rev_round15_screening.py checks both).

usage (from results/):  P3_DATA_CIFAR=../data/cifar10 python3 ../src/rev_round15_cnn_records.py CONFIG [R]
Output rev_round15_cnn_records_<CONFIG>.csv.  15 to 25 minutes per run.
"""
import sys, csv, time, os
import numpy as np
import torch
from rev_train import diagnostics, CsvTable, T, SIGMA, A_TRUE, REF_MARKOV, REF_IID, POIS
from rev_round14_cnn import CONFIGS, run_once, Net, get_flat, local_update, N_CLIENTS, M, DATA
from certified2 import snr_lower_bound
from rev_round15_train_records import extra

COLS = ["config", "rep", "d", "n_clients", "m_canary", "law", "partition", "noise_r", "C_clip", "acc", "clip_frac",
        "a_lo", "covered_nominal", "a_hat", "s_hat", "lag1", "xcorr_mean", "xcorr_sd", "xcorr_maxabs", "ks", "res_part", "res_nonpart",
        "erasure", "cert", "uncert", "claim", "true_loss_r", "poisson", "above_true_r", "flag_claim", "flag_poisson",
        "mean_a", "mean_a_norm", "lag1_200", "a_lo_c", "cert_c", "n0", "n_abs", "sum_a", "sumsq_a", "lagsum_a", "n_obs", "sum_obs", "n_pres", "sum_counts_pres", "sum_S_pres"]

if __name__ == "__main__":
    name = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 10; law, r, seed = CONFIGS[name]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    z = np.load(os.path.join(DATA, "cifar10_jpeg_mirror.npz"))
    Xtr = z["X_train"].astype(np.float32) / 255.0; Xte = z["X_test"].astype(np.float32) / 255.0
    mu = Xtr.mean(axis=(0, 1, 2), keepdims=True); Xtr -= mu; Xte -= mu
    Xtr = torch.from_numpy(np.ascontiguousarray(Xtr.transpose(0, 3, 1, 2))); Xte = torch.from_numpy(np.ascontiguousarray(Xte.transpose(0, 3, 1, 2)))
    ytr = torch.from_numpy(z["y_train"]); yte = torch.from_numpy(z["y_test"]); data = (Xtr, ytr, Xte, yte)
    net = Net(); torch.manual_seed(seed); net0 = Net(); theta0 = get_flat(net0).clone(); per = len(ytr) // N_CLIENTS; norms = []
    for k in range(20):
        idx = torch.as_tensor(rng.permutation(len(ytr))[:per]); norms.append(float(local_update(net, theta0, Xtr[idx], ytr[idx], rng).norm()))
    C_clip = float(np.median(norms)) * 0.7
    tab = CsvTable("step14_table_iid.csv" if law == "iid" else "step14_table_markov.csv"); ref = REF_IID if law == "iid" else REF_MARKOV
    out_path = f"rev_round15_cnn_records_{name}.csv"; new = not os.path.exists(out_path)
    done = 0 if new else len(list(csv.DictReader(open(out_path))))
    f = open(out_path, "a", newline=""); w = csv.writer(f)
    if new: w.writerow(COLS)
    t0 = time.time()
    print(f"{name}: CNN d={theta0.numel()} clients={N_CLIENTS} canaries={M} law={law} noise_r={r} C_clip={C_clip:.3f}; threads {torch.get_num_threads()}; resuming at rep {done}", flush=True)
    for rep in range(R):
        rng = np.random.default_rng(seed + 1000 * (rep + 1))
        if rep < done: continue
        res = run_once(law, r, data, C_clip, rng, net)
        X, present, B = res["X"], res["present"], res["B"]
        a_lo = snr_lower_bound(X, present, B, 0.045); a_v2 = snr_lower_bound(X, present, B, 0.05)
        dg = diagnostics(X, present, B); ex = extra(X, present, B)
        S = X.sum(axis=1) / np.sqrt(T); counts = B.sum(axis=1)
        er = float(S[present].mean() / (counts[present].mean() / np.sqrt(T)))
        cert = tab.cert(a_lo); unc = tab.plain_interp(a_v2); cert_c = tab.cert(ex["a_lo_c"]); true_r = tab.plain_interp(min(A_TRUE / r, tab.a[-1]))
        w.writerow([name, rep, res["d"], N_CLIENTS, M, law, "iid", r, f"{C_clip:.4f}", f"{res['acc']:.4f}", f"{res['clip']:.3f}",
                    f"{a_lo:.4f}", int(a_lo <= A_TRUE), f"{dg['a_hat']:.4f}", f"{dg['s_hat']:.4f}", f"{dg['lag1']:.4f}",
                    f"{dg['xcorr_mean']:.5f}", f"{dg['xcorr_sd']:.5f}", f"{dg['xcorr_maxabs']:.4f}", f"{dg['ks']:.5f}",
                    f"{dg['res_part']:.4f}", f"{dg['res_nonpart']:.4f}", f"{er:.3f}", f"{cert:.3f}", f"{unc:.3f}",
                    f"{ref:.3f}", f"{true_r:.3f}", f"{POIS:.3f}", int(cert > true_r), int(cert > ref), int(cert > POIS),
                    repr(ex["mean_a"]), repr(ex["mean_a_norm"]), repr(ex["lag1_200"]), repr(ex["a_lo_c"]), repr(cert_c), ex["n0"], ex["n_abs"], repr(ex["sum_a"]), repr(ex["sumsq_a"]),
                    repr(ex["lagsum_a"]), ex["n_obs"], repr(ex["sum_obs"]), ex["n_pres"], ex["sum_counts_pres"], repr(ex["sum_S_pres"])]); f.flush()
        print(f"  rep {rep}: acc={res['acc']:.3f} clip={res['clip']:.2f} a_lo={a_lo:.4f} s_hat={dg['s_hat']:.4f} mean_a={ex['mean_a']:.5f} lag1={dg['lag1']:.4f}/{ex['lag1_200']:.4f} "
              f"ks={dg['ks']:.5f} erasure={er:.3f} cert={cert:.3f} cert_c={cert_c:.3f} [{time.time()-t0:.0f}s]", flush=True)
    f.close()
