"""Step 14b: real DP-FedAvg training (MLP 784-64-10, Fashion-MNIST, random canary directions) audited with the
merged certified path-aware auditor (KL certification, 0.01 grid, alpha 0.045/0.005) read from the step14 tables,
alongside path-aware v2 (plain interpolation, alpha 0.05) and the Gaussian-fit audit.
usage: python fl_mlp2.py [R]"""
import numpy as np, csv, time, sys
from corraudit import *
from auditors2 import *
from certified2 import snr_lower_bound
from fl_sim import load_idx, DATA
from fl_mlp import init_params, flat, local_update, run_once


class CsvTable:
    def __init__(self, path):
        rows = list(csv.DictReader(open(path)))
        self.a = np.array([float(r["a"]) for r in rows]); self.plain = np.array([float(r["eps_plain"]) for r in rows])
        self.bern = np.array([float(r["eps_cert_bern_0.005"]) for r in rows])

    def cert(self, a):
        if not np.isfinite(a): return 0.0                 # non-finite estimate: certificate 0 (round 12)
        i = np.searchsorted(self.a - 1e-9, a, side="right") - 1
        return 0.0 if i < 0 else float(self.bern[i])

    def plain_interp(self, a):
        return float(np.interp(a, self.a, self.plain, left=0.0))


if __name__ == "__main__":
    T, q, sigma_noise = 200, 0.1, 2.0
    n_clients, m_canary, R = 200, 5000, int(sys.argv[1]) if len(sys.argv) > 1 else 3
    DELTA = 1e-4
    rng = np.random.default_rng(7777)
    Xtr = load_idx(f"{DATA}/train-images-idx3-ubyte.gz"); ytr = load_idx(f"{DATA}/train-labels-idx1-ubyte.gz")
    Xte = load_idx(f"{DATA}/t10k-images-idx3-ubyte.gz"); yte = load_idx(f"{DATA}/t10k-labels-idx1-ubyte.gz")
    mu = Xtr.mean(axis=0); data = (Xtr - mu, ytr, Xte - mu, yte)
    p0 = init_params(rng); norms = []
    for k in range(20):
        idx = rng.permutation(len(ytr))[:300]; pk = local_update(p0, data[0][idx], ytr[idx], 0.05, 30, rng)
        norms.append(np.linalg.norm(flat(pk) - flat(p0)))
    C_clip = float(np.median(norms)) * 0.7
    print(f"C_clip={C_clip:.3f}", flush=True)
    refs = {}
    for r in csv.DictReader(open("pilot_a_results.csv")):
        if float(r["delta"]) == DELTA and int(r["T"]) == T:
            refs[(r["process"], float(r["param"]))] = (float(r["eps_full_mc"]), float(r["eps_poisson_pld"]))
    tables = {"iid": CsvTable("step14_table_iid.csv"), "markov": CsvTable("step14_table_markov.csv")}
    out = open("fl_mlp_5k_results.csv", "w", newline=""); w = csv.writer(out)
    w.writerow(["process", "param", "rep", "acc", "clip_frac", "erasure_ratio", "eps_poisson", "eps_true_model1", "a_lo_045", "gauss_sum", "path_aware_v2", "merged_certified"])
    t0 = time.time()
    for kind, par in [("iid", 0.0), ("markov", 0.9)]:
        ef, ep = refs[(kind, float(par))]; tab = tables[kind]
        print(f"{kind} {par}: poisson={ep:.2f} model1-true={ef:.2f}", flush=True)
        for rep in range(R):
            res = run_once(kind, par, T, q, sigma_noise, C_clip, n_clients, m_canary, data, rng)
            X, present, counts, B = res["X"], res["present"], res["counts"], res["B"]
            S = scores_sum(X)
            g = gauss_sum_audit(X, present, DELTA, 0.05)
            a_v2 = snr_lower_bound(X, present, B, 0.05); a_c = snr_lower_bound(X, present, B, 0.045)
            pa = tab.plain_interp(a_v2); mc = tab.cert(a_c)
            pred = counts[present].mean() / np.sqrt(T); er = S[present].mean() / pred
            w.writerow([kind, par, rep, f"{res['acc']:.4f}", f"{res['clip_frac']:.3f}", f"{er:.3f}", f"{ep:.3f}", f"{ef:.3f}", f"{a_c:.4f}", f"{g:.3f}", f"{pa:.3f}", f"{mc:.3f}"]); out.flush()
            print(f"  rep {rep}: acc={res['acc']:.3f} erasure={er:.2f} a_lo={a_c:.3f} | gauss={g:.2f} v2={pa:.2f} merged_cert={mc:.2f} (true {ef:.2f}, Poisson {ep:.2f}) [{time.time()-t0:.0f}s]", flush=True)
    out.close()
