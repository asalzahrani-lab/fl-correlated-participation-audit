import os
"""Step 17: generalisation of the real-training audit — second dataset (MNIST digits) and non-IID client data
(label-skewed shards: each client holds examples of 2 classes only), MLP 784-64-10, DP-FedAvg, random canary directions,
merged certified auditor (tables from step 14) vs path-aware v2 vs Gaussian fit.
usage: python fl_mlp3.py [R]"""
import numpy as np, csv, time, sys
from corraudit import *
from auditors2 import *
from certified2 import snr_lower_bound
from fl_sim import load_idx
from fl_mlp import init_params, flat, unflat, local_update, accuracy
from fl_mlp2 import CsvTable

DATASETS = {"fmnist": os.environ.get("P3_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "fmnist")), "mnist": os.environ.get("P3_DATA_MNIST", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mnist"))}


def make_shards(ytr, n_clients, kind, rng, classes_per_client=2):
    n = len(ytr)
    if kind == "iid":
        perm = rng.permutation(n); per = n // n_clients
        return [perm[i * per:(i + 1) * per] for i in range(n_clients)]
    # label-skew: sort by label, cut into 2*n_clients shards, give each client 2 shards (McMahan et al. 2017 style)
    order = np.argsort(ytr, kind="stable"); nsh = n_clients * classes_per_client; per = n // nsh
    shards = [order[i * per:(i + 1) * per] for i in range(nsh)]
    sp = rng.permutation(nsh)
    return [np.concatenate([shards[sp[classes_per_client * i + j]] for j in range(classes_per_client)]) for i in range(n_clients)]


def run_once(kind, par, T, q, sigma_noise, C_clip, shards, m_canary, data, rng, lr=0.05):
    Xtr, ytr, Xte, yte = data; n_clients = len(shards)
    P, pi, emit = iid_chain(q) if kind == "iid" else two_state_chain(q, par)
    Breal = sample_paths(P, pi, emit, T, n_clients, rng); Bcan = sample_paths(P, pi, emit, T, m_canary, rng)
    present = rng.random(m_canary) < 0.5
    p = init_params(rng); d = flat(p).size
    U = rng.standard_normal((m_canary, d)).astype(np.float32); U /= np.linalg.norm(U, axis=1, keepdims=True)
    Xobs = np.zeros((m_canary, T), np.float32); denom = n_clients * q; clipf = []
    for t in range(T):
        agg = np.zeros(d, np.float32); nclip = 0; nsel = 0
        for k in np.flatnonzero(Breal[:, t]):
            idx = shards[k]
            pk = local_update(p, Xtr[idx], ytr[idx], lr, 30, rng)
            delta = flat(pk) - flat(p); nrm = np.linalg.norm(delta); nsel += 1
            if nrm > C_clip: delta *= C_clip / nrm; nclip += 1
            agg += delta
        clipf.append(nclip / max(nsel, 1))
        active = np.flatnonzero(Bcan[:, t] & present)
        if len(active): agg += C_clip * U[active].sum(axis=0)
        agg += (sigma_noise * C_clip * rng.standard_normal(d)).astype(np.float32)
        Xobs[:, t] = (U @ agg) / C_clip
        p = unflat(flat(p) + agg / denom, p)
    return dict(X=Xobs, present=present, counts=Bcan.sum(axis=1), B=Bcan, acc=accuracy(p, Xte, yte), clip_frac=float(np.mean(clipf)))


if __name__ == "__main__":
    T, q, sigma_noise = 200, 0.1, 2.0
    n_clients, m_canary, R = 200, 1000, int(sys.argv[1]) if len(sys.argv) > 1 else 8
    DELTA = 1e-4
    rng = np.random.default_rng(1717)
    refs = {}
    for r in csv.DictReader(open("pilot_a_results.csv")):
        if float(r["delta"]) == DELTA and int(r["T"]) == T:
            refs[(r["process"], float(r["param"]))] = (float(r["eps_full_mc"]), float(r["eps_poisson_pld"]))
    tables = {"iid": CsvTable("step14_table_iid.csv"), "markov": CsvTable("step14_table_markov.csv")}
    out = open("fl_mlp3_results.csv", "w", newline=""); w = csv.writer(out)
    w.writerow(["dataset", "partition", "process", "param", "rep", "acc", "clip_frac", "erasure_ratio", "eps_poisson", "eps_true_model1", "a_lo_045", "gauss_sum", "path_aware_v2", "merged_certified"])
    t0 = time.time()
    for ds, part in [("mnist", "iid"), ("mnist", "noniid2"), ("fmnist", "noniid2")]:
        D = DATASETS[ds]
        Xtr = load_idx(f"{D}/train-images-idx3-ubyte.gz"); ytr = load_idx(f"{D}/train-labels-idx1-ubyte.gz")
        Xte = load_idx(f"{D}/t10k-images-idx3-ubyte.gz"); yte = load_idx(f"{D}/t10k-labels-idx1-ubyte.gz")
        mu = Xtr.mean(axis=0); data = (Xtr - mu, ytr, Xte - mu, yte)
        shards = make_shards(ytr, n_clients, "iid" if part == "iid" else "noniid", rng)
        p0 = init_params(rng); norms = []
        for k in range(20):
            idx = shards[k][:300]; pk = local_update(p0, data[0][idx], ytr[idx], 0.05, 30, rng)
            norms.append(np.linalg.norm(flat(pk) - flat(p0)))
        C_clip = float(np.median(norms)) * 0.7
        print(f"== {ds} {part}: C_clip={C_clip:.3f}", flush=True)
        for kind, par in [("iid", 0.0), ("markov", 0.9)]:
            ef, ep = refs[(kind, float(par))]; tab = tables[kind]
            for rep in range(R):
                res = run_once(kind, par, T, q, sigma_noise, C_clip, shards, m_canary, data, rng)
                X, present, counts, B = res["X"], res["present"], res["counts"], res["B"]
                S = scores_sum(X)
                g = gauss_sum_audit(X, present, DELTA, 0.05)
                a_v2 = snr_lower_bound(X, present, B, 0.05); a_c = snr_lower_bound(X, present, B, 0.045)
                pa = tab.plain_interp(a_v2); mc = tab.cert(a_c)
                pred = counts[present].mean() / np.sqrt(T); er = S[present].mean() / pred
                w.writerow([ds, part, kind, par, rep, f"{res['acc']:.4f}", f"{res['clip_frac']:.3f}", f"{er:.3f}", f"{ep:.3f}", f"{ef:.3f}", f"{a_c:.4f}", f"{g:.3f}", f"{pa:.3f}", f"{mc:.3f}"]); out.flush()
                print(f"  {ds} {part} {kind} rep {rep}: acc={res['acc']:.3f} clip={res['clip_frac']:.2f} erasure={er:.2f} | gauss={g:.2f} v2={pa:.2f} merged={mc:.2f} (true {ef:.2f}, Poisson {ep:.2f}) [{time.time()-t0:.0f}s]", flush=True)
    out.close()
