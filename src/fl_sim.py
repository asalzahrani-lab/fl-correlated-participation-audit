import os
"""Step 6-7: DP-FedAvg on Fashion-MNIST (PCA-50 logistic regression) with canary clients following a
participation process (i.i.d., Markov lambda=0.9, cyclic period 10).  Audits the per-round releases with the
four auditors and reports them against Poisson accounting and the count-averaged (Jensen) upper bound."""
import numpy as np, gzip, csv, time, sys, os
from scipy.stats import norm
from corraudit import *
from auditors2 import *
from theory import eps_jensen_full, h_gauss

DATA = os.environ.get("P3_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "fmnist"))


def load_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[:4], "big")
    if magic == 2051:
        n, r, c = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(3)]
        return np.frombuffer(data[16:], dtype=np.uint8).reshape(n, r * c).astype(np.float32) / 255.0
    n = int.from_bytes(data[4:8], "big")
    return np.frombuffer(data[8:], dtype=np.uint8).astype(np.int64)


def prepare(dim=50, seed=0, raw=False, null_from=150):
    """raw=False: PCA-dim standardized features, canary directions random (Vnull=None).
    raw=True: centered raw pixels (784), canary directions drawn from the bottom PCA components (index >= null_from)."""
    Xtr = load_idx(f"{DATA}/train-images-idx3-ubyte.gz"); ytr = load_idx(f"{DATA}/train-labels-idx1-ubyte.gz")
    Xte = load_idx(f"{DATA}/t10k-images-idx3-ubyte.gz"); yte = load_idx(f"{DATA}/t10k-labels-idx1-ubyte.gz")
    mu = Xtr.mean(axis=0); Xc = Xtr - mu
    U, S, Vt = np.linalg.svd(Xc[:20000], full_matrices=False)
    if raw:
        return Xc, ytr, Xte - mu, yte, Vt[null_from:].T
    Pm = Vt[:dim].T
    Ztr = Xc @ Pm; Zte = (Xte - mu) @ Pm
    sd = Ztr.std(axis=0) + 1e-8
    return Ztr / sd, ytr, Zte / sd, yte, None


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True); E = np.exp(Z); return E / E.sum(axis=1, keepdims=True)


def local_update(W, b, X, y, lr=0.1, batch=30, epochs=1, rng=None):
    W = W.copy(); b = b.copy(); n = len(y)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            j = idx[s:s + batch]; P = softmax(X[j] @ W + b); P[np.arange(len(j)), y[j]] -= 1
            W -= lr * X[j].T @ P / len(j); b -= lr * P.mean(axis=0)
    return W, b


def accuracy(W, b, X, y):
    return float((np.argmax(X @ W + b, axis=1) == y).mean())


def run_once(kind, par, T, q, sigma_noise, C_clip, n_clients, m_canary, data, rng, lr=0.1):
    Ztr, ytr, Zte, yte, Vnull = data
    dim = Ztr.shape[1]; d = dim * 10 + 10
    P, pi, emit = iid_chain(q) if kind == "iid" else (two_state_chain(q, par) if kind == "markov" else minsep_chain(q, int(par)))
    # client data: IID shards
    perm = rng.permutation(len(ytr)); per = len(ytr) // n_clients
    shards = [perm[i * per:(i + 1) * per] for i in range(n_clients)]
    # participation paths for real clients and canaries
    Breal = sample_paths(P, pi, emit, T, n_clients, rng)
    Bcan = sample_paths(P, pi, emit, T, m_canary, rng)
    present = rng.random(m_canary) < 0.5
    if Vnull is None:
        U = rng.standard_normal((m_canary, d))
    else:
        G = rng.standard_normal((m_canary, Vnull.shape[1], 10))
        U = np.concatenate([np.einsum('pk,mkc->mpc', Vnull, G).reshape(m_canary, dim * 10), np.zeros((m_canary, 10))], axis=1)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    W = np.zeros((dim, 10)); b = np.zeros(10)
    Xobs = np.zeros((m_canary, T))
    denom = n_clients * q
    clip_frac = []
    w0 = np.concatenate([W.ravel(), b])
    for t in range(T):
        agg = np.zeros(d)
        nclip = 0; nsel = 0
        for k in np.flatnonzero(Breal[:, t]):
            idx = shards[k]
            Wk, bk = local_update(W, b, Ztr[idx], ytr[idx], lr=lr, rng=rng)
            delta = np.concatenate([(Wk - W).ravel(), bk - b])
            nrm = np.linalg.norm(delta); nsel += 1
            if nrm > C_clip:
                delta *= C_clip / nrm; nclip += 1
            agg += delta
        clip_frac.append(nclip / max(nsel, 1))
        active = np.flatnonzero(Bcan[:, t] & present)
        if len(active):
            agg += C_clip * U[active].sum(axis=0)
        agg += sigma_noise * C_clip * rng.standard_normal(d)
        Xobs[:, t] = (U @ agg) / C_clip
        upd = agg / denom
        W += upd[:dim * 10].reshape(dim, 10); b += upd[dim * 10:]
    wT = np.concatenate([W.ravel(), b])
    cos = (U @ (wT - w0)) / (np.linalg.norm(wT - w0) + 1e-12)
    return dict(X=Xobs, present=present, counts=Bcan.sum(axis=1), P=P, pi=pi, emit=emit,
                acc=accuracy(W, b, Zte, yte), clip_frac=float(np.mean(clip_frac)), cos=cos, d=d)


def andrew_estimate(cos, present, d, delta):
    """Andrew et al. Alg. 2: null N(0, 1/d) vs Gaussian fitted to observed canaries' cosines."""
    obs = cos[present]
    mu1, s1 = obs.mean(), obs.std(ddof=1)
    return eps_gauss_pair(0.0, 1 / np.sqrt(d), mu1, s1, delta)


if __name__ == "__main__":
    T, q, sigma_noise, C_clip = 200, 0.1, 2.0, 0.5
    n_clients, m_canary, R = 200, 400, int(sys.argv[1]) if len(sys.argv) > 1 else 10
    DELTA, ALPHA = 1e-4, 0.05
    rng = np.random.default_rng(2026)
    data = prepare(raw=True, null_from=150)
    print("data ready", data[0].shape, flush=True)
    a_grid = np.linspace(0.2, 0.7, 11)
    refs = {}
    for r in csv.DictReader(open("pilot_a_results.csv")):
        if float(r["delta"]) == DELTA and int(r["T"]) == T:
            refs[(r["process"], float(r["param"]))] = (float(r["eps_full_mc"]), float(r["eps_poisson_pld"]))
    out = open("fl_results.csv", "a", newline=""); w = csv.writer(out)
    if os.path.getsize("fl_results.csv") == 0:
        w.writerow(["process", "param", "rep", "acc", "clip_frac", "absent_sd", "present_mean", "pred_present_mean", "eps_poisson", "eps_true_model1", "eps_jensen_ub",
                    "gauss_sum", "count_cond_known", "path_aware", "lr_threshold", "andrew_est"])
    t0 = time.time()
    for kind, par in [("iid", 0.0), ("markov", 0.9), ("minsep", 9)]:
        P, pi, emit = iid_chain(q) if kind == "iid" else (two_state_chain(q, par) if kind == "markov" else minsep_chain(q, int(par)))
        pn = count_distribution(P, pi, emit, T)
        table = PathAwareTable(P, pi, emit, T, a_grid, DELTA, n=100_000, seed=3)
        ef, ep = refs[(kind, float(par))]
        ub = eps_jensen_full(pn, 1.0, sigma_noise, DELTA)
        print(f"{kind} {par}: poisson={ep:.2f} model1-true={ef:.2f} jensen-ub={ub:.2f} table(0.5)={table(0.5):.2f} [{time.time()-t0:.0f}s]", flush=True)
        for rep in range(R):
            res = run_once(kind, par, T, q, sigma_noise, C_clip, n_clients, m_canary, data, rng, lr=0.05)
            X, present, counts = res["X"], res["present"], res["counts"]
            S = scores_sum(X)
            g = gauss_sum_audit(X, present, DELTA, ALPHA)
            cc = count_cond_audit(X, present, counts, pn, DELTA, ALPHA)
            pa = path_aware_audit(X, present, counts, T, table, ALPHA)
            lt = lr_threshold_audit(X, present, P, pi, emit, 1.0, sigma_noise, DELTA, ALPHA)
            an = andrew_estimate(res["cos"], present, res["d"], DELTA)
            absent_sd = S[~present].std(ddof=1); pres_mean = S[present].mean(); pred = counts[present].mean() / np.sqrt(T)
            w.writerow([kind, par, rep, f"{res['acc']:.4f}", f"{res['clip_frac']:.3f}", f"{absent_sd:.3f}", f"{pres_mean:.3f}", f"{pred:.3f}", f"{ep:.3f}", f"{ef:.3f}", f"{ub:.3f}",
                        f"{g:.3f}", f"{cc:.3f}", f"{pa:.3f}", f"{lt:.3f}", f"{an:.3f}"])
            out.flush()
            print(f"  rep {rep}: acc={res['acc']:.3f} clip={res['clip_frac']:.2f} absent_sd={absent_sd:.2f} (σ=2) present_mean={pres_mean:.2f} (pred {pred:.2f}) | gauss={g:.2f} countcond={cc:.2f} path={pa:.2f} lr_thr={lt:.2f} andrew={an:.2f} [{time.time()-t0:.0f}s]", flush=True)
    out.close()
