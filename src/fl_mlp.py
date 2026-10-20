"""Non-convex check: DP-FedAvg with a 1-hidden-layer MLP (784-64-10, ~51k params) on Fashion-MNIST, RANDOM canary
directions (no null-space construction).  Measures canary erasure and runs the audits (Gaussian fit, count-conditioned,
path-aware v2) under i.i.d. and Markov lambda=0.9 participation."""
import numpy as np, csv, time, sys, os
from corraudit import *
from auditors2 import *
from fl_sim import load_idx, DATA

H = 64


def init_params(rng, dim=784):
    W1 = rng.standard_normal((dim, H)).astype(np.float32) * np.sqrt(2.0 / dim); b1 = np.zeros(H, np.float32)
    W2 = rng.standard_normal((H, 10)).astype(np.float32) * np.sqrt(1.0 / H); b2 = np.zeros(10, np.float32)
    return [W1, b1, W2, b2]


def flat(p): return np.concatenate([x.ravel() for x in p])


def unflat(v, p):
    out = []; s = 0
    for x in p:
        out.append(v[s:s + x.size].reshape(x.shape).astype(np.float32)); s += x.size
    return out


def forward(p, X):
    W1, b1, W2, b2 = p
    Hh = np.maximum(X @ W1 + b1, 0.0)
    Z = Hh @ W2 + b2
    return Hh, Z


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True); E = np.exp(Z); return E / E.sum(axis=1, keepdims=True)


def local_update(p, X, y, lr, batch, rng, epochs=1):
    W1, b1, W2, b2 = [x.copy() for x in p]
    n = len(y)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            j = idx[s:s + batch]; Xb = X[j]
            Hh = np.maximum(Xb @ W1 + b1, 0.0); Pm = softmax(Hh @ W2 + b2); Pm[np.arange(len(j)), y[j]] -= 1; Pm /= len(j)
            gW2 = Hh.T @ Pm; gb2 = Pm.sum(axis=0)
            dH = (Pm @ W2.T) * (Hh > 0)
            gW1 = Xb.T @ dH; gb1 = dH.sum(axis=0)
            W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return [W1, b1, W2, b2]


def accuracy(p, X, y):
    return float((np.argmax(forward(p, X)[1], axis=1) == y).mean())


def run_once(kind, par, T, q, sigma_noise, C_clip, n_clients, m_canary, data, rng, lr=0.05):
    Xtr, ytr, Xte, yte = data
    P, pi, emit = iid_chain(q) if kind == "iid" else two_state_chain(q, par)
    perm = rng.permutation(len(ytr)); per = len(ytr) // n_clients
    shards = [perm[i * per:(i + 1) * per] for i in range(n_clients)]
    Breal = sample_paths(P, pi, emit, T, n_clients, rng); Bcan = sample_paths(P, pi, emit, T, m_canary, rng)
    present = rng.random(m_canary) < 0.5
    p = init_params(rng); d = flat(p).size
    U = np.empty((m_canary, d), np.float32)                      # chunked: same stream as one call, no float64 temp
    for i0 in range(0, m_canary, 500):
        U[i0:i0 + 500] = rng.standard_normal((min(500, m_canary - i0), d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
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
    return dict(X=Xobs, present=present, counts=Bcan.sum(axis=1), B=Bcan, acc=accuracy(p, Xte, yte), clip_frac=float(np.mean(clipf)), d=d)


if __name__ == "__main__":
    T, q, sigma_noise = 200, 0.1, 2.0
    n_clients, m_canary, R = 200, 1000, int(sys.argv[1]) if len(sys.argv) > 1 else 8
    DELTA, ALPHA = 1e-4, 0.05
    rng = np.random.default_rng(4242)
    Xtr = load_idx(f"{DATA}/train-images-idx3-ubyte.gz"); ytr = load_idx(f"{DATA}/train-labels-idx1-ubyte.gz")
    Xte = load_idx(f"{DATA}/t10k-images-idx3-ubyte.gz"); yte = load_idx(f"{DATA}/t10k-labels-idx1-ubyte.gz")
    mu = Xtr.mean(axis=0); data = (Xtr - mu, ytr, Xte - mu, yte)
    # clip norm from typical round-0 update norms
    p0 = init_params(rng); norms = []
    for k in range(20):
        idx = rng.permutation(len(ytr))[:300]; pk = local_update(p0, data[0][idx], ytr[idx], 0.05, 30, rng)
        norms.append(np.linalg.norm(flat(pk) - flat(p0)))
    C_clip = float(np.median(norms)) * 0.7
    print(f"update norm median {np.median(norms):.3f} -> C_clip={C_clip:.3f}", flush=True)
    refs = {}
    for r in csv.DictReader(open("pilot_a_results.csv")):
        if float(r["delta"]) == DELTA and int(r["T"]) == T:
            refs[(r["process"], float(r["param"]))] = (float(r["eps_full_mc"]), float(r["eps_poisson_pld"]))
    a_grid = np.linspace(0.2, 0.7, 11)
    out = open("fl_mlp_results.csv", "w", newline=""); w = csv.writer(out)
    w.writerow(["process", "param", "rep", "acc", "clip_frac", "absent_sd", "present_mean", "pred_present_mean", "erasure_ratio", "eps_poisson", "eps_true_model1", "gauss_sum", "count_cond_known", "path_aware_v2"])
    t0 = time.time()
    for kind, par in [("iid", 0.0), ("markov", 0.9)]:
        P, pi, emit = iid_chain(q) if kind == "iid" else two_state_chain(q, par)
        pn = count_distribution(P, pi, emit, T)
        table = PathAwareTable(P, pi, emit, T, a_grid, DELTA, n=100_000, seed=11)
        ef, ep = refs[(kind, float(par))]
        print(f"{kind} {par}: poisson={ep:.2f} model1-true={ef:.2f} [{time.time()-t0:.0f}s]", flush=True)
        for rep in range(R):
            res = run_once(kind, par, T, q, sigma_noise, C_clip, n_clients, m_canary, data, rng)
            X, present, counts, B = res["X"], res["present"], res["counts"], res["B"]
            S = scores_sum(X)
            g = gauss_sum_audit(X, present, DELTA, ALPHA); cc = count_cond_audit(X, present, counts, pn, DELTA, ALPHA)
            pa = path_aware_audit_v2(X, present, B, T, table, ALPHA)
            absent_sd = S[~present].std(ddof=1); pres_mean = S[present].mean(); pred = counts[present].mean() / np.sqrt(T)
            w.writerow([kind, par, rep, f"{res['acc']:.4f}", f"{res['clip_frac']:.3f}", f"{absent_sd:.3f}", f"{pres_mean:.3f}", f"{pred:.3f}", f"{pres_mean/pred:.3f}", f"{ep:.3f}", f"{ef:.3f}", f"{g:.3f}", f"{cc:.3f}", f"{pa:.3f}"]); out.flush()
            print(f"  rep {rep}: acc={res['acc']:.3f} clip={res['clip_frac']:.2f} absent_sd={absent_sd:.2f} present/pred={pres_mean/pred:.2f} | gauss={g:.2f} countcond={cc:.2f} path_v2={pa:.2f} [{time.time()-t0:.0f}s]", flush=True)
    out.close()
