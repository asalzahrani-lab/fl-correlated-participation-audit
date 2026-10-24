"""Revision C: real DP-FedAvg training with canary clients, extended after the pre-submission reviews.

New relative to fl_mlp2.py / fl_mlp3.py:
  * orthonormal canary directions (randomised Hadamard rows on a random coordinate subset of size 2^k <= d), so that
    the projections of the shared Gaussian noise on different canaries are exactly independent N(0, sigma^2 C^2);
  * configurable MLP depth/width, number of real clients and number of canaries (canary : real-client ratio);
  * per-run diagnostics of the observation model: coverage of the lower confidence limit a_lo <= a (= 1/sigma),
    the point estimate a_hat and the noise-scale estimate s_hat, the lag-1 autocorrelation and the cross-canary
    correlation of absent-canary observations, a KS distance of the standardised absent observations from N(0,1),
    and the mean residual of present canaries on participating vs non-participating rounds.
Server update, clipping calibration, local SGD and the audits are those of fl_mlp2.py.

usage: python rev_train.py CONFIG R      (CONFIG in base_orth, ratio_1to10, ratio_1to2, medium, medium_iid, base_rand,
                                          ctrl200_share01, ctrl200_share05)"""
import sys, csv, time, os
import numpy as np
from scipy.stats import kstest, norm
from corraudit import two_state_chain, iid_chain, sample_paths
from certified2 import snr_lower_bound
from fl_sim import load_idx, DATA

T, Q, SIGMA, A_TRUE, DELTA = 200, 0.1, 2.0, 0.5, 1e-4
REF_MARKOV, REF_IID, POIS = 11.869, 2.869, 2.866          # 10M-sample reference (paper) and exact PLD values
CONFIGS = {
    #               hidden      n_clients  m_canary  directions  law
    "base_orth":   ((64,),      200,       1000,     "hadamard", "markov"),
    "base_rand":   ((64,),      200,       1000,     "random",   "markov"),
    "ratio_1to10": ((64,),      1000,      200,      "hadamard", "markov"),
    "ratio_1to2":  ((64,),      1000,      1000,     "hadamard", "markov"),
    "medium":      ((256, 128), 500,       1000,     "hadamard", "markov"),
    "medium_iid":  ((256, 128), 500,       1000,     "hadamard", "iid"),
    # round 11: accuracy controls at 200 real clients (canary share 0.1 and 0.5 instead of base_orth's 2.5)
    "ctrl200_share01": ((64,),  200,       40,       "hadamard", "markov"),
    "ctrl200_share05": ((64,),  200,       200,      "hadamard", "markov"),
}


# ---------------------------------------------------------------- MLP
def init_params(rng, hidden, dim=784, out=10):
    sizes = [dim] + list(hidden) + [out]; p = []
    for i in range(len(sizes) - 1):
        p.append((rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / sizes[i])).astype(np.float32))
        p.append(np.zeros(sizes[i + 1], np.float32))
    return p


def flat(p): return np.concatenate([x.ravel() for x in p])


def unflat(v, p):
    out = []; s = 0
    for x in p:
        out.append(v[s:s + x.size].reshape(x.shape).astype(np.float32)); s += x.size
    return out


def logits(p, X):
    h = X
    for i in range(0, len(p) - 2, 2):
        h = np.maximum(h @ p[i] + p[i + 1], 0.0)
    return h @ p[-2] + p[-1]


def local_update(p, X, y, lr, batch, rng):
    p = [x.copy() for x in p]; n = len(y); idx = rng.permutation(n); L = len(p) // 2
    for s in range(0, n, batch):
        j = idx[s:s + batch]; acts = [X[j]]
        for i in range(L - 1):
            acts.append(np.maximum(acts[-1] @ p[2 * i] + p[2 * i + 1], 0.0))
        Z = acts[-1] @ p[-2] + p[-1]; Z -= Z.max(axis=1, keepdims=True); P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(j)), y[j]] -= 1; g = P / len(j)
        for i in range(L - 1, -1, -1):
            gW = acts[i].T @ g; gb = g.sum(axis=0)
            if i > 0:
                g = (g @ p[2 * i].T) * (acts[i] > 0)
            p[2 * i] -= lr * gW; p[2 * i + 1] -= lr * gb
    return p


def accuracy(p, X, y): return float((np.argmax(logits(p, X), axis=1) == y).mean())


# ---------------------------------------------------------------- canary directions
def fwht(x):
    """Unnormalised fast Walsh-Hadamard transform along the last axis (length a power of 2)."""
    x = x.copy(); n = x.shape[-1]; h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = x[..., 0, :].copy(), x[..., 1, :].copy()
        x[..., 0, :] = a + b; x[..., 1, :] = a - b
        x = x.reshape(*x.shape[:-3], n); h *= 2
    return x


class HadamardCanaries:
    """u_i = (H D)[k_i, :] / sqrt(n) placed on a random coordinate subset of size n = 2^floor(log2 d): orthonormal."""
    def __init__(self, d, m, rng):
        self.n = 1 << int(np.floor(np.log2(d))); assert m <= self.n
        self.coords = rng.choice(d, self.n, replace=False); self.D = rng.choice([-1.0, 1.0], self.n).astype(np.float32)
        self.k = rng.choice(self.n, m, replace=False); self.d = d

    def project(self, v):                       # U @ v
        return fwht(self.D * v[self.coords])[self.k] / np.sqrt(self.n)

    def sum_dirs(self, active):                 # sum_{i in active} u_i  (length d)
        e = np.zeros(self.n, np.float32); e[self.k[active]] = 1.0
        out = np.zeros(self.d, np.float32); out[self.coords] = self.D * fwht(e) / np.sqrt(self.n)
        return out


class RandomCanaries:
    def __init__(self, d, m, rng):
        self.U = np.empty((m, d), np.float32)
        for i0 in range(0, m, 500):
            self.U[i0:i0 + 500] = rng.standard_normal((min(500, m - i0), d))
        self.U /= np.linalg.norm(self.U, axis=1, keepdims=True)

    def project(self, v): return self.U @ v

    def sum_dirs(self, active): return self.U[active].sum(axis=0)


# ---------------------------------------------------------------- one run
def run_once(cfg, data, C_clip, rng, lr=0.05, batch=30):
    hidden, n_clients, m, dirs, law = cfg
    Xtr, ytr, Xte, yte = data
    P, pi, emit = iid_chain(Q) if law == "iid" else two_state_chain(Q, 0.9)
    perm = rng.permutation(len(ytr)); per = len(ytr) // n_clients
    shards = [perm[i * per:(i + 1) * per] for i in range(n_clients)]
    Breal = sample_paths(P, pi, emit, T, n_clients, rng); Bcan = sample_paths(P, pi, emit, T, m, rng)
    present = rng.random(m) < 0.5
    p = init_params(rng, hidden); d = flat(p).size
    can = HadamardCanaries(d, m, rng) if dirs == "hadamard" else RandomCanaries(d, m, rng)
    X = np.zeros((m, T), np.float32); denom = n_clients * Q; clipf = []
    for t in range(T):
        agg = np.zeros(d, np.float32); nclip = nsel = 0
        for k in np.flatnonzero(Breal[:, t]):
            pk = local_update(p, Xtr[shards[k]], ytr[shards[k]], lr, batch, rng)
            delta = flat(pk) - flat(p); nrm = np.linalg.norm(delta); nsel += 1
            if nrm > C_clip: delta *= C_clip / nrm; nclip += 1
            agg += delta
        clipf.append(nclip / max(nsel, 1))
        active = np.flatnonzero(Bcan[:, t] & present)
        if len(active): agg += C_clip * can.sum_dirs(active)
        agg += (SIGMA * C_clip * rng.standard_normal(d)).astype(np.float32)
        X[:, t] = can.project(agg) / C_clip
        p = unflat(flat(p) + agg / denom, p)
    return dict(X=X, present=present, B=Bcan, acc=accuracy(p, Xte, yte), clip=float(np.mean(clipf)), d=d)


def diagnostics(X, present, B):
    """Observation-model diagnostics in noise units (X is in units of C; the model says X = B + sigma Z)."""
    Xa = X[~present] / SIGMA                                  # absent: pure noise + residual of the real clients
    s_hat = float(Xa.std())
    lag1 = float(np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in Xa]))
    sub = Xa[:200]; Cm = np.corrcoef(sub); off = Cm[~np.eye(len(sub), dtype=bool)]
    ks = float(kstest(((Xa - Xa.mean()) / s_hat).ravel(), "norm").statistic)
    Bp = B[present].astype(bool); Xp = X[present] / SIGMA
    a_hat = float(Xp[Bp].mean())                              # regression estimate of a in noise units
    res_part = float(Xp[Bp].mean() - A_TRUE); res_non = float(Xp[~Bp].mean())
    return dict(s_hat=s_hat, lag1=lag1, xcorr_mean=float(off.mean()), xcorr_sd=float(off.std()),
                xcorr_maxabs=float(np.abs(off).max()), ks=ks, a_hat=a_hat, res_part=res_part, res_nonpart=res_non)


class CsvTable:
    def __init__(self, path):
        rows = list(csv.DictReader(open(path)))
        self.a = np.array([float(r["a"]) for r in rows]); self.plain = np.array([float(r["eps_plain"]) for r in rows])
        self.cert_ = np.array([float(r["eps_cert_bern_0.005"]) for r in rows])     # KL-form certificate (see README)

    def cert(self, a):
        i = np.searchsorted(self.a - 1e-9, a, side="right") - 1
        return 0.0 if i < 0 else float(self.cert_[i])

    def plain_interp(self, a): return float(np.interp(a, self.a, self.plain, left=0.0))


if __name__ == "__main__":
    name, R = sys.argv[1], int(sys.argv[2]); cfg = CONFIGS[name]
    rng = np.random.default_rng({"base_orth": 901, "base_rand": 902, "ratio_1to10": 903, "ratio_1to2": 904, "medium": 905, "medium_iid": 906, "ctrl200_share01": 901, "ctrl200_share05": 901}[name])   # controls use base_orth's seed, hence its clipping norm
    Xtr = load_idx(f"{DATA}/train-images-idx3-ubyte.gz"); ytr = load_idx(f"{DATA}/train-labels-idx1-ubyte.gz")
    Xte = load_idx(f"{DATA}/t10k-images-idx3-ubyte.gz"); yte = load_idx(f"{DATA}/t10k-labels-idx1-ubyte.gz")
    mu = Xtr.mean(axis=0); data = (Xtr - mu, ytr, Xte - mu, yte)
    per = len(ytr) // cfg[1]
    p0 = init_params(rng, cfg[0]); norms = []
    for k in range(20):
        idx = rng.permutation(len(ytr))[:per]; pk = local_update(p0, data[0][idx], ytr[idx], 0.05, 30, rng)
        norms.append(np.linalg.norm(flat(pk) - flat(p0)))
    C_clip = float(np.median(norms)) * 0.7
    tab = CsvTable("step14_table_iid.csv" if cfg[4] == "iid" else "step14_table_markov.csv")
    ref = REF_IID if cfg[4] == "iid" else REF_MARKOV
    out_path = f"rev_train_{name}.csv"; new = not os.path.exists(out_path)
    done = 0 if new else len(list(csv.DictReader(open(out_path))))
    f = open(out_path, "a", newline=""); w = csv.writer(f)
    if new:
        w.writerow(["config", "rep", "d", "n_clients", "m_canary", "dirs", "law", "C_clip", "acc", "clip_frac", "a_lo", "covered",
                    "a_hat", "s_hat", "lag1", "xcorr_mean", "xcorr_sd", "xcorr_maxabs", "ks", "res_part", "res_nonpart",
                    "erasure", "cert", "uncert", "ref", "poisson", "above_ref", "flag"])
    t0 = time.time()
    print(f"{name}: hidden={cfg[0]} clients={cfg[1]} canaries={cfg[2]} dirs={cfg[3]} law={cfg[4]} C_clip={C_clip:.3f}; resuming at rep {done}", flush=True)
    for rep in range(R):
        res = run_once(cfg, data, C_clip, rng)            # always run, so that the random stream is the same on resume
        if rep < done:
            continue
        X, present, B = res["X"], res["present"], res["B"]
        a_lo = snr_lower_bound(X, present, B, 0.045); a_v2 = snr_lower_bound(X, present, B, 0.05)
        dg = diagnostics(X, present, B)
        S = X.sum(axis=1) / np.sqrt(T); counts = B.sum(axis=1)
        er = float(S[present].mean() / (counts[present].mean() / np.sqrt(T)))
        cert = tab.cert(a_lo); unc = tab.plain_interp(a_v2)
        w.writerow([name, rep, res["d"], cfg[1], cfg[2], cfg[3], cfg[4], f"{C_clip:.4f}", f"{res['acc']:.4f}", f"{res['clip']:.3f}",
                    f"{a_lo:.4f}", int(a_lo <= A_TRUE), f"{dg['a_hat']:.4f}", f"{dg['s_hat']:.4f}", f"{dg['lag1']:.4f}",
                    f"{dg['xcorr_mean']:.5f}", f"{dg['xcorr_sd']:.5f}", f"{dg['xcorr_maxabs']:.4f}", f"{dg['ks']:.5f}",
                    f"{dg['res_part']:.4f}", f"{dg['res_nonpart']:.4f}", f"{er:.3f}", f"{cert:.3f}", f"{unc:.3f}",
                    f"{ref:.3f}", f"{POIS:.3f}", int(cert > ref), int(cert > POIS)]); f.flush()
        print(f"  rep {rep}: d={res['d']} acc={res['acc']:.3f} clip={res['clip']:.2f} a_lo={a_lo:.3f} a_hat={dg['a_hat']:.3f} s_hat={dg['s_hat']:.3f} "
              f"lag1={dg['lag1']:.3f} xcorr={dg['xcorr_mean']:.4f}±{dg['xcorr_sd']:.4f} ks={dg['ks']:.4f} erasure={er:.2f} cert={cert:.2f} unc={unc:.2f} [{time.time()-t0:.0f}s]", flush=True)
    f.close()
