"""Round 14: DP-FedAvg with canary clients on CIFAR-10 with a convolutional network (the reviews of TIFS v12 and v13
asked for a model that is not a small MLP on MNIST-type data).

Protocol: that of rev_train.py with the network replaced.  Network: conv 3x3 (3 -> 32) + ReLU + max-pool 2, conv 3x3
(32 -> 64) + ReLU + max-pool 2, fully connected 4,096 -> 128 + ReLU, 128 -> 10 (545,098 parameters), weights by
PyTorch's default initialisation (seeded).  Data: CIFAR-10 (50,000 training, 10,000 test images; pixels scaled to [0, 1]
and centred by the training mean per channel), 200 real clients with IID shards of 250 images.  T = 200 rounds; real
clients and canaries draw their participation paths from the stated law (rate 0.1; Markov lambda = 0.9 or i.i.d.).  A
participating real client runs one epoch of SGD (learning rate 0.05, batch 30, cross-entropy) from the global model and
its update is clipped to norm C, fixed before training at 0.7 times the median norm of 20 local updates of the initial
model on random batches of 250 training images (the canaries do not enter the calibration).  m = 1,000 canaries, each
present with probability 1/2, contribute C u_i in the rounds they participate, injected after clipping; u_i are
orthonormal rows of a randomised Hadamard matrix on a random set of 2^19 coordinates.  The server adds N(0, (r sigma C)^2)
with sigma = 2 and r = 1 (r < 1 in the noise-error configuration) and updates theta <- theta + (clipped sum + canaries +
noise) / (n q).  The auditor projects each aggregate on u_i and divides by C (the diagnostics divide by sigma, as in
rev_train.py); the certified audit, the diagnostics and the tables are those of rev_train.py (claim: the correct accountant's value for the canaries' law at the nominal
noise, 11.869 Markov / rev_train.REF_IID = 2.869 i.i.d., the Monte-Carlo reference against the exact 2.866; the Poisson
claim 2.866 is also recorded; no flag count differs).

The CIFAR-10 copy used here is the JPEG re-encoding hosted at github.com/YoongiKim/CIFAR-10-images (the official
archive is not reachable from the build machine); file cifar10_jpeg_mirror.npz, P3_DATA_CIFAR points to its folder.

usage (from results/):  P3_DATA_CIFAR=/path python3 ../src/rev_round14_cnn.py CONFIG [R]
   CONFIG in cnn_markov (seed 1431), cnn_iid (1432), cnn_markov_r0.90 (1433).  About 15 to 25 minutes per run on one or two cores.
Output: rev_round14_cnn_<CONFIG>.csv, one row per run (columns as rev_round14_train.py; true_loss_r is the step-14
table's value at min(a/r, 0.55), a lower estimate of the realised loss), appended on resume.
"""
import sys, csv, time, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from rev_train import HadamardCanaries, diagnostics, CsvTable, T, Q, SIGMA, A_TRUE, REF_MARKOV, REF_IID, POIS
from corraudit import two_state_chain, iid_chain, sample_paths
from certified2 import snr_lower_bound

torch.set_num_threads(int(os.environ.get("P3_THREADS", "2")))
DATA = os.environ.get("P3_DATA_CIFAR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cifar10"))
CONFIGS = {"cnn_markov": ("markov", 1.00, 1431), "cnn_iid": ("iid", 1.00, 1432), "cnn_markov_r0.90": ("markov", 0.90, 1433)}
N_CLIENTS, M, LR, BATCH = 200, 1000, 0.05, 30


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1); self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.f1 = nn.Linear(64 * 8 * 8, 128); self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f2(F.relu(self.f1(x.flatten(1))))


def get_flat(net): return torch.cat([p.data.view(-1) for p in net.parameters()])


def set_flat(net, v):
    i = 0
    for p in net.parameters():
        n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n


def local_update(net, theta, X, y, rng):
    set_flat(net, theta); opt = torch.optim.SGD(net.parameters(), lr=LR)
    perm = rng.permutation(len(y))
    for s in range(0, len(y), BATCH):
        idx = perm[s:s + BATCH]; opt.zero_grad()
        F.cross_entropy(net(X[idx]), y[idx]).backward(); opt.step()
    return get_flat(net) - theta


@torch.no_grad()
def accuracy(net, theta, X, y):
    set_flat(net, theta); correct = 0
    for s in range(0, len(y), 1000): correct += int((net(X[s:s + 1000]).argmax(1) == y[s:s + 1000]).sum())
    return correct / len(y)


def run_once(law, r, data, C_clip, rng, net):
    Xtr, ytr, Xte, yte = data
    P, pi, emit = iid_chain(Q) if law == "iid" else two_state_chain(Q, 0.9)
    perm = rng.permutation(len(ytr)); per = len(ytr) // N_CLIENTS
    shards = [torch.as_tensor(perm[i * per:(i + 1) * per]) for i in range(N_CLIENTS)]
    Breal = sample_paths(P, pi, emit, T, N_CLIENTS, rng); Bcan = sample_paths(P, pi, emit, T, M, rng)
    present = rng.random(M) < 0.5
    torch.manual_seed(int(rng.integers(2**31))); net0 = Net(); theta = get_flat(net0).clone(); d = theta.numel()
    can = HadamardCanaries(d, M, rng)
    X = np.zeros((M, T), np.float32); denom = N_CLIENTS * Q; clipf = []
    for t in range(T):
        agg = torch.zeros(d); nclip = nsel = 0
        for k in np.flatnonzero(Breal[:, t]):
            delta = local_update(net, theta, Xtr[shards[k]], ytr[shards[k]], rng)
            nrm = float(delta.norm()); nsel += 1
            if nrm > C_clip: delta *= C_clip / nrm; nclip += 1
            agg += delta
        clipf.append(nclip / max(nsel, 1))
        agg_np = agg.numpy().astype(np.float32)
        active = np.flatnonzero(Bcan[:, t] & present)
        if len(active): agg_np += C_clip * can.sum_dirs(active)
        agg_np += (r * SIGMA * C_clip * rng.standard_normal(d)).astype(np.float32)
        X[:, t] = can.project(agg_np) / C_clip
        theta = theta + torch.from_numpy(agg_np) / denom
    return dict(X=X, present=present, B=Bcan, acc=accuracy(net, theta, Xte, yte), clip=float(np.mean(clipf)), d=d)


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
    out_path = f"rev_round14_cnn_{name}.csv"; new = not os.path.exists(out_path)
    done = 0 if new else len(list(csv.DictReader(open(out_path))))
    f = open(out_path, "a", newline=""); w = csv.writer(f)
    if new:
        w.writerow(["config", "rep", "d", "n_clients", "m_canary", "law", "partition", "noise_r", "C_clip", "acc", "clip_frac",
                    "a_lo", "covered_nominal", "a_hat", "s_hat", "lag1", "xcorr_mean", "xcorr_sd", "xcorr_maxabs", "ks", "res_part", "res_nonpart",
                    "erasure", "cert", "uncert", "claim", "true_loss_r", "poisson", "above_true_r", "flag_claim", "flag_poisson"])
    t0 = time.time()
    print(f"{name}: CNN d={theta0.numel()} clients={N_CLIENTS} canaries={M} law={law} noise_r={r} C_clip={C_clip:.3f}; resuming at rep {done}", flush=True)
    for rep in range(R):
        if rep < done:                                     # keep the random stream aligned on resume
            rng = np.random.default_rng(seed + 1000 * (rep + 1)); continue
        rng = np.random.default_rng(seed + 1000 * (rep + 1))
        res = run_once(law, r, data, C_clip, rng, net)
        X, present, B = res["X"], res["present"], res["B"]
        a_lo = snr_lower_bound(X, present, B, 0.045); a_v2 = snr_lower_bound(X, present, B, 0.05)
        dg = diagnostics(X, present, B)
        S = X.sum(axis=1) / np.sqrt(T); counts = B.sum(axis=1)
        er = float(S[present].mean() / (counts[present].mean() / np.sqrt(T)))
        cert = tab.cert(a_lo); unc = tab.plain_interp(a_v2); true_r = tab.plain_interp(min(A_TRUE / r, tab.a[-1]))
        w.writerow([name, rep, res["d"], N_CLIENTS, M, law, "iid", r, f"{C_clip:.4f}", f"{res['acc']:.4f}", f"{res['clip']:.3f}",
                    f"{a_lo:.4f}", int(a_lo <= A_TRUE), f"{dg['a_hat']:.4f}", f"{dg['s_hat']:.4f}", f"{dg['lag1']:.4f}",
                    f"{dg['xcorr_mean']:.5f}", f"{dg['xcorr_sd']:.5f}", f"{dg['xcorr_maxabs']:.4f}", f"{dg['ks']:.5f}",
                    f"{dg['res_part']:.4f}", f"{dg['res_nonpart']:.4f}", f"{er:.3f}", f"{cert:.3f}", f"{unc:.3f}",
                    f"{ref:.3f}", f"{true_r:.3f}", f"{POIS:.3f}", int(cert > true_r), int(cert > ref), int(cert > POIS)]); f.flush()
        print(f"  rep {rep}: acc={res['acc']:.3f} clip={res['clip']:.2f} a_lo={a_lo:.3f} a_hat={dg['a_hat']:.3f} s_hat={dg['s_hat']:.3f} lag1={dg['lag1']:.3f} "
              f"ks={dg['ks']:.4f} erasure={er:.2f} cert={cert:.2f} (claim {ref:.2f}) [{time.time()-t0:.0f}s]", flush=True)
    f.close()
