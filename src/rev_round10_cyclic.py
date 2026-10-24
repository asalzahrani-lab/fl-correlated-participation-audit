"""Revision round 10 (internal review), M10c: the certified path-aware audit on the cyclic law (Table III, row "Cyclic").
The cyclic law is the minimum-separation renewal law at b = 9 and q = 0.1: a period-10 cycle with a uniformly random
phase.  It is represented exactly as the uniform mixture over its 10 phase paths (EmpiricalPathLaw), whose likelihood
ratio is checked against the forward algorithm of the b = 9 chain.  As for the other laws (step16_delta5.py,
step19_rerun_audits.py): a certified table on the grid a = 0.40, ..., 0.52 (step 0.01) from N_MC shared samples per
grid point at delta = 1e-4 and 1e-5 (KL-certified at alpha_mc = 0.005, the step-function certification of step 14 and
the plain Monte-Carlo value), an independent reference run at a = 0.5 with a two-sided 99% interval (rev_common.eps_interval),
and 120 audits (m = 1,000, T = 200, a = 0.5) of the certified, uncertified and step-function auditors.
Outputs rev_round10_cyclic_table.csv, rev_round10_cyclic_reference.csv, rev_round10_cyclic_results.csv."""
import csv, sys, time
import numpy as np
from corraudit import minsep_chain, eps_poisson_pld, eps_sum_exact, count_distribution
from certified import EmpiricalPathLaw, ChainLaw, plain_eps
from certified2 import CertifiedTable2, snr_lower_bound, eps_cert_bern
from certified import delta_lo_curve, eps_from_lower_curve
from rev_common import eps_interval

T, a_true, m, R, q = 200, 0.5, 1000, 120, 0.1
A_EST, A_V2, A_MC = 0.045, 0.05, 0.005
N_MC = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
N_REF = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000_000
a_grid = np.round(np.arange(0.40, 0.5201, 0.01), 2)
DELTAS = [1e-4, 1e-5]
rng = np.random.default_rng(1010)
t0 = time.time(); log = lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True)

# ---- the law: 10 phase paths, uniform
paths = np.zeros((10, T), dtype=np.int8)
for ph in range(10): paths[ph, ph::10] = 1
law = EmpiricalPathLaw(paths)
P, pi, emit = minsep_chain(q, 9); chain = ChainLaw(P, pi, emit, T)
Xc = rng.standard_normal((200, T)); Xc[:100] += 0.5 * law.sample(100, rng)
dl = np.abs(law.loglr(Xc, 0.5) - chain.loglr(Xc, 0.5)).max()
log(f"cyclic law as 10 phase paths; max |log LR (paths) - log LR (b=9 chain)| on 200 test points: {dl:.2e}")
assert dl < 1e-3
pn = count_distribution(P, pi, emit, T)
claim = {d: eps_poisson_pld(q, T, 1.0, 1.0 / a_true, d) for d in DELTAS}
fm = {d: eps_sum_exact(pn, 1.0, 1.0 / a_true, T, d) for d in DELTAS}
log(f"Poisson claim {claim}; sum statistic (exact) {fm}")

# ---- independent reference at a = 0.5 with a two-sided 99% interval
lp, lq = [], []
for s in range(0, N_REF, 200_000):
    k = min(200_000, N_REF - s)
    B = law.sample(k, rng); Zp = rng.standard_normal((k, T)).astype(np.float32); Zq = rng.standard_normal((k, T)).astype(np.float32)
    lp.append(law.loglr(a_true * B + Zp, a_true)); lq.append(law.loglr(Zq, a_true))
lp = np.concatenate(lp); lq = np.concatenate(lq)
ref = {}
with open("rev_round10_cyclic_reference.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["law", "delta", "samples", "eps", "lo99", "hi99", "poisson", "ratio", "ratio_lo99", "sum_statistic_exact"])
    for d in DELTAS:
        e, lo, hi = eps_interval(lp, lq, d, alpha=0.01); ref[d] = (e, lo, hi)
        w.writerow(["cyclic_b9", f"{d:g}", N_REF, f"{e:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{claim[d]:.4f}", f"{e / claim[d]:.3f}", f"{lo / claim[d]:.3f}", f"{fm[d]:.4f}"])
        log(f"reference delta={d:g}: {e:.3f} [{lo:.3f}, {hi:.3f}] ({N_REF} samples); Poisson {claim[d]:.3f}; ratio {e / claim[d]:.2f} (lower 99% {lo / claim[d]:.2f})")
del lp, lq

# ---- certified table (delta = 1e-5 in the constructor; 1e-4 recomputed from the same samples)
tab = CertifiedTable2(law, a_grid, 1e-5, N_MC, A_MC, rng, chunk=200_000, log=log, keep_samples=True)
cert4, step4, plain4 = [], [], []
for gi, a in enumerate(a_grid):
    lpg = tab.lps[gi].astype(np.float64); lqg = tab.lqs[gi].astype(np.float64)
    cert4.append(eps_cert_bern(lpg, lqg, 1e-4, A_MC)); step4.append(eps_from_lower_curve(delta_lo_curve(lpg, lqg, A_MC), 1e-4)); plain4.append(plain_eps(lpg, lqg, 1e-4))
    log(f"  a={a:.2f}: 1e-4 plain {plain4[-1]:.3f} cert {cert4[-1]:.3f} step {step4[-1]:.3f} | 1e-5 plain {tab.eps_plain[gi]:.3f} cert {tab.eps_bern[gi]:.3f}")
cert4, step4, plain4 = map(np.array, (cert4, step4, plain4))
del tab.lps, tab.lqs
with open("rev_round10_cyclic_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["a", "eps_plain_1e-4", "eps_cert_1e-4", "eps_step_1e-4", "eps_plain_1e-5", "eps_cert_1e-5"])
    for gi, a in enumerate(a_grid):
        w.writerow([f"{a:.2f}", f"{plain4[gi]:.4f}", f"{cert4[gi]:.4f}", f"{step4[gi]:.4f}", f"{tab.eps_plain[gi]:.4f}", f"{tab.eps_bern[gi]:.4f}"])

# ---- 120 audits
def floor(arr, x):
    i = np.searchsorted(a_grid - 1e-9, x, side="right") - 1
    return 0.0 if i < 0 else float(arr[i])
res = {k: [] for k in ["cert_1e-4", "v2_1e-4", "step_1e-4", "cert_1e-5", "v2_1e-5", "alo"]}
for rep in range(R):
    present = rng.random(m) < 0.5
    B = law.sample(m, rng)
    X = (B * present[:, None]) * a_true + rng.standard_normal((m, T)).astype(np.float32)
    a_c = snr_lower_bound(X, present, B, A_EST); a_v = snr_lower_bound(X, present, B, A_V2); res["alo"].append(a_c)
    res["cert_1e-4"].append(floor(cert4, a_c)); res["step_1e-4"].append(floor(step4, a_c)); res["v2_1e-4"].append(float(np.interp(a_v, a_grid, plain4, left=0.0)))
    res["cert_1e-5"].append(floor(tab.eps_bern, a_c)); res["v2_1e-5"].append(float(np.interp(a_v, a_grid, tab.eps_plain, left=0.0)))
out = []
for est, v in res.items():
    if est == "alo": continue
    v = np.array(v); d = 1e-5 if est.endswith("1e-5") else 1e-4; r, ep = ref[d][0], claim[d]
    out.append(["cyclic_b9", f"{d:g}", est, R, f"{v.mean():.4f}", f"{v.std():.4f}", f"{v.min():.4f}", f"{v.max():.4f}", f"{r:.4f}", f"{ep:.4f}",
                f"{v.mean() / r:.4f}", int(np.sum(v > r)), int(np.sum(v > ep)), int(np.sum(v > ref[d][2]))])
    log(f"cyclic {est:10s}: {v.mean():.3f}+-{v.std():.3f} [{v.min():.2f},{v.max():.2f}] ref {r:.3f} recovery {v.mean() / r:.3f} "
        f"above ref {int(np.sum(v > r))}/{R} flags {int(np.sum(v > ep))}/{R}")
al = np.array(res["alo"]); log(f"a_lo mean {al.mean():.4f} sd {al.std():.4f} min {al.min():.4f} max {al.max():.4f}; {1 - al.mean() / a_true:.3f} below a; a_lo > a in {int(np.sum(al > a_true))}")
with open("rev_round10_cyclic_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["law", "delta", "estimator", "R", "mean", "sd", "min", "max", "reference", "poisson", "recovery", "n_above_reference",
                                   "n_above_poisson", "n_above_reference_hi99"]); w.writerows(out)
np.savez("rev_round10_cyclic_audits.npz", **{k: np.array(v) for k, v in res.items()})
log("done")
