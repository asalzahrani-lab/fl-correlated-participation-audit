"""Step 8: misspecification of the participation process assumed by the path-aware auditor.
True canary process: Markov lambda=0.9 (T=200).  Auditor builds its eps_full(a) table at lambda in {0.8, 0.9, 0.95}
and (second case) the count-conditioned audit uses the count law of the assumed process."""
import numpy as np, csv, time
from corraudit import *
from auditors2 import *

C, sigma, q, T = 1.0, 2.0, 0.1, 200
DELTA, ALPHA, m, R = 1e-4, 0.05, 1000, 100
rng = np.random.default_rng(88)
true_lam = 0.9
Pt, pit, emitt = two_state_chain(q, true_lam)
eps_true = 12.082
a_grid = np.linspace(0.2, 0.7, 11)
tables, pns = {}, {}
t0 = time.time()
for lam in [0.8, 0.9, 0.95]:
    P, pi, emit = two_state_chain(q, lam)
    tables[lam] = PathAwareTable(P, pi, emit, T, a_grid, DELTA, n=100_000, seed=5)
    pns[lam] = count_distribution(P, pi, emit, T)
    print(f"table lambda={lam}: eps(0.5)={tables[lam](0.5):.2f} [{time.time()-t0:.0f}s]", flush=True)
out = open("step8_results.csv", "w", newline=""); w = csv.writer(out)
w.writerow(["assumed_lambda", "estimator", "mean", "sd", "coverage_vs_true", "flag_poisson"])
res = {(lam, est): [] for lam in tables for est in ["path_aware", "count_cond"]}
for rep in range(R):
    present = rng.random(m) < 0.5
    B = sample_paths(Pt, pit, emitt, T, m, rng); counts = B.sum(axis=1)
    X = (B * present[:, None]) * C + sigma * rng.standard_normal((m, T))
    for lam in tables:
        res[(lam, "path_aware")].append(path_aware_audit(X, present, counts, T, tables[lam], ALPHA))
        res[(lam, "count_cond")].append(count_cond_audit(X, present, counts, pns[lam], DELTA, ALPHA))
for (lam, est), v in res.items():
    v = np.array(v)
    w.writerow([lam, est, f"{v.mean():.3f}", f"{v.std():.3f}", f"{np.mean(v <= eps_true):.3f}", f"{np.mean(v > 2.866):.3f}"])
    print(f"assumed lambda={lam} {est:11s}: {v.mean():.2f}±{v.std():.2f} coverage={np.mean(v<=eps_true):.3f} flag={np.mean(v>2.866):.2f}", flush=True)
out.close()
