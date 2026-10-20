"""Step 5b: path-regression estimator of a (v2) vs the mean-gap version (v1)."""
import numpy as np, csv, time
from corraudit import *
from auditors2 import *
C, sigma, q = 1.0, 2.0, 0.1
DELTA, ALPHA = 1e-4, 0.05
rng = np.random.default_rng(31)
refs = {}
for r in csv.DictReader(open("pilot_a_results.csv")):
    if float(r["delta"]) == DELTA:
        refs[(int(r["T"]), r["process"], float(r["param"]))] = (float(r["eps_full_mc"]), float(r["eps_poisson_pld"]))
cells = [(200, "iid", 0.0, 100), (200, "markov", 0.9, 100), (200, "minsep", 9, 100), (1000, "markov", 0.9, 40), (1000, "minsep", 9, 40)]
a_grid = np.linspace(0.2, 0.7, 11)
out = open("step5b_results.csv", "w", newline=""); w = csv.writer(out)
w.writerow(["T", "process", "param", "m", "R", "eps_true", "eps_poisson", "estimator", "mean", "sd", "coverage", "flag_poisson", "ratio"])
t0 = time.time()
for (T, kind, par, R) in cells:
    P, pi, emit = iid_chain(q) if kind == "iid" else (two_state_chain(q, par) if kind == "markov" else minsep_chain(q, int(par)))
    ef, ep = refs[(T, kind, float(par))]
    table = PathAwareTable(P, pi, emit, T, a_grid, DELTA, n=100_000 if T == 200 else 40_000, seed=2)
    res = {"path_aware_v1": [], "path_aware_v2": []}
    for m in [1000]:
        for rep in range(R):
            present = rng.random(m) < 0.5
            B = sample_paths(P, pi, emit, T, m, rng); counts = B.sum(axis=1)
            X = (B * present[:, None]) * C + sigma * rng.standard_normal((m, T))
            res["path_aware_v1"].append(path_aware_audit(X, present, counts, T, table, ALPHA))
            res["path_aware_v2"].append(path_aware_audit_v2(X, present, B, T, table, ALPHA))
    for name, v in res.items():
        v = np.array(v)
        w.writerow([T, kind, par, 1000, R, f"{ef:.3f}", f"{ep:.3f}", name, f"{v.mean():.3f}", f"{v.std():.3f}", f"{np.mean(v <= ef):.3f}", f"{np.mean(v > ep):.3f}", f"{v.mean()/ef:.3f}"])
        print(f"T={T} {kind} {par}: true={ef:.2f} | {name}: {v.mean():.2f}±{v.std():.2f} cov={np.mean(v<=ef):.3f} flag={np.mean(v>ep):.2f} ratio={v.mean()/ef:.2f} [{time.time()-t0:.0f}s]", flush=True)
    out.flush()
out.close()
