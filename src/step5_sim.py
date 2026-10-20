"""Step 5: compare the four auditors in simulation (Model 1), T=200 and T=1000."""
import numpy as np, csv, time, sys
from corraudit import *
from auditors2 import *

C, sigma, q = 1.0, 2.0, 0.1
DELTA, ALPHA = 1e-4, 0.05
rng = np.random.default_rng(21)
refs = {}
for r in csv.DictReader(open("pilot_a_results.csv")):
    if float(r["delta"]) == DELTA:
        refs[(int(r["T"]), r["process"], float(r["param"]))] = (float(r["eps_full_mc"]), float(r["eps_sum_exact"]), float(r["eps_poisson_pld"]))

cells = [(200, "iid", 0.0, 1000, 200), (200, "markov", 0.9, 1000, 200), (200, "minsep", 9, 1000, 200),
         (1000, "markov", 0.9, 1000, 60), (1000, "minsep", 9, 1000, 60)]
a_grid = np.linspace(0.2, 0.7, 11)   # true a = C/sigma = 0.5
out = open("step5_results.csv", "w", newline="")
w = csv.writer(out)
w.writerow(["T", "process", "param", "m", "R", "eps_true", "eps_sum", "eps_poisson", "estimator", "mean", "sd", "coverage", "flag_poisson", "ratio"])
t0 = time.time()
for (T, kind, par, m, R) in cells:
    P, pi, emit = iid_chain(q) if kind == "iid" else (two_state_chain(q, par) if kind == "markov" else minsep_chain(q, int(par)))
    pn = count_distribution(P, pi, emit, T)
    ef, es, ep = refs[(T, kind, float(par))]
    table = PathAwareTable(P, pi, emit, T, a_grid, DELTA, n=100_000 if T == 200 else 40_000, seed=1)
    print(f"table T={T} {kind} {par}: eps(a) = {np.round(table.eps, 2)} (true a=0.5 -> {table(0.5):.2f}; MC ref {ef:.2f}) [{time.time()-t0:.0f}s]", flush=True)
    res = {"gauss_sum": [], "count_cond_known": [], "path_aware": [], "lr_threshold": []}
    for rep in range(R):
        present = rng.random(m) < 0.5
        B = sample_paths(P, pi, emit, T, m, rng)
        counts = B.sum(axis=1)
        X = (B * present[:, None]) * C + sigma * rng.standard_normal((m, T))
        res["gauss_sum"].append(gauss_sum_audit(X, present, DELTA, ALPHA))
        res["count_cond_known"].append(count_cond_audit(X, present, counts, pn, DELTA, ALPHA))
        res["path_aware"].append(path_aware_audit(X, present, counts, T, table, ALPHA))
        res["lr_threshold"].append(lr_threshold_audit(X, present, P, pi, emit, C, sigma, DELTA, ALPHA))
    for name, v in res.items():
        v = np.array(v)
        w.writerow([T, kind, par, m, R, f"{ef:.3f}", f"{es:.3f}", f"{ep:.3f}", name, f"{v.mean():.3f}", f"{v.std():.3f}", f"{np.mean(v <= ef):.3f}", f"{np.mean(v > ep):.3f}", f"{v.mean()/ef:.3f}"])
        print(f"T={T} {kind} {par}: true={ef:.2f} poisson={ep:.2f} | {name:17s} {v.mean():6.2f}±{v.std():.2f} cov={np.mean(v<=ef):.3f} flag={np.mean(v>ep):.3f} ratio={v.mean()/ef:.2f} [{time.time()-t0:.0f}s]", flush=True)
    out.flush()
out.close()
