"""Step 7b: real training with m = 2000 canaries so that the nonparametric (proof-grade) audit has enough power."""
import numpy as np, csv, time, sys
from corraudit import *
from auditors2 import *
from theory import eps_jensen_full
from fl_sim import prepare, run_once, andrew_estimate

T, q, sigma_noise, C_clip = 200, 0.1, 2.0, 0.5
n_clients, m_canary, R = 200, 2000, int(sys.argv[1]) if len(sys.argv) > 1 else 10
DELTA, ALPHA = 1e-4, 0.05
rng = np.random.default_rng(777)
data = prepare(raw=True, null_from=150)
a_grid = np.linspace(0.2, 0.7, 11)
refs = {}
for r in csv.DictReader(open("pilot_a_results.csv")):
    if float(r["delta"]) == DELTA and int(r["T"]) == T:
        refs[(r["process"], float(r["param"]))] = (float(r["eps_full_mc"]), float(r["eps_poisson_pld"]))
out = open("fl_results_m2000.csv", "w", newline=""); w = csv.writer(out)
w.writerow(["process", "param", "rep", "acc", "absent_sd", "present_mean", "pred_present_mean", "eps_poisson", "eps_true_model1", "eps_jensen_ub",
            "gauss_sum", "count_cond_known", "path_aware", "lr_threshold", "andrew_est"])
t0 = time.time()
for kind, par in [("markov", 0.9), ("minsep", 9)]:
    P, pi, emit = two_state_chain(q, par) if kind == "markov" else minsep_chain(q, int(par))
    pn = count_distribution(P, pi, emit, T)
    table = PathAwareTable(P, pi, emit, T, a_grid, DELTA, n=100_000, seed=9)
    ef, ep = refs[(kind, float(par))]
    ub = eps_jensen_full(pn, 1.0, sigma_noise, DELTA)
    print(f"{kind} {par}: poisson={ep:.2f} model1-true={ef:.2f} jensen-ub={ub:.2f} [{time.time()-t0:.0f}s]", flush=True)
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
        w.writerow([kind, par, rep, f"{res['acc']:.4f}", f"{absent_sd:.3f}", f"{pres_mean:.3f}", f"{pred:.3f}", f"{ep:.3f}", f"{ef:.3f}", f"{ub:.3f}",
                    f"{g:.3f}", f"{cc:.3f}", f"{pa:.3f}", f"{lt:.3f}", f"{an:.3f}"])
        out.flush()
        print(f"  rep {rep}: acc={res['acc']:.3f} absent_sd={absent_sd:.2f} present_mean={pres_mean:.2f} (pred {pred:.2f}) | gauss={g:.2f} countcond={cc:.2f} path={pa:.2f} lr_thr={lt:.2f} andrew={an:.2f} [{time.time()-t0:.0f}s]", flush=True)
out.close()
