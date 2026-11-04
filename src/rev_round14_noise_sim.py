"""Round 14: what the audit adds to accounting from the logs -- detection of an implementation noise error (simulation).

The accountant reads the participation law and the configuration (sigma = 2, C = 1, T = 200, delta = 1e-4) and reports
the correct value for the canaries' law at the nominal noise: 11.869 for the Markov law of Section V (the 10M-sample
reference of Table II), 2.866 for i.i.d. participation (exact PLD), 10.69 for the trace law (Table II).  The server
actually adds noise of scale r times the nominal, r in {1.10, 1.00, 0.98, 0.95, 0.90, 0.80}: at r = 1 the claim is true,
at r > 1 conservative, at r < 1 the realised loss is eps(a / r) with a = 0.5.  The accountant's number does not move
with r; the audit estimates the signal-to-noise ratio from the released aggregates and maps it through the certified
table of the law (the merged tables of Table III, rev_round14_common.py), so its report does.  120 audits per cell (m = 1,000 canaries; and m = 200 for
the Markov law), alpha_est = 0.045, alpha_mc = 0.005, seed 14500 + cell.  Reported: certified mean +- sd, the realised
loss at r (the table's plain Monte-Carlo value at a / r, or a 400k-sample estimate beyond the grid; PLD for i.i.d.),
recovery against it, flags of the claim (false alarms at r >= 1, power at r < 1), exceedances of the realised loss,
exceedances of the NOMINAL reference's upper 99% limit (above_nominal_upper99: 12.033 for the Markov law, the claim
itself for the trace law, empty for i.i.d.; at r < 1 this is not an error count, since the realised loss lies above it),
and how many certificates sit at the grid's last entry (a / r above the grid is floored to it, a valid, conservative
certificate).  The 400k-sample estimates beyond the grid carry a sampling error of about 0.1 (the two Markov cells at
r = 0.9 give 15.02 and 14.88 for the same eps(0.5556)).

    cd results && P3_TRACE=../trace python3 ../src/rev_round14_noise_sim.py        # about 15 minutes
Output rev_round14_noise_sim.csv.
"""
import csv, time, os
import numpy as np
from corraudit import two_state_chain, iid_chain, eps_poisson_pld
from certified import ChainLaw, EmpiricalPathLaw
from certified2 import snr_lower_bound, plain_eps
from rev_common import window, apply_policy
from rev_round14_common import paper_table

T, A_TRUE, DELTA, Q, R = 200, 0.5, 1e-4, 0.1, 120
A_EST = 0.045
RS = [1.10, 1.00, 0.98, 0.95, 0.90, 0.80]
CLAIM = {"markov": 11.869, "iid": 2.866, "trace": 10.69}
UPPER = {"markov": 12.033, "iid": None, "trace": 10.69}     # 99% upper limit of the independent re-run (Table II); trace re-run 10.42 [10.17, 10.69]
t0 = time.time()
log = lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True)



def plain_mc(L, a, n, rng, chunk=20_000):
    lp = np.empty(n); lq = np.empty(n)
    for s in range(0, n, chunk):
        k = min(chunk, n - s); B = L.sample(k, rng).astype(np.float64)
        lp[s:s + k] = L.loglr(a * B + rng.standard_normal((k, T)), a); lq[s:s + k] = L.loglr(rng.standard_normal((k, T)), a)
    return plain_eps(lp, lq, DELTA)


if __name__ == "__main__":
    laws = {}
    P, pi, emit = two_state_chain(Q, 0.9); laws["markov"] = ChainLaw(P, pi, emit, T)
    P, pi, emit = iid_chain(Q); laws["iid"] = ChainLaw(P, pi, emit, T)
    A, _, _ = window(10, 3); Bt = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0))
    starts = np.arange(0, Bt.shape[1] - T + 1, 10); W = np.concatenate([Bt[:, s:s + T] for s in starts], axis=0).astype(np.int8)
    laws["trace"] = EmpiricalPathLaw(W); q_trace = float(W.mean())
    tabs = {k: paper_table(k) for k in laws}
    rows = []; cell = 0
    for name in ("markov", "iid", "trace"):
        for m in ((1000, 200) if name == "markov" else (1000,)):
            for r in RS:
                L, tab = laws[name], tabs[name]; a_eff = A_TRUE / r
                if name == "iid": true_r = eps_poisson_pld(Q, T, 1.0, 1.0 / a_eff, DELTA)
                elif a_eff <= tab.a[-1] + 1e-9: true_r = tab.plain(a_eff)
                else: true_r = plain_mc(L, a_eff, 400_000, np.random.default_rng(14600 + cell)); log(f"  {name}: plain eps at a={a_eff:.4f} from 400k samples: {true_r:.3f}")
                rng = np.random.default_rng(14500 + cell); cell += 1; ests = []; alos = []; capped = 0
                for _ in range(R):
                    present = rng.random(m) < 0.5; B = L.sample(m, rng)
                    X = (B * present[:, None]) * A_TRUE + r * rng.standard_normal((m, T)).astype(np.float32)
                    alo = snr_lower_bound(X, present, B, A_EST); alos.append(alo); ests.append(tab.cert(alo)); capped += alo >= tab.a[-1]
                e = np.array(ests); claim = CLAIM[name]; up = UPPER[name]
                row = [name, m, r, f"{a_eff:.4f}", R, f"{claim:.3f}", f"{true_r:.3f}", f"{e.mean():.4f}", f"{e.std():.4f}", f"{e.mean() / true_r:.4f}",
                       int((e > claim).sum()), int((e > true_r).sum()), int((e > up).sum()) if up else "", capped, f"{np.mean(alos):.4f}"]
                rows.append(row)
                log(f"{name} m={m} r={r}: a/r={a_eff:.3f}; realised loss {true_r:.2f}; certified {e.mean():.2f} +- {e.std():.2f} ({e.mean() / true_r:.1%}); "
                    f"flags of the claim {claim}: {int((e > claim).sum())}/{R}; above the realised loss {int((e > true_r).sum())}; at the grid cap {capped}")
    with open("rev_round14_noise_sim.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["law", "m", "noise_r", "a_over_r", "R", "claim", "realised_loss", "mean_cert", "sd_cert", "recovery", "flags", "above_realised", "above_nominal_upper99", "at_grid_cap", "a_lo_mean"]); w.writerows(rows)
    log("done")
