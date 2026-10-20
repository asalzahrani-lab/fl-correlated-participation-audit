"""Step 19: consolidated audit tables for the manuscript (after the Section-5 review).
One reference eps per law and delta = the largest-sample Monte-Carlo value, with its (delta-propagated) standard error;
corrected confidence limits (alpha/2 per one-sided limit); all baselines in the same runs; the cyclic law included.
Tables: KL-certified entries read from step14/step16 CSVs (each entry individually certified, so mixing sample sizes is
valid); references: Markov 1e-4 11.869 / 1e-5 15.053 (10M), trace 1e-4 10.685 / 1e-5 14.975 (3M), i.i.d. exact PLD.
Outputs step19_results.csv (one row per law x delta x auditor) and step19_audits.npz (per-audit values)."""
import numpy as np, csv, json, sys, time
import os; sys.path.insert(0, os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")))
from scipy.stats import binom
from corraudit import two_state_chain, iid_chain, minsep_chain, sample_paths, count_distribution, eps_poisson_pld, eps_sum_exact, eps_from_delta_curve
from auditors2 import gauss_sum_audit, count_cond_audit
from certified import EmpiricalPathLaw, ChainLaw
from certified2 import snr_lower_bound

T, a_true, m, R = 200, 0.5, 1000, 120
A_EST, A_V2 = 0.045, 0.05
rng = np.random.default_rng(1919)
t0 = time.time(); log = lambda s: print(f"{s} [{time.time()-t0:.0f}s]", flush=True)


def read_table(path, col_plain, col_cert):
    rows = list(csv.DictReader(open(path)))
    a = np.array([float(r["a"]) for r in rows]); return a, np.array([float(r[col_plain]) for r in rows]), np.array([float(r[col_cert]) for r in rows])


class Table:
    def __init__(self, parts):
        """parts: list of (a_grid, plain, cert); entries merged on a (later parts override earlier ones)."""
        d = {}
        for a, p, c in parts:
            for ai, pi_, ci in zip(a, p, c): d[round(float(ai), 2)] = (pi_, ci)
        self.a = np.array(sorted(d)); self.plain = np.array([d[k][0] for k in self.a]); self.cert = np.array([d[k][1] for k in self.a])

    def cert_floor(self, x):
        i = np.searchsorted(self.a - 1e-9, x, side="right") - 1
        return 0.0 if i < 0 else float(self.cert[i])

    def plain_interp(self, x):
        return float(np.interp(x, self.a, self.plain, left=0.0))


# ---------- tables
t14m = read_table("step14_table_markov.csv", "eps_plain", "eps_cert_bern_0.005")
t16m4 = read_table("step16_table_markov.csv", "eps_plain_1e-4", "eps_cert_1e-4"); t16m5 = read_table("step16_table_markov.csv", "eps_plain_1e-5", "eps_cert_1e-5")
t14i = read_table("step14_table_iid.csv", "eps_plain", "eps_cert_bern_0.005")
t14t = read_table("step14_table_trace.csv", "eps_plain", "eps_cert_bern_0.005")
t16t4 = read_table("step16_table_trace.csv", "eps_plain_1e-4", "eps_cert_1e-4"); t16t5 = read_table("step16_table_trace.csv", "eps_plain_1e-5", "eps_cert_1e-5")
tables = {("markov", 1e-4): Table([t14m, t16m4]), ("markov", 1e-5): Table([t16m5]),
          ("iid", 1e-4): Table([t14i]),
          ("trace", 1e-4): Table([t14t, t16t4]), ("trace", 1e-5): Table([t16t5])}
# references (largest-n Monte Carlo at a = 0.5) and their standard errors propagated from the delta estimate:
# se(eps) ~ (d eps / d ln delta) * sqrt((1 - delta) / (n delta)), slope from the two-delta tables
def ref_and_se(eps4, eps5, delta, n):
    slope = (eps5 - eps4) / np.log(10)                       # d eps / d ln delta  (magnitude)
    return slope * np.sqrt((1 - delta) / (n * delta))
refs = {("markov", 1e-4): (11.869, ref_and_se(11.869, 15.053, 1e-4, 10_000_000), "10M"),
        ("markov", 1e-5): (15.053, ref_and_se(11.869, 15.053, 1e-5, 10_000_000), "10M"),
        ("trace", 1e-4): (10.685, ref_and_se(10.685, 14.975, 1e-4, 3_000_000), "3M"),
        ("trace", 1e-5): (14.975, ref_and_se(10.685, 14.975, 1e-5, 3_000_000), "3M"),
        ("iid", 1e-4): (eps_poisson_pld(0.1, T, 1.0, 1.0 / a_true, 1e-4), 0.0, "exact (PLD)")}
for k, v in refs.items(): log(f"reference {k}: {v[0]:.3f} +- {v[1]:.3f} ({v[2]})")

# ---------- laws
P_m, pi_m, e_m = two_state_chain(0.1, 0.9); P_i, pi_i, e_i = iid_chain(0.1); P_c, pi_c, e_c = minsep_chain(0.1, 9)
pn = {"markov": count_distribution(P_m, pi_m, e_m, T), "iid": count_distribution(P_i, pi_i, e_i, T), "cyclic": count_distribution(P_c, pi_c, e_c, T)}
laws = {"markov": ChainLaw(P_m, pi_m, e_m, T), "iid": ChainLaw(P_i, pi_i, e_i, T), "cyclic": ChainLaw(P_c, pi_c, e_c, T)}
from trace_analysis import ready_intervals, availability_grid
d = json.load(open(os.path.join(os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")), "state_traces.json")))
devs = []
for k, v in d.items():
    iv, tf, tl = ready_intervals(v["messages"])
    if tf is None or tl - tf < 2 * 86400: continue
    devs.append((k, iv, tf, tl))
step = 600; t0_ = min(tf for (_, _, tf, _) in devs); t1_ = max(tl for (_, _, _, tl) in devs)
times = np.arange(t0_, t1_, step); Rn = len(times)
Amat = np.zeros((len(devs), Rn), dtype=bool); cover = np.zeros((len(devs), Rn), dtype=bool)
for i, (k, iv, tf, tl) in enumerate(devs):
    Amat[i] = availability_grid(iv, t0_, t1_, step); cover[i] = (times >= tf) & (times <= tl)
W = 3 * 144; best = max(range(0, Rn - W, 36), key=lambda s0: (cover[:, s0:s0 + W].all(axis=1)).sum())
sel = cover[:, best:best + W].all(axis=1); A = Amat[sel, best:best + W]; N, Tw = A.shape
rng_pol = np.random.default_rng(0); Bpol = np.zeros_like(A); Epool = A.sum(axis=0).mean(); K = int(round(0.6 * Epool))
for t in range(Tw):
    idx = np.flatnonzero(A[:, t]); chosen = idx if len(idx) <= K else rng_pol.choice(idx, K, replace=False); Bpol[chosen, t] = True
paths = np.array([Bpol[i, s0:s0 + T] for i in range(N) for s0 in range(0, Tw - T + 1, 10)], dtype=np.int8)
laws["trace"] = EmpiricalPathLaw(paths); pn["trace"] = np.bincount(paths.sum(axis=1), minlength=T + 1) / len(paths)
q_trace = paths.mean()
poisson = {"markov": eps_poisson_pld(0.1, T, 1.0, 2.0, 1e-4), "iid": eps_poisson_pld(0.1, T, 1.0, 2.0, 1e-4), "cyclic": eps_poisson_pld(0.1, T, 1.0, 2.0, 1e-4), "trace": eps_poisson_pld(q_trace, T, 1.0, 2.0, 1e-4)}
poisson5 = {"markov": eps_poisson_pld(0.1, T, 1.0, 2.0, 1e-5), "trace": eps_poisson_pld(q_trace, T, 1.0, 2.0, 1e-5)}
# cyclic per-round reference (from pilot_a, 300k MC) and its final-model exact value
cyc_ref = None
for r in csv.DictReader(open("pilot_a_results.csv")):
    if r["process"] == "minsep" and float(r["param"]) == 9 and float(r["delta"]) == 1e-4 and int(r["T"]) == T: cyc_ref = float(r["eps_full_mc"])
log(f"cyclic per-round reference (pilot_a, 300k): {cyc_ref}; final-model exact {eps_sum_exact(pn['cyclic'], 1.0, 2.0, T, 1e-4):.3f}")

out = []; store = {}
for law_name in ["markov", "iid", "trace", "cyclic"]:
    law = laws[law_name]
    res = {k: [] for k in ["cert_1e-4", "v2_1e-4", "cert_1e-5", "v2_1e-5", "gauss_1e-4", "countcond_1e-4", "alo"]}
    for rep in range(R):
        present = rng.random(m) < 0.5
        B = law.sample(m, rng)
        X = (B * present[:, None]) * a_true + rng.standard_normal((m, T)).astype(np.float32)
        counts = B.sum(axis=1)
        if law_name != "cyclic":
            a_c = snr_lower_bound(X, present, B, A_EST); a_v = snr_lower_bound(X, present, B, A_V2); res["alo"].append(a_c)
            tb = tables[(law_name, 1e-4)]; res["cert_1e-4"].append(tb.cert_floor(a_c)); res["v2_1e-4"].append(tb.plain_interp(a_v))
            if (law_name, 1e-5) in tables:
                tb5 = tables[(law_name, 1e-5)]; res["cert_1e-5"].append(tb5.cert_floor(a_c)); res["v2_1e-5"].append(tb5.plain_interp(a_v))
        res["gauss_1e-4"].append(gauss_sum_audit(X.astype(float), present, 1e-4, 0.05))
        res["countcond_1e-4"].append(count_cond_audit(X.astype(float), present, counts, pn[law_name], 1e-4, 0.05))
    for est, v in res.items():
        if not v or est == "alo": continue
        v = np.array(v); delta = 1e-5 if est.endswith("1e-5") else 1e-4
        if law_name == "cyclic": ref, se, nref = cyc_ref, float("nan"), "300k"
        else: ref, se, nref = refs[(law_name, delta)]
        ep = poisson5[law_name] if delta == 1e-5 else poisson[law_name]
        store[f"{law_name}_{est}"] = v
        out.append([law_name, f"{delta:g}", est, R, f"{v.mean():.4f}", f"{v.std():.4f}", f"{v.min():.4f}", f"{v.max():.4f}", f"{ref:.4f}", f"{se:.4f}", nref, f"{ep:.4f}", f"{v.mean()/ref:.4f}", int(np.sum(v > ref)), int(np.sum(v > ep))])
        log(f"{law_name:7s} {est:15s}: {v.mean():.3f}+-{v.std():.3f} [{v.min():.2f},{v.max():.2f}] ref {ref:.3f}+-{se:.3f} recovery {v.mean()/ref:.3f} false {int(np.sum(v > ref))}/{R} flags {int(np.sum(v > ep))}/{R}")
    if res["alo"]:
        al = np.array(res["alo"]); log(f"{law_name}: a_lo(0.045) mean {al.mean():.4f} sd {al.std():.4f} min {al.min():.4f}; relative to a: {1 - al.mean()/a_true:.3f} below")
with open("step19_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["law", "delta", "estimator", "R", "mean", "sd", "min", "max", "reference", "reference_se", "reference_samples", "poisson", "recovery", "n_above_reference", "n_above_poisson"]); w.writerows(out)
np.savez("step19_audits.npz", **store)
log("done")
