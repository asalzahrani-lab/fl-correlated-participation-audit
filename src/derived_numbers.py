"""Recompute the derived numbers quoted in the text from the result files (see PROVENANCE.md).

Prints: the Gaussian-pair eps without subsampling (50.5); the thinning identity D_N = (1-s) + s D_{N^A} against the
measured policy table; the OLS slope/intercept/R^2 of D against T for the two horizon runs and Var(p)/E(p) from the
tail file; the far-tail thresholds eps_0 = log L_p(cT) of the threshold-criterion Proposition (full version: Proposition 4)
for the Binomial law and the lambda = 0.9 chain; the aggregate audit counts of the revised version (1,320 simulated,
274 training runs, 1,594 certifications on 994 independent audit data sets, violated claims flagged in 1,416 of 1,417 certifications (816 of 817 data sets), 0 of 177 correct claims flagged; the cyclic-law audits of rev_round10_cyclic_results.csv included) from the
audit result files; and the i.i.d. Monte-Carlo checks.
Run from src/ after the result CSVs are in place (they are copied in ../results/)."""
import os, csv, sys
import numpy as np
from scipy.stats import binom
from corraudit import two_state_chain, count_distribution

RES = os.environ.get("P3_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
def rows(name): return list(csv.DictReader(open(os.path.join(RES, name))))

# 1. eps of the Gaussian pair with shift a*sqrt(T), no subsampling (Section I / III-B: 50.5)
from vecgauss import eps_gauss_pair_vec
T, a, delta = 200, 0.5, 1e-4
print(f"Gaussian pair, shift {a}*sqrt({T}) = {a*np.sqrt(T):.3f}: eps({delta}) = {float(eps_gauss_pair_vec(0.0, 1.0, a*np.sqrt(T), 1.0, delta)):.2f}")

# 2. thinning identity (Section VI-B, eq. 18)
pol = {r["policy"]: r for r in rows("step21b_policies.csv")}
D_all = float(pol["thinning 1.00"]["D"]) if "thinning 1.00" in pol else float(pol["take all available"]["D"])
for s in (0.05, 0.20, 0.50):
    key = [k for k in pol if k.startswith("thinning") and abs(float(k.split()[1]) - s) < 1e-9][0]
    print(f"thinning s={s:.2f}: identity predicts D = {(1-s) + s*D_all:.2f}, measured {float(pol[key]['D']):.2f}")

# 3. OLS of D on T for the two horizon runs (Section VI-C)
hz = rows("step21_horizon.csv")
for rm in ("10", "5"):
    sub = [r for r in hz if r["round_min"] == rm]
    Tv = np.array([float(r["T"]) for r in sub]); Dv = np.array([float(r["D"]) for r in sub])
    sl, ic = np.polyfit(Tv, Dv, 1); resid = Dv - (sl*Tv + ic)
    print(f"{rm}-min run: D = {sl:.3f} T + {ic:.1f}, R^2 = {1 - resid.var()/Dv.var():.4f}")
tails = rows("step21b_tails.csv")
for rm in ("10", "5"):
    v = sorted({r["VarP_over_EP"] for r in tails if r["round_min"] == rm})
    print(f"{rm}-min run: Var(p_i)/E[p_i] = {', '.join(v)}")

# 4. far-tail thresholds eps_0 = log L_p(cT), c = a/sqrt(T) (condition s* >= cT of the threshold-criterion Proposition)
def eps0(pn, T, a):
    c = a/np.sqrt(T); x = c*T; n = np.arange(T+1)
    return float(np.log(np.sum(pn*np.exp(c*n*x - (c*n)**2/2))))
T, q, a = 200, 0.1, 0.5
pn_bin = binom.pmf(np.arange(T+1), T, q)
P, pi, emit = two_state_chain(q, 0.9); pn_chain = count_distribution(P, pi, emit, T)
print(f"far-tail threshold eps_0: Binomial {eps0(pn_bin, T, a):.2f}, chain lambda=0.9 {eps0(pn_chain, T, a):.1f}")

# 5. aggregate audit counts
sim = rows("step19_results.csv")
cert = [r for r in sim if r["estimator"].startswith("cert")]
cert += [dict(r, law="cyclic") for r in rows("rev_round10_cyclic_results.csv") if r["estimator"].startswith("cert")]   # tenth round
n_corr = sum(int(r["R"]) for r in cert if r["law"] != "iid")
n_iid = sum(int(r["R"]) for r in cert if r["law"] == "iid")
above = sum(int(r["n_above_reference"]) for r in cert)
flags = sum(int(r["n_above_poisson"]) for r in cert if r["law"] != "iid")
false = sum(int(r["n_above_poisson"]) for r in cert if r["law"] == "iid")
for f in ("step16_results_markov_T1000.csv", "step16_results_trace_T1000.csv"):
    for r in rows(f):
        if r["estimator"].startswith("cert"):
            n_corr += int(r["R"]); above += int(r["n_above_reference"]); flags += int(r["n_above_poisson"])
print(f"simulated certified audits on correlated laws: {n_corr} (flags {flags}), i.i.d. control: {n_iid} (false flags {false}), above reference: {above}")
tr = rows("fl_mlp2_results.csv") + rows("fl_mlp3_results.csv") + rows("fl_mlp_5k_results.csv")
# NB: the column eps_true_model1 of these files holds the in-script reference of the training scripts (12.08, an
# earlier 2M-sample value); the paper's reference is the 10M-sample 11.869 (Markov) and the exact PLD 2.866 (i.i.d.).
REF = {"markov": 11.869, "iid": 2.866}; POIS = 2.866
mk = [r for r in tr if r["process"] == "markov"]; ii = [r for r in tr if r["process"] == "iid"]
fl_flags = sum(float(r["merged_certified"]) > POIS for r in mk)
fl_above = sum(float(r["merged_certified"]) > REF[r["process"]] for r in tr)
fl_false = sum(float(r["merged_certified"]) > POIS for r in ii)
# revision runs with orthogonal canaries (rev_train_*.csv)
rv = []
for f in ("rev_train_base_orth.csv", "rev_train_base_rand.csv", "rev_train_ratio_1to2.csv", "rev_train_ratio_1to10.csv",
          "rev_train_medium.csv", "rev_train_medium_iid.csv"):
    rv += rows(f)
rmk = [r for r in rv if r["law"] == "markov"]; rii = [r for r in rv if r["law"] == "iid"]
rv_flags = sum(int(r["flag"]) for r in rmk); rv_false = sum(int(r["flag"]) for r in rii); rv_above = sum(int(r["above_ref"]) for r in rv)
rv_cover = sum(int(r["covered"]) for r in rv)
print(f"training runs (first set): {len(tr)} ({len(mk)} Markov: flags {fl_flags}; {len(ii)} i.i.d.: false flags {fl_false}); above reference {fl_above}")
print(f"training runs (orthogonal canaries): {len(rv)} ({len(rmk)} Markov: flags {rv_flags}; {len(rii)} i.i.d.: false flags {rv_false}); above reference {rv_above}; a_lo <= a in {rv_cover}")
n_tr = len(tr) + len(rv); n_mk = len(mk) + len(rmk); n_ii = len(ii) + len(rii)
print(f"totals: simulated {n_corr + n_iid}, training runs {n_tr}, all audits {n_corr + n_iid + n_tr}, above reference {above + fl_above + rv_above}, "
      f"violated claims flagged {flags + fl_flags + rv_flags} of {n_corr + n_mk}, correct claims flagged {false + fl_false + rv_false} of {n_iid + n_ii}")
# 5b. independent audit data sets: every simulated data set on a correlated law is certified at both delta (one audit
# stream per law and horizon), so the delta = 1e-5 evaluations re-use the delta = 1e-4 data; training runs and the
# i.i.d. control are evaluated once.
c4 = [r for r in cert if float(r["delta"]) == 1e-4]
s_corr = sum(int(r["R"]) for r in c4 if r["law"] != "iid"); s_flags = sum(int(r["n_above_poisson"]) for r in c4 if r["law"] != "iid")
s_above_any = above   # 0 at both deltas
for f in ("step16_results_markov_T1000.csv", "step16_results_trace_T1000.csv"):
    for r in rows(f):
        if r["estimator"].startswith("cert") and float(r["delta"]) == 1e-4:
            s_corr += int(r["R"]); s_flags += int(r["n_above_poisson"])
print(f"independent audit data sets: {s_corr + n_iid + n_tr} (simulated {s_corr} on correlated laws + {n_iid} i.i.d., training {n_tr}); "
      f"violated claims flagged {s_flags + fl_flags + rv_flags} of {s_corr + n_mk}; correct claims flagged {false + fl_false + rv_false} of {n_iid + n_ii}; "
      f"above the reference at either delta: {s_above_any + fl_above + rv_above}")

# 6. i.i.d. Monte-Carlo checks against the PLD accountant
for r in rows("theory_perround_bound.csv"):
    if r["process"] == "iid":
        print(f"T={r['T']} i.i.d.: Monte-Carlo {float(r['eps_true_mc']):.3f} vs PLD {float(r['eps_poisson']):.3f} ({100*(float(r['eps_true_mc'])/float(r['eps_poisson'])-1):+.1f}%)")
