"""Reviewer check: re-derive every headline number of the paper from the stored outputs in ../results/ and print
PASS / FAIL for each.  No heavy computation is repeated (that is what the generating scripts are for); this script
checks that the numbers in the manuscripts (TIFS v16 main + supplement, extended v21) follow from the files that the
generating scripts wrote, and that the random streams that generate the certificate tables are disjoint from those
that generate the audit data.  Runs in a few seconds:  cd src && python verify_claims.py
Exit status 0 if every check passes."""
import csv, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("P3_RESULTS", os.path.join(HERE, "..", "results"))
LOGS = os.path.join(RES, "logs")
fails = 0


def rows(name):
    return list(csv.DictReader(open(os.path.join(RES, name))))


def one(name, **kw):
    out = [r for r in rows(name) if all(str(r[k]) == str(v) for k, v in kw.items())]
    assert len(out) == 1, (name, kw, len(out))
    return out[0]


def check(label, value, expected, tol, source):
    global fails
    ok = abs(value - expected) <= tol
    fails += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {value:.4g} (paper {expected:g}, tolerance {tol:g}) <- {source}")


def check_true(label, cond, source):
    global fails
    fails += not cond
    print(f"[{'PASS' if cond else 'FAIL'}] {label} <- {source}")


print("== 1. Pooled accounting on the trace (abstract, Sections I, VI-B)")
take_all = one("step21b_policies.csv", policy="take all available")
check("final-model ratio, take-all policy, 1e-4", float(take_all["ratio_1e-4"]), 9.4, 0.05, "step21b_policies.csv ratio_1e-4")
acc = one("step16_accounting_trace.csv", delta="0.0001")
check("retained per-round reference, running example, 1e-4 (3M samples)", float(acc["perround_reference_eps"]), 10.69, 0.005, "step16_accounting_trace.csv")
check("per-round ratio, estimate, 1e-4", float(acc["perround_reference_eps"]) / float(acc["poisson_perround_eps"]), 8.8, 0.05, "step16_accounting_trace.csv")
re8 = one("rev_refs_trace200.csv", delta="0.0001")
check_true("independent re-run uses 8M samples", int(re8["n_mc"]) == 8_000_000, "rev_refs_trace200.csv n_mc")
check("lower 99% limit of the ratio, 1e-4", float(re8["ci99_lo"]) / float(re8["eps_poisson"]), 8.35, 0.005, "rev_refs_trace200.csv ci99_lo / eps_poisson")
pol = one("step21b_policies.csv", policy="fixed cohort 0.6 x pool")
check("upper end (Theorem 2 on the count law) / Poisson, 1e-4", float(pol["thm3_bound_1e-4"]) / float(pol["perround_poisson_1e-4"]), 16.4, 0.05, "step21b_policies.csv thm3_bound_1e-4")
check("upper end / Poisson, 1e-5", float(pol["thm3_bound_1e-5"]) / float(pol["perround_poisson_1e-5"]), 16.7, 0.05, "step21b_policies.csv thm3_bound_1e-5")
st = one("rev_refs_stored.csv", law="trace_T1000", delta="0.0001")
check("T = 1,000 lower 99% limit of the ratio", float(st["ci99_lo"]) / float(st["eps_poisson"]), 17.07, 0.005, "rev_refs_stored.csv")
br = one("rev_t1000_bracket.csv", delta="0.0001")
check("T = 1,000 upper end / Poisson", float(br["ratio_upper"]), 20.8, 0.05, "rev_t1000_bracket.csv ratio_upper")
r200 = float(acc["perround_reference_eps"]) / float(acc["poisson_perround_eps"])
check("T = 1,000 vs T = 200 point estimates, 1e-4 (extended VI-F)", float(br["ratio_lower"]) / r200, 1.98, 0.006, "rev_t1000_bracket.csv / step16_accounting_trace.csv")
br5 = one("rev_t1000_bracket.csv", delta="1e-05"); acc5 = one("step16_accounting_trace.csv", delta="1e-05")
check("T = 1,000 vs T = 200 point estimates, 1e-5", float(br5["ratio_lower"]) / (float(acc5["perround_reference_eps"]) / float(acc5["poisson_perround_eps"])), 1.65, 0.006, "rev_t1000_bracket.csv / step16_accounting_trace.csv")
check("T = 1,000 vs T = 200 upper ends, 1e-4", float(br["ratio_upper"]) / (float(pol["thm3_bound_1e-4"]) / float(pol["perround_poisson_1e-4"])), 1.27, 0.006, "rev_t1000_bracket.csv / step21b_policies.csv")
check("T = 1,000 vs T = 200 upper ends, 1e-5", float(br5["ratio_upper"]) / (float(pol["thm3_bound_1e-5"]) / float(pol["perround_poisson_1e-5"])), 1.20, 0.006, "rev_t1000_bracket.csv / step21b_policies.csv")

print("== 2. Per-device bracket (abstract, Sections I, VI-F, Table VI)")
lo = one("rev_peruser_exact2_summary.csv", quantity="pr_lo", delta="0.0001")
hi = one("rev_peruser_exact2_summary.csv", quantity="thm3", delta="0.0001")
check("lower end, worst device (optimistic PLD, discretisation 1e-3)", float(lo["max"]), 26.99, 0.005, "rev_peruser_exact2_summary.csv pr_lo max")
check("lower end / claim, median device", float(lo["median_over_reference"]), 6.0, 0.05, "rev_peruser_exact2_summary.csv")
check("devices whose lower end exceeds the claim", float(lo["n_above_reference"]), 397, 0, "rev_peruser_exact2_summary.csv")
log = open(os.path.join(LOGS, "rev_peruser_exact2.log")).read()
m = re.search(r"worst device 335 at discretisation 1e-4: \[([\d.]+), ([\d.]+)\] at 1e-4", log)
check_true("worst device recomputed at discretisation 1e-4 is logged", m is not None, "logs/rev_peruser_exact2.log")
if m:
    check("worst device lower end at 1e-4 discretisation / claim", float(m.group(1)) / float(lo["reference"]), 22.2, 0.05, "logs/rev_peruser_exact2.log")
check("upper end, maximum over devices and windows", float(hi["max"]), 34.53, 0.005, "rev_peruser_exact2_summary.csv thm3 max")
check("upper end / claim", float(hi["max"]) / float(hi["reference"]), 28.3, 0.05, "rev_peruser_exact2_summary.csv")
dev = rows("rev_peruser_exact2_devices.csv")
check_true("window-search certificate holds for every device", all(r["certified"] == "1" for r in dev), "rev_peruser_exact2_devices.csv certified")
gap = max(float(r["pr_hi_0.0001"]) - float(r["pr_lo_0.0001"]) for r in dev)
check_true(f"pessimistic minus optimistic PLD at most 0.10 on every device (max {gap:.4f}; values stored to 4 decimals)", gap <= 0.1 + 1e-4, "rev_peruser_exact2_devices.csv pr_hi - pr_lo")
top = rows("rev_peruser_exact2_top20.csv")
check_true("all-window check of the 20 largest devices agrees with the certified search", len(top) == 20 and all(r["same"] == "1" for r in top), "rev_peruser_exact2_top20.csv same")
up = rows("rev_round5_upper_check.csv")
dmax = max(abs(float(r["upper_1e-4_independent"]) - float(r["upper_1e-4_published"])) for r in up)
check_true(f"independent recomputation of the upper end agrees on all {len(up)} devices to the 6 printed decimals (max printed difference {dmax:.1e})", dmax <= 1.5e-6 and len(up) == 402, "rev_round5_upper_check.csv")
mu = re.search(r"delta=0.0001: .* max \|independent - published\| over devices ([\d.e-]+)", open(os.path.join(LOGS, "rev_round5_upper_check.log")).read())
check("unrounded maximum difference, 1e-4 (paper: 2.5e-7)", float(mu.group(1)), 2.5e-7, 5e-9, "logs/rev_round5_upper_check.log")

print("== 3. Timers (Sections I, VI-F, VII; Sec. S-VI)")
sch = one("rev_timer_schema.csv", policy="fixed cohort 0.6, timer 20", delta="0.0001")
check("per-device schema bound, median (an upper bound)", float(sch["schema_eps_median"]), 3.21, 0.005, "rev_timer_schema.csv")
check("per-device schema bound, maximum", float(sch["schema_eps_max"]), 6.62, 0.005, "rev_timer_schema.csv")
check("median / claim", float(sch["median_ratio"]), 20, 0.5, "rev_timer_schema.csv")
check("maximum / claim", float(sch["max_ratio"]), 41, 0.5, "rev_timer_schema.csv")
check("devices whose schema count equals the realised count", float(sch["schema_count_equals_realised"]), 383, 0, "rev_timer_schema.csv")
check("largest pool over draws at a later round (K = 27)", float(sch["max_pool_t_ge2_over_draws"]), 28, 0, "rev_timer_schema.csv")
t20 = one("rev_refs_timer20.csv", delta="0.0001")
check("pooled per-round ratio under the 20-round timer", float(t20["ratio"]), 1.10, 0.005, "rev_refs_timer20.csv")
check("its lower 99% limit", float(t20["ratio_lo"]), 1.07, 0.005, "rev_refs_timer20.csv")
check("its upper 99% limit", float(t20["ratio_hi"]), 1.13, 0.005, "rev_refs_timer20.csv")

print("== 4. Audits (abstract, Sections V-E, V-F)")
sim = rows("step19_results.csv")
cyc = rows("rev_round10_cyclic_results.csv")          # tenth round: the certified audit on the cyclic law (Table III)
cert = [r for r in sim if r["estimator"].startswith("cert")] + [dict(r, law="cyclic") for r in cyc if r["estimator"].startswith("cert")]
t1k = [r for f in ("step16_results_markov_T1000.csv", "step16_results_trace_T1000.csv") for r in rows(f) if r["estimator"].startswith("cert")]
recs = [float(r["recovery"]) for r in cert + t1k]
check("lowest certified recovery", min(recs), 0.74, 0.005, "step19_results.csv, step16_results_*_T1000.csv")
check("highest certified recovery", max(recs), 0.90, 0.005, "step19_results.csv, step16_results_*_T1000.csv")
gauss = [float(r["recovery"]) for r in sim if r["estimator"].startswith("gauss") and r["law"] in ("markov", "trace")]
cyc_ref = float(cyc[0]["reference"])                   # retained 20M-sample reference of the cyclic law (7.9371)
gauss_cyc = [float(r["mean"]) / cyc_ref for r in sim if r["estimator"] == "gauss_1e-4" and r["law"] == "cyclic"]
check("Gaussian fit, lowest recovery on correlated laws (abstract: 3 %)", min(gauss + gauss_cyc), 0.03, 0.005, "step19_results.csv gauss")
check("Gaussian fit, highest recovery on the Markov and trace laws (Contribution 2: 15 %)", max(gauss), 0.15, 0.005, "step19_results.csv gauss")
check("Gaussian fit, recovery on the cyclic law (abstract: 22 %; Table III)", max(gauss_cyc), 0.215, 0.005, "step19_results.csv gauss (cyclic) / rev_round10_cyclic_results.csv reference")
n_corr = sum(int(r["R"]) for r in cert + t1k if r["law"] != "iid"); n_iid = sum(int(r["R"]) for r in cert if r["law"] == "iid")
above = sum(int(r["n_above_reference"]) for r in cert + t1k)
flags = sum(int(r["n_above_poisson"]) for r in cert + t1k if r["law"] != "iid")
false = sum(int(r["n_above_poisson"]) for r in cert if r["law"] == "iid")
tr = rows("fl_mlp2_results.csv") + rows("fl_mlp3_results.csv") + rows("fl_mlp_5k_results.csv")
REF = {"markov": 11.869, "iid": 2.866}; POIS = 2.866
fl_flags = sum(float(r["merged_certified"]) > POIS for r in tr if r["process"] == "markov")
fl_false = sum(float(r["merged_certified"]) > POIS for r in tr if r["process"] == "iid")
fl_above = sum(float(r["merged_certified"]) > REF[r["process"]] for r in tr)
rv = [r for f in ("rev_train_base_orth.csv", "rev_train_base_rand.csv", "rev_train_ratio_1to2.csv", "rev_train_ratio_1to10.csv",
                  "rev_train_medium.csv", "rev_train_medium_iid.csv") for r in rows(f)]
rv_flags = sum(int(r["flag"]) for r in rv if r["law"] == "markov"); rv_false = sum(int(r["flag"]) for r in rv if r["law"] == "iid")
rv_above = sum(int(r["above_ref"]) for r in rv)
n_mk = sum(r["process"] == "markov" for r in tr) + sum(r["law"] == "markov" for r in rv)
n_ii = sum(r["process"] == "iid" for r in tr) + sum(r["law"] == "iid" for r in rv)
check("training runs", len(tr) + len(rv), 274, 0, "fl_mlp*_results.csv, rev_train_*.csv")
check("all certified audit evaluations", n_corr + n_iid + len(tr) + len(rv), 1594, 0, "step19, rev_round10_cyclic, step16 T=1000, training files")
check("estimates above the reference", above + fl_above + rv_above, 0, 0, "same files")
check("violated claims flagged", flags + fl_flags + rv_flags, 1416, 0, f"same files (of {n_corr + n_mk})")
check("violated claims tested", n_corr + n_mk, 1417, 0, "same files")
check("correct claims flagged", false + fl_false + rv_false, 0, 0, f"same files (of {n_iid + n_ii})")
check("correct claims tested", n_iid + n_ii, 177, 0, "same files")
# independent data sets: each simulated data set on a correlated law is certified at both delta (one audit stream per
# law and horizon: step19_rerun_audits.py seed 1919, step20b_parts.py seeds 4041/4042), so count the delta = 1e-4 rows only
s_corr = sum(int(r["R"]) for r in cert if float(r["delta"]) == 1e-4 and r["law"] != "iid") + sum(int(r["R"]) for r in t1k if float(r["delta"]) == 1e-4)
s_flags = sum(int(r["n_above_poisson"]) for r in cert if float(r["delta"]) == 1e-4 and r["law"] != "iid") + sum(int(r["n_above_poisson"]) for r in t1k if float(r["delta"]) == 1e-4)
check("independent audit data sets (abstract, Sections I, V-E, VII)", s_corr + n_iid + len(tr) + len(rv), 994, 0, "same files, delta = 1e-4 rows")
check("violated claims flagged, independent data sets", s_flags + fl_flags + rv_flags, 816, 0, f"same files (of {s_corr + n_mk})")
check("violated claims tested, independent data sets", s_corr + n_mk, 817, 0, "same files")
ts_ = rows("rev_train_summary.csv")
check("training, lowest configuration-mean recovery (Section VII: 73 to 93%)", min(float(r["recovery"]) for r in ts_), 0.73, 0.005, "rev_train_summary.csv")
k5 = [float(r["merged_certified"]) / 2.866 for r in rows("fl_mlp_5k_results.csv") if r["process"] == "iid"]
check("training, highest configuration-mean recovery (5,000 canaries, i.i.d.)", sum(k5) / len(k5), 0.93, 0.0051, "fl_mlp_5k_results.csv")

print("== 5. Monte-Carlo provenance (Table II, Table S2)")
mk = one("step16_accounting_markov.csv", delta="0.0001")
check("retained Markov reference, 1e-4 (10M samples, step16_delta5.py)", float(mk["perround_reference_eps"]), 11.87, 0.005, "step16_accounting_markov.csv")
for f, n in (("rev_refs_markov200.csv", 10_000_000), ("rev_refs_trace200.csv", 8_000_000)):
    check_true(f"independent re-run {f} has {n:,} samples at both deltas", all(int(r["n_mc"]) == n for r in rows(f)), f)
src = {s: open(os.path.join(HERE, s)).read() for s in ("step14_merged.py", "step16_delta5.py", "step19_rerun_audits.py", "step20b_parts.py", "rev_refs.py")}
seed_of = lambda s, pat: {int(x) for x in re.findall(pat, src[s])}
table_seeds = seed_of("step14_merged.py", r'"(?:markov|trace|iid)": (\d+)(?![\d.])') | seed_of("step16_delta5.py", r"default_rng\((\d+) if") | seed_of("step16_delta5.py", r"else (\d+)\)")
audit_seeds = seed_of("step19_rerun_audits.py", r"(?m)^rng = np\.random\.default_rng\((\d+)\)")   # seed 0 elsewhere is the policy draw that defines the law, shared by design
t1k_table = seed_of("step20b_parts.py", r"seed = \((\d+) if") | seed_of("step20b_parts.py", r"else (\d+)\) \* 1000"); t1k_audit = seed_of("step20b_parts.py", r"default_rng\((\d+) if") | seed_of("step20b_parts.py", r'else (\d+)\)\n')
print(f"       certificate-table seeds (T=200): {sorted(table_seeds)}; audit-data seeds (T=200): {sorted(audit_seeds)}; "
      f"T=1,000 table seeds x1000+17T+part: {sorted(t1k_table)}; T=1,000 audit seeds: {sorted(t1k_audit)}")
check_true("audit data and certificate tables are drawn from disjoint seeds", audit_seeds.isdisjoint(table_seeds) and t1k_audit.isdisjoint(t1k_table) and len(audit_seeds) > 0 and len(t1k_audit) > 0,
           "step14_merged.py, step16_delta5.py, step19_rerun_audits.py, step20b_parts.py")

print("== 6. Fifth-round checks")
lg = open(os.path.join(LOGS, "rev_round5_checks.log")).read()
m1 = re.search(r"delta=0.0001: final-model eps, equal-dispersion law ([\d.]+); Binomial ([\d.]+)", lg)
check("equal-dispersion count law, final-model eps at 1e-4", float(m1.group(1)), 23.0, 0.05, "logs/rev_round5_checks.log")
check("Binomial(200, 0.1), final-model eps at 1e-4", float(m1.group(2)), 2.69, 0.005, "logs/rev_round5_checks.log")
m2 = re.search(r"eps=0.0: allocation H\(P\|\|Q\) ([\d.]+).*independent H\(P\|\|Q\) ([\d.]+)", lg)
check("random allocation, total variation (T = 2)", float(m2.group(1)), 0.1440, 0.00005, "logs/rev_round5_checks.log")
check("independent participation, total variation (T = 2)", float(m2.group(2)), 0.1398, 0.00005, "logs/rev_round5_checks.log")

print("== 7. Seventh-round checks (TIFS V-F, VI-G; extended V-F, VI-G)")
au = rows("rev_round7_audit_sens.csv")
pw = {float(r["lambda_canaries"]): r for r in au if r["mode"] == "power"}
for lam, ratio, flags, rec in ((0.8, 2.29, 120, 0.86), (0.6, 1.36, 120, 0.88), (0.3, 1.08, 11, 0.90)):
    r = pw[lam]
    check(f"power, lambda={lam}: true loss / Poisson claim (reference / 2.8662)", float(r["eps_true_table_law"]) / float(r["poisson"]), ratio, 0.005, "rev_round7_audit_sens.csv")
    check(f"power, lambda={lam}: flags of 120", int(r["flags"]), flags, 0, "rev_round7_audit_sens.csv")
    check(f"power, lambda={lam}: recovery", float(r["recovery"]), rec, 0.005, "rev_round7_audit_sens.csv")
    check(f"power, lambda={lam}: estimates above the reference's upper 99% limit", int(r["above_true"]), 0, 0, "rev_round7_audit_sens.csv")
ms = {float(r["lambda_table"]): r for r in au if r["mode"] == "misspec"}
for lt, above, rec in ((0.85, 0, 0.592), (0.89, 0, 0.775), (0.9, 1, 0.845), (0.91, 16, 0.923), (0.95, 120, 1.463)):
    r = ms[lt]
    check(f"misspecification, table lambda'={lt}: estimates above eps_true(0.9) = 11.869, of 120", int(r["above_true"]), above, 0, "rev_round7_audit_sens.csv")
    check(f"misspecification, table lambda'={lt}: mean estimate / 11.869", float(r["recovery"]), rec, 0.005, "rev_round7_audit_sens.csv")
import numpy as np
e9 = np.load(os.path.join(RES, "rev_round7_misspec_0.9_est.npy"))
check("misspecification control (table lambda' = 0.9): the one estimate above 11.869", float(e9.max()), 11.894, 0.0005, "rev_round7_misspec_0.9_est.npy")
check_true("... it lies below the upper 99% limit 12.033 of the 10M re-run", float(e9.max()) <= 12.033 and int((e9 > 11.869).sum()) == 1, "rev_round7_misspec_0.9_est.npy, rev_refs_markov200.csv")
for lam, mx in ((0.8, 6.086), (0.6, 3.672), (0.3, 2.914)):
    ep = np.load(os.path.join(RES, f"rev_round7_power_{lam}_est.npy"))
    check(f"power, lambda={lam}: largest of 120 estimates", float(ep.max()), mx, 0.0005, f"rev_round7_power_{lam}_est.npy")
    check_true(f"power, lambda={lam}: no estimate above the point reference", bool((ep <= float(pw[lam]["eps_true_table_law"])).all()), f"rev_round7_power_{lam}_est.npy")
ls = rows("rev_round7_lamse.csv")[0]
check("sd of the maximum-likelihood lambda (1,000 devices x 200 rounds)", float(ls["sd_hat"]), 0.0021, 0.0001, "rev_round7_lamse.csv")
check("99% range of the estimate, lower", float(ls["q005"]), 0.895, 0.0005, "rev_round7_lamse.csv")
check("99% range of the estimate, upper", float(ls["q995"]), 0.905, 0.0005, "rev_round7_lamse.csv")
ts = rows("rev_round7_trace_sens.csv")
rl = {int(r["round_min"]): r for r in ts if r["check"] == "round_length"}
for rmin, r4, r5 in ((5, 8.91, 10.28), (10, 5.16, 6.25), (30, 2.47, 2.99)):
    check(f"round length {rmin} min, final-model ratio 1e-4", float(rl[rmin]["fm_eps_1e-4"]) / float(rl[rmin]["fm_binom_1e-4"]), r4, 0.005, "rev_round7_trace_sens.csv")
    check(f"round length {rmin} min, final-model ratio 1e-5", float(rl[rmin]["fm_eps_1e-5"]) / float(rl[rmin]["fm_binom_1e-5"]), r5, 0.005, "rev_round7_trace_sens.csv")
import statistics
for n, med, lo, hi in ((286, 5.41, 3.72, 7.29), (458, 5.70, 4.97, 7.20)):
    v = [float(r["fm_ratio_1e-4"]) for r in ts if r["check"] == "device_subsample" and int(r["devices"]) == n]
    check(f"device subsample ({n} of 572, 20 subsets): number of subsets", len(v), 20, 0, "rev_round7_trace_sens.csv")
    check(f"device subsample ({n} of 572): median ratio 1e-4", statistics.median(v), med, 0.005, "rev_round7_trace_sens.csv")
    check(f"device subsample ({n} of 572): min ratio 1e-4", min(v), lo, 0.005, "rev_round7_trace_sens.csv")
    check(f"device subsample ({n} of 572): max ratio 1e-4", max(v), hi, 0.005, "rev_round7_trace_sens.csv")
s7 = open(os.path.join(HERE, "rev_round7_audit_sens.py")).read()
K7 = {float(k): int(v) for k, v in re.findall(r"([\d.]+): (\d+)", re.search(r"^K = \{([^}]*)\}", s7, re.M).group(1))}
base = lambda pat: int(re.search(pat, s7, re.S).group(1))
tab7 = {base(r"def table\(.*?default_rng\((\d{4}) \+ K") + k for k in K7.values()} | {base(r"def reference\(.*?default_rng\((\d{4}) \+ K") + k for k in K7.values()}
aud7 = {base(r"alo = audits\(lam, (\d{4}) \+ K") + K7[l] for l in (0.3, 0.6, 0.8)} | {int(x) for x in re.findall(r"audits\(0\.9, (\d+)\)", s7)}
print(f"       seventh-round table/reference seeds: {sorted(tab7)}; audit seeds: {sorted(aud7)}")
check_true("seventh-round audit data and certificate tables are drawn from disjoint seeds", tab7.isdisjoint(aud7) and len(aud7) == 4 and len(tab7) == 16,
           "rev_round7_audit_sens.py")

print("== 8. Ninth-round checks (certificate implementation, Section VI-E)")
s9 = open(os.path.join(HERE, "rev_peruser_exact2.py")).read()
check_true("inclusion rates are grouped exactly (np.unique on the values, no np.round)", "np.unique(r, return_counts=True)" in s9 and "np.round(r" not in s9, "rev_peruser_exact2.py")
check_true("optimistic PLD is composed with tail_mass_truncation=0 within and across rates", "self_compose(int(cnt), tail_mass_truncation=0.0) if not pessimistic" in s9 and "pld.compose(p, tail_mass_truncation=0.0) if not pessimistic" in s9, "rev_peruser_exact2.py")
cmp9 = {r[0]: r for r in csv.reader(open(os.path.join(RES, "rev_round9_compare.csv")))}
check("largest change of any per-device value against release v8 (1e-4 and 1e-5, both ends)", max(float(cmp9[c][1]) for c in ("pr_lo_0.0001", "pr_lo_1e-05", "pr_hi_0.0001", "pr_hi_1e-05")), 0, 1e-7, "rev_round9_compare.csv")
check_true("worst windows, refinement counts and certificates unchanged", cmp9["same_worst_window"][1] == cmp9["same_worst_window"][2] and cmp9["same_n_refined"][1] == cmp9["same_n_refined"][2] and cmp9["all_certified"][1] == "1", "rev_round9_compare.csv")
hi9 = one("rev_peruser_exact2_summary.csv", quantity="thm3", delta="0.0001")
check("upper end quoted as 28.3x (34.527 / 1.2189, not rounded down)", round(float(hi9["max"]) / float(hi9["reference"]), 1), 28.3, 0.001, "rev_peruser_exact2_summary.csv")

print("== 9. Tenth-round checks (smaller delta, cyclic audit, erasure, canaries)")
sd = {(r["law"], r["delta"]): r for r in rows("rev_round10_small_delta.csv")}
for law, d, col, val, tol in (("chain0.9_T200", "0.0001", "sum_stat_ratio", 3.09, 0.005), ("trace_T200", "0.0001", "sum_stat_ratio", 5.16, 0.005),
                              ("trace_T200", "0.0001", "thm2_over_poisson", 16.4, 0.05), ("chain0.9_T200", "1e-08", "sum_stat_ratio", 4.93, 0.005),
                              ("trace_T200", "1e-08", "sum_stat_ratio", 7.91, 0.005), ("chain0.9_T200", "1e-10", "sum_stat_ratio", 5.58, 0.005),
                              ("trace_T200", "1e-10", "sum_stat_ratio", 8.37, 0.005), ("trace_T1000", "0.0001", "thm2_over_poisson", 20.8, 0.05),
                              ("trace_T1000", "1e-10", "thm2_over_poisson", 17.6, 0.05)):
    check(f"{law} {col} at delta={d}", float(sd[(law, d)][col]), val, tol, "rev_round10_small_delta.csv")
DS = ("0.0001", "1e-05", "1e-06", "1e-08", "1e-10")
v = [float(sd[("chain0.9_T200", d)]["thm2_over_poisson"]) for d in DS]
check_true("chain, T = 200: Theorem 2 / Poisson rises at every step from 1e-4 to 1e-10", all(v[k + 1] > v[k] for k in range(4)), "rev_round10_small_delta.csv")
check("chain, T = 200: Theorem 2 / Poisson at 1e-4 (7.7; supplement 7.73)", v[0], 7.73, 0.005, "rev_round10_small_delta.csv")
check("chain, T = 200: Theorem 2 / Poisson at 1e-10 (8.6; supplement 8.64)", v[-1], 8.64, 0.005, "rev_round10_small_delta.csv")
v = [float(sd[("trace_T200", d)]["thm2_over_poisson"]) for d in DS]
check("trace, T = 200: smallest Theorem 2 / Poisson over 1e-4..1e-10 (16.3; supplement 16.27)", min(v), 16.27, 0.005, "rev_round10_small_delta.csv")
check("trace, T = 200: largest Theorem 2 / Poisson over 1e-4..1e-10 (16.7; supplement 16.74)", max(v), 16.74, 0.005, "rev_round10_small_delta.csv")
v = [float(sd[("trace_T1000", d)]["sum_stat_ratio"]) for d in DS]
check("trace, T = 1,000: final-model ratio at 1e-4", v[0], 7.31, 0.005, "rev_round10_small_delta.csv")
check("trace, T = 1,000: final-model ratio at 1e-8 (levels off at 7.9)", v[3], 7.87, 0.005, "rev_round10_small_delta.csv")
check("trace, T = 1,000: final-model ratio at 1e-10", v[4], 7.86, 0.005, "rev_round10_small_delta.csv")
check_true("numerical checks of the small-delta file (finer PLD, finer grid) below 1e-4", all(float(r["check_pld_finer"]) < 1e-4 and float(r["check_grid_finer"]) < 1e-4 for r in sd.values()),
           "rev_round10_small_delta.csv")
pol = rows("rev_round10_small_delta_policies.csv"); ds = ["0.0001", "1e-05", "1e-06", "1e-08", "1e-10"]
check_true("every policy's final-model ratio grows as delta falls from 1e-4 to 1e-10", all(all(float(r[f"sum_stat_ratio_{ds[k + 1]}"]) > float(r[f"sum_stat_ratio_{ds[k]}"]) for k in range(4)) for r in pol),
           "rev_round10_small_delta_policies.csv")
ta = [r for r in pol if r["policy"] == "take all available"][0]
check("take-all final-model ratio at 1e-10", float(ta["sum_stat_ratio_1e-10"]), 12.4, 0.05, "rev_round10_small_delta_policies.csv")
cr = {r["delta"]: r for r in rows("rev_round10_cyclic_reference.csv")}
check("cyclic per-round reference at 1e-4 (20M samples)", float(cr["0.0001"]["eps"]), 7.94, 0.005, "rev_round10_cyclic_reference.csv")
check("... lower 99% limit", float(cr["0.0001"]["lo99"]), 7.89, 0.005, "rev_round10_cyclic_reference.csv")
check("... upper 99% limit", float(cr["0.0001"]["hi99"]), 7.98, 0.005, "rev_round10_cyclic_reference.csv")
check("cyclic reference at 1e-5", float(cr["1e-05"]["eps"]), 9.19, 0.005, "rev_round10_cyclic_reference.csv")
check("... lower 99% limit at 1e-5", float(cr["1e-05"]["lo99"]), 9.07, 0.005, "rev_round10_cyclic_reference.csv")
check("... upper 99% limit at 1e-5", float(cr["1e-05"]["hi99"]), 9.33, 0.005, "rev_round10_cyclic_reference.csv")
check("cyclic per-round ratio at 1e-4 (falls ...)", float(cr["0.0001"]["ratio"]), 2.77, 0.005, "rev_round10_cyclic_reference.csv")
check("... to 1e-5", float(cr["1e-05"]["ratio"]), 2.74, 0.005, "rev_round10_cyclic_reference.csv")
cy = {r["estimator"]: r for r in cyc}
check("cyclic certified audit, mean at 1e-4", float(cy["cert_1e-4"]["mean"]), 7.16, 0.005, "rev_round10_cyclic_results.csv")
check("cyclic certified audit, recovery at 1e-4", float(cy["cert_1e-4"]["recovery"]), 0.90, 0.005, "rev_round10_cyclic_results.csv")
check("cyclic certified audit, flags of 120 at 1e-4 and 1e-5", int(cy["cert_1e-4"]["n_above_poisson"]) + int(cy["cert_1e-5"]["n_above_poisson"]), 240, 0, "rev_round10_cyclic_results.csv")
check("cyclic certified audit, estimates above the reference", int(cy["cert_1e-4"]["n_above_reference"]) + int(cy["cert_1e-5"]["n_above_reference"]), 0, 0, "rev_round10_cyclic_results.csv")
unc = sum(int(r["n_above_reference"]) for r in sim if r["estimator"] == "v2_1e-4") + int(cy["v2_1e-4"]["n_above_reference"])
check("uncertified estimates above the reference at 1e-4, four laws (11 of 480)", unc, 11, 0, "step19_results.csv, rev_round10_cyclic_results.csv")
c1k = {r["delta"]: r for r in rows("rev_round10_cyclic_T1000.csv")}
check("cyclic per-round ratio at T = 1,000, 1e-4 (3.9 times)", float(c1k["0.0001"]["ratio"]), 3.9, 0.05, "rev_round10_cyclic_T1000.csv")
check("cyclic Theorem 2 (= schema) over the reference at T = 1,000", float(c1k["0.0001"]["thm2_over_reference"]), 1.08, 0.005, "rev_round10_cyclic_T1000.csv")
check("cyclic per-round ratio at T = 1,000, 1e-4 (falls ...)", float(c1k["0.0001"]["ratio"]), 3.85, 0.005, "rev_round10_cyclic_T1000.csv")
check("... to 1e-5", float(c1k["1e-05"]["ratio"]), 3.72, 0.005, "rev_round10_cyclic_T1000.csv")
pd = {(r["quantity"], r["delta"]): r for r in rows("rev_round10_peruser_summary.csv")}
for q, d, val in (("pr_lo", "1e-06", 19.0), ("thm3", "1e-06", 24.1), ("pr_lo", "1e-08", 17.5), ("thm3", "1e-08", 22.0)):
    check(f"per device, worst {q} / claim at {d}", float(pd[(q, d)]["max"]) / float(pd[(q, d)]["reference"]), val, 0.05, "rev_round10_peruser_summary.csv")
check("per device, worst upper end at 1e-8 (value)", float(pd[("thm3", "1e-08")]["max"]), 45.51, 0.005, "rev_round10_peruser_summary.csv")
check_true("largest lower end at 1e-6 and 1e-8 is device 335, largest upper end device 176 (two different devices)",
           all(pd[("pr_lo", d)]["argmax_device"] == "335" and pd[("thm3", d)]["argmax_device"] == "176" for d in ("1e-06", "1e-08")), "rev_round10_peruser_summary.csv")
check("per device, worst lower end at 1e-8 (value)", float(pd[("pr_lo", "1e-08")]["max"]), 36.18, 0.005, "rev_round10_peruser_summary.csv")
for d, val in (("1e-06", 5.5), ("1e-08", 5.3)):
    check(f"per device, median lower end / claim at {d}", float(pd[("pr_lo", d)]["median_over_reference"]), val, 0.05, "rev_round10_peruser_summary.csv")
dv = rows("rev_round10_peruser_devices.csv")
check_true("small-delta window-search certificate holds for every device", all(r["certified"] == "1" for r in dv), "rev_round10_peruser_devices.csv certified")
t20 = rows("rev_round10_peruser_top20.csv")
check_true("small-delta all-window check of the 20 largest devices agrees with the certified search", len(t20) == 20 and all(r["same"] == "1" for r in t20), "rev_round10_peruser_top20.csv")
m10 = re.search(r"worst device 335 at discretisation 1e-4: \[([\d.]+), ([\d.]+)\] at 1e-6, \[([\d.]+), ([\d.]+)\] at 1e-8", open(os.path.join(LOGS, "rev_round10_peruser_delta.log")).read())
check_true("worst device recomputed at discretisation 1e-4 (small delta) is logged", m10 is not None, "logs/rev_round10_peruser_delta.log")
if m10:
    check("worst device at 1e-4 discretisation, 1e-8, over the claim", float(m10.group(3)) / float(pd[("pr_lo", "1e-08")]["reference"]), 17.5, 0.05, "logs/rev_round10_peruser_delta.log")
cn = {(r["law"], int(r["m"])): r for r in rows("rev_round10_canaries.csv")}
for law, recs in (("markov", (0.68, 0.77, 0.83, 0.86, 0.90, 0.92)), ("trace", (0.44, 0.54, 0.66, 0.74, 0.77, 0.83))):
    for m, val in zip((100, 200, 500, 1000, 2000, 5000), recs):
        check(f"canary sweep, {law}, m={m}: certified recovery", float(cn[(law, m)]["recovery"]), val, 0.005, "rev_round10_canaries.csv")
check("canary sweep: audits that do not flag the claim (of 1,440)", sum(120 - int(r["flags"]) for r in cn.values()), 1, 0, "rev_round10_canaries.csv")
check("canary sweep: estimates above the reference (trace, m = 100)", int(cn[("trace", 100)]["above_reference"]), 2, 0, "rev_round10_canaries.csv")
check("canary sweep: estimates above the reference elsewhere", sum(int(r["above_reference"]) for k, r in cn.items() if k != ("trace", 100)), 0, 0, "rev_round10_canaries.csv")
gr = [float(r["alo_gap_rel"]) / float(r["rule_gap_rel"]) for r in cn.values()]
check("canary sweep: smallest gap of a_lo over z/sqrt(n_obs)", min(gr), 1.0, 0.05, "rev_round10_canaries.csv")
check("canary sweep: largest gap of a_lo over z/sqrt(n_obs)", max(gr), 1.3, 0.05, "rev_round10_canaries.csv")
er = rows("rev_round10_erasure.csv")
for cfg, val, n in (("pca50_random", 0.50, 10), ("raw_lowvar", 0.95, 4)):
    v = [float(r["erasure_ratio"]) for r in er if r["config"] == cfg]
    check(f"erasure ratio, {cfg}: runs", len(v), n, 0, "rev_round10_erasure.csv")
    check(f"erasure ratio, {cfg}: mean", sum(v) / len(v), val, 0.005, "rev_round10_erasure.csv")

print("== 10. Eleventh-round checks (step-function column of Table III, accuracy controls)")
for law, n in (("markov", 9), ("trace", 5)):
    t = rows(f"rev_round11_step_table_{law}.csv")
    check_true(f"step-16 replay, {law}: {len(t)} grid points, every published KL-certified entry reproduced at 1e-4 and 1e-5",
               len(t) == n and all(r["reproduced"] == "1" for r in t), f"rev_round11_step_table_{law}.csv reproduced")
    check_true(f"step-16 replay, {law}: step-function value below the KL value at every grid point (1e-4)",
               all(float(r["eps_step_1e-4"]) < float(r["eps_cert_1e-4_published"]) for r in t), f"rev_round11_step_table_{law}.csv")
st = {r["law"]: r for r in rows("rev_round11_step_results.csv")}
for law, mean, rec, flags in (("markov", 9.54, 0.80, 120), ("iid", 2.41, 0.84, 0), ("trace", 6.91, 0.65, 120), ("cyclic", 6.74, 0.85, 120)):
    check(f"step-function certified, {law}: mean at 1e-4", float(st[law]["mean"]), mean, 0.005, "rev_round11_step_results.csv")
    check(f"step-function certified, {law}: recovery", float(st[law]["recovery"]), rec, 0.005, "rev_round11_step_results.csv")
    check(f"step-function certified, {law}: flags of 120", int(st[law]["n_above_poisson"]), flags, 0, "rev_round11_step_results.csv")
kl_rec = {r["law"]: float(r["recovery"]) for r in sim if r["estimator"] == "cert_1e-4"}; kl_rec["cyclic"] = float(cy["cert_1e-4"]["recovery"])
un_rec = {r["law"]: float(r["recovery"]) for r in sim if r["estimator"] == "v2_1e-4"}; un_rec["cyclic"] = float(cy["v2_1e-4"]["recovery"])
gap_kl = [100 * (kl_rec[l] - float(st[l]["recovery"])) for l in st]; gap_un = [100 * (un_rec[l] - float(st[l]["recovery"])) for l in st]
check("step-function below KL certification, fewest points (extended V-F: 5 to 9)", min(gap_kl), 5, 0.5, "rev_round11_step_results.csv, step19_results.csv")
check("step-function below KL certification, most points", max(gap_kl), 9, 0.5, "same files")
check("step-function below the uncertified estimate, fewest points (8 to 16)", min(gap_un), 8, 0.5, "same files")
check("step-function below the uncertified estimate, most points", max(gap_un), 16, 0.5, "same files")
check("step-function certified: estimates above the reference, four laws", sum(int(r["n_above_reference"]) for r in st.values()), 0, 0, "rev_round11_step_results.csv")
acc = {}
for f in ("rev_train_base_orth.csv", "rev_train_ctrl200_share05.csv", "rev_train_ctrl200_share01.csv"):
    a = [float(r["acc"]) for r in rows(f)]; acc[f] = sum(a) / len(a)
    check(f"accuracy at 200 clients, {f}: runs", len(a), 50, 0, f)
check("mean accuracy, 2.5 present canaries per real client (1,000 canaries)", acc["rev_train_base_orth.csv"], 0.730, 0.0005, "rev_train_base_orth.csv")
check("mean accuracy, 0.5 (200 canaries, control)", acc["rev_train_ctrl200_share05.csv"], 0.729, 0.0005, "rev_train_ctrl200_share05.csv")
check("mean accuracy, 0.1 (40 canaries, control)", acc["rev_train_ctrl200_share01.csv"], 0.731, 0.0005, "rev_train_ctrl200_share01.csv")
c5, c1 = rows("rev_train_ctrl200_share05.csv"), rows("rev_train_ctrl200_share01.csv")
check_true("controls use base_orth's clipping norm", all(abs(float(r["C_clip"]) - float(rows("rev_train_base_orth.csv")[0]["C_clip"])) < 1e-9 for r in c5 + c1), "rev_train_ctrl200_*.csv C_clip")
check("control audits above the reference (of 100)", sum(int(r["above_ref"]) for r in c5 + c1), 0, 0, "rev_train_ctrl200_*.csv")
check("control flags, 200 canaries (of 50)", sum(int(r["flag"]) for r in c5), 49, 0, "rev_train_ctrl200_share05.csv")
check("control flags, 40 canaries (of 50)", sum(int(r["flag"]) for r in c1), 19, 0, "rev_train_ctrl200_share01.csv")
check("control, 40 canaries: runs below the table's grid (report 0)", sum(float(r["cert"]) == 0 for r in c1), 31, 0, "rev_train_ctrl200_share01.csv")

print("== 11. Twelfth-round checks (canary-free accuracy baseline, below-grid run, zero-return convention)")
c0 = rows("rev_train_ctrl200_share00.csv")
check("baseline without canaries: runs", len(c0), 50, 0, "rev_train_ctrl200_share00.csv")
check_true("baseline without canaries: every row has 0 canaries and no audit columns", all(r["m_canary"] == "0" and r["cert"] == "" for r in c0), "rev_train_ctrl200_share00.csv")
check("mean accuracy, no canaries (baseline; main text V-F, Table S4 row 5)", sum(float(r["acc"]) for r in c0) / len(c0), 0.731, 0.0005, "rev_train_ctrl200_share00.csv")
a0 = [float(r["acc"]) for r in c0]
check("baseline accuracy range, lower end (Table S4: 0.708)", min(a0), 0.708, 0.0005, "rev_train_ctrl200_share00.csv")
check("baseline accuracy range, upper end (Table S4: 0.754)", max(a0), 0.754, 0.0005, "rev_train_ctrl200_share00.csv")
check_true("baseline uses base_orth's seed and clipping norm", all(abs(float(r["C_clip"]) - float(rows("rev_train_base_orth.csv")[0]["C_clip"])) < 1e-9 for r in c0), "rev_train_ctrl200_share00.csv C_clip")
check("control, 200 canaries: runs below the table's grid (Table S4 caption: one, repetition 26)", sum(float(r["cert"]) == 0 for r in c5), 1, 0, "rev_train_ctrl200_share05.csv")
check_true("control, 200 canaries: the below-grid run is repetition 26 with a_lo = 0.3867", any(r["rep"] == "26" and abs(float(r["a_lo"]) - 0.3867) < 5e-5 and float(r["cert"]) == 0 for r in c5), "rev_train_ctrl200_share05.csv")
check("control flags, both arms (text S-V: 68 of 100)", sum(int(r["flag"]) for r in c5 + c1), 68, 0, "rev_train_ctrl200_*.csv")
allv = [float(r["a_lo"]) for f in ("rev_train_base_orth.csv", "rev_train_base_rand.csv", "rev_train_ratio_1to2.csv", "rev_train_ratio_1to10.csv", "rev_train_medium.csv", "rev_train_medium_iid.csv", "rev_train_ctrl200_share05.csv", "rev_train_ctrl200_share01.csv") for r in rows(f)]
check_true("every stored lower confidence limit of the 300 audited training runs is finite (the NaN path of the pre-v12 code was never reached)", len(allv) == 300 and all(v == v and abs(v) != float("inf") for v in allv), "rev_train_*.csv a_lo")
check_true("zero-return convention test script present (cd src && python test_lower_limit.py)", os.path.exists(os.path.join(HERE, "test_lower_limit.py")), "src/test_lower_limit.py")

print("== 12. Thirteenth-round checks (bootstrap over devices; the audit pointed at the worst device's law)")
import statistics
bs = rows("rev_round13_bootstrap.csv")
def q(vals, p):
    v = sorted(vals); k = (len(v) - 1) * p; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)
for pol, pt in (("fixed cohort 0.6", {"fm_ratio_0.0001": 5.16, "fm_ratio_1e-05": 6.25, "thm2_0.0001": 19.94, "thm2_ratio_0.0001": 16.38, "D": 21.3}),
                ("take all available", {"fm_ratio_0.0001": 9.44, "thm2_ratio_0.0001": 13.67, "D": 38.1})):
    base = [r for r in bs if r["policy"] == pol and r["replicate"] == "-1"]
    reps = [r for r in bs if r["policy"] == pol and int(r["replicate"]) >= 0]
    check_true(f"bootstrap, {pol}: one analysis-population row and 1,000 replicates", len(base) == 1 and len(reps) == 1000, "rev_round13_bootstrap.csv")
    for k, v in pt.items():
        check(f"bootstrap, {pol}: analysis population reproduces the paper's {k}", float(base[0][k]), v, 0.006 if v < 10 else 0.06, "rev_round13_bootstrap.csv replicate -1")
    check_true(f"bootstrap, {pol}: replicate seeds 13001+r give distinct populations (q varies)", len({r["q"] for r in reps}) > 900, "rev_round13_bootstrap.csv")
ivals = {("fixed cohort 0.6", "fm_ratio_0.0001"): (3.98, 7.34), ("fixed cohort 0.6", "fm_ratio_1e-05"): (4.60, 8.76), ("fixed cohort 0.6", "thm2_ratio_0.0001"): (14.73, 19.21),
         ("fixed cohort 0.6", "thm2_ratio_1e-05"): (14.78, 19.70), ("fixed cohort 0.6", "D"): (19.3, 24.7), ("take all available", "fm_ratio_0.0001"): (6.58, 11.75),
         ("take all available", "fm_ratio_1e-05"): (7.03, 12.98), ("take all available", "thm2_ratio_0.0001"): (11.93, 15.52), ("take all available", "thm2_ratio_1e-05"): (11.87, 15.81),
         ("take all available", "D"): (33.9, 42.2)}
for (pol, k), (lo, hi) in ivals.items():
    v = [float(r[k]) for r in bs if r["policy"] == pol and int(r["replicate"]) >= 0]
    check(f"bootstrap 95% interval, {pol}, {k}: lower end (Sec. S-VI / VI-G)", q(v, 0.025), lo, 0.06, "rev_round13_bootstrap.csv")
    check(f"bootstrap 95% interval, {pol}, {k}: upper end", q(v, 0.975), hi, 0.06, "rev_round13_bootstrap.csv")
sm = rows("rev_round13_bootstrap_summary.csv")
check_true("bootstrap summary file lists 1,000 replicates for every quantity", all(r["replicates"] == "1000" for r in sm), "rev_round13_bootstrap_summary.csv")
check_true("bootstrap: no final-model or Theorem 2 interval reaches 1", all(float(r["q025"]) > 1.5 for r in sm if r["quantity"].endswith("ratio_0.0001") or r["quantity"].endswith("ratio_1e-05")), "rev_round13_bootstrap_summary.csv")
wt = rows("rev_round13_worst_table.csv"); wa = rows("rev_round13_worst_audits.csv")
check("worst-device table: grid points 0.40 to 0.52 (13)", len(wt), 13, 0, "rev_round13_worst_table.csv")
for d in ("1e-4", "1e-5"):
    lo = [float(r[f"eps_lo_{d}"]) for r in wt]; hi = [float(r[f"eps_hi_{d}"]) for r in wt]
    check_true(f"worst-device table at {d}: increasing in a (the floor rule is valid)", all(b > a for a, b in zip(lo, lo[1:])), "rev_round13_worst_table.csv")
    check(f"worst-device table at {d}: pessimistic exceeds optimistic by at most 0.09", max(h - l for h, l in zip(hi, lo)), 0.086, 0.005, "rev_round13_worst_table.csv")
check("worst-device audits: 120 audits", len(wa), 120, 0, "rev_round13_worst_audits.csv")
check("worst-device audits: mean n_obs (text: about 58,000)", sum(int(r["n_obs"]) for r in wa) / len(wa), 57785, 50, "rev_round13_worst_audits.csv")
check("worst-device audits: coverage of a_lo <= a = 0.5", sum(float(r["a_lo"]) <= 0.5 for r in wa) / len(wa), 0.983, 0.001, "rev_round13_worst_audits.csv")
c4 = [float(r["cert_0.0001"]) for r in wa]; c5 = [float(r["cert_1e-05"]) for r in wa]
check("worst-device audit, certified mean at 1e-4 (text: 25.77)", statistics.mean(c4), 25.77, 0.005, "rev_round13_worst_audits.csv")
check("worst-device audit, certified sd at 1e-4 (text: 0.44)", statistics.pstdev(c4), 0.44, 0.005, "rev_round13_worst_audits.csv")
check("worst-device audit, recovery at 1e-4 against the reference 27.07 (text: 95%)", statistics.mean(c4) / 27.0672, 0.952, 0.001, "rev_round13_worst_audits.csv, log")
check("worst-device audit, certified mean at 1e-5 (text: 28.32)", statistics.mean(c5), 28.32, 0.005, "rev_round13_worst_audits.csv")
check("worst-device audit, recovery at 1e-5 against 29.71 (text: 95%)", statistics.mean(c5) / 29.7062, 0.953, 0.001, "rev_round13_worst_audits.csv, log")
check("worst-device audit: flags against the claim 1.219 at 1e-4 (of 120)", sum(int(r["flag_0.0001"]) for r in wa), 120, 0, "rev_round13_worst_audits.csv")
check("worst-device audit: audits above the reference at 1e-4", sum(int(r["above_ref_0.0001"]) for r in wa), 0, 0, "rev_round13_worst_audits.csv")
check("worst-device audit: audits above the reference at 1e-5", sum(int(r["above_ref_1e-05"]) for r in wa), 0, 0, "rev_round13_worst_audits.csv")
check_true("worst-device audit: every certified value is a table value (floor rule; the table file is rounded to 4 decimals)", all(any(abs(float(r["cert_0.0001"]) - float(t["eps_lo_1e-4"])) < 6e-5 for t in wt) for r in wa), "rev_round13_worst_audits.csv")
check_true("worst-device audit: the maximum certified value 26.99 is below the reference 27.07", max(c4) < 27.0672, "rev_round13_worst_audits.csv")
wl = open(os.path.join(LOGS, "rev_round13_worst_device_audit.log")).read()
check_true("worst-device log: device 335, window 21, K=27, reference 27.0672 / 29.7062 (optimistic, 1e-4 discretisation) bracketed by 27.0758 / 29.7148", all(x in wl for x in ("device 335", "window 21", "K=27", "optimistic 27.0672 / 29.7062", "pessimistic 27.0758 / 29.7148")), "logs/rev_round13_worst_device_audit.log")
dev335 = one("rev_peruser_exact2_devices.csv", device="335")
check("worst device: the stored lower end at 1e-4 (Table VI: 26.99 at discretisation 1e-3)", float(dev335["pr_lo_0.0001"]), 26.99, 0.005, "rev_peruser_exact2_devices.csv")


print("== 13. Fourteenth-round checks (noise error; operating region; comparators; a law learned from logs; heterogeneous data; CNN)")
ns = rows("rev_round14_noise_sim.csv")
check("noise error, simulation: 6 noise scales x 3 laws at m=1000 plus 6 at m=200", len(ns), 24, 0, "rev_round14_noise_sim.csv")
exp_flags = {("markov", "1000"): [0, 0, 1, 40, 120, 120], ("iid", "1000"): [0, 0, 0, 5, 111, 120], ("trace", "1000"): [0, 0, 0, 7, 76, 120], ("markov", "200"): [0, 0, 1, 9, 47, 120]}
for (law, m), fl in exp_flags.items():
    got = [int(r["flags"]) for r in ns if r["law"] == law and r["m"] == m]
    check_true(f"noise error, {law} m={m}: flags at r=1.10,1.00,0.98,0.95,0.90,0.80 are {fl} (Sec. V-F, S-V)", got == fl, "rev_round14_noise_sim.csv flags")
check("noise error: certified estimates above the realised loss eps(a/r) over the 24 cells (text: one of 2,880 simulated audits, Markov r=1.10)", sum(int(r["above_realised"]) for r in ns), 1, 0, "rev_round14_noise_sim.csv above_realised")
check("noise error: 2,880 simulated audits in all", sum(int(r["R"]) for r in ns), 2880, 0, "rev_round14_noise_sim.csv")
mk = [r for r in ns if r["law"] == "markov" and r["m"] == "1000"]
for r, (mc, rl) in zip(mk, [(8.15, 9.60), (10.31, 11.87), (10.80, 12.41), (11.65, 13.28), (13.18, 15.02), (14.22, 19.65)]):
    check(f"noise error, Markov r={r['noise_r']}: certified mean (text {mc}) and realised loss (text {rl})", float(r["mean_cert"]), mc, 0.006, "rev_round14_noise_sim.csv")
    check(f"noise error, Markov r={r['noise_r']}: realised loss", float(r["realised_loss"]), rl, 0.006, "rev_round14_noise_sim.csv")
for cfg, fl in (("noise_r1.00", 0), ("noise_r0.95", 2), ("noise_r0.90", 10), ("noise_r0.80", 10), ("noise_iid_r1.00", 0), ("noise_iid_r0.90", 8)):
    tr = rows(f"rev_round14_train_{cfg}.csv")
    check(f"noise error, training {cfg}: ten runs, flags of the claim (text {fl})", sum(int(r["flag_claim"]) for r in tr), fl, 0, f"rev_round14_train_{cfg}.csv")
    check_true(f"noise error, training {cfg}: no run above the realised loss", len(tr) == 10 and all(int(r["above_true_r"]) == 0 for r in tr), f"rev_round14_train_{cfg}.csv")
for cfg, rr in (("noise_r0.95", 0.950), ("noise_r0.90", 0.900), ("noise_r0.80", 0.801)):
    tr = rows(f"rev_round14_train_{cfg}.csv")
    check(f"noise error, training {cfg}: the noise-scale diagnostic reads r (text {rr})", statistics.mean(float(r["s_hat"]) for r in tr), rr, 0.003, f"rev_round14_train_{cfg}.csv s_hat")
rs = rows("rev_round14_residual_sim.csv")
check("operating region: 1 null + 4 classes x 3 rho + AR(1) at two phi x 3 rho = 19 cells", len(rs), 19, 0, "rev_round14_residual_sim.csv")
def cell(k, rho, phi="0.0"): return one("rev_round14_residual_sim.csv", **{"class": k, "rho": rho, "phi": phi})
check("operating region, null: certified mean (Table S2: 10.35)", float(cell("none", "0.0")["mean_cert"]), 10.35, 0.006, "rev_round14_residual_sim.csv")
check("operating region, offset 0.05: estimates above the reference (text 189 of 200)", int(cell("mean", "0.05")["above_ref_all"]), 189, 0, "rev_round14_residual_sim.csv")
check("operating region, offset 0.05: rejected by the frozen rule (text 200)", int(cell("mean", "0.05")["rejected"]), 200, 0, "rev_round14_residual_sim.csv")
check_true("operating region: the centred estimator is never above the reference in the offset cells and recovers 86-87%", all(int(cell("mean", rho)["above_ref_centred"]) == 0 and 0.855 <= float(cell("mean", rho)["recovery_centred"]) <= 0.875 for rho in ("0.05", "0.1", "0.25")), "rev_round14_residual_sim.csv")
check_true("operating region: white and AR(1) residuals never above the modified release's loss, recovery within 7 points of the null (0.872)", all(int(r["above_ref_all"]) == 0 and float(r["recovery"]) >= 0.802 for r in rs if r["class"] in ("white", "ar")), "rev_round14_residual_sim.csv")
check_true("operating region: reinforcement never above eps(a+rho); pull-back never above eps(a-rho)", all(int(r["above_ref_all"]) == 0 for r in rs if r["class"] in ("presence+", "presence-")), "rev_round14_residual_sim.csv")
check_true("operating region: the rule rejects every audit at rho=0.25 for white, offset and AR(1)", all(int(r["rejected"]) == 200 for r in rs if r["rho"] == "0.25" and r["class"] in ("white", "mean", "ar")), "rev_round14_residual_sim.csv")
check_true("operating region: the centred estimator agrees with the estimator of Sec. V-C within one point of recovery outside the offset cells", all(abs(float(r["recovery_centred"]) - float(r["recovery"])) <= 0.0151 for r in rs if r["class"] != "mean"), "rev_round14_residual_sim.csv")
rule = rows("rev_round14_residual_rule.csv")
check("operating region: the frozen rule's null sd of the absent mean (text 0.003)", float(one("rev_round14_residual_rule.csv", statistic="mean_a")["null_sd"]), 0.003, 0.0006, "rev_round14_residual_rule.csv")
cm = rows("rev_round14_comparators.csv")
check("comparators: 4 auditors x 4 laws", len(cm), 16, 0, "rev_round14_comparators.csv")
exp_cm = {("markov", "path"): (10.30, 0.867, 1, 120), ("markov", "sum-law"): (8.93, 0.753, 0, 117), ("markov", "count-cond"): (4.16, 0.351, 0, 110), ("markov", "max-fit"): (0.13, 0.011, 0, 0),
          ("iid", "path"): (2.56, 0.894, 0, 0), ("iid", "sum-law"): (2.00, 0.699, 0, 0), ("iid", "count-cond"): (1.81, 0.632, 0, 0), ("iid", "max-fit"): (0.13, 0.044, 0, 0),
          ("trace", "path"): (7.68, 0.719, 0, 120), ("trace", "sum-law"): (5.17, 0.483, 0, 110), ("trace", "count-cond"): (1.22, 0.114, 0, 45), ("trace", "max-fit"): (0.08, 0.007, 0, 0),
          ("cyclic", "path"): (7.14, 0.899, 0, 120), ("cyclic", "sum-law"): (5.59, 0.704, 1, 107), ("cyclic", "count-cond"): (1.74, 0.220, 0, 0), ("cyclic", "max-fit"): (0.15, 0.018, 0, 0)}
for (law, aud), (mc, rec, ab, fl) in exp_cm.items():
    r = one("rev_round14_comparators.csv", law=law, auditor=aud)
    check(f"comparators, {law} {aud}: certified mean (Sec. S-V: {mc})", float(r["mean"]), mc, 0.006, "rev_round14_comparators.csv")
    check(f"comparators, {law} {aud}: recovery", float(r["recovery"]), rec, 0.0015, "rev_round14_comparators.csv")
    check_true(f"comparators, {law} {aud}: above the reference {ab}, flags {fl}", int(r["above_ref"]) == ab and int(r["flags"]) == fl, "rev_round14_comparators.csv")
for law, ir in (("markov", 0.18), ("iid", 0.10), ("trace", 0.15), ("cyclic", 0.10)):
    check(f"comparators, {law}: information ratio of the sum statistic (text {ir})", float(one("rev_round14_comparators.csv", law=law, auditor="sum-law")["info_ratio"]), ir, 0.005, "rev_round14_comparators.csv")
for cfg, (rec, fl, fp, acc) in (("dir01_m100", (0.64, 0, 9, 0.754)), ("dir01_m500", (0.82, 0, 10, 0.753)), ("iid_m100", (0.52, 0, 8, 0.807)), ("iid_m500", (0.82, 0, 10, 0.808))):
    tr = rows(f"rev_round14_train_{cfg}.csv")
    check(f"heterogeneous data, {cfg}: ten runs, certified recovery (text {rec:.0%})", statistics.mean(float(r["cert"]) for r in tr) / 11.869, rec, 0.006, f"rev_round14_train_{cfg}.csv")
    check_true(f"heterogeneous data, {cfg}: flags of the correct claim {fl}, of the Poisson claim {fp}, none above the reference", sum(int(r["flag_claim"]) for r in tr) == fl and sum(int(r["flag_poisson"]) for r in tr) == fp and all(int(r["above_true_r"]) == 0 for r in tr), f"rev_round14_train_{cfg}.csv")
    check(f"heterogeneous data, {cfg}: mean accuracy (text {acc})", statistics.mean(float(r["acc"]) for r in tr), acc, 0.002, f"rev_round14_train_{cfg}.csv")
dr = rows("rev_round14_train_dir01_m100.csv") + rows("rev_round14_train_dir01_m500.csv")
check_true("heterogeneous data: the Dirichlet(0.1) shards range from empty to 1,183 examples over both configurations and 85% of updates are clipped", all(int(r["shard_min"]) == 0 for r in dr) and max(int(r["shard_max"]) for r in dr) == 1183 and abs(statistics.mean(float(r["clip_frac"]) for r in dr) - 0.85) < 0.01, "rev_round14_train_dir01_*.csv")
ll = rows("rev_round14_learned_law.csv"); lh = rows("rev_round14_lamhat.csv")
check("learned law: 3 log sizes x 3 rules", len(ll), 9, 0, "rev_round14_learned_law.csv")
check("learned law: sd of lambda_hat over 1,000 logs of N=100 devices (text 0.0064)", statistics.pstdev(float(r["lambda_hat"]) for r in lh if r["devices"] == "100"), 0.0064, 0.0002, "rev_round14_lamhat.csv")
r = one("rev_round14_learned_law.csv", devices="100", rule="oracle")
check("learned law, N=100 oracle: recovery (text 86%)", float(r["recovery"]), 0.8552, 0.0015, "rev_round14_learned_law.csv")
check_true("learned law, N=100 oracle: replicates 1000, above 11.869 = 0, above 12.033 = 0, flags 1000, abstentions 0", r["replicates"] == "1000" and r["above_true"] == "0" and r["above_hi"] == "0" and r["flags"] == "1000" and r["abstentions"] == "0", "rev_round14_learned_law.csv")
r = one("rev_round14_learned_law.csv", devices="100", rule="plugin")
check("learned law, N=100 plugin: recovery (text 85%)", float(r["recovery"]), 0.8509, 0.0015, "rev_round14_learned_law.csv")
check_true("learned law, N=100 plugin: replicates 1000, above 11.869 = 19, above 12.033 = 2, flags 1000, abstentions 0", r["replicates"] == "1000" and r["above_true"] == "19" and r["above_hi"] == "2" and r["flags"] == "1000" and r["abstentions"] == "0", "rev_round14_learned_law.csv")
r = one("rev_round14_learned_law.csv", devices="100", rule="cons")
check("learned law, N=100 cons: recovery (text 73%)", float(r["recovery"]), 0.7313, 0.0015, "rev_round14_learned_law.csv")
check_true("learned law, N=100 cons: replicates 1000, above 11.869 = 0, above 12.033 = 0, flags 1000, abstentions 0", r["replicates"] == "1000" and r["above_true"] == "0" and r["above_hi"] == "0" and r["flags"] == "1000" and r["abstentions"] == "0", "rev_round14_learned_law.csv")
check("learned law: sd of lambda_hat over 1,000 logs of N=500 devices (text 0.0029)", statistics.pstdev(float(r["lambda_hat"]) for r in lh if r["devices"] == "500"), 0.0029, 0.0002, "rev_round14_lamhat.csv")
r = one("rev_round14_learned_law.csv", devices="500", rule="oracle")
check("learned law, N=500 oracle: recovery (text 86%)", float(r["recovery"]), 0.8563, 0.0015, "rev_round14_learned_law.csv")
check_true("learned law, N=500 oracle: replicates 1000, above 11.869 = 0, above 12.033 = 0, flags 1000, abstentions 0", r["replicates"] == "1000" and r["above_true"] == "0" and r["above_hi"] == "0" and r["flags"] == "1000" and r["abstentions"] == "0", "rev_round14_learned_law.csv")
r = one("rev_round14_learned_law.csv", devices="500", rule="plugin")
check("learned law, N=500 plugin: recovery (text 85%)", float(r["recovery"]), 0.8542, 0.0015, "rev_round14_learned_law.csv")
check_true("learned law, N=500 plugin: replicates 1000, above 11.869 = 0, above 12.033 = 0, flags 1000, abstentions 0", r["replicates"] == "1000" and r["above_true"] == "0" and r["above_hi"] == "0" and r["flags"] == "1000" and r["abstentions"] == "0", "rev_round14_learned_law.csv")
r = one("rev_round14_learned_law.csv", devices="500", rule="cons")
check("learned law, N=500 cons: recovery (text 78%)", float(r["recovery"]), 0.7823, 0.0015, "rev_round14_learned_law.csv")
check_true("learned law, N=500 cons: replicates 1000, above 11.869 = 0, above 12.033 = 0, flags 1000, abstentions 0", r["replicates"] == "1000" and r["above_true"] == "0" and r["above_hi"] == "0" and r["flags"] == "1000" and r["abstentions"] == "0", "rev_round14_learned_law.csv")
check("learned law: sd of lambda_hat over 1,000 logs of N=2000 devices (text 0.0015)", statistics.pstdev(float(r["lambda_hat"]) for r in lh if r["devices"] == "2000"), 0.0015, 0.0002, "rev_round14_lamhat.csv")
r = one("rev_round14_learned_law.csv", devices="2000", rule="oracle")
check("learned law, N=2000 oracle: recovery (text 86%)", float(r["recovery"]), 0.8553, 0.0015, "rev_round14_learned_law.csv")
check_true("learned law, N=2000 oracle: replicates 1000, above 11.869 = 0, above 12.033 = 0, flags 1000, abstentions 0", r["replicates"] == "1000" and r["above_true"] == "0" and r["above_hi"] == "0" and r["flags"] == "1000" and r["abstentions"] == "0", "rev_round14_learned_law.csv")
r = one("rev_round14_learned_law.csv", devices="2000", rule="plugin")
check("learned law, N=2000 plugin: recovery (text 85%)", float(r["recovery"]), 0.8544, 0.0015, "rev_round14_learned_law.csv")
check_true("learned law, N=2000 plugin: replicates 1000, above 11.869 = 0, above 12.033 = 0, flags 1000, abstentions 0", r["replicates"] == "1000" and r["above_true"] == "0" and r["above_hi"] == "0" and r["flags"] == "1000" and r["abstentions"] == "0", "rev_round14_learned_law.csv")
r = one("rev_round14_learned_law.csv", devices="2000", rule="cons")
check("learned law, N=2000 cons: recovery (text 81%)", float(r["recovery"]), 0.8078, 0.0015, "rev_round14_learned_law.csv")
check_true("learned law, N=2000 cons: replicates 1000, above 11.869 = 0, above 12.033 = 0, flags 1000, abstentions 0", r["replicates"] == "1000" and r["above_true"] == "0" and r["above_hi"] == "0" and r["flags"] == "1000" and r["abstentions"] == "0", "rev_round14_learned_law.csv")
la = rows("rev_round14_learned_law_audits.csv")
check("learned law: 3,000 audit rows", len(la), 3000, 0, "rev_round14_learned_law_audits.csv")
check_true("learned law: the conservative table never lies above lambda_hat - z_0.99 se, and is empty exactly when the rule abstains", all((r["lambda_cons_table"] == "" and float(r["cert_cons"]) == 0.0) or (r["lambda_cons_table"] != "" and float(r["lambda_cons_table"]) <= float(r["lambda_lo"]) + 1e-9) for r in la), "rev_round14_learned_law_audits.csv")
check_true("learned law: the plug-in table is the grid point nearest to lambda_hat", all(abs(float(r["lambda_plugin_table"]) - float(r["lambda_hat"])) <= 0.0025 + 1e-9 for r in la), "rev_round14_learned_law_audits.csv")
tabs = {l: rows(f"rev_round14_table_lam_{l:.3f}.csv") for l in (0.86, 0.87, 0.875, 0.88, 0.885, 0.89, 0.895, 0.90, 0.905, 0.91, 0.915, 0.92, 0.93)}
check_true("learned law: thirteen tables on the grid 0.43..0.51 (9 points each)", all(len(t) == 9 and t[0]["a"] == "0.43" and t[-1]["a"] == "0.51" for t in tabs.values()), "rev_round14_table_lam_*.csv")
check_true("learned law: the certified curve increases in lambda at every grid point (the monotonicity the conservative rule rests on)", all(float(tabs[a][i]["eps_cert_bern_0.005"]) < float(tabs[b][i]["eps_cert_bern_0.005"]) for a, b in zip(sorted(tabs), sorted(tabs)[1:]) for i in range(9)), "rev_round14_table_lam_*.csv")
check_true("learned law: every table increases in a (the floor rule is valid)", all(float(t[i]["eps_cert_bern_0.005"]) < float(t[i + 1]["eps_cert_bern_0.005"]) for t in tabs.values() for i in range(8)), "rev_round14_table_lam_*.csv")
o = rows("rev_round14_outfamily.csv")[0]
check("out of family: certified recovery of the trace reference (text 12%)", float(o["recovery"]), 0.1243, 0.0015, "rev_round14_outfamily.csv")
check_true("out of family: 120 audits, 0 above the reference, 105 flags; fitted chain q = 0.0471, lambda = 0.528", o["R"] == "120" and o["above_ref"] == "0" and o["flags"] == "105" and abs(float(o["q_hat"]) - 0.0471) < 6e-5 and abs(float(o["lambda_hat"]) - 0.528) < 6e-4, "rev_round14_outfamily.csv")
cn = rows("rev_round14_cnn_cnn_markov.csv")
check_true("CNN cnn_markov: 5 runs of the 545,098-parameter network", len(cn) == 5 and all(r["d"] == "545098" for r in cn), "rev_round14_cnn_cnn_markov.csv")
check("CNN cnn_markov: certified recovery (text 85%)", statistics.mean(float(r["cert"]) for r in cn) / 11.869, 0.8547, 0.0015, "rev_round14_cnn_cnn_markov.csv")
check_true("CNN cnn_markov: flags of the Poisson claim 5, of the correct claim 0, none above the realised loss, accuracy 0.373 to 0.403", sum(int(r["flag_poisson"]) for r in cn) == 5 and sum(int(r["flag_claim"]) for r in cn) == 0 and all(int(r["above_true_r"]) == 0 for r in cn) and abs(min(float(r["acc"]) for r in cn) - 0.373) < 6e-4 and abs(max(float(r["acc"]) for r in cn) - 0.403) < 6e-4, "rev_round14_cnn_cnn_markov.csv")
cn = rows("rev_round14_cnn_cnn_iid.csv")
check_true("CNN cnn_iid: 3 runs of the 545,098-parameter network", len(cn) == 3 and all(r["d"] == "545098" for r in cn), "rev_round14_cnn_cnn_iid.csv")
check("CNN cnn_iid: certified recovery (text 89%)", statistics.mean(float(r["cert"]) for r in cn) / 2.866, 0.8911, 0.0015, "rev_round14_cnn_cnn_iid.csv")
check_true("CNN cnn_iid: flags of the Poisson claim 0, of the correct claim 0, none above the realised loss, accuracy 0.387 to 0.388", sum(int(r["flag_poisson"]) for r in cn) == 0 and sum(int(r["flag_claim"]) for r in cn) == 0 and all(int(r["above_true_r"]) == 0 for r in cn) and abs(min(float(r["acc"]) for r in cn) - 0.387) < 6e-4 and abs(max(float(r["acc"]) for r in cn) - 0.388) < 6e-4, "rev_round14_cnn_cnn_iid.csv")
cn = rows("rev_round14_cnn_cnn_markov_r0.90.csv")
check_true("CNN cnn_markov_r0.90: 2 runs of the 545,098-parameter network", len(cn) == 2 and all(r["d"] == "545098" for r in cn), "rev_round14_cnn_cnn_markov_r0.90.csv")
check("CNN cnn_markov_r0.90: certified recovery (text 110%)", statistics.mean(float(r["cert"]) for r in cn) / 11.869, 1.1043, 0.0015, "rev_round14_cnn_cnn_markov_r0.90.csv")
check_true("CNN cnn_markov_r0.90: flags of the Poisson claim 2, of the correct claim 2, none above the realised loss, accuracy 0.368 to 0.404", sum(int(r["flag_poisson"]) for r in cn) == 2 and sum(int(r["flag_claim"]) for r in cn) == 2 and all(int(r["above_true_r"]) == 0 for r in cn) and abs(min(float(r["acc"]) for r in cn) - 0.368) < 6e-4 and abs(max(float(r["acc"]) for r in cn) - 0.404) < 6e-4, "rev_round14_cnn_cnn_markov_r0.90.csv")
check_true("CNN: the diagnostics match Model 1 (scale within 0.01 of 1, lag-1 within 0.01 of -0.005, correlation sd within 0.002 of 0.0709, erasure 0.90 to 1.15, the per-run spread of the MLP runs)", all(abs(float(r["s_hat"]) - 1) < 0.01 and abs(float(r["lag1"]) + 0.005) < 0.01 and abs(float(r["xcorr_sd"]) - 0.0709) < 0.002 and 0.90 <= float(r["erasure"]) <= 1.15 for k in ("markov", "iid") for r in rows(f"rev_round14_cnn_cnn_{k}.csv")), "rev_round14_cnn_cnn_*.csv")

print("== 14. Fifteenth-round checks (per-audit records; the noise-shortfall audits before and after the rule; the scale alarm; realised-loss intervals; the device decomposition)")
rec = rows("rev_round15_records.csv")
check("records: 2,880 noise + 3,800 residual + 800 null = 7,480 rows", len(rec), 7480, 0, "rev_round15_records.csv")
ids = [r["audit_id"] for r in rec]
check_true("records: no duplicate or missing audit id (120 / 200 / 200 reps per cell)", len(set(ids)) == len(ids) and all(sum(1 for r in rec if r["experiment"] == e and r["cell"] == c) == n for e, n in (("noise", 120), ("residual", 200), ("null", 200)) for c in {r["cell"] for r in rec if r["experiment"] == e}), "rev_round15_records.csv")
noise14 = rows("rev_round14_noise_sim.csv")
for r14 in noise14:
    cellrows = [r for r in rec if r["experiment"] == "noise" and r["law"] == r14["law"] and r["m"] == r14["m"] and float(r["noise_r"]) == float(r14["noise_r"])]
    e = [float(r["cert"]) for r in cellrows]
    check_true(f"records reproduce the round-14 noise cell {r14['law']} m={r14['m']} r={r14['noise_r']} (mean, flags, exceedances)", len(e) == 120 and abs(statistics.mean(e) - float(r14["mean_cert"])) < 5e-5 and sum(x > float(r14["claim"]) for x in e) == int(r14["flags"]) and sum(x > float(r14["realised_loss"]) for x in e) == int(r14["above_realised"]), "rev_round15_records.csv vs rev_round14_noise_sim.csv")
sn = rows("rev_round15_screening_noise.csv")
def scell(law, m, r, rule="A", est="eq11"): return one("rev_round15_screening_noise.csv", law=law, m=str(m), noise_r=f"{r:.2f}", rule=rule, estimator=est)
check_true("screening: raw flags of every cell equal the round-14 flags", all(int(scell(r14["law"], r14["m"], float(r14["noise_r"]))["raw_flags"]) == int(r14["flags"]) for r14 in noise14), "rev_round15_screening_noise.csv")
check_true("scale alarm: 120 of 120 at every r != 1 with 1,000 canaries, 0 of 360 at r = 1 (text V-F, S-V.E)", all(int(scell(l, 1000, r)["scale_alarms"]) == 120 for l in ("markov", "iid", "trace") for r in (1.10, 0.98, 0.95, 0.90, 0.80)) and sum(int(scell(l, 1000, 1.00)["scale_alarms"]) for l in ("markov", "iid", "trace")) == 0, "rev_round15_screening_noise.csv scale_alarms")
check_true("scale alarm, 200 canaries, own null band: 56 of 120 at r = 0.98, 120 at r <= 0.95 and 1.10, 0 at r = 1", int(scell("markov", 200, 0.98, "B_cal")["scale_alarms"]) == 56 and all(int(scell("markov", 200, r, "B_cal")["scale_alarms"]) == 120 for r in (1.10, 0.95, 0.90, 0.80)) and int(scell("markov", 200, 1.00, "B_cal")["scale_alarms"]) == 0, "rev_round15_screening_noise.csv")
check_true("rule A (as frozen) keeps no flag at any r != 1 and withholds every report there (1,000 canaries)", all(int(scell(l, 1000, r)["A_flag_kept"]) == 0 and int(scell(l, 1000, r)["abstentions"]) == 120 for l in ("markov", "iid", "trace") for r in (1.10, 0.98, 0.95, 0.90, 0.80)), "rev_round15_screening_noise.csv rule A")
check_true("rule A at r = 1: abstentions 0, 0, 7 (Markov, iid, trace; the 7 by erasure)", [int(scell(l, 1000, 1.00)["abstentions"]) for l in ("markov", "iid", "trace")] == [0, 0, 7] and scell("trace", 1000, 1.00)["reasons"].strip() == "erasure:7", "rev_round15_screening_noise.csv rule A")
check_true("rule B keeps 120, 110, 74 of 120, 111, 76 flags at r = 0.90; all 40, 5, 7 at 0.95; 120, 120, 118 at 0.80", [int(scell(l, 1000, 0.90, "B")["A_flag_kept"]) for l in ("markov", "iid", "trace")] == [120, 110, 74] and [int(scell(l, 1000, 0.95, "B")["A_flag_kept"]) for l in ("markov", "iid", "trace")] == [40, 5, 7] and [int(scell(l, 1000, 0.80, "B")["A_flag_kept"]) for l in ("markov", "iid", "trace")] == [120, 120, 118], "rev_round15_screening_noise.csv rule B")
check_true("rule B: one iid audit withheld by KS and two trace audits by erasure at r = 0.90; 7 of 360 abstentions at r = 1", scell("iid", 1000, 0.90, "B")["reasons"].strip() == "ks:1" and scell("trace", 1000, 0.90, "B")["reasons"].strip() == "erasure:4" and int(scell("trace", 1000, 0.90, "B")["B_flag_withheld"]) == 2 and sum(int(scell(l, 1000, 1.00, "B")["abstentions"]) for l in ("markov", "iid", "trace")) == 7, "rev_round15_screening_noise.csv rule B")
check_true("per-configuration nulls (rule B_cal): 2 of 360 abstentions at r = 1 (KS), 75 trace flags kept at r = 0.90", sum(int(scell(l, 1000, 1.00, "B_cal")["abstentions"]) for l in ("markov", "iid", "trace")) == 2 and int(scell("trace", 1000, 0.90, "B_cal")["A_flag_kept"]) == 75, "rev_round15_screening_noise.csv rule B_cal")
check_true("200 canaries: transferred rule A withholds 101 of 120 at r = 1, rule B 98, the m = 200 rule 1; the m = 200 rule keeps all 47 flags at r = 0.90", int(scell("markov", 200, 1.00, "A")["abstentions"]) == 101 and int(scell("markov", 200, 1.00, "B")["abstentions"]) == 98 and int(scell("markov", 200, 1.00, "B_cal")["abstentions"]) == 1 and int(scell("markov", 200, 0.90, "B_cal")["A_flag_kept"]) == 47 and int(scell("markov", 200, 0.90)["raw_flags"]) == 47, "rev_round15_screening_noise.csv")
check_true("flags against the Markov claim's upper limit 12.033 equal the flags against 11.869 in every Markov cell", all(int(r["raw_flags_upper_limit"]) == int(r["raw_flags"]) for r in sn if r["law"] == "markov" and r["estimator"] == "eq11" and r["rule"] == "A"), "rev_round15_screening_noise.csv")
eqA = [r for r in sn if r["rule"] == "A" and r["estimator"] == "eq11"]
check_true("coverage: 1 of 2,880 above the point reference, 0 above the independent interval's upper end, 50 inside an interval", sum(int(r["above_point"]) for r in eqA) == 1 and sum(int(r["above_interval"]) for r in eqA) == 0 and sum(int(r["inside_interval"]) for r in eqA) == 50, "rev_round15_screening_noise.csv")
check_true("centred estimator: 120, 108, 69 raw flags at r = 0.90 and 120, 107, 67 after rule B", [int(scell(l, 1000, 0.90, "B", "centred")["raw_flags"]) for l in ("markov", "iid", "trace")] == [120, 108, 69] and [int(scell(l, 1000, 0.90, "B", "centred")["A_flag_kept"]) for l in ("markov", "iid", "trace")] == [120, 107, 67], "rev_round15_screening_noise.csv centred")
iv = rows("rev_round15_intervals.csv")
check_true("intervals: 18 (law, r) points; iid exact; every round-14 point value lies inside its independent 99% interval", len(iv) == 18 and all(r["samples"] == "exact" for r in iv if r["law"] == "iid") and all(float(r["lo99"]) <= float(one("rev_round14_noise_sim.csv", law=r["law"], m="1000", noise_r=str(float(r["noise_r"])))["realised_loss"]) <= float(r["hi99"]) for r in iv if r["law"] != "iid"), "rev_round15_intervals.csv vs rev_round14_noise_sim.csv")
ru = rows("rev_round15_rules.csv")
fil = {r["statistic"]: r for r in rows("rev_round14_residual_rule.csv")}
ruleA = {r["statistic"]: r for r in ru if r["rule"] == "A"}
check_true("rule A recomputed from the regenerated null cell equals the round-14 rule file (means and sds to 1e-5; KS max)", all(abs(float(ruleA[k]["null_mean"]) - float(fil[f]["null_mean"])) < 1e-5 and abs(float(ruleA[k]["null_sd"]) - float(fil[f]["null_sd"])) < 1e-5 for k, f in (("s_hat", "s_hat"), ("lag1_200", "lag1"), ("mean_a", "mean_a"), ("erasure", "erasure"))) and abs(float(ruleA["ks"]["upper"]) - float(fil["ks"]["upper"])) < 1e-5, "rev_round15_rules.csv vs rev_round14_residual_rule.csv")
nulls = {(r["law"], r["m"]): r for r in ru if r["rule"] == "A_cal" and r["statistic"] == "s_hat"}
check_true("independent nulls: s_hat sd 0.0023 (Markov 1,000), 0.0049 (Markov 200), 0.0022 (iid), 0.0022 (trace) (text: sd 0.002; band 1 +- 0.020 at m = 200)", abs(float(nulls[("markov", "1000")]["null_sd"]) - 0.00234) < 5e-5 and abs(float(nulls[("markov", "200")]["null_sd"]) - 0.00492) < 5e-5 and abs(float(nulls[("iid", "1000")]["null_sd"]) - 0.00217) < 5e-5 and abs(float(nulls[("trace", "1000")]["null_sd"]) - 0.00224) < 5e-5, "rev_round15_rules.csv")
ksn = {(r["law"], r["m"]): float(r["upper"]) for r in ru if r["rule"] == "A_cal" and r["statistic"] == "ks"}
check_true("independent nulls: KS maximum 0.0081 at m = 200 against 0.0038 for the round-14 rule (twice as large); erasure sd 0.137 (trace) against 0.063 (Markov)", abs(ksn[("markov", "200")] - 0.00807) < 5e-5 and abs(float(fil["ks"]["upper"]) - 0.00383) < 5e-5 and abs(float(one("rev_round15_rules.csv", rule="A_cal", law="trace", m="1000", statistic="erasure")["null_sd"]) - 0.1367) < 5e-4 and abs(float(fil["erasure"]["null_sd"]) - 0.0627) < 5e-4, "rev_round15_rules.csv")
sr = rows("rev_round15_screening_residual.csv")
def rcell(k, rho, rule, est="eq11"): return one("rev_round15_screening_residual.csv", **{"class": k, "rho": rho, "rule": rule, "estimator": est})
check_true("residual cells: white rho = 0.10 / 0.25 rejected 13 / 200 by rule A (s_hat alone) and 0 / 2 by rule B", int(rcell("white", "0.1", "A")["rejected"]) == 13 and int(rcell("white", "0.25", "A")["rejected"]) == 200 and int(rcell("white", "0.1", "B")["rejected"]) == 0 and int(rcell("white", "0.25", "B")["rejected"]) == 2 and rcell("white", "0.1", "A")["reasons"].strip() == "s_hat:13", "rev_round15_screening_residual.csv")
def rcellp(k, rho, phi, rule, est="eq11"): return one("rev_round15_screening_residual.csv", **{"class": k, "rho": rho, "phi": phi, "rule": rule, "estimator": est})
check_true("residual cells: every round-14 rejection and exceedance count reproduced under rule A (19 cells)", all(int(rcellp(r14["class"], r14["rho"], r14["phi"], "A")["rejected"]) == int(r14["rejected"]) and int(rcellp(r14["class"], r14["rho"], r14["phi"], "A")["above_ref_all"]) == int(r14["above_ref_all"]) and int(rcellp(r14["class"], r14["rho"], r14["phi"], "A")["above_ref_passed"]) == int(r14["above_ref_passed"]) for r14 in rows("rev_round14_residual_sim.csv")), "rev_round15_screening_residual.csv vs rev_round14_residual_sim.csv")
check_true("residual cells: the centred estimator exceeds the offset cells' reference in none, before or after the rule", all(int(rcell("mean", rho, rl, "centred")["above_ref_all"]) == 0 for rho in ("0.05", "0.1", "0.25") for rl in ("A", "B")), "rev_round15_screening_residual.csv")
st = rows("rev_round15_screening_training.csv")
def tcell(cfg, rule, est="eq11"): return one("rev_round15_screening_training.csv", config=cfg, rule=rule, estimator=est)
for cfg, raw, alarm in (("noise_r1.00", 0, 0), ("noise_r0.95", 2, 10), ("noise_r0.90", 10, 10), ("noise_r0.80", 10, 10), ("noise_iid_r1.00", 0, 0), ("noise_iid_r0.90", 8, 10)):
    t = tcell(cfg, "A"); tb = tcell(cfg, "B")
    check_true(f"training {cfg}: raw flags {raw}, scale alarms {alarm}, rule A keeps none at r < 1, rule B keeps all, no abstention at r = 1", int(t["raw_flags"]) == raw and int(t["scale_alarms"]) == alarm and int(tb["A_flag_kept"]) == raw and (int(t["A_flag_kept"]) == 0 if alarm else int(t["abstentions"]) == 0 and int(tb["abstentions"]) == 0), "rev_round15_screening_training.csv")
for cfg in ("noise_r1.00", "noise_r0.95", "noise_r0.90", "noise_r0.80", "noise_iid_r1.00", "noise_iid_r0.90"):
    a = rows(f"rev_round15_train_records_{cfg}.csv"); b = rows(f"rev_round14_train_{cfg}.csv")
    check_true(f"training records {cfg}: 10 runs identical to the round-14 file on every shared column", len(a) == 10 and len(b) == 10 and all(all(x[k] == y[k] for k in y) for x, y in zip(a, b)), f"rev_round15_train_records_{cfg}.csv")
hd = rows("rev_round15_horizon_decomp.csv"); hs = {r["round_min"]: r for r in rows("rev_round15_horizon_slopes.csv")}
check_true("decomposition: B + W = Var(N_T) to 1e-9 at every horizon", all(abs(float(r["B_plus_W_minus_V"])) < 1e-9 for r in hd), "rev_round15_horizon_decomp.csv")
check("decomposition: slope of D, 10 min (text 0.064)", float(hs["10"]["slope_D"]), 0.0639, 0.0005, "rev_round15_horizon_slopes.csv")
check("decomposition: slope of D, 5 min (text 0.058)", float(hs["5"]["slope_D"]), 0.0578, 0.0005, "rev_round15_horizon_slopes.csv")
check("decomposition: slope of B/mean, 10 min (text 0.085)", float(hs["10"]["slope_B_over_mean"]), 0.0848, 0.0005, "rev_round15_horizon_slopes.csv")
check("decomposition: slope of B/mean, 5 min (text 0.074)", float(hs["5"]["slope_B_over_mean"]), 0.0743, 0.0005, "rev_round15_horizon_slopes.csv")
check("decomposition: slope of W/mean, 10 min (text -0.021)", float(hs["10"]["slope_W_over_mean"]), -0.0209, 0.0005, "rev_round15_horizon_slopes.csv")
check("decomposition: slope of W/mean, 5 min (text -0.017)", float(hs["5"]["slope_W_over_mean"]), -0.0166, 0.0005, "rev_round15_horizon_slopes.csv")
check_true("decomposition: the two component slopes add to the slope of D (both panels)", all(abs(float(hs[p]["slope_sum_minus_slope_D"])) < 1e-9 for p in ("10", "5")), "rev_round15_horizon_slopes.csv")
check("decomposition: Var(p)/E[p], 10 min (text 0.083)", float(hs["10"]["VarP_over_EP"]), 0.0832, 0.0005, "rev_round15_horizon_slopes.csv")
check("decomposition: Var(p)/E[p], 5 min (text 0.072)", float(hs["5"]["VarP_over_EP"]), 0.0720, 0.0005, "rev_round15_horizon_slopes.csv")
check_true("decomposition: the v14 '77% and 80%' were the ratios of the fitted slope to Var(p)/E[p] (0.768, 0.802), not shares", abs(float(hs["10"]["slopeD_over_VarPEP"]) - 0.768) < 0.002 and abs(float(hs["5"]["slopeD_over_VarPEP"]) - 0.802) < 0.002, "rev_round15_horizon_slopes.csv")
def hrow(rm, T): return one("rev_round15_horizon_decomp.csv", round_min=str(rm), T=str(T))
check_true("decomposition: W/mean 7.1 and 14.7 at the shortest horizon, 0.2 and 0.8 at the longest; 4 and 31 windows per device there, none disjoint", abs(float(hrow(10, 50)["W_over_mean"]) - 7.05) < 0.01 and abs(float(hrow(5, 100)["W_over_mean"]) - 14.72) < 0.01 and abs(float(hrow(10, 400)["W_over_mean"]) - 0.19) < 0.01 and abs(float(hrow(5, 1000)["W_over_mean"]) - 0.81) < 0.01 and hrow(10, 400)["windows_per_device"] == "4" and hrow(5, 1000)["windows_per_device"] == "31" and hrow(10, 400)["nonoverlap_windows_per_device"] == "1" and hrow(5, 1000)["nonoverlap_windows_per_device"] == "1", "rev_round15_horizon_decomp.csv")
check_true("decomposition: B/Var(N_T) 0.38, 0.58, 0.83, 0.96, 0.99 (10 min) and 0.33, 0.52, 0.76, 0.90, 0.96, 0.99 (5 min)", all(abs(float(hrow(10, T)["B_over_V"]) - v) < 0.006 for T, v in ((50, 0.375), (100, 0.583), (200, 0.829), (300, 0.959), (400, 0.994))) and all(abs(float(hrow(5, T)["B_over_V"]) - v) < 0.006 for T, v in ((100, 0.331), (200, 0.522), (400, 0.756), (600, 0.904), (800, 0.960), (1000, 0.989))), "rev_round15_horizon_decomp.csv")
check_true("decomposition: disjoint-window shares 0.38, 0.58, 0.78 (T = 50, 100, 200) and 0.33, 0.53, 0.77 (T = 100, 200, 400)", all(abs(float(hrow(10, T)["B_over_V_nonoverlap"]) - v) < 0.006 for T, v in ((50, 0.384), (100, 0.576), (200, 0.777))) and all(abs(float(hrow(5, T)["B_over_V_nonoverlap"]) - v) < 0.006 for T, v in ((100, 0.334), (200, 0.526), (400, 0.765))), "rev_round15_horizon_decomp.csv")
check_true("decomposition: the between-device term equals T Var(p)/E[p] within 7% at every horizon (edge effects)", all(abs(float(r["B_over_mean"]) / float(r["T_VarP_over_EP"]) - 1) < 0.07 for r in hd), "rev_round15_horizon_decomp.csv")

print(f"\n{'ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
