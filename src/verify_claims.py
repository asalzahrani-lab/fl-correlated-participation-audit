"""Reviewer check: re-derive every headline number of the paper from the stored outputs in ../results/ and print
PASS / FAIL for each.  No heavy computation is repeated (that is what the generating scripts are for); this script
checks that the numbers in the manuscripts (TIFS v9 main + supplement, extended v14) follow from the files that the
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
cert = [r for r in sim if r["estimator"].startswith("cert")]
t1k = [r for f in ("step16_results_markov_T1000.csv", "step16_results_trace_T1000.csv") for r in rows(f) if r["estimator"].startswith("cert")]
recs = [float(r["recovery"]) for r in cert + t1k]
check("lowest certified recovery", min(recs), 0.74, 0.005, "step19_results.csv, step16_results_*_T1000.csv")
check("highest certified recovery", max(recs), 0.90, 0.005, "step19_results.csv, step16_results_*_T1000.csv")
gauss = [float(r["recovery"]) for r in sim if r["estimator"].startswith("gauss") and r["law"] in ("markov", "trace")]
check("Gaussian fit, lowest recovery on correlated laws", min(gauss), 0.03, 0.005, "step19_results.csv gauss")
check("Gaussian fit, highest recovery on correlated laws", max(gauss), 0.15, 0.005, "step19_results.csv gauss")
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
check("all certified audit evaluations", n_corr + n_iid + len(tr) + len(rv), 1354, 0, "step19, step16 T=1000, training files")
check("estimates above the reference", above + fl_above + rv_above, 0, 0, "same files")
check("violated claims flagged", flags + fl_flags + rv_flags, 1176, 0, f"same files (of {n_corr + n_mk})")
check("violated claims tested", n_corr + n_mk, 1177, 0, "same files")
check("correct claims flagged", false + fl_false + rv_false, 0, 0, f"same files (of {n_iid + n_ii})")
check("correct claims tested", n_iid + n_ii, 177, 0, "same files")
# independent data sets: each simulated data set on a correlated law is certified at both delta (one audit stream per
# law and horizon: step19_rerun_audits.py seed 1919, step20b_parts.py seeds 4041/4042), so count the delta = 1e-4 rows only
s_corr = sum(int(r["R"]) for r in cert if float(r["delta"]) == 1e-4 and r["law"] != "iid") + sum(int(r["R"]) for r in t1k if float(r["delta"]) == 1e-4)
s_flags = sum(int(r["n_above_poisson"]) for r in cert if float(r["delta"]) == 1e-4 and r["law"] != "iid") + sum(int(r["n_above_poisson"]) for r in t1k if float(r["delta"]) == 1e-4)
check("independent audit data sets (abstract, Sections I, V-E, VII)", s_corr + n_iid + len(tr) + len(rv), 874, 0, "same files, delta = 1e-4 rows")
check("violated claims flagged, independent data sets", s_flags + fl_flags + rv_flags, 696, 0, f"same files (of {s_corr + n_mk})")
check("violated claims tested, independent data sets", s_corr + n_mk, 697, 0, "same files")
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

print(f"\n{'ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
