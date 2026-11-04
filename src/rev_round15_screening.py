"""Round 15: the noise-error audits before and after the diagnostic rule, with the scale alarm separated from the
privacy report; the residual cells under the same rules; the training runs as a transfer test.

Inputs (results/): rev_round15_records.csv (one row per audit: the noise-error cells, the residual cells and the
independent null calibration cells; rev_round15_records.py), rev_round14_residual_rule.csv (the rule frozen in round
14), rev_round14_noise_sim.csv and rev_round14_residual_sim.csv (the round-14 aggregates, reproduced here from the
rows), rev_round15_intervals.csv (independent 99% intervals of the realised loss; rev_round15_intervals.py),
rev_round15_train_records_*.csv and rev_round15_cnn_records_*.csv (the training runs with their full records).

Rules.  Every rule is a fixed function of the diagnostics of one audit, with thresholds frozen on a null cell:
  A      the round-14 rule as frozen: s_hat, lag1_200, mean_a and erasure within the null mean +- 4 sd, ks at most
         the null maximum (thresholds recomputed at full precision from the regenerated null cell, residual cell 0,
         which is the same data as in round 14; the rounded file rev_round14_residual_rule.csv is checked against it);
  B      the scale statistic s_hat taken out of the rejection and reported as a separate SCALE ALARM (s_hat outside
         its rule-A band); the shape diagnostics decide the abstention: lag1_200, mean_a_norm = mean_a / s_hat and
         erasure within the null mean +- 4 sd, ks (already computed on (x - mean) / s_hat) at most the null maximum --
         frozen on the same null cell (residual cell 0), evaluated on cells that did not set it;
  A_cal, B_cal   the same two forms with thresholds frozen on the independent null cells of rev_round15_records.py
         (seeds 15000 to 15003, one per (law, m) configuration), so that a configuration other than (Markov, 1,000)
         is screened by a rule calibrated for it rather than transferred.
For each cell and rule the four counts of the plan: A (raw flag, passed), B (raw flag, rejected), C (no flag, passed),
D (no flag, rejected); raw flags = A + B, flags after screening = A, abstentions = B + D; Wilson 95% intervals for
the raw rate, the after-screening rate and the withheld-flag rate B / N; the reason counts (which statistics failed;
more than one may fail in one audit).  The raw flag is the strict exceedance of the claim by the full-precision
certificate; the flag of the documented upper limit (12.033 for the Markov law) is counted alongside.  Coverage is
classed against the point reference of round 14 (above / not) and against the independent interval (above its upper
end / inside / below its lower end).  Both estimators (eq. (11) and the centred variant) are screened.

The training runs (Fashion-MNIST, 10 per cell; CIFAR-10, 2 to 5) are screened by the rules frozen on the simulation's
null cells: a TRANSFER test, since no training null calibrated them (the statistics are defined identically; the
training erasure is the ratio of the mean present-canary sum score to its expected value, as in the simulation).

Checks (every one aborts the script on failure): the per-cell aggregates of the rows reproduce rev_round14_noise_sim.csv
(mean, sd, flags, exceedances) and rev_round14_residual_sim.csv (mean, sd, rejected, exceedances among all and among
passers, the centred columns); a_lo, the centred a_lo, s_hat, mean_a and erasure of every row are recomputed from its
sufficient statistics; the Fashion-MNIST training records reproduce rev_round14_train_*.csv on every shared column and
the CNN records reproduce rev_round14_cnn_*.csv on a_lo, s_hat, lag-1, the certificate and every flag, within 0.003 on
accuracy and within five units of the last printed digit on erasure, KS, the uncertified value and the correlation
sample (PyTorch's CPU reductions are not bit-reproducible); no audit id is missing or duplicated.

    cd results && python3 ../src/rev_round15_screening.py
Outputs rev_round15_screening_noise.csv, rev_round15_screening_residual.csv, rev_round15_screening_training.csv,
rev_round15_rules.csv (every threshold used), and a log of the summary on stdout.
"""
import csv, glob, os, sys
import numpy as np
from scipy.stats import norm, chi2

T, A_TRUE, A_EST = 200, 0.5, 0.045
KEYS_A = ("s_hat", "lag1_200", "mean_a", "erasure")
KEYS_B = ("lag1_200", "mean_a_norm", "erasure")
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg); print("CHECK FAILED: " + msg, flush=True)


def wilson(k, n, z=1.959964):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt_ci(k, n):
    lo, hi = wilson(k, n); return f"{lo:.3f}-{hi:.3f}"


def read(name):
    return list(csv.DictReader(open(name)))


def f(x): return float(x)


def recompute(r):
    """a_lo, centred a_lo, s_hat, mean_a, erasure from the sufficient statistics of one row."""
    n0 = int(r["n0"]); sum_a = f(r["sum_a"]); sumsq = f(r["sumsq_a"]); n_obs = int(r["n_obs"]); sum_obs = f(r["sum_obs"])
    mean_a = sum_a / n0; var_pop = sumsq / n0 - mean_a * mean_a; s_hat = np.sqrt(max(var_pop, 0.0))
    sd = np.sqrt(max((sumsq - n0 * mean_a * mean_a) / (n0 - 1), 0.0)); sd_hi = sd * np.sqrt((n0 - 1) / chi2.ppf(A_EST / 2, n0 - 1)); z = norm.ppf(1 - A_EST / 2)
    shift = sum_obs / n_obs if n_obs > 0 else 0.0
    a_lo = max(shift - z * sd_hi / np.sqrt(n_obs), 0.0) / sd_hi if n_obs > 0 else 0.0
    a_lo_c = max(shift - mean_a - z * sd_hi * np.sqrt(1.0 / n_obs + 1.0 / n0), 0.0) / sd_hi if n_obs > 0 else 0.0
    n_pres = int(r["n_pres"]); er = (f(r["sum_S_pres"]) / n_pres) / (A_TRUE * (int(r["sum_counts_pres"]) / n_pres) / np.sqrt(T))
    return a_lo, a_lo_c, s_hat, mean_a, er


def freeze(rows, keys, ks_key="ks"):
    """Thresholds from a null cell: (mean, sd, lower, upper) per key at +- 4 sd, and the KS maximum."""
    rule = {}
    for k in keys:
        v = np.array([f(r[k]) for r in rows]); rule[k] = (v.mean(), v.std(), v.mean() - 4 * v.std(), v.mean() + 4 * v.std())
    rule["ks_max"] = max(f(r[ks_key]) for r in rows)
    return rule


def apply(rule, keys, r):
    """Returns the list of failed statistics (empty = passed)."""
    bad = [k for k in keys if not (rule[k][2] <= f(r[k]) <= rule[k][3])]
    if f(r["ks"]) > rule["ks_max"]: bad.append("ks")
    return bad


def abcd(rows, passed, flagkey):
    fl = np.array([int(r[flagkey]) for r in rows], bool); ps = np.asarray(passed, bool)
    return int((fl & ps).sum()), int((fl & ~ps).sum()), int((~fl & ps).sum()), int((~fl & ~ps).sum())


if __name__ == "__main__":
    rec = read("rev_round15_records.csv")
    ids = [r["audit_id"] for r in rec]; check(len(ids) == len(set(ids)), "duplicate audit ids")
    noise = [r for r in rec if r["experiment"] == "noise"]; resid = [r for r in rec if r["experiment"] == "residual"]; null = [r for r in rec if r["experiment"] == "null"]
    check(len(noise) == 2880, f"noise rows {len(noise)}"); check(len(resid) == 3800, f"residual rows {len(resid)}"); check(len(null) == 800, f"null rows {len(null)}")
    for cellset, R in ((noise, 120), (resid, 200), (null, 200)):
        for c in sorted({r["cell"] for r in cellset}):
            reps = sorted(int(r["rep"]) for r in cellset if r["cell"] == c); check(reps == list(range(R)), f"cell {cellset[0]['experiment']} {c}: reps {reps[:3]}... ({len(reps)})")

    # ---- sufficient-statistics recomputation on every row
    worst = 0.0
    for r in rec:
        a_lo, a_lo_c, s_hat, mean_a, er = recompute(r)
        for name, got, want in (("a_lo", a_lo, f(r["a_lo"])), ("a_lo_c", a_lo_c, f(r["a_lo_c"])), ("s_hat", s_hat, f(r["s_hat"])), ("mean_a", mean_a, f(r["mean_a"])), ("erasure", er, f(r["erasure"]))):
            worst = max(worst, abs(got - want)); check(abs(got - want) <= 1e-7 * max(1.0, abs(want)), f"{r['audit_id']}: {name} recomputed {got} vs stored {want}")
    print(f"sufficient statistics: {len(rec)} rows recomputed, largest absolute discrepancy {worst:.2e}", flush=True)

    # ---- reproduction of the round-14 aggregates
    r14 = {(r["law"], int(r["m"]), f(r["noise_r"])): r for r in read("rev_round14_noise_sim.csv")}
    cells_n = sorted({(r["law"], int(r["m"]), f(r["noise_r"]), int(r["cell"])) for r in noise}, key=lambda x: x[3])
    for law, m, r_, c in cells_n:
        rows = [x for x in noise if int(x["cell"]) == c]; e = np.array([f(x["cert"]) for x in rows]); old = r14[(law, m, r_)]
        claim = f(rows[0]["claim"]); ref = f(rows[0]["reference"])
        check(abs(e.mean() - f(old["mean_cert"])) < 5e-5 and abs(e.std() - f(old["sd_cert"])) < 5e-5, f"noise cell {c}: mean/sd {e.mean():.4f}/{e.std():.4f} vs v14 {old['mean_cert']}/{old['sd_cert']}")
        check(int((e > claim).sum()) == int(old["flags"]) and int((e > ref).sum()) == int(old["above_realised"]), f"noise cell {c}: flags/above {int((e > claim).sum())}/{int((e > ref).sum())} vs v14 {old['flags']}/{old['above_realised']}")
    print("round-14 noise aggregates reproduced in all 24 cells", flush=True)
    r14r = {(r["class"], f(r["rho"]), f(r["phi"])): r for r in read("rev_round14_residual_sim.csv")}
    null0 = [x for x in resid if int(x["cell"]) == 0]
    ruleA = freeze(null0, KEYS_A); ruleB = freeze(null0, KEYS_B)
    filed = {{"lag1": "lag1_200"}.get(r["statistic"], r["statistic"]): r for r in read("rev_round14_residual_rule.csv")}
    for k in KEYS_A:
        check(abs(ruleA[k][0] - f(filed[k]["null_mean"])) < 1e-5 and abs(ruleA[k][1] - f(filed[k]["null_sd"])) < 1e-5, f"rule A {k}: {ruleA[k][:2]} vs file {filed[k]['null_mean']}, {filed[k]['null_sd']}")
    check(abs(ruleA["ks_max"] - f(filed["ks"]["upper"])) < 1e-5, f"rule A ks max {ruleA['ks_max']} vs file {filed['ks']['upper']}")
    n_margin = 0
    for x in resid + noise:
        full = len(apply(ruleA, KEYS_A, x)) == 0
        rounded = all(f(filed[k]["lower"]) <= f(x[k]) <= f(filed[k]["upper"]) for k in KEYS_A) and f(x["ks"]) <= f(filed["ks"]["upper"])
        n_margin += full != rounded
    print(f"rule A at full precision vs the rounded file: {n_margin} of {len(resid) + len(noise)} audits decided differently", flush=True)
    cells_r = sorted({(r["residual"], f(r["rho"]), f(r["phi"]), int(r["cell"])) for r in resid}, key=lambda x: x[3])
    for kind, rho, phi, c in cells_r:
        rows = [x for x in resid if int(x["cell"]) == c]; e = np.array([f(x["cert"]) for x in rows]); ec = np.array([f(x["cert_c"]) for x in rows]); old = r14r[(kind, rho, phi)]
        ref = f(rows[0]["reference"]); passed = np.array([len(apply(ruleA, KEYS_A, x)) == 0 for x in rows])
        check(abs(e.mean() - f(old["mean_cert"])) < 5e-5 and abs(e.std() - f(old["sd_cert"])) < 5e-5, f"residual cell {c}: mean/sd vs v14")
        check(int((~passed).sum()) == int(old["rejected"]), f"residual cell {c} {kind} {rho}: rejected {int((~passed).sum())} vs v14 {old['rejected']}")
        check(int((e > ref).sum()) == int(old["above_ref_all"]) and int((e[passed] > ref).sum()) == int(old["above_ref_passed"]), f"residual cell {c}: exceedances vs v14")
        check(int((e > 11.869).sum()) == int(old["above_nominal_all"]) and int((e[passed] > 11.869).sum()) == int(old["above_nominal_passed"]), f"residual cell {c}: nominal exceedances vs v14")
        check(abs(ec.mean() - f(old["mean_cert_centred"])) < 5e-5 and int((ec > ref).sum()) == int(old["above_ref_centred"]), f"residual cell {c}: centred vs v14")
    print("round-14 residual aggregates reproduced in all 19 cells (rule A at full precision)", flush=True)

    # ---- calibrated rules from the independent null cells
    cal = {}
    for k in sorted({int(r["cell"]) for r in null}):
        rows = [x for x in null if int(x["cell"]) == k]; cfg = (rows[0]["law"], int(rows[0]["m"]))
        cal[cfg] = (freeze(rows, KEYS_A), freeze(rows, KEYS_B))
    with open("rev_round15_rules.csv", "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["rule", "calibration", "law", "m", "statistic", "null_mean", "null_sd", "lower", "upper"])
        for name, rule, keys in (("A", ruleA, KEYS_A), ("B", ruleB, KEYS_B)):
            for k in keys: w.writerow([name, "round-14 null (residual cell 0, seed 14700)", "markov", 1000, k, f"{rule[k][0]:.6f}", f"{rule[k][1]:.6f}", f"{rule[k][2]:.6f}", f"{rule[k][3]:.6f}"])
            w.writerow([name, "round-14 null (residual cell 0, seed 14700)", "markov", 1000, "ks", "", "", "", f"{rule['ks_max']:.6f}"])
        for cfg, (ra, rb) in cal.items():
            for name, rule, keys in (("A_cal", ra, KEYS_A), ("B_cal", rb, KEYS_B)):
                for k in keys: w.writerow([name, "independent null (seeds 15000+)", cfg[0], cfg[1], k, f"{rule[k][0]:.6f}", f"{rule[k][1]:.6f}", f"{rule[k][2]:.6f}", f"{rule[k][3]:.6f}"])
                w.writerow([name, "independent null (seeds 15000+)", cfg[0], cfg[1], "ks", "", "", "", f"{rule['ks_max']:.6f}"])
    ra_m, rb_m = cal[("markov", 1000)]
    print("rule A (round-14 null): " + ", ".join(f"{k} {ruleA[k][0]:.5f}+-{ruleA[k][1]:.5f}" for k in KEYS_A) + f", ks max {ruleA['ks_max']:.5f}")
    print("rule A_cal (Markov 1,000, independent null): " + ", ".join(f"{k} {ra_m[k][0]:.5f}+-{ra_m[k][1]:.5f}" for k in KEYS_A) + f", ks max {ra_m['ks_max']:.5f}")
    for cfg, (ra, rb) in cal.items():
        print(f"  null {cfg}: s_hat sd {ra['s_hat'][1]:.5f}, mean_a sd {ra['mean_a'][1]:.5f}, lag1 sd {ra['lag1_200'][1]:.5f}, erasure sd {ra['erasure'][1]:.4f}, ks max {ra['ks_max']:.5f}")

    # ---- the noise cells
    iv = {}
    if os.path.exists("rev_round15_intervals.csv"):
        for r in read("rev_round15_intervals.csv"): iv[(r["law"], f(r["noise_r"]))] = (f(r["eps"]), f(r["lo99"]), f(r["hi99"]))
    rules_all = {"A": (ruleA, KEYS_A), "B": (ruleB, KEYS_B)}
    out = []
    print("\nNOISE CELLS (per rule: A = flag kept, B = flag withheld, C = no flag reported, D = withheld without flag; reasons in brackets)", flush=True)
    for law, m, r_, c in cells_n:
        rows = [x for x in noise if int(x["cell"]) == c]; N = len(rows); claim = f(rows[0]["claim"]); ref = f(rows[0]["reference"])
        e = np.array([f(x["cert"]) for x in rows]); ec = np.array([f(x["cert_c"]) for x in rows])
        scale_alarm = int(sum(not (ruleA["s_hat"][2] <= f(x["s_hat"]) <= ruleA["s_hat"][3]) for x in rows))
        scale_alarm_cal = int(sum(not (cal[(law, m)][0]["s_hat"][2] <= f(x["s_hat"]) <= cal[(law, m)][0]["s_hat"][3]) for x in rows))
        raw = int((e > claim).sum()); raw_up = int((e > f(rows[0]["claim_upper"])).sum()) if rows[0]["claim_upper"] not in ("nan", "") else ""
        raw_c = int((ec > claim).sum())
        above_pt = int((e > ref).sum()); above_pt_c = int((ec > ref).sum())
        if (law, r_) in iv:
            pe, lo, hi = iv[(law, r_)]; above_iv = int((e > hi).sum()); undec = int(((e >= lo) & (e <= hi)).sum()); above_iv_c = int((ec > hi).sum()); undec_c = int(((ec >= lo) & (ec <= hi)).sum())
        else: pe = lo = hi = float("nan"); above_iv = undec = above_iv_c = undec_c = ""
        line = f"{law:6s} m={m:4d} r={r_:.2f}: claim {claim:.3f}, realised {ref:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if (law, r_) in iv else "") + f"; raw flags {raw}/{N} (upper limit {raw_up}); centred raw {raw_c}; scale alarms {scale_alarm} (calibrated {scale_alarm_cal}); above point {above_pt}" + (f", above interval {above_iv}, inside {undec}" if (law, r_) in iv else "")
        for rname in ("A", "B", "A_cal", "B_cal"):
            rule, keys = rules_all[rname] if rname in rules_all else ((cal[(law, m)][0], KEYS_A) if rname == "A_cal" else (cal[(law, m)][1], KEYS_B))
            bads = [apply(rule, keys, x) for x in rows]; passed = [len(b) == 0 for b in bads]
            reasons = {k: sum(k in b for b in bads) for k in list(keys) + ["ks"]}
            for est, flagkey in (("eq11", "flag_claim"), ("centred", None)):
                if est == "centred":
                    fl = (ec > claim); ps = np.array(passed); a_, b_, c_, d_ = int((fl & ps).sum()), int((fl & ~ps).sum()), int((~fl & ps).sum()), int((~fl & ~ps).sum())
                else: a_, b_, c_, d_ = abcd(rows, passed, flagkey)
                out.append([law, m, f"{r_:.2f}", c, N, f"{claim:.3f}", f"{ref:.3f}", f"{lo:.3f}" if (law, r_) in iv else "", f"{hi:.3f}" if (law, r_) in iv else "", rname, est,
                            a_ + b_, fmt_ci(a_ + b_, N), a_, fmt_ci(a_, N), b_, fmt_ci(b_, N), c_, d_, b_ + d_, scale_alarm if rname in ("A", "B") else scale_alarm_cal,
                            raw_up if est == "eq11" else "", above_pt if est == "eq11" else above_pt_c, above_iv if est == "eq11" else above_iv_c, undec if est == "eq11" else undec_c,
                            " ".join(f"{k}:{v}" for k, v in reasons.items() if v)])
            a1, b1, c1, d1 = abcd(rows, passed, "flag_claim")
            line += f"\n      rule {rname:5s}: eq. 11 A {a1:3d} B {b1:3d} C {c1:3d} D {d1:3d} (after screening {a1}/{N}); centred A {a_:3d} B {b_:3d} C {c_:3d} D {d_:3d} [" + ", ".join(f"{k} {v}" for k, v in reasons.items() if v) + "]"
        print(line, flush=True)
    with open("rev_round15_screening_noise.csv", "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["law", "m", "noise_r", "cell", "N", "claim", "realised_point", "realised_lo99", "realised_hi99", "rule", "estimator", "raw_flags", "raw_rate_ci95",
                                        "A_flag_kept", "after_rate_ci95", "B_flag_withheld", "withheld_rate_ci95", "C_no_flag", "D_withheld_no_flag", "abstentions", "scale_alarms",
                                        "raw_flags_upper_limit", "above_point", "above_interval", "inside_interval", "reasons"]); w.writerows(out)

    # ---- the residual cells
    out = []
    print("\nRESIDUAL CELLS (Markov, m = 1,000; rules A and B frozen on the null cell; A_cal / B_cal on the independent Markov-1,000 null)", flush=True)
    for kind, rho, phi, c in cells_r:
        rows = [x for x in resid if int(x["cell"]) == c]; N = len(rows); ref = f(rows[0]["reference"])
        e = np.array([f(x["cert"]) for x in rows]); ec = np.array([f(x["cert_c"]) for x in rows])
        scale_alarm = int(sum(not (ruleA["s_hat"][2] <= f(x["s_hat"]) <= ruleA["s_hat"][3]) for x in rows))
        line = f"{kind:9s} rho={rho:.2f} phi={phi:.1f}: ref {ref:.3f}; eq.11 {e.mean():.3f} above {int((e > ref).sum())}; centred {ec.mean():.3f} above {int((ec > ref).sum())}; scale alarms {scale_alarm}"
        for rname, (rule, keys) in (("A", (ruleA, KEYS_A)), ("B", (ruleB, KEYS_B)), ("A_cal", (ra_m, KEYS_A)), ("B_cal", (rb_m, KEYS_B))):
            bads = [apply(rule, keys, x) for x in rows]; ps = np.array([len(b) == 0 for b in bads]); reasons = {k: sum(k in b for b in bads) for k in list(keys) + ["ks"]}
            for est, v in (("eq11", e), ("centred", ec)):
                out.append([kind, rho, phi, c, N, f"{ref:.3f}", rname, est, int((~ps).sum()), int(ps.sum()), int((v > ref).sum()), int((v[ps] > ref).sum()), int((v > 11.869).sum()), int((v[ps] > 11.869).sum()),
                            f"{v.mean():.4f}", f"{v[ps].mean():.4f}" if ps.any() else "", scale_alarm, " ".join(f"{k}:{n} " for k, n in reasons.items() if n)])
            line += f"\n      rule {rname:5s}: rejected {int((~ps).sum()):3d}; above ref among passers eq.11 {int((e[ps] > ref).sum())}, centred {int((ec[ps] > ref).sum())} [" + ", ".join(f"{k} {n}" for k, n in reasons.items() if n) + "]"
        print(line, flush=True)
    with open("rev_round15_screening_residual.csv", "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["class", "rho", "phi", "cell", "N", "reference", "rule", "estimator", "rejected", "passed", "above_ref_all", "above_ref_passed", "above_nominal_all",
                                        "above_nominal_passed", "mean_cert_all", "mean_cert_passed", "scale_alarms", "reasons"]); w.writerows(out)

    # ---- the training runs (transfer test)
    out = []
    print("\nTRAINING RUNS (rules transferred from the simulation null cells; mean_a from the round-15 records)", flush=True)
    files = sorted(glob.glob("rev_round15_train_records_*.csv")) + sorted(glob.glob("rev_round15_cnn_records_*.csv"))
    for path in files:
        rows = read(path)
        if not rows: print(f"{path}: no rows yet (skipped)", flush=True); continue
        name = rows[0]["config"]; N = len(rows); law = rows[0]["law"]
        old_path = path.replace("rev_round15_train_records_", "rev_round14_train_").replace("rev_round15_cnn_records_", "rev_round14_cnn_")
        old = read(old_path) if os.path.exists(old_path) else []
        shared = [k for k in old[0].keys() if k in rows[0]] if old else []
        exact_cols = ("config", "rep", "d", "n_clients", "m_canary", "law", "partition", "noise_r", "C_clip", "a_lo", "covered_nominal", "a_hat", "s_hat", "lag1", "res_part", "res_nonpart", "cert", "claim", "true_loss_r", "poisson", "above_true_r", "flag_claim", "flag_poisson", "clip_frac")
        def tol(k, v):                                                  # accuracy within 0.003; otherwise five units of the last printed decimal
            return 0.003 if k == "acc" else 5 * 10.0 ** -(len(v.split(".")[1]) if "." in v else 0)
        def same(o, r, k):
            if k in exact_cols or "cnn" not in path: return o[k] == r[k]
            try: return abs(float(o[k]) - float(r[k])) <= tol(k, o[k]) + 1e-12
            except ValueError: return o[k] == r[k]
        n_same = sum(all(o[k] == r[k] for k in shared) for o, r in zip(old, rows)) if old else 0
        n_audit = sum(all(same(o, r, k) for k in shared) for o, r in zip(old, rows)) if old else 0
        partial = "cnn" in path and len(rows) < len(old)
        # the Fashion-MNIST records must be identical on every shared column; the CNN records (PyTorch) must be identical on
        # a_lo, s_hat, lag-1, the certificate and every flag, within 0.003 on accuracy and within five units of the last
        # printed digit on erasure, KS, the uncertified value and the 200-canary correlation sample
        check(len(old) == 0 or (n_audit == len(rows) and (len(rows) == len(old) or partial)), f"{name}: {n_audit} of {len(rows)} records match {old_path} ({len(old)} rows)")
        print(f"{name}: {n_same} of {len(rows)} records identical to {os.path.basename(old_path)} on all {len(shared)} shared columns" + (f", {n_audit} within the tolerances on the non-decision columns" if "cnn" in path else "") + (f" ({len(old) - len(rows)} runs still to regenerate)" if partial else ""), flush=True)
        claim = f(rows[0]["claim"]); e = np.array([f(x["cert"]) for x in rows]); ec = np.array([f(x["cert_c"]) for x in rows]); r_ = f(rows[0]["noise_r"])
        # the training lag1 of the rule's definition is lag1_200; s_hat, mean_a, ks, erasure as stored
        cfg = (law, 1000)
        scale_alarm = int(sum(not (ruleA["s_hat"][2] <= f(x["s_hat"]) <= ruleA["s_hat"][3]) for x in rows))
        line = f"{name:18s} N={N} r={r_:.2f} claim {claim:.3f}: raw flags {int((e > claim).sum())}; centred raw {int((ec > claim).sum())}; scale alarms {scale_alarm}; mean_a {np.mean([f(x['mean_a']) for x in rows]):+.5f}"
        for rname, (rule, keys) in (("A", (ruleA, KEYS_A)), ("B", (ruleB, KEYS_B)), ("A_cal", (cal[cfg][0], KEYS_A)), ("B_cal", (cal[cfg][1], KEYS_B))):
            bads = [apply(rule, keys, x) for x in rows]; ps = np.array([len(b) == 0 for b in bads]); reasons = {k: sum(k in b for b in bads) for k in list(keys) + ["ks"]}
            for est, v in (("eq11", e), ("centred", ec)):
                fl = v > claim; a_, b_, c_, d_ = int((fl & ps).sum()), int((fl & ~ps).sum()), int((~fl & ps).sum()), int((~fl & ~ps).sum())
                out.append([name, law, f"{r_:.2f}", N, f"{claim:.3f}", rname, est, a_ + b_, a_, b_, c_, d_, b_ + d_, scale_alarm, " ".join(f"{k}:{n}" for k, n in reasons.items() if n)])
            fl = e > claim; a_, b_ = int((fl & ps).sum()), int((fl & ~ps).sum())
            line += f"\n      rule {rname:5s}: A {a_} B {b_} C {int((~fl & ps).sum())} D {int((~fl & ~ps).sum())} [" + ", ".join(f"{k} {n}" for k, n in reasons.items() if n) + "]"
        print(line, flush=True)
    with open("rev_round15_screening_training.csv", "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["config", "law", "noise_r", "N", "claim", "rule", "estimator", "raw_flags", "A_flag_kept", "B_flag_withheld", "C_no_flag", "D_withheld_no_flag", "abstentions", "scale_alarms", "reasons"]); w.writerows(out)
    print(f"\n{len(fails)} check(s) failed" if fails else "\nall checks passed", flush=True)
    sys.exit(1 if fails else 0)
