"""Per-device table of the paper (ever-available devices of the running example). Reads rev_peruser_devices.csv.
Output rev_peruser_table.csv."""
import csv
import numpy as np
rows = list(csv.DictReader(open("rev_peruser_devices.csv")))
ever = np.array([float(r["avail"]) > 0 for r in rows])
REF = {("fm", 1e-4): 1.134, ("fm", 1e-5): 1.360, ("pr", 1e-4): 1.219, ("pr", 1e-5): 1.462}
pooled = {("fm", 1e-4): 5.841, ("fm", 1e-5): 8.50, ("pr", 1e-4): 10.69, ("pr", 1e-5): 14.97}
out = []
for col, kind, label in [("a_fm", "fm", "(a) model known, final model"), ("emp_fm", "fm", "(a') own window law, final model"),
                         ("b_fm", "fm", "(b) availability known, final model"), ("a_thm3", "pr", "(a) Thm 3 / joint release, per round"),
                         ("b_pr", "pr", "(b) availability known, per round")]:
    for d in (1e-4, 1e-5):
        v = np.array([float(r[f"{col}_{d:g}"]) for r in rows])[ever]
        ref = REF[(kind, d)]
        out.append([label, d, len(v), f"{np.median(v):.2f}", f"{np.quantile(v, .9):.2f}", f"{np.quantile(v, .99):.2f}", f"{v.max():.2f}",
                     f"{ref:.2f}", f"{v.max()/ref:.1f}", f"{(v > ref).mean():.3f}", f"{(v > pooled[(kind, d)]).mean():.3f}", pooled[(kind, d)]])
        print(" | ".join(map(str, out[-1])))
with open("rev_peruser_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["law", "delta", "devices", "median", "p90", "p99", "max", "poisson_or_binomial_ref", "max_over_ref", "share_above_ref", "share_above_pooled", "pooled"]); w.writerows(out)
r36 = [r for r in rows if r["device"] == "36"][0]
print("device 36:", {k: r36[k] for k in ["avail", "part_rate", "s_i", "a_fm_0.0001", "emp_fm_0.0001", "b_fm_0.0001", "b_pr_0.0001", "b_expected_count"]})
