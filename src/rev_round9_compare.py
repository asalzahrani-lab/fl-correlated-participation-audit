"""Revision round 9: compare the regenerated per-device bracket (rev_peruser_exact2.py, release v9: exact grouping of the
inclusion rates, optimistic PLD self-composed with tail_mass_truncation=0) with the release-v8 outputs (rates rounded to
9 decimals, library default truncation 1e-15 added to the infinity mass).  Prints the largest absolute change per
column and writes rev_round9_compare.csv.  The v8 outputs are read from rev_peruser_exact2_devices_v8.csv if present,
otherwise from the repository's tag v8 (git show v8:results/rev_peruser_exact2_devices.csv).  Run from results/ or src/."""
import csv, io, os, subprocess, numpy as np
RES = os.environ.get("P3_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
RES = RES if os.path.isdir(RES) else "."
p8 = os.path.normpath(os.path.join(RES, "rev_peruser_exact2_devices_v8.csv"))
if os.path.exists(p8):
    old_f = open(p8)
else:
    r = subprocess.run(["git", "show", "v8:results/rev_peruser_exact2_devices.csv"], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.abspath(__file__)))
    if r.returncode != 0:
        raise SystemExit("rev_round9_compare.py needs the release-v8 table: run it in a full clone with tag v8 (git fetch --tags; "
                         "a shallow clone or a zip download has no tags), or save results/rev_peruser_exact2_devices.csv from "
                         "release v8 as " + p8 + ".")
    old_f = io.StringIO(r.stdout)
old = {r["device"]: r for r in csv.DictReader(old_f)}
new = {r["device"]: r for r in csv.DictReader(open(os.path.join(RES, "rev_peruser_exact2_devices.csv")))}
rows = []
for c in ["pr_lo_0.0001", "pr_lo_1e-05", "pr_hi_0.0001", "pr_hi_1e-05", "pr_expw_lo_0.0001", "max_coarse_unrefined_0.0001", "thm3_0.0001", "thm3_1e-05", "fm_0.0001", "fm_1e-05"]:
    d = np.array([float(new[k][c]) - float(old[k][c]) for k in old])
    rows.append([c, f"{np.abs(d).max():.3e}", f"{d.mean():+.3e}"]); print(f"{c:28s} max |v9 - v8| {np.abs(d).max():.3e}  mean {d.mean():+.3e}")
same_w = sum(new[k]["w_pr_0.0001"] == old[k]["w_pr_0.0001"] for k in old); same_n = sum(new[k]["n_refined"] == old[k]["n_refined"] for k in old)
cert = all(new[k]["certified"] == "1" for k in new)
print(f"same worst window {same_w} of {len(old)}; same number of refined windows {same_n} of {len(old)}; all certificates hold: {cert}")
with open(os.path.join(RES, "rev_round9_compare.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["column", "max_abs_change", "mean_change"]); w.writerows(rows)
    w.writerow(["same_worst_window", same_w, len(old)]); w.writerow(["same_n_refined", same_n, len(old)]); w.writerow(["all_certified", int(cert), ""])
