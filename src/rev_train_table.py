"""LaTeX rows for the paper's training-diagnostics table, from rev_train_summary.csv."""
import csv
R = {r["config"]: r for r in csv.DictReader(open("rev_train_summary.csv"))}
LAB = {"base_orth": ("64", "200", "1,000", "orth.", "Markov"), "base_rand": ("64", "200", "1,000", "random", "Markov"),
       "ratio_1to2": ("64", "1,000", "1,000", "orth.", "Markov"), "ratio_1to10": ("64", "1,000", "200", "orth.", "Markov"),
       "medium": ("256--128", "500", "1,000", "orth.", "Markov"), "medium_iid": ("256--128", "500", "1,000", "orth.", "\\iid")}
for k in ["base_orth", "base_rand", "ratio_1to2", "ratio_1to10", "medium", "medium_iid"]:
    if k not in R: continue
    r = R[k]; h, n, m, dirs, law = LAB[k]; runs = int(r["runs"])
    flag = f"flags {r['flags']}/{runs}" if law != "\\iid" else f"flags {r['flags']}/{runs}"
    print(f"{h} & {n} & {m} & {float(r['canary_per_real']):.1f} & {dirs} & {law} & {runs} & ${float(r['cert']):.2f}\\pm{float(r['cert_sd']):.2f}$ ({float(r['recovery'])*100:.0f}\\%); {flag}; above {r['above']}/{runs} "
          f"& {float(r['coverage'])*100:.0f}\\% & {float(r['a_hat']):.3f} & {float(r['s_hat']):.3f} & {float(r['lag1']):.3f} & {float(r['xcorr_sd']):.4f} & {float(r['ks']):.4f} & ${float(r['erasure']):.2f}\\pm{float(r['erasure_sd']):.2f}$ & {r['acc'].replace('-', '--')} \\\\")
