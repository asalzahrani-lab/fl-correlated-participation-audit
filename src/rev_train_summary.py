"""Summary of the revision training runs (rev_train_*.csv) for the paper's training table and diagnostics paragraph.
Output rev_train_summary.csv (one row per configuration plus pooled rows)."""
import csv, glob
import numpy as np

ORDER = ["base_orth", "base_rand", "ratio_1to2", "ratio_1to10", "medium", "medium_iid"]
DESC = {"base_orth": "784-64-10, 200 clients, 1,000 canaries, orthogonal", "base_rand": "784-64-10, 200 clients, 1,000 canaries, random",
        "ratio_1to2": "784-64-10, 1,000 clients, 1,000 canaries", "ratio_1to10": "784-64-10, 1,000 clients, 200 canaries",
        "medium": "784-256-128-10, 500 clients, 1,000 canaries", "medium_iid": "784-256-128-10, 500 clients, 1,000 canaries, iid"}
rows = {}
for f in glob.glob("rev_train_*.csv"):
    if f.endswith("summary.csv"): continue
    for r in csv.DictReader(open(f)):
        rows.setdefault(r["config"], []).append(r)
out = []
def summ(name, R):
    g = lambda k: np.array([float(r[k]) for r in R])
    cert, unc, ref, pois = g("cert"), g("uncert"), g("ref"), g("poisson")
    n_real, m = float(R[0]["n_clients"]), float(R[0]["m_canary"])
    ratio = (m * 0.5 * 0.1) / (n_real * 0.1)          # present canaries per round / real clients per round
    viol = ref[0] > pois[0] * 1.01
    d = dict(config=name, desc=DESC.get(name, name), law=R[0]["law"], runs=len(R), d=int(R[0]["d"]), canary_per_real=f"{ratio:.2f}",
             cert=f"{cert.mean():.2f}", cert_sd=f"{cert.std():.2f}", recovery=f"{cert.mean()/ref[0]:.3f}",
             flags=int((cert > pois).sum()), above=int((cert > ref).sum()), unc=f"{unc.mean():.2f}", unc_recovery=f"{unc.mean()/ref[0]:.3f}",
             unc_above=int((unc > ref).sum()), unc_flags=int((unc > pois).sum()), violated=int(viol),
             coverage=f"{g('covered').mean():.3f}", a_lo=f"{g('a_lo').mean():.4f}", a_hat=f"{g('a_hat').mean():.4f}", a_hat_sd=f"{g('a_hat').std():.4f}",
             s_hat=f"{g('s_hat').mean():.4f}", s_hat_range=f"{g('s_hat').min():.3f}-{g('s_hat').max():.3f}", lag1=f"{g('lag1').mean():.4f}",
             xcorr_mean=f"{g('xcorr_mean').mean():.5f}", xcorr_sd=f"{g('xcorr_sd').mean():.4f}", xcorr_maxabs=f"{g('xcorr_maxabs').max():.3f}",
             ks=f"{g('ks').mean():.4f}", ks_max=f"{g('ks').max():.4f}", res_nonpart=f"{g('res_nonpart').mean():.4f}",
             erasure=f"{g('erasure').mean():.3f}", erasure_sd=f"{g('erasure').std():.3f}", erasure_range=f"{g('erasure').min():.2f}-{g('erasure').max():.2f}",
             acc=f"{g('acc').min():.3f}-{g('acc').max():.3f}", clip=f"{g('clip_frac').mean():.2f}", zero_cert=int((cert == 0).sum()))
    out.append(d)
    print(f"{name:12s} runs {d['runs']:3d} c/r {d['canary_per_real']} cert {d['cert']}±{d['cert_sd']} ({float(d['recovery']):.0%}) flags {d['flags']}/{d['runs']} above {d['above']} "
          f"unc {d['unc']} ({float(d['unc_recovery']):.0%}) above {d['unc_above']} | cover {d['coverage']} a_lo {d['a_lo']} a_hat {d['a_hat']}±{d['a_hat_sd']} s_hat {d['s_hat']} "
          f"lag1 {d['lag1']} xcorr {d['xcorr_mean']}±{d['xcorr_sd']} max {d['xcorr_maxabs']} ks {d['ks']} (max {d['ks_max']}) erasure {d['erasure']}±{d['erasure_sd']} [{d['erasure_range']}] acc {d['acc']} clip {d['clip']} zero {d['zero_cert']}")
for name in ORDER:
    if name in rows: summ(name, rows[name])
mk = [r for n in ORDER if n in rows and n != "medium_iid" for r in rows[n]]
if mk: summ("all_markov", mk)
allr = [r for n in ORDER if n in rows for r in rows[n]]
cov = np.mean([float(r["covered"]) for r in allr]) if allr else float("nan")
print(f"all runs: {len(allr)}; coverage of a_lo <= a: {cov:.3f}")
with open("rev_train_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); [w.writerow(d) for d in out]
