"""Figures for the manuscript (matplotlib, print-oriented).
Fig. 1: regime map -- per-round ratio eps_true/eps_Poisson vs lambda for sigma in {1,2,4,8} (regime_map.log, T=200).
Fig. 2: horizon -- final-model ratio vs hours for the trace (10-min and 5-min runs) and the lambda=0.9 chain (T=200,1000).
Fig. 3: audits -- recovery (mean/sd) per auditor and law at delta 1e-4 (step19_results.csv)."""
import os, re, csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("P3_RESULTS", os.path.join(HERE, "..", "results"))   # stored outputs (results/ and results/logs/)
OUT = os.environ.get("P3_FIGDIR", HERE)                                   # figures are written next to this script


def src(name):
    for d in (RES, os.path.join(RES, "logs")):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    raise SystemExit(f"{name} not found in {RES} or {RES}/logs (set P3_RESULTS)")


C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 200,
                     "pdf.fonttype": 42, "ps.fonttype": 42})   # TrueType, not Type 3 (IEEE PDF requirements)

# ---------------- Fig 1: regime map
rows = []
for line in open(src("regime_map.log")):
    m = re.match(r"T=(\d+) sigma=([\d.]+) lam=([\d.]+): true=([\d.]+) poisson=([\d.]+) ratio=([\d.]+)", line)
    if m: rows.append(tuple(float(x) for x in m.groups()))
rows = np.array(rows)
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for i, sig in enumerate([1.0, 2.0, 4.0, 8.0]):
    sel = (rows[:, 0] == 200) & (rows[:, 1] == sig)
    lam, ratio = rows[sel, 2], rows[sel, 5]
    ax.plot(lam, ratio, "-o", color=C[i], lw=1.2, ms=3.5, label=f"$\\sigma={sig:g}$")
    ax.annotate(f"$\\sigma={sig:g}$", (lam[-1], ratio[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color="#0b0b0b", va="center")
sel = (rows[:, 0] == 1000) & (rows[:, 1] == 2.0)
r1000 = rows[sel, 5].copy(); r1000[rows[sel, 2] == 0.9] = 2.89   # 3M-sample value (step20b)
ax.plot(rows[sel, 2], r1000, "--s", color=C[1], lw=1.0, ms=3, label="$\\sigma=2$, $T=1{,}000$")
ax.axhline(1.0, color="#999", lw=0.6, ls=":")
ax.set_xlabel("lag-1 autocorrelation $\\lambda$ of participation")
ax.set_ylabel("$\\varepsilon_{\\rm true}/\\varepsilon_{\\rm Poisson}$ (per round)")
ax.set_xlim(-0.03, 1.08); ax.set_ylim(0.8, 6.6)
ax.legend(loc="upper left", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_regime.pdf")); plt.close(fig)

# ---------------- Fig 2: horizon
hz = list(csv.DictReader(open(src("step21_horizon.csv"))))
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for i, (rm, lab) in enumerate([("10", "trace, 10-min rounds, $K=27$"), ("5", "trace, 5-min rounds, $K=23$")]):
    h = np.array([float(r["hours"]) for r in hz if r["round_min"] == rm]); rt = np.array([float(r["ratio_1e-4"]) for r in hz if r["round_min"] == rm])
    ax.plot(h, rt, "-o", color=C[i], lw=1.2, ms=3.5, label=lab)
# chain: final-model ratio at T=200 (33.3 h at 10-min) and T=1000 (166.7 h) -> plot vs rounds converted at 10 min/round
ax.plot([200 * 10 / 60, 1000 * 10 / 60], [3.09, 1.73], "--s", color=C[2], lw=1.0, ms=3.5, label="two-state chain $\\lambda=0.9$ (10-min rounds)")
ax.axhline(1.0, color="#999", lw=0.6, ls=":")
ax.set_xlabel("training horizon (hours)")
ax.set_ylabel("final-model ratio, $\\delta=10^{-4}$")
ax.set_xlim(0, 175); ax.set_ylim(0, 10.5)
ax.legend(loc="upper right", fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_horizon.pdf")); plt.close(fig)

# ---------------- Fig 3: audit recovery
res = {(r["law"], r["estimator"]): r for r in csv.DictReader(open(src("step19_results.csv")))}
laws = [("markov", "Markov $\\lambda=0.9$"), ("iid", "i.i.d."), ("trace", "FLASH trace law")]
ests = [("cert_1e-4", "certified path-aware"), ("v2_1e-4", "uncertified path-aware"), ("countcond_1e-4", "count-conditioned"), ("gauss_1e-4", "Gaussian fit")]
fig, ax = plt.subplots(figsize=(3.4, 2.7))
w = 0.19; x = np.arange(len(laws))
for j, (ek, el) in enumerate(ests):
    means = [float(res[(lk, ek)]["mean"]) / float(res[(lk, ek)]["reference"]) for lk, _ in laws]
    sds = [float(res[(lk, ek)]["sd"]) / float(res[(lk, ek)]["reference"]) for lk, _ in laws]
    ax.bar(x + (j - 1.5) * w, means, w * 0.92, color=C[j], label=el, yerr=sds, error_kw={"lw": 0.6, "capsize": 1.5}, edgecolor="white", linewidth=0.5)
ax.axhline(1.0, color="#999", lw=0.6, ls=":")
# Poisson claim as fraction of truth
for i, (lk, _) in enumerate(laws):
    r = res[(lk, "cert_1e-4")]; frac = float(r["poisson"]) / float(r["reference"])
    ax.plot([i - 0.42, i + 0.42], [frac, frac], color="#0b0b0b", lw=0.8)
    txt = "Poisson claim = truth" if lk == "iid" else "Poisson claim"
    ax.annotate(txt, (i - 0.42, frac), xytext=(1, 2 if lk != "iid" else -8), textcoords="offset points", fontsize=5.5, ha="left", va="bottom")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in laws])
ax.set_ylabel("reported $\\hat\\varepsilon$ / $\\varepsilon_{\\rm true}$")
ax.set_ylim(0, 1.08)
ax.legend(loc="upper center", fontsize=6, ncol=2, bbox_to_anchor=(0.5, -0.16))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_audits.pdf")); plt.close(fig)
print("figures written")
