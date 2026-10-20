import csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("pilot_a_results.csv")))
rows = [r for r in rows if float(r["delta"]) == 1e-4]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, T in zip(axes, [200, 1000]):
    rr = [r for r in rows if int(r["T"]) == T]
    labels, full, ssum, pois = [], [], [], []
    order = [("minsep", 9), ("minsep", 5), ("minsep", 2), ("iid", 0.0), ("markov", 0.3), ("markov", 0.6), ("markov", 0.9), ("markov", 0.97), ("markov", 0.99)]
    for proc, par in order:
        m = [r for r in rr if r["process"] == proc and float(r["param"]) == float(par)]
        if not m:
            continue
        r = m[0]
        lab = {"iid": "i.i.d.", "markov": f"Markov λ={par}", "minsep": f"min-sep b={int(par)}"}[proc]
        labels.append(lab); full.append(float(r["eps_full_mc"])); ssum.append(float(r["eps_sum_exact"])); pois.append(float(r["eps_poisson_pld"]))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, full, 0.4, label="true ε (full release, Markov/min-sep)", color="#2b5797")
    ax.bar(x + 0.2, ssum, 0.4, label="ε of the sum statistic (final model)", color="#8fb3e0")
    ax.axhline(pois[0], color="#c0392b", ls="--", label=f"Poisson accounting at q=0.1 ({pois[0]:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_title(f"T={T}, q=0.1, σ=2, C=1, δ=1e-4"); ax.set_ylabel("ε(δ)")
    ax.set_ylim(0, min(max(full) * 1.08, 62))
    for xi, v in zip(x, full):
        ax.text(xi - 0.2, min(v, 60) + 0.3, ">60" if v >= 59.9 else f"{v:.1f}", ha="center", fontsize=7)
axes[0].legend(fontsize=7, loc="upper left")
plt.tight_layout(); plt.savefig("fig_pilot_a.png", dpi=160)
print("saved")
