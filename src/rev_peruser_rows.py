"""Round 3: regenerate the position-(b) rows of Tables XI / S5 / TIFS VI and the bracket column of Tables XII / S6 from
rev_peruser_exact2_*.csv (lower end = optimistic PLD estimate), and, with --patch FILE..., patch the given manuscript .tex files in place.
Also fills the [[GAPMAX]] and [[NREF]] placeholders."""
import csv, re, sys
import numpy as np

dev = {int(r["device"]): r for r in csv.DictReader(open("rev_peruser_exact2_devices.csv"))}
summ = {(r["quantity"], float(r["delta"])): r for r in csv.DictReader(open("rev_peruser_exact2_summary.csv"))}
ever = [r for r in dev.values() if float(r["avail"]) > 0]
pooled = {1e-4: 10.69, 1e-5: 14.97}; pooled_fm = {1e-4: 5.841, 1e-5: 8.496}   # running example final-model values (step21b_policies.csv)


def row(label, col, d):
    s = summ[(col, d)]; vals = np.array([float(r[f"{col}_{d:g}"]) for r in ever])
    share_pooled = (vals > (pooled_fm if col == "fm" else pooled)[d]).mean()
    return (f"(b) availability, {label} & {float(s['median']):.2f} & {float(s['p90']):.2f} & {float(s['max']):.2f} & "
            f"{float(s['max_over_reference']):.1f} & {100*float(s['share_above_reference']):.0f}\\% & {100*share_pooled:.0f}\\% \\\\")


rows = {}
for d in (1e-4, 1e-5):
    rows[d] = [row("FM", "fm", d), row("PR lower", "pr_lo", d), row("PR upper", "thm3", d)]
    print(f"delta {d:g}:"); [print("  " + r) for r in rows[d]]
gap = np.array([float(r["pr_hi_0.0001"]) - float(r["pr_lo_0.0001"]) for r in ever]); gapmax = gap.max()
nref = np.mean([int(r["n_refined"]) for r in ever])
print(f"GAPMAX {gapmax:.3f} (relative max {max(gap[i]/float(ever[i]['pr_lo_0.0001']) for i in range(len(ever))):.4f}); NREF {nref:.1f}; certified fails {sum(1-int(r['certified']) for r in ever)}")
brackets = {i: f"[{float(dev[i]['pr_lo_0.0001']):.2f}, {float(dev[i]['thm3_0.0001']):.2f}]" for i in (36, 206, 335, 286, 176, 342, 368, 240, 433, 548)}
print(brackets)

if "--patch" in sys.argv:
    old = {1e-4: ["(b) availability, FM & 2.09 & 7.53 & 23.93 & 21.1 & 67\\% & 16\\% \\\\",
                  "(b) availability, PR lower & 7.34 & 14.59 & 27.08 & 22.2 & 99\\% & 24\\% \\\\",
                  "(b) availability, PR upper & 9.62 & 18.82 & 34.53 & 28.3 & 100\\% & 42\\% \\\\"],
           1e-5: ["(b) availability, FM & 2.46 & 8.54 & 26.38 & 19.4 & 67\\% & 10\\% \\\\",
                  "(b) availability, PR lower & 8.39 & 16.32 & 29.72 & 20.3 & 99\\% & 12\\% \\\\",
                  "(b) availability, PR upper & 10.93 & 20.90 & 37.67 & 25.8 & 100\\% & 28\\% \\\\"]}
    # the manuscript sources are not part of this release: pass them on the command line,
    # e.g. python3 rev_peruser_rows.py --patch sec5_grounding.tex t6_grounding.tex supp_body.tex response_to_reviewers.tex
    texfiles = sys.argv[sys.argv.index("--patch") + 1:]
    if not texfiles:
        raise SystemExit("--patch needs the .tex files to patch (the manuscript sources are not in this release)")
    for f in texfiles:
        s = open(f).read()
        for d in (1e-4, 1e-5):
            for o, n in zip(old[d], rows[d]):
                if o in s: s = s.replace(o, n)
        for i, b in brackets.items():
            s = re.sub(rf"(^{i} & [0-9.]+ & [0-9.]+ & [0-9.]+ \$\[[0-9.,]+\]\$ & [0-9.]+ & )\[[0-9., ]+\]", lambda m: m.group(1) + b, s, flags=re.M)
        s = s.replace("[[GAPMAX]]", f"{gapmax:.2f}").replace("[[NREF]]", f"{nref:.1f}")
        open(f, "w").write(s)
    print("patched")
