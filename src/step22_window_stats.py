"""Step 22: descriptive statistics of the FLASH trace and of the 3-day analysis window used in Section 6.1
(device counts, trace span, population diurnal profile, window pool range, per-device availability quantiles,
lag-1 autocorrelation of availability in the window).  Prints to stdout; no random numbers involved."""
import json, numpy as np, datetime, sys
import os; sys.path.insert(0, os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")))
from trace_analysis import ready_intervals, availability_grid, lag1

d = json.load(open(os.path.join(os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")), "state_traces.json")))
print("devices in file:", len(d))
devs = []; spans = []
for k, v in d.items():
    iv, tf, tl = ready_intervals(v["messages"])
    if tf is None: continue
    spans.append((tl - tf) / 86400)
    if tl - tf < 2 * 86400: continue
    devs.append((k, iv, tf, tl))
spans = np.array(spans)
print(f"devices with any trace {len(spans)}, span days median {np.median(spans):.1f} IQR {np.percentile(spans,25):.1f}-{np.percentile(spans,75):.1f}; >= 2 days: {len(devs)}")
step = 600
t0 = min(tf for (_, _, tf, _) in devs); t1 = max(tl for (_, _, _, tl) in devs)
print("overall span", datetime.datetime.fromtimestamp(t0), "to", datetime.datetime.fromtimestamp(t1), f"{(t1-t0)/86400:.2f} days")
times = np.arange(t0, t1, step); R = len(times)
Amat = np.zeros((len(devs), R), dtype=bool); cover = np.zeros((len(devs), R), dtype=bool)
for i, (k, iv, tf, tl) in enumerate(devs):
    Amat[i] = availability_grid(iv, t0, t1, step); cover[i] = (times >= tf) & (times <= tl)
covered = cover.sum(axis=0); pool = (Amat & cover).sum(axis=0); frac = pool / np.maximum(covered, 1)
hours = ((times - t0) / 3600 + datetime.datetime.fromtimestamp(t0).hour) % 24
prof = [frac[(hours >= h) & (hours < h + 3)].mean() for h in range(0, 24, 3)]
print(f"population grid: {R} rounds of 10 min; available fraction mean {frac.mean():.4f} min {frac.min():.3f} max {frac.max():.3f}; lag1 {lag1(frac):.3f}")
print("diurnal available fraction by 3-h block:", np.round(prof, 3))
W = 3 * 144; best = max(range(0, R - W, 36), key=lambda s0: (cover[:, s0:s0 + W].all(axis=1)).sum())
sel = cover[:, best:best + W].all(axis=1); A = Amat[sel, best:best + W]
print("3-day window start", datetime.datetime.fromtimestamp(times[best]), "devices", int(sel.sum()), f"mean availability {A.mean():.4f}",
      f"E[pool] {A.sum(axis=0).mean():.1f} pool min/max {A.sum(axis=0).min()}/{A.sum(axis=0).max()}")
pA = A.mean(axis=1)
print(f"per-device availability in window: median {np.median(pA):.4f} IQR {np.percentile(pA,25):.4f}-{np.percentile(pA,75):.4f}; share > 0.5: {(pA>0.5).mean():.4f}; share 0: {(pA==0).mean():.4f}")
lamA = np.array([lag1(A[i]) for i in range(A.shape[0])]); print(f"lambda_A median in window {np.nanmedian(lamA):.3f}")
hw = ((np.arange(W) * 10 / 60) + datetime.datetime.fromtimestamp(times[best]).hour) % 24
pw = A.sum(axis=0); print("window pool by 3-h block:", [round(float(pw[(hw >= h) & (hw < h + 3)].mean()), 1) for h in range(0, 24, 3)])
