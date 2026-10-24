"""Revision round 10: the cyclic law's per-round reference at T = 1,000 from a larger sample (the earlier value 27.79 came
from the pilot run of pilot_b_1000.log), with a two-sided 99% interval, as rev_round10_cyclic.py does at T = 200.
Also the count-averaged bound (Theorem 2), which for the cyclic law equals the schema bound h^{-1}(a sqrt(T/10)).
Output rev_round10_cyclic_T1000.csv."""
import csv, sys, time
import numpy as np
from corraudit import eps_poisson_pld, eps_from_delta_curve
from certified import EmpiricalPathLaw
from theory import h_gauss
from rev_common import eps_interval

T, a, q = 1000, 0.5, 0.1
N_REF = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000_000
rng = np.random.default_rng(1012)
t0 = time.time(); log = lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True)
paths = np.zeros((10, T), dtype=np.int8)
for ph in range(10): paths[ph, ph::10] = 1
law = EmpiricalPathLaw(paths)
lp, lq = [], []
for s in range(0, N_REF, 50_000):
    k = min(50_000, N_REF - s)
    B = law.sample(k, rng); Zp = rng.standard_normal((k, T)).astype(np.float32); Zq = rng.standard_normal((k, T)).astype(np.float32)
    lp.append(law.loglr(a * B + Zp, a)); lq.append(law.loglr(Zq, a))
lp = np.concatenate(lp); lq = np.concatenate(lq)
rows = []
for d in [1e-4, 1e-5]:
    e, lo, hi = eps_interval(lp, lq, d, alpha=0.01)
    ep = eps_poisson_pld(q, T, 1.0, 1.0 / a, d)
    thm2 = eps_from_delta_curve(lambda x: float(h_gauss(np.array([a * np.sqrt(T / 10)]), x)[0]), d, 0.0, 400.0)
    rows.append(["cyclic_b9", T, f"{d:g}", N_REF, f"{e:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{ep:.4f}", f"{e / ep:.3f}", f"{thm2:.4f}", f"{thm2 / e:.3f}"])
    log(f"T={T} delta={d:g}: reference {e:.3f} [{lo:.3f}, {hi:.3f}]; Poisson {ep:.3f} (ratio {e / ep:.2f}); Theorem 2 = schema {thm2:.3f} ({thm2 / e:.2f}x the reference)")
with open("rev_round10_cyclic_T1000.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["law", "T", "delta", "samples", "eps", "lo99", "hi99", "poisson", "ratio", "thm2_schema", "thm2_over_reference"]); w.writerows(rows)
log("done")
