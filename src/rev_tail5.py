"""Reviewer question R4: the devices that carry the tail at T = 1,000 (5-minute rounds, 4-day window, fixed cohort 0.6,
seed 0): their availability, inclusion rate given availability, and largest window count.  Output printed (rev_tail5.log)."""
import numpy as np
from rev_common import window, apply_policy
A, _, _ = window(5, 4, verbose=False); N, Tw = A.shape
B = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0)); T = 1000
maxN = np.array([max(B[i, s:s + T].sum() for s in range(0, Tw - T + 1, 5)) for i in range(N)])
avail = A.mean(axis=1); tail = np.flatnonzero(maxN >= 200)
rank = np.array([(avail > avail[i]).sum() for i in tail])
incl = np.array([B[i][A[i]].mean() for i in tail])
print(f"devices {N}, window {Tw} rounds; tail devices (max N >= 200 at T=1000): {len(tail)}")
print(f"their availability {avail[tail].min():.3f}-{avail[tail].max():.3f} (median {np.median(avail[tail]):.3f}); population median {np.median(avail):.3f}, mean {avail.mean():.3f}")
print(f"their availability rank: within the top {(rank.max()+1)/N:.3f} of devices")
print(f"inclusion given available: median {np.median(incl):.3f} range {incl.min():.3f}-{incl.max():.3f}; population {B[A].mean():.3f}")
print(f"max N of tail devices: {sorted(maxN[tail].tolist())}")
print(f"share of the window overlap at T=1000 stride 5: consecutive windows share {(T-5)/T:.3f}; windows per device {(Tw-T)//5+1}")
