"""Revision round 4, B1: a STRUCTURAL (draw-free) statement about the timer policies against an adversary who knows
availability, replacing the 100-seed sensitivity check as the basis of any exactness claim.

Policy: fixed cohort K = round(0.6 * E[pool]) with a b-round timer (b = 20 or 5): in round t the eligible set is
E_t = {i : A[i,t] = 1 and t - last_i > b}; if |E_t| <= K every eligible device is taken, otherwise K are drawn uniformly.

Status after round 1.  For each device i let s_i be its status after rounds 0 and 1: 'n' (taken in neither), '0' (taken
in round 0) or '1' (taken in round 1; a device taken in round 0 is blocked in round 1 because b >= 1).  The ALLOWED set
S_i of statuses is what some draw of rounds 0-1 can produce:
   '0' allowed iff A[i,0] = 1, and it is forced if round 0 cannot bind (|A[:,0]| <= K);
   '1' allowed iff A[i,1] = 1 and i can be outside round 0's cohort, forced among those if round 1 cannot bind;
   'n' allowed iff i can be outside both cohorts.
Deterministic evolution.  If no round t >= 2 binds, every eligible device is taken from round 2 on, so device i's whole
path is a deterministic function P_i(s_i) of its own availability row and s_i.
Certificate that no round t >= 2 can bind under ANY draw.  Let e_i(t; s) = 1 if, evolving deterministically from s,
device i is eligible at t.  UB_t = the maximum of sum_i e_i(t; s_i) over all status assignments the draws of rounds 0-1
can produce (exactly K devices take '0'; among the devices of A[:,1] that did not, min(K, their number) take '1'),
computed exactly by a dynamic programme; it bounds |E_t| for every draw under which rounds 2..t-1 did not bind; if
UB_t <= K for all t >= 2, induction on t shows that no round t >= 2 binds for any draw, and the
set of possible paths of device i is exactly {P_i(s) : s in S_i}.
Consequences.  (1) A device with |S_i| = 1 has the same path under every draw: against the availability-aware adversary
its per-round loss over a window is exactly that of the Gaussian pair with shift a*sqrt(N_i).  (2) For the others the
path ranges over at most three known paths; the adversary who also sees the cohort draws faces one of them, and the
availability-only adversary faces their mixture, whose curve is at most sum_s w_s h_eps(a sqrt N_i(s)) <= max_s
h_eps(a sqrt N_i(s)) (Proposition 3), a deterministic upper bound needing no Monte Carlo.
Result.  The certificate FAILS: under some draws of rounds 0-1 the round-t pool can exceed K in later rounds (one round
for the 20-round timer, ten for the 5-round timer), so path invariance cannot be established structurally for any
device, and the 100-seed study (rev_timer_seeds.py) remains an empirical sensitivity check.  What IS certified is the
per-device schema bound: the greedy count N_sched (the most participations the device's availability allows in a
window under the timer) bounds the count of every draw, so h_eps^{-1}(a sqrt N_sched) bounds the per-round loss against
every adversary up to one who also sees the cohort draws.
Outputs rev_timer_structural.csv, rev_timer_schema.csv and a printed summary; cross-checked against rev_timer_seeds.py."""
import csv
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from rev_common import window, apply_policy
from corraudit import eps_poisson_pld

T, A_SIG, DELTAS, STRIDE = 200, 0.5, [1e-4, 1e-5], 10
def h(mu, e): return norm.cdf(mu / 2 - e / mu) - np.exp(e) * norm.cdf(-mu / 2 - e / mu) if mu > 0 else 0.0
def eps_gauss(mu, d):
    if mu <= 0 or h(mu, 0.0) <= d: return 0.0
    return brentq(lambda e: h(mu, e) - d, 0.0, 200.0)


def evolve(row, s, b):
    """Deterministic path of one device from status s after round 1, assuming no round >= 2 binds."""
    Tw = len(row); path = np.zeros(Tw, dtype=bool); last = -10**9
    if s == "0": path[0] = True; last = 0
    if s == "1": path[1] = True; last = 1
    elig = np.zeros(Tw, dtype=bool)
    for t in range(2, Tw):
        if row[t] and t - last > b:
            elig[t] = True; path[t] = True; last = t
    return path, elig


if __name__ == "__main__":
    A, _, _ = window(10, 3, verbose=False); N, Tw = A.shape
    K = int(round(0.6 * A.sum(axis=0).mean()))
    starts = list(range(0, Tw - T + 1, STRIDE)); rows = []; sched_rows = []
    for name, b in [("fixed cohort 0.6, timer 20", 20), ("fixed cohort 0.6, timer 5", 5)]:
        bind0 = A[:, 0].sum() > K; bind1_possible = A[:, 1].sum() > K
        allowed = []
        for i in range(N):
            S = set()
            a0, a1 = bool(A[i, 0]), bool(A[i, 1])
            can_skip0 = (not a0) or bind0                      # i can be outside round 0's cohort
            if a0: S.add("0")
            if a0 and not bind0: S = {"0"}                      # forced
            else:
                if a1 and can_skip0: S.add("1")
                if can_skip0 and ((not a1) or bind1_possible): S.add("n")
            allowed.append(sorted(S))
        paths = {}; eligs = {}
        for i in range(N):
            for s in allowed[i]:
                paths[i, s], eligs[i, s] = evolve(A[i], s, b)
        # Exact maximum of the round-t pool over all feasible draws of rounds 0-1 (assuming rounds 2..t-1 did not bind):
        # devices with one allowed status contribute a fixed e_i(t); the variable devices (A[i,0] or A[i,1] = 1) are
        # assigned statuses subject to the cohort sizes: exactly K take '0' (round 0 binds); among the m variable devices
        # in A[:,1] that did not take '0', exactly min(K, m) take '1' and the rest 'n'.  Dynamic programme over the
        # variable devices with state (c0, c1, m).
        var = [i for i in range(N) if len(allowed[i]) > 1]; fixed = [i for i in range(N) if len(allowed[i]) == 1]
        fixed_e = np.zeros(Tw, dtype=int)
        for i in fixed: fixed_e += eligs[i, allowed[i][0]].astype(int)
        k0 = min(K, int(A[:, 0].sum()))
        M = int(A[:, 1].sum()) + 1
        UB = fixed_e.copy()
        diff_t = [t for t in range(2, Tw) if any(len({bool(eligs[i, s][t]) for s in allowed[i]}) > 1 for i in var)]
        NEG = -10**6
        for t in range(2, Tw):
            if t not in diff_t:
                UB[t] += sum(int(eligs[i, allowed[i][0]][t]) for i in var); continue
            dp = np.full((k0 + 1, K + 1, M), NEG, dtype=np.int64); dp[0, 0, 0] = 0
            for i in var:
                new = np.full_like(dp, NEG); in1 = bool(A[i, 1])
                for s in allowed[i]:
                    v = int(eligs[i, s][t])
                    if s == "0":
                        new[1:, :, :] = np.maximum(new[1:, :, :], dp[:-1, :, :] + v)
                    elif s == "1":
                        new[:, 1:, 1:] = np.maximum(new[:, 1:, 1:], dp[:, :-1, :-1] + v)
                    else:
                        if in1: new[:, :, 1:] = np.maximum(new[:, :, 1:], dp[:, :, :-1] + v)
                        else:   new = np.maximum(new, dp + v)
                dp = new
            best = NEG
            for m in range(M):
                c1 = min(K, m)
                if dp[k0, c1, m] > best: best = dp[k0, c1, m]
            UB[t] += int(best)
        worst_t = int(np.argmax(UB[2:])) + 2
        certified = bool((UB[2:] <= K).all())
        print(f"   rounds t>=2 at which some variable device's eligibility depends on its status: {len(diff_t)} (last {max(diff_t) if diff_t else None}); "
              f"max over t>=2 of the exact pool maximum: {UB[2:].max()} at t={worst_t}; rounds >= 2 that could bind: {int((UB[2:] > K).sum())}")
        n_multi = sum(1 for i in range(N) if len(allowed[i]) > 1)
        maxN = {(i, s): max(int(paths[i, s][s0:s0 + T].sum()) for s0 in starts) for i in range(N) for s in allowed[i]}
        ever = np.array([max(maxN[i, s] for s in allowed[i]) > 0 for i in range(N)])
        invariant = np.array([len(allowed[i]) == 1 for i in range(N)]) & ever
        nrange = np.array([max(maxN[i, s] for s in allowed[i]) - min(maxN[i, s] for s in allowed[i]) for i in range(N)])
        # cross-check: seed-0 policy path must be one of the allowed paths
        Bs = apply_policy(A, "fixed", 0.6, b, np.random.default_rng(0))
        consistent = all(any((Bs[i] == paths[i, s]).all() for s in allowed[i]) for i in range(N))
        q = Bs.mean(); claim = {d: eps_poisson_pld(q, T, 1.0, 2.0, d) for d in DELTAS}
        # Per-device schema bound (draw-free): the largest number of participations the device's availability allows in
        # any window under the timer, i.e. the greedy count started unblocked at the window start (earliest-available
        # selection maximises the number of points with separation > b).  Every draw of the policy gives the device at
        # most this many participations in that window, so h_eps^{-1}(a sqrt(N_sched)) bounds its per-round loss against
        # every adversary up to one who also sees the cohort draws (Proposition 2(i): sum_n p_n h(a sqrt n) <= h(a sqrt n_max)).
        def greedy(row):
            best = 0
            for s0 in starts:
                last = -10**9; c = 0
                for t in range(s0, s0 + T):
                    if row[t] and t - last > b: c += 1; last = t
                best = max(best, c)
            return best
        Nsched = np.array([greedy(A[i]) for i in range(N)])
        Nreal = np.array([max(int(Bs[i, s0:s0 + T].sum()) for s0 in starts) for i in range(N)])
        assert (Nreal <= Nsched).all()
        print(f"   per-device schema count: equals the seed-0 count for {(Nsched[Nreal > 0] == Nreal[Nreal > 0]).sum()} of {(Nreal > 0).sum()} participating devices; "
              f"max excess {int((Nsched - Nreal).max())}; median N_sched {int(np.median(Nsched[Nreal > 0]))}, max {Nsched.max()}")
        print(f"{name}: K={K}; round 0 binds {bind0}, round 1 can bind {bind1_possible}; max_t>=2 UB_t = {UB[2:].max()} at t={worst_t} "
              f"(<= K: {certified}); devices with more than one allowed status {n_multi}; participating {ever.sum()}; "
              f"structurally invariant {invariant.sum()}; largest-window count can change for {(nrange[ever] > 0).sum()} (by at most {nrange.max()}); "
              f"seed-0 path among the allowed paths for every device: {consistent}")
        for d in DELTAS:
            e_lo = np.array([min(eps_gauss(A_SIG * np.sqrt(maxN[i, s]), d) for s in allowed[i]) for i in range(N)])
            e_hi = np.array([max(eps_gauss(A_SIG * np.sqrt(maxN[i, s]), d) for s in allowed[i]) for i in range(N)])
            e_inv = e_hi[invariant]
            e_sched = np.array([eps_gauss(A_SIG * np.sqrt(n), d) for n in Nsched])
            print(f"  delta {d:g}: per-device schema bound (certified, every draw): median {np.median(e_sched[Nreal > 0]):.2f}, max {e_sched.max():.2f} "
                  f"(x{np.median(e_sched[Nreal > 0])/claim[d]:.1f}, x{e_sched.max()/claim[d]:.1f}); seed-0 realised-path value: median "
                  f"{np.median(np.array([eps_gauss(A_SIG*np.sqrt(n), d) for n in Nreal])[Nreal > 0]):.2f}")
            sched_rows.append([name, d, f"{claim[d]:.3f}", int((Nreal > 0).sum()), int((Nsched[Nreal > 0] == Nreal[Nreal > 0]).sum()), int((Nsched - Nreal).max()),
                               int(np.median(Nsched[Nreal > 0])), int(Nsched.max()), f"{np.median(e_sched[Nreal > 0]):.2f}", f"{e_sched.max():.2f}",
                               f"{np.median(e_sched[Nreal > 0])/claim[d]:.1f}", f"{e_sched.max()/claim[d]:.1f}", int(UB[2:].max()), int((UB[2:] > K).sum())])
            worst = int(np.argmax(e_hi))
            rows.append([name, d, K, int(bind0), int(bind1_possible), int(UB[2:].max()), int(certified), int(ever.sum()), int(invariant.sum()),
                         int((nrange[ever] > 0).sum()), int(nrange.max()), f"{claim[d]:.3f}", f"{np.median(e_hi[ever]):.2f}", f"{e_hi.max():.2f}", worst,
                         f"{e_lo[worst]:.2f}", f"{np.median(e_inv):.2f}", f"{e_inv.max():.2f}", f"{np.median(e_hi[ever])/claim[d]:.1f}",
                         f"{e_hi.max()/claim[d]:.1f}", int(consistent)])
            print(f"  delta {d:g}: claim {claim[d]:.3f}; certified upper value (max over allowed paths): median {np.median(e_hi[ever]):.2f}, "
                  f"max {e_hi.max():.2f} (device {worst}; its smallest allowed value {e_lo[worst]:.2f}); invariant devices: median {np.median(e_inv):.2f}, "
                  f"max {e_inv.max():.2f}; ratios median {np.median(e_hi[ever])/claim[d]:.1f}, max {e_hi.max()/claim[d]:.1f}")
    with open("rev_timer_structural.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "delta", "K", "round0_binds", "round1_can_bind", "max_UB_t_ge2", "no_later_binding_certified", "participating",
                    "structurally_invariant", "count_can_change", "max_count_change", "poisson_claim", "upper_median", "upper_max", "worst_device",
                    "worst_device_min", "invariant_median", "invariant_max", "median_ratio", "max_ratio", "seed0_consistent"])
        w.writerows(rows)
    with open("rev_timer_schema.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "delta", "poisson_claim", "participating", "schema_count_equals_realised", "max_excess", "median_N_sched", "max_N_sched",
                    "schema_eps_median", "schema_eps_max", "median_ratio", "max_ratio", "max_pool_t_ge2_over_draws", "rounds_t_ge2_that_can_bind"])
        w.writerows(sched_rows)
