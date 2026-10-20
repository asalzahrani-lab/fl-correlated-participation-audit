"""Regime grounding on the FLASH state trace (1,000 devices sampled from the 136k-device trace of Yang et al., WWW 2021).
Availability per Google's criteria as implemented in FLASH's Timer: locked AND wifi AND charging.
Outputs: per-device availability rate p_A and lag-1 autocorrelation lambda_A at several round lengths; pool-size
dynamics; participation autocorrelation lambda_B and count dispersion under cohort policies; exact final-model eps
(Theorem 1) under the empirical count law vs Poisson at the same mean rate."""
import json, sys, datetime, csv, time
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from corraudit import eps_sum_exact, two_state_chain, count_distribution, eps_poisson_pld
from theory import eps_jensen_full

FMT = "%Y-%m-%d %H:%M:%S"


def ready_intervals(messages):
    """Return list of (start_ts, end_ts) in epoch seconds where locked & wifi & charged, plus trace span."""
    locked = wifi = charged = False
    st = None; out = []; t_first = t_last = None
    for mes in messages.split("\n"):
        if not mes.strip():
            continue
        try:
            t, s = mes.strip().split("\t")
        except ValueError:
            continue
        s = s.strip().lower()
        ts = datetime.datetime.strptime(t.strip(), FMT).timestamp()
        if t_first is None: t_first = ts
        t_last = ts
        if s == "battery_charged_on": charged = True
        elif s == "battery_charged_off": charged = False
        elif s == "wifi": wifi = True
        elif s in ("unknown", "4g", "3g", "2g", "5g"): wifi = False
        elif s == "screen_on": pass
        elif s == "screen_off": pass
        elif s == "screen_lock": locked = True
        elif s == "screen_unlock": locked = False
        elif s.endswith("%"): pass
        else: continue
        ok = locked and wifi and charged
        if ok and st is None:
            st = ts
        if st is not None and not ok:
            out.append((st, ts)); st = None
    if st is not None and t_last is not None:
        out.append((st, t_last))
    return out, t_first, t_last


def availability_grid(intervals, t0, t1, step):
    """Boolean vector: available at each grid time t0 + k*step (for the whole step? use 'at start of round')."""
    times = np.arange(t0, t1, step)
    A = np.zeros(len(times), dtype=bool)
    for (a, b) in intervals:
        i0 = int(np.ceil((a - t0) / step)); i1 = int(np.floor((b - t0) / step))
        if i1 >= i0:
            A[max(i0, 0):min(i1 + 1, len(times))] = True
    return A


def lag1(x):
    x = x.astype(float)
    if x.std() == 0: return np.nan
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def main():
    d = json.load(open(os.path.join(os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")), "state_traces.json")))
    devs = []
    for k, v in d.items():
        iv, tf, tl = ready_intervals(v["messages"])
        if tf is None or tl - tf < 2 * 86400:   # need at least 2 days
            continue
        devs.append((k, iv, tf, tl))
    print(f"devices with >= 2 days of trace: {len(devs)}", flush=True)
    rows = []
    for step_min in [5, 10, 30]:
        step = step_min * 60
        pA, lamA = [], []
        for (k, iv, tf, tl) in devs:
            A = availability_grid(iv, tf, tl, step)
            if A.mean() in (0.0, 1.0): continue
            pA.append(A.mean()); lamA.append(lag1(A))
        pA = np.array(pA); lamA = np.array(lamA)
        print(f"round={step_min} min: devices {len(pA)}, p_A median {np.median(pA):.3f} (IQR {np.percentile(pA,25):.3f}-{np.percentile(pA,75):.3f}); "
              f"lambda_A median {np.nanmedian(lamA):.3f} (IQR {np.nanpercentile(lamA,25):.3f}-{np.nanpercentile(lamA,75):.3f}); share lambda_A>0.9: {np.mean(lamA>0.9):.2f}", flush=True)
        rows.append(["availability", step_min, len(pA), f"{np.median(pA):.4f}", f"{np.nanmedian(lamA):.4f}", f"{np.nanpercentile(lamA,25):.4f}", f"{np.nanpercentile(lamA,75):.4f}", f"{np.mean(lamA>0.9):.3f}"])
    # ---------------- population grid on a common window (10-min rounds)
    step = 600
    t0 = min(tf for (_, _, tf, _) in devs); t1 = max(tl for (_, _, _, tl) in devs)
    times = np.arange(t0, t1, step); R = len(times)
    Amat = np.zeros((len(devs), R), dtype=bool); cover = np.zeros((len(devs), R), dtype=bool)
    for i, (k, iv, tf, tl) in enumerate(devs):
        Amat[i] = availability_grid(iv, t0, t1, step)
        cover[i] = (times >= tf) & (times <= tl)
    covered = cover.sum(axis=0); pool = (Amat & cover).sum(axis=0)
    frac = pool / np.maximum(covered, 1)
    hours = ((times - t0) / 3600 + datetime.datetime.fromtimestamp(t0).hour) % 24
    print(f"population: window {R} rounds of 10 min ({R/144:.1f} days); covered devices median {np.median(covered):.0f}; available fraction mean {frac.mean():.3f}, min {frac.min():.3f}, max {frac.max():.3f}; lag-1 autocorr of pool fraction {lag1(frac):.3f}", flush=True)
    # diurnal profile
    prof = [frac[(hours >= h) & (hours < h + 3)].mean() for h in range(0, 24, 3)]
    print("diurnal available fraction by 3-h block:", np.round(prof, 3), flush=True)
    rows.append(["pool", 10, R, f"{frac.mean():.4f}", f"{frac.min():.4f}", f"{frac.max():.4f}", f"{lag1(frac):.4f}", " ".join(f"{p:.3f}" for p in prof)])
    # ---------------- participation under cohort policies (devices covered for the whole window used; else restrict to a 3-day sub-window with max coverage)
    # choose the 3-day sub-window with the most fully-covered devices
    W = 3 * 144
    best = max(range(0, R - W, 36), key=lambda s0: (cover[:, s0:s0 + W].all(axis=1)).sum())
    sel = cover[:, best:best + W].all(axis=1)
    A = Amat[sel, best:best + W]; N, Tw = A.shape
    print(f"policy window: {Tw} rounds (3 days), {N} fully covered devices; mean available fraction {A.mean():.3f}", flush=True)
    rng = np.random.default_rng(0)
    T_eps = 200; delta = 1e-4; C, sigma = 1.0, 2.0
    print("policy | q (mean participation) | lambda_B (median over devices) | dispersion Var(N_T)/E[N_T] over T=200 | exact final-model eps (empirical count law) vs Poisson at same q", flush=True)
    policies = []
    for s in [0.05, 0.2, 0.5, 1.0]:
        policies.append(("thinning s=%.2f" % s, ("thin", s, 0)))
    for c in [0.1, 0.3, 0.6, 1.0]:
        policies.append(("fixed cohort K=%.1f*E[pool]" % c, ("fixed", c, 0)))
    for b in [5, 20]:
        policies.append(("fixed cohort K=0.6*E[pool], min-sep b=%d" % b, ("fixed", 0.6, b)))
    Epool = A.sum(axis=1).mean() if False else A.sum(axis=0).mean()
    for name, (kind, par, b) in policies:
        B = np.zeros_like(A)
        last = np.full(N, -10**9)
        for t in range(Tw):
            elig = A[:, t] & (t - last > b)
            idx = np.flatnonzero(elig)
            if kind == "thin":
                chosen = idx[rng.random(len(idx)) < par]
            else:
                K = int(round(par * Epool))
                chosen = idx if len(idx) <= K else rng.choice(idx, K, replace=False)
            B[chosen, t] = True; last[chosen] = t
        q = B.mean()
        lamB = np.array([lag1(B[i]) for i in range(N)])
        lamB_med = np.nanmedian(lamB)
        # count law over windows of T_eps rounds (all devices, all start offsets in steps of 20)
        counts = []
        for i in range(N):
            for s0 in range(0, Tw - T_eps, 20):
                counts.append(B[i, s0:s0 + T_eps].sum())
        counts = np.array(counts); m = counts.mean(); v = counts.var()
        pn = np.bincount(counts, minlength=T_eps + 1) / len(counts)
        e_emp = eps_sum_exact(pn, C, sigma, T_eps, delta)
        qd = m / T_eps
        # Poisson at the same mean rate: Binomial count law
        from scipy.stats import binom
        pn_pois = binom.pmf(np.arange(T_eps + 1), T_eps, qd)
        e_pois = eps_sum_exact(pn_pois, C, sigma, T_eps, delta)
        ub = eps_jensen_full(pn, C, sigma, delta); ep_round = eps_poisson_pld(qd, T_eps, C, sigma, delta)
        np.save(os.path.join(HERE, f"pn_{name.replace(' ', '_').replace('*','x').replace('=','').replace(',','')}.npy"), pn)
        # two-state Markov fit with (qd, lamB_med) for the map
        print(f"{name:40s} | q={qd:.3f} | lambda_B={lamB_med:.3f} (IQR {np.nanpercentile(lamB,25):.2f}-{np.nanpercentile(lamB,75):.2f}) | D={v/m if m>0 else np.nan:.2f} | eps_sum emp={e_emp:.3f} vs Poisson={e_pois:.3f} (ratio {e_emp/e_pois if e_pois>0 else np.nan:.2f}) | per-round: Poisson={ep_round:.3f} Thm3-bound={ub:.3f} (bound/Poisson {ub/ep_round:.2f})", flush=True)
        rows.append(["policy", name, f"{qd:.4f}", f"{lamB_med:.4f}", f"{np.nanpercentile(lamB,25):.4f}", f"{np.nanpercentile(lamB,75):.4f}", f"{v/m if m>0 else np.nan:.4f}", f"{e_emp:.4f}", f"{e_pois:.4f}", f"{ep_round:.4f}", f"{ub:.4f}"])
    with open(os.path.join(HERE, "trace_results.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    main()


def extra():
    """Per-policy: Theorem-3 (count-averaged) upper bound for the per-round release, Poisson per-round eps at the
    same q, and the effective two-state-Markov lambda implied by the dispersion index."""
    import csv
    from theory import eps_jensen_full
    from corraudit import eps_poisson_pld
    rows = list(csv.reader(open(os.path.join(HERE, "trace_results.csv"))))
    # recompute count laws quickly by re-running the policy loop (same seed) -> simpler: reuse printed D, q
    print("policy | q | D | effective lambda = (D' - 1)/(D' + 1) with D' = D/(1-q) | Poisson per-round eps | Theorem-3 bound needs the count law -> computed in main2")
    for r in rows:
        if r[0] != "policy": continue
        name, q, lamB, D = r[1], float(r[2]), float(r[3]), float(r[6])
        Dp = D / (1 - q); lam_eff = (Dp - 1) / (Dp + 1)
        ep = eps_poisson_pld(q, 200, 1.0, 2.0, 1e-4)
        print(f"{name:40s} | q={q:.3f} | D={D:.1f} | lambda_eff={lam_eff:.2f} | Poisson per-round eps(T=200)={ep:.3f}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "extra":
    extra()
