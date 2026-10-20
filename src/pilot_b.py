"""Pilot B: coverage of one-run audit estimators under correlated canary participation."""
import numpy as np, time, csv
from scipy.stats import norm, chi2
from scipy.stats import t as student_t
from corraudit import *
from corraudit import _bonf_box
from vecgauss import eps_gauss_pair_vec

C, sigma, q = 1.0, 2.0, 0.1
DELTA = 1e-4
M = 1000          # canaries per run
R = 300           # replications (independent one-run audits)
ALPHA = 0.05
rng = np.random.default_rng(7)

# references from pilot A
refs = {}
with open("pilot_a_results.csv") as f:
    for row in csv.DictReader(f):
        if float(row["delta"]) == DELTA:
            refs[(int(row["T"]), row["process"], float(row["param"]))] = (
                float(row["eps_full_mc"]), float(row["eps_sum_exact"]), float(row["eps_poisson_pld"]))

cells = [(200, "iid", 0.0), (200, "markov", 0.6), (200, "markov", 0.9), (200, "markov", 0.97),
         (200, "minsep", 5), (200, "minsep", 9),
         (1000, "iid", 0.0), (1000, "markov", 0.9), (1000, "markov", 0.99), (1000, "minsep", 9)]


def gauss_audit_fast(S, present, delta, alpha=ALPHA, ngrid=5):
    a = alpha / 4
    (m0l, m0h), (s0l, s0h) = _bonf_box(S[~present], a)
    (m1l, m1h), (s1l, s1h) = _bonf_box(S[present], a)
    g = np.meshgrid(np.linspace(m0l, m0h, ngrid), np.linspace(s0l, s0h, ngrid),
                    np.linspace(m1l, m1h, ngrid), np.linspace(s1l, s1h, ngrid), indexing="ij")
    e = eps_gauss_pair_vec(g[0].ravel(), g[1].ravel(), g[2].ravel(), g[3].ravel(), delta)
    point = float(eps_gauss_pair_vec(S[~present].mean(), S[~present].std(ddof=1),
                                     S[present].mean(), S[present].std(ddof=1), delta))
    return float(e.min()), point


def mixture_eps_fast(means, wts, s, mu0, s0, delta, grid=4001):
    keep = wts > 0
    means, wts = means[keep], wts[keep] / wts[keep].sum()
    lo = min(means.min(), mu0) - 10 * max(s, s0)
    hi = max(means.max(), mu0) + 10 * max(s, s0)
    x = np.linspace(lo, hi, grid); dx = x[1] - x[0]
    p1 = (wts[None, :] * norm.pdf(x[:, None], means[None, :], s)).sum(axis=1)
    p0 = norm.pdf(x, mu0, s0)
    lo_e, hi_e = 0.0, 60.0
    def d(e):
        return max(np.sum(np.clip(p1 - np.exp(e) * p0, 0, None)) * dx,
                   np.sum(np.clip(p0 - np.exp(e) * p1, 0, None)) * dx)
    if d(0.0) <= delta:
        return 0.0
    for _ in range(50):
        mid = 0.5 * (lo_e + hi_e)
        if d(mid) <= delta:
            hi_e = mid
        else:
            lo_e = mid
    return hi_e


def count_conditioned_fast(S, present, counts, delta, alpha=ALPHA, ngrid=3):
    """Present law = empirical count mixture sum_n p_hat_n N(mu0 + beta n, s0^2).
    beta (per-participation shift) is estimated from the present-vs-absent mean gap divided by the
    mean present count; Bonferroni box on (mu0, s0, beta), alpha/3 each."""
    a = alpha / 3
    (m0l, m0h), (s0l, s0h) = _bonf_box(S[~present], a)
    Sp, Sa = S[present], S[~present]
    nn = counts[present].astype(float)
    gap = Sp.mean() - Sa.mean()
    se = np.sqrt(Sp.var(ddof=1) / len(Sp) + Sa.var(ddof=1) / len(Sa))
    tq = student_t.ppf(1 - a / 2, min(len(Sp), len(Sa)) - 1)
    nbar = nn.mean()
    bl, bh = max((gap - tq * se) / nbar, 0.0), max((gap + tq * se) / nbar, 0.0)
    pn = np.bincount(counts) / len(counts)
    n = np.arange(len(pn))
    best = np.inf
    for mu0 in np.linspace(m0l, m0h, ngrid):
        for s0 in np.linspace(s0l, s0h, ngrid):
            for b in np.linspace(bl, bh, ngrid):
                best = min(best, mixture_eps_fast(mu0 + b * n, pn, s0, mu0, s0, delta))
    return best, gap / nbar


if __name__ == "__main__":
    import os
    done = set()
    if os.path.exists("pilot_b_results.csv"):
        for row in csv.DictReader(open("pilot_b_results.csv")):
            done.add((int(row["T"]), row["process"], float(row["param"])))
    out = open("pilot_b_results.csv", "a", newline="")
    w = csv.writer(out)
    if not done: w.writerow(["T", "process", "param", "eps_full", "eps_sum", "eps_poisson", "estimator",
                "mean_hat", "sd_hat", "cov_vs_full", "exceed_vs_sum", "flag_vs_poisson", "ratio_to_full"])
    raw = open("pilot_b_raw.csv", "a", newline="")
    wr = csv.writer(raw)
    if not done: wr.writerow(["T", "process", "param", "rep", "gauss_box", "gauss_point", "countcond", "steinke"])
    t0 = time.time()
    for (T, proc, par) in cells:
        if (T, proc, float(par)) in done or (T, proc, float(par)) not in refs:
            print("skip", T, proc, par, flush=True); continue
        if proc == "iid":
            P, pi, emit = iid_chain(q)
        elif proc == "markov":
            P, pi, emit = two_state_chain(q, par)
        else:
            P, pi, emit = minsep_chain(q, int(par))
        ef, es, ep = refs[(T, proc, float(par))]
        res = {"gauss_box": [], "gauss_point": [], "countcond": [], "steinke": []}
        for r in range(R):
            S, present, counts = simulate_scores(P, pi, emit, T, C, sigma, M, rng)
            gb, gp = gauss_audit_fast(S, present, DELTA)
            cc, _ = count_conditioned_fast(S, present, counts, DELTA)
            st = steinke_audit(S, present, DELTA, ALPHA, k=100)
            for k_, v in zip(res, (gb, gp, cc, st)):
                res[k_].append(v)
            wr.writerow([T, proc, par, r, f"{gb:.4f}", f"{gp:.4f}", f"{cc:.4f}", f"{st:.4f}"])
        raw.flush()
        for name, v in res.items():
            v = np.array(v)
            w.writerow([T, proc, par, f"{ef:.3f}", f"{es:.3f}", f"{ep:.3f}", name, f"{v.mean():.3f}", f"{v.std():.3f}",
                        f"{np.mean(v <= ef):.3f}", f"{np.mean(v > es):.3f}", f"{np.mean(v > ep):.3f}", f"{v.mean()/ef:.3f}"])
            print(f"T={T} {proc} {par}: full={ef:.2f} sum={es:.2f} poisson={ep:.2f} | {name:12s} mean={v.mean():.2f}±{v.std():.2f} cov(full)={np.mean(v<=ef):.3f} exceed(sum)={np.mean(v>es):.3f} flag(poisson)={np.mean(v>ep):.3f} [{time.time()-t0:.0f}s]", flush=True)
        out.flush()
    out.close(); raw.close()
