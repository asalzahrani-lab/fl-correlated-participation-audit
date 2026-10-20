"""Steps 1-3: exact accountant for the sum release, dispersion law, Jensen (count-averaged) upper bound
for the per-round release, and the sign / crossover of the Gaussian-fit bias."""
import numpy as np, csv, time
from scipy.stats import norm
from scipy.optimize import brentq
from corraudit import *

C, sigma, q = 1.0, 2.0, 0.1


def h_gauss(mu, eps):
    """Hockey-stick divergence of the Gaussian mechanism with GDP parameter mu at eps (both directions equal)."""
    mu = np.asarray(mu, float)
    out = np.zeros_like(mu)
    pos = mu > 0
    m = mu[pos]
    out[pos] = norm.cdf(m / 2 - eps / m) - np.exp(eps) * norm.cdf(-m / 2 - eps / m)
    return np.clip(out, 0, None)


def eps_jensen_sum(pn, C, sigma, T, delta):
    """Upper bound for the sum release: delta(eps) <= E_N[h(N C /(sqrt(T) sigma), eps)]."""
    n = np.arange(len(pn)); mu = n * C / (np.sqrt(T) * sigma)
    f = lambda e: float(np.sum(pn * h_gauss(mu, e)))
    return eps_from_delta_curve(f, delta)


def eps_jensen_full(pn, C, sigma, delta):
    """Upper bound for the per-round release: delta(eps) <= E_N[h(sqrt(N) C / sigma, eps)]."""
    n = np.arange(len(pn)); mu = np.sqrt(n) * C / sigma
    f = lambda e: float(np.sum(pn * h_gauss(mu, e)))
    return eps_from_delta_curve(f, delta)


def popfit_eps(pn, C, sigma, T, delta):
    n = np.arange(len(pn)); means = C * n / np.sqrt(T)
    mu1 = float(np.sum(pn * means)); s1 = float(np.sqrt(sigma**2 + np.sum(pn * means**2) - mu1**2))
    return eps_gauss_pair(0.0, sigma, mu1, s1, delta)


def process(kind, par):
    if kind == "iid":
        return iid_chain(q)
    if kind == "markov":
        return two_state_chain(q, par)
    return minsep_chain(q, int(par))


if __name__ == "__main__":
    t0 = time.time()
    # ---------------- Step 1: dispersion law for the sum release (T = 200, delta = 1e-4 and 1e-6)
    procs = [("minsep", 9), ("minsep", 7), ("minsep", 5), ("minsep", 3), ("minsep", 2), ("minsep", 1),
             ("markov", -0.1), ("iid", 0.0), ("markov", 0.15), ("markov", 0.3), ("markov", 0.45), ("markov", 0.6),
             ("markov", 0.75), ("markov", 0.9), ("markov", 0.95), ("markov", 0.97)]
    T = 200
    with open("theory_dispersion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["process", "param", "var_N", "dispersion_index", "eps_sum_1e-4", "eps_sum_1e-6", "jensen_sum_1e-4", "jensen_full_1e-4", "popfit_1e-4"])
        for kind, par in procs:
            P, pi, emit = process(kind, par)
            pn = count_distribution(P, pi, emit, T)
            n = np.arange(T + 1); m = np.sum(n * pn); v = np.sum(n * n * pn) - m**2
            e4 = eps_sum_exact(pn, C, sigma, T, 1e-4); e6 = eps_sum_exact(pn, C, sigma, T, 1e-6)
            js = eps_jensen_sum(pn, C, sigma, T, 1e-4); jf = eps_jensen_full(pn, C, sigma, 1e-4)
            pf = popfit_eps(pn, C, sigma, T, 1e-4)
            w.writerow([kind, par, f"{v:.3f}", f"{v/m:.4f}", f"{e4:.4f}", f"{e6:.4f}", f"{js:.4f}", f"{jf:.4f}", f"{pf:.4f}"])
            print(f"{kind:7s} {par:>5}: Var={v:8.2f} D={v/m:6.3f}  eps_sum(1e-4)={e4:.3f} (1e-6: {e6:.3f})  Jensen_sum={js:.3f}  Jensen_full={jf:.3f}  popfit={pf:.3f}", flush=True)
    # ---------------- Step 3: Jensen upper bound for the per-round release vs Monte-Carlo truth (from pilot A)
    rows = [r for r in csv.DictReader(open("pilot_a_results.csv")) if float(r["delta"]) == 1e-4]
    with open("theory_perround_bound.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T", "process", "param", "eps_true_mc", "eps_poisson", "jensen_full", "schema_maxp", "jensen/true"])
        for r in rows:
            Tr = int(r["T"]); kind = r["process"]; par = float(r["param"])
            P, pi, emit = process(kind, par)
            pn = count_distribution(P, pi, emit, Tr)
            jf = eps_jensen_full(pn, C, sigma, 1e-4)
            tr = float(r["eps_full_mc"])
            w.writerow([Tr, kind, par, f"{tr:.3f}", r["eps_poisson_pld"], f"{jf:.3f}", r["eps_schema_maxp"], f"{jf/tr:.2f}"])
            print(f"T={Tr} {kind} {par}: true={tr:.2f} poisson={float(r['eps_poisson_pld']):.2f} Jensen_full={jf:.2f} schema={float(r['eps_schema_maxp']):.2f}", flush=True)
    # ---------------- Step 2: crossover delta* where the Gaussian fit becomes anti-conservative (sum release, T=200)
    deltas = 10.0 ** np.arange(-3, -15, -1)
    with open("theory_crossover.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["process", "param", "var_N", "delta_star", "ratio_fit_exact_at_1e-4", "ratio_at_1e-8", "ratio_at_1e-12"])
        for kind, par in [("minsep", 9), ("minsep", 5), ("minsep", 2), ("markov", -0.1), ("iid", 0.0), ("markov", 0.3), ("markov", 0.6), ("markov", 0.9), ("markov", 0.97)]:
            P, pi, emit = process(kind, par)
            pn = count_distribution(P, pi, emit, T)
            n = np.arange(T + 1); m = np.sum(n * pn); v = np.sum(n * n * pn) - m**2
            ratios = {}
            dstar = None
            for d in deltas:
                ex = eps_sum_exact(pn, C, sigma, T, d, grid=40001); pf = popfit_eps(pn, C, sigma, T, d)
                ratios[d] = pf / ex if ex > 0 else np.nan
                if dstar is None and pf > ex * (1 + 1e-4):
                    dstar = d
            w.writerow([kind, par, f"{v:.2f}", dstar, f"{ratios[1e-4]:.4f}", f"{ratios[1e-8]:.4f}", f"{ratios[1e-12]:.4f}"])
            print(f"crossover {kind} {par}: Var={v:.1f} delta*={dstar}  fit/exact: 1e-4 {ratios[1e-4]:.3f} | 1e-8 {ratios[1e-8]:.3f} | 1e-12 {ratios[1e-12]:.3f}", flush=True)
    print("done", time.time() - t0)
