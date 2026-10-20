import numpy as np, time
from corraudit import *

rng = np.random.default_rng(1)
q, T, C, sigma = 0.1, 200, 1.0, 1.0

# 1. count distribution vs simulation
for name, (P, pi, emit) in {"iid": iid_chain(q), "markov0.9": two_state_chain(q, 0.9), "minsep5": minsep_chain(q, 5)}.items():
    pn = count_distribution(P, pi, emit, T)
    n = np.arange(T + 1)
    B = sample_paths(P, pi, emit, T, 200_000, rng)
    c = B.sum(axis=1)
    print(f"{name:10s} exact mean {np.sum(n*pn):.3f} var {np.sum(n*n*pn)-np.sum(n*pn)**2:.3f} | sim mean {c.mean():.3f} var {c.var():.3f} | stationary rate sim {B.mean():.4f} | tmix {mixing_time(P, pi)}")

# 2. Gaussian pair vs analytic Gaussian mechanism
from scipy.stats import norm
mu = 2.0
delta = 1e-5
e = eps_gauss_pair(0, 1, mu, 1, delta)
# analytic: delta(eps) = Phi(mu/2 - eps/mu) - e^eps Phi(-mu/2 - eps/mu)
f = lambda ee: norm.cdf(mu/2 - ee/mu) - np.exp(ee)*norm.cdf(-mu/2 - ee/mu)
print("gauss pair eps", e, "analytic delta at that eps", f(e), "(target", delta, ")")

# 3. full-release MC for iid vs PLD
t0 = time.time()
P, pi, emit = iid_chain(q)
for d in [1e-3, 1e-4]:
    e_mc = eps_full_mc(P, pi, emit, T, C, sigma, d, n=300_000, rng=rng)
    e_pld = eps_poisson_pld(q, T, C, sigma, d)
    e_sum = eps_sum_exact(count_distribution(P, pi, emit, T), C, sigma, T, d)
    print(f"iid delta={d}: eps_full_mc {e_mc:.3f} | PLD {e_pld:.3f} | eps_sum_exact {e_sum:.3f}")
print("time", time.time() - t0)
