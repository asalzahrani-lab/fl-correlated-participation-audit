"""Regression tests for the zero-return convention of the lower confidence limit a_lo and the table lookups
(round 12).  The paper (Section V-C) sets a_lo := 0, hence eps_hat = 0, when there is no participating
present-canary observation (n_obs = 0) or fewer than two absent-canary observations (n_0 < 2).  Before round 12
the code divided first and produced NaN, which np.searchsorted mapped to the LAST grid entry, i.e. a positive
certificate.  No published configuration (at least 40 canaries over 200 rounds) can reach these cases; the
tests pin the convention for the reusable code.

    cd src && python test_lower_limit.py        # prints PASS/FAIL per case, exit status 0 if all pass
"""
import sys
import numpy as np
from scipy.stats import chi2, norm
import certified2, certified

fails = 0


def check(name, cond):
    global fails
    fails += not cond
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


T = 200
rng = np.random.default_rng(27102026)
grid = np.round(np.arange(0.40, 0.5201, 0.01), 2)
table_vals = np.linspace(5.0, 8.3928, len(grid))     # any increasing table; the last entry is the trap


class Lookup:                                        # the floor-rule lookup of CertifiedTable2
    a_grid = grid; eps_bern = table_vals
    _floor = certified2.CertifiedTable2._floor
    cert = certified2.CertifiedTable2.cert


lk = Lookup()

# 1. one canary, never present: n_obs = 0
B = np.zeros((1, T), np.int8); B[0, ::10] = 1
X = 0.5 * B + rng.standard_normal((1, T))
a = certified2.snr_lower_bound(X, np.array([False]), B, 0.045)
check("n_obs = 0: a_lo is 0 (not NaN)", a == 0.0)
check("n_obs = 0: certificate is 0", lk.cert(a) == 0.0)
a_old = certified.snr_lower_bound(X, np.array([False]), B, 0.05)
check("n_obs = 0 (certified.py): a_lo is 0", a_old == 0.0)

# 2. one canary, present: n_0 = 0
a = certified2.snr_lower_bound(X, np.array([True]), B, 0.045)
check("n_0 = 0: a_lo is 0", a == 0.0 and lk.cert(a) == 0.0)

# 3. two canaries, both present, one absent observation only is impossible with T = 200; build n_0 = 1 directly
X1 = np.vstack([X, rng.standard_normal((1, 1)).repeat(T, axis=1)])   # second canary absent, but T observations
Xsmall = np.array([[0.7], [0.1]]); Bsmall = np.array([[1], [0]], np.int8)   # T = 1: one absent observation
a = certified2.snr_lower_bound(Xsmall, np.array([True, False]), Bsmall, 0.045)
check("n_0 = 1: a_lo is 0", a == 0.0)

# 4. a direct non-finite lookup cannot return a positive certificate
check("direct NaN lookup: certificate 0", lk.cert(float("nan")) == 0.0)
check("direct +inf lookup: certificate 0", lk.cert(float("inf")) == 0.0)
check("direct -inf lookup: certificate 0", lk.cert(float("-inf")) == 0.0)
check("below-grid lookup: certificate 0", lk.cert(0.39) == 0.0)
check("on-grid lookup unchanged", lk.cert(0.52) == table_vals[-1] and lk.cert(0.455) == table_vals[5])

# 5. an ordinary valid case: the guarded function equals the unguarded formula
m = 1000
Bm = (rng.random((m, T)) < 0.1).astype(np.int8); pres = rng.random(m) < 0.5
Xm = (Bm * pres[:, None]) * 0.5 + rng.standard_normal((m, T)).astype(np.float32)
a = certified2.snr_lower_bound(Xm, pres, Bm, 0.045)
Bp = Bm[pres].astype(float); Xp = Xm[pres].astype(float); nobs = Bp.sum(); shift = (Bp * Xp).sum() / nobs
xa = Xm[~pres].astype(float).ravel(); n0 = len(xa); sd = xa.std(ddof=1)
sd_hi = sd * np.sqrt((n0 - 1) / chi2.ppf(0.045 / 2, n0 - 1)); z = norm.ppf(1 - 0.045 / 2)
a_ref = max(shift - z * sd_hi / np.sqrt(nobs), 0.0) / sd_hi
check(f"valid case: a_lo = {a:.6f} equals the unguarded formula ({a_ref:.6f})", abs(a - a_ref) < 1e-12 and 0.4 < a < 0.5)

print("ALL TESTS PASSED" if fails == 0 else f"{fails} TEST(S) FAILED")
sys.exit(1 if fails else 0)
