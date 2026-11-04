"""Round 15: the growth of the dispersion D = Var(N_T) / E[N_T] with the horizon, decomposed by device (Sections VI-C
and S-VI.B).

Setting: the two running-policy panels of step21b_consistent.py (fixed cohort 0.6 x pool, seed 0): 10-minute rounds on
the 3-day window (572 devices, 432 rounds, T = 50 to 400, stride 10) and 5-minute rounds on the 4-day window (486
devices, 1,152 rounds, T = 100 to 1,000, stride 5).  Every device contributes the same number of windows per horizon,
so the device weights are equal and the law of total variance holds exactly for the window counts:
    V = B + W,   B = Var_i(mu_i),   W = E_i[v_i],
mu_i the mean and v_i the (population) variance of device i's window counts.  Reported per horizon: the number of
windows per device (overlapping, at the panel's stride; and non-overlapping, stride T), the mean count, V, B, W,
B + W - V, D = V / mean, B / mean, W / mean, the between-device share B / V, and the same quantities on the
non-overlapping windows where at least two exist.  Per panel: the unweighted OLS regressions on T (the fit of
derived_numbers.py, with an intercept) of D, B / mean and W / mean, whose slopes add exactly (the regression is
linear), the Binomial benchmark coefficient Var(p_i) / E[p_i] from the devices' realised rates over the whole window
(the quantity step21b_tails.csv calls VarP_over_EP), and the ratio of the fitted slope of D to that coefficient --
the number the v14 text printed as "77 % and 80 % of the slopes".

What the decomposition shows (v15 text): the between-device component alone, T Var(p_i) / E[p_i] up to the window's
edge effects, has a slope above the fitted slope of D, and the within-device component falls with T because a device's
overlapping windows coincide more as T approaches the log's length; the fitted slopes alone do not identify a share of
the growth due to differences between devices.  The share B / V is what the text reports, with the number of windows
behind it.

    cd results && P3_TRACE=../trace python3 ../src/rev_round15_horizon_decomp.py        # under a minute
Outputs rev_round15_horizon_decomp.csv (one row per panel and horizon) and rev_round15_horizon_slopes.csv (one row
per panel).
"""
import csv, time
import numpy as np
from rev_common import window, apply_policy

PANELS = [(10, 3, [50, 100, 200, 300, 400], 10), (5, 4, [100, 200, 400, 600, 800, 1000], 5)]
t0 = time.time()
log = lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True)


def window_counts(B, T, stride):
    """Device x window matrix of participation counts over windows of length T at the given stride."""
    N, Tw = B.shape
    return np.array([[B[i, s0:s0 + T].sum() for s0 in range(0, Tw - T + 1, stride)] for i in range(N)], dtype=np.float64)


def decompose(c):
    mu = c.mean(); V = c.var(); Bv = c.mean(axis=1).var(); W = c.var(axis=1).mean()
    return mu, V, Bv, W


def ols(T, y):
    sl, ic = np.polyfit(T, y, 1); res = y - (sl * T + ic)
    return sl, ic, 1.0 - res.var() / y.var()


if __name__ == "__main__":
    rows, slopes = [], []
    for round_min, win_days, Ts, stride in PANELS:
        A, _, _ = window(round_min, win_days)
        B = apply_policy(A, "fixed", 0.6, 0, np.random.default_rng(0)); N, Tw = B.shape
        p = B.mean(axis=1); VE = p.var() / p.mean(); Ep = p.mean()
        log(f"{round_min}-min panel: {N} devices, {Tw} rounds, E[p] {Ep:.4f}, Var(p)/E[p] {VE:.4f}, Binomial intercept 1 - E[p] = {1 - Ep:.3f}")
        D_, Bm_, Wm_ = [], [], []
        for T in Ts:
            c = window_counts(B, T, stride); nw = c.shape[1]; mu, V, Bv, W = decompose(c)
            cn = window_counts(B, T, T); nwn = cn.shape[1]
            if nwn >= 2:
                mun, Vn, Bn, Wn = decompose(cn); Dn, shn = Vn / mun, Bn / Vn
            else:
                Dn = shn = float("nan")
            D_.append(V / mu); Bm_.append(Bv / mu); Wm_.append(W / mu)
            rows.append([round_min, T, N, nw, nwn, f"{mu:.4f}", f"{V:.4f}", f"{Bv:.4f}", f"{W:.4f}", f"{Bv + W - V:.3e}", f"{V / mu:.4f}", f"{Bv / mu:.4f}", f"{W / mu:.4f}",
                         f"{Bv / V:.5f}", f"{T * VE:.4f}", f"{Dn:.4f}" if nwn >= 2 else "", f"{shn:.5f}" if nwn >= 2 else ""])
            log(f"  T={T}: windows/device {nw} (non-overlapping {nwn}); mean {mu:.2f}; V {V:.2f} = B {Bv:.2f} + W {W:.2f} (residual {Bv + W - V:.1e}); "
                f"D {V / mu:.2f}; B/mean {Bv / mu:.2f} (T Var(p)/E[p] = {T * VE:.2f}); W/mean {W / mu:.2f}; B/V {Bv / V:.3f}"
                + (f"; non-overlapping: D {Dn:.2f}, B/V {shn:.3f}" if nwn >= 2 else "; non-overlapping: one window per device"))
        Tv = np.array(Ts, float)
        sD, iD, rD = ols(Tv, np.array(D_)); sB, iB, rB = ols(Tv, np.array(Bm_)); sW, iW, rW = ols(Tv, np.array(Wm_))
        slopes.append([round_min, N, Tw, f"{Ep:.5f}", f"{VE:.5f}", f"{sD:.5f}", f"{iD:.3f}", f"{rD:.5f}", f"{sB:.5f}", f"{iB:.3f}", f"{rB:.5f}", f"{sW:.5f}", f"{iW:.3f}", f"{rW:.5f}",
                       f"{sB + sW - sD:.2e}", f"{sD / VE:.4f}", f"{VE / sD:.4f}"])
        log(f"  OLS on T: D slope {sD:.4f} intercept {iD:.2f} R2 {rD:.4f}; B/mean slope {sB:.4f} intercept {iB:.2f}; W/mean slope {sW:.4f} intercept {iW:.2f}; "
            f"slopes add to {sB + sW:.4f}; fitted slope / Var(p)/E[p] = {sD / VE:.4f} (the v14 '77 % / 80 %'); Var(p)/E[p] / slope = {VE / sD:.4f}")
    with open("rev_round15_horizon_decomp.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["round_min", "T", "n_devices", "windows_per_device", "nonoverlap_windows_per_device", "mean_count", "V", "B", "W", "B_plus_W_minus_V",
                                       "D", "B_over_mean", "W_over_mean", "B_over_V", "T_VarP_over_EP", "D_nonoverlap", "B_over_V_nonoverlap"]); w.writerows(rows)
    with open("rev_round15_horizon_slopes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["round_min", "n_devices", "n_rounds", "E_p", "VarP_over_EP", "slope_D", "intercept_D", "R2_D", "slope_B_over_mean", "intercept_B_over_mean", "R2_B",
                                       "slope_W_over_mean", "intercept_W_over_mean", "R2_W", "slope_sum_minus_slope_D", "slopeD_over_VarPEP", "VarPEP_over_slopeD"]); w.writerows(slopes)
    log("done")
