# Headline claims → computation

Each row is a number in the abstract or the conclusions of the journal version (TIFS v11), the script that produces it, the
output file, and the value it contains. "Certified" = a deterministic bound or a bound holding with stated confidence;
"estimate" = a Monte-Carlo point estimate.

| Claim (abstract / conclusion) | Type | Script | Output → value |
|---|---|---|---|
| Model exact for full-norm updates, whatever their directions, with exogenous, independent inclusions (so, under these hypotheses, the model's per-round curve is the guarantee of nonlinear DP-FedAvg with fixed clipping; the Monte-Carlo values estimate it and their upper 99 % limits bound it; under fixed cohorts the guarantee is the upper end); fixed-cohort bracket; exogeneity necessary for the law-based results, while the schema bound survives selection from released outputs (fully adaptive GDP composition, Smith–Thakurta) | proof + examples | `rev_cohort_example.py` | `results/logs/rev_cohort_example.log`: two-client 1.498 / 1.825 (scalar 1.091 / 1.354, cohort seen 1.600 / 1.908); orthogonal other update 1.186; adaptive selection 1.600 = ½h(a) vs 1.091 |
| Pooled sum-statistic (final-model) loss up to 9.4× over independent participation (δ = 1e-4), more at smaller δ | exact (Theorem 1) | `step21b_consistent.py`; `rev_round10_small_delta.py` | `results/step21b_policies.csv`, row "take all available": `ratio_1e-4` = 9.44; `rev_round10_small_delta_policies.csv`: every policy's ratio grows from 1e-4 to 1e-10 (take-all 12.39 at 1e-10) |
| Per-round, running example, T = 200: estimated 8.8× | estimate (3M MC) | `step19_rerun_audits.py`, `step21b_consistent.py` | reference 10.685 / Poisson 1.2188 = 8.77 (`step19_results.csv`, `step21b_policies.csv`) |
| … at least 8.35× with 99 % confidence | certified (KL interval) | `rev_refs.py` (law `trace`, 8M) | `results/rev_refs_trace200.csv`: lower limit 10.174 / 1.2188 = 8.35 |
| … upper end 16.4× | exact (Theorem 2) | `step21b_consistent.py` | `step21b_policies.csv` `thm3_bound_1e-4` = 19.942 / 1.218 = 16.4 |
| T = 1,000: 17 to 21× | certified lower (99 %) / exact upper | `step20b_parts.py`, `rev_refs.py stored`, `rev_t1000_bracket.py` | lower limit 50.43 / 2.954 = 17.07 (`rev_refs_stored.csv`); Theorem 2: 61.34 / 2.954 = 20.8 (`rev_t1000_bracket.csv`) |
| Worst device: claim must cover ≥ 22× | certified numerically (optimistic PLD, rates grouped exactly, composed with no tail truncation; v9) | `rev_peruser_exact2.py` | `rev_peruser_exact2_summary.csv` `pr_lo` max 26.990 (1e-3; 22.1×); `rev_peruser_exact2.log`: device 335 at discretisation 1e-4 [27.067, 27.076] → 22.2× of 1.2189 |
| Worst device: guarantee to report 28.3× (upper end 34.53) | exact (Theorem 2 on the Poisson-binomial law) | `rev_peruser_exact2.py` | `rev_peruser_exact2_summary.csv` `thm3` max 34.527 (device 176) / 1.2189 = 28.3 |
| Worst-window search for the lower end (certified) | branch-and-bound with pessimistic 1e-2 screen | `rev_peruser_exact2.py` | `rev_peruser_exact2_devices.csv`: per device `n_refined` (windows computed at 1e-3), `max_coarse_unrefined_<δ>` (largest screen value left unrefined), `certified` = 1 for all 402; `rev_peruser_exact2_top20.csv`: all 24 windows at 1e-3 for the 20 largest devices, `same` = 1 for all 20 |
| Upper end maximised over windows | exact on every window (no screening) | `rev_peruser_exact2.py` | array `thm3` over all 24 windows; `rev_peruser_exact2_devices.csv` `thm3_<δ>`, `w_thm3` |
| Audit recovers 74–90 %; Gaussian fit 3–15 % | estimates over 120 audits | `step19_rerun_audits.py` | `step19_results.csv` |
| Main evaluation, 994 independent audit data sets (1,594 certifications: each simulated data set under a violated claim is certified at both δ; the cyclic-law audits of `rev_round10_cyclic.py` included): violated claims (2.7× to 17×) flagged on 816 of 817 data sets (1,416 of 1,417 certifications), correct claims on 0 of 177; never above the reference | counts | `derived_numbers.py` | printed by `src/derived_numbers.py` (section 5) |
| 20-round timer: pooled per-round ratio 1.10 [1.07, 1.13] | estimate with 99 % interval | `rev_refs.py` (law `timer20`) | `rev_refs_timer20.csv` |
| Timer: pool exceeds the cohort in 2 of 432 rounds (realised run); later rounds can bind under some draws | count / exact search | `rev_timer_peruser.py`, `rev_timer_structural.py` | `rev_timer_peruser.log` (binding 2/432); `rev_timer_structural.log` (max pool 28 > K = 27 at t = 21, one round; ten rounds under the 5-round timer) |
| Timer: per-device schema bound median 20× (3.21 vs 0.162), max 41× (6.62) | certified upper bound (every draw); not a lower bound for an adversary who knows availability only | `rev_timer_structural.py` | `rev_timer_schema.csv` |
| Timer: realised paths (median 3.21, max 6.62; 4.85 / 14.45 for the 5-round timer) | exact for an adversary who sees availability and the draws, conditional on them | `rev_timer_peruser.py` | `rev_timer_peruser.csv` |
| Timer: 100-seed sensitivity (344 / 334 of 402 devices with unchanged paths; count changed by one for 3 / 12) | empirical sensitivity, not a bound | `rev_timer_seeds.py` | `rev_timer_seeds.csv`, `results/logs/rev_timer_seeds.log` |
| Upper end recomputed independently (all 402 devices × 24 windows, both δ) | exact, second implementation | `rev_round5_upper_check.py` | `rev_round5_upper_check.csv`; `results/logs/rev_round5_upper_check.log`: max 34.5266 (device 176, window 0) at 1e-4, 37.6676 at 1e-5; max abs difference from `thm3_<δ>` 2.5e-7 |
| Dispersion does not determine the final-model curve: P(19) = 18/181, P(20) = 0.9, P(200) = 1/1810 (mean 20, variance 18, D = 0.9) | exact (Theorem 1) | `rev_round5_checks.py` | `rev_round5_checks.csv`, `results/logs/rev_round5_checks.log`: 23.0086 vs Binomial 2.6888 at 1e-4; 31.4378 vs 3.1534 at 1e-5 |
| A deterministic count does not by itself dominate rate-matched Poisson: T = 2, a = 0.5, rate 1/2, random allocation vs independent participation | numerical quadrature (4,001² grid) | `rev_round5_checks.py` | `rev_round5_checks.log`: total variation 0.1440283 vs 0.1397986; at ε = 1, H(P‖Q) 6.46e-4 vs 2.04e-3 and H(Q‖P) 4.79e-4 vs 4.9e-7 |

### Tenth-round computations (TIFS IV-E, V-E, VI-E, Supplementary Material; not in the abstract except where noted)

| Claim | Type | Script | Output → value |
|---|---|---|---|
| Smaller δ: final-model ratio 4.93 / 7.91 at 1e-8 and 5.58 / 8.37 at 1e-10 (chain / trace); at T = 200, Theorem 2 over the Poisson value rises from 7.73 to 8.64 (chain) and stays at 16.27–16.74 (trace, the upper end of the pooled per-round bracket); ratios that fall: the cyclic law's per-round ratio (2.77 → 2.74 at 1e-5; 3.85 → 3.72 at T = 1,000), the trace's at T = 1,000 (MC 17.4 → 16.9; Theorem 2 20.8 → 17.6 at 1e-10), and the per-device ratios | exact / closed form | `rev_round10_small_delta.py` | `rev_round10_small_delta.csv` |
| Per device at 1e-8: largest lower end over devices 17.5× (device 335) and largest upper end 22.0× (device 176) the claim (the claim grows faster than the losses, 36.18 and 45.51) | certified numerically / exact | `rev_round10_peruser_delta.py` | `rev_round10_peruser_summary.csv` |
| Cyclic law: certified audit 7.16 (90 %), 120 of 120 flags, none above the reference 7.94 [7.89, 7.98]; T = 1,000 ratio 3.9 | estimates over 120 audits; references with 99 % intervals (20M samples) | `rev_round10_cyclic.py`, `rev_round10_cyclic_T1000.py` | `rev_round10_cyclic_results.csv`, `rev_round10_cyclic_reference.csv`, `rev_round10_cyclic_T1000.csv` |
| Where Model 1 fails: erasure ratio 0.50 (510-parameter softmax regression, random directions) against 0.95 (low-variance directions) | 10 and 4 training runs | `rev_round10_erasure.py` | `rev_round10_erasure.csv` |
| Audit power against the number of canaries: certified recovery 68–92 % (Markov) and 44–83 % (trace) for 100 to 5,000 canaries; gap of a_lo about z/√n_obs | estimates over 120 audits per cell | `rev_round10_canaries.py` | `rev_round10_canaries.csv` |
### Seventh-round computations (TIFS V-F, VI-G and Section I; not in the abstract)

| Claim | Type | Script | Output → value |
|---|---|---|---|
| The audit flags 120/120 at λ = 0.8 and 0.6 (true loss 2.29× and 1.36× the claim) and 11/120 at λ = 0.3 (1.08×); detection beyond about 1/0.88 ≈ 1.14× | estimates over 120 audits; references with 99 % intervals (2M samples) | `rev_round7_audit_sens.py power` | `rev_round7_audit_sens.csv` (`mode` = power) |
| Tables for λ' = 0.85, 0.89, 0.9, 0.91, 0.95 with canaries at λ = 0.9: 0, 0, 1, 16, 120 of 120 estimates above ε_true = 11.869 | estimates over the same 120 audit data sets | `rev_round7_audit_sens.py misspec` | `rev_round7_audit_sens.csv` (`mode` = misspec, `above_true`) |
| The λ fitted to 1,000 devices' logs (200 rounds) has standard deviation 0.002 | 200 simulated replicates | `rev_round7_audit_sens.py lamse` | `rev_round7_lamse.csv` `sd_hat` = 0.00215 |
| The factors are those of 10-minute rounds and grow as rounds shorten: 8.9, 5.2, 2.5 at 5-, 10-, 30-minute rounds (33 h) | exact (Theorem 1) | `rev_round7_trace_sens.py` | `rev_round7_trace_sens.csv` (`check` = round_length) |
| Over 20 random halves of the devices the final-model ratio is 3.7 to 7.3 | exact (Theorem 1) per subset | `rev_round7_trace_sens.py` | `rev_round7_trace_sens.csv` (`check` = device_subsample, 286 devices) |

All rows above are re-derived from the stored outputs by `src/verify_claims.py` (PASS / FAIL per number; log in
`results/logs/verify_claims.log`); `REVIEWER_CHECKS.md` maps the artifact checks to files and commands.
