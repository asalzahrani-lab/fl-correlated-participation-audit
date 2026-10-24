# Reviewer checks: where each claim can be verified

This file answers the artifact check requested in the sixth internal pre-submission review (updated after the seventh). It covers six items: version identity, traceable
results, PLD certification, window maximisation, timer accounting, and Monte-Carlo provenance. Each item lists the
files to open and the command to run.

The quickest check is:

```
cd src && python verify_claims.py      # a few seconds; exit status 0 if every check passes
```

`verify_claims.py` re-derives every headline number of the paper from the stored outputs in `results/` and prints
PASS or FAIL against the value in the manuscripts. It covers:

- the pooled ratios 9.4, 8.8, 8.35, 16.4, 16.7, 17.07 and 20.8;
- the per-device 22.2× and 28.3×, the window-search certificate and the PLD gap;
- the timer schema bound (3.21 / 6.62, 20× / 41×) and the 1.10 [1.07, 1.13] ratio;
- the audit counts (994 independent data sets and 1,594 certifications; 816 of 817 and 1,416 of 1,417; 0 of 177; none above the reference) and recoveries (74–90 %, Gaussian fit 3–15 %);
- the Monte-Carlo sample sizes, the disjointness of certificate and audit seeds, and the fifth-round counterexamples;
- the seventh-round computations: audit power at small violations, audits with a table built for a misspecified law,
  the sampling error of the law estimated from logs, round length at a fixed horizon and device subsampling, and the
  disjointness of their table and audit seeds;
- the tenth-round computations: ratios at smaller δ (to 1e-10), the per-device bracket at 1e-6 and 1e-8, the certified
  audit on the cyclic law and its 20M-sample references, the audit power against the number of canaries, and the
  failure of the observation model in a 510-parameter model.

The output of the last run is in `results/logs/verify_claims.log`.

The generating scripts recompute the underlying values; their run times range from seconds to hours (see the script
docstrings).

## 1. Version identity

This is release **v10**. It matches **TIFS v10** (main text and Supplementary Material) and the **extended version v15**
(`VERSION.md`).

The stored outputs were produced with NumPy 2.5.3, SciPy 1.18.1 and dp-accounting 0.6.0 (`VERSION.md`).
`requirements.txt` pins dp-accounting 0.6.0 and gives lower bounds for the other libraries. `verify_claims.py` was
last run with Python 3.13.16 and the same library versions.

## 2. Traceable results

- **Tables and figures.** `PROVENANCE.md` maps every table, every figure and every number in the text to its
  generating script, its output file and the column.
- **Headline claims.** `CLAIMS.md` gives, for each number in the abstract and the conclusion, the script, the output
  file and the value it contains.
- **Parameters.** Each script states its parameters in its docstring and at the top of the file: T, q, σ, C, δ, the
  grid, the number of Monte-Carlo samples and the number of audits.
- **Random seeds.** See the table below.

| Computation | Script | Seed(s) |
|---|---|---|
| Trace policy draw, running example and all seed-0 results | `rev_common.apply_policy`, called by `step21b_consistent.py`, `rev_peruser_exact2.py`, `rev_t1000_bracket.py`, `rev_timer_structural.py` | 0 |
| Twenty draws per policy | `rev_trace_robust.py` | 0–19 |
| Timer sensitivity | `rev_timer_seeds.py` | 0–99 |
| Certificate tables, T = 200, δ = 1e-4 | `step14_merged.py` | 2027 (Markov), 2028 (trace), 2029 (i.i.d.) |
| Retained references (10M Markov, 3M trace) and certificate tables, δ = 1e-5 | `step16_delta5.py` | 3031 (Markov), 3032 (trace) |
| Simulated audit data, T = 200 (the 120 audits per cell of Tables II, III, S2) | `step19_rerun_audits.py` | 1919 |
| Certificate tables, T = 1,000 (one process per part) | `step20b_parts.py` | (3031 or 3032) × 1000 + 17T + part |
| Simulated audit data, T = 1,000 | `step20b_parts.py` (audit stage) | 4041 (Markov), 4042 (trace) |
| Independent reference re-runs with 99 % intervals | `rev_refs.py` | 7101–7105 × 1000 + part |
| Cyclic law (tenth round): reference (20M), then the certified table (10M per grid point), then the 120 audit data sets, drawn in that order from one stream | `rev_round10_cyclic.py` | 1010 |
| Cyclic law, T = 1,000 reference | `rev_round10_cyclic_T1000.py` | 1012 |
| Canary sweep (tenth round): table extensions, then the audit data, in that order from one stream | `rev_round10_canaries.py` | 1011 |
| Erasure runs (tenth round) | `rev_round10_erasure.py` | 1013 |
| Guess-counting audits | `rev_guess2.py` | 8101–8103 |
| Training runs | `fl_mlp2.py`, `fl_mlp3.py`, `fl_mlp_5k.py`, `rev_train.py` | 777; 1717; 7777; 901–906 |
| Timer-cost training | `rev_timer_utility.py` | 5000 + s, 6000 + s (s = 0–4) |
| Mean-preserving spreads | `step15_convex_order.py` | 15 |
| Seventh round: certificate tables / references for the chain at λ = 0.3, 0.6, 0.8, 0.85, 0.89, 0.91, 0.95, 0.9 | `rev_round7_audit_sens.py` | 9100 + k / 9200 + k (k = 1–8 in that order) |
| Seventh round: audit data (power at λ = 0.3, 0.6, 0.8; misspecification, one stream at λ = 0.9 for every table) | `rev_round7_audit_sens.py` | 9301–9303; 9399 |
| Seventh round: sampling error of the log-estimated λ | `rev_round7_audit_sens.py lamse` | 9400 |
| Seventh round: device subsets | `rev_round7_trace_sens.py` | 0–19 (subset), 0 (policy draw) |

## 3. PLD certification (per-device lower end)

- **Script.** `rev_peruser_exact2.py`. The docstring states the construction.
  - The optimistic estimate of `dp_accounting` (privacy buckets: rounding down), with every composition run with
    `tail_mass_truncation=0` so that no mass is truncated or added to the infinity mass, is a lower estimate of ε(δ) for
    the mechanism with the rates as computed (grouped exactly, no rounding). It is the published lower end, at
    discretisation 1e-3. (Release v8 rounded the rates to 9 decimals and used the library default 1e-15, which
    dp-accounting 0.6.0 adds to the infinity mass even in optimistic mode; the change is at most 1.1e-8, section 7.)
  - The pessimistic estimate (connect-the-dots: rounding up, truncated tail mass added to the infinity mass) is an
    upper estimate.
- **Per-device values.** `results/rev_peruser_exact2_devices.csv`: columns `pr_lo_<δ>` (optimistic) and `pr_hi_<δ>`
  (pessimistic). The largest gap is 0.100 over the 402 devices.
- **Worst device refined.** Device 335 recomputed at discretisation 1e-4 gives [27.067, 27.076] at δ = 1e-4
  (`results/logs/rev_peruser_exact2.log`, second-to-last line).

## 4. Window maximisation

- **Upper end.** The upper end (Theorem 2 on the Poisson-binomial law) is computed exactly on all 24 windows of every
  device, independently of the lower-end search. The values are the array `thm3` in `rev_peruser_exact2.py`, and the
  per-device maximum and its window are the columns `thm3_<δ>` and `w_thm3`.
- **Independent recomputation of the upper end.** `rev_round5_upper_check.py` rebuilds each count law by the discrete
  Fourier transform of its characteristic function and inverts by bisection. Its per-device maxima agree with the
  published ones to 2.5e-7 (`results/rev_round5_upper_check.csv`, `results/logs/rev_round5_upper_check.log`).
- **Lower-end search.** All 24 windows are screened with the pessimistic estimate at discretisation 1e-2, which is an
  upper bound on each window's value.
  - Candidates are refined at 1e-3, and so is every window whose screen value exceeds the best certified lower value.
  - The per-device record is in the columns `n_refined`, `max_coarse_unrefined_<δ>` and `certified`; `certified` = 1
    for all 402 devices.
  - For the 20 largest devices all 24 windows were computed directly (`rev_peruser_exact2_top20.csv`, `same` = 1 for
    all 20).

## 5. Timer accounting

- **Draw-free schema bound.** `rev_timer_structural.py` computes it.
  - An exact dynamic programme over every draw of the two binding rounds shows that later rounds can bind.
  - The per-device greedy count, started unblocked in each window, bounds every draw's count.
  - Outputs: `results/rev_timer_structural.csv` and `results/rev_timer_schema.csv` (median 3.21, maximum 6.62 at
    δ = 1e-4 under the 20-round timer).
  - It is an upper bound, not a lower bound for an adversary who knows availability only.
- **Realised paths.** `rev_timer_peruser.py` → `rev_timer_peruser.csv`. These values are exact for an adversary who
  also sees the cohort draws, conditionally on those draws.
- **100-seed study.** `rev_timer_seeds.py` → `rev_timer_seeds.csv`. This is an empirical sensitivity check, not a
  bound.

## 6. Monte-Carlo provenance

- **Retained references.** These are the values in Table II and Table S2.
  - Markov law: 11.869 / 15.053 from 10M samples.
  - Trace law: 10.685 / 14.975 from 3M samples.
  - The sample sizes are the defaults of `step16_delta5.py` and are recorded in the column `reference_samples` of
    `step19_results.csv`.
  - Script `step16_delta5.py` (seeds 3031 / 3032); outputs `step16_accounting_markov.csv` and
    `step16_accounting_trace.csv`.
- **Independent re-runs with two-sided 99 % Chernoff–Hoeffding intervals.**
  - `rev_refs.py`, with sample sizes in column `n_mc`: Markov 10M (`rev_refs_markov200.csv`), trace 8M
    (`rev_refs_trace200.csv`).
  - The T = 1,000 intervals are computed from the stored samples (`rev_refs_stored.csv`: 3M Markov, 2M trace).
- **Audit data versus certificate samples.** These come from separate random streams.
  - The certificate tables use seeds 2027–2029 and 3031–3032; the T = 1,000 parts use the formula above.
  - The audit data use seed 1919 (T = 200) and 4041–4042 (T = 1,000).
  - `verify_claims.py` reads all of these seeds from the script sources and checks that the two sets are disjoint. The
    policy draw with seed 0, which defines the participation law, is shared by design.
  - The validity argument of the certified audit needs this independence: the table is fixed before, and independently
    of, the audit data.

## 7. Seventh- and ninth-round computations

- **Power at small violations.** `python rev_round7_audit_sens.py power 0.3` (and 0.6, 0.8; about 7 minutes each on one
  core). Output `rev_round7_audit_sens.csv`, rows with `mode` = power; per-audit estimates in
  `rev_round7_power_<λ>_est.npy`.
- **Misspecified law.** `python rev_round7_audit_sens.py misspec 0.91` (and 0.85, 0.89, 0.9, 0.95). The canaries follow
  λ = 0.9 and every table certifies the same 120 audit data sets (one stream, seed 9399), so the rows differ only by the
  table. The logs list the certified table value at each grid point; these values increase with the assumed λ' at all
  nine grid points.
- **Sampling error of the log-estimated law.** `python rev_round7_audit_sens.py lamse` (seconds) → `rev_round7_lamse.csv`.
- **Round length and device subsampling.** `python rev_round7_trace_sens.py` (under a minute; needs the FLASH trace in `trace/`,
  fetched by `bash trace/fetch_trace.sh`) → `rev_round7_trace_sens.csv`.
- After the parts are written, `python rev_round7_audit_sens.py combine` collects them into `rev_round7_audit_sens.csv`.
- `verify_claims.py`, section 7, re-derives every number of these computations quoted in the manuscripts and checks that
  their table and reference seeds (9101–9108, 9201–9208) are disjoint from their audit seeds (9301–9303, 9399).
- **Ninth round (certificate implementation).** `rev_peruser_exact2.py` now groups the inclusion rates exactly and
  runs every composition of the optimistic PLD with `tail_mass_truncation=0` (dp-accounting 0.6.0 otherwise truncates
  1e-15 and, in `self_compose`, adds it to the infinity mass even in optimistic mode). `python rev_round9_compare.py` compares the regenerated outputs with the
  v8 outputs (read from the repository's tag v8): largest change 1.1e-8, same windows and certificates.
  `verify_claims.py`, section 8, checks the source and the comparison.
