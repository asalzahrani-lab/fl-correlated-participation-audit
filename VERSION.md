# Version

Release **v12** (2026-10-28), matching the journal submission **TIFS v12** (main text and Supplementary Material) and the
**extended version v17**. Earlier releases: v11 (TIFS v11 / extended v16), v10 (TIFS v10 / extended v15), v9 (TIFS v9 / extended v14), v8 (TIFS v8 / extended v13), v7 (TIFS v7 / extended v12), v6 (TIFS v6 / extended v11), v5 (TIFS v5 / v10), v4 (TIFS v4 / v9), v3 (TIFS v3 / v8),
v2 (TIFS v2 / v7).

What changed in v12 (twelfth internal pre-submission review, three independent reports on TIFS v11): one defect fixed, one
test script added, one configuration added, one summary extended. No published value changed. **Defect.** With no
participating observation of a present canary (`n_obs = 0`) or fewer than two absent-canary observations (`n_0 < 2`),
`snr_lower_bound` in `certified2.py` and `certified.py` divided first and returned NaN, which `np.searchsorted` mapped to
the last grid entry, so the table lookup returned the certificate of the largest grid value (8.39 at a = 0.52) instead of
the 0 that Section V-C of the paper defines. Both functions now return 0 in those cases and whenever the estimate is not
finite; `CertifiedTable2.cert`, `CsvTable.cert` (`rev_train.py`), `cert_floor` (`step19_rerun_audits.py`), the lookups in
`fl_mlp2.py`, `fl_mlp_5k.py`, `rev_round10_canaries.py` and `rev_round10_cyclic.py`, and the inline lookups of the
table-building scripts `step14_merged.py`, `step16_delta5.py`, `step20_T1000.py` and `step20b_parts.py` return 0 for a
non-finite argument.
`test_lower_limit.py` pins the convention with eleven cases (`cd src && python test_lower_limit.py`). No stored result is
affected: every lower confidence limit of the 300 audited training runs is finite (checked by `verify_claims.py`), and no
configuration of the paper (at least 40 canaries over 200 rounds) can reach the case. **Baseline.** `rev_train.py` gains
`ctrl200_share00`, fifty trainings at the 200 real clients of `base_orth` with no canaries and no audit (seed 901, hence
the same clipping norm 0.4199): mean accuracy 0.7309 (s.d. 0.0100, s.e. 0.0014, range 0.708-0.754), against 0.730 with
1,000 canaries, 0.729 with 200 and 0.731 with 40. `rev_train_summary.py` writes it to `rev_train_controls_summary.csv` with
the training columns only; `rev_train_table.py` prints it as row 5 of Table S4 (extended Table VII). `verify_claims.py`
gains a twelfth-round section (11 checks; 226 in all, all passing) and its sections are now numbered consecutively.
**Text.** The abstract (249 words, no symbols), the "stateless" scope of the equality claim, the credit for the Monte-Carlo
accountant (Wang et al., NeurIPS 2023), the gate sentence (keeps Theorem 3's guarantee under its assumptions), the
certified-uncertified gap over all settings ("3 to 9 points", from 2.7 to 9.5), the control audits (100 outside the 994
data sets, none above the reference, 68 flags; the one below-grid run of the 200-canary arm), and the initialisation and
data split in the supplement's protocol paragraph.

Values that differ between release v11 and release v12, with the reason:

| Quantity | v11 | v12 | Reason |
|---|---|---|---|
| Lower confidence limit with `n_obs = 0` or `n_0 < 2` (reusable code only) | NaN, looked up as the last grid entry (a positive certificate) | 0, certificate 0 | zero-return convention of Section V-C; no published configuration reaches the case |
| Accuracy at 200 real clients without canaries | not run | 0.731 (50 runs, range 0.708-0.754) | twelfth-round baseline; Section V-F, Table S4 row 5 |
| Certified-uncertified gap "across all settings" (extended V-F) | "4 to 9 points" (stale) | "3 to 9 points" | the gaps over every setting are 2.7, 3.4, 3.4, 3.7, 4.8, 5.2, 6.1, 6.1, 7.1, 9.2, 9.5 |
| Abstract, Gaussian-fit recovery | 3-15 % | 3-22 % | the fit recovers 15.0 % (Markov), 3.3 % (trace), 21.5 % (cyclic); the old range covered the first two |
| `verify_claims.py` sections | 1-8, 10, 11 | 1-11 | consecutive numbering |

What changed in v11 (eleventh internal pre-submission review): one script added, two configurations added, two
summary scripts extended. No earlier output changed except the step-function column of Table III, which now uses the
protocol of every other column, and the stored precision of one column of `rev_train_summary.csv` (see the table below). `rev_round11_step.py` replays the certified tables of `step16_delta5.py` from their seeds
(3031 Markov, 10M samples; 3032 trace, 3M), checks that every published KL-certified entry is reproduced (all 9 + 5 grid
points at 1e-4 and 1e-5, to the printed four decimals), computes the step-function (Clopper-Pearson) certificate on the
same samples, and certifies the same 120 audits per law as the other columns (`step19_audits.npz`; the cyclic law's
step-function entry is that of `rev_round10_cyclic.py`). `rev_train.py` gains `ctrl200_share05` and `ctrl200_share01`,
accuracy controls at the 200 real clients of `base_orth` (seed 901, hence the same clipping norm). `rev_train_summary.py`
writes their summary to a separate file, `rev_train_controls_summary.csv`, so `rev_train_summary.csv` keeps the 200 runs of
the table's text, and now stores `a_hat` with five decimals so that the three-decimal table entries are rounded once
(no printed value changes). `rev_train_table.py` prints the two control rows. `verify_claims.py` gains an eleventh-round
section (32 checks; 214 in all, all passing). The first attempt at trace grid points 3-4 was stopped by the environment at 2.8M of 3M samples, so it
was rerun from the start; its log is kept as `rev_round11_step_trace_34_first_attempt_stopped.log`.

Values that differ between release v10 and release v11, with the reason (no v10 value was reported in a submission):

| Quantity | v10 | v11 | Reason |
|---|---|---|---|
| Table III, step-function column, Markov / i.i.d. / trace | 8.38 ± 0.18 (71 %) / 2.25 ± 0.05 (78 %) / 5.38 ± 1.17 (50 %) | 9.54 ± 0.51 (80 %) / 2.41 ± 0.07 (84 %) / 6.91 ± 0.79 (65 %) | recomputed with the current protocol: the same 120 audits per law as the other columns (v10: step 14's own audits), the 0.045/0.005 α split (v10: 0.025/0.025) and the 0.01 grid floor (v10: 0.05), on the same Monte-Carlo samples as the KL column |
| Table III, step-function column, cyclic | — | 6.74 ± 0.26 (85 %) | the tenth-round cyclic run already computed it with the current protocol; shown now that the column is uniform |
| Cost of the step-function certificate (extended V-F) | 16–31 points below the uncertified estimate | 8–16 points (5–9 below the KL certificate) | follows from the new column |
| KL-certificate cost quoted in extended V-F | 4–7 points | 3–7 points | the cyclic law (2.7 points) was not yet in that sentence; the journal text already said 3 to 7 |
| `rev_train_summary.csv`, column `a_hat` | four decimals | five decimals | single rounding of the table entries; no printed value changes |

What changed in v10 (tenth internal pre-submission review): six scripts, all additions. No earlier output changed except
that the cyclic law's per-round references were recomputed from 20M samples: 7.94 [7.89, 7.98] at T = 200 and 28.04 at
T = 1,000, against 8.02 and 27.79 from the 300k-sample pilots, so the derived ratios become 2.8x and 3.9x and the Theorem 2
looseness 1.29 and 1.08. `rev_round10_small_delta.py`: the pooled ratios at delta down to 1e-10 (exact or closed form).
`rev_round10_peruser_delta.py` gives the per-device bracket at 1e-6 and 1e-8; the losses grow, but the claim grows faster: the
largest lower and upper ends over devices, devices 335 and 176, are 17.5x and 22.0x the claim at 1e-8. `rev_round10_cyclic.py` and `rev_round10_cyclic_T1000.py`
run the certified audit on the cyclic law, not run before, which now enters the main evaluation (994 independent data
sets, 1,594 certifications, 816 of 817 violated claims flagged). `rev_round10_canaries.py`: audit power against the number
of canaries. `rev_round10_erasure.py` re-runs and logs the configuration in which the observation model fails.
`verify_claims.py` gains a tenth-round section and counts the cyclic audits; `derived_numbers.py` counts them too. The
extended version now uses the journal numbering of the theorems (see Numbering below).

Values that differ between release v9 and release v10, with the reason (no v9 value was reported in a submission):

| Quantity | v9 | v10 | Reason |
|---|---|---|---|
| Cyclic law, per-round reference, T = 200, delta = 1e-4 | 8.02 (300k samples) | 7.94 [7.89, 7.98] (20M) | recomputed with a 99 % interval; 8.02 lies outside it |
| Cyclic law, per-round ratio, T = 200 / T = 1,000 | 2.8 / 3.8 | 2.8 / 3.9 | T = 1,000 reference recomputed from 20M samples: 27.79 → 28.04 |
| Theorem 2 over the reference, cyclic law, T = 200 / T = 1,000 | 1.28 / 1.09 | 1.29 / 1.08 | follows from the two new references |
| Independent audit data sets / certifications | 874 / 1,354 | 994 / 1,594 | the 120 cyclic-law audits added, each certified at both delta |
| Violated claims flagged | 696 of 697 | 816 of 817 | same |
| Range of violations in the main evaluation | 2.9- to 17-fold | 2.7- to 17-fold | the cyclic law's violation is 2.77-fold |
| Uncertified estimates above the reference | 6 of 360 | 11 of 480 | the cyclic law adds 5 of 120 |
| Table build time quoted in the text | 32 min to 2.6 h | 15 min to 2.6 h | the range now starts at the i.i.d. table (15 min in Table S1 of both versions); no time was re-measured |

What changed in v9 (ninth internal pre-submission review, a code inspection): `rev_peruser_exact2.py` groups the per-round
inclusion rates exactly (v8 rounded them to 9 decimals first) and composes the optimistic PLD, within a rate and across rates, with
`tail_mass_truncation=0`; dp-accounting 0.6.0 otherwise applies a default truncation of 1e-15 in both compositions and adds it to
the infinity mass in `self_compose` even in the optimistic mode. The lower end of the per-device bracket is then a lower estimate of the mechanism with the rates as
computed. All per-device outputs were regenerated (`rev_peruser_exact2_*.csv`, log). The largest change in any per-device value is 1.1e-8, none at the printed precision; worst windows, refinement counts and certificates unchanged (rev_round9_compare.py). The guarantee to report
is now quoted as 28.3× (upper end 34.53) instead of the rounded-down 28×. No other computation changed.

Post-release documentation fixes on `main` (2026-10-22; no code path, result or number changed, tag v9 unchanged): `PROVENANCE.md` no longer names an internal runner script that is not in the release (`rev_run_train.sh` runs both training batches). `README.md` and `rev_round9_compare.py` state that the comparison with v8 needs tag v8 (a full clone) and stop with that message instead of a traceback. 2026-10-23: `paper/make_figures.py` reads the stored outputs from `results/` (override with `P3_RESULTS`) and writes the figures next to itself, where it used absolute paths of the author's machine before; it embeds TrueType fonts in place of Type 3 (IEEE PDF requirements). The three figures were regenerated from the same data; by pixel comparison only glyph rasterisation differs. `src/rev_peruser_rows.py --patch` takes the manuscript files to patch as arguments.

Review history. Every "review round" or "revision round" named in this package (file names `rev_*`, script docstrings,
this file, README.md, PROVENANCE.md, REVIEWER_CHECKS.md) refers to internal pre-submission reviews of the manuscript.
The manuscript has not been submitted to a journal before. "TIFS vN" and "extended vN" are internal draft labels.

Computed with NumPy 2.5.3, SciPy 1.18.1, dp-accounting 0.6.0, on one or two CPU cores; no GPU. `src/verify_claims.py`
was last run with Python 3.13.16 and these library versions.

What changed after the eighth internal pre-submission review (2026-10-20, same release v8: the manuscript had not been
submitted): `verify_claims.py` and `derived_numbers.py` also count independent audit data sets (874; violated claims
flagged on 696 of 697), because each simulated data set on a correlated law is certified at both δ (1,354 certifications).
They check the training recovery range 73–93 % too. No computed number changed.

What changed in v8 (seventh internal pre-submission review): `rev_round7_audit_sens.py` and
`rev_round7_trace_sens.py`. The first covers the power of the certified audit at small violations (λ = 0.3, 0.6, 0.8), audits of λ = 0.9 canaries certified with tables built for
λ' = 0.85, 0.89, 0.9, 0.91, 0.95, and the sampling error of the maximum-likelihood λ from the logs of 1,000 devices over 200
rounds; outputs `rev_round7_audit_sens.csv`, `rev_round7_lamse.csv`, logs in `results/logs/`. The second
covers round length at a fixed 33-hour horizon (5-, 10- and 30-minute rounds) and 20 random halves
and four-fifths of the devices; output `rev_round7_trace_sens.csv`. `verify_claims.py` gains a section of
seventh-round checks, including the disjointness of their table and audit seeds. No published number changed. The
new numbers are in TIFS Sections V-F and VI-G and extended Sections V-F and VI-G.

What changed in v7 (sixth internal pre-submission review): `src/verify_claims.py`, which re-derives every headline number from the stored outputs
and checks that certificate and audit seeds are disjoint (output `results/logs/verify_claims.log`, all checks pass), and
`REVIEWER_CHECKS.md`, which maps the six artifact checks of the review to files, columns and commands, with a table of random
seeds. No published number changed.

Numbering. Propositions are numbered alike in both versions (1 transcript reduction, 2 fixed-size cohorts, 3 path
released, 4 pooled laws, 5 far tail, 6 certified map; 7 no-union only in the extended version). Theorems (since extended
v15): 1 final-model accountant, 2 count-averaged bound, 3 validity of the audit, S1 Gaussian fit, in both versions; in
extended v14 and earlier these were Theorems 1, 3, 4 and 2. File and column names keep the earlier labels (`thm3` =
count-averaged bound). Tables: see PROVENANCE.md.

What changed in v6 (fifth internal pre-submission review): `rev_round5_checks.py` and `rev_round5_upper_check.py`. The first holds the fifth-round review's two counterexamples:
an equal-dispersion count law with final-model eps 23.0 against 2.69, and random allocation against independent
participation at T = 2 in both hockey-stick directions. The second recomputes the per-device upper end independently,
by DFT of the characteristic function for the Poisson-binomial law, log-space h_eps and bisection;
its per-device maxima over the 24 windows agree with `rev_peruser_exact2.py` to 2.5e-7 on all 402 devices. No published number changed.

What changed in v5 (fourth internal pre-submission review): `rev_timer_structural.py` (exact search over all draws of the binding rounds;
per-device schema bound), the adaptive-selection example in `rev_cohort_example.py`, and CLAIMS.md (every headline
number of the abstract → script → output file → value).
