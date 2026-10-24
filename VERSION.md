# Version

Release **v10** (2026-10-24), matching the journal submission **TIFS v10** (main text and Supplementary Material) and the
**extended version v15**. Earlier releases: v9 (TIFS v9 / extended v14), v8 (TIFS v8 / extended v13), v7 (TIFS v7 / extended v12), v6 (TIFS v6 / extended v11), v5 (TIFS v5 / v10), v4 (TIFS v4 / v9), v3 (TIFS v3 / v8),
v2 (TIFS v2 / v7).

What changed in v10 (tenth internal pre-submission review): six scripts, all additions; no earlier output changed except
that the cyclic law's per-round references were recomputed from 20M samples (7.94 [7.89, 7.98] at T = 200 and 28.04 at
T = 1,000, against 8.02 and 27.79 from the 300k-sample pilots; the derived ratios become 2.8x and 3.9x, and the Theorem 2
looseness 1.29 and 1.08). `rev_round10_small_delta.py`: the pooled ratios at delta down to 1e-10 (exact or closed form);
`rev_round10_peruser_delta.py`: the per-device bracket at 1e-6 and 1e-8 (the losses grow, the claim grows faster: the
largest lower and upper ends over devices, devices 335 and 176, are 17.5x and 22.0x the claim at 1e-8); `rev_round10_cyclic.py` and `rev_round10_cyclic_T1000.py`:
the certified audit on the cyclic law (previously not run), which enters the main evaluation (now 994 independent data
sets, 1,594 certifications, 816 of 817 violated claims flagged); `rev_round10_canaries.py`: audit power against the number
of canaries; `rev_round10_erasure.py`: the configuration in which the observation model fails, re-run and logged.
`verify_claims.py` gains a tenth-round section and counts the cyclic audits; `derived_numbers.py` counts them too. The
extended version now uses the journal numbering of the theorems (see Numbering below).

What changed in v9 (ninth internal pre-submission review, a code inspection): `rev_peruser_exact2.py` groups the per-round
inclusion rates exactly (v8 rounded them to 9 decimals first) and composes the optimistic PLD, within a rate and across rates, with
`tail_mass_truncation=0` (dp-accounting 0.6.0 applies a default truncation of 1e-15 in both compositions and adds it to
the infinity mass in `self_compose` even in the optimistic mode), so the lower end of the per-device bracket is a lower estimate of the mechanism with the rates as
computed; all per-device outputs were regenerated (`rev_peruser_exact2_*.csv`, log). The largest change in any per-device value is 1.1e-8 (none at the printed precision; worst windows, refinement counts and certificates unchanged; rev_round9_compare.py). The guarantee to report
is now quoted as 28.3× (upper end 34.53) instead of the rounded-down 28×. No other computation changed.

Post-release documentation fixes on `main` (2026-10-22; no code path, result or number changed, tag v9 unchanged): `PROVENANCE.md` no longer names an internal runner script that is not in the release (`rev_run_train.sh` runs both training batches); `README.md` and `rev_round9_compare.py` state that the comparison with v8 needs tag v8 (a full clone) and stop with that message instead of a traceback. 2026-10-23: `paper/make_figures.py` reads the stored outputs from `results/` (override with `P3_RESULTS`) and writes the figures next to itself instead of using absolute paths of the author's machine, and embeds TrueType instead of Type 3 fonts (IEEE PDF requirements); the three figures were regenerated from the same data (pixel comparison: only glyph rasterisation differs). `src/rev_peruser_rows.py --patch` takes the manuscript files to patch as arguments.

Review history. Every "review round" or "revision round" named in this package (file names `rev_*`, script docstrings,
this file, README.md, PROVENANCE.md, REVIEWER_CHECKS.md) refers to internal pre-submission reviews of the manuscript;
the manuscript has not been submitted to a journal before. "TIFS vN" and "extended vN" are internal draft labels.

Computed with NumPy 2.5.3, SciPy 1.18.1, dp-accounting 0.6.0, on one or two CPU cores; no GPU. `src/verify_claims.py`
was last run with Python 3.13.16 and these library versions.

What changed after the eighth internal pre-submission review (2026-10-20, same release v8: the manuscript had not been
submitted): `verify_claims.py` and `derived_numbers.py` also count independent audit data sets (874; violated claims
flagged on 696 of 697), since each simulated data set on a correlated law is certified at both δ (1,354 certifications),
and check the training recovery range 73–93 %. No computed number changed.

What changed in v8 (seventh internal pre-submission review): `rev_round7_audit_sens.py` (power of the
certified audit at small violations, λ = 0.3, 0.6, 0.8; audits of λ = 0.9 canaries certified with tables built for
λ' = 0.85, 0.89, 0.9, 0.91, 0.95; sampling error of the maximum-likelihood λ from the logs of 1,000 devices over 200
rounds; outputs `rev_round7_audit_sens.csv`, `rev_round7_lamse.csv`, logs in `results/logs/`) and
`rev_round7_trace_sens.py` (round length at a fixed 33-hour horizon: 5-, 10- and 30-minute rounds; 20 random halves
and four-fifths of the devices; output `rev_round7_trace_sens.csv`). `verify_claims.py` gains a section of
seventh-round checks, including the disjointness of their table and audit seeds. No published number changed; the
new numbers are in TIFS Sections V-F and VI-G and extended Sections V-F and VI-G.

What changed in v7 (sixth internal pre-submission review): `src/verify_claims.py` (re-derives every headline number from the stored outputs
and checks that certificate and audit seeds are disjoint; output `results/logs/verify_claims.log`, all checks pass) and
`REVIEWER_CHECKS.md` (the six artifact checks of the review mapped to files, columns and commands, with a table of random
seeds). No published number changed.

Numbering. Propositions are numbered alike in both versions (1 transcript reduction, 2 fixed-size cohorts, 3 path
released, 4 pooled laws, 5 far tail, 6 certified map; 7 no-union only in the extended version). Theorems (since extended
v15): 1 final-model accountant, 2 count-averaged bound, 3 validity of the audit, S1 Gaussian fit, in both versions; in
extended v14 and earlier these were Theorems 1, 3, 4 and 2. File and column names keep the earlier labels (`thm3` =
count-averaged bound). Tables: see PROVENANCE.md.

What changed in v6 (fifth internal pre-submission review): `rev_round5_checks.py` (the two counterexamples of the fifth-round review:
an equal-dispersion count law with final-model eps 23.0 against 2.69, and random allocation against independent
participation at T = 2 in both hockey-stick directions) and `rev_round5_upper_check.py` (an independent recomputation of
the per-device upper end: DFT of the characteristic function for the Poisson-binomial law, log-space h_eps, bisection;
the per-device maxima over the 24 windows agree with `rev_peruser_exact2.py` to 2.5e-7 on all 402 devices). No published number changed.

What changed in v5 (fourth internal pre-submission review): `rev_timer_structural.py` (exact search over all draws of the binding rounds;
per-device schema bound), the adaptive-selection example in `rev_cohort_example.py`, and CLAIMS.md (every headline
number of the abstract → script → output file → value).
