# When Clients Come Back: Privacy Accounting and Auditing of Federated Learning under Temporally Correlated Participation

Code, certified tables and trace pipeline for the paper

> A. S. Alzahrani, "When Clients Come Back: Privacy Accounting and Auditing of Federated Learning under Temporally Correlated Participation," 2026. arXiv: *(to be inserted)*.

Every number in the paper is traced to a script and an output file in [`PROVENANCE.md`](PROVENANCE.md); the headline numbers of the abstract are mapped one by one in [`CLAIMS.md`](CLAIMS.md), and [`VERSION.md`](VERSION.md) states which manuscript version this release matches. Reviewers: [`REVIEWER_CHECKS.md`](REVIEWER_CHECKS.md) maps each artifact check to files and commands, and `cd src && python verify_claims.py` re-derives every headline number from the stored outputs in a few seconds.

## What is here

| Directory | Contents |
|---|---|
| `src/` | All analysis code (pure NumPy/SciPy; `dp_accounting` for the Poisson values). Library modules: `corraudit.py` (participation laws, exact count laws, HMM likelihood ratios, Theorem 1 final-model accountant, Poisson PLD accountant), `theory.py` (dispersion law, Theorem 3 count-averaged bound, Gaussian-fit crossover), `certified.py` / `certified2.py` (certified Monte-Carlo tables; default certificate = Chernoff–Hoeffding bound in KL form), `auditors2.py` (Gaussian-fit, count-conditioned and path-aware auditors), `vecgauss.py`. Experiment scripts `step*.py`, `pilot_*.py`, `regime_map.py`, `fl_*.py` (real DP-FedAvg training with canary clients). |
| `trace/` | `trace_analysis.py`: eligibility mapping of the FLASH device trace (locked ∧ Wi-Fi ∧ charging), availability processes, selection policies and count laws. `pn_*.npy`: the ten count laws of Table VIII (full version). The trace itself is **not** redistributed here; see `trace/README.md`. |
| `results/` | Every CSV the paper's tables and figures are built from, the run logs (`results/logs/`), `step19_audits.npz` (per-audit values of the consolidated audit tables), `step18_pn_*.npy` (count laws of the six availability models) and `theory_notes.md` (working notes with the proofs as first written). |
| `paper/` | `make_figures.py` and the three figures. |
| `data/` | Empty; MNIST / Fashion-MNIST in IDX format go here for the training runs (`data/README.md`). |

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bash trace/fetch_trace.sh        # clones FLASH (BSD-2-Clause) and copies data/state_traces.json into trace/
```
Python ≥ 3.10. The code was run with numpy 2.5, scipy 1.18, dp-accounting 0.6.0 on one CPU core; nothing uses a GPU.

Paths: scripts look for the trace in `trace/` (override with `P3_TRACE=/path`) and for the image data in `data/fmnist`, `data/mnist` (override with `P3_DATA`, `P3_DATA_MNIST`). Run every script from inside `src/`.

## Reproducing the paper

Times are for one CPU core. Outputs land in `src/` with the same names as the copies kept in `results/`.

```bash
cd src
# Section IV: exact accountant, dispersion law, Theorem 3 looseness, Gaussian-fit crossover
python3 theory.py                               # theory_dispersion.csv, theory_perround_bound.csv, theory_crossover.csv (minutes)
python3 regime_map.py                           # regime_map.csv: Fig. 1 (hours; 300k samples per point)
python3 step15_convex_order.py                  # step15_*.csv: 900 random spreads, 12 standard laws, designed pairs (hours)
python3 step15b_replay.py && python3 step15c_lrcheck.py   # Appendix B / Sec. S-II certification counts

# Section V: certified tables, simulated audits
python3 step14_merged.py markov 2000000 120     # 2M-sample tables at 1e-4 (32 min + audits); also iid, trace
python3 step16_delta5.py markov 10000000 120    # 10M-sample Markov tables at 1e-4/1e-5 (~2 h); trace: 3M
python3 step19_rerun_audits.py                  # consolidated audit tables: step19_results.csv (Tables II, III, V; Fig. 2)
# restart-safe T=1000 and fine-grid runs (parts, then combine):
bash run_parts.sh markov 12 250000 1000         # Markov T=1000, 3M samples
bash run_parts.sh trace 20 100000 1000 5 4 5    # trace T=1000, 5-min rounds, 4-day window, 2M samples
bash run_parts.sh trace 10 100000 fine          # trace T=200, 0.01 grid, 1M samples

# Real training (needs data/): 68 runs with 1,000 canaries + 6 with 5,000
python3 fl_mlp2.py 10      # Fashion-MNIST IID, 10 runs per law        -> fl_mlp2_results.csv
python3 fl_mlp3.py 8       # MNIST IID / 2-class, F-MNIST 2-class      -> fl_mlp3_results.csv
python3 fl_mlp_5k.py 3     # 5,000 canaries                            -> fl_mlp_5k_results.csv

# Section VI: trace
python3 step22_window_stats.py                  # descriptive statistics of the trace and the 3-day window
python3 step21_trace_unified.py                 # policies + horizon curves (step21_policies.csv, step21_horizon.csv)
python3 step21b_consistent.py                   # seed-0 policy table with Thm-3 bound, tail support, 5-min window (Table VIII)
python3 step18_availability_models.py           # six availability models (Table X)

# Revision runs (after internal pre-submission reviews; the manuscript has not been submitted before); see PROVENANCE.md for what each feeds
python3 rev_threshold.py                        # threshold criterion on the 900 spreads (Proposition 4, Appendix B)
python3 rev_dongganesh.py                       # cross-check with the Dong-Ganesh recursion (b = 2, 5, 9)
python3 rev_renewal_ci.py                       # renewal laws b = 2, 5, 9 with 99% intervals
python3 rev_guess.py                            # guess-counting audit with the sum and the path-aware scores (superseded by rev_guess2.py)
bash rev_run_refs.sh                            # reference re-runs with 99% intervals: Markov 10M, timers, leave-out, trace 8M (hours)
python3 rev_refs.py stored                      # intervals on the stored T = 1,000 reference samples
python3 rev_peruser.py && python3 rev_peruser_notes.py && python3 rev_peruser_table.py   # per-device privacy, positions (a), (a') (Section VI-E)
# Second revision round
python3 rev_cohort_example.py                   # fixed-cohort two-client example and the worst-window example (Sections III-C, VI-E; seconds)
python3 rev_peruser_exact.py                    # position (b) with exact inclusion rates, maximum over 24 windows, PLD lower end and Theorem-3 upper end (Tables XI-XII; ~45 min)
python3 rev_guess2.py                           # guess-counting with three scores against matched targets (Table IV)
# Third revision round
python3 rev_peruser_exact2.py                   # certified per-device bracket: optimistic/pessimistic PLD, certified window search, top-20 full check (Tables XI-XII; ~1.5 h on two cores)
python3 rev_timer_seeds.py                      # timer policies under 100 seeds: path invariance given availability (Section VI-E)
python3 rev_t1000_bracket.py                    # Theorem-3 upper end at T = 1,000 for the running example (Section VI-F)
# Fourth revision round
python3 rev_timer_structural.py                 # exact search over all draws of the binding rounds + per-device schema bound (Section VI-E; minutes)
python3 rev_cohort_example.py                   # now also example (4): why participation must be exogenous to the noise
python3 rev_trace_robust.py && python3 rev_trace_window.py   # 20 draws, leave-out, non-overlapping windows, later window
python3 rev_tail5.py                            # the tail devices at T = 1,000
bash rev_run_train.sh                           # 200 training runs with orthogonal canaries (needs data/; about 2 h)
python3 rev_train_summary.py && python3 rev_train_table.py
python3 rev_timer_utility.py 5                  # utility under the timers (5 seeds)
python3 rev_timer_peruser.py                    # per-device loss under the timers with availability known
python3 rev_validity_lower.py                   # audits against the lower 99% limits of the references
python3 derived_numbers.py                      # aggregate counts: 874 independent audit data sets (1,354 certifications), 696 of 697 / 0 of 177

# Fifth to seventh revision rounds
python3 rev_round5_checks.py && python3 rev_round5_upper_check.py   # fifth round: counterexamples; independent upper-end check
python3 verify_claims.py                        # sixth round: re-derive every headline number from results/ (seconds)
for l in 0.3 0.6 0.8; do python3 rev_round7_audit_sens.py power $l; done                 # seventh round: audit power (~7 min each)
for l in 0.85 0.89 0.9 0.91 0.95; do python3 rev_round7_audit_sens.py misspec $l; done   # seventh round: misspecified law
python3 rev_round7_audit_sens.py lamse && python3 rev_round7_audit_sens.py combine          # log-estimated law; collect
python3 rev_round7_trace_sens.py                # seventh round: round length at a fixed horizon, device subsampling

# Figures
cd ../paper && python3 make_figures.py
```

`results/theory_notes.md` records the derivations and the review corrections in the order they were made.

`src/derived_numbers.py` recomputes from the result files the numbers that are derived in the text rather than read from a table (the no-subsampling ε of 50.5, the thinning identity, the OLS slopes of D on T, the far-tail thresholds of the threshold-criterion proposition, the aggregate audit counts (874 independent data sets, 1,354 certifications; 696 of 697 and 0 of 177), and the i.i.d. Monte-Carlo checks).

## Naming notes
* `eps_cert_bern*` columns and `merged_bern_*` estimator names are historical: since step 14 the certificate is the Chernoff–Hoeffding (KL-form) lower confidence bound (`certified2.kl_lo`, the default of `eps_cert_bern`); the empirical-Bernstein variant is kept as `bernstein_lo` but was not used for any reported number.
* `path_aware_v2` = the uncertified path-aware estimator (plain Monte-Carlo curve, linear interpolation); `merged_certified` / `cert_*` = the certified path-aware auditor; `gauss_sum` = Gaussian-fit audit of the final-model scores; `count_cond*` = count-conditioned audit; `old_certified` / `step_*` = the step-function certification of step 12.
* "FM" = final model, "PR" = per-round in table headers.
* `vecgauss.py` in this release evaluates Gaussian upper tails as `ndtr(-z)` rather than `1 - ndtr(z)`; the change affects only ε above about 12 (no reported Gaussian-fit or count-conditioned value exceeds 5) and is documented in the file.

## License
MIT (see `LICENSE`). The FLASH trace and simulator are BSD-2-Clause, © their authors (Yang et al., WWW 2021).
