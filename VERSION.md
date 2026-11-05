# Version

Release **v17** (2026-11-05), matching the journal submission **TIFS v17** (main text and Supplementary Material) and the
**extended version v22**. Earlier releases: v16 (TIFS v16 / extended v21), v15 (TIFS v15 / extended v20), v14 (TIFS v14 / extended v19), v13 (TIFS v13 / extended v18), v12 (TIFS v12 / extended v17), v11 (TIFS v11 / extended v16), v10 (TIFS v10 / extended v15), v9 (TIFS v9 / extended v14), v8 (TIFS v8 / extended v13), v7 (TIFS v7 / extended v12), v6 (TIFS v6 / extended v11), v5 (TIFS v5 / v10), v4 (TIFS v4 / v9), v3 (TIFS v3 / v8),
v2 (TIFS v2 / v7).

What changed in v17 (seventeenth internal pre-submission review, an automated full-manuscript review of TIFS v16 that found
no mathematical or numerical error and verified all 58 references, and asked for self-contained definitions, the window rule
and the per-round reference on non-overlapping windows): one script, one result, seven checks. **The per-round reference on
non-overlapping windows** (`rev_round17_nonoverlap_ref.py`, seed 7106, 8M samples at a = 0.5): on the 1,144 windows
(starts 0 and 200 of the 3-day window, two per device; D = 21.83 as in `rev_robust_nonoverlap.csv`) the secret-path
reference is 14.154 [13.886, 14.437] at 1e-4 and 18.711 [18.171, 19.412] at 1e-5 (ratios 11.6 and 12.8 to the Poisson
values 1.219 and 1.462) against 10.69 and 14.97 on the 13,728 stride-10 windows (8.8 and 10.3), and the count-averaged
upper end (Theorem 2) is 21.209 and 25.898 (17.4 and 17.7 times) against 19.94 and 24.33: an empirical law that mixes
fewer distinct paths gives the secret-path adversary less hidden sampling, so the reference rises toward the upper end,
which depends on the count law only; the paper's factors (8.8 to 16.4) remain the conservative ones. Output
`results/rev_round17_nonoverlap_ref.csv`; `verify_claims.py` section 15 (7 checks; 509 in all). No other stored result
changed. The manuscript (TIFS v17 / extended v22) adds the stride rule (VI-A), the non-overlapping reference (VI-G), the
definition of the erasure ratio and the thresholds of the shape diagnostics (V-F), the centred estimator in closed form
(V-F), the orthonormality condition m <= d and the canary injection point (V-A, V-C), the cost of building the certified
table (V-D), the setting of the timer accuracy figures (VII), and caption and notation fixes (Fig. 1, Tables II, IV, V;
h^{-1}_delta); the byline drops the membership grade and the acknowledgment is the author's own wording (from v16's
final edits of 2026-11-05); the Supplementary Material is set at 10 pt and shortened to the six pages the SPS rule allows.

What changed in v16 (sixteenth internal pre-submission review, a readiness review of TIFS v15 that re-ran the checks on
commit 0f3902b): documentation only -- no script, stored result or check changed, `verify_claims.py` still 502 checks.
The manuscript (TIFS v16 / extended v21) names the estimator behind the 62 to 100 % detection range after shape screening
(eq. (11): 120 / 110 / 74 of 120 at r = 0.90) and adds the centred estimator's range, 56 to 100 % (120 / 107 / 67, both in
`rev_round15_screening_noise.csv`), in the abstract, Contribution 2, V-F and the conclusion; VII practice (4) says that an
audit of the sum release's own privacy can miss a violation confined to the per-round transcript while a sum estimator
given the law, the counts and the same table flags it with less power; V-C writes the quantity that eq. (11) bounds as
theta = mu / s (a in the model, a / r under a noise scale r times the declared). CLAIMS.md carries the centred range.

What changed in v15 (fifteenth internal pre-submission review, a referee report and re-analysis plan on TIFS v14): six
scripts added, no new experimental condition (the only new computations are four null calibration cells and twelve
400k-sample reference runs), no simulated or training number changed; two statements of the text corrected and the
noise-shortfall claim restated with the diagnostic rule applied. **Per-audit records** (`rev_round15_records.py`): the
2,880 noise-error audits (seeds 14500 + cell) and the 3,800 residual audits (seeds 14700 + cell) of v14 regenerated with
one record each (identifier, cell, seed, law, m, r, claim, the sha256 identity of the merged certified table,
full-precision a_lo and certificate for eq. (11) and for the centred estimator, the five diagnostics with the
definitions of `rev_round14_residual_sim.py`, and sufficient statistics that recompute a_lo, the centred a_lo, the
noise scale, the absent mean and the erasure exactly; KS and the per-canary lag-1 are reproduced from the seed), plus
four null calibration cells of 200 audits at the declared noise for (Markov, 1,000), (Markov, 200), (iid, 1,000),
(trace, 1,000) (seeds 15000-15003, generated after the test cells). **Screening** (`rev_round15_screening.py`):
recomputes every record from its sufficient statistics (largest discrepancy 3e-16), reproduces every per-cell aggregate
of `rev_round14_noise_sim.csv` and `rev_round14_residual_sim.csv`, and applies the rule of v14 as frozen (rule A; the
rounded rule file and the full-precision thresholds decide 1 of 6,680 audits differently), rule B (the noise scale
reported as a separate scale alarm, the shape diagnostics -- lag-1, the absent mean over the scale, the normalised KS,
the erasure -- deciding with thresholds frozen on the same null cell) and the two forms with thresholds frozen on the independent
null of each (law, m) (`rev_round15_rules.csv`). Four counts per cell (flag kept / flag withheld / no flag / withheld
without flag), reasons, Wilson 95 % intervals. Results (1,000 canaries): the scale statistic, which estimates r with sd
0.002, leaves its band 1 +- 0.0085 in 120 / 120 audits at every r != 1 (0.98 and 1.10 included) and 0 / 360 at r = 1;
rule A therefore keeps no flag at any r != 1 (the v14 flag counts were counts before screening) and withholds 0 / 0 / 7
of 120 reports at r = 1 (Markov / iid / trace; the 7 by the erasure band, sd 0.063 on the Markov null against 0.137
under the trace law); rule B keeps 120 / 110 / 74 of the 120 / 111 / 76 flags at r = 0.90, all 40 / 5 / 7 at 0.95,
120 / 120 / 118 at 0.80, and withholds 7 / 360 at r = 1 (2 / 360 with per-configuration nulls, which also keep 75
trace flags at 0.90); 200 canaries: the Markov-1,000 thresholds withhold 101 (rule A) and 98 (rule B) of 120 reports at
r = 1 (null KS maximum 0.0081 against 0.0038), the m = 200 rule withholds 1 and keeps all 47 flags at r = 0.90; the
centred estimator flags 120 / 108 / 69 at r = 0.90 before and 120 / 107 / 67 after rule B. Flags against the Markov
claim's upper limit 12.033 equal those against 11.869 in every cell. **Realised-loss intervals**
(`rev_round15_intervals.py`): independent 400k-sample two-sided 99 % intervals at every (law, r) (seeds 14900-14911;
iid exact): 1 of 2,880 certificates above the point reference (Markov r = 1.10, as in v14), 0 above an interval's upper
end, 50 inside an interval (undecided); every v14 point value lies inside its interval. **Training records**
(`rev_round15_train_records.py`, `rev_round15_cnn_records.py`): the six Fashion-MNIST noise configurations and the
CIFAR-10 runs regenerated with the absent mean, lag1_200, the centred estimator and the sufficient statistics recorded
(same seeds; the Fashion-MNIST rows reproduce `rev_round14_train_*.csv` on all 33 shared columns; the CNN rows reproduce
a_lo, s_hat, lag-1, the certificate and every flag exactly, while accuracy (within 0.003), erasure, KS, the uncertified
value and the correlation sample (a few units of the last printed digit) differ between PyTorch runs); screened as a transfer test:
scale alarm 10 / 10 at r < 1 and 0 / 10 at r = 1, rule A withholds every report at r < 1, rule B none, no abstention
at r = 1. **Device decomposition of the horizon curves** (`rev_round15_horizon_decomp.py`): Var(N_T) = B + W exactly
(equal device weights); the between-device term B / E[N_T] grows at 0.085 and 0.074 per round (Var(p)/E[p] = 0.083 and
0.072 from the realised rates), above the fitted slopes of D (0.064, 0.058); the within-device term is 7.05 and 14.72
at the shortest horizon and 0.19 and 0.81 at the longest (4 and 31 overlapping windows per device, none disjoint),
slope -0.021 and -0.017; the two slopes add to the slope of D; B / Var(N_T) = 0.375, 0.583, 0.829, 0.959, 0.994
(10-minute panel, T = 50 ... 400) and 0.331, 0.522, 0.756, 0.904, 0.960, 0.989 (5-minute, T = 100 ... 1,000); on disjoint
windows 0.384, 0.576, 0.777 and 0.334, 0.526, 0.765 where at least two exist. The v14 sentence "the realised rates
account for 77 % and 80 % of the slopes" was the ratio of the fitted slope to Var(p)/E[p] (0.768, 0.802) read as a
share; it is replaced by the decomposition. `verify_claims.py` gains a fifteenth-round section (72 checks; 502 in all,
all passing). **Text.** Abstract, Contribution 2, V-F and the conclusion: a 10 % shortfall trips the scale alarm in every
audit and is flagged by the certified report in 62 to 100 % of the audits by law after screening ("can detect"
replaced); V-F states the two outputs, rule A's abstention and rule B's counts; V-A: observations in declared-noise
units, so the noise scale reads r and the limit bounds a/r; Corollary 3 bounds the probability of a flag when the claim
is correct or conservative; VI-C and S-VI.B carry the decomposition; VII: for a stateful client with an exogenous path
the count-averaged bound survives (fully adaptive composition given the path), and what the logs give (rates: brackets;
count law: sum release and count-averaged bound; joint law: the exact per-round value); S-V.E carries the re-analysis;
S-V.F the AR(1) scoping, the scale alarm on the white cells and the centred estimator's certificate; S-V.D names the
maximum-over-rounds comparator as in the manner of Algorithm 3 of Andrew et al. with the fixed noise-unit normalisation
in place of their per-round cosine; S-V.C calls the CNN a small-sample second-architecture check and notes the clipping
calibration's own accounting; S-V.G notes the delta-method limit. To hold the budgets (main 13 pages, supplement 6):
Table S3 (per-device values at 1e-5 and 1e-8) and the two 500-client rows of Table S1 are prose in S-VI.D and S-V.C; the
main text drops VI-B's accuracy sentence (VII, S-VI keep it), VI-G's fixed-horizon comparison (extended only), V-A's
n_obs sentence, V-E's population-level sentence, V-F's softmax detail and VI-E's 3 % / 11 % (Table III). The extended
version adds a screening table and a decomposition table.

Values that differ between release v14 and release v15, with the reason:

| Quantity | v14 | v15 | Reason |
|---|---|---|---|
| Noise-shortfall detection (abstract, I, V-F, VII) | "can detect a 10 % shortfall" (flags 120 / 111 / 76 of 120 at r = 0.90, counts before the diagnostic rule) | scale alarm in 120 / 120 audits at every r != 1 and 0 / 360 at r = 1; rule A as frozen withholds every report at r != 1; with the scale as alarm and the shape diagnostics deciding, 120 / 110 / 74 flags remain, 7 / 360 reports withheld at r = 1 (2 with per-configuration nulls) | fifteenth-round re-analysis; the simulated numbers are unchanged |
| Within-device / between-device attribution (VI-C, S-VI.B) | "the realised rates account for 77 % and 80 % of the slopes" | the between-device term alone grows at 0.085 / 0.074 per round, above the fitted slopes; the within-device term shrinks with the window overlap; B / Var(N_T) = 0.38 ... 0.99 (as in v14) and 0.38, 0.58, 0.78 on disjoint windows | the v14 ratio was slope / coefficient, not a share |
| Coverage statement of the noise experiment | 1 of 2,880 above the realised loss (point value) | 1 above the point value, 0 above an independent 99 % interval, 50 inside | new intervals |
| Corollary 3 | "a flag is a false alarm with probability at most alpha" | "if the claim is correct or conservative, the probability of a flag is at most alpha" | wording |
| Stateful client (VII) | "only the schema bound survives" | the count-averaged bound of Theorem 2 survives with an exogenous path; the schema bound alone without exogeneity | wording |
| `verify_claims.py` checks | 430 | 502 | fifteenth-round section |

What changed in v14 (fourteenth internal pre-submission review, two reports on TIFS v13): eight source files added (seven scripts and the CIFAR-10 conversion), six
computations added, no published value changed. The noise, residual and comparator simulations use the merged certified
tables of Table III (`rev_round14_common.py`: the step-14 and step-16 tables merged as `step19_rerun_audits.py` merges
them); the learned-law study builds its own 1M-sample tables; the training and CNN scripts use the step-14 tables as
`rev_train.py` does; m = 1,000, T = 200, a = 0.5, delta = 1e-4 throughout. **Implementation noise error** (`rev_round14_noise_sim.py`, `rev_round14_train.py`
noise_*): the server adds noise of scale r x the declared sigma = 2 while accountant and auditor are told sigma; the
claim is the correct accountant's value for the canaries' own law at the declared noise (11.869 Markov, 2.866 iid,
10.69 trace; the training scripts take the iid claim from `rev_train.REF_IID` = 2.869, the Monte-Carlo reference, and no
flag count differs between 2.866 and 2.869), so it is true at r = 1 and short of the realised loss eps(a/r) at r < 1. Flags at r = 1.10 / 1.00 /
0.98 / 0.95 / 0.90 / 0.80: Markov 0 / 0 / 1 / 40 / 120 / 120, iid 0 / 0 / 0 / 5 / 111 / 120, trace 0 / 0 / 0 / 7 / 76 /
120 (120 audits per cell; Markov with 200 canaries 0 / 0 / 1 / 9 / 47 / 120); one estimate of 2,880 above the realised loss (Markov r = 1.10, a_lo above a/r);
training (Fashion-MNIST, 784-64-10, ten runs per cell) 0 / 2 / 10 / 10 flags at r = 1, 0.95, 0.90, 0.80 (Markov) and
0 / 8 at r = 1, 0.90 (iid). **Operating region** (`rev_round14_residual_sim.py`): residuals added to every canary
observation (white, constant offset, AR(1) with phi = 0.5 and 0.9, reinforcing and pull-back on participating rounds) at
rho = 0.05, 0.10, 0.25 of the noise scale, 200 audits per cell, a diagnostic rule frozen on the null cell (mean +- 4 sd
of the absent canaries' scale, lag-1, mean and the erasure ratio; KS at most the null maximum), and a centred
estimator (`snr_lower_bound_centred`: absent mean subtracted, variance term 1/n_0). White residuals never exceed the modified
release's loss and AR(1) residuals never exceed the nominal 11.869 (no scalar reference describes that release; at
most 7 points of recovery lost; both rejected by the rule at 0.25); reinforcement
never exceeds eps(a + rho); the pull-back costs power only; the constant offset of 0.05 puts 189 of 200 estimates above
the reference and is rejected 200 / 200 by the rule; the centred estimator recovers 86-87 % with none above in every
offset cell and agrees with eq. (11) of the journal version within one point elsewhere (2 of 200 above the
pull-back reference at rho = 0.10, within the allowance). **Comparators on the same information**
(`rev_round14_comparators.py`): on the same 120 data sets per law, the path-aware audit, a sum audit with the same law
and the same table (regression of the sum on the known counts; information ratio (sum N_i^2 / T) / sum N_i), the
count-conditioned audit and the maximum-over-rounds score with the Gaussian fit: Markov 87 / 75 / 35 / 1 %, iid
89 / 70 / 63 / 4 %, trace 72 / 48 / 11 / 1 %, cyclic 90 / 70 / 22 / 2 % (information ratios 0.18 / 0.10 / 0.15 /
0.10); one of 480 path audits and one of 480 sum audits above a reference. **A law learned from logs**
(`rev_round14_learned_law.py`): canaries at lambda = 0.9 with q known; lambda_hat = p11 - p01 by maximum likelihood from the logs of N = 100 / 500 / 2,000 devices (sd 0.0064 / 0.0029 / 0.0015); thirteen tables at lambda' = 0.86 ... 0.93 (1M samples per point, seed 14100 + 1000 lambda'); 1,000 replicates per N (fresh logs, seed 14200; fresh audits, seed 14300 + r + 10 N). Oracle table: recovery 86, 86, 86 %, above 11.869: 0, 0, 0; plug-in table (nearest lambda'): recovery 85, 85, 85 %, above 11.869: 19, 0, 0, above 12.033: 2, 0, 0, flags 1000, 1000, 1000; conservative table (largest lambda' <= lambda_hat - z_0.99 se, level 0.01 + 0.035 + 0.005): recovery 73, 78, 81 %, above 11.869: 0, 0, 0, above 12.033: 0, 0, 0, flags 1000, 1000, 1000, abstentions 0, 0, 0. Out of family (canaries on the FLASH trace law, table for a chain fitted to the trace logs, q = 0.0471, lambda = 0.528): certified 1.33 +- 0.08 (12 % of 10.69), 0 / 120 above, 105 flags. A sensitivity study resting on the observed monotonicity in lambda, not a guarantee. The pooled chain cannot carry the trace law's between-device dispersion (eightfold under-statement, safe here but for no general reason): the table must be built from the logged paths, not from a family fitted to them. **Heterogeneous data** (`rev_round14_train.py` dir01_*, iid_*):
784-256-128-10, 500 clients, Dirichlet(0.1) label-skewed shards (empty to 1,183 examples; 85 % of updates clipped) or
IID shards, 100 / 500 canaries, ten runs each: certified recovery 64 / 82 % (Dirichlet) and 52 / 82 % (IID), 9, 10, 8
and 10 flags of the Poisson claim, none above the reference, diagnostics unchanged. **A convolutional network on
CIFAR-10** (`rev_round14_cnn.py`, PyTorch, CPU): conv 32, conv 64, fc 128, 10 (545098 parameters), 200 clients with IID shards of 250 images, 1,000 orthonormal canaries, sigma = 2, T = 200 (JPEG re-encoding of CIFAR-10 from github.com/YoongiKim/CIFAR-10-images, converted by `make_cifar_npz.py`; the official archive was unreachable from the build machine): cnn_markov (5 runs, seed 1431) certified 10.14 +- 0.36 (85 % of 11.869), 5 flags, none above the reference, accuracy 0.373 to 0.403, clip fraction 1.00, erasure 0.94 to 1.02; cnn_iid (3 runs, seed 1432) 2.55 +- 0.00 (89 % of 2.866), none above; cnn_markov_r0.90 (2 runs, seed 1433) 13.11 +- 0.00, 2 / 2 flags of the claim 11.869, the table's lower estimate of the realised loss 14.63 (the 400k-sample estimate is 15.02); every update clipped (C calibrated on the initial model). Outputs `rev_round14_cnn_cnn_<config>.csv`. Accuracy is low because at 20 clients per round the noise norm (sigma C sqrt(d)) is 74 times the bound 20 C on the clipped sum, against 22 for the 50,890-parameter network. `verify_claims.py` gains a fourteenth-round section
(150 checks; 430 in all, all passing). **Text.** VI-F states that the schema bound's maximum under the 20-round timer is the timer's own
cap (10 participations), so it holds for every device and population, and III-C says the schema bound holds for every
population once its cap is set by policy; Contribution 2 gives the Gaussian fit's range as 3 to 22 %; the abstract says
"mutually independent paths" and carries the noise-shortfall sentence; V-A states the measured cost of the sum
statistic given the law and the counts; Table III's caption defines the count-conditioned audit; V-F gains the
paragraphs "What the audit adds to the logs" and "Operating region", the learned-law sentence and the heterogeneous-data
and CNN sentence; VII and the conclusion carry the noise-shortfall clause. The supplement (6 pages, set at 9 points) gains Secs. S-V.D to
S-V.G, Table S2 (residuals) and the new rows of Table S1; its former protocol, 1e-5-audit and guess-counting tables are prose. The extended version gains a
novelty table (Section II) and the tables of every new experiment. TIFS v14 main text 13 pages, supplement 6, abstract
250 words.

Values that differ between release v13 and release v14, with the reason:

| Quantity | v13 | v14 | Reason |
|---|---|---|---|
| Cost of the sum statistic given the law | "a tenth of the path's information" (N^2/T = 2 against N = 20, an estimate) | measured: 11 to 24 recovery points less than the path, 2.6 to 10 times the dispersion; information ratio 0.10 to 0.18 | fourteenth-round request; new script |
| Sensitivity to the table's law | tables at lambda' = 0.85 ... 0.95: 0, 0, 1, 16, 120 of 120 above (kept) | plus a law learned from logs of 100 / 500 / 2,000 devices with plug-in and conservative rules (plug-in above 11.869: 19, 0, 0 of 1,000; conservative 0, 0, 0, recovery 73, 78, 81 %, abstentions 0, 0, 0; out of family (chain fitted to the trace logs) 12 %, 0/120 above) | fourteenth-round request; new script |
| Training settings | MLPs on MNIST / Fashion-MNIST, IID or 2-class shards | plus Dirichlet(0.1) shards (64 / 82 %) and a 545k-parameter CNN on CIFAR-10 (85 % Markov, 89 % iid, accuracy 0.37-0.40) | fourteenth-round request; new scripts |
| `verify_claims.py` checks | 280 | 430 | fourteenth-round section |

What changed in v13 (thirteenth internal pre-submission review, two independent reports on TIFS v12): two scripts added,
two computations added, no published value changed. **Bootstrap over devices** (`rev_round13_bootstrap.py`). The 572
devices of the analysis window are resampled with replacement 1,000 times (replicate r with seed 13001 + r); the policy is
re-applied to each resampled population with its own cohort size (fixed cohort: K = round(0.6 x the resampled mean pool),
policy seed 0), the count law is taken over the 24 windows, and Theorem 1 (exact final-model eps), Theorem 2 (count-averaged
bound) and the Poisson figure at the replicate's own rate are evaluated at 1e-4 and 1e-5. The analysis population
reproduces the paper (5.16, 6.25, 19.94, 16.38 for the running example; 9.44, 13.67 for take-all). 95 % percentile
intervals: running example, final-model ratio 3.98 to 7.34 (1e-4) and 4.60 to 8.76 (1e-5), Theorem 2 over Poisson 14.73 to
19.21 and 14.78 to 19.70, D 19.3 to 24.7; take-all, final-model ratio 6.58 to 11.75 and 7.03 to 12.98, Theorem 2 over
Poisson 11.93 to 15.52 and 11.87 to 15.81, D 33.9 to 42.2. Outputs `rev_round13_bootstrap.csv` (one row per replicate and
policy; replicate -1 is the analysis population) and `rev_round13_bootstrap_summary.csv`. It replaces the "20 random halves"
summary of the text (TIFS VI-G and Sec. S-VI; the halves and four-fifths subsets stay in the extended version). **The
audit pointed at the worst device's law** (`rev_round13_worst_device_audit.py`). Canaries are given device 335's law under
position (b) over its worst window (window 21, rounds 210 to 409; K = 27; 115 expected participations). Given the rates the
rounds are independent, so eps(delta; a) comes from the PLD accountant on the 0.01 grid of a at discretisation
1e-3 (the table is the optimistic estimate, a lower estimate; the pessimistic estimate exceeds it by at most 0.086) and no Monte-Carlo
certificate is needed; the reference is the optimistic estimate at a = 0.5 and discretisation 1e-4, 27.0672 at 1e-4
(bracketed by the pessimistic 27.0758) and 29.7062 at 1e-5. 120 audits of 1,000 canaries (seed 13002, level 0.05, the whole
level on the estimate): certified 25.77 +- 0.44 (95.2 % of the reference), 120 flags of the Poisson claim 1.219, none above
the reference; at 1e-5, 28.32 +- 0.47 (95.3 % of 29.71). Outputs `rev_round13_worst_table.csv` and
`rev_round13_worst_audits.csv`. `verify_claims.py` gains a thirteenth-round section (54 checks; 280 in all, all passing).
**Text.** The reported per-device figure is now called "the bound to report from the logs" (conditional on the logged
availability) rather than "the guarantee to report", and Section III-C states the three kinds of number (a measurement
for a given law, a bound for the run from the logs, the schema bound for every path); the stateless-client scope and the
audit's independence from it are stated in Sections V-E and VII; the training scope ("small vision models, so the
mechanism is architecture-independent and the numbers are not"), the timer's accuracy cost in the conclusion, and the
step-function certificate as "a Clopper-Pearson limit on one tail probability" are stated; references with more than six
authors use "et al." (IEEE Reference Guide). TIFS v13 main text 13 pages, supplement 6, abstract 250 words.

Values that differ between release v12 and release v13, with the reason:

| Quantity | v12 | v13 | Reason |
|---|---|---|---|
| Sampling uncertainty of the trace numbers | 20 random halves: final-model ratio 3.7 to 7.3 (median 5.4) | bootstrap over devices: 95 % interval 3.98 to 7.34 about 5.16; Theorem 2 over Poisson 14.73 to 19.21 about 16.38 | thirteenth-round request; the halves summary stays in the extended version |
| Audit of one device's law | not computed | device 335 under (b): certified 25.77 +- 0.44 of 27.07 (95 %), 120/120 flags, 0 above | thirteenth-round request; new script |
| `verify_claims.py` checks | 226 | 280 | thirteenth-round section |

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

Computed with NumPy 2.5.3, SciPy 1.18.1, dp-accounting 0.6.0, on one or two CPU cores; no GPU (the CNN runs of the
fourteenth and fifteenth rounds with PyTorch 2.15, CPU, and Pillow for the CIFAR-10 conversion). `src/verify_claims.py` was last run
with Python 3.13.16 and these library versions.

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
