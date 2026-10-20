# Theory notes for P3 (working, 2026-10-14)

Setting (Model 1, scalar projection). Rounds t = 1..T. Canary participation indicators B = (B_1..B_T) follow a
finite-state process with known law; absent world B = 0. Per-round observation X_t = a B_t + Z_t, Z_t i.i.d. N(0,1),
with a = C/σ the per-participation signal-to-noise ratio (C = clip norm = canary update norm, σ = noise multiplier).
Count N_T = Σ_t B_t with law p_n (computable exactly in O(T² |S|) for a finite-state chain by the forward recursion).

## Theorem 1 (exact privacy of the final-model release under any participation process)
Let S = T^{-1/2} Σ_t X_t. Then Law(S | absent) = N(0,1) and Law(S | present) = Σ_n p_n N(a n /√T, 1).
Write μ_n = a n/√T, G_0 = N(0,1), M = Σ_n p_n N(μ_n,1). The likelihood ratio dM/dG_0(s) = Σ_n p_n exp(μ_n s − μ_n²/2)
is nondecreasing in s (all μ_n ≥ 0), hence for every ε the optimal rejection regions are half-lines and
  δ_1(ε) = Σ_n p_n Φ̄(s*(ε) − μ_n) − e^ε Φ̄(s*(ε)),   s*(ε) the unique root of log(dM/dG_0)(s) = ε,
  δ_2(ε) = Φ(s**(ε)) − e^ε Σ_n p_n Φ(s**(ε) − μ_n),   s**(ε) the root of log(dM/dG_0)(s) = −ε,
and ε(δ) = inf{ε ≥ 0 : max(δ_1(ε), δ_2(ε)) ≤ δ}. The privacy of the final-model release depends on the participation
process only through the count law p_n. [Proof: monotone likelihood ratio of a mixture of shifted Gaussians with
nonnegative shifts; Neyman–Pearson; hockey-stick as the optimal-test gap.]

Corollary 1.1 (count-averaged upper bound). By convexity of P ↦ H_ε(P‖Q) and Jensen,
  H_ε(M‖G_0) ≤ Σ_n p_n h_ε(μ_n),  h_ε(μ) = Φ(μ/2 − ε/μ) − e^ε Φ(−μ/2 − ε/μ),
and the same argument for the per-round release gives H_ε(P_full‖Q_full) ≤ Σ_n p_n h_ε(a√n)   (Theorem 3 below).

Numerical law (dispersion). At fixed (T, q, a, δ) across the two-state Markov family (λ ∈ [−0.1, 0.97]) and the
b-min-separation renewal family (b ∈ [1, 9]), ε(δ) is monotone in Var(N_T) (theory_dispersion.csv):
Var 0 → 2.53, 4.1 → 2.57, 18 (i.i.d.) → 2.69, 71 → 3.25, 326 → 8.30, 988 → 27.1 (T = 200, q = 0.1, a = 0.5, δ = 1e-4).
Conjecture 1.2: if N' ≥_cx N (convex order, equal means) then ε_{N'}(δ) ≥ ε_N(δ). Supported numerically; a proof would
need H_ε(Σ p_n G_n ‖ G_0) to be Schur-convex in p along the shift direction — open.

## Theorem 2 (sign of the Gaussian-fit bias)
Let M̃ = N(μ̄, 1 + v) with μ̄ = Σ p_n μ_n, v = Var(μ_N) (moment-matched Gaussian, the population limit of the
Agrawal et al. fit). As s → ∞, log dM̃/dG_0(s) ~ (v/(2(1+v))) s² while log dM/dG_0(s) ~ μ_max s − μ_max²/2 with
μ_max = a·n_max/√T; hence for every process with v > 0 there is δ*(p) > 0 such that ε_fit(δ) > ε_exact(δ) for δ < δ*.
Numerically δ* is below 1e-13 for every process tested with Var(N_T) ≤ 33 (min-sep, i.i.d., λ ≤ 0.3), and the fit is
conservative by a factor 0.65 (λ = 0.9) to 0.43 (λ = 0.97) at δ = 1e-4, 0.50 / 0.54 at 1e-8, 0.47 / 0.65 at 1e-12
(theory_crossover.csv). Statement for the paper: under positive autocorrelation the Gaussian-score audit is
conservative at every practical δ and loses power as (1+λ)/(1−λ) grows; its failure mode is power, not validity.
[Proof of the asymptotic part: tail comparison of a Gaussian with variance > 1 against a finite mixture of unit-variance
Gaussians with bounded shifts.]

## Theorem 3 (count-averaged accountant for the per-round release)
For the per-round release (X_1..X_T), for any participation process with count law p_n,
  δ(ε) ≤ Σ_n p_n h_ε(a√n).
Proof: condition on the path; given the path with n participations the mechanism is the adaptive composition of n
Gaussian mechanisms with sensitivity a (in noise units), i.e. a√n-GDP, whose hockey-stick is h_ε(a√n); the present
law is the p-mixture over paths; convexity of the hockey-stick in its first argument and Jensen give the bound.
Numerics (T = 200, a = 0.5, δ = 1e-4; theory_perround_bound.csv): bound/true = 3.97 (i.i.d.), 1.83 (λ = 0.9),
1.24 (λ = 0.97), 1.28 (cyclic b = 9); at T = 1000: 4.4 (i.i.d.), 2.26 (λ = 0.9), 1.09 (cyclic). Always below the
schema worst case (max participations), and tight where correlation is strong — exactly where Poisson accounting
is wrong by 3–10×.

## Proposition 4 (validity of the count-conditioned audit)
Auditor draws m canaries, each present with probability 1/2 independently, each with its own participation path from
the known process; observes X^{(i)}; computes S_i. Under Model 1 the true (μ_0, s_0, β) = (0, 1, a/√T) in score units
and Law(S_i | present, n_i) = N(μ_0 + β n_i, s_0²). Let R be a (1−α) joint confidence set for (μ_0, s_0, β) (Bonferroni
over a t-interval for μ_0, a χ²-interval for s_0, a t-interval for β from the present-vs-absent mean gap divided by
the mean present count — unbiased since E[gap] = β E[N]). Report ε̂ = min_{θ ∈ R} ε_θ(δ), where ε_θ is Theorem 1's
ε computed with the known p_n and parameters θ. Then P(ε̂ ≤ ε_mech(δ)) ≥ 1 − α, where ε_mech is the privacy of the
per-round release. Proof: with probability ≥ 1−α the true θ* ∈ R, so ε̂ ≤ ε_{θ*}(δ) = ε of the final-model statistic
= ε of a post-processing of the per-round release ≤ ε_mech(δ).

## Proposition 5 (validity of the path-aware audit)
Same setup; the auditor estimates a with a (1−α) lower confidence limit a_lo (from β_lo√T / s_0,hi, Bonferroni) and
reports ε̂ = ε_full(a_lo), with ε_full(a) the privacy of the per-round release at signal-to-noise a under the known
process (computed by Monte Carlo on the hidden-Markov likelihood ratio, with the Monte-Carlo error treated as in
Theorem 3's bound or by a conservative sample size). ε_full is nondecreasing in a (the present law at a' < a is a
post-processing... more precisely: X at a' equals X at a with the canary signal scaled, a data-processing argument on
the shift). Then P(ε̂ ≤ ε_mech(δ)) ≥ 1 − α.
Simulation (T = 200, λ = 0.9, m = 1000, 200 audits): ε̂ = 5.72 ± 1.50 (47% of 12.08; coverage 1.00; flags the Poisson
claim in 99% of audits) vs 4.06 (count-conditioned), 1.78 (Gaussian fit), 1.39 (nonparametric LR threshold).

## Proposition 6 (nonparametric likelihood-ratio audit)
With ℓ_i = log dP/dQ(X^{(i)}) computed at the claimed (C, σ) and the known process (forward algorithm), for a fixed
threshold grid {t_j} and Clopper–Pearson bounds at level α/(2J): if p_lo(t_j) − e^ε q_hi(t_j) > δ for some j then
the mechanism is not (ε, δ)-DP, with probability of a false rejection ≤ α. No shape assumption; valid for the real
training mechanism given independent canaries. Its power is capped by the sample size: with m/2 absent canaries and
zero exceedances, ε̂ ≤ log(p_lo / q_hi) ≈ log(0.5 · m/2 / 3.7).

## Empirical finding E1 (canary erasure by training dynamics)
On Fashion-MNIST logistic regression with PCA-50 features (d = 510) and random canary directions, the per-round
canary signal observed in the aggregate is attenuated by ≈ 42% over 200 rounds (present-score mean 0.81 vs 1.40
predicted): subsequent local training pulls the model back along the canary direction (curvature u^T H u of the
population loss). With canary directions in the bottom-634 PCA subspace of pixel space (d = 7850), where the data
variance and hence the curvature are negligible, the attenuation disappears (1.58 and 1.32 vs 1.40, two runs).
This is the "identical background" idealization of Agrawal et al. (Sec. 6) failing in low dimension, and a design
rule for canaries: choose directions of negligible curvature.

## Step 14 — merged certified path-aware auditor (replaces the step-function certification of Step 12)

**Setting.** Known participation law (chain or empirical path law), per-canary observations X = aB + Z in noise units, m canaries with independent presence coins. The auditor computes â_lo (regression on the known paths, confidence 1−α_est) and maps it to ε through a Monte-Carlo table of the true privacy curve of the law.

**Proposition 7 (certified map).** Let lp_1..lp_n ~ log LR(X) with X ~ P_a (present world) and lq_1..lq_n with X ~ Q (absent world), independent of the audit data. For fixed ε let
  m̂_P(ε) = (1/n) Σ_i (1 − e^{ε − lp_i})_+ ,  m̂_Q(ε) = (1/n) Σ_i (1 − e^{ε + lq_i})_+ ,
both averages of [0,1]-valued i.i.d. variables with means H_ε(P_a‖Q) and H_ε(Q‖P_a). Define μ_lo(m̂, n, α) = min{μ : n·kl(m̂, μ) ≤ ln(1/α)} (Chernoff–Hoeffding, KL form; valid for [0,1]-valued variables since E e^{sX} ≤ 1 − μ + μe^s) and
  δ_lo(ε) = max{ μ_lo(m̂_P(ε), n, α_mc/2), μ_lo(m̂_Q(ε), n, α_mc/2) } ,
  ε_cert(a) = sup{ ε : δ_lo(ε) > δ } .
Then (i) δ_lo is nonincreasing in ε (m̂_P, m̂_Q are, and μ_lo is increasing in m̂); (ii) for every fixed ε, P[δ_lo(ε) > δ(ε)] ≤ α_mc where δ(ε) = max(H_ε(P_a‖Q), H_ε(Q‖P_a)); (iii) hence P[ε_cert(a) > ε_true(a)] ≤ α_mc, where ε_true(a) = inf{ε : δ(ε) ≤ δ}.
Proof of (iii): if ε_cert > ε_true then by (i) δ_lo(ε_true) > δ ≥ δ(ε_true), an event of probability ≤ α_mc by (ii) applied at the fixed point ε_true. ∎

**Proposition 8 (grid floor, no union bound).** Let a_1 < … < a_G be a fixed grid and ε_cert(a_k) computed from independent samples for each k. The audit reports ε̂ = ε_cert(a_{k*}) with k* = max{k : a_k ≤ â_lo}. Then P[ε̂ > ε_true(a)] ≤ α_est + α_mc. Proof: ε_true(·) is nondecreasing (more signal per participation is less private), so on the event {â_lo ≤ a} (probability ≥ 1−α_est) we have a_{k*} ≤ a and ε_true(a_{k*}) ≤ ε_true(a); k* is a function of the audit data, independent of the Monte-Carlo samples, so conditionally on k* Proposition 7 gives P[ε_cert(a_{k*}) > ε_true(a_{k*}) | k*] ≤ α_mc. No Bonferroni over the grid is needed. ∎

**Cost of certification (where the losses were).** Step 12 used δ_lo(ε) = max_c (1−e^{−c}) CP_lo[P(log LR > ε+c)], a step-function minorant of the integrand: at 2M samples it certifies 9.31 where the plain Monte-Carlo value is 10.32 (−10%). The KL-form bound certifies 9.94 (−3.7%) from the same samples; the remaining gap is statistical (δ_lo = 0.77δ at n = 2M, 0.69δ at 1M, 0.59δ at 500k) and shrinks as n^{-1/2}. The 0.05 grid floor cost 10–15%; the 0.01 grid costs < 2%. Splitting α as 0.045/0.005 instead of 0.025/0.025 raises â_lo slightly at a negligible cost on the Monte-Carlo side (ln(1/α) enters only logarithmically).

## Step 15 — the dispersion law as a mathematical statement (final-model release)

**Setting.** Sum statistic S = Σ_t X_t/√T; present law M_p = Σ_n p_n N(c n, 1) with c = a/√T, absent law Q = N(0,1); ε_sum(δ; p) is a functional of the count law p alone (Theorem 1). Likelihood ratio L_p(x) = E_p[g_x(N)], g_x(n) = exp(c n x − c² n²/2), increasing in x.

**What the numerics say (step15_*.csv; T = 200, q = 0.1, a = 0.5, δ ∈ {10⁻², …, 10⁻⁶}).**
1. Convex order: along 36 chains of 25 random mean-preserving spreads each (900 steps; each step p ≤_cx p′), ε_sum never decreased beyond the solver's tolerance at any of the five δ (largest apparent decrease < 5×10⁻⁵, re-examined on a 20× finer grid in step15b).
2. Dispersion index: among the 13 standard laws with mean 20 (Binomial, four Markov chains, three min-sep renewal laws, three beta-binomials, all-or-none, deterministic), 76 of 78 pairs are convex-ordered and ε_sum follows D in all of them; the two non-ordered pairs contain a counterexample to the D-ordering: Markov λ = 0.97 (D = 49.4) is less private than beta-binomial ρ = 0.3 (D = 54.6) at δ ≤ 10⁻⁴ (27.1 vs 26.9 at 10⁻⁴; 38.4 vs 36.6 at 10⁻⁶).
3. Adversarial pairs with equal mean and equal variance (±0.2%) but different upper tails: the heavier upper tail is less private at every δ, with the gap widening as δ falls (trial 32: 2.95 vs 2.73 at 10⁻⁴, 3.98 vs 3.20 at 10⁻⁵, 5.48 vs 3.63 at 10⁻⁶) even though its D is marginally *smaller*.
Conclusion: the law is a statement about the convex (stop-loss) order of the count law, not about its variance; D orders every standard family and is the right one-number summary for a regime map, but the theorem must be stated in convex order. The two extreme elements of the convex order on {0,…,T} with mean qT are both consistent: the deterministic count (cyclic participation) is the least private (2.533 at 10⁻⁴, the minimum over all laws tested) and all-or-none (frozen chain) the most (43.7).

**Proposition 9 (far-tail monotonicity, rigorous).** Fix p ≤_cx p′ with equal means. Let x*_p(ε) be the threshold with L_p(x) > e^ε ⇔ x > x*_p(ε). If x*_p(ε) ≥ c(T+1) then H_ε(M_{p′}‖Q) ≥ H_ε(M_p‖Q).
Proof. g_x″(n) = g_x(n)·[(cx − c²n)² − c²] ≥ 0 whenever |x − cn| ≥ 1; for x ≥ c(T+1) this holds for every n ∈ [0, T], so g_x is convex on the support and E_{p′}[g_x(N)] ≥ E_p[g_x(N)], i.e. L_{p′}(x) ≥ L_p(x) for all x > x*_p(ε). Hence (L_{p′}(x) − e^ε)_+ ≥ (L_p(x) − e^ε)_+ pointwise on {x > x*_p}, and the right-hand side vanishes elsewhere; integrate against Q. ∎
Remarks. (i) The condition is a lower bound on ε: it holds for all ε ≥ ε₀(p) := log L_p(c(T+1)), i.e. for δ below a threshold that depends on p (for the Binomial at these parameters ε₀ ≈ 4.8, δ₀ ≈ 10⁻⁹; for heavier laws δ₀ is larger). (ii) The reverse direction H_ε(Q‖M_p) = E_Q[(1 − e^ε L_p)_+] is *anti*-monotone on {x ≤ −1} by the same convexity argument; what the auditor and accountant report is the maximum of the two directions, and in the tested range the forward direction is the active one. (iii) Between δ₀ and the tested range the law is supported by item 1 above but not proved; the paper states Proposition 9 and labels the full-range statement Conjecture 1 with the 900-step evidence.

**Conjecture 1.** For every δ ∈ (0, 1) and all count laws p ≤_cx p′ on {0,…,T} with equal means, ε_sum(δ; p) ≤ ε_sum(δ; p′).

**Step 15c — exact certificates along the chains (step15c_lrcheck.csv).** For each of the 900 spreads p → p′ and each δ, the pointwise condition of Proposition 9 was checked on a 400,001-point grid: L_{p′}(x) ≥ L_p(x) for all x on the rejection region {x > x*_p(ε_p(δ))}. Where it holds, H_ε(M_{p′}‖Q) ≥ H_ε(M_p‖Q) follows exactly (no integration error). It holds in 900/900 steps at δ = 10⁻⁶, 896/900 at 10⁻⁵, 879/900 at 10⁻⁴, 811/900 at 10⁻³ and 648/900 at 10⁻²; in the remaining steps the sufficient condition fails (the non-convex window of g_x intersects the rejection region) while the computed ε still did not decrease (step 15). So the convex-order law is proved step-by-step on the whole sample at δ ≤ 10⁻⁵ up to four steps, and supported numerically elsewhere.

## Step 18 — availability models as accountant input (FLASH window, fixed cohort K = 27)
Empirical path law: D = 21.3, ε_sum(10⁻⁴) = 5.84 (5.16× Poisson-count), ε_sum(10⁻⁵) = 8.50.
M1 per-device two-state Markov (p_i, λ_i): D 26.9, ε 7.43 (1.27×, conservative). M2 pooled alternating renewal: 8.18 (1.40×). M3 renewal with per-device off-law scaling: 7.44 (1.27×). M4 pooled hour-of-day inhomogeneous Markov: 3.37 (0.58×, anti-conservative; best KS 0.033 but erases heterogeneity). M5 M4 + heuristic per-device factor: 4.59 (0.79×). M6 M4 + per-device factor calibrated to p_i (periodic steady state by iteration, bisection on the factor): 5.83 (1.00×) at 10⁻⁴, 7.86 (0.93×) at 10⁻⁵.
Reading: over a 33-hour horizon the upper tail of the count law is carried by the always-available minority; any model that pools devices loses it and under-states ε; any model that keeps p_i per device is within 27–40% and conservative unless it also models the nightly pool shrinkage, which brings it to the empirical value. Recommendation for the accountant: per-device Markov fit as the conservative default; calibrated diurnal chain when accuracy matters; never a homogeneous population model.

## Corrections after the Section-3 review (2026-10-16)
- Proposition 9: the convexity condition is x ≥ cT + 1 (|x − cn| ≥ 1 for all n ≤ T), not c(T+1). Thresholds under the corrected condition: Binomial(200, 0.1), a = 0.5: ε₀ = log L(cT+1) = 6.08, δ₀ = 1.8e-14; Markov λ = 0.9: ε₀ = 18.6, δ₀ = 6e-8. The proposition orders the forward direction only; the ordering of ε(δ) needs the forward direction to be the active one at ε(δ) (true in every computed configuration; δ₂ = 0 once e^{−ε} ≤ p₀).
- Step 15c's check is a numerical pointwise verification of L_{p′} ≥ L_p on the rejection region (400,001-point grid), not a check of Prop. 9's hypothesis; wording: "verified pairwise on the sample to grid precision".
- Theorem 2 restated as the asymptotic claim only: for v > 0 there is ε₀ with δ_fit(ε) ≥ δ_exact(ε) for ε ≥ ε₀ (quadratic vs linear log-likelihood ratio, pointwise domination of (L̃ − e^ε)₊ on the half-line, integrate). The "conservative at practical δ" statement is numerical and holds for positively autocorrelated laws; for the most concentrated laws (min-sep 5, 9) the fit is above the exact value by 0.01–0.07% at every δ computed (theory_crossover.csv).
- Theorem 1: δ₂(ε) = 0 whenever e^{−ε} ≤ p₀ (L ≥ p₀); h_ε(0) := 0; forward recursion O(T²|S|²).
- Trace pipeline unified (step21): policy table with stride 10 on the 572-device 3-day window; fixed-cohort-0.6 ratio 5.75 (this draw) vs 5.16 (the draw used for the audit references) — run-to-run variability of one realisation ≈ 10%. Horizon curves at fixed round length and K: 10-min, K=27: D 11.3→33.8, ratio 5.1→6.2 over 8–67 h; 5-min, K=23: D 22→74.6, ratio 9.4→7.3 over 8–83 h (δ=1e-4). The trace error does not decay with the horizon; the chain's does (final-model 3.09× at T=200 → 1.73× at T=1000).
- Schema/true ratio (pilot_a, eps_schema_maxp / eps_full_mc): 1.3–5.3× (T=200 laws), 1.0–1.3× for the cycle.

## Corrections after the Section-6 review (2026-10-16)
- Policy table convention: one fresh random stream (seed 0) per policy (step21b_policies.csv); the shared-stream table (step21_policies.csv) differs on the thinning-0.2/0.5 and cohort-0.3/0.6/1.0 rows (e.g. 0.6-cohort final-model ratio 5.75 vs 5.16). All sections now use the seed-0 draw; Table 3.2 and Table 6.2 updated.
- Theorem 3 bound on the running example (seed-0 draw, stride 10): 19.94 at 1e-4 (1.87x the MC reference 10.69), 24.33 at 1e-5 (1.62x of 14.98); it is an UPPER bound and cannot place the loss above Poisson.
- Step 18 verdict corrected: per-device time-homogeneous models (M1, M3) conservative 1.17–1.27x; pooled renewal M2 1.40x for an artefactual reason (exchangeable devices inherit the heaviest on-run tail; support to N = 160 vs 113); pooled diurnal M4 0.57–0.58x, M5 0.74–0.79x; calibrated M6 1.00x / 0.93x. Tail mass P(N>=80): empirical 0.0028, M1 0.0071, M2 0.0045, M3 0.0067, M4 0.0003, M5 0.0010, M6 0.0031.
- Heterogeneity decomposition (step21b_tails.csv): Var(p_i)/E[p_i] = 0.083 (10-min) and 0.072 (5-min) against fitted slopes 0.064 / 0.058 (77–80%); OLS intercepts 8.3 / 16.4 vs chain value 2.7 at lambda 0.48; between-device share of Var(N_T): 0.38 (T=50) → 0.83 (200) → 0.99 (400); 0.33 (T=100) → 0.99 (1000).
- Tail support of the running example (T=200): N>=60 from 19 devices (1.3% of windows), N>=80 from 6 (0.28%), N>=100 from 2 (0.06%); T=1000 (5-min): N>=200 from 19 devices, N>=300 from 5.
- Horizon ranges (final-model ratio at 1e-4): 10-min 4.9–6.2 over 8–67 h (no trend); 5-min 9.4 → 6.4–7.3, declining then flat beyond 50 h.
- MC reference of the trace law grows with sample size (10.35 at 1M, 10.69 at 3M at 1e-4): heavy-tailed LR, far tail under-sampled; references are lower estimates, recoveries upper estimates, validity unaffected.
- Trace clock phase never used (users in several time zones); FLASH paper describes one week from 31 Jan 2020 while the sample spans 23 Jan – 5 Feb.
- Timings on the count law: Theorem 1 exact 0.07 s, Theorem 3 bound 0.002 s.
- Manuscript cross-references: the chain's decay with T is a regime-map numeric (Section 3.6), not a theorem; the sum-audit blindness follows from Theorem 1, not Lemma 1.

## Fine-grid check of the trace row at delta=1e-5 (2026-10-17)
A 0.01 grid (0.38–0.52, 1M samples per point) on the trace law gives certified 11.31±1.02 at 1e-5 (0/120 above the 1M reference 15.705; 120/120 flags) and 7.82±0.82 at 1e-4, against the paper's coarse-grid 11.35±1.03 and 7.88±0.84. The refinement changes nothing; one sentence recording it was added to the protocol appendix (full version) and Sec. S-III (TIFS supplement). The 1M reference of this run (15.705 at 1e-5) is above the 3M value (14.97) the paper uses, another instance of the heavy-tailed trace reference.
