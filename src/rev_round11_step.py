"""Revision round 11 (internal review of v10): the step-function column of Table III under the current protocol.

Earlier releases took the step-function entries of Table III from older runs (step 14, confidence limits at level
alpha_est/2, a 0.05 grid).  This script recomputes them with the protocol of every other column: the same 120 audit data
sets per law (step19_rerun_audits.py, seed 1919; the cyclic law from rev_round10_cyclic.py), the same lower limit
a_lo (alpha_est = 0.045), the same grid and floor rule, and a step-function (Clopper-Pearson) certificate at
alpha_mc = 0.005 computed from the same Monte-Carlo samples as the KL-certified entries.

Stage "table" (python rev_round11_step.py table markov|trace [grid indices, e.g. 0,1,2]):
  replays the certified tables of step16_delta5.py with the same seeds (3031 Markov, 3032 trace), the same law, grid,
  sample size and chunking, so that the samples are identical; checks that the KL-certified values reproduce the
  published step16_table_<law>.csv, and writes the step-function certified values at delta = 1e-4 and 1e-5 from those
  samples to rev_round11_step_table_<law>[_part<indices>].csv.  Each process replays the whole random stream and computes
  the grid points it is given, so the grid can be split over processes; the parts are concatenated into
  rev_round11_step_table_<law>.csv (stage "join").
Stage "apply" (python rev_round11_step.py apply):
  merges the step-function tables exactly as step19_rerun_audits.py merges the KL tables (step 14, 2M samples, below
  the step-16 grid; step 16 on its grid), recovers each audit's grid point from its stored certified value (the
  certified table is strictly increasing in a, so the value identifies the grid point read), reads the step-function
  value at that grid point, and writes rev_round11_step_results.csv (one row per law at delta = 1e-4)."""
import numpy as np, csv, time, sys, json, os
sys.path.insert(0, os.environ.get("P3_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace")))

stage = sys.argv[1]
t0 = time.time()
log = lambda s: print(f"{s} [{time.time()-t0:.0f}s]", flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("P3_RESULTS", os.path.join(HERE, "..", "results"))

if stage == "table":
    from corraudit import two_state_chain
    from certified import EmpiricalPathLaw, ChainLaw, plain_eps, delta_lo_curve, eps_from_lower_curve
    from certified2 import CertifiedTable2, eps_cert_bern
    which = sys.argv[2]
    # ---- identical to step16_delta5.py up to the table (same seeds, law, grid, sample size, chunking)
    T, a_true, q = 200, 0.5, 0.1
    N_MC = 10_000_000 if which == "markov" else 3_000_000
    A_EST, A_MC = 0.045, 0.005
    rng = np.random.default_rng(3031 if which == "markov" else 3032)
    a_grid = np.round(np.arange(0.44, 0.5201, 0.01), 2) if which == "markov" else np.array([0.42, 0.44, 0.46, 0.48, 0.50])
    DELTAS = [1e-5, 1e-4]
    if which == "trace":
        from trace_analysis import ready_intervals, availability_grid
        d = json.load(open(os.path.join(os.environ.get("P3_TRACE", os.path.join(HERE, "..", "trace")), "state_traces.json")))
        devs = []
        for k, v in d.items():
            iv, tf, tl = ready_intervals(v["messages"])
            if tf is None or tl - tf < 2 * 86400: continue
            devs.append((k, iv, tf, tl))
        step = 600
        t0_ = min(tf for (_, _, tf, _) in devs); t1_ = max(tl for (_, _, _, tl) in devs)
        times = np.arange(t0_, t1_, step); Rn = len(times)
        Amat = np.zeros((len(devs), Rn), dtype=bool); cover = np.zeros((len(devs), Rn), dtype=bool)
        for i, (k, iv, tf, tl) in enumerate(devs):
            Amat[i] = availability_grid(iv, t0_, t1_, step); cover[i] = (times >= tf) & (times <= tl)
        W = 3 * 144
        best = max(range(0, Rn - W, 36), key=lambda s0: (cover[:, s0:s0 + W].all(axis=1)).sum())
        sel = cover[:, best:best + W].all(axis=1)
        A = Amat[sel, best:best + W]; N, Tw = A.shape
        rng_pol = np.random.default_rng(0)
        Bpol = np.zeros_like(A); Epool = A.sum(axis=0).mean(); K = int(round(0.6 * Epool))
        for t in range(Tw):
            idx = np.flatnonzero(A[:, t])
            chosen = idx if len(idx) <= K else rng_pol.choice(idx, K, replace=False)
            Bpol[chosen, t] = True
        paths = np.array([Bpol[i, s0:s0 + T] for i in range(N) for s0 in range(0, Tw - T + 1, 10)], dtype=np.int8)
        law = EmpiricalPathLaw(paths); name = "trace_cohort0.6"
        log(f"trace law: {len(paths)} paths, q={paths.mean():.4f}")
    else:
        P, pi, emit = two_state_chain(q, 0.9); law = ChainLaw(P, pi, emit, T); name = "markov0.9"
    # grid points computed by this process (all by default); every process replays the full random stream, so that the
    # samples at each grid point are identical to those of step16_delta5.py (the loop below is CertifiedTable2's)
    sub = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(len(a_grid)))
    log(f"== {name}: replaying {N_MC} samples, grid points {[f'{a_grid[i]:.2f}' for i in sub]}, seed {3031 if which == 'markov' else 3032}")
    from corraudit import loglr_hmm
    chunk = 4000 if which == "trace" else 25_000
    lps = np.empty((len(sub), N_MC), np.float32); lqs = np.empty((len(sub), N_MC), np.float32)
    done = 0; is_emp = isinstance(law, EmpiricalPathLaw)
    if is_emp:
        BT = law.B.T; const0 = law.logw.astype(np.float32); npath = law.n_path.astype(np.float32)
    while done < N_MC:
        k = min(chunk, N_MC - done)
        B = law.sample(k, rng)
        Zp = rng.standard_normal((k, law.T)).astype(np.float32)
        Zq = rng.standard_normal((k, law.T)).astype(np.float32)
        if is_emp:
            Gp = B.astype(np.float32) @ BT; ZpB = Zp @ BT; ZqB = Zq @ BT
            for j, gi in enumerate(sub):
                a = a_grid[gi]; const = const0 - 0.5 * a * a * npath
                Zm = a * (a * Gp + ZpB) + const[None, :]
                m_ = Zm.max(axis=1, keepdims=True)
                lps[j, done:done + k] = (m_ + np.log(np.exp(Zm - m_).sum(axis=1, keepdims=True)))[:, 0]
                Zm = a * ZqB + const[None, :]
                m_ = Zm.max(axis=1, keepdims=True)
                lqs[j, done:done + k] = (m_ + np.log(np.exp(Zm - m_).sum(axis=1, keepdims=True)))[:, 0]
        else:
            Bf = B.astype(np.float64); Zpd = Zp.astype(np.float64); Zqd = Zq.astype(np.float64)
            for j, gi in enumerate(sub):
                a = a_grid[gi]
                lps[j, done:done + k] = loglr_hmm(Bf * a + Zpd, law.P, law.pi, law.emit, a, 1.0)
                lqs[j, done:done + k] = loglr_hmm(Zqd, law.P, law.pi, law.emit, a, 1.0)
        done += k
        if (done // chunk) % 100 == 0: log(f"  samples {done}/{N_MC}")
    pub = {r["a"]: r for r in csv.DictReader(open(os.path.join(RES, f"step16_table_{which}.csv")))}
    out = []
    for j, gi in enumerate(sub):
        a = a_grid[gi]
        lp = lps[j].astype(np.float64); lq = lqs[j].astype(np.float64)
        cert4 = eps_cert_bern(lp, lq, DELTAS[1], A_MC); cert5 = eps_cert_bern(lp, lq, DELTAS[0], A_MC)
        step4 = eps_from_lower_curve(delta_lo_curve(lp, lq, A_MC), DELTAS[1]); step5 = eps_from_lower_curve(delta_lo_curve(lp, lq, A_MC), DELTAS[0])
        p = pub[f"{a:.2f}"]
        same = (f"{cert4:.4f}" == p["eps_cert_1e-4"]) and (f"{cert5:.4f}" == p["eps_cert_1e-5"])
        out.append([f"{a:.2f}", f"{cert4:.4f}", p["eps_cert_1e-4"], f"{cert5:.4f}", p["eps_cert_1e-5"], int(same), f"{step4:.4f}", f"{step5:.4f}"])
        log(f"  a={a:.2f}: KL 1e-4 {cert4:.4f} (published {p['eps_cert_1e-4']}), 1e-5 {cert5:.4f} (published {p['eps_cert_1e-5']}) "
            f"{'reproduced' if same else 'DIFFERENT'}; step-function 1e-4 {step4:.4f}, 1e-5 {step5:.4f}")
    part = "" if len(sys.argv) <= 3 else "_part" + sys.argv[3].replace(",", "")
    with open(f"rev_round11_step_table_{which}{part}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "eps_cert_1e-4_replay", "eps_cert_1e-4_published", "eps_cert_1e-5_replay", "eps_cert_1e-5_published",
                    "reproduced", "eps_step_1e-4", "eps_step_1e-5"])
        w.writerows(out)
    log(f"all KL values reproduced: {all(r[5] == 1 for r in out)}")

elif stage == "join":
    which = sys.argv[2]
    import glob
    parts = sorted(glob.glob(f"rev_round11_step_table_{which}_part*.csv"))
    allr = [r for p in parts for r in csv.DictReader(open(p))]
    allr.sort(key=lambda r: float(r["a"]))
    with open(f"rev_round11_step_table_{which}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allr[0].keys())); w.writeheader(); w.writerows(allr)
    log(f"{which}: {len(allr)} grid points from {len(parts)} parts; all reproduced: {all(r['reproduced'] == '1' for r in allr)}")

elif stage == "apply":
    def rows(name):
        return list(csv.DictReader(open(os.path.join(RES, name))))

    def merged(parts):
        """parts: list of (a list, value list); later parts override earlier ones, as in step19_rerun_audits.Table."""
        d = {}
        for a, v in parts:
            for ai, vi in zip(a, v): d[round(float(ai), 2)] = float(vi)
        g = np.array(sorted(d)); return g, np.array([d[k] for k in g])

    def col(name, c):
        r = rows(name); return [x["a"] for x in r], [x[c] for x in r]
    t14 = {w: rows(f"step14_table_{w}.csv") for w in ("markov", "iid", "trace")}
    t16 = {w: rows(f"step16_table_{w}.csv") for w in ("markov", "trace")}
    s16 = {w: rows(f"rev_round11_step_table_{w}.csv") for w in ("markov", "trace")}
    assert all(int(r["reproduced"]) == 1 for w in s16 for r in s16[w]), "step-16 replay did not reproduce the published KL table"
    kl, st = {}, {}
    for w in ("markov", "iid", "trace"):
        p_kl = [([r["a"] for r in t14[w]], [r["eps_cert_bern_0.005"] for r in t14[w]])]
        p_st = [([r["a"] for r in t14[w]], [r["eps_cert_step_0.005"] for r in t14[w]])]
        if w in t16:
            p_kl.append(([r["a"] for r in t16[w]], [r["eps_cert_1e-4"] for r in t16[w]]))
            p_st.append(([r["a"] for r in s16[w]], [r["eps_step_1e-4"] for r in s16[w]]))
        kl[w] = merged(p_kl); st[w] = merged(p_st)
        assert np.array_equal(kl[w][0], st[w][0]) and np.all(np.diff(kl[w][1]) > 0)
    aud = np.load(os.path.join(RES, "step19_audits.npz"))
    refs = {"markov": 11.869, "iid": 2.8662, "trace": 10.685}
    claims = {"markov": 2.8662, "iid": 2.8662, "trace": 1.2177}
    out = []
    for w in ("markov", "iid", "trace"):
        g, kv = kl[w]; sv = st[w][1]
        cert = aud[f"{w}_cert_1e-4"]; vals = []
        for c in cert:
            if c == 0.0: vals.append(0.0); continue
            i = np.flatnonzero(np.abs(kv - round(float(c), 4)) < 5e-5)
            assert len(i) == 1, (w, c); vals.append(float(sv[i[0]]))
        v = np.array(vals)
        out.append([w, len(v), f"{v.mean():.4f}", f"{v.std():.4f}", f"{refs[w]:.4f}", f"{v.mean() / refs[w]:.4f}", int(np.sum(v > refs[w])), int(np.sum(v > claims[w]))])
        log(f"{w}: step-function certified {v.mean():.3f} +- {v.std():.3f} recovery {v.mean() / refs[w]:.3f} above ref {int(np.sum(v > refs[w]))}/{len(v)} flags {int(np.sum(v > claims[w]))}/{len(v)}")
    cy = {r["estimator"]: r for r in rows("rev_round10_cyclic_results.csv")}["step_1e-4"]
    out.append(["cyclic", int(cy["R"]), cy["mean"], cy["sd"], cy["reference"], cy["recovery"], int(cy["n_above_reference"]), int(cy["n_above_poisson"])])
    log(f"cyclic (rev_round10_cyclic.py, same protocol): {float(cy['mean']):.3f} +- {float(cy['sd']):.3f} recovery {float(cy['recovery']):.3f}")
    with open("rev_round11_step_results.csv", "w", newline="") as f:
        w_ = csv.writer(f); w_.writerow(["law", "R", "mean", "sd", "reference", "recovery", "n_above_reference", "n_above_poisson"]); w_.writerows(out)
