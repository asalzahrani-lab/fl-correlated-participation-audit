# Device-availability trace

The experiments of Section VI use the 1,000-device state trace distributed with the FLASH simulator
(Yang et al., "Characterizing Impacts of Heterogeneity in Federated Learning upon Large-Scale Smartphone Data", WWW 2021),
file `data/state_traces.json` of https://github.com/PKU-Chengxu/FLASH (BSD-2-Clause). It is not copied into this
repository; `bash fetch_trace.sh` clones FLASH (shallow) and places the file here as `trace/state_traces.json`
(37 MB; the MD5 of the copy we used is recorded in `fetch_trace.sh`).

`trace_analysis.py` applies FLASH's reading of the Bonawitz et al. eligibility rule (locked ∧ Wi-Fi ∧ charging, sampled
at the start of each round), computes per-device availability and lag-1 autocorrelation at several round lengths,
the eligible-pool dynamics, and the count laws under the selection policies (`pn_*.npy`, used by Table V of the full
version). The window selection, policies with a fresh random stream per policy, the horizon runs and the availability
models are in `src/step21_trace_unified.py`, `src/step21b_consistent.py`, `src/step22_window_stats.py` and
`src/step18_availability_models.py`, which import `trace_analysis.py` from this directory.

The trace clock's time zone is not documented by FLASH; the paper uses the shape of the diurnal profile, never its phase.
