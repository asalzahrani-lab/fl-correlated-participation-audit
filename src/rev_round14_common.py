"""Round 14: the certified tables of the main evaluation (Tables II and III), merged as step19_rerun_audits.py merges them:
the step-14 table (1M to 2M samples per grid point) with the step-16 entries (10M Markov, 3M trace) overriding it where
both exist, so that every round-14 simulation certifies with the paper's own tables."""
import csv
import numpy as np


def _read(path, col_plain, col_cert):
    rows = list(csv.DictReader(open(path)))
    return (np.array([float(r["a"]) for r in rows]), np.array([float(r[col_plain]) for r in rows]), np.array([float(r[col_cert]) for r in rows]))


class MergedTable:
    def __init__(self, parts):
        d = {}
        for a, p, c in parts:
            for ai, pi_, ci in zip(a, p, c): d[round(float(ai), 2)] = (pi_, ci)
        self.a = np.array(sorted(d)); self.p = np.array([d[k][0] for k in self.a]); self.c = np.array([d[k][1] for k in self.a])

    def cert(self, x):
        if not np.isfinite(x): return 0.0
        i = np.searchsorted(self.a - 1e-9, x, side="right") - 1
        return 0.0 if i < 0 else float(self.c[i])

    def plain(self, x): return float(np.interp(x, self.a, self.p, left=0.0))


def paper_table(law):
    """law in markov, iid, trace, cyclic; the table used for Table III at delta = 1e-4 (the only delta merged here)."""
    if law == "markov":
        return MergedTable([_read("step14_table_markov.csv", "eps_plain", "eps_cert_bern_0.005"), _read("step16_table_markov.csv", "eps_plain_1e-4", "eps_cert_1e-4")])
    if law == "iid":
        return MergedTable([_read("step14_table_iid.csv", "eps_plain", "eps_cert_bern_0.005")])
    if law == "trace":
        return MergedTable([_read("step14_table_trace.csv", "eps_plain", "eps_cert_bern_0.005"), _read("step16_table_trace.csv", "eps_plain_1e-4", "eps_cert_1e-4")])
    if law == "cyclic":
        return MergedTable([_read("rev_round10_cyclic_table.csv", "eps_plain_1e-4", "eps_cert_1e-4")])
    raise ValueError(law)
