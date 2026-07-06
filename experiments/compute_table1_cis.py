"""Standalone: patient-bootstrap 95% CI of AUROC for the Table-I discrimination ladder,
from already-committed saved out-of-fold scores. Same convention as bench.py / boot_auc:
RandomState(0), n=2000 resamples, rng.choice(len(y), len(y), True), skip resamples that
lose a class, percentile [2.5, 97.5]. Writes results/table1_cis.csv. Does not retrain anything.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core import config

RES = config.RESULTS_DIR          # code/results (or $AFNET_RESULTS) - NOT experiments/results
N_BOOT, SEED = 2000, 0


def boot_auc(y, s, n=N_BOOT, seed=SEED):
    """Identical to bench.py-family boot_auc (patient/row bootstrap)."""
    rng = np.random.RandomState(seed)
    v = []
    for _ in range(n):
        b = rng.choice(len(y), len(y), True)
        if len(np.unique(y[b])) == 2:
            v.append(roc_auc_score(y[b], s[b]))
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# (ladder model label, npz file, key) -- only rows whose per-patient scores are cached.
SPEC = "bench_spectral_icd_h365.npz"   # gold ICD-10 ladder: spectral, n=9,275, prev 0.20, 1-yr, 5-seed ensemble
ROWS = [
    ("clinical (age/sex + ECG intervals)", SPEC, "clinical"),
    ("plain BTTR (spectral)", SPEC, "bttr"),
    ("CNN (black-box)", SPEC, "cnn"),
    ("conv-BTTN (ECG-only)", SPEC, "bttn"),
    ("conv-BTTN + SSL", SPEC, "bttn_ssl"),
    ("conv-BTTN + clinical", SPEC, "bttn_cov"),
    ("conv-BTTN + SSL + clinical (full)", SPEC, "bttn_ssl_cov"),
]

# The gold 7-method run caches per-patient (seed-ensemble) scores for every ladder row.
NOT_CACHED = []

out = []
for label, fname, key in ROWS:
    d = np.load(RES / fname)
    y, s = d["y"], d[key]
    auc, lo, hi = boot_auc(y, s)
    out.append(dict(model=label, auroc=round(auc, 4), ci_lo=round(lo, 4),
                    ci_hi=round(hi, 4), source_artifact=fname, n_boot=N_BOOT))

df = pd.DataFrame(out, columns=["model", "auroc", "ci_lo", "ci_hi", "source_artifact", "n_boot"])
df.to_csv(RES / "table1_cis.csv", index=False)
print(df.to_string(index=False))
print("\nNot cached locally (no per-patient scores -> no CI computed):")
for r in NOT_CACHED:
    print("  -", r)
