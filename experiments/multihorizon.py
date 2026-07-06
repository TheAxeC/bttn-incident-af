"""Multi-horizon incident-AF risk from ONE survival BTTN.

Trains the survival BTTN once on continuous time-to-AF (Cox loss), produces an
out-of-fold risk score per patient, then reads discrimination at 1/2/5-year horizons
on the SAME patients (at-risk filtering per horizon) plus the overall C-index. A
clinical Cox model (age/sex + ECG intervals) is the multi-horizon baseline. This is
the capability a per-horizon binary CNN lacks: a single interpretable model that
gives calibrated risk at any horizon.

Run:  python multihorizon.py --source ecg --tensor spectral
"""
from __future__ import annotations

import argparse
import time as _time

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

from core import config
from core import data
from core.bttn import bttn_fold

# Horizons read off the SAME survival risk score (no per-horizon retraining).
#   120d = "90-day onset (post-blank)": positive = first AF in (index+30d, index+120d],
#          a true short-onset INCIDENT-prediction construct that preserves the 30-day
#          blank (the survival cohort already excludes onset <= BLANKING_DAYS, so
#          event<=120 == onset in (30,120]). NOT near-term detection.
#   365/730/1825d = 1/2/5-yr incident horizons.
HORIZONS = [120, 365, 730, 1825]
HORIZON_LABEL = {120: "90-day onset (post-blank)", 365: "1yr", 730: "2yr", 1825: "5yr"}
CLIN = ["anchor_age", "sex_m", "rr_interval", "p_dur", "qrs_dur", "qt", "p_axis", "qrs_axis", "t_axis"]


def horizon_auroc(risk, time, event, H):
    """Cases = event by H; controls = AF-free through H; censored-before-H excluded."""
    lab = np.full(len(time), -1)
    lab[(event == 1) & (time <= H)] = 1
    lab[time > H] = 0                                  # event or censor after H -> AF-free through H
    m = lab >= 0
    if len(np.unique(lab[m])) < 2:
        return np.nan, int(m.sum())
    return roc_auc_score(lab[m], risk[m]), int(m.sum())


def horizon_auroc_ci(risk, time, event, H, n=2000, seed=0):
    """Patient-bootstrap 95% CI for horizon AUROC (same resampling as bench.boot_paired).
    Returns (auc, lo, hi, n_eval, n_pos) with auc==nan if a horizon has <2 classes."""
    lab = np.full(len(time), -1)
    lab[(event == 1) & (time <= H)] = 1
    lab[time > H] = 0
    m = lab >= 0
    y, sc = lab[m], risk[m]
    n_pos = int((y == 1).sum())
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, int(m.sum()), n_pos
    rng = np.random.RandomState(seed); idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        d.append(roc_auc_score(y[b], sc[b]))
    d = np.array(d)
    return (float(roc_auc_score(y, sc)), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), int(m.sum()), n_pos)


def main(source, tensor, seeds):
    coh = pd.read_csv(config.RESULTS_DIR / f"cohort_{source}_survival.csv")
    coh["sex_m"] = (coh.get("gender", pd.Series(index=coh.index)).astype(str).str.upper() == "M").astype(float)
    coh["time"] = np.where(coh.label == 1, coh["time_to_event_days"], coh["fu_days"]).astype(float)
    coh["event"] = coh.label.astype(int)

    cache = config.RESULTS_DIR / f"X_{tensor}_{source}_survival_n{len(coh)}.npy"
    if cache.exists():
        X = np.load(cache); print(f"[cache] {X.shape}")
    else:
        t = _time.time()
        recs = data.load_many(coh["path"].tolist())
        ok = [i for i, r in enumerate(recs) if r is not None]
        coh = coh.iloc[ok].reset_index(drop=True)
        from core.preprocess import build_spectral_tensor, build_tensor
        X = (build_spectral_tensor([recs[i] for i in ok]) if tensor == "spectral"
             else build_tensor([recs[i] for i in ok], mode="median_beat"))
        np.save(cache, X); print(f"[built] {X.shape} in {_time.time()-t:.0f}s")

    time_a, event_a = coh.time.to_numpy(), coh.event.to_numpy()
    y_naive = coh.label.to_numpy()                     # only used for early-stopping AUROC proxy
    rng = np.random.RandomState(0)
    fold = np.empty(len(coh), int)
    for c in (0, 1):
        idx = np.where(event_a == c)[0]; rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % config.N_FOLDS

    cov_all = coh[[c for c in CLIN if c in coh]].to_numpy(float)   # clinical features for the fusion model
    risk_bttn = np.zeros(len(coh)); risk_fuse = np.zeros(len(coh)); risk_clin = np.zeros(len(coh))
    for f in range(config.N_FOLDS):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        # early-stopping proxy label = event within 1yr among training patients
        y_es = ((event_a[tr] == 1) & (time_a[tr] <= 365)).astype(int)
        risk_bttn[te] = bttn_fold(X[tr], time_a[tr], event_a[tr], X[te], y_es, seed=0)
        risk_fuse[te] = bttn_fold(X[tr], time_a[tr], event_a[tr], X[te], y_es, seed=0,
                                  cov_tr=cov_all[tr], cov_te=cov_all[te])   # BTTN + clinical fusion
        # clinical Cox
        dfc = coh.iloc[tr][[c for c in CLIN if c in coh]].copy()
        dfc["time"] = time_a[tr]; dfc["event"] = event_a[tr]
        dfc = dfc.fillna(dfc.median(numeric_only=True))
        keep = [c for c in dfc.columns[:-2] if dfc[c].std() > 1e-9]
        cox = CoxPHFitter(penalizer=0.1).fit(dfc[keep + ["time", "event"]], "time", "event")
        dte = coh.iloc[te][keep].fillna(dfc[keep].median())
        risk_clin[te] = cox.predict_partial_hazard(dte).to_numpy().ravel()
        print(f"  fold {f} done")

    print("\n=== Multi-horizon AUROC (one survival model) ===")
    rows = []
    for H in HORIZONS:
        ab, lo, hi, nb, npos = horizon_auroc_ci(risk_bttn, time_a, event_a, H)
        af, flo, fhi, nf, _ = horizon_auroc_ci(risk_fuse, time_a, event_a, H)
        ac, nc = horizon_auroc(risk_clin, time_a, event_a, H)
        lab = HORIZON_LABEL.get(H, f"{H}d")
        rows.append({"horizon": lab, "horizon_days": H, "horizon_yr": round(H / 365, 3),
                     "n_eval": nb, "n_pos": npos,
                     "BTTN": round(ab, 3), "BTTN_lo": round(lo, 3), "BTTN_hi": round(hi, 3),
                     "BTTN_clin": round(af, 3), "BTTN_clin_lo": round(flo, 3), "BTTN_clin_hi": round(fhi, 3),
                     "clinical": round(ac, 3)})
        print(f"  {lab:24s} (n={nb:5d}, pos={npos:4d}):  "
              f"BTTN {ab:.3f} [{lo:.3f}-{hi:.3f}]   BTTN+clin {af:.3f} [{flo:.3f}-{fhi:.3f}]   clinical {ac:.3f}")
    cidx_b = concordance_index(time_a, -risk_bttn, event_a)
    cidx_f = concordance_index(time_a, -risk_fuse, event_a)
    cidx_c = concordance_index(time_a, -risk_clin, event_a)
    print(f"\n  C-index:  BTTN {cidx_b:.3f}   BTTN+clin {cidx_f:.3f}   clinical {cidx_c:.3f}")
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / f"multihorizon_{source}_{tensor}.csv", index=False)
    np.savez(config.RESULTS_DIR / f"multihorizon_{source}_{tensor}.npz",
             risk_bttn=risk_bttn, risk_fuse=risk_fuse, risk_clin=risk_clin, time=time_a, event=event_a)
    print(f"[saved] multihorizon_{source}_{tensor}.csv  (C-index BTTN {cidx_b:.3f} / fuse {cidx_f:.3f} / clin {cidx_c:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ecg")
    ap.add_argument("--tensor", default="spectral", choices=["spectral", "median"])
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    main(a.source, a.tensor, a.seeds)
