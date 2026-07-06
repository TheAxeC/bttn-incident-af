"""Model-improvement benchmark for incident-AF prediction.

Caches the median-beat tensor once (waveform extraction is the slow part), then runs
several methods through identical stratified patient-level CV over multiple seeds, with
AUROC + patient-bootstrap CI and a paired test of each method vs binary BTTR and vs the
clinical baseline. The goal is a genuine algorithm-level win for glass-box BTTR, in line
with the PTB-XL paper's accepted "matches-or-exceeds at no accuracy cost".

Run:  python bench.py --source ecg --horizon 365 --seeds 5 --methods clinical,bttr,survbttr
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from core import config
from core import data
from core.bttr_model import BTTRClassifier
from core.preprocess import build_tensor
from core.survival import SurvBTTR

CLIN_COLS = ["anchor_age", "sex_m", "rr_interval", "p_dur", "qrs_dur", "qt", "p_axis", "qrs_axis", "t_axis"]


def load_cohort_tensor(source, horizon, max_n, tensor="median", neg_ratio=2):
    coh = pd.read_csv(config.RESULTS_DIR / f"cohort_{source}_h{horizon}.csv")
    coh["sex_m"] = (coh.get("gender", pd.Series(index=coh.index)).astype(str).str.upper() == "M").astype(float)
    pos = coh[coh.label == 1]; neg = coh[coh.label == 0]
    n_neg = min(len(neg), neg_ratio * len(pos))
    coh = pd.concat([pos, neg.sample(n_neg, random_state=0)]).reset_index(drop=True)
    if max_n and len(coh) > max_n:
        frac = max_n / len(coh)
        coh = pd.concat([coh[coh.label == 1].sample(frac=frac, random_state=0),
                         coh[coh.label == 0].sample(frac=frac, random_state=0)]).reset_index(drop=True)

    cache = config.RESULTS_DIR / f"X_{tensor}_{source}_h{horizon}_r{neg_ratio}_n{len(coh)}.npy"
    if cache.exists():
        X = np.load(cache)
        print(f"[cache] {X.shape} from {cache.name}")
    else:
        t = time.time()
        recs = data.load_many(coh["path"].tolist())
        ok = [i for i, r in enumerate(recs) if r is not None]
        coh = coh.iloc[ok].reset_index(drop=True)
        if tensor == "multibeat":
            from core.preprocess import build_multibeat_tensor
            X = build_multibeat_tensor([recs[i] for i in ok])
        elif tensor == "window":
            X = build_tensor([recs[i] for i in ok], mode="window", decimate_factor=2)  # full 10s @250Hz
        elif tensor == "spectral":
            from core.preprocess import build_spectral_tensor
            X = build_spectral_tensor([recs[i] for i in ok])
        else:
            X = build_tensor([recs[i] for i in ok], mode="median_beat")
        np.save(cache, X)
        print(f"[built] {X.shape} in {time.time()-t:.0f}s -> {cache.name}")
    # event time: positives -> time_to_event, controls -> follow-up
    coh["time"] = np.where(coh.label == 1, coh["time_to_event_days"], coh["fu_days"]).astype(float)
    coh["event"] = coh.label.astype(int)
    return coh, X


def folds(y, n, seed):
    rng = np.random.RandomState(seed)
    f = np.empty(len(y), int)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]; rng.shuffle(idx)
        f[idx] = np.arange(len(idx)) % n
    return f


def run_method(name, X, coh, tr, te):
    y = coh.label.to_numpy()
    if name == "clinical":
        Xc = coh[[c for c in CLIN_COLS if c in coh]].to_numpy(float)
        keep = ~np.all(np.isnan(Xc[tr]), axis=0)
        Xc = Xc[:, keep]
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(Xc[tr], y[tr]); return clf.predict_proba(Xc[te])[:, 1]
    if name == "bttr":
        m = BTTRClassifier().fit(X[tr], y[tr]); return m.decision_function(X[te])
    if name == "survbttr":
        m = SurvBTTR().fit(X[tr], coh.time.to_numpy()[tr], coh.event.to_numpy()[tr])
        return m.risk(X[te])
    if name == "survbttr_raw":
        m = SurvBTTR(cox_head=False).fit(X[tr], coh.time.to_numpy()[tr], coh.event.to_numpy()[tr])
        return m.risk(X[te])
    if name == "cnn":
        from core.cnn import cnn_fold
        return cnn_fold(X[tr], y[tr], X[te])
    if name == "gbbttr":
        from gbbttr import GBBTTR
        m = GBBTTR().fit(X[tr], y[tr]); return m.decision_function(X[te])
    if name == "bttn":
        from core.bttn import bttn_fold
        return bttn_fold(X[tr], coh.time.to_numpy()[tr], coh.event.to_numpy()[tr],
                         X[te], y[tr])
    if name in ("bttn_cov", "bttn_ssl_cov"):              # BTTN + the full clinical feature vector
        # cov = the SAME risk factors the clinical baseline uses (age, sex, ECG intervals+axes),
        # so BTTN+cov is a fair superset of the clinical model plus the deep ECG substrate. The
        # covariate weights are warm-started from a logistic fit (clinical-strength) inside bttn_fold.
        from core.bttn import bttn_fold
        cov = coh[[c for c in CLIN_COLS if c in coh]].to_numpy(float)
        pre = str(config.RESULTS_DIR / "ssl_encoder.pt") if name == "bttn_ssl_cov" else None
        return bttn_fold(X[tr], coh.time.to_numpy()[tr], coh.event.to_numpy()[tr],
                         X[te], y[tr], pretrained=pre, cov_tr=cov[tr], cov_te=cov[te])
    if name == "bttn_ssl":
        from core.bttn import bttn_fold
        return bttn_fold(X[tr], coh.time.to_numpy()[tr], coh.event.to_numpy()[tr],
                         X[te], y[tr], pretrained=str(config.RESULTS_DIR / "ssl_encoder.pt"))
    if name == "bttn_attn":
        from core.bttn import bttn_fold
        return bttn_fold(X[tr], coh.time.to_numpy()[tr], coh.event.to_numpy()[tr],
                         X[te], y[tr], head_type="attn")
    if name == "bttn_ace":
        from core.bttn import bttn_fold
        return bttn_fold(X[tr], coh.time.to_numpy()[tr], coh.event.to_numpy()[tr],
                         X[te], y[tr], ace_warm=True)
    raise ValueError(name)


def boot_paired(y, sa, sb, n=2000, seed=0):
    rng = np.random.RandomState(seed); y = np.asarray(y); idx = np.arange(len(y))
    d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        d.append(roc_auc_score(y[b], sa[b]) - roc_auc_score(y[b], sb[b]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def main(source, horizon, seeds, methods, max_n, tensor="median", neg_ratio=2):
    coh, X = load_cohort_tensor(source, horizon, max_n, tensor, neg_ratio)
    y = coh.label.to_numpy()
    print(f"[bench] {len(coh)} pts, prevalence {y.mean():.3f}, methods={methods}, seeds={seeds}")
    scores = {m: np.zeros((seeds, len(y))) for m in methods}
    for s in range(seeds):
        f = folds(y, config.N_FOLDS, s)
        for fold in range(config.N_FOLDS):
            tr, te = np.where(f != fold)[0], np.where(f == fold)[0]
            for m in methods:
                scores[m][s, te] = run_method(m, X, coh, tr, te)
        line = "  ".join(f"{m} {np.mean([roc_auc_score(y, scores[m][s])]):.3f}" for m in methods)
        print(f"  seed {s}: {line}")

    print("\n=== AUROC (mean over seeds, patient-bootstrap CI) ===")
    auroc = {}
    for m in methods:
        per = [roc_auc_score(y, scores[m][s]) for s in range(seeds)]
        auroc[m] = np.mean(per)
        print(f"  {m:14s} {np.mean(per):.3f} +/- {np.std(per):.3f}")
    # paired vs bttr and vs clinical, using seed-averaged scores
    avg = {m: scores[m].mean(0) for m in methods}
    for ref in ("bttr", "clinical", "cnn"):
        if ref not in methods:
            continue
        print(f"\n=== paired AUROC delta vs {ref} ===")
        for m in methods:
            if m == ref:
                continue
            md, lo, hi, p = boot_paired(y, avg[m], avg[ref])
            star = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {m:14s} - {ref:9s}  {md:+.4f} [{lo:+.4f},{hi:+.4f}]  P(>{ref})={p:.2f} {star}")

    np.savez(config.RESULTS_DIR / f"bench_{tensor}_{source}_h{horizon}.npz",
             **{m: avg[m] for m in methods}, y=y)
    print(f"\n[saved] bench_{tensor}_{source}_h{horizon}.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ecg", choices=["icd", "ecg"])
    ap.add_argument("--horizon", type=int, default=365)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max_n", type=int, default=0)
    ap.add_argument("--methods", default="clinical,bttr,survbttr")
    ap.add_argument("--tensor", default="median", choices=["median", "multibeat", "window", "spectral"])
    ap.add_argument("--neg_ratio", type=int, default=2)
    a = ap.parse_args()
    main(a.source, a.horizon, a.seeds, a.methods.split(","), a.max_n or None, a.tensor, a.neg_ratio)
