"""Faithfulness probe CONTROLS.

PROBE INPUT (stated precisely). The faithfulness claim is that a LINEAR readout of the
BTTN's interpretable block factor scores recovers ~96% of full AUROC. Those scores are the
STATIC, HEAD-INDEPENDENT block-factor projections:
    s(X) = mean_t ( X x1 A_k^T x3 C_k^T )          == BlockTermLayer.block_scores(X)
i.e. the pure multilinear projection of the spectral tensor onto the learned per-block
frequency factors A_k and lead factors C_k, time-averaged. These are exactly the quantities
the glass-box substrate map is built from. They are computed WITHOUT the nonlinear conv/MLP
head -- so a probe on them cannot smuggle in head capacity (no concept-bottleneck leakage).

This script CALIBRATES that ~96% number: is task-recovery special to the BTTN factor space,
or is incident-AF near-linear in ANY reasonable space? We fit the SAME linear logistic readout
(patient-level CV, patient-bootstrap CI) on five probe sources:
  (1) bttn_factors  : BTTN static block factor scores               [the claim]
  (2) raw_flat      : the raw flattened spectral tensor (F*T*L)       [is the task just linear?]
  (3) cnn_penult    : the black-box CNN's 64-d penultimate features   [upper-ish bound]
  (4) random_proj   : a fixed random projection of the flattened input to the SAME dim as the
                      BTTN factors                                    [matched-dim null]
  (5) labelblind    : block factors from a LABEL-BLIND block layer (SSL-pretrained if available,
                      else random-init -- never saw the labels)       [is supervision needed?]

Output: results/faithfulness_controls.csv
  probe_source, auroc, ci_lo, ci_hi, recovery_vs_full
("full" = the supervised BTTN full-model AUROC, computed here on the same folds.)

Run (HPC, cached spectral tensor):  python faithfulness_controls.py --seeds 3
"""
from __future__ import annotations
import argparse, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
from core import config
from core.bench import load_cohort_tensor, folds
from core.bttn import train_model, predict_risk, BlockTermLayer, DEV


def boot_auc(y, s, n=2000, seed=0):
    """Patient-bootstrap 95% CI of AUROC (one ECG per patient -> row bootstrap)."""
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        b = rng.choice(len(y), len(y), True)
        if len(np.unique(y[b])) == 2:
            v.append(roc_auc_score(y[b], s[b]))
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _bttn_factors(model, X, stats):
    m, s = stats; Xs = (X.astype(np.float32) - m) / s
    with torch.no_grad():
        return model.block_scores(torch.tensor(Xs, device=DEV)).cpu().numpy()


def _block_layer_factors(block, X, stats):
    """block_scores of an arbitrary (e.g. label-blind) BlockTermLayer."""
    m, s = stats; Xs = (X.astype(np.float32) - m) / s
    with torch.no_grad():
        x = torch.tensor(Xs, device=DEV)
        if x.ndim == 4:                      # (B,F,T,L) -> already correct for block_scores
            return block.block_scores(x).cpu().numpy()
        return block.block_scores(x[:, None]).cpu().numpy()


def probe_cv(feat_fn, y, t, e, X, seeds, full_scores=None):
    """Out-of-fold linear logistic probe AUROC for a feature extractor feat_fn(tr,te)->(Ftr,Fte).
    Returns seed-averaged out-of-fold probe scores."""
    probe = np.zeros((seeds, len(y)))
    for sd in range(seeds):
        f = folds(y, config.N_FOLDS, sd)
        for fold in range(config.N_FOLDS):
            tr, te = np.where(f != fold)[0], np.where(f == fold)[0]
            Ftr, Fte = feat_fn(tr, te, sd)
            lr = LogisticRegression(max_iter=2000).fit(Ftr, y[tr])
            probe[sd, te] = lr.predict_proba(Fte)[:, 1]
    return probe.mean(0)


def main(source, horizon, neg_ratio, seeds):
    coh, X = load_cohort_tensor(source, horizon, None, "spectral", neg_ratio)
    coh["time"] = np.where(coh.label == 1, coh["time_to_event_days"], coh["fu_days"]).astype(float)
    coh["event"] = coh.label.astype(int)
    y = coh.label.to_numpy(); t = coh.time.to_numpy(); e = coh.event.to_numpy()
    print(f"[faithfulness_controls] n={len(coh)} prev={y.mean():.3f} seeds={seeds}")

    # --- supervised full BTTN: full AUROC + the (1) bttn_factors probe in ONE pass --------
    full = np.zeros((seeds, len(y))); bttn_probe = np.zeros((seeds, len(y)))
    blind_probe = np.zeros((seeds, len(y)))
    ssl_path = config.RESULTS_DIR / "ssl_encoder.pt"
    has_ssl = ssl_path.exists()
    for sd in range(seeds):
        fld = folds(y, config.N_FOLDS, sd)
        for fold in range(config.N_FOLDS):
            tr, te = np.where(fld != fold)[0], np.where(fld == fold)[0]
            model, stats = train_model(X[tr], t[tr], e[tr], y[tr], seed=sd)
            full[sd, te] = predict_risk(model, X[te], stats)
            Ftr = _bttn_factors(model, X[tr], stats); Fte = _bttn_factors(model, X[te], stats)
            lr = LogisticRegression(max_iter=2000).fit(Ftr, y[tr])
            bttn_probe[sd, te] = lr.predict_proba(Fte)[:, 1]
            # (5) label-blind block layer: SSL-pretrained if available else random-init; never fit on labels
            xtr4 = X[tr] if X[tr].ndim == 4 else X[tr][:, None]
            blind = BlockTermLayer(*xtr4.shape[1:]).to(DEV)
            if has_ssl:
                blind.load_state_dict(torch.load(ssl_path, map_location=DEV), strict=False)
            blind.eval()
            Btr = _block_layer_factors(blind, X[tr], stats); Bte = _block_layer_factors(blind, X[te], stats)
            lrb = LogisticRegression(max_iter=2000).fit(Btr, y[tr])
            blind_probe[sd, te] = lrb.predict_proba(Bte)[:, 1]

    full_avg = full.mean(0); full_auc, _, _ = boot_auc(y, full_avg)
    bttn_avg = bttn_probe.mean(0); blind_avg = blind_probe.mean(0)
    Xflat = X.reshape(len(X), -1).astype(np.float32)

    # dim of the block factors (matched dim for the random projection control)
    from core.bttn import BTTN as _BTTN
    dummy = _BTTN(*(X.shape[1:] if X.ndim == 4 else (1, *X.shape[1:])))
    d_block = dummy.block.K * dummy.block.rf * dummy.block.rl

    # --- (2) raw flattened spectral tensor probe (standardised per-fold) -------------------
    def raw_feat(tr, te, sd):
        mu = Xflat[tr].mean(0); sg = Xflat[tr].std(0) + 1e-6
        return (Xflat[tr] - mu) / sg, (Xflat[te] - mu) / sg
    raw_avg = probe_cv(raw_feat, y, t, e, X, seeds)

    # --- (4) random projection of flattened input to d_block dims (fixed across folds) -----
    rngP = np.random.RandomState(0)
    P = rngP.randn(Xflat.shape[1], d_block).astype(np.float32) / np.sqrt(Xflat.shape[1])
    Xproj = Xflat @ P

    def proj_feat(tr, te, sd):
        mu = Xproj[tr].mean(0); sg = Xproj[tr].std(0) + 1e-6
        return (Xproj[tr] - mu) / sg, (Xproj[te] - mu) / sg
    proj_avg = probe_cv(proj_feat, y, t, e, X, seeds)

    # --- (3) CNN penultimate features probe ----------------------------------------------
    from core.cnn import cnn_features_fold

    def cnn_feat(tr, te, sd):
        fe, ftr = cnn_features_fold(X[tr], y[tr], X[te], seed=sd)
        # standardise on train features
        mu = ftr.mean(0); sg = ftr.std(0) + 1e-6
        return (ftr - mu) / sg, (fe - mu) / sg
    cnn_avg = probe_cv(cnn_feat, y, t, e, X, seeds)

    sources = {
        "bttn_factors": bttn_avg,
        "raw_flat": raw_avg,
        "cnn_penult": cnn_avg,
        "random_proj": proj_avg,
        "labelblind": blind_avg,
    }
    rows = []
    print(f"\n[full BTTN AUROC] {full_auc:.3f}  (block factor dim={d_block}, "
          f"label-blind source={'SSL' if has_ssl else 'random-init'})")
    for name, sc in sources.items():
        a, lo, hi = boot_auc(y, sc)
        rows.append({"probe_source": name, "auroc": round(a, 3),
                     "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                     "recovery_vs_full": round(a / full_auc, 3)})
        print(f"  {name:13s} AUROC {a:.3f} [{lo:.3f},{hi:.3f}]  recovery {a/full_auc:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "faithfulness_controls.csv", index=False)
    print(df.to_string(index=False)); print("[saved] results/faithfulness_controls.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ecg"); ap.add_argument("--horizon", type=int, default=365)
    ap.add_argument("--neg_ratio", type=int, default=4); ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    main(a.source, a.horizon, a.neg_ratio, a.seeds)
