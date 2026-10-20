"""Architecture sweep: does extending the neural part help - and what does it cost in
interpretability? Variants of BTTN's temporal model / head, each scored on:
  - full AUROC (5-fold patient-level, multi-seed)
  - FAITHFULNESS probe = AUROC of a logistic readout of the PURE multilinear block scores
    (the interpretable A_k/C_k projections the substrate map reports). faithfulness = probe/full;
    a richer nonlinear part that lifts full AUROC but not the probe is adding BLACK-BOX capacity
    (the concept-bottleneck leakage failure mode).

Variants: conv (current) | gru (recurrent over spectrogram time) | deephead (wider/deeper MLP).
Run on HPC (cached spectral tensor):  python arch_sweep.py --seeds 3
"""
from __future__ import annotations
import argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from core import config
from core.bench import load_cohort_tensor, folds
from core.bttn import train_model, predict_risk, std_stats, DEV
import torch

VARIANTS = {"conv": {}, "gru": {"temporal": "gru"}, "deephead": {"deep_head": True}}


def probe_scores(model, X, stats):
    m, s = stats; Xs = ((X.astype(np.float32) - m) / s)
    with torch.no_grad():
        return model.block_scores(torch.tensor(Xs, device=DEV)).cpu().numpy()


def main(source, horizon, neg_ratio, seeds):
    coh, X = load_cohort_tensor(source, horizon, None, "spectral", neg_ratio)
    coh["time"] = np.where(coh.label == 1, coh["time_to_event_days"], coh["fu_days"]).astype(float)
    coh["event"] = coh.label.astype(int)
    y = coh.label.to_numpy(); t = coh.time.to_numpy(); e = coh.event.to_numpy()
    print(f"[arch_sweep] n={len(coh)} prev={y.mean():.3f}")
    rows = []
    for name, kw in VARIANTS.items():
        full = np.zeros((seeds, len(y))); probe = np.zeros((seeds, len(y)))
        for sd in range(seeds):
            f = folds(y, config.N_FOLDS, sd)
            for fold in range(config.N_FOLDS):
                tr, te = np.where(f != fold)[0], np.where(f == fold)[0]
                model, stats = train_model(X[tr], t[tr], e[tr], y[tr], seed=sd, **kw)
                full[sd, te] = predict_risk(model, X[te], stats)
                lr = LogisticRegression(max_iter=1000).fit(probe_scores(model, X[tr], stats), y[tr])
                probe[sd, te] = lr.predict_proba(probe_scores(model, X[te], stats))[:, 1]
        fa = np.mean([roc_auc_score(y, full[s]) for s in range(seeds)])
        fsd = np.std([roc_auc_score(y, full[s]) for s in range(seeds)])
        pa = np.mean([roc_auc_score(y, probe[s]) for s in range(seeds)])
        rows.append({"variant": name, "full_auroc": round(fa, 3), "full_sd": round(fsd, 3),
                     "probe_auroc": round(pa, 3), "faithfulness": round(pa / fa, 3)})
        print(f"  {name:9s} full {fa:.3f}+/-{fsd:.3f}  probe {pa:.3f}  faithfulness {pa/fa:.3f}")
    df = pd.DataFrame(rows); df.to_csv(config.RESULTS_DIR / "arch_sweep.csv", index=False)
    print(df.to_string(index=False)); print("[saved] results/arch_sweep.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ecg"); ap.add_argument("--horizon", type=int, default=365)
    ap.add_argument("--neg_ratio", type=int, default=4); ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    main(a.source, a.horizon, a.neg_ratio, a.seeds)
