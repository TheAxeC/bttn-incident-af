"""Robustness / ablation suite for the BTTN incident-AF result.

Runs on the CACHED spectral cohort tensor (reuses bench.load_cohort_tensor so labels align
with the cache; no waveform extraction). Reports AUROC@1yr mean±sd over seeds + patient-
bootstrap CI for:
  --mode K        : block-count sensitivity (K = 4/8/12/16/24) - the main interpretability knob
  --mode horizon  : 1/2/5-yr (via the survival cohort + multihorizon, see that script)
Outputs results/ablation_<mode>.csv.

Run:  python ablation.py --mode K --source ecg --horizon 365 --neg_ratio 4 --seeds 3
"""
from __future__ import annotations
import argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from core import config
from core.bench import load_cohort_tensor, folds, boot_paired
from core.bttn import bttn_fold

K_GRID = [4, 8, 12, 16, 24]


def cv_scores(X, coh, K, seeds):
    y = coh.label.to_numpy(); t = coh.time.to_numpy(); e = coh.event.to_numpy()
    per_seed = np.zeros((seeds, len(y)))
    for s in range(seeds):
        f = folds(y, config.N_FOLDS, s)
        for fold in range(config.N_FOLDS):
            tr, te = np.where(f != fold)[0], np.where(f == fold)[0]
            per_seed[s, te] = bttn_fold(X[tr], t[tr], e[tr], X[te], y[tr], seed=s, K=K)
    aucs = [roc_auc_score(y, per_seed[s]) for s in range(seeds)]
    return np.mean(aucs), np.std(aucs), per_seed.mean(0)


def main(mode, source, horizon, neg_ratio, seeds):
    coh, X = load_cohort_tensor(source, horizon, None, "spectral", neg_ratio)
    coh["time"] = np.where(coh.label == 1, coh["time_to_event_days"], coh["fu_days"]).astype(float)
    coh["event"] = coh.label.astype(int)
    y = coh.label.to_numpy()
    print(f"[ablation:{mode}] n={len(coh)} prevalence={y.mean():.3f} seeds={seeds}")
    rows = []
    if mode == "K":
        for K in K_GRID:
            m, sd, _ = cv_scores(X, coh, K, seeds)
            rows.append({"K": K, "auroc": round(m, 3), "sd": round(sd, 3)})
            print(f"  K={K:2d}: AUROC {m:.3f} ± {sd:.3f}")
        df = pd.DataFrame(rows)
    else:
        raise SystemExit("use multihorizon.py for the horizon ablation")
    df.to_csv(config.RESULTS_DIR / f"ablation_{mode}.csv", index=False)
    print(df.to_string(index=False))
    print(f"[saved] results/ablation_{mode}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="K", choices=["K"])
    ap.add_argument("--source", default="ecg"); ap.add_argument("--horizon", type=int, default=365)
    ap.add_argument("--neg_ratio", type=int, default=4); ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    main(a.mode, a.source, a.horizon, a.neg_ratio, a.seeds)
