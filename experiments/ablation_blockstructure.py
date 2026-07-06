"""Block-term-STRUCTURE ablation.

Does the block-term parameterisation of the interpretable first layer DO WORK, or is it
just dim-reduction? We hold the entire downstream model identical (multi-scale temporal
conv + nonlinear MLP head + Cox loss) and swap ONLY the first interpretable layer between:

  blockterm : K blocks, each a learnable rank-(rf,rl) Tucker freq/lead subspace
              (A_k in R^{F x rf}, C_k in R^{L x rl}) -- the contribution / current model.
  cp        : CP rank-1-per-block (rf=rl=1) -> K rank-1 freq(x)lead atoms; isolates whether
              the multilinear RANK (>1) inside a block matters.
  fullrank  : a single full-rank learned F x L -> ch linear projection per time step (no block,
              no per-mode low-rank structure) -> plain dim-reduction with the SAME output width.

For each variant we report, on the same 5-fold x N-seed protocol as the faithfulness sweep
(arch_sweep.py):
  - full AUROC (+ patient-bootstrap 95% CI, reuse bench.boot_auc-style resampling)
  - linear-probe recovery = AUROC of a logistic readout of the PURE interpretable block factor
    scores (block.block_scores) / full AUROC  (the faithfulness ratio).

Output: results/ablation_blockstructure.csv
  variant, full_auroc, full_ci_lo, full_ci_hi, probe_auroc, faithfulness_ratio, seeds, folds

Run (HPC, cached spectral tensor):  python ablation_blockstructure.py --seeds 3
"""
from __future__ import annotations
import argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
from core import config
from core.bench import load_cohort_tensor, folds
from core.bttn import train_model, predict_risk, DEV

VARIANTS = ["blockterm", "cp", "fullrank"]


def boot_auc(y, s, n=2000, seed=0):
    """Patient-bootstrap 95% CI of AUROC (one ECG per patient -> row bootstrap)."""
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        b = rng.choice(len(y), len(y), True)
        if len(np.unique(y[b])) == 2:
            v.append(roc_auc_score(y[b], s[b]))
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def probe_scores(model, X, stats):
    m, s = stats; Xs = ((X.astype(np.float32) - m) / s)
    with torch.no_grad():
        return model.block_scores(torch.tensor(Xs, device=DEV)).cpu().numpy()


def main(source, horizon, neg_ratio, seeds):
    coh, X = load_cohort_tensor(source, horizon, None, "spectral", neg_ratio)
    coh["time"] = np.where(coh.label == 1, coh["time_to_event_days"], coh["fu_days"]).astype(float)
    coh["event"] = coh.label.astype(int)
    y = coh.label.to_numpy(); t = coh.time.to_numpy(); e = coh.event.to_numpy()
    print(f"[ablation_blockstructure] n={len(coh)} prev={y.mean():.3f} seeds={seeds} folds={config.N_FOLDS}")
    rows = []
    for variant in VARIANTS:
        full = np.zeros((seeds, len(y))); probe = np.zeros((seeds, len(y)))
        for sd in range(seeds):
            f = folds(y, config.N_FOLDS, sd)
            for fold in range(config.N_FOLDS):
                tr, te = np.where(f != fold)[0], np.where(f == fold)[0]
                model, stats = train_model(X[tr], t[tr], e[tr], y[tr], seed=sd, block_mode=variant)
                full[sd, te] = predict_risk(model, X[te], stats)
                lr = LogisticRegression(max_iter=1000).fit(probe_scores(model, X[tr], stats), y[tr])
                probe[sd, te] = lr.predict_proba(probe_scores(model, X[te], stats))[:, 1]
        avg_full = full.mean(0); avg_probe = probe.mean(0)
        fa, lo, hi = boot_auc(y, avg_full)
        pa = roc_auc_score(y, avg_probe)
        rows.append({"variant": variant, "full_auroc": round(fa, 3),
                     "full_ci_lo": round(lo, 3), "full_ci_hi": round(hi, 3),
                     "probe_auroc": round(pa, 3), "faithfulness_ratio": round(pa / fa, 3),
                     "seeds": seeds, "folds": config.N_FOLDS})
        print(f"  {variant:9s} full {fa:.3f} [{lo:.3f},{hi:.3f}]  probe {pa:.3f}  "
              f"faithfulness {pa/fa:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "ablation_blockstructure.csv", index=False)
    print(df.to_string(index=False)); print("[saved] results/ablation_blockstructure.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ecg"); ap.add_argument("--horizon", type=int, default=365)
    ap.add_argument("--neg_ratio", type=int, default=4); ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    main(a.source, a.horizon, a.neg_ratio, a.seeds)
