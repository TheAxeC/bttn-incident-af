"""Intervention / sufficiency test.

A faithful glass-box map should be CAUSAL for the prediction: if the substrate it highlights
truly carries the future-AF signal, ablating the TOP-ranked interpretable factors should
degrade the full model proportionally, and ablating the BOTTOM-ranked ones should barely move
it. This is the sufficiency/necessity counterpart to the linear-probe recovery number.

Protocol (held-out, trained-once BTTN -- same train procedure as interpret_bttn.py):
  1. Rank the interpretable input axes by occlusion importance: for each frequency bin and each
     lead, zero it in the held-out input and record the AUROC drop (the substrate-map ranking).
  2. Sweep n = 0,1,2,...: zero the top-n ranked (freq bin OR lead) axes jointly in the held-out
     input, recompute full-model AUROC -> the intervention curve. A 'bottom-n' (least-important
     first) curve is recorded as a control (should degrade much more slowly).

Output: results/intervention_curve.csv
  n_factors_removed, auroc, delta_vs_baseline   [+ order column: 'top' / 'bottom']

Run (HPC, cached spectral tensor):  python intervention_curve.py
"""
from __future__ import annotations
import argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from core import config
from core.bench import load_cohort_tensor, folds
from core.bttn import train_model, predict_risk, DEV

FMAX = 40.0


def _auc(model, stats, Xin, y):
    return roc_auc_score(y, predict_risk(model, Xin, stats))


def main(source, horizon, neg_ratio, seed):
    coh, X = load_cohort_tensor(source, horizon, None, "spectral", neg_ratio)
    coh["time"] = np.where(coh.label == 1, coh["time_to_event_days"], coh["fu_days"]).astype(float)
    coh["event"] = coh.label.astype(int)
    y = coh.label.to_numpy(); t = coh.time.to_numpy(); e = coh.event.to_numpy()
    F, T, L = X.shape[1:]
    print(f"[intervention] n={len(coh)} prev={y.mean():.3f} F={F} L={L}")

    # single held-out split (one trained model -> faithful occlusion + intervention)
    fld = folds(y, config.N_FOLDS, seed)
    tr, te = np.where(fld != 0)[0], np.where(fld == 0)[0]
    model, stats = train_model(X[tr], t[tr], e[tr], y[tr], seed=seed)
    Xte, yte = X[te], y[te]
    base = _auc(model, stats, Xte, yte)
    print(f"[intervention] held-out base AUROC {base:.3f}")

    # --- rank interpretable axes by single-axis occlusion importance ----------------------
    # axes: F frequency bins + L leads, each identified by ('freq', i) / ('lead', j).
    axes = []
    for fi in range(F):
        Xo = Xte.copy(); Xo[:, fi] = 0.0
        axes.append((("freq", fi), base - _auc(model, stats, Xo, yte)))
    for li in range(L):
        Xo = Xte.copy(); Xo[..., li] = 0.0
        axes.append((("lead", li), base - _auc(model, stats, Xo, yte)))
    ranked = sorted(axes, key=lambda kv: kv[1], reverse=True)        # most important first
    order_top = [a for a, _ in ranked]
    order_bottom = [a for a, _ in sorted(axes, key=lambda kv: kv[1])]  # least important first

    def sweep(order, tag):
        rows = []
        Xo = Xte.copy()
        rows.append({"n_factors_removed": 0, "auroc": round(base, 4),
                     "delta_vs_baseline": 0.0, "order": tag})
        for n, (kind, idx) in enumerate(order, start=1):
            if kind == "freq":
                Xo[:, idx] = 0.0
            else:
                Xo[..., idx] = 0.0
            a = _auc(model, stats, Xo, yte)
            rows.append({"n_factors_removed": n, "auroc": round(a, 4),
                         "delta_vs_baseline": round(a - base, 4), "order": tag})
        return rows

    rows = sweep(order_top, "top") + sweep(order_bottom, "bottom")
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "intervention_curve.csv", index=False)
    # concise summary
    top = df[df.order == "top"]; bot = df[df.order == "bottom"]
    for k in (1, 3, 5, 10):
        if (top.n_factors_removed == k).any():
            dt = float(top.loc[top.n_factors_removed == k, "delta_vs_baseline"].iloc[0])
            db = float(bot.loc[bot.n_factors_removed == k, "delta_vs_baseline"].iloc[0])
            print(f"  remove {k:2d}: top {dt:+.4f}   bottom {db:+.4f}")
    print("[saved] results/intervention_curve.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ecg"); ap.add_argument("--horizon", type=int, default=365)
    ap.add_argument("--neg_ratio", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.source, a.horizon, a.neg_ratio, a.seed)
