"""Glass-box incident-AF substrate map from the BTTN.

Two complementary readouts, both on a held-out split:
  1. learned per-block (freq x lead) signatures (BlockTermLayer.maps) + the temporal
     conv kernels -> the interpretable block structure;
  2. occlusion importance -- zero each frequency band / each lead in the held-out
     input and measure the AUROC drop -> a model-faithful answer to "which spectral
     bands and which leads carry the future-AF signal in a normal sinus ECG".

Output: results/bttn_interpret.png + results/bttn_importance.csv
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from core import config
from core.bttn import BTTN, _std, _bttn_train_one, cox_ph_loss, DEV, MAX_EPOCH, BATCH

FMAX, NPERSEG, FS = 40.0, 128, 125.0           # spectral builder: fs=250/decimate2=125
LEADS = config.LEAD_NAMES


def main(horizon=365, source="ecg", neg_ratio=4):
    # reuse the same spectral tensor cache as the ladder (load_cohort_tensor handles build/load and
    # returns coh aligned with X, with time/event set) -- fixes the old hardcoded _n7854 provisional cache.
    from core.bench import load_cohort_tensor
    coh, X = load_cohort_tensor(source, horizon, None, "spectral", neg_ratio)
    time = coh["time"].to_numpy()
    y = coh.label.to_numpy()

    rng = np.random.RandomState(0); idx = rng.permutation(len(X))
    cut = int(0.8 * len(idx)); tr, te = idx[:cut], idx[cut:]
    Xtr_raw, Xte_raw = X[tr].astype(np.float32), X[te].astype(np.float32)
    F, T, L = X.shape[1:]

    # Train a small ensemble with the SAME procedure as the reported model (train_model does the
    # internal standardisation + an 85/15 validation split + early stopping on val AUROC with
    # best-state restore), so the occlusion importances reflect the ACTUAL model, not a single net
    # run to the last epoch with no early stopping (the old readout, which under-performed).
    from core.bttn import train_model, predict_risk
    N_ENS = 3
    models = [train_model(Xtr_raw, time[tr], y[tr], y[tr].astype(int), seed=sd)
              for sd in range(N_ENS)]

    def auc_of(Xin_raw):                            # ensemble-averaged risk on RAW input (predict_risk std's it)
        s = np.mean([predict_risk(m, Xin_raw, st) for m, st in models], axis=0)
        return roc_auc_score(y[te], s)

    base = auc_of(Xte_raw)
    freqs = np.linspace(0, FMAX, F)
    fimp = []
    for fi in range(F):
        Xo = Xte_raw.copy(); Xo[:, fi] = 0.0        # zero this frequency band's power
        fimp.append(base - auc_of(Xo))
    limp = []
    for li in range(L):
        Xo = Xte_raw.copy(); Xo[..., li] = 0.0      # zero this lead
        limp.append(base - auc_of(Xo))

    imp = pd.DataFrame({"freq_hz": np.round(freqs, 1), "freq_importance": fimp})
    imp_l = pd.DataFrame({"lead": LEADS, "lead_importance": limp})
    imp.to_csv(config.RESULTS_DIR / "bttn_importance_freq.csv", index=False)
    imp_l.to_csv(config.RESULTS_DIR / "bttn_importance_lead.csv", index=False)

    maps = np.mean([m.block.maps() for m, _ in models], axis=0)   # ensemble-averaged block signatures (K, F, L)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(freqs, fimp, width=freqs[1] - freqs[0]); ax[0].set_title(f"Frequency importance (base AUROC {base:.3f})")
    ax[0].set_xlabel("Hz"); ax[0].set_ylabel("AUROC drop when occluded")
    ax[1].barh(LEADS, limp); ax[1].set_title("Lead importance"); ax[1].invert_yaxis()
    agg = np.abs(maps).mean(0)                  # mean |block signature| (F x L)
    im = ax[2].imshow(agg, aspect="auto", origin="lower",
                      extent=[0, len(LEADS), 0, FMAX], cmap="magma")
    ax[2].set_title("BTTN block signature |freq x lead|"); ax[2].set_xlabel("lead idx"); ax[2].set_ylabel("Hz")
    fig.colorbar(im, ax=ax[2]); fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "bttn_interpret.png", dpi=130)
    print(f"[interpret] base AUROC {base:.3f}")
    print("top frequency bands (Hz):", imp.sort_values("freq_importance", ascending=False).head(5).to_dict("records"))
    print("top leads:", imp_l.sort_values("lead_importance", ascending=False).head(4).to_dict("records"))
    print("[saved] results/bttn_interpret.png + importance CSVs")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ecg", choices=["icd", "ecg"])
    ap.add_argument("--horizon", type=int, default=365)
    ap.add_argument("--neg_ratio", type=int, default=4)
    a = ap.parse_args()
    main(a.horizon, a.source, a.neg_ratio)
