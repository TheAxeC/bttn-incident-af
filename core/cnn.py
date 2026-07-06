"""Compact, regularised 1D-CNN ceiling for incident-AF prediction.

Black-box deep-learning reference (Attia-style) on the SAME median-beat tensor BTTR
sees -> a fair head-to-head that tells us the accuracy ceiling and hence whether an
interpretable model can match it ("interpretability at no accuracy cost") or whether
there is a real gap to close. Unlike the PTB-XL pilot (42 positives -> the CNN
overfits), here there are thousands of patients, so the CNN gets an honest shot.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

DEV = torch.device("cuda" if torch.cuda.is_available()
                   else ("mps" if torch.backends.mps.is_available() else "cpu"))
MAX_EPOCH, PATIENCE, N_ENS, BATCH = 80, 12, 3, 256


class SmallCNN(nn.Module):
    def __init__(self, n_leads):
        super().__init__()
        # split the trunk so the 64-d penultimate features can be read out (used as a
        # control probe source in the faithfulness analysis, faithfulness_controls.py).
        self.trunk = nn.Sequential(
            nn.Conv1d(n_leads, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout(0.4),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Dropout(0.5))
        self.fc = nn.Linear(64, 1)

    def features(self, x):                  # (B, 64) penultimate (pre-logit) features
        return self.trunk(x)

    def forward(self, x):
        return self.fc(self.trunk(x)).squeeze(-1)


def _std(Xtr, Xte):
    m = Xtr.mean((0, 2), keepdims=True); s = Xtr.std((0, 2), keepdims=True) + 1e-6
    return (Xtr - m) / s, (Xte - m) / s


def _train_one(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    model = SmallCNN(Xtr.shape[1]).to(DEV)
    pos_w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEV)
    n = len(Xtr_t); best_auc, best_state, bad = -1, None, 0
    for ep in range(MAX_EPOCH):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[b]), ytr_t[b]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = torch.sigmoid(model(Xva_t)).cpu().numpy()
        auc = roc_auc_score(yva, va) if len(np.unique(yva)) == 2 else 0.5
        if auc > best_auc:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    return model


def cnn_fold(Xtr, ytr, Xte, seed=0):
    """Xtr/Xte: (N, T, L) -> CNN expects (N, L, T). Inner val split for early stop,
    3-seed ensemble. Returns test scores."""
    if Xtr.ndim == 4:                       # spectral (N,F,T,L) -> (N, F*L, T) channels
        N, F, T, L = Xtr.shape
        Xtr = np.transpose(Xtr, (0, 1, 3, 2)).reshape(N, F * L, T)
        Xte = np.transpose(Xte, (0, 1, 3, 2)).reshape(Xte.shape[0], F * L, T)
    else:                                   # (N,T,L) -> (N,L,T)
        Xtr = np.transpose(Xtr, (0, 2, 1)); Xte = np.transpose(Xte, (0, 2, 1))
    rng = np.random.RandomState(seed)
    idx = np.arange(len(ytr)); rng.shuffle(idx)
    cut = int(0.85 * len(idx)); tri, vai = idx[:cut], idx[cut:]
    Xt, Xv = _std(Xtr[tri], Xtr[vai]); _, Xe = _std(Xtr[tri], Xte)
    ens = np.zeros(len(Xte))
    for sd in range(N_ENS):
        m = _train_one(Xt, ytr[tri], Xv, ytr[vai], seed * 10 + sd)
        with torch.no_grad():
            ens += torch.sigmoid(m(torch.tensor(Xe, dtype=torch.float32, device=DEV))).cpu().numpy()
    return ens / N_ENS


def cnn_features_fold(Xtr, ytr, Xte, seed=0):
    """Train one CNN (same trunk/early-stop as cnn_fold) and return its 64-d penultimate
    features for the TEST fold -> a control probe source for the faithfulness analysis.
    Returns (feat_te, feat_tr) so the probe can fit on train, evaluate on test."""
    if Xtr.ndim == 4:                       # spectral (N,F,T,L) -> (N, F*L, T)
        N, F, T, L = Xtr.shape
        Xtr = np.transpose(Xtr, (0, 1, 3, 2)).reshape(N, F * L, T)
        Xte = np.transpose(Xte, (0, 1, 3, 2)).reshape(Xte.shape[0], F * L, T)
    else:
        Xtr = np.transpose(Xtr, (0, 2, 1)); Xte = np.transpose(Xte, (0, 2, 1))
    rng = np.random.RandomState(seed)
    idx = np.arange(len(ytr)); rng.shuffle(idx)
    cut = int(0.85 * len(idx)); tri, vai = idx[:cut], idx[cut:]
    Xt, Xv = _std(Xtr[tri], Xtr[vai]); _, Xe = _std(Xtr[tri], Xte)
    _, Xtr_all = _std(Xtr[tri], Xtr)        # all train rows standardised on the same stats
    m = _train_one(Xt, ytr[tri], Xv, ytr[vai], seed * 10)
    m.eval()
    with torch.no_grad():
        ft = m.features(torch.tensor(Xtr_all, dtype=torch.float32, device=DEV)).cpu().numpy()
        fe = m.features(torch.tensor(Xe, dtype=torch.float32, device=DEV)).cpu().numpy()
    return fe, ft
