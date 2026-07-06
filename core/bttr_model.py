"""Thin sklearn-style wrapper around the vendored, real BTTR (bttrlib).

BTTR is a regression: we map the binary AF label to {-1, +1}, fit on the training
fold, and use the continuous prediction as the decision score (threshold-free
AUROC/AUPRC are the headline; a 0-threshold gives a point classifier). This is
fully INDUCTIVE - `fit` never sees test data - which is the whole reason to use
real BTTR instead of the thesis's transductive CP-features surrogate.

Interpretability: after fitting, `block_maps()` returns the K per-block weight
tensors reshaped to (time x lead) plus their regression coefficients b_k, i.e. the
glass-box AF signatures the paper leads with.
"""
from __future__ import annotations

import numpy as np

from core import config
from bttrlib import BTTR


class BTTRClassifier:
    def __init__(self, nfactor=None, enforce_rank1=None, snrs=None, ratios=None):
        self.nfactor = nfactor or config.BTTR_NFACTOR
        self.enforce_rank1 = config.BTTR_ENFORCE_RANK1 if enforce_rank1 is None else enforce_rank1
        self.snrs = snrs or config.BTTR_SNRS
        self.ratios = ratios or config.BTTR_RATIOS
        self._m = None
        self._shape = None         # (T, L) of the training tensor, for reshaping factors

    def fit(self, X, y):
        """X: (N, T, L) float tensor; y: (N,) in {0,1}."""
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        self._shape = X.shape[1:]
        ymap = np.where(np.asarray(y) == 1, 1.0, -1.0)[:, None]
        self._m = BTTR()
        self._m.train(X, ymap, nFactor=self.nfactor,
                      SNRs=self.snrs, ratios=self.ratios,
                      enforce_rank_1=self.enforce_rank1)
        # train() breaks early when the residual norm collapses, so fewer than
        # nfactor blocks may exist; predict() loops over nFactor -> clamp it.
        self._m.nFactor = len(self._m.E.models)
        return self

    def fit_continuous(self, X, f):
        """Fit BTTR against an arbitrary continuous target f (N,) -- e.g. survival
        martingale residuals. Same supervised, inductive block extraction; only the
        regression objective's target changes (binary {-1,+1} -> continuous risk)."""
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        self._shape = X.shape[1:]
        f = np.asarray(f, dtype=float).reshape(-1, 1)
        self._m = BTTR()
        self._m.train(X, f, nFactor=self.nfactor, SNRs=self.snrs, ratios=self.ratios,
                      enforce_rank_1=self.enforce_rank1)
        self._m.nFactor = len(self._m.E.models)
        return self

    def transform(self, X):
        """Block scores T = X_(1) @ W  -> (N, K). The supervised latent features the
        downstream survival/risk head consumes."""
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if self._m is None or len(self._m.E.models) == 0:
            return np.zeros((X.shape[0], 1))
        W = np.asarray(self._m.getModel()[0], dtype=float)          # (features, K)
        return X.reshape((X.shape[0], -1), order="F") @ W

    def decision_function(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        out = self._m.predict(X)
        if not out or out[-1] is None:        # BTTR broke early / built 0 blocks
            return np.zeros(X.shape[0])
        return np.asarray(out[-1], dtype=float).ravel()

    def predict_proba(self, X):
        s = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-s))          # squashing for a proba-like column
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

    # --- interpretability -------------------------------------------------- #
    def block_maps(self):
        """Return (maps, b) where maps is (K, T, L) per-block weight tensors and
        b is (K,) the per-block regression coefficients. Each map is a clinically-
        readable spatio-temporal AF signature (reshape uses BTTR's order='F')."""
        modelX, modelY = self._m.getModel()           # (T*L, K), (K, 1)
        modelX = np.asarray(modelX, dtype=float)
        b = np.asarray(modelY, dtype=float).ravel()
        K = modelX.shape[1]
        maps = np.stack([modelX[:, k].reshape(self._shape, order="F") for k in range(K)], 0)
        return maps, b              # (K, *self._shape) -- 2D (T,L) or N-D (B,T,L)

    @property
    def n_blocks(self):
        return 0 if self._m is None else len(getattr(self._m, "ace_results", []))
