"""Survival-driven Block-Term Tensor Regression (Surv-BTTR).

Incident-AF prediction is a *time-to-event* problem: a patient who develops AF at
40 days and one who is censored AF-free at 5 years are both "label=1 within horizon"
to a binary model, which throws the timing away. Surv-BTTR changes the regression
OBJECTIVE the block decomposition optimises from a binary {-1,+1} fit to a survival
fit, so deflation chases tensor structure correlated with *excess hazard*:

  1. null Nelson-Aalen cumulative hazard H0(t) from the training (time, event);
     martingale residual  m_i = event_i - H0(t_i)  (continuous "more/less event than
     expected by t_i" target -- Therneau's residual trick that linearises Cox PLS).
  2. BTTR.train against f = m  -> supervised, survival-aware block extraction.
  3. risk head: a Cox PH model on the K block SCORES (proper partial likelihood),
     giving a calibrated risk score; fall back to the raw BTTR output if Cox fails.

Glass-box is preserved: each block is still a (time x lead) map, now a *hazard*
substrate. This is the genuine algorithm-level novelty of the follow-up (tensor
survival regression for ECG), not a tweak of inputs or an ensemble.
"""
from __future__ import annotations

import numpy as np
from lifelines import CoxPHFitter, NelsonAalenFitter
import pandas as pd

from core.bttr_model import BTTRClassifier


def martingale_residuals(time, event):
    """m_i = event_i - H0(t_i) from a null (covariate-free) Nelson-Aalen fit."""
    na = NelsonAalenFitter().fit(np.asarray(time, float), np.asarray(event, int))
    H0 = na.cumulative_hazard_at_times(np.asarray(time, float)).to_numpy().ravel()
    return np.asarray(event, float) - H0


class SurvBTTR:
    """Survival-driven BTTR with a Cox head on block scores."""

    def __init__(self, nfactor=None, cox_head=True, cox_penalizer=0.1):
        self.nfactor = nfactor
        self.cox_head = cox_head
        self.cox_penalizer = cox_penalizer
        self._b = None
        self._cox = None

    def fit(self, X, time, event):
        m = martingale_residuals(time, event)
        self._b = BTTRClassifier(nfactor=self.nfactor).fit_continuous(X, m)
        if self.cox_head:
            T = self._b.transform(X)
            df = pd.DataFrame(T, columns=[f"b{k}" for k in range(T.shape[1])])
            df["time"] = np.asarray(time, float)
            df["event"] = np.asarray(event, int)
            # guard degenerate columns
            keep = [c for c in df.columns[:-2] if df[c].std() > 1e-9]
            self._cols = keep
            try:
                self._cox = CoxPHFitter(penalizer=self.cox_penalizer).fit(
                    df[keep + ["time", "event"]], "time", "event")
            except Exception:
                self._cox = None
        return self

    def risk(self, X):
        """Higher = higher AF risk."""
        if self.cox_head and self._cox is not None:
            T = self._b.transform(X)
            df = pd.DataFrame(T, columns=[f"b{k}" for k in range(T.shape[1])])[self._cols]
            return self._cox.predict_partial_hazard(df).to_numpy().ravel()
        return self._b.decision_function(X)         # raw martingale-fit output

    def block_maps(self):
        return self._b.block_maps()
