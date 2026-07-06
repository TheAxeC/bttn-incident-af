"""Clinical-grade evaluation of the incident-AF risk scores (no gold needed).

Clinical venues (J-BHI) want more than AUROC: calibration, risk
stratification, decision-curve net benefit, and operating-point sensitivity/PPV.
Computed from the saved GOLD ICD-10 survival scores (`multihorizon_icd_spectral.npz`:
risk_fuse / risk_bttn / risk_clin / time / event). Headline = the ECG+clinical fusion
(risk_fuse). Probabilities via 2-fold cross-fitted Platt scaling (honest, no full-data
optimism).

Out: results/clin_eval.png + results/clin_eval_operating_points.csv + risk_strata.csv
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from core import config

H = 365
d = np.load(config.RESULTS_DIR / "multihorizon_icd_spectral.npz")   # gold ICD-10 survival scores
time, event = d["time"], d["event"]
# 1-yr binary label with at-risk filtering (drop censored-before-H)
lab = np.where((event == 1) & (time <= H), 1, np.where(time > H, 0, -1))
m = lab >= 0
y = lab[m].astype(int)
# headline = the ECG+clinical fusion (risk_fuse); keep ECG-only + clinical for reference
scores = {"BTTN+clin": d["risk_fuse"][m], "BTTN": d["risk_bttn"][m], "clinical": d["risk_clin"][m]}
prev = y.mean()
# Natural 1-yr AF incidence among the eligible sinus-ECG cohort BEFORE the control cap
# (cohort.py gold build: 1,855 incident cases / 41,770 AF-free controls). The modelling cohort is
# balanced to ~0.20; PPV/NNM are recalibrated back to this rate below (transportable numbers).
TRUE_PREV = 1855 / (1855 + 41770)
print(f"[cohort] n={len(y)} balanced-prevalence={prev:.3f}  true-incidence={TRUE_PREV:.4f}")


def cross_platt(s, y, seed=0):
    """2-fold cross-fitted logistic recalibration -> probabilities (no optimism)."""
    rng = np.random.RandomState(seed); idx = rng.permutation(len(y)); half = len(idx) // 2
    p = np.zeros(len(y))
    for a, b in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
        lr = LogisticRegression(max_iter=1000).fit(s[a].reshape(-1, 1), y[a])
        p[b] = lr.predict_proba(s[b].reshape(-1, 1))[:, 1]
    return p


def boot_auc(y, s, n=2000, seed=0):
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        b = rng.choice(len(y), len(y), True)
        if len(np.unique(y[b])) == 2: v.append(roc_auc_score(y[b], s[b]))
    return np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)


probs = {k: cross_platt(s, y) for k, s in scores.items()}

# ---- operating points (flag top-q% as high-risk) + risk strata ----
rows = []
for k, p in probs.items():
    a, lo, hi = boot_auc(y, p)
    for q in (0.05, 0.10, 0.20, 0.30):
        thr = np.quantile(p, 1 - q); pred = p >= thr
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        sens = tp / (tp + fn + 1e-9); spec = tn / (tn + fp + 1e-9)
        ppv = tp / (tp + fp + 1e-9); npv = tn / (tn + fn + 1e-9)
        # TRUE-INCIDENCE recalibration: sens/spec are prevalence-independent, so map PPV/NNM
        # to the natural 1-yr AF rate via Bayes (the balanced-cohort ppv above is NOT transportable).
        ppv_true = sens * TRUE_PREV / (sens * TRUE_PREV + (1 - spec) * (1 - TRUE_PREV) + 1e-12)
        nnm = (1.0 / ppv_true) if ppv_true > 0 else float("inf")           # number-needed-to-monitor
        flagged_true = sens * TRUE_PREV + (1 - spec) * (1 - TRUE_PREV)      # fraction flagged at true rate
        rows.append(dict(model=k, auroc=round(a, 3), ci=f"[{lo:.3f},{hi:.3f}]", flagged_bal=f"{q:.0%}",
                         sens=round(sens, 3), spec=round(spec, 3), ppv_bal=round(ppv, 3), npv=round(npv, 3),
                         ppv_true=round(ppv_true, 4), nnm=round(nnm, 1), flagged_true=round(flagged_true, 4),
                         brier=round(brier_score_loss(y, p), 4)))
op = pd.DataFrame(rows); op.to_csv(config.RESULTS_DIR / "clin_eval_operating_points.csv", index=False)
print(op.to_string(index=False))

# risk strata (deciles of the fusion model's prob -> observed event rate)
pb = probs["BTTN+clin"]; dec = pd.qcut(pb, 10, labels=False, duplicates="drop")
strata = pd.DataFrame({"decile": dec, "y": y}).groupby("decile").agg(
    n=("y", "size"), observed_rate=("y", "mean"), mean_pred=("y", lambda v: pb[v.index].mean()))
strata.to_csv(config.RESULTS_DIR / "risk_strata.csv")

# ---- figure: reliability + decision curve + risk strata ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
# (a) reliability
for k, p in probs.items():
    bins = np.quantile(p, np.linspace(0, 1, 11)); bins[-1] += 1e-9
    bi = np.clip(np.digitize(p, bins) - 1, 0, 9)
    mp = [p[bi == i].mean() for i in range(10)]; ob = [y[bi == i].mean() for i in range(10)]
    ax[0].plot(mp, ob, "o-", label=f"{k} (Brier {brier_score_loss(y,p):.3f})")
ax[0].plot([0, prev * 3], [0, prev * 3], "k--", lw=1); ax[0].set_title("Calibration (1-yr)")
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("observed"); ax[0].legend(fontsize=8)
# (b) decision curve (net benefit)
pts = np.linspace(0.01, 0.5, 50)
for k, p in probs.items():
    nb = [((p >= t).astype(int) @ y) / len(y) - ((p >= t).astype(int) @ (1 - y)) / len(y) * (t / (1 - t)) for t in pts]
    ax[1].plot(pts, nb, label=k)
ax[1].plot(pts, [prev - (1 - prev) * (t / (1 - t)) for t in pts], "k:", lw=1, label="treat all")
ax[1].axhline(0, color="gray", lw=1, ls="--", label="treat none")
ax[1].set_ylim(-0.02, prev * 1.1); ax[1].set_title("Decision curve (net benefit)")
ax[1].set_xlabel("threshold probability"); ax[1].legend(fontsize=8)
# (c) risk strata
ax[2].bar(strata.index, strata.observed_rate, color="crimson", alpha=0.8)
ax[2].axhline(prev, color="gray", ls="--", label=f"cohort prev {prev:.2f}")
ax[2].set_title("Observed 1-yr AF by fusion-model risk decile"); ax[2].set_xlabel("risk decile")
ax[2].set_ylabel("observed AF rate"); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "clin_eval.png", dpi=140)
print("[saved] results/clin_eval.png + clin_eval_operating_points.csv + risk_strata.csv")
top, bot = strata.observed_rate.iloc[-1], strata.observed_rate.iloc[0]
print(f"[risk gradient] top decile {top:.3f} vs bottom {bot:.3f}  ({top/max(bot,1e-3):.1f}x)")
