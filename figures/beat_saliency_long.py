"""Longitudinal beat-aligned, ENERGY-NORMALIZED saliency (the de-biased 'changes over time').

Two panels, both using the same energy-normalized occlusion as beat_saliency.py:
  (A) heatmap: cardiac-phase (P/PR/QRS/ST/T) x time-before-AF-onset - how each phase's
      per-unit-energy importance evolves as AF approaches (mean over leads & case patients).
  (B) a few example case patients: their serial sinus ECGs (median beat) with de-biased
      saliency shading, columns = time-before-onset.

Reuses beat_saliency() from beat_saliency.py. Run AFTER beat_saliency.py (avoids CPU thrash).
  python beat_saliency_long.py --source icd
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from core import config, data
from core import cohort as C
from core.bttn import train_model, predict_risk
from beat_saliency import beat_saliency, SEG, PRE_MS, POST_MS, STEP_MS, LEADS

BINS = [(0, 180), (180, 365), (365, 730), (730, 1460)]
BIN_LBL = ["0-0.5y", "0.5-1y", "1-2y", "2-4y"]
N_PAT = 80           # case patients for the heatmap
centers = np.arange(-PRE_MS, POST_MS, STEP_MS)


def seg_norm(sal, eng):
    """(L, nwin) raw sal+eng -> (n_seg,) energy-normalized importance, mean over leads."""
    saln = np.clip(sal, 0, None) / (eng + eng.mean() * 0.1 + 1e-9)
    out = []
    for a, b in SEG.values():
        m = (centers >= a) & (centers < b)
        out.append(saln[:, m].mean())
    return np.array(out)


def main(source="icd"):
    coh = pd.read_csv(config.RESULTS_DIR / f"cohort_{source}_survival.csv")
    X = np.load(config.RESULTS_DIR / f"X_spectral_{source}_survival_n{len(coh)}.npy")
    time = np.where(coh.label == 1, coh.time_to_event_days, coh.fu_days).astype(float)
    event = coh.label.to_numpy(); y1 = ((event == 1) & (time <= 365)).astype(int)
    print("[train] BTTN ...")
    model, stats = train_model(X, time, event, y1, seed=0)

    df = C._load_ecg_meta(); onset = df.loc[df.is_af].groupby("subject_id")["ecg_time"].min()
    sr = df[df.is_sinus & ~df.is_nonsinus]
    # rank case patients by index-ECG risk (so "high-risk" exemplars are clean)
    cand = [sid for sid in onset.index if sid in set(sr.subject_id)]

    # ---- (A) heatmap: segment x time-to-onset ----
    H = np.zeros((len(SEG), len(BINS))); cnt = np.zeros(len(BINS)); done = 0
    for sid in cand:
        if done >= N_PAT:
            break
        recs = sr[sr.subject_id == sid].sort_values("ecg_time")
        recs = recs[recs.ecg_time < onset[sid]]
        if len(recs) < 2:
            continue
        used = False
        for _, row in recs.tail(4).iterrows():
            tto = (onset[sid] - row.ecg_time).days
            bi = next((j for j, (a, b) in enumerate(BINS) if a <= tto < b), None)
            if bi is None:
                continue
            try:
                rec = data.load(row.path)
            except Exception:
                continue
            sal, eng, _ = beat_saliency(model, stats, rec)
            if sal is None:
                continue
            H[:, bi] += seg_norm(sal, eng); cnt[bi] += 1; used = True
        done += 1 if used else 0
    H /= np.maximum(cnt, 1)
    np.savez(config.RESULTS_DIR / "beat_saliency_long.npz", H=H, cnt=cnt)
    print("[heatmap] per-bin n:", cnt.astype(int).tolist())

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(H, aspect="auto", cmap="magma", origin="lower")
    ax.set_xticks(range(len(BINS))); ax.set_xticklabels(BIN_LBL)
    ax.set_yticks(range(len(SEG))); ax.set_yticklabels(list(SEG.keys()))
    ax.set_xlabel("time before AF onset"); ax.set_ylabel("cardiac phase")
    ax.set_title(f"Energy-normalized saliency by cardiac phase vs time-to-onset (n={int(cnt.sum())} ECGs)")
    fig.colorbar(im, label="risk-drop per unit energy"); fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "beat_saliency_long_heatmap.png", dpi=140)
    print("[saved] results/beat_saliency_long_heatmap.png")

    # ---- (B) example serial ECGs (3 patients, lead II) with de-biased shading ----
    expat = []
    for sid in cand:
        recs = sr[sr.subject_id == sid].sort_values("ecg_time")
        recs = recs[recs.ecg_time < onset[sid]]
        if len(recs) >= 3:
            expat.append((sid, recs))
        if len(expat) >= 3:
            break
    lead = LEADS.index("II")
    fig, axes = plt.subplots(len(expat), 3, figsize=(14, 3 * len(expat)), squeeze=False)
    for r, (sid, recs) in enumerate(expat):
        picks = recs.iloc[np.linspace(0, len(recs) - 1, 3).astype(int)]
        for cax, (_, row) in zip(axes[r], picks.iterrows()):
            try:
                rec = data.load(row.path)
            except Exception:
                continue
            sal, eng, mb = beat_saliency(model, stats, rec)
            if sal is None:
                continue
            saln = np.clip(sal[lead], 0, None) / (eng[lead] + eng.mean() * 0.1 + 1e-9)
            tb = np.linspace(-PRE_MS, POST_MS, mb.shape[0]); smax = saln.max() + 1e-9
            cax.plot(tb, mb[:, lead], "k", lw=1.1)
            for c, ms in enumerate(centers):
                if saln[c] / smax > 0.05:
                    cax.axvspan(ms - STEP_MS / 2, ms + STEP_MS / 2, color="crimson", alpha=0.6 * saln[c] / smax)
            tto = (onset[sid] - row.ecg_time).days
            cax.set_title(f"pt {sid}  ({tto}d before onset)", fontsize=8); cax.set_yticks([])
    fig.suptitle("Example pre-AF trajectories: lead II, energy-normalized BTTN saliency (red drives risk)")
    fig.supxlabel("ms relative to R-peak"); fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "beat_saliency_long_examples.png", dpi=140)
    print("[saved] results/beat_saliency_long_examples.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="icd", choices=["icd", "ecg"])  # icd = gold ICD-10 (the paper); ecg = provisional
    a = ap.parse_args()
    main(a.source)
