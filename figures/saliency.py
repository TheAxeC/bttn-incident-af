"""Time-domain saliency on the raw ECG (the 'which areas matter' figure).

The substrate map (interpret_bttn) lives in the freq x lead domain. This projects the
BTTN's attention back onto the RAW 12-lead waveform: occlude each (lead x time-window) of
the signal, rebuild the spectrogram, measure the drop in predicted risk -> a (lead x time)
saliency overlaid as shading on the ECG trace. Two outputs:
  (A) one high-risk patient's 12-lead ECG, saliency-shaded  -> 'which areas are important'
  (B) saliency aggregated across cases by time-before-AF-onset -> 'which changes over time
      are important' (longitudinal)

Run (local; needs the MIMIC zip + the cached survival spectral tensor):
  python saliency.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from core import config, data
from core.preprocess import build_spectral_tensor, clean
from core.bttn import train_model, predict_risk

LEADS = config.LEAD_NAMES
WIN_S = 0.4          # occlusion window (s)
FS_RAW = 250         # display sampling (after decimate 2 from 500)


def saliency_one(model, stats, rec, win_s=WIN_S):
    """(L, nwin) risk-drop saliency + the cleaned display signal (T,L)."""
    sig = clean(rec.signal, rec.fs, decimate_factor=2)              # (T, L) @250Hz, what we display
    base = predict_risk(model, build_spectral_tensor([rec], decimate_factor=2), stats)[0]
    T = sig.shape[0]; w = int(win_s * FS_RAW); nwin = T // w
    sal = np.zeros((len(LEADS), nwin))
    for l in range(len(LEADS)):
        for k in range(nwin):
            r2 = config.Recording(signal=rec.signal.copy(), fs=rec.fs)
            a = int(k * win_s * rec.fs); b = int((k + 1) * win_s * rec.fs)
            r2.signal[a:b, l] = 0.0                                 # occlude raw window in this lead
            risk = predict_risk(model, build_spectral_tensor([r2], decimate_factor=2), stats)[0]
            sal[l, k] = base - risk
    return sig, sal, base


def overlay(sig, sal, fname, title, seg_s=3.0):
    """12-lead grid, saliency-shaded, first seg_s seconds."""
    n = int(seg_s * FS_RAW); nwin = sal.shape[1]; w = sig.shape[0] // nwin
    smax = np.abs(sal).max() + 1e-9
    fig, axes = plt.subplots(4, 3, figsize=(15, 9), sharex=True)
    for l, ax in enumerate(axes.T.flatten()):
        x = np.arange(min(n, sig.shape[0])) / FS_RAW
        ax.plot(x, sig[:len(x), l], color="black", lw=0.7)
        for k in range(nwin):
            a = k * w
            if a >= len(x):
                break
            imp = max(0, sal[l, k]) / smax                          # positive saliency = risk-driving
            if imp > 0.02:
                ax.axvspan(a / FS_RAW, min((a + w) / FS_RAW, x[-1]), color="crimson", alpha=0.55 * imp)
        ax.set_title(LEADS[l], fontsize=8, loc="left"); ax.set_yticks([])
    fig.suptitle(title, fontsize=13); fig.supxlabel("time (s)")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / fname, dpi=140); plt.close(fig)
    print(f"[saved] results/{fname}")


def main():
    coh = pd.read_csv(config.RESULTS_DIR / "cohort_ecg_survival.csv")
    X = np.load(config.RESULTS_DIR / f"X_spectral_ecg_survival_n{len(coh)}.npy")
    time = np.where(coh.label == 1, coh.time_to_event_days, coh.fu_days).astype(float)
    event = coh.label.to_numpy(); y1 = ((event == 1) & (time <= 365)).astype(int)
    print("[train] BTTN ...")
    model, stats = train_model(X, time, event, y1, seed=0)
    risk = predict_risk(model, X, stats)

    # (A) highest-risk event patient with a loadable waveform
    order = np.argsort(-risk)
    for i in order:
        if event[i] == 1:
            rec = data.load(coh.iloc[i]["path"])
            if rec is not None:
                sig, sal, base = saliency_one(model, stats, rec)
                overlay(sig, sal, "saliency_ecg.png",
                        f"BTTN saliency on a high-risk pre-AF sinus ECG (risk={base:.2f}): red = drives AF risk")
                break

    # (B) longitudinal: mean per-lead saliency by time-to-onset, across top-risk case patients
    from core import cohort as C
    df = C._load_ecg_meta(); onset = df.loc[df.is_af].groupby("subject_id")["ecg_time"].min()
    sr = df[df.is_sinus & ~df.is_nonsinus]
    BINS = [(0, 180), (180, 365), (365, 730), (730, 1460)]
    agg = np.zeros((len(LEADS), len(BINS))); cnt = np.zeros(len(BINS))
    done = 0
    for sid in onset.index:
        if done >= 150:
            break
        recs = sr[sr.subject_id == sid].sort_values("ecg_time")
        recs = recs[recs.ecg_time < onset[sid]]
        if len(recs) < 2:
            continue
        for _, row in recs.tail(4).iterrows():
            tto = (onset[sid] - row.ecg_time).days
            bi = next((j for j, (a, b) in enumerate(BINS) if a <= tto < b), None)
            if bi is None:
                continue
            try:
                rec = data.load(row.path)
            except Exception:
                continue
            _, sal, _ = saliency_one(model, stats, rec)
            agg[:, bi] += np.clip(sal, 0, None).sum(1); cnt[bi] += 1
        done += 1
    agg /= np.maximum(cnt, 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(agg, aspect="auto", cmap="magma",
                   extent=[0, len(BINS), 0, len(LEADS)])
    ax.set_xticks(np.arange(len(BINS)) + 0.5); ax.set_xticklabels(["0-0.5y", "0.5-1y", "1-2y", "2-4y"])
    ax.set_yticks(np.arange(len(LEADS)) + 0.5); ax.set_yticklabels(LEADS[::-1])
    ax.set_xlabel("time before AF onset"); ax.set_title("Per-lead saliency vs time-to-onset (n=%d case patients)" % int(cnt.sum()))
    fig.colorbar(im, label="mean per-lead saliency"); fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "saliency_longitudinal.png", dpi=140)
    print("[saved] results/saliency_longitudinal.png  (per-bin n:", cnt.astype(int).tolist(), ")")


if __name__ == "__main__":
    main()
