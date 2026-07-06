"""Beat-aligned, averaged BTTN saliency (the clean 'which part of the heartbeat' figure).

For many high-risk pre-AF sinus ECGs: occlude FINE (25 ms) windows defined RELATIVE TO THE
R-PEAK across every beat in a lead, rebuild the spectrogram, measure the risk drop -> a
(lead x cardiac-cycle-phase) saliency. Average over ECGs and overlay on the mean median beat,
with P / PR / QRS / ST / T marked. Far more legible than per-strip saliency.

Run (needs the survival spectral tensor cache built by multihorizon.py):  python beat_saliency.py --source icd
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from core import config, data
from core.preprocess import build_spectral_tensor, clean, detect_rpeaks
from core.bttn import train_model, predict_risk

LEADS = config.LEAD_NAMES
PRE_MS, POST_MS, STEP_MS, WIN_MS = 350, 450, 25, 25       # beat window + fine occlusion step/width
N_ECG = 120
SEG = {"P": (-230, -130), "PR": (-130, -50), "QRS": (-50, 50), "ST": (50, 150), "T": (150, 400)}


def beat_saliency(model, stats, rec):
    fs = rec.fs
    peaks = detect_rpeaks(clean(rec.signal, fs, decimate_factor=1)[:, config.LEAD_NAMES.index("II")], fs)
    if len(peaks) < 3:
        return None, None, None
    base = predict_risk(model, build_spectral_tensor([rec], decimate_factor=2), stats)[0]
    centers = np.arange(-PRE_MS, POST_MS, STEP_MS)
    sal = np.zeros((len(LEADS), len(centers)))
    eng = np.zeros((len(LEADS), len(centers)))               # energy removed (to de-bias occlusion)
    half = int(WIN_MS / 2 / 1000 * fs)
    for l in range(len(LEADS)):
        for c, ms in enumerate(centers):
            r2 = config.Recording(signal=rec.signal.copy(), fs=fs)
            off = int(ms / 1000 * fs); e = 0.0
            for p in peaks:                                  # occlude this phase in EVERY beat
                a, b = p + off - half, p + off + half
                if 0 <= a and b <= r2.signal.shape[0]:
                    e += float(np.sum(r2.signal[a:b, l] ** 2)); r2.signal[a:b, l] = 0.0
            risk = predict_risk(model, build_spectral_tensor([r2], decimate_factor=2), stats)[0]
            sal[l, c] = base - risk; eng[l, c] = e
    # mean median beat for display (cleaned @250Hz scale not needed; use full-fs cleaned, resample to centers grid)
    sig = clean(rec.signal, fs, decimate_factor=1)
    a = int(PRE_MS / 1000 * fs)
    beats = [sig[p - a:p + int(POST_MS / 1000 * fs)] for p in peaks if p - a >= 0 and p + int(POST_MS / 1000 * fs) <= len(sig)]
    if not beats:
        return None, None, None
    mb = np.median(np.stack(beats, 0), axis=0)               # (beatlen, L)
    return sal, eng, mb


def main(source="icd"):
    coh = pd.read_csv(config.RESULTS_DIR / f"cohort_{source}_survival.csv")
    X = np.load(config.RESULTS_DIR / f"X_spectral_{source}_survival_n{len(coh)}.npy")
    time = np.where(coh.label == 1, coh.time_to_event_days, coh.fu_days).astype(float)
    event = coh.label.to_numpy(); y1 = ((event == 1) & (time <= 365)).astype(int)
    print("[train] BTTN ...")
    model, stats = train_model(X, time, event, y1, seed=0)
    risk = predict_risk(model, X, stats)
    order = [i for i in np.argsort(-risk) if event[i] == 1]

    centers = np.arange(-PRE_MS, POST_MS, STEP_MS)
    SAL = np.zeros((len(LEADS), len(centers))); ENG = np.zeros((len(LEADS), len(centers))); MB = None; n = 0
    for i in order:
        if n >= N_ECG:
            break
        try:
            rec = data.load(coh.iloc[i]["path"])
        except Exception:
            continue
        sal, eng, mb = beat_saliency(model, stats, rec)
        if sal is None:
            continue
        SAL += sal; ENG += eng
        bl = mb.shape[0]
        MB = mb if MB is None else (MB[:min(len(MB), bl)] + mb[:min(len(MB), bl)])
        n += 1
        if n % 20 == 0:
            print(f"  {n}/{N_ECG} ECGs")
    SAL /= max(n, 1); ENG /= max(n, 1); MB /= max(n, 1)
    # energy-normalized saliency: risk-drop per unit energy removed (de-biases the QRS energy)
    SAL_N = np.clip(SAL, 0, None) / (ENG + ENG.mean() * 0.1 + 1e-9)
    print(f"[done] averaged over {n} high-risk pre-AF ECGs")
    np.savez(config.RESULTS_DIR / "beat_saliency.npz", SAL=SAL, ENG=ENG, SAL_N=SAL_N, MB=MB, centers=centers)
    rows = {}
    for name, (a, b) in SEG.items():
        m = (centers >= a) & (centers < b)
        rows[name + "_raw"] = float(np.clip(SAL[:, m], 0, None).mean())
        rows[name + "_norm"] = float(SAL_N[:, m].mean())
    print("[raw seg]", {k.split('_')[0]: round(v, 4) for k, v in rows.items() if k.endswith('raw')})
    print("[ENERGY-NORM seg]", {k.split('_')[0]: round(v, 4) for k, v in rows.items() if k.endswith('norm')})
    pd.DataFrame([rows]).to_csv(config.RESULTS_DIR / "beat_saliency_segments.csv", index=False)
    SAL = SAL_N                                              # plot the de-biased version

    # figure: 12-lead grid, median beat + saliency shading
    tb = np.linspace(-PRE_MS, POST_MS, MB.shape[0])
    smax = np.clip(SAL, 0, None).max() + 1e-9
    fig, axes = plt.subplots(4, 3, figsize=(15, 9), sharex=True)
    for l, ax in enumerate(axes.T.flatten()):
        ax.plot(tb, MB[:, l], color="black", lw=1.1)
        for c, ms in enumerate(centers):
            imp = max(0, SAL[l, c]) / smax
            if imp > 0.03:
                ax.axvspan(ms - STEP_MS / 2, ms + STEP_MS / 2, color="crimson", alpha=0.6 * imp)
        ax.set_title(LEADS[l], fontsize=8, loc="left"); ax.set_yticks([])
        ax.axvline(0, color="gray", lw=0.5, ls=":")
    fig.suptitle(f"Beat-aligned BTTN saliency, energy-normalized (mean of {n} high-risk pre-AF sinus ECGs) — red = drives future-AF risk per unit signal energy", fontsize=11)
    fig.supxlabel("time relative to R-peak (ms)")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "beat_saliency.png", dpi=140)
    print("[saved] results/beat_saliency.png + beat_saliency_segments.csv")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="icd", choices=["icd", "ecg"])  # icd = gold ICD-10 (the paper); ecg = provisional
    a = ap.parse_args()
    main(a.source)
