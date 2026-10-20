"""External validation on PTB-XL (independent dataset / country / device).

PTB-XL has NO follow-up, so it cannot validate incident *prediction*. What it CAN validate:
  (1) METHOD TRANSFER - the BTTN spectral pipeline runs on an independent dataset and
      discriminates AF (SR-vs-AF detection), i.e. the approach isn't MIMIC-specific;
  (2) INTERPRETABILITY TRANSFER - the glass-box occlusion substrate highlights the SAME
      frequency bands (f-wave / P-wave) and leads (V1) as on MIMIC, so the read-outs are not
      a dataset artifact.

Honest scope: this is detection / AF-in-SR on PTB-XL, NOT external incident-AF prediction
(that needs a dated-outcome cohort). Output: results/external_ptbxl_*.csv + .png.

Run:  python external_ptbxl.py --task detect      # SR-vs-AF, balanced (built from raw metadata)
      python external_ptbxl.py --task afsr        # AF-in-SR identification (built from raw metadata)
"""
from __future__ import annotations
import argparse, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, wfdb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from core import config
from core.preprocess import build_spectral_tensor
from core.bttn import train_model_binary, predict_risk, DEV

PTB = Path("~/data/ptb-xl").expanduser()
FMAX = 40.0; AF = ("AFIB", "AFLT")


def build_cohort(task, seed=0):
    """Build the PTB-XL evaluation cohort DIRECTLY from the raw ptbxl_database.csv (self-contained,
    no pre-made cohort file). Rhythm labels come from the scp_codes dict-string: AF = AFIB or AFLT,
    SR = the sinus-rhythm key. Records are read at 100 Hz (filename_lr, full coverage of both classes)."""
    db = pd.read_csv(PTB / "ptbxl_database.csv")
    scp = db["scp_codes"].astype(str)
    db["is_af"] = scp.str.contains("AFIB|AFLT")
    db["is_sr"] = scp.str.contains("'SR'") & ~db["is_af"]      # sinus rhythm, and not co-coded AF
    db["date"] = pd.to_datetime(db["recording_date"], errors="coerce")
    rng = np.random.RandomState(seed)
    if task == "detect":                                       # SR vs AF, class-balanced
        af = db[db.is_af]; sr = db[db.is_sr]
        neg = sr.sample(n=min(len(af), len(sr)), random_state=seed)
        df = pd.concat([af.assign(label=1), neg.assign(label=0)]).reset_index(drop=True)
    else:                                                      # afsr: a sinus index ECG is positive if the
        af_dates = db[db.is_af].groupby("patient_id")["date"].apply(list)   # SAME patient has AF within +/-31 days
        df = db[db.is_sr].copy()
        df["label"] = df.apply(lambda r: int(any(
            pd.notnull(d) and abs((d - r["date"]).days) <= 31 for d in af_dates.get(r["patient_id"], []))), axis=1)
        df = df.reset_index(drop=True)
    df = df[df["filename_lr"].apply(lambda p: (PTB / (p + ".hea")).exists())].reset_index(drop=True)
    print(f"[cohort:{task}] n={len(df)} pos={int(df.label.sum())} (built from raw ptbxl_database.csv)")
    return df


def load(task):
    df = build_cohort(task)
    cache = config.RESULTS_DIR / f"X_ptbxl_{task}_n{len(df)}.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        recs = []
        for fn in df["filename_lr"]:
            r = wfdb.rdrecord(str(PTB / fn))
            recs.append(config.Recording(signal=np.nan_to_num(np.asarray(r.p_signal, float)), fs=int(r.fs)))
        X = build_spectral_tensor(recs, decimate_factor=1)   # records100 already at 100 Hz
        np.save(cache, X)
    return df, X


def main(task, seeds):
    df, X = load(task); y = df["label"].to_numpy()
    print(f"[ptbxl:{task}] n={len(y)} pos={int(y.sum())} X={X.shape}")
    # (1) method transfer: 5-fold CV AUROC
    rng = np.random.RandomState(0); fold = np.empty(len(y), int)
    for c in (0, 1):
        idx = np.where(y == c)[0]; rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % 5
    sc = np.zeros(len(y))
    for f in range(5):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        model, stats = train_model_binary(X[tr], y[tr], seed=0)
        sc[te] = predict_risk(model, X[te], stats)
    auc = roc_auc_score(y, sc)
    print(f"[method transfer] PTB-XL {task} AUROC = {auc:.3f}")

    # (2) interpretability transfer: train on all, occlude each freq band / lead
    model, stats = train_model_binary(X, y, seed=0)
    base = roc_auc_score(y, predict_risk(model, X, stats))
    F, L = X.shape[1], X.shape[3]; freqs = np.linspace(0, FMAX, F)
    fimp = [base - roc_auc_score(y, predict_risk(model, (lambda Z: (Z.__setitem__((slice(None), fi), 0) or Z))(X.copy()), stats)) for fi in range(F)]
    limp = [base - roc_auc_score(y, predict_risk(model, (lambda Z: (Z.__setitem__((slice(None), slice(None), slice(None), li), 0) or Z))(X.copy()), stats)) for li in range(L)]
    impf = pd.DataFrame({"freq_hz": np.round(freqs, 1), "importance": fimp})
    impl = pd.DataFrame({"lead": config.LEAD_NAMES, "importance": limp})
    impf.to_csv(config.RESULTS_DIR / f"external_ptbxl_{task}_freq.csv", index=False)
    impl.to_csv(config.RESULTS_DIR / f"external_ptbxl_{task}_lead.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(freqs, fimp, width=freqs[1] - freqs[0]); ax[0].set_title(f"PTB-XL {task} freq importance (AUROC {base:.3f})")
    ax[0].set_xlabel("Hz"); ax[0].set_ylabel("AUROC drop")
    ax[1].barh(config.LEAD_NAMES, limp); ax[1].invert_yaxis(); ax[1].set_title("lead importance")
    fig.suptitle("External validation on PTB-XL: does the glass-box substrate reproduce MIMIC?")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / f"external_ptbxl_{task}.png", dpi=140)
    print("top freq:", impf.sort_values("importance", ascending=False).head(4).to_dict("records"))
    print("top leads:", impl.sort_values("importance", ascending=False).head(4).to_dict("records"))
    print(f"[saved] results/external_ptbxl_{task}.* (compare to MIMIC: f-wave 8Hz+P-wave 2Hz, V1)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="detect", choices=["detect", "afsr"]); ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    main(a.task, a.seeds)
