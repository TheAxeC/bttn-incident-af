"""External validation on CODE-15 (Brazil, ~345k 12-lead ECGs) - runs on the HPC.

A large, out-of-distribution, free external cohort (different country / device / population).
Big enough (4,237 AF-in-SR positives, 7,033 AF) to externally validate BOTH discrimination
and the glass-box substrate map for the SINUS-RHYTHM task - which PTB-XL (42 positives) could
not. CODE-15 has no per-exam dates, so no incident prediction; validates detection + AF-in-SR
identification + the substrate map.

Tasks:
  detect : SR-vs-AF (AF column), one exam/patient.
  afsr   : AF-in-SR identification - a non-AF exam is positive if the same patient has an AF
           exam elsewhere; negative = patient never has AF. One exam/patient (no leakage).

Data: ~/data/code15/{exams.csv, exams_part*.hdf5} (tracings (n,4096,12) @400Hz, leads = MIMIC order).
Out: results/external_code15_<task>_{freq,lead}.csv + .png.  Run via job.slurm on HPC.
"""
from __future__ import annotations
import argparse, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from core import config
from core.preprocess import build_spectral_tensor
from core.bttn import train_model_binary, predict_risk

C15 = Path("~/data/code15").expanduser()
FMAX = 40.0; MAXPOS, MAXNEG = 6000, 6000


def build_cohort(task, seed=0):
    d = pd.read_csv(C15 / "exams.csv"); rng = np.random.RandomState(seed)
    afpat = set(d.loc[d["AF"] == True, "patient_id"])
    if task == "detect":
        pos = d[d["AF"] == True].drop_duplicates("patient_id")
        neg = d[(d["AF"] == False) & (~d["patient_id"].isin(afpat))].drop_duplicates("patient_id")
    else:  # afsr
        pos = d[(d["AF"] == False) & (d["patient_id"].isin(afpat))].drop_duplicates("patient_id")
        neg = d[(d["AF"] == False) & (~d["patient_id"].isin(afpat))].drop_duplicates("patient_id")
    pos = pos.sample(min(len(pos), MAXPOS), random_state=rng)
    neg = neg.sample(min(len(neg), MAXNEG), random_state=rng)
    coh = pd.concat([pos.assign(label=1), neg.assign(label=0)]).sample(frac=1, random_state=rng).reset_index(drop=True)
    print(f"[cohort:{task}] n={len(coh)} pos={int(coh.label.sum())}")
    return coh


def load_X(coh):
    cache = config.RESULTS_DIR / f"X_code15_{len(coh)}.npy"
    recs = [None] * len(coh)
    eid2row = {int(e): i for i, e in enumerate(coh["exam_id"])}
    for part, grp in coh.groupby("trace_file"):
        with h5py.File(C15 / part, "r") as f:
            idx = {int(e): j for j, e in enumerate(f["exam_id"][:])}
            for e in grp["exam_id"]:
                j = idx.get(int(e))
                if j is not None:
                    recs[eid2row[int(e)]] = config.Recording(signal=np.nan_to_num(np.asarray(f["tracings"][j], float)), fs=400)
    keep = [i for i, r in enumerate(recs) if r is not None]
    X = build_spectral_tensor([recs[i] for i in keep], decimate_factor=1)
    return X, keep


def occl(model, X, y, stats, axis_setter, n):
    base = roc_auc_score(y, predict_risk(model, X, stats))
    out = []
    for i in range(n):
        Z = X.copy(); axis_setter(Z, i)
        out.append(base - roc_auc_score(y, predict_risk(model, Z, stats)))
    return base, out


def main(task, seeds):
    coh = build_cohort(task)
    X, keep = load_X(coh); coh = coh.iloc[keep].reset_index(drop=True); y = coh["label"].to_numpy()
    print(f"[X] {X.shape}")
    # (1) method transfer: 5-fold patient-disjoint (one exam/patient already) CV
    rng = np.random.RandomState(0); fold = np.empty(len(y), int)
    for c in (0, 1):
        ix = np.where(y == c)[0]; rng.shuffle(ix); fold[ix] = np.arange(len(ix)) % 5
    sc = np.zeros(len(y))
    for f in range(5):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        m, st = train_model_binary(X[tr], y[tr], seed=0)
        sc[te] = predict_risk(m, X[te], st)
    auc = roc_auc_score(y, sc); print(f"[method transfer] CODE-15 {task} AUROC = {auc:.3f}")
    # (2) interpretability transfer: train on all, occlude freq bands / leads
    m, st = train_model_binary(X, y, seed=0)
    F, L = X.shape[1], X.shape[3]; freqs = np.linspace(0, FMAX, F)
    base, fimp = occl(m, X, y, st, lambda Z, i: Z.__setitem__((slice(None), i), 0), F)
    _, limp = occl(m, X, y, st, lambda Z, i: Z.__setitem__((slice(None), slice(None), slice(None), i), 0), L)
    pd.DataFrame({"freq_hz": np.round(freqs, 1), "importance": fimp}).to_csv(config.RESULTS_DIR / f"external_code15_{task}_freq.csv", index=False)
    pd.DataFrame({"lead": config.LEAD_NAMES, "importance": limp}).to_csv(config.RESULTS_DIR / f"external_code15_{task}_lead.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(freqs, fimp, width=freqs[1] - freqs[0]); ax[0].set_title(f"CODE-15 {task} freq importance (AUROC {auc:.3f}, n={len(y)})")
    ax[0].set_xlabel("Hz"); ax[1].barh(config.LEAD_NAMES, limp); ax[1].invert_yaxis(); ax[1].set_title("lead importance")
    fig.suptitle("External validation on CODE-15 — does the glass-box substrate reproduce MIMIC?")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / f"external_code15_{task}.png", dpi=140)
    impf = pd.DataFrame({"hz": np.round(freqs, 1), "imp": fimp}).sort_values("imp", ascending=False)
    impl = pd.DataFrame({"lead": config.LEAD_NAMES, "imp": limp}).sort_values("imp", ascending=False)
    print("top freq:", impf.head(4).to_dict("records"))
    print("top leads:", impl.head(4).to_dict("records"))
    print(f"[saved] results/external_code15_{task}.*  (MIMIC ref: f-wave 8Hz + P-wave 2Hz, V1)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="afsr", choices=["detect", "afsr"]); ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    main(a.task, a.seeds)
