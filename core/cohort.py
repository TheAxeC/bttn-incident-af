"""Build the incident-AF cohort from MIMIC-IV-ECG metadata (+ ICD-10 when available).

Output: results/cohort.csv with one row per eligible patient (one index ECG each):
  subject_id, study_id, path, index_time, label, time_to_event_days, horizon_hit,
  age, gender, plus ECG-interval covariates (rr_interval, p_dur, qrs_dur, axes) for
  the fair clinical baseline. No waveforms are read here -- pure metadata.

Run:  python cohort.py            # uses config.OUTCOME_SOURCE
      python cohort.py --source ecg   # self-contained provisional outcome (no hosp module)
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd

from core import config

AF_RE = re.compile(r"atrial fib|atrial flutter|a-?fib|afib|\baf\b|fibrillation/flutter", re.I)
SINUS_RE = re.compile(r"sinus", re.I)
# rhythms that are neither clean SR nor AF -> ECG cannot serve as an AF-naive SR index
NONSINUS_RE = re.compile(r"paced|pacemaker|junctional|atrial tachycardia|svt|"
                         r"supraventricular tachycardia|wandering|ectopic atrial", re.I)


def _load_ecg_meta():
    rl = pd.read_csv(config.MIMIC_ECG_DIR / "record_list.csv",
                     usecols=["subject_id", "study_id", "ecg_time", "path"])
    mm = pd.read_csv(config.MIMIC_ECG_DIR / "machine_measurements.csv", low_memory=False)
    rep_cols = [c for c in mm.columns if c.startswith("report_")]
    mm["rep"] = mm[rep_cols].fillna("").agg(" ".join, axis=1)
    keep = ["subject_id", "study_id", "rep", "rr_interval", "p_onset", "p_end",
            "qrs_onset", "qrs_end", "t_end", "p_axis", "qrs_axis", "t_axis"]
    df = rl.merge(mm[keep], on=["subject_id", "study_id"], how="inner")
    df["ecg_time"] = pd.to_datetime(df["ecg_time"], errors="coerce")
    df = df.dropna(subset=["ecg_time"]).reset_index(drop=True)
    df["is_af"] = df["rep"].str.contains(AF_RE, na=False)
    df["is_sinus"] = df["rep"].str.contains(SINUS_RE, na=False) & ~df["is_af"]
    df["is_nonsinus"] = df["rep"].str.contains(NONSINUS_RE, na=False)
    # ECG interval covariates (P-wave duration etc. are established AF predictors)
    df["p_dur"] = df["p_end"] - df["p_onset"]
    df["qrs_dur"] = df["qrs_end"] - df["qrs_onset"]
    df["qt"] = df["t_end"] - df["qrs_onset"]
    return df


def _icd_first_af():
    """First I48 (ICD-10) / 4273x (ICD-9) onset date per subject, via hosp module."""
    import gzip
    dx = pd.read_csv(config.MIMIC_HOSP_DIR / "diagnoses_icd.csv.gz",
                     usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
                     dtype={"icd_code": str})
    adm = pd.read_csv(config.MIMIC_HOSP_DIR / "admissions.csv.gz",
                      usecols=["subject_id", "hadm_id", "admittime"])
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    code = dx["icd_code"].str.replace(".", "", regex=False).str.strip()
    is_af = ((dx["icd_version"] == 10) & code.str.startswith(config.I48_PREFIX)) | \
            ((dx["icd_version"] == 9) & code.isin(config.I48_ICD9))
    af_adm = dx.loc[is_af, ["subject_id", "hadm_id"]].merge(
        adm, on=["subject_id", "hadm_id"], how="left").dropna(subset=["admittime"])
    return af_adm.groupby("subject_id")["admittime"].min()


def _patient_followup(df):
    """Last observed contact per subject (here: last ECG). With hosp module also
    consider last discharge / death; last ECG is the conservative lower bound."""
    last_ecg = df.groupby("subject_id")["ecg_time"].max()
    fu = last_ecg.rename("fu_end").to_frame()
    try:
        pts = pd.read_csv(config.MIMIC_HOSP_DIR / "patients.csv.gz",
                          usecols=["subject_id", "anchor_age", "gender", "dod"])
        pts["dod"] = pd.to_datetime(pts["dod"], errors="coerce")
        fu = fu.merge(pts.set_index("subject_id"), left_index=True, right_index=True, how="left")
        # death extends confirmed AF-free observation
        fu["fu_end"] = fu[["fu_end", "dod"]].max(axis=1)
    except FileNotFoundError:
        fu["anchor_age"] = np.nan
        fu["gender"] = np.nan
    return fu


def build(source, survival=False):
    df = _load_ecg_meta()
    print(f"[meta] {len(df):,} ECGs / {df.subject_id.nunique():,} patients")

    # ---- outcome onset date per patient -------------------------------------
    if source == "icd":
        if not (config.MIMIC_HOSP_DIR / "diagnoses_icd.csv.gz").exists():
            sys.exit(f"hosp module missing at {config.MIMIC_HOSP_DIR} -- run the wget, "
                     f"or use --source ecg for the provisional self-contained outcome.")
        onset = _icd_first_af()
        print(f"[icd] {len(onset):,} patients with an I48 diagnosis date")
    else:  # ecg: first AF-reading ECG is the onset proxy
        onset = df.loc[df.is_af].groupby("subject_id")["ecg_time"].min()
        print(f"[ecg] {len(onset):,} patients with an AF-reading ECG")

    fu = _patient_followup(df)

    # ---- index ECG = earliest AF-naive sinus ECG per patient ----------------
    cand = df[df.is_sinus & ~df.is_nonsinus].sort_values("ecg_time")
    index = cand.groupby("subject_id", as_index=False).first()
    index = index.set_index("subject_id")
    index["onset"] = onset.reindex(index.index)
    index["fu_end"] = fu["fu_end"].reindex(index.index)
    index = index.join(fu[["anchor_age", "gender"]], how="left")

    # any AF (ECG report) at/before index disqualifies (must be AF-naive at index)
    af_first_ecg = df.loc[df.is_af].groupby("subject_id")["ecg_time"].min()
    index["af_ecg_before"] = af_first_ecg.reindex(index.index)

    t0 = index["ecg_time"]
    blank = pd.Timedelta(days=config.BLANKING_DAYS)
    onset_gap = (index["onset"] - t0).dt.days

    # prevalent/concurrent: onset (icd or ecg) at/before index+blanking -> exclude
    prevalent = index["onset"].notna() & (onset_gap <= config.BLANKING_DAYS)
    # also exclude if an AF ECG was recorded at/before index (not AF-naive)
    prevalent |= index["af_ecg_before"].notna() & (index["af_ecg_before"] <= t0 + blank)

    horizon = config.HORIZON_DAYS
    fu_days = (index["fu_end"] - t0).dt.days
    if survival:
        # survival cohort: event = incident AF at ANY future time (no horizon cap);
        # control = censored AF-free with adequate follow-up. Keep continuous time.
        positive = index["onset"].notna() & (onset_gap > config.BLANKING_DAYS)
        control = index["onset"].isna() & (fu_days >= config.SURV_MIN_FU)
    else:
        positive = index["onset"].notna() & (onset_gap > config.BLANKING_DAYS) & (onset_gap <= horizon)
        # control: never any onset, observed AF-free for >= horizon past index
        control = index["onset"].isna() & (fu_days >= config.MIN_FU_FOR_CONTROL)

    index["label"] = np.where(positive, 1, np.where(control, 0, -1))
    index["time_to_event_days"] = onset_gap
    index["fu_days"] = fu_days
    index = index[~prevalent]
    coh = index[index["label"] >= 0].copy()
    print(f"[cohort] positives={int((coh.label==1).sum()):,}  controls={int((coh.label==0).sum()):,}  "
          f"(excluded prevalent={int(prevalent.sum()):,})")

    # ---- patient-level subsample + fold assignment --------------------------
    rng = np.random.RandomState(config.RANDOM_STATE)
    pos = coh[coh.label == 1]; neg = coh[coh.label == 0]
    if config.MAX_POS and len(pos) > config.MAX_POS:
        pos = pos.sample(config.MAX_POS, random_state=rng)
    if config.MAX_NEG and len(neg) > config.MAX_NEG:
        neg = neg.sample(config.MAX_NEG, random_state=rng)
    coh = pd.concat([pos, neg]).sample(frac=1, random_state=rng)
    coh["fold"] = (np.arange(len(coh)) % config.N_FOLDS)
    print(f"[final] {len(coh):,} patients  ({int((coh.label==1).sum()):,} pos / "
          f"{int((coh.label==0).sum()):,} neg)  prevalence={coh.label.mean():.3f}")

    out_cols = ["study_id", "path", "ecg_time", "label", "time_to_event_days", "fu_days",
                "anchor_age", "gender", "rr_interval", "p_dur", "qrs_dur", "qt",
                "p_axis", "qrs_axis", "t_axis", "fold"]
    coh = coh.reset_index()[["subject_id"] + out_cols]
    coh = coh.rename(columns={"ecg_time": "index_time"})
    config.RESULTS_DIR.mkdir(exist_ok=True)
    fn = config.RESULTS_DIR / (f"cohort_{source}_survival.csv" if survival
                               else f"cohort_{source}_h{horizon}.csv")
    coh.to_csv(fn, index=False)
    print(f"[saved] {fn}")
    return coh


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=config.OUTCOME_SOURCE, choices=["icd", "ecg"])
    ap.add_argument("--horizon", type=int, default=config.HORIZON_DAYS)
    ap.add_argument("--blanking", type=int, default=config.BLANKING_DAYS,
                    help="days after index within which AF counts as concurrent and is EXCLUDED. "
                         "Default 30 = incident PREDICTION. Set 0 for Attia-style near-term "
                         "identification (includes concurrent/paroxysmal AF) -- a different, easier task.")
    ap.add_argument("--survival", action="store_true")
    a = ap.parse_args()
    config.HORIZON_DAYS = a.horizon
    config.BLANKING_DAYS = a.blanking
    config.MIN_FU_FOR_CONTROL = a.horizon
    build(a.source, survival=a.survival)
