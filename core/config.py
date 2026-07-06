"""Configuration for the MIMIC-IV-ECG incident-AF prediction study (Dries follow-up).

Distinct from the PTB-XL paper: that one does AF-in-SR *identification* (concurrent
paroxysmal AF, Attia 31-day window, n~42-55) because PTB-XL has no dated outcomes.
Here we have MIMIC-IV-ECG (800k ECGs / 161k patients) linked to ICD-10 I48 diagnosis
DATES -> genuine *future-onset* incident-AF prediction from a sinus-rhythm ECG.

Design (one ECG per patient, patient-level, leakage-free):
  index ECG  = patient's EARLIEST sinus-rhythm ECG that is AF-naive (no AF on ECG and
               no I48 diagnosis at/before that ECG, within BLANKING_DAYS after either).
  positive   = first I48 onset in (index + BLANKING_DAYS, index + HORIZON_DAYS].
  control    = no I48 ever, AND observed AF-free for >= HORIZON_DAYS past index.
  excluded   = prevalent/concurrent AF (I48 at/before index+blanking); inadequate
               follow-up controls (cannot confirm AF-free through the horizon).

The model is BTTN (Block-Term Tensor Network): a learnable, nonlinear, survival
generalization of block-term tensor regression on an alignment-free spectral tensor
(frequency x time x lead). The contribution is the interpretable-factor +
measured-faithfulness framework at parity with a black-box CNN, instantiated by the
block-term layer, plus the glass-box incident-AF substrate map. The vendored linear
BTTR (bttrlib) is the baseline that shows why the nonlinear generalization is needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths  (data under ~/data; results live inside code/results - see ../../README.md)    #
# --------------------------------------------------------------------------- #
MIMIC_ECG_DIR = Path("~/data/mimic-ecg").expanduser()      # record_list.csv, machine_measurements.csv
# ECG waveform archive: on the HPC it lives at ~/data/archive/ (34 GB matched subset); on Axel's Mac it is
# ~/Downloads/. Default to the HPC location (where the runs happen); override with AFNET_ECG_ZIP elsewhere.
MIMIC_ECG_ZIP = Path(os.environ.get("AFNET_ECG_ZIP", "~/data/archive/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip")).expanduser()
# hosp module: PhysioNet-canonical layout is ~/data/mimic-iv/hosp (full module lives there on the HPC);
# override with AFNET_HOSP_DIR if the three files (diagnoses_icd/admissions/patients .csv.gz) live elsewhere.
MIMIC_HOSP_DIR = Path(os.environ.get("AFNET_HOSP_DIR", "~/data/mimic-iv/hosp")).expanduser()
WAVE_CACHE = Path("~/data/mimic-ecg/files").expanduser()    # per-record .hea/.dat extracted on demand
# config.py lives at code/core/ -> parents[1] is code/; results/ lives INSIDE code/ (self-contained).
RESULTS_DIR = Path(os.environ.get("AFNET_RESULTS", Path(__file__).resolve().parents[1] / "results"))

# --------------------------------------------------------------------------- #
# Outcome definition                                                          #
# --------------------------------------------------------------------------- #
OUTCOME_SOURCE = "icd"          # "icd" (gold; needs hosp module) or "ecg" (later ECG reads AF; self-contained)
HORIZON_DAYS = 365              # primary prediction horizon (1 yr); also evaluate 2 yr / 5 yr
# 120 = "90-day onset (post-blank)": first AF in (index+BLANKING_DAYS, index+120d];
# a short-onset INCIDENT-prediction horizon read off the SAME survival curve (not detection).
HORIZONS_EVAL = [120, 365, 730, 1825]
BLANKING_DAYS = 30              # AF inside this window after index = concurrent, NOT incident -> excluded
I48_PREFIX = "I48"              # ICD-10 atrial fibrillation and flutter
I48_ICD9 = {"42731", "42732"}   # ICD-9 AF (4273x) for older admissions

# --------------------------------------------------------------------------- #
# Signal / leads (MIMIC-IV-ECG: 12-lead, 500 Hz, 10 s -- same layout as PTB-XL) #
# --------------------------------------------------------------------------- #
PTBXL_FS = 500                  # name kept so vendored preprocess.py works unchanged
LOWRES_FS = 100
DECIMATE_FACTOR = 5             # 500 -> 100 Hz; keeps P/f-wave band
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
RHYTHM_LEAD = "II"
WAVELET = "sym4"; WAVELET_LEVEL = 3; WAVELET_THRESHOLD = 7.0

# --------------------------------------------------------------------------- #
# Cohort caps (for tractable compute; sampled patient-level, logged if applied) #
# --------------------------------------------------------------------------- #
MAX_POS = 4000                  # cap incident-AF cases (None = all ~ up to horizon)
MAX_NEG = 16000                 # cap controls (deep models want scale; bench picks ratio)
MIN_FU_FOR_CONTROL = HORIZON_DAYS
SURV_MIN_FU = 365               # survival-cohort controls need >= this censoring follow-up

# --------------------------------------------------------------------------- #
# BTTR / tensorisation                                                        #
# --------------------------------------------------------------------------- #
BTTR_NFACTOR = 8
BTTR_ENFORCE_RANK1 = True
BTTR_SNRS = range(20, 41, 5)
BTTR_RATIOS = (95.0, 97.0, 99.0)
TENSORISATION = "median_beat"   # morphology-only; SR index ECG has no AF irregularity to exploit

# --------------------------------------------------------------------------- #
# Reproducibility / CV                                                        #
# --------------------------------------------------------------------------- #
SEEDS = list(range(5))
RANDOM_STATE = 0
N_FOLDS = 5                      # patient-grouped (one ECG/patient here, so simple stratified)


@dataclass
class Recording:
    signal: "object"            # (T, L) float
    fs: int
