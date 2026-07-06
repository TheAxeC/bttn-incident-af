"""Cohort accounting + differential follow-up.

LIGHT metadata/pandas job (no waveforms). Reuses cohort.py's metadata logic so the numbers
align exactly with the cohorts bench.py/multihorizon.py consume. For BOTH the binary (incident,
horizon-capped) and survival (continuous time-to-AF) designs it produces:

  - a CONSORT-style flow: eligible N -> exclusions by each criterion -> final N patients;
  - incident events at 1 / 2 / 5 yr; controls; explicit neg:pos ratio;
  - censoring distribution (follow-up quantiles for controls); median follow-up;
  - DIFFERENTIAL FOLLOW-UP: fraction of "controls" with ANY later ECG vs none, and a
    sensitivity count restricting controls to those with a documented later AF-free ECG.

Outputs: results/cohort_accounting.csv (one row per cohort/metric) + results/cohort_flow.txt.

Run (LOCAL, metadata only):  python cohort_accounting.py --source ecg
"""
from __future__ import annotations
import argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from core import config
from core.cohort import _load_ecg_meta, _icd_first_af, _patient_followup


def _build_index(df, source):
    """Replicate cohort.build's index-ECG selection + onset/follow-up join (metadata only).
    Returns (index df keyed by subject_id, prevalent boolean mask, onset series)."""
    if source == "icd":
        onset = _icd_first_af()
    else:
        onset = df.loc[df.is_af].groupby("subject_id")["ecg_time"].min()
    fu = _patient_followup(df)
    cand = df[df.is_sinus & ~df.is_nonsinus].sort_values("ecg_time")
    index = cand.groupby("subject_id", as_index=False).first().set_index("subject_id")
    index["onset"] = onset.reindex(index.index)
    index["fu_end"] = fu["fu_end"].reindex(index.index)
    index = index.join(fu[["anchor_age", "gender"]], how="left")
    af_first_ecg = df.loc[df.is_af].groupby("subject_id")["ecg_time"].min()
    index["af_ecg_before"] = af_first_ecg.reindex(index.index)
    t0 = index["ecg_time"]
    blank = pd.Timedelta(days=config.BLANKING_DAYS)
    index["onset_gap"] = (index["onset"] - t0).dt.days
    index["fu_days"] = (index["fu_end"] - t0).dt.days
    prevalent = (index["onset"].notna() & (index["onset_gap"] <= config.BLANKING_DAYS)) | \
                (index["af_ecg_before"].notna() & (index["af_ecg_before"] <= t0 + blank))
    return index, prevalent, onset


def _later_ecg_stats(df, index, control_ids):
    """For control patients: do they have ANY ECG strictly after their index ECG, and is the
    latest one AF-free (differential-follow-up sensitivity)."""
    t0 = index["ecg_time"]
    # later ECGs per subject (after the index time)
    sub = df[df.subject_id.isin(control_ids)].copy()
    sub = sub.merge(t0.rename("t0"), left_on="subject_id", right_index=True, how="left")
    later = sub[sub.ecg_time > sub.t0]
    has_later = later.subject_id.unique()
    # of those, the latest later ECG is AF-free (not an AF reading)
    latest = later.sort_values("ecg_time").groupby("subject_id").last()
    later_affree = latest.index[~latest["is_af"].astype(bool)]
    return set(has_later), set(later_affree)


def _accounting(df, source, survival):
    index, prevalent, _onset = _build_index(df, source)
    eligible_sinus = len(index)                 # patients with an AF-naive-candidate sinus index ECG
    n_prevalent = int(prevalent.sum())
    horizon = config.HORIZON_DAYS
    og = index["onset_gap"]; fud = index["fu_days"]

    if survival:
        positive = index["onset"].notna() & (og > config.BLANKING_DAYS)
        control = index["onset"].isna() & (fud >= config.SURV_MIN_FU)
        excl_short_fu = int((index["onset"].isna() & (fud < config.SURV_MIN_FU)).sum())
        min_fu = config.SURV_MIN_FU
    else:
        positive = index["onset"].notna() & (og > config.BLANKING_DAYS) & (og <= horizon)
        control = index["onset"].isna() & (fud >= config.MIN_FU_FOR_CONTROL)
        # positives whose onset is beyond the horizon are neither incident-pos nor clean control
        excl_beyond_horizon = int((index["onset"].notna() & (og > horizon)).sum())
        excl_short_fu = int((index["onset"].isna() & (fud < config.MIN_FU_FOR_CONTROL)).sum())
        min_fu = config.MIN_FU_FOR_CONTROL

    keep = ~prevalent
    pos = positive & keep; ctl = control & keep
    n_pos = int(pos.sum()); n_ctl = int(ctl.sum())

    # incident events at 1/2/5 yr among eligible (non-prevalent) patients with an onset
    onset_days = og[keep & index["onset"].notna()]
    incident = onset_days > config.BLANKING_DAYS
    ev1 = int((incident & (onset_days <= 365)).sum())
    ev2 = int((incident & (onset_days <= 730)).sum())
    ev5 = int((incident & (onset_days <= 1825)).sum())

    # censoring distribution among controls (follow-up days)
    ctl_fu = fud[ctl]
    med_fu = float(np.median(ctl_fu)) if len(ctl_fu) else float("nan")
    fu_q = np.percentile(ctl_fu, [25, 50, 75]) if len(ctl_fu) else [np.nan] * 3

    # differential follow-up: controls with ANY later ECG vs none
    ctl_ids = index.index[ctl]
    has_later, later_affree = _later_ecg_stats(df, index, ctl_ids)
    n_ctl_with_later = len(set(ctl_ids) & has_later)
    n_ctl_no_later = n_ctl - n_ctl_with_later
    n_ctl_affree_doc = len(set(ctl_ids) & later_affree)   # sensitivity: documented AF-free later ECG

    rows = [
        ("eligible_sinus_index_patients", eligible_sinus),
        ("excluded_prevalent_concurrent_AF", n_prevalent),
        ("excluded_inadequate_followup_controls", excl_short_fu),
        ("final_positives", n_pos),
        ("final_controls", n_ctl),
        ("final_total", n_pos + n_ctl),
        ("neg_pos_ratio", round(n_ctl / max(1, n_pos), 3)),
        ("incident_events_1yr", ev1),
        ("incident_events_2yr", ev2),
        ("incident_events_5yr", ev5),
        ("control_median_followup_days", round(med_fu, 1)),
        ("control_followup_q25_days", round(float(fu_q[0]), 1)),
        ("control_followup_q75_days", round(float(fu_q[2]), 1)),
        ("min_followup_for_control_days", min_fu),
        ("controls_with_any_later_ecg", n_ctl_with_later),
        ("controls_with_no_later_ecg", n_ctl_no_later),
        ("controls_frac_with_later_ecg", round(n_ctl_with_later / max(1, n_ctl), 3)),
        ("controls_with_documented_AFfree_later_ecg", n_ctl_affree_doc),
        ("controls_frac_documented_AFfree_later", round(n_ctl_affree_doc / max(1, n_ctl), 3)),
    ]
    if not survival:
        rows.insert(3, ("excluded_onset_beyond_horizon", excl_beyond_horizon))
    return rows


def main(source, horizon):
    config.HORIZON_DAYS = horizon
    config.MIN_FU_FOR_CONTROL = horizon
    df = _load_ecg_meta()
    print(f"[meta] {len(df):,} ECGs / {df.subject_id.nunique():,} patients")

    all_rows = []; flow_lines = []
    for survival in (False, True):
        cohort = "survival" if survival else f"binary_h{horizon}"
        rows = _accounting(df, source, survival)
        for metric, value in rows:
            all_rows.append({"cohort": cohort, "metric": metric, "value": value})
        flow_lines.append(f"=== {cohort} cohort (source={source}) ===")
        for metric, value in rows:
            flow_lines.append(f"  {metric:42s} {value}")
        flow_lines.append("")
        print("\n".join(flow_lines[-(len(rows) + 2):]))

    out_csv = config.RESULTS_DIR / "cohort_accounting.csv"
    out_txt = config.RESULTS_DIR / "cohort_flow.txt"
    config.RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    out_txt.write_text("\n".join(flow_lines))
    print(f"[saved] {out_csv}")
    print(f"[saved] {out_txt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ecg", choices=["icd", "ecg"])
    ap.add_argument("--horizon", type=int, default=config.HORIZON_DAYS)
    a = ap.parse_args()
    main(a.source, a.horizon)
