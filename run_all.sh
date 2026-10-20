#!/bin/bash
# run_all.sh - FULL from-scratch reproduction of the BTTN incident-AF paper on the UTwente SLURM cluster.
#
# A stranger with (a) the credentialed data and (b) the packed env can reproduce EVERY reported number,
# table, and figure with a single command:
#
#     bash run_all.sh                 # submit the whole pipeline as a SLURM dependency graph
#     FRESH=1 bash run_all.sh         # additionally wipe any existing project intermediates first
#
# Nothing is assumed pre-computed. Every cohort, every spectral tensor (extracted from the 34 GB zip),
# the SSL encoder, every trained model, and every table/figure is built by the jobs below, in dependency
# order, so no job ever reads a cache another job has not built yet. On a fresh account the on-demand
# waveform cache (~/data/mimic-ecg/files/) starts empty and is filled from the zip automatically.
# When the graph finishes, `python paper.py` assembles the figures/tables from code/results/.
#
# This is HPC-only: the headline numbers need the credentialed 34 GB waveforms + a GPU. There is no
# local shortcut, and no step may be hand-run - if you hit a "cache does not exist" error, that is a
# reproduction bug, not a user error (see the research-hub CLAUDE.md "Reproducibility" rule).
set -euo pipefail

ROOT="$HOME/projects/afnet/code"
LOGS="$HOME/projects/afnet/logs"
DATA="$HOME/data"
ENV_TARBALL="$HOME/envs/afnet_env3.tar.gz"
cd "$ROOT"
mkdir -p "$LOGS" results

# --------------------------------------------------------------------------- #
# 0. Prerequisites that cannot be shipped. Fail LOUDLY here with the exact fix - never silently, never
#    mid-run. These are the ONLY manual steps, and only to OBTAIN inputs, never to build intermediates.
# --------------------------------------------------------------------------- #
fail() { echo "[run_all] MISSING PREREQUISITE: $1" >&2; echo "  fix: $2" >&2; exit 1; }

[ -f "$ENV_TARBALL" ] || fail "packed env $ENV_TARBALL" \
  "build it once on a login node from requirements.txt, then pack it:
     python3 -m venv /tmp/\$USER/afnet_env && /tmp/\$USER/afnet_env/bin/pip install --no-cache-dir -r $ROOT/requirements.txt
     tar czf $ENV_TARBALL -C /tmp/\$USER afnet_env
   (torch must match the cluster CUDA; see requirements.txt + docs/HPC-UTwente-howto.md)"

ECG_ZIP="${AFNET_ECG_ZIP:-$DATA/mimic-ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip}"
[ -f "$ECG_ZIP" ] || fail "MIMIC-IV-ECG waveform zip ($ECG_ZIP)" \
  "download the credentialed 'MIMIC-IV-ECG matched subset 1.0' from PhysioNet to that path (or set AFNET_ECG_ZIP)"

HOSP="${AFNET_HOSP_DIR:-$DATA/mimic-iv/hosp}"
for f in diagnoses_icd admissions patients; do
  [ -f "$HOSP/$f.csv.gz" ] || fail "MIMIC-IV hosp file $HOSP/$f.csv.gz" \
    "download the credentialed MIMIC-IV hosp module from PhysioNet to $HOSP/ (or set AFNET_HOSP_DIR)"
done

[ -f "$DATA/mimic-ecg/record_list.csv" ] || fail "$DATA/mimic-ecg/record_list.csv" \
  "extract record_list.csv + machine_measurements.csv from the MIMIC-IV-ECG release into $DATA/mimic-ecg/"
[ -d "$DATA/code15" ]  || echo "[run_all] NOTE: $DATA/code15 not found -> the CODE-15 external step will fail; download CODE-15 there to include it." >&2

# PTB-XL (public): the zip PhysioNet serves lives in $DATA/ptb-xl under PhysioNet's own name, $P.zip,
# and holds one top folder, $P. Its database and 100 Hz records are extracted into $DATA/ptb-xl
# itself with that folder stripped, so ptbxl_database.csv and records100/ sit at the folder's top,
# where every reader expects them. A file already there is kept, and the database is written last,
# so its presence means the records before it are complete.
P="ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
PTBXL_ZIP="${AFNET_PTBXL_ZIP:-$DATA/ptb-xl/$P.zip}"
if [ ! -f "$DATA/ptb-xl/ptbxl_database.csv" ]; then
  if [ -f "$PTBXL_ZIP" ]; then
    echo "[run_all] extracting PTB-XL (database + 100 Hz records) from $PTBXL_ZIP into $DATA/ptb-xl (one-time) ..."
    python3 - "$PTBXL_ZIP" "$DATA/ptb-xl" "$P" <<'PY'
import os
import shutil
import sys
import zipfile

zip_path, out, top = sys.argv[1], sys.argv[2], sys.argv[3] + "/"
with zipfile.ZipFile(zip_path) as archive:
    wanted = []
    for info in archive.infolist():
        rel = info.filename[len(top):] if info.filename.startswith(top) else ""
        if not info.is_dir() and (rel == "ptbxl_database.csv" or rel.startswith("records100/")):
            wanted.append((rel == "ptbxl_database.csv", rel, info))
    for _, rel, info in sorted(wanted, key=lambda one: one[0]):
        target = os.path.join(out, rel)
        if os.path.exists(target):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with archive.open(info) as source, open(target, "wb") as sink:
            shutil.copyfileobj(source, sink)
PY
  else
    echo "[run_all] NOTE: no PTB-XL zip at $PTBXL_ZIP -> the PTB-XL external step will fail; download PTB-XL 1.0.3 from PhysioNet." >&2
  fi
fi

# --------------------------------------------------------------------------- #
# 1. Optional: wipe existing project intermediates so nothing is reused (true from-scratch re-test).
#    On a genuinely fresh account there is nothing to wipe - the graph builds everything regardless.
# --------------------------------------------------------------------------- #
if [ "${FRESH:-0}" = "1" ]; then
  echo "[run_all] FRESH=1 -> removing project intermediates (cohorts, tensor caches, encoder, results)"
  rm -f results/*.npy results/*.npz results/*.pt results/*.csv results/*.png
  # DEEP=1 also drops the on-demand waveform cache so records re-extract from the zip (adds hours).
  [ "${DEEP:-0}" = "1" ] && { echo "[run_all] DEEP=1 -> also clearing the extracted waveform cache"; rm -rf "$DATA/mimic-ecg/files"; }
fi

# helper: submit a job through hpc/job.slurm, route logs to $LOGS, return the job id
sub() { sbatch --parsable --chdir="$ROOT" --output="$LOGS/%x_%j.out" "$@"; }
J() { echo "hpc/job.slurm"; }   # the generic env-staging GPU runner: sbatch ... job.slurm <script.py> [args]

echo "[run_all] submitting the reproduction graph ..."

# --------------------------------------------------------------------------- #
# 2. Cohorts (gold ICD-10 I48 labels from the hosp module; metadata-only, minutes).
# --------------------------------------------------------------------------- #
C1=$(sub --job-name=bttn-cohort-h365  "$(J)" core/cohort.py --source icd --horizon 365)
C2=$(sub --job-name=bttn-cohort-surv  "$(J)" core/cohort.py --source icd --survival)
C3=$(sub --job-name=bttn-cohort-h1825 "$(J)" core/cohort.py --source icd --horizon 1825)
CA=$(sub --job-name=bttn-cohort-acct  --dependency=afterok:"$C1" "$(J)" core/cohort_accounting.py --source icd --horizon 365)

# --------------------------------------------------------------------------- #
# 3. SSL pretrain -> results/ssl_encoder.pt (builds the 30k unlabeled spectral pool from the zip).
# --------------------------------------------------------------------------- #
S=$(sub --job-name=bttn-ssl "$(J)" core/ssl_pretrain.py --n 30000 --epochs 60 --batch 256)

# --------------------------------------------------------------------------- #
# 4. Discrimination ladders (each BUILDS its spectral tensor cache from the zip on first use).
#    Table I (1-yr, 7 methods) needs the h365 cohort + the SSL encoder; 5-yr needs the h1825 cohort.
# --------------------------------------------------------------------------- #
B1=$(sub --job-name=bttn-table1 --dependency=afterok:"$C1":"$S" "$(J)" core/bench.py --source icd --horizon 365 --seeds 5 --methods clinical,bttr,cnn,bttn,bttn_ssl,bttn_cov,bttn_ssl_cov --tensor spectral --neg_ratio 4)
B5=$(sub --job-name=bttn-5yr    --dependency=afterok:"$C3":"$S" "$(J)" core/bench.py --source icd --horizon 1825 --seeds 5 --methods clinical,bttn,bttn_ssl,bttn_cov,bttn_ssl_cov --tensor spectral --neg_ratio 4)

# --------------------------------------------------------------------------- #
# 5. Multi-horizon survival (one Cox model; builds the survival tensor cache).
# --------------------------------------------------------------------------- #
MH=$(sub --job-name=bttn-multihorizon --dependency=afterok:"$C2":"$S" "$(J)" experiments/multihorizon.py --source icd --tensor spectral)

# --------------------------------------------------------------------------- #
# 6. Interpretability + ablation (reuse the h365 tensor cache the Table-I job built -> depend on it).
# --------------------------------------------------------------------------- #
FA=$(sub --job-name=bttn-faith     --dependency=afterok:"$B1" "$(J)" experiments/faithfulness_controls.py --source icd --horizon 365 --neg_ratio 4)
IV=$(sub --job-name=bttn-interv    --dependency=afterok:"$B1" "$(J)" experiments/intervention_curve.py --source icd --horizon 365 --neg_ratio 4)
AB=$(sub --job-name=bttn-ablat     --dependency=afterok:"$B1" "$(J)" experiments/ablation_blockstructure.py --source icd --horizon 365 --neg_ratio 4)
KS=$(sub --job-name=bttn-ksweep    --dependency=afterok:"$B1" "$(J)" experiments/ablation.py --mode K --source icd --horizon 365 --neg_ratio 4 --seeds 3)
IN=$(sub --job-name=bttn-interpret --dependency=afterok:"$B1" "$(J)" experiments/interpret_bttn.py --source icd --horizon 365 --neg_ratio 4)
T1=$(sub --job-name=bttn-table1cis --dependency=afterok:"$B1" "$(J)" experiments/compute_table1_cis.py)

# --------------------------------------------------------------------------- #
# 7. Clinical utility + transportable PPV/NNM (reads the multi-horizon survival npz).
# --------------------------------------------------------------------------- #
CE=$(sub --job-name=bttn-clineval --dependency=afterok:"$MH" "$(J)" experiments/clin_eval.py)

# --------------------------------------------------------------------------- #
# 7b. Cardiac-phase saliency figures (reuse the survival tensor cache the multi-horizon job built).
#     beat_saliency_long imports beat_saliency, so run it after (avoids CPU thrash + reuses the cache).
# --------------------------------------------------------------------------- #
BS=$(sub  --job-name=bttn-beatsal     --dependency=afterok:"$MH" "$(J)" figures/beat_saliency.py --source icd)
BSL=$(sub --job-name=bttn-beatsallong --dependency=afterok:"$BS" "$(J)" figures/beat_saliency_long.py --source icd)

# --------------------------------------------------------------------------- #
# 8. External identification/substrate validation (independent CODE-15 + PTB-XL data).
# --------------------------------------------------------------------------- #
X1=$(sub --job-name=bttn-code15-afsr   "$(J)" experiments/external_code15.py --task afsr   --seeds 3)
X2=$(sub --job-name=bttn-code15-detect "$(J)" experiments/external_code15.py --task detect --seeds 3)
X3=$(sub --job-name=bttn-ptbxl-detect  "$(J)" experiments/external_ptbxl.py  --task detect --seeds 3)

echo "[run_all] submitted. jobs: cohorts=$C1,$C2,$C3 acct=$CA ssl=$S table1=$B1 5yr=$B5 multihorizon=$MH"
echo "[run_all]   faith=$FA interv=$IV ablat=$AB ksweep=$KS interpret=$IN table1cis=$T1 clineval=$CE beatsal=$BS,$BSL external=$X1,$X2,$X3"
echo "[run_all] monitor:  squeue -u \$USER   |   tail -f $LOGS/bttn-*.out"
echo "[run_all] when everything is COMPLETED, assemble figures/tables:  cd $ROOT && python paper.py"
