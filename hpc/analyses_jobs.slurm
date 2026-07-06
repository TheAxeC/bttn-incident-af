#!/bin/bash
#SBATCH --job-name=afanalyses
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/faesayl/projects/afnet/logs/%x_%j.out
#
# Supporting-analysis GPU jobs: block-structure ablation + faithfulness controls
# + intervention/sufficiency curve. Mirrors job.slurm (afnet_env3 staged to /local,
# system python3 + PYTHONPATH, partition main-gpu). Self-contained: runs all three.
#
# ============================ EXACT SUBMIT SEQUENCE (run when HPC access returns) ========
# From this code/ dir on the laptop (UT network / EduVPN up):
#   # 1) push code, NOT caches (exclude __pycache__, *.npy, *.log):
#   rsync -az --exclude '__pycache__' --exclude '*.npy' --exclude '*.log' \
#         code/ utwente:~/projects/afnet/code/
#   # 2) ship the cached spectral tensor + cohort csv these jobs need (already built locally;
#   #    NEVER ship the 36 GB zip). The r4 spectral tensor + the SSL encoder are required:
#   rsync -a results/X_spectral_ecg_h365_r4_n13090.npy \
#            results/cohort_ecg_h365.csv \
#            results/ssl_encoder.pt utwente:~/projects/afnet/code/results/
#   # (only if rebuilding multihorizon from scratch -- otherwise the 120d row is already
#   #  computed locally from results/multihorizon_ecg_spectral.npz):
#   rsync -a results/X_spectral_ecg_survival_n20000.npy \
#            results/cohort_ecg_survival.csv utwente:~/projects/afnet/code/results/
#   # 3) launch:
#   ssh utwente 'cd ~/projects/afnet/code && sbatch analyses_jobs.slurm'
#   # 4) monitor + pull results back:
#   ssh utwente 'squeue -u $USER; tail -f ~/projects/afnet/logs/afrev_*.out'
#   rsync -a utwente:~/projects/afnet/code/results/ablation_blockstructure.csv \
#            utwente:~/projects/afnet/code/results/faithfulness_controls.csv \
#            utwente:~/projects/afnet/code/results/intervention_curve.csv \
#            utwente:~/projects/afnet/code/results/multihorizon_ecg_spectral.csv \
#            utwente:~/projects/afnet/code/results/multihorizon_ecg_spectral.npz results/
# ========================================================================================
set -e
STAGE=/local/$USER/afnet3          # env: afnet_env3.tar.gz (torch + tensorly + h5py)
mkdir -p "$STAGE"
( flock 9
  if [ ! -f "$STAGE/.ready" ]; then
    echo "[$(date +%T)] extracting env to $STAGE ..."
    tar xzf ~/envs/afnet_env3.tar.gz -C "$STAGE"
    touch "$STAGE/.ready"
  fi
) 9>/tmp/afnet_env3.lock
export PYTHONPATH="$STAGE/afnet_env/lib/python3.10/site-packages"
cd ~/projects/afnet/code
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "[$(date +%T)] node $(hostname)  GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
/usr/bin/python3 -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

# (1) R5 -- block-term-STRUCTURE ablation (blockterm | cp | fullrank), 5-fold x 3-seed
echo "[$(date +%T)] === ablation_blockstructure ==="
/usr/bin/python3 -u experiments/ablation_blockstructure.py --source ecg --horizon 365 --neg_ratio 4 --seeds 3

# (2a) R3 -- faithfulness probe controls, 5-fold x 3-seed
echo "[$(date +%T)] === faithfulness_controls ==="
/usr/bin/python3 -u experiments/faithfulness_controls.py --source ecg --horizon 365 --neg_ratio 4 --seeds 3

# (2b) R3 -- intervention / sufficiency curve (single held-out trained model)
echo "[$(date +%T)] === intervention_curve ==="
/usr/bin/python3 -u experiments/intervention_curve.py --source ecg --horizon 365 --neg_ratio 4 --seed 0

# (3) Multi-horizon survival eval -- now includes the 120d "90-day onset (post-blank)"
#     short-onset incident horizon alongside 1/2/5yr (HORIZONS=[120,365,730,1825] in
#     multihorizon.py). One survival model, read at each horizon -- NO per-horizon retrain.
#     NOTE: the 120d row was already computed LOCALLY (CPU) from the saved per-patient risk
#     scores in results/multihorizon_ecg_spectral.npz; this rerun only refreshes it if the
#     survival model / spectral survival tensor are rebuilt. Needs X_spectral_ecg_survival_n*.npy
#     + cohort_ecg_survival.csv staged in results/ if a full rebuild is wanted.
echo "[$(date +%T)] === multihorizon (incl. 120d 90-day-onset horizon) ==="
/usr/bin/python3 -u experiments/multihorizon.py --source ecg --tensor spectral

echo "[$(date +%T)] DONE"
