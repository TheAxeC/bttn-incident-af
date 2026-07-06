#!/bin/bash
#SBATCH --job-name=afnet
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# ^ username-agnostic (lands in the submission dir); run_all.sh overrides with an absolute logs path.
# Usage: sbatch [--job-name=NAME] job.slurm <script.py> [args...]
set -e
STAGE=/local/$USER/afnet3          # env now includes tensorly + h5py (afnet_env3.tar.gz)
mkdir -p "$STAGE"
# stage the env tarball to fast node-local disk (shared/guarded across array tasks)
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
echo "[$(date +%T)] running: $@"
/usr/bin/python3 -u "$@"
echo "[$(date +%T)] DONE"
