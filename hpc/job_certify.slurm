#!/bin/bash
#SBATCH --job-name=afnet-cert
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/faesayl/projects/afnet/logs/%x_%j.out
# Quick package-wiring certification (NOT a result): proves env-stage + PYTHONPATH +
# core package import (incl. vendored bttrlib via bttr_model) on the migrated
# ~/projects/afnet/code package layout. No data needed.
set -e
STAGE=/local/$USER/afnet3; mkdir -p "$STAGE"
( flock 9
  if [ ! -f "$STAGE/.ready" ]; then tar xzf ~/envs/afnet_env3.tar.gz -C "$STAGE"; touch "$STAGE/.ready"; fi
) 9>/tmp/afnet_env3.lock
export PYTHONPATH="$STAGE/afnet_env/lib/python3.10/site-packages"
cd ~/projects/afnet/code
export PYTHONPATH="$PWD:$PYTHONPATH"
/usr/bin/python3 -c "import torch; from core import config, bench; from core.bttn import BTTN; from core.bttr_model import BTTR; import core.cnn, core.survival; print('AFNET_PKG_IMPORT_OK cuda?', torch.cuda.is_available(), '| RESULTS_DIR', config.RESULTS_DIR)"
echo "AFNET_CERT_DONE"
