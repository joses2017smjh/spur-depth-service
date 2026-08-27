#!/bin/bash
#SBATCH --job-name=spur_da2_onnx
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.err

# DA2 ONNX only. Latency already measured exclusive on V100 (job 21004172).
# Job 21004172 failed because DA2's MemEffAttention called CUDA xFormers on CPU.
# to_onnx.py now forces XFORMERS_AVAILABLE=False on that module.

set -euo pipefail
source /nfs/hpc/share/sanchej7/miniforge3_fixed/etc/profile.d/conda.sh
conda activate /nfs/stak/users/sanchej7/miniforge3/envs/depth-env

export COMPUTER_VISION_ROOT=/nfs/hpc/share/sanchej7/Computer_Vision
export SPUR_ROOT=/nfs/stak/users/sanchej7/hpc-share/spur-depth-service
export PYTHONPATH="${SPUR_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export XFORMERS_DISABLED=1
export PYTHONUNBUFFERED=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export TMPDIR=/tmp/${USER}_spur_${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR" "${SPUR_ROOT}/engines"

export SPUR_DA2_CKPT="${COMPUTER_VISION_ROOT}/checkpoints/full_spur_2tex_all_3view_seed1/best.pth"
export DA2_ROOT="${COMPUTER_VISION_ROOT}/depth-anything-v2/metric_depth"

echo "===== DA2 ONNX START ====="
date; hostname; nvidia-smi -L || true
cp -v "$SPUR_DA2_CKPT" "$TMPDIR/da2_best.pth"
python -m spur_depth.export.to_onnx \
    --graph da2 \
    --da2-ckpt "$TMPDIR/da2_best.pth" \
    --da2-root "$DA2_ROOT" \
    --da2-h 518 --da2-w 924 \
    --out "${SPUR_ROOT}/engines"
ls -lh "${SPUR_ROOT}/engines/da2.onnx"
echo "===== DA2 ONNX END ====="
date
