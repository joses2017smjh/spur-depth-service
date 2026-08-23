#!/bin/bash
#SBATCH --job-name=spur_p1p2
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.err

# P1 eval_rmse (torch) on the 2026-08-20 re-rendered val trees, then P2
# encoder+fuse ONNX export from the seed-1 3-pair checkpoint.
# A re-render will not reproduce 0.0445 m bit-for-bit.

set -euo pipefail

source /nfs/hpc/share/sanchej7/miniforge3_fixed/etc/profile.d/conda.sh
conda activate /nfs/stak/users/sanchej7/miniforge3/envs/depth-env

export COMPUTER_VISION_ROOT=/nfs/hpc/share/sanchej7/Computer_Vision
export SPUR_ROOT=/nfs/stak/users/sanchej7/hpc-share/spur-depth-service
export PYTHONPATH="${SPUR_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export INPUT_DEPTH_SUBDIR=Da2Finetune
export SPUR_CKPT="${COMPUTER_VISION_ROOT}/checkpoints/dino_da2ft_3pair_fusion_nopose_spur_seed1/exp1/seed_01/best_epoch_0023.pt"
VAL_MANIFEST="${COMPUTER_VISION_ROOT}/manifests/dino_da2ft_3pair_fusion_nopose_spur_seed1/stereo_val_boxfam.csv"

mkdir -p "${COMPUTER_VISION_ROOT}/logs" "${SPUR_ROOT}/engines" "${SPUR_ROOT}/bench/results"

echo "===== JOB START spur_p1p2 ====="
date
hostname
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "===== P1 eval_rmse torch ====="
python -m spur_depth.bench.eval_rmse \
    --backend torch \
    --ckpt "$SPUR_CKPT" \
    --val-manifest "$VAL_MANIFEST" \
    --no-pro-calib \
    --seed 0

echo "===== P1 latency fp32 + fp16 on this GPU ====="
python -m spur_depth.bench.latency --stage refiner --precision fp32 --ckpt "$SPUR_CKPT"
python -m spur_depth.bench.latency --stage refiner --precision fp16 --ckpt "$SPUR_CKPT"

echo "===== P2 ONNX encoder + fuse_decode ====="
python -m spur_depth.export.to_onnx --graph both --ckpt "$SPUR_CKPT" --out "${SPUR_ROOT}/engines"

echo "===== P2 eval_rmse onnx (if graphs exist) ====="
python -m spur_depth.bench.eval_rmse \
    --backend onnx \
    --ckpt "$SPUR_CKPT" \
    --val-manifest "$VAL_MANIFEST" \
    --onnx-dir "${SPUR_ROOT}/engines" \
    --no-pro-calib \
    --seed 0 \
    --out "${SPUR_ROOT}/bench/results/onnx_eval.json" || true

echo "===== JOB END spur_p1p2 ====="
date
