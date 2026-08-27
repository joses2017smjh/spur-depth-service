#!/bin/bash
#SBATCH --job-name=spur_holdout
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.err

# Hold-out α/β, drift baseline from real DA2 depths, TinyUNet trunk
# segmenter, Farneback stereo fusion. 24k train set is still missing —
# N is the restored 9-tree val render, not orchard-scale.

set -euo pipefail
source /nfs/hpc/share/sanchej7/miniforge3_fixed/etc/profile.d/conda.sh
conda activate /nfs/stak/users/sanchej7/miniforge3/envs/depth-env

export SPUR_ROOT=/nfs/stak/users/sanchej7/hpc-share/spur-depth-service
export DATA_ROOT=/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur
export PYTHONPATH="${SPUR_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export XFORMERS_DISABLED=1

mkdir -p "${SPUR_ROOT}/bench/results" "${SPUR_ROOT}/weights" \
         /nfs/hpc/share/sanchej7/Computer_Vision/logs

echo "===== HOLD-OUT RIGOR START ====="
date; hostname; nvidia-smi -L || true

echo "===== TinyUNet trunk segmenter (7-tree train / paper-val test) ====="
python -m spur_depth.pipeline.train_seg \
    --data-root "$DATA_ROOT" \
    --epochs 8 --batch 8 \
    --out "${SPUR_ROOT}/weights/trunk_unet.pt"

echo "===== hold-out α/β + LOO + Farneback stereo + drift ====="
python -m spur_depth.bench.rigorous \
    --data-root "$DATA_ROOT" \
    --flow farneback \
    --stereo-limit 40 \
    --out "${SPUR_ROOT}/bench/results/holdout_affine_2026-08-22.json"

echo "===== HOLD-OUT RIGOR END ====="
date
