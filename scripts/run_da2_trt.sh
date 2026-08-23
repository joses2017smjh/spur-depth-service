#!/bin/bash
#SBATCH --job-name=spur_da2_trt
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.err

# Fatter than the 8 GB interactive job: DA2 ViT-L export (~3.8 GB ckpt) plus
# TensorRT engine builds. Does not invent a TRT number if trtexec is missing.

set -euo pipefail

source /nfs/hpc/share/sanchej7/miniforge3_fixed/etc/profile.d/conda.sh
conda activate /nfs/stak/users/sanchej7/miniforge3/envs/depth-env

export COMPUTER_VISION_ROOT=/nfs/hpc/share/sanchej7/Computer_Vision
export SPUR_ROOT=/nfs/stak/users/sanchej7/hpc-share/spur-depth-service
export PYTHONPATH="${SPUR_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export XFORMERS_DISABLED=1
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp/${USER}_spur_${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR" "${SPUR_ROOT}/engines" "${SPUR_ROOT}/bench/results" \
         "${COMPUTER_VISION_ROOT}/logs"

export SPUR_CKPT="${COMPUTER_VISION_ROOT}/checkpoints/dino_da2ft_3pair_fusion_nopose_spur_seed1/exp1/seed_01/best_epoch_0023.pt"
export SPUR_DA2_CKPT="${COMPUTER_VISION_ROOT}/checkpoints/full_spur_2tex_all_3view_seed1/best.pth"
export DA2_ROOT="${COMPUTER_VISION_ROOT}/depth-anything-v2/metric_depth"

echo "===== JOB START spur_da2_trt ====="
date
hostname
nvidia-smi -L || true
free -h | head -2
which python
python -c "print('python-ok')"
# NVML check avoids creating a CUDA context during import (can hang 10+ min on a cold dgx2).
export PYTORCH_NVML_BASED_CUDA_CHECK=1
timeout 90 python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || echo "torch probe timed out; continuing into export"

echo "===== hunt trtexec / tensorrt ====="
command -v trtexec || true
python -c "import tensorrt as t; print('tensorrt', t.__version__)" 2>/dev/null || echo "no python tensorrt"
module avail -t 2>&1 | grep -iE 'tensorrt|tensor.rt' || true

# Latency does not need DA2. Run it first so a slow ONNX export cannot eat the wallclock
# and leave us with no numbers (job 20999884 TIMED OUT after 4h still inside torch.load).
echo "===== exclusive V100 latency (clean p95/p99) ====="
python -m spur_depth.bench.latency --stage refiner --precision fp32 --ckpt "$SPUR_CKPT"
python -m spur_depth.bench.latency --stage refiner --precision fp16 --ckpt "$SPUR_CKPT"

echo "===== copy DA2 ckpt to node-local /tmp (3.8G; NFS torch.load hung 4h) ====="
cp -v "$SPUR_DA2_CKPT" "$TMPDIR/da2_best.pth"
export SPUR_DA2_CKPT="$TMPDIR/da2_best.pth"
ls -lh "$SPUR_DA2_CKPT"

echo "===== DA2 ONNX (Engine A) ====="
python -m spur_depth.export.to_onnx \
    --graph da2 \
    --da2-ckpt "$SPUR_DA2_CKPT" \
    --da2-root "$DA2_ROOT" \
    --da2-h 518 --da2-w 924 \
    --out "${SPUR_ROOT}/engines"

echo "===== TensorRT fuse_decode fp32 then fp16 ====="
python -m spur_depth.export.build_engine \
    --onnx "${SPUR_ROOT}/engines/fuse_decode.onnx" \
    --out-dir "${SPUR_ROOT}/engines" || echo "TRT fuse fp32 SKIPPED"
python -m spur_depth.export.build_engine \
    --onnx "${SPUR_ROOT}/engines/fuse_decode.onnx" \
    --out-dir "${SPUR_ROOT}/engines" --fp16 || echo "TRT fuse fp16 SKIPPED"

echo "===== TensorRT encoder (1.2 GB ONNX) fp32 then fp16 ====="
python -m spur_depth.export.build_engine \
    --onnx "${SPUR_ROOT}/engines/encoder.onnx" \
    --out-dir "${SPUR_ROOT}/engines" \
    --min "rgb:1x3x280x512,d_pro:1x1x280x512" \
    --opt "rgb:6x3x280x512,d_pro:6x1x280x512" \
    --max "rgb:6x3x280x512,d_pro:6x1x280x512" \
    --workspace-mib 12288 || echo "TRT encoder fp32 SKIPPED"
python -m spur_depth.export.build_engine \
    --onnx "${SPUR_ROOT}/engines/encoder.onnx" \
    --out-dir "${SPUR_ROOT}/engines" --fp16 \
    --min "rgb:1x3x280x512,d_pro:1x1x280x512" \
    --opt "rgb:6x3x280x512,d_pro:6x1x280x512" \
    --max "rgb:6x3x280x512,d_pro:6x1x280x512" \
    --workspace-mib 12288 || echo "TRT encoder fp16 SKIPPED"

echo "===== TensorRT da2 fp32 then fp16 ====="
if [ -f "${SPUR_ROOT}/engines/da2.onnx" ]; then
    python -m spur_depth.export.build_engine \
        --onnx "${SPUR_ROOT}/engines/da2.onnx" \
        --out-dir "${SPUR_ROOT}/engines" || echo "TRT da2 fp32 SKIPPED"
    python -m spur_depth.export.build_engine \
        --onnx "${SPUR_ROOT}/engines/da2.onnx" \
        --out-dir "${SPUR_ROOT}/engines" --fp16 || echo "TRT da2 fp16 SKIPPED"
fi

echo "===== engines on disk ====="
ls -lh "${SPUR_ROOT}/engines" || true
echo "===== JOB END spur_da2_trt ====="
date
