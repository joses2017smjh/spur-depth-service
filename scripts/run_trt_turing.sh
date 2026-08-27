#!/bin/bash
#SBATCH --job-name=spur_trt_turing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx8000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.err

# TensorRT 11.2 needs SM >= 7.5. V100 (SM 7.0) on job 21015984 failed.
# 21017775: dropped EXPLICIT_BATCH. 21018041: dropped platform_has_fast_fp16.
# This retry uses network_creation_flags() + builder_has_fp16(). Fuse/decode only.

set -euo pipefail
SPUR_ROOT="${SPUR_ROOT:-/nfs/hpc/share/sanchej7/spur-depth-service}"
# shellcheck source=/dev/null
source "${SPUR_ROOT}/scripts/scratch_env.sh"
spur_export_scratch

source /nfs/hpc/share/sanchej7/miniforge3_fixed/etc/profile.d/conda.sh
conda activate /nfs/stak/users/sanchej7/miniforge3/envs/depth-env

export PYTHONPATH="${SPUR_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export SPUR_TRT_PATH="${SPUR_TRT_PATH:-/nfs/hpc/share/sanchej7/opt/py-tensorrt}"
export LD_LIBRARY_PATH="${SPUR_TRT_PATH}/tensorrt_libs:${LD_LIBRARY_PATH:-}"

echo "===== TRT TURING START ====="
date; hostname; nvidia-smi -L || true
python -c "import torch; print('sm', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"

python -m spur_depth.export.trt_runtime \
    --onnx "${SPUR_ROOT}/engines/fuse_decode.onnx" \
    --out-dir "${SPUR_ENGINE_DIR}" \
    --fp16 --bench

shopt -s nullglob
for f in "${SPUR_ENGINE_DIR}"/*.plan "${SPUR_ENGINE_DIR}"/*.json \
         "${SPUR_ARTIFACT_DIR}"/trt_fuse_*.json; do
    [ -f "$f" ] || continue
    bytes=$(stat -c%s "$f")
    if [ "$bytes" -lt 83886080 ]; then
        case "$f" in
            *.json) dest="${SPUR_ROOT}/bench/results/" ;;
            *) dest="${SPUR_ROOT}/engines/" ;;
        esac
        cp -v "$f" "$dest"
    else
        echo "skip large $f ($bytes bytes)"
    fi
done
echo "===== TRT TURING END ====="
date
