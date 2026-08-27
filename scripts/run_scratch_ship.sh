#!/bin/bash
#SBATCH --job-name=spur_scratch_ship
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.err

# TensorRT fuse-only, C2 inspect, YOLO-nano, Huber/median affine.
# Everything heavy lands on this node's /scratch. Only small artifacts
# copy back to the repo. Does not pull a 10 GB NGC image. Does not claim
# a TensorRT end-to-end number. 24k train set is still missing.

set -euo pipefail

SPUR_ROOT="${SPUR_ROOT:-/nfs/hpc/share/sanchej7/spur-depth-service}"
# shellcheck source=/dev/null
source "${SPUR_ROOT}/scripts/scratch_env.sh"
spur_export_scratch

source /nfs/hpc/share/sanchej7/miniforge3_fixed/etc/profile.d/conda.sh
conda activate /nfs/stak/users/sanchej7/miniforge3/envs/depth-env

export DATA_ROOT=/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur
export PYTHONPATH="${SPUR_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export XFORMERS_DISABLED=1
export SPUR_TRT_PATH="${SPUR_TRT_PATH:-/nfs/hpc/share/sanchej7/opt/py-tensorrt}"
export LD_LIBRARY_PATH="${SPUR_TRT_PATH}/tensorrt_libs:${LD_LIBRARY_PATH:-}"
cd "$SPUR_ROOT"

mkdir -p /nfs/hpc/share/sanchej7/Computer_Vision/logs \
         "${SPUR_ROOT}/bench/results" "${SPUR_ROOT}/weights"

echo "===== SCRATCH SHIP START ====="
date; hostname; nvidia-smi -L || true
echo "SPUR_SCRATCH=$SPUR_SCRATCH"
df -h /scratch 2>/dev/null | head -2 || true

echo "===== 1. YOLO-nano (no 1080p copies) ====="
python -m spur_depth.pipeline.train_yolo \
    --data-root "$DATA_ROOT" \
    --epochs 20 --batch 8 \
    --out "${SPUR_ARTIFACT_DIR}/yolo_nano.pt"
cp -v "${SPUR_ARTIFACT_DIR}/yolo_nano.pt" "${SPUR_ROOT}/weights/yolo_nano.pt"
cp -v "${SPUR_ARTIFACT_DIR}/yolo_nano.json" "${SPUR_ROOT}/weights/yolo_nano.json"

echo "===== 2. Huber / median-of-trees / box-anchor / box-gated field ====="
python -m spur_depth.bench.improve \
    --data-root "$DATA_ROOT" \
    --unet "${SPUR_ROOT}/weights/trunk_unet.pt" \
    --yolo "${SPUR_ARTIFACT_DIR}/yolo_nano.pt" \
    --out "${SPUR_ARTIFACT_DIR}/holdout_rigor2_2026-08-23.json"
cp -v "${SPUR_ARTIFACT_DIR}/holdout_rigor2_2026-08-23.json" \
      "${SPUR_ROOT}/bench/results/holdout_rigor2_2026-08-23.json"

echo "===== 3. TensorRT fuse_decode only (Python API, pip wheel) ====="
set +e
python -m spur_depth.export.trt_runtime \
    --onnx "${SPUR_ROOT}/engines/fuse_decode.onnx" \
    --out-dir "${SPUR_ENGINE_DIR}" \
    --fp16 --bench
TRT_RC=$?
set -e
echo "trt_runtime exit $TRT_RC"
ls -lh "${SPUR_ENGINE_DIR}" || true
# Copy back only small plans + their sidecar JSON.
shopt -s nullglob
for f in "${SPUR_ENGINE_DIR}"/*.plan "${SPUR_ENGINE_DIR}"/*.json; do
    bytes=$(stat -c%s "$f")
    if [ "$bytes" -lt 83886080 ]; then
        cp -v "$f" "${SPUR_ROOT}/engines/"
    else
        echo "skip large $f ($bytes bytes)"
    fi
done
cp -v "${SPUR_ARTIFACT_DIR}"/trt_fuse_*.json "${SPUR_ROOT}/bench/results/" 2>/dev/null || true

echo "===== 4. C2 inspect any plan we built ====="
set +e
for plan in "${SPUR_ENGINE_DIR}"/*.plan; do
    python -m spur_depth.export.c2_harness "$plan" \
        | tee "${SPUR_ARTIFACT_DIR}/c2_$(basename "$plan").json"
    cp -v "${SPUR_ARTIFACT_DIR}/c2_$(basename "$plan").json" \
          "${SPUR_ROOT}/bench/results/" || true
done
set -e

echo "===== 5. Apptainer GPU smoke (best-effort; may lack network) ====="
set +e
bash "${SPUR_ROOT}/scripts/run_gpu_container.sh" || echo "container smoke SKIPPED"
set -e
cp -v "${SPUR_ARTIFACT_DIR}"/gpu_container_smoke.json "${SPUR_ROOT}/bench/results/" 2>/dev/null || true

echo "===== artifacts copied back ====="
ls -lh "${SPUR_ROOT}/weights/yolo_nano.pt" "${SPUR_ROOT}/engines"/*.plan 2>/dev/null || true
ls -lh "${SPUR_ROOT}/bench/results"/holdout_rigor2_*.json \
       "${SPUR_ROOT}/bench/results"/c2_*.json \
       "${SPUR_ROOT}/bench/results"/trt_fuse_*.json 2>/dev/null || true
echo "===== SCRATCH SHIP END ====="
date
