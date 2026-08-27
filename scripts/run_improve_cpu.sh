#!/bin/bash
#SBATCH --job-name=spur_improve_cpu
#SBATCH --partition=dgx2
# CPU-only re-score. Fine to sit in queue behind GPU jobs.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.err

# Re-score Huber/median on the n=78 GT-mask protocol. No GPU (YOLO/UNet skipped).
# Does not replace 0.0344 m unless holdout_rigor2 JSON says so.

set -euo pipefail
SPUR_ROOT="${SPUR_ROOT:-/nfs/hpc/share/sanchej7/spur-depth-service}"
# shellcheck source=/dev/null
source "${SPUR_ROOT}/scripts/scratch_env.sh"
spur_export_scratch

source /nfs/hpc/share/sanchej7/miniforge3_fixed/etc/profile.d/conda.sh
conda activate /nfs/stak/users/sanchej7/miniforge3/envs/depth-env

export PYTHONPATH="${SPUR_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DATA_ROOT="${DATA_ROOT:-/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur}"

echo "===== IMPROVE CPU START ====="
date; hostname

python -m spur_depth.bench.improve \
    --data-root "$DATA_ROOT" \
    --skip-detectors \
    --out "${SPUR_ARTIFACT_DIR}/holdout_rigor2_2026-08-23.json"

cp -v "${SPUR_ARTIFACT_DIR}/holdout_rigor2_2026-08-23.json" \
      "${SPUR_ROOT}/bench/results/holdout_rigor2_2026-08-23.json"
echo "===== IMPROVE CPU END ====="
date
