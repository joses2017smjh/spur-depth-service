#!/bin/bash
#SBATCH --job-name=spur_24k_affine
#SBATCH --partition=dgx2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%j.err

# Orchard-scale α/β + drift. Exits 2 until the 24k train set is mounted
# (Depot or SPUR_TRAIN_ROOT). Does not reprint the 9-tree numbers as 24k.

set -euo pipefail
SPUR_ROOT="${SPUR_ROOT:-/nfs/hpc/share/sanchej7/spur-depth-service}"
# shellcheck source=/dev/null
source "${SPUR_ROOT}/scripts/scratch_env.sh"
spur_export_scratch

source /nfs/hpc/share/sanchej7/miniforge3_fixed/etc/profile.d/conda.sh
conda activate /nfs/stak/users/sanchej7/miniforge3/envs/depth-env
export PYTHONPATH="${SPUR_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export DATA_ROOT="${DATA_ROOT:-/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur}"

python - <<'PY'
import json, os, sys
from pathlib import Path
from spur_depth.data.restore_index import discover_train_or_restore
root = Path(os.environ["DATA_ROOT"])
rows = discover_train_or_restore(root)
print(f"indexed {len(rows)} frames under {root} SPUR_TRAIN_ROOT={os.environ.get('SPUR_TRAIN_ROOT')}")
if len(rows) < 1000:
    print(
        "24k train set is still missing. "
        "Mount it on Depot and set DATA_ROOT or SPUR_TRAIN_ROOT. Refusing to fake orchard-scale α/β.",
        file=sys.stderr,
    )
    sys.exit(2)
sys.exit(0)
PY

python -m spur_depth.bench.improve \
    --data-root "$DATA_ROOT" \
    --min-orchard-frames 1000 \
    --out "${SPUR_ARTIFACT_DIR}/orchard_affine.json"
cp -v "${SPUR_ARTIFACT_DIR}/orchard_affine.json" \
      "${SPUR_ROOT}/bench/results/orchard_affine.json"
