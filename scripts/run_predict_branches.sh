#!/bin/bash
#SBATCH -J spur-bpredict
#SBATCH -p ampere,dgxh,gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o /nfs/hpc/share/sanchej7/spur-branch-runs/%x-%j.out

# Full-frame BranchNet predictions from an existing checkpoint, from a frozen commit:
#   sbatch --export=NONE scripts/run_predict_branches.sh <sha> <ckpt> <out> [predict args...]
# e.g. the Envy-only model on the UFO val and test trees (protocol v3):
#   ... <sha> /nfs/.../21532184/best.pt /nfs/.../21532184/pred_ufo --trees lpy_ufo_00028 ...

set -euo pipefail
COMMIT=${1:?commit} CKPT=${2:?checkpoint} OUT=${3:?output dir}
export PATH=/usr/bin:/bin
export TORCH_HOME=/nfs/hpc/share/sanchej7/.cache/torch XDG_CACHE_HOME=/nfs/hpc/share/sanchej7/.cache
export HF_HOME=/nfs/hpc/share/sanchej7/.cache/huggingface MPLCONFIGDIR="/tmp/mpl-$SLURM_JOB_ID"
export PYTHONUNBUFFERED=1
PY=/nfs/hpc/share/sanchej7/miniforge3/envs/depth-env/bin/python
REPO=/nfs/hpc/share/sanchej7/spur-depth-service
CACHE=/nfs/hpc/share/sanchej7/spur-branch-cache/v1
SRC=$(mktemp -d "/tmp/${USER}-bpredict-${SLURM_JOB_ID}-XXXX")
trap 'rm -rf -- "$SRC"' EXIT
git clone -q --shared "$REPO" "$SRC/src"
git -C "$SRC/src" checkout -q "$COMMIT"
mkdir -p "$OUT" "$MPLCONFIGDIR"
echo "spur-bpredict $CKPT -> $OUT commit $COMMIT on $(hostname) $(date -Is): ${*:4}"
nvidia-smi -L || true
PYTHONPATH="$SRC/src" "$PY" "$SRC/src/scripts/train_branches.py" predict \
    --ckpt "$CKPT" --cache "$CACHE" --out "$OUT" "${@:4}"
echo "done $(date -Is)"
