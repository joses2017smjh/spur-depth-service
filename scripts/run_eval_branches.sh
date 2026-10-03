#!/bin/bash
#SBATCH -J spur-beval
#SBATCH -p share
#SBATCH -c 2
#SBATCH --mem=10G
#SBATCH -t 02:00:00
#SBATCH -o /nfs/hpc/share/sanchej7/spur-branch-eval/logs/%x-%A_%a.out

# Score branch perception (scripts/eval_branches.py) from a frozen commit: array task i
# scores frames i, i+N, ... of the split. Rows land in $OUT/frames; run
#   eval_branches.py --summarize-only --out $OUT --split $SPLIT --cfg "$CFG"
# once every task has finished. Submit (N = array size):
#   sbatch --export=NONE --array=0-11 scripts/run_eval_branches.sh <sha> <split> <out> \
#       <cfg-json-file> <pred-dir|none> <method,method> <depth,depth>

set -euo pipefail
export PATH=/usr/bin:/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl-$SLURM_JOB_ID
export TORCH_HOME=/nfs/hpc/share/sanchej7/.cache/torch XDG_CACHE_HOME=/nfs/hpc/share/sanchej7/.cache
COMMIT=$1 SPLIT=$2 OUT=$3 CFG_FILE=$4 PRED=$5
# Method and depth lists are comma-separated: sbatch re-splits quoted arguments.
METHODS=${6//,/ } DEPTHS=${7//,/ }
N=${SLURM_ARRAY_TASK_COUNT:?run as an array}
REPO=/nfs/hpc/share/sanchej7/spur-depth-service
PY=/nfs/hpc/share/sanchej7/miniforge3/envs/depth-env/bin/python
SRC=/nfs/hpc/share/sanchej7/spur-branch-eval/src-$COMMIT
mkdir -p "$(dirname "$SRC")" "$OUT"
if [ ! -d "$SRC" ]; then
    tmp=$(mktemp -d "$SRC.tmp.XXXXXX")
    git clone -q --shared "$REPO" "$tmp"
    git -C "$tmp" checkout -q "$COMMIT"
    mkdir -p "$tmp/weights" && cp "$REPO/weights/trunk_unet_100tree.pt" "$tmp/weights/"
    mv -T "$tmp" "$SRC" 2>/dev/null || rm -rf -- "$tmp"
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$(git -C "$REPO" rev-parse "$COMMIT")" ]
PRED_ARG=()
[ "$PRED" != "none" ] && PRED_ARG=(--pred-dir "$PRED")
echo "spur-beval $SPLIT shard $SLURM_ARRAY_TASK_ID/$N commit $COMMIT on $(hostname) $(date -Is)"
cd "$SRC"
ulimit -d 9000000
# shellcheck disable=SC2086
PYTHONPATH=$SRC "$PY" scripts/eval_branches.py --split "$SPLIT" --methods $METHODS \
    --depths $DEPTHS --cfg "$(cat "$CFG_FILE")" "${PRED_ARG[@]}" --out "$OUT" \
    --shard "$SLURM_ARRAY_TASK_ID" --n-shards "$N"
echo "finished $(date -Is)"
