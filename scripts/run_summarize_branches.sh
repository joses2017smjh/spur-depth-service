#!/bin/bash
#SBATCH -J spur-bsum
#SBATCH -p share
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /nfs/hpc/share/sanchej7/spur-branch-eval/logs/%x-%j.out

# summary.json for rows written by run_eval_branches.sh, from the SAME frozen clone
# (src-<sha>), so the scorer fingerprints match:
#   sbatch --export=NONE scripts/run_summarize_branches.sh <rows-sha> <split> <out> \
#       <cfg-json-file> <pred-dir|none> <method,method> <depth,depth>

set -euo pipefail
COMMIT=$1 SPLIT=$2 OUT=$3 CFG_FILE=$4 PRED=$5
METHODS=${6//,/ } DEPTHS=${7//,/ }
export PATH=/usr/bin:/bin OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl-$SLURM_JOB_ID
PY=/nfs/hpc/share/sanchej7/miniforge3/envs/depth-env/bin/python
SRC=/nfs/hpc/share/sanchej7/spur-branch-eval/src-$COMMIT
[ -d "$SRC" ] || { echo "no frozen clone $SRC"; exit 2; }
PRED_ARG=()
[ "$PRED" != "none" ] && PRED_ARG=(--pred-dir "$PRED")
cd "$SRC"
# shellcheck disable=SC2086
PYTHONPATH=$SRC "$PY" scripts/eval_branches.py --split "$SPLIT" --methods $METHODS \
    --depths $DEPTHS --cfg "$(cat "$CFG_FILE")" "${PRED_ARG[@]}" --out "$OUT" --summarize-only
echo "summary $OUT/summary.json $(date -Is)"
