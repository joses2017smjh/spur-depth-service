#!/bin/bash
#SBATCH -J spur-bcache
#SBATCH -p share
#SBATCH --array=0-9
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:30:00
#SBATCH -o /nfs/hpc/share/sanchej7/spur-branch-cache/logs/%x-%A_%a.out

# Branch label cache (scripts/build_branch_cache.py): array task i builds shard i of 10
# (trees k % 10 == i, 600 frames) with 4 workers. Every task runs from a frozen clone at
# $COMMIT, so edits to the live repo cannot leak into a running array. Submit with:
#   env -u SLURM_JOB_ID ... sbatch --export=NONE,COMMIT=<sha> scripts/run_build_branch_cache.sh
# The log directory above must exist before sbatch.

set -euo pipefail
export PATH=/usr/bin:/bin
: "${COMMIT:?set COMMIT=<sha>: sbatch --export=NONE,COMMIT=<sha> ...}"
: "${SLURM_ARRAY_TASK_ID:?run as an sbatch array task}"

REPO=/nfs/hpc/share/sanchej7/spur-depth-service
BASE=/nfs/hpc/share/sanchej7/spur-branch-cache
PY=/nfs/hpc/share/sanchej7/miniforge3/envs/depth-env/bin/python
COMMIT=$(git -C "$REPO" rev-parse --verify --quiet "${COMMIT}^{commit}") ||
    { echo "error: COMMIT is not a commit in $REPO" >&2; exit 2; }
SRC=$BASE/src-$COMMIT
mkdir -p "$BASE"

export TORCH_HOME=/nfs/hpc/share/sanchej7/.cache/torch
export HF_HOME=/nfs/hpc/share/sanchej7/.cache/huggingface
export XDG_CACHE_HOME=/nfs/hpc/share/sanchej7/.cache
export MPLCONFIGDIR=/tmp/mpl-$SLURM_JOB_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
ulimit -d 15000000
mkdir -p "$MPLCONFIGDIR"

# Concurrent array tasks may all find no clone: each clones to its own temp dir and
# renames it into place; rename fails on an existing non-empty dir, so losers discard theirs.
if [ ! -d "$SRC" ]; then
    tmp=$(mktemp -d "$SRC.tmp.XXXXXX")
    git clone -q --shared "$REPO" "$tmp"
    git -C "$tmp" checkout -q "$COMMIT"
    mv -T "$tmp" "$SRC" 2>/dev/null || rm -rf -- "$tmp"
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" ] ||
    { echo "error: $SRC is not at $COMMIT" >&2; exit 2; }
for f in spur_depth/branches/gt.py spur_depth/branches/tree.py scripts/build_branch_cache.py; do
    [ -f "$SRC/$f" ] || { echo "error: $f is not in commit $COMMIT; commit it first" >&2; exit 2; }
done

cd "$SRC"
echo "spur-bcache shard $SLURM_ARRAY_TASK_ID/10 commit $COMMIT on $(hostname) at $(date -Is)"
PYTHONPATH=$SRC "$PY" scripts/build_branch_cache.py \
    --shard "$SLURM_ARRAY_TASK_ID" --n-shards 10 --workers 4
echo "finished at $(date -Is)"
