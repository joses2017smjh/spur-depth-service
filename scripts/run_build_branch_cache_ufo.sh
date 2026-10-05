#!/bin/bash
#SBATCH -J spur-bcache-ufo
#SBATCH -p share
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:30:00
#SBATCH -o /nfs/hpc/share/sanchej7/spur-branch-cache/logs/%x-%A_%a.out

# Label cache for the rendered UFO trees (protocol v3), from a frozen commit, next to the
# Envy cache in v1/ with its own manifests (manifest_ufo_shardNN.json):
#   sbatch --export=NONE --array=0-9 scripts/run_build_branch_cache_ufo.sh <sha> <tree-list-file> [tag]
# (tag: manifest_<tag>shardNN.json, default "ufo_"; orchard keys use "orchard_")
# Array task i builds trees k % N == i of the list (N = array size), skipping complete frames.

set -euo pipefail
COMMIT=${1:?commit} LIST=${2:?tree list} TAG=${3:-ufo_}
export PATH=/usr/bin:/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl-$SLURM_JOB_ID
export TORCH_HOME=/nfs/hpc/share/sanchej7/.cache/torch XDG_CACHE_HOME=/nfs/hpc/share/sanchej7/.cache
N=${SLURM_ARRAY_TASK_COUNT:?run as an array}
REPO=/nfs/hpc/share/sanchej7/spur-depth-service
PY=/nfs/hpc/share/sanchej7/miniforge3/envs/depth-env/bin/python
SRC=$(mktemp -d "/tmp/${USER}-bcache-${SLURM_JOB_ID}-XXXX")
trap 'rm -rf -- "$SRC"' EXIT
git clone -q --shared "$REPO" "$SRC/src"
git -C "$SRC/src" checkout -q "$COMMIT"
mapfile -t TREES < "$LIST"
echo "spur-bcache-ufo shard $SLURM_ARRAY_TASK_ID/$N commit $COMMIT on $(hostname) $(date -Is)"
cd "$SRC/src"
ulimit -d 15000000
PYTHONPATH="$SRC/src" "$PY" scripts/build_branch_cache.py --trees "${TREES[@]}" \
    --shard "$SLURM_ARRAY_TASK_ID" --n-shards "$N" --workers 4 --manifest-tag "$TAG"
echo "finished $(date -Is)"
