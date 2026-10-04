#!/bin/bash
#SBATCH -J spur-proxy
#SBATCH -p share
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 02:30:00
#SBATCH -o /nfs/hpc/share/sanchej7/spur-branch-eval/logs/%x-%j.out

# Isaac cut-success CPU proxy (scripts/isaac_cut_proxy.py) from a frozen commit:
#   sbatch --export=NONE scripts/run_isaac_cut_proxy.sh <sha> <split> <out> <cfg-json-file> \
#       <pred-dir|none> <method,method> <depth,depth>
# One output directory per configuration: the result is <out>/<split>.json and the graph
# cache <out>/graphs. The registered truth length is read from CAL below.

set -euo pipefail
export PATH=/usr/bin:/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl-$SLURM_JOB_ID
export TORCH_HOME=/nfs/hpc/share/sanchej7/.cache/torch XDG_CACHE_HOME=/nfs/hpc/share/sanchej7/.cache
COMMIT=$1 SPLIT=$2 OUT=$3 CFG_FILE=$4 PRED=$5
# Comma-separated lists: sbatch re-splits quoted arguments.
METHODS=${6//,/ } DEPTHS=${7//,/ }
REPO=/nfs/hpc/share/sanchej7/spur-depth-service
PY=/nfs/hpc/share/sanchej7/miniforge3/envs/depth-env/bin/python
SRC=/nfs/hpc/share/sanchej7/spur-branch-eval/src-$COMMIT
CAL=/nfs/hpc/share/sanchej7/spur-branch-eval/isaac_proxy/length_calibration.json
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
echo "spur-proxy $SPLIT $METHODS / $DEPTHS commit $COMMIT on $(hostname) $(date -Is)"
cd "$SRC"
ulimit -d 10000000
# shellcheck disable=SC2086
PYTHONPATH=$SRC "$PY" scripts/isaac_cut_proxy.py --split "$SPLIT" --methods $METHODS \
    --depths $DEPTHS --cfg "$(cat "$CFG_FILE")" "${PRED_ARG[@]}" --out "$OUT" --calibration "$CAL"
echo "finished $(date -Is)"
