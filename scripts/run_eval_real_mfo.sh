#!/bin/bash
#SBATCH -J spur-mfo
#SBATCH -p ampere,dgxh,gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -o /nfs/hpc/share/sanchej7/spur-realdata/logs/%x-%j.out

# Real MFO cherry frames, zero-shot, for one checkpoint, from a frozen commit:
#   sbatch --export=NONE scripts/run_eval_real_mfo.sh <sha> <ckpt> <root>
# <root> gets results/ and a symlink mfo_cherry_ufo -> the shared, hash-manifested data.
# Selection on MFO val first (eval_real_mfo.py select), then test and train (score);
# scripts/rescore_mfo_mapping.py then reads the saved counts under either class mapping.

set -euo pipefail
COMMIT=${1:?commit} CKPT=${2:?checkpoint} ROOT=${3:?results root}
export PATH=/usr/bin:/bin PYTHONUNBUFFERED=1
export TORCH_HOME=/nfs/hpc/share/sanchej7/.cache/torch XDG_CACHE_HOME=/nfs/hpc/share/sanchej7/.cache
export MPLCONFIGDIR="/tmp/mpl-$SLURM_JOB_ID"
PY=/nfs/hpc/share/sanchej7/miniforge3/envs/depth-env/bin/python
REPO=/nfs/hpc/share/sanchej7/spur-depth-service
DATA=/nfs/hpc/share/sanchej7/spur-realdata/mfo_cherry_ufo
SRC=$(mktemp -d "/tmp/${USER}-mfo-${SLURM_JOB_ID}-XXXX")
trap 'rm -rf -- "$SRC"' EXIT
git clone -q --shared "$REPO" "$SRC/src"
git -C "$SRC/src" checkout -q "$COMMIT"
mkdir -p "$ROOT/results" "$MPLCONFIGDIR"
[ -e "$ROOT/mfo_cherry_ufo" ] || ln -s "$DATA" "$ROOT/mfo_cherry_ufo"
echo "spur-mfo $CKPT -> $ROOT commit $COMMIT on $(hostname) $(date -Is)"
cd "$SRC/src"
export PYTHONPATH="$SRC/src"
"$PY" scripts/eval_real_mfo.py --root "$ROOT" --ckpt "$CKPT" select
"$PY" scripts/eval_real_mfo.py --root "$ROOT" --ckpt "$CKPT" score --splits test train
for m in v2 ufo; do
    "$PY" scripts/rescore_mfo_mapping.py "$ROOT/results" --mapping $m \
        --out "$ROOT/results/mfo_rescore_$m.json"
done
echo "done $(date -Is)"
