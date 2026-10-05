#!/bin/bash
#SBATCH -J spur-branches
#SBATCH -p ampere,dgxh,gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:50:00
#SBATCH -o /nfs/hpc/share/sanchej7/spur-branch-runs/%x-%j.out

# Train BranchNet from a frozen commit, then predict the val + paper-test trees.
#
#   sbatch --export=NONE,COMMIT=<sha> scripts/run_train_branches.sh
#   sbatch --export=NONE scripts/run_train_branches.sh <sha> [max_minutes] [train args...]
#
# Arguments after max_minutes go to `train_branches.py train` unchanged, e.g. a warm
# start: <sha> 80 --init /nfs/hpc/share/sanchej7/spur-branch-runs/<job>/best.pt --lr 2e-4
#
# `man sbatch` (25.11) says NONE accepts no explicit variables; if COMMIT does not
# arrive that way, the positional form always works. The source is a shared clone
# checked out at COMMIT, so later edits to the working tree cannot leak into the run.
# Budget: training (incl. per-epoch validation) stops itself after MAX_MIN minutes;
# prediction of 6 trees x 3 depth sources follows within the 110-minute limit.

set -euo pipefail

COMMIT="${COMMIT:-${1:-}}"
MAX_MIN="${MAX_MIN:-${2:-75}}"
if [ -z "$COMMIT" ]; then
    echo "COMMIT is required (--export=NONE,COMMIT=<sha> or first argument)" >&2
    exit 2
fi

export PATH=/usr/bin:/bin
export TORCH_HOME=/nfs/hpc/share/sanchej7/.cache/torch
export XDG_CACHE_HOME=/nfs/hpc/share/sanchej7/.cache
export HF_HOME=/nfs/hpc/share/sanchej7/.cache/huggingface
export MPLCONFIGDIR="/tmp/mpl-$SLURM_JOB_ID"
export PYTHONUNBUFFERED=1

PY=/nfs/hpc/share/sanchej7/miniforge3/envs/depth-env/bin/python
REPO=/nfs/hpc/share/sanchej7/spur-depth-service
CACHE=/nfs/hpc/share/sanchej7/spur-branch-cache/v1
RUN="/nfs/hpc/share/sanchej7/spur-branch-runs/$SLURM_JOB_ID"

mkdir -p "$RUN" "$MPLCONFIGDIR"
git clone -q --shared "$REPO" "$RUN/src"
git -C "$RUN/src" checkout -q "$COMMIT"
export PYTHONPATH="$RUN/src"

echo "===== spur-branches job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "commit $(git -C "$RUN/src" rev-parse HEAD) (requested $COMMIT), max train minutes $MAX_MIN"
echo "extra train args: ${*:3}"
nvidia-smi || true

"$PY" "$RUN/src/scripts/train_branches.py" train \
    --cache "$CACHE" --out "$RUN" --max-minutes "$MAX_MIN" "${@:3}"
echo "===== train done at $(date -Is)"

PRED_EXTRA=()
for a in "${@:3}"; do
    case "$a" in --ufo | --orchard) PRED_EXTRA+=("$a") ;; esac
done
"$PY" "$RUN/src/scripts/train_branches.py" predict \
    --ckpt "$RUN/best.pt" --cache "$CACHE" --out "$RUN/pred" "${PRED_EXTRA[@]}"
echo "===== predict done at $(date -Is)"

date -Is > "$RUN/DONE"
