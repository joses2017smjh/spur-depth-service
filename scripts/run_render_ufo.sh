#!/bin/bash
#SBATCH -J spur-ufo-render
#SBATCH -p dgx2,ampere,gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:30:00
#SBATCH -o /nfs/hpc/share/sanchej7/spur-ufo/logs/%x-%A_%a.out

# Render synthetic UFO trees (L-Py lpy_ufo_*) exactly as the Envy set was rendered
# (Computer_Vision/generate_orchard_bark02.sh: generate_tree2.py, bark_brown_02, rigs
# box + box_cam1-4, 6 heights x stereo l/r = 60 frames, then DA2-ft depth), with one change:
# the generator's hierarchy walk also keeps "tertiarybranch_" parts (UFO laterals), which it
# would otherwise drop with their spurs (scripts/blender/generate_tree2_ufo.patch).
#
#   sbatch --export=NONE --array=0-0 scripts/run_render_ufo.sh <commit> <tree-list-file>
#
# Array task i renders line i of the tree-list file. Computer_Vision is read-only: the
# generator is copied (sha256 checked) and patched on node-local disk, Blender's user
# config and Python byte-code stay off it, and outputs go to OUT on hpc-share.

set -euo pipefail
COMMIT=${1:?commit} LIST=${2:?tree list file}
export PATH=/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CV=/nfs/hpc/share/sanchej7/Computer_Vision
REPO=/nfs/hpc/share/sanchej7/spur-depth-service
OUT=/nfs/hpc/share/sanchej7/spur-ufo/full_spur_ufo
BLENDER=/usr/local/apps/blender/4.2.19/blender
PY=/nfs/hpc/share/sanchej7/miniforge3/envs/depth-env/bin/python
GEN_SHA=2dedbbc1dd8bd8e720668ed21d8720ceb196215cdc2af542ad9dd4321495f324
BLEND_SHA=88352362755f4280aee3f5be2b7c7b2fc4d8892771384fbb4fa02c26e9fe0fb4
CKPT=$CV/checkpoints/full_spur_2tex_all_3view_seed1/best.pth
BARK=bark_brown_02
NBOX=4
EXPECTED=$(( (1 + NBOX) * 6 * 2 ))

TREE=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" "$LIST")
[ -n "$TREE" ] || { echo "no tree on line $SLURM_ARRAY_TASK_ID of $LIST"; exit 2; }
echo "===== spur-ufo-render $TREE task $SLURM_ARRAY_TASK_ID on $(hostname) $(date -Is) commit $COMMIT"
nvidia-smi -L || true

if [ "$(find "$OUT/Da2Finetune/$BARK/$TREE" -name '*.npy' 2>/dev/null | wc -l)" -ge "$EXPECTED" ] \
    && [ -f "$OUT/manifests/$TREE.json" ]; then
    echo "[SKIP] $TREE already complete"
    exit 0
fi

# Storage preflight: the project's 2 TiB hard limit is the refusal line; keep 100 GiB clear.
used_kb=$(lfs quota -p 30762 /nfs/hpc/share | awk 'NR==3 {gsub(/\*/, "", $2); print $2}')
if [ -z "$used_kb" ] || [ "$used_kb" -gt $(( (2048 - 100) * 1024 * 1024 )) ]; then
    echo "refusing: project 30762 uses ${used_kb:-?} KiB (limit 2 TiB minus 100 GiB)"
    exit 3
fi

# Frozen inputs: the patch from the committed source, the generator by hash.
SRC=$(mktemp -d "/tmp/${USER}-ufo-${SLURM_JOB_ID}-XXXX")
trap 'rm -rf -- "$SRC"' EXIT
git -C "$REPO" show "$COMMIT:scripts/blender/generate_tree2_ufo.patch" > "$SRC/gen.patch"
cp "$CV/Dataloader/generate_tree2.py" "$SRC/generate_tree2.py"
echo "$GEN_SHA  $SRC/generate_tree2.py" | sha256sum -c -
echo "$BLEND_SHA  $CV/orchard_template.blend" | sha256sum -c -
patch -s "$SRC/generate_tree2.py" < "$SRC/gen.patch"
grep -q 'tertiarybranch_' "$SRC/generate_tree2.py"

LOCAL="$SRC/out"
mkdir -p "$LOCAL" "$SRC/tmp" "$SRC/cfg" "$SRC/scripts" "$OUT/manifests"
export COMPUTER_VISION_ROOT=$CV CV_OUTPUT_DIR=$LOCAL CV_TREE_ID_FILTER=ufo TREE_ID=$TREE
export BARK_NAME=$BARK CV_SKIP_CAM=1 CV_NUM_BOX_CAM_POSES=$NBOX CV_FORCE_RENDER=1
export BLENDER_USER_CONFIG=$SRC/cfg BLENDER_USER_SCRIPTS=$SRC/scripts TMPDIR=$SRC/tmp
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

t0=$(date +%s)
echo "===== STAGE 1 blender $(date -Is)"
"$BLENDER" -b "$CV/orchard_template.blend" -P "$SRC/generate_tree2.py"
t1=$(date +%s)
n_of=$(find "$LOCAL/Optical_flow/$BARK/$TREE" -name '*.png' 2>/dev/null | wc -l)
echo "Optical_flow frames: $n_of (expected $EXPECTED)"
[ "$n_of" -ge "$EXPECTED" ] || { echo "blender wrote $n_of / $EXPECTED frames"; exit 4; }

echo "===== STAGE 2 DA2-ft $(date -Is)"
PYTHONPATH="$CV/depth-anything-v2:$CV/depth-anything-v2/metric_depth:$CV" \
HF_HOME=/nfs/hpc/share/sanchej7/.cache/huggingface TORCH_HOME=/nfs/hpc/share/sanchej7/.cache/torch \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XFORMERS_DISABLED=1 MPLCONFIGDIR=$SRC/tmp \
"$PY" "$CV/infer_monocular_tree.py" --data-root "$LOCAL" --bark "$BARK" --tree "$TREE" \
    --models da2ft --ckpt "$CKPT" --expected-of "$EXPECTED"
t2=$(date +%s)
n_da2=$(find "$LOCAL/Da2Finetune/$BARK/$TREE" -name '*.npy' | wc -l)
[ "$n_da2" -ge "$EXPECTED" ] || { echo "DA2 wrote $n_da2 / $EXPECTED"; exit 5; }

echo "===== copy-back $(date -Is)"
for kind in Da2Finetune depth mask ann box_mask Optical_flow; do
    if [ -d "$LOCAL/$kind/$BARK/$TREE" ]; then
        mkdir -p "$OUT/$kind/$BARK/$TREE"
        rsync -a "$LOCAL/$kind/$BARK/$TREE/" "$OUT/$kind/$BARK/$TREE/"
    fi
done
mkdir -p "$OUT/cylinders_world/$BARK"
cp "$LOCAL/cylinders_world/$BARK/$TREE.json" "$OUT/cylinders_world/$BARK/"
"$PY" - "$OUT/manifests/$TREE.json" <<EOF
import json, sys
json.dump({
    "tree": "$TREE", "commit": "$COMMIT", "job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
    "host": "$(hostname)", "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
    "generator_sha256": "$GEN_SHA", "template_sha256": "$BLEND_SHA",
    "patch_sha256": "$(sha256sum "$SRC/gen.patch" | cut -c1-64)",
    "patched_generator_sha256": "$(sha256sum "$SRC/generate_tree2.py" | cut -c1-64)",
    "da2_ckpt": "$CKPT", "bark": "$BARK", "rigs": ["box"] + [f"box_cam{i}" for i in range(1, $NBOX + 1)],
    "frames": $n_of, "da2_maps": $n_da2,
    "blender_s": $(( t1 - t0 )), "da2_s": $(( t2 - t1 )),
}, open(sys.argv[1], "w"), indent=2)
EOF
du -sh "$OUT/depth/$BARK/$TREE" "$OUT/Da2Finetune/$BARK/$TREE" "$OUT/Optical_flow/$BARK/$TREE"
echo "===== done $TREE $(date -Is) blender $(( t1 - t0 ))s da2 $(( t2 - t1 ))s"
