#!/bin/bash
# GPU "clean machine" smoke via Apptainer. Docker is not usable on CoE DGX
# (no daemon access). The SIF and layer cache stay on node scratch.
#
#   source scripts/scratch_env.sh && spur_export_scratch
#   bash scripts/run_gpu_container.sh
#
# Exit 2 if the image cannot be built or pulled. Do not quote a GPU-docker
# latency unless bench/results/gpu_container_smoke.json says ok=true.

set -euo pipefail

SPUR_ROOT="${SPUR_ROOT:-/nfs/hpc/share/sanchej7/spur-depth-service}"
# shellcheck source=/dev/null
source "${SPUR_ROOT}/scripts/scratch_env.sh"
if [ -z "${SPUR_SCRATCH:-}" ]; then
  spur_export_scratch
fi

SIF="${SPUR_SCRATCH}/spur-depth-gpu.sif"
JSON="${SPUR_ARTIFACT_DIR:-${SPUR_ROOT}/bench/results}/gpu_container_smoke.json"
mkdir -p "$(dirname "$JSON")" "$SPUR_SCRATCH"

echo "[container] scratch=$SPUR_SCRATCH"
command -v apptainer >/dev/null || {
  echo '{"ok": false, "reason": "apptainer not on PATH"}' > "$JSON"
  exit 2
}

# Prefer the Apptainer def. Dockerfile.gpu is Docker syntax and fails as a .def
# (job 21015984). Build from $SPUR_ROOT so %files paths resolve.
cd "${SPUR_ROOT}"
DEF="${SPUR_ROOT}/apptainer/gpu.def"
if [ -f "$DEF" ]; then
  SRC="$DEF"
else
  SRC="${SPUR_ROOT}/Dockerfile.gpu"
fi
set +e
timeout 25m apptainer build --force "$SIF" "$SRC"
BUILD_RC=$?
set -e

if [ "$BUILD_RC" -ne 0 ]; then
  echo "[container] Dockerfile build failed ($BUILD_RC); trying base pytorch pull"
  set +e
  timeout 20m apptainer pull --force "$SIF" \
      docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
  PULL_RC=$?
  set -e
  if [ "${PULL_RC:-1}" -ne 0 ]; then
    printf '%s\n' "{\"ok\": false, \"reason\": \"apptainer build $BUILD_RC pull ${PULL_RC:-missing}\"}" > "$JSON"
    echo "wrote $JSON"
    exit 2
  fi
fi

set +e
OUT=$(apptainer exec --nv -B "${SPUR_ROOT}:/app" "$SIF" \
    python -c "import torch; print('cuda', torch.cuda.is_available())" 2>&1)
RC=$?
set -e
echo "$OUT"

ok=false
if [ "$RC" -eq 0 ] && echo "$OUT" | grep -q "cuda True"; then
  ok=true
fi

python - "$JSON" "$ok" "$OUT" "$SIF" <<'PY'
import json, sys
from pathlib import Path
path, ok, out, sif = sys.argv[1], sys.argv[2] == "true", sys.argv[3], sys.argv[4]
payload = {
    "ok": ok,
    "sif": sif,
    "sif_bytes": Path(sif).stat().st_size if Path(sif).is_file() else 0,
    "cuda_probe": out[-400:],
    "note": "Apptainer --nv smoke. Not a TensorRT millisecond. SIF stays on scratch.",
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n")
print("wrote", path)
raise SystemExit(0 if ok else 2)
PY
