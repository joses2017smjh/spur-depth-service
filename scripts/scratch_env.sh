# Node-local scratch for engines / Apptainer / YOLO. Source from SLURM jobs.
# Do not put SIF images or TensorRT workspaces on /nfs/stak or hpc-share.
spur_scratch() {
  local user="${USER:-sanchej7}"
  local job="${SLURM_JOB_ID:-local}"
  if [ -n "${SPUR_SCRATCH:-}" ]; then
    mkdir -p "$SPUR_SCRATCH"
    printf '%s\n' "$SPUR_SCRATCH"
    return 0
  fi
  local d
  for d in \
      "/scratch/${user}/spur/${job}" \
      "/raid/${user}/spur/${job}" \
      "${TMPDIR:-/tmp}/spur_${job}" \
      "/tmp/${user}_spur_${job}"; do
    if mkdir -p "$d" 2>/dev/null && touch "$d/.spur_write" 2>/dev/null; then
      rm -f "$d/.spur_write"
      printf '%s\n' "$d"
      return 0
    fi
  done
  d="/tmp/${user}_spur_${job}"
  mkdir -p "$d"
  printf '%s\n' "$d"
}

spur_export_scratch() {
  export SPUR_SCRATCH="$(spur_scratch)"
  export TMPDIR="${SPUR_SCRATCH}/tmp"
  export PIP_CACHE_DIR="${SPUR_SCRATCH}/pip"
  export XDG_CACHE_HOME="${SPUR_SCRATCH}/xdg"
  export APPTAINER_CACHEDIR="${SPUR_SCRATCH}/apptainer"
  export APPTAINER_TMPDIR="${SPUR_SCRATCH}/apptainer-tmp"
  export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
  export SPUR_ENGINE_DIR="${SPUR_SCRATCH}/engines"
  export SPUR_ARTIFACT_DIR="${SPUR_SCRATCH}/out"
  mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" \
           "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" \
           "$SPUR_ENGINE_DIR" "$SPUR_ARTIFACT_DIR"
}
