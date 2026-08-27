"""NFS vs node-local scratch. Engines, caches, and YOLO runs stay off stak.

CoE home is 25 GB. hpc-share is the 1 TB scratch share and is already near
the ceiling. DGX ``/scratch`` (also ``/raid``) is job-local and the right
place for TensorRT builds, Apptainer layers, and detector training.

Set ``SPUR_SCRATCH`` to pin a directory. Jobs source ``scripts/scratch_env.sh``.
"""

from __future__ import annotations

import os
from pathlib import Path


def scratch_root() -> Path:
    """Writable job-local directory. Never the 25 GB stak home."""
    pinned = os.environ.get("SPUR_SCRATCH")
    candidates = []
    if pinned:
        candidates.append(Path(pinned))
    user = os.environ.get("USER", "sanchej7")
    job = os.environ.get("SLURM_JOB_ID", "local")
    tmp = os.environ.get("TMPDIR")
    candidates.extend(
        [
            Path("/scratch") / user / "spur" / job,
            Path(tmp) / f"spur_{job}" if tmp else None,
            Path("/tmp") / f"{user}_spur_{job}",
        ]
    )
    for cand in candidates:
        if cand is None:
            continue
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".spur_write"
            probe.write_text("ok")
            probe.unlink()
            return cand
        except OSError:
            continue
    fallback = Path("/tmp") / f"{user}_spur_{job}"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def engine_dir() -> Path:
    env = os.environ.get("SPUR_ENGINE_DIR")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return scratch_root() / "engines"


def artifact_dir() -> Path:
    env = os.environ.get("SPUR_ARTIFACT_DIR")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return scratch_root() / "out"
