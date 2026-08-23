"""Fan-out experiment tracking: W&B, MLflow, and a local JSONL fallback.

None of the backends is required to import this module. Missing extras
become no-ops so trainers and the service can share the same calls.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class Tracker:
    """``log_params`` / ``log_metrics`` / ``log_artifact`` to every backend."""

    def __init__(
        self,
        *,
        jsonl_path: str | Path | None = None,
        wandb_run: Any | None = None,
        mlflow_run: bool = False,
    ) -> None:
        self._jsonl = Path(jsonl_path) if jsonl_path else None
        if self._jsonl:
            self._jsonl.parent.mkdir(parents=True, exist_ok=True)
        self._wandb = wandb_run
        self._mlflow = mlflow_run
        if mlflow_run:
            try:
                import mlflow  # noqa: F401
            except ImportError as exc:
                raise ImportError("mlflow extra is not installed") from exc

    def _append(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self._jsonl is None:
            return
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **dict(payload),
        }
        with self._jsonl.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._append("params", {"params": dict(params)})
        if self._wandb is not None:
            self._wandb.config.update(dict(params), allow_val_change=True)
        if self._mlflow:
            import mlflow

            mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        row: dict[str, Any] = {"metrics": dict(metrics)}
        if step is not None:
            row["step"] = step
        self._append("metrics", row)
        if self._wandb is not None:
            self._wandb.log(dict(metrics), step=step)
        if self._mlflow:
            import mlflow

            mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        p = Path(path)
        self._append("artifact", {"path": str(p), "name": name or p.name})
        if self._wandb is not None:
            import wandb

            art = wandb.Artifact(name or p.stem, type="file")
            art.add_file(str(p))
            self._wandb.log_artifact(art)
        if self._mlflow:
            import mlflow

            mlflow.log_artifact(str(p))


def default_tracker(jsonl_path: str | Path | None = None) -> Tracker:
    """JSONL always; W&B if a run is already active; MLflow if ``SPUR_MLFLOW=1``."""
    wandb_run = None
    try:
        import wandb

        if wandb.run is not None:
            wandb_run = wandb.run
    except ImportError:
        pass
    mlflow_on = os.environ.get("SPUR_MLFLOW", "") in {"1", "true", "TRUE"}
    if jsonl_path is None:
        jsonl_path = os.environ.get("SPUR_TRACK_JSONL", "")
    return Tracker(
        jsonl_path=jsonl_path or None,
        wandb_run=wandb_run,
        mlflow_run=mlflow_on,
    )
