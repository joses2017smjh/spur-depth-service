"""JSONL backend works without wandb/mlflow installed."""

from __future__ import annotations

import json
from pathlib import Path

from spur_depth.tracking import Tracker


def test_jsonl_params_metrics_artifact(tmp_path: Path):
    log = tmp_path / "track.jsonl"
    art = tmp_path / "card.json"
    art.write_text("{}")
    t = Tracker(jsonl_path=log)
    t.log_params({"n_views": 6, "use_pose": False})
    t.log_metrics({"val/rmse": 0.0443}, step=23)
    t.log_artifact(art)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    kinds = [r["kind"] for r in rows]
    assert kinds == ["params", "metrics", "artifact"]
    assert rows[0]["params"]["n_views"] == 6
    assert rows[1]["metrics"]["val/rmse"] == 0.0443
    assert rows[1]["step"] == 23
