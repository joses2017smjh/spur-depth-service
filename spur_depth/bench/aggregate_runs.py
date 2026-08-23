"""Build the README accuracy table from W&B, checkpoints, or a frozen JSON.

The dashboard and the paper then cannot disagree, because one generates the
other. Default source is the committed ``bench/results/seed_rmse.json``
(values read from the five ``best_epoch_*.pt`` files). ``--source wandb``
needs ``WANDB_API_KEY`` in the environment — never a script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_FROZEN = _REPO / "bench" / "results" / "seed_rmse.json"


def sample_mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 1:
        raise ValueError("no seeds")
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var)


def _rows_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    seeds = sorted(payload["seeds"], key=lambda r: int(r["seed"]))
    rmses = [float(r["best_rmse_m"]) for r in seeds]
    mean, std = sample_mean_std(rmses)
    return {
        "variant": payload.get("variant", ""),
        "dataset": payload.get("dataset", ""),
        "eval_mask": payload.get("eval_mask", "trunk, 0.5–10.0 m"),
        "source": payload.get("source", ""),
        "seeds": seeds,
        "mean_rmse_m": mean,
        "std_rmse_m": std,
    }


def from_frozen(path: Path = _FROZEN) -> dict[str, Any]:
    return _rows_from_payload(json.loads(path.read_text()))


def from_checkpoints(
    root: Path, variant: str = "dino_da2ft_3pair_fusion_nopose_spur"
) -> dict[str, Any]:
    """Read ``best_rmse`` out of ``best_epoch_*.pt`` under ``root``.

    Directory layout matches the HPC tree:
    ``{variant}_seedN/exp1/seed_0N/best_epoch_XXXX.pt``.
    """
    import torch

    seeds = []
    for seed in range(1, 6):
        d = root / f"{variant}_seed{seed}"
        pts = sorted(d.glob("**/best_epoch_*.pt"))
        if not pts:
            continue
        latest = pts[-1]
        blob = torch.load(latest, map_location="cpu", weights_only=False)
        epoch = int(blob.get("epoch", int(latest.stem.split("_")[-1])))
        rmse = float(blob["best_rmse"])
        seeds.append({"seed": seed, "epoch": epoch, "best_rmse_m": rmse, "path": str(latest)})
    if not seeds:
        raise FileNotFoundError(f"no {variant}_seedN checkpoints under {root}")
    return _rows_from_payload(
        {
            "variant": variant,
            "dataset": "full_spur",
            "source": f"torch.load best_rmse under {root}",
            "seeds": seeds,
        }
    )


def from_wandb(
    project: str,
    *,
    entity: str | None = None,
    group: str = "dino_da2ft_3pair_fusion_nopose",
) -> dict[str, Any]:
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed; pip install wandb") from exc
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is not set (do not hard-code it)")

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    seeds = []
    for run in api.runs(path):
        cfg = {
            k: (v.get("value") if isinstance(v, dict) and "value" in v else v)
            for k, v in dict(run.config).items()
        }
        name = str(cfg.get("wandb_run_name") or run.name or "")
        grp = str(cfg.get("wandb_group") or getattr(run, "group", "") or "")
        if group not in name and group not in grp:
            continue
        rmse = (
            run.summary.get("val/rmse")
            or run.summary.get("best_rmse")
            or run.summary.get("val_rmse")
        )
        if rmse is None:
            continue
        seed = int(cfg.get("seed") or 0)
        seeds.append(
            {
                "seed": seed,
                "epoch": int(run.summary.get("epoch") or 0),
                "best_rmse_m": float(rmse),
                "run_id": run.id,
                "run_name": name,
            }
        )
    if not seeds:
        raise SystemExit(f"no runs matching group={group!r} in {path}")
    return _rows_from_payload(
        {
            "variant": group,
            "source": f"wandb {path}",
            "seeds": seeds,
        }
    )


def markdown_table(agg: dict[str, Any]) -> str:
    lines = [
        "| seed | epoch | best_rmse (m) |",
        "| --- | --- | --- |",
    ]
    for r in agg["seeds"]:
        epoch = r.get("epoch", "")
        lines.append(f"| {r['seed']} | {epoch} | {float(r['best_rmse_m']):.6f} |")
    lines.append(
        f"| **mean ± sample std** | | **{agg['mean_rmse_m']:.5f} ± {agg['std_rmse_m']:.5f}** |"
    )
    return "\n".join(lines)


def emit(agg: dict[str, Any], as_json: bool) -> str:
    if as_json:
        return json.dumps(agg, indent=2)
    header = (
        f"variant: {agg.get('variant')}\n"
        f"source:  {agg.get('source')}\n"
        f"mask:    {agg.get('eval_mask')}\n"
    )
    return header + markdown_table(agg) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("frozen", "checkpoints", "wandb"), default="frozen")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ckpt-root", type=Path, default=None)
    ap.add_argument("--wandb-project", default="spur-da2ft-depth-ablation")
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    ap.add_argument("--group", default="dino_da2ft_3pair_fusion_nopose")
    ap.add_argument("--frozen", type=Path, default=_FROZEN)
    args = ap.parse_args(argv)

    if args.source == "frozen":
        agg = from_frozen(args.frozen)
    elif args.source == "checkpoints":
        root = args.ckpt_root or Path(os.environ.get("SPUR_CKPT_ROOT", ""))
        if not root:
            print("--ckpt-root or SPUR_CKPT_ROOT is required", file=sys.stderr)
            return 2
        agg = from_checkpoints(root)
    else:
        agg = from_wandb(args.wandb_project, entity=args.wandb_entity, group=args.group)
    sys.stdout.write(emit(agg, args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
