"""PSI / KS drift scoring against a committed baseline.

Warn at PSI > 0.1, alert at PSI > 0.25. The baseline shipped with this
slice is computed from the 6 fixture views — ``Data/full_spur`` is not
mounted, so this is a contract test, not a 24k-frame population. Recompute
with ``python -m spur_depth.serve.drift --from-dir ...`` when the dataset
is restored.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from spur_depth.serve.input_stats import request_stats

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = Path(__file__).with_name("baseline_stats.json")
PSI_WARN = 0.1
PSI_ALERT = 0.25
WINDOW = 500
FEATURE_KEYS = (
    "luminance_mean",
    "luminance_std",
    "focus_var",
    "saturated_frac",
    "depth_p50",
    "oor_frac",
)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index of ``actual`` vs ``expected``.

    Equal-width bins over the combined range. Empty bins get ``eps`` so a
    missing category does not explode.
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if expected.size == 0 or actual.size == 0:
        return float("nan")
    lo = float(min(expected.min(), actual.min()))
    hi = float(max(expected.max(), actual.max()))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e = e_hist.astype(np.float64) / max(e_hist.sum(), 1)
    a = a_hist.astype(np.float64) / max(a_hist.sum(), 1)
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def ks_d(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic D (no p-value)."""
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if x.size == 0 or y.size == 0:
        return float("nan")
    data_all = np.concatenate([x, y])
    cdf1 = np.searchsorted(x, data_all, side="right") / x.size
    cdf2 = np.searchsorted(y, data_all, side="right") / y.size
    return float(np.max(np.abs(cdf1 - cdf2)))


def load_baseline(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_BASELINE
    if not p.is_file():
        return {"features": {}, "n_frames": 0, "source": "missing"}
    return json.loads(p.read_text())


class DriftMonitor:
    """Rolling window of request stats vs a frozen baseline."""

    def __init__(
        self,
        baseline: Mapping[str, Any] | None = None,
        *,
        window: int = WINDOW,
        log_path: str | Path | None = None,
    ) -> None:
        self.baseline = dict(baseline or load_baseline())
        self.window = window
        self._lock = threading.Lock()
        self._buf: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._n = 0
        env_log = os.environ.get("SPUR_REQUEST_LOG", "")
        self.log_path = Path(log_path) if log_path else (Path(env_log) if env_log else None)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def observe(self, stats: Mapping[str, Any]) -> dict[str, Any]:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **{k: stats[k] for k in stats if k != "ts"},
        }
        with self._lock:
            for k in FEATURE_KEYS:
                if k in stats and isinstance(stats[k], (int, float)):
                    self._buf[k].append(float(stats[k]))
            self._n += 1
            snapshot = self._score_unlocked()
            if self.log_path is not None:
                with self.log_path.open("a") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")
        return snapshot

    def _score_unlocked(self) -> dict[str, Any]:
        features = self.baseline.get("features") or {}
        out: dict[str, Any] = {"n_logged": self._n, "window": self.window, "features": {}}
        any_warn = False
        any_alert = False
        for k in FEATURE_KEYS:
            expected = np.asarray(features.get(k, {}).get("values", []), dtype=np.float64)
            actual = np.asarray(list(self._buf[k]), dtype=np.float64)
            if expected.size == 0 or actual.size == 0:
                continue
            p = psi(expected, actual)
            d = ks_d(expected, actual)
            warn = bool(p > PSI_WARN)
            alert = bool(p > PSI_ALERT)
            any_warn = any_warn or warn
            any_alert = any_alert or alert
            out["features"][k] = {
                "psi": p,
                "ks_d": d,
                "warn": warn,
                "alert": alert,
                "n_window": int(actual.size),
            }
        out["warn"] = any_warn
        out["alert"] = any_alert
        return out

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._score_unlocked()


def baseline_from_stats(rows: Iterable[Mapping[str, Any]], source: str) -> dict[str, Any]:
    collected: dict[str, list[float]] = defaultdict(list)
    n = 0
    for row in rows:
        n += 1
        for k in FEATURE_KEYS:
            if k in row:
                collected[k].append(float(row[k]))
    features = {}
    for k, vals in collected.items():
        arr = np.asarray(vals, dtype=np.float64)
        features[k] = {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std()) if arr.size else 0.0,
            "values": [float(v) for v in vals],
        }
    return {"n_frames": n, "source": source, "features": features}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-dir", type=Path, help="directory of RGB PNGs (writes baseline JSON)")
    ap.add_argument(
        "--from-data-root",
        type=Path,
        help="restored Data/full_spur root; uses real DA2-ft depths, not the 2.0 m placeholder",
    )
    ap.add_argument(
        "--holdout-only", action="store_true", help="with --from-data-root, only paper val trees"
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_BASELINE)
    args = ap.parse_args(argv)
    if args.from_dir is None and args.from_data_root is None:
        print(json.dumps(load_baseline(), indent=2))
        return 0
    from PIL import Image

    rows = []
    source = ""
    if args.from_data_root is not None:
        from spur_depth.data.restore_index import PAPER_VAL, discover_da2_triples, split_holdout

        discovered = discover_da2_triples(args.from_data_root)
        if args.holdout_only:
            _, discovered = split_holdout(discovered)
        for rec in discovered:
            if rec["rgb"] is None:
                continue
            img = Image.open(rec["rgb"])
            depth = np.load(rec["pred"]).astype(np.float32)
            depth[depth >= 1e9] = 0.0
            rows.append(request_stats(images=img, depths=depth))
        source = f"full_spur restore n={len(rows)} real DA2-ft"
        if args.holdout_only:
            source += f" holdout {PAPER_VAL}"
    else:
        pngs = sorted(args.from_dir.glob("view_*.png")) or sorted(
            p for p in args.from_dir.glob("*.png") if "mask" not in p.name.lower()
        )
        for p in pngs:
            img = Image.open(p)
            # Placeholder depth at the training mid-range so oor_frac is defined.
            depth = np.full((img.size[1], img.size[0]), 2.0, dtype=np.float32)
            rows.append(request_stats(images=img, depths=depth))
        source = str(args.from_dir)
    payload = baseline_from_stats(rows, source=source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out} n_frames={payload['n_frames']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
