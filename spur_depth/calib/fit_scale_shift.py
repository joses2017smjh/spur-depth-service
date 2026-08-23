"""Streaming least-squares scale/shift: calibrated = α · raw + β.

One constant-memory pass per image, accumulating ``n, Σx, Σy, Σxy, Σx²``
over valid pixels, then a 2×2 solve. Per-image moments are merged in
sorted-path order so a later OpenMP port can be bit-identical across
thread counts (parallelise files; reduce in file-index order).

Valid pixel: trunk mask eroded by ``erode_r``, finite pred/GT, GT in
``[min_depth, max_depth]``. An image is skipped when its GT std on those
pixels is below ``min_gt_std`` (the original filter: a flat board does
not constrain a scale/shift).

``Data/full_spur`` is not currently mounted. Without pairs this module
exits 2 rather than reprinting the training-time literals and calling
that a reproduction.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from spur_depth.calib import load_pro_config


def erode_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square-kernel binary erosion. ``radius=0`` is a no-op.

    Implemented with a False-padded AND-reduce so it matches the C++ port
    (out-of-bounds is invalid). OpenCV's default erode border is *not*
    that — ``morphologyDefaultBorderValue`` is +inf, so a full-ones mask
    would not shrink.
    """
    m = np.asarray(mask).astype(bool, copy=False)
    r = int(radius)
    if r <= 0:
        return m
    h, w = m.shape[-2], m.shape[-1]
    pad = np.pad(m, r, mode="constant", constant_values=False)
    out = np.ones((h, w), dtype=bool)
    k = 2 * r + 1
    for dr in range(k):
        for dc in range(k):
            out &= pad[dr : dr + h, dc : dc + w]
    return out


@dataclass(frozen=True)
class Moments:
    n: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    sxy: float = 0.0
    sx2: float = 0.0

    def merge(self, other: "Moments") -> "Moments":
        return Moments(
            n=self.n + other.n,
            sx=self.sx + other.sx,
            sy=self.sy + other.sy,
            sxy=self.sxy + other.sxy,
            sx2=self.sx2 + other.sx2,
        )


def moments_from_xy(x: np.ndarray, y: np.ndarray) -> Moments:
    """Five moments in float64. ``x`` is raw PRO, ``y`` is GT."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return Moments(
        n=float(x.size),
        sx=float(np.sum(x)),
        sy=float(np.sum(y)),
        sxy=float(np.sum(x * y)),
        sx2=float(np.sum(x * x)),
    )


def valid_pixels(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    *,
    erode_r: int,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
    valid = erode_binary(np.asarray(mask), erode_r)
    if valid.shape != pred.shape:
        raise ValueError(f"mask shape {valid.shape} != pred shape {pred.shape}")
    valid &= np.isfinite(pred) & np.isfinite(gt)
    valid &= (gt >= min_depth) & (gt <= max_depth)
    return valid


def moments_from_image(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    *,
    erode_r: int = 10,
    min_gt_std: float = 0.05,
    min_depth: float = 0.5,
    max_depth: float = 10.0,
) -> Moments | None:
    valid = valid_pixels(pred, gt, mask, erode_r=erode_r, min_depth=min_depth, max_depth=max_depth)
    n = int(valid.sum())
    if n < 2:
        return None
    y = np.asarray(gt, dtype=np.float64)[valid]
    if float(y.std()) < min_gt_std:
        return None
    x = np.asarray(pred, dtype=np.float64)[valid]
    return moments_from_xy(x, y)


def solve_scale_shift(m: Moments) -> tuple[float, float]:
    """Return (alpha, beta) for y ≈ alpha * x + beta.

    Raises ValueError on empty or singular systems.
    """
    if m.n < 2:
        raise ValueError("no valid pixels: empty or fully-masked input")
    det = m.sx2 * m.n - m.sx * m.sx
    if abs(det) < 1e-18:
        raise ValueError("singular normal equations (pred has no variance)")
    alpha = (m.sxy * m.n - m.sx * m.sy) / det
    beta = (m.sx2 * m.sy - m.sx * m.sxy) / det
    return float(alpha), float(beta)


def merge_in_order(parts: Iterable[Moments]) -> Moments:
    acc = Moments()
    for p in parts:
        acc = acc.merge(p)
    return acc


def fit_pairs(
    pairs: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    **kwargs,
) -> tuple[float, float, Moments]:
    parts: list[Moments] = []
    for pred, gt, mask in pairs:
        m = moments_from_image(pred, gt, mask, **kwargs)
        if m is not None:
            parts.append(m)
    acc = merge_in_order(parts)
    alpha, beta = solve_scale_shift(acc)
    return alpha, beta, acc


def _load_mask(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    from PIL import Image

    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != shape:
        raise ValueError(f"{path}: mask shape {arr.shape} != {shape}")
    return arr


def fit_files(
    triples: Sequence[tuple[Path, Path, Path]],
    **kwargs,
) -> tuple[float, float, Moments, int]:
    """Fit over on-disk triples ``(pred.npy, gt.npy, mask)``.

    Triples are sorted by pred path so C++ and Python agree on merge order.
    """
    ordered = sorted(triples, key=lambda t: str(t[0]))
    parts: list[Moments] = []
    used = 0
    for pred_p, gt_p, mask_p in ordered:
        pred = np.load(pred_p)
        gt = np.load(gt_p)
        mask = _load_mask(mask_p, pred.shape)
        m = moments_from_image(pred, gt, mask, **kwargs)
        if m is None:
            continue
        parts.append(m)
        used += 1
    acc = merge_in_order(parts)
    alpha, beta = solve_scale_shift(acc)
    return alpha, beta, acc, used


def _read_pairs_csv(path: Path) -> list[tuple[Path, Path, Path]]:
    rows = []
    with path.open() as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if not row or row[0].startswith("#"):
                continue
            if i == 0 and row[0].lower() in {"pred", "pro", "x"}:
                continue
            if len(row) < 3:
                raise ValueError(f"{path}:{i + 1}: expected pred,gt,mask")
            rows.append((Path(row[0].strip()), Path(row[1].strip()), Path(row[2].strip())))
    return rows


def main(argv: list[str] | None = None) -> int:
    cfg = load_pro_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-csv", type=Path, help="CSV of pred.npy,gt.npy,mask")
    ap.add_argument("--erode-r", type=int, default=int(cfg["erode_r"]))
    ap.add_argument("--min-gt-std", type=float, default=float(cfg["min_gt_std"]))
    ap.add_argument("--min-depth", type=float, default=float(cfg["min_depth"]))
    ap.add_argument("--max-depth", type=float, default=float(cfg["max_depth"]))
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args(argv)

    if args.pairs_csv is None or not args.pairs_csv.is_file():
        print(
            "Data/full_spur is not mounted and --pairs-csv was not given. "
            "Refusing to reprint the training-time α/β as if they were re-fit. "
            "See spur_depth/calib/pro_best_config.json.",
            file=sys.stderr,
        )
        return 2

    kwargs = dict(
        erode_r=args.erode_r,
        min_gt_std=args.min_gt_std,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    try:
        alpha, beta, acc, used = fit_files(_read_pairs_csv(args.pairs_csv), **kwargs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = {
        "alpha": alpha,
        "beta": beta,
        "n_images_used": used,
        "n_pixels": acc.n,
        "erode_r": args.erode_r,
        "min_gt_std": args.min_gt_std,
        "fit_space": "depth",
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"alpha={alpha:.16g}  beta={beta:.16g}  n_images={used}  n_pixels={acc.n:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
