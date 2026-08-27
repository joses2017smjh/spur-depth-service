"""Robust α/β on top of the C++-matched least-squares moments.

The shipped fitter is ordinary LS. One leaky mask pixel at 8 m can yank
β. These variants re-use ``Moments`` so the 2×2 solve stays identical:

* Huber IRLS — down-weight |r| > δ (default 5 cm)
* median-of-trees — one LS fit per fit-tree, then median α and β
* box-anchor — scale only, from the 30 cm box (no GT depth at test time)

Headline in ``SHIPPING.md`` stays the LS hold-out until a GPU job measures
these on the paper val trees and they win. Do not swap numbers here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from spur_depth.calib.fit_scale_shift import (
    Moments,
    merge_in_order,
    moments_from_image,
    solve_scale_shift,
    valid_pixels,
)

HUBER_DELTA_M = 0.05


def moments_weighted(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Moments:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    return Moments(
        n=float(np.sum(w)),
        sx=float(np.sum(w * x)),
        sy=float(np.sum(w * y)),
        sxy=float(np.sum(w * x * y)),
        sx2=float(np.sum(w * x * x)),
    )


def huber_weights(residual: np.ndarray, delta: float = HUBER_DELTA_M) -> np.ndarray:
    r = np.abs(np.asarray(residual, dtype=np.float64))
    w = np.ones_like(r)
    big = r > delta
    w[big] = delta / np.clip(r[big], 1e-12, None)
    return w


def huber_irls(
    triples: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    delta: float = HUBER_DELTA_M,
    iters: int = 6,
    erode_r: int = 10,
    min_gt_std: float = 0.05,
    min_depth: float = 0.5,
    max_depth: float = 10.0,
) -> tuple[float, float, Moments]:
    """Two-pass-capable IRLS. Callers can stream triples each iteration."""
    cached = list(triples)
    parts = [
        moments_from_image(p, g, m, erode_r=erode_r, min_gt_std=min_gt_std) for p, g, m in cached
    ]
    parts = [p for p in parts if p is not None]
    if not parts:
        raise ValueError("no valid pixels for Huber IRLS")
    alpha, beta = solve_scale_shift(merge_in_order(parts))
    acc = merge_in_order(parts)
    for _ in range(iters):
        wparts: list[Moments] = []
        for pred, gt, mask in cached:
            valid = valid_pixels(
                pred, gt, mask, erode_r=erode_r, min_depth=min_depth, max_depth=max_depth
            )
            if int(valid.sum()) < 2:
                continue
            x = np.asarray(pred, dtype=np.float64)[valid]
            y = np.asarray(gt, dtype=np.float64)[valid]
            if float(y.std()) < min_gt_std:
                continue
            resid = y - (alpha * x + beta)
            wparts.append(moments_weighted(x, y, huber_weights(resid, delta)))
        if not wparts:
            break
        acc = merge_in_order(wparts)
        alpha, beta = solve_scale_shift(acc)
    return alpha, beta, acc


def median_of_groups(
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    **kwargs,
) -> tuple[float, float, dict[str, tuple[float, float]]]:
    """Independent LS per group; return (median α, median β, per-group)."""
    per: dict[str, tuple[float, float]] = {}
    for key, triples in grouped.items():
        parts = [moments_from_image(p, g, m, **kwargs) for p, g, m in triples]
        parts = [p for p in parts if p is not None]
        if not parts:
            continue
        per[key] = solve_scale_shift(merge_in_order(parts))
    if not per:
        raise ValueError("no group had valid pixels")
    alphas = np.array([a for a, _ in per.values()], dtype=np.float64)
    betas = np.array([b for _, b in per.values()], dtype=np.float64)
    return float(np.median(alphas)), float(np.median(betas)), per


def group_by(keys: list[str], triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    out: dict[str, list] = defaultdict(list)
    for k, t in zip(keys, triples):
        out[k].append(t)
    return dict(out)
