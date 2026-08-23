"""Per-request input and prediction statistics. Images are not retained."""

from __future__ import annotations

from typing import Any

import numpy as np

DEPTH_RANGE = (0.5, 10.0)


def _as_rgb_uint8(image) -> np.ndarray:
    from PIL import Image

    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
    return arr.astype(np.uint8, copy=False)


def image_stats(image) -> dict[str, float]:
    """Luminance, focus (Laplacian variance), saturated-pixel fraction."""
    import cv2

    rgb = _as_rgb_uint8(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lum = gray.astype(np.float64) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    saturated = ((rgb >= 254).all(axis=-1) | (rgb <= 1).all(axis=-1)).mean()
    return {
        "luminance_mean": float(lum.mean()),
        "luminance_std": float(lum.std()),
        "focus_var": float(lap.var()),
        "saturated_frac": float(saturated),
    }


def depth_stats(depth: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    d = np.asarray(depth, dtype=np.float64).reshape(-1)
    lo, hi = DEPTH_RANGE
    qs = np.quantile(d, [0.01, 0.05, 0.50, 0.95, 0.99])
    out: dict[str, float] = {
        "depth_p1": float(qs[0]),
        "depth_p5": float(qs[1]),
        "depth_p50": float(qs[2]),
        "depth_p95": float(qs[3]),
        "depth_p99": float(qs[4]),
        "oor_frac": float(((d < lo) | (d > hi)).mean()),
    }
    if mask is not None:
        m = np.asarray(mask).reshape(-1) > 0
        out["mask_frac"] = float(m.mean()) if m.size else 0.0
    return out


def request_stats(
    *,
    images,
    depths: np.ndarray,
    group_id: str | None = None,
    masks=None,
) -> dict[str, Any]:
    """Aggregate stats for one /predict or /predict/group call.

    ``images`` is a single PIL image or a sequence of them.
    ``depths`` is (H, W) or (V, H, W).
    """
    from PIL import Image

    if isinstance(images, Image.Image):
        imgs = [images]
    else:
        imgs = list(images)
    img_rows = [image_stats(im) for im in imgs]
    keys = img_rows[0].keys()
    merged = {k: float(np.mean([r[k] for r in img_rows])) for k in keys}

    d = np.asarray(depths)
    mask_arr = None
    if masks is not None:
        mask_arr = (
            np.stack([np.array(m) for m in masks], axis=0)
            if not isinstance(masks, np.ndarray)
            else masks
        )
    merged.update(depth_stats(d, mask_arr))
    if group_id is not None:
        merged["group_id"] = group_id
    merged["n_views"] = len(imgs)
    return merged
