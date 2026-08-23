"""Aggregate metric maps from more than one sensor.

Industry stacks (AgriNav-Sim2Real, ZED2i + IMU + NIR, orchard RGB-D + ToF)
do not pick a winner — they weight. Inverse-variance fusion is the
minimum honest aggregator: each source brings a per-pixel variance, the
fused depth is the precision-weighted mean, and out-of-range pixels stay
out. IMU / lidar enter as extra maps with their own variance, not as
comments in a README.
"""

from __future__ import annotations

import numpy as np


def fuse_depths(
    depths: list[np.ndarray],
    variances: list[np.ndarray] | None = None,
    valid: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fused_metres, fused_variance). Empty support → NaN."""
    if not depths:
        raise ValueError("fuse_depths needs at least one map")
    stack = np.stack([d.astype(np.float32) for d in depths], axis=0)
    if variances is None:
        var = np.ones_like(stack)
    else:
        var = np.stack([v.astype(np.float32) for v in variances], axis=0)
        var = np.clip(var, 1e-6, None)
    if valid is None:
        ok = np.isfinite(stack) & (stack > 0)
    else:
        ok = np.stack([v.astype(bool) for v in valid], axis=0) & np.isfinite(stack)
    prec = np.where(ok, 1.0 / var, 0.0)
    num = np.sum(prec * stack, axis=0)
    den = np.sum(prec, axis=0)
    fused = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0).astype(np.float32)
    fused_var = np.divide(1.0, den, out=np.full_like(den, np.nan), where=den > 0).astype(np.float32)
    return fused, fused_var
