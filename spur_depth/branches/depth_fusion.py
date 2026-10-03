"""Sensor depth + DA2-ft depth: keep the sensor where it agrees, the network where it fails.

Stereo RGB-D cameras are metric but lose thin wood (dropouts, background
bleeding, flying pixels). DA2-ft, fine-tuned on these orchards, is smooth on
thin wood but carries a scene-level scale/shift error. Fit that affine on
pixels where both are valid, with Huber IRLS so the sensor's thin-wood
failures do not pull the fit; then take the sensor value wherever it lies
within a noise-scaled band of the aligned network, else the aligned network.
Same idea as using sparse metric depth to prompt a monocular model (Prompt
Depth Anything, CVPR 2025), in closed form and without retraining.
"""

from __future__ import annotations

import numpy as np


def huber_affine(x: np.ndarray, y: np.ndarray, delta: float = 0.05, iters: int = 10):
    """(a, b) minimising a Huber loss of a*x + b - y."""
    w = np.ones_like(x)
    a, b = 1.0, 0.0
    for _ in range(iters):
        A = np.stack([x * w, w], 1)
        sol, *_ = np.linalg.lstsq(A, y * w, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = np.abs(a * x + b - y)
        w = np.where(r <= delta, 1.0, np.sqrt(delta / np.maximum(r, 1e-9)))
    return a, b


def fuse_sensor_mono(
    sensor: np.ndarray,
    mono: np.ndarray,
    region: np.ndarray | None = None,
    sigma_coef: float = 0.0038,
    band_sigmas: float = 3.0,
    band_floor_m: float = 0.03,
    max_fit_px: int = 200_000,
) -> tuple[np.ndarray, dict]:
    """Fused metric depth (float32, 0 = invalid) and fit diagnostics.

    ``region`` limits the affine fit (e.g. predicted wood pixels); the band
    uses the sensor's own error model, sigma_z = sigma_coef * z^2.
    """
    sensor = np.asarray(sensor, dtype=np.float32)
    mono = np.asarray(mono, dtype=np.float32)
    both = (sensor > 0) & (mono > 0) & np.isfinite(mono)
    fit = both & region if region is not None and (both & region).sum() > 500 else both
    ys, xs = np.nonzero(fit)
    if len(ys) < 50:
        return np.where(sensor > 0, sensor, mono).astype(np.float32), {"fit_px": int(len(ys))}
    if len(ys) > max_fit_px:
        pick = np.random.default_rng(0).choice(len(ys), max_fit_px, replace=False)
        ys, xs = ys[pick], xs[pick]
    a, b = huber_affine(mono[ys, xs].astype(np.float64), sensor[ys, xs].astype(np.float64))
    aligned = np.where(mono > 0, a * mono + b, 0.0).astype(np.float32)
    band = np.maximum(band_floor_m, band_sigmas * sigma_coef * aligned * aligned)
    agree = (sensor > 0) & (np.abs(sensor - aligned) <= band)
    fused = np.where(agree, sensor, aligned).astype(np.float32)
    info = {
        "fit_px": int(len(ys)),
        "a": a,
        "b": b,
        "sensor_kept_frac": float(agree[region].mean())
        if region is not None and region.any()
        else float(agree.mean()),
    }
    return fused, info
