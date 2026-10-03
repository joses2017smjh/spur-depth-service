"""Simulated stereo RGB-D sensor depth, so "RGB-D" is not read as "perfect depth".

Profile ``d435`` follows the active-stereo error model Intel documents for the
D400 series: RMS depth error = z^2 * subpixel / (f_px * baseline). At 848x480
(f ~= 424 px for the 90 deg HFOV), 50 mm baseline and 0.08 px subpixel,
sigma_z ~= 0.0038 * z^2 m, i.e. 1.5 cm at 2 m, which matches the ~0.75 %
standard deviation measured at 2 m (Rustler et al., 2025). Matching runs at
~1/2.26 of our 1920 px width, so the error field is correlated at that scale.

Thin wood is where stereo fails first (Jain et al., ICRA 2025, and the
RRT-Connect point clouds there missing tertiary branches): any tree pixel
whose local width at sensor resolution is under ``thin_px`` is dropped in
coherent blocks, half to invalid (0) and half to the background behind it.
Silhouette pixels get flying-pixel depth, and a few low-frequency holes are
cut everywhere. Output is quantized to 1 mm like the D435 depth unit.

Deterministic: pass ``seed`` (e.g. ``frame_seed(key)``) and the same frame
always gets the same noisy depth in training, evaluation and figures.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

D435 = {
    "sigma_coef": 0.0038,  # m^-1: sigma_z = sigma_coef * z^2
    "subsample": 1920.0 / 848.0,  # our pixels per sensor pixel
    "thin_px": 3.0,  # sensor pixels; narrower wood fails to match
    "thin_drop": 0.7,  # fraction of thin blocks lost
    "edge_flying": 0.5,  # fraction of silhouette pixels mixed with background
    "hole_frac": 0.02,  # low-frequency holes anywhere
}


def frame_seed(key: str) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


def _correlated_noise(rng, shape, cell: float) -> np.ndarray:
    h, w = shape
    hs, ws = max(2, int(round(h / cell))), max(2, int(round(w / cell)))
    n = rng.standard_normal((hs, ws)).astype(np.float32)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)


def _background_fill(depth: np.ndarray, fg: np.ndarray, k: int = 31) -> np.ndarray:
    """Depth just behind the foreground: grey dilation of the non-tree depth."""
    bg = np.where(fg | (depth <= 0), 0.0, depth).astype(np.float32)
    return cv2.dilate(bg, np.ones((k, k), np.uint8))


def simulate_sensor(
    depth: np.ndarray,
    tree_mask: np.ndarray,
    seed: int,
    profile: dict | None = None,
) -> np.ndarray:
    """Noisy metric depth (float32, 0 = invalid) from clean planar depth."""
    p = dict(D435, **(profile or {}))
    rng = np.random.default_rng(seed)
    z = np.asarray(depth, dtype=np.float32)
    valid = z > 0
    fg = np.asarray(tree_mask, dtype=bool) & valid
    out = z.copy()

    # 1. Axial error, correlated at sensor resolution.
    n = _correlated_noise(rng, z.shape, p["subsample"])
    out = out + n * (p["sigma_coef"] * z * z)

    # 2. Thin wood: local width (2x medial distance) below thin_px sensor pixels.
    dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 3)
    width = 2.0 * cv2.dilate(dist, np.ones((5, 5), np.uint8)) / p["subsample"]
    thin = fg & (width < p["thin_px"])
    blocks = _correlated_noise(rng, z.shape, 8.0 * p["subsample"])
    lost = thin & (blocks < np.quantile(blocks, p["thin_drop"]))
    behind = _background_fill(z, fg)
    to_bg = lost & (_correlated_noise(rng, z.shape, 4.0 * p["subsample"]) > 0)
    out[lost] = 0.0
    out[to_bg] = behind[to_bg]

    # 3. Flying pixels on the tree silhouette.
    edge = fg & ~cv2.erode(fg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    fly = edge & (rng.random(z.shape) < p["edge_flying"]) & (behind > 0)
    a = rng.random(int(fly.sum())).astype(np.float32)
    out[fly] = a * out[fly] + (1.0 - a) * behind[fly]

    # 4. Low-frequency holes anywhere.
    holes = _correlated_noise(rng, z.shape, 24.0 * p["subsample"])
    out[holes < np.quantile(holes, p["hole_frac"])] = 0.0

    out[~valid & ~to_bg] = 0.0
    out = np.round(np.clip(out, 0.0, 65.0) * 1000.0) / 1000.0
    return out.astype(np.float32)
