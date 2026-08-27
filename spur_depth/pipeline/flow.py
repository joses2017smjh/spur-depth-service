"""Dense correspondence. RAFT when the weights are present, Farneback otherwise.

Teed & Deng, RAFT (ECCV 2020) is the flow model orchard visuomotor work
still calls in 2025 (Jain, Grimm, Lee — ICRA 2025: wrist-cam RAFT beats a
depth camera on thin dormant wood, and transfers sim→real). This repo's
``Optical_flow/`` folder is stereo RGB, not precomputed flow — so we compute
it. torchvision.raft_large is used if importable; OpenCV Farneback is the
CPU path that always runs in CI.
"""

from __future__ import annotations

import numpy as np


def dense_flow(prev_rgb: np.ndarray, next_rgb: np.ndarray, backend: str = "auto") -> np.ndarray:
    """Return (H, W, 2) float32 flow in pixels (dx, dy)."""
    if backend == "farneback":
        return _farneback(prev_rgb, next_rgb)
    if backend == "raft":
        return _raft(prev_rgb, next_rgb)
    try:
        return _raft(prev_rgb, next_rgb)
    except Exception:
        return _farneback(prev_rgb, next_rgb)


def _farneback(prev_rgb: np.ndarray, next_rgb: np.ndarray) -> np.ndarray:
    import cv2

    a = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow.astype(np.float32)


def _raft(prev_rgb: np.ndarray, next_rgb: np.ndarray) -> np.ndarray:
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    from torchvision.transforms.functional import resize, to_tensor

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).eval()
    prep = weights.transforms()
    h, w = prev_rgb.shape[:2]
    # RAFT wants multiples of 8.
    nh, nw = (h // 8) * 8, (w // 8) * 8
    t0 = to_tensor(resize(prev_rgb, [nh, nw]))
    t1 = to_tensor(resize(next_rgb, [nh, nw]))
    img0, img1 = prep(t0[None], t1[None])
    with torch.no_grad():
        pred = model(img0, img1)[-1][0]
    flow = pred.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    if flow.shape[:2] != (h, w):
        import cv2

        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= w / nw
        flow[..., 1] *= h / nh
    return flow


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    import cv2

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 8, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
