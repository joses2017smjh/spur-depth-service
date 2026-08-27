"""SSD-lineage box proposals from the synthetic masks this dataset already has.

Liu et al. 2016 (SSD) made single-shot detection the default. Orchard papers
through 2025–2026 prefer YOLO for trunks (e.g. YOLOv5n+SENet pear trunks,
YOLOv9c orchard obstacles) because thin branches starve SSD's default anchors.
The *training target* is still a box around a mask. That is what this module
emits — so a later YOLO/SSD head can train without inventing labels.

Returns SSD-style tuples ``(x1, y1, x2, y2, score, cls)`` in pixel coords.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CLS_TRUNK = 1
CLS_BOX = 2


@dataclass(frozen=True)
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int


def boxes_from_mask(mask: np.ndarray, cls: int = CLS_TRUNK, min_area: int = 64) -> list[Det]:
    """Connected components → axis-aligned boxes. Score is fill / box area."""
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out: list[Det] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if int(area) < min_area:
            continue
        fill = float(area) / float(max(w * h, 1))
        out.append(Det(float(x), float(y), float(x + w), float(y + h), fill, cls))
    out.sort(key=lambda d: d.score * (d.x2 - d.x1) * (d.y2 - d.y1), reverse=True)
    return out


def box_to_mask(shape: tuple[int, ...], det: Det, erode_px: int = 10) -> np.ndarray:
    """Rectangle interior of a box. Field scoring uses this, not a speckled mask."""
    h, w = int(shape[0]), int(shape[1])
    m = np.zeros((h, w), dtype=bool)
    x1 = max(0, int(np.floor(det.x1)))
    y1 = max(0, int(np.floor(det.y1)))
    x2 = min(w, int(np.ceil(det.x2)))
    y2 = min(h, int(np.ceil(det.y2)))
    if x2 <= x1 or y2 <= y1:
        return m
    m[y1:y2, x1:x2] = True
    if erode_px > 0:
        from spur_depth.calib.fit_scale_shift import erode_binary

        m = erode_binary(m, erode_px)
    return m


def keep_largest_component(mask: np.ndarray, min_area: int = 2048, close_k: int = 5) -> np.ndarray:
    """Drop speckles. Field RMSE on a speckled UNet mask is not a trunk metric."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    if int(stats[idx, cv2.CC_STAT_AREA]) < min_area:
        return np.zeros_like(m, dtype=bool)
    return labels == idx


def draw_boxes(rgb: np.ndarray, dets: list[Det], color=(33, 150, 243)) -> np.ndarray:
    vis = rgb.copy()
    for d in dets:
        c = (255, 112, 67) if d.cls == CLS_BOX else color
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(vis, p1, p2, c, 3)
    return vis
