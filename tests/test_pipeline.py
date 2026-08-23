"""Stack units: boxes from a mask, Farneback flow, inverse-variance fuse."""

from __future__ import annotations

import numpy as np

from spur_depth.pipeline.detect import CLS_TRUNK, boxes_from_mask
from spur_depth.pipeline.flow import dense_flow
from spur_depth.pipeline.sensors import fuse_depths
from spur_depth.pipeline.sim2real import box_anchor_scale


def test_boxes_from_mask_finds_the_blob():
    m = np.zeros((64, 80), dtype=np.uint8)
    m[10:30, 20:50] = 1
    dets = boxes_from_mask(m, CLS_TRUNK, min_area=10)
    assert len(dets) == 1
    d = dets[0]
    assert d.x1 == 20 and d.y1 == 10 and d.x2 == 50 and d.y2 == 30
    assert d.cls == CLS_TRUNK


def test_farneback_shift_is_rightward():
    a = np.zeros((48, 64, 3), dtype=np.uint8)
    b = np.zeros((48, 64, 3), dtype=np.uint8)
    a[16:32, 16:32] = 200
    b[16:32, 20:36] = 200
    flow = dense_flow(a, b)
    assert flow.shape == (48, 64, 2)
    assert float(flow[24, 24, 0]) > 1.0


def test_fuse_depths_precision_weighted():
    lo = np.full((4, 4), 2.0, dtype=np.float32)
    hi = np.full((4, 4), 4.0, dtype=np.float32)
    fused, var = fuse_depths(
        [lo, hi],
        variances=[np.full((4, 4), 0.25), np.full((4, 4), 1.0)],
    )
    # weights 4 and 1 → (8+4)/5 = 2.4
    assert abs(float(fused[0, 0]) - 2.4) < 1e-5
    assert abs(float(var[0, 0]) - 0.2) < 1e-5


def test_box_anchor_scale_recovers_width():
    z = np.full((20, 40), 2.0, dtype=np.float32)
    mask = np.zeros((20, 40), dtype=bool)
    mask[:, 10:30] = True  # 20 px wide
    K = np.array([[100.0, 0, 20], [0, 100.0, 10], [0, 0, 1]], dtype=np.float32)
    # implied = 2.0 * 20 / 100 = 0.40 m; true 0.30 → scale 0.40/0.30
    out = box_anchor_scale(z, mask, true_width_m=0.30, K=K)
    assert abs(out["implied_width_m"] - 0.40) < 1e-5
    assert abs(out["scale"] - 0.40 / 0.30) < 1e-5
