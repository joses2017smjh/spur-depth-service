"""YOLO-nano encode/decode and k-means anchors. CPU only."""

from __future__ import annotations

import numpy as np
import torch

from spur_depth.pipeline.detect import CLS_TRUNK, Det
from spur_depth.pipeline.yolo import (
    DEFAULT_ANCHORS,
    INPUT,
    YOLONano,
    decode_raw,
    encode_boxes,
    kmeans_anchors,
    nms_dets,
)


def test_forward_shape():
    m = YOLONano()
    x = torch.zeros(2, 3, INPUT, INPUT)
    raw = m(x)
    assert raw.shape == (2, 3, 6, 16, 16)


def test_encode_decode_roundtrip():
    det = Det(40, 30, 90, 200, 1.0, CLS_TRUNK)
    anchors = torch.tensor(DEFAULT_ANCHORS, dtype=torch.float32)
    target, obj = encode_boxes([det], anchors)
    assert float(obj.max()) == 1.0
    raw = target[None]
    dec = decode_raw(raw, anchors)[0]
    pos = obj.reshape(-1) > 0
    boxes = dec.reshape(-1, 6)[pos]
    assert boxes.shape[0] == 1
    x1, y1, x2, y2 = [float(v) for v in boxes[0, :4]]
    assert abs(x1 - 40) < 1.5 and abs(y1 - 30) < 1.5
    assert abs(x2 - 90) < 1.5 and abs(y2 - 200) < 1.5


def test_nms_keeps_the_stronger_of_two_overlaps():
    boxes = torch.tensor(
        [
            [10.0, 10.0, 50.0, 80.0, 0.9, 0.9],
            [12.0, 12.0, 52.0, 82.0, 0.4, 0.4],
            [200.0, 10.0, 240.0, 80.0, 0.8, 0.8],
        ]
    )
    dets = nms_dets(boxes, conf=0.1, iou_thr=0.4)
    assert len(dets) == 2
    assert dets[0].score >= dets[1].score


def test_kmeans_returns_k_sorted_by_area():
    rng = np.random.default_rng(0)
    skinny = rng.normal([20, 120], 2, size=(30, 2))
    fat = rng.normal([60, 180], 3, size=(30, 2))
    out = kmeans_anchors(np.vstack([skinny, fat]), k=2, iters=15)
    assert out.shape == (2, 2)
    assert out[0, 0] * out[0, 1] <= out[1, 0] * out[1, 1]
