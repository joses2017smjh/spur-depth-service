"""In-repo YOLO-nano (1 class: trunk). No Ultralytics, no copied 1080p frames.

Orchard papers ship YOLOv5n/v8n/v9c. This is the same assignment — a
grid of anchor boxes trained on ``boxes_from_mask`` — small enough to
train on the 7-tree restore without a second dataset copy. It is not a
COCO YOLOv8n checkpoint.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from spur_depth.pipeline.detect import CLS_TRUNK, Det

INPUT = 256
STRIDE = 16
GRID = INPUT // STRIDE  # 16
N_ANCHORS = 3
# Tall trunks after a 256 resize of 1080p (w, h) in pixels.
DEFAULT_ANCHORS = ((28.0, 140.0), (40.0, 180.0), (56.0, 210.0))


class ConvBN(nn.Module):
    def __init__(self, i: int, o: int, k: int = 3, s: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, k, stride=s, padding=k // 2, bias=False),
            nn.BatchNorm2d(o),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class YOLONano(nn.Module):
    """Single-scale YOLOv3-tiny cousin: stride 16, 3 anchors, 1 class."""

    def __init__(self, anchors: tuple[tuple[float, float], ...] = DEFAULT_ANCHORS):
        super().__init__()
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        self.backbone = nn.Sequential(
            ConvBN(3, 32, s=2),
            ConvBN(32, 64, s=2),
            ConvBN(64, 128, s=2),
            ConvBN(128, 256, s=2),
            ConvBN(256, 256, s=1),
        )
        self.head = nn.Conv2d(256, N_ANCHORS * 6, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, A, 6, G, G) — tx, ty, tw, th, obj, cls
        raw = self.head(self.backbone(x))
        b = raw.shape[0]
        return raw.view(b, N_ANCHORS, 6, GRID, GRID)


def encode_boxes(
    dets: list[Det],
    anchors: torch.Tensor,
    *,
    ignore_iou: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (target A×6×G×G, obj_mask A×G×G) in the network's raw space."""
    target = torch.zeros(N_ANCHORS, 6, GRID, GRID)
    obj_mask = torch.zeros(N_ANCHORS, GRID, GRID)
    if not dets:
        return target, obj_mask
    anc = anchors.detach().cpu().numpy()
    for d in dets:
        if d.cls != CLS_TRUNK:
            continue
        bw = max(d.x2 - d.x1, 1.0)
        bh = max(d.y2 - d.y1, 1.0)
        cx = 0.5 * (d.x1 + d.x2)
        cy = 0.5 * (d.y1 + d.y2)
        gj = min(GRID - 1, max(0, int(cy / STRIDE)))
        gi = min(GRID - 1, max(0, int(cx / STRIDE)))
        ious = _anchor_iou(bw, bh, anc)
        a = int(np.argmax(ious))
        # Pre-activation targets so decode's sigmoid/exp is a true inverse.
        target[a, 0, gj, gi] = _logit((cx / STRIDE) - gi)
        target[a, 1, gj, gi] = _logit((cy / STRIDE) - gj)
        target[a, 2, gj, gi] = math.log(bw / max(float(anc[a, 0]), 1.0))
        target[a, 3, gj, gi] = math.log(bh / max(float(anc[a, 1]), 1.0))
        target[a, 4, gj, gi] = 1.0
        target[a, 5, gj, gi] = 1.0
        obj_mask[a, gj, gi] = 1.0
        for k, iou in enumerate(ious):
            if k != a and iou > ignore_iou:
                obj_mask[k, gj, gi] = -1.0  # ignore, not background
    return target, obj_mask


def decode_raw(raw: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """(B,A,6,G,G) → (B, A*G*G, 6) as x1,y1,x2,y2,obj,cls in 256-px coords."""
    b, a, _, gy, gx = raw.shape
    device = raw.device
    yy, xx = torch.meshgrid(
        torch.arange(gy, device=device), torch.arange(gx, device=device), indexing="ij"
    )
    tx, ty, tw, th = raw[:, :, 0], raw[:, :, 1], raw[:, :, 2], raw[:, :, 3]
    obj = raw[:, :, 4].sigmoid()
    cls = raw[:, :, 5].sigmoid()
    cx = (xx[None, None] + tx.sigmoid()) * STRIDE
    cy = (yy[None, None] + ty.sigmoid()) * STRIDE
    aw = anchors[None, :, 0, None, None]
    ah = anchors[None, :, 1, None, None]
    pw = tw.exp() * aw
    ph = th.exp() * ah
    x1, y1 = cx - pw / 2, cy - ph / 2
    x2, y2 = cx + pw / 2, cy + ph / 2
    boxes = torch.stack([x1, y1, x2, y2, obj, cls], dim=-1)
    return boxes.reshape(b, a * gy * gx, 6)


def nms_dets(decoded: torch.Tensor, *, conf=0.25, iou_thr=0.45, max_det=12) -> list[Det]:
    """``decoded`` is (N, 6) on CPU/GPU."""
    if decoded.numel() == 0:
        return []
    score = decoded[:, 4] * decoded[:, 5]
    keep = score >= conf
    boxes, score = decoded[keep, :4], score[keep]
    if boxes.numel() == 0:
        return []
    try:
        from torchvision.ops import nms

        idx = nms(boxes, score, iou_thr)[:max_det]
    except Exception:
        idx = torch.argsort(score, descending=True)[:max_det]
    out = []
    for i in idx.tolist():
        x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
        out.append(Det(x1, y1, x2, y2, float(score[i]), CLS_TRUNK))
    return out


def yolo_loss(raw: torch.Tensor, target: torch.Tensor, obj_mask: torch.Tensor) -> torch.Tensor:
    pos = obj_mask > 0
    bg = obj_mask == 0
    obj_pred = raw[:, :, 4]
    obj_tgt = target[:, :, 4]
    loss_obj = torch.tensor(0.0, device=raw.device)
    if pos.any():
        loss_obj = loss_obj + torch.nn.functional.binary_cross_entropy_with_logits(
            obj_pred[pos], obj_tgt[pos]
        )
    if bg.any():
        loss_obj = loss_obj + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
            obj_pred[bg], obj_tgt[bg]
        )
    if not pos.any():
        return loss_obj
    box_pred = raw[:, :, :4].permute(0, 1, 3, 4, 2)[pos]
    box_tgt = target[:, :, :4].permute(0, 1, 3, 4, 2)[pos]
    loss_box = torch.nn.functional.mse_loss(box_pred, box_tgt)
    cls_pred = raw[:, :, 5][pos]
    cls_tgt = target[:, :, 5][pos]
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(cls_pred, cls_tgt)
    return loss_box + loss_obj + 0.5 * loss_cls


def kmeans_anchors(wh: np.ndarray, k: int = 3, iters: int = 20, seed: int = 0) -> np.ndarray:
    """IoU k-means on (w, h) in 256-px space."""
    rng = np.random.default_rng(seed)
    wh = np.asarray(wh, dtype=np.float64)
    if len(wh) < k:
        return np.asarray(DEFAULT_ANCHORS, dtype=np.float32)
    cents = wh[rng.choice(len(wh), size=k, replace=False)]
    for _ in range(iters):
        ious = np.array([[_pair_iou(w, h, cw, ch) for cw, ch in cents] for w, h in wh])
        lab = np.argmax(ious, axis=1)
        for i in range(k):
            pts = wh[lab == i]
            if len(pts):
                cents[i] = pts.mean(axis=0)
    order = np.argsort(cents[:, 0] * cents[:, 1])
    return cents[order].astype(np.float32)


def scale_dets(dets: list[Det], *, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> list[Det]:
    sy = dst_hw[0] / max(src_hw[0], 1)
    sx = dst_hw[1] / max(src_hw[1], 1)
    return [Det(d.x1 * sx, d.y1 * sy, d.x2 * sx, d.y2 * sy, d.score, d.cls) for d in dets]


def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _anchor_iou(bw: float, bh: float, anchors: np.ndarray) -> np.ndarray:
    return np.array([_pair_iou(bw, bh, aw, ah) for aw, ah in anchors], dtype=np.float64)


def _pair_iou(w1, h1, w2, h2) -> float:
    inter = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return float(inter / max(union, 1e-6))
