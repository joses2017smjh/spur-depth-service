"""Preprocessing, re-implemented explicitly for serving.

The training-time preprocessing lives inside a ``torch.utils.data.Dataset`` and
inside ``DepthAnythingV2.infer_image``. Neither is exportable, and neither is
reachable from a request handler. This module restates both contracts as plain
functions so the service does exactly what training did.

This is the single most likely way a correct model becomes a wrong service, so
each function documents the contract it mirrors and ``tests/test_preprocess.py``
pins the parts that are easy to get subtly wrong.

⚠ A NOTE ON THE INPUT-DEPTH RESIZE
──────────────────────────────────
It is tempting to resize the *input* depth with ``nearest``, because the loader
carries a prominent comment saying bilinear "inflates masked RMSE ~20x". That
comment is real, but it is attached to ``_load_depth`` — the **ground-truth**
depth used for scoring — not to ``_load_pro_depth``, which produces the tensor
the model actually consumes.

The distinction matters:

  * GT depth (``load_gt_depth`` here): background is set to 0 before resizing,
    so bilinear would average ~2 m of trunk against 0 m of background at the
    silhouette and corrupt the metric. NEAREST.
  * Input depth (``load_input_depth`` here): no zeroing step, and the shipped
    launcher leaves ``INPUT_DEPTH_INTERP`` unset, so the loader default —
    ``bilinear`` — is what every reported number was trained and validated
    under. BILINEAR.

Serving the input depth with ``nearest`` would feed the network a distribution
it never saw. We match training instead.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize, to_tensor

from spur_depth.calib import PRO_ALPHA, PRO_BETA, PRO_DEPTH_EPS

# Shipped refiner resolution (run_spur_dino_da2ft_3pair_fusion_dgx2.sh)
REFINER_H, REFINER_W = 280, 512

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Blender writes background depth as a sentinel rather than an actual distance.
DEPTH_INF_THRESH = 1e9

# Floor + α/β: loaded from spur_depth/calib/pro_best_config.json (C1).
# The shipped run passes --no_pro_calib, so these are NOT applied by default;
# they exist for reproducing the raw-PRO ablations.


# ──────────────────────────────────────────────────────────────────────────────
# Refiner inputs
# ──────────────────────────────────────────────────────────────────────────────


def load_rgb(path_or_image, H: int = REFINER_H, W: int = REFINER_W) -> torch.Tensor:
    """RGB → (3, H, W) ImageNet-normalised float32.

    Mirrors ``TrunkStereoTripletMVPDataset._load_rgb``::

        PIL.convert("RGB").resize((W, H), BILINEAR) → to_tensor → normalize

    Note PIL's ``resize`` takes (width, height), which is the reverse of the
    torch convention used everywhere else in this file.
    """
    img = path_or_image
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB").resize((W, H), Image.BILINEAR)
    return normalize(to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)


def load_input_depth(
    arr_or_path,
    H: int = REFINER_H,
    W: int = REFINER_W,
    pro_calib: bool = False,
) -> torch.Tensor:
    """Predicted (DA2-fine-tuned) depth → (1, H, W) float32, in metres.

    Mirrors ``TrunkStereoTripletMVPDataset._load_pro_depth``. BILINEAR, for the
    reason given in this module's docstring — do not "fix" it to nearest.

    Args:
        pro_calib: apply the alpha/beta PRO calibration. False for every
            shipped configuration, which runs ``--no_pro_calib`` because
            DA2-fine-tuned depth is already metric.
    """
    arr = arr_or_path
    if not isinstance(arr, np.ndarray):
        arr = np.load(arr)
    arr = arr.astype(np.float32)

    t = torch.from_numpy(arr).reshape(1, 1, *arr.shape[-2:])
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    t = t.squeeze(0)

    if pro_calib:
        # alpha is negative — raw PRO is inversely correlated with GT depth.
        t = PRO_ALPHA * t + PRO_BETA
    return t.clamp(min=PRO_DEPTH_EPS)


# ──────────────────────────────────────────────────────────────────────────────
# Scoring-only inputs (never fed to the network)
# ──────────────────────────────────────────────────────────────────────────────


def load_gt_depth(arr_or_path, H: int = REFINER_H, W: int = REFINER_W) -> torch.Tensor:
    """Blender ground-truth depth → (1, H, W) float32, in metres.

    Mirrors ``_load_depth``: sentinel background → 0, then NEAREST resize.
    Bilinear here would average ~2 m of trunk with 0 m of background across the
    silhouette, corrupting ~5% of trunk-mask edge pixels and inflating masked
    RMSE roughly 20x.
    """
    arr = arr_or_path
    if not isinstance(arr, np.ndarray):
        arr = np.load(arr)
    arr = arr.astype(np.float32).copy()
    arr[arr >= DEPTH_INF_THRESH] = 0.0

    t = torch.from_numpy(arr).reshape(1, 1, *arr.shape[-2:])
    t = F.interpolate(t, size=(H, W), mode="nearest")
    return t.squeeze(0)


def load_mask(path_or_image, H: int = REFINER_H, W: int = REFINER_W) -> torch.Tensor:
    """Trunk mask → (1, H, W) float32 in {0, 1}. NEAREST; a blurred mask is not
    a mask. An absent path yields all-ones, matching the loader."""
    if path_or_image is None or (
        isinstance(path_or_image, (str, os.PathLike)) and not os.path.isfile(path_or_image)
    ):
        return torch.ones(1, H, W, dtype=torch.float32)

    img = path_or_image
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    arr = np.array(img, dtype=np.float32)

    t = torch.from_numpy(arr).reshape(1, 1, *arr.shape[-2:])
    t = F.interpolate(t, size=(H, W), mode="nearest")
    return (t.squeeze(0) > 0).float()


# ──────────────────────────────────────────────────────────────────────────────
# DA2 (Engine A)
# ──────────────────────────────────────────────────────────────────────────────


def da2_resize_shape(h: int, w: int, input_size: int = 518) -> Tuple[int, int]:
    """DA2's Resize(keep_aspect_ratio, ensure_multiple_of=14, 'lower_bound').

    Mirrors ``depth_anything_v2.util.transform.Resize.get_size`` +
    ``constrain_to_multiple_of`` exactly: scale so both sides are at least
    ``input_size``, then round each to a multiple of 14, bumping up with
    ``ceil`` only when rounding fell below the lower bound.

    Returns (new_h, new_w).
    """
    scale = max(input_size / h, input_size / w)

    def _constrain(x: float, min_val: int) -> int:
        y = int(np.round(x / 14.0) * 14)
        if y < min_val:
            y = int(np.ceil(x / 14.0) * 14)
        return max(14, y)

    return _constrain(h * scale, input_size), _constrain(w * scale, input_size)


def preprocess_da2(bgr: np.ndarray, input_size: int = 518) -> torch.Tensor:
    """OpenCV BGR uint8 image → (1, 3, H14, W14) float32, ready for DA2.

    Mirrors ``DepthAnythingV2.image2tensor``::

        BGR → RGB → /255 → Resize(518, keep_aspect, mult-of-14, INTER_CUBIC)
            → Normalize(ImageNet) → CHW

    ``infer_image`` performs this inside the model and is therefore not
    exportable; the graph we export starts at the tensor this returns.
    """
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
    h, w = rgb.shape[:2]
    new_h, new_w = da2_resize_shape(h, w, input_size)

    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    rgb = (rgb - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(np.float32)
    return torch.from_numpy(chw)


def postprocess_da2(depth: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """DA2 output → source resolution.

    ``align_corners=True`` is not a style choice: it is what ``infer_image``
    uses, and switching it shifts the sampling grid by up to half a pixel,
    which moves depth at the trunk silhouette.
    """
    if depth.dim() == 3:
        depth = depth[:, None]
    return F.interpolate(depth, (out_h, out_w), mode="bilinear", align_corners=True)
