"""Input-depth resize is bilinear; GT-depth resize is nearest.

The two look interchangeable until you put a 2 m trunk next to a 0 m
background sentinel. Bilinear on GT inflates masked RMSE ~20×. Bilinear
on the *network input* is what every reported number was trained under.
This file pins both so a well-meaning cleanup cannot swap them.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from spur_depth.serve.preprocess import (
    DEPTH_INF_THRESH,
    da2_resize_shape,
    load_gt_depth,
    load_input_depth,
    load_mask,
    load_rgb,
)


def test_gt_depth_resize_is_nearest_so_silhouette_is_not_averaged_with_zero_background():
    """GT background is zeroed before resize; bilinear would smear 2 m into 0 m."""
    arr = np.zeros((7, 7), dtype=np.float32)
    arr[:, :3] = 2.0
    arr[0, 6] = DEPTH_INF_THRESH  # sentinel, must become 0 before nearest

    got = load_gt_depth(arr, H=4, W=4)
    cleaned = arr.copy()
    cleaned[cleaned >= DEPTH_INF_THRESH] = 0.0
    src = torch.from_numpy(cleaned)[None, None]
    nearest = F.interpolate(src, size=(4, 4), mode="nearest").squeeze(0)
    bilinear = F.interpolate(src, size=(4, 4), mode="bilinear", align_corners=False).squeeze(0)

    assert got.shape == (1, 4, 4)
    assert torch.equal(got, nearest)
    assert torch.max((got - bilinear).abs()).item() > 0.1
    assert float(got[0, 0, -1]) == 0.0  # sentinel became 0, then nearest


def test_input_depth_resize_is_bilinear_matching_the_shipped_loader_default():
    """INPUT_DEPTH_INTERP was unset on the 0.0445 m run → bilinear."""
    rng = np.random.default_rng(0)
    arr = rng.random((40, 80)).astype(np.float32) * 5 + 0.5
    got = load_input_depth(arr, H=10, W=20)
    expect = F.interpolate(
        torch.from_numpy(arr)[None, None],
        size=(10, 20),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    assert torch.max((got - expect).abs()).item() < 1e-6


def test_mask_resize_is_nearest():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, :4] = 255
    img = Image.fromarray(mask, mode="L")
    got = load_mask(img, H=4, W=4).numpy()
    assert set(np.unique(got).tolist()) <= {0.0, 1.0}
    assert got[0, :, 0].min() == 1.0
    assert got[0, :, -1].max() == 0.0


def test_load_rgb_imagenet_normalises_and_uses_pil_wh_order():
    img = Image.new("RGB", (16, 8), color=(255, 0, 0))  # W=16, H=8
    t = load_rgb(img, H=4, W=8)
    assert t.shape == (3, 4, 8)
    # Red channel after ImageNet norm: (1 - 0.485) / 0.229
    assert float(t[0].mean()) > 1.0
    assert float(t[1].mean()) < 0.0


def test_da2_resize_shape_matches_round_then_ceil_contract():
    # Square already at 518 → 518.
    assert da2_resize_shape(518, 518) == (518, 518)
    # 280×512 orchard frame: scale by 518/280, width rounds to a 14-multiple.
    h, w = da2_resize_shape(280, 512)
    assert h % 14 == 0 and w % 14 == 0
    assert h >= 518 and w >= 518
