"""BranchNet, its losses and its input contract: CPU-only, no dataset needed."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spur_depth.branches.losses import BranchLoss, dilate, skeleton_recall_loss
from spur_depth.branches.model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    BranchNet,
    pad_to_multiple,
    prepare_input,
    unpad,
)
from spur_depth.branches.tree import IGNORE


@pytest.fixture(scope="module")
def net() -> BranchNet:
    torch.manual_seed(0)
    return BranchNet(pretrained=False).eval()


def _axes(b: int = 2, h: int = 32, w: int = 48) -> torch.Tensor:
    skel = torch.zeros(b, h, w, dtype=torch.uint8)
    skel[:, 10, 4:40] = 2  # a branch
    skel[:, 12:30, 20] = 4  # a spur
    skel[0, 3:9, 30] = 1  # trunk in one image only
    return skel


def test_forward_shapes(net):
    with torch.no_grad():
        out = net(torch.randn(1, 5, 64, 96))
    assert out["seg"].shape == (1, 5, 64, 96)
    assert out["ctr"].shape == (1, 1, 64, 96)


def test_forward_rejects_unpadded_input(net):
    with pytest.raises(ValueError):
        net(torch.randn(1, 5, 60, 96))


def test_depth_channels_start_as_damped_mean_rgb_filter(net):
    w = net.encoder.conv1.weight.detach()
    expect = 0.5 * w[:, :3].mean(1)
    assert torch.allclose(w[:, 3], expect) and torch.allclose(w[:, 4], expect)


def test_losses_finite_and_backward():
    torch.manual_seed(0)
    b, h, w = 2, 32, 48
    seg = torch.randn(b, 5, h, w, requires_grad=True)
    ctr_logit = torch.randn(b, 1, h, w, requires_grad=True)
    cls = torch.randint(0, 5, (b, h, w))
    cls[:, :4, :6] = IGNORE
    skel = _axes(b, h, w)
    ctr = dilate((skel > 0).float().unsqueeze(1))[:, 0]
    total, comps = BranchLoss()({"seg": seg, "ctr": ctr_logit}, cls, skel, ctr)
    assert set(comps) == {"ce", "dice", "skel", "ctr_bce", "ctr_dice"}
    assert torch.isfinite(total) and all(torch.isfinite(v) for v in comps.values())
    total.backward()
    assert torch.isfinite(seg.grad).all() and torch.isfinite(ctr_logit.grad).all()
    assert seg.grad.abs().sum() > 0 and ctr_logit.grad.abs().sum() > 0
    # Ignored pixels get no segmentation gradient from CE or Dice; only skeleton recall could
    # touch them, and there is no axis there.
    assert seg.grad[:, :, :4, :6].abs().max() == 0


def test_losses_finite_when_every_pixel_is_ignored():
    seg = torch.randn(1, 5, 16, 16, requires_grad=True)
    cls = torch.full((1, 16, 16), IGNORE)
    skel = torch.zeros(1, 16, 16, dtype=torch.uint8)
    total, _ = BranchLoss()({"seg": seg, "ctr": torch.zeros(1, 1, 16, 16)}, cls, skel)
    assert torch.isfinite(total)
    total.backward()


def test_skeleton_recall_zero_on_skeleton_one_when_empty():
    skel = _axes()
    prob = torch.zeros(2, 5, *skel.shape[1:])
    for c in (1, 2, 3, 4):
        prob[:, c] = dilate((skel == c).float().unsqueeze(1))[:, 0]
    assert skeleton_recall_loss(prob, skel).item() < 1e-4
    empty = torch.zeros_like(prob)
    empty[:, 0] = 1.0  # everything predicted background
    assert skeleton_recall_loss(empty, skel).item() > 0.999
    # Covering only the undilated 1-px axis recalls about a third of the 3-px band.
    thin = torch.zeros_like(prob)
    for c in (1, 2, 4):
        thin[:, c] = (skel == c).float()
    assert 0.5 < skeleton_recall_loss(thin, skel).item() < 0.8


def test_prepare_input_channels():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (6, 7, 3), dtype=np.uint8)
    depth = rng.uniform(0.5, 3.0, (6, 7)).astype(np.float32)
    depth[0, :3] = 0.0
    depth[1, 0] = np.nan
    x = prepare_input(rgb, depth)
    assert x.shape == (5, 6, 7) and x.dtype == torch.float32
    for c in range(3):
        expect = (rgb[..., c] / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        np.testing.assert_allclose(x[c].numpy(), expect, rtol=1e-5, atol=1e-5)
    valid = np.isfinite(depth) & (depth > 0)
    np.testing.assert_array_equal(x[4].numpy(), valid.astype(np.float32))
    np.testing.assert_allclose(x[3].numpy()[valid], 1.0 / depth[valid], rtol=1e-6)
    assert (x[3].numpy()[~valid] == 0).all()
    none = prepare_input(rgb, None)
    assert (none[3:] == 0).all() and torch.equal(none[:3], x[:3])


def test_pad_to_multiple_round_trip():
    x = torch.randn(2, 5, 50, 70)
    xp, hw = pad_to_multiple(x, 32)
    assert xp.shape == (2, 5, 64, 96) and hw == (50, 70)
    assert torch.equal(unpad(xp, hw), x)
    assert (xp[..., 50:, :] == 0).all() and (xp[..., :, 70:] == 0).all()
    same, hw2 = pad_to_multiple(torch.randn(1, 5, 64, 32), 32)
    assert same.shape[-2:] == (64, 32) and hw2 == (64, 32)
