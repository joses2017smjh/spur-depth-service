"""The restored model must load the trained checkpoint exactly, and offline.

Complements ``test_source_parity.py``. That test proves the restored source
matches the deleted original; this one proves the restored source matches the
*trained weights*, which is the claim the service actually depends on.

It also pins the two structural facts that make offline serving possible:

  1. The checkpoint carries the frozen DINOv2 ViT-L backbone (343 tensors,
     ~304.4M params under ``encoder.dino.*``). So the container never needs to
     download LVD-142M weights.
  2. The vendored architecture's keys and shapes match those tensors exactly,
     so ``pretrained=False`` + ``load_state_dict(strict=True)`` is lossless.

Unlike the parity test this runs on any supported Python — no bytecode needed.
It does need the checkpoint; point ``SPUR_CKPT`` at it, or let the default
cluster path resolve.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_DEFAULT_CKPT = Path(
    "/nfs/hpc/share/sanchej7/Computer_Vision/checkpoints/"
    "dino_da2ft_3pair_fusion_nopose_spur_seed1/exp1/seed_01/best_epoch_0023.pt"
)
_CKPT = Path(os.environ.get("SPUR_CKPT", _DEFAULT_CKPT))

pytestmark = [
    pytest.mark.ckpt,
    pytest.mark.skipif(not _CKPT.is_file(), reason=f"checkpoint not available at {_CKPT}"),
]

# The shipped configuration, from run_spur_dino_da2ft_3pair_fusion_dgx2.sh
SHIPPED = dict(n_views=6, use_pose=False, no_fusion=False, pred_mode="absolute")
_H, _W = 280, 512


@pytest.fixture(scope="module")
def state_dict():
    return torch.load(_CKPT, map_location="cpu", weights_only=False)["model"]


def test_checkpoint_carries_frozen_backbone(state_dict):
    """The 1.3 GB checkpoint is mostly frozen ViT-L — that is the point.

    If this ever fails, the container can no longer be built without network
    access, and the Docker story in the README becomes false.
    """
    dino = {k: v for k, v in state_dict.items() if k.startswith("encoder.dino.")}
    assert len(dino) == 343, f"expected 343 backbone tensors, got {len(dino)}"

    n_dino = sum(v.numel() for v in dino.values())
    n_total = sum(v.numel() for v in state_dict.values())
    assert n_dino / 1e6 == pytest.approx(304.4, abs=0.1)
    assert n_total / 1e6 == pytest.approx(313.0, abs=0.1)


def test_vendored_backbone_matches_checkpoint(state_dict):
    """Vendored DINOv2 keys/shapes == checkpoint keys/shapes, exactly.

    This is what licenses dropping ``torch.hub.load(..., pretrained=True)``.
    """
    from spur_depth.models.dinov2 import dinov2_vitl14

    vendored = {k: tuple(v.shape) for k, v in dinov2_vitl14().state_dict().items()}
    ckpt = {
        k[len("encoder.dino.") :]: tuple(v.shape)
        for k, v in state_dict.items()
        if k.startswith("encoder.dino.")
    }
    assert set(vendored) == set(ckpt), "backbone key sets differ"
    assert vendored == ckpt, "backbone shapes differ"


def test_pretrained_true_is_refused():
    """Reaching the network from inside the container must fail loudly."""
    from spur_depth.models.dinov2 import dinov2_vitl14

    with pytest.raises(ValueError, match="pretrained=True"):
        dinov2_vitl14(pretrained=True)


def test_model_loads_strict(state_dict):
    """The end-to-end P0 claim: restored source + trained weights, strict=True.

    ``strict=True`` is the whole assertion. A missing or unexpected key here
    would mean the recovered architecture differs from the trained one, and
    every downstream number would be measured against the wrong model.
    """
    from spur_depth.models import MVStereoDINOUNet

    model = MVStereoDINOUNet(**SHIPPED)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected


def test_shipped_config_has_no_pose_params(state_dict):
    """--no_pose resolves use_pose to False, so no pose MLP is ever built.

    Worth pinning: the exported graph's input signature depends on it. With
    pose off, the ONNX graph needs only (rgb, d_pro) — no pose_T, no K.
    """
    assert not any("pose" in k.lower() for k in state_dict)

    from spur_depth.models import MVStereoDINOUNet

    model = MVStereoDINOUNet(**SHIPPED)
    assert model.use_pose is False
    assert not hasattr(model, "pose_project")


@pytest.mark.parametrize("n_views", [6])
def test_forward_shape_and_positivity(state_dict, n_views):
    """A real forward pass on the loaded weights, at the shipped resolution.

    ``pred_mode="absolute"`` puts a softplus on the head, so every predicted
    depth must be strictly positive — a negative metre reading would be a
    silent contract break for any downstream cutter.
    """
    from spur_depth.models import MVStereoDINOUNet

    model = MVStereoDINOUNet(**SHIPPED).eval()
    model.load_state_dict(state_dict, strict=True)

    torch.manual_seed(0)
    rgb = torch.randn(1, n_views, 3, _H, _W)
    d_pro = torch.rand(1, n_views, 1, _H, _W) * 5.0 + 0.5

    with torch.no_grad():
        out = model(rgb, d_pro)

    assert out.shape == (1, n_views, 1, _H, _W)
    assert torch.isfinite(out).all(), "non-finite depth in output"
    assert (out > 0).all(), "softplus head produced a non-positive depth"


def test_token_grid_arithmetic():
    """280x512 -> 20x37 tokens. Pinned because ONNX export depends on it.

    DINOv2 pads to a multiple of 14: 280 % 14 == 0 (20 tokens exactly), while
    512 % 14 == 8, so pad_w = 6 -> 518 -> 37 tokens. The bottleneck is
    therefore (B, 512, 20, 37) and every shape in the exported graph is static.
    """
    H, W = _H, _W
    pad_h = (14 - H % 14) % 14
    pad_w = (14 - W % 14) % 14
    assert (pad_h, pad_w) == (0, 6)
    assert ((H + pad_h) // 14, (W + pad_w) // 14) == (20, 37)
