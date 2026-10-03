"""BranchNet: full-resolution part segmentation and centerlines from RGB + depth.

Spurs are 2-5 px wide at 1920x1080, so both heads predict at input resolution:
a ResNet-34 U-Net climbs back from 1/32 to 1/1, and a light stem on the raw
input supplies the full-resolution edges that ResNet's stride-2 conv1 has
already blurred. Depth enters as inverse depth (bounded, largest on the near
wood) plus a valid flag, so "no reading" is not confused with "far away" and
one checkpoint serves clean GT, sensor-like, DA2 and no-depth inputs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 5
IN_CHANNELS = 5
STRIDE = 32
# Readings closer than this are sensor garbage; 1/z would otherwise reach 1000+.
MIN_DEPTH_M = 0.05


def prepare_input(rgb: np.ndarray, depth_m: np.ndarray | None) -> torch.Tensor:
    """(5, H, W) float32: ImageNet-normalised RGB, 1/z (0 where invalid), valid flag.

    The single preprocessing path for training crops, validation and inference.
    ``depth_m`` is metric depth with 0 = invalid; ``None`` means no depth at all.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8 HxWx3, got {rgb.dtype} {rgb.shape}")
    h, w = rgb.shape[:2]
    out = np.zeros((IN_CHANNELS, h, w), dtype=np.float32)
    for c in range(3):
        out[c] = (rgb[..., c].astype(np.float32) / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    if depth_m is not None:
        z = np.asarray(depth_m, dtype=np.float32)
        if z.shape != (h, w):
            raise ValueError(f"depth shape {z.shape} != rgb shape {(h, w)}")
        valid = np.isfinite(z) & (z > MIN_DEPTH_M)
        np.divide(1.0, z, out=out[3], where=valid)
        out[4] = valid
    return torch.from_numpy(out)


def pad_to_multiple(
    x: torch.Tensor, multiple: int = STRIDE
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Zero-pad bottom/right so H and W divide ``multiple``; also returns the original (H, W).

    Zero is the ImageNet mean colour and "no depth", the least informative input.
    """
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    return x, (h, w)


def unpad(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Crop a padded tensor (or a network output) back to ``hw``."""
    return x[..., : hw[0], : hw[1]]


def _conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class _Up(nn.Module):
    """x2 bilinear upsample, concatenate the skip, two conv3x3-BN-ReLU."""

    def __init__(self, cin: int, cskip: int, cout: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(_conv_bn_relu(cin + cskip, cout), _conv_bn_relu(cout, cout))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class _Encoder(nn.Module):
    """ResNet-34 with a 5-channel conv1; returns features at 1/2, 1/4, 1/8, 1/16, 1/32."""

    def __init__(self, pretrained: bool, in_channels: int) -> None:
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet34(weights=weights)
        rgb_w = net.conv1.weight.detach()
        conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv1.weight[:, :3] = rgb_w
            # Depth channels start as a damped luminance filter: they respond to
            # structure from the first step without swamping the pretrained RGB response.
            conv1.weight[:, 3:] = 0.5 * rgb_w.mean(dim=1, keepdim=True)
        self.conv1, self.bn1, self.relu, self.maxpool = conv1, net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(self.maxpool(f0))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]


class BranchNet(nn.Module):
    """U-Net with a ResNet-34 encoder; ``forward`` returns ``{"seg": (B,5,H,W), "ctr": (B,1,H,W)}`` logits.

    H and W must be multiples of 32 (see ``pad_to_multiple``).
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        in_channels: int = IN_CHANNELS,
        stem_channels: int = 24,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 48, 32),
    ) -> None:
        super().__init__()
        d = tuple(int(c) for c in decoder_channels)
        if len(d) != 5:
            raise ValueError("decoder_channels needs 5 entries (1/16 ... 1/1)")
        # Constructor arguments minus ``pretrained``: enough to rebuild from a checkpoint.
        self.config = {
            "num_classes": int(num_classes),
            "in_channels": int(in_channels),
            "stem_channels": int(stem_channels),
            "decoder_channels": list(d),
        }
        self.encoder = _Encoder(pretrained, in_channels)
        self.stem = nn.Sequential(
            _conv_bn_relu(in_channels, stem_channels), _conv_bn_relu(stem_channels, stem_channels)
        )
        self.up4 = _Up(512, 256, d[0])  # 1/16
        self.up3 = _Up(d[0], 128, d[1])  # 1/8
        self.up2 = _Up(d[1], 64, d[2])  # 1/4
        self.up1 = _Up(d[2], 64, d[3])  # 1/2
        self.up0 = _Up(d[3], stem_channels, d[4])  # 1/1
        self.seg_head = nn.Conv2d(d[4], num_classes, 1)
        self.ctr_head = nn.Conv2d(d[4], 1, 1)
        # Centerline pixels are a few percent of a frame; start near that prior so the
        # first steps are not spent pushing every logit down.
        nn.init.constant_(self.ctr_head.bias, -3.0)

    def decoder_parameters(self):
        """Everything outside the pretrained encoder (trained at the full learning rate)."""
        return (p for n, p in self.named_parameters() if not n.startswith("encoder."))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.shape[-2] % STRIDE or x.shape[-1] % STRIDE:
            raise ValueError(f"input H, W must be multiples of {STRIDE}: {tuple(x.shape)}")
        f0, f1, f2, f3, f4 = self.encoder(x)
        y = self.up4(f4, f3)
        y = self.up3(y, f2)
        y = self.up2(y, f1)
        y = self.up1(y, f0)
        y = self.up0(y, self.stem(x))
        return {"seg": self.seg_head(y), "ctr": self.ctr_head(y)}
