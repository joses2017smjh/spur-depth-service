"""Decoder building blocks.

Restored from ``_mvp_precompiled.pyc`` (Python 3.10 bytecode); the original
source was deleted. Docstrings are verbatim from the bytecode's embedded
constants, and structural equivalence is asserted by
``tests/test_source_parity.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3×3 convolution layers with BatchNorm and ReLU.

    Pattern: Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU. Each conv preserves
    spatial dims (padding=1). bias=False because BatchNorm has its own bias.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Decoder upsampling step: double resolution, concatenate skip, ConvBlock.

    1. Bilinear upsample x to the skip connection's spatial size
    2. Concatenate with skip along the channel dim
    3. ConvBlock to process the combined features
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
