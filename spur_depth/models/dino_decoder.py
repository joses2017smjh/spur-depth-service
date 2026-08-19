"""Decoder matched to the DINO encoder's uniform 256-channel skips.

Restored from ``_mvp_precompiled.pyc``; docstrings verbatim from the bytecode.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .blocks import Up
from .channels import _BOTT_CH, _SKIP_CH


class DINODecoder(nn.Module):
    """Decoder matched to DINOv2ViTLEncoder skip channel sizes.

    Identical logic to UNetDecoder but uses uniform _SKIP_CH=256 at all
    skip levels (instead of the 64/128/256/512 progression of the CNN path).

    Channel flow:
      b      (B, 512, H/14, W/14)  ← fused bottleneck
      up3  + s3(256) → (B, 256, H/8,  W/8)
      up2  + s2(256) → (B, 256, H/4,  W/4)
      up1  + s1(256) → (B, 128, H/2,  W/2)
      up0  + s0(256) → (B,  64, H,    W  )
      head            → (B,   1, H,    W  )  ← depth prediction
    """

    def __init__(self) -> None:
        super().__init__()
        self.up3 = Up(_BOTT_CH, _SKIP_CH, 256)  # 512 + 256 → 256
        self.up2 = Up(256, _SKIP_CH, 256)  # 256 + 256 → 256
        self.up1 = Up(256, _SKIP_CH, 128)  # 256 + 256 → 128
        self.up0 = Up(128, _SKIP_CH, 64)  # 128 + 256 → 64
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(
        self, b: torch.Tensor, skips: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Args:
            b:     (B, 512, H/14, W/14) — fused multi-view bottleneck
            skips: (s0, s1, s2, s3) — per-view skip connections

        Returns:
            (B, 1, H, W) — refined depth for one view
        """
        s0, s1, s2, s3 = skips
        x = self.up3(b, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.head(x)
