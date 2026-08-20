"""Trainable depth-encoding branch that runs alongside the frozen ViT.

Restored from ``_mvp_precompiled.pyc``; docstrings verbatim from the bytecode.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DepthSideBranch(nn.Module):
    """Small CNN that extracts multi-scale features from the PRO depth map.

    DINOv2 only accepts RGB, so depth is processed here in a separate branch
    and fused with ViT features at the skip-connection level.

    WHY 3 LAYERS?
    PRO depth is already a processed, meaningful signal — not raw sensor noise.
    Three strided convolutions are enough to produce features at the 3 spatial
    scales (H, H/2, H/4) where depth structure is most useful to fuse.
    Deeper processing would overfit on the small dataset.

    Outputs (3 spatial scales, matched to ViT skip targets):
        f0: (B,  32, H,    W  )  — full-res depth gradients / edges
        f1: (B,  64, H/2,  W/2)  — mid-res depth shape
        f2: (B, 128, H/4,  W/4)  — coarse depth regions
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            d: (B, 1, H, W) — PRO depth map for one view

        Returns:
            f0: (B,  32, H,    W  )
            f1: (B,  64, H/2,  W/2)
            f2: (B, 128, H/4,  W/4)
        """
        f0 = self.layer1(d)
        f1 = self.layer2(f0)
        f2 = self.layer3(f1)
        return f0, f1, f2
