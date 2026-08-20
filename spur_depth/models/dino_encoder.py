"""Frozen DINOv2 ViT-L encoder with a trainable depth side branch.

Restored from ``_mvp_precompiled.pyc``; docstrings verbatim from the bytecode.

ONE DELIBERATE CHANGE vs the original bytecode
───────────────────────────────────────────────
The original ``__init__`` built the backbone with::

    self.dino = torch.hub.load(
        'facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True
    )

which reaches the network at model-construction time. That makes the model
unbuildable in an offline container, on a CI runner, or on any machine
without the hub cache warmed. We construct the vendored architecture with
``pretrained=False`` instead.

This is safe *and* verifiable: the trained checkpoint stores the frozen
backbone (343 tensors, 304.4M params under ``encoder.dino.*``), and the
vendored architecture's keys and shapes match it exactly. Whatever random
init we build is fully overwritten by ``load_state_dict``. The random init
does matter for anyone constructing this class without a checkpoint — see
``pretrained_backbone`` below.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .channels import _BOTT_CH, _DINO_DIM, _SKIP_CH
from .depth_side_branch import DepthSideBranch
from .dinov2 import dinov2_vitl14


class DINOv2ViTLEncoder(nn.Module):
    """DINOv2 ViT-L encoder (frozen) + trainable depth side branch fusion.

    Replaces UNetEncoder. Produces the same interface: bottleneck + 4 skips.
    The spatial resolution of the output depth map is NOT affected by the
    larger channel sizes — the decoder's Up blocks use F.interpolate to
    upsample spatial dims independently of channel count.

    ┌──────────────────────────────────────────────────────────────────┐
    │  RGB (3ch)  ──► DINOv2 ViT-L (FROZEN)                              │
    │                 │  extract tokens at blocks 5,11,17,23             │
    │                 │  reshape (B,N,1024) → (B,1024,h_tok,w_tok)       │
    │                 │  project 1024 → 256 (skips) / 512 (bottleneck)   │
    │                 │  upsample to target skip resolutions             │
    │                                                                    │
    │  depth (1ch) ──► DepthSideBranch (TRAINABLE)                       │
    │                  features at H, H/2, H/4                           │
    │                                                                    │
    │  FUSION (TRAINABLE): cat(ViT_skip, depth_feat) → 1×1 conv          │
    │    s0: 256+32  → 256  at (H,   W  )                                │
    │    s1: 256+64  → 256  at (H/2, W/2)                                │
    │    s2: 256+128 → 256  at (H/4, W/4)                                │
    │    s3: 256           at (H/8, W/8)  ← no depth fusion (too coarse) │
    │    b:  512           at (H/14,W/14) ← bottleneck                   │
    └──────────────────────────────────────────────────────────────────┘

    out_ch = 512 so fuse_conv in MVDINOv2PoseConcat is identical to
    the original: 3×(512+32) → 512.
    """

    _HOOK_BLOCKS = (5, 11, 17, 23)  # block indices to extract (0-indexed)

    def __init__(self, pretrained_backbone: bool = False) -> None:
        """
        Args:
            pretrained_backbone: kept False in every shipping path. The frozen
                LVD-142M weights arrive via the refiner checkpoint, not via a
                download. Set True only when training a *new* refiner from
                scratch on a machine that is allowed to reach the network.
        """
        super().__init__()
        # Frozen DINOv2 ViT-L backbone (vendored architecture; see dinov2/)
        self.dino = dinov2_vitl14(pretrained=pretrained_backbone)
        for p in self.dino.parameters():
            p.requires_grad = False

        # 1×1 projections: 1024 ViT channels → skip / bottleneck channels
        self.proj_l6 = nn.Conv2d(_DINO_DIM, _SKIP_CH, 1)
        self.proj_l12 = nn.Conv2d(_DINO_DIM, _SKIP_CH, 1)
        self.proj_l18 = nn.Conv2d(_DINO_DIM, _SKIP_CH, 1)
        self.proj_l24 = nn.Conv2d(_DINO_DIM, _SKIP_CH, 1)
        self.proj_b = nn.Conv2d(_DINO_DIM, _BOTT_CH, 1)

        # Trainable depth side branch
        self.depth_branch = DepthSideBranch()

        # Trainable fusion: concat(ViT skip, depth feat) → 1×1 conv → 256
        self.fuse_s0 = nn.Conv2d(_SKIP_CH + 32, _SKIP_CH, 1)
        self.fuse_s1 = nn.Conv2d(_SKIP_CH + 64, _SKIP_CH, 1)
        self.fuse_s2 = nn.Conv2d(_SKIP_CH + 128, _SKIP_CH, 1)

        self.out_ch = _BOTT_CH

    @staticmethod
    def _tokens_to_spatial(tokens: torch.Tensor, h_tok: int, w_tok: int) -> torch.Tensor:
        """Reshape flat token sequence → 2D spatial map.

        DINOv2 pads images to the nearest multiple of patch_size (14) before
        tokenising, so w_tok = ceil(W/14), h_tok = ceil(H/14).

        Args:
            tokens: (B, N, 1024)  N = h_tok * w_tok
            h_tok:  token grid height
            w_tok:  token grid width
        Returns:
            (B, 1024, h_tok, w_tok)
        """
        B, N, C = tokens.shape
        return tokens.permute(0, 2, 1).reshape(B, C, h_tok, w_tok)

    def forward(
        self, rgb: torch.Tensor, depth: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            rgb:   (B, 3, H, W) — ImageNet-normalised RGB for one view
            depth: (B, 1, H, W) — PRO depth for one view

        Returns:
            b:     (B, 512, H/14, W/14) — bottleneck (spatial size uses ceil)
            skips: (s0, s1, s2, s3) — all (B, 256, *)
                   s0 at (H, W), s1 at (H/2, W/2),
                   s2 at (H/4, W/4), s3 at (H/8, W/8)
        """
        B, _, H, W = rgb.shape

        # Pad RGB to a multiple of patch_size=14
        pad_h = (14 - H % 14) % 14
        pad_w = (14 - W % 14) % 14
        rgb_in = F.pad(rgb, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else rgb
        h_tok = (H + pad_h) // 14
        w_tok = (W + pad_w) // 14

        # ---- RGB stream: frozen DINOv2 features at 4 block depths ----
        raw = self.dino.get_intermediate_layers(
            rgb_in, n=self._HOOK_BLOCKS, return_class_token=False
        )
        f6, f12, f18, f24 = [self._tokens_to_spatial(t, h_tok, w_tok) for t in raw]

        # Project 1024 → 256 / 512
        f6 = self.proj_l6(f6)
        f12 = self.proj_l12(f12)
        f18 = self.proj_l18(f18)
        s3 = self.proj_l24(f24)  # deepest skip — no depth fusion
        b = self.proj_b(f24)  # bottleneck, stays at (h_tok, w_tok)

        # Upsample ViT skips to UNet skip resolutions
        s0_vit = F.interpolate(f6, size=(H, W), mode="bilinear", align_corners=False)
        s1_vit = F.interpolate(f12, size=(H // 2, W // 2), mode="bilinear", align_corners=False)
        s2_vit = F.interpolate(f18, size=(H // 4, W // 4), mode="bilinear", align_corners=False)
        s3 = F.interpolate(s3, size=(H // 8, W // 8), mode="bilinear", align_corners=False)

        # ---- Depth stream: trainable side branch ----
        d0, d1, d2 = self.depth_branch(depth)

        # ---- RGB-DEPTH SKIP FUSION: concat then 1×1 conv ----
        s0 = self.fuse_s0(torch.cat([s0_vit, d0], dim=1))
        s1 = self.fuse_s1(torch.cat([s1_vit, d1], dim=1))
        s2 = self.fuse_s2(torch.cat([s2_vit, d2], dim=1))
        # s3 has no depth fusion (too coarse)

        return b, (s0, s1, s2, s3)
