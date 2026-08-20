"""Trace-friendly graphs for ONNX export.

The refiner's ``forward`` loops over views in Python. Tracing that unrolls
six copies of ViT-L into one graph (~144 transformer blocks), which exports
but is miserable to build as a TensorRT engine. We split at the boundary the
architecture already has:

  EncoderWrapper     one ViT-L; fold views into batch  (N, 3, H, W)
  FuseDecodeWrapper  1×1 fuse over V bottlenecks + per-view decode

Neither wrapper writes ``self._last_aux``. That attribute write is harmless
under ``torch.onnx.export`` for ``pred_mode="absolute"`` (the aux dict is
``None``) and is the kind of thing that later fails under ``torch.export``.
Keep it out of the graph.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spur_depth.models.dino_decoder import DINODecoder
from spur_depth.models.dino_encoder import DINOv2ViTLEncoder
from spur_depth.models.mv_stereo_dino import MVStereoDINOUNet


class EncoderWrapper(nn.Module):
    """``DINOv2ViTLEncoder`` with tuple outputs flattened for ONNX.

    Inputs:
        rgb    (N, 3, H, W)
        d_pro  (N, 1, H, W)
    Outputs:
        bottleneck (N, 512, h_tok, w_tok)
        s0 (N, 256, H, W)
        s1 (N, 256, H/2, W/2)
        s2 (N, 256, H/4, W/4)
        s3 (N, 256, H/8, W/8)
    """

    def __init__(self, encoder: DINOv2ViTLEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else DINOv2ViTLEncoder()

    def forward(
        self, rgb: torch.Tensor, d_pro: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, (s0, s1, s2, s3) = self.encoder(rgb, d_pro)
        return b, s0, s1, s2, s3


class FuseDecodeWrapper(nn.Module):
    """Cross-view 1×1 fuse + per-view ``DINODecoder`` + absolute (softplus) head.

    View count is fixed at construction — the fuse conv's in-channels are
    ``n_views * 512``. The shipped graph is V=6.

    Inputs:
        bottlenecks (B, V, 512, h_tok, w_tok)
        s0 (B, V, 256, H, W)
        s1 (B, V, 256, H/2, W/2)
        s2 (B, V, 256, H/4, W/4)
        s3 (B, V, 256, H/8, W/8)
    Output:
        depth (B, V, 1, H, W)  metres, strictly positive
    """

    def __init__(
        self,
        n_views: int = 6,
        fuse_conv: nn.Conv2d | None = None,
        decoder: DINODecoder | None = None,
    ) -> None:
        super().__init__()
        self.n_views = n_views
        self.fuse_conv = (
            fuse_conv
            if fuse_conv is not None
            else nn.Conv2d(n_views * 512, 512, kernel_size=1, bias=True)
        )
        self.decoder = decoder if decoder is not None else DINODecoder()

    def forward(
        self,
        bottlenecks: torch.Tensor,
        s0: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        s3: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_views, channels, height, width = bottlenecks.shape
        fused = self.fuse_conv(bottlenecks.reshape(bsz, n_views * channels, height, width))
        preds = []
        for i in range(n_views):
            raw = self.decoder(fused, (s0[:, i], s1[:, i], s2[:, i], s3[:, i]))
            # pred_mode="absolute" — identical clamp/softplus as MVStereoDINOUNet
            pred, _aux = (
                F.softplus(raw.clamp(-20, 20)).clamp(min=1e-3),
                None,
            )
            preds.append(pred)
        return torch.stack(preds, dim=1)


def split_refiner(
    model: MVStereoDINOUNet,
) -> Tuple[EncoderWrapper, FuseDecodeWrapper]:
    """Share weights with an already-loaded refiner. No copies of ViT-L."""
    if model.use_pose:
        raise ValueError("split export is only defined for the shipped --no_pose graph")
    if model.no_fusion or not hasattr(model, "fuse_conv"):
        raise ValueError("split export needs the fusion 1×1 conv")
    if model.pred_mode != "absolute":
        raise ValueError("split export implements the absolute (softplus) head only")
    encoder = EncoderWrapper(model.encoder)
    fuse = FuseDecodeWrapper(
        n_views=model.n_views, fuse_conv=model.fuse_conv, decoder=model.decoder
    )
    return encoder, fuse
