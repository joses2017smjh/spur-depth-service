"""Model definitions for the SPUR depth refiner.

Every class here except ``MVStereoDINOUNet`` and the losses was recovered by
disassembling ``_mvp_precompiled.pyc`` after the original source was deleted.
``tests/test_source_parity.py`` proves the restored source and the bytecode
build the same model, to < 1e-6 max abs forward difference.
"""

from .blocks import ConvBlock, Up
from .channels import _BOTT_CH, _DINO_DIM, _SKIP_CH
from .depth_side_branch import DepthSideBranch
from .dino_decoder import DINODecoder
from .dino_encoder import DINOv2ViTLEncoder
from .losses import masked_rmse_nview, silog_loss_nview, stereo_mv_consistency_loss
from .mv_stereo_dino import MVStereoDINOUNet
from .pose import _CAM_R_BASE, _CAM_ROT_Z_BASE, _CAM_Z_BASE, PoseProject

__all__ = [
    "ConvBlock", "Up", "DepthSideBranch", "DINODecoder", "DINOv2ViTLEncoder",
    "PoseProject", "MVStereoDINOUNet",
    "silog_loss_nview", "stereo_mv_consistency_loss", "masked_rmse_nview",
    "_DINO_DIM", "_SKIP_CH", "_BOTT_CH",
    "_CAM_Z_BASE", "_CAM_ROT_Z_BASE", "_CAM_R_BASE",
]
