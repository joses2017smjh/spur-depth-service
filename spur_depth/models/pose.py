"""Pose embedding for cross-view fusion.

Restored from ``_mvp_precompiled.pyc`` by disassembling ``PoseProject`` and
reading its embedded docstrings. The class body is confirmed by the
``__init__`` bytecode:

    LOAD nn.Sequential
      nn.Linear(in_dim, 64)
      nn.ReLU(inplace=True)
      nn.Linear(64, pose_ch)
    STORE_ATTR net

Note that the shipped 3-pair configuration runs with ``--no_pose``, so this
module is unused on that path — ``MVStereoDINOUNet`` resolves
``use_pose = use_pose and not no_fusion`` to False and never builds it. It is
restored because the CNN refiner imports it at module scope, and because the
pose-enabled ablations need it to load.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Reconstructed from annotation sampling (original source unrecoverable).
# Mean camera height ≈ 1.43 m; horizontal radius ≈ 11.66 m; uniform orbit → 0.0.
_CAM_Z_BASE = 1.43
_CAM_ROT_Z_BASE = 0.0
_CAM_R_BASE = 11.66


class PoseProject(nn.Module):
    """Project a flattened 4×4 pose matrix into a compact embedding for fusion.

    The 4×4 relative pose matrix (camera-to-reference transform) encodes:
      - Rotation: how much the camera rotated between views (3×3 submatrix)
      - Translation: how far the camera moved (last column, 3 values)

    We flatten this to 16 numbers, then use a small MLP to project it into
    a learned embedding space (pose_ch dimensions, default 32).

    This embedding is later spatially broadcast and concatenated with the
    bottleneck feature map, so the decoder knows the geometric relationship
    between views.

    Architecture: Linear(16→64) → ReLU → Linear(64→32)

    Args:
        in_dim:  16 (flattened 4×4 matrix)
        pose_ch: Output embedding size (default 32)
    """

    def __init__(self, in_dim: int = 16, pose_ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, pose_ch),
        )

    def forward(self, pose_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose_vec: (B, 16) — flattened relative pose matrix

        Returns:
            (B, pose_ch) — pose embedding vector
        """
        return self.net(pose_vec)
