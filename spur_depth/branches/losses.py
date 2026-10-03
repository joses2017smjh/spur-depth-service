"""BranchNet training loss: region terms for the parts, recall terms for their thin axes.

Cross-entropy and soft Dice get the bulk of each part right, but a 2-5 px spur
that is predicted one pixel too thin, or broken once, costs almost nothing in
either. Skeleton Recall (Kirchhoff et al., ECCV 2024) scores the fraction of
each class's GT axis (dilated by one pixel) that the class probability covers,
so gaps in thin parts become expensive. The centerline head is trained on the
same 1-px-dilated axes with BCE (pos_weight for the rare positives) + soft Dice.

Everything is computed in float32 on batch-level sums, whatever the autocast dtype.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spur_depth.branches.tree import IGNORE

CLASS_WEIGHTS = (1.0, 2.0, 1.5, 3.0, 3.0)  # background, trunk, branch, shoot, spur
TREE_CLASSES = (1, 2, 3, 4)


def dilate(mask: torch.Tensor, px: int = 1) -> torch.Tensor:
    """Binary dilation of a float (B, C, H, W) mask with a (2*px+1)^2 square."""
    return F.max_pool2d(mask, 2 * px + 1, stride=1, padding=px)


def _one_hot(labels: torch.Tensor, classes: tuple[int, ...]) -> torch.Tensor:
    """(B, H, W) integer map -> (B, len(classes), H, W) float; IGNORE matches no class."""
    return torch.stack([labels == c for c in classes], 1).float()


def _mean_over_present(per_class: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Mean of ``per_class`` over classes with ``present``; 0 when none is (no host sync)."""
    present = present.float()
    return (per_class * present).sum() / present.sum().clamp(min=1.0)


def skeleton_recall_loss(
    prob: torch.Tensor,
    skel: torch.Tensor,
    classes: tuple[int, ...] = TREE_CLASSES,
    eps: float = 1e-6,
) -> torch.Tensor:
    """1 - sum(p_c * S_c) / (sum(S_c) + eps), averaged over the classes whose skeleton is present.

    ``prob``: (B, C, H, W) class probabilities. ``skel``: (B, H, W) class-coded 1-px axes;
    S_c is the class-c axis dilated by 1 px (3x3). Sums run over the whole batch.
    """
    s = dilate(_one_hot(skel, classes))
    p = prob[:, list(classes)].float()
    n = s.sum((0, 2, 3))
    recall = (p * s).sum((0, 2, 3)) / (n + eps)
    return _mean_over_present(1.0 - recall, n > 0)


def soft_dice_loss(
    prob: torch.Tensor,
    target: torch.Tensor,
    classes: tuple[int, ...] = TREE_CLASSES,
    smooth: float = 1.0,
) -> torch.Tensor:
    """1 - soft Dice per class over non-ignored pixels, averaged over classes present in the batch."""
    valid = (target != IGNORE).unsqueeze(1).float()
    p = prob[:, list(classes)].float() * valid
    t = _one_hot(target, classes)
    inter = (p * t).sum((0, 2, 3))
    n_t = t.sum((0, 2, 3))
    dice = (2.0 * inter + smooth) / (p.sum((0, 2, 3)) + n_t + smooth)
    return _mean_over_present(1.0 - dice, n_t > 0)


def binary_soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0):
    p = torch.sigmoid(logits.float())
    inter = (p * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (p.sum() + target.sum() + smooth)


class BranchLoss(nn.Module):
    """Weighted sum of segmentation and centerline terms; ``forward`` returns (total, components)."""

    def __init__(
        self,
        w_ce: float = 1.0,
        w_dice: float = 0.5,
        w_skel: float = 0.5,
        w_ctr_bce: float = 1.0,
        w_ctr_dice: float = 0.5,
        class_weights: tuple[float, ...] = CLASS_WEIGHTS,
        ctr_pos_weight: float = 4.0,
    ) -> None:
        super().__init__()
        self.w = {
            "ce": w_ce,
            "dice": w_dice,
            "skel": w_skel,
            "ctr_bce": w_ctr_bce,
            "ctr_dice": w_ctr_dice,
        }
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.register_buffer("pos_weight", torch.tensor([ctr_pos_weight], dtype=torch.float32))

    def forward(
        self,
        out: dict[str, torch.Tensor],
        cls: torch.Tensor,
        skel: torch.Tensor,
        ctr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``out``: BranchNet logits; ``cls`` (B,H,W) int64 with IGNORE; ``skel`` (B,H,W)
        class-coded axes; ``ctr`` (B,H,W) binary axes dilated by 1 px (derived from skel if None).
        """
        logits = out["seg"].float()
        ctr_logits = out["ctr"].float()[:, 0]
        cls = cls.long()
        if ctr is None:
            ctr = dilate((skel > 0).float().unsqueeze(1))[:, 0]
        ctr = ctr.float()

        # Weighted CE as F.cross_entropy(weight=...) defines it, but 0 (not NaN)
        # when every pixel of the batch is ignored.
        valid = cls != IGNORE
        ce_px = F.cross_entropy(
            logits, cls, weight=self.class_weights, ignore_index=IGNORE, reduction="none"
        )
        w_px = self.class_weights[torch.where(valid, cls, 0)] * valid
        ce = ce_px.sum() / w_px.sum().clamp(min=1e-6)

        prob = torch.softmax(logits, 1)
        comps = {
            "ce": ce,
            "dice": soft_dice_loss(prob, cls),
            "skel": skeleton_recall_loss(prob, skel),
            "ctr_bce": F.binary_cross_entropy_with_logits(
                ctr_logits, ctr, pos_weight=self.pos_weight
            ),
            "ctr_dice": binary_soft_dice_loss(ctr_logits, ctr),
        }
        total = sum(self.w[k] * v for k, v in comps.items())
        return total, {k: v.detach() for k, v in comps.items()}
