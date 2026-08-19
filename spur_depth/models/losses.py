"""Losses and metrics for the multi-view refiner.

Extracted verbatim from ``MVP_MODEL/mvp_stereo_model.py`` (which survived as
real source). Only the import block is new. Kept in the service package so
``bench/eval_rmse.py`` scores exactly the quantity the trainer reported —
``masked_rmse_nview`` is the function behind every RMSE in the paper.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def _make_pixel_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
    """(3, H*W) homogeneous pixel coordinate grid [u, v, 1]^T per column."""
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    return torch.stack([u, v, torch.ones_like(u)], dim=0).reshape(3, H * W)


def _camera_baseline(T_i_wc: torch.Tensor, T_j_wc: torch.Tensor) -> torch.Tensor:
    """Per-batch Euclidean distance (m) between two camera centres."""
    C_i = torch.inverse(T_i_wc)[:, :3, 3]
    C_j = torch.inverse(T_j_wc)[:, :3, 3]
    return torch.norm(C_i - C_j, dim=1)



# ──────────────────────────────────────────────────────────────────────────────

def silog_loss_nview(
    d_pred: torch.Tensor,       # (B, V, 1, H, W)
    d_gt:   torch.Tensor,       # (B, V, 1, H, W)
    mask:   torch.Tensor,       # (B, V, 1, H, W)
    variance_focus: float = 0.85,
    eps: float = 1e-6,
    min_depth: float = 0.5,
    max_depth: float = 10.0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Scale-invariant log loss summed over all V views.

    Only trunk-masked pixels whose GT depth falls in [min_depth, max_depth]
    contribute.  This excludes background seen through canopy gaps (Blender
    reports their true far depth inside the silhouette mask).
    """
    V = d_pred.shape[1]
    per_view: List[torch.Tensor] = []
    for v in range(V):
        pred  = d_pred[:, v].clamp(min=eps)
        gt    = d_gt[:, v].clamp(min=eps)
        m     = (mask[:, v].float()
                 * (d_gt[:, v] > min_depth).float()
                 * (d_gt[:, v] < max_depth).float())
        n     = m.sum() + eps
        d     = (torch.log(pred) - torch.log(gt)) * m
        var   = (d ** 2).sum() / n - variance_focus * (d.sum() / n) ** 2
        per_view.append(torch.sqrt(var.clamp(min=0.0)))
    total = sum(per_view)
    return total, per_view


# Keep 2-view alias for backwards compatibility
silog_loss_2view = silog_loss_nview

def stereo_mv_consistency_loss(
    d_pred: torch.Tensor,   # (B, V, 1, H, W)  V = 2*n_pairs (views flat)
    d_gt:   torch.Tensor,   # (B, V, 1, H, W)
    mask:   torch.Tensor,   # (B, V, 1, H, W)
    K:      torch.Tensor,   # (B, V, 3, 3)
    T_wc:   torch.Tensor,   # (B, V, 4, 4)
    min_depth:    float = 0.001,
    max_depth:    float = 20.0,
    max_baseline: float = 0.80,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Bidirectional GT-warp MV consistency loss applied within each stereo pair.

    Views are expected flat: views 0,1 = pair 0 (L,R); 2,3 = pair 1; 4,5 = pair 2 …
    For each pair, two directions are computed:
      L→R: warp GT_L into R frame, compare pred_R vs z_R
      R→L: warp GT_R into L frame, compare pred_L vs z_L

    Gradient flows through d_pred only; d_gt is used purely for geometry.
    """
    device = d_pred.device
    B, V, _, H, W = d_pred.shape
    n_pairs = V // 2

    pix = _make_pixel_grid(H, W, device)  # (3, H*W)
    total        = torch.tensor(0.0, device=device)
    n_valid_dirs = 0

    for p in range(n_pairs):
        i_l, i_r = 2 * p, 2 * p + 1

        pred1 = d_pred[:, i_l, 0];  pred2 = d_pred[:, i_r, 0]
        gt1   = d_gt[:,   i_l, 0];  gt2   = d_gt[:,   i_r, 0]
        m1    = mask[:,   i_l, 0];  m2    = mask[:,   i_r, 0]
        K1, K2       = K[:,    i_l], K[:,    i_r]
        T1_wc, T2_wc = T_wc[:, i_l], T_wc[:, i_r]

        baseline = _camera_baseline(T1_wc, T2_wc)
        close    = baseline <= max_baseline
        if not close.any():
            continue

        # Direction pairs: (src_gt, src_mask, src_K, src_T, tgt_pred, tgt_K, tgt_T, tgt_mask)
        directions = [
            (gt2, m2, K2, T2_wc, pred1, K1, T1_wc, m1),  # R→L
            (gt1, m1, K1, T1_wc, pred2, K2, T2_wc, m2),  # L→R
        ]

        for (gt_src, mask_src, K_src, T_src, pred_tgt, K_tgt, T_tgt, mask_tgt) in directions:
            K_src_inv = torch.inverse(K_src)
            T_src_cw  = torch.inverse(T_src)
            T_st      = T_tgt @ T_src_cw

            gt_flat   = gt_src.reshape(B, 1, H * W)
            pts_src   = (K_src_inv @ pix.unsqueeze(0)) * gt_flat

            ones      = torch.ones(B, 1, H * W, device=device, dtype=pts_src.dtype)
            pts_src_h = torch.cat([pts_src, ones], dim=1)
            pts_tgt_h = T_st @ pts_src_h
            pts_tgt   = pts_tgt_h[:, :3]
            z_tgt     = pts_tgt[:, 2:3]

            proj = K_tgt @ pts_tgt
            u    = proj[:, 0:1] / (z_tgt + eps)
            v    = proj[:, 1:2] / (z_tgt + eps)

            u_n  = 2.0 * u / (W - 1) - 1.0
            v_n  = 2.0 * v / (H - 1) - 1.0
            grid = torch.cat([u_n, v_n], dim=1).permute(0, 2, 1).reshape(B, H, W, 2)

            pred_tgt_sampled = F.grid_sample(
                pred_tgt.unsqueeze(1), grid,
                mode='bilinear', padding_mode='zeros', align_corners=True,
            ).squeeze(1)

            mask_tgt_sampled = F.grid_sample(
                mask_tgt.unsqueeze(1).float(), grid,
                mode='nearest', padding_mode='zeros', align_corners=True,
            ).squeeze(1)

            z_map = z_tgt.reshape(B, H, W)
            in_bounds = (
                (u_n.reshape(B, H, W) >= -1) & (u_n.reshape(B, H, W) <= 1) &
                (v_n.reshape(B, H, W) >= -1) & (v_n.reshape(B, H, W) <= 1)
            )
            valid = (
                in_bounds &
                (mask_src > 0.5) &
                (mask_tgt_sampled > 0.5) &
                (gt_src > min_depth) & (gt_src < max_depth) &
                (z_map > min_depth) & (z_map < max_depth) &
                (pred_tgt_sampled > min_depth) & (pred_tgt_sampled < max_depth)
            ) & close.view(B, 1, 1)

            if valid.sum() < 10:
                continue

            log_diff = torch.abs(
                torch.log(pred_tgt_sampled[valid]) - torch.log(z_map[valid])
            )
            total        = total + log_diff.mean()
            n_valid_dirs += 1

    return total / max(n_valid_dirs, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def masked_rmse_nview(
    d_pred: torch.Tensor,   # (B, V, 1, H, W)
    d_gt:   torch.Tensor,   # (B, V, 1, H, W)
    mask:   torch.Tensor,   # (B, V, 1, H, W)
    eps: float = 1e-6,
    min_depth: float = 0.5,
    max_depth: float = 10.0,
) -> Tuple[float, List[float]]:
    """RMSE on trunk-masked, depth-valid pixels, per view and average."""
    V = d_pred.shape[1]
    per_view: List[float] = []
    for v in range(V):
        m     = (mask[:, v].float()
                 * (d_gt[:, v] > min_depth).float()
                 * (d_gt[:, v] < max_depth).float())
        diff2 = ((d_pred[:, v] - d_gt[:, v]) ** 2) * m
        rmse  = torch.sqrt(diff2.sum() / (m.sum() + eps))
        per_view.append(rmse.item())
    return sum(per_view) / V, per_view


# Keep 2-view alias for backwards compatibility
masked_rmse_2view = masked_rmse_nview
