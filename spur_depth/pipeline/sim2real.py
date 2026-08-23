"""Synthetic → usable in a real orchard. Two checks, not a CycleGAN.

Jain/Grimm/Lee (ICRA 2025) transfer a RAFT policy by never trusting RGB
photorealism. CycleGAN / digital-twin papers (DT/MARS) try to paint the
other way. This service was trained on one bark texture and one camera
sweep — every axis of that sentence is a drift axis.

What we can measure without field GT:

1. Appearance stats (luminance, focus, saturation) vs the committed baseline.
2. A 30 cm box in the frame. Known size + predicted metres ⇒ implied scale.
   That is the only number a cutter's tolerance understands.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spur_depth.serve.input_stats import image_stats

BOX_TRUE_M = 0.30


def appearance_gap(rgb: np.ndarray) -> dict[str, Any]:
    return {"stats": image_stats(rgb)}


def box_anchor_scale(
    depth_m: np.ndarray,
    box_mask: np.ndarray,
    true_width_m: float = BOX_TRUE_M,
    K: np.ndarray | None = None,
) -> dict[str, float]:
    """Implied metric scale from a box of known width.

    If K is given, width in metres is median(Z) * pixel_width / fx.
    Without K we report only the median predicted depth on the box.
    """
    m = box_mask.astype(bool) & np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(m):
        return {
            "n": 0.0,
            "median_z_m": float("nan"),
            "implied_width_m": float("nan"),
            "scale": float("nan"),
        }
    z = float(np.median(depth_m[m]))
    ys, xs = np.where(m)
    pix_w = float(xs.max() - xs.min() + 1)
    if K is None:
        return {
            "n": float(m.sum()),
            "median_z_m": z,
            "implied_width_m": float("nan"),
            "scale": float("nan"),
        }
    fx = float(K[0, 0])
    implied = z * pix_w / max(fx, 1e-6)
    return {
        "n": float(m.sum()),
        "median_z_m": z,
        "implied_width_m": implied,
        "scale": implied / true_width_m if true_width_m else float("nan"),
    }
