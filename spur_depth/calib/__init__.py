"""PRO scale/shift calibration (C1).

The loaders used to carry α and β as bare literals. They now read
``pro_best_config.json``. The fitter that produces those constants is
``fit_scale_shift.py``; the C++17 port lives in ``cpp/scale_shift/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).with_name("pro_best_config.json")

# Fallbacks match the training-time literals if the JSON is missing.
_FALLBACK = {
    "alpha": -0.06610793956568871,
    "beta": 1.555980697834118,
    "erode_r": 10,
    "min_gt_std": 0.05,
    "min_depth": 0.5,
    "max_depth": 10.0,
    "fit_space": "depth",
    "pro_depth_eps": 1.0e-3,
}


def load_pro_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else _CONFIG_PATH
    if not p.is_file():
        return dict(_FALLBACK)
    data = json.loads(p.read_text())
    out = dict(_FALLBACK)
    out.update({k: data[k] for k in _FALLBACK if k in data})
    return out


_CFG = load_pro_config()
PRO_ALPHA = float(_CFG["alpha"])
PRO_BETA = float(_CFG["beta"])
PRO_DEPTH_EPS = float(_CFG["pro_depth_eps"])
PRO_ERODE_R = int(_CFG["erode_r"])
PRO_MIN_GT_STD = float(_CFG["min_gt_std"])
