"""Run dummy inference on the bundled sample and write demo_depth.png."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    os.environ["SPUR_SKIP_WEIGHTS"] = "1"
    sys.path.insert(0, str(_REPO))
    from spur_depth.serve.runner import DummyRunner

    sample = _REPO / "samples" / "view_01.png"
    if not sample.is_file():
        print(f"missing {sample}", file=sys.stderr)
        return 2
    img = Image.open(sample)
    depth = DummyRunner().predict_one(img)
    vis = (np.clip(depth / (depth.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
    out = _REPO / "demo_depth.png"
    Image.fromarray(vis).save(out)
    print(f"wrote {out}  shape={depth.shape}  mean={float(depth.mean()):.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
