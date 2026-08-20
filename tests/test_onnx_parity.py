"""ONNX graph regression for the tiny fuse+decode split.

No weights, no dataset: a randomly initialised ``FuseDecodeWrapper`` still
contains the real 1×1 fuse conv, the real ``DINODecoder``, and the absolute
(softplus) head. If that graph stops exporting, every later TensorRT engine
is fiction.

The ViT-L encoder graph is too large for CI; export it locally with
``python -m spur_depth.export.to_onnx --graph encoder --ckpt $SPUR_CKPT``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from spur_depth.export.to_onnx import _export  # noqa: E402
from spur_depth.export.wrappers import FuseDecodeWrapper  # noqa: E402

# Tiny spatial size so CI does not export a 280×512 decoder graph.
_H, _W = 32, 64


@pytest.mark.onnx
def test_fuse_decode_onnx_matches_torch(tmp_path: Path):
    # Random FuseDecodeWrapper — real fuse conv + DINODecoder, no ViT-L.
    torch.manual_seed(0)
    fuse = FuseDecodeWrapper(n_views=6).eval()
    bottlenecks = torch.randn(1, 6, 512, 4, 6)
    s0 = torch.randn(1, 6, 256, _H, _W)
    s1 = torch.randn(1, 6, 256, _H // 2, _W // 2)
    s2 = torch.randn(1, 6, 256, _H // 4, _W // 4)
    s3 = torch.randn(1, 6, 256, _H // 8, _W // 8)
    onnx_path = tmp_path / "fuse_shared.onnx"
    _export(
        fuse,
        (bottlenecks, s0, s1, s2, s3),
        ["bottlenecks", "s0", "s1", "s2", "s3"],
        ["depth"],
        onnx_path,
    )

    with torch.no_grad():
        torch_out = fuse(bottlenecks, s0, s1, s2, s3).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (ort_out,) = sess.run(
        None,
        {
            "bottlenecks": bottlenecks.numpy(),
            "s0": s0.numpy(),
            "s1": s1.numpy(),
            "s2": s2.numpy(),
            "s3": s3.numpy(),
        },
    )
    delta = float(np.max(np.abs(torch_out - ort_out)))
    assert torch_out.shape == (1, 6, 1, _H, _W)
    assert delta < 1e-4, f"max abs {delta} >= 1e-4"
