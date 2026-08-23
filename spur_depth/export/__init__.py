"""Export helpers.

Keep this module lazy: ``python -m spur_depth.export.to_onnx`` must set
``XFORMERS_DISABLED=1`` *before* DINOv2 attention is imported, otherwise
tracing hits CUDA-only flash-attention kernels.
"""

from __future__ import annotations

from typing import Any

__all__ = ["EncoderWrapper", "FuseDecodeWrapper", "split_refiner"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .wrappers import EncoderWrapper, FuseDecodeWrapper, split_refiner

        return {
            "EncoderWrapper": EncoderWrapper,
            "FuseDecodeWrapper": FuseDecodeWrapper,
            "split_refiner": split_refiner,
        }[name]
    raise AttributeError(name)
