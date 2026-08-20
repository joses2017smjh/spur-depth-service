"""Channel widths shared by the DINO encoder and decoder.

These were module-level constants in the original ``mvp_depth_refine_model``;
they are hoisted here so the encoder and decoder can agree on them without
importing each other.
"""

_DINO_DIM = 1024  # ViT-L token dimension
_SKIP_CH = 256  # projected skip channel size
_BOTT_CH = 512  # bottleneck channel size

__all__ = ["_DINO_DIM", "_SKIP_CH", "_BOTT_CH"]
