"""Dataset loaders used by the eval harness.

The 3-pair loader is the one behind the shipped 0.0445 m number. It is
ported verbatim from ``dataset/trunk_stereo_triplet_mvp.py``; only comments
around ``INPUT_DEPTH_INTERP`` were clarified (the default is bilinear).
"""

from .trunk_stereo_triplet import TrunkStereoTripletMVPDataset

__all__ = ["TrunkStereoTripletMVPDataset"]
