"""Perception stack around the metric-depth service.

Lineage the modules implement, not slogans:

* detect   — SSD-shaped box proposals from the synthetic trunk / box masks
             (field papers have moved to YOLO; the *labels* are unchanged)
* flow     — dense correspondences (RAFT when torchvision ships it, Farneback else)
* refine   — the shipped DA2-ft + DINO RGB+D service
* reconstruct — back-project metric depth with Blender K, T_wc
* sensors  — inverse-variance fusion of metric maps + optional lidar/IMU
* sim2real — appearance PSI + 30 cm box-anchor implied scale
"""

from spur_depth.pipeline.detect import box_to_mask, boxes_from_mask
from spur_depth.pipeline.flow import dense_flow, flow_to_rgb
from spur_depth.pipeline.reconstruct import Frame, unproject, merge_clouds
from spur_depth.pipeline.sensors import fuse_depths
from spur_depth.pipeline.sim2real import appearance_gap, box_anchor_scale

__all__ = [
    "box_to_mask",
    "boxes_from_mask",
    "dense_flow",
    "flow_to_rgb",
    "Frame",
    "unproject",
    "merge_clouds",
    "fuse_depths",
    "appearance_gap",
    "box_anchor_scale",
]
