"""Cut targets for the Isaac pruning workflow.

isaac-sim-pruning-workflow consumes flat per-target records, not graphs:
``task/loop.py`` ``EpisodeTarget(position_w, axis_w, radius_m, length_m,
confidence, source)`` and ``geometry/cut_point.py`` ``CutPoint`` use a Z-up
world frame in metres, the frame of the L-Py cylinders and of these renders.
``cut_targets`` turns a camera-frame TreeGraph into those records, one per
non-trunk part that has a parent, best first.
"""

from __future__ import annotations

import numpy as np

from spur_depth.branches.tree import CLASSES, CUT_OFFSET_M, TRUNK, TreeGraph


def cut_targets(
    graph: TreeGraph,
    T_wc: np.ndarray | None = None,
    classes: tuple[str, ...] = ("spur", "shoot", "branch"),
    offset_m: float = CUT_OFFSET_M,
    source: str = "spur_depth.branches",
) -> list[dict]:
    """Records with ``position_w``/``axis_w`` in the world frame (camera frame if ``T_wc`` is None).

    ``T_wc`` is world-to-camera (OpenCV), as ``reconstruct.load_pose`` returns it.
    """
    if T_wc is not None:
        T = np.asarray(T_wc, dtype=np.float64)
        R, t = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]
    else:
        R, t = np.eye(3), np.zeros(3)
    out = []
    for p in graph.parts.values():
        name = CLASSES[p.cls]
        if p.cls == TRUNK or p.parent is None or name not in classes or p.length <= 0:
            continue
        point, axis = p.cut(offset_m)
        out.append(
            {
                "record_id": int(p.pid),
                "part_name": f"{name}_{p.pid}",
                "part_class": name,
                "parent_id": int(p.parent),
                "position_w": (R @ point + t).round(5).tolist(),
                "axis_w": (R @ axis).round(5).tolist(),
                "radius_m": round(float(np.median(p.radius)), 5),
                "length_m": round(float(p.length), 5),
                "confidence": round(float(p.score), 4),
                "source": source,
                "frame": "world" if T_wc is not None else "camera",
            }
        )
    out.sort(key=lambda r: (-r["confidence"], r["record_id"]))
    return out
