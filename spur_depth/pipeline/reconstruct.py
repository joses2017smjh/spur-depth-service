"""Metric RGB-D → world points. This is the tree, not a pretty mesh.

MVS nets (MSA-MVSNet and friends) learn a cost volume. We already have
metric depth and Blender K / T_wc on every frame, so the geometry step is
back-projection + merge — the same fusion those nets emit after depth
regression. Poses come from ``ann/*.json`` (OpenCV world-to-camera).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spur_depth.data.trunk_stereo_triplet import _euler_xyz_to_T


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    T_wc: np.ndarray


def load_pose(ann: dict) -> tuple[np.ndarray, np.ndarray]:
    cam = ann["camera"]
    K = np.asarray(cam["intrinsics"]["K"], dtype=np.float32)
    T_wc = _euler_xyz_to_T(cam["location"], cam["rotation_euler"])
    return K, T_wc


def unproject(frame: Frame, stride: int = 4, max_points: int = 80_000) -> np.ndarray:
    """Return (N, 6) xyzrgb. Depth is metres. Invalid / unmasked pixels drop."""
    h, w = frame.depth.shape[:2]
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = frame.depth[ys, xs]
    m = frame.mask[ys, xs] & np.isfinite(z) & (z > 0.3) & (z < 12.0)
    xs, ys, z = xs[m].astype(np.float32), ys[m].astype(np.float32), z[m].astype(np.float32)
    fx, fy = float(frame.K[0, 0]), float(frame.K[1, 1])
    cx, cy = float(frame.K[0, 2]), float(frame.K[1, 2])
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    cam = np.stack([x, y, z], axis=1)
    R = frame.T_wc[:3, :3]
    t = frame.T_wc[:3, 3]
    # T_wc is world-to-camera: X_c = R X_w + t  →  X_w = R^T (X_c − t)
    world = (R.T @ (cam - t).T).T
    rgb = frame.rgb[ys.astype(int), xs.astype(int)].astype(np.float32) / 255.0
    pts = np.concatenate([world, rgb], axis=1)
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(pts.shape[0], max_points, replace=False)]
    return pts.astype(np.float32)


def merge_clouds(clouds: list[np.ndarray]) -> np.ndarray:
    if not clouds:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(clouds, axis=0)


def write_ply(path, pts: np.ndarray) -> None:
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    rgb = np.clip(pts[:, 3:6] * 255.0, 0, 255).astype(np.uint8)
    with open(path, "w") as fh:
        fh.write(header)
        for (x, y, z), (r, g, b) in zip(pts[:, :3], rgb):
            fh.write(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}\n")
