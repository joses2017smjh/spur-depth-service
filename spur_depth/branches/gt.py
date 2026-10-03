"""Exact branch ground truth from the L-Py metadata the renderer used.

The generator builds each tree mesh from ``cylinder_data`` (only parts
reachable from ``hierarchy["root"]``; spurs are leaves) and places it with
``location`` + ``rotation_euler = (-17.143 deg, 0, 0)``. Every annotation
carries the world centroids of all cylinders in metadata order, so the
world transform of any tree is recoverable without the 9 sidecar files.

Per-pixel labels come from geometry, not from a second render: GT depth is
back-projected with the render camera's K and each tree pixel takes the
part whose cylinder surface it lies on (0.1 mm median residual).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from spur_depth.branches.tree import (
    IGNORE,
    PREFIX_TO_CLASS,
    TRUNK,
    Part,
    TreeGraph,
    closest_point_on_polyline,
    resample_polyline,
)

CV_ROOT = Path("/nfs/hpc/share/sanchej7/Computer_Vision")
METADATA_DIR = CV_ROOT / "trees" / "metadata"
DATA_ROOT = CV_ROOT / "Data" / "full_spur"
BARK = "bark_brown_02"
TILT_RAD = math.radians(-17.143)
RIGS = ("box", "box_cam1", "box_cam2", "box_cam3", "box_cam4")
SHOTS = tuple(f"shot{i:02d}" for i in range(1, 7))
SIDES = ("l", "r")
PAPER_TEST = ("lpy_envy_00042", "lpy_envy_00065")
# Tuning trees for the graph assembly and model selection; never scored as test.
VAL_TREES = ("lpy_envy_00001", "lpy_envy_00009", "lpy_envy_00015", "lpy_envy_00041")
SAMPLE_STEP_M = 0.01
MIN_PART_PIXELS = 25


def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def reachable_parts(hierarchy: dict) -> list[str]:
    """Parts the generator renders: BFS from root; branch/nontrunk/trunk recurse, spurs are leaves."""
    out: list[str] = []
    seen: set[str] = set()
    queue = list(hierarchy.get("root", []))
    for name in queue:
        seen.add(name)
    while queue:
        parent = queue.pop(0)
        out.append(parent)
        for child in hierarchy.get(parent, []):
            if child in seen:
                continue
            low = child.lower()
            if low.startswith(("branch_", "nontrunk_", "trunk")):
                seen.add(child)
                queue.append(child)
            elif low.startswith("spur_"):
                seen.add(child)
                out.append(child)
    return out


def part_class(name: str) -> int:
    return PREFIX_TO_CLASS[name.split("_")[0].lower()]


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points p (N,3) to segments a-b (N,3) row-wise."""
    d = b - a
    t = np.clip(((p - a) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-18), 0.0, 1.0)
    return np.linalg.norm(a + t[..., None] * d - p, axis=-1)


@dataclass
class TreeModel:
    """World-frame ground truth for one tree."""

    tree_id: str
    location: np.ndarray
    seg_a: np.ndarray  # (M, 3) cylinder axis endpoints, world, reachable parts only
    seg_b: np.ndarray
    seg_r: np.ndarray  # (M,)
    seg_part: np.ndarray  # (M,) index into part_names
    part_names: list[str]
    part_cls: np.ndarray  # (P,)
    part_parent: np.ndarray  # (P,) index or -1
    part_points: list[np.ndarray]  # world polylines, base -> tip
    part_radius: list[np.ndarray]
    part_attach: list[np.ndarray | None]
    fit_residual_m: float = 0.0
    _kd: cKDTree | None = field(default=None, repr=False)
    _kd_seg: np.ndarray | None = field(default=None, repr=False)

    def segment_index(self) -> tuple[cKDTree, np.ndarray]:
        if self._kd is None:
            pts, owner = [], []
            for i, (a, b) in enumerate(zip(self.seg_a, self.seg_b)):
                n = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.005)) + 1)
                t = np.linspace(0.0, 1.0, n)[:, None]
                pts.append(a + t * (b - a))
                owner.append(np.full(n, i))
            self._kd = cKDTree(np.concatenate(pts))
            self._kd_seg = np.concatenate(owner)
        return self._kd, self._kd_seg


def load_metadata(tree_id: str, meta_dir: Path = METADATA_DIR) -> dict:
    return json.loads((Path(meta_dir) / f"{tree_id}_metadata.json").read_text())


def _world_centroids(tree_id: str, ann: dict | None, data_root: Path) -> np.ndarray:
    if ann is not None and ann.get("cylinders_world"):
        return np.asarray(ann["cylinders_world"], dtype=np.float64)
    side = Path(data_root) / "cylinders_world" / BARK / f"{tree_id}.json"
    if side.is_file():
        return np.asarray([c["centroid"] for c in json.loads(side.read_text())], dtype=np.float64)
    raise FileNotFoundError(f"no world centroids for {tree_id}: pass an annotation")


def _chain_part(cent: np.ndarray, length: np.ndarray, radius: np.ndarray, start: int):
    """Order cylinder centroids into a base->tip polyline (greedy nearest neighbour)."""
    n = len(cent)
    order = [start]
    left = set(range(n)) - {start}
    while left:
        cur = cent[order[-1]]
        nxt = min(left, key=lambda j: float(np.sum((cent[j] - cur) ** 2)))
        order.append(nxt)
        left.remove(nxt)
    c = cent[order]
    r = radius[order]
    if n == 1:
        return c, r, order
    d0 = c[0] - c[1]
    d0 /= max(np.linalg.norm(d0), 1e-12)
    d1 = c[-1] - c[-2]
    d1 /= max(np.linalg.norm(d1), 1e-12)
    base = c[0] + 0.5 * length[order[0]] * d0
    tip = c[-1] + 0.5 * length[order[-1]] * d1
    return np.vstack([base, c, tip]), np.concatenate([[r[0]], r, [r[-1]]]), order


def load_tree_model(
    tree_id: str,
    ann: dict | None = None,
    meta_dir: Path = METADATA_DIR,
    data_root: Path = DATA_ROOT,
) -> TreeModel:
    meta = load_metadata(tree_id, meta_dir)
    cyl = list(meta["cylinder_data"].values())
    C = np.array([c["centroid"] for c in cyl], dtype=np.float64)
    O = np.array([c["orientation"] for c in cyl], dtype=np.float64)
    L = np.array([c["length"] for c in cyl], dtype=np.float64)
    R = np.array([c["radius"] for c in cyl], dtype=np.float64)
    names = [c["part_name"] for c in cyl]

    Rt = rot_x(TILT_RAD)
    world = _world_centroids(tree_id, ann, data_root)
    if world.shape != C.shape:
        raise ValueError(f"{tree_id}: {len(world)} world centroids for {len(C)} cylinders")
    loc = np.median(world - C @ Rt.T, axis=0)
    resid = float(np.abs(C @ Rt.T + loc - world).max())
    if resid > 2e-3:
        raise ValueError(f"{tree_id}: world transform residual {resid:.4f} m (tilt changed?)")
    Cw = C @ Rt.T + loc
    Ow = O @ Rt.T
    A = Cw - 0.5 * L[:, None] * Ow
    B = Cw + 0.5 * L[:, None] * Ow

    hierarchy = meta["hierarchy"]
    parent_of = {c: p for p, cs in hierarchy.items() for c in cs}
    by_part: dict[str, list[int]] = {}
    for i, nm in enumerate(names):
        by_part.setdefault(nm, []).append(i)
    order = [p for p in reachable_parts(hierarchy) if p in by_part]
    index = {p: k for k, p in enumerate(order)}

    seg_keep = np.array([nm in index for nm in names])
    seg_part = np.array([index.get(nm, -1) for nm in names])[seg_keep]

    part_points, part_radius, part_attach = [], [], []
    part_parent = np.full(len(order), -1, dtype=np.int64)
    for k, name in enumerate(order):
        idx = np.array(by_part[name])
        par = parent_of.get(name)
        if par in index:
            part_parent[k] = index[par]
            pidx = np.array(by_part[par])
            # Start from the cylinder nearest the parent's axis.
            d = np.array(
                [
                    _segment_distance(np.repeat(Cw[i][None], len(pidx), 0), A[pidx], B[pidx]).min()
                    for i in idx
                ]
            )
            start = int(np.argmin(d))
        else:
            start = int(np.argmin(Cw[idx, 2]))
        pts, rad, _ = _chain_part(Cw[idx], L[idx], R[idx], start)
        part_points.append(pts)
        part_radius.append(rad)
    for k in range(len(order)):
        if part_parent[k] < 0:
            part_attach.append(None)
            continue
        _, q, _ = closest_point_on_polyline(part_points[part_parent[k]], part_points[k][0])
        part_attach.append(q)

    return TreeModel(
        tree_id=tree_id,
        location=loc,
        seg_a=A[seg_keep],
        seg_b=B[seg_keep],
        seg_r=R[seg_keep],
        seg_part=seg_part,
        part_names=order,
        part_cls=np.array([part_class(p) for p in order]),
        part_parent=part_parent,
        part_points=part_points,
        part_radius=part_radius,
        part_attach=part_attach,
        fit_residual_m=resid,
    )


@dataclass
class FrameGT:
    part_id: np.ndarray  # (H, W) int32: -1 background, -2 tree pixel without a part
    cls: np.ndarray  # (H, W) uint8 class id, IGNORE where unmatched
    graph: TreeGraph  # camera frame, visible parts only
    pixel_counts: dict[int, int]
    surface_residual_m: float


def backproject(us, vs, z, K) -> np.ndarray:
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=-1)


def label_frame(
    model: TreeModel,
    K: np.ndarray,
    T_wc: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    tol_m: float = 0.003,
    min_part_pixels: int = MIN_PART_PIXELS,
) -> FrameGT:
    K = np.asarray(K, dtype=np.float64)
    T_wc = np.asarray(T_wc, dtype=np.float64)
    h, w = depth.shape
    valid = mask & np.isfinite(depth) & (depth > 0.05) & (depth < 1e8)
    vs, us = np.nonzero(valid)
    z = depth[vs, us].astype(np.float64)
    R_wc, t_wc = T_wc[:3, :3], T_wc[:3, 3]
    pts_w = (backproject(us.astype(np.float64), vs.astype(np.float64), z, K) - t_wc) @ R_wc

    kd, owner = model.segment_index()
    _, nn = kd.query(pts_w, k=8)
    cand = owner[nn]  # (N, 8) segment ids
    a, b = model.seg_a[cand], model.seg_b[cand]
    surf = np.abs(_segment_distance(pts_w[:, None, :], a, b) - model.seg_r[cand])
    best = np.argmin(surf, axis=1)
    seg = cand[np.arange(len(cand)), best]
    err = surf[np.arange(len(cand)), best]
    ok = err < tol_m + 0.002 * z

    part_id = np.full((h, w), -1, dtype=np.int32)
    part_id[mask] = -2
    part_id[vs[ok], us[ok]] = model.seg_part[seg[ok]]
    cls = np.zeros((h, w), dtype=np.uint8)
    cls[mask] = IGNORE
    cls[vs[ok], us[ok]] = model.part_cls[model.seg_part[seg[ok]]]

    counts = np.bincount(model.seg_part[seg[ok]], minlength=len(model.part_names))
    graph = TreeGraph(frame="camera", meta={"tree": model.tree_id})
    labelled_v, labelled_u = vs[ok], us[ok]
    labelled_p = model.seg_part[seg[ok]]
    sort = np.argsort(labelled_p, kind="stable")
    labelled_v, labelled_u, labelled_p = labelled_v[sort], labelled_u[sort], labelled_p[sort]
    bounds = np.searchsorted(labelled_p, np.arange(len(model.part_names) + 1))

    for k in np.nonzero(counts >= min_part_pixels)[0]:
        pts_c = model.part_points[k] @ R_wc.T + t_wc
        pts, rad = resample_polyline(pts_c, SAMPLE_STEP_M, model.part_radius[k])
        zc = np.maximum(pts[:, 2], 1e-6)
        u = K[0, 0] * pts[:, 0] / zc + K[0, 2]
        v = K[1, 1] * pts[:, 1] / zc + K[1, 2]
        inside = (pts[:, 2] > 0.05) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        pix = np.stack(
            [labelled_u[bounds[k] : bounds[k + 1]], labelled_v[bounds[k] : bounds[k + 1]]], 1
        )
        vis = np.zeros(len(pts), dtype=bool)
        if inside.any() and len(pix):
            r_px = K[0, 0] * rad / zc
            win = np.maximum(2.0, 1.5 * r_px + 1.0)
            d, _ = cKDTree(pix).query(np.stack([u, v], 1)[inside])
            vis[inside] = d <= win[inside]
        if not vis.any():
            continue
        att = model.part_attach[k]
        par = int(model.part_parent[k])
        graph.add(
            Part(
                pid=int(k),
                cls=int(model.part_cls[k]),
                points=pts,
                radius=rad,
                parent=None if par < 0 else par,
                attach=None if att is None else R_wc @ att + t_wc,
                name=model.part_names[k],
                visible=vis,
            )
        )
    residual = float(np.median(err[ok])) if ok.any() else float("nan")
    return FrameGT(
        part_id=part_id,
        cls=cls,
        graph=graph,
        pixel_counts={int(k): int(c) for k, c in enumerate(counts) if c},
        surface_residual_m=residual,
    )


# ------------------------------------------------------------------ dataset index


@dataclass(frozen=True)
class FrameRef:
    tree: str
    rig: str
    shot: str
    side: str

    @property
    def stem(self) -> str:
        return f"{self.tree}_{self.shot}_{self.side}"

    def path(self, kind: str, root: Path = DATA_ROOT) -> Path:
        ext = {"Optical_flow": "png", "mask": "png", "ann": "json"}.get(kind, "npy")
        return Path(root) / kind / BARK / self.tree / self.rig / f"{self.stem}.{ext}"

    @property
    def key(self) -> str:
        return f"{self.tree}/{self.rig}/{self.stem}"


def list_frames(trees, rigs=RIGS, shots=SHOTS, sides=SIDES, root: Path = DATA_ROOT):
    out = []
    for t in trees:
        for r in rigs:
            for s in shots:
                for d in sides:
                    f = FrameRef(t, r, s, d)
                    if f.path("ann", root).is_file() and f.path("depth", root).is_file():
                        out.append(f)
    return out


def all_trees(root: Path = DATA_ROOT) -> list[str]:
    d = Path(root) / "ann" / BARK
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def load_frame(ref: FrameRef, root: Path = DATA_ROOT, depth_kind: str = "depth"):
    """(rgb uint8 HxWx3, depth float32 metres with 0 = invalid, mask bool, ann dict)."""
    from PIL import Image

    rgb = np.asarray(Image.open(ref.path("Optical_flow", root)).convert("RGB"))
    depth = np.load(ref.path(depth_kind, root)).astype(np.float32)
    depth[~np.isfinite(depth) | (depth >= 1e8)] = 0.0
    mask = np.asarray(Image.open(ref.path("mask", root)).convert("L")) > 0
    ann = json.loads(ref.path("ann", root).read_text())
    return rgb, depth, mask, ann


def frame_gt(ref: FrameRef, root: Path = DATA_ROOT, model: TreeModel | None = None):
    """Convenience: load one frame and label it. Returns (rgb, depth, mask, ann, K, T_wc, FrameGT)."""
    from spur_depth.pipeline.reconstruct import load_pose

    rgb, depth, mask, ann = load_frame(ref, root)
    K, T = load_pose(ann)
    model = model or load_tree_model(ref.tree, ann)
    return (
        rgb,
        depth,
        mask,
        ann,
        K.astype(np.float64),
        T.astype(np.float64),
        label_frame(model, K, T, depth, mask),
    )


def trunk_root(graph: TreeGraph) -> int | None:
    trunks = [p.pid for p in graph.parts.values() if p.cls == TRUNK]
    return trunks[0] if trunks else None
