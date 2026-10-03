"""Figures for the branch module: 2D graph overlays and 3D orbits.

Colours follow the reference data-viz palette: three categorical slots (the
most that stay distinguishable for colour-blind readers in every pair), so
shoots and spurs share the tertiary colour and differ by line width. Ground
truth is drawn in muted grey; text stays in ink colours.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from spur_depth.branches.tree import BRANCH, SHOOT, SPUR, TRUNK, TreeGraph

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HEX = {TRUNK: "#2a78d6", BRANCH: "#eb6834", SHOOT: "#1baf7a", SPUR: "#1baf7a"}
WIDTH = {TRUNK: 2.6, BRANCH: 2.0, SHOOT: 1.4, SPUR: 1.0}
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"


def _rgb(hexcode: str) -> tuple[int, int, int]:
    h = hexcode.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def project(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.maximum(points[:, 2], 1e-6)
    u = K[0, 0] * points[:, 0] / z + K[0, 2]
    v = K[1, 1] * points[:, 1] / z + K[1, 2]
    return np.stack([u, v], 1), points[:, 2] > 0.05


def draw_graph_2d(
    rgb: np.ndarray,
    graph: TreeGraph,
    K: np.ndarray,
    scale: float = 1.0,
    dim: float = 0.45,
    cuts: bool = True,
    gt: TreeGraph | None = None,
) -> np.ndarray:
    """RGB (dimmed) with parts by class, junctions and cut points. Returns uint8 RGB."""
    img = (rgb.astype(np.float32) * dim + 255.0 * (1 - dim) * 0.15).clip(0, 255).astype(np.uint8)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    Ks = K.copy()
    Ks[:2] *= scale
    aa = cv2.LINE_AA
    if gt is not None:
        for p in gt.parts.values():
            uv, ok = project(p.points, Ks)
            vis = ok & (p.visible if p.visible is not None else True)
            for i in range(len(uv) - 1):
                if vis[i] and vis[i + 1]:
                    cv2.line(
                        img, tuple(np.int32(uv[i])), tuple(np.int32(uv[i + 1])), _rgb(MUTED), 1, aa
                    )
    order = sorted(graph.parts.values(), key=lambda q: -float(np.median(q.points[:, 2])))
    for p in order:
        uv, ok = project(p.points, Ks)
        col = _rgb(HEX[p.cls])
        th = max(1, int(round(WIDTH[p.cls] * scale * 1.6)))
        pts = np.int32(uv[ok]).reshape(-1, 1, 2)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, col, th, aa)
    for p in graph.parts.values():
        if p.attach is not None:
            (uv,), ok = project(p.attach[None], Ks)
            if ok[0]:
                cv2.circle(
                    img, tuple(np.int32(uv)), max(2, int(3 * scale * 1.6)), (255, 255, 255), 1, aa
                )
        if cuts and p.cls != TRUNK and p.parent is not None and p.length > 0:
            c, d = p.cut()
            seg = np.stack([c - 0.02 * d, c + 0.02 * d])
            uv, ok = project(seg, Ks)
            if ok.all():
                cv2.line(
                    img,
                    tuple(np.int32(uv[0])),
                    tuple(np.int32(uv[1])),
                    (255, 255, 255),
                    max(1, int(2 * scale)),
                    aa,
                )
    return img


def label(img: np.ndarray, text: str, y: int = 22, x: int = 12, size: float = 0.6) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_gif(frames: list[np.ndarray], path: Path, duration: int = 500, colors: int = 128) -> None:
    ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=colors) for f in frames]
    ims[0].save(
        path, save_all=True, append_images=ims[1:], duration=duration, loop=0, optimize=True
    )


def orbit_frames(
    world_graphs: list[TreeGraph],
    gt_segments: tuple[np.ndarray, np.ndarray] | None = None,
    azimuths=range(-90, 270, 12),
    elev: float = 12.0,
    size=(6.4, 6.0),
    title: str = "",
    cut_points: bool = True,
) -> list[np.ndarray]:
    """Matplotlib 3D orbit of graphs already in one frame (e.g. world). Z is up."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    segs = {c: [] for c in (TRUNK, BRANCH, SHOOT, SPUR)}
    cuts = []
    allpts = []
    for g in world_graphs:
        for p in g.parts.values():
            if len(p.points) > 1:
                segs[p.cls].append(np.stack([p.points[:-1], p.points[1:]], 1))
                allpts.append(p.points)
            if cut_points and p.cls != TRUNK and p.parent is not None and p.length > 0:
                cuts.append(p.cut()[0])
    pts = np.concatenate(allpts) if allpts else np.zeros((1, 3))
    lo = np.percentile(pts, 1, axis=0) - 0.05
    hi = np.percentile(pts, 99, axis=0) + 0.05
    frames = []
    for az in azimuths:
        fig = plt.figure(figsize=size, facecolor=SURFACE)
        ax = fig.add_axes([0.0, 0.0, 1.0, 0.92], projection="3d", facecolor=SURFACE)
        if gt_segments is not None:
            a, b = gt_segments
            ax.add_collection3d(Line3DCollection(np.stack([a, b], 1), colors=MUTED, linewidths=0.5))
        for c in (SPUR, SHOOT, BRANCH, TRUNK):
            if segs[c]:
                ax.add_collection3d(
                    Line3DCollection(np.concatenate(segs[c]), colors=HEX[c], linewidths=WIDTH[c])
                )
        if cuts:
            cp = np.array(cuts)
            ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], s=7, c=INK, depthshade=False, linewidths=0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-3)))
        ax.view_init(elev=elev, azim=az)
        ax.set_axis_off()
        if title:
            fig.text(0.5, 0.95, title, ha="center", va="center", color=INK, fontsize=10)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return frames


def legend_strip(width: int, height: int = 30) -> np.ndarray:
    """Plain legend row: class colours, grey GT, white ring = junction, tick = cut axis."""
    img = np.full((height, width, 3), _rgb(SURFACE), np.uint8)
    x = 12
    items = [
        (HEX[TRUNK], "trunk"),
        (HEX[BRANCH], "branch"),
        (HEX[SPUR], "shoot / spur"),
        (MUTED, "ground truth"),
    ]
    for col, text in items:
        cv2.line(img, (x, height // 2), (x + 22, height // 2), _rgb(col), 3, cv2.LINE_AA)
        cv2.putText(
            img,
            text,
            (x + 28, height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            _rgb(INK_2),
            1,
            cv2.LINE_AA,
        )
        x += 40 + 9 * len(text)
    cv2.circle(img, (x + 6, height // 2), 5, _rgb(INK_2), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        "junction   white tick: cut point + axis",
        (x + 16, height // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        _rgb(INK_2),
        1,
        cv2.LINE_AA,
    )
    return img
