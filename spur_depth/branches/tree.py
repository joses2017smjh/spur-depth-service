"""Tree graph: what the perception module returns and what the scorer compares.

A ``Part`` is one botanical organ (trunk, scaffold branch, shoot, spur) as a
metric axis polyline ordered base -> tip, with a radius per vertex. Parts
point at their parent; ``attach`` is the junction on the parent's axis. The
pruning-facing output is ``Part.cut()``: a point on the axis a fixed arc
length above the junction plus the local axis direction, which is what a
cutter needs to reach the point with its jaws perpendicular to the wood
(Jain, Grimm, Lee, ICRA 2025).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLASSES = ("background", "trunk", "branch", "shoot", "spur")
TRUNK, BRANCH, SHOOT, SPUR = 1, 2, 3, 4
# L-Py part-name prefix -> class id. "nontrunk" parts are the 0.3 m shoots on scaffolds.
PREFIX_TO_CLASS = {"trunk": TRUNK, "branch": BRANCH, "nontrunk": SHOOT, "spur": SPUR}
IGNORE = 255
CUT_OFFSET_M = 0.015


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def resample_polyline(points: np.ndarray, step: float, *extra: np.ndarray):
    """Evenly spaced samples along a polyline; ``extra`` per-vertex arrays are interpolated."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return (np.zeros((0, 3)), *[np.zeros((0,) + np.asarray(e).shape[1:]) for e in extra])
    if len(points) == 1:
        return (points.copy(), *[np.asarray(e, dtype=np.float64).copy() for e in extra])
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(np.floor(s[-1] / step)) + 1, 2)
    t = np.linspace(0.0, s[-1], n)
    out = np.stack([np.interp(t, s, points[:, k]) for k in range(3)], axis=1)
    res = [out]
    for e in extra:
        e = np.asarray(e, dtype=np.float64)
        if e.dtype == bool or e.ndim == 1:
            res.append(np.interp(t, s, e.astype(np.float64)))
        else:
            res.append(np.stack([np.interp(t, s, e[:, k]) for k in range(e.shape[1])], 1))
    return tuple(res)


def point_at_arclength(points: np.ndarray, s_target: float) -> tuple[np.ndarray, np.ndarray]:
    """(point, unit tangent) at arc length ``s_target`` from the first vertex."""
    points = np.asarray(points, dtype=np.float64)
    seg = np.diff(points, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-9
    if not keep.any():
        return points[0].copy(), np.array([0.0, 0.0, 1.0])
    seg, seg_len = seg[keep], seg_len[keep]
    starts = points[:-1][keep]
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    s_target = float(np.clip(s_target, 0.0, s[-1]))
    i = int(np.clip(np.searchsorted(s, s_target, side="right") - 1, 0, len(seg) - 1))
    frac = (s_target - s[i]) / seg_len[i]
    # Tangent over a ~3 cm window so a single kinked vertex does not dominate.
    lo = point_at_arclength_raw(starts, seg, s, max(s_target - 0.015, 0.0))
    hi = point_at_arclength_raw(starts, seg, s, min(s_target + 0.015, s[-1]))
    d = hi - lo
    n = np.linalg.norm(d)
    tangent = d / n if n > 1e-9 else seg[i] / seg_len[i]
    return starts[i] + frac * seg[i], tangent


def point_at_arclength_raw(starts, seg, s, s_target):
    i = int(np.clip(np.searchsorted(s, s_target, side="right") - 1, 0, len(seg) - 1))
    seg_len = np.linalg.norm(seg[i])
    frac = 0.0 if seg_len < 1e-12 else (s_target - s[i]) / seg_len
    return starts[i] + frac * seg[i]


def closest_point_on_polyline(points: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray, float]:
    """(distance, closest point, arc length of that point) from ``q`` to a polyline."""
    points = np.asarray(points, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if len(points) == 1:
        return float(np.linalg.norm(points[0] - q)), points[0].copy(), 0.0
    a, b = points[:-1], points[1:]
    d = b - a
    dd = np.maximum((d * d).sum(1), 1e-18)
    t = np.clip(((q - a) * d).sum(1) / dd, 0.0, 1.0)
    proj = a + t[:, None] * d
    dist = np.linalg.norm(proj - q, axis=1)
    j = int(np.argmin(dist))
    s = np.concatenate([[0.0], np.cumsum(np.sqrt(dd))])
    return float(dist[j]), proj[j], float(s[j] + t[j] * np.sqrt(dd[j]))


@dataclass
class Part:
    pid: int
    cls: int
    points: np.ndarray  # (N, 3) metres, base -> tip
    radius: np.ndarray  # (N,) metres
    parent: int | None = None
    attach: np.ndarray | None = None  # (3,) junction on the parent axis
    score: float = 1.0
    name: str = ""
    visible: np.ndarray | None = None  # (N,) bool; ground truth only
    # Predictions only: a cut direction fitted away from the junction (assemble.fit_cut_axes);
    # ground truth never sets it, so GT cuts keep the local tangent.
    cut_axis: np.ndarray | None = None

    @property
    def length(self) -> float:
        return polyline_length(self.points)

    def cut(self, offset: float = CUT_OFFSET_M) -> tuple[np.ndarray, np.ndarray]:
        """Cut point ``offset`` metres above the base and the axis direction there."""
        point, tangent = point_at_arclength(self.points, min(offset, 0.5 * self.length))
        return point, (tangent if self.cut_axis is None else self.cut_axis.copy())

    def to_json(self) -> dict:
        out = {
            "id": int(self.pid),
            "class": CLASSES[self.cls],
            "parent": None if self.parent is None else int(self.parent),
            "points_m": np.round(self.points, 5).tolist(),
            "radius_m": np.round(self.radius, 5).tolist(),
            "length_m": round(self.length, 5),
            "score": round(float(self.score), 4),
        }
        if self.name:
            out["name"] = self.name
        if self.attach is not None:
            out["attach_m"] = np.round(self.attach, 5).tolist()
        if self.cls != TRUNK and self.length > 0:
            p, d = self.cut()
            out["cut"] = {"point_m": np.round(p, 5).tolist(), "axis": np.round(d, 5).tolist()}
        return out


@dataclass
class TreeGraph:
    """Parts in one frame. ``frame`` is "camera" (OpenCV: x right, y down, z forward)."""

    parts: dict[int, Part] = field(default_factory=dict)
    frame: str = "camera"
    meta: dict = field(default_factory=dict)

    def add(self, part: Part) -> None:
        self.parts[part.pid] = part

    def edges(self) -> list[tuple[int, int]]:
        return [(p.pid, p.parent) for p in self.parts.values() if p.parent is not None]

    def children(self, pid: int) -> list[int]:
        return [p.pid for p in self.parts.values() if p.parent == pid]

    def roots(self) -> list[int]:
        return [p.pid for p in self.parts.values() if p.parent is None]

    def transformed(self, R: np.ndarray, t: np.ndarray, frame: str) -> TreeGraph:
        """Apply ``x -> R x + t`` to every point (e.g. camera -> world)."""
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        out = TreeGraph(frame=frame, meta=dict(self.meta))
        for p in self.parts.values():
            out.add(
                Part(
                    pid=p.pid,
                    cls=p.cls,
                    points=p.points @ R.T + t,
                    radius=p.radius.copy(),
                    parent=p.parent,
                    attach=None if p.attach is None else R @ p.attach + t,
                    score=p.score,
                    name=p.name,
                    visible=None if p.visible is None else p.visible.copy(),
                    cut_axis=None if p.cut_axis is None else R @ p.cut_axis,
                )
            )
        return out

    def to_json(self) -> dict:
        counts = {c: 0 for c in CLASSES[1:]}
        for p in self.parts.values():
            counts[CLASSES[p.cls]] += 1
        return {
            "frame": self.frame,
            "units": "m",
            "convention": "OpenCV camera: x right, y down, z forward"
            if self.frame == "camera"
            else self.frame,
            "n_parts": len(self.parts),
            "counts": counts,
            "edges": [[c, p] for c, p in self.edges()],
            "parts": [p.to_json() for p in sorted(self.parts.values(), key=lambda q: q.pid)],
            "meta": self.meta,
        }

    @staticmethod
    def from_json(d: dict) -> TreeGraph:
        g = TreeGraph(frame=d.get("frame", "camera"), meta=d.get("meta", {}))
        for p in d["parts"]:
            g.add(
                Part(
                    pid=int(p["id"]),
                    cls=CLASSES.index(p["class"]),
                    points=np.asarray(p["points_m"], dtype=np.float64).reshape(-1, 3),
                    radius=np.asarray(p["radius_m"], dtype=np.float64),
                    parent=p.get("parent"),
                    attach=None if p.get("attach_m") is None else np.asarray(p["attach_m"]),
                    score=float(p.get("score", 1.0)),
                    name=p.get("name", ""),
                )
            )
        return g
