"""Multi-view fusion: per-frame predicted graphs -> one world-frame tree graph.

A sweep of frames of one tree (e.g. one rig's six heights, both stereo eyes)
gives several noisy, partial views of every part. Each frame's graph is moved
to the world frame with its logged pose, then:

1. parts are associated across frames by 3D overlap (a part joins the track
   whose samples cover most of its own);
2. each track's members are merged into one polyline, the per-centimetre
   median of all member samples along the longest member (plus whatever other
   members see beyond its ends), so independent depth noise averages out;
3. every member votes for its own frame's parent, mapped to that parent's
   track, and a maximum spanning arborescence over the votes connects the
   tracks (a raw vote can cycle); tracks with no parent votes become roots.

``view_graph`` puts the fused graph back into one frame and keeps the parts
that frame's own segmentation sees, so the fused model is scored per frame
with the same scorer and ground truth as a single-view prediction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2
import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from spur_depth.branches.tree import (
    TRUNK,
    Part,
    TreeGraph,
    closest_point_on_polyline,
    polyline_length,
    resample_polyline,
)


@dataclass
class FuseConfig:
    assoc_tol_m: float = 0.03  # a sample is covered by a track within this distance
    assoc_min_frac: float = 0.5  # share of a part's samples a track must cover to absorb it
    min_views: int = 1  # tracks seen in fewer distinct frames are dropped
    step_m: float = 0.01  # merged polylines are sampled at this spacing
    orphan_weight: float = 0.1  # root-edge weight per member that had no parent
    smooth_m: float = 0.0  # Gaussian smoothing of merged axes along arc length (sigma)
    # Extend a merged axis past its base with what other views see there. Off keeps the base
    # (the junction end) of the longest member, since skeletons bend into the parent there.
    extend_base: bool = True
    view_min_samples: int = 3  # view_graph keeps parts with this many samples on wood
    view_dilate_px: int = 4
    # "seen": every fused part the frame's wood mask sees; "anchored": only the parts this
    # frame itself detected (fused geometry and parents, no parts added from other views).
    view_mode: str = "seen"


def to_world(graph: TreeGraph, T_wc: np.ndarray) -> TreeGraph:
    """Camera-frame graph -> world frame, for a world-to-camera pose ``T_wc``."""
    R, t = np.asarray(T_wc)[:3, :3], np.asarray(T_wc)[:3, 3]
    return graph.transformed(R.T, -R.T @ t, "world")


def to_camera(graph: TreeGraph, T_wc: np.ndarray) -> TreeGraph:
    R, t = np.asarray(T_wc)[:3, :3], np.asarray(T_wc)[:3, 3]
    return graph.transformed(R, t, "camera")


@dataclass
class _Member:
    frame: int
    pid: int
    pts: np.ndarray  # resampled at step_m
    rad: np.ndarray
    cls: int
    score: float
    parent: int | None


def _merge(
    members: list[_Member], tol: float, step: float, extend_base: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Median polyline of the members along the longest one, extended past its ends."""
    ref = max(members, key=lambda m: len(m.pts))
    if len(ref.pts) < 2 or len(members) == 1:
        return ref.pts.copy(), ref.rad.copy()
    n_ref = len(ref.pts)
    l_ref = polyline_length(ref.pts)
    spacing = l_ref / (n_ref - 1)
    bins: dict[int, list] = defaultdict(list)  # index along ref (may be <0 or >=n_ref)
    t_base = ref.pts[0] - ref.pts[min(2, n_ref - 1)]
    t_tip = ref.pts[-1] - ref.pts[max(n_ref - 3, 0)]
    t_base /= max(np.linalg.norm(t_base), 1e-9)
    t_tip /= max(np.linalg.norm(t_tip), 1e-9)
    for m in members:
        for q, r in zip(m.pts, m.rad):
            d, _, s = closest_point_on_polyline(ref.pts, q)
            beyond_b = float((q - ref.pts[0]) @ t_base)
            beyond_t = float((q - ref.pts[-1]) @ t_tip)
            if s <= 1e-9 and beyond_b > 0.5 * step:
                lateral = np.linalg.norm((q - ref.pts[0]) - beyond_b * t_base)
                if extend_base and lateral <= tol:
                    bins[-int(np.ceil(beyond_b / step))].append((q, r))
                continue
            if s >= l_ref - 1e-6 and beyond_t > 0.5 * step:
                lateral = np.linalg.norm((q - ref.pts[-1]) - beyond_t * t_tip)
                if lateral <= tol:
                    bins[n_ref - 1 + int(np.ceil(beyond_t / step))].append((q, r))
                continue
            if d <= tol:
                bins[int(np.clip(round(s / spacing), 0, n_ref - 1))].append((q, r))
    keys = sorted(bins)
    pts = np.array([np.median(np.array([q for q, _ in bins[k]]), axis=0) for k in keys])
    rad = np.array([np.median([r for _, r in bins[k]]) for k in keys])
    if len(pts) < 2:
        return ref.pts.copy(), ref.rad.copy()
    (pts, rad) = resample_polyline(pts, step, rad)
    return pts, rad


def fuse(graphs: list[TreeGraph], cfg: FuseConfig | None = None) -> TreeGraph:
    """World-frame graphs of one tree -> one fused world-frame graph."""
    cfg = cfg or FuseConfig()
    tracks: list[list[_Member]] = []
    member_track: dict[tuple[int, int], int] = {}
    for f, g in enumerate(graphs):
        # Tracks from earlier frames only: fragments of one frame never absorb each other.
        if tracks:
            samples = np.concatenate([m.pts for t in tracks for m in t])
            owner = np.concatenate(
                [np.full(len(m.pts), k) for k, t in enumerate(tracks) for m in t]
            )
            tree = cKDTree(samples)
        else:
            tree = None
        new_tracks: list[list[_Member]] = []
        for p in sorted(g.parts.values(), key=lambda q: -q.length):
            if len(p.points) < 2:
                continue
            pts, rad = resample_polyline(p.points, cfg.step_m, p.radius)
            m = _Member(f, p.pid, pts, rad, p.cls, p.score, p.parent)
            best, best_frac = None, 0.0
            if tree is not None:
                hits = tree.query_ball_point(pts, cfg.assoc_tol_m)
                cover: Counter = Counter()
                for h in hits:
                    for k in set(owner[h].tolist()):
                        cover[k] += 1
                if cover:
                    best, n = cover.most_common(1)[0]
                    best_frac = n / len(pts)
            if best is not None and best_frac >= cfg.assoc_min_frac:
                tracks[best].append(m)
                member_track[(f, p.pid)] = best
            else:
                member_track[(f, p.pid)] = len(tracks) + len(new_tracks)
                new_tracks.append([m])
        tracks.extend(new_tracks)

    keep = [k for k, t in enumerate(tracks) if len({m.frame for m in t}) >= cfg.min_views]
    merged: dict[int, tuple[np.ndarray, np.ndarray, int, float]] = {}
    for k in keep:
        t = tracks[k]
        pts, rad = _merge(t, cfg.assoc_tol_m, cfg.step_m, cfg.extend_base)
        if cfg.smooth_m > 0 and len(pts) > 2:
            sig = cfg.smooth_m / max(polyline_length(pts) / (len(pts) - 1), 1e-6)
            pts = gaussian_filter1d(pts, sig, axis=0, mode="nearest")
        votes: Counter = Counter()
        for m in t:
            votes[m.cls] += len(m.pts)
        merged[k] = (pts, rad, votes.most_common(1)[0][0], float(np.mean([m.score for m in t])))

    # Parent votes -> maximum spanning arborescence from a virtual root.
    G = nx.DiGraph()
    G.add_node("root")
    for k in merged:
        G.add_node(k)
    pv: Counter = Counter()
    orphan: Counter = Counter()
    for k in merged:
        for m in tracks[k]:
            par = None if m.parent is None else member_track.get((m.frame, m.parent))
            if par is None or par not in merged or par == k:
                orphan[k] += 1
            else:
                pv[(par, k)] += 1
    for k, (_, _, c, _) in merged.items():
        w = 1e-3 + cfg.orphan_weight * orphan[k]
        G.add_edge("root", k, weight=1e3 if c == TRUNK else w)
    for (par, k), n in pv.items():
        if merged[k][2] != TRUNK:
            G.add_edge(par, k, weight=float(n))
    arb = nx.maximum_spanning_arborescence(G, attr="weight")
    parent_of = {v: u for u, v in arb.edges()}

    frames = {k: sorted({m.frame for m in tracks[k]}) for k in merged}
    out = TreeGraph(
        frame="world",
        meta={"fused_views": len(graphs), "tracks": len(merged), "track_frames": frames},
    )
    for k, (pts, rad, c, score) in merged.items():
        out.add(Part(pid=k, cls=c, points=pts, radius=rad, score=score))
    for k, part in out.parts.items():
        par = parent_of.get(k)
        if par is None or par == "root":
            continue
        part.parent = par
        _, part.attach, _ = closest_point_on_polyline(out.parts[par].points, part.points[0])
    return out


def view_graph(
    fused: TreeGraph,
    T_wc: np.ndarray,
    K: np.ndarray,
    wood: np.ndarray,
    cfg: FuseConfig | None = None,
    frame: int | None = None,
) -> TreeGraph:
    """The fused graph in one camera, keeping the parts that camera's own wood mask sees.

    ``frame`` is the camera's index in the list given to ``fuse``; "anchored" mode needs it.
    """
    cfg = cfg or FuseConfig()
    if cfg.view_mode not in ("seen", "anchored"):
        raise ValueError(f"unknown view_mode {cfg.view_mode!r}")
    if cfg.view_mode == "anchored" and frame is None:
        raise ValueError("view_mode 'anchored' needs the frame index")
    cam = to_camera(fused, T_wc)
    track_frames = fused.meta.get("track_frames", {})
    h, w = wood.shape
    k = 2 * cfg.view_dilate_px + 1
    seen = cv2.dilate(wood.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    keep = set()
    for p in cam.parts.values():
        if cfg.view_mode == "anchored" and frame not in track_frames.get(p.pid, ()):
            continue
        z = p.points[:, 2]
        ok = z > 0.05
        u = np.round(K[0, 0] * p.points[ok, 0] / z[ok] + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * p.points[ok, 1] / z[ok] + K[1, 2]).astype(int)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if int(seen[v[inside], u[inside]].sum()) >= cfg.view_min_samples:
            keep.add(p.pid)
    out = TreeGraph(frame="camera", meta=dict(cam.meta))
    for pid in keep:
        p = cam.parts[pid]
        if p.parent not in keep:
            p.parent, p.attach = None, None
        out.add(p)
    return out
