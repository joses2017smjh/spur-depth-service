"""Class map + metric depth -> TreeGraph (detect, 3D-localize, connect).

1. Skeletonize the predicted wood mask and split it into pixel segments.
2. Lift every skeleton vertex to 3D: robust depth over the local cross-section
   plus the radius, because a depth camera sees the bark, not the axis.
3. A pixel junction is only a junction if its segment ends also meet in 3D;
   ends more than a few centimetres apart in depth are a crossing in the
   image, not wood touching wood. This is where depth earns its place.
4. Straight, same-class continuations through a junction are one part;
   same-class collinear parts across a short gap are one occluded part.
5. Every part then picks its parent jointly with all others: a minimum-cost
   spanning arborescence (Edmonds) over 3D attachment candidates, with a
   botanical prior (spurs and shoots on branches, branches on the trunk) and
   an explicit orphan option so out-of-view parents do not force bad edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import networkx as nx
import numpy as np

from spur_depth.branches.skeleton import (
    merge_close_junctions,
    prune_spurs,
    skeleton_graph,
    thin,
)
from spur_depth.branches.tree import (
    BRANCH,
    IGNORE,
    SHOOT,
    SPUR,
    TRUNK,
    Part,
    TreeGraph,
    closest_point_on_polyline,
)

# Child -> parent compatibility cost; missing pairs are not allowed.
CLASS_COST = {
    (SPUR, BRANCH): 0.0,
    (SPUR, SHOOT): 0.0,
    (SPUR, TRUNK): 0.2,
    (SPUR, SPUR): 1.0,
    (SHOOT, BRANCH): 0.0,
    (SHOOT, TRUNK): 0.4,
    (SHOOT, SHOOT): 1.0,
    (BRANCH, TRUNK): 0.0,
    (BRANCH, BRANCH): 0.8,
}


@dataclass
class AssembleConfig:
    min_component_px: int = 40
    whisker_ratio: float = 2.0
    whisker_min_px: float = 6.0
    merge_ratio: float = 2.5
    merge_min_px: float = 8.0
    skip_radii: float = 2.0  # tangents start this many radii away from a junction
    vertex_step_px: int = 3
    depth_pad_px: int = 1
    surface_to_axis: float = 0.7  # axis depth = robust surface depth + k * radius
    depth_percentile: float = 25.0
    z_min: float = 0.2
    z_max: float = 12.0
    glue_gap_m: float = 0.03
    continue_deg: float = 30.0
    bridge_gap_m: float = 0.06
    bridge_deg: float = 20.0
    attach_tol_m: float = 0.04
    attach_depth_frac: float = 0.01
    parallel_penalty: float = 0.3
    junction_bonus: float = 0.3
    orphan_cost: float = 2.0
    min_part_len_m: float = 0.015
    stub_radii: float = 4.0  # childless parts shorter than this many radii are stubs
    min_thick_len_m: float = 0.05  # a childless trunk/branch piece shorter than this is a stub
    tangent_len_m: float = 0.03
    geometric_classes: bool = False  # baseline only: classes from radius/length


@dataclass
class _Seg:
    cls: int
    pts: np.ndarray  # (k, 3)
    rad: np.ndarray  # (k,)
    pix: np.ndarray  # (k, 2) y, x
    a: int  # skeleton node ids at the two ends
    b: int
    conf: float


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def _end_dir(pts: np.ndarray, at_start: bool, length: float, skip: float = 0.0) -> np.ndarray:
    """Unit direction pointing from an end into the polyline.

    Measured from arc length ``skip`` to ``skip + length`` so the first
    vertices, which thinning bends toward a junction, do not set the angle.
    Short polylines fall back to their chord.
    """
    p = pts if at_start else pts[::-1]
    if len(p) < 2:
        return np.zeros(3)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s0 = min(skip, 0.3 * s[-1])
    s1 = min(s0 + length, s[-1])
    a = np.array([np.interp(s0, s, p[:, k]) for k in range(3)])
    b = np.array([np.interp(s1, s, p[:, k]) for k in range(3)])
    v = _unit(b - a)
    return v if np.any(v) else _unit(p[-1] - p[0])


def _robust_z(z: np.ndarray, k: int = 9) -> np.ndarray:
    """Median-filter a depth profile, mark outliers and fill them by interpolation."""
    ok = z > 0
    if ok.sum() < 2:
        return np.where(ok, z, 0.0)
    idx = np.arange(len(z))
    zi = np.interp(idx, idx[ok], z[ok])
    k = min(k, len(z) if len(z) % 2 else len(z) - 1) if len(z) > 2 else 1
    pad = np.pad(zi, k // 2, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    mad = np.median(np.abs(zi[ok] - med[ok])) + 1e-6
    good = ok & (np.abs(z - med) < np.maximum(0.03, 4.0 * 1.4826 * mad))
    if good.sum() < 2:
        return np.where(good, z, 0.0)
    return np.interp(idx, idx[good], z[good])


def _lift(cls_map, depth, dt, K, path, cls, cfg: AssembleConfig):
    """3D vertices and radii along one pixel path (every ``vertex_step_px``)."""
    h, w = depth.shape
    sel = np.unique(np.r_[np.arange(0, len(path), cfg.vertex_step_px), len(path) - 1])
    pix = path[sel]
    f = float(K[0, 0])
    zs = np.zeros(len(pix))
    rp = np.zeros(len(pix))
    for i, (y, x) in enumerate(pix):
        r = max(float(dt[y, x]) - 0.5, 0.5)
        half = int(np.ceil(r)) + cfg.depth_pad_px
        y0, y1, x0, x1 = (
            max(y - half, 0),
            min(y + half + 1, h),
            max(x - half, 0),
            min(x + half + 1, w),
        )
        dz = depth[y0:y1, x0:x1]
        same = (cls_map[y0:y1, x0:x1] == cls) & (dz > cfg.z_min) & (dz < cfg.z_max)
        # Sensor failures on thin wood read too far (background or flying pixels),
        # so a low percentile of the cross-section is the robust surface depth.
        zs[i] = float(np.percentile(dz[same], cfg.depth_percentile)) if same.any() else 0.0
        rp[i] = r
    zs = _robust_z(zs)
    ok = zs > 0
    if ok.sum() < 2:
        return None
    # Vertices without usable depth are dropped, never placed at the camera centre.
    zs, rp, pix = zs[ok], rp[ok], pix[ok]
    rad = rp * zs / f
    za = zs + cfg.surface_to_axis * rad
    x = (pix[:, 1] - K[0, 2]) / K[0, 0] * za
    y = (pix[:, 0] - K[1, 2]) / K[1, 1] * za
    return np.stack([x, y, za], 1), rad, pix


def _split_by_class(path: np.ndarray, cls_map: np.ndarray, dt: np.ndarray, ratio: float = 2.5):
    """Runs of constant class along a path.

    A child's skeleton starts on its parent's medial axis and crosses the
    parent's own pixels first, so runs shorter than ``ratio`` local radii are
    absorbed into their longer neighbour instead of becoming stub parts.
    """
    labels = cls_map[path[:, 0], path[:, 1]].astype(int)
    labels = np.where(labels == IGNORE, 0, labels)
    # Fill unknown with the nearest known label along the path.
    known = labels > 0
    if not known.any():
        return []
    idx = np.arange(len(labels))
    labels = labels[known][np.abs(idx[:, None] - idx[known][None, :]).argmin(1)]
    runs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append([start, i, int(labels[start])])
            start = i
    width = dt[path[:, 0], path[:, 1]]

    def too_short(s: int, e: int) -> bool:
        return e - s < max(4.0, ratio * float(width[s:e].max()))

    # Absorb short runs into their longer neighbour.
    while len(runs) > 1:
        short = [k for k, (s, e, _) in enumerate(runs) if too_short(s, e)]
        if not short:
            break
        k = short[0]
        left = runs[k - 1][1] - runs[k - 1][0] if k > 0 else -1
        right = runs[k + 1][1] - runs[k + 1][0] if k + 1 < len(runs) else -1
        nb = k - 1 if left >= right else k + 1
        lo, hi = min(k, nb), max(k, nb)
        runs[lo] = [runs[lo][0], runs[hi][1], runs[nb][2]]
        del runs[hi]
    return runs


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.p[self.find(i)] = self.find(j)


def _chain(segs: list[_Seg], members: list[int], glued: set[tuple[int, int, int, int]]):
    """Order glued segments into one polyline. ``glued`` holds (seg, end, seg, end)."""
    link: dict[tuple[int, int], tuple[int, int]] = {}
    for s1, e1, s2, e2 in glued:
        if s1 in members:
            link[(s1, e1)] = (s2, e2)
            link[(s2, e2)] = (s1, e1)
    start = None
    for s in members:
        for e in (0, 1):
            if (s, e) not in link:
                start = (s, e)
                break
        if start:
            break
    if start is None:
        start = (members[0], 0)
    order, seen, cur = [], set(), start
    while cur and cur[0] not in seen:
        s, e = cur
        seen.add(s)
        order.append((s, e))
        cur = link.get((s, 1 - e))
    for s in members:  # anything a cycle left out
        if s not in seen:
            order.append((s, 0))
    pts, rad, pix = [], [], []
    for s, e in order:
        sg = segs[s]
        p, r, q = (sg.pts, sg.rad, sg.pix) if e == 0 else (sg.pts[::-1], sg.rad[::-1], sg.pix[::-1])
        pts.append(p)
        rad.append(r)
        pix.append(q)
    return np.concatenate(pts), np.concatenate(rad), np.concatenate(pix), order


def _geometric_class(pts: np.ndarray, rad: np.ndarray) -> int:
    """Baseline-only classes from L-Py regularities (10 mm wood vs 3 mm, 0.2 m spurs)."""
    r = float(np.median(rad))
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    if r >= 0.0065:
        return BRANCH
    return SPUR if length < 0.25 else SHOOT


def assemble(
    cls_map: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    conf: np.ndarray | None = None,
    cfg: AssembleConfig | None = None,
) -> TreeGraph:
    """Build a camera-frame TreeGraph from a per-pixel class map (0 = background)."""
    cfg = cfg or AssembleConfig()
    K = np.asarray(K, dtype=np.float64)
    cls_map = np.asarray(cls_map)
    depth = np.asarray(depth, dtype=np.float64)
    fg = (cls_map > 0) & (cls_map != 0)
    n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    small = np.nonzero(stats[:, cv2.CC_STAT_AREA] < cfg.min_component_px)[0]
    fg &= ~np.isin(cc, small[small > 0])
    graph = TreeGraph(frame="camera")
    if not fg.any():
        return graph
    # The image border counts as background, so radii stay finite on a full mask.
    dt = cv2.distanceTransform(np.pad(fg, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    sk = skeleton_graph(thin(fg))

    def whisker_len(s) -> float:
        ends = [s.path[0], s.path[-1]]
        return max(cfg.whisker_min_px, cfg.whisker_ratio * max(dt[y, x] for y, x in ends))

    sk = prune_spurs(sk, whisker_len)

    def ladder_len(s) -> float:
        ends = [s.path[0], s.path[-1]]
        return max(cfg.merge_min_px, cfg.merge_ratio * max(dt[y, x] for y, x in ends))

    sk = merge_close_junctions(sk, ladder_len)

    # Segments split where the class changes; split points become new nodes.
    segs: list[_Seg] = []
    next_node = max(sk.nodes, default=0) + 10_000_000
    for s in sk.segments:
        runs = _split_by_class(s.path, cls_map, dt)
        for k, (i0, i1, c) in enumerate(runs):
            if c not in (TRUNK, BRANCH, SHOOT, SPUR):
                continue
            path = s.path[max(i0 - (k > 0), 0) : i1]
            if len(path) < 2:
                continue
            a = s.a if k == 0 else next_node + k - 1
            b = s.b if k == len(runs) - 1 else next_node + k
            lifted = _lift(cls_map, depth, dt, K, path, c, cfg)
            if lifted is None:
                continue
            pts, rad, pix = lifted
            cf = float(np.mean(conf[path[:, 0], path[:, 1]])) if conf is not None else 1.0
            segs.append(_Seg(c, pts, rad, pix, a, b, cf))
        next_node += max(len(runs), 1)
    if not segs:
        return graph

    # Ends meeting at the same skeleton node.
    ends_at: dict[int, list[tuple[int, int]]] = {}
    for i, s in enumerate(segs):
        ends_at.setdefault(s.a, []).append((i, 0))
        ends_at.setdefault(s.b, []).append((i, 1))

    def end_pt(i: int, e: int) -> np.ndarray:
        return segs[i].pts[0] if e == 0 else segs[i].pts[-1]

    def end_dir(i: int, e: int) -> np.ndarray:
        r = float(segs[i].rad[0 if e == 0 else -1])
        return _end_dir(segs[i].pts, e == 0, cfg.tangent_len_m, cfg.skip_radii * r)

    uf = _UF(len(segs))
    glued: set[tuple[int, int, int, int]] = set()
    evidence: set[tuple[int, int]] = set()  # (seg, other seg) met in 3D at a junction
    for ends in ends_at.values():
        if len(ends) < 2:
            continue
        # Depth-consistent clusters of ends (single linkage).
        m = len(ends)
        cl = _UF(m)
        for x in range(m):
            for y in range(x + 1, m):
                (i, e), (j, f) = ends[x], ends[y]
                r = segs[i].rad[0 if e == 0 else -1] + segs[j].rad[0 if f == 0 else -1]
                pi, pj = end_pt(i, e), end_pt(j, f)
                # A crossing in the image is a jump in depth, not a lateral offset.
                tol = cfg.glue_gap_m + 2.0 * r + cfg.attach_depth_frac * float(pi[2])
                if abs(float(pi[2] - pj[2])) < tol:
                    cl.union(x, y)
        groups: dict[int, list[int]] = {}
        for x in range(m):
            groups.setdefault(cl.find(x), []).append(x)
        for grp in groups.values():
            cand = []
            for ix, x in enumerate(grp):
                for y in grp[ix + 1 :]:
                    (i, e), (j, f) = ends[x], ends[y]
                    if i == j or segs[i].cls != segs[j].cls:
                        continue
                    cosang = float(np.dot(end_dir(i, e), -end_dir(j, f)))
                    defl = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
                    if defl < cfg.continue_deg:
                        cand.append((defl, x, y))
            used: set[int] = set()
            for _, x, y in sorted(cand):
                if x in used or y in used:
                    continue
                used.update((x, y))
                (i, e), (j, f) = ends[x], ends[y]
                glued.add((i, e, j, f))
                uf.union(i, j)
            for x in grp:
                for y in grp:
                    if x != y:
                        evidence.add((ends[x][0], ends[y][0]))

    # Parts = glued chains.
    members: dict[int, list[int]] = {}
    for i in range(len(segs)):
        members.setdefault(uf.find(i), []).append(i)
    parts = []
    for mem in members.values():
        pts, rad, pix, order = _chain(segs, mem, glued)
        cls = segs[mem[0]].cls
        conf_m = float(np.mean([segs[i].conf for i in mem]))
        part = {"pts": pts, "rad": rad, "pix": pix, "cls": cls, "segs": mem, "conf": conf_m}
        parts.append(_finish(part, cfg))

    parts = _bridge(parts, cfg)
    if cfg.geometric_classes:
        _assign_geometric_classes(parts)

    seg_part = {}
    for k, p in enumerate(parts):
        for i in p["segs"]:
            seg_part[i] = k
    part_evidence = {(seg_part[i], seg_part[j]) for i, j in evidence if seg_part[i] != seg_part[j]}
    parents = _arborescence(parts, part_evidence, cfg)

    for k, p in enumerate(parts):
        par, base_end = parents.get(k, (None, None))
        pts, rad = p["pts"], p["rad"]
        if par is None:
            # Roots: base at the end lower in the image (larger camera y).
            if pts[-1, 1] > pts[0, 1]:
                pts, rad = pts[::-1], rad[::-1]
            attach = None
        else:
            if base_end == 1:
                pts, rad = pts[::-1], rad[::-1]
            _, attach, _ = closest_point_on_polyline(parts[par]["pts"], pts[0])
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0
        has_child = any(v[0] == k for v in parents.values() if v[0] is not None)
        min_len = max(cfg.min_part_len_m, cfg.stub_radii * p["r_med"])
        if p["cls"] in (TRUNK, BRANCH):
            min_len = max(min_len, cfg.min_thick_len_m)
        if length < min_len and not has_child:
            continue
        graph.add(
            Part(
                pid=k,
                cls=int(p["cls"]),
                points=pts,
                radius=rad,
                parent=par,
                attach=attach,
                score=p["conf"],
            )
        )
    # Parents that were filtered out make their children roots.
    for p in graph.parts.values():
        if p.parent is not None and p.parent not in graph.parts:
            p.parent, p.attach = None, None
    return graph


def _finish(part: dict, cfg: AssembleConfig) -> dict:
    """Cache end points, inward end directions and median radius of a part."""
    pts, rad = part["pts"], part["rad"]
    part["ends"] = np.stack([pts[0], pts[-1]])
    part["dirs"] = np.stack(
        [
            _end_dir(pts, True, cfg.tangent_len_m, cfg.skip_radii * float(rad[0])),
            _end_dir(pts, False, cfg.tangent_len_m, cfg.skip_radii * float(rad[-1])),
        ]
    )
    part["r_med"] = float(np.median(rad))
    part["arc"] = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return part


def _bridge(parts: list[dict], cfg: AssembleConfig) -> list[dict]:
    """Join same-class parts whose ends nearly touch and line up (occlusion gaps)."""
    from scipy.spatial import cKDTree

    while len(parts) > 1:
        ends = np.concatenate([p["ends"] for p in parts])
        dirs = np.concatenate([p["dirs"] for p in parts])
        best = None
        for x, y in cKDTree(ends).query_pairs(cfg.bridge_gap_m):
            a, ea, b, eb = x // 2, x % 2, y // 2, y % 2
            if a == b or parts[a]["cls"] != parts[b]["cls"]:
                continue
            gap = float(np.linalg.norm(ends[x] - ends[y]))
            # Collinear: the two ends face each other and the gap lines up with both.
            defl = np.degrees(np.arccos(np.clip(np.dot(dirs[x], -dirs[y]), -1, 1)))
            if gap > 1e-6:
                g = _unit(ends[y] - ends[x])
                defl = max(defl, np.degrees(np.arccos(np.clip(np.dot(-dirs[x], g), -1, 1))))
            if defl < cfg.bridge_deg and (best is None or gap < best[0]):
                best = (gap, a, ea, b, eb)
        if best is None:
            return parts
        _, a, ea, b, eb = best
        pa, pb = parts[a], parts[b]
        # Orient so pa ends at ea=1 and pb starts at eb=0.
        A = {k: (v[::-1] if ea == 0 else v) for k, v in pa.items() if k in ("pts", "rad", "pix")}
        B = {k: (v[::-1] if eb == 1 else v) for k, v in pb.items() if k in ("pts", "rad", "pix")}
        merged = {
            "pts": np.concatenate([A["pts"], B["pts"]]),
            "rad": np.concatenate([A["rad"], B["rad"]]),
            "pix": np.concatenate([A["pix"], B["pix"]]),
            "cls": pa["cls"],
            "segs": pa["segs"] + pb["segs"],
            "conf": 0.5 * (pa["conf"] + pb["conf"]),
        }
        parts = [p for k, p in enumerate(parts) if k not in (a, b)] + [_finish(merged, cfg)]
    return parts


def _assign_geometric_classes(parts: list[dict]) -> None:
    for p in parts:
        p["cls"] = _geometric_class(p["pts"], p["rad"])
    thick = [k for k, p in enumerate(parts) if p["cls"] == BRANCH]
    if not thick:
        return

    # Trunk: the thick part that is longest and most vertical in the camera frame.
    def score(k: int) -> float:
        pts = parts[k]["pts"]
        span = pts[-1] - pts[0]
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        return length * (0.5 + abs(_unit(span)[1]))

    parts[max(thick, key=score)]["cls"] = TRUNK


def _arborescence(parts: list[dict], evidence: set[tuple[int, int]], cfg: AssembleConfig):
    """Return {child: (parent or None, base end 0/1)} from a min-cost arborescence."""
    from scipy.spatial import cKDTree

    G = nx.DiGraph()
    G.add_node("root")
    best_end: dict[tuple[int, int], int] = {}
    owner = np.concatenate([np.full(len(p["pts"]), k) for k, p in enumerate(parts)])
    kd = cKDTree(np.concatenate([p["pts"] for p in parts]))
    max_rad = max(p["r_med"] for p in parts)
    for c, pc in enumerate(parts):
        G.add_node(c)
        G.add_edge("root", c, weight=0.0 if pc["cls"] == TRUNK else cfg.orphan_cost)
        near: set[int] = set()
        for e in (0, 1):
            end = pc["pts"][0 if e == 0 else -1]
            r = cfg.attach_tol_m + 2.0 * max_rad + cfg.attach_depth_frac * float(end[2]) + 0.05
            near.update(int(owner[i]) for i in kd.query_ball_point(end, r))
        for q in near:
            pq = parts[q]
            if q == c:
                continue
            prior = CLASS_COST.get((pc["cls"], pq["cls"]))
            if prior is None:
                continue
            best = None
            for e in (0, 1):
                end = pc["ends"][e]
                d, attach, s = closest_point_on_polyline(pq["pts"], end)
                tol = cfg.attach_tol_m + 2.0 * pq["r_med"] + cfg.attach_depth_frac * float(end[2])
                if d > tol:
                    continue
                into_child = pc["dirs"][e]
                j = int(np.clip(np.searchsorted(pq["arc"], s), 1, len(pq["pts"]) - 1))
                tangent = _unit(pq["pts"][j] - pq["pts"][j - 1])
                cost = (
                    d / tol + prior + cfg.parallel_penalty * abs(float(np.dot(into_child, tangent)))
                )
                if (c, q) in evidence:
                    cost -= cfg.junction_bonus
                if best is None or cost < best[0]:
                    best = (cost, e)
            if best is not None:
                G.add_edge(q, c, weight=best[0])
                best_end[(q, c)] = best[1]
    arb = nx.minimum_spanning_arborescence(G, attr="weight", default=1e9)
    out: dict[int, tuple[int | None, int | None]] = {}
    for q, c in arb.edges():
        out[c] = (None, None) if q == "root" else (q, best_end[(q, c)])
    return out
