"""Binary mask -> one-pixel skeleton -> pixel graph of segments between nodes.

Zhang-Suen thinning (CACM 1984) with a 256-entry lookup table per
sub-iteration, so each pass is a handful of whole-array numpy ops. Graph
edges use m-adjacency (a diagonal neighbour only counts when no shared
4-neighbour links the pair), which keeps staircase corners from posing as
junctions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Neighbour order P2..P9 (N, NE, E, SE, S, SW, W, NW) as (dy, dx).
_NB = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _luts() -> tuple[np.ndarray, np.ndarray]:
    lut1 = np.zeros(256, dtype=bool)
    lut2 = np.zeros(256, dtype=bool)
    for code in range(256):
        p = [(code >> i) & 1 for i in range(8)]  # p[0]=P2 ... p[7]=P9
        b = sum(p)
        a = sum(1 for i in range(8) if p[i] == 0 and p[(i + 1) % 8] == 1)
        if not (2 <= b <= 6 and a == 1):
            continue
        p2, p4, p6, p8 = p[0], p[2], p[4], p[6]
        lut1[code] = p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
        lut2[code] = p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
    return lut1, lut2


_LUT1, _LUT2 = _luts()


def _codes(img: np.ndarray) -> np.ndarray:
    p = np.pad(img, 1)
    h, w = img.shape
    code = np.zeros((h, w), dtype=np.uint8)
    for bit, (dy, dx) in enumerate(_NB):
        code |= p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] << bit
    return code


def thin(mask: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """Zhang-Suen skeleton of a boolean mask (computed on its bounding box)."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img = mask[y0:y1, x0:x1].astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for lut in (_LUT1, _LUT2):
            rm = (img == 1) & lut[_codes(img)]
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            break
    out[y0:y1, x0:x1] = img.astype(bool)
    return out


@dataclass
class Segment:
    a: int  # node id at path[0]
    b: int  # node id at path[-1]
    path: np.ndarray  # (k, 2) int (y, x), ordered a -> b


@dataclass
class SkeletonGraph:
    nodes: dict[int, np.ndarray] = field(default_factory=dict)  # node id -> (m, 2) pixels
    kind: dict[int, str] = field(default_factory=dict)  # "junction" | "tip" | "loop"
    segments: list[Segment] = field(default_factory=list)

    def degree(self, node: int) -> int:
        return sum((s.a == node) + (s.b == node) for s in self.segments)


def _m_neighbours(skel: np.ndarray):
    ys, xs = np.nonzero(skel)
    n = len(ys)
    h, w = skel.shape
    idx = np.full((h + 2, w + 2), -1, dtype=np.int64)
    idx[ys + 1, xs + 1] = np.arange(n)
    nbrs = []
    for dy, dx in _NB:
        j = idx[ys + 1 + dy, xs + 1 + dx]
        if dy != 0 and dx != 0:
            # Drop a diagonal link when a shared 4-neighbour already connects the pair.
            shared = (idx[ys + 1 + dy, xs + 1] >= 0) | (idx[ys + 1, xs + 1 + dx] >= 0)
            j = np.where(shared, -1, j)
        nbrs.append(j)
    nb = np.stack(nbrs, axis=1)  # (n, 8)
    return ys, xs, nb


def skeleton_graph(skel: np.ndarray) -> SkeletonGraph:
    """Segments between junction clusters and tips of a one-pixel skeleton."""
    g = SkeletonGraph()
    ys, xs, nb = _m_neighbours(np.asarray(skel, dtype=bool))
    n = len(ys)
    if n == 0:
        return g
    deg = (nb >= 0).sum(1)
    is_node = deg != 2

    # Junction pixels that touch each other form one node (union-find).
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    junction = np.nonzero(deg >= 3)[0]
    for i in junction:
        for j in nb[i]:
            if j >= 0 and deg[j] >= 3:
                ri, rj = find(i), find(int(j))
                if ri != rj:
                    parent[ri] = rj
    node_of = np.full(n, -1, dtype=np.int64)
    members: dict[int, list[int]] = {}
    for i in np.nonzero(is_node)[0]:
        r = find(int(i)) if deg[i] >= 3 else int(i)
        node_of[i] = r
        members.setdefault(r, []).append(int(i))
    for r, mem in members.items():
        g.nodes[r] = np.stack([ys[mem], xs[mem]], 1)
        g.kind[r] = "junction" if deg[mem[0]] >= 3 else ("tip" if deg[mem[0]] == 1 else "loop")

    visited = np.zeros(n, dtype=bool)  # degree-2 pixels already on a segment
    for start in np.nonzero(is_node)[0]:
        for nxt in nb[start]:
            if nxt < 0:
                continue
            nxt = int(nxt)
            if is_node[nxt]:
                # Direct node-node link; keep it once and never inside one junction cluster.
                if node_of[nxt] != node_of[start] and start < nxt:
                    g.segments.append(
                        Segment(
                            int(node_of[start]),
                            int(node_of[nxt]),
                            np.array([[ys[start], xs[start]], [ys[nxt], xs[nxt]]]),
                        )
                    )
                continue
            if visited[nxt]:
                continue
            path = [int(start), nxt]
            prev, cur = int(start), nxt
            while not is_node[cur]:
                visited[cur] = True
                cand = [int(j) for j in nb[cur] if j >= 0 and j != prev]
                if not cand:
                    break
                prev, cur = cur, cand[0]
                path.append(cur)
            if not is_node[cur]:
                continue
            p = np.array(path)
            g.segments.append(
                Segment(int(node_of[start]), int(node_of[cur]), np.stack([ys[p], xs[p]], 1))
            )

    # Closed loops with no node: cut each at an arbitrary pixel.
    for i in np.nonzero(~is_node & ~visited)[0]:
        if visited[i]:
            continue
        path = [int(i)]
        visited[i] = True
        prev, cur = -1, int(i)
        while True:
            cand = [int(j) for j in nb[cur] if j >= 0 and j != prev and not visited[j]]
            if not cand:
                break
            prev, cur = cur, cand[0]
            visited[cur] = True
            path.append(cur)
        node_id = int(n + i)
        p = np.array(path)
        g.nodes[node_id] = np.array([[ys[i], xs[i]]])
        g.kind[node_id] = "loop"
        g.segments.append(Segment(node_id, node_id, np.stack([ys[p], xs[p]], 1)))
    return g


def prune_spurs(g: SkeletonGraph, min_len, rounds: int = 2) -> SkeletonGraph:
    """Drop whiskers: tip-to-junction segments shorter than ``min_len(segment)`` pixels.

    Thinning a blob leaves short tip branches that are not wood. After each
    round, junctions left with two segments are dissolved by joining them.
    """
    for _ in range(rounds):
        keep = []
        for s in g.segments:
            ka, kb = g.kind.get(s.a), g.kind.get(s.b)
            whisker = (ka == "tip") != (kb == "tip") and "junction" in (ka, kb)
            if whisker and len(s.path) < min_len(s):
                continue
            keep.append(s)
        changed = len(keep) != len(g.segments)
        g.segments = keep
        g = _dissolve_degree2(g)
        if not changed:
            break
    return g


def merge_close_junctions(g: SkeletonGraph, max_len) -> SkeletonGraph:
    """Collapse junction-to-junction segments shorter than ``max_len(segment)`` pixels.

    Thinning a limb that carries several children within its own width leaves
    a ladder of junctions a few pixels apart; they are one branching region.
    """
    while True:
        hit = None
        for k, s in enumerate(g.segments):
            if (
                s.a != s.b
                and g.kind.get(s.a) == "junction"
                and g.kind.get(s.b) == "junction"
                and len(s.path) < max_len(s)
            ):
                hit = k
                break
        if hit is None:
            return _dissolve_degree2(g)
        s = g.segments.pop(hit)
        keep, gone = s.a, s.b
        g.nodes[keep] = np.concatenate([g.nodes[keep], g.nodes.pop(gone), s.path], 0)
        g.kind.pop(gone, None)
        for t in g.segments:
            if t.a == gone:
                t.a = keep
            if t.b == gone:
                t.b = keep


def _dissolve_degree2(g: SkeletonGraph) -> SkeletonGraph:
    while True:
        incident: dict[int, list[int]] = {}
        for k, s in enumerate(g.segments):
            incident.setdefault(s.a, []).append(k)
            if s.b != s.a:
                incident.setdefault(s.b, []).append(k)
        target = None
        for node, segs in incident.items():
            if g.kind.get(node) == "junction" and len(segs) == 2 and segs[0] != segs[1]:
                target = (node, segs)
                break
        if target is None:
            return g
        node, (i, j) = target
        si, sj = g.segments[i], g.segments[j]
        pi = si.path if si.b == node else si.path[::-1]
        pj = sj.path if sj.a == node else sj.path[::-1]
        a = si.a if si.b == node else si.b
        b = sj.b if sj.a == node else sj.a
        merged = Segment(a, b, np.concatenate([pi, pj[1:]], 0))
        g.segments = [s for k, s in enumerate(g.segments) if k not in (i, j)] + [merged]
        g.kind[node] = "dissolved"
