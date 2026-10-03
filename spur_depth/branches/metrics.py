"""Scorer: predicted TreeGraph vs ground-truth TreeGraph, one camera frame.

Definitions are fixed in docs/BRANCH_PROTOCOL.md before any test frame is
scored. Everything is in metres in the camera frame. Ground truth only
counts what the camera can see: GT part samples carry ``visible`` flags
(projected axis inside the image and on that part's own pixels). A GT part
is *scored* when at least 3 of its 1 cm samples are visible.

* Skeleton P/R/F1 @ tau: predicted axis samples (1 cm) within tau of a GT
  axis of a visible part (precision), and visible GT samples within tau of a
  predicted axis (recall). Distances are to axes densified to 2 mm, so they
  do not depend on where the 1 cm counting samples fall.
* Parts: one-to-one Hungarian matching on a symmetric 3D overlap score;
  a match needs score >= 0.3. Detection P/R, class accuracy of matches,
  axis direction and radius error of matched pairs. A predicted part lying
  mostly on an unscored GT part is don't-care.
* Connectivity: every predicted part maps to its dominant GT part (the one
  at least half its samples lie on, many-to-one). An edge (child -> parent)
  is correct when the two map to a GT child and its GT parent; each GT edge
  is credited once, edges inside one GT part (fragment to fragment) are
  neutral, edges whose child lies on an unscored part are don't-care. Recall
  counts only observable GT edges (both parts scored, child base and parent
  near the junction visible). ``edge_*_strict`` uses the one-to-one matches.
  Floating rate: matched predicted parts with an observable GT parent edge
  but no predicted parent.
* Cut points (the pruning objective): each non-trunk part's point 1.5 cm
  above its base plus the axis direction there. Each tier gets its own
  one-to-one assignment that maximises successes; success follows Jain et
  al. (ICRA 2025): within 5 cm and the cut axis within 30 deg; the strict
  tier is 2 cm and 15 deg. Recall counts GT cuts with an observable edge; a
  predicted cut that succeeds only on a non-observable GT cut is don't-care.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from spur_depth.branches.tree import CLASSES, IGNORE, TRUNK, TreeGraph, resample_polyline

STEP_M = 0.01
DENSE_STEP_M = 0.002
TAUS_M = (0.01, 0.02, 0.05)
MATCH_TAU_M = 0.03
MATCH_MIN_SCORE = 0.3
MIN_VISIBLE_SAMPLES = 3
BASE_VISIBLE_M = 0.03
PARENT_NEAR_M = 0.04
CUT_GATE_M = 0.05
CUT_TIERS = {"jain": (0.05, 30.0), "strict": (0.02, 15.0)}
TREE_CLASSES = (1, 2, 3, 4)


def _samples(graph: TreeGraph, visible_only: bool):
    """1 cm counting samples: (points, owner pid, class)."""
    pts, owner, cls = [], [], []
    for p in graph.parts.values():
        if len(p.points) == 0:
            continue
        if p.visible is not None:
            q = p.points[p.visible] if visible_only else p.points
        else:
            (q,) = resample_polyline(p.points, STEP_M)
        if len(q) == 0:
            continue
        pts.append(q)
        owner.append(np.full(len(q), p.pid))
        cls.append(np.full(len(q), p.cls))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return np.concatenate(pts), np.concatenate(owner), np.concatenate(cls)


def _dense(graph: TreeGraph):
    """Axes densified to 2 mm for distance queries: (KD-tree or None, owner pid)."""
    pts, owner = [], []
    for p in graph.parts.values():
        if len(p.points) == 0:
            continue
        (q,) = resample_polyline(p.points, DENSE_STEP_M)
        pts.append(q)
        owner.append(np.full(len(q), p.pid))
    if not pts:
        return None, np.zeros(0, dtype=int)
    return cKDTree(np.concatenate(pts)), np.concatenate(owner)


def _query(kd, owner, pts):
    if kd is None or len(pts) == 0:
        return np.full(len(pts), np.inf), np.full(len(pts), -1)
    d, j = kd.query(pts)
    return d, owner[j]


def _axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two lines (sign-free), degrees."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 90.0
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)) / (na * nb), 0.0, 1.0))))


def _principal(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(3)
    c = points - points.mean(0)
    return np.linalg.svd(c, full_matrices=False)[2][0]


def _base_visible(part) -> bool:
    if part.visible is None:
        return True
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(part.points, axis=0), axis=1))]
    return bool(np.any(part.visible[s <= BASE_VISIBLE_M]))


def _parent_visible_near(parent, point: np.ndarray) -> bool:
    if parent.visible is None:
        return True
    d = np.linalg.norm(parent.points - point, axis=1)
    return bool(np.any(parent.visible & (d <= PARENT_NEAR_M)))


def scored_parts(gt: TreeGraph) -> set[int]:
    return {
        p.pid
        for p in gt.parts.values()
        if p.visible is None or int(p.visible.sum()) >= MIN_VISIBLE_SAMPLES
    }


def observable_edges(gt: TreeGraph, scored: set[int] | None = None) -> set[tuple[int, int]]:
    scored = scored_parts(gt) if scored is None else scored
    out = set()
    for p in gt.parts.values():
        if p.parent is None or p.parent not in gt.parts or p.attach is None:
            continue
        if p.pid not in scored or p.parent not in scored:
            continue
        if _base_visible(p) and _parent_visible_near(gt.parts[p.parent], p.attach):
            out.add((p.pid, p.parent))
    return out


def segmentation_counts(pred_cls: np.ndarray, gt_cls: np.ndarray, per_class: bool = True) -> dict:
    """Intersection / union pixel counts (GT IGNORE pixels excluded).

    ``per_class=False`` keeps only the wood (any tree class) entry, for methods
    whose pixels carry no part classes.
    """
    valid = gt_cls != IGNORE
    out = {}
    if per_class:
        for c in TREE_CLASSES:
            p = (pred_cls == c) & valid
            g = (gt_cls == c) & valid
            out[CLASSES[c]] = {"inter": int((p & g).sum()), "union": int((p | g).sum())}
    p = (pred_cls > 0) & (pred_cls != IGNORE) & valid
    g = (gt_cls > 0) & valid
    out["wood"] = {"inter": int((p & g).sum()), "union": int((p | g).sum())}
    return out


def _cut_counts(gt: TreeGraph, pred: TreeGraph, obs: set[tuple[int, int]], scored: set[int]):
    gt_cuts = []  # (point, axis, class, observable)
    for p in gt.parts.values():
        if p.cls == TRUNK or p.parent is None or p.parent not in gt.parts or p.pid not in scored:
            continue
        pt, ax = p.cut()
        gt_cuts.append((pt, ax, p.cls, (p.pid, p.parent) in obs))
    pred_cuts = [
        p.cut()
        for p in pred.parts.values()
        if p.cls != TRUNK and p.parent is not None and p.length > 0
    ]
    observable = np.array([g[3] for g in gt_cuts], dtype=bool)
    gcls = [g[2] for g in gt_cuts]
    out = {"n_gt": int(observable.sum()), "n_pred_all": len(pred_cuts)}
    for c in TREE_CLASSES[1:]:
        out[f"n_gt_{CLASSES[c]}"] = int(sum(bool(k == c and o) for k, o in zip(gcls, observable)))
    if gt_cuts and pred_cuts:
        G = np.array([g[0] for g in gt_cuts])
        P = np.array([p[0] for p in pred_cuts])
        D = np.linalg.norm(G[:, None, :] - P[None, :, :], axis=-1)
        A = np.array([[_axis_angle_deg(g[1], p[1]) for p in pred_cuts] for g in gt_cuts])
    else:
        D = A = np.zeros((len(gt_cuts), len(pred_cuts)))
    for name, (dmax, amax) in CUT_TIERS.items():
        succ_obs = 0
        dont_care = 0
        per_cls = {c: 0 for c in TREE_CLASSES[1:]}
        if D.size:
            # Every success costs ~0 and every non-success 1: the assignment maximises
            # successes, and among those prefers the closest cuts.
            S = (D <= dmax) & (A <= amax)
            cost = np.where(S, 1e-3 * D / dmax, 1.0)
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if not S[i, j]:
                    continue
                if observable[i]:
                    succ_obs += 1
                    per_cls[gcls[i]] += 1
                else:
                    dont_care += 1
        out[f"success_{name}"] = succ_obs
        out[f"n_pred_{name}"] = len(pred_cuts) - dont_care
        for k, v in per_cls.items():
            out[f"success_{name}_{CLASSES[k]}"] = v
    # Diagnostics only: distance-gated pairs over observable GT cuts.
    dists, angs = [], []
    if D.size and observable.any():
        idx = np.nonzero(observable)[0]
        Do = D[idx]
        r, c = linear_sum_assignment(np.where(Do <= CUT_GATE_M, Do, 1e6))
        for i, j in zip(r, c):
            if Do[i, j] <= CUT_GATE_M:
                dists.append(float(Do[i, j]))
                angs.append(float(A[idx[i], j]))
    out["n_paired"] = len(dists)
    out["sum_dist_m"] = float(np.sum(dists))
    out["sum_angle_deg"] = float(np.sum(angs))
    return out


def evaluate_frame(pred: TreeGraph, gt: TreeGraph) -> dict:
    """Counts for one frame; aggregate with ``aggregate``."""
    res: dict = {}
    scored = scored_parts(gt)
    gp_vis, go_vis, gc_vis = _samples(gt, visible_only=True)
    pp, po, _ = _samples(pred, visible_only=False)
    kd_gt, own_gt = _dense(gt)
    kd_pred, own_pred = _dense(pred)
    d_p2g, j_p2g = _query(kd_gt, own_gt, pp)  # pred sample -> nearest GT axis, its part
    d_g2p, j_g2p = _query(kd_pred, own_pred, gp_vis)  # visible GT sample -> nearest pred axis

    # ---- skeleton precision / recall
    sk = {"n_pred": int(len(pp)), "n_gt": int(len(gp_vis))}
    for t in TAUS_M:
        key = f"{int(round(t * 100))}cm"
        sk[f"tp_pred_{key}"] = int((d_p2g <= t).sum())
        sk[f"tp_gt_{key}"] = int((d_g2p <= t).sum())
    for c in TREE_CLASSES:
        m = gc_vis == c
        sk[f"n_gt_{CLASSES[c]}"] = int(m.sum())
        sk[f"tp_gt_2cm_{CLASSES[c]}"] = int((d_g2p[m] <= 0.02).sum())
    near = d_p2g[np.isfinite(d_p2g) & (d_p2g <= MATCH_TAU_M)]
    sk["sum_err_matched_m"] = float(near.sum())
    sk["n_err_matched"] = int(len(near))
    res["skeleton"] = sk

    # ---- part matching (one-to-one) and dominant mapping (many-to-one)
    gt_all = [p.pid for p in gt.parts.values()]
    gt_ids = [k for k in gt_all if k in scored]
    pred_ids = [p.pid for p in pred.parts.values()]
    ga = {k: i for i, k in enumerate(gt_all)}
    pi = {k: i for i, k in enumerate(pred_ids)}
    n_pred_s = np.zeros(len(pred_ids))
    for own in po:
        n_pred_s[pi[own]] += 1
    n_gt_s = np.zeros(len(gt_all))
    for own in go_vis:
        n_gt_s[ga[own]] += 1
    own_ov = np.zeros((len(pred_ids), len(gt_all)))  # pred samples on each GT part
    cov_ov = np.zeros((len(pred_ids), len(gt_all)))  # GT visible samples on each pred part
    for k in np.nonzero(d_p2g <= MATCH_TAU_M)[0]:
        own_ov[pi[po[k]], ga[j_p2g[k]]] += 1
    for k in np.nonzero(d_g2p <= MATCH_TAU_M)[0]:
        cov_ov[pi[j_g2p[k]], ga[go_vis[k]]] += 1
    score = (own_ov + cov_ov) / np.maximum(n_pred_s[:, None] + n_gt_s[None, :], 1)
    sel = np.array([ga[k] for k in gt_ids], dtype=int)
    match: dict[int, int] = {}
    if len(pred_ids) and len(sel):
        sc = score[:, sel]
        rows, cols = linear_sum_assignment(-sc)
        for r, c in zip(rows, cols):
            if sc[r, c] >= MATCH_MIN_SCORE:
                match[pred_ids[r]] = gt_ids[c]
    dominant: dict[int, int] = {}
    if len(gt_all):
        for r, pid in enumerate(pred_ids):
            j = int(np.argmax(own_ov[r]))
            if own_ov[r, j] > 0 and own_ov[r, j] >= 0.5 * n_pred_s[r]:
                dominant[pid] = gt_all[j]
    # Unmatched predictions that lie on GT wood too occluded to score are don't-care.
    dont_care = {p for p in pred_ids if p not in match and dominant.get(p, -1) in ga}
    dont_care = {p for p in dont_care if dominant[p] not in scored}

    parts = {"n_pred": len(pred_ids) - len(dont_care), "n_gt": len(gt_ids), "tp": len(match)}
    parts["tp_class"] = sum(pred.parts[p].cls == gt.parts[g].cls for p, g in match.items())
    for c in TREE_CLASSES:
        parts[f"n_gt_{CLASSES[c]}"] = sum(gt.parts[g].cls == c for g in gt_ids)
        parts[f"tp_{CLASSES[c]}"] = sum(gt.parts[g].cls == c for g in match.values())
    ang, rad = [], []
    for p, g in match.items():
        P, G = pred.parts[p], gt.parts[g]
        (pq,) = resample_polyline(P.points, STEP_M)
        gq = G.points[G.visible] if G.visible is not None else G.points
        if len(gq) < 2 or len(pq) < 2:
            continue
        near_p = pq[cKDTree(gq).query(pq)[0] <= MATCH_TAU_M]
        near_g = gq[cKDTree(pq).query(gq)[0] <= MATCH_TAU_M]
        if len(near_p) >= 2 and len(near_g) >= 2:
            ang.append(_axis_angle_deg(_principal(near_p), _principal(near_g)))
        rad.append(abs(float(np.median(P.radius)) - float(np.median(G.radius))))
    parts["sum_angle_deg"] = float(np.sum(ang))
    parts["n_angle"] = len(ang)
    parts["angle_le_15"] = int(np.sum(np.array(ang) <= 15.0))
    parts["sum_radius_err_m"] = float(np.sum(rad))
    parts["n_radius"] = len(rad)
    res["parts"] = parts

    # ---- connectivity
    obs = observable_edges(gt, scored)
    gt_edges = {(p.pid, p.parent) for p in gt.parts.values() if p.parent is not None}
    pred_edges = pred.edges()
    credited: set[tuple[int, int]] = set()
    neutral = ignored = 0
    for c, q in pred_edges:
        gc, gq = dominant.get(c), dominant.get(q)
        if gc is not None and gc not in scored:
            ignored += 1
        elif gc is not None and gc == gq:
            neutral += 1
        elif gc is not None and gq is not None and (gc, gq) in gt_edges:
            credited.add((gc, gq))  # a second edge for the same GT edge stays a false positive
    strict_any = strict_obs = 0
    junction_err = []
    for c, q in pred_edges:
        gc, gq = match.get(c), match.get(q)
        if gc is None or gq is None or (gc, gq) not in gt_edges:
            continue
        strict_any += 1
        if (gc, gq) in obs:
            strict_obs += 1
            pa, gat = pred.parts[c].attach, gt.parts[gc].attach
            if pa is not None and gat is not None:
                junction_err.append(float(np.linalg.norm(pa - gat)))
    floating = float_den = 0
    for p, g in match.items():
        G = gt.parts[g]
        if G.parent is not None and (g, G.parent) in obs:
            float_den += 1
            floating += pred.parts[p].parent is None
    res["edges"] = {
        "n_pred": len(pred_edges) - neutral - ignored,
        "n_pred_all": len(pred_edges),
        "n_gt_obs": len(obs),
        "geo_tp_pred": len(credited),
        "geo_tp_gt": len(credited & obs),
        "neutral": neutral,
        "ignored": ignored,
        "tp_pred": strict_any,
        "tp_gt": strict_obs,
        "sum_junction_err_m": float(np.sum(junction_err)),
        "n_junction": len(junction_err),
        "junction_le_2cm": int(np.sum(np.array(junction_err) <= 0.02)),
        "floating": int(floating),
        "floating_den": int(float_den),
    }

    res["cuts"] = _cut_counts(gt, pred, obs, scored)
    res["n_gt_parts_visible"] = len(gt_ids)
    return res


def _sum(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, dict):
                out[k] = _sum([out[k], v]) if k in out else _sum([v])
            elif isinstance(v, (int, float, np.integer, np.floating)):
                out[k] = (
                    out.get(k, 0) + v.item() if isinstance(v, np.generic) else out.get(k, 0) + v
                )
    return out


def _ratio(a: float, b: float) -> float | None:
    return float(a) / float(b) if b else None


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None and r is None:
        return None
    if not p or not r:
        return 0.0
    return 2 * p * r / (p + r)


def summarize(totals: dict) -> dict:
    """Headline rates from summed counts (micro-average over frames)."""
    s, pa, e, c = totals["skeleton"], totals["parts"], totals["edges"], totals["cuts"]
    out: dict = {}
    for t in TAUS_M:
        key = f"{int(round(t * 100))}cm"
        p, r = _ratio(s[f"tp_pred_{key}"], s["n_pred"]), _ratio(s[f"tp_gt_{key}"], s["n_gt"])
        out[f"skeleton_precision_{key}"], out[f"skeleton_recall_{key}"] = p, r
        out[f"skeleton_f1_{key}"] = _f1(p, r)
    for cl in TREE_CLASSES:
        n = CLASSES[cl]
        out[f"skeleton_recall_2cm_{n}"] = _ratio(s[f"tp_gt_2cm_{n}"], s[f"n_gt_{n}"])
    out["axis_error_mm_matched"] = (
        1000.0 * s["sum_err_matched_m"] / s["n_err_matched"] if s["n_err_matched"] else None
    )
    p, r = _ratio(pa["tp"], pa["n_pred"]), _ratio(pa["tp"], pa["n_gt"])
    out["part_precision"], out["part_recall"], out["part_f1"] = p, r, _f1(p, r)
    out["part_class_accuracy"] = _ratio(pa["tp_class"], pa["tp"])
    for cl in TREE_CLASSES:
        n = CLASSES[cl]
        out[f"part_recall_{n}"] = _ratio(pa[f"tp_{n}"], pa[f"n_gt_{n}"])
    out["part_axis_angle_deg_mean"] = _ratio(pa["sum_angle_deg"], pa["n_angle"])
    out["part_axis_angle_le_15_frac"] = _ratio(pa["angle_le_15"], pa["n_angle"])
    rr = _ratio(pa["sum_radius_err_m"], pa["n_radius"])
    out["part_radius_err_mm_mean"] = None if rr is None else 1000.0 * rr
    p, r = _ratio(e["geo_tp_pred"], e["n_pred"]), _ratio(e["geo_tp_gt"], e["n_gt_obs"])
    out["edge_precision"], out["edge_recall"], out["edge_f1"] = p, r, _f1(p, r)
    p, r = _ratio(e["tp_pred"], e["n_pred_all"]), _ratio(e["tp_gt"], e["n_gt_obs"])
    out["edge_precision_strict"], out["edge_recall_strict"] = p, r
    out["edge_f1_strict"] = _f1(p, r)
    jr = _ratio(e["sum_junction_err_m"], e["n_junction"])
    out["junction_err_mm_mean"] = None if jr is None else 1000.0 * jr
    out["junction_le_2cm_frac"] = _ratio(e["junction_le_2cm"], e["n_junction"])
    out["floating_rate"] = _ratio(e["floating"], e["floating_den"])
    for name in CUT_TIERS:
        r = _ratio(c[f"success_{name}"], c["n_gt"])
        p = _ratio(c[f"success_{name}"], c[f"n_pred_{name}"])
        out[f"cut_recall_{name}"], out[f"cut_precision_{name}"] = r, p
        out[f"cut_f1_{name}"] = _f1(p, r)
        for cl in TREE_CLASSES[1:]:
            n = CLASSES[cl]
            out[f"cut_recall_{name}_{n}"] = _ratio(c[f"success_{name}_{n}"], c[f"n_gt_{n}"])
    dm = _ratio(c["sum_dist_m"], c["n_paired"])
    out["cut_dist_mm_mean_paired"] = None if dm is None else 1000.0 * dm
    out["cut_angle_deg_mean_paired"] = _ratio(c["sum_angle_deg"], c["n_paired"])
    if "seg" in totals:
        for k, v in totals["seg"].items():
            out[f"iou_{k}"] = _ratio(v["inter"], v["union"])
        ious = [out.get(f"iou_{CLASSES[k]}") for k in TREE_CLASSES]
        out["miou_tree"] = float(np.mean(ious)) if all(v is not None for v in ious) else None
    return out


def aggregate(frame_results: list[dict]) -> dict:
    totals = _sum(frame_results)
    return {"totals": totals, "summary": summarize(totals), "n_frames": len(frame_results)}
