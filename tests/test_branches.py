"""Branch module on synthetic scenes: no dataset, no weights, CPU only."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spur_depth.branches.assemble import AssembleConfig, assemble
from spur_depth.branches.depth_fusion import fuse_sensor_mono, huber_affine
from spur_depth.branches.metrics import aggregate, evaluate_frame
from spur_depth.branches.skeleton import skeleton_graph, thin
from spur_depth.branches.tree import BRANCH, SPUR, TRUNK, Part, TreeGraph
from spur_depth.camera import blender_intrinsics, scale_intrinsics

K = scale_intrinsics(blender_intrinsics(), 1.0 / 3.0)  # 640 x 360, f ~ 498 px
H, W = 360, 640
Z = 1.5


def _line(p0, p1, n=200):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) * np.asarray(p0, float) + t * np.asarray(p1, float)


def _scene():
    """Trunk (vertical) with one branch and one spur on the branch, all at depth Z."""
    trunk = _line([0.0, 0.35, Z], [0.0, -0.30, Z])
    branch = _line([0.0, 0.05, Z], [0.40, -0.05, Z])
    spur = _line([0.20, 0.00, Z], [0.24, -0.12, Z])
    return [(TRUNK, trunk, 0.012), (BRANCH, branch, 0.010), (SPUR, spur, 0.004)]


def _render(scene):
    """Class map and depth (surface = axis depth - radius) for the synthetic scene."""
    cls = np.zeros((H, W), np.uint8)
    depth = np.full((H, W), 6.0, np.float32)
    for c, pts, r in scene:
        u = K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2]
        v = K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]
        th = max(1, int(round(2 * r * K[0, 0] / Z)))
        for i in range(len(u) - 1):
            p, q = (
                (int(round(u[i])), int(round(v[i]))),
                (int(round(u[i + 1])), int(round(v[i + 1]))),
            )
            cv2.line(cls, p, q, int(c), th)
            cv2.line(depth, p, q, float(Z - r), th)
    return cls, depth


def _gt_graph(scene):
    g = TreeGraph()
    parents = {0: None, 1: 0, 2: 1}
    for k, (c, pts, r) in enumerate(scene):
        attach = None
        if parents[k] is not None:
            attach = pts[0].copy()
        from spur_depth.branches.tree import resample_polyline

        (q,) = resample_polyline(pts, 0.01)
        g.add(Part(k, c, q, np.full(len(q), r), parents[k], attach, visible=np.ones(len(q), bool)))
    return g


def test_thin_gives_one_pixel_y():
    m = np.zeros((80, 80), np.uint8)
    cv2.line(m, (40, 75), (40, 40), 1, 7)
    cv2.line(m, (40, 40), (12, 6), 1, 4)
    cv2.line(m, (40, 40), (70, 6), 1, 4)
    sk = thin(m > 0)
    assert sk.sum() < (m > 0).sum() / 3
    g = skeleton_graph(sk)
    kinds = sorted(g.kind.values())
    assert kinds.count("tip") == 3 and kinds.count("junction") == 1
    assert len(g.segments) == 3


def test_assemble_recovers_parts_depth_and_parents():
    scene = _scene()
    cls, depth = _render(scene)
    pred = assemble(cls, depth, K, cfg=AssembleConfig(min_component_px=10))
    by_cls = {}
    for p in pred.parts.values():
        by_cls.setdefault(p.cls, []).append(p)
    assert len(by_cls[TRUNK]) == 1 and len(by_cls[BRANCH]) == 1 and len(by_cls[SPUR]) == 1
    trunk, branch, spur = by_cls[TRUNK][0], by_cls[BRANCH][0], by_cls[SPUR][0]
    assert branch.parent == trunk.pid and spur.parent == branch.pid and trunk.parent is None
    # Axis depth = surface + radius correction, within a few mm of Z.
    assert abs(float(np.median(branch.points[:, 2])) - Z) < 0.01
    # Base of the branch is at the trunk end.
    assert abs(branch.points[0, 0]) < 0.03


def test_metrics_perfect_and_shifted():
    gt = _gt_graph(_scene())
    s = aggregate([evaluate_frame(gt, gt)])["summary"]
    assert s["skeleton_f1_2cm"] == pytest.approx(1.0)
    assert s["part_precision"] == pytest.approx(1.0) and s["part_recall"] == pytest.approx(1.0)
    assert s["edge_f1"] == pytest.approx(1.0) and s["edge_f1_strict"] == pytest.approx(1.0)
    assert s["cut_recall_jain"] == pytest.approx(1.0) and s["cut_recall_strict"] == pytest.approx(
        1.0
    )

    shifted = TreeGraph()
    for p in gt.parts.values():
        shifted.add(Part(p.pid, p.cls, p.points + [0.0, 0.0, 0.03], p.radius, p.parent, p.attach))
    s = aggregate([evaluate_frame(shifted, gt)])["summary"]
    assert s["skeleton_precision_2cm"] == pytest.approx(0.0)
    assert s["skeleton_f1_5cm"] == pytest.approx(1.0)
    assert s["cut_recall_jain"] == pytest.approx(1.0)
    assert s["cut_recall_strict"] == pytest.approx(0.0)


def test_metrics_charge_missing_and_wrong_edges():
    gt = _gt_graph(_scene())
    orphan = TreeGraph()
    for p in gt.parts.values():
        par = None if p.cls == SPUR else p.parent
        orphan.add(Part(p.pid, p.cls, p.points, p.radius, par, p.attach))
    s = aggregate([evaluate_frame(orphan, gt)])["summary"]
    assert s["edge_recall"] == pytest.approx(0.5)
    assert s["floating_rate"] == pytest.approx(0.5)
    # A spur with its base and tip swapped puts the cut point at the wrong end.
    flipped = TreeGraph()
    for p in gt.parts.values():
        pts = p.points[::-1] if p.cls == SPUR else p.points
        flipped.add(Part(p.pid, p.cls, pts, p.radius, p.parent, p.attach))
    s = aggregate([evaluate_frame(flipped, gt)])["summary"]
    assert s["cut_recall_jain"] == pytest.approx(0.5)


def test_tree_graph_json_round_trip():
    gt = _gt_graph(_scene())
    back = TreeGraph.from_json(gt.to_json())
    assert sorted(back.edges()) == sorted(gt.edges())
    for pid, p in gt.parts.items():
        assert np.allclose(back.parts[pid].points, p.points, atol=1e-5)
    cut = gt.to_json()["parts"][2]["cut"]
    assert len(cut["point_m"]) == 3 and abs(np.linalg.norm(cut["axis"]) - 1.0) < 1e-3


def test_fusion_keeps_good_sensor_and_repairs_thin_wood():
    rng = np.random.default_rng(0)
    true = 1.5 + rng.random((60, 80)).astype(np.float32)
    mono = (true - 0.05) / 1.1  # scale and shift error
    sensor = true + rng.normal(0, 0.003, true.shape).astype(np.float32)
    bad = rng.random(true.shape) < 0.2
    sensor[bad] = 8.0  # bled to background
    sensor[rng.random(true.shape) < 0.1] = 0.0  # dropouts
    a, b = huber_affine(mono[~bad].ravel(), sensor[~bad].ravel())
    assert a == pytest.approx(1.1, abs=0.02) and b == pytest.approx(0.05, abs=0.03)
    fused, info = fuse_sensor_mono(sensor, mono)
    assert np.abs(fused - true).max() < 0.03
    assert 0.6 < info["sensor_kept_frac"] < 0.8


def test_service_contract_with_random_weights():
    from spur_depth.branches.model import BranchNet
    from spur_depth.branches.service import BranchService

    svc = BranchService()
    with pytest.raises(NotImplementedError):
        svc.run(np.zeros((64, 96, 3), np.uint8), np.ones((64, 96), np.float32), K)
    svc.model = BranchNet(pretrained=False).eval()
    rgb = np.zeros((64, 96, 3), np.uint8)
    with pytest.raises(ValueError):
        svc.run(rgb, np.ones((32, 96), np.float32), K)
    body = svc.run(rgb, np.full((64, 96), 1.5, np.float32), K, mono_depth=np.ones((64, 96)))
    assert body["frame"] == "camera" and body["units"] == "m"
    assert set(body["meta"]["latency_ms"]) == {"network", "graph"}


def test_cut_targets_world_frame_round_trip():
    from spur_depth.branches.export import cut_targets

    gt = _gt_graph(_scene())
    # World-to-camera: rotate 90 deg about x and shift; targets must come back in world.
    c, s = 0.0, 1.0
    T = np.eye(4)
    T[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    T[:3, 3] = [0.1, -0.2, 0.3]
    cam = gt.transformed(T[:3, :3], T[:3, 3], "camera")
    recs = cut_targets(cam, T)
    assert {r["part_class"] for r in recs} == {"branch", "spur"}
    for r in recs:
        p, _ = gt.parts[r["record_id"]].cut()
        assert np.allclose(r["position_w"], p, atol=1e-4)
        assert abs(np.linalg.norm(r["axis_w"]) - 1.0) < 1e-3


def test_cut_success_is_not_stolen_by_a_wrong_axis_stub():
    gt = _gt_graph(_scene())
    pred = TreeGraph()
    for p in gt.parts.values():
        pred.add(Part(p.pid, p.cls, p.points, p.radius, p.parent, p.attach))
    spur = gt.parts[2]
    cut, axis = spur.cut()
    side = np.cross(axis, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    stub = np.stack([cut - 0.015 * side, cut + 0.015 * side])
    pred.add(Part(9, SPUR, stub, np.full(2, 0.003), parent=1, attach=stub[0]))
    s = aggregate([evaluate_frame(pred, gt)])["summary"]
    assert s["cut_recall_jain"] == pytest.approx(1.0)


def test_empty_prediction_scores_zero_not_none():
    gt = _gt_graph(_scene())
    s = aggregate([evaluate_frame(TreeGraph(), gt)])["summary"]
    for k in ("skeleton_f1_2cm", "part_f1", "edge_f1", "cut_f1_jain"):
        assert s[k] == 0.0


def test_fragment_chain_is_not_worse_than_star():
    gt = _gt_graph(_scene())
    branch = gt.parts[1].points
    thirds = np.array_split(np.arange(len(branch)), 3)
    frags = [branch[i[0] : i[-1] + 1] for i in thirds]

    def graph(parents):
        g = TreeGraph()
        g.add(Part(0, TRUNK, gt.parts[0].points, gt.parts[0].radius))
        for k, (f, par) in enumerate(zip(frags, parents)):
            g.add(Part(10 + k, BRANCH, f, np.full(len(f), 0.01), par, f[0]))
        return g

    chain = aggregate([evaluate_frame(graph([0, 10, 11]), gt)])["summary"]
    star = aggregate([evaluate_frame(graph([0, 0, 0]), gt)])["summary"]
    assert chain["edge_precision"] >= star["edge_precision"]


def test_small_hole_does_not_split_a_trunk():
    cls = np.zeros((H, W), np.uint8)
    depth = np.full((H, W), 6.0, np.float32)
    cv2.line(cls, (320, 330), (320, 30), int(TRUNK), 21)
    depth[cls > 0] = Z
    cls[180, 320] = 0  # one background pixel inside the trunk
    pred = assemble(cls, depth, K, cfg=AssembleConfig(min_component_px=10))
    trunks = [p for p in pred.parts.values() if p.cls == TRUNK]
    assert len(trunks) == 1


def test_self_loop_is_not_dissolved_as_degree_two():
    from spur_depth.branches.skeleton import Segment, SkeletonGraph, _dissolve_degree2

    g = SkeletonGraph()
    g.nodes = {0: np.array([[0, 0]]), 1: np.array([[0, 5]])}
    g.kind = {0: "junction", 1: "tip"}
    loop = Segment(0, 0, np.array([[0, 0], [1, 1], [2, 0], [1, -1], [0, 0]]))
    tail = Segment(0, 1, np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]))
    g.segments = [loop, tail]
    out = _dissolve_degree2(g)
    assert len(out.segments) == 2 and out.kind[0] == "junction"


def test_millimetre_depth_is_rejected():
    cls, depth = _render(_scene())
    with pytest.raises(ValueError):
        assemble(cls, depth * 1000.0, K)


def test_diagonal_two_pixel_band_keeps_its_skeleton():
    m = np.zeros((160, 160), bool)
    for i in range(10, 140):
        m[i, i] = m[i, i + 1] = True
    assert thin(m).sum() > 100


def _deg(a, b):
    c = abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(c, 1.0))))


def test_fitted_cut_axis_skips_the_junction_bend_and_leaves_gt_untouched():
    from spur_depth.branches.assemble import fit_cut_axes, fitted_axis

    # A spur whose first 2 cm bend toward the parent, as thinning bends a skeleton at a junction.
    s = np.linspace(0.0, 0.12, 121)[:, None]
    d = np.array([0.3, -1.0, 0.2]) / np.linalg.norm([0.3, -1.0, 0.2])
    pts = s * d + np.clip(0.02 - s, 0.0, None) * np.array([1.0, 0.0, 0.0]) + [0.1, 0.0, Z]
    part = Part(0, SPUR, pts, np.full(len(pts), 0.004), parent=1, attach=pts[0])
    point, tangent = part.cut()
    assert _deg(tangent, d) > 10.0 and _deg(fitted_axis(pts, 0.03, 0.08), d) < 0.5
    g = TreeGraph()
    g.add(part)
    fit_cut_axes(g, 0.03, 0.08)
    point2, axis2 = part.cut()
    assert np.array_equal(point, point2) and _deg(axis2, d) < 0.5
    # The axis follows a rigid transform; ground truth never carries one.
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moved = g.transformed(R, np.zeros(3), "world").parts[0]
    assert _deg(moved.cut()[1], R @ d) < 0.5
    assert all(p.cut_axis is None for p in _gt_graph(_scene()).parts.values())


def test_assemble_cut_axis_option():
    cls, depth = _render(_scene())
    base = assemble(cls, depth, K, cfg=AssembleConfig(min_component_px=10))
    fit = assemble(cls, depth, K, cfg=AssembleConfig(min_component_px=10, cut_axis="fit"))
    assert all(p.cut_axis is None for p in base.parts.values())
    assert any(p.cut_axis is not None for p in fit.parts.values() if p.cls != TRUNK)
    for pid, p in base.parts.items():
        assert np.array_equal(p.points, fit.parts[pid].points)
    with pytest.raises(ValueError):
        assemble(cls, depth, K, cfg=AssembleConfig(min_component_px=10, cut_axis="nope"))
