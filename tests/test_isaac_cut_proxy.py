"""scripts/isaac_cut_proxy.py: pose rule and the pruning repo's geometry on synthetic branches.

Skips when the read-only pruning clone the proxy imports its geometry from is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

proxy = pytest.importorskip("isaac_cut_proxy")
if not (proxy.ISAAC_SRC / "source" / "isaaclab_pruning").is_dir():
    pytest.skip(f"pruning clone absent: {proxy.ISAAC_SRC}", allow_module_level=True)
pytest.importorskip("torch")

from spur_depth.branches.tree import BRANCH, SPUR, Part  # noqa: E402


@pytest.fixture(scope="module")
def isaac():
    return proxy.Isaac(proxy.ISAAC_SRC)


def _spur(base, axis, length=0.2, radius=0.004, pid=1):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    pts = np.asarray(base, float) + np.linspace(0.0, length, 21)[:, None] * axis
    return Part(pid=pid, cls=SPUR, points=pts, radius=np.full(len(pts), radius), parent=0)


def _wood(*segments):
    """Other wood as (a, b, radius, part id); a far-away dummy keeps the shapes non-empty."""
    segs = [*segments, ((50.0, 50.0, 50.0), (50.1, 50.0, 50.0), 0.01, 99)]
    return {
        "a": np.array([s[0] for s in segs], float),
        "b": np.array([s[1] for s in segs], float),
        "r": np.array([s[2] for s in segs], float),
        "part": np.array([s[3] for s in segs]),
    }


def test_tool_frame_follows_the_pose_rule():
    point, axis = np.array([0.1, -0.2, 1.3]), np.array([0.3, 0.9, 0.2])
    R, degenerate = proxy.tool_frame(point, axis)
    u = axis / np.linalg.norm(axis)
    assert not degenerate
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert abs(R[:, 2] @ u) < 1e-12  # approach perpendicular to the perceived axis
    assert R[:, 2] @ point > 0  # and pointing away from the camera
    assert np.allclose(R[:, 0], np.cross(R[:, 2], u) / np.linalg.norm(np.cross(R[:, 2], u)))
    assert np.allclose(R[:, 1], -u)
    _, degenerate = proxy.tool_frame(point, point)  # axis along the viewing ray
    assert degenerate


def test_perfect_target_succeeds_and_a_depth_error_fails(isaac):
    spur = _spur(base=(0.05, 0.02, 1.2), axis=(1.0, -0.3, 0.1))
    pt, ax = spur.cut()
    good = proxy.make_item(pt, ax, spur)
    deep = proxy.make_item(pt, ax, spur)
    deep["p"] = pt - 0.010 * deep["R"][:, 2]  # perceived 10 mm short along the approach
    res = proxy.evaluate_items(isaac, [good, deep], _wood(), (0.02, 0.10))
    for li in range(2):
        assert res[0]["legacy"]["success"][li] and res[0]["demo"]["success"][li]
        # 10 mm short puts the true branch past the legacy mouth (window 42.5-49.9 mm deep)
        assert not res[1]["legacy"]["jaw_hit"][li]
        # and outside the demo's 8 mm mouth tolerance
        assert not res[1]["demo"]["jaw_hit"][li] and not res[1]["demo"]["gate_only"][li]


def test_axis_error_fails_perpendicularity(isaac):
    spur = _spur(base=(0.0, 0.0, 1.0), axis=(1.0, 0.0, 0.0))
    pt, ax = spur.cut()
    tilt = np.radians(20.0)  # perceived axis rotated 20 deg about the viewing ray
    item = proxy.make_item(pt, np.array([np.cos(tilt), np.sin(tilt), 0.0]), spur)
    (res,) = proxy.evaluate_items(isaac, [item], _wood(), (0.10,))
    assert not res["legacy"]["perpendicular"][0] and not res["demo"]["perpendicular"][0]
    assert 19.0 < res["legacy_perp_deg"][0] < 21.0
    # A tilt toward the camera leaves the closing axis perpendicular to the true axis.
    item = proxy.make_item(pt, np.array([np.cos(tilt), 0.0, np.sin(tilt)]), spur)
    (res,) = proxy.evaluate_items(isaac, [item], _wood(), (0.10,))
    assert res["legacy_perp_deg"][0] < 1.5


def test_other_wood_in_the_cutter_fails_clearance(isaac):
    spur = _spur(base=(0.0, 0.0, 1.0), axis=(1.0, 0.0, 0.0))
    pt, ax = spur.cut()
    item = proxy.make_item(pt, ax, spur)
    R = item["R"]
    # A branch through the demo mouth along the closing axis, 5 mm behind the cut point.
    centre = pt + 0.005 * R[:, 1]
    across = (centre - 0.05 * R[:, 0], centre + 0.05 * R[:, 0], 0.01, 7)
    (clear,) = proxy.evaluate_items(isaac, [item], _wood(), (0.10,))
    (hit,) = proxy.evaluate_items(isaac, [item], _wood(across), (0.10,))
    assert clear["demo"]["success"][0] and clear["legacy"]["success"][0]
    assert hit["demo"]["gate_only"][0] and not hit["demo"]["success"][0]
    assert hit["demo_hits"] == [7]
    # The legacy failure box sits in front of the mouth: the same branch 40 mm toward the
    # tool (inside it) fails the legacy clearance only.
    front = pt - 0.040 * R[:, 2]
    near_tool = (front - 0.05 * R[:, 0], front + 0.05 * R[:, 0], 0.01, 8)
    (res,) = proxy.evaluate_items(isaac, [item], _wood(near_tool), (0.10,))
    assert res["legacy"]["success_isolated"][0] and not res["legacy"]["success"][0]
    assert res["legacy_hits"] == [8] and res["demo"]["success"][0]


def test_truth_segment_is_clipped_at_the_base():
    branch = Part(
        pid=3,
        cls=BRANCH,
        points=np.array([[0.0, 0.0, 1.0], [0.3, 0.0, 1.0]]),
        radius=np.full(2, 0.01),
        parent=0,
    )
    a, b = proxy.truth_segment(branch, 0.10)
    assert np.allclose(a, [0.0, 0.0, 1.0]) and np.allclose(b, [0.065, 0.0, 1.0])
    a, b = proxy.truth_segment(branch, 0.02)
    assert np.allclose(a, [0.005, 0.0, 1.0]) and np.allclose(b, [0.025, 0.0, 1.0])
