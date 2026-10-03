"""The annotated K is a placeholder; the render camera is a 28 mm lens."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from spur_depth.camera import (
    K_REF_ANNOTATED,
    blender_intrinsics,
    intrinsics_from_ann,
    is_annotated_k_ref,
    scale_intrinsics,
)
from spur_depth.data.trunk_stereo_triplet import _euler_xyz_to_T
from spur_depth.pipeline.reconstruct import load_pose

FIXTURE = Path(__file__).parent / "fixtures" / "k_fix" / "lpy_envy_00042_box_cam1_shot03_l.npz"


def _ann(K):
    return {
        "camera": {
            "location": [0.0, 0.0, 2.0],
            "rotation_euler": [1.5, 0.0, 0.0],
            "intrinsics": {"width": 1920, "height": 1080, "K": np.asarray(K).tolist()},
        }
    }


def test_blender_intrinsics_is_28mm_on_36mm_sensor():
    K = blender_intrinsics()
    assert K[0, 0] == pytest.approx(28.0 / 36.0 * 1920.0)
    assert K[1, 1] == pytest.approx(K[0, 0])
    assert (K[0, 2], K[1, 2]) == (959.5, 539.5)


def test_annotation_k_ref_is_replaced():
    assert is_annotated_k_ref(K_REF_ANNOTATED)
    K = intrinsics_from_ann(_ann(K_REF_ANNOTATED))
    assert np.allclose(K, blender_intrinsics())
    kept = intrinsics_from_ann(_ann(K_REF_ANNOTATED), trust_annotation=True)
    assert np.allclose(kept, K_REF_ANNOTATED)


def test_other_k_is_kept_with_warning():
    other = [[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        K = intrinsics_from_ann(_ann(other))
    assert np.allclose(K, other)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_load_pose_uses_render_camera():
    K, T = load_pose(_ann(K_REF_ANNOTATED))
    assert K[0, 0] == pytest.approx(1493.333, abs=1e-2)
    K_old, _ = load_pose(_ann(K_REF_ANNOTATED), trust_annotation_k=True)
    assert K_old[0, 0] == pytest.approx(2666.667, abs=1e-2)
    assert T.shape == (4, 4)


def test_scale_intrinsics_keeps_pixel_centres():
    K = scale_intrinsics(blender_intrinsics(), 0.5)
    assert K[0, 0] == pytest.approx(1493.333 / 2, abs=1e-2)
    assert (K[0, 2], K[1, 2]) == (479.5, 269.5)


def _surface_error_m(K: np.ndarray) -> np.ndarray:
    d = np.load(FIXTURE)
    u, v, z = d["u"].astype(np.float64), d["v"].astype(np.float64), d["depth"].astype(np.float64)
    T = _euler_xyz_to_T(d["location"], d["rotation_euler"]).astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pts = (T[:3, :3].T @ (np.stack([x, y, z], 1) - T[:3, 3]).T).T
    c, o = d["centroid"].astype(np.float64), d["orientation"].astype(np.float64)
    half = 0.5 * d["length"].astype(np.float64)
    rel = pts[:, None, :] - c[None, :, :]
    t = np.clip(np.einsum("nmk,mk->nm", rel, o), -half, half)
    axis_dist = np.linalg.norm(rel - t[..., None] * o[None], axis=-1)
    return np.abs(axis_dist - d["radius"][None, :]).min(axis=1)


def test_fixture_depth_lands_on_cylinders_only_with_render_k():
    good = _surface_error_m(blender_intrinsics())
    bad = _surface_error_m(np.load(FIXTURE)["K_annotated"])
    assert np.median(good) < 1e-3
    assert np.percentile(good, 95) < 5e-3
    assert np.median(bad) > 0.05
