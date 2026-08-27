"""Huber IRLS and median-of-trees recover a known affine; outliers do not yank LS."""

from __future__ import annotations

import numpy as np

from spur_depth.calib.fit_scale_shift import fit_pairs
from spur_depth.calib.robust import huber_irls, huber_weights, median_of_groups
from spur_depth.paths import scratch_root


def _clean(h=24, w=32, alpha=1.1, beta=0.2):
    pred = np.linspace(1.0, 3.0, w, dtype=np.float64)
    pred = np.broadcast_to(pred, (h, w)).copy()
    gt = alpha * pred + beta
    mask = np.ones((h, w), dtype=bool)
    return pred.astype(np.float32), gt.astype(np.float32), mask


def test_huber_recovers_clean_affine():
    pred, gt, mask = _clean()
    a, b, _ = huber_irls([(pred, gt, mask)], erode_r=0, min_gt_std=0.01, iters=4)
    assert abs(a - 1.1) < 1e-5
    assert abs(b - 0.2) < 1e-5


def test_huber_resists_sky_pixels_better_than_ls():
    pred, gt, mask = _clean()
    pred = pred.copy()
    gt = gt.copy()
    gt[:3, :] = 8.0  # leaked far band (~12% of pixels)
    ls_a, ls_b, _ = fit_pairs([(pred, gt, mask)], erode_r=0, min_gt_std=0.01)
    hu_a, hu_b, _ = huber_irls([(pred, gt, mask)], erode_r=0, min_gt_std=0.01, delta=0.05, iters=8)
    ls_err = abs(ls_a - 1.1) + abs(ls_b - 0.2)
    hu_err = abs(hu_a - 1.1) + abs(hu_b - 0.2)
    assert hu_err < ls_err
    assert hu_err < 0.05


def test_median_of_trees_ignores_one_bad_cohort():
    good = [_clean(alpha=1.05, beta=-0.03) for _ in range(3)]
    bad_p, bad_g, bad_m = _clean(alpha=1.05, beta=-0.03)
    bad_g = bad_g + 0.8
    grouped = {"a": [good[0]], "b": [good[1]], "c": [good[2]], "bad": [(bad_p, bad_g, bad_m)]}
    a, b, per = median_of_groups(grouped, erode_r=0, min_gt_std=0.01)
    assert "bad" in per
    assert abs(a - 1.05) < 0.03
    assert abs(b + 0.03) < 0.05


def test_huber_weights_clip_large_residuals():
    r = np.array([0.01, 0.05, 0.20])
    w = huber_weights(r, delta=0.05)
    assert abs(w[0] - 1.0) < 1e-9
    assert abs(w[1] - 1.0) < 1e-9
    assert abs(w[2] - 0.25) < 1e-9


def test_scratch_root_is_writable():
    root = scratch_root()
    p = root / "probe.txt"
    p.write_text("ok")
    assert p.read_text() == "ok"
    p.unlink()
