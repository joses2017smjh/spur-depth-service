"""C1: recover a known affine map; empty/masked inputs fail closed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spur_depth.calib import PRO_ALPHA, PRO_BETA, load_pro_config
from spur_depth.calib.fit_scale_shift import (
    Moments,
    fit_files,
    fit_pairs,
    moments_from_image,
    solve_scale_shift,
)


def _ramp(h=32, w=48, alpha=PRO_ALPHA, beta=PRO_BETA, seed=0):
    rng = np.random.default_rng(seed)
    # Mild variation so GT std clears min_gt_std.
    xs = 0.8 + 0.02 * np.arange(w, dtype=np.float64)
    pred = np.broadcast_to(xs, (h, w)).copy()
    pred = pred + rng.normal(0, 1e-12, pred.shape)  # keep well-conditioned
    gt = alpha * pred + beta
    mask = np.ones((h, w), dtype=np.uint8)
    return pred, gt, mask


def test_known_answer_recovers_training_literals():
    pred, gt, mask = _ramp()
    alpha, beta, acc = fit_pairs([(pred, gt, mask)], erode_r=0, min_gt_std=0.0)
    assert abs(alpha - PRO_ALPHA) < 1e-9
    assert abs(beta - PRO_BETA) < 1e-9
    assert acc.n == pred.size


def test_erode_drops_a_border():
    pred, gt, mask = _ramp(h=21, w=21)
    m0 = moments_from_image(pred, gt, mask, erode_r=0, min_gt_std=0.0)
    m2 = moments_from_image(pred, gt, mask, erode_r=2, min_gt_std=0.0)
    assert m0 is not None and m2 is not None
    # 21x21 eroded by 2 → 17x17
    assert m2.n == 17 * 17
    assert m0.n == 21 * 21


def test_flat_gt_is_skipped_by_min_gt_std():
    pred = np.ones((16, 16), dtype=np.float64)
    gt = np.full((16, 16), 2.0)
    mask = np.ones((16, 16), dtype=np.uint8)
    assert moments_from_image(pred, gt, mask, erode_r=0, min_gt_std=0.05) is None


def test_all_masked_raises():
    pred, gt, _ = _ramp()
    mask = np.zeros_like(pred, dtype=np.uint8)
    with pytest.raises(ValueError, match="no valid pixels"):
        fit_pairs([(pred, gt, mask)], erode_r=0, min_gt_std=0.0)


def test_empty_moments_raise():
    with pytest.raises(ValueError, match="no valid pixels"):
        solve_scale_shift(Moments())


def test_config_json_matches_literals():
    cfg = load_pro_config()
    assert cfg["alpha"] == PRO_ALPHA
    assert cfg["beta"] == PRO_BETA
    assert cfg["erode_r"] == 10
    raw = json.loads(
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("spur_depth/calib/pro_best_config.json")
        .read_text()
    )
    assert raw["verified_on_full_spur"] is False


def test_fit_files_merge_order_is_sorted_path(tmp_path: Path):
    triples = []
    for i in (3, 1, 2):
        pred, gt, mask = _ramp(seed=i)
        pp = tmp_path / f"pred_{i}.npy"
        gp = tmp_path / f"gt_{i}.npy"
        mp = tmp_path / f"mask_{i}.npy"
        np.save(pp, pred.astype(np.float32))
        np.save(gp, gt.astype(np.float32))
        np.save(mp, mask.astype(np.uint8))
        triples.append((pp, gp, mp))
    a, b, acc, used = fit_files(triples, erode_r=0, min_gt_std=0.0)
    assert used == 3
    assert abs(a - PRO_ALPHA) < 1e-6
    assert abs(b - PRO_BETA) < 1e-6


def test_cli_without_pairs_exits_2():
    from spur_depth.calib.fit_scale_shift import main

    assert main([]) == 2
