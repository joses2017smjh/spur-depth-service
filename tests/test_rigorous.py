"""Hold-out split and affine helpers do not peek at the paper val trees."""

from __future__ import annotations

import numpy as np

from spur_depth.bench.rigorous import apply_affine, masked_rmse
from spur_depth.calib.fit_scale_shift import fit_pairs
from spur_depth.data.restore_index import PAPER_VAL, split_holdout


def test_split_holdout_keeps_paper_val_out_of_fit():
    rows = [
        {"tree": "lpy_envy_00001"},
        {"tree": "lpy_envy_00042"},
        {"tree": "lpy_envy_00065"},
        {"tree": "lpy_envy_00092"},
    ]
    fit, hold = split_holdout(rows)
    assert {r["tree"] for r in hold} == set(PAPER_VAL)
    assert {r["tree"] for r in fit} == {"lpy_envy_00001", "lpy_envy_00092"}


def test_gt_mask_score_keeps_low_std_frames_xy_drops():
    from spur_depth.bench.improve import _score, _score_gt_mask, headline_stays_ls

    pred = np.full((80, 80), 2.0, dtype=np.float32)
    gt = pred + 0.01  # GT std is 0, so the xy gate drops the frame
    mask = np.ones((80, 80), dtype=bool)
    rec = {"xy": None, "pred": pred, "gt": gt, "mask": mask}
    assert _score([{"xy": None}], 1.0, 0.0)["n"] == 0
    scored = _score_gt_mask([rec], 1.0, 0.0)
    assert scored["n"] == 1
    assert scored["rmse_m"] is not None
    assert headline_stays_ls("huber_irls", 0.0338) is True
    assert headline_stays_ls("huber_irls", 0.0320) is False
    assert headline_stays_ls("ls", 0.0320) is True


def test_holdout_affine_recovers_known_scale_shift():
    rng = np.random.default_rng(0)
    pred = rng.uniform(1.0, 3.0, size=(32, 32)).astype(np.float32)
    gt = (1.1 * pred + 0.2).astype(np.float32)
    mask = np.ones((32, 32), dtype=bool)
    alpha, beta, _ = fit_pairs([(pred, gt, mask)], erode_r=0, min_gt_std=0.01)
    cal = apply_affine(pred, alpha, beta)
    err = masked_rmse(cal, gt, mask, erode_r=0)
    assert err is not None and err < 1e-5
    raw = masked_rmse(pred, gt, mask, erode_r=0)
    assert raw is not None and raw > 0.05
