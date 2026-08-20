"""Fixture group: 6 synthetic 280×512 views, dummy inference is deterministic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from spur_depth.serve.preprocess import REFINER_H, REFINER_W, load_input_depth, load_rgb
from spur_depth.serve.runner import DummyRunner

_FIXTURE = Path(__file__).parent / "fixtures" / "group"


def test_fixture_group_dummy_is_finite_and_shaped():
    images = [Image.open(_FIXTURE / f"view_{i:02d}.png") for i in range(1, 7)]
    depths = [np.load(_FIXTURE / f"depth_{i:02d}.npy") for i in range(1, 7)]
    out = DummyRunner().predict_group(images, depths)
    assert out.shape == (6, REFINER_H, REFINER_W)
    assert np.isfinite(out).all()
    # depths were already 280×512; bilinear is identity on that grid aside from dtype
    for i, d in enumerate(depths):
        got = load_input_depth(d.astype(np.float32)).squeeze(0).numpy()
        assert np.allclose(out[i], got, atol=1e-5)


def test_fixture_rgb_matches_refiner_contract():
    t = load_rgb(_FIXTURE / "view_01.png")
    assert t.shape == (3, REFINER_H, REFINER_W)
