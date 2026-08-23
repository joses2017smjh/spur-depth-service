"""PSI / KS contract: identical distributions stay quiet; a shift alerts."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from spur_depth.serve.drift import DriftMonitor, ks_d, psi
from spur_depth.serve.input_stats import image_stats, request_stats


def test_psi_identical_is_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    assert psi(x, x.copy()) < 0.02


def test_psi_mean_shift_alerts():
    rng = np.random.default_rng(1)
    expected = rng.normal(0, 1, 2000)
    actual = rng.normal(3, 1, 2000)
    assert psi(expected, actual) > 0.25


def test_ks_identical_is_zero():
    x = np.linspace(0, 1, 50)
    assert ks_d(x, x) == 0.0


def test_ks_separated_is_one():
    assert ks_d(np.zeros(20), np.ones(20)) == 1.0


def test_monitor_flags_shifted_luminance(tmp_path):
    rng = np.random.default_rng(2)
    n = 200
    baseline = {
        "n_frames": n,
        "source": "test",
        "features": {
            "luminance_mean": {"values": rng.normal(0.4, 0.02, n).tolist()},
            "luminance_std": {"values": rng.normal(0.1, 0.01, n).tolist()},
            "focus_var": {"values": rng.normal(50, 5, n).tolist()},
            "saturated_frac": {"values": np.abs(rng.normal(0.01, 0.002, n)).tolist()},
            "depth_p50": {"values": rng.normal(2.0, 0.1, n).tolist()},
            "oor_frac": {"values": np.abs(rng.normal(0.02, 0.005, n)).tolist()},
        },
    }
    mon = DriftMonitor(baseline, window=50, log_path=tmp_path / "req.jsonl")
    keys = list(baseline["features"])
    # Replay the baseline itself so PSI stays quiet.
    for i in range(50):
        snap = mon.observe({k: baseline["features"][k]["values"][i] for k in keys})
    assert snap["features"]["luminance_mean"]["alert"] is False
    for _ in range(50):
        row = {k: baseline["features"][k]["values"][0] for k in keys}
        row["luminance_mean"] = float(rng.normal(0.9, 0.01))
        snap = mon.observe(row)
    assert snap["features"]["luminance_mean"]["alert"] is True
    lines = (tmp_path / "req.jsonl").read_text().strip().splitlines()
    assert len(lines) == 100
    json.loads(lines[0])


def test_image_stats_keys():
    img = Image.new("RGB", (32, 16), color=(10, 20, 30))
    s = image_stats(img)
    assert 0.0 <= s["luminance_mean"] <= 1.0
    assert s["focus_var"] >= 0.0
    depth = np.full((16, 32), 2.0, np.float32)
    row = request_stats(images=img, depths=depth, group_id="t")
    assert row["group_id"] == "t"
    assert row["oor_frac"] == 0.0
