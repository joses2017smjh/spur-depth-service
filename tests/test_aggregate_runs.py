"""Accuracy table is generated from the frozen seed JSON, not typed by hand."""

from __future__ import annotations

from spur_depth.bench.aggregate_runs import from_frozen, markdown_table, sample_mean_std


def test_frozen_table_matches_checkpoint_seeds():
    agg = from_frozen()
    assert [r["seed"] for r in agg["seeds"]] == [1, 2, 3, 4, 5]
    rmses = [r["best_rmse_m"] for r in agg["seeds"]]
    mean, std = sample_mean_std(rmses)
    assert abs(mean - 0.0445244) < 1e-6
    assert abs(std - 0.00570) < 5e-5
    md = markdown_table(agg)
    assert "0.044330" in md
    assert "mean ± sample std" in md


def test_cli_frozen_exits_0():
    from spur_depth.bench.aggregate_runs import main

    assert main(["--source", "frozen", "--json"]) == 0
