"""Results chart + markdown table from an eval_branches.py summary.json.

One panel per depth source; bars compare the existing stack (TinyUNet, muted)
with BranchNet (blue) on the primary metrics; the oracle-class ceiling is a tick.

python scripts/render_branch_results.py SUMMARY.json --out docs/readme/branch_results.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = [
    ("skeleton_f1_2cm", "skeleton F1\n@2 cm"),
    ("edge_f1", "edge F1"),
    ("cut_recall_jain", "cut recall\n5 cm, 30°"),
    ("cut_recall_strict", "cut recall\n2 cm, 15°"),
]
DEPTHS = [
    ("sensor", "simulated D435"),
    ("fused", "D435 + DA2-ft fused"),
    ("da2", "DA2-ft (RGB only)"),
    ("gt", "rendered depth"),
]
SURFACE, INK, INK_2, MUTED, GRID, BLUE = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#2a78d6",
)


def _get(res: dict, method: str, depth: str, key: str):
    r = res.get(f"{method}/{depth}")
    return None if r is None else r["summary"].get(key)


def chart(summary: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = summary["results"]
    depths = [d for d in DEPTHS if any(f"{m}/{d[0]}" in res for m in ("branchnet", "tinyunet"))]
    fig, axes = plt.subplots(
        1, len(depths), figsize=(3.3 * len(depths), 3.4), facecolor=SURFACE, sharey=True
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(METRICS))
    w = 0.36
    for ax, (dk, title) in zip(axes, depths):
        ax.set_facecolor(SURFACE)
        for k, (method, col, label) in enumerate(
            (("tinyunet", MUTED, "existing stack (TinyUNet)"), ("branchnet", BLUE, "BranchNet"))
        ):
            if f"{method}/{dk}" not in res:
                continue
            vals = [(_get(res, method, dk, m) or 0.0) for m, _ in METRICS]
            bars = ax.bar(x + (k - 0.5) * w, vals, w * 0.92, color=col, label=label, zorder=2)
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.015,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=INK_2,
                )
        ceil = [_get(res, "oracle", dk, m) for m, _ in METRICS]
        for xi, c in zip(x, ceil):
            if c is not None:
                ax.plot([xi - w, xi + w], [c, c], color=INK, lw=1.2, zorder=3)
        ax.set_xticks(x, [lbl for _, lbl in METRICS], fontsize=7, color=INK_2)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(axis="y", labelsize=7, colors=INK_2)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left", labelcolor=INK_2)
    fig.text(
        0.99,
        0.01,
        "black tick: same assembly on ground-truth classes (ceiling)",
        ha="right",
        fontsize=7,
        color=INK_2,
    )
    fig.suptitle(
        f"{summary['split'].capitalize()} trees {', '.join(t[-5:] for t in summary['trees'])}: "
        f"{summary_frames(summary)} frames",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def summary_frames(summary: dict) -> int:
    return max(r["n_frames"] for r in summary["results"].values())


def table(summary: dict) -> str:
    cols = [
        ("skeleton_f1_2cm", "Skeleton F1 @2 cm"),
        ("edge_f1", "Edge F1"),
        ("cut_recall_jain", "Cut recall 5 cm/30°"),
        ("cut_f1_jain", "Cut F1 5 cm/30°"),
        ("cut_recall_strict", "Cut recall 2 cm/15°"),
        ("iou_spur", "Spur IoU"),
        ("miou_tree", "mIoU (4 classes)"),
    ]
    head = "| Classes | Depth | " + " | ".join(c[1] for c in cols) + " |"
    sep = "| --- | --- | " + " | ".join("---" for _ in cols) + " |"
    rows = [head, sep]
    names = {"oracle": "GT (ceiling)", "tinyunet": "TinyUNet (existing)", "branchnet": "BranchNet"}
    for method in ("tinyunet", "branchnet", "oracle"):
        for dk, dname in DEPTHS:
            r = summary["results"].get(f"{method}/{dk}")
            if r is None:
                continue
            s = r["summary"]
            vals = []
            for key, _ in cols:
                v = s.get(key)
                if method == "oracle" and key in ("iou_spur", "miou_tree"):
                    vals.append("1.00")
                elif method == "tinyunet" and key in ("iou_spur", "miou_tree"):
                    vals.append("n/a")
                else:
                    vals.append("–" if v is None else f"{v:.2f}")
            rows.append(f"| {names[method]} | {dname} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/readme/branch_results.png"))
    args = ap.parse_args(argv)
    summary = json.loads(args.summary.read_text())
    chart(summary, args.out)
    print(table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
