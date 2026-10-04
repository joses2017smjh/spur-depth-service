"""v2 chart: what the fitted cut axis, the fine-tuned BranchNet and CDM depth change on test.

One panel per depth source. Bars: v1 BranchNet (tangent cut axis, muted), the same
predictions with the fitted cut axis (blue), the fine-tuned BranchNet with the
fitted axis (dark blue); the black tick is GT classes with the fitted axis.

python scripts/render_branch_v2.py --v1 V1.json --fit FIT.json [FIT_CDM.json] \
    --ft FT.json [FT_CDM.json] --out docs/readme/branch_v2.png
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
    ("cdm", "D435 + CDM refined"),
    ("gt", "rendered depth"),
]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
BARS = [
    ("v1", "#b9b7ae", "v1: BranchNet, tangent cut axis"),
    ("fit", "#2a78d6", "v2: same predictions, fitted cut axis"),
    ("ft", "#14427a", "v2: fine-tuned BranchNet, fitted cut axis"),
]


def _results(paths: list[Path]) -> dict:
    out: dict = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        if not d.get("complete", False):
            raise ValueError(f"{p}: incomplete summary")
        out.update(d["results"])
    return out


def chart(sets: dict[str, dict], out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(DEPTHS), figsize=(3.4 * len(DEPTHS), 3.5), facecolor=SURFACE, sharey=True
    )
    x = np.arange(len(METRICS))
    w = 0.27
    for ax, (dk, dname) in zip(axes, DEPTHS):
        ax.set_facecolor(SURFACE)
        shown = [(k, c, lbl) for k, c, lbl in BARS if f"branchnet/{dk}" in sets.get(k, {})]
        for i, (k, col, lbl) in enumerate(shown):
            s = sets[k][f"branchnet/{dk}"]["summary"]
            vals = [s.get(m) or 0.0 for m, _ in METRICS]
            off = (i - (len(shown) - 1) / 2) * w
            bars = ax.bar(x + off, vals, w * 0.92, color=col, label=lbl, zorder=2)
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.012,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.8,
                    color=INK_2,
                )
        ceil = sets.get("fit", {}).get(f"oracle/{dk}")
        if ceil:
            for xi, (m, _) in zip(x, METRICS):
                v = ceil["summary"].get(m)
                if v is not None:
                    ax.plot([xi - 1.5 * w, xi + 1.5 * w], [v, v], color=INK, lw=1.1, zorder=3)
        ax.set_xticks(x, [lbl for _, lbl in METRICS], fontsize=7, color=INK_2)
        ax.set_ylim(0, 1.05)
        ax.set_title(dname, fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c3c2b7")
        ax.tick_params(axis="y", labelsize=7, colors=INK_2)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left", labelcolor=INK_2)
    fig.text(
        0.99,
        0.01,
        "black tick: ground-truth classes, fitted cut axis (ceiling)",
        ha="right",
        fontsize=7,
        color=INK_2,
    )
    fig.suptitle(title, fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def table(sets: dict[str, dict]) -> str:
    cols = [
        ("skeleton_f1_2cm", "Skeleton F1 @2 cm"),
        ("edge_f1", "Edge F1"),
        ("cut_recall_jain", "Cut recall 5 cm/30°"),
        ("cut_f1_jain", "Cut F1 5 cm/30°"),
        ("cut_recall_strict", "Cut recall 2 cm/15°"),
        ("cut_f1_strict", "Cut F1 2 cm/15°"),
        ("cut_angle_deg_mean_paired", "Cut axis error (deg)"),
        ("miou_tree", "mIoU (4 classes)"),
    ]
    names = {
        "v1": "BranchNet v1, tangent axis",
        "fit": "BranchNet v1, fitted axis",
        "ft": "BranchNet fine-tuned, fitted axis",
    }
    rows = [
        "| Model | Depth | " + " | ".join(c for _, c in cols) + " |",
        "| --- | --- | " + " | ".join("---" for _ in cols) + " |",
    ]
    for k in ("v1", "fit", "ft"):
        for dk, dname in DEPTHS:
            r = sets.get(k, {}).get(f"branchnet/{dk}")
            if r is None:
                continue
            s = r["summary"]
            vals = [
                "–"
                if s.get(c) is None
                else (f"{s[c]:.1f}" if c.endswith("_deg_mean_paired") else f"{s[c]:.2f}")
                for c, _ in cols
            ]
            rows.append(f"| {names[k]} | {dname} | " + " | ".join(vals) + " |")
    for dk, dname in DEPTHS:
        r = sets.get("fit", {}).get(f"oracle/{dk}")
        if r is not None:
            s = r["summary"]
            vals = [
                "–"
                if s.get(c) is None
                else (f"{s[c]:.1f}" if c.endswith("_deg_mean_paired") else f"{s[c]:.2f}")
                for c, _ in cols
            ]
            vals[-1] = "1.00"
            rows.append(
                f"| GT classes (ceiling), fitted axis | {dname} | " + " | ".join(vals) + " |"
            )
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=Path, nargs="+", required=True)
    ap.add_argument("--fit", type=Path, nargs="+", required=True)
    ap.add_argument("--ft", type=Path, nargs="*", default=[])
    ap.add_argument("--title", default="Test trees 00042, 00065: 120 frames, assembly config C")
    ap.add_argument("--out", type=Path, default=Path("docs/readme/branch_v2.png"))
    args = ap.parse_args(argv)
    sets = {"v1": _results(args.v1), "fit": _results(args.fit)}
    if args.ft:
        sets["ft"] = _results(args.ft)
    chart(sets, args.out, args.title)
    print(table(sets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
