"""Real MFO cherry frames, zero-shot: one panel per measure, one bar per BranchNet version.

Reads the rescoring evidence (scripts/rescore_mfo_mapping.py, UFO mapping) of each model and
plots the MFO test split at each model's val-chosen resize factor. Numbers only: no MFO image.

python scripts/render_branch_mfo.py A.json:"label" B.json:"label" ... --out docs/readme/branch_mfo.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Ordinal blue ramp (one hue, light -> dark = older -> newer), validated with the dataviz
# skill's palette checker (--ordinal, light surface #fcfcfb).
RAMP = ("#86b6ef", "#2a78d6", "#104281")
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
PANELS = [
    ("miou3", "mIoU: leader, sidebranch, spur", "higher is better"),
    ("wood_iou_t", "wood IoU", "higher is better"),
    ("labelled_wood_recall", "labelled wood found", "higher is better"),
    ("background_predicted_wood_rate", "background called wood", "lower is better"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="RESCORE.json:label, oldest first")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if len(args.runs) > len(RAMP):
        raise SystemExit(f"at most {len(RAMP)} models")
    runs = []
    for spec in args.runs:
        path, label = spec.split(":", 1)
        d = json.loads(Path(path).read_text())
        if d["mapping"] != "ufo":
            raise SystemExit(f"{path}: expected the UFO mapping")
        runs.append((label, d["headline"][args.split], d["chosen_factor"]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PANELS), figsize=(11.5, 3.3), facecolor=SURFACE)
    for ax, (key, title, note) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        vals = [r[1][key] for r in runs]
        x = range(len(runs))
        bars = ax.bar(
            x, vals, width=0.78, color=RAMP[: len(runs)], edgecolor=SURFACE, linewidth=2, zorder=2
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.012,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK_2,
            )
        ax.set_title(f"{title}\n({note})", fontsize=8.5, color=INK)
        ax.set_xticks([])
        ax.set_ylim(0, 1.0)  # every measure is a fraction: one shared scale
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
        ax.tick_params(axis="y", labelsize=7, colors=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in RAMP[: len(runs)]]
    labels = [f"{r[0]} (resize x{r[2]:g})" for r in runs]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(runs),
        frameon=False,
        fontsize=8,
        labelcolor=INK_2,
    )
    fig.suptitle(
        f"Real MFO cherry frames, zero-shot, {args.split} split (videos 174-184): BranchNet trained on renders only",
        fontsize=9.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
