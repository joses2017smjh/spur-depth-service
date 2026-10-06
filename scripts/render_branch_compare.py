"""Before/after sweep GIF: two BranchNet checkpoints on the same frames (run after test scoring).

--mode graph (default): RGB | model A graph | model B graph | ground truth, assembled with the
scoring config (fused D435 + DA2-ft depth, as in render_branch_figures).
--mode classes: RGB | model A pixel classes | model B pixel classes | ground-truth classes, from
the saved predictions and the label cache only (same-row orchard neighbours, IGNORE in the
ground truth, are drawn white), with each frame's pixel mIoU.

python scripts/render_branch_compare.py --pred-a A/pred --label-a "Envy only" \
    --pred-b B/pred --label-b "Envy + UFO" --tree lpy_ufo_00035 --cfg CFG.json --out OUT.gif
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_branch_figures as F  # noqa: E402

from spur_depth.branches import gt as G  # noqa: E402
from spur_depth.branches.assemble import AssembleConfig  # noqa: E402
from spur_depth.branches.metrics import aggregate, evaluate_frame  # noqa: E402
from spur_depth.branches.viz import draw_graph_2d, label, legend_strip, save_gif  # noqa: E402


def _overlay(rgb: np.ndarray, cls: np.ndarray) -> np.ndarray:
    from spur_depth.branches.tree import IGNORE
    from spur_depth.branches.viz import HEX

    out = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    for c, h in HEX.items():
        out[cls == c] = tuple(int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    out[cls == IGNORE] = (245, 245, 245)
    return out


def _class_legend(width: int, height: int = 30) -> np.ndarray:
    import cv2

    from spur_depth.branches.tree import BRANCH, SPUR, TRUNK
    from spur_depth.branches.viz import HEX, INK_2, SURFACE

    def rgb(h):
        return tuple(int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    img = np.full((height, width, 3), rgb(SURFACE), np.uint8)
    x = 12
    items = [
        (rgb(HEX[TRUNK]), "trunk"),
        (rgb(HEX[BRANCH]), "branch"),
        (rgb(HEX[SPUR]), "shoot / spur"),
        ((205, 205, 205), "same-row trees (ignored in the ground truth)"),
        ((40, 40, 40), "background, incl. the 3 rows behind"),
    ]
    for col, text in items:
        cv2.rectangle(img, (x, height // 2 - 5), (x + 22, height // 2 + 5), col, -1)
        cv2.putText(
            img,
            text,
            (x + 28, height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            rgb(INK_2),
            1,
            cv2.LINE_AA,
        )
        x += 40 + int(8.6 * len(text))
    return img


def _miou(pred: np.ndarray, gt: np.ndarray) -> float:
    from spur_depth.branches.tree import IGNORE

    ok = gt != IGNORE
    ious = []
    for c in range(1, 5):
        g, p = (gt == c) & ok, (pred == c) & ok
        if g.any():
            ious.append((g & p).sum() / max((g | p).sum(), 1))
    return float(np.mean(ious)) if ious else float("nan")


def classes_mode(args, refs) -> int:
    import cv2

    from spur_depth.branches.dataset import read_png, read_rgb

    cache = Path("/nfs/hpc/share/sanchej7/spur-branch-cache/v1")
    tiles, report = [], []
    for k, ref in enumerate(refs):
        rgb = read_rgb(ref)
        gt = read_png(cache / ref.tree / ref.rig / f"{ref.stem}.cls.png")
        pa = read_png(args.pred_a / "sensor" / ref.tree / ref.rig / f"{ref.stem}.cls.png")
        pb = read_png(args.pred_b / "sensor" / ref.tree / ref.rig / f"{ref.stem}.cls.png")
        s = args.panel_h / rgb.shape[0]

        def small(img):
            return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        ma, mb = _miou(pa, gt), _miou(pb, gt)
        panels = [
            label(small(rgb), "RGB"),
            label(small(_overlay(rgb, pa)), f"BranchNet, {args.label_a}: mIoU {ma:.2f}"),
            label(small(_overlay(rgb, pb)), f"BranchNet, {args.label_b}: mIoU {mb:.2f}"),
            label(small(_overlay(rgb, gt)), "ground truth (white: same-row trees, ignored)"),
        ]
        row = np.concatenate(panels, 1)
        row = label(row, f"camera height {k + 1}/6", y=row.shape[0] - 10, x=row.shape[1] - 210)
        tiles.append(np.concatenate([row, _class_legend(row.shape[1])], 0))
        report.append(
            {
                "frame": ref.key,
                args.label_a: ma if np.isfinite(ma) else None,
                args.label_b: mb if np.isfinite(mb) else None,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_gif(tiles, args.out, duration=1000)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {"tree": args.tree, "rig": args.rig, "mode": "classes", "frames": report}, indent=2
        )
        + "\n"
    )
    print("wrote", args.out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("graph", "classes"), default="graph")
    ap.add_argument("--pred-a", type=Path, required=True)
    ap.add_argument("--label-a", required=True)
    ap.add_argument("--pred-b", type=Path, required=True)
    ap.add_argument("--label-b", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--rig", default="box_cam1")
    ap.add_argument("--side", default="l")
    ap.add_argument("--cfg", type=Path, default=None, help="AssembleConfig JSON (graph mode)")
    ap.add_argument("--panel-h", type=int, default=380)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    refs = [G.FrameRef(args.tree, args.rig, s, args.side) for s in G.SHOTS]
    if args.mode == "classes":
        return classes_mode(args, refs)
    if args.cfg is None:
        ap.error("--mode graph needs --cfg")
    cfg = replace(AssembleConfig(), **json.loads(args.cfg.read_text()))
    model, fa, fb = None, [], []
    for ref in refs:
        a = F._frame(ref, args.pred_a, cfg, model, "branchnet")
        model = a["model"]
        fa.append(a)
        fb.append(F._frame(ref, args.pred_b, cfg, model, "branchnet"))
    x0, y0, x1, y1 = F._tree_box(fa)
    s = args.panel_h / float(y1 - y0)
    tiles, report = [], []
    for k, (a, b) in enumerate(zip(fa, fb)):
        rgb = a["rgb"][y0:y1, x0:x1]
        K = a["K"].copy()
        K[0, 2] -= x0
        K[1, 2] -= y0
        import cv2

        panels = [label(cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), "RGB")]
        row_rep = {"frame": refs[k].key}
        for f, name in ((a, args.label_a), (b, args.label_b)):
            m = aggregate([evaluate_frame(f["pred"], f["gt"])])["summary"]
            row_rep[name] = {
                key: m[key] for key in ("skeleton_f1_2cm", "edge_f1", "cut_recall_jain")
            }
            panels.append(label(draw_graph_2d(rgb, f["pred"], K, scale=s), f"BranchNet, {name}"))
        panels.append(label(draw_graph_2d(rgb, F._visible(a["gt"]), K, scale=s), "ground truth"))
        row = np.concatenate(panels, 1)
        row = label(row, f"camera height {k + 1}/6", y=row.shape[0] - 10, x=row.shape[1] - 210)
        tiles.append(np.concatenate([row, legend_strip(row.shape[1])], 0))
        report.append(row_rep)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_gif(tiles, args.out, duration=900)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "tree": args.tree,
                "rig": args.rig,
                "side": args.side,
                "pred_a": str(args.pred_a),
                "pred_b": str(args.pred_b),
                "cfg": json.loads(args.cfg.read_text()),
                "frames": report,
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
