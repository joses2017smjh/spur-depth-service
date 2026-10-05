"""Before/after sweep GIF: two BranchNet checkpoints on the same frames (run after test scoring).

Per camera height: RGB | model A classes + graph | model B classes + graph | ground truth, with
the assembly config used for scoring (fused D435 + DA2-ft depth, as in render_branch_figures).

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-a", type=Path, required=True)
    ap.add_argument("--label-a", required=True)
    ap.add_argument("--pred-b", type=Path, required=True)
    ap.add_argument("--label-b", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--rig", default="box_cam1")
    ap.add_argument("--side", default="l")
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--panel-h", type=int, default=380)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = replace(AssembleConfig(), **json.loads(args.cfg.read_text()))
    refs = [G.FrameRef(args.tree, args.rig, s, args.side) for s in G.SHOTS]
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
