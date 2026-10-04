"""Cut-target GIF: where the cutter goes and how its jaws line up, v1 tangent vs fitted axis.

One rig's 6-height sweep (run after test scoring, never before). Each frame shows
the same BranchNet graph twice, with the cut axis from the v1 rule (tangent
+-1.5 cm at the cut point) and from the fitted rule (cut_axis="fit"). Each
predicted cut is a short segment along its axis (the jaws close across it),
coloured by the scorer's own success-maximising assignment: strict success
(2 cm, 15 deg), Jain success only (5 cm, 30 deg), or miss. Ground-truth cuts
of observable edges are white.

python scripts/render_branch_cuts.py --pred-dir RUN/pred --cfg '{...}' --tree lpy_envy_00065 \
    --out docs/readme/branch_cuts.gif
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_branches as ev  # noqa: E402

from spur_depth.branches import gt as G  # noqa: E402
from spur_depth.branches.assemble import AssembleConfig, assemble, fit_cut_axes  # noqa: E402
from spur_depth.branches.dataset import frame_depth, load_cached_frame  # noqa: E402
from spur_depth.branches.depth_fusion import fuse_sensor_mono  # noqa: E402
from spur_depth.branches.metrics import CUT_TIERS, observable_edges, scored_parts  # noqa: E402
from spur_depth.branches.tree import IGNORE, TRUNK  # noqa: E402
from spur_depth.branches.viz import CRITICAL, GOOD, INK_2, label, project, save_gif  # noqa: E402

AMBER = "#e0a100"
WHITE = (255, 255, 255)


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    c = abs(float(a @ b)) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    return float(np.degrees(np.arccos(min(c, 1.0))))


def outcomes(gt_cuts: list, pred_cuts: list) -> list[str]:
    """'strict' / 'jain' / 'miss' per predicted cut, from the scorer's assignment rule."""
    out = ["miss"] * len(pred_cuts)
    if not gt_cuts or not pred_cuts:
        return out
    D = np.array([[np.linalg.norm(g[0] - p[0]) for p in pred_cuts] for g in gt_cuts])
    A = np.array([[_angle(g[1], p[1]) for p in pred_cuts] for g in gt_cuts])
    for name in ("jain", "strict"):  # strict overwrites jain
        dmax, amax = CUT_TIERS[name]
        S = (D <= dmax) & (A <= amax)
        r, c = linear_sum_assignment(np.where(S, 1e-3 * D / dmax, 1.0))
        for i, j in zip(r, c):
            if S[i, j]:
                out[j] = name
    return out


def frame_cuts(ref, pred_dir: Path, cfg: AssembleConfig, lo: float, hi: float) -> dict:
    gtg, gt_cls, K = ev.gt_graph(ref, ev.CACHE)
    frame = load_cached_frame(ref, ev.CACHE, skel=False)
    cls, conf, tweak, _ = ev.predicted_classes(
        "branchnet", ref, "sensor", gt_cls, frame, pred_dir, None
    )
    wood = (cls > 0) & (cls != IGNORE)
    depth, _ = fuse_sensor_mono(
        frame_depth("sensor", frame, ref), frame_depth("da2", frame, ref), wood
    )
    pred = assemble(cls, depth, K, conf=conf, cfg=replace(cfg, cut_axis="tangent", **tweak))
    obs = observable_edges(gtg, scored_parts(gtg))
    gt_cuts = [p.cut() for p in gtg.parts.values() if p.cls != TRUNK and (p.pid, p.parent) in obs]
    parts = [
        p for p in pred.parts.values() if p.cls != TRUNK and p.parent is not None and p.length > 0
    ]
    tangent = [p.cut() for p in parts]
    fit_cut_axes(pred, lo, hi)
    fitted = [p.cut() for p in parts]
    return {
        "rgb": frame["rgb"],
        "K": K,
        "gt": gt_cuts,
        "tangent": (tangent, outcomes(gt_cuts, tangent)),
        "fitted": (fitted, outcomes(gt_cuts, fitted)),
    }


def _draw(rgb, K, gt_cuts, cuts, outs, box, scale) -> np.ndarray:
    x0, y0, x1, y1 = box
    img = (rgb[y0:y1, x0:x1].astype(np.float32) * 0.55).astype(np.uint8)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    Ks = K.copy()
    Ks[0, 2] -= x0
    Ks[1, 2] -= y0
    Ks[:2] *= scale
    aa = cv2.LINE_AA
    half = 0.03
    for c, d in gt_cuts:
        uv, ok = project(np.stack([c - half * d, c, c + half * d]), Ks)
        if ok.all():
            cv2.line(img, tuple(np.int32(uv[0])), tuple(np.int32(uv[2])), WHITE, 1, aa)
            cv2.circle(img, tuple(np.int32(uv[1])), 4, WHITE, 1, aa)
    col = {"strict": _rgb(GOOD), "jain": _rgb(AMBER), "miss": _rgb(CRITICAL)}
    for (c, d), o in sorted(zip(cuts, outs), key=lambda t: t[1] != "miss"):
        uv, ok = project(np.stack([c - half * d, c, c + half * d]), Ks)
        if ok.all():
            cv2.line(img, tuple(np.int32(uv[0])), tuple(np.int32(uv[2])), col[o], 3, aa)
            cv2.circle(img, tuple(np.int32(uv[1])), 3, col[o], -1, aa)
    return img


def _legend(width: int, height: int = 30) -> np.ndarray:
    strip = np.full((height, width, 3), 252, np.uint8)
    x = 12
    for text, c in (
        ("strict success: 2 cm + 15 deg", _rgb(GOOD)),
        ("Jain success only: 5 cm + 30 deg", _rgb(AMBER)),
        ("miss", _rgb(CRITICAL)),
        ("ground-truth cut and axis", (150, 150, 150)),
    ):
        cv2.line(strip, (x, height // 2), (x + 22, height // 2), c, 3, cv2.LINE_AA)
        cv2.putText(
            strip,
            text,
            (x + 30, height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            _rgb(INK_2),
            1,
            cv2.LINE_AA,
        )
        x += 40 + int(8.2 * len(text))
    return strip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--cfg", type=str, required=True, help="JSON AssembleConfig incl. cut_fit_*")
    ap.add_argument("--tree", default=G.PAPER_TEST[1])
    ap.add_argument("--rig", default="box_cam1")
    ap.add_argument("--side", default="l")
    ap.add_argument("--panel-h", type=int, default=460)
    ap.add_argument("--out", type=Path, default=Path("docs/readme/branch_cuts.gif"))
    args = ap.parse_args(argv)
    cfg = replace(AssembleConfig(), **json.loads(args.cfg))
    lo, hi = cfg.cut_fit_lo_m, cfg.cut_fit_hi_m
    refs = [G.FrameRef(args.tree, args.rig, s, args.side) for s in G.SHOTS]
    frames = [frame_cuts(r, args.pred_dir, cfg, lo, hi) for r in refs]
    # One crop for the sweep: every GT cut point of every height.
    uv = np.concatenate(
        [project(np.array([c for c, _ in f["gt"]]), f["K"])[0] for f in frames if f["gt"]]
    )
    h, w = frames[0]["rgb"].shape[:2]
    pad = 40
    box = (
        int(max(0, np.percentile(uv[:, 0], 1) - pad)),
        int(max(0, np.percentile(uv[:, 1], 1) - pad)),
        int(min(w, np.percentile(uv[:, 0], 99) + pad)),
        int(min(h, np.percentile(uv[:, 1], 99) + pad)),
    )
    scale = args.panel_h / float(box[3] - box[1])
    tiles, report = [], []
    for k, f in enumerate(frames):
        n = len(f["gt"])
        panels = []
        fit_title = f"v2: axis fitted {lo * 100:.0f}-{hi * 100:.0f} cm past the junction"
        for key, title in (("tangent", "v1: axis = tangent at the cut"), ("fitted", fit_title)):
            cuts, outs = f[key]
            img = _draw(f["rgb"], f["K"], f["gt"], cuts, outs, box, scale)
            ns = sum(o == "strict" for o in outs)
            panels.append(label(img, f"{title}   strict {ns}/{n}"))
            report.append(
                {
                    "frame": refs[k].key,
                    "rule": key,
                    "gt_cuts": n,
                    "strict": ns,
                    "jain_or_better": sum(o != "miss" for o in outs),
                }
            )
        row = np.concatenate(panels, 1)
        row = label(
            row,
            f"camera height {k + 1}/6 (BranchNet, D435 + DA2-ft fused depth)",
            y=row.shape[0] - 10,
        )
        tiles.append(np.concatenate([row, _legend(row.shape[1])], 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_gif(tiles, args.out, duration=1100)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "tree": args.tree,
                "rig": args.rig,
                "side": args.side,
                "cfg": json.loads(args.cfg),
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
