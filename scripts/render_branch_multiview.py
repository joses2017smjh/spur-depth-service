"""Multi-view figure: one rig's 12 single-frame graphs vs the fused tree, orbiting in 3D.

Every frame of the rig (6 heights x 2 stereo eyes) is assembled exactly as in
eval_branches_multiview.py (BranchNet classes, D435 + DA2-ft fused depth), placed
in the world frame with its logged pose, and fused (spur_depth.branches.multiview).
Left: the 12 per-frame graphs overlaid, which is what a robot sees without
fusion (duplicated, depth-noisy parts). Right: the fused tree. Grey: ground-truth
axes of every part visible in at least one frame. Run after test scoring.

python scripts/render_branch_multiview.py --pred-dir RUN/pred --cfg CFG.json --fuse FUSE.json \
    --tree lpy_envy_00065 --rig box_cam1 --out docs/readme/branch_multiview.gif
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_branches as ev  # noqa: E402
import eval_branches_multiview as mv  # noqa: E402

from spur_depth.branches import gt as G  # noqa: E402
from spur_depth.branches.assemble import AssembleConfig, assemble  # noqa: E402
from spur_depth.branches.dataset import frame_depth, load_cached_frame  # noqa: E402
from spur_depth.branches.depth_fusion import fuse_sensor_mono  # noqa: E402
from spur_depth.branches.multiview import FuseConfig, fuse, to_world  # noqa: E402
from spur_depth.branches.tree import BRANCH, IGNORE, SHOOT, SPUR, TRUNK  # noqa: E402
from spur_depth.branches.viz import HEX, INK, INK_2, MUTED, SURFACE, WIDTH, save_gif  # noqa: E402


def world_graphs(refs, pred_dir: Path, cfg: AssembleConfig):
    out, seen = [], set()
    for ref in refs:
        gtg, gt_cls, K = ev.gt_graph(ref, ev.CACHE)
        seen |= {p.pid for p in gtg.parts.values() if p.visible is not None and p.visible.any()}
        frame = load_cached_frame(ref, ev.CACHE, skel=False)
        cls, conf, tweak, _ = ev.predicted_classes(
            "branchnet", ref, "sensor", gt_cls, frame, pred_dir, None
        )
        wood = (cls > 0) & (cls != IGNORE)
        depth, _ = fuse_sensor_mono(
            frame_depth("sensor", frame, ref), frame_depth("da2", frame, ref), wood
        )
        pred = assemble(cls, depth, K, conf=conf, cfg=replace(cfg, **tweak))
        out.append(to_world(pred, mv.frame_pose(ref, ev.CACHE)))
    return out, seen


def _segments(graphs):
    segs = {c: [] for c in (TRUNK, BRANCH, SHOOT, SPUR)}
    cuts = []
    for g in graphs:
        for p in g.parts.values():
            if len(p.points) > 1:
                segs[p.cls].append(np.stack([p.points[:-1], p.points[1:]], 1))
            if p.cls != TRUNK and p.parent is not None and p.length > 0:
                cuts.append(p.cut()[0])
    return segs, np.array(cuts).reshape(-1, 3)


def face_azimuth(a: np.ndarray, b: np.ndarray) -> float:
    """Azimuth (deg) looking at the tree face-on: along the least-spread horizontal direction."""
    xy = np.concatenate([a, b])[:, :2]
    w, v = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))
    n = v[:, 0]
    return float(np.degrees(np.arctan2(n[1], n[0])))


def orbit(panels, gt_ab, titles, suptitle, n_frames=40, swing=55.0, elev=10.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    a, b = gt_ab
    pts = np.concatenate([a, b])
    lo = np.percentile(pts, 1, axis=0) - 0.05
    hi = np.percentile(pts, 99, axis=0) + 0.05
    drawn = [_segments(g) for g in panels]
    face = face_azimuth(a, b)
    azimuths = face + swing * np.sin(np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False))
    frames = []
    for az in azimuths:
        fig = plt.figure(figsize=(11.0, 6.4), facecolor=SURFACE)
        for k, ((segs, cuts), title) in enumerate(zip(drawn, titles)):
            ax = fig.add_axes([0.5 * k, 0.0, 0.5, 0.9], projection="3d", facecolor=SURFACE)
            ax.add_collection3d(Line3DCollection(np.stack([a, b], 1), colors=MUTED, linewidths=0.5))
            for c in (SPUR, SHOOT, BRANCH, TRUNK):
                if segs[c]:
                    ax.add_collection3d(
                        Line3DCollection(
                            np.concatenate(segs[c]), colors=HEX[c], linewidths=WIDTH[c]
                        )
                    )
            if len(cuts):
                ax.scatter(
                    cuts[:, 0], cuts[:, 1], cuts[:, 2], s=5, c=INK, depthshade=False, linewidths=0
                )
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-3)), zoom=1.12)
            ax.view_init(elev=elev, azim=az)
            ax.set_axis_off()
            fig.text(0.25 + 0.5 * k, 0.9, title, ha="center", color=INK, fontsize=10)
        fig.text(0.5, 0.965, suptitle, ha="center", color=INK, fontsize=10.5)
        fig.text(
            0.5,
            0.02,
            "blue trunk, orange branch, green shoot / spur, black dot: cut point; grey: ground-truth axes",
            ha="center",
            color=INK_2,
            fontsize=8.5,
        )
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--cfg", type=Path, required=True, help="AssembleConfig overrides JSON file")
    ap.add_argument("--fuse", type=Path, required=True, help="FuseConfig overrides JSON file")
    ap.add_argument("--tree", default=G.PAPER_TEST[1])
    ap.add_argument("--rig", default="box_cam1")
    ap.add_argument("--out", type=Path, default=Path("docs/readme/branch_multiview.gif"))
    args = ap.parse_args(argv)
    cfg = replace(AssembleConfig(), **json.loads(args.cfg.read_text()))
    fcfg = replace(FuseConfig(), **json.loads(args.fuse.read_text()))
    refs = [r for r in G.list_frames([args.tree]) if r.rig == args.rig]
    views, seen = world_graphs(refs, args.pred_dir, cfg)
    fused = fuse(views, fcfg)
    if cfg.cut_axis == "fit":
        from spur_depth.branches.assemble import fit_cut_axes

        fit_cut_axes(fused, cfg.cut_fit_lo_m, cfg.cut_fit_hi_m)
    rgb, depth, mask, ann = G.load_frame(refs[0])
    model = G.load_tree_model(args.tree, ann)
    keep = np.isin(model.seg_part, sorted(seen))
    n_single = sum(len(g.parts) for g in views)
    frames = orbit(
        [views, [fused]],
        (model.seg_a[keep], model.seg_b[keep]),
        [
            f"{len(views)} single-frame graphs, logged poses ({n_single} parts)",
            f"fused tree ({len(fused.parts)} parts)",
        ],
        f"{args.tree} {args.rig}: BranchNet + fused D435/DA2-ft depth, 6 heights x 2 eyes",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, args.out, duration=140, colors=96)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "tree": args.tree,
                "rig": args.rig,
                "frames": [r.key for r in refs],
                "assemble_config": json.loads(args.cfg.read_text()),
                "fuse_config": json.loads(args.fuse.read_text()),
                "pred_dir": str(args.pred_dir),
                "single_frame_parts": n_single,
                "fused_parts": len(fused.parts),
                "gt_parts_visible_in_any_frame": len(seen),
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", args.out, n_single, "->", len(fused.parts), "parts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
