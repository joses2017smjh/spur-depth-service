"""README figures for the branch module (run after test scoring, never before).

Writes into --out (default docs/readme):
  branch_sweep.gif    one rig's 6-height camera sweep: RGB | prediction | ground truth
  branch_orbit.gif    the 6 heights of that rig, predicted graphs placed in the world frame
                      with logged poses, over the GT axes in grey
  branch_depth.png    one frame lifted with each depth source, seen from above (x-z)

python scripts/render_branch_figures.py --pred-dir RUN/pred --tree lpy_envy_00065
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from spur_depth.branches import gt as G
from spur_depth.branches.assemble import AssembleConfig, assemble
from spur_depth.branches.depth_fusion import fuse_sensor_mono
from spur_depth.branches.depth_noise import frame_seed, simulate_sensor
from spur_depth.branches.tree import IGNORE, TreeGraph
from spur_depth.branches.viz import (
    HEX,
    INK,
    INK_2,
    MUTED,
    SURFACE,
    draw_graph_2d,
    label,
    legend_strip,
    orbit_frames,
    save_gif,
)
from spur_depth.pipeline.reconstruct import load_pose


def _cls(pred_dir: Path, source: str, ref: G.FrameRef):
    base = pred_dir / source / ref.tree / ref.rig / ref.stem
    cls = cv2.imread(f"{base}.cls.png", cv2.IMREAD_UNCHANGED)
    conf = cv2.imread(f"{base}.conf.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    return cls, conf


_TINY = None


def _classes(method: str, ref: G.FrameRef, pred_dir: Path | None, rgb, fg):
    """(class map, confidence, assembly tweaks) for the method shown in the figures."""
    global _TINY
    if method == "branchnet":
        cls, conf = _cls(pred_dir, "sensor", ref)
        return cls, conf, {}
    if method == "oracle":
        return fg.cls, None, {}
    if _TINY is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "eval_branches", Path(__file__).with_name("eval_branches.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _TINY = mod.TinyUNetBaseline()
    return _TINY(rgb).astype(np.uint8), None, {"geometric_classes": True}


def _frame(ref: G.FrameRef, pred_dir: Path, cfg: AssembleConfig, model=None, method="branchnet"):
    rgb, depth, mask, ann = G.load_frame(ref)
    K, T = load_pose(ann)
    K, T = K.astype(np.float64), T.astype(np.float64)
    model = model or G.load_tree_model(ref.tree, ann)
    fg = G.label_frame(model, K, T, depth, mask)
    # Same depth realisation the scorer and BranchNet used: the cached mm depth.
    depth = np.round(np.minimum(depth, 65.535) * 1000.0) / 1000.0
    sensor = simulate_sensor(depth, fg.cls > 0, frame_seed(ref.key))
    da2 = np.load(ref.path("Da2Finetune")).astype(np.float32)
    cls, conf, tweak = _classes(method, ref, pred_dir, rgb, fg)
    cfg = replace(cfg, **tweak)
    wood = (cls > 0) & (cls != IGNORE) if method != "oracle" else fg.cls > 0
    fused, _ = fuse_sensor_mono(sensor, da2, wood)
    pred = assemble(cls, fused, K, conf=conf, cfg=cfg)
    return {
        "rgb": rgb,
        "K": K,
        "T": T,
        "gt": fg.graph,
        "pred": pred,
        "cls": cls,
        "depths": {"gt": depth, "sensor": sensor, "da2": da2, "fused": fused},
        "model": model,
        "cfg": cfg,
    }


def sweep_gif(frames: list[dict], out: Path) -> None:
    tiles = []
    for k, f in enumerate(frames):
        a = label(
            cv2.resize(f["rgb"], None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA), "RGB"
        )
        b = label(draw_graph_2d(f["rgb"], f["pred"], f["K"], scale=1 / 3), "predicted graph")
        c = label(draw_graph_2d(f["rgb"], _visible(f["gt"]), f["K"], scale=1 / 3), "ground truth")
        row = np.concatenate([a, b, c], 1)
        row = label(row, f"camera height {k + 1}/6", y=row.shape[0] - 10, x=row.shape[1] - 210)
        tiles.append(np.concatenate([row, legend_strip(row.shape[1])], 0))
    save_gif(tiles, out, duration=900)


def _visible(g: TreeGraph) -> TreeGraph:
    """Ground truth restricted to its visible samples, for drawing."""
    out = TreeGraph(frame=g.frame)
    for p in g.parts.values():
        if p.visible is None or p.visible.sum() < 2:
            continue
        q = p.points[p.visible]
        out.add(type(p)(p.pid, p.cls, q, p.radius[p.visible], p.parent, p.attach))
    return out


def orbit_gif(frames: list[dict], out: Path, title: str) -> None:
    world = []
    for f in frames:
        R, t = f["T"][:3, :3], f["T"][:3, 3]
        world.append(f["pred"].transformed(R.T, -R.T @ t, "world"))
    m = frames[0]["model"]
    keep = np.isin(m.seg_part, np.unique([p.pid for f in frames for p in f["gt"].parts.values()]))
    shots = orbit_frames(world, (m.seg_a[keep], m.seg_b[keep]), title=title)
    save_gif(shots, out, duration=140, colors=96)


def depth_png(f: dict, out: Path, cfg: AssembleConfig) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spur_depth.branches.metrics import aggregate, evaluate_frame

    names = {
        "gt": "rendered depth",
        "sensor": "simulated D435",
        "da2": "DA2-ft (RGB only)",
        "fused": "D435 + DA2-ft fused",
    }
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), facecolor=SURFACE, sharey=True)
    report = {}
    gpts = np.concatenate([p.points[p.visible] for p in f["gt"].parts.values() if p.visible.any()])
    for ax, (key, title) in zip(axes, names.items()):
        g = assemble(f["cls"], f["depths"][key], f["K"], cfg=f["cfg"])
        s = aggregate([evaluate_frame(g, f["gt"])])["summary"]
        report[key] = {k: s[k] for k in ("skeleton_f1_2cm", "edge_f1", "cut_recall_jain")}
        ax.scatter(gpts[:, 0], gpts[:, 2], s=0.3, c=MUTED, linewidths=0)
        for p in g.parts.values():
            ax.plot(p.points[:, 0], p.points[:, 2], color=HEX[p.cls], lw=0.8)
        ax.set_title(f"{title}\nskeleton F1@2cm {s['skeleton_f1_2cm']:.2f}", color=INK, fontsize=9)
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_2, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7")
        ax.set_xlabel("x (m)", color=INK_2, fontsize=8)
    axes[0].set_ylabel("depth z (m)", color=INK_2, fontsize=8)
    lo = np.percentile(gpts[:, 2], 1) - 0.4
    axes[0].set_ylim(lo, np.percentile(gpts[:, 2], 99) + 0.8)
    fig.suptitle(
        "Same predicted classes, four depth sources, seen from above (grey: GT axes)",
        color=INK,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--method", choices=("branchnet", "tinyunet", "oracle"), default="branchnet")
    ap.add_argument("--tree", default=G.PAPER_TEST[1])
    ap.add_argument("--rig", default="box_cam1")
    ap.add_argument("--side", default="l")
    ap.add_argument("--cfg", type=str, default="{}")
    ap.add_argument("--out", type=Path, default=Path("docs/readme"))
    args = ap.parse_args(argv)
    cfg = replace(AssembleConfig(), **json.loads(args.cfg))
    args.out.mkdir(parents=True, exist_ok=True)
    refs = [G.FrameRef(args.tree, args.rig, s, args.side) for s in G.SHOTS]
    model = None
    frames = []
    for ref in refs:
        f = _frame(ref, args.pred_dir, cfg, model, args.method)
        model = f["model"]
        frames.append(f)
        print("framed", ref.key, len(f["pred"].parts), flush=True)
    sweep_gif(frames, args.out / "branch_sweep.gif")
    orbit_gif(
        frames,
        args.out / "branch_orbit.gif",
        f"{args.tree}: 6 camera heights placed in the world frame with their logged poses",
    )
    rep = depth_png(frames[2], args.out / "branch_depth.png", cfg)
    (args.out / "branch_figures.json").write_text(
        json.dumps(
            {"tree": args.tree, "rig": args.rig, "depth_png_frame": refs[2].key, **rep}, indent=2
        )
        + "\n"
    )
    Image.open(args.out / "branch_depth.png").close()
    print("wrote figures to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
