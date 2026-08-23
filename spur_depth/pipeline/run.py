"""Run the perception stack on one val tree and write README demos.

    python -m spur_depth.pipeline.run --out docs/readme
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from spur_depth.pipeline.detect import CLS_TRUNK, boxes_from_mask, draw_boxes
from spur_depth.pipeline.flow import dense_flow, flow_to_rgb
from spur_depth.pipeline.reconstruct import Frame, load_pose, merge_clouds, unproject, write_ply
from spur_depth.pipeline.sensors import fuse_depths
from spur_depth.pipeline.sim2real import appearance_gap, box_anchor_scale

DATA = Path("/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur")
BARK, TREE, SHOT = "bark_brown_02", "lpy_envy_00042", "shot01"
BLUE = (33, 150, 243)


def _load(kind: str, set_id: str, side: str, ext: str):
    p = DATA / kind / BARK / TREE / set_id / f"{TREE}_{SHOT}_{side}.{ext}"
    return p


def _rgb(set_id: str, side: str) -> np.ndarray:
    return np.asarray(Image.open(_load("Optical_flow", set_id, side, "png")).convert("RGB"))


def _mask(set_id: str, side: str) -> np.ndarray:
    return np.asarray(Image.open(_load("mask", set_id, side, "png")).convert("L")) > 0


def _depth(kind: str, set_id: str, side: str) -> np.ndarray:
    arr = np.load(_load(kind, set_id, side, "npy")).astype(np.float32)
    arr[arr >= 1e9] = 0.0
    return arr


def _ann(set_id: str, side: str) -> dict:
    p = DATA / "ann" / BARK / TREE / set_id / f"{TREE}_{SHOT}_{side}.json"
    return json.loads(p.read_text())


def _gif(frames: list[np.ndarray], path: Path, duration: int = 400) -> None:
    ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=duration, loop=0, optimize=True)


def _orbit_png(pts: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.2, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    n = min(len(pts), 12000)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), n, replace=False) if len(pts) > n else np.arange(len(pts))
    p = pts[idx]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 3:6], s=0.4, depthshade=False)
    ax.set_axis_off()
    ax.view_init(elev=12, azim=-40)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _orbit_gif(pts: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(pts), 8000)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), n, replace=False) if len(pts) > n else np.arange(len(pts))
    p = pts[idx]
    frames = []
    for az in range(0, 360, 30):
        fig = plt.figure(figsize=(5.2, 3.8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 3:6], s=0.35, depthshade=False)
        ax.set_axis_off()
        ax.view_init(elev=14, azim=az)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        frames.append(buf)
    _gif(frames, path, duration=180)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="docs/readme")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    views = [("box", "l"), ("box", "r"), ("box_cam1", "l"), ("box_cam1", "r")]
    rgb0 = _rgb("box", "l")
    mask0 = _mask("box", "l")
    da0 = _depth("Da2Finetune", "box", "l")
    gt0 = _depth("depth", "box", "l")

    dets = boxes_from_mask(mask0, CLS_TRUNK)
    det_vis = draw_boxes(rgb0, dets[:8])
    Image.fromarray(det_vis).save(out / "detect_boxes.png")

    flow = dense_flow(_rgb("box", "l"), _rgb("box", "r"))
    Image.fromarray(flow_to_rgb(flow)).save(out / "flow_stereo.png")

    flow_frames = []
    for set_id in ("box", "box_cam1", "box_cam2"):
        a, b = _rgb(set_id, "l"), _rgb(set_id, "r")
        flow_frames.append(flow_to_rgb(dense_flow(a, b)))
        flow_frames.append(draw_boxes(a, boxes_from_mask(_mask(set_id, "l"))[:6]))
    _gif(flow_frames, out / "stack_orbit.gif", duration=450)

    clouds = []
    for set_id, side in views:
        rgb = _rgb(set_id, side)
        depth = _depth("Da2Finetune", set_id, side)
        mask = _mask(set_id, side)
        K, T = load_pose(_ann(set_id, side))
        clouds.append(unproject(Frame(rgb, depth, mask, K, T), stride=6))
    cloud = merge_clouds(clouds)
    write_ply(out / "tree_cloud.ply", cloud[:40000] if len(cloud) > 40000 else cloud)
    _orbit_png(cloud, out / "reconstruct.png")
    _orbit_gif(cloud, out / "reconstruct.gif")

    fused, _ = fuse_depths([da0, gt0], variances=[np.full_like(da0, 0.01), np.full_like(gt0, 1e-4)], valid=[mask0, mask0])
    appear = appearance_gap(rgb0)
    box_p = DATA / "mask" / BARK / TREE / "box" / f"{TREE}_{SHOT}_l.png"
    box_scale = box_anchor_scale(da0, mask0, K=load_pose(_ann("box", "l"))[0])
    (out / "pipeline_report.json").write_text(
        json.dumps(
            {
                "tree": TREE,
                "n_boxes": len(dets),
                "flow_mean_px": float(np.linalg.norm(flow, axis=-1).mean()),
                "cloud_points": int(cloud.shape[0]),
                "fused_median_m": float(np.nanmedian(fused[mask0])),
                "appearance": appear,
                "box_anchor": box_scale,
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", sorted(x.name for x in out.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
