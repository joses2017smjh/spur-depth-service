"""Before/after figures for the intrinsics fix, plus the corrected reconstruction.

Writes into docs/readme/:
  k_fix_overlay.png    GT cylinder axes projected onto one RGB frame with each K
  k_fix_orbit.gif      GT depth back-projected with the annotated K (left) and the
                       render camera's K (right), GT cylinder axes overlaid
  reconstruct.gif/.png the README's DA2-ft four-view cloud, now with the render K
  k_fix_report.json    surface error of each cloud against the GT cylinders

python scripts/render_k_fix.py --out docs/readme
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from spur_depth.pipeline.reconstruct import Frame, load_pose, merge_clouds, unproject
from spur_depth.pipeline.run import _orbit_gif, _orbit_png

DATA = Path("/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur")
BARK, TREE, SHOT = "bark_brown_02", "lpy_envy_00042", "shot01"
VIEWS = [("box", "l"), ("box", "r"), ("box_cam1", "l"), ("box_cam1", "r")]

# Reference palette (dataviz skill), first three categorical slots + chrome.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
CLASS_COLOR = {"trunk": "#2a78d6", "branch": "#eb6834", "tertiary": "#1baf7a"}


def _frame(set_id: str, side: str, depth_kind: str, trust_k: bool) -> Frame:
    stem = f"{TREE}_{SHOT}_{side}"
    rgb = np.asarray(
        Image.open(DATA / "Optical_flow" / BARK / TREE / set_id / f"{stem}.png").convert("RGB")
    )
    mask = np.asarray(Image.open(DATA / "mask" / BARK / TREE / set_id / f"{stem}.png").convert("L"))
    depth = np.load(DATA / depth_kind / BARK / TREE / set_id / f"{stem}.npy").astype(np.float32)
    depth[depth >= 1e9] = 0.0
    ann = json.loads((DATA / "ann" / BARK / TREE / set_id / f"{stem}.json").read_text())
    K, T = load_pose(ann, trust_annotation_k=trust_k)
    return Frame(rgb, depth, mask > 0, K, T)


def _cloud(depth_kind: str, trust_k: bool, stride: int) -> np.ndarray:
    frames = [_frame(s, side, depth_kind, trust_k) for s, side in VIEWS]
    return merge_clouds([unproject(f, stride=stride) for f in frames])


def _axes():
    cyl = json.loads((DATA / "cylinders_world" / BARK / f"{TREE}.json").read_text())
    c = np.array([r["centroid"] for r in cyl], dtype=np.float64)
    o = np.array([r["orientation"] for r in cyl], dtype=np.float64)
    half = 0.5 * np.array([r["length"] for r in cyl], dtype=np.float64)[:, None]
    rad = np.array([r["radius"] for r in cyl], dtype=np.float64)
    kind = [r["part_name"].split("_")[0] for r in cyl]
    return c - half * o, c + half * o, rad, kind


def _surface_error(pts: np.ndarray, a: np.ndarray, b: np.ndarray, rad: np.ndarray) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, 21)
    samples = (a[:, None, :] + ts[None, :, None] * (b - a)[:, None, :]).reshape(-1, 3)
    dist, j = cKDTree(samples).query(pts[:, :3])
    return np.abs(dist - np.repeat(rad, len(ts))[j])


def _class_of(kind: str) -> str:
    return {"trunk": "trunk", "branch": "branch"}.get(kind, "tertiary")


def _render_pair(before, after, a, b, kind, err_before, err_after, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    lo = np.percentile(after[:, :3], 2, axis=0) - 0.05
    hi = np.percentile(after[:, :3], 98, axis=0) + 0.05
    rng = np.random.default_rng(0)
    pick_b = rng.choice(len(before), min(len(before), 9000), replace=False)
    pick_a = rng.choice(len(after), min(len(after), 9000), replace=False)
    centre = 0.5 * (a + b)
    keep = np.all((centre > lo) & (centre < hi), axis=1)
    panels = [
        (
            before[pick_b],
            f"annotated K  (fx 2667, fy 1500)\nmedian surface error {err_before:.1f} cm",
        ),
        (after[pick_a], f"render K  (f 1493, 28 mm lens)\nmedian surface error {err_after:.2f} cm"),
    ]
    frames = []
    for az in range(-80, 280, 15):
        fig = plt.figure(figsize=(9.6, 6.0), facecolor=SURFACE)
        for k, (pts, title) in enumerate(panels):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d", facecolor=SURFACE)
            ax.set_position([0.5 * k - 0.02, 0.06, 0.54, 0.84])
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=MUTED, s=0.6, alpha=0.6, linewidths=0)
            for cls, lw in (("tertiary", 0.8), ("branch", 1.4), ("trunk", 2.0)):
                sel = [i for i in np.nonzero(keep)[0] if _class_of(kind[i]) == cls]
                segs = np.stack([a[sel], b[sel]], axis=1)
                ax.add_collection3d(Line3DCollection(segs, colors=CLASS_COLOR[cls], linewidths=lw))
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(tuple(hi - lo))
            ax.view_init(elev=12, azim=az)
            ax.set_axis_off()
            ax.set_title(title, color=INK, fontsize=10)
        fig.text(
            0.5,
            0.03,
            "grey: GT depth back-projected from 4 views   lines: GT cylinder axes "
            "(blue trunk, orange branch, aqua spur/shoot)",
            ha="center",
            color=INK_2,
            fontsize=8,
        )
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=160, loop=0, optimize=True)


def _render_overlay(a, b, kind, path: Path) -> None:
    """GT cylinder axes projected onto one RGB frame with each K."""
    set_id, side = "box_cam1", "l"
    stem = f"{TREE}_{SHOT}_{side}"
    rgb = np.asarray(
        Image.open(DATA / "Optical_flow" / BARK / TREE / set_id / f"{stem}.png").convert("RGB")
    )
    ann = json.loads((DATA / "ann" / BARK / TREE / set_id / f"{stem}.json").read_text())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.9), facecolor=SURFACE)
    for ax, trust, title in (
        (axes[0], True, "annotated K (fx 2667, fy 1500)"),
        (axes[1], False, "render K (f 1493, 28 mm lens)"),
    ):
        K, T = load_pose(ann, trust_annotation_k=trust)
        K, T = K.astype(np.float64), T.astype(np.float64)
        pa = (T[:3, :3] @ a.T).T + T[:3, 3]
        pb = (T[:3, :3] @ b.T).T + T[:3, 3]
        ok = (pa[:, 2] > 0.2) & (pb[:, 2] > 0.2)
        ua = (K[:2, :2] @ (pa[:, :2] / pa[:, 2:3]).T).T + K[:2, 2]
        ub = (K[:2, :2] @ (pb[:, :2] / pb[:, 2:3]).T).T + K[:2, 2]
        ax.imshow(rgb)
        for cls, lw in (("tertiary", 0.7), ("branch", 1.2), ("trunk", 1.6)):
            sel = ok & np.array([_class_of(k) == cls for k in kind])
            segs = np.stack([ua[sel], ub[sel]], axis=1)
            ax.add_collection(LineCollection(segs, colors=CLASS_COLOR[cls], linewidths=lw))
        ax.set_xlim(0, rgb.shape[1])
        ax.set_ylim(rgb.shape[0], 0)
        ax.set_axis_off()
        ax.set_title(title, color=INK, fontsize=10)
    fig.text(
        0.5,
        0.02,
        "GT cylinder axes projected into lpy_envy_00042 box_cam1 shot01_l "
        "(blue trunk, orange branch, aqua spur/shoot)",
        ha="center",
        color=INK_2,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.07, wspace=0.03)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="docs/readme")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    a, b, rad, kind = _axes()
    _render_overlay(a, b, kind, out / "k_fix_overlay.png")
    gt_before = _cloud("depth", trust_k=True, stride=4)
    gt_after = _cloud("depth", trust_k=False, stride=4)
    e_before = _surface_error(gt_before, a, b, rad)
    e_after = _surface_error(gt_after, a, b, rad)
    _render_pair(
        gt_before,
        gt_after,
        a,
        b,
        kind,
        float(np.median(e_before)) * 100,
        float(np.median(e_after)) * 100,
        out / "k_fix_orbit.gif",
    )

    da2_before = _cloud("Da2Finetune", trust_k=True, stride=6)
    da2_after = _cloud("Da2Finetune", trust_k=False, stride=6)
    _orbit_png(da2_after, out / "reconstruct.png")
    _orbit_gif(da2_after, out / "reconstruct.gif")

    def stats(e):
        return {
            "n_points": int(e.size),
            "median_m": float(np.median(e)),
            "p90_m": float(np.percentile(e, 90)),
            "frac_within_5mm": float(np.mean(e < 0.005)),
        }

    report = {
        "tree": TREE,
        "shot": SHOT,
        "views": [f"{s}_{side}" for s, side in VIEWS],
        "surface_error_vs_gt_cylinders": {
            "gt_depth_annotated_K": stats(e_before),
            "gt_depth_render_K": stats(e_after),
            "da2ft_depth_annotated_K": stats(_surface_error(da2_before, a, b, rad)),
            "da2ft_depth_render_K": stats(_surface_error(da2_after, a, b, rad)),
        },
        "render_K": load_pose(
            json.loads((DATA / "ann" / BARK / TREE / "box" / f"{TREE}_{SHOT}_l.json").read_text())
        )[0].tolist(),
    }
    (out / "k_fix_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["surface_error_vs_gt_cylinders"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
