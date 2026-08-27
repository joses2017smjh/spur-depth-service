#!/usr/bin/env python3
"""Render README figures from the restored val tree and committed bench JSON.

Palette matches Computer_Vision/plot_ablation.py and the architecture diagrams:
  trunk / primary   #2196F3
  contrast          #FF7043
  seed / ink        #333333
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

BLUE = "#2196F3"
ORANGE = "#FF7043"
INK = "#333333"
MUTED = "#6B6B6B"
GRID = "#DDDDDD"

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "readme"
DATA = Path("/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur")
BARK, TREE, SET_ID, SHOT = "bark_brown_02", "lpy_envy_00042", "box", "shot01"
CV = Path("/nfs/hpc/share/sanchej7/Computer_Vision")
PAIRS = (("box", "L"), ("box", "R"), ("box_cam1", "L"), ("box_cam1", "R"), ("box_cam2", "L"), ("box_cam2", "R"))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 160,
        }
    )


def _stem(set_id: str = SET_ID, side: str = "l") -> str:
    return f"{TREE}_{SHOT}_{side}"


def _rgb(set_id: str, side: str, max_w: int | None = None) -> np.ndarray:
    p = DATA / "Optical_flow" / BARK / TREE / set_id / f"{TREE}_{SHOT}_{side.lower()}.png"
    im = Image.open(p).convert("RGB")
    if max_w and im.width > max_w:
        im.thumbnail((max_w, max_w))
    return np.asarray(im)


def _save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.name}  {path.stat().st_size} bytes", flush=True)


def _npy(kind: str, set_id: str = SET_ID, side: str = "l") -> np.ndarray:
    p = DATA / kind / BARK / TREE / set_id / f"{_stem(set_id, side)}.npy"
    arr = np.load(p).astype(np.float32)
    arr[arr >= 1e9] = 0.0
    return arr


def _mask(set_id: str = SET_ID, side: str = "l") -> np.ndarray:
    p = DATA / "mask" / BARK / TREE / set_id / f"{_stem(set_id, side)}.png"
    return np.asarray(Image.open(p).convert("L")) > 0


def _load_primary():
    return _rgb(SET_ID, "l"), _npy("Da2Finetune"), _npy("depth"), _mask()


def _depth_show(ax, depth, mask, title, vmin, vmax):
    vis = np.ma.array(depth, mask=~mask)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="#1A1A1A")
    im = ax.imshow(vis, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()
    return im


def hero_strip(rgb, da, gt, mask) -> None:
    trunk = gt[mask]
    vmin, vmax = np.percentile(trunk, [2, 98]) if trunk.size else (0.5, 6.0)
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.55), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB  ·  box cam, shot 01")
    axes[0].set_axis_off()
    im = _depth_show(axes[1], da, mask, "DA2 fine-tune  ·  ~0.060 m", vmin, vmax)
    _depth_show(axes[2], gt, mask, "Blender GT", vmin, vmax)
    overlay = rgb.copy()
    tint = np.zeros_like(rgb)
    tint[..., 0], tint[..., 1], tint[..., 2] = 33, 150, 243
    alpha = (0.48 * mask).astype(np.float32)[..., None]
    blend = (overlay.astype(np.float32) * (1 - alpha) + tint.astype(np.float32) * alpha).astype(np.uint8)
    axes[3].imshow(blend)
    axes[3].set_title("Trunk mask  ·  loss support")
    axes[3].set_axis_off()
    cbar = fig.colorbar(im, ax=axes[1:3], fraction=0.046, pad=0.02, shrink=0.86)
    cbar.set_label("metres")
    fig.suptitle(
        f"{TREE}  /  {SET_ID}  /  {SHOT}  —  same pixels, three depths",
        fontsize=12,
        fontweight="bold",
        y=1.06,
    )
    _save(fig, "hero_strip.png")


def da2_residual(da, gt, mask) -> None:
    err = np.abs(da - gt)
    vis = np.ma.array(err, mask=~mask)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cmap = plt.get_cmap("Oranges").copy()
    cmap.set_bad(color="#1A1A1A")
    im = ax.imshow(vis, cmap=cmap, vmin=0, vmax=float(np.percentile(err[mask], 95)), interpolation="nearest")
    ax.set_axis_off()
    trunk_rmse = float(np.sqrt(np.mean(err[mask] ** 2)))
    ax.set_title(f"|DA2-ft − GT| on trunk  ·  this frame RMSE {trunk_rmse:.3f} m")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="metres")
    _save(fig, "da2_residual.png")


def nearest_vs_bilinear(gt, mask) -> None:
    import cv2

    h, w = 140, 256
    near = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
    bili = cv2.resize(gt, (w, h), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    cols = np.where(m.any(axis=0))[0]
    col = int(np.median(cols)) if cols.size else w // 2
    sl = slice(max(0, h // 2 - 42), min(h, h // 2 + 42))
    sc = slice(max(0, col - 42), min(w, col + 42))
    vmin, vmax = 0.0, float(np.percentile(gt[mask], 90)) if mask.any() else 4.0

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.45), constrained_layout=True)
    axes[0].imshow(near[sl, sc], cmap="turbo", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title("GT resize: nearest")
    axes[1].imshow(bili[sl, sc], cmap="turbo", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].set_title("GT resize: bilinear")
    diff = np.abs(near - bili)
    im = axes[2].imshow(
        diff[sl, sc],
        cmap="Oranges",
        vmin=0,
        vmax=max(0.25, float(np.percentile(diff[m], 95) if m.any() else 0.4)),
        interpolation="nearest",
    )
    axes[2].set_title("|nearest − bilinear|")
    for ax in axes:
        ax.set_axis_off()
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02, label="metres")
    fig.suptitle(
        "Bilinear on a zeroed background smears the trunk silhouette. Scoring uses nearest.",
        fontsize=11,
        fontweight="bold",
        y=1.04,
    )
    _save(fig, "nearest_vs_bilinear.png")


def accuracy_seeds() -> None:
    blob = json.loads((REPO / "bench" / "results" / "seed_rmse.json").read_text())
    seeds = [r["seed"] for r in blob["seeds"]]
    vals = [r["best_rmse_m"] for r in blob["seeds"]]
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    rerender = 0.0467

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(seeds))
    bars = ax.bar(
        x,
        vals,
        color=BLUE,
        alpha=0.75,
        width=0.55,
        zorder=2,
        yerr=None,
    )
    rng = np.random.default_rng(42)
    for i, v in enumerate(vals):
        ax.scatter(x[i] + rng.uniform(-0.08, 0.08), v, color=INK, s=36, zorder=4)
        ax.text(
            x[i],
            v + 0.0018,
            f"{v * 100:.2f} cm",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.axhline(mean, color=ORANGE, linewidth=1.6, linestyle="--", zorder=3, label=f"mean  {mean:.4f} m")
    ax.fill_between([-0.55, 4.55], mean - std, mean + std, color=ORANGE, alpha=0.12, zorder=1)
    ax.scatter(
        [0],
        [rerender],
        facecolors="none",
        edgecolors=INK,
        s=70,
        linewidths=1.4,
        zorder=5,
        label=f"re-render seed-1  {rerender:.4f} m",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Val RMSE (m)")
    ax.set_ylim(0, 0.072)
    ax.set_xlim(-0.55, 4.55)
    ax.yaxis.grid(True, linestyle="--", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("DINO RGB+D, 3 stereo pairs, trunk mask 0.5–10.0 m")
    seed_dot = plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor=INK, markersize=7, label="checkpoint best_rmse"
    )
    ax.legend(
        handles=[
            mpatches.Patch(color=BLUE, alpha=0.75, label="per-seed RMSE"),
            plt.Line2D([0], [0], color=ORANGE, linestyle="--", label=f"mean {mean:.4f} m"),
            seed_dot,
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor=INK,
                markersize=8,
                label=f"re-render seed-1  {rerender:.4f} m",
            ),
        ],
        frameon=False,
        loc="upper right",
        fontsize=8.5,
    )
    _save(fig, "accuracy_seeds.png")


def da2_vs_dino() -> None:
    # Paper numbers. DINO seeds from seed_rmse.json.
    blob = json.loads((REPO / "bench" / "results" / "seed_rmse.json").read_text())
    dino = [r["best_rmse_m"] for r in blob["seeds"]]
    labels = ["DA2-ft\nsingle view", "DINO RGB+D\n3 stereo pairs"]
    means = [0.0598, float(np.mean(dino))]
    stds = [0.0, float(np.std(dino, ddof=1))]
    colors = [ORANGE, BLUE]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(2)
    bars = ax.bar(
        x,
        means,
        yerr=[0, stds[1]],
        width=0.55,
        color=colors,
        alpha=0.75,
        capsize=6,
        error_kw=dict(elinewidth=1.4, ecolor="#222222"),
        zorder=2,
    )
    rng = np.random.default_rng(42)
    ax.scatter(np.zeros(1), [0.0598], color=INK, s=28, zorder=3)
    jitter = rng.uniform(-0.12, 0.12, size=len(dino))
    ax.scatter(np.ones(len(dino)) + jitter, dino, color=INK, s=28, zorder=3)
    for bar, s, lab in zip(
        bars,
        stds,
        ["5.98 cm", f"{means[1]*100:.2f}±{stds[1]*100:.2f} cm"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.002,
            lab,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Val RMSE (m)")
    ax.set_ylim(0, 0.085)
    ax.yaxis.grid(True, linestyle="--", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Same trees, same mask. Refinement is the 1.5 cm.")
    seed_dot = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=INK, markersize=6, label="Individual seed")
    fig.legend(
        handles=[
            mpatches.Patch(color=ORANGE, alpha=0.75, label="DA2-ft  (real-time cut)"),
            mpatches.Patch(color=BLUE, alpha=0.75, label="DINO 3-pair  (offline plan)"),
            seed_dot,
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    _save(fig, "da2_vs_dino.png")


def orchard_affine() -> None:
    """DA2-ft vs GT trunk mask after hold-out affine. Not the DINO 0.0445 m number."""
    labels = ["9-tree restore\nLS affine", "100 trees, 1 bark\nLS affine", "DINO 3-pair\n(different model)"]
    vals = [0.034366, 0.032517, 0.04452]
    colors = [ORANGE, BLUE, MUTED]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(3)
    bars = ax.bar(x, vals, width=0.55, color=colors, alpha=0.8, zorder=2)
    for bar, v, lab in zip(bars, vals, ["3.44 cm", "3.25 cm", "4.45 cm"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0012,
            lab,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hold-out RMSE (m)")
    ax.set_ylim(0, 0.058)
    ax.yaxis.grid(True, linestyle="--", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Same val trees (00042 + 00065). One bark is not 24k.")
    _save(fig, "orchard_affine.png")


def latency_quadro() -> None:
    path = REPO / "bench" / "results" / "quadro-rtx-8000_2026-08-18.json"
    rows = json.loads(path.read_text())
    labels = ["paper\nDGX fp32"]
    p50 = [484.7]
    p95 = [np.nan]
    p99 = [np.nan]
    for row in rows:
        labels.append(f"Quadro\n{row['precision']}")
        p50.append(row["p50_ms"])
        p95.append(row["p95_ms"])
        p99.append(row["p99_ms"])
    labels += ["V100\nfp32", "V100\nfp16"]
    p50 += [393.5, 156.0]
    p95 += [397.2, 156.2]
    p99 += [397.8, 156.3]

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, p50, width=w, color=BLUE, alpha=0.85, label="p50", zorder=2)
    ax.bar(x, p95, width=w, color=ORANGE, alpha=0.85, label="p95", zorder=2)
    ax.bar(x + w, p99, width=w, color=INK, alpha=0.55, label="p99", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("milliseconds  (6-view group)")
    ax.set_title("Same checkpoint. Each column names its GPU.")
    ax.yaxis.grid(True, linestyle="--", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _save(fig, "latency_p50.png")


def six_views() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.6), constrained_layout=True)
    titles = [
        "box  L",
        "box_cam1  L",
        "box_cam2  L",
        "box  R",
        "box_cam1  R",
        "box_cam2  R",
    ]
    sides = ["l", "l", "l", "r", "r", "r"]
    cams = ["box", "box_cam1", "box_cam2", "box", "box_cam1", "box_cam2"]
    for ax, title, side, cam in zip(axes.ravel(), titles, sides, cams):
        ax.imshow(_rgb(cam, side, max_w=640))
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(
        "One orbital position, three neighbouring rigs  —  the 6-view group",
        fontsize=12,
        fontweight="bold",
    )
    _save(fig, "six_views.png")


def onnx_split() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 3.15))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    def box(x, y, w, h, color, title, body):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.14, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.18, y + h - 0.32, title, fontsize=10, fontweight="bold", color=color, fontfamily="monospace")
        ax.text(x + 0.18, y + 0.28, body, fontsize=9, color=INK, va="bottom")

    box(0.2, 1.55, 2.5, 1.55, INK, "RGB + D", "6 × 280×512\nImageNet RGB\nbilinear depth")
    box(3.3, 1.55, 3.1, 1.55, BLUE, "encoder.onnx", "ViT-L per view\n1.2 GB graph\nmax |Δ| 1.53e-5")
    box(7.1, 1.55, 2.7, 1.55, ORANGE, "fuse_decode.onnx", "26 MB graph\nmax |Δ| 9.5e-7\nmetres out")
    box(10.4, 1.55, 1.55, 1.55, BLUE, "depth", "float32\n.npy")
    box(0.2, 0.15, 5.2, 1.15, ORANGE, "da2.onnx  (Engine A)", "single RGB  ·  518×924  ·  ~0.060 m  ·  POST /predict")
    for x0, x1 in ((2.7, 3.3), (6.4, 7.1), (9.8, 10.4)):
        ax.annotate("", xy=(x1, 2.3), xytext=(x0, 2.3), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.set_title("The Python loop over 6 views would unroll 144 ViT-L blocks. We split instead.")
    _save(fig, "onnx_split.png")


def two_regimes(rgb, da, mask) -> None:
    trunk = da[mask]
    vmin, vmax = np.percentile(trunk, [2, 98]) if trunk.size else (0.5, 6.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("POST /predict  ·  one RGB frame")
    axes[0].set_axis_off()
    _depth_show(axes[1], da, mask, "POST /predict  ·  DA2-ft metres on this frame", vmin, vmax)
    _save(fig, "two_regimes.png")


def copy_static() -> None:
    src = CV / "arch_dino_dsb.jpg"
    if src.is_file():
        Image.open(src).convert("RGB").save(OUT / "arch_dino.jpg", quality=90)
        print("wrote arch_dino.jpg", flush=True)
    abl = CV / "ablation_rmse.png"
    if abl.is_file():
        shutil.copy2(abl, OUT / "view_diversity_ablation.png")
        print("wrote view_diversity_ablation.png", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading val frame from Data/full_spur ...", flush=True)
    _style()
    rgb, da, gt, mask = _load_primary()
    hero_strip(rgb, da, gt, mask)
    da2_residual(da, gt, mask)
    nearest_vs_bilinear(gt, mask)
    accuracy_seeds()
    da2_vs_dino()
    orchard_affine()
    latency_quadro()
    six_views()
    onnx_split()
    two_regimes(rgb, da, mask)
    copy_static()
    print("wrote", sorted(p.name for p in OUT.iterdir() if p.is_file()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
