"""Zero-shot BranchNet on real orchard frames: MFO cherry labels and Envy_2 tree9.

BranchNet has only seen Blender renders of L-Py Envy trees. This script scores the
unchanged checkpoint on the pixel-labelled real cherry frames of the MFO dataset
(MFO dataset, CVPRW 2025; "Labelled Data/UFORozaVideos" on its Box share) and
draws a qualitative GIF on unlabelled real Envy frames (Envy_2 tree9, Azure
Kinect). The MFO data ships without a licence file, so every input and output
stays outside the repo (``--root``; writing inside the repo is refused).

Protocol, fixed before any prediction existed:
  data     the 386 current UFORozaVideos frames (``video_184/old/`` is a superseded
           copy and is skipped); splits by video id as shipped with the data:
           train 89-163, val 164-173, test 174-184
  labels   labelme polygons rasterised as labelme does (PIL polygon, outline and
           fill), later on top: nonbranch, other, leader, sidebranch, spur; leader ->
           trunk, sidebranch -> branch, spur -> spur, other -> ignored for the
           per-class scores, nonbranch and unlabelled -> background; a label outside
           these five names is treated as "other" and counted in the manifest
  input    RGB only: depth "none" (1/z = 0, valid = 0), one of the four inputs
           BranchNet trained with (10% of crops); the frames carry no metric depth
  predict  RGB upsampled bicubic (cv2.INTER_CUBIC) by f in {1, 1.6, 2.4}, full frame
           (``forward_full`` pads to the stride), softmax resized back to 640x480
           (bilinear, align_corners=False, no antialias), argmax; predicted shoot
           is scored as branch
  metrics  micro-averaged pixel counts per split: IoU of trunk, branch and spur
           with "other" pixels ignored, their mean (mIoU over the classes present),
           binary wood IoU against ``N_t.png`` (predicted wood = any non-background
           class), recall of labelled wood, and the raw confusion matrix of GT
           {background, leader, sidebranch, spur, other, nonbranch} x predicted
           {background, trunk, branch, shoot, spur}
  select   the factor with the highest val 3-class mIoU (ties: the smaller factor)
           is written to ``results/mfo_selection.json`` before any train or test
           frame is predicted; it is the headline, all factors are reported

python scripts/eval_real_mfo.py manifest --inventory labelled_inventory.json \
    --envy-lists envy_tree9_rgb.lst envy_tree9_depth.lst
python scripts/eval_real_mfo.py select        # val only, writes the selection
python scripts/eval_real_mfo.py score         # test then train; refuses without the selection
python scripts/eval_real_mfo.py panels        # figures/mfo_test_panels.png
python scripts/eval_real_mfo.py envy-predict  # GPU: Envy class maps
python scripts/eval_real_mfo.py envy --graph  # figures/envy_tree9_branchnet.gif (CPU)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from spur_depth.branches.assemble import AssembleConfig, assemble
from spur_depth.branches.tree import BRANCH, SHOOT, SPUR, TRUNK
from spur_depth.branches.viz import HEX, INK_2, MUTED, SURFACE, draw_graph_2d, label, save_gif

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/nfs/hpc/share/sanchej7/spur-realdata")
CKPT = Path("/nfs/hpc/share/sanchej7/spur-branch-runs/21523238/best.pt")
BOX_URL = "https://oregonstate.app.box.com/index.php?rm=box_download_shared_file&shared_name={share}&file_id={fid}"
CHERRY_SHARE = "krarnn9v3exzqg2f7w1gj6xhnfycww1r"
ENVY_SHARE = "0ulhgjkpk6y6yxsdbfmyeod9kfqafyd6"
ENVY_RECORDING_ID = "f_1808407794002"  # tree9 recording; its first 4 MiB hold the calibration
SPLITS = {"train": (89, 163), "val": (164, 173), "test": (174, 184)}
FACTORS = (1.0, 1.6, 2.4)
FRAME_KINDS = {"json": ".json", "png": ".png", "t": "_t.png"}
SKIPPED_KINDS = ("_d.png", "_f_0.png", "_f_1.png", "_helper.png")
LAYERS = ("nonbranch", "other", "leader", "sidebranch", "spur")  # later on top
GT_ROWS = ("background", "leader", "sidebranch", "spur", "other", "nonbranch")
PRED_COLS = ("background", "trunk", "branch", "shoot", "spur")
ROW = {name: i for i, name in enumerate(GT_ROWS)}
# scored class -> (GT row, predicted columns counted as that class)
SCORED = {
    "trunk": (ROW["leader"], (TRUNK,)),
    "branch": (ROW["sidebranch"], (BRANCH, SHOOT)),
    "spur": (ROW["spur"], (SPUR,)),
}
WOOD_ROWS = ("leader", "sidebranch", "spur", "other")
TRAIN_FX = 1493.3  # render focal length at 1920x1080 (README, intrinsics correction)
ASSEMBLE_CFG = {"orphan_cost": 1.2, "junction_bonus": 0.0, "depth_percentile": 50.0}
INTERP = {
    "upsample": "cv2.resize INTER_CUBIC on uint8 RGB",
    "downsample": "torch.nn.functional.interpolate(prob, mode='bilinear', align_corners=False, antialias=False)",
    "amp": "torch.autocast as in spur_depth.branches.infer.forward_full (fp16 on Turing)",
}


# ---------------------------------------------------------------- utilities


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _guard(path: Path) -> Path:
    """Refuse outputs inside the repo: the MFO data has no licence file."""
    path = Path(path).resolve()
    if path == REPO or REPO in path.parents:
        raise SystemExit(f"refusing to write MFO-derived files inside the repo: {path}")
    return path


def _write_json(path: Path, obj: dict) -> None:
    _guard(path).parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def git_state() -> dict:
    def run(*a: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *a], capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "tracked_files_modified": run("status", "--porcelain", "--untracked-files=no"),
        "script_status": run("status", "--porcelain", "--", "scripts/eval_real_mfo.py"),
    }


def split_of(video: int) -> str | None:
    for name, (lo, hi) in SPLITS.items():
        if lo <= video <= hi:
            return name
    return None


def _hex(code: str) -> np.ndarray:
    h = code.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


# ---------------------------------------------------------------- data


def build_manifest(data: Path, inventory: list) -> dict:
    """Frame list with Box file ids, sizes and sha256 of the files downloaded."""
    ids = {p: (fid, int(size)) for p, fid, size in inventory}
    stems = sorted(
        p[: -len(".json")]
        for p in ids
        if p.startswith("Labelled Data/UFORozaVideos/") and p.endswith(".json") and "/old/" not in p
    )
    frames, labels, shape_types = [], {}, {}
    t_iou = []
    same_image = 0
    for stem in stems:
        rel = stem[len("Labelled Data/") :]
        m = re.fullmatch(r"UFORozaVideos/video_(\d+)/(\d+)", rel)
        video, frame = int(m.group(1)), int(m.group(2))
        split = split_of(video)
        files = {}
        for kind, suffix in FRAME_KINDS.items():
            fid, size = ids[stem + suffix]
            path = data / (rel + suffix)
            if path.stat().st_size != size:
                raise SystemExit(f"size mismatch: {path}")
            files[kind] = {"path": rel + suffix, "file_id": fid, "size": size, "sha256": _sha(path)}
        doc = json.loads((data / (rel + ".json")).read_text())
        for s in doc["shapes"]:
            key = f"{split}:{s['label']}"
            labels[key] = labels.get(key, 0) + 1
            st = s.get("shape_type") or "polygon"
            shape_types[st] = shape_types.get(st, 0) + 1
        rgb = load_rgb(data / (rel + ".png"))
        emb = np.asarray(Image.open(io.BytesIO(base64.b64decode(doc["imageData"]))).convert("RGB"))
        same_image += int(emb.shape == rgb.shape and np.array_equal(emb, rgb))
        raw = rasterise(doc, *rgb.shape[:2])
        wood_t = load_wood(data / (rel + "_t.png"))
        poly = np.isin(raw, [ROW[n] for n in WOOD_ROWS])
        union = (poly | wood_t).sum()
        t_iou.append(float((poly & wood_t).sum() / union) if union else 1.0)
        frames.append(
            {
                "key": f"video_{video}/{frame}",
                "video": video,
                "frame": frame,
                "split": split,
                "files": files,
            }
        )
    counts = {s: sum(f["split"] == s for f in frames) for s in SPLITS}
    t_iou = np.asarray(t_iou)
    return {
        "dataset": "MFO cherry, Labelled Data/UFORozaVideos (Wang et al., CVPRW 2025)",
        "box_share": f"https://oregonstate.box.com/s/{CHERRY_SHARE}",
        "download_url": BOX_URL.format(share=CHERRY_SHARE, fid="<file_id>"),
        "licence": "none shipped with the data; research use, cite the paper",
        "split_rule": {k: f"video {lo}-{hi}" for k, (lo, hi) in SPLITS.items()},
        "frames_per_split": counts,
        "excluded": "UFORozaVideos/video_184/old/33.* (superseded copy of video_184/33)",
        "kinds_skipped": list(SKIPPED_KINDS),
        "polygon_counts_by_split_label": dict(sorted(labels.items())),
        "unknown_labels_as_other": sorted({k.split(":", 1)[1] for k in labels} - set(LAYERS)),
        "shape_types": shape_types,
        "checks": {
            "png_equals_embedded_imageData": f"{same_image}/{len(frames)}",
            "t_png_vs_polygon_wood_iou": {
                "min": float(t_iou.min()),
                "median": float(np.median(t_iou)),
                "mean": float(t_iou.mean()),
            },
        },
        "frames": frames,
    }


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_wood(path: Path) -> np.ndarray:
    """``N_t.png`` is a 3-channel black/white mask, white = tree wood."""
    t = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if t is None:
        raise FileNotFoundError(path)
    return (t if t.ndim == 2 else t[..., 0]) > 127


def rasterise(doc: dict, h: int, w: int) -> np.ndarray:
    """(H, W) uint8 index into GT_ROWS; polygons drawn layer by layer, later on top."""
    raw = np.zeros((h, w), np.uint8)
    for layer in LAYERS:
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for s in doc["shapes"]:
            name = s["label"] if s["label"] in LAYERS else "other"
            if name != layer:
                continue
            if (s.get("shape_type") or "polygon") != "polygon" or len(s["points"]) < 3:
                raise ValueError(f"unsupported shape {s.get('shape_type')} ({s['label']})")
            draw.polygon([(float(x), float(y)) for x, y in s["points"]], outline=1, fill=1)
        raw[np.asarray(mask, dtype=bool)] = ROW[layer]
    return raw


def load_frame(data: Path, entry: dict, verify: bool = True) -> dict:
    files = entry["files"]
    if verify:
        for f in files.values():
            if _sha(data / f["path"]) != f["sha256"]:
                raise SystemExit(f"sha256 mismatch against the manifest: {f['path']}")
    rgb = load_rgb(data / files["png"]["path"])
    doc = json.loads((data / files["json"]["path"]).read_text())
    return {
        "rgb": rgb,
        "raw": rasterise(doc, *rgb.shape[:2]),
        "wood": load_wood(data / files["t"]["path"]),
    }


# ---------------------------------------------------------------- network


class Net:
    """BranchNet on one GPU, RGB only; times the wall-clock while it holds the device."""

    def __init__(self, ckpt: Path, device: str) -> None:
        import torch

        from spur_depth.branches.infer import amp_dtype, load_model

        self.t0 = time.time()
        self.device = torch.device(device)
        self.model = load_model(ckpt, self.device)
        self.dtype = str(amp_dtype(self.device))
        self.forward_s = 0.0

    def prob(self, rgb: np.ndarray, factor: float, out_hw: tuple[int, int] | None = None):
        """(5, H, W) float32 softmax on the device at ``out_hw`` (default: the input size)."""
        import torch
        import torch.nn.functional as F

        from spur_depth.branches.infer import forward_full
        from spur_depth.branches.model import prepare_input

        h, w = rgb.shape[:2]
        out_hw = out_hw or (h, w)
        size = (round(w * factor), round(h * factor))
        x = rgb if size == (w, h) else cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC)
        t = time.time()
        seg, _ = forward_full(self.model, prepare_input(x, None)[None], self.device)
        prob = torch.softmax(seg, 1)
        if tuple(prob.shape[-2:]) != tuple(out_hw):
            prob = F.interpolate(prob, size=out_hw, mode="bilinear", align_corners=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.forward_s += time.time() - t
        return prob[0]

    def report(self) -> dict:
        import torch

        info = {"wall_s": round(time.time() - self.t0, 2), "forward_s": round(self.forward_s, 2)}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            info["wall_s"] = round(time.time() - self.t0, 2)
            props = torch.cuda.get_device_properties(self.device)
            info["device"] = torch.cuda.get_device_name(self.device)
            info["pci_bus_id"] = f"{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
            info["max_memory_gb"] = round(torch.cuda.max_memory_allocated(self.device) / 1e9, 2)
        info["amp_dtype"] = self.dtype
        info["torch"] = torch.__version__
        return info


def fkey(factor: float) -> str:
    return f"x{factor:g}"


def predict_split(net: Net, data: Path, entries: list[dict], pred_dir: Path) -> list[dict]:
    """Per frame and factor: the 6x5 confusion counts and the wood counts against N_t.png."""
    rows = []
    for k, e in enumerate(entries):
        fr = load_frame(data, e)
        row = {"key": e["key"], "video": e["video"], "frame": e["frame"], "factors": {}}
        for f in FACTORS:
            cls = net.prob(fr["rgb"], f).argmax(0).to("cpu").numpy().astype(np.uint8)
            out = pred_dir / fkey(f) / f"{e['key']}.cls.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out), cls):
                raise OSError(out)
            cm = np.bincount(
                fr["raw"].ravel().astype(np.int64) * len(PRED_COLS) + cls.ravel(),
                minlength=len(GT_ROWS) * len(PRED_COLS),
            ).reshape(len(GT_ROWS), len(PRED_COLS))
            wood_p = cls > 0
            wood = [
                int((wood_p & fr["wood"]).sum()),
                int((wood_p & ~fr["wood"]).sum()),
                int((~wood_p & fr["wood"]).sum()),
            ]
            row["factors"][fkey(f)] = {"cm": cm.tolist(), "wood_t": wood}
        rows.append(row)
        if (k + 1) % 25 == 0 or k + 1 == len(entries):
            print(f"  {k + 1}/{len(entries)} {e['key']}", flush=True)
    return rows


# ---------------------------------------------------------------- metrics


def metrics(cm: np.ndarray, wood: np.ndarray) -> dict:
    """Scores from summed counts (micro average); IoU is None for a class with no pixels."""
    cm = np.asarray(cm, dtype=np.int64)
    rows_scored = [r for r in range(len(GT_ROWS)) if r != ROW["other"]]
    out: dict = {}
    ious = {}
    for name, (row, cols) in SCORED.items():
        cols = list(cols)
        tp = int(cm[row, cols].sum())
        fn = int(cm[row].sum()) - tp
        fp = int(sum(cm[r, cols].sum() for r in rows_scored if r != row))
        den = tp + fp + fn
        ious[name] = tp / den if den else None
        out[name] = {
            "iou": ious[name],
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    present = [n for n, v in ious.items() if v is not None]
    out["miou3"] = float(np.mean([ious[n] for n in present])) if present else None
    out["miou3_classes"] = present
    tp, fp, fn = (int(v) for v in wood)
    out["wood_vs_t"] = {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    lab = cm[[ROW[n] for n in WOOD_ROWS]]
    scored = cm[[ROW[n] for n in ("leader", "sidebranch", "spur")]]
    out["labelled_wood_recall"] = float(lab[:, 1:].sum() / lab.sum()) if lab.sum() else None
    out["labelled_wood_recall_excl_other"] = (
        float(scored[:, 1:].sum() / scored.sum()) if scored.sum() else None
    )
    out["wood_recall_by_label"] = {
        n: float(cm[ROW[n], 1:].sum() / cm[ROW[n]].sum()) if cm[ROW[n]].sum() else None
        for n in GT_ROWS
        if n != "background"
    }
    out["background_predicted_wood_rate"] = (
        float(cm[ROW["background"], 1:].sum() / cm[ROW["background"]].sum())
        if cm[ROW["background"]].sum()
        else None
    )
    out["confusion"] = {
        "rows_gt": list(GT_ROWS),
        "cols_pred": list(PRED_COLS),
        "counts": cm.tolist(),
    }
    return out


def split_metrics(rows: list[dict]) -> dict:
    out = {}
    for f in FACTORS:
        k = fkey(f)
        cm = sum(np.asarray(r["factors"][k]["cm"], dtype=np.int64) for r in rows)
        wood = sum(np.asarray(r["factors"][k]["wood_t"], dtype=np.int64) for r in rows)
        out[k] = metrics(cm, wood)
    return out


def per_video(rows: list[dict], factor: float) -> dict:
    k = fkey(factor)
    out = {}
    for v in sorted({r["video"] for r in rows}):
        rs = [r for r in rows if r["video"] == v]
        cm = sum(np.asarray(r["factors"][k]["cm"], dtype=np.int64) for r in rs)
        wood = sum(np.asarray(r["factors"][k]["wood_t"], dtype=np.int64) for r in rs)
        m = metrics(cm, wood)
        out[f"video_{v}"] = {
            "frames": len(rs),
            "miou3": m["miou3"],
            "iou": {n: m[n]["iou"] for n in SCORED},
            "wood_iou_t": m["wood_vs_t"]["iou"],
            "labelled_wood_recall": m["labelled_wood_recall"],
        }
    return out


# ---------------------------------------------------------------- commands


def _paths(args) -> dict:
    root = _guard(args.root)
    return {
        "data": root / "mfo_cherry_ufo",
        "manifest": root / "mfo_cherry_ufo" / "manifest.json",
        "results": root / "results",
        "pred": root / "results" / "pred",
        "figures": root / "figures",
        "envy": root / "envy2_tree9",
        "selection": root / "results" / "mfo_selection.json",
        "summary": root / "results" / "mfo_zero_shot.json",
    }


def _entries(paths: dict, split: str, limit: int | None) -> list[dict]:
    man = json.loads(paths["manifest"].read_text())
    es = sorted(
        (e for e in man["frames"] if e["split"] == split), key=lambda e: (e["video"], e["frame"])
    )
    return es[:limit] if limit else es


def _provenance(args, paths: dict) -> dict:
    script = Path(__file__).resolve()
    return {
        "checkpoint": {"path": str(args.ckpt), "sha256": _sha(args.ckpt)},
        "data_manifest": {"path": str(paths["manifest"]), "sha256": _sha(paths["manifest"])},
        "script": {"path": str(script.relative_to(REPO)), "sha256": _sha(script)},
        "repo": git_state(),
        "argv": sys.argv,
        "interpolation": INTERP,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def cmd_manifest(args) -> int:
    paths = _paths(args)
    inventory = json.loads(Path(args.inventory).read_text())
    man = build_manifest(paths["data"], inventory)
    _write_json(paths["manifest"], man)
    print(json.dumps({k: v for k, v in man.items() if k != "frames"}, indent=2))
    if args.envy_lists:
        ids = {}
        for lst in args.envy_lists:
            for line in Path(lst).read_text().splitlines():
                t = line.split()
                if len(t) == 4 and t[0] == "file":
                    ids[t[1]] = (t[3], int(t[2]))
        files = []
        for p in sorted(paths["envy"].glob("*.png")):
            fid, size = ids[p.name]
            if p.stat().st_size != size:
                raise SystemExit(f"size mismatch: {p}")
            files.append({"path": p.name, "file_id": fid, "size": size, "sha256": _sha(p)})
        for name, note in (
            ("tree9_head.mkv", f"bytes 0-4194303 of {ENVY_RECORDING_ID} (tree9 recording)"),
            ("tree9_calibration.json", "K4A_CALIBRATION_FILE attachment of tree9_head.mkv"),
        ):
            p = paths["envy"] / name
            files.append(
                {"path": name, "source": note, "size": p.stat().st_size, "sha256": _sha(p)}
            )
        _write_json(
            paths["envy"] / "manifest.json",
            {
                "dataset": "MFO Envy_2, BeforePruning tree9 extracted frames (Azure Kinect)",
                "box_share": f"https://oregonstate.box.com/s/{ENVY_SHARE}",
                "download_url": BOX_URL.format(share=ENVY_SHARE, fid="<file_id>"),
                "recording_modes": {"color": "MJPG_1080P", "depth": "NFOV_UNBINNED"},
                "files": files,
            },
        )
        print("envy files:", len(files))
    return 0


def cmd_select(args) -> int:
    paths = _paths(args)
    if paths["selection"].exists():
        raise SystemExit(f"{paths['selection']} exists; the selection is written once")
    for split in ("test", "train"):
        for e in _entries(paths, split, None):
            if any((paths["pred"] / fkey(f) / f"{e['key']}.cls.png").exists() for f in FACTORS):
                raise SystemExit(f"{split} predictions exist already; selection must come first")
    entries = _entries(paths, "val", args.limit)
    net = Net(args.ckpt, args.device)
    print(f"val: {len(entries)} frames", flush=True)
    rows = predict_split(net, paths["data"], entries, paths["pred"])
    gpu = net.report()
    del net
    m = split_metrics(rows)
    miou = {k: m[k]["miou3"] for k in m}
    best = max(FACTORS, key=lambda f: (miou[fkey(f)], -f))
    _write_json(paths["results"] / "mfo_frames_val.json", {"split": "val", "frames": rows})
    sel = {
        "rule": (
            "factor with the highest val 3-class mIoU (trunk, branch incl. predicted shoot, "
            "spur; 'other' ignored; micro-averaged pixel counts); ties -> smaller factor"
        ),
        "factors": list(FACTORS),
        "val_miou3": miou,
        "chosen_factor": best,
        "val_frames": len(rows),
        "val_videos": sorted({r["video"] for r in rows}),
        "limit": args.limit,
        "written_at": _now(),
        "test_or_train_predicted_before": False,
        "gpu": gpu,
        "provenance": _provenance(args, paths),
    }
    _write_json(paths["selection"], sel)
    print(
        json.dumps(
            {k: sel[k] for k in ("val_miou3", "chosen_factor", "val_frames", "gpu")}, indent=2
        )
    )
    return 0


def cmd_score(args) -> int:
    paths = _paths(args)
    if not paths["selection"].exists():
        raise SystemExit("no selection: run `select` (val only) first")
    sel = json.loads(paths["selection"].read_text())
    prov = _provenance(args, paths)
    same_script = prov["script"]["sha256"] == sel["provenance"]["script"]["sha256"]
    if not same_script and not args.allow_script_change:
        raise SystemExit(
            "the script changed since the selection; pass --allow-script-change to record it"
        )
    net = Net(args.ckpt, args.device)
    rows = {}
    for split in args.splits:
        entries = _entries(paths, split, args.limit)
        print(f"{split}: {len(entries)} frames", flush=True)
        rows[split] = predict_split(net, paths["data"], entries, paths["pred"])
        _write_json(
            paths["results"] / f"mfo_frames_{split}.json", {"split": split, "frames": rows[split]}
        )
    gpu = net.report()
    del net
    rows["val"] = json.loads((paths["results"] / "mfo_frames_val.json").read_text())["frames"]
    best = sel["chosen_factor"]
    splits = {}
    for split in ("val", *args.splits):
        rs = rows[split]
        splits[split] = {
            "frames": len(rs),
            "videos": sorted({r["video"] for r in rs}),
            "factors": split_metrics(rs),
            "per_video_at_chosen_factor": per_video(rs, best),
        }
    head = {
        s: {
            "miou3": splits[s]["factors"][fkey(best)]["miou3"],
            "iou": {n: splits[s]["factors"][fkey(best)][n]["iou"] for n in SCORED},
            "wood_iou_t": splits[s]["factors"][fkey(best)]["wood_vs_t"]["iou"],
            "labelled_wood_recall": splits[s]["factors"][fkey(best)]["labelled_wood_recall"],
        }
        for s in splits
    }
    gpu_total = sel["gpu"]["wall_s"] + gpu["wall_s"]
    summary = {
        "what": "BranchNet (synthetic-only training) zero-shot on MFO cherry UFORozaVideos labels",
        "protocol": __doc__.split("Protocol, fixed before any prediction existed:")[1]
        .split("python scripts")[0]
        .strip(),
        "gt_mapping": {
            "leader": "trunk (1)",
            "sidebranch": "branch (2)",
            "spur": "spur (4)",
            "other": "ignored in per-class IoU; wood for N_t.png and labelled-wood recall",
            "nonbranch": "background (0)",
            "unlabelled": "background (0)",
            "rasterisation_order": list(LAYERS),
        },
        "depth_input": "none (1/z = 0, valid = 0)",
        "factors": list(FACTORS),
        "selection": {
            "file": str(paths["selection"]),
            "sha256": _sha(paths["selection"]),
            "chosen_factor": best,
            "val_miou3": sel["val_miou3"],
            "written_at": sel["written_at"],
            "script_unchanged_since_selection": same_script,
        },
        "headline": {"factor": best, **head},
        "splits": splits,
        "gpu": {
            "select_wall_s": sel["gpu"]["wall_s"],
            "score_wall_s": gpu["wall_s"],
            "minutes": round(gpu_total / 60.0, 2),
            "score": gpu,
        },
        "provenance": prov,
        "disclosures": args.disclosure or [],
        "written_at": _now(),
    }
    _write_json(paths["summary"], summary)
    print(
        json.dumps(
            {"headline": summary["headline"], "gpu_min": summary["gpu"]["minutes"]}, indent=2
        )
    )
    return 0


# ---------------------------------------------------------------- figures

PRED_COLORS = {TRUNK: HEX[TRUNK], BRANCH: HEX[BRANCH], SHOOT: HEX[BRANCH], SPUR: HEX[SPUR]}
GT_COLORS = {
    ROW["leader"]: HEX[TRUNK],
    ROW["sidebranch"]: HEX[BRANCH],
    ROW["spur"]: HEX[SPUR],
    ROW["other"]: MUTED,
}
RAW_COLORS = {TRUNK: HEX[TRUNK], BRANCH: HEX[BRANCH], SHOOT: HEX[SHOOT], SPUR: HEX[SPUR]}


def overlay(rgb: np.ndarray, cls: np.ndarray, colors: dict, alpha: float = 0.85) -> np.ndarray:
    """Dimmed RGB with the class pixels painted in the class colours."""
    out = rgb.astype(np.float32) * 0.5 + 255.0 * 0.5 * 0.15
    for c, code in colors.items():
        m = cls == c
        out[m] = (1.0 - alpha) * out[m] + alpha * _hex(code)
    return out.clip(0, 255).astype(np.uint8)


def legend(
    width: int, items: list[tuple[str, str]], note: str = "", height: int = 30
) -> np.ndarray:
    img = np.full((height, width, 3), _hex(SURFACE).astype(np.uint8), np.uint8)
    x = 12
    ink = tuple(int(v) for v in _hex(INK_2))
    for code, text in items:
        cv2.rectangle(
            img,
            (x, height // 2 - 6),
            (x + 18, height // 2 + 6),
            tuple(int(v) for v in _hex(code)),
            -1,
        )
        cv2.putText(
            img,
            text,
            (x + 24, height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            ink,
            1,
            cv2.LINE_AA,
        )
        x += 40 + 8 * len(text)
    if note:
        cv2.putText(
            img, note, (x, height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ink, 1, cv2.LINE_AA
        )
    return img


def cmd_panels(args) -> int:
    paths = _paths(args)
    if args.split != "val" and not paths["summary"].exists():
        raise SystemExit("figures come after scoring: run `score` first")
    sel = json.loads(paths["selection"].read_text())
    f = sel["chosen_factor"]
    entries = _entries(paths, args.split, None)
    # Fixed rule, not chosen by result: the first labelled frame of each video in the split.
    first = {}
    for e in entries:
        first.setdefault(e["video"], e)
    picks = list(first.values())[: args.max_rows]
    rows = []
    for e in picks:
        fr = load_frame(paths["data"], e)
        cls = cv2.imread(str(paths["pred"] / fkey(f) / f"{e['key']}.cls.png"), cv2.IMREAD_UNCHANGED)
        rows.append(
            np.concatenate(
                [
                    label(fr["rgb"], f"{e['key']} ({args.split})"),
                    label(overlay(fr["rgb"], cls, PRED_COLORS), f"BranchNet, RGB only, x{f:g}"),
                    label(overlay(fr["rgb"], fr["raw"], GT_COLORS), "MFO labels"),
                ],
                1,
            )
        )
    grid = np.concatenate(rows, 0)
    leg = legend(
        grid.shape[1],
        [
            (HEX[TRUNK], "trunk / leader"),
            (HEX[BRANCH], "branch (+ shoot) / sidebranch"),
            (HEX[SPUR], "spur"),
            (MUTED, "other (ignored)"),
        ],
        note="nonbranch (wires) and unlabelled = background",
    )
    grid = np.concatenate([grid, leg], 0)
    name = "mfo_test_panels.png" if args.split == "test" else f"mfo_{args.split}_panels.png"
    out = _guard(paths["figures"] / name)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out, optimize=True)
    _write_json(
        out.with_suffix(".json"),
        {
            "frames": [e["key"] for e in picks],
            "rule": "first labelled frame of each video in the split (not chosen by result)",
            "factor": f,
            "columns": ["RGB", "BranchNet argmax, shoot shown as branch (as scored)", "MFO labels"],
            "script_sha256": _sha(Path(__file__).resolve()),
            "written_at": _now(),
        },
    )
    print("wrote", out, grid.shape)
    return 0


def kinect_cameras(cal_path: Path) -> dict:
    """Pinhole K + OpenCV distortion for colour (MJPG_1080P) and depth (NFOV_UNBINNED).

    Raw Azure Kinect intrinsics are normalised by the sensor size; the per-mode
    conversion follows the Azure Kinect SDK: colour 4096x3072 is binned to
    1920x1440 and cropped by (0, 180) to 1920x1080; depth NFOV unbinned is the
    1024x1024 sensor cropped by (192, 180) to 640x576; pixel centres at integers.
    The extrinsic ``Rt`` of the colour camera maps depth-camera points to it.
    """
    cams = {
        c["Location"]: c
        for c in json.loads(Path(cal_path).read_text())["CalibrationInformation"]["Cameras"]
    }

    def conv(cam: dict, binned: tuple[int, int], crop: tuple[int, int]):
        p = cam["Intrinsics"]["ModelParameters"]
        fx, fy = p[2] * binned[0], p[3] * binned[1]
        cx, cy = p[0] * binned[0] - crop[0] - 0.5, p[1] * binned[1] - crop[1] - 0.5
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        # k4a order: cx cy fx fy k1..k6 codx cody p2 p1 -> OpenCV k1 k2 p1 p2 k3 k4 k5 k6
        dist = np.array([p[4], p[5], p[13], p[12], p[6], p[7], p[8], p[9]], dtype=np.float64)
        return K, dist

    Kc, dc = conv(cams["CALIBRATION_CameraLocationPV0"], (1920, 1440), (0, 180))
    Kd, dd = conv(cams["CALIBRATION_CameraLocationD0"], (1024, 1024), (192, 180))
    rt = cams["CALIBRATION_CameraLocationPV0"]["Rt"]
    R = np.asarray(rt["Rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(rt["Translation"], dtype=np.float64)
    return {"Kc": Kc, "dc": dc, "Kd": Kd, "dd": dd, "R": R, "t": t}


def register_depth(depth_mm: np.ndarray, cam: dict, shape=(1080, 1920), invert: bool = False):
    """Kinect depth (mm, depth camera) -> metric depth in the colour image, z-buffered."""
    v, u = np.nonzero(depth_mm > 0)
    z = depth_mm[v, u].astype(np.float64) / 1000.0
    pts = np.stack([u, v], 1).astype(np.float64)[:, None]
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)
    n = cv2.undistortPointsIter(pts, cam["Kd"], cam["dd"], None, None, crit)[:, 0]
    X = np.stack([n[:, 0] * z, n[:, 1] * z, z], 1)
    R, t = cam["R"], cam["t"]
    Xc = (X - t) @ R if invert else X @ R.T + t
    front = Xc[:, 2] > 0.1
    Xc = Xc[front]
    uv = cv2.projectPoints(Xc, np.zeros(3), np.zeros(3), cam["Kc"], cam["dc"])[0][:, 0]
    zbuf = np.full(shape[0] * shape[1], np.inf)
    # each depth pixel covers ~1.8 colour pixels (fx 505 vs 913): splat 2x2
    for dy in (0, 1):
        for dx in (0, 1):
            uu = np.floor(uv[:, 0]).astype(np.int64) + dx
            vv = np.floor(uv[:, 1]).astype(np.int64) + dy
            ok = (uu >= 0) & (uu < shape[1]) & (vv >= 0) & (vv < shape[0])
            np.minimum.at(zbuf, vv[ok] * shape[1] + uu[ok], Xc[ok, 2])
    zbuf[~np.isfinite(zbuf)] = 0.0
    return zbuf.reshape(shape).astype(np.float32)


def registration_check(rgb: np.ndarray, depth: np.ndarray, title: str) -> np.ndarray:
    """RGB with the registered depth's edges (jumps > 10 cm and valid borders) in magenta."""
    valid = depth > 0
    inv = np.where(valid, 1.0 / np.maximum(depth, 1e-3), 0.0).astype(np.float32)
    gx = np.abs(cv2.Sobel(inv, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(inv, cv2.CV_32F, 0, 1, ksize=3))
    edge = (gx + gy) > 0.08
    out = rgb.copy()
    out[edge] = (255, 0, 255)
    return label(out, title, size=1.0, y=40)


def _envy_setup(args) -> tuple[dict, dict, float, list[str], Path]:
    paths = _paths(args)
    cam = kinect_cameras(paths["envy"] / "tree9_calibration.json")
    factor = TRAIN_FX / float(cam["Kc"][0, 0])
    names = sorted(p.name for p in paths["envy"].glob("rgb*.png"))[: args.limit]
    return paths, cam, factor, names, _guard(paths["envy"] / "pred")


def cmd_envy_predict(args) -> int:
    """GPU step: class map and confidence per Envy frame, RGB only at the focal-ratio factor."""
    paths, cam, factor, names, pred = _envy_setup(args)
    pred.mkdir(parents=True, exist_ok=True)
    net = Net(args.ckpt, args.device)
    for name in names:
        rgb = load_rgb(paths["envy"] / name)
        conf, cls = (a.to("cpu").numpy() for a in net.prob(rgb, factor, rgb.shape[:2]).max(0))
        stem = name[: -len(".png")]
        for suffix, img in (
            (".cls.png", cls.astype(np.uint8)),
            (".conf.png", np.round(conf * 255.0).astype(np.uint8)),
        ):
            if not cv2.imwrite(str(pred / f"{stem}{suffix}"), img):
                raise OSError(pred / f"{stem}{suffix}")
        print(name, flush=True)
    gpu = net.report()
    del net
    _write_json(
        pred / "predict.json",
        {
            "frames": names,
            "input": "RGB only (depth none), full frame",
            "factor": factor,
            "gpu": gpu,
            "checkpoint_sha256": _sha(args.ckpt),
            "script_sha256": _sha(Path(__file__).resolve()),
            "written_at": _now(),
        },
    )
    print(json.dumps(gpu))
    return 0


def cmd_envy(args) -> int:
    """CPU step: RGB | classes (| graph from registered Kinect depth) as a GIF."""
    paths, cam, factor, names, pred = _envy_setup(args)
    out_dir = paths["figures"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.check_registration:
        rgb = load_rgb(paths["envy"] / names[0])
        d = cv2.imread(str(paths["envy"] / names[0].replace("rgb", "depth")), cv2.IMREAD_UNCHANGED)
        tiles = []
        for inv in (False, True):
            reg = register_depth(d, cam, rgb.shape[:2], invert=inv)
            tiles.append(
                registration_check(rgb, reg, f"{names[0]} Rt {'inverted' if inv else 'as given'}")
            )
        y0, y1, x0, x1 = args.check_crop
        img = np.concatenate([t[y0:y1, x0:x1] for t in tiles], 1)
        out = _guard(out_dir / "envy_tree9_registration_check.png")
        Image.fromarray(img).save(out)
        print("wrote", out)
        return 0
    info = json.loads((pred / "predict.json").read_text())
    cfg = AssembleConfig(**ASSEMBLE_CFG)
    s = args.scale
    frames, record = [], []
    for name in names:
        stem = name[: -len(".png")]
        rgb = load_rgb(paths["envy"] / name)
        h, w = rgb.shape[:2]
        cls = cv2.imread(str(pred / f"{stem}.cls.png"), cv2.IMREAD_UNCHANGED)
        conf = cv2.imread(str(pred / f"{stem}.conf.png"), cv2.IMREAD_UNCHANGED) / 255.0
        small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ov = cv2.resize(
            overlay(rgb, cls, RAW_COLORS), None, fx=s, fy=s, interpolation=cv2.INTER_AREA
        )
        panels = [
            label(small, f"Envy_2 tree9 frame {stem[3:]} (real, Azure Kinect)"),
            label(ov, f"BranchNet, RGB only, x{factor:.2f}"),
        ]
        rec = {"frame": name, "class_pixels": np.bincount(cls.ravel(), minlength=5).tolist()}
        if args.graph:
            d = cv2.imread(str(paths["envy"] / name.replace("rgb", "depth")), cv2.IMREAD_UNCHANGED)
            reg = register_depth(d, cam, (h, w), invert=args.invert_rt)
            cls_g = cls
            if args.graph_max_depth:
                near = ((reg > 0) & (reg < args.graph_max_depth)).astype(np.uint8)
                k = 2 * args.graph_dilate + 1
                near = cv2.dilate(near, np.ones((k, k), np.uint8)) > 0
                cls_g = np.where(near, cls, 0).astype(np.uint8)
            t0 = time.time()
            g = assemble(cls_g, reg, cam["Kc"], conf=conf.astype(np.float32), cfg=cfg)
            rec["assemble_s"] = round(time.time() - t0, 1)
            panels.append(
                label(draw_graph_2d(rgb, g, cam["Kc"], scale=s), "graph, registered Kinect depth")
            )
            rec["parts"] = len(g.parts)
            rec["registered_valid_frac"] = float((reg > 0).mean())
        row = np.concatenate(panels, 1)
        items = [(HEX[TRUNK], "trunk"), (HEX[BRANCH], "branch"), (HEX[SPUR], "shoot / spur")]
        note = "white tick: cut point and axis" if args.graph else "real frames: qualitative only"
        frames.append(np.concatenate([row, legend(row.shape[1], items, note=note)], 0))
        record.append(rec)
        print(name, rec, flush=True)
    out = _guard(out_dir / "envy_tree9_branchnet.gif")
    save_gif(frames, out, duration=900, colors=args.colors)
    meta = {
        "frames": record,
        "input": info["input"],
        "factor": factor,
        "factor_rule": (
            f"render focal {TRAIN_FX} px / Kinect colour focal {cam['Kc'][0, 0]:.2f} px at 1080p "
            "(tree9_calibration.json): branches appear at training scale; fixed by the focal "
            "ratio, not tuned"
        ),
        "colour_K": cam["Kc"].tolist(),
        "depth_K": cam["Kd"].tolist(),
        "graph": bool(args.graph),
        "assemble_config": ASSEMBLE_CFG if args.graph else None,
        "graph_max_depth_m": args.graph_max_depth if args.graph else None,
        "graph_dilate_px": args.graph_dilate if args.graph and args.graph_max_depth else None,
        "rt_inverted": bool(args.invert_rt) if args.graph else None,
        "scale": s,
        "gif_bytes": out.stat().st_size,
        "gpu_predict": info["gpu"],
        "checkpoint_sha256": _sha(args.ckpt),
        "script_sha256": _sha(Path(__file__).resolve()),
        "written_at": _now(),
    }
    _write_json(out.with_suffix(".json"), meta)
    print("wrote", out, meta["gif_bytes"], "bytes")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--device", default="cuda")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--inventory", required=True, help="JSON list of [box path, file id, size]")
    m.add_argument("--envy-lists", nargs="*", default=[])
    s = sub.add_parser("select")
    s.add_argument("--limit", type=int, default=None, help="smoke tests only")
    c = sub.add_parser("score")
    c.add_argument("--splits", nargs="+", default=["test", "train"])
    c.add_argument("--limit", type=int, default=None, help="smoke tests only")
    c.add_argument("--allow-script-change", action="store_true")
    c.add_argument("--disclosure", action="append")
    p = sub.add_parser("panels")
    p.add_argument("--split", default="test")
    p.add_argument("--max-rows", type=int, default=6)
    ep = sub.add_parser("envy-predict")
    ep.add_argument("--limit", type=int, default=None)
    e = sub.add_parser("envy")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--graph-max-depth", type=float, default=None)
    e.add_argument("--graph-dilate", type=int, default=7)
    e.add_argument("--scale", type=float, default=0.4)
    e.add_argument("--colors", type=int, default=128)
    e.add_argument("--graph", action="store_true", help="registered Kinect depth + assembly")
    e.add_argument("--invert-rt", action="store_true")
    e.add_argument("--check-registration", action="store_true")
    e.add_argument("--check-crop", type=int, nargs=4, default=(0, 1080, 0, 1920))
    args = ap.parse_args(argv)
    cmds = {
        "manifest": cmd_manifest,
        "select": cmd_select,
        "score": cmd_score,
        "panels": cmd_panels,
        "envy-predict": cmd_envy_predict,
        "envy": cmd_envy,
    }
    return cmds[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
