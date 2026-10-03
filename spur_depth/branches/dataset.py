"""Training crops and whole frames from the branch label cache (v1).

Labels, GT depth and DA2-ft depth come from the cache
(``{root}/{tree}/{rig}/{stem}.{cls,skel,depth,da2}.png``); RGB is read from the
dataset itself. Each training crop draws its depth input from the sources the
model meets at test time - clean GT, a simulated D435 (``depth_noise``), DA2-ft
and none - so one checkpoint serves all of them and degrades gracefully when
depth is missing.

Randomness: every crop draws from ``default_rng((seed, epoch, index))``, so a
run is reproducible whatever the worker count; ``worker_init_fn`` pins thread
pools and seeds the legacy global RNGs.
"""

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from spur_depth.branches import gt
from spur_depth.branches.depth_noise import frame_seed, simulate_sensor
from spur_depth.branches.model import prepare_input

# A frame is usable when its labels and GT depth exist and the builder has written
# meta.json (its last write, so it marks a finished frame). DA2 is optional: the
# builder skips it when the DA2 .npy is missing, and a crop that draws "da2" for
# such a frame falls back to the simulated sensor. Depth PNGs are uint16 mm with
# 0 = invalid and 65535 = at or beyond 65.535 m, which stays a valid far reading.
REQUIRED_SUFFIXES = (".cls.png", ".skel.png", ".depth.png", ".meta.json")
DEPTH_SOURCES = ("gt", "sensor", "da2", "none")
DEPTH_PROBS = (0.15, 0.45, 0.30, 0.10)


def cache_path(cache_root: Path, ref: gt.FrameRef, suffix: str) -> Path:
    return Path(cache_root) / ref.tree / ref.rig / f"{ref.stem}{suffix}"


def cached_frames(cache_root: Path, trees) -> list[gt.FrameRef]:
    """Frames of ``trees`` with complete cache entries (one listdir per rig, no per-file stat)."""
    root = Path(cache_root)
    out: list[gt.FrameRef] = []
    for tree in trees:
        tdir = root / tree
        if not tdir.is_dir():
            continue
        prefix = f"{tree}_"
        for rig in sorted(p.name for p in tdir.iterdir() if p.is_dir()):
            names = set(os.listdir(tdir / rig))
            stems = sorted(n[: -len(".cls.png")] for n in names if n.endswith(".cls.png"))
            for stem in stems:
                if not stem.startswith(prefix):
                    continue
                if not all(stem + s in names for s in REQUIRED_SUFFIXES):
                    continue
                # Tree ids contain underscores: strip the tree, then split shot/side once.
                shot, _, side = stem[len(prefix) :].rpartition("_")
                if shot and side:
                    out.append(gt.FrameRef(tree, rig, shot, side))
    return out


def split_frames(
    cache_root: Path, data_root: Path = gt.DATA_ROOT
) -> tuple[list[gt.FrameRef], list[gt.FrameRef], list[gt.FrameRef]]:
    """(train, val, test) cached frames: test = PAPER_TEST, val = VAL_TREES, train = the rest."""
    held = set(gt.PAPER_TEST) | set(gt.VAL_TREES)
    train_trees = [t for t in gt.all_trees(data_root) if t not in held]
    return (
        cached_frames(cache_root, train_trees),
        cached_frames(cache_root, gt.VAL_TREES),
        cached_frames(cache_root, gt.PAPER_TEST),
    )


def read_png(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"missing or unreadable: {path}")
    return img


def read_depth_m(path: Path) -> np.ndarray:
    """uint16 millimetre PNG -> float32 metres, 0 = invalid."""
    mm = read_png(path)
    if mm.dtype != np.uint16:
        raise ValueError(f"{path}: expected uint16 depth, got {mm.dtype}")
    return mm.astype(np.float32) * np.float32(0.001)


def read_rgb(ref: gt.FrameRef, data_root: Path = gt.DATA_ROOT) -> np.ndarray:
    """(H, W, 3) uint8 RGB.

    cv2 rather than PIL: the PNG decode is the largest per-crop cost (104 vs 158 ms at
    1920x1080 on one core) and the pixels are identical to ``Image.convert("RGB")``
    for these 8-bit RGB renders (checked on 5 frames across trees and rigs).
    """
    path = ref.path("Optical_flow", data_root)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"missing or unreadable: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_cached_frame(
    ref: gt.FrameRef,
    cache_root: Path,
    data_root: Path = gt.DATA_ROOT,
    skel: bool = True,
    da2: bool = True,
) -> dict:
    """Whole frame: rgb (H,W,3) uint8, cls/skel (H,W) uint8, depth/da2 (H,W) float32 m.

    ``da2`` is None when the cache has no DA2 prediction for the frame.
    """
    out = {
        "rgb": read_rgb(ref, data_root),
        "cls": read_png(cache_path(cache_root, ref, ".cls.png")),
        "depth": read_depth_m(cache_path(cache_root, ref, ".depth.png")),
        "skel": read_png(cache_path(cache_root, ref, ".skel.png")) if skel else None,
        "da2": None,
    }
    p = cache_path(cache_root, ref, ".da2.png")
    if da2 and p.is_file():
        out["da2"] = read_depth_m(p)
    if out["rgb"].shape[:2] != out["cls"].shape:
        raise ValueError(f"{ref.key}: rgb {out['rgb'].shape} vs labels {out['cls'].shape}")
    return out


def sensor_depth(depth_m: np.ndarray, cls: np.ndarray, seed: int) -> np.ndarray:
    """Simulated D435 depth; the tree mask is every non-background label, including IGNORE."""
    return simulate_sensor(depth_m, cls > 0, seed)


def frame_depth(source: str, frame: dict, ref: gt.FrameRef) -> np.ndarray | None:
    """Full-frame depth input for one evaluation source; 'sensor' is seeded by the frame key."""
    if source == "gt":
        return frame["depth"]
    if source == "sensor":
        return sensor_depth(frame["depth"], frame["cls"], frame_seed(ref.key))
    if source == "da2":
        return frame["da2"]
    if source == "none":
        return np.zeros_like(frame["depth"])
    raise ValueError(f"unknown depth source {source!r}")


def worker_init_fn(worker_id: int) -> None:
    """One thread per loader worker: N workers x default cv2/torch pools oversubscribe the CPUs."""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def colour_jitter(
    rgb: np.ndarray, rng: np.random.Generator, bcs: float = 0.2, hue: float = 0.03
) -> np.ndarray:
    """Brightness, contrast, saturation in [1-bcs, 1+bcs] and hue shift in [-hue, hue] turns."""
    luma = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    img = rgb.astype(np.float32) * np.float32(rng.uniform(1 - bcs, 1 + bcs))
    mean = float((img @ luma).mean())
    img = (img - mean) * np.float32(rng.uniform(1 - bcs, 1 + bcs)) + mean
    grey = (img @ luma)[..., None]
    img = (img - grey) * np.float32(rng.uniform(1 - bcs, 1 + bcs)) + grey
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    shift = int(round(rng.uniform(-hue, hue) * 180.0))  # OpenCV hue spans 0..179
    if shift:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 180).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img


class BranchCrops(Dataset):
    """One random crop per frame per epoch: dict(x (5,h,w), cls, skel, ctr, src).

    ``cls`` int64 with IGNORE, ``skel`` uint8 class-coded 1-px axes, ``ctr`` float32
    binary axes dilated by 1 px, ``src`` index into ``DEPTH_SOURCES``.
    """

    def __init__(
        self,
        refs: list[gt.FrameRef],
        cache_root: Path,
        crop: int = 512,
        p_center: float = 0.8,
        depth_probs: tuple[float, ...] = DEPTH_PROBS,
        p_flip: float = 0.5,
        p_blur: float = 0.15,
        seed: int = 0,
        data_root: Path = gt.DATA_ROOT,
    ) -> None:
        if not refs:
            raise ValueError("no frames")
        if len(depth_probs) != len(DEPTH_SOURCES) or abs(sum(depth_probs) - 1.0) > 1e-6:
            raise ValueError(f"depth_probs must be {len(DEPTH_SOURCES)} probabilities")
        self.refs = list(refs)
        self.cache_root = Path(cache_root)
        self.data_root = Path(data_root)
        self.crop = int(crop)
        self.p_center = p_center
        self.depth_probs = np.asarray(depth_probs, dtype=np.float64)
        self.p_flip, self.p_blur = p_flip, p_blur
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call before each DataLoader iterator is created (workers get a copy of the dataset)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng((self.seed, self.epoch, int(index)))
        for attempt in range(5):
            ref = self.refs[index]
            try:
                return self._crop(ref, rng)
            except (OSError, ValueError) as e:
                # One bad file must not kill a 75-minute job: warn and draw another frame.
                if attempt == 4:
                    raise
                warnings.warn(f"{ref.key}: {e}; drawing another frame", RuntimeWarning)
                index = int(rng.integers(len(self.refs)))
        raise AssertionError("unreachable")

    def _crop(self, ref: gt.FrameRef, rng: np.random.Generator) -> dict:
        cls = read_png(cache_path(self.cache_root, ref, ".cls.png"))
        skel = read_png(cache_path(self.cache_root, ref, ".skel.png"))
        h, w = cls.shape
        if skel.shape != (h, w):
            raise ValueError(f"skel {skel.shape} vs cls {cls.shape}")
        ch, cw = min(self.crop, h), min(self.crop, w)
        tree_px = np.flatnonzero((cls >= 1) & (cls <= 4)) if rng.random() < self.p_center else []
        if len(tree_px):
            cy, cx = divmod(int(tree_px[rng.integers(len(tree_px))]), w)
            y0 = int(np.clip(cy - ch // 2, 0, h - ch))
            x0 = int(np.clip(cx - cw // 2, 0, w - cw))
        else:
            y0, x0 = int(rng.integers(h - ch + 1)), int(rng.integers(w - cw + 1))
        win = (slice(y0, y0 + ch), slice(x0, x0 + cw))

        rgb = read_rgb(ref, self.data_root)
        if rgb.shape[:2] != (h, w):
            raise ValueError(f"rgb {rgb.shape} vs labels {cls.shape}")
        rgb, cls, skel = rgb[win], cls[win], skel[win]

        src = int(rng.choice(len(DEPTH_SOURCES), p=self.depth_probs))
        sensor_seed = int(rng.integers(2**31))  # drawn always, so the stream does not shift
        depth = None
        if DEPTH_SOURCES[src] == "da2":
            p = cache_path(self.cache_root, ref, ".da2.png")
            if p.is_file():
                depth = read_depth_m(p)[win]
            else:
                src = DEPTH_SOURCES.index("sensor")
        if DEPTH_SOURCES[src] in ("gt", "sensor"):
            depth = read_depth_m(cache_path(self.cache_root, ref, ".depth.png"))[win]
            if DEPTH_SOURCES[src] == "sensor":
                depth = sensor_depth(depth, cls, sensor_seed)

        if rng.random() < self.p_flip:
            rgb, cls, skel = rgb[:, ::-1], cls[:, ::-1], skel[:, ::-1]
            depth = None if depth is None else depth[:, ::-1]
        rgb = colour_jitter(np.ascontiguousarray(rgb), rng)
        if rng.random() < self.p_blur:
            rgb = cv2.GaussianBlur(rgb, (0, 0), float(rng.uniform(0.3, 1.0)))

        cls = np.ascontiguousarray(cls)
        skel = np.ascontiguousarray(skel)
        ctr = cv2.dilate((skel > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
        return {
            "x": prepare_input(rgb, None if depth is None else np.ascontiguousarray(depth)),
            "cls": torch.from_numpy(cls.astype(np.int64)),
            "skel": torch.from_numpy(skel),
            "ctr": torch.from_numpy(ctr.astype(np.float32)),
            "src": src,
        }


class FullFrames(Dataset):
    """Whole cached frames plus the depth input for each requested source.

    Exists so DataLoader workers do the PNG decoding and the full-frame sensor
    simulation (together ~0.5-0.7 s of CPU per frame) in parallel for validation
    and prediction.
    """

    def __init__(
        self,
        refs: list[gt.FrameRef],
        cache_root: Path,
        sources: tuple[str, ...],
        data_root: Path = gt.DATA_ROOT,
        labels: bool = True,
    ) -> None:
        unknown = set(sources) - set(DEPTH_SOURCES)
        if unknown:
            raise ValueError(f"unknown depth sources {sorted(unknown)}")
        self.refs = list(refs)
        self.cache_root = Path(cache_root)
        self.sources = tuple(sources)
        self.data_root = Path(data_root)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict:
        """Frame dict, or ``{"index", "error"}`` for an unreadable frame (callers skip it)."""
        ref = self.refs[index]
        try:
            frame = load_cached_frame(
                ref, self.cache_root, self.data_root, skel=self.labels, da2="da2" in self.sources
            )
        except (OSError, ValueError) as e:
            return {"index": int(index), "error": str(e)}
        return {
            "index": int(index),
            "rgb": frame["rgb"],
            "cls": frame["cls"] if self.labels else None,
            "skel": frame["skel"],
            "depth": {s: frame_depth(s, frame, ref) for s in self.sources},
        }


def identity_collate(item):
    """``batch_size=None`` loaders: keep numpy arrays as they are."""
    return item
