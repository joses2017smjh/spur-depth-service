"""Unsupervised adaptation to real frames (protocol v5): EMA-teacher self-training, optional MIC.

Unlabelled real RGB frames (no metric depth: the network's "none" depth input) are labelled
on the fly by an exponential-moving-average teacher; the student learns from the confident
pseudo-labels next to the synthetic supervised loss.

  selftrain  student sees a photometrically strong view of the same crop the teacher saw;
             each image's pseudo-label loss is weighted by the share of pixels whose
             teacher confidence exceeds ``tau`` (DACS, Tranheden et al. 2021)
  mic        selftrain + ClassMix of synthetic and real crops (half of a synthetic crop's
             classes pasted onto a real crop, DACS) + masked image consistency: the student
             sees the real crop with random patches masked and must reproduce the teacher's
             pseudo-labels (MIC, Hoyer et al. 2023, the method behind MFO's adapted model)

Selection uses MFO val (``mfo_val_score``): 3-class mIoU under the UFO class mapping.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from spur_depth.branches.dataset import colour_jitter
from spur_depth.branches.model import prepare_input
from spur_depth.branches.tree import BRANCH, IGNORE, SHOOT, SPUR, TRUNK

UDA_MODES = ("none", "selftrain", "mic")


class TargetFrames(Dataset):
    """Random ``crop`` x ``crop`` windows of real frames resized by ``factor``: weak + strong views."""

    def __init__(self, paths: list[Path], factor: float, crop: int, seed: int = 0) -> None:
        self.paths = [Path(p) for p in paths]
        self.factor, self.crop, self.seed = factor, crop, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng((self.seed, self.epoch, index))
        bgr = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"unreadable target frame {self.paths[index]}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        size = (round(w * self.factor), round(h * self.factor))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC)
        c = self.crop
        if rgb.shape[0] < c or rgb.shape[1] < c:
            rgb = cv2.copyMakeBorder(
                rgb, 0, max(0, c - rgb.shape[0]), 0, max(0, c - rgb.shape[1]), cv2.BORDER_REFLECT
            )
        y = int(rng.integers(0, rgb.shape[0] - c + 1))
        x = int(rng.integers(0, rgb.shape[1] - c + 1))
        crop = np.ascontiguousarray(rgb[y : y + c, x : x + c])
        if rng.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1])
        strong = colour_jitter(crop, rng, bcs=0.4, hue=0.08)
        if rng.random() < 0.5:
            k = int(rng.choice([3, 5]))
            strong = cv2.GaussianBlur(strong, (k, k), 0)
        return {
            "weak": prepare_input(crop, None),
            "strong": prepare_input(np.ascontiguousarray(strong), None),
        }


class EMATeacher:
    """Frozen copy of the student updated as ``t = a t + (1 - a) s`` after every step."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: torch.nn.Module, alpha: float) -> None:
        for t, s in zip(self.model.parameters(), student.parameters()):
            t.mul_(alpha).add_(s.detach(), alpha=1.0 - alpha)
        for t, s in zip(self.model.buffers(), student.buffers()):
            t.copy_(s)

    @torch.no_grad()
    def pseudo_labels(self, x: torch.Tensor, tau: float, amp_dtype=None):
        """(labels (B,H,W) int64, per-image weight (B,) = share of pixels above ``tau``)."""
        with torch.autocast(x.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            seg = self.model(x)["seg"].float()
        prob = torch.softmax(seg, 1)
        conf, lab = prob.max(1)
        return lab, (conf > tau).float().mean((1, 2))


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Pixel cross-entropy (IGNORE skipped) times a per-pixel weight, averaged over pixels."""
    ce = F.cross_entropy(logits.float(), labels, ignore_index=IGNORE, reduction="none")
    valid = (labels != IGNORE).float()
    return (ce * weight * valid).sum() / valid.sum().clamp_min(1.0)


def classmix(xs, ys, xt, yt, wt, generator: random.Random):
    """Paste half of each synthetic crop's classes onto a real crop (DACS ClassMix).

    Returns the mixed input, labels and per-pixel weights (1 on pasted synthetic pixels,
    the real crop's pseudo-label weight elsewhere).
    """
    b = xt.shape[0]
    xm, ym, wm = xt.clone(), yt.clone(), wt[:, None, None].expand_as(yt).clone()
    for i in range(b):
        present = [int(c) for c in torch.unique(ys[i]).tolist() if c != IGNORE]
        if not present:
            continue
        chosen = generator.sample(present, max(1, len(present) // 2))
        m = torch.isin(ys[i], torch.tensor(chosen, device=ys.device))
        xm[i] = torch.where(m[None], xs[i], xt[i])
        ym[i] = torch.where(m, ys[i], yt[i])
        wm[i] = torch.where(m, torch.ones_like(wm[i]), wm[i])
    return xm, ym, wm


def mic_mask(x: torch.Tensor, patch: int, ratio: float, generator: torch.Generator) -> torch.Tensor:
    """Zero out (= dataset mean after normalisation) a random ``ratio`` of ``patch``-px patches."""
    b, _, h, w = x.shape
    gh, gw = -(-h // patch), -(-w // patch)
    keep = (torch.rand(b, 1, gh, gw, generator=generator, device="cpu") > ratio).float()
    keep = F.interpolate(keep, scale_factor=patch, mode="nearest")[:, :, :h, :w].to(x.device)
    return x * keep


# ---------------------------------------------------------------- MFO val selection

# MFO GT rows: background, leader, sidebranch, spur, other, nonbranch. UFO mapping: leader =
# predicted trunk or branch, sidebranch = shoot, spur = spur; "other" ignored.
UFO_MAPPING = {"leader": (1, (TRUNK, BRANCH)), "sidebranch": (2, (SHOOT,)), "spur": (3, (SPUR,))}
OTHER_ROW = 4


def load_mfo_split(root: Path, split: str) -> list[dict]:
    """[{rgb, raw}] for one MFO split (labels rasterised as scripts/eval_real_mfo.py does)."""
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "eval_real_mfo", Path(__file__).resolve().parents[2] / "scripts" / "eval_real_mfo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = Path(root) / "mfo_cherry_ufo"
    man = json.loads((data / "manifest.json").read_text())
    entries = sorted(
        (e for e in man["frames"] if e["split"] == split), key=lambda e: (e["video"], e["frame"])
    )
    out = []
    for e in entries:
        fr = mod.load_frame(data, e, verify=False)
        out.append({"rgb": fr["rgb"], "raw": fr["raw"]})
    return out


@torch.no_grad()
def mfo_val_score(model, frames: list[dict], factor: float, device, amp: bool = True) -> dict:
    """3-class mIoU under the UFO mapping, micro-averaged over ``frames`` (RGB only)."""
    from spur_depth.branches.infer import forward_full

    model.eval()
    cm = np.zeros((6, 5), dtype=np.int64)
    for fr in frames:
        rgb, raw = fr["rgb"], fr["raw"]
        h, w = rgb.shape[:2]
        size = (round(w * factor), round(h * factor))
        x = rgb if size == (w, h) else cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC)
        seg, _ = forward_full(model, prepare_input(x, None)[None], device, amp)
        prob = F.interpolate(torch.softmax(seg.float(), 1), size=(h, w), mode="bilinear")
        pred = prob[0].argmax(0).cpu().numpy()
        cm += np.bincount(raw.astype(np.int64) * 5 + pred, minlength=30).reshape(6, 5)
    model.train()
    rows = [r for r in range(6) if r != OTHER_ROW]
    ious = {}
    for name, (row, cols) in UFO_MAPPING.items():
        cols = list(cols)
        tp = int(cm[row, cols].sum())
        fn = int(cm[row].sum()) - tp
        fp = int(sum(cm[r, cols].sum() for r in rows if r != row))
        ious[name] = tp / (tp + fp + fn) if tp + fp + fn else None
    vals = [v for v in ious.values() if v is not None]
    return {
        "miou3_ufo": float(np.mean(vals)) if vals else None,
        "iou": ious,
        "n_frames": len(frames),
    }
