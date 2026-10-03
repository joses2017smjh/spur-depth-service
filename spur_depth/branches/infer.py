"""Full-frame BranchNet inference and the on-disk prediction format.

Per frame ``save_prediction`` writes ``{stem}.cls.png`` (argmax), ``{stem}.conf.png``
(max probability x 255), ``{stem}.ctr.png`` (centerline probability x 255) and
with ``save_prob`` ``{stem}.prob.npz`` (``prob``: (5, H, W) float16 softmax), so
graph assembly can re-threshold without re-running the network.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from spur_depth.branches.model import BranchNet, pad_to_multiple, prepare_input, unpad


def amp_dtype(device: torch.device | str) -> torch.dtype | None:
    """bf16 on GPUs with native bf16 (sm80+: A40, H100), fp16 otherwise (RTX 8000), None on CPU.

    ``torch.cuda.is_bf16_supported()`` counts emulation by default and says True on
    Turing, where bf16 convolutions then run emulated; ask for hardware support only.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return None
    with torch.cuda.device(device):
        native = torch.cuda.is_bf16_supported(including_emulation=False)
    return torch.bfloat16 if native else torch.float16


def load_model(ckpt_path: Path | str, device: torch.device | str = "cpu") -> BranchNet:
    """BranchNet from a ``train_branches.py`` checkpoint, in eval mode on ``device``."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = dict(blob.get("config", {}).get("arch", {}))
    model = BranchNet(pretrained=False, **arch)
    model.load_state_dict(blob["model"])
    device = torch.device(device)
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


@torch.no_grad()
def forward_full(
    model: BranchNet,
    x: torch.Tensor,
    device: torch.device | str,
    amp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(seg logits (B,5,H,W), ctr logits (B,H,W)) in float32 on ``device`` for any H, W."""
    device = torch.device(device)
    x, hw = pad_to_multiple(x.to(device, non_blocking=True))
    if device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    dtype = amp_dtype(device) if amp else None
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
        out = model(x)
    return unpad(out["seg"], hw).float(), unpad(out["ctr"], hw)[:, 0].float()


@torch.no_grad()
def predict(
    model: BranchNet,
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    device: torch.device | str,
    amp: bool = True,
) -> dict[str, np.ndarray]:
    """dict(prob=(5,H,W) float32 softmax, ctr=(H,W) float32 sigmoid) for one full frame."""
    seg, ctr = forward_full(model, prepare_input(rgb, depth_m)[None], device, amp)
    return {
        "prob": torch.softmax(seg[0], 0).cpu().numpy(),
        "ctr": torch.sigmoid(ctr[0]).cpu().numpy(),
    }


def _imwrite(path: Path, img: np.ndarray) -> None:
    if not cv2.imwrite(str(path), img):
        raise OSError(f"could not write {path}")


def save_prediction(
    out_dir: Path | str, stem: str, pred: dict[str, np.ndarray], save_prob: bool = False
) -> None:
    """Write the per-frame prediction files (see the module docstring).

    The float16 class probabilities (``.prob.npz``) cost 1-2 s of deflate and up
    to 10 MB per frame; the scorer reads only the PNGs, so they are opt-in.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prob = np.asarray(pred["prob"], dtype=np.float32)
    _imwrite(out_dir / f"{stem}.cls.png", prob.argmax(0).astype(np.uint8))
    _imwrite(out_dir / f"{stem}.conf.png", np.round(prob.max(0) * 255.0).astype(np.uint8))
    ctr = np.clip(np.asarray(pred["ctr"], dtype=np.float32), 0.0, 1.0)
    _imwrite(out_dir / f"{stem}.ctr.png", np.round(ctr * 255.0).astype(np.uint8))
    if save_prob:
        np.savez_compressed(out_dir / f"{stem}.prob.npz", prob=prob.astype(np.float16))
