"""Inference runners.

One GPU, one in-flight batch. Do not start this process with ``--workers 4``:
four processes each construct a ViT-L context (~1.3 GB of weights, plus any
engine) and the GPU serialises them anyway. Use one worker, ``InferenceGate``
in ``app.py``, and 503 when the queue is full.

``DummyRunner`` exists so CI and ``SPUR_SKIP_WEIGHTS=1`` can exercise the
HTTP contract without the 1.3 GB checkpoint. It is deterministic in the
inputs so a preprocess regression changes the output.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from PIL import Image

from spur_depth.serve.preprocess import (
    REFINER_H,
    REFINER_W,
    load_input_depth,
    load_rgb,
)

N_VIEWS = 6


@dataclass
class RunnerInfo:
    sha: str
    precision: str
    backend: str


class Runner(Protocol):
    info: RunnerInfo

    def warmup(self) -> None: ...

    def predict_one(self, image: Image.Image) -> np.ndarray:
        """Return (H, W) float32 metres."""

    def predict_group(
        self, images: list[Image.Image], depths: list[np.ndarray] | None = None
    ) -> np.ndarray:
        """Return (V, H, W) float32 metres."""


class DummyRunner:
    """Constant-time stand-in used by CI and ``SPUR_SKIP_WEIGHTS=1``.

    Output is a function of ImageNet-normalised RGB mean so a silent switch
    from bilinear to nearest (or a dropped normalisation) fails the fixture
    test. Not a depth model.
    """

    def __init__(self) -> None:
        self.info = RunnerInfo(sha="dummy", precision="dummy", backend="dummy")

    def warmup(self) -> None:
        return None

    def _map_from_rgb(self, rgb: torch.Tensor) -> np.ndarray:
        # rgb: (3, H, W) ImageNet-normalised
        level = float(rgb.mean().item())
        depth = np.full((rgb.shape[1], rgb.shape[2]), 2.0 + 0.1 * level, dtype=np.float32)
        return depth

    def _maybe_delay(self) -> None:
        delay_ms = float(os.environ.get("SPUR_INFER_DELAY_MS", "0") or 0)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    def predict_one(self, image: Image.Image) -> np.ndarray:
        self._maybe_delay()
        rgb = load_rgb(image)
        return self._map_from_rgb(rgb)

    def predict_group(
        self, images: list[Image.Image], depths: list[np.ndarray] | None = None
    ) -> np.ndarray:
        maps = [self.predict_one(im) for im in images]
        if depths is not None:
            # If the client supplied metric depth, pass it through the same
            # bilinear resize the refiner was trained on and ignore RGB. This
            # path exists so /predict/group can be tested without DA2.
            maps = [load_input_depth(d).squeeze(0).numpy() for d in depths]
        return np.stack(maps, axis=0)


class TorchRefinerRunner:
    """Shipped 6-view DINO refiner on PyTorch (CPU or CUDA)."""

    def __init__(self, ckpt: str, device: str | None = None) -> None:
        from spur_depth.models import MVStereoDINOUNet

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = blob["model"]
        sha = hashlib.sha256()
        with open(ckpt, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                sha.update(chunk)
        self.model = MVStereoDINOUNet(
            n_views=N_VIEWS, use_pose=False, no_fusion=False, pred_mode="absolute"
        )
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device).eval()
        self.info = RunnerInfo(
            sha=sha.hexdigest()[:16],
            precision="fp32",
            backend=f"torch-{self.device.type}",
        )
        self._da2 = None

    def warmup(self) -> None:
        rgb = torch.randn(1, N_VIEWS, 3, REFINER_H, REFINER_W, device=self.device)
        d_pro = torch.rand(1, N_VIEWS, 1, REFINER_H, REFINER_W, device=self.device) + 0.5
        with torch.no_grad():
            self.model(rgb, d_pro)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def predict_one(self, image: Image.Image) -> np.ndarray:
        """Real-time path is DA2-ft, which is not loaded in this slice.

        Returning a 501 from the runner lets the HTTP layer speak 501 with a
        stable message rather than pretending the refiner is a single-view
        model.
        """
        raise NotImplementedError(
            "POST /predict is the DA2-ft single-view path (Engine A). "
            "This runner only has the 6-view refiner; use POST /predict/group "
            "or load a DA2 checkpoint (P2 Engine A)."
        )

    def predict_group(
        self, images: list[Image.Image], depths: list[np.ndarray] | None = None
    ) -> np.ndarray:
        if depths is None or len(depths) != len(images):
            raise ValueError(
                "the 6-view refiner needs per-view metric depth "
                "(DA2-ft .npy, already in metres). Pass `depths` or run Engine A first."
            )
        rgbs = [load_rgb(im) for im in images]
        dpros = [load_input_depth(d) for d in depths]
        rgb = torch.stack(rgbs, dim=0).unsqueeze(0).to(self.device)
        d_pro = torch.stack(dpros, dim=0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(rgb, d_pro)
        return out.squeeze(0).squeeze(1).cpu().numpy()


def build_runner() -> Runner:
    if os.environ.get("SPUR_SKIP_WEIGHTS", "") in {"1", "true", "TRUE"}:
        return DummyRunner()
    ckpt = os.environ.get("SPUR_CKPT", "")
    if not ckpt or not os.path.isfile(ckpt):
        # Prefer a dummy over a crash on first boot of an empty container.
        # /readyz still reports ready; /model.train_domain tells the client
        # this is not the paper checkpoint.
        return DummyRunner()
    return TorchRefinerRunner(ckpt)
