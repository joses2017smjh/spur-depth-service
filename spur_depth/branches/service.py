"""RGB-D frame -> tree graph JSON, the code behind ``POST /branches``.

BranchNet labels every pixel, optional monocular depth repairs the sensor's
thin-wood failures (``depth_fusion``), and ``assemble`` returns the parts with
their 3D axes, parents, junctions and cut points in the camera frame.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from spur_depth.branches.assemble import AssembleConfig, assemble
from spur_depth.branches.depth_fusion import fuse_sensor_mono


class BranchService:
    def __init__(self, ckpt: str | os.PathLike | None = None, device: str | None = None):
        self.ckpt = Path(ckpt) if ckpt else None
        self.device = device
        self.model = None
        self.sha = None
        self.cfg = AssembleConfig()

    @classmethod
    def from_env(cls) -> BranchService:
        return cls(os.environ.get("SPUR_BRANCH_CKPT") or None, os.environ.get("SPUR_DEVICE"))

    @property
    def configured(self) -> bool:
        return self.ckpt is not None or self.model is not None

    def _load(self):
        if self.model is not None:
            return self.model
        if self.ckpt is None:
            raise NotImplementedError(
                "POST /branches needs SPUR_BRANCH_CKPT (BranchNet checkpoint)"
            )
        import torch

        from spur_depth.branches.infer import load_model

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(self.ckpt, torch.device(dev))
        self.sha = hashlib.sha256(self.ckpt.read_bytes()).hexdigest()[:12]
        return self.model

    def run(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        mono_depth: np.ndarray | None = None,
    ) -> dict:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"image must be HxWx3, got {rgb.shape}")
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"depth {depth.shape} does not match image {rgb.shape[:2]}")
        if mono_depth is not None and mono_depth.shape != depth.shape:
            raise ValueError(f"mono_depth {mono_depth.shape} does not match depth {depth.shape}")
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("K must be a 3x3 pinhole matrix with positive focal lengths")
        model = self._load()
        import torch

        from spur_depth.branches.infer import predict

        t0 = time.perf_counter()
        dev = next(model.parameters()).device
        out = predict(model, rgb, depth, dev, amp=dev.type == "cuda")
        cls = out["prob"].argmax(0).astype(np.uint8)
        conf = out["prob"].max(0)
        t1 = time.perf_counter()
        lift = depth
        fusion = None
        if mono_depth is not None:
            lift, fusion = fuse_sensor_mono(depth, mono_depth, cls > 0)
        graph = assemble(cls, lift, K, conf=conf, cfg=self.cfg)
        t2 = time.perf_counter()
        body = graph.to_json()
        body["meta"] = {
            "model_sha256_prefix": self.sha,
            "assemble_config": asdict(self.cfg),
            "depth_fusion": fusion,
            "latency_ms": {"network": 1e3 * (t1 - t0), "graph": 1e3 * (t2 - t1)},
            "torch": torch.__version__,
        }
        return body
