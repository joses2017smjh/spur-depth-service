"""Torch vs ONNX Runtime max-abs check for a split-graph export."""

from __future__ import annotations

import os

os.environ["XFORMERS_DISABLED"] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spur_depth.export.wrappers import split_refiner
from spur_depth.models import MVStereoDINOUNet

SHIPPED = dict(n_views=6, use_pose=False, no_fusion=False, pred_mode="absolute")
_H, _W = 280, 512


def _ort_session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def fuse_decode_parity(onnx_path: Path, ckpt: str | None, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = MVStereoDINOUNet(**SHIPPED).eval()
    if ckpt:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"], strict=True)
    _, fuse = split_refiner(model)

    bsz, n_views = 1, 6
    bottlenecks = torch.randn(bsz, n_views, 512, 20, 37)
    s0 = torch.randn(bsz, n_views, 256, _H, _W)
    s1 = torch.randn(bsz, n_views, 256, _H // 2, _W // 2)
    s2 = torch.randn(bsz, n_views, 256, _H // 4, _W // 4)
    s3 = torch.randn(bsz, n_views, 256, _H // 8, _W // 8)

    with torch.no_grad():
        torch_out = fuse(bottlenecks, s0, s1, s2, s3).numpy()

    sess = _ort_session(onnx_path)
    (ort_out,) = sess.run(
        None,
        {
            "bottlenecks": bottlenecks.numpy(),
            "s0": s0.numpy(),
            "s1": s1.numpy(),
            "s2": s2.numpy(),
            "s3": s3.numpy(),
        },
    )
    delta = np.max(np.abs(torch_out - ort_out))
    return {
        "graph": "fuse_decode",
        "max_abs": float(delta),
        "torch_shape": list(torch_out.shape),
        "onnx_shape": list(ort_out.shape),
        "pass": bool(delta < 1e-4),
    }


def encoder_parity(onnx_path: Path, ckpt: str | None, seed: int = 0) -> dict:
    """Compare torch vs ORT without holding both graphs in RAM at once.

    The ViT-L encoder ONNX is ~1.2 GB. Loading it together with the PyTorch
    checkpoint on an 8 GB interactive job gets the process SIGKILL'd. Run
    torch on CUDA, dump activations, free, then ORT.
    """
    import gc

    torch.manual_seed(seed)
    rgb = torch.randn(1, 3, _H, _W)
    d_pro = torch.rand(1, 1, _H, _W) * 5 + 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MVStereoDINOUNet(**SHIPPED).eval()
    if ckpt:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"], strict=True)
        del blob
    encoder, _ = split_refiner(model)
    del model
    encoder = encoder.to(device)
    with torch.no_grad():
        torch_outs = [t.detach().cpu().numpy() for t in encoder(rgb.to(device), d_pro.to(device))]
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sess = _ort_session(onnx_path)
    ort_outs = sess.run(None, {"rgb": rgb.numpy(), "d_pro": d_pro.numpy()})
    del sess
    deltas = [float(np.max(np.abs(a - b))) for a, b in zip(torch_outs, ort_outs)]
    delta = max(deltas)
    return {
        "graph": "encoder",
        "max_abs": delta,
        "max_abs_per_output": deltas,
        "pass": bool(delta < 1e-4),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--graph", choices=["fuse_decode", "encoder"], default="fuse_decode")
    args = p.parse_args(argv)
    fn = fuse_decode_parity if args.graph == "fuse_decode" else encoder_parity
    report = fn(Path(args.onnx), args.ckpt)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
