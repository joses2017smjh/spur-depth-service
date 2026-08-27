"""Export the split refiner graphs to ONNX (opset 18, dynamo=False).

python -m spur_depth.export.to_onnx --graph fuse_decode --out engines/
python -m spur_depth.export.to_onnx --graph encoder --ckpt $SPUR_CKPT --out engines/

xFormers' flash-attention kernels are CUDA-only and are not ONNX ops. Disable
them *before* importing the model so tracing uses ``scaled_dot_product_attention``.
"""

from __future__ import annotations

import os

os.environ["XFORMERS_DISABLED"] = "1"

import argparse
import json
import sys
from pathlib import Path

import torch

from spur_depth.export.wrappers import FuseDecodeWrapper, split_refiner
from spur_depth.models import MVStereoDINOUNet

SHIPPED = dict(n_views=6, use_pose=False, no_fusion=False, pred_mode="absolute")
_H, _W = 280, 512
_OPSET = 18


def _export(model: torch.nn.Module, args, input_names, output_names, path: Path) -> None:
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    # torch 2.1+ accepts dynamo=; older ignores it via TypeError fallback.
    try:
        torch.onnx.export(
            model,
            args,
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=_OPSET,
            do_constant_folding=True,
            dynamo=False,
            **extra,
        )
    except TypeError:
        torch.onnx.export(
            model,
            args,
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=_OPSET,
            do_constant_folding=True,
        )


def _maybe_check(path: Path) -> dict:
    info = {"path": str(path), "bytes": path.stat().st_size}
    # onnx.checker materialises the full protobuf. ViT-L encoder is ~1.2 GB and
    # the checker has OOM-killed this process on a 32 GB interactive job.
    if info["bytes"] > 400 * 1024 * 1024:
        info["checker"] = "skipped (>400 MB; run onnx.checker on a fatter node)"
        info["nodes"] = None
        return info
    try:
        import onnx

        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        info["ir_version"] = graph.ir_version
        info["opset"] = [op.version for op in graph.opset_import]
        info["nodes"] = len(graph.graph.node)
        info["checker"] = "ok"
    except ImportError:
        info["checker"] = "onnx package not installed; skipped"
    except Exception as exc:
        info["checker"] = f"FAIL: {exc}"
        raise
    return info


def _load_refiner(ckpt: str | None) -> MVStereoDINOUNet:
    model = MVStereoDINOUNet(**SHIPPED)
    if ckpt:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        model.load_state_dict(sd, strict=True)
    else:
        print(
            "[export] no --ckpt: exporting randomly initialised weights "
            "(fine for graph tests, not for serving)",
            file=sys.stderr,
        )
    return model.eval()


def export_fuse_decode(out_dir: Path, ckpt: str | None) -> dict:
    # fuse+decode is ~5 M params. Do not construct ViT-L just to export it.
    if ckpt:
        _, fuse = split_refiner(_load_refiner(ckpt))
    else:
        fuse = FuseDecodeWrapper(n_views=6)
    bsz, n_views = 1, fuse.n_views
    bottlenecks = torch.randn(bsz, n_views, 512, 20, 37)
    s0 = torch.randn(bsz, n_views, 256, _H, _W)
    s1 = torch.randn(bsz, n_views, 256, _H // 2, _W // 2)
    s2 = torch.randn(bsz, n_views, 256, _H // 4, _W // 4)
    s3 = torch.randn(bsz, n_views, 256, _H // 8, _W // 8)
    path = out_dir / "fuse_decode.onnx"
    _export(
        fuse,
        (bottlenecks, s0, s1, s2, s3),
        ["bottlenecks", "s0", "s1", "s2", "s3"],
        ["depth"],
        path,
    )
    return _maybe_check(path)


def export_encoder(out_dir: Path, ckpt: str | None) -> dict:
    model = _load_refiner(ckpt)
    encoder, _ = split_refiner(model)
    rgb = torch.randn(1, 3, _H, _W)
    d_pro = torch.rand(1, 1, _H, _W) * 5 + 0.5
    path = out_dir / "encoder.onnx"
    _export(
        encoder,
        (rgb, d_pro),
        ["rgb", "d_pro"],
        ["bottleneck", "s0", "s1", "s2", "s3"],
        path,
    )
    return _maybe_check(path)


def _disable_da2_xformers() -> None:
    """Make DA2's MemEffAttention use the ONNX-safe matmul path on CPU."""
    import depth_anything_v2.dinov2_layers.attention as attn  # noqa: E402

    attn.XFORMERS_AVAILABLE = False

    def _plain(self, x, attn_bias=None):
        if attn_bias is not None:
            raise AssertionError("xFormers attn_bias is not supported in the ONNX path")
        return attn.Attention.forward(self, x)

    attn.MemEffAttention.forward = _plain
    try:
        import depth_anything_v2.dinov2_layers.block as blk  # noqa: E402

        blk.XFORMERS_AVAILABLE = False
    except Exception:
        pass


def export_da2(
    out_dir: Path,
    ckpt: str | None,
    da2_root: str,
    height: int,
    width: int,
    max_depth: float,
) -> dict:
    """Engine A: DepthAnythingV2.forward at a static multiple-of-14 size."""
    if not ckpt or not Path(ckpt).is_file():
        raise FileNotFoundError(
            "DA2 export needs --da2-ckpt (the fine-tune best.pth). "
            "Refusing to export randomly initialised metric depth."
        )
    if height % 14 or width % 14:
        raise ValueError(f"DA2 ONNX size must be a multiple of 14, got {height}x{width}")

    sys.path.insert(0, da2_root)
    # DA2 vendors its own DINOv2. It does not honour XFORMERS_DISABLED; its
    # MemEffAttention calls CUDA-only xFormers kernels and ONNX tracing dies
    # on CPU (job 21004172). Force the plain Attention.forward path *and*
    # replace MemEffAttention.forward so a later re-import cannot flip it.
    _disable_da2_xformers()
    from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

    from spur_depth.export.wrappers import DA2ForwardWrapper

    print(f"[da2] construct ViT-L  ckpt={ckpt}", file=sys.stderr, flush=True)
    raw = DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        max_depth=max_depth,
    )
    print("[da2] torch.load (CPU) ...", file=sys.stderr, flush=True)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    print(f"[da2] load_state_dict  keys={len(sd)}", file=sys.stderr, flush=True)
    raw.load_state_dict(sd, strict=True)
    raw.eval()
    wrapped = DA2ForwardWrapper(raw)
    rgb = torch.randn(1, 3, height, width)
    path = out_dir / "da2.onnx"
    print(f"[da2] onnx.export {height}x{width} -> {path}", file=sys.stderr, flush=True)
    _export(wrapped, (rgb,), ["rgb"], ["depth"], path)
    print(f"[da2] wrote {path.stat().st_size} bytes", file=sys.stderr, flush=True)
    info = _maybe_check(path)
    info["graph"] = "da2"
    info["H"] = height
    info["W"] = width
    info["max_depth"] = max_depth
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--graph", choices=["fuse_decode", "encoder", "da2", "both"], default="fuse_decode")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--da2-ckpt", type=str, default=os.environ.get("SPUR_DA2_CKPT"))
    p.add_argument(
        "--da2-root",
        type=str,
        default=os.environ.get(
            "DA2_ROOT",
            "/nfs/hpc/share/sanchej7/Computer_Vision/depth-anything-v2/metric_depth",
        ),
    )
    p.add_argument("--da2-h", type=int, default=518)
    p.add_argument("--da2-w", type=int, default=924, help="518×924 is 1080×1920 after DA2's 14-multiple resize")
    p.add_argument("--max-depth", type=float, default=20.0)
    p.add_argument("--out", type=str, default="engines")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    reports = []
    if args.graph in ("fuse_decode", "both"):
        reports.append(export_fuse_decode(out_dir, args.ckpt))
    if args.graph in ("encoder", "both"):
        reports.append(export_encoder(out_dir, args.ckpt))
    if args.graph == "da2":
        reports.append(export_da2(out_dir, args.da2_ckpt, args.da2_root, args.da2_h, args.da2_w, args.max_depth))
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
