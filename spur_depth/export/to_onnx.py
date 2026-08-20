"""Export the split refiner graphs to ONNX (opset 18, dynamo=False).

python -m spur_depth.export.to_onnx --graph fuse_decode --out engines/
python -m spur_depth.export.to_onnx --graph encoder --ckpt $SPUR_CKPT --out engines/
"""

from __future__ import annotations

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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--graph", choices=["fuse_decode", "encoder", "both"], default="fuse_decode")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--out", type=str, default="engines")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    reports = []
    if args.graph in ("fuse_decode", "both"):
        reports.append(export_fuse_decode(out_dir, args.ckpt))
    if args.graph in ("encoder", "both"):
        reports.append(export_encoder(out_dir, args.ckpt))
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
