"""Build TensorRT engines from the split ONNX graphs.

Usage on a GPU node with TensorRT installed (trtexec or the Python API):

    python -m spur_depth.export.build_engine --onnx engines/fuse_decode.onnx --fp16
    python -m spur_depth.export.build_engine --onnx engines/encoder.onnx \\
        --min rgb:1x3x280x512,d_pro:1x1x280x512 \\
        --opt rgb:6x3x280x512,d_pro:6x1x280x512 \\
        --max rgb:6x3x280x512,d_pro:6x1x280x512 --fp16
    python -m spur_depth.export.build_engine --onnx engines/da2.onnx --fp16

Engines are not portable across GPU / TRT / driver. The output filename is
``<stem>_<fp32|fp16>_<gpu-slug>.plan`` so two cards never clobber each other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path


def _find_trtexec() -> str | None:
    env = os.environ.get("TRTEXEC")
    if env and Path(env).is_file():
        return env
    which = shutil.which("trtexec")
    if which:
        return which
    for candidate in (
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
        "/opt/tensorrt/bin/trtexec",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


def _gpu_slug() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        name = out.stdout.strip().splitlines()[0]
        return "-".join(name.lower().replace("-", " ").split()) or "gpu"
    except Exception:
        return "gpu"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _trtexec_version(trtexec: str) -> str:
    try:
        out = subprocess.run([trtexec], capture_output=True, text=True, timeout=20)
        for line in (out.stdout + out.stderr).splitlines():
            if "TensorRT" in line or "trtexec" in line.lower():
                return line.strip()[:200]
    except Exception:
        pass
    return "unknown"


def build_with_trtexec(
    trtexec: str,
    onnx: Path,
    plan: Path,
    fp16: bool,
    min_shapes: str | None,
    opt_shapes: str | None,
    max_shapes: str | None,
    workspace_mib: int,
) -> dict:
    cmd = [
        trtexec,
        f"--onnx={onnx}",
        f"--saveEngine={plan}",
        f"--memPoolSize=workspace:{workspace_mib}M",
        "--builderOptimizationLevel=4",
    ]
    if fp16:
        cmd.append("--fp16")
    if min_shapes:
        cmd.append(f"--minShapes={min_shapes}")
    if opt_shapes:
        cmd.append(f"--optShapes={opt_shapes}")
    if max_shapes:
        cmd.append(f"--maxShapes={max_shapes}")
    print("[trt]", " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path = plan.with_suffix(".trtexec.log")
    log_path.write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"trtexec failed ({proc.returncode}). Last 40 lines:\n"
            + "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
        )
    return {"tool": "trtexec", "cmd": cmd, "log": str(log_path)}


def build_with_python_api(onnx: Path, plan: Path, fp16: bool, workspace_mib: int) -> dict:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "Neither trtexec nor the tensorrt Python package is available. "
            "Load a TensorRT module or use an nvcr.io/nvidia/tensorrt image."
        ) from exc

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with onnx.open("rb") as fh:
        if not parser.parse(fh.read()):
            errs = [parser.get_error(i).desc() for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed: " + "; ".join(errs[:8]))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib << 20)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("tensorrt build_serialized_network returned None")
    plan.write_bytes(bytes(serialized))
    return {"tool": f"tensorrt-python {trt.__version__}"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="engines")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--min", dest="min_shapes", default=None)
    p.add_argument("--opt", dest="opt_shapes", default=None)
    p.add_argument("--max", dest="max_shapes", default=None)
    p.add_argument("--workspace-mib", type=int, default=8192)
    args = p.parse_args(argv)

    onnx = Path(args.onnx)
    if not onnx.is_file():
        print(f"missing ONNX: {onnx}", file=sys.stderr)
        return 2

    precision = "fp16" if args.fp16 else "fp32"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = out_dir / f"{onnx.stem}_{precision}_{_gpu_slug()}.plan"

    trtexec = _find_trtexec()
    try:
        if trtexec:
            tool = build_with_trtexec(
                trtexec,
                onnx,
                plan,
                args.fp16,
                args.min_shapes,
                args.opt_shapes,
                args.max_shapes,
                args.workspace_mib,
            )
        else:
            tool = build_with_python_api(onnx, plan, args.fp16, args.workspace_mib)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    record = {
        "onnx": str(onnx),
        "onnx_sha256": _sha256(onnx),
        "plan": str(plan),
        "plan_bytes": plan.stat().st_size,
        "precision": precision,
        "gpu": _gpu_slug(),
        "date": date.today().isoformat(),
        "trtexec": _trtexec_version(trtexec) if trtexec else None,
        **tool,
    }
    meta = plan.with_suffix(".json")
    meta.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
