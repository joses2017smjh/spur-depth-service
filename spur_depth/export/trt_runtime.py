"""TensorRT runtime that does not assume a system `trtexec`.

The HPC module is TensorRT 7 / CUDA 10. We load a pip wheel from
``SPUR_TRT_PATH`` (default ``/nfs/hpc/share/sanchej7/opt/py-tensorrt``)
so the 3 GB NGC TensorRT image never has to land on disk.

Only ``fuse_decode.onnx`` (~26 MB) is built by default. Encoder and DA2
ONNX are 1.2–1.3 GB each; their engines would clone that on disk and OOM
an 8 GB interactive job. Do not quote a fuse-only number as end-to-end.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_DEFAULT_WHEEL = Path("/nfs/hpc/share/sanchej7/opt/py-tensorrt")


def _trt_lib_dirs(extra: Path) -> list[Path]:
    return [p for p in (extra / "tensorrt_libs", extra / "nvidia" / "tensorrt_libs") if p.is_dir()]


def bootstrap_tensorrt(*, reexec: bool = True) -> None:
    extra = Path(os.environ.get("SPUR_TRT_PATH", _DEFAULT_WHEEL))
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
    libs = _trt_lib_dirs(extra)
    if not libs:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [str(p) for p in libs if str(p) not in cur.split(":")]
    if not parts:
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts + ([cur] if cur else []))
    # Linux only honours LD_LIBRARY_PATH at process start. Re-exec once.
    if reexec and os.environ.get("SPUR_TRT_REEXEC") != "1":
        os.environ["SPUR_TRT_REEXEC"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])


def import_trt():
    bootstrap_tensorrt()
    import tensorrt as trt  # noqa: WPS433

    return trt


def _cuda_sm() -> tuple[int, int] | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
    except Exception:
        return None
    return None


def network_creation_flags(trt) -> int:
    """TRT 8–9 need ``EXPLICIT_BATCH``; TRT 10+ made it the default and dropped the flag.

    Job 21017775 on RTX 8000 died here: TensorRT 11.2 bindings have
    ``NetworkDefinitionCreationFlag`` with no ``EXPLICIT_BATCH``.
    """
    enum = getattr(trt, "NetworkDefinitionCreationFlag", None)
    if enum is None:
        return 0
    flag = getattr(enum, "EXPLICIT_BATCH", None)
    if flag is None:
        return 0
    return 1 << int(flag)


def builder_has_fp16(builder) -> bool:
    """TRT 11 dropped ``Builder.platform_has_fast_fp16`` (job 21018041)."""
    attr = getattr(builder, "platform_has_fast_fp16", None)
    if attr is None:
        return True
    return bool(attr() if callable(attr) else attr)


def enable_fp16(trt, config) -> bool:
    """TRT 11 removed ``BuilderFlag.FP16`` (job 21018057). Strong typing is the default.

    Returns True only if a weak-typing FP16 flag was actually set. A TRT 11
    engine built without it is FP32-from-ONNX, not an FP16 speedup. Do not
    quote it as TensorRT FP16.
    """
    flag = getattr(getattr(trt, "BuilderFlag", None), "FP16", None)
    if flag is None:
        return False
    config.set_flag(flag)
    return True


def build_plan(
    onnx: Path,
    plan: Path,
    *,
    fp16: bool = True,
    workspace_mib: int = 2048,
) -> dict:
    sm = _cuda_sm()
    # TensorRT 11.x wheels refuse Volta (sm_70). Job 21015984 died here on V100.
    if sm is not None and sm < (7, 5):
        raise RuntimeError(
            f"GPU SM {sm[0]}.{sm[1]} is below TensorRT 11 minimum SM 7.5 "
            "(Volta V100 is 7.0). Rebuild fuse_decode on Turing/Ampere/Hopper "
            "(RTX 8000, A40, A100, H100). Do not quote a TRT number from a V100."
        )
    trt = import_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if builder is None:
        raise RuntimeError("tensorrt.Builder returned None (unsupported GPU / CUDA)")
    network = builder.create_network(network_creation_flags(trt))
    parser = trt.OnnxParser(network, logger)
    raw = Path(onnx).read_bytes()
    if not parser.parse(raw):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed: " + "; ".join(errs[:8]))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib << 20)
    used_fp16 = False
    if fp16 and builder_has_fp16(builder):
        used_fp16 = enable_fp16(trt, config)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    plan.parent.mkdir(parents=True, exist_ok=True)
    Path(plan).write_bytes(bytes(serialized))
    return {
        "onnx": str(onnx),
        "plan": str(plan),
        "plan_bytes": Path(plan).stat().st_size,
        "fp16": bool(used_fp16),
        "tensorrt": str(trt.__version__),
        "workspace_mib": workspace_mib,
    }


class TrtEngine:
    """Run a serialized engine with torch CUDA buffers (no pycuda)."""

    def __init__(self, plan: Path):
        import torch

        trt = import_trt()
        self.torch = torch
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(Path(plan).read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize {plan}")
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.input_names = []
        self.output_names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, feeds: dict[str, "object"]) -> dict[str, "object"]:
        torch = self.torch
        ctx = self.context
        for name, arr in feeds.items():
            t = arr if torch.is_tensor(arr) else torch.from_numpy(np.ascontiguousarray(arr))
            t = t.to(device="cuda", dtype=torch.float32, non_blocking=True).contiguous()
            ctx.set_tensor_address(name, t.data_ptr())
            feeds[name] = t
        outs = {}
        for name in self.output_names:
            shape = tuple(ctx.get_tensor_shape(name))
            out = torch.empty(shape, device="cuda", dtype=torch.float32)
            ctx.set_tensor_address(name, out.data_ptr())
            outs[name] = out
        with torch.cuda.stream(self.stream):
            ok = ctx.execute_async_v3(self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("execute_async_v3 failed")
        self.stream.synchronize()
        return outs


def _ort_fuse(onnx: Path, feeds: dict[str, np.ndarray]) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    out = sess.run(None, {k: feeds[k] for k in names})
    return out[0]


def bench_fuse(onnx: Path, plan: Path, n: int = 50, warmup: int = 10) -> dict:
    import onnx as ox
    import torch

    graph = ox.load(str(onnx), load_external_data=False)
    feeds_np = {}
    for inp in graph.graph.input:
        dims = [d.dim_value or 1 for d in inp.type.tensor_type.shape.dim]
        dims = [max(int(x), 1) for x in dims]
        feeds_np[inp.name] = np.random.randn(*dims).astype(np.float32)

    cpu = _ort_fuse(onnx, feeds_np)
    eng = TrtEngine(plan)
    trt_out = eng.infer(dict(feeds_np))
    trt_np = next(iter(trt_out.values())).detach().cpu().numpy()
    max_abs = float(np.max(np.abs(cpu - trt_np)))

    for _ in range(warmup):
        eng.infer(dict(feeds_np))
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        eng.infer(dict(feeds_np))
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        "graph": "fuse_decode",
        "n": n,
        "p50_ms": ts[len(ts) // 2],
        "p95_ms": ts[int(len(ts) * 0.95)],
        "max_abs_vs_ort_cpu": max_abs,
        "plan": str(plan),
        "plan_bytes": Path(plan).stat().st_size,
        "note": "fuse_decode only. Not end-to-end refiner latency.",
    }


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--onnx", type=Path, default=Path("engines/fuse_decode.onnx"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--bench", action="store_true")
    args = p.parse_args(argv)
    if args.out_dir is None:
        from spur_depth.paths import engine_dir

        args.out_dir = engine_dir()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fp16 = not args.fp32
    slug = "tesla-v100-sxm3-32gb"
    try:
        import torch

        if torch.cuda.is_available():
            slug = "-".join(torch.cuda.get_device_name(0).lower().replace("/", " ").split())
    except Exception:
        pass
    plan = args.out_dir / f"{args.onnx.stem}_{'fp16' if fp16 else 'fp32'}_{slug}.plan"
    rec: dict = {}
    if not args.skip_build:
        rec = build_plan(args.onnx, plan, fp16=fp16)
        print(json.dumps(rec, indent=2), flush=True)
    if args.bench:
        if not plan.is_file():
            print(f"missing plan {plan}", file=sys.stderr)
            return 2
        bench = bench_fuse(args.onnx, plan)
        rec.update(bench)
        from spur_depth.paths import artifact_dir

        out = artifact_dir() / f"trt_fuse_{slug}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps(bench, indent=2), flush=True)
        print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
