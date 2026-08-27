"""C2: prove a ``.plan`` is a TensorRT engine. No invented millisecond.

The pip wheel at ``SPUR_TRT_PATH`` ships ``libnvinfer`` but not ``NvInfer.h``,
so the C++ binary is skipped in CI. This Python entry deserializes the plan
and prints I/O names — same contract as ``cpp/tensorrt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def inspect_plan(plan: Path) -> dict:
    from spur_depth.export.trt_runtime import import_trt

    trt = import_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    blob = Path(plan).read_bytes()
    if len(blob) < 8:
        raise RuntimeError("file too small to be a TensorRT engine")
    engine = runtime.deserialize_cuda_engine(blob)
    if engine is None:
        raise RuntimeError("deserializeCudaEngine failed (wrong TRT version or corrupt plan)")
    ios = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        kind = "in" if mode == trt.TensorIOMode.INPUT else "out"
        ios.append({"name": name, "io": kind})
    return {"plan": str(plan), "bytes": len(blob), "io": ios, "tensorrt": str(trt.__version__)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("plan", type=Path)
    args = p.parse_args(argv)
    if not args.plan.is_file():
        print(f"missing plan {args.plan}", file=sys.stderr)
        return 2
    try:
        rec = inspect_plan(args.plan)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
