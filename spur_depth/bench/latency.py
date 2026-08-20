"""Latency benchmark for the depth pipeline.

Why this exists
───────────────
The paper reports 484.7 ms for a 6-view group and ~80 ms/view, measured on a
DGX A100/H100 partition. Those numbers are not a usable baseline for an export
speedup claim: a TensorRT number measured on one GPU against a PyTorch number
measured on another is not a speedup, it is two unrelated numbers. This script
re-measures the PyTorch baseline on whatever GPU you actually export on, so
every row of the README latency table shares hardware.

Methodology
───────────
  * 20 warmup iterations, discarded. The first CUDA call initialises the
    context and cuDNN autotunes; timing those makes the mean meaningless.
  * 200 timed iterations, each bracketed by CUDA events. Events time the
    device, not the host, so they are not perturbed by Python overhead.
  * Reports p50/p95/p99 alongside mean±std. Percentiles are what a latency
    SLA is written against; a mean hides the tail that actually hurts.
  * Synthetic inputs. Latency depends on tensor shape, not tensor content,
    and using random tensors keeps disk I/O out of the measurement.

Every result is written to ``bench/results/<gpu-slug>_<date>.json`` with the
full environment (torch, CUDA, driver, GPU, git SHA) so a number can never be
separated from the machine that produced it.

Usage
─────
    python -m spur_depth.bench.latency --stage refiner --ckpt /path/best.pt
    python -m spur_depth.bench.latency --stage refiner --precision fp16
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Environment provenance
# ──────────────────────────────────────────────────────────────────────────────


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def gpu_slug(name: str) -> str:
    """'Quadro RTX 8000' -> 'quadro-rtx-8000', for use in a filename."""
    return "-".join(name.lower().replace("-", " ").split())


def environment() -> dict:
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "gpu": dev,
        "driver": _driver_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "git_sha": _git_sha(Path(__file__).resolve().parents[2]),
        "date": date.today().isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Timing core
# ──────────────────────────────────────────────────────────────────────────────


def time_callable(fn: Callable[[], object], warmup: int, iters: int) -> list[float]:
    """Return per-iteration wall time in ms, measured with CUDA events."""
    use_cuda = torch.cuda.is_available()

    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            # Sync on the event, not the whole device: we want this iteration's
            # elapsed time, not the time until every queued kernel drains.
            end.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
    return times


def summarize(times: list[float]) -> dict:
    s = sorted(times)

    def pct(p: float) -> float:
        # Nearest-rank percentile; unambiguous and matches what trtexec reports.
        idx = min(len(s) - 1, max(0, int(round(p / 100.0 * len(s) + 0.5)) - 1))
        return s[idx]

    return {
        "n": len(s),
        "mean_ms": statistics.fmean(s),
        "std_ms": statistics.stdev(s) if len(s) > 1 else 0.0,
        "min_ms": s[0],
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "max_ms": s[-1],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Stages
# ──────────────────────────────────────────────────────────────────────────────


def build_refiner(args, device, dtype):
    """The shipped 3-pair DINO refiner: 6 views, no pose, fusion on."""
    from spur_depth.models import MVStereoDINOUNet

    model = MVStereoDINOUNet(
        n_views=args.n_views, use_pose=False, no_fusion=False, pred_mode="absolute"
    )
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(sd, strict=True)
    else:
        # Latency is shape-driven, so random weights time identically. Say so
        # in the JSON rather than letting a reader assume it was the real model.
        print("[latency] no --ckpt given: timing randomly-initialised weights", file=sys.stderr)
    model = model.to(device=device, dtype=dtype).eval()

    rgb = torch.randn(1, args.n_views, 3, args.H, args.W, device=device, dtype=dtype)
    d_pro = torch.rand(1, args.n_views, 1, args.H, args.W, device=device, dtype=dtype) * 5 + 0.5

    def run():
        with torch.no_grad():
            model(rgb, d_pro)

    return run, {"n_views": args.n_views, "H": args.H, "W": args.W, "batch": 1}


def build_da2(args, device, dtype):
    """Depth Anything V2 ViT-L metric head, one frame at 518x518."""
    sys.path.insert(0, str(Path(args.da2_root).resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

    model = DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        max_depth=args.max_depth,
    )
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
    model = model.to(device=device, dtype=dtype).eval()

    x = torch.randn(1, 3, 518, 518, device=device, dtype=dtype)

    def run():
        with torch.no_grad():
            model(x)

    return run, {"H": 518, "W": 518, "batch": 1, "max_depth": args.max_depth}


STAGES = {"refiner": build_refiner, "da2": build_da2}


# ──────────────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=sorted(STAGES), default="refiner")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument(
        "--da2-root",
        type=str,
        default="/nfs/hpc/share/sanchej7/Computer_Vision/depth-anything-v2/metric_depth",
    )
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--n-views", dest="n_views", type=int, default=6)
    p.add_argument("--H", type=int, default=280)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--max-depth", type=float, default=20.0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--out-dir", type=str, default=None, help="default: <repo>/bench/results")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print(
            "WARNING: no CUDA device; timing on CPU is not comparable to any "
            "GPU row in the README table.",
            file=sys.stderr,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.precision == "fp16" else torch.float32

    run, shape_meta = STAGES[args.stage](args, device, dtype)
    times = time_callable(run, args.warmup, args.iters)
    stats = summarize(times)

    env = environment()
    record = {
        "stage": args.stage,
        "backend": "torch",
        "precision": args.precision,
        "weights": "trained" if args.ckpt else "random-init",
        "shape": shape_meta,
        "warmup": args.warmup,
        **stats,
        "env": env,
    }
    if args.stage == "refiner":
        record["per_view_ms"] = stats["p50_ms"] / args.n_views

    print(json.dumps(record, indent=2))
    print(
        f"\n{args.stage} [{args.precision}, torch] on {env['gpu']}:  "
        f"p50 {stats['p50_ms']:.1f} ms | p95 {stats['p95_ms']:.1f} | "
        f"p99 {stats['p99_ms']:.1f} | mean {stats['mean_ms']:.1f} ± {stats['std_ms']:.1f}",
        file=sys.stderr,
    )

    if not args.no_save:
        out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else Path(__file__).resolve().parents[2] / "bench" / "results"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{gpu_slug(env['gpu'])}_{env['date']}.json"
        # Append rather than overwrite: one file per GPU per day accumulates
        # every stage/precision measured on that machine.
        blob = json.loads(path.read_text()) if path.is_file() else []
        blob.append(record)
        path.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"[latency] appended to {path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
