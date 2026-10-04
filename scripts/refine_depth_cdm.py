"""Refine the simulated D435 depth with a Camera Depth Model (CDM) and write it as a depth source.

CDM (arXiv 2509.02530; code github.com/ByteDance-Seed/manip-as-in-sim-suite, Apache-2.0;
weights huggingface.co/depth-anything/camera-depth-model-*, CC-BY-NC-4.0) maps RGB plus a
raw sensor depth map to a cleaned metric depth map. The input here is the scorer's own
simulated sensor, ``frame_depth("sensor", frame, ref)``; the output uses the label cache's
depth encoding, so ``spur_depth.branches.dataset.read_depth_m`` reads it back:

    OUT/{tree}/{rig}/{stem}.depth.png   uint16 mm, 0 = invalid, 65535 = at or beyond 65.535 m
    OUT/manifest.json                   checkpoint, code commits, setting, frames, timing

Model contract (CDM code at commit 757b2ea):
* input and output are inverse depth (1/m) with 0 = invalid; the network output is
  upsampled to the frame size with nearest interpolation, then inverted here;
* ``RGBDDepth.infer_image`` swaps BGR to RGB itself and the official ``infer.py`` passes
  it an RGB array, so on the official path the network sees BGR ("official");
  "flipped" passes BGR, so the network sees RGB;
* weights load strictly: Lightning ``.ckpt`` keys carry a ``pipeline.`` prefix,
  ``cdm_base.pth`` keys a ``module.`` prefix;
* the official CLI zeroes depth beyond 25 m before inverting it; this script does not:
  every positive sensor value (the simulator clips at 65 m) is passed as valid.

Optional sky fill (``--sky-fill-px D --sky-fill-m F --wood-pred-dir DIR``, off by default):
invalid sensor pixels farther than D px (Euclidean) from a classifier's predicted wood
(classes 1..4, 255 counted as wood; never GT) are set to F metres before inversion, so CDM
sees open sky as far rather than as a hole; holes within D px of the wood stay invalid.

One GPU holds the model; two loader threads decode frames and simulate the sensor.

PYTHONPATH=. python scripts/refine_depth_cdm.py --model d435 --input-size 518 \
    --precision fp16 --trees lpy_envy_00001 --out /nfs/hpc/share/sanchej7/spur-branch-eval/depth/dev
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import torch

from spur_depth.branches import gt as G
from spur_depth.branches.dataset import frame_depth, load_cached_frame, read_png

REPO = Path(__file__).resolve().parents[1]
CACHE = Path("/nfs/hpc/share/sanchej7/spur-branch-cache/v1")
CDM_SRC = Path("/nfs/hpc/share/sanchej7/tools/manip-as-in-sim-suite/cdm")
# key: (Hugging Face repo, file, pinned revision)
MODELS = {
    "d435": (
        "depth-anything/camera-depth-model-d435",
        "cdm_d435.ckpt",
        "e6aa46d86d2bc623558b7d2659e5ef907252801c",
    ),
    "l515": (
        "depth-anything/camera-depth-model-l515",
        "cdm_l515.ckpt",
        "f9a13d79e8941da60bcfcefb3b16121cca21d782",
    ),
    "base": (
        "depth-anything/camera-depth-model-base",
        "cdm_base.pth",
        "e0b202315a51d927aa50b9e7ce1f5f9b0fb44b96",
    ),
}
CHANNEL_ORDERS = {"official": "BGR", "flipped": "RGB"}  # what the network sees
MIN_DISP = 1e-6  # the ReLU head can return 0 (inf depth): such pixels are written as invalid


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_path(model: str, hf_home: Path) -> Path:
    """The pinned checkpoint inside the Hugging Face cache (no network access)."""
    repo, filename, revision = MODELS[model]
    path = Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
    path = path / revision / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing: hf download {repo} {filename} --revision {revision} (HF_HOME set)"
        )
    return path


def load_cdm(ckpt: Path, cdm_src: Path = CDM_SRC, device: str = "cuda") -> torch.nn.Module:
    """CDM ViT-L with the checkpoint's weights, loaded strictly, in eval mode on ``device``."""
    if str(cdm_src) not in sys.path:
        sys.path.insert(0, str(cdm_src))
    from rgbddepth.dpt import RGBDDepth

    with torch.device(device):  # build on the GPU: no 2.6 GB fp32 copy in host RAM
        model = RGBDDepth(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    # mmap: a Lightning .ckpt also holds ~2.8 GB of optimizer state that is never read
    ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    if "state_dict" in ck:  # cdm_{d435,l515,...}.ckpt
        sd = {k.removeprefix("pipeline."): v for k, v in ck["state_dict"].items()}
    else:  # cdm_base.pth
        sd = {k.removeprefix("module."): v for k, v in ck["model"].items()}
    for k in ("_mean", "_std"):  # the training wrapper's normalisation buffers, unused here
        sd.pop(k, None)
    model.load_state_dict(sd, strict=True)
    del ck, sd
    return model.eval()


def network_input_hw(input_size: int, hw: tuple[int, int], cdm_src: Path = CDM_SRC):
    """(h, w) the CDM transform resizes an (H, W) frame to (short side, multiple of 14)."""
    if str(cdm_src) not in sys.path:
        sys.path.insert(0, str(cdm_src))
    from rgbddepth.util.transform import Resize

    resize = Resize(
        input_size,
        input_size,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method="lower_bound",
    )
    w, h = resize.get_size(hw[1], hw[0])
    return int(h), int(w)


@torch.no_grad()
def cdm_refine(
    model: torch.nn.Module,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    input_size: int = 518,
    precision: str = "fp16",
    channel_order: str = "official",
) -> np.ndarray:
    """Refined depth (H, W) float32 metres, 0 = invalid, from RGB uint8 and depth in metres."""
    if channel_order not in CHANNEL_ORDERS or precision not in ("fp16", "fp32"):
        raise ValueError(f"bad setting {channel_order!r} / {precision!r}")
    d = np.nan_to_num(np.asarray(depth_m, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    inv = np.zeros_like(d)
    pos = d > 0
    inv[pos] = 1.0 / d[pos]
    img = rgb if channel_order == "official" else rgb[..., ::-1]
    with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
        disp = model.infer_image(np.ascontiguousarray(img), inv, input_size=input_size)
    disp = np.nan_to_num(disp.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(disp)
    ok = disp > MIN_DISP
    out[ok] = 1.0 / disp[ok]
    return out


def depth_to_mm(depth: np.ndarray) -> np.ndarray:
    """The label cache encoding (``build_branch_cache.depth_to_mm``): uint16 mm, 0 = invalid."""
    d = np.asarray(depth, dtype=np.float64)
    mm = np.rint(np.where(np.isfinite(d) & (d > 0), d, 0.0) * 1000.0)
    return np.clip(mm, 0, 65535).astype(np.uint16)


def encode_depth_png(depth: np.ndarray) -> bytes:
    """PNG bytes of ``depth_to_mm(depth)``, checked to decode to the same array."""
    mm = depth_to_mm(depth)
    ok, buf = cv2.imencode(".png", mm)
    if not ok:
        raise RuntimeError("PNG encode failed")
    back = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if back is None or back.dtype != np.uint16 or not np.array_equal(back, mm):
        raise RuntimeError("PNG round trip failed")
    return buf.tobytes()


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def predicted_wood(cls: np.ndarray) -> np.ndarray:
    """Predicted wood: classes 1..4, with 255 (ignore) counted as wood."""
    return ((cls >= 1) & (cls <= 4)) | (cls == 255)


def load_pred_wood(ref: G.FrameRef, pred_dir: Path) -> np.ndarray:
    """Predicted wood mask from a classifier's ``{tree}/{rig}/{stem}.cls.png`` (never GT)."""
    return predicted_wood(read_png(Path(pred_dir) / ref.tree / ref.rig / f"{ref.stem}.cls.png"))


def sky_fill(sensor: np.ndarray, wood: np.ndarray, dilate_px: float, far_m: float) -> np.ndarray:
    """Invalid sensor pixels farther than ``dilate_px`` from predicted wood set to ``far_m``.

    Distance is Euclidean to the nearest wood pixel (a disk dilation of radius
    ``dilate_px``); holes within it stay 0 for CDM to fill, valid pixels are unchanged.
    """
    if wood.shape != sensor.shape:
        raise ValueError(f"wood mask {wood.shape} vs sensor {sensor.shape}")
    if wood.any():
        dist = cv2.distanceTransform((~wood).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        near = dist <= dilate_px
    else:
        near = np.zeros(sensor.shape, dtype=bool)
    out = np.array(sensor, dtype=np.float32, copy=True)
    out[~(out > 0) & ~near] = np.float32(far_m)
    return out


def load_input(ref: G.FrameRef, cache: Path, fill=None) -> tuple[np.ndarray, np.ndarray]:
    """(RGB uint8, the scorer's simulated sensor depth in metres) for one frame.

    ``fill`` = (wood_pred_dir, dilate_px, far_m) applies ``sky_fill`` with the frame's
    predicted wood; None (the default) returns the sensor depth unchanged.
    """
    frame = load_cached_frame(ref, cache, skel=False, da2=False)
    sensor = frame_depth("sensor", frame, ref)
    if fill is not None:
        pred_dir, dilate_px, far_m = fill
        sensor = sky_fill(sensor, load_pred_wood(ref, pred_dir), dilate_px, far_m)
    return frame["rgb"], sensor


def iter_inputs(refs, cache: Path, workers: int = 2, ahead: int = 4, fill=None):
    """Yield (ref, (rgb, sensor)) in order, decoded by ``workers`` threads, ``ahead`` in flight."""
    it = iter(refs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        todo = deque((r, pool.submit(load_input, r, cache, fill)) for r in islice(it, ahead))
        while todo:
            ref, fut = todo.popleft()
            nxt = next(it, None)
            if nxt is not None:
                todo.append((nxt, pool.submit(load_input, nxt, cache, fill)))
            yield ref, fut.result()


def frame_refs(trees, every: int = 1, limit: int = 0) -> list[G.FrameRef]:
    refs = []
    for tree in trees:
        fr = G.list_frames([tree])[::every]
        refs += fr[:limit] if limit else fr
    return refs


def git(*args: str, cwd: Path = REPO) -> str:
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    return r.stdout.strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODELS), default="d435")
    ap.add_argument("--trees", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument("--channel-order", choices=sorted(CHANNEL_ORDERS), default="official")
    ap.add_argument("--every", type=int, default=1, help="every Nth frame per tree")
    ap.add_argument("--limit", type=int, default=0, help="first N frames per tree (0 = all)")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--cdm-src", type=Path, default=CDM_SRC)
    ap.add_argument("--hf-home", type=Path, default=os.environ.get("HF_HOME"))
    ap.add_argument("--loaders", type=int, default=2)
    ap.add_argument("--sky-fill-px", type=int, default=None, help="sky fill: distance D (px)")
    ap.add_argument("--sky-fill-m", type=float, default=None, help="sky fill: far depth F (m)")
    ap.add_argument(
        "--wood-pred-dir",
        type=Path,
        default=None,
        help="sky fill: predicted class maps {tree}/{rig}/{stem}.cls.png (never GT)",
    )
    ap.add_argument(
        "--selection",
        type=Path,
        default=None,
        help="val selection JSON: its winner must match this setting; its GPU time is added",
    )
    args = ap.parse_args(argv)
    if args.hf_home is None:
        ap.error("set HF_HOME (or --hf-home): the checkpoint is read from the Hugging Face cache")
    if not torch.cuda.is_available():
        ap.error("CDM needs a CUDA device")
    fill_args = (args.sky_fill_px, args.sky_fill_m, args.wood_pred_dir)
    if any(a is not None for a in fill_args) and any(a is None for a in fill_args):
        ap.error("--sky-fill-px, --sky-fill-m and --wood-pred-dir go together")
    fill = None
    if args.sky_fill_px is not None:
        fill = (args.wood_pred_dir, args.sky_fill_px, args.sky_fill_m)
    setting = {
        "model": args.model,
        "input_size": args.input_size,
        "precision": args.precision,
        "channel_order": args.channel_order,
    }
    if fill is not None:
        setting["sky_fill"] = {
            "dilate_px": args.sky_fill_px,
            "far_m": args.sky_fill_m,
            "wood_pred_dir": str(args.wood_pred_dir),
        }
    selection = None
    if args.selection is not None:
        sel = json.loads(args.selection.read_text())
        if sel.get("winner") != setting:
            sys.exit(f"setting {setting} is not the selection winner {sel.get('winner')}")
        selection = {
            "path": str(args.selection),
            "sha256": sha256_file(args.selection),
            "gpu_minutes": sel["gpu"]["minutes_total"],
        }

    started = _now()
    refs = frame_refs(args.trees, args.every, args.limit)
    if not refs:
        sys.exit(f"no frames for trees {args.trees}")
    repo_id, filename, revision = MODELS[args.model]
    ckpt = checkpoint_path(args.model, args.hf_home)
    print(f"{len(refs)} frames; hashing {ckpt.name}", flush=True)
    ckpt_sha = sha256_file(ckpt)

    t_load = time.perf_counter()
    model = load_cdm(ckpt, args.cdm_src)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load
    print(f"model loaded in {load_s:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)

    rows = []
    t_loop = time.perf_counter()
    inputs = iter_inputs(refs, args.cache, args.loaders, fill=fill)
    for n, (ref, (rgb, sensor)) in enumerate(inputs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        depth = cdm_refine(model, rgb, sensor, args.input_size, args.precision, args.channel_order)
        sec = time.perf_counter() - t0
        out = args.out / ref.tree / ref.rig / f"{ref.stem}.depth.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(out, encode_depth_png(depth))
        rows.append({"key": ref.key, "inference_s": round(sec, 4)})
        print(f"[{n + 1}/{len(refs)}] {ref.key} {sec:.2f}s", flush=True)
    loop_s = time.perf_counter() - t_loop
    resident_s = time.perf_counter() - t_load
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    del model
    torch.cuda.empty_cache()

    infer_s = float(sum(r["inference_s"] for r in rows))
    gpu_min = infer_s / 60.0
    hw = rgb.shape[:2]
    input_desc = (
        "spur_depth.branches.dataset.frame_depth('sensor', frame, ref) on "
        "load_cached_frame(ref, cache, skel=False, da2=False); the network gets 1/z where "
        "z > 0, else 0; no 25 m clamp (the official CDM CLI zeroes z > 25 m)"
    )
    if fill is not None:
        input_desc += (
            f"; sky fill first: sensor pixels that are not > 0 and lie farther than "
            f"{args.sky_fill_px} px (Euclidean) from predicted wood (classes 1..4 or 255 in "
            f"{args.wood_pred_dir}/{{tree}}/{{rig}}/{{stem}}.cls.png) are set to "
            f"{args.sky_fill_m} m; holes within that distance stay 0"
        )
    manifest = {
        "tool": "scripts/refine_depth_cdm.py",
        "script_sha256": sha256_file(Path(__file__)),
        "repo": {
            "path": str(REPO),
            "commit": git("rev-parse", "HEAD"),
            "tracked_files_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "script_git_status": git("status", "--porcelain", "--", "scripts/refine_depth_cdm.py")
            or "tracked, unmodified",
        },
        "checkpoint": {
            "key": args.model,
            "repo": repo_id,
            "file": filename,
            "revision": revision,
            "path": str(ckpt),
            "bytes": ckpt.stat().st_size,
            "sha256": ckpt_sha,
            "license": "CC-BY-NC-4.0",
        },
        "cdm_code": {
            "path": str(args.cdm_src),
            "commit": git("rev-parse", "HEAD", cwd=args.cdm_src),
            "dirty": bool(git("status", "--porcelain", cwd=args.cdm_src)),
            "license": "Apache-2.0",
        },
        "setting": {
            **setting,
            "frame_hw": list(hw),
            "network_input_hw": list(network_input_hw(args.input_size, hw, args.cdm_src)),
            "network_sees": CHANNEL_ORDERS[args.channel_order],
            "fp16_mode": "torch.autocast float16" if args.precision == "fp16" else None,
        },
        "input": input_desc,
        "output": {
            "layout": "{tree}/{rig}/{stem}.depth.png",
            "encoding": "uint16 mm, 0 = invalid, 65535 = at or beyond 65.535 m "
            "(build_branch_cache.depth_to_mm); read with dataset.read_depth_m",
            "invalid": f"network inverse depth <= {MIN_DISP:g} or non-finite",
            "upsampling": "nearest (CDM infer_image), in inverse depth",
        },
        "cache": str(args.cache),
        "trees": list(args.trees),
        "every": args.every,
        "limit": args.limit,
        "n_frames": len(rows),
        "frames": rows,
        "timing": {
            "inference_s_total": round(infer_s, 2),
            "inference_s_median": round(float(np.median([r["inference_s"] for r in rows])), 4),
            "gpu_minutes": round(gpu_min, 3),
            "gpu_minutes_definition": "sum of per-frame cdm_refine wall time (GPU synced)",
            "model_load_s": round(load_s, 2),
            "loop_s": round(loop_s, 2),
            "model_resident_minutes": round(resident_s / 60.0, 3),
            "peak_gpu_gib": round(peak_gib, 2),
        },
        "selection": selection,
        "gpu_minutes_with_selection": round(gpu_min + selection["gpu_minutes"], 3)
        if selection
        else None,
        "env": {
            "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cv2": cv2.__version__,
            "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "argv": sys.argv,
        "started": started,
        "finished": _now(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_atomic(args.out / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(f"{len(rows)} frames, {gpu_min:.2f} GPU-min inference, peak {peak_gib:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
