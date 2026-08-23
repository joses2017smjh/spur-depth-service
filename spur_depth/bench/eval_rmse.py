"""Reproduce the paper's masked RMSE from a checkpoint and a val manifest.

The number the trainer logged as ``best_rmse`` is ``masked_rmse_nview``
averaged over the 6 views, then averaged over val batches, with the loader's
own mask (trunk pixels whose GT depth is in ``[min_depth, max_depth]``).
This script calls that same function, so a disagreement is a real bug, not a
metric-definition drift.

Backends
────────
``torch``  the restored ``MVStereoDINOUNet``. This is the P1 gate.
``onnx``   split encoder + fuse/decode graphs from ``spur_depth.export``.
``trt``    not in the minimum slice; raises with a pointer to P3.

The 88 GB Blender dataset is not in git. Without ``DATA_ROOT`` (or a
resolving val manifest) this script exits 2 with a message, rather than
inventing a number. ``--synthetic`` exists only as a plumbing smoke test
and is labelled as such in the JSON.

Usage
─────
    python -m spur_depth.bench.eval_rmse --backend torch --ckpt /path/best.pt \\
        --val-manifest /path/stereo_val_boxfam.csv --no-pro-calib
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spur_depth.bench.latency import environment, gpu_slug
from spur_depth.models.losses import masked_rmse_nview

_RERENDER_NOTE = (
    "Val trees were re-rendered 2026-08-20 (job val9_chain-20976470). "
    "Original Da2Finetune .npy files are gone, so this number is not expected "
    "to match the paper's 0.0445 m bit-for-bit. Label it as a re-render."
)

SHIPPED = dict(n_views=6, use_pose=False, no_fusion=False, pred_mode="absolute")
PAPER_RMSE = 0.0445
PAPER_RMSE_STD = 0.0057


def _load_torch_model(ckpt: str, device: torch.device):
    from spur_depth.models import MVStereoDINOUNet

    model = MVStereoDINOUNet(**SHIPPED)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def _predict_torch(model, rgb, d_pro, device):
    rgb = rgb.to(device)
    d_pro = d_pro.to(device)
    with torch.no_grad():
        return model(rgb, d_pro).cpu()


def _predict_onnx(session_enc, session_fuse, rgb, d_pro):
    import numpy as np

    b, v, _, h, w = rgb.shape
    rgb_n = rgb.reshape(b * v, 3, h, w).numpy()
    d_n = d_pro.reshape(b * v, 1, h, w).numpy()
    enc_out = session_enc.run(None, {"rgb": rgb_n, "d_pro": d_n})
    bott, s0, s1, s2, s3 = enc_out
    c_b = bott.shape[1]
    bott = bott.reshape(b, v, c_b, bott.shape[2], bott.shape[3])
    s0 = s0.reshape(b, v, s0.shape[1], s0.shape[2], s0.shape[3])
    s1 = s1.reshape(b, v, s1.shape[1], s1.shape[2], s1.shape[3])
    s2 = s2.reshape(b, v, s2.shape[1], s2.shape[2], s2.shape[3])
    s3 = s3.reshape(b, v, s3.shape[1], s3.shape[2], s3.shape[3])
    (pred,) = session_fuse.run(
        None,
        {
            "bottlenecks": bott.astype(np.float32),
            "s0": s0.astype(np.float32),
            "s1": s1.astype(np.float32),
            "s2": s2.astype(np.float32),
            "s3": s3.astype(np.float32),
        },
    )
    return torch.from_numpy(pred)


def _score_loader(predict_fn, loader, min_depth: float, max_depth: float) -> dict:
    view_sums = None
    n = 0
    for batch in loader:
        rgb = batch["rgb"]
        d_pro = batch["d_pro"]
        d_gt = batch["d_gt"]
        mask = batch["mask"]
        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(0)
            d_pro = d_pro.unsqueeze(0)
            d_gt = d_gt.unsqueeze(0)
            mask = mask.unsqueeze(0)
        pred = predict_fn(rgb, d_pro)
        mean_rmse, per_view = masked_rmse_nview(
            pred, d_gt, mask, min_depth=min_depth, max_depth=max_depth
        )
        if view_sums is None:
            view_sums = [0.0] * len(per_view)
        for i, v in enumerate(per_view):
            view_sums[i] += v
        n += 1
        if n == 1 or n % 5 == 0:
            print(
                f"[eval] batch {n}  running_rmse={sum(view_sums)/len(view_sums)/n:.4f} m",
                file=sys.stderr,
                flush=True,
            )
        _ = mean_rmse
    if not n:
        raise RuntimeError("val loader was empty — manifests resolved to zero samples")
    per_view = [s / n for s in view_sums]
    return {
        "n_batches": n,
        "rmse_mean_m": sum(per_view) / len(per_view),
        "rmse_per_view_m": per_view,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["torch", "onnx", "trt"], default="torch")
    p.add_argument("--ckpt", type=str, default=os.environ.get("SPUR_CKPT"))
    p.add_argument("--val-manifest", type=str, default=None)
    p.add_argument("--path-remap", type=str, default="full_trunk:full_spur")
    p.add_argument("--no-pro-calib", action="store_true", default=True)
    p.add_argument("--min-depth", type=float, default=0.5)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--H", type=int, default=280)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--onnx-dir", type=str, default=None)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="numpy seed for inverse-distance neighbour sampling in the loader",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="write JSON here; default bench/results/<gpu-slug>_<date>_eval.json",
    )
    p.add_argument(
        "--synthetic",
        type=int,
        default=0,
        metavar="N",
        help="score N random tensors (NOT comparable to the paper; plumbing only)",
    )
    args = p.parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.backend == "trt":
        print(
            "TensorRT eval is P3 and is not in this slice. Measure fp32/fp16 "
            "ONNX first; do not invent a TRT number.",
            file=sys.stderr,
        )
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        if args.backend != "torch" or not args.ckpt:
            print("--synthetic requires --backend torch and --ckpt", file=sys.stderr)
            return 2
        model = _load_torch_model(args.ckpt, device)
        rmses = []
        for i in range(args.synthetic):
            torch.manual_seed(i)
            rgb = torch.randn(1, 6, 3, args.H, args.W)
            d_pro = torch.rand(1, 6, 1, args.H, args.W) * 5 + 0.5
            d_gt = d_pro.clone()
            mask = torch.ones_like(d_gt)
            pred = _predict_torch(model, rgb, d_pro, device)
            mean_rmse, _ = masked_rmse_nview(
                pred, d_gt, mask, min_depth=args.min_depth, max_depth=args.max_depth
            )
            rmses.append(mean_rmse)
        record = {
            "backend": "torch",
            "mode": "synthetic-plumbing",
            "paper_comparable": False,
            "rmse_mean_m": float(sum(rmses) / len(rmses)),
            "env": environment(),
        }
        print(json.dumps(record, indent=2))
        print(
            "WARNING: --synthetic is not the paper metric. It only proves the harness runs.",
            file=sys.stderr,
        )
        return 0

    if not args.val_manifest or not Path(args.val_manifest).is_file():
        print(
            "No val manifest. The 88 GB dataset is not in git (see TRANSFER.md "
            "in the training repo). Pass --val-manifest when Data/ is restored. "
            "Refusing to print a fake RMSE.",
            file=sys.stderr,
        )
        return 2

    os.environ.setdefault("INPUT_DEPTH_SUBDIR", "Da2Finetune")

    from spur_depth.data import TrunkStereoTripletMVPDataset

    ds = TrunkStereoTripletMVPDataset(
        args.val_manifest,
        H=args.H,
        W=args.W,
        path_remap=args.path_remap,
        pro_calib=not args.no_pro_calib,
    )
    if len(ds) == 0:
        print(
            f"Val manifest {args.val_manifest} resolved to 0 samples. "
            "Input-depth files are missing (INPUT_DEPTH_SUBDIR="
            f"{os.environ.get('INPUT_DEPTH_SUBDIR')}).",
            file=sys.stderr,
        )
        return 2

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    if args.backend == "torch":
        if not args.ckpt:
            print("--ckpt is required for --backend torch", file=sys.stderr)
            return 2
        model = _load_torch_model(args.ckpt, device)

        def predict(rgb, d_pro):
            return _predict_torch(model, rgb, d_pro, device)

    else:
        import onnxruntime as ort

        onnx_dir = Path(args.onnx_dir or "engines")
        enc_path = onnx_dir / "encoder.onnx"
        fuse_path = onnx_dir / "fuse_decode.onnx"
        if not enc_path.is_file() or not fuse_path.is_file():
            print(
                f"ONNX graphs not found in {onnx_dir} (need encoder.onnx and fuse_decode.onnx).",
                file=sys.stderr,
            )
            return 2
        sess_opt = ort.SessionOptions()
        providers = ["CPUExecutionProvider"]
        session_enc = ort.InferenceSession(
            str(enc_path), sess_options=sess_opt, providers=providers
        )
        session_fuse = ort.InferenceSession(
            str(fuse_path), sess_options=sess_opt, providers=providers
        )

        def predict(rgb, d_pro):
            return _predict_onnx(session_enc, session_fuse, rgb, d_pro)

    stats = _score_loader(predict, loader, args.min_depth, args.max_depth)
    env = environment()
    record = {
        "backend": args.backend,
        "paper_comparable": False,
        "paper_comparable_reason": _RERENDER_NOTE,
        "paper_rmse_m": PAPER_RMSE,
        "paper_rmse_std_m": PAPER_RMSE_STD,
        "checkpoint_best_rmse_m": 0.044330,
        "eval_mask": f"trunk, {args.min_depth}–{args.max_depth} m",
        "n_views": 6,
        "n_samples": len(ds),
        "seed": args.seed,
        "ckpt": args.ckpt,
        "val_manifest": args.val_manifest,
        **stats,
        "env": env,
    }
    print(json.dumps(record, indent=2))
    delta = abs(stats["rmse_mean_m"] - PAPER_RMSE)
    print(
        f"val RMSE {stats['rmse_mean_m']:.4f} m  "
        f"(paper {PAPER_RMSE:.4f} ± {PAPER_RMSE_STD:.4f}, "
        f"|Δ|={delta:.4f}; re-rendered val, not a bit-match)",
        file=sys.stderr,
    )
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[2]
        / "bench"
        / "results"
        / f"{gpu_slug(env['gpu'])}_{env['date']}_eval.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[eval] wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
