"""Train BranchNet on the branch label cache, and predict held-out trees from a checkpoint.

    train    time-budgeted training with a fast full-frame validation after every
             epoch -> OUT/best.pt, OUT/last.pt, OUT/history.json
    predict  full-frame predictions for every cached frame of --trees and each
             depth source -> OUT/{source}/{tree}/{rig}/{stem}.{cls,conf,ctr}.png (+ .prob.npz with --save-prob)

The budget is wall time, not epochs, because the GPU job has a hard limit and
the per-epoch cost depends on the node (A40 / H100 / RTX 8000) and on Lustre.
`train --init CKPT` warm-starts from an earlier run's weights (strict load); the
optimizer and the learning-rate schedule start fresh.
Both subcommands also run on CPU with tiny settings, e.g.

    train   --device cpu --max-minutes 0.5 --batch 2 --crop 128 --workers 0 --val-frames 2
    predict --device cpu --limit-frames 1 --workers 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from spur_depth.branches import gt
from spur_depth.branches.dataset import (
    DEPTH_SOURCES,
    BranchCrops,
    FullFrames,
    cached_frames,
    identity_collate,
    split_frames,
    worker_init_fn,
)
from spur_depth.branches.infer import (
    amp_dtype,
    forward_full,
    load_model,
    predict,
    save_prediction,
)
from spur_depth.branches.losses import BranchLoss
from spur_depth.branches.model import BranchNet, prepare_input
from spur_depth.branches.tree import CLASSES

SRC_ROOT = Path(__file__).resolve().parents[1]
CTR_TOL_PX = 2.0
# Seconds kept free after the last validation for two checkpoint writes to NFS.
SAVE_MARGIN_S = 30.0
LOSS_KEYS = ("total", "ce", "dice", "skel", "ctr_bce", "ctr_dice")


def git_commit(src: Path = SRC_ROOT) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loader_kwargs(workers: int) -> dict:
    """Spawned workers: a loader is rebuilt every epoch, and forking after validation has
    started OpenCV/OpenMP thread pools deadlocks the new workers (job 21520501 hung at the
    start of epoch 2). The timeout turns any future stall into an error, not an idle GPU."""
    if workers <= 0:
        return {"num_workers": 0}
    return {
        "num_workers": workers,
        "worker_init_fn": worker_init_fn,
        "prefetch_factor": 2,
        "multiprocessing_context": "spawn",
        "timeout": 600,
    }


def _finite(v: float) -> float | None:
    return float(v) if math.isfinite(v) else None


# ------------------------------------------------------------------ validation


def centerline_counts(pred: np.ndarray, axis: np.ndarray, tol: float = CTR_TOL_PX) -> np.ndarray:
    """[pred px within tol of the GT axis, pred px, GT axis px within tol of pred, GT axis px]."""
    n_pred, n_axis = int(pred.sum()), int(axis.sum())
    hit_pred = hit_axis = 0
    if n_pred and n_axis:
        # distanceTransform measures the distance to the nearest ZERO pixel.
        d_axis = cv2.distanceTransform((~axis).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        hit_pred = int((d_axis[pred] <= tol).sum())
        d_pred = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        hit_axis = int((d_pred[axis] <= tol).sum())
    return np.array([hit_pred, n_pred, hit_axis, n_axis], dtype=np.int64)


@torch.no_grad()
def validate(model: BranchNet, frames: list[dict], device: torch.device, amp: bool) -> dict:
    """Per-class IoU (ignoring IGNORE) and 2-px-tolerance centerline F1, micro over frames."""
    model.eval()
    n = len(CLASSES)
    conf = torch.zeros(n * n, dtype=torch.int64, device=device)
    ctr = np.zeros(4, dtype=np.int64)
    for fr in frames:
        x = prepare_input(fr["rgb"], fr["depth"]["sensor"])[None]
        seg, ctr_logit = forward_full(model, x, device, amp)
        pred = seg[0].argmax(0)
        target = torch.from_numpy(fr["cls"]).to(device).long()
        ok = target < n  # drops IGNORE
        conf += torch.bincount(target[ok] * n + pred[ok], minlength=n * n)
        ctr += centerline_counts((ctr_logit[0] > 0).cpu().numpy(), fr["skel"] > 0)
    model.train()
    cm = conf.view(n, n).cpu().numpy().astype(np.float64)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    tree_iou = iou[1:]
    miou = float(np.nanmean(tree_iou)) if np.isfinite(tree_iou).any() else float("nan")
    p = ctr[0] / ctr[1] if ctr[1] else 0.0
    r = ctr[2] / ctr[3] if ctr[3] else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {
        "miou_tree": _finite(miou),
        "iou": {name: _finite(v) for name, v in zip(CLASSES, iou)},
        "ctr_precision": float(p),
        "ctr_recall": float(r),
        "ctr_f1": float(f1),
        "n_frames": len(frames),
    }


def pick_val(refs: list[gt.FrameRef], k: int) -> list[gt.FrameRef]:
    """A fixed subset spread evenly over the (tree, rig, shot) ordering."""
    if k <= 0 or k >= len(refs):
        return list(refs)
    return [refs[i] for i in np.unique(np.linspace(0, len(refs) - 1, k).round().astype(int))]


def load_frames(ds: FullFrames, workers: int) -> list[dict]:
    loader = DataLoader(
        ds, batch_size=None, collate_fn=identity_collate, **loader_kwargs(min(workers, len(ds)))
    )
    frames = []
    for item in loader:
        if "error" in item:
            print(f"skip val frame {ds.refs[item['index']].key}: {item['error']}", flush=True)
        else:
            frames.append(item)
    return frames


# ------------------------------------------------------------------ training


class Schedule:
    """Linear warmup, then cosine to ``floor`` over ``total`` iterations once that is known."""

    def __init__(self, warmup: int, floor: float = 0.01) -> None:
        self.warmup, self.floor = max(int(warmup), 1), floor
        self.total: int | None = None

    def factor(self, it: int) -> float:
        if self.total is None or it < self.warmup:
            return min(1.0, (it + 1) / self.warmup)
        t = min(1.0, (it - self.warmup) / max(self.total - self.warmup, 1))
        return self.floor + (1.0 - self.floor) * 0.5 * (1.0 + math.cos(math.pi * t))


def fit_iters(it: int, time_left: float, sec_per_iter: float, iters_per_epoch: int, val_s: float):
    """Total iterations reachable when every remaining epoch also pays one validation."""
    per_iter = sec_per_iter + val_s / max(iters_per_epoch, 1)
    return it + int(max(time_left - val_s, 0.0) / max(per_iter, 1e-6))


def load_init(model: BranchNet, path: Path) -> dict:
    """Warm start: copy an earlier checkpoint's weights into ``model`` (strict)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"], strict=True)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return {
        "path": str(path),
        "sha256": h.hexdigest(),
        "epoch": blob.get("epoch"),
        "git_commit": blob.get("git_commit"),
        "val_miou_tree": (blob.get("val") or {}).get("miou_tree"),
    }


def save_checkpoint(path: Path, model: BranchNet, **meta) -> None:
    """Atomic write, so a job killed mid-save never leaves a truncated best.pt."""
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save({"model": state, **meta}, tmp)
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def train(args: argparse.Namespace) -> int:
    t0 = time.time()
    budget_s = args.max_minutes * 60.0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    cuda = device.type == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)

    train_refs, val_refs, test_refs = split_frames(args.cache, args.data_root, ufo=args.ufo)
    if args.limit_train > 0:
        train_refs = train_refs[: args.limit_train]
    print(
        f"frames: train {len(train_refs)} val {len(val_refs)} test {len(test_refs)} "
        f"(cache {args.cache})",
        flush=True,
    )
    if len(train_refs) < args.batch or not val_refs:
        print(json.dumps({"error": "not enough cached frames", "cache": str(args.cache)}))
        return 2
    val_refs = pick_val(val_refs, args.val_frames)

    dtype = None if args.no_amp else amp_dtype(device)
    model = BranchNet(pretrained=not (args.no_pretrained or args.init))
    init = load_init(model, args.init) if args.init else None
    if init:
        print(
            f"warm start from {init['path']} (epoch {init['epoch']}, val mIoU-tree "
            f"{init['val_miou_tree']}, commit {init['git_commit']})",
            flush=True,
        )
    model = model.to(device)
    if cuda:
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    groups = []
    enc_lr = args.lr * args.encoder_lr_mult
    for params, lr in (
        (list(model.encoder.parameters()), enc_lr),
        (list(model.decoder_parameters()), args.lr),
    ):
        # No weight decay on BN affine parameters and biases.
        groups.append(
            {"params": [p for p in params if p.ndim > 1], "base_lr": lr, "weight_decay": args.wd}
        )
        groups.append(
            {"params": [p for p in params if p.ndim <= 1], "base_lr": lr, "weight_decay": 0.0}
        )
    for g in groups:
        g["lr"] = g["base_lr"]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    loss_fn = BranchLoss().to(device)
    commit = git_commit()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        arch=model.config,
        amp_dtype=str(dtype).replace("torch.", "") if dtype is not None else "float32",
        device=str(device),
        gpu=torch.cuda.get_device_name(device) if cuda else None,
        n_params=n_params,
        torch=torch.__version__,
        n_train_frames=len(train_refs),
        init_checkpoint=init,
        val_keys=[r.key for r in val_refs],
    )
    print(
        f"BranchNet {n_params / 1e6:.2f} M params on {device} ({config['gpu']}), "
        f"amp {config['amp_dtype']}, commit {commit}",
        flush=True,
    )

    tv = time.time()
    val_frames = load_frames(
        FullFrames(val_refs, args.cache, ("sensor",), args.data_root), args.workers
    )
    print(
        f"val: {len(val_frames)} frames, sensor depth cached in {time.time() - tv:.1f}s", flush=True
    )
    if not val_frames:
        print(json.dumps({"error": "no readable val frames"}))
        return 2

    ds = BranchCrops(
        train_refs, args.cache, crop=args.crop, seed=args.seed, data_root=args.data_root
    )
    gen = torch.Generator().manual_seed(args.seed)
    iters_per_epoch = len(ds) // args.batch
    sched = Schedule(args.warmup)
    # Validation cost before the first one is measured; only feeds the stop rule and estimate.
    val_s = (0.5 if cuda else 5.0) * len(val_frames)
    history: list[dict] = []
    best_score, best_epoch = -math.inf, 0
    it, epoch, stop = 0, 0, False
    t_mark = None  # wall time at iteration 5, after worker start-up and cuDNN autotuning
    first_wait = 0.0

    def time_left() -> float:
        return budget_s - (time.time() - t0)

    while not stop:
        if epoch >= 1 and time_left() < val_s + SAVE_MARGIN_S + 60.0:
            break  # not even a minute of training would fit before the closing validation
        epoch += 1
        ds.set_epoch(epoch)
        loader = DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=True,
            drop_last=True,
            pin_memory=cuda,
            generator=gen,
            **loader_kwargs(args.workers),
        )
        model.train()
        sums = dict.fromkeys(LOSS_KEYS, 0.0)
        window = dict.fromkeys(LOSS_KEYS, 0.0)
        src_counts = np.zeros(len(DEPTH_SOURCES), dtype=np.int64)
        steps = skipped = win_steps = 0
        t_ep = t_last = t_win = time.time()
        wait = win_wait = 0.0
        for batch in loader:
            now = time.time()
            wait += now - t_last
            win_wait += now - t_last
            if epoch == 1 and steps == 0:
                first_wait = now - t_ep
            # Never stop before the first step: best.pt must exist on every exit path.
            if (steps or epoch > 1) and time_left() < val_s + SAVE_MARGIN_S:
                stop = True
                break
            f = sched.factor(it)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            x = batch["x"].to(device, non_blocking=True)
            if cuda:
                x = x.contiguous(memory_format=torch.channels_last)
            cls = batch["cls"].to(device, non_blocking=True)
            skel = batch["skel"].to(device, non_blocking=True)
            ctr = batch["ctr"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                out = model(x)
            loss, comps = loss_fn(out, cls, skel, ctr)
            vals = torch.stack([loss.detach(), *comps.values()]).tolist()
            if not all(math.isfinite(v) for v in vals):
                skipped += 1
                if skipped > 50:
                    raise RuntimeError(f"{skipped} non-finite losses in epoch {epoch}")
                t_last = time.time()
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            it += 1
            steps += 1
            win_steps += 1
            for k, v in zip(LOSS_KEYS, vals):
                sums[k] += v
                window[k] += v
            src_counts += np.bincount(batch["src"].numpy(), minlength=len(DEPTH_SOURCES))
            if it == 5:
                t_mark = time.time()
            if it == 25 and t_mark is not None:
                sec_it = (time.time() - t_mark) / 20.0
                est = fit_iters(it, time_left() - SAVE_MARGIN_S, sec_it, iters_per_epoch, val_s)
                sched.total = max(int(0.97 * est), it + 1)
                print(
                    f"timing: {sec_it:.3f} s/it over iters 5-25 -> schedule total {sched.total} "
                    f"iters (~{sched.total / max(iters_per_epoch, 1):.1f} epochs of {iters_per_epoch})",
                    flush=True,
                )
            if it % args.log_every == 0:
                dt = time.time() - t_win
                mem = torch.cuda.max_memory_allocated(device) / 2**30 if cuda else 0.0
                avg = {k: window[k] / max(win_steps, 1) for k in LOSS_KEYS}
                print(
                    f"ep {epoch} it {it} [{steps}/{iters_per_epoch}] loss {avg['total']:.4f} "
                    f"(ce {avg['ce']:.3f} dice {avg['dice']:.3f} skel {avg['skel']:.3f} "
                    f"cbce {avg['ctr_bce']:.3f} cdice {avg['ctr_dice']:.3f}) "
                    f"lr {args.lr * f:.2e} {win_steps / max(dt, 1e-9):.2f} it/s "
                    f"data-wait {win_wait / max(dt, 1e-9):.0%} mem {mem:.1f}G "
                    f"t {(time.time() - t0) / 60:.1f}m",
                    flush=True,
                )
                window = dict.fromkeys(LOSS_KEYS, 0.0)
                win_steps, win_wait, t_win = 0, 0.0, time.time()
            t_last = time.time()
        train_s = time.time() - t_ep
        del loader
        if steps == 0:
            break  # stopped at the top of an epoch: nothing new to validate

        tv = time.time()
        metrics = validate(model, val_frames, device, amp=dtype is not None)
        score = metrics["miou_tree"] if metrics["miou_tree"] is not None else -1.0
        if args.ufo:
            # Protocol v3: each domain counts equally in checkpoint selection.
            by_domain = {}
            for name in ("envy", "ufo"):
                sub = [
                    f
                    for f in val_frames
                    if val_refs[f["index"]].tree.startswith("lpy_ufo_") == (name == "ufo")
                ]
                by_domain[name] = validate(model, sub, device, amp=dtype is not None)
            metrics["by_domain"] = by_domain
            ms = [by_domain[d]["miou_tree"] for d in ("envy", "ufo")]
            score = float(np.mean(ms)) if all(m is not None for m in ms) else -1.0
            metrics["selection_score"] = score
        val_s = time.time() - tv
        is_best = score > best_score
        rec = {
            "epoch": epoch,
            "iter": it,
            "steps": steps,
            "partial": stop,
            "skipped_nonfinite": skipped,
            "train": {k: sums[k] / steps for k in LOSS_KEYS},
            "depth_sources": dict(zip(DEPTH_SOURCES, src_counts.tolist())),
            "lr_factor": sched.factor(it),
            "schedule_total": sched.total,
            "val": metrics,
            "best": is_best,
            "train_s": round(train_s, 1),
            "data_wait_s": round(wait, 1),
            "val_s": round(val_s, 1),
            "elapsed_min": round((time.time() - t0) / 60.0, 2),
        }
        history.append(rec)
        meta = {"config": config, "epoch": epoch, "iter": it, "val": metrics, "git_commit": commit}
        save_checkpoint(args.out / "last.pt", model, **meta)
        if is_best:
            best_score, best_epoch = score, epoch
            save_checkpoint(args.out / "best.pt", model, **meta)
        write_json(args.out / "history.json", history)
        ious = " ".join(
            f"{c} {'n/a' if v is None else f'{v:.3f}'}" for c, v in metrics["iou"].items()
        )
        print(
            f"== epoch {epoch} ({'partial, ' if stop else ''}{steps} steps, train {train_s:.0f}s, "
            f"wait {wait:.0f}s, val {val_s:.0f}s) loss {rec['train']['total']:.4f} | val mIoU-tree "
            f"{score:.4f} [{ious}] ctr P/R/F1 {metrics['ctr_precision']:.3f}/"
            f"{metrics['ctr_recall']:.3f}/{metrics['ctr_f1']:.3f}{' *best*' if is_best else ''} "
            f"| t {(time.time() - t0) / 60:.1f}m",
            flush=True,
        )
        if epoch == 1 and not stop and t_mark is not None and it > 5:
            # Re-fit the cosine length with the real epoch cost (validation and start-up included).
            sec_it = (t_ep + train_s - t_mark) / (it - 5)
            est = fit_iters(
                it, time_left() - SAVE_MARGIN_S, sec_it, iters_per_epoch, val_s + first_wait
            )
            sched.total = max(int(0.97 * est), it + 1)
            print(f"schedule total re-fit after epoch 1: {sched.total} iters", flush=True)

    summary = {
        "best_epoch": best_epoch,
        "best_miou_tree": _finite(best_score),
        "epochs": epoch,
        "iters": it,
        "minutes": round((time.time() - t0) / 60.0, 2),
        "out": str(args.out),
    }
    print(json.dumps(summary), flush=True)
    return 0


# ------------------------------------------------------------------ prediction


def predict_frames(args: argparse.Namespace) -> int:
    t0 = time.time()
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = load_model(args.ckpt, device)
    blob_meta = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_info = {k: blob_meta.get(k) for k in ("epoch", "iter", "val", "git_commit")}
    del blob_meta
    if args.ufo:
        args.trees = list(args.trees) + list(gt.UFO_VAL_TREES + gt.UFO_TEST)
    refs = cached_frames(args.cache, args.trees)
    if args.limit_frames > 0:
        refs = refs[: args.limit_frames]
    print(
        f"predict: {len(refs)} frames x {args.sources} on {device}, ckpt epoch "
        f"{ckpt_info['epoch']} commit {ckpt_info['git_commit']}",
        flush=True,
    )
    if not refs:
        print(json.dumps({"error": "no cached frames for", "trees": args.trees}))
        return 2
    ds = FullFrames(refs, args.cache, tuple(args.sources), args.data_root, labels=False)
    loader = DataLoader(
        ds, batch_size=None, collate_fn=identity_collate, **loader_kwargs(args.workers)
    )
    amp = not args.no_amp
    written = dict.fromkeys(args.sources, 0)
    missing = dict.fromkeys(args.sources, 0)
    failed: list[str] = []
    t_wait = t_net = t_save = 0.0
    pending: deque = deque()
    t_last = time.time()
    with ThreadPoolExecutor(max_workers=args.save_threads) as pool:
        for k, item in enumerate(loader):
            t_wait += time.time() - t_last
            ref = refs[item["index"]]
            if "error" in item:
                failed.append(ref.key)
                print(f"skip {ref.key}: {item['error']}", flush=True)
                t_last = time.time()
                continue
            for src in args.sources:
                depth = item["depth"][src]
                if depth is None:
                    missing[src] += 1
                    continue
                tn = time.time()
                pred = predict(model, item["rgb"], depth, device, amp=amp)
                t_net += time.time() - tn
                out_dir = args.out / src / ref.tree / ref.rig
                pending.append(
                    pool.submit(save_prediction, out_dir, ref.stem, pred, args.save_prob)
                )
                written[src] += 1
                ts = time.time()
                while len(pending) > 2 * args.save_threads:
                    pending.popleft().result()
                t_save += time.time() - ts
            if (k + 1) % 20 == 0 or k + 1 == len(refs):
                el = time.time() - t0
                print(
                    f"  {k + 1}/{len(refs)} frames, {el:.0f}s ({el / (k + 1):.2f} s/frame; "
                    f"data-wait {t_wait:.0f}s net {t_net:.0f}s save-wait {t_save:.0f}s)",
                    flush=True,
                )
            t_last = time.time()
        while pending:
            pending.popleft().result()
    summary = {
        "ckpt": str(args.ckpt),
        "checkpoint": ckpt_info,
        "trees": list(args.trees),
        "sources": list(args.sources),
        "frames": len(refs),
        "written": written,
        "missing_depth": missing,
        "failed_frames": failed,
        "device": str(device),
        "seconds": {
            "total": round(time.time() - t0, 1),
            "data_wait": round(t_wait, 1),
            "network": round(t_net, 1),
            "save_wait": round(t_save, 1),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "predict_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if not failed else 1


# ------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="time-budgeted training with per-epoch validation")
    t.add_argument("--cache", type=Path, required=True, help="label cache root (v1)")
    t.add_argument("--out", type=Path, required=True, help="run directory")
    t.add_argument("--data-root", type=Path, default=gt.DATA_ROOT, help="RGB dataset root")
    t.add_argument("--max-minutes", type=float, default=75.0, help="wall budget incl. validation")
    t.add_argument("--batch", type=int, default=12)
    t.add_argument("--crop", type=int, default=512)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--lr", type=float, default=6e-4, help="decoder + heads")
    t.add_argument("--encoder-lr-mult", type=float, default=0.33)
    t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--warmup", type=int, default=300, help="iterations")
    t.add_argument("--clip", type=float, default=5.0, help="gradient norm clip")
    t.add_argument("--val-frames", type=int, default=48)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None, help="default: cuda if available")
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--no-pretrained", action="store_true", help="random encoder init")
    t.add_argument("--limit-train", type=int, default=0, help="first N train frames (tests)")
    t.add_argument(
        "--init", type=Path, default=None, help="warm start from this checkpoint's weights"
    )
    t.add_argument(
        "--ufo", action="store_true", help="add the protocol-v3 UFO train/val trees (v3)"
    )

    p = sub.add_parser("predict", help="full-frame predictions for held-out trees")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="prediction directory")
    p.add_argument("--data-root", type=Path, default=gt.DATA_ROOT)
    p.add_argument("--trees", nargs="+", default=list(gt.VAL_TREES + gt.PAPER_TEST))
    p.add_argument("--ufo", action="store_true", help="also predict the UFO val and test trees")
    p.add_argument("--sources", nargs="+", default=["gt", "sensor", "da2"], choices=DEPTH_SOURCES)
    p.add_argument("--limit-frames", type=int, default=0, help="first N frames (tests)")
    p.add_argument("--workers", type=int, default=4, help="loading + sensor simulation")
    p.add_argument("--save-threads", type=int, default=6, help="npz deflate releases the GIL")
    p.add_argument("--save-prob", action="store_true", help="also write float16 .prob.npz")
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")
    return ap


def main(argv=None) -> int:
    cv2.setNumThreads(1)  # no OpenCV thread pool in the main process
    args = build_parser().parse_args(argv)
    return train(args) if args.cmd == "train" else predict_frames(args)


if __name__ == "__main__":
    raise SystemExit(main())
