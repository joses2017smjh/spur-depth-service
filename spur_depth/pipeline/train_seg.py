"""Tiny trunk segmenter trained on the 7 non-paper trees, tested on 00042/00065.

This is the detector the field papers would call YOLO. We train a 4-level
U-Net (no extra deps) on the Blender trunk masks, then report IoU / Dice
on the paper val trees. Boxes come from the predicted mask via
``boxes_from_mask`` — same SSD-shaped output, now from a learned head.

Also reports a field-like DA2 RMSE that scores on the *predicted* mask
instead of the GT mask, which is what you have on a real tree.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from spur_depth.bench.rigorous import load_triple, masked_rmse
from spur_depth.data.restore_index import DEFAULT_ROOT, PAPER_VAL, discover_da2_triples
from spur_depth.pipeline.detect import CLS_TRUNK, boxes_from_mask, keep_largest_component

H = W = 256


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
            )

        self.e1, self.e2, self.e3 = block(3, 16), block(16, 32), block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.b = block(64, 128)
        self.u3, self.u2, self.u1 = block(128 + 64, 64), block(64 + 32, 32), block(32 + 16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.b(self.pool(c))
        x = self.u3(torch.cat([F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False), c], 1))
        x = self.u2(torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), b], 1))
        x = self.u1(torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), a], 1))
        return self.head(x)


class MaskDS(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = [r for r in rows if r["rgb"] is not None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rgb = np.asarray(Image.open(r["rgb"]).convert("RGB").resize((W, H), Image.BILINEAR))
        mask = np.asarray(Image.open(r["mask"]).convert("L").resize((W, H), Image.NEAREST)) > 0
        x = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        y = torch.from_numpy(mask.astype(np.float32))[None]
        return x, y


def _dice(logits, y, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * y).sum()
    den = p.sum() + y.sum() + eps
    return num / den


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    dices, ious = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).float()
        inter = (pred * y).sum().item()
        union = (pred + y).clamp(0, 1).sum().item()
        ious.append(inter / union if union else 1.0)
        dices.append(_dice(logits, y).item())
    return float(np.mean(ious)), float(np.mean(dices))


@torch.no_grad()
def predict_mask(model, rgb: np.ndarray, device) -> np.ndarray:
    h0, w0 = rgb.shape[:2]
    x = torch.from_numpy(
        np.asarray(Image.fromarray(rgb).resize((W, H), Image.BILINEAR)).transpose(2, 0, 1).astype(np.float32) / 255.0
    )[None].to(device)
    pred = (torch.sigmoid(model(x))[0, 0].cpu().numpy() > 0.5)
    return np.asarray(Image.fromarray((pred * 255).astype(np.uint8)).resize((w0, h0), Image.NEAREST)) > 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("weights/trunk_unet.pt"))
    ap.add_argument("--eval-ckpt", type=Path, default=None, help="skip training; score an existing TinyUNet")
    args = ap.parse_args(argv)

    rows = discover_da2_triples(args.data_root)
    train = [r for r in rows if r["tree"] not in PAPER_VAL and r["rgb"] is not None]
    val = [r for r in rows if r["tree"] in PAPER_VAL and r["rgb"] is not None]
    if len(train) < 16 or len(val) < 4:
        print(json.dumps({"error": "not enough frames", "n_train": len(train), "n_val": len(val)}))
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_tr, ds_va = MaskDS(train), MaskDS(val)
    ld_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)
    ld_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)
    model = TinyUNet().to(device)
    history = []
    if args.eval_ckpt is not None:
        blob = torch.load(args.eval_ckpt, map_location=device, weights_only=False)
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        model.load_state_dict(sd, strict=True)
        history = blob.get("history", []) if isinstance(blob, dict) else []
        args.out = args.eval_ckpt
        iou, dice = _eval(model, ld_va, device)
        print(f"eval-ckpt  val IoU {iou:.3f}  Dice {dice:.3f}", flush=True)
        if not history:
            history = [{"epoch": 0, "train_loss": None, "val_iou": iou, "val_dice": dice}]
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(1, args.epochs + 1):
            model.train()
            losses = []
            for x, y in ld_tr:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y) + (1.0 - _dice(logits, y))
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            iou, dice = _eval(model, ld_va, device)
            history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_iou": iou, "val_dice": dice})
            print(f"epoch {ep}  loss {np.mean(losses):.4f}  val IoU {iou:.3f}  Dice {dice:.3f}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "history": history, "hold_trees": list(PAPER_VAL)}, args.out)

    model.eval()
    field_gt, field_pred, field_clean, demo = [], [], [], []
    rng = random.Random(0)
    sample = rng.sample(val, k=min(24, len(val)))
    for r in sample:
        rgb = np.asarray(Image.open(r["rgb"]).convert("RGB"))
        pred_m = predict_mask(model, rgb, device)
        da2, gt, gt_m = load_triple(r)
        if pred_m.shape != gt_m.shape:
            pred_m = np.asarray(
                Image.fromarray((pred_m.astype(np.uint8) * 255)).resize((gt_m.shape[1], gt_m.shape[0]), Image.NEAREST)
            ) > 0
        e_gt = masked_rmse(da2, gt, gt_m)
        cleaned = keep_largest_component(pred_m, min_area=2048)
        e_pr = masked_rmse(da2, gt, pred_m)
        e_clean = masked_rmse(da2, gt, cleaned)
        if e_gt is not None:
            field_gt.append(e_gt)
        if e_pr is not None:
            field_pred.append(e_pr)
        if e_clean is not None:
            field_clean.append(e_clean)
        demo.append(
            {
                "path": str(r["rgb"]),
                "n_boxes_raw": len(boxes_from_mask(pred_m, CLS_TRUNK)),
                "n_boxes_cleaned": len(boxes_from_mask(cleaned, CLS_TRUNK, min_area=2048)),
            }
        )

    report = {
        "n_train": len(train),
        "n_val": len(val),
        "device": str(device),
        "history": history,
        "best_val_iou": max(h["val_iou"] for h in history),
        "weights": str(args.out),
        "field_rmse": {
            "n": len(sample),
            "da2_on_gt_mask_m": float(np.mean(field_gt)) if field_gt else None,
            "da2_on_pred_mask_m": float(np.mean(field_pred)) if field_pred else None,
            "da2_on_cleaned_pred_mask_m": float(np.mean(field_clean)) if field_clean else None,
        },
        "demo_boxes": demo,
        "note": "Trained on 7 trees, tested on paper val pair. Not a YOLO checkpoint.",
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("n_train", "n_val", "best_val_iou", "field_rmse", "weights")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
