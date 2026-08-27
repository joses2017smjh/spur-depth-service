"""Second-pass rigor on the 9-tree restore. Does not cache 1080p maps.

Stores only eroded trunk (x, y) vectors (~2k pixels / frame). Fits:

* ordinary LS (must match holdout_affine JSON)
* Huber IRLS (δ = 5 cm)
* median-of-trees α/β
* 30 cm box-anchor scale (no GT depth)

Scores paper val trees. Also reports box-gated field RMSE: GT box, UNet
box (if ``weights/trunk_unet.pt``), YOLO box (if ``weights/yolo_nano.pt``).

The 24k train set is still missing. This file will use it the moment
``discover_train_or_restore`` sees more trees — it does not invent them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from spur_depth.bench.rigorous import apply_affine, load_triple, masked_rmse
from spur_depth.calib.fit_scale_shift import (
    merge_in_order,
    moments_from_xy,
    solve_scale_shift,
    valid_pixels,
)
from spur_depth.calib.robust import HUBER_DELTA_M, huber_irls, median_of_groups
from spur_depth.data.restore_index import (
    DEFAULT_ROOT,
    PAPER_VAL,
    discover_train_or_restore,
    split_holdout,
)
from spur_depth.pipeline.detect import CLS_TRUNK, box_to_mask, boxes_from_mask
from spur_depth.pipeline.reconstruct import load_pose
from spur_depth.pipeline.sim2real import box_anchor_scale

# Job 21004733, n=78 GT-mask on paper val trees. Do not replace this from n=55.
HEADLINE_LS_M = 0.034366251874510464


def headline_stays_ls(
    best_method: str | None, rmse_m: float | None, *, ls_m: float = HEADLINE_LS_M
) -> bool:
    """Keep 0.0344 m unless another method beats it by more than 1 mm on n=78."""
    if best_method is None or rmse_m is None or best_method == "ls":
        return True
    return not (rmse_m < ls_m - 0.001)


def _xy(pred, gt, mask):
    v = valid_pixels(pred, gt, mask, erode_r=10, min_depth=0.5, max_depth=10.0)
    if int(v.sum()) < 32:
        return None
    y = np.asarray(gt, dtype=np.float64)[v]
    if float(y.std()) < 0.05:
        return None
    x = np.asarray(pred, dtype=np.float64)[v]
    return x, y


def _summarize(errs: list[float]) -> dict:
    if not errs:
        return {"n": 0, "rmse_m": None}
    arr = np.asarray(errs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "rmse_m": float(arr.mean()),
        "rmse_std_m": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
    }


def _score(hold: list[dict], alpha: float, beta: float) -> dict:
    errs = []
    for rec in hold:
        if rec["xy"] is None:
            continue
        x, y = rec["xy"]
        d = (alpha * x + beta) - y
        errs.append(float(np.sqrt(np.mean(d * d))))
    return _summarize(errs)


def _score_gt_mask(hold: list[dict], alpha: float, beta: float) -> dict:
    """Same gate as ``holdout_affine_2026-08-22.json`` (n=78), not the xy-std filter (n=55)."""
    errs = []
    for rec in hold:
        if "pred" not in rec:
            continue
        e = masked_rmse(apply_affine(rec["pred"], alpha, beta), rec["gt"], rec["mask"])
        if e is not None:
            errs.append(e)
    return _summarize(errs)


def _load_box_mask(row: dict, shape: tuple[int, int]) -> np.ndarray | None:
    p = row.get("box_mask")
    if p is None or not Path(p).is_file():
        return None
    m = np.asarray(Image.open(p).convert("L")) > 0
    if m.shape != shape:
        m = (
            np.asarray(
                Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (shape[1], shape[0]), Image.NEAREST
                )
            )
            > 0
        )
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--unet", type=Path, default=Path("weights/trunk_unet.pt"))
    ap.add_argument("--yolo", type=Path, default=Path("weights/yolo_nano.pt"))
    ap.add_argument("--min-orchard-frames", type=int, default=1000)
    ap.add_argument(
        "--skip-detectors",
        action="store_true",
        help="Skip UNet/YOLO box-gated RMSE (CPU re-score of α/β only).",
    )
    args = ap.parse_args(argv)

    rows = discover_train_or_restore(args.data_root)
    if not rows:
        print("no Da2Finetune triples", flush=True)
        return 2
    fit_rows, hold_rows = split_holdout(rows)
    orchard = len(rows) >= args.min_orchard_frames

    fit_xy: list[dict] = []
    hold_xy: list[dict] = []
    box_scales = []
    for i, r in enumerate(rows):
        pred, gt, mask = load_triple(r)
        rec = {"row": r, "xy": _xy(pred, gt, mask), "raw": masked_rmse(pred, gt, mask)}
        if r["tree"] not in PAPER_VAL:
            fit_xy.append(rec)
            bm = _load_box_mask(r, pred.shape)
            if bm is not None and r["ann"].is_file():
                K, _ = load_pose(json.loads(r["ann"].read_text()))
                sc = box_anchor_scale(pred, bm, K=K)
                if np.isfinite(sc.get("scale", np.nan)):
                    box_scales.append(float(sc["scale"]))
        else:
            rec["pred"], rec["gt"], rec["mask"] = pred, gt, mask
            hold_xy.append(rec)
        if (i + 1) % 50 == 0:
            print(f"[improve] loaded {i + 1}/{len(rows)}", flush=True)

    # Huber / median need fit triples. Rebuild from xy for Huber via fake 1-D
    # images so we never keep 1080p on the fit cohort.
    fit_1d = []
    fit_keys = []
    for rec in fit_xy:
        if rec["xy"] is None:
            continue
        x, y = rec["xy"]
        pred = x.reshape(1, -1).astype(np.float32)
        gt = y.reshape(1, -1).astype(np.float32)
        mask = np.ones_like(pred, dtype=bool)
        fit_1d.append((pred, gt, mask))
        fit_keys.append(rec["row"]["tree"])

    if not fit_1d:
        print("no valid fit pixels", flush=True)
        return 2

    ls_parts = [moments_from_xy(p.reshape(-1), g.reshape(-1)) for p, g, _ in fit_1d]
    ls_acc = merge_in_order(ls_parts)
    ls_a, ls_b = solve_scale_shift(ls_acc)
    hub_a, hub_b, hub_acc = huber_irls(
        fit_1d, delta=HUBER_DELTA_M, iters=6, erode_r=0, min_gt_std=0.0
    )
    grouped = {}
    for k, t in zip(fit_keys, fit_1d):
        grouped.setdefault(k, []).append(t)
    med_a, med_b, per_tree = median_of_groups(grouped, erode_r=0, min_gt_std=0.0)
    box_a = float(np.median(box_scales)) if box_scales else None
    box_b = 0.0

    hold_for_score = hold_xy
    payload = {
        "n_frames_total": len(rows),
        "n_fit_frames": len(fit_rows),
        "n_hold_frames": len(hold_rows),
        "orchard_scale": orchard,
        "four_bark_24k": False,
        "note": (
            "One bark (bark_brown_02) crossed 1000 frames. That is not the paper 4-bark 24k set."
            if orchard
            else "Fewer than 1000 frames. 9-tree restore only."
        ),
        "methods": {
            "ls": {
                "alpha": ls_a,
                "beta": ls_b,
                "n_pixels": ls_acc.n,
                "holdout": _score(hold_for_score, ls_a, ls_b),
                "holdout_gt_mask": _score_gt_mask(hold_xy, ls_a, ls_b),
            },
            "huber_irls": {
                "alpha": hub_a,
                "beta": hub_b,
                "delta_m": HUBER_DELTA_M,
                "n_pixels": hub_acc.n,
                "holdout": _score(hold_for_score, hub_a, hub_b),
                "holdout_gt_mask": _score_gt_mask(hold_xy, hub_a, hub_b),
            },
            "median_of_trees": {
                "alpha": med_a,
                "beta": med_b,
                "per_tree": {k: {"alpha": a, "beta": b} for k, (a, b) in per_tree.items()},
                "holdout": _score(hold_for_score, med_a, med_b),
                "holdout_gt_mask": _score_gt_mask(hold_xy, med_a, med_b),
            },
            "box_anchor_scale": {
                "alpha": box_a,
                "beta": box_b,
                "n_fit_boxes": len(box_scales),
                "holdout": _score(hold_for_score, box_a, box_b)
                if box_a is not None
                else {"n": 0, "rmse_m": None},
                "holdout_gt_mask": _score_gt_mask(hold_xy, box_a, box_b)
                if box_a is not None
                else {"n": 0, "rmse_m": None},
                "note": "Scale only from the 30 cm box on the fit trees. No GT depth.",
            },
        },
    }

    # Winner on the n=55 xy-std split (diagnostic). Headline uses n=78 GT-mask.
    cands = [("ls", ls_a, ls_b, payload["methods"]["ls"]["holdout"])]
    for name, a, b in (("huber_irls", hub_a, hub_b), ("median_of_trees", med_a, med_b)):
        cands.append((name, a, b, payload["methods"][name]["holdout"]))
    if box_a is not None:
        cands.append(
            ("box_anchor_scale", box_a, box_b, payload["methods"]["box_anchor_scale"]["holdout"])
        )
    best = min(
        (c for c in cands if c[3].get("rmse_m") is not None),
        key=lambda c: c[3]["rmse_m"],
        default=None,
    )
    payload["best_holdout"] = (
        {"method": best[0], "alpha": best[1], "beta": best[2], **best[3]} if best else None
    )
    mask_cands = []
    for name in ("ls", "huber_irls", "median_of_trees", "box_anchor_scale"):
        rec = payload["methods"].get(name, {}).get("holdout_gt_mask") or {}
        if rec.get("rmse_m") is not None:
            mask_cands.append((name, rec["rmse_m"], rec.get("n")))
    best_mask = min(mask_cands, key=lambda c: c[1]) if mask_cands else None
    payload["best_holdout_gt_mask"] = (
        {"method": best_mask[0], "rmse_m": best_mask[1], "n": best_mask[2]} if best_mask else None
    )
    payload["headline_stays_ls_unless_best_wins"] = headline_stays_ls(
        best_mask[0] if best_mask else None,
        best_mask[1] if best_mask else None,
    )
    payload["headline_ls_rmse_m"] = HEADLINE_LS_M

    if args.skip_detectors:
        payload["field_box_gated"] = {
            "skipped": True,
            "note": "Detector-gated RMSE left to the GPU job. This run is α/β only.",
        }
    else:
        payload["field_box_gated"] = _field_box_gated(hold_xy, args.unet, args.yolo, ls_a, ls_b)

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}", flush=True)
    return 0


def _field_box_gated(hold_xy, unet_path: Path, yolo_path: Path, alpha: float, beta: float) -> dict:
    gt_box, unet_box, yolo_box, gt_mask = [], [], [], []
    device = None
    unet = yolo = None
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if unet_path.is_file():
            from spur_depth.pipeline.train_seg import TinyUNet, predict_mask

            unet = TinyUNet().to(device)
            blob = torch.load(unet_path, map_location=device, weights_only=False)
            sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
            unet.load_state_dict(sd, strict=True)
            unet.eval()
        if yolo_path.is_file():
            from spur_depth.pipeline.train_yolo import predict_dets
            from spur_depth.pipeline.yolo import YOLONano

            blob = torch.load(yolo_path, map_location=device, weights_only=False)
            anc = blob.get("anchors")
            yolo = YOLONano(
                tuple(map(tuple, anc)) if anc else ((28.0, 140.0), (40.0, 180.0), (56.0, 210.0))
            ).to(device)
            yolo.load_state_dict(blob["model"], strict=True)
            yolo.eval()
    except Exception as exc:
        return {"error": str(exc)}

    for rec in hold_xy:
        if "pred" not in rec:
            continue
        pred = apply_affine(rec["pred"], alpha, beta)
        gt, mask = rec["gt"], rec["mask"]
        e = masked_rmse(pred, gt, mask)
        if e is not None:
            gt_mask.append(e)
        dets = boxes_from_mask(mask, CLS_TRUNK, min_area=2048)
        if dets:
            e_b = masked_rmse(pred, gt, box_to_mask(gt.shape, dets[0], erode_px=10))
            if e_b is not None:
                gt_box.append(e_b)
        rgb_p = rec["row"].get("rgb")
        if rgb_p is None:
            continue
        rgb = np.asarray(Image.open(rgb_p).convert("RGB"))
        if unet is not None:
            from spur_depth.pipeline.train_seg import predict_mask

            pm = predict_mask(unet, rgb, device)
            pd = boxes_from_mask(pm, CLS_TRUNK, min_area=2048)
            if pd:
                e_u = masked_rmse(pred, gt, box_to_mask(gt.shape, pd[0], erode_px=10))
                if e_u is not None:
                    unet_box.append(e_u)
        if yolo is not None:
            from spur_depth.pipeline.train_yolo import predict_dets

            pd = predict_dets(yolo, rgb, device)
            if pd:
                e_y = masked_rmse(pred, gt, box_to_mask(gt.shape, pd[0], erode_px=10))
                if e_y is not None:
                    yolo_box.append(e_y)
    return {
        "da2_affine_on_gt_mask_m": _summarize(gt_mask),
        "da2_affine_on_gt_box_m": _summarize(gt_box),
        "da2_affine_on_unet_box_m": _summarize(unet_box),
        "da2_affine_on_yolo_box_m": _summarize(yolo_box),
        "note": (
            "GT-box gated is the oracle a perfect detector can reach. "
            "UNet/YOLO boxes are what you have in the field."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
