"""Leave-one-cohort scale/shift, stereo fusion, and hold-out RMSE.

The 24k train set is not on disk. Nine val-render trees are. This script
fits α/β on the seven trees that are *not* the paper val pair
(``lpy_envy_00042``, ``lpy_envy_00065``) and reports RMSE only on those
two. That is a real hold-out, not a fit-on-the-number-you-publish.

Also reports:

* raw DA2-ft vs GT (no affine)
* hold-out α/β applied to DA2
* per-tree LOO α/β (diagnostic; not the headline)
* stereo depth from L→R Farneback + camera baseline, fused with DA2
  by inverse-variance

Applying this α/β to the *refiner* input is a different experiment — the
shipped DINO run was trained without a DA2 affine. Headline numbers here
are DA2-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from spur_depth.calib.fit_scale_shift import (
    Moments,
    merge_in_order,
    moments_from_image,
    solve_scale_shift,
    valid_pixels,
)
from spur_depth.data.restore_index import (
    DEFAULT_ROOT,
    PAPER_VAL,
    discover_da2_triples,
    split_holdout,
)
from spur_depth.pipeline.flow import dense_flow
from spur_depth.pipeline.reconstruct import load_pose
from spur_depth.pipeline.sensors import fuse_depths
from spur_depth.serve.drift import baseline_from_stats
from spur_depth.serve.input_stats import request_stats


def _resize_to(arr: np.ndarray, hw: tuple[int, int], nearest: bool) -> np.ndarray:
    if arr.shape[:2] == hw:
        return arr
    img = Image.fromarray(arr if arr.ndim == 2 else arr)
    resample = Image.NEAREST if nearest else Image.BILINEAR
    out = np.asarray(img.resize((hw[1], hw[0]), resample))
    return out.astype(arr.dtype, copy=False)


def load_triple(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.load(row["pred"]).astype(np.float32)
    gt = np.load(row["gt"]).astype(np.float32)
    pred[pred >= 1e9] = 0.0
    gt[gt >= 1e9] = 0.0
    mask = np.asarray(Image.open(row["mask"]).convert("L")) > 0
    if pred.shape != gt.shape:
        pred = _resize_to(pred, gt.shape[:2], nearest=False)
    if mask.shape != gt.shape[:2]:
        mask = _resize_to(mask.astype(np.uint8), gt.shape[:2], nearest=True) > 0
    return pred, gt, mask


def masked_rmse(pred, gt, mask, erode_r=10) -> float | None:
    m = valid_pixels(pred, gt, mask, erode_r=erode_r, min_depth=0.5, max_depth=10.0)
    if int(m.sum()) < 32:
        return None
    d = (pred.astype(np.float64) - gt.astype(np.float64))[m]
    return float(np.sqrt(np.mean(d * d)))


def apply_affine(pred: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return (alpha * pred.astype(np.float64) + beta).astype(np.float32)


def _fit_from_moments(parts: list[Moments]) -> tuple[float, float, Moments]:
    acc = merge_in_order(parts)
    alpha, beta = solve_scale_shift(acc)
    return alpha, beta, acc


def _summarize(errs: list[float]) -> dict:
    if not errs:
        return {"n": 0, "rmse_m": None}
    arr = np.asarray(errs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "rmse_m": float(arr.mean()),
        "rmse_std_m": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
    }


def _camera_center(T_wc: np.ndarray) -> np.ndarray:
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    return (-R.T @ t).astype(np.float64)


def stereo_depth(
    row_l: dict, row_r: dict, *, flow_backend: str, max_dy: float = 2.0, min_disp: float = 1.0
) -> tuple[np.ndarray, np.ndarray] | None:
    """L→R Farneback/RAFT depth plus per-pixel variance from 0.5 px disparity noise.

    Pixels fail the gate when vertical flow exceeds ``max_dy`` (not epipolar)
    or disparity is under ``min_disp``. Those stay NaN so inverse-variance
    fusion cannot be pulled around by a 1.5 m Farneback map.
    """
    if row_l["rgb"] is None or row_r["rgb"] is None:
        return None
    if not (row_l["ann"].is_file() and row_r["ann"].is_file()):
        return None
    rgb_l = np.asarray(Image.open(row_l["rgb"]).convert("RGB"))
    rgb_r = np.asarray(Image.open(row_r["rgb"]).convert("RGB"))
    flow = dense_flow(rgb_l, rgb_r, backend=flow_backend)
    Kl, Tl = load_pose(json.loads(row_l["ann"].read_text()))
    _, Tr = load_pose(json.loads(row_r["ann"].read_text()))
    B = float(np.linalg.norm(_camera_center(Tl) - _camera_center(Tr)))
    fx = float(Kl[0, 0])
    if B < 1e-4 or fx < 1.0:
        return None
    dx, dy = flow[..., 0], flow[..., 1]
    disp = -dx
    ok = (disp > min_disp) & (np.abs(dy) <= max_dy)
    z = np.full(disp.shape, np.nan, dtype=np.float32)
    z[ok] = ((fx * B) / disp[ok]).astype(np.float32)
    z = np.where(np.isfinite(z) & (z > 0.3) & (z < 12.0), z, np.nan).astype(np.float32)
    sigma_d = 0.5
    var = np.full(disp.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(z)
    var[finite] = (z[finite] * sigma_d / np.clip(disp[finite], 1e-3, None)) ** 2
    var = np.clip(var, 1e-4, 100.0)
    return z, var


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-stereo", action="store_true")
    ap.add_argument("--skip-loo", action="store_true")
    ap.add_argument("--stereo-limit", type=int, default=40, help="max L/R pairs for stereo (NFS)")
    ap.add_argument("--flow", choices=["farneback", "raft", "auto"], default="farneback")
    ap.add_argument("--max-per-tree", type=int, default=0, help="0 = all frames (debug cap)")
    ap.add_argument("--holdout-only", action="store_true", help="load paper val trees only (needs --alpha --beta)")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    args = ap.parse_args(argv)

    rows = discover_da2_triples(args.data_root)
    if args.holdout_only:
        if args.alpha is None or args.beta is None:
            print("--holdout-only needs --alpha and --beta", file=sys.stderr)
            return 2
        rows = [r for r in rows if r["tree"] in PAPER_VAL]
    if args.max_per_tree:
        capped: list[dict] = []
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            if counts[r["tree"]] >= args.max_per_tree:
                continue
            counts[r["tree"]] += 1
            capped.append(r)
        rows = capped
    if not rows:
        print("no Da2Finetune/depth/mask triples", file=sys.stderr)
        return 2

    fit_rows, hold_rows = split_holdout(rows)
    per_row: list[dict] = []
    hold_cache: list[dict] = []
    for i, r in enumerate(rows):
        pred, gt, mask = load_triple(r)
        mom = moments_from_image(pred, gt, mask)
        raw = masked_rmse(pred, gt, mask)
        rec = {"row": r, "moments": mom, "raw_rmse": raw}
        per_row.append(rec)
        if r["tree"] in PAPER_VAL:
            if not args.skip_stereo:
                rec["pred"], rec["gt"], rec["mask"] = pred, gt, mask
            hold_cache.append(rec)
        if (i + 1) % 50 == 0:
            print(f"[rigorous] loaded {i + 1}/{len(rows)}", file=sys.stderr, flush=True)

    fit_moms = [p["moments"] for p in per_row if p["row"]["tree"] not in PAPER_VAL and p["moments"] is not None]
    if args.alpha is not None and args.beta is not None:
        alpha, beta = float(args.alpha), float(args.beta)
        acc = merge_in_order([m for m in (p["moments"] for p in per_row) if m is not None]) or Moments()
        n_fit_images = 0
    else:
        if not fit_moms:
            print("no valid pixels on the fit cohort", file=sys.stderr)
            return 2
        alpha, beta, acc = _fit_from_moments(fit_moms)
        n_fit_images = sum(1 for p in per_row if p["row"]["tree"] not in PAPER_VAL and p["moments"] is not None)

    raw_hold = _summarize([p["raw_rmse"] for p in hold_cache if p["raw_rmse"] is not None])
    cal_hold_errs = []
    for p in hold_cache:
        if "pred" in p:
            pred, gt, mask = p["pred"], p["gt"], p["mask"]
        else:
            pred, gt, mask = load_triple(p["row"])
            if not args.skip_stereo:
                p["pred"], p["gt"], p["mask"] = pred, gt, mask
        err = masked_rmse(apply_affine(pred, alpha, beta), gt, mask)
        if err is not None:
            cal_hold_errs.append(err)
    cal_hold = _summarize(cal_hold_errs)
    raw_fit = _summarize(
        [p["raw_rmse"] for p in per_row if p["row"]["tree"] not in PAPER_VAL and p["raw_rmse"] is not None]
    )

    trees = sorted({r["tree"] for r in rows})
    loo = []
    if not args.skip_loo:
        by_tree: dict[str, list[dict]] = defaultdict(list)
        for p in per_row:
            by_tree[p["row"]["tree"]].append(p)
        for tree in trees:
            others = [p["moments"] for t, ps in by_tree.items() if t != tree for p in ps if p["moments"] is not None]
            a, b, _ = _fit_from_moments(others)
            errs = []
            for p in by_tree[tree]:
                if "pred" in p:
                    pred, gt, mask = p["pred"], p["gt"], p["mask"]
                else:
                    pred, gt, mask = load_triple(p["row"])
                err = masked_rmse(apply_affine(pred, a, b), gt, mask)
                if err is not None:
                    errs.append(err)
            loo.append({"tree": tree, "alpha": a, "beta": b, **_summarize(errs)})
            print(f"[loo] {tree} α={a:.4f} β={b:.4f}", file=sys.stderr, flush=True)

    stereo_hold = {"n": 0, "rmse_m": None, "fused_rmse_m": None, "flow": args.flow}
    if not args.skip_stereo:
        by_key = {(r["tree"], r["set_id"], r["shot"], r["side"]): r for r in hold_rows}
        fused_err, st_err = [], []
        n_st = 0
        for r in hold_rows:
            if r["side"] != "l" or n_st >= args.stereo_limit:
                continue
            rr = by_key.get((r["tree"], r["set_id"], r["shot"], "r"))
            if rr is None:
                continue
            n_st += 1
            packed = stereo_depth(r, rr, flow_backend=args.flow)
            p = next((x for x in hold_cache if x["row"] is r or x["row"]["pred"] == r["pred"]), None)
            if p is None or "pred" not in p:
                pred, gt, mask = load_triple(r)
            else:
                pred, gt, mask = p["pred"], p["gt"], p["mask"]
            cal = apply_affine(pred, alpha, beta)
            if packed is None:
                continue
            z, var_st_map = packed
            if z.shape != cal.shape:
                z = _resize_to(np.nan_to_num(z, nan=0.0), cal.shape[:2], nearest=False)
                z = np.where(z > 0.3, z, np.nan).astype(np.float32)
                var_st_map = _resize_to(np.nan_to_num(var_st_map, nan=1.0), cal.shape[:2], nearest=False)
            e_s = masked_rmse(z, gt, mask)
            if e_s is not None:
                st_err.append(e_s)
            var_da2 = np.full_like(cal, float((cal_hold.get("rmse_m") or 0.04) ** 2))
            # Floor stereo variance at 1 m² so a 1.5 m Farneback map cannot
            # outweigh a 3.4 cm DA2 prior. Analytic 0.5 px noise is optimistic.
            var_st = np.where(np.isfinite(var_st_map), np.maximum(var_st_map, 1.0), 100.0).astype(np.float32)
            fused, _ = fuse_depths(
                [cal, np.nan_to_num(z, nan=0.0)],
                variances=[var_da2, var_st],
                valid=[mask, np.isfinite(z) & mask],
            )
            fused = np.where(np.isfinite(fused), fused, cal)
            e_f = masked_rmse(fused, gt, mask)
            if e_f is not None:
                fused_err.append(e_f)
            print(f"[stereo] {n_st}/{args.stereo_limit} {r['tree']} {r['shot']}", file=sys.stderr, flush=True)
        stereo_hold = {
            "n": n_st,
            "rmse_m": float(np.mean(st_err)) if st_err else None,
            "fused_rmse_m": float(np.mean(fused_err)) if fused_err else None,
            "flow": args.flow,
            "note": (
                "Farneback on this rig is not a metric depth source. "
                "Stereo variance is floored at 1 m² so fusion cannot beat DA2 around. "
                "Do not ship stereo fusion; headline is DA2 hold-out affine."
            ),
        }

    drift_rows = []
    for p in hold_cache:
        r = p["row"]
        if r["rgb"] is None:
            continue
        img = Image.open(r["rgb"])
        if "pred" in p:
            depth = apply_affine(p["pred"], alpha, beta)
        else:
            pred, _, _ = load_triple(r)
            depth = apply_affine(pred, alpha, beta)
        drift_rows.append(request_stats(images=img, depths=depth))
    drift = baseline_from_stats(
        drift_rows,
        source=f"holdout {PAPER_VAL} n={len(drift_rows)} calibrated DA2",
    )

    payload = {
        "n_frames_total": len(rows),
        "n_fit_frames": len(fit_rows),
        "n_hold_frames": len(hold_rows),
        "fit_trees": sorted({r["tree"] for r in fit_rows}),
        "hold_trees": list(PAPER_VAL),
        "note": (
            "24k train set is still missing. Fit is 7 restored trees; "
            "headline RMSE is the 2 paper val trees. DA2-only — not the DINO refiner."
        ),
        "alpha_beta_holdout_fit": {
            "alpha": alpha,
            "beta": beta,
            "n_pixels": acc.n,
            "n_images": n_fit_images if args.alpha is None else None,
            "source": "least-squares on 7 non-paper trees, trunk mask, erode_r=10, GT 0.5–10 m",
        },
        "holdout_rmse": {
            "da2_raw_m": raw_hold,
            "da2_holdout_affine_m": cal_hold,
            "fit_cohort_raw_m": raw_fit,
        },
        "loo_per_tree": loo,
        "stereo_on_holdout": stereo_hold,
        "drift_baseline": {"n_frames": drift["n_frames"], "source": drift["source"]},
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        drift_path = args.out.with_name("baseline_stats_holdout.json")
        drift_path.write_text(json.dumps(drift, indent=2) + "\n")
        print(f"wrote {args.out} and {drift_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
