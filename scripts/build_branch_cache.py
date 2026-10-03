"""Cache per-frame branch labels so training never runs ``label_frame`` online.

Labelling one 1920x1080 frame against its L-Py tree costs ~2 s of KD-tree
queries, far more than a training step can spend, so every frame is labelled
once here. Per frame, under ``OUT/{tree}/{rig}/``:

* ``{stem}.cls.png``    uint8 class map (0 bg, 1 trunk, 2 branch, 3 shoot, 4 spur, 255 ignore)
* ``{stem}.skel.png``   uint8 1-px centerlines of the visible GT parts, value = class id
* ``{stem}.depth.png``  uint16 GT depth in mm (0 invalid, 65535 = at or beyond 65.535 m)
* ``{stem}.da2.png``    uint16 DA2-ft depth in mm, same encoding (absent if the .npy is)
* ``{stem}.graph.json`` GT TreeGraph + per-sample ``visible`` (paper-test and val trees only)
* ``{stem}.meta.json``  pose, render K, label stats; written last, so it marks a finished frame

Trees are sharded ``k % n_shards`` over the sorted tree list for the sbatch
array in ``scripts/run_build_branch_cache.sh``. Files are written atomically
and finished frames are skipped, so a re-run resumes where a shard stopped.

python scripts/build_branch_cache.py --trees lpy_envy_00042 --max-frames 3 --out /tmp/bc
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import cv2
import numpy as np

from spur_depth.branches import gt
from spur_depth.branches.tree import CLASSES, IGNORE, TreeGraph
from spur_depth.pipeline.reconstruct import load_pose

DEFAULT_OUT = Path("/nfs/hpc/share/sanchej7/spur-branch-cache/v1")
EVAL_TREES = frozenset(gt.PAPER_TEST + gt.VAL_TREES)
NAMES = CLASSES[1:]
MIN_Z_M = 0.05
_STATE: dict = {}


def output_paths(ref: gt.FrameRef, out: Path) -> dict[str, Path]:
    d = out / ref.tree / ref.rig
    kinds = {"cls": "png", "skel": "png", "depth": "png", "da2": "png"}
    kinds |= {"graph": "json", "meta": "json"}
    return {k: d / f"{ref.stem}.{k}.{ext}" for k, ext in kinds.items()}


def is_complete(ref: gt.FrameRef, out: Path) -> bool:
    paths = output_paths(ref, out)
    need = ["cls", "skel", "depth", "meta"]
    need += ["graph"] if ref.tree in EVAL_TREES else []
    need += ["da2"] if ref.path("Da2Finetune").is_file() else []
    return all(paths[k].is_file() for k in need)


def depth_to_mm(depth: np.ndarray) -> tuple[np.ndarray, int]:
    """(uint16 mm, 0 = invalid, clipped at 65535; number of clipped pixels)."""
    d = np.asarray(depth, dtype=np.float64)
    mm = np.rint(np.where(np.isfinite(d) & (d > 0), d, 0.0) * 1000.0)
    return np.clip(mm, 0, 65535).astype(np.uint16), int((mm > 65535).sum())


def draw_skeleton(graph: TreeGraph, K: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """1-px centerlines between consecutive visible samples; far parts first, near overwrite."""
    skel = np.zeros(shape, dtype=np.uint8)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    order = []
    for p in graph.parts.values():
        z = p.points[:, 2]
        vis = np.asarray(p.visible, dtype=bool) & (z > MIN_Z_M)
        if vis.any():
            order.append((-float(z[vis].mean()), p.pid, p, vis))
    for *_, p, vis in sorted(order, key=lambda o: o[:2]):
        z = np.where(vis, p.points[:, 2], 1.0)
        u = np.rint(fx * p.points[:, 0] / z + cx).astype(np.int64)
        v = np.rint(fy * p.points[:, 1] / z + cy).astype(np.int64)
        for i in np.nonzero(vis[:-1] & vis[1:])[0]:
            a, b = (int(u[i]), int(v[i])), (int(u[i + 1]), int(v[i + 1]))
            cv2.line(skel, a, b, int(p.cls), 1, cv2.LINE_8)
    return skel


def graph_json(graph: TreeGraph) -> dict:
    """``TreeGraph.to_json`` plus each part's per-sample ``visible`` flags (0/1)."""
    js = graph.to_json()
    for d in js["parts"]:
        vis = np.asarray(graph.parts[d["id"]].visible, dtype=bool)
        if len(vis) != len(d["points_m"]):
            raise ValueError(f"part {d['id']}: {len(vis)} visible flags, {len(d['points_m'])} pts")
        d["visible"] = vis.astype(int).tolist()
    return js


def _finite(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def _json(obj: dict, indent: int | None = None) -> bytes:
    sep = None if indent else (",", ":")
    return json.dumps(obj, indent=indent, separators=sep, allow_nan=False).encode()


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _init(model: gt.TreeModel, out: Path) -> None:
    _STATE.update(model=model, out=out)


def build_frame(ref: gt.FrameRef) -> dict:
    """Label one frame and write its files; returns the manifest row."""
    t0 = time.perf_counter()
    paths = output_paths(ref, _STATE["out"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, depth, mask, ann = gt.load_frame(ref)
        K, T_wc = (np.asarray(a, dtype=np.float64) for a in load_pose(ann))
        t1 = time.perf_counter()
        fg = gt.label_frame(_STATE["model"], K, T_wc, depth, mask)
    t2 = time.perf_counter()
    skel = draw_skeleton(fg.graph, K, depth.shape)
    depth_mm, depth_clip = depth_to_mm(depth)
    da2_src = ref.path("Da2Finetune")
    da2_mm, da2_clip = depth_to_mm(np.load(da2_src)) if da2_src.is_file() else (None, None)

    mask_px = int(mask.sum())
    labelled = int(((fg.cls > 0) & (fg.cls != IGNORE)).sum())
    part_cls = np.array([p.cls for p in fg.graph.parts.values()], dtype=np.int64)
    n_vis = np.bincount(part_cls, minlength=len(CLASSES))
    skel_px = np.bincount(skel.ravel(), minlength=len(CLASSES))
    meta = {
        "tree": ref.tree,
        "rig": ref.rig,
        "shot": ref.shot,
        "side": ref.side,
        "key": ref.key,
        "rgb": str(ref.path("Optical_flow")),
        "da2": str(da2_src) if da2_mm is not None else None,
        "size": list(depth.shape),
        "K": K.tolist(),
        "T_wc": T_wc.tolist(),
        "mask_px": mask_px,
        "labelled_frac": labelled / mask_px if mask_px else None,
        "surface_residual_m": _finite(fg.surface_residual_m),
        "n_visible": {n: int(n_vis[c]) for c, n in enumerate(NAMES, 1)},
        "skel_px": {n: int(skel_px[c]) for c, n in enumerate(NAMES, 1)},
        "clipped_px": {"depth": depth_clip, "da2": da2_clip},
    }
    if caught:
        meta["warnings"] = sorted({f"{w.category.__name__}: {w.message}" for w in caught})

    paths["meta"].parent.mkdir(parents=True, exist_ok=True)
    write_atomic(paths["cls"], _png(fg.cls))
    write_atomic(paths["skel"], _png(skel))
    write_atomic(paths["depth"], _png(depth_mm))
    if da2_mm is not None:
        write_atomic(paths["da2"], _png(da2_mm))
    if ref.tree in EVAL_TREES:
        write_atomic(paths["graph"], _json(graph_json(fg.graph)))
    write_atomic(paths["meta"], _json(meta, indent=1))
    t3 = time.perf_counter()
    times = {"load": t1 - t0, "label": t2 - t1, "draw_write": t3 - t2}
    return summary_row(meta, "done", t3 - t0, {k: round(v, 3) for k, v in times.items()})


def summary_row(
    meta: dict, status: str, seconds: float | None = None, t: dict | None = None
) -> dict:
    keys = ("labelled_frac", "surface_residual_m", "n_visible", "skel_px", "warnings")
    row = {"key": meta["key"], "status": status}
    row["seconds"] = None if seconds is None else round(seconds, 3)
    row |= {"t": t} if t else {}
    row |= {k: meta[k] for k in keys if k in meta}
    return row | {"da2": meta["da2"] is not None}


def safe_build(ref: gt.FrameRef) -> dict:
    try:
        return build_frame(ref)
    except Exception:
        return {"key": ref.key, "status": "error", "error": traceback.format_exc()}


def _git_head() -> str | None:
    src = Path(__file__).resolve().parents[1]
    r = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--trees", nargs="+", help="subset of trees to shard (default: all)")
    ap.add_argument("--max-frames", type=int, help="first n frames of each tree (testing)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    if not 0 <= args.shard < args.n_shards:
        ap.error(f"--shard must be in [0, {args.n_shards})")
    known = gt.all_trees()
    if unknown := sorted(set(args.trees or []) - set(known)):
        ap.error(f"unknown trees under {gt.DATA_ROOT}: {unknown}")
    trees = sorted(args.trees) if args.trees else known
    trees = [t for k, t in enumerate(trees) if k % args.n_shards == args.shard]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    rows: list[dict] = []
    model_s: dict[str, float] = {}
    tree_s: dict[str, float] = {}
    ctx = mp.get_context("fork")  # workers inherit the tree model and its KD-tree, no pickling
    for j, tree in enumerate(trees):
        t_tree = time.perf_counter()
        frames = gt.list_frames([tree])[: args.max_frames]
        todo = [f for f in frames if not is_complete(f, out)]
        done = [f for f in frames if f not in todo]
        tree_rows = []
        for f in done:
            meta = json.loads(output_paths(f, out)["meta"].read_text())
            tree_rows.append(summary_row(meta, "skipped"))
        if todo:
            try:
                t = time.perf_counter()
                model = gt.load_tree_model(tree, json.loads(todo[0].path("ann").read_text()))
                model.segment_index()
                model_s[tree] = round(time.perf_counter() - t, 3)
            except Exception:
                err = traceback.format_exc()
                tree_rows += [{"key": f.key, "status": "error", "error": err} for f in todo]
                todo = []
        if todo and (args.workers <= 1 or len(todo) == 1):
            _init(model, out)
            tree_rows += [safe_build(f) for f in todo]
        elif todo:
            with ctx.Pool(min(args.workers, len(todo)), _init, (model, out)) as pool:
                tree_rows += list(pool.imap_unordered(safe_build, todo))
        tree_s[tree] = round(time.perf_counter() - t_tree, 3)
        rows += tree_rows
        st = [r["status"] for r in tree_rows]
        n_da2 = sum(not r.get("da2", True) for r in tree_rows)
        print(
            f"[shard {args.shard}/{args.n_shards}] tree {j + 1}/{len(trees)} {tree}: "
            f"{st.count('done')} done, {st.count('skipped')} skipped, {st.count('error')} errors, "
            f"{n_da2} without DA2, {tree_s[tree]:.1f} s",
            flush=True,
        )

    rows.sort(key=lambda r: r["key"])
    secs = [r["seconds"] for r in rows if r["status"] == "done"]
    errors = [r["key"] for r in rows if r["status"] == "error"]
    manifest = {
        "shard": args.shard,
        "n_shards": args.n_shards,
        "trees": trees,
        "out": str(out),
        "data_root": str(gt.DATA_ROOT),
        "source": {"commit": _git_head(), "gt": gt.__file__},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "host": socket.gethostname(),
        "slurm": {k: os.environ.get(k) for k in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")},
        "encoding": {
            "cls": "uint8 class id, 255 ignore",
            "skel": "uint8 class id on 1-px centerlines, 0 none",
            "depth/da2": "uint16 mm, 0 invalid, 65535 = >= 65.535 m",
        },
        "counts": {
            "frames": len(rows),
            **{s: sum(r["status"] == s for r in rows) for s in ("done", "skipped", "error")},
            "da2_missing": sum(not r.get("da2", True) for r in rows),
        },
        "timing": {
            "wall_s": round(time.perf_counter() - t_start, 3),
            "frame_s_mean": round(float(np.mean(secs)), 3) if secs else None,
            "frame_s_median": round(float(np.median(secs)), 3) if secs else None,
            "model_load_s": model_s,
            "tree_wall_s": tree_s,
        },
        "da2_missing": [r["key"] for r in rows if not r.get("da2", True)],
        "errors": errors,
        "frames": rows,
    }
    write_atomic(out / f"manifest_shard{args.shard:02d}.json", _json(manifest, indent=1))
    print(f"[shard {args.shard}/{args.n_shards}] {manifest['counts']}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
