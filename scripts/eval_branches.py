"""Score branch perception on held-out trees: methods x depth sources.

Methods (what produces the per-pixel classes):
  oracle    ground-truth class map; isolates graph assembly + depth
  tinyunet  the repo's existing 256x256 tree-mask UNet, classes from geometry
  branchnet the new full-resolution RGB-D network (predictions from the GPU job)
Depth sources (what lifts pixels to metres), read from the label cache exactly as
BranchNet's prediction step read them:
  gt        rendered depth (a perfect RGB-D camera)
  sensor    simulated D435-style depth (spur_depth.branches.depth_noise)
  da2       DA2-ft monocular metric depth, i.e. RGB only
  fused     sensor depth where it agrees with affine-aligned DA2-ft, DA2-ft elsewhere
            (spur_depth.branches.depth_fusion); the classifier still sees raw sensor depth
  cdm       Camera Depth Model (CDM-D435, ICLR 2026) refinement of the same simulated
            sensor frame, precomputed by scripts/refine_depth_cdm.py into --cdm-dir;
            lift only, the classifier still sees raw sensor depth

Every row carries a fingerprint of the scorer code, assembly config, inputs and
method; a row is reused only when its fingerprint matches, and the summary
refuses rows from other fingerprints or an incomplete frame set.

python scripts/eval_branches.py --split val --methods oracle --depths gt sensor \
    --out /nfs/hpc/share/sanchej7/spur-branch-eval/dev
Per-frame rows go to OUT/frames/*.json; OUT/summary.json holds the aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from spur_depth.branches import gt as G
from spur_depth.branches.assemble import AssembleConfig, assemble
from spur_depth.branches.dataset import frame_depth, load_cached_frame, read_depth_m
from spur_depth.branches.depth_fusion import fuse_sensor_mono
from spur_depth.branches.metrics import aggregate, evaluate_frame, segmentation_counts
from spur_depth.branches.tree import CLASSES, IGNORE, TreeGraph

REPO = Path(__file__).resolve().parents[1]
CACHE = Path("/nfs/hpc/share/sanchej7/spur-branch-cache/v1")
CDM_DIR = Path("/nfs/hpc/share/sanchej7/spur-branch-eval/depth/cdm")
TINYUNET = REPO / "weights" / "trunk_unet_100tree.pt"
SCORER_FILES = (
    *sorted(str(p.relative_to(REPO)) for p in (REPO / "spur_depth" / "branches").glob("*.py")),
    "scripts/eval_branches.py",
    "docs/BRANCH_PROTOCOL.md",
)
METHODS = ("oracle", "tinyunet", "branchnet")
DEPTHS = ("gt", "sensor", "da2", "fused", "cdm")
LIFT_ONLY = ("fused", "cdm")  # depth sources the classifier never sees: it gets raw sensor


def _sha(path: Path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _png(path: Path) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    return a


def gt_graph(ref: G.FrameRef, cache: Path) -> tuple[TreeGraph, np.ndarray, np.ndarray]:
    """(GT graph with visible flags, GT class map, render K) from the label cache."""
    base = cache / ref.tree / ref.rig / ref.stem
    meta = json.loads(Path(f"{base}.meta.json").read_text())
    d = json.loads(Path(f"{base}.graph.json").read_text())
    g = TreeGraph.from_json(d)
    for p in d["parts"]:
        g.parts[int(p["id"])].visible = np.asarray(p["visible"], dtype=bool)
    return g, _png(Path(f"{base}.cls.png")), np.asarray(meta["K"], dtype=np.float64)


class TinyUNetBaseline:
    """The existing tree-mask model, run exactly as deployed (256x256, NEAREST upsampling)."""

    def __init__(self, path: Path = TINYUNET):
        import torch

        from spur_depth.pipeline.train_seg import TinyUNet

        self.model = TinyUNet()
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        from spur_depth.pipeline.train_seg import predict_mask

        return predict_mask(self.model, rgb, "cpu")


def predicted_classes(method: str, ref, net_source: str, gt_cls, frame, pred_dir, tiny):
    """(class map, confidence or None, assembly config tweaks, per-class seg scoring)."""
    if method == "oracle":
        return gt_cls, None, {}, True
    if method == "tinyunet":
        return tiny(frame["rgb"]).astype(np.uint8), None, {"geometric_classes": True}, False
    if method == "branchnet":
        base = Path(pred_dir) / net_source / ref.tree / ref.rig / ref.stem
        cls = _png(Path(f"{base}.cls.png"))
        conf = _png(Path(f"{base}.conf.png")).astype(np.float32) / 255.0
        return cls, conf, {}, True
    raise ValueError(method)


def git_state(repo: Path) -> dict:
    def run(*a):
        return subprocess.run(
            ["git", "-C", str(repo), *a], capture_output=True, text=True
        ).stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=all", "--", *SCORER_FILES)
    return {
        "commit": run("rev-parse", "HEAD"),
        "scorer_files_status": status,
        "dirty": bool(status),
    }


def provenance(args, cfg: AssembleConfig) -> dict:
    pred = {}
    if args.pred_dir is not None:
        summ = Path(args.pred_dir) / "predict_summary.json"
        pred = {"dir": str(args.pred_dir), "predict_summary_sha256": _sha(summ)}
        if summ.is_file():
            ck = json.loads(summ.read_text()).get("ckpt")
            pred["ckpt"] = ck
            pred["ckpt_sha256"] = _sha(Path(ck)) if ck else None
    return {
        "scorer_sha256": {f: _sha(REPO / f) for f in SCORER_FILES},
        "tinyunet_sha256": _sha(TINYUNET) if "tinyunet" in args.methods else None,
        "pred": pred,
        "cache": str(args.cache),
        "assemble_config": asdict(cfg),
        **(
            {
                "cdm": {
                    "dir": str(args.cdm_dir),
                    "manifest_sha256": _sha(args.cdm_dir / "manifest.json"),
                }
            }
            if "cdm" in args.depths
            else {}
        ),
    }


def fingerprint(prov: dict, method: str, source: str) -> str:
    blob = json.dumps({"prov": prov, "method": method, "source": source}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def frame_list(args) -> list[G.FrameRef]:
    trees = G.PAPER_TEST if args.split == "test" else G.VAL_TREES
    frames = []
    for t in trees:
        fr = G.list_frames([t])[:: args.every]
        frames += fr[: args.limit] if args.limit else fr
    return frames


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--methods", nargs="+", default=["oracle"], choices=METHODS)
    ap.add_argument("--depths", nargs="+", default=["gt"], choices=DEPTHS)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--cdm-dir", type=Path, default=CDM_DIR, help="refined depth for 'cdm'")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N frames per tree (val only)")
    ap.add_argument("--every", type=int, default=1, help="every Nth frame per tree (val only)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--cfg", type=str, default="{}", help="JSON AssembleConfig overrides")
    ap.add_argument("--summarize-only", action="store_true", help="rebuild summary.json")
    args = ap.parse_args(argv)
    cfg = replace(AssembleConfig(), **json.loads(args.cfg))
    if "branchnet" in args.methods and args.pred_dir is None:
        ap.error("--methods branchnet needs --pred-dir")
    if args.split == "test":
        if args.limit or args.every != 1:
            ap.error("--split test scores every frame: no --limit / --every")
        state = git_state(REPO)
        if state["dirty"]:
            sys.exit(
                f"refusing test scoring with uncommitted scorer files:\n{state['scorer_files_status']}"
            )
    prov = provenance(args, cfg)
    if args.summarize_only:
        return 0 if summarize_dir(args, prov)["complete"] else 1

    frames = frame_list(args)[args.shard :: args.n_shards]
    tiny = TinyUNetBaseline() if "tinyunet" in args.methods else None
    out_frames = args.out / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)
    fps = {(m, s): fingerprint(prov, m, s) for m in args.methods for s in args.depths}

    t0 = time.time()
    for n, ref in enumerate(frames):
        todo = []
        for source in args.depths:
            for method in args.methods:
                row_p = out_frames / f"{method}__{source}__{ref.key.replace('/', '__')}.json"
                if row_p.is_file():
                    try:
                        if (
                            json.loads(row_p.read_text())["meta"]["fingerprint"]
                            == fps[(method, source)]
                        ):
                            continue
                    except (json.JSONDecodeError, KeyError):
                        pass
                todo.append((method, source, row_p))
        if not todo:
            continue
        gtg, gt_cls, K = gt_graph(ref, args.cache)
        frame = load_cached_frame(ref, args.cache, skel=False)
        depths = {}

        def get_depth(src: str) -> np.ndarray:
            if src not in depths:
                depths[src] = frame_depth(src, frame, ref)
            return depths[src]

        for method, source, row_p in todo:
            net_source = "sensor" if source in LIFT_ONLY else source
            cls, conf, tweak, per_class = predicted_classes(
                method, ref, net_source, gt_cls, frame, args.pred_dir, tiny
            )
            if source == "fused":
                wood = (cls > 0) & (cls != IGNORE) if method != "oracle" else gt_cls > 0
                depth, _ = fuse_sensor_mono(get_depth("sensor"), get_depth("da2"), wood)
            elif source == "cdm":
                depth = read_depth_m(args.cdm_dir / ref.tree / ref.rig / f"{ref.stem}.depth.png")
            else:
                depth = get_depth(source)
            ts = time.time()
            pred = assemble(cls, depth, K, conf=conf, cfg=replace(cfg, **tweak))
            row = evaluate_frame(pred, gtg)
            row["seg"] = segmentation_counts(np.where(cls == IGNORE, 0, cls), gt_cls, per_class)
            row["meta"] = {
                "key": ref.key,
                "tree": ref.tree,
                "method": method,
                "depth": source,
                "fingerprint": fps[(method, source)],
                "assemble_s": round(time.time() - ts, 3),
                "n_parts_pred": len(pred.parts),
            }
            _write_atomic(row_p, json.dumps(row))
        print(f"[{n + 1}/{len(frames)}] {ref.key} {time.time() - t0:.0f}s", flush=True)

    if args.n_shards == 1:
        return 0 if summarize_dir(args, prov)["complete"] else 1
    return 0


def summarize_dir(args, prov: dict) -> dict:
    """Aggregate rows of the current fingerprint over the full expected frame set."""
    expected = {r.key for r in frame_list(args)}
    groups: dict[tuple[str, str], dict[str, dict]] = {
        (m, s): {} for m in args.methods for s in args.depths
    }
    stale = 0
    for p in sorted((args.out / "frames").glob("*.json")):
        r = json.loads(p.read_text())
        m = r["meta"]
        key = (m["method"], m["depth"])
        if key not in groups or m["key"] not in expected:
            continue
        if m["fingerprint"] != fingerprint(prov, *key):
            stale += 1
            continue
        groups[key][m["key"]] = r
    result = {
        "protocol": "docs/BRANCH_PROTOCOL.md v2",
        "split": args.split,
        "trees": list(G.PAPER_TEST if args.split == "test" else G.VAL_TREES),
        "frames_expected": len(expected),
        "frame_list_sha256": hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest(),
        "argv": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "git": git_state(REPO),
        "provenance": prov,
        "stale_rows_ignored": stale,
        "classes": list(CLASSES),
        "results": {},
        "missing": {},
    }
    complete = True
    for (method, source), rows in sorted(groups.items()):
        missing = sorted(expected - set(rows))
        if missing:
            complete = False
            result["missing"][f"{method}/{source}"] = missing
        if not rows:
            continue
        clean = [{k: v for k, v in r.items() if k != "meta"} for r in rows.values()]
        agg = aggregate(clean)
        per_tree = {}
        for t in sorted({r["meta"]["tree"] for r in rows.values()}):
            sub = [
                {k: v for k, v in r.items() if k != "meta"}
                for r in rows.values()
                if r["meta"]["tree"] == t
            ]
            per_tree[t] = aggregate(sub)["summary"]
        result["results"][f"{method}/{source}"] = {
            "n_frames": agg["n_frames"],
            "fingerprint": fingerprint(prov, method, source),
            "summary": agg["summary"],
            "per_tree": per_tree,
            "totals": agg["totals"],
            "assemble_s_mean": float(np.mean([r["meta"]["assemble_s"] for r in rows.values()])),
        }
    result["complete"] = complete
    args.out.mkdir(parents=True, exist_ok=True)
    _write_atomic(args.out / "summary.json", json.dumps(result, indent=2) + "\n")
    if not complete:
        print(f"INCOMPLETE: missing rows for {sorted(result['missing'])}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
